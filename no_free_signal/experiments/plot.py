"""Figure generator for the experiment results.

Reads a directory of ``<arm>_seed<n>.jsonl`` files (produced by
``harness.run_one``) and writes PNGs into ``--out``. Each figure
aggregates across seeds within an arm.

Five figures, in the order the paper presents them:

  Fig 1 — Fitness ladder. Per-arm pop_AUC mean ± bootstrap 95% CI as a
          horizontal bar chart. Headline visual: D wins, but E ≈ F nearly
          match D, and B ≈ A.
  Fig 2 — Pairwise comparison probabilities. P(A>B) bars for the six
          pre-defined comparisons, with 50% (no effect) and 95%
          (significance threshold) reference lines.
  Fig 3 — Substrate-LLM alignment over time, with null-shuffled baseline
          shown as a dashed horizontal line per arm panel.
  Fig 4 — Communication activity over time (vocalize emits per snapshot
          window).
  Fig 5 — Population dynamics over time (supplementary).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from no_free_signal.experiments import stats as _stats

# Grayscale palette for print-friendly figures. Seven distinguishable
# shades from black to light gray; combined with arm labels in legends
# this is readable in B&W. Per-arm line styles (solid/dashed/dotted)
# are also varied below where line plots stack arms on the same axes.
ARM_COLORS = {
    "A": "#000000",
    "B": "#262626",
    "C": "#4d4d4d",
    "D": "#737373",
    "E": "#969696",
    "F": "#bdbdbd",
    "G": "#d9d9d9",
}
ARM_LINESTYLES = {
    "A": "-",
    "B": "--",
    "C": ":",
    "D": "-.",
    "E": (0, (3, 1, 1, 1)),
    "F": (0, (5, 2)),
    "G": (0, (1, 1)),
}
ARM_LABELS = {
    "A": "A: mute",
    "B": "B: fixed substrate + LLM",
    "C": "C: evolvable substrate, no LLM",
    "D": "D: full stack (treatment)",
    "E": "E: scrambled LLM",
    "F": "F: context-randomized LLM",
    "G": "G: random emitter, no LLM",
}


def _load_run(path: Path) -> dict[str, Any]:
    header: dict[str, Any] | None = None
    snapshots: list[dict[str, Any]] = []
    summary: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("kind")
            if kind == "header":
                header = obj
            elif kind == "snapshot":
                snapshots.append(obj)
            elif kind == "summary":
                summary = obj
    return {"path": path, "header": header, "snapshots": snapshots, "summary": summary}


def _load_dir(in_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Group runs by arm. Returns {arm: [run_dict, ...]}.

    Falls back to filename-derived arm if the run lacks a header record."""
    runs: dict[str, list[dict[str, Any]]] = {a: [] for a in ARM_COLORS}
    for jsonl in sorted(in_dir.glob("*.jsonl")):
        if jsonl.name == "manifest.jsonl":
            continue
        run = _load_run(jsonl)
        if run["header"] is not None:
            arm = run["header"].get("arm", "?").upper()
        elif run["summary"] is not None:
            arm = str(run["summary"].get("arm", jsonl.name[0])).upper()
        else:
            arm = jsonl.name[0].upper()
        if arm not in runs:
            runs[arm] = []
        runs[arm].append(run)
    return runs


def _aligned_grid(runs: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (ticks, values[seeds, ticks]) aligned across runs.

    Truncates to the shortest run since extinction kills runs at varying
    ticks; the mean line therefore reflects only seeds still alive at
    each tick."""
    if not runs:
        return np.array([]), np.zeros((0, 0))
    min_len = min(len(r["snapshots"]) for r in runs if r["snapshots"]) if runs else 0
    if min_len == 0:
        return np.array([]), np.zeros((0, 0))
    grid = np.zeros((len(runs), min_len), dtype=np.float64)
    ticks = np.array([])
    for i, r in enumerate(runs):
        snaps = r["snapshots"][:min_len]
        ticks = np.array([s["tick"] for s in snaps], dtype=np.float64)
        grid[i] = np.array([s.get(key, 0.0) for s in snaps], dtype=np.float64)
    return ticks, grid


# ---------------------------------------------------------------------------
# Fig 1 — Fitness ladder (headline figure)
# ---------------------------------------------------------------------------

def fig_fitness_ladder(arm_data: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Horizontal bar chart of per-arm pop_AUC mean ± bootstrap 95% CI.

    The treatment arm D is highlighted with a slightly thicker edge so
    the visual reads as 'D vs everyone else' at a glance."""
    arms = [a for a in _stats.ARM_ORDER if a in arm_data]
    if not arms:
        return

    means, lows, highs = [], [], []
    for a in arms:
        m, lo, hi = _stats.boot_ci(arm_data[a]["pop_aucs"])
        means.append(m)
        lows.append(m - lo)
        highs.append(hi - m)

    _stats._seed_rng()  # reproducible bootstrap

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(arms))
    colors = [ARM_COLORS[a] for a in arms]
    edge = ["black" if a == "D" else "none" for a in arms]
    lw = [2.0 if a == "D" else 0.5 for a in arms]
    ax.barh(y, means, xerr=[lows, highs], color=colors,
            edgecolor=edge, linewidth=lw,
            error_kw={"ecolor": "#333", "capsize": 4, "elinewidth": 1.2})
    ax.set_yticks(y)
    ax.set_yticklabels([ARM_LABELS[a] for a in arms])
    ax.invert_yaxis()  # arm A at top
    ax.set_xlabel("population AUC (creatures × ticks)")
    ax.set_title("Fig 1 — Fitness ladder: pop_AUC by arm\n"
                 "(mean ± bootstrap 95% CI, n=20 seeds per arm)")
    ax.grid(True, axis="x", alpha=0.3)

    # Annotate each bar with its mean.
    for i, (a, m) in enumerate(zip(arms, means)):
        ax.text(m, i, f"  {m:,.0f}", va="center", fontsize=9, color="#333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — Pairwise comparison probabilities
# ---------------------------------------------------------------------------

def fig_pairwise_probabilities(arm_data: dict[str, dict[str, Any]], out_path: Path) -> None:
    """For each pre-defined comparison, plot P(A>B) as a horizontal bar.

    Reference lines at 50% (no effect) and 95% (conventional significance).
    Bars use grayscale fills (darkest = significant, mid = directional,
    light = below 50%) for print-friendly rendering."""
    pairs = _stats.pairwise_results(arm_data, metric_key="pop_aucs")
    if not pairs:
        return

    # Sort so most positive at top.
    pairs.sort(key=lambda d: d["p_a_gt_b"], reverse=True)

    labels = [f"{p['a']} > {p['b']}\n({p['desc']})" for p in pairs]
    probs = [p["p_a_gt_b"] * 100 for p in pairs]

    def _shade(pct: float) -> str:
        if pct >= 95:
            return "#1a1a1a"  # near-black: significant
        if pct >= 50:
            return "#666666"  # mid-gray: directional
        return "#cccccc"      # light gray: reversed / no effect

    colors = [_shade(p) for p in probs]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))
    ax.barh(y, probs, color=colors, edgecolor="#000000", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("P(A > B) on population AUC, bootstrap 10k")
    ax.set_xlim(0, 100)
    ax.axvline(50, color="#444", linestyle=":", linewidth=1, label="50% (no effect)")
    ax.axvline(95, color="#000", linestyle="--", linewidth=1.2, label="95% (significance threshold)")
    ax.set_title("Fig 2 — Pairwise bootstrap probabilities by comparison")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    for i, p in enumerate(probs):
        ax.text(p + 1.5, i, f"{p:.0f}%", va="center", fontsize=9, color="#222")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — Substrate-LLM alignment (with null baseline)
# ---------------------------------------------------------------------------

def fig_substrate_alignment(runs_by_arm: dict[str, list[dict]],
                              alignment_stats: dict[str, dict[str, float]],
                              out_path: Path) -> None:
    """One panel per arm: cos(attention, LLM emit) over time, plus a
    dashed horizontal line at the null-shuffled baseline. If the
    observed cosine never sits clearly above the dashed line, the
    substrate is not directionally aligning with the LLM beyond what a
    bin-permutation null would produce."""
    arms_present = [a for a in _stats.ARM_ORDER if runs_by_arm.get(a)]
    if not arms_present:
        return
    n = len(arms_present)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False)

    for i, arm in enumerate(arms_present):
        ax = axes[i // cols][i % cols]
        runs = runs_by_arm[arm]
        ticks, llm_cos = _aligned_grid(runs, "cos_attention_vs_llm")
        if llm_cos.shape[0] == 0:
            ax.set_visible(False)
            continue
        m = llm_cos.mean(axis=0)
        sd = llm_cos.std(axis=0)
        ax.plot(ticks, m, color=ARM_COLORS[arm], linewidth=2,
                linestyle=ARM_LINESTYLES[arm],
                label="cos(attention, LLM emit)")
        ax.fill_between(ticks, m - sd, m + sd, color=ARM_COLORS[arm], alpha=0.18)

        # Null-shuffled baseline (dashed horizontal line).
        st = alignment_stats.get(arm, {})
        null_mean = st.get("null_mean", 0.0)
        if null_mean > 0:
            ax.axhline(null_mean, color="#222", linestyle="--", linewidth=1.0,
                       label=f"null-shuffled mean ({null_mean:.2f})")

        # Annotate the z-score in the corner.
        z = st.get("z", 0.0)
        ax.text(0.97, 0.04, f"z = {z:+.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#aaa", "boxstyle": "round,pad=0.3"})

        ax.set_title(ARM_LABELS[arm], fontsize=10)
        ax.set_xlabel("tick")
        ax.set_ylabel("cosine similarity")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=7)

    for j in range(len(arms_present), rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Fig 3 — Substrate-LLM alignment over time, with null-shuffled baseline\n"
                 "(z near 0 ⇒ observed alignment indistinguishable from bin-permutation null)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — Communication activity over time
# ---------------------------------------------------------------------------

def fig_communication_activity(runs_by_arm: dict[str, list[dict]], out_path: Path) -> None:
    """Vocalize emissions per snapshot window over time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for arm in _stats.ARM_ORDER:
        runs = runs_by_arm.get(arm, [])
        if not runs:
            continue
        ticks, grid = _aligned_grid(runs, "vocalize_emits_total")
        if grid.shape[0] == 0:
            continue
        rate = np.diff(grid, axis=1, prepend=0.0)
        m = rate.mean(axis=0)
        lo = np.quantile(rate, 0.25, axis=0)
        hi = np.quantile(rate, 0.75, axis=0)
        ax.plot(ticks, m, color=ARM_COLORS[arm], label=ARM_LABELS[arm],
                linewidth=2, linestyle=ARM_LINESTYLES[arm])
        ax.fill_between(ticks, lo, hi, color=ARM_COLORS[arm], alpha=0.13)
    ax.set_xlabel("tick")
    ax.set_ylabel("vocalize emissions per snapshot window")
    ax.set_title("Fig 4 — Communication activity by arm (median ± IQR)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 5 — Population dynamics (supplementary)
# ---------------------------------------------------------------------------

def fig_population_dynamics(runs_by_arm: dict[str, list[dict]], out_path: Path) -> None:
    """Population over time per arm, mean ± std across surviving seeds."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for arm in _stats.ARM_ORDER:
        runs = runs_by_arm.get(arm, [])
        if not runs:
            continue
        ticks, grid = _aligned_grid(runs, "n_creatures")
        if grid.shape[0] == 0:
            continue
        m = grid.mean(axis=0)
        sd = grid.std(axis=0)
        ax.plot(ticks, m, color=ARM_COLORS[arm], label=ARM_LABELS[arm],
                linewidth=2, linestyle=ARM_LINESTYLES[arm])
        ax.fill_between(ticks, m - sd, m + sd, color=ARM_COLORS[arm], alpha=0.12)
    ax.set_xlabel("tick")
    ax.set_ylabel("population size")
    ax.set_title("Fig 5 — Population dynamics by arm (supplementary; mean ± std)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 6 — Kaplan-Meier survival curves
# ---------------------------------------------------------------------------

def fig_survival_curves(arm_data: dict[str, dict[str, Any]],
                         out_path: Path,
                         horizon: int = 15_000) -> None:
    """Per-arm Kaplan-Meier-style survival curves built from per-seed
    extinction times. Runs that completed without extinction are
    right-censored at `horizon`. The curve is the proportion of seeds
    whose population was still alive at each tick, drawn as a step
    function. With 20 seeds per arm this is a coarse curve, but the
    arm-level separation is the relevant comparison."""
    arms = [a for a in _stats.ARM_ORDER if a in arm_data]
    if not arms:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arm in arms:
        d = arm_data[arm]
        ticks = d.get("ticks_completed") or []
        extinct = d.get("extinct_flags") or []
        if not ticks:
            continue
        # For non-extinct runs, treat their ticks_completed as right-censoring
        # at horizon: they contribute to "still alive" for their whole run.
        events: list[tuple[int, bool]] = []
        for t, e in zip(ticks, extinct):
            events.append((int(t), bool(e)))

        n = len(events)
        sorted_events = sorted(events, key=lambda x: x[0])
        # Step over sorted event times; at each extinction event, drop the
        # surviving fraction by 1/n. Censored events do not drop the curve.
        xs: list[float] = [0.0]
        ys: list[float] = [1.0]
        alive = n
        for t, e in sorted_events:
            if e:
                alive -= 1
                xs.append(float(t))
                ys.append(alive / n)
        # Extend the curve out to horizon.
        xs.append(float(horizon))
        ys.append(alive / n)
        ax.step(xs, ys, where="post",
                color=ARM_COLORS[arm], linewidth=2,
                linestyle=ARM_LINESTYLES[arm],
                label=f"{ARM_LABELS[arm]} ({alive}/{n} survived)")

    ax.set_xlabel("tick")
    ax.set_ylabel("fraction of seeds with non-extinct population")
    ax.set_xlim(0, horizon)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title("Fig 6 — Kaplan-Meier survival curves by arm\n"
                 "(step drops at extinction events; runs reaching the "
                 f"{horizon:,}-tick horizon are right-censored)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 7 — System architecture diagram
# ---------------------------------------------------------------------------

def fig_architecture_diagram(out_path: Path) -> None:
    """Boxes-and-arrows diagram of the experiment's control loop.

    Drawn as a static matplotlib figure rather than imported from a
    vector source so it tracks the repo and never goes stale. Reads:

        genome → substrate traits (production / perception / reflex)
               → observation + audio perception (8-bin sampling)
               → action selector (reflex > LLM > brain hierarchy)
               → emission + action (8-bin wave + 10-action choice)
               → environment (audio physics, hunger, predation)
               → survival / reproduction → genome inheritance

    Each box is colored by category (substrate / controllers /
    environment) so a reviewer can see the loop at a glance."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # B&W architecture: three grayscale fills distinguishable when
    # printed greyscale. The reader still gets the category cue from
    # box position and labels.
    SUB = "#e6e6e6"   # substrate boxes — light gray
    CTRL = "#bfbfbf"  # controllers — medium gray
    ENV = "#999999"   # environment — darker gray

    def box(x, y, w, h, label, color):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.04,rounding_size=0.18",
                               linewidth=1.2, edgecolor="#333",
                               facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=9, wrap=True)

    def arrow(x0, y0, x1, y1, label="", curve=0.0):
        rad = curve
        a = FancyArrowPatch((x0, y0), (x1, y1),
                            arrowstyle="-|>", mutation_scale=14,
                            connectionstyle=f"arc3,rad={rad}",
                            linewidth=1.1, color="#333")
        ax.add_patch(a)
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.18, label, ha="center", va="bottom",
                    fontsize=8, color="#444")

    # Top row: genome → substrate traits
    box(0.4, 5.0, 2.2, 1.0, "Genome\n(heritable traits)", SUB)
    box(3.2, 5.0, 3.2, 1.0,
        "Substrate traits\nproduction bias · perception\nattention · reflex thresholds", SUB)

    # Middle row: observation → action selector → emission
    box(0.4, 3.2, 2.2, 1.0, "Observation\n(audio + visual)", CTRL)
    box(3.2, 3.2, 3.2, 1.0,
        "Action selector\nreflex > LLM > brain", CTRL)
    box(7.0, 3.2, 3.5, 1.0,
        "Emission + action\n8-bin wave · 10 actions", CTRL)

    # Bottom row: environment → survival / reproduction
    box(0.4, 1.4, 3.0, 1.0,
        "Environment\naudio physics · food ·\npredation · death", ENV)
    box(4.0, 1.4, 3.0, 1.0,
        "Survival /\nreproduction", ENV)
    box(7.6, 1.4, 2.9, 1.0,
        "Genome inheritance\n(parent traits ± mutation)", SUB)

    # Arrows — the control loop, in order
    arrow(2.6, 5.5, 3.2, 5.5)                # genome → substrate traits
    arrow(4.8, 5.0, 4.8, 4.2)                # substrate traits → action selector (down)
    arrow(2.6, 3.7, 3.2, 3.7)                # observation → action selector
    arrow(6.4, 3.7, 7.0, 3.7)                # action selector → emission/action
    arrow(8.75, 3.2, 7.0, 2.4, curve=-0.15)  # emission/action → survival/reproduction
    arrow(4.0, 1.9, 3.4, 1.9)                # survival/reproduction → environment
    arrow(1.9, 2.4, 1.9, 3.2)                # environment → observation (close inner loop)
    arrow(7.0, 1.9, 7.6, 1.9)                # survival/reproduction → genome inheritance
    arrow(9.05, 2.4, 1.5, 5.0, curve=0.30)   # genome inheritance → genome (close outer loop)

    # Legend
    legend_y = 0.4
    box(0.4, legend_y, 1.7, 0.55, "substrate", SUB)
    box(2.3, legend_y, 1.7, 0.55, "controller", CTRL)
    box(4.2, legend_y, 1.7, 0.55, "environment", ENV)

    ax.set_title("Fig 7 — System architecture\n"
                 "(genome → substrate → perception → action → environment → "
                 "selection → genome)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def render_all(in_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Load runs, compute stats, write all figures. Returns a dict with
    the loaded run set, arm_data, and alignment_stats so the report
    generator can reuse them without re-loading."""
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_by_arm = _load_dir(in_dir)
    arm_data = _stats.compute_arm_data(runs_by_arm)
    alignment_stats = _stats.arm_alignment_stats(runs_by_arm)

    fig_fitness_ladder(arm_data, out_dir / "fig1_fitness_ladder.png")
    fig_pairwise_probabilities(arm_data, out_dir / "fig2_pairwise_probabilities.png")
    fig_substrate_alignment(runs_by_arm, alignment_stats, out_dir / "fig3_alignment.png")
    fig_communication_activity(runs_by_arm, out_dir / "fig4_communication.png")
    fig_population_dynamics(runs_by_arm, out_dir / "fig5_population.png")
    fig_survival_curves(arm_data, out_dir / "fig6_survival.png")
    fig_architecture_diagram(out_dir / "fig7_architecture.png")

    return {
        "runs_by_arm": runs_by_arm,
        "arm_data": arm_data,
        "alignment_stats": alignment_stats,
        "figures_dir": out_dir,
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="no_free_signal.experiments.plot")
    p.add_argument("--in", dest="in_dir", default="results")
    p.add_argument("--out", dest="out_dir", default=None,
                   help="output directory for PNG files (default: <in>/figures)")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    payload = render_all(in_dir, out_dir)
    n_total = sum(len(v) for v in payload["runs_by_arm"].values())
    print(f"[plot] loaded {n_total} runs across "
          f"{sum(1 for v in payload['runs_by_arm'].values() if v)} arms")
    print(f"[plot] wrote 7 figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
