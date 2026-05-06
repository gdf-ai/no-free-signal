"""Behavioral receiver-response analysis.

Reads ``kind=emit_event`` records from ``results_15k_behavioral/`` runs
and answers the question: when a creature *hears* an emission, does its
behavior change in fitness-relevant ways relative to creatures that
don't hear it?

Each emit_event record carries a list of receivers within the audio
radius. Each receiver is tagged with:

- ``attention_weighted_strength`` — how much it heard, post-attention
- ``heard_threshold_{005,010,025}`` — three thresholds the smoke
  established as informative
- ``self_hearing`` — emitter and receiver are the same creature
- pre-emit state (energy, predator distance, food distance) at emission tick
- outcomes for t+1, t+2, t+3, t+10, t+25, t+50 (action, predator
  distance, survived, energy delta, food eaten, reproduced)

We split each receiver into a "heard" or "no-heard" bucket per threshold
(receivers with strength <= threshold are natural matched controls — they
were within audio radius of the same emitter, so spatial/temporal
context is similar by construction). Per-seed metrics are computed,
then bootstrap across seeds within each arm. Cross-arm comparisons
(D vs G, F vs G, etc.) follow the same paired-seed bootstrap pattern
used in the fitness analysis (`stats.boot_diff_paired`).

Output is one JSON file (``behavior_results.json``) with the per-arm
and pairwise tables, plus a printed summary suitable for the report.

Pseudoreplication note: every metric is aggregated to a per-seed
scalar before bootstrap. We never bootstrap over individual receiver
events. This means the effective N for any test is the number of
seeds (20 per arm), not the number of receiver events (often >100k
per arm)."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Audio radius of the simulator (matches AudioField.ATTENUATION_RADIUS).
AUDIO_RADIUS = 6
# 9999 is the sentinel for "no predator within sight" used by the
# behavioral logger. Filter these out of predator-conditioned metrics.
NO_PREDATOR_SENTINEL = 9999

ARM_ORDER = ("A", "B", "C", "D", "E", "F", "G")
THRESHOLDS = (0.05, 0.10, 0.25)

# Pairwise comparisons we care about for the receiver-response story.
# Same shape as stats.PAIRWISE_COMPARISONS but tuned to behavior.
PAIRWISE = (
    ("D", "G", "LLM emissions vs matched random noise: behavioral effect"),
    ("F", "G", "context-randomized LLM vs random noise (cleanest internal control)"),
    ("D", "B", "evolvable substrate vs frozen substrate, both with LLM"),
    ("D", "C", "full stack vs no-emitter control"),
    ("D", "A", "full stack vs mute floor"),
    ("E", "G", "scrambled LLM vs noise"),
)

# Behavioral metrics we extract per receiver record.
# Each is a function (receiver_dict) -> float | None (None means skip).
def _metric_flee_3tick(r: dict) -> float | None:
    """1.0 if predator distance increased over t..t+3, else 0.0.
    None if no predator at t (sentinel) -- meaningless to ask if it
    fled when there was nothing to flee from."""
    p0 = r.get("receiver_dist_predator_t")
    p3 = r.get("predator_dist_t3")
    if p0 is None or p3 is None:
        return None
    if p0 >= NO_PREDATOR_SENTINEL or p3 >= NO_PREDATOR_SENTINEL:
        return None
    return 1.0 if p3 > p0 else 0.0


def _metric_flee_strict(r: dict) -> float | None:
    """1.0 if predator distance increased by >=2 over t..t+3."""
    p0 = r.get("receiver_dist_predator_t")
    p3 = r.get("predator_dist_t3")
    if p0 is None or p3 is None:
        return None
    if p0 >= NO_PREDATOR_SENTINEL or p3 >= NO_PREDATOR_SENTINEL:
        return None
    return 1.0 if (p3 - p0) >= 2 else 0.0


def _metric_predator_dist_delta_t3(r: dict) -> float | None:
    p0 = r.get("receiver_dist_predator_t")
    p3 = r.get("predator_dist_t3")
    if p0 is None or p3 is None:
        return None
    if p0 >= NO_PREDATOR_SENTINEL or p3 >= NO_PREDATOR_SENTINEL:
        return None
    return float(p3 - p0)


def _metric_survived_next_25(r: dict) -> float | None:
    v = r.get("survived_next_25")
    return None if v is None else (1.0 if v else 0.0)


def _metric_survived_next_50(r: dict) -> float | None:
    v = r.get("survived_next_50")
    return None if v is None else (1.0 if v else 0.0)


def _metric_energy_delta_next_10(r: dict) -> float | None:
    v = r.get("energy_delta_next_10")
    return None if v is None else float(v)


def _metric_food_eaten_next_10(r: dict) -> float | None:
    v = r.get("food_eaten_next_10")
    return None if v is None else float(v)


def _metric_reproduced_next_50(r: dict) -> float | None:
    v = r.get("reproduced_next_50")
    return None if v is None else (1.0 if v else 0.0)


METRICS: dict[str, Any] = {
    "flee_3tick": _metric_flee_3tick,
    "flee_strict": _metric_flee_strict,
    "predator_dist_delta_t3": _metric_predator_dist_delta_t3,
    "survived_next_25": _metric_survived_next_25,
    "survived_next_50": _metric_survived_next_50,
    "energy_delta_next_10": _metric_energy_delta_next_10,
    "food_eaten_next_10": _metric_food_eaten_next_10,
    "reproduced_next_50": _metric_reproduced_next_50,
}


def stream_emit_events(path: Path) -> Iterable[dict]:
    """Yield emit_event dicts from a JSONL file. Skips other record kinds."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("kind") == "emit_event":
                yield obj


def per_seed_metrics(path: Path,
                     threshold: float,
                     include_self_hearing: bool = False
                     ) -> dict[str, Any]:
    """Compute per-(arm, seed) behavioral metrics for one run.

    Returns a dict with:
      - arm, seed
      - heard / no_heard sample counts
      - For each metric: heard_mean, no_heard_mean, diff
        (None if neither bucket has any valid records for that metric)
    """
    arm: str | None = None
    seed: int | None = None
    # Collect per-receiver metric values into "heard" vs "no_heard" buckets.
    heard: dict[str, list[float]] = {m: [] for m in METRICS}
    no_heard: dict[str, list[float]] = {m: [] for m in METRICS}

    for ev in stream_emit_events(path):
        if arm is None:
            arm = ev.get("arm")
            seed = ev.get("seed")
        for r in ev.get("receivers", []):
            if not include_self_hearing and r.get("self_hearing"):
                continue
            strength = float(r.get("attention_weighted_strength", 0.0))
            bucket = heard if strength > threshold else no_heard
            for m, fn in METRICS.items():
                v = fn(r)
                if v is not None:
                    bucket[m].append(v)

    out: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "n_heard": len(heard["energy_delta_next_10"]),
        "n_no_heard": len(no_heard["energy_delta_next_10"]),
    }
    for m in METRICS:
        h = heard[m]
        n = no_heard[m]
        out[m] = {
            "heard_mean": statistics.mean(h) if h else None,
            "no_heard_mean": statistics.mean(n) if n else None,
            "n_heard": len(h),
            "n_no_heard": len(n),
        }
    return out


def boot_ci(values: list[float], n: int = 10_000, ci: float = 95) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(n):
        s = [random.choice(values) for _ in values]
        means.append(sum(s) / len(s))
    means.sort()
    lo = means[int(n * (50 - ci / 2) / 100)]
    hi = means[int(n * (50 + ci / 2) / 100)]
    return statistics.mean(values), lo, hi


def boot_diff_paired(a_vals: list[float], b_vals: list[float],
                     n: int = 10_000) -> tuple[float, float, float, float]:
    """Paired-seed bootstrap of a - b. a_vals[i] and b_vals[i] must
    refer to the same seed (i.e., already aligned by seed)."""
    if not a_vals or not b_vals or len(a_vals) != len(b_vals):
        return 0.0, 0.0, 0.0, 0.5
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    boots = []
    for _ in range(n):
        s = [random.choice(diffs) for _ in diffs]
        boots.append(sum(s) / len(s))
    boots.sort()
    lo = boots[int(n * 0.025)]
    hi = boots[int(n * 0.975)]
    p_pos = sum(1 for x in boots if x > 0) / len(boots)
    return statistics.mean(boots), lo, hi, p_pos


def per_seed_diffs(per_seed_dicts: list[dict[str, Any]],
                   metric: str) -> list[tuple[int, float]]:
    """For each seed in this arm, return (seed, heard_mean - no_heard_mean).
    Skips seeds where either bucket is empty for the metric."""
    out = []
    for d in per_seed_dicts:
        sd = d[metric]
        if sd["heard_mean"] is None or sd["no_heard_mean"] is None:
            continue
        out.append((d["seed"], sd["heard_mean"] - sd["no_heard_mean"]))
    return out


def bootstrap_arm(per_seed_dicts: list[dict[str, Any]],
                  metric: str) -> dict[str, Any] | None:
    """Bootstrap per-seed (heard - no_heard) diffs across seeds within
    one arm. Returns mean diff, 95% CI, P(heard > no_heard), n_seeds."""
    diffs = [d for _, d in per_seed_diffs(per_seed_dicts, metric)]
    if len(diffs) < 2:
        return None
    mean, lo, hi = boot_ci(diffs)
    p_pos = sum(1 for x in diffs if x > 0) / len(diffs)
    return {
        "n_seeds": len(diffs),
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_positive": p_pos,
    }


def cross_arm_compare(arm_a_seeds: list[dict[str, Any]],
                       arm_b_seeds: list[dict[str, Any]],
                       metric: str) -> dict[str, Any] | None:
    """Paired-seed bootstrap of (arm_a's behavioral effect) vs
    (arm_b's behavioral effect) on a metric. Both lists must be sorted
    by seed and contain the same set of seeds."""
    a_diffs = per_seed_diffs(arm_a_seeds, metric)
    b_diffs = per_seed_diffs(arm_b_seeds, metric)
    a_lookup = {s: d for s, d in a_diffs}
    b_lookup = {s: d for s, d in b_diffs}
    common = sorted(set(a_lookup) & set(b_lookup))
    if len(common) < 2:
        return None
    a_paired = [a_lookup[s] for s in common]
    b_paired = [b_lookup[s] for s in common]
    mean, lo, hi, p = boot_diff_paired(a_paired, b_paired)
    return {
        "n_seeds_paired": len(common),
        "mean_diff_of_diffs": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_a_gt_b": p,
    }


def analyze_dir(in_dir: Path, threshold: float) -> dict[str, Any]:
    """Run the full pipeline for one heard-strength threshold."""
    random.seed(42)

    files = sorted(in_dir.glob("*_seed*.jsonl"))
    files = [p for p in files if not p.name.startswith("_")]
    print(f"[behavior] threshold={threshold}: scanning {len(files)} runs", file=sys.stderr)

    # arm -> list of per-seed dicts (sorted by seed)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for i, p in enumerate(files):
        if i % 20 == 0:
            print(f"[behavior]   ({i}/{len(files)}) {p.name}", file=sys.stderr)
        try:
            ps = per_seed_metrics(p, threshold)
        except Exception as e:
            print(f"[behavior]   ERROR reading {p.name}: {e!r}", file=sys.stderr)
            continue
        if ps["arm"] in ARM_ORDER:
            by_arm[ps["arm"]].append(ps)

    for arm in by_arm:
        by_arm[arm].sort(key=lambda d: d["seed"] or 0)

    # Per-arm bootstrap of (heard - no_heard) effects.
    per_arm_results: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        runs = by_arm.get(arm, [])
        if not runs:
            continue
        per_arm_results[arm] = {
            "n_runs": len(runs),
            "n_heard_total": sum(d["n_heard"] for d in runs),
            "n_no_heard_total": sum(d["n_no_heard"] for d in runs),
            "metrics": {},
        }
        for m in METRICS:
            r = bootstrap_arm(runs, m)
            if r is not None:
                per_arm_results[arm]["metrics"][m] = r

    # Cross-arm pairwise comparisons (diff-of-diffs).
    pairwise_results: list[dict[str, Any]] = []
    for a, b, desc in PAIRWISE:
        a_runs = by_arm.get(a, [])
        b_runs = by_arm.get(b, [])
        if not a_runs or not b_runs:
            continue
        for m in METRICS:
            cmp = cross_arm_compare(a_runs, b_runs, m)
            if cmp is not None:
                pairwise_results.append({
                    "a": a, "b": b, "desc": desc, "metric": m, **cmp,
                })

    return {
        "threshold": threshold,
        "per_arm": per_arm_results,
        "pairwise": pairwise_results,
    }


def _fmt_ci(d: dict[str, Any] | None) -> str:
    if d is None:
        return "(no data)"
    if "mean_diff" in d:
        return f"{d['mean_diff']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  P>0={d['p_positive']:.0%}  n={d['n_seeds']}"
    return f"{d['mean_diff_of_diffs']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  P(a>b)={d['p_a_gt_b']:.0%}  n={d['n_seeds_paired']}"


def print_summary(results_by_threshold: list[dict[str, Any]]) -> None:
    """Pretty-print headline tables to stdout."""
    for results in results_by_threshold:
        thr = results["threshold"]
        print()
        print("=" * 80)
        print(f"BEHAVIORAL RECEIVER-RESPONSE -- threshold > {thr}")
        print("=" * 80)
        print()
        print("Per-arm: P(metric | heard) - P(metric | no-heard); positive = heard helps")
        print("-" * 80)
        # Headline metrics in order.
        headline = ["flee_3tick", "predator_dist_delta_t3", "survived_next_25",
                    "energy_delta_next_10", "food_eaten_next_10",
                    "reproduced_next_50"]
        for m in headline:
            print(f"\n  metric: {m}")
            for arm in ARM_ORDER:
                arm_data = results["per_arm"].get(arm, {})
                md = arm_data.get("metrics", {}).get(m)
                if md is None:
                    print(f"    {arm}: (no data)")
                    continue
                print(f"    {arm}: {_fmt_ci(md)}")

        print()
        print("=" * 80)
        print(f"PAIRWISE diff-of-diffs (positive => arm A's heard-effect > arm B's)")
        print("=" * 80)
        # Group pairwise by metric.
        by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for p in results["pairwise"]:
            by_metric[p["metric"]].append(p)
        for m in headline:
            if m not in by_metric:
                continue
            print(f"\n  metric: {m}")
            for p in by_metric[m]:
                tag = f"{p['a']} vs {p['b']}"
                print(f"    {tag:7s}: {p['mean_diff_of_diffs']:+.4f} [{p['ci_lo']:+.4f}, {p['ci_hi']:+.4f}]  P({p['a']}>{p['b']})={p['p_a_gt_b']:.0%}  n={p['n_seeds_paired']}  ({p['desc']})")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="no_free_signal.experiments.behavior_analysis")
    ap.add_argument("--in", dest="in_dir", default="results_15k_behavioral")
    ap.add_argument("--out-json", default="results_15k_behavioral/behavior_results.json")
    ap.add_argument("--thresholds", default=",".join(str(t) for t in THRESHOLDS),
                    help="comma-separated heard-strength thresholds")
    args = ap.parse_args(argv)

    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        print(f"input dir not found: {in_dir}", file=sys.stderr)
        return 1

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    results = [analyze_dir(in_dir, t) for t in thresholds]
    print_summary(results)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[behavior] wrote {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
