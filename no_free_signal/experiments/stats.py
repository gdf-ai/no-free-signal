"""Offline statistics over completed run sets.

Bootstrap CIs, pairwise bootstrap differences with P(A>B), null-shuffled
alignment z-scores. Per-arm aggregations of pop_AUC, ticks_completed,
extinction rate, etc. Pure functions on already-loaded run dicts (the
shape produced by ``plot._load_run``).

Both ``plot.py`` (figures) and ``report.py`` (inline stats text) import
from this module so the numbers in the figures and the markdown body
agree by construction.
"""
from __future__ import annotations

import random
import statistics
from typing import Any, Iterable

import numpy as np

ARM_ORDER = ("A", "B", "C", "D", "E", "F", "G")

_RNG_SEED = 42


def _seed_rng() -> None:
    random.seed(_RNG_SEED)
    np.random.seed(_RNG_SEED)


def boot_ci(values: list[float], n: int = 10_000, ci: float = 95) -> tuple[float, float, float]:
    """Bootstrap mean and (1-ci/100)-symmetric CI."""
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


def boot_diff(a_vals: list[float], b_vals: list[float], n: int = 10_000) -> tuple[float, float, float, float]:
    """Bootstrap difference of means a - b. Returns (mean, lo, hi, P(a>b))."""
    if not a_vals or not b_vals:
        return 0.0, 0.0, 0.0, 0.5
    diffs = []
    for _ in range(n):
        a_s = sum(random.choice(a_vals) for _ in a_vals) / len(a_vals)
        b_s = sum(random.choice(b_vals) for _ in b_vals) / len(b_vals)
        diffs.append(a_s - b_s)
    diffs.sort()
    p_pos = sum(1 for x in diffs if x > 0) / len(diffs)
    lo = diffs[int(n * 0.025)]
    hi = diffs[int(n * 0.975)]
    return statistics.mean(diffs), lo, hi, p_pos


def boot_diff_paired(a_vals: list[float], b_vals: list[float],
                     n: int = 10_000) -> tuple[float, float, float, float]:
    """Paired-seed bootstrap of a - b. Resamples per-seed differences
    rather than independent draws from each arm. Returns (mean, lo, hi,
    P(a>b)). Requires len(a_vals) == len(b_vals) and the i-th element of
    each list to refer to the same seed; otherwise falls back to the
    independent boot_diff."""
    if not a_vals or not b_vals or len(a_vals) != len(b_vals):
        return boot_diff(a_vals, b_vals, n=n)
    diffs_per_seed = [a - b for a, b in zip(a_vals, b_vals)]
    boots = []
    for _ in range(n):
        s = [random.choice(diffs_per_seed) for _ in diffs_per_seed]
        boots.append(sum(s) / len(s))
    boots.sort()
    p_pos = sum(1 for x in boots if x > 0) / len(boots)
    lo = boots[int(n * 0.025)]
    hi = boots[int(n * 0.975)]
    return statistics.mean(boots), lo, hi, p_pos


def cohens_d(a_vals: list[float], b_vals: list[float]) -> float:
    """Cohen's d effect size for the difference of means a - b. Uses
    pooled standard deviation. Conventional thresholds: 0.2 small,
    0.5 medium, 0.8 large."""
    if len(a_vals) < 2 or len(b_vals) < 2:
        return 0.0
    ma = statistics.mean(a_vals)
    mb = statistics.mean(b_vals)
    va = statistics.variance(a_vals)
    vb = statistics.variance(b_vals)
    na = len(a_vals)
    nb = len(b_vals)
    pooled = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    if pooled <= 0:
        return 0.0
    return (ma - mb) / (pooled ** 0.5)


def _pop_auc(snapshots: list[dict[str, Any]]) -> float:
    if len(snapshots) < 2:
        return 0.0
    ps = [s.get("n_creatures", 0) for s in snapshots]
    ts = [s.get("tick", 0) for s in snapshots]
    return sum((ps[i] + ps[i + 1]) / 2 * (ts[i + 1] - ts[i]) for i in range(len(ps) - 1))


def _seed_of(run: dict[str, Any]) -> int | None:
    h = run.get("header") or {}
    if "seed" in h:
        try:
            return int(h["seed"])
        except (TypeError, ValueError):
            return None
    return None


def _mean_population(snapshots: list[dict[str, Any]]) -> float:
    if not snapshots:
        return 0.0
    pops = [s.get("n_creatures", 0) for s in snapshots]
    return statistics.mean(pops) if pops else 0.0


def compute_arm_data(runs_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """For each arm with completed runs, return per-arm stats and the raw
    per-seed values needed for bootstrap comparisons.

    Per-seed lists (`pop_aucs`, `ticks_completed`, `final_pops`, `emits`,
    `mean_pops`, `seeds`) are sorted by seed number so element i in any
    list refers to the same seed across all lists for that arm — and,
    because seeds are matched across arms in the harness, also across
    arms. That alignment is what enables `boot_diff_paired`."""
    out: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        runs = runs_by_arm.get(arm, [])
        completed = [r for r in runs if r.get("summary") is not None and r.get("snapshots")]
        if not completed:
            continue
        # Sort by seed so paired-seed analyses can index by position.
        completed.sort(key=lambda r: (_seed_of(r) is None, _seed_of(r) or 0))
        seeds = [_seed_of(r) for r in completed]
        ticks = [int(r["summary"].get("ticks_completed", 0)) for r in completed]
        extinct = [bool(r["summary"].get("extinction", False)) for r in completed]
        final_pops = [int(r["snapshots"][-1].get("n_creatures", 0)) for r in completed]
        pop_aucs = [_pop_auc(r["snapshots"]) for r in completed]
        api_calls = [int(r["summary"].get("llm_api_calls", 0)) for r in completed]
        emits = [int(r["summary"].get("llm_emits_with_wave", 0)) for r in completed]
        cost = [float(r["summary"].get("llm_cost_usd", 0.0)) for r in completed]
        mean_pops = [_mean_population(r["snapshots"]) for r in completed]
        out[arm] = {
            "n": len(completed),
            "seeds": seeds,
            "n_extinct": sum(extinct),
            "extinct_pct": 100 * sum(extinct) / len(extinct) if extinct else 0.0,
            "ticks_completed": ticks,
            "pop_aucs": pop_aucs,
            "final_pops": final_pops,
            "api_calls": api_calls,
            "emits": emits,
            "cost_usd": cost,
            "mean_pops": mean_pops,
            "extinct_flags": extinct,
            "mean_ticks": statistics.mean(ticks) if ticks else 0,
            "median_ticks": statistics.median(ticks) if ticks else 0,
            "mean_pop_auc": statistics.mean(pop_aucs) if pop_aucs else 0,
            "mean_final_pop": statistics.mean(final_pops) if final_pops else 0,
            "mean_calls": statistics.mean(api_calls) if api_calls else 0,
            "mean_emits": statistics.mean(emits) if emits else 0,
            "mean_cost": statistics.mean(cost) if cost else 0,
        }
    return out


def compute_emission_rates(arm_data: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-arm emission cadence summary. Reports total emits per run,
    total ticks survived per run, mean population while alive, and the
    composite per-creature-per-tick rate.

    Necessary because raw `llm_emits_with_wave` differs across arms
    largely because populations live for different durations and reach
    different sizes. Per-capita-per-tick rate normalizes that out and
    is the actual cadence-matching test for the G control."""
    out: dict[str, dict[str, float]] = {}
    for arm, d in arm_data.items():
        emits = d.get("emits", []) or []
        ticks = d.get("ticks_completed", []) or []
        mean_pops = d.get("mean_pops", []) or []
        if not emits or not ticks or not mean_pops:
            continue
        per_run_rates: list[float] = []
        for e, t, p in zip(emits, ticks, mean_pops):
            denom = t * p
            if denom > 0:
                per_run_rates.append(e / denom)
        out[arm] = {
            "mean_emits_per_run": statistics.mean(emits) if emits else 0.0,
            "mean_ticks_per_run": statistics.mean(ticks) if ticks else 0.0,
            "mean_pop": statistics.mean(mean_pops) if mean_pops else 0.0,
            "mean_per_capita_per_tick": (statistics.mean(per_run_rates)
                                         if per_run_rates else 0.0),
            "stdev_per_capita_per_tick": (statistics.stdev(per_run_rates)
                                          if len(per_run_rates) > 1 else 0.0),
            "n_runs": len(per_run_rates),
        }
    return out


def null_z_alignment(runs: list[dict[str, Any]], n_shuffles: int = 100) -> dict[str, float]:
    """Compute null-adjusted alignment z-score for one arm.

    Reads the ``mean_audio_attention`` and ``llm_emit_mean`` 8-vectors
    from each completed run's last snapshot. Builds a null distribution
    by shuffling the LLM emit vector ``n_shuffles`` times per run and
    computing cos against (un-shuffled) audio attention. The z-score is
    (terminal_cos_mean - null_mean) / null_std. Near-zero means observed
    alignment is no better than random bin-permutation."""
    completed = [r for r in runs if r.get("summary") is not None]
    init_coses, term_coses, null_distribution = [], [], []
    for r in completed:
        if not r["snapshots"]:
            continue
        first = r["snapshots"][0]
        last = r["snapshots"][-1]
        ic = float(first.get("cos_attention_vs_llm", 0.0))
        tc = float(last.get("cos_attention_vs_llm", 0.0))
        init_coses.append(ic)
        term_coses.append(tc)
        att = last.get("mean_audio_attention", [])
        emit = last.get("llm_emit_mean", [])
        if att and emit and len(att) == 8 and len(emit) == 8 and any(emit):
            a = np.array(att, dtype=float)
            e = np.array(emit, dtype=float)
            for _ in range(n_shuffles):
                shuf = e.copy()
                np.random.shuffle(shuf)
                den = np.linalg.norm(a) * np.linalg.norm(shuf)
                if den > 1e-9:
                    null_distribution.append(float(a @ shuf) / den)
    if not term_coses:
        return {"init_cos": 0.0, "term_cos": 0.0, "delta": 0.0,
                "null_mean": 0.0, "null_std": 0.0, "z": 0.0, "n_null": 0}
    ic_m = statistics.mean(init_coses)
    tc_m = statistics.mean(term_coses)
    if null_distribution:
        nl_m = statistics.mean(null_distribution)
        nl_s = statistics.stdev(null_distribution) if len(null_distribution) > 1 else 0.001
        z = (tc_m - nl_m) / nl_s if nl_s > 0 else 0.0
    else:
        nl_m = 0.0
        nl_s = 0.0
        z = 0.0
    return {
        "init_cos": ic_m, "term_cos": tc_m, "delta": tc_m - ic_m,
        "null_mean": nl_m, "null_std": nl_s, "z": z,
        "n_null": len(null_distribution),
    }


def arm_alignment_stats(runs_by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    """One null_z_alignment dict per arm. Skips no-LLM arms (returns
    zeros) since they have no emit vectors to compare against."""
    out = {}
    for arm in ARM_ORDER:
        runs = runs_by_arm.get(arm, [])
        if not runs:
            continue
        out[arm] = null_z_alignment(runs)
    return out


PAIRWISE_COMPARISONS = (
    ("D", "B", "evolvable vs frozen substrate"),
    ("B", "A", "frozen-substrate LLM vs mute"),
    ("D", "F", "full LLM vs replay-randomized"),
    ("D", "E", "full LLM vs scrambled bins"),
    ("D", "A", "full stack vs mute"),
    ("D", "C", "any emission vs no-emitter"),
    ("D", "G", "LLM vs cadence-targeted noise"),
    ("G", "C", "noise emission vs no-emitter"),
    ("F", "G", "replayed LLM vs uniform noise"),
)


def _align_paired(a_arm: dict[str, Any], b_arm: dict[str, Any],
                  metric_key: str) -> tuple[list[float], list[float]]:
    """Return per-seed-aligned lists of metric values for the two arms,
    keeping only seeds present in both arms. Order is by seed."""
    a_seeds = a_arm.get("seeds") or []
    b_seeds = b_arm.get("seeds") or []
    a_vals_full = a_arm.get(metric_key) or []
    b_vals_full = b_arm.get(metric_key) or []
    if (not a_seeds or not b_seeds
            or len(a_seeds) != len(a_vals_full)
            or len(b_seeds) != len(b_vals_full)):
        return [], []
    b_lookup = {s: v for s, v in zip(b_seeds, b_vals_full) if s is not None}
    a_paired: list[float] = []
    b_paired: list[float] = []
    for s, v in zip(a_seeds, a_vals_full):
        if s is None:
            continue
        if s in b_lookup:
            a_paired.append(v)
            b_paired.append(b_lookup[s])
    return a_paired, b_paired


def pairwise_results(arm_data: dict[str, dict[str, Any]],
                     metric_key: str = "pop_aucs",
                     paired: bool = True) -> list[dict[str, Any]]:
    """Run bootstrap difference of means for each pre-defined pair on the
    given metric. By default uses the paired-seed bootstrap (seeds are
    matched across arms in this experiment); pass `paired=False` for the
    independent bootstrap. Returns dicts with mean diff, CI, P(A>B), and
    Cohen's d."""
    _seed_rng()
    out = []
    for a, b, desc in PAIRWISE_COMPARISONS:
        if a not in arm_data or b not in arm_data:
            continue
        if paired:
            a_vals, b_vals = _align_paired(arm_data[a], arm_data[b], metric_key)
            if not a_vals or not b_vals:
                a_vals = arm_data[a].get(metric_key, [])
                b_vals = arm_data[b].get(metric_key, [])
                m, lo, hi, p = boot_diff(a_vals, b_vals)
                method = "independent (fallback)"
            else:
                m, lo, hi, p = boot_diff_paired(a_vals, b_vals)
                method = "paired"
        else:
            a_vals = arm_data[a].get(metric_key, [])
            b_vals = arm_data[b].get(metric_key, [])
            if not a_vals or not b_vals:
                continue
            m, lo, hi, p = boot_diff(a_vals, b_vals)
            method = "independent"
        d_eff = cohens_d(a_vals, b_vals)
        out.append({
            "a": a, "b": b, "desc": desc,
            "mean_diff": m, "lo": lo, "hi": hi,
            "p_a_gt_b": p,
            "cohens_d": d_eff,
            "method": method,
            "metric": metric_key,
        })
    return out
