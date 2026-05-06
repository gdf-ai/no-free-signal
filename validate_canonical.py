"""Cadence audit + completeness gate for an experiment dataset.

Usage:  python validate_canonical.py <dataset_dir>

Hard pass criteria (exits non-zero if any fail; the report step should
not run unless this exits 0):
  - 140 files (7 arms x 20 seeds)
  - 0 stubs (<1 KB)
  - 20/20 per arm, all 20 seeds present
  - summary record in every file
  - For LLM-active arms (B/D/E/F): mean skip ratio < 20%
    (skip_ratio = skipped_refreshes / (skipped_refreshes + llm_api_calls)).
    A high skip ratio indicates the cadence-protection rate limiter or
    Bedrock latency is silently collapsing calls/tick.

Soft signals (printed for inspection, not gating):
  - per-arm mean calls/tick, emits/creature-tick
  - per-arm 429s and throttle retries (should generally be 0; sustained
    >0 means either the rate limit is too high or Bedrock is squeezing).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def tail_summary(p: Path):
    size = p.stat().st_size
    if size == 0:
        return None
    with p.open("rb") as f:
        f.seek(max(0, size - 16384))
        tail = f.read().decode("utf-8", errors="ignore")
    for line in reversed(tail.split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("kind") == "summary":
            return obj
    return None


def main() -> int:
    dataset_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results_15k_behavioral")
    if not dataset_dir.is_dir():
        print(f"ERROR: dataset dir not found: {dataset_dir}")
        return 2

    print("=" * 80)
    print(f"CADENCE AUDIT + COMPLETENESS — {dataset_dir}")
    print("=" * 80)

    files = sorted(dataset_dir.glob("*_seed*.jsonl"))
    n_files = len(files)
    stubs = [p for p in files if p.stat().st_size < 1024]
    n_stubs = len(stubs)

    per_arm: dict[str, list] = {}
    no_summary: list[str] = []
    sizes: list[int] = []
    for p in files:
        arm = p.name[0]
        seed = int(p.stem.split("seed")[1])
        s = tail_summary(p)
        sizes.append(p.stat().st_size)
        if s is None:
            no_summary.append(p.name)
            continue
        per_arm.setdefault(arm, []).append((seed, s))

    print(f"\nfiles:           {n_files}")
    print(f"stubs (<1KB):    {n_stubs}")
    print(f"no summary:      {len(no_summary)}")
    if no_summary:
        print(f"  first 10: {no_summary[:10]}")

    print("\nper-arm completeness (target 20/20):")
    completeness_ok = True
    for arm in "ABCDEFG":
        runs = per_arm.get(arm, [])
        n = len(runs)
        seeds_present = sorted(s for s, _ in runs)
        missing = [i for i in range(20) if i not in seeds_present]
        if n == 20 and not missing:
            flag = "OK"
        else:
            flag = f"MISSING {missing}"
            completeness_ok = False
        print(f"  {arm}: {n}/20  {flag}")

    print("\n" + "=" * 80)
    print("CADENCE AUDIT TABLE — distributions")
    print("=" * 80)
    print(f"{'arm':3s} {'n':>3s} {'mean_skip%':>10s} {'med_skip%':>9s} "
          f"{'max_skip%':>9s} {'>30%':>5s} {'>50%':>5s} "
          f"{'calls/tick':>10s} {'emits/c/t':>10s} {'429s':>6s} {'ext%':>5s}")
    print("-" * 95)

    cadence_ok = True
    high_skip_arms: list[str] = []
    nonzero_throttle_arms: list[str] = []
    outliers: dict[str, list[tuple[int, float]]] = {}  # arm -> [(seed, ratio)]
    per_arm_metrics: dict[str, dict] = {}
    for arm in "ABCDEFG":
        runs = per_arm.get(arm, [])
        if not runs:
            continue
        summaries = [s for _, s in runs]
        seeds_for_arm = [seed for seed, _ in runs]
        n = len(summaries)
        ticks = [s.get("ticks_completed", 0) or 0 for s in summaries]
        calls = [s.get("llm_api_calls", 0) or 0 for s in summaries]
        emits = [s.get("llm_emits_with_wave", 0) or 0 for s in summaries]
        skips = [s.get("skipped_refreshes", 0) or 0 for s in summaries]
        t429s = [s.get("throttled_429s", 0) or 0 for s in summaries]
        ext_flags = [bool(s.get("extinction")) for s in summaries]
        extinct = sum(1 for f in ext_flags if f) / n * 100

        # Per-run skip ratio for distribution stats.
        per_run_skip = []
        for c, sk in zip(calls, skips):
            attempts = c + sk
            per_run_skip.append((sk / attempts * 100) if attempts else 0.0)
        mean_calls = statistics.mean(calls)
        mean_emits = statistics.mean(emits)
        mean_ticks = statistics.mean(ticks) or 1
        calls_per_tick = mean_calls / mean_ticks if mean_ticks else 0
        emits_per_tick = mean_emits / mean_ticks if mean_ticks else 0
        emits_per_creature_per_tick = emits_per_tick / 25.0
        sum_429 = sum(t429s)

        if sum_429 > 0:
            nonzero_throttle_arms.append(arm)

        # Distribution stats per arm.
        mean_skip = statistics.mean(per_run_skip)
        med_skip = statistics.median(per_run_skip)
        max_skip = max(per_run_skip)
        n_over_30 = sum(1 for x in per_run_skip if x > 30.0)
        n_over_50 = sum(1 for x in per_run_skip if x > 50.0)

        # Outlier seeds (>30% skip).
        outliers[arm] = sorted(
            [(seed, ratio) for seed, ratio in zip(seeds_for_arm, per_run_skip)
             if ratio > 30.0],
            key=lambda x: -x[1],
        )

        # Hard cadence criterion: weighted-mean skip ratio for LLM arms.
        if arm in ("B", "D", "E", "F") and mean_skip > 20.0:
            high_skip_arms.append(f"{arm}({mean_skip:.1f}%)")
            cadence_ok = False

        per_arm_metrics[arm] = {
            "n": n,
            "ticks_completed": ticks,
            "calls": calls,
            "skips": skips,
            "extinction": ext_flags,
            "skip_ratios": per_run_skip,
            "seeds": seeds_for_arm,
        }

        print(f"{arm:3s} {n:>3d} {mean_skip:>9.1f}% {med_skip:>8.1f}% "
              f"{max_skip:>8.1f}% {n_over_30:>5d} {n_over_50:>5d} "
              f"{calls_per_tick:>10.4f} {emits_per_creature_per_tick:>10.4f} "
              f"{sum_429:>6d} {extinct:>4.0f}%")

    # Outlier list — by arm.
    any_outliers = any(outliers.get(a) for a in "BDEF")
    if any_outliers:
        print("\nHigh-skip outliers (>30%) by LLM arm:")
        for arm in "BDEF":
            ol = outliers.get(arm, [])
            if ol:
                seeds_str = ", ".join(f"seed{s}({r:.0f}%)" for s, r in ol[:10])
                print(f"  {arm}: {seeds_str}")

    # ----- sensitivity analysis: does the fitness ranking hold? -----
    print("\n" + "=" * 80)
    print("SENSITIVITY: does fitness ranking hold under high-skip exclusion?")
    print("=" * 80)

    def _ranking(filter_fn) -> list[tuple[str, float, int]]:
        """Compute (arm, mean_ticks, n_kept) ordered descending by mean_ticks
        after applying `filter_fn(skip_ratio)` per run (True = keep)."""
        out = []
        for arm in "ABCDEFG":
            m = per_arm_metrics.get(arm)
            if not m:
                continue
            kept = [(t, sk_r) for t, sk_r in zip(m["ticks_completed"], m["skip_ratios"])
                    if filter_fn(sk_r)]
            if not kept:
                continue
            ticks_kept = [t for t, _ in kept]
            out.append((arm, statistics.mean(ticks_kept), len(ticks_kept)))
        return sorted(out, key=lambda x: -x[1])

    scenarios = [
        ("all runs",      lambda r: True),
        ("exclude >50%",  lambda r: r <= 50.0),
        ("exclude >30%",  lambda r: r <= 30.0),
    ]
    rankings = []
    for name, fn in scenarios:
        rk = _ranking(fn)
        rk_str = " > ".join(f"{a}({m:.0f},n={n})" for a, m, n in rk)
        print(f"  {name:18s}: {rk_str}")
        rankings.append([a for a, _, _ in rk])

    # Robustness check: are the top-3 and bottom-3 stable across scenarios?
    if rankings:
        ranking_robust = all(rankings[0] == r for r in rankings[1:])
        if ranking_robust:
            print("  =>ranking is IDENTICAL across all sensitivity scenarios.")
        else:
            print("  =>WARNING: ranking changes when high-skip runs are excluded.")
            print("    Disclose high-skip outliers in the report and re-run any "
                  "arm/seed materially affected.")

    # ----- pass/fail summary -----
    print("\n" + "=" * 80)
    print("HARD PASS CRITERIA")
    print("=" * 80)
    fail_reasons: list[str] = []
    if n_files != 140:
        fail_reasons.append(f"file count {n_files} != 140")
    if n_stubs > 0:
        fail_reasons.append(f"{n_stubs} stubs present (must be 0)")
    if not completeness_ok:
        fail_reasons.append("not all arms have 20/20 seeds with summary")
    if no_summary:
        fail_reasons.append(f"{len(no_summary)} files lack a summary record")
    if not cadence_ok:
        fail_reasons.append(
            f"high skip ratio (>20%) on LLM-active arms: "
            f"{', '.join(high_skip_arms)} — cadence collapsed silently"
        )

    if fail_reasons:
        print("FAILED:")
        for r in fail_reasons:
            print(f"  - {r}")
        print("\nThis dataset is NOT cleared for the report. Do not generate "
              "REPORT.md until these are resolved.")
        return 1

    print("PASSED — all completeness and cadence criteria.")
    if nonzero_throttle_arms:
        print(f"NOTE: throttle activity (429s/retries) seen in arms: "
              f"{', '.join(nonzero_throttle_arms)}. The retries kept the "
              f"run going but if 429s keep climbing on future runs, lower "
              f"NFS_BEDROCK_RPS below 13.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
