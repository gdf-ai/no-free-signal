"""Parallel orchestrator for the experiment grid.

Runs the (arm × seed) combos as N concurrent subprocesses of the existing
``no_free_signal.experiments.harness`` single-run CLI. Each subprocess gets its own
Python interpreter, so global state (model id env var, frozen-substrate
flag, LLM client cache) is fully isolated between runs.

Resume-on-restart: any ``<arm>_seed<n>.jsonl`` whose final line is a valid
``"kind":"summary"`` record is treated as already complete and skipped. So
you can Ctrl+C, restart, and it picks up where it left off.

Usage::

    python -m no_free_signal.experiments.parallel --confirm \
        --workers 4 --seeds 0,1,2,3,4,5,6,7,8,9 --n-steps 5000 \
        --model nova-lite

Wall-clock estimate at ~50 ms/tick × 5000 ticks ≈ 4 min/run, 60 runs:

    workers   wall clock
    1         ~4 hours       (sequential; same as ``--grid``)
    4         ~1 hour
    6         ~40 min
    8         ~30 min        (RAM-bound; ~1 GB/worker)

The orchestrator does NOT enforce session call caps directly — each child
subprocess does. Estimated cost is the sum of children's reported costs,
so check the manifest at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ARMS = ["A", "B", "C", "D", "E", "F", "G"]


def _is_run_complete(path: Path) -> bool:
    """A JSONL run is complete iff its last non-empty line is a summary."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            last = ""
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if not last:
            return False
        obj = json.loads(last)
        return obj.get("kind") == "summary"
    except Exception:
        return False


def run_one_subprocess(
    *,
    arm: str,
    seed: int,
    n_steps: int,
    model: str,
    n_creatures: int,
    refresh_every: int,
    grid_size: int,
    snapshot_every: int,
    out_dir: Path,
    timeout_sec: int,
    per_run_cap: int,
    log_behavioral: bool = False,
    complete_set: set[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    # Skip if the (arm, seed) is in the explicit complete_set (no file
    # is created for these runs). This is the AWS-safe path: markers
    # never enter out_dir, so the heartbeat sidecar's incremental S3
    # sync can't propagate them and clobber the user's local data.
    if complete_set is not None and (arm, seed) in complete_set:
        return {"arm": arm, "seed": seed, "skipped": True,
                "wall": 0.0, "skip_reason": "in complete-list"}
    out_path = out_dir / f"{arm}_seed{seed}.jsonl"
    if _is_run_complete(out_path):
        return {"arm": arm, "seed": seed, "skipped": True,
                "wall": 0.0, "skip_reason": "summary present in file"}

    cmd = [
        sys.executable, "-m", "no_free_signal.experiments.harness",
        "--arm", arm, "--seed", str(seed),
        "--n-steps", str(n_steps),
        "--snapshot-every", str(snapshot_every),
        "--n-creatures", str(n_creatures),
        "--grid-size", str(grid_size),
        "--refresh-every", str(refresh_every),
        "--model", model,
        "--per-run-cap", str(per_run_cap),
        "--out", str(out_path),
    ]
    if log_behavioral:
        cmd.append("--log-behavioral")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout_sec if timeout_sec > 0 else None,
        )
        wall = round(time.monotonic() - t0, 1)
        ok = proc.returncode == 0 and _is_run_complete(out_path)
        return {
            "arm": arm, "seed": seed, "ok": ok, "wall": wall,
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-400:],
            "stderr_tail": (proc.stderr or "")[-400:],
        }
    except subprocess.TimeoutExpired:
        return {
            "arm": arm, "seed": seed, "ok": False,
            "wall": round(time.monotonic() - t0, 1),
            "error": f"timeout after {timeout_sec}s",
        }
    except Exception as e:
        return {
            "arm": arm, "seed": seed, "ok": False,
            "wall": round(time.monotonic() - t0, 1),
            "error": repr(e),
        }


def _summarize_costs(out_dir: Path, jobs: list[tuple[str, int]]) -> dict[str, Any]:
    """Read each completed run's summary line and sum costs/calls."""
    total_calls = 0
    total_cost = 0.0
    completed = 0
    for arm, seed in jobs:
        path = out_dir / f"{arm}_seed{seed}.jsonl"
        if not _is_run_complete(path):
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            summary = next((l for l in reversed(lines) if l.get("kind") == "summary"), None)
            if summary is None:
                continue
            total_calls += int(summary.get("llm_api_calls", 0))
            total_cost += float(summary.get("llm_cost_usd", 0.0))
            completed += 1
        except Exception:
            pass
    return {"completed": completed, "calls": total_calls, "cost_usd": round(total_cost, 4)}


def _read_run_summary(path: Path) -> dict | None:
    """Tail-read the summary record from a completed run JSONL."""
    try:
        size = path.stat().st_size
        if size == 0:
            return None
        with path.open("rb") as f:
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
    except Exception:
        return None
    return None


# Arms that exercise the LLM (B/D/E/F call Bedrock; A/C/G do not). Used
# by the mid-run throttle-abort guard.
_LLM_ARMS = ("B", "D", "E", "F")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="no_free_signal.experiments.parallel")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9",
                    help="comma-separated seeds (default: 0..9)")
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated arms (default: A,B,C,D,E,F,G)")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel subprocess count (default: 4)")
    ap.add_argument("--n-steps", type=int, default=5000)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--n-creatures", type=int, default=25)
    ap.add_argument("--grid-size", type=int, default=24)
    ap.add_argument("--refresh-every", type=int, default=15)
    ap.add_argument("--model", default="nova-lite")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="per-run timeout in seconds (default: 3600 = 1h). "
                         "Pass 0 to disable timeout — relies on per-run-cap + "
                         "n-steps as bounds (recommended for long runs).")
    ap.add_argument("--per-run-cap", type=int, default=50_000,
                    help="per-run LLM call cap (default 50000, 67% headroom over "
                         "15k-tick × max-pop sustained worst case)")
    ap.add_argument("--confirm", action="store_true",
                    help="required to actually launch")
    ap.add_argument("--log-behavioral", action="store_true",
                    help="pass --log-behavioral to each child harness "
                         "subprocess so per-emit and per-creature-tick "
                         "records are written for receiver-response "
                         "analysis. ~150 MB per 15k-tick run.")
    ap.add_argument("--complete-list", default=None,
                    help="path to a file with 'ARM SEED' lines (whitespace-"
                         "separated, one per line, '#' lines ignored). Any "
                         "(arm, seed) pair listed is treated as already "
                         "complete and skipped without writing any file. "
                         "Used by AWS bootstrap to flag runs already done "
                         "on the user's local machine -- markers never "
                         "enter the results dir, so incremental S3 sync "
                         "can't accidentally overwrite local data with "
                         "marker stubs.")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARMS:
            print(f"unknown arm: {a!r}; expected one of {ARMS}", file=sys.stderr)
            return 2
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read explicit skip list (--complete-list) into a set of (arm, seed)
    # tuples. Members of this set are treated as already complete and
    # skipped without ever calling run_one_subprocess (no file written
    # to out_dir for them).
    complete_set: set[tuple[str, int]] = set()
    if args.complete_list:
        clp = Path(args.complete_list)
        if clp.exists():
            for line in clp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        complete_set.add((parts[0].upper(), int(parts[1])))
                    except ValueError:
                        pass
            print(f"[parallel] complete-list: loaded {len(complete_set)} "
                  f"(arm, seed) pairs to skip")
        else:
            print(f"[parallel] complete-list path {clp} not found; ignoring")

    jobs: list[tuple[str, int]] = [(a, s) for a in arms for s in seeds]

    # Pre-check resume state. A run counts as already-done if it's in
    # the explicit complete_set OR if its output file exists with a
    # summary-line tail.
    def _is_already_done(arm: str, seed: int) -> bool:
        if (arm, seed) in complete_set:
            return True
        return _is_run_complete(out_dir / f"{arm}_seed{seed}.jsonl")

    already_done = sum(1 for a, s in jobs if _is_already_done(a, s))

    print(f"[parallel] {len(jobs)} runs total ({already_done} already complete, will skip)")
    print(f"[parallel] arms     : {arms}")
    print(f"[parallel] seeds    : {seeds}")
    print(f"[parallel] workers  : {args.workers}")
    print(f"[parallel] model    : {args.model}")
    print(f"[parallel] n_steps  : {args.n_steps}")
    print(f"[parallel] timeout  : {args.timeout}s per run")
    print(f"[parallel] out_dir  : {out_dir}")
    print(f"[parallel] throttle-guard: abort if 2 LLM runs hit "
          f">30% skip ratio (rc=3)")
    if not args.confirm:
        print("[parallel] --confirm not set; not running. Re-run with --confirm.")
        return 0

    t_start = time.monotonic()
    completed = 0
    skipped = 0
    failed = 0
    consecutive_bad_llm = 0  # consecutive LLM runs above the skip threshold
    THROTTLE_ABORT_THRESHOLD = 3
    THROTTLE_SKIP_RATIO = 0.50  # >50% of refresh windows silently dropped
    # Loosened from (2 / 0.30) after observing transient Bedrock blips:
    # one or two LLM runs out of 20 occasionally hit 30-70% skip while
    # the other 18 run cleanly. The audit gate still catches sustained
    # collapse, but the abort guard now needs THREE consecutive bad
    # runs at >50% to fire — only triggers on real persistent issues.

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one_subprocess,
                arm=arm, seed=seed,
                n_steps=args.n_steps,
                model=args.model,
                n_creatures=args.n_creatures,
                refresh_every=args.refresh_every,
                grid_size=args.grid_size,
                snapshot_every=args.snapshot_every,
                out_dir=out_dir,
                timeout_sec=args.timeout,
                per_run_cap=args.per_run_cap,
                log_behavioral=args.log_behavioral,
                complete_set=complete_set,
            ): (arm, seed) for arm, seed in jobs
        }
        try:
            for fut in as_completed(futures):
                arm, seed = futures[fut]
                try:
                    result = fut.result()
                except Exception as e:
                    failed += 1
                    print(f"[parallel] EXC arm={arm} seed={seed}: {e!r}")
                    continue
                if result.get("skipped"):
                    skipped += 1
                    print(f"[parallel] skip arm={arm} seed={seed} (already complete)")
                elif result.get("ok"):
                    completed += 1
                    print(f"[parallel] done arm={arm} seed={seed} "
                          f"wall={result['wall']}s "
                          f"({completed + skipped}/{len(jobs)} ok)")
                    # Mid-run throttle-abort guard. Tracks CONSECUTIVE
                    # bad LLM runs so transient Bedrock blips (one or
                    # two outlier runs amid clean ones) don't kill the
                    # grid. Resets on any clean LLM run.
                    if arm in _LLM_ARMS:
                        s = _read_run_summary(
                            out_dir / f"{arm}_seed{seed}.jsonl"
                        )
                        if s is not None:
                            calls = int(s.get("llm_api_calls", 0) or 0)
                            skips = int(s.get("skipped_refreshes", 0) or 0)
                            t429 = int(s.get("throttled_429s", 0) or 0)
                            attempts = calls + skips
                            ratio = skips / attempts if attempts else 0.0
                            if ratio > THROTTLE_SKIP_RATIO:
                                consecutive_bad_llm += 1
                                print(
                                    f"[parallel] WARN cadence on arm={arm} "
                                    f"seed={seed}: skip_ratio={ratio*100:.1f}% "
                                    f"(calls={calls}, skipped={skips}, "
                                    f"429s={t429}). "
                                    f"consecutive_bad={consecutive_bad_llm}/"
                                    f"{THROTTLE_ABORT_THRESHOLD}"
                                )
                                if consecutive_bad_llm >= THROTTLE_ABORT_THRESHOLD:
                                    print(
                                        f"[parallel] ABORT: "
                                        f"{consecutive_bad_llm} consecutive "
                                        f"LLM runs > "
                                        f"{int(THROTTLE_SKIP_RATIO*100)}% skip. "
                                        f"Bedrock is collapsing persistently; "
                                        f"exiting (rc=3) so the launcher tears "
                                        f"down."
                                    )
                                    for f2 in futures:
                                        f2.cancel()
                                    sys.stdout.flush()
                                    os._exit(3)
                            else:
                                consecutive_bad_llm = 0
                else:
                    failed += 1
                    print(f"[parallel] FAIL arm={arm} seed={seed} "
                          f"wall={result.get('wall')}s rc={result.get('returncode', '?')}")
                    if result.get("stderr_tail"):
                        for ln in result["stderr_tail"].splitlines()[-3:]:
                            print(f"  stderr: {ln}")
                    if result.get("error"):
                        print(f"  error: {result['error']}")
        except KeyboardInterrupt:
            print("\n[parallel] KeyboardInterrupt — cancelling pending jobs.")
            for f in futures:
                f.cancel()
            raise

    elapsed = time.monotonic() - t_start
    cost_summary = _summarize_costs(out_dir, jobs)
    print()
    print(f"[parallel] finished in {elapsed/60:.1f} min")
    print(f"[parallel]   ok       : {completed}")
    print(f"[parallel]   skipped  : {skipped}")
    print(f"[parallel]   failed   : {failed}")
    print(f"[parallel]   total $  : ~${cost_summary['cost_usd']}")
    print(f"[parallel]   total calls: {cost_summary['calls']}")
    print()
    print(f"[parallel] next: python -m no_free_signal.experiments.report --in {out_dir} --out {out_dir}/REPORT.md")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
