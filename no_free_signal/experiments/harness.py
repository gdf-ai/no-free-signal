"""Experiment runner: ``python -m no_free_signal.experiments.harness ...``

Single-arm, single-seed runs that dump JSONL incrementally so a crash mid-
run doesn't lose data. The aggregate driver (``run_grid``) loops over arms
and seeds with a budget guard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from no_free_signal.experiments.ablations import apply_arm
from no_free_signal.experiments.behavioral_logger import BehavioralLogger
from no_free_signal.experiments.llm_emit_logger import EmitLog
from no_free_signal.experiments.metrics import compute_snapshot
from no_free_signal.world import World
import no_free_signal.llm_controller as _llm_ctrl
from foresight.envs.unified_world import WorldConfig

ARMS = ["A", "B", "C", "D", "E", "F", "G"]

# Per-call USD by model id. Calibrated against AWS CloudWatch (2026-04-30):
#   Nova Lite measured 842 input + 113 output tokens/call avg → $0.0000776/call.
# The earlier 593-token estimate didn't account for social_signal,
# personal_events and overheard utterances that grow the prompt at runtime.
# Other rates extrapolated proportionally from AWS Bedrock published per-
# token pricing assuming similar prompt growth:
#   Nova Lite : $0.06 in / $0.24 out  → ~$0.0000776/call (verified)
#   Nova Micro: $0.035 in / $0.14 out → ~$0.0000452/call
#   Haiku 4.5 : $1.00 in / $5.00 out  → ~$0.001407/call
#   Haiku 3.5 : $0.80 in / $4.00 out  → ~$0.001125/call
#   Haiku 3   : $0.25 in / $1.25 out  → ~$0.000352/call
COST_PER_CALL = {
    "us.amazon.nova-lite-v1:0": 0.0000776,
    "us.amazon.nova-micro-v1:0": 0.0000452,
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": 0.001407,
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": 0.001125,
    "us.anthropic.claude-3-haiku-20240307-v1:0": 0.000352,
}

# Friendly aliases for the --model CLI flag. Bedrock inference-profile ids
# (the ``us.`` prefix routes through cross-region inference, which is
# required for several newer models). Note word order: claude-3-5-haiku,
# NOT claude-haiku-3-5.
MODEL_ALIASES = {
    "nova-lite":    "us.amazon.nova-lite-v1:0",
    "nova-micro":   "us.amazon.nova-micro-v1:0",
    "haiku":        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "haiku-4-5":    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "haiku-3-5":    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "haiku-3":      "us.anthropic.claude-3-haiku-20240307-v1:0",
}


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)

# Hard caps enforced by the harness, not by trust. ``SESSION_CALL_CAP`` is
# the upper bound across the entire grid run; the high-end cost estimate
# must come in below it or ``run_grid`` aborts. 1M calls at Nova Lite is
# ~$25 — generous headroom for any sane plan up to ~20 seeds × 10000 ticks.
# Override via ``--session-cap N`` when tightening for a specific run.
PER_RUN_CALL_CAP = 50_000
SESSION_CALL_CAP = 2_000_000

DEFAULT_MODEL = "us.amazon.nova-lite-v1:0"
DEFAULT_N_STEPS = 3000
DEFAULT_SNAPSHOT_EVERY = 50
DEFAULT_N_CREATURES = 25
# Hard cap on population during the experiment. Without this the world's
# default 50-creature ceiling kicks in and ~doubles LLM call rate when
# births outpace deaths in early generations.
DEFAULT_MAX_CREATURES = 30
DEFAULT_GRID_SIZE = 24
DEFAULT_REFRESH_EVERY = 15
# Generous per-run daily-limit override. Without this, the LLMController
# global default of 500/day silently no-ops most calls in the second arm,
# wrecking the experiment.
PER_RUN_DAILY_LIMIT_FLOOR = 50_000


def _cost_for_model(model_id: str) -> float:
    return COST_PER_CALL.get(model_id, COST_PER_CALL[DEFAULT_MODEL])


def _arm_uses_llm(arm: str) -> bool:
    return arm.upper() in ("B", "D", "E", "F")


def estimate_cost(
    arms: list[str], seeds: list[int], n_steps: int, n_creatures: int,
    refresh_every: int, model_id: str, max_creatures: int = DEFAULT_MAX_CREATURES,
) -> dict[str, Any]:
    """Print and return the projected total LLM cost for a grid run, as
    a low / mid / high range. No API calls are made.

    - low:  initial population (n_creatures) for the entire run.
    - mid:  population averaged at ~80% of max_creatures (realistic).
    - high: population pinned at max_creatures for the entire run.
    """
    llm_runs = sum(1 for arm in arms for _ in seeds if _arm_uses_llm(arm))
    per_call = _cost_for_model(model_id)

    def _calls_for_pop(pop: float) -> int:
        return int(llm_runs * (pop / max(1, refresh_every)) * n_steps)

    low_calls = _calls_for_pop(n_creatures)
    mid_calls = _calls_for_pop(0.8 * max_creatures)
    high_calls = _calls_for_pop(max_creatures)

    return {
        "model_id": model_id,
        "per_call_usd": per_call,
        "arms": arms,
        "seeds": seeds,
        "max_creatures": max_creatures,
        "llm_runs": llm_runs,
        "calls_low": low_calls,
        "calls_mid": mid_calls,
        "calls_high": high_calls,
        "usd_low": round(low_calls * per_call, 4),
        "usd_mid": round(mid_calls * per_call, 4),
        "usd_high": round(high_calls * per_call, 4),
    }


def run_one(
    *,
    arm: str,
    seed: int,
    out_path: Path,
    n_steps: int = DEFAULT_N_STEPS,
    snapshot_every: int = DEFAULT_SNAPSHOT_EVERY,
    n_creatures: int = DEFAULT_N_CREATURES,
    max_creatures: int = DEFAULT_MAX_CREATURES,
    grid_size: int = DEFAULT_GRID_SIZE,
    refresh_every: int = DEFAULT_REFRESH_EVERY,
    model_id: str = DEFAULT_MODEL,
    per_run_cap: int = PER_RUN_CALL_CAP,
    log_emit_events: bool = False,
    log_behavioral: bool = False,
    behavioral_state_log_every: int = 3,
) -> dict[str, Any]:
    """Run one (arm, seed) and stream JSONL snapshots to ``out_path``."""
    arm = arm.upper()
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    model_id = resolve_model(model_id)

    os.environ["NFS_BEDROCK_MODEL_ID"] = model_id
    # Raise the daily-call limit floor so a stale env var doesn't cap the
    # session at 500. The user can override NFS_LLM_DAILY_LIMIT explicitly
    # for tighter budgets.
    cur_limit = int(os.environ.get("NFS_LLM_DAILY_LIMIT", "0") or 0)
    if cur_limit < PER_RUN_DAILY_LIMIT_FLOOR:
        os.environ["NFS_LLM_DAILY_LIMIT"] = str(PER_RUN_DAILY_LIMIT_FLOOR)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    emit_log = EmitLog()
    world = World(
        seed=seed,
        n_creatures=n_creatures,
        grid_size=grid_size,
        enable_brains=True,
        llm_emit_logger=emit_log,
    )
    # Hard population cap for the experiment.
    world._cfg.max_creatures = max_creatures
    world._env.config.max_creatures = max_creatures
    # Snapshot the global LLM call counter at run start so we can compute
    # the *actual* API calls made by this run (not just successful
    # emissions). emit_log undercounts because many LLM responses don't
    # include a valid vocalize_wave field.
    api_calls_at_start = _llm_ctrl._today_count()
    # Instrumentation maps consulted by metrics.compute_snapshot.
    world._age_at_death = {}
    world._counters = {
        "vocalize_actions": 0,
        "vocalize_emits": 0,
        "per_cid_call_counts": {},
    }

    apply_arm(world, arm, seed, emit_log, refresh_every_ticks=refresh_every)

    # behavioral_logger is constructed AFTER the JSONL file handle is
    # opened (inside the `with out_path.open(...)` block below) so its
    # write_callback can stream finalized records straight to disk.
    # This bounds the logger's RAM footprint to only the in-flight
    # outcome window (~50 ticks of pending records, <1 MB).
    behavioral_logger = None

    if _arm_uses_llm(arm):
        # _compute_refresh_every_ticks inverts target_cps → refresh_every via
        #   ticks_per_call = 5 * n_creatures / target_cps
        # so to make each controller refresh every ``refresh_every`` ticks we
        # set target_cps = 5 * n_creatures / refresh_every. This is approximate
        # — population fluctuations at runtime change the live `n`, but per-
        # controller refresh stays close to the requested cadence.
        world.enable_auto_attach_llm(
            target_cps=max(0.05, 5.0 * n_creatures / max(1, refresh_every)),
        )

    t_start = time.monotonic()
    cap_hit = False
    extinct = False
    # Tick-rate cap. AWS Linux on c7i.* runs ticks ~5x faster than
    # Windows local (54 vs ~10 ticks/sec at small populations), which
    # outruns Bedrock latency: each LLM call takes ~1s, so on AWS
    # most refresh attempts skip because the previous call hasn't
    # returned yet. Cap matches local Windows pacing so each LLM
    # refresh window has time to complete. Override via the
    # NFS_TICK_RATE env var (per-process); 0 disables.
    try:
        _tick_rate_cap = float(os.environ.get("NFS_TICK_RATE", "0"))
    except ValueError:
        _tick_rate_cap = 0.0
    _min_tick_interval = (1.0 / _tick_rate_cap) if _tick_rate_cap > 0 else 0.0
    with out_path.open("w", encoding="utf-8") as f:
        # Header — run params for offline analysis.
        f.write(json.dumps({
            "kind": "header",
            "arm": arm,
            "seed": seed,
            "n_steps": n_steps,
            "snapshot_every": snapshot_every,
            "n_creatures": n_creatures,
            "grid_size": grid_size,
            "refresh_every": refresh_every,
            "model_id": model_id,
            "per_call_usd": _cost_for_model(model_id),
        }) + "\n")
        f.flush()

        # Now that the JSONL is open, set up the behavioral logger
        # with a streaming write callback. Each finalized record goes
        # straight to disk; nothing accumulates in RAM beyond the
        # outcome window. This is the fix for the previous OOM where
        # a 32-worker run filled 128 GB of RAM with in-memory event
        # buffers and crashed the EC2 instance.
        if log_behavioral:
            def _write_record(record: dict) -> None:
                f.write(json.dumps(record) + "\n")
                # NOTE: no flush per-record — kernel page cache
                # absorbs writes; we'd lose throughput flushing every
                # event. The OS flushes on its own cadence and the
                # incremental S3 sync syncs to S3 every 60s.
            behavioral_logger = BehavioralLogger(
                arm=arm, seed=seed,
                state_log_every=behavioral_state_log_every,
                write_callback=_write_record,
            )
            world.attach_behavioral_logger(behavioral_logger)

        for tick in range(n_steps):
            tick_t0 = time.monotonic()
            ages_before = {cid: c.age for cid, c in world._env.creatures.items()}
            result = world.step(n=1)
            # Track deaths for lifespan metric.
            for ev in result.get("events", []):
                if ev.get("kind") == "death":
                    cid = ev.get("creature_id")
                    age = ages_before.get(cid, 0)
                    if cid is not None:
                        world._age_at_death[cid] = (tick, age)
            # Tally per-tick vocalization counters.
            for cid, c in world._env.creatures.items():
                if c.vocalized_last_tick:
                    world._counters["vocalize_emits"] += 1
                    world._counters["per_cid_call_counts"][cid] = (
                        world._counters["per_cid_call_counts"].get(cid, 0) + 1
                    )
            # Snapshot.
            if tick % snapshot_every == 0:
                api_calls_this_run = _llm_ctrl._today_count() - api_calls_at_start
                snap = compute_snapshot(world, emit_log, tick)
                snap["kind"] = "snapshot"
                snap["arm"] = arm
                snap["seed"] = seed
                snap["llm_api_calls"] = api_calls_this_run
                snap["llm_emits_with_wave"] = emit_log.total_calls()
                snap["llm_cost_usd"] = round(
                    api_calls_this_run * _cost_for_model(model_id), 6,
                )
                f.write(json.dumps(snap) + "\n")
                f.flush()
            # Tick-rate cap: pace the loop so on faster hardware the
            # simulation doesn't outrun Bedrock. No-op when the cap is
            # disabled (NFS_TICK_RATE=0 / unset) or when this tick
            # already took longer than the budget (large populations
            # naturally pace themselves).
            if _min_tick_interval > 0:
                spent = time.monotonic() - tick_t0
                remaining = _min_tick_interval - spent
                if remaining > 0:
                    time.sleep(remaining)
            # Stop if extinction or budget cap (use the *real* API counter,
            # not emit_log which only counts successful structured emits).
            if not world._env.creatures:
                extinct = True
                break
            api_calls_this_run = _llm_ctrl._today_count() - api_calls_at_start
            if api_calls_this_run >= per_run_cap:
                cap_hit = True
                break

        # Optional per-emission event dump for behavioral analysis.
        # Off by default; the records are kind="emit" and contain
        # (tick, creature_id, wave, amp). Receiver-response analyses
        # can join these against per-tick action records (not yet
        # logged — separate instrumentation work).
        if log_emit_events:
            n_events = 0
            for ev_tick, cid, wave, amp in emit_log.snapshot():
                f.write(json.dumps({
                    "kind": "emit",
                    "tick": int(ev_tick),
                    "creature_id": int(cid),
                    "wave": list(wave),
                    "amp": float(amp),
                }) + "\n")
                n_events += 1
            f.flush()
            print(f"[harness] dumped {n_events} kind=emit records "
                  f"(--log-emit-events)")

        # Behavioral logger: flush any in-flight pending records
        # whose outcome windows hadn't yet closed. Records were
        # already streamed to `f` via the write_callback during the
        # run, so we just flush the file and confirm zero leftover
        # in-memory state.
        if behavioral_logger is not None:
            try:
                behavioral_logger.finalize_remaining(
                    env=world._env, tick=int(world._env.steps),
                )
            except Exception as exc:
                print(f"[harness] behavioral_logger.finalize_remaining "
                      f"failed: {exc!r}")
            f.flush()
            # Sanity: when streaming via write_callback, the in-memory
            # lists must stay empty. Print a warning if not (would
            # indicate a regression where some path bypassed the
            # callback).
            n_leftover = (len(behavioral_logger.events)
                          + len(behavioral_logger.creature_ticks))
            if n_leftover > 0:
                print(f"[harness] WARNING: {n_leftover} behavioral "
                      f"records leaked to in-memory buffer (should "
                      f"have been streamed). Flushing now.")
                for record in behavioral_logger.events:
                    f.write(json.dumps(record) + "\n")
                for record in behavioral_logger.creature_ticks:
                    f.write(json.dumps(record) + "\n")
                f.flush()

        # Final summary.
        final_api_calls = _llm_ctrl._today_count() - api_calls_at_start
        final_emits = emit_log.total_calls()
        # Cadence-protection counters (skipped refreshes from
        # _refresh_in_flight collisions, 429s seen, retries done).
        # Per-process and reset at subprocess start, so these are
        # naturally per-run in the parallel harness.
        cadence = _llm_ctrl.get_runtime_counters()
        summary = {
            "kind": "summary",
            "arm": arm,
            "seed": seed,
            "ticks_completed": tick + 1,
            "wall_seconds": round(time.monotonic() - t_start, 2),
            "llm_api_calls": final_api_calls,
            "llm_emits_with_wave": final_emits,
            "llm_cost_usd": round(final_api_calls * _cost_for_model(model_id), 6),
            "skipped_refreshes": cadence["skipped_refreshes"],
            "throttled_429s": cadence["throttled_429s"],
            "throttle_retries": cadence["throttle_retries"],
            "extinction": extinct,
            "per_run_cap_hit": cap_hit,
            "model_id": model_id,
        }
        f.write(json.dumps(summary) + "\n")
        f.flush()
    return summary


def run_grid(
    *,
    out_dir: Path,
    arms: list[str],
    seeds: list[int],
    n_steps: int = DEFAULT_N_STEPS,
    snapshot_every: int = DEFAULT_SNAPSHOT_EVERY,
    n_creatures: int = DEFAULT_N_CREATURES,
    grid_size: int = DEFAULT_GRID_SIZE,
    refresh_every: int = DEFAULT_REFRESH_EVERY,
    model_id: str = DEFAULT_MODEL,
    confirm: bool = False,
    session_cap: int = SESSION_CALL_CAP,
    per_run_cap: int = PER_RUN_CALL_CAP,
) -> list[dict[str, Any]]:
    est = estimate_cost(arms, seeds, n_steps, n_creatures, refresh_every,
                         model_id, max_creatures=DEFAULT_MAX_CREATURES)
    print(f"[harness] grid plan:")
    print(f"  arms     : {arms}")
    print(f"  seeds    : {seeds}")
    print(f"  ticks    : {n_steps}  per-run snapshot every {snapshot_every}")
    print(f"  pop      : initial {n_creatures}, max {DEFAULT_MAX_CREATURES}, grid {grid_size}")
    print(f"  refresh  : every {refresh_every} ticks per LLM controller")
    print(f"  model    : {est['model_id']}  @ ${est['per_call_usd']:.6f}/call")
    print(f"  llm runs : {est['llm_runs']}")
    print(f"  cost low / mid / high :")
    print(f"    low  (initial pop)     : ~${est['usd_low']:.2f}  ({est['calls_low']:>7d} calls)")
    print(f"    mid  (avg ~80% of max) : ~${est['usd_mid']:.2f}  ({est['calls_mid']:>7d} calls)")
    print(f"    high (always at max)   : ~${est['usd_high']:.2f}  ({est['calls_high']:>7d} calls)")
    print(f"  session call cap: {session_cap}")
    if est["calls_high"] > session_cap:
        print(f"  ! ABORT: high-end estimate ({est['calls_high']}) exceeds session cap ({session_cap}).")
        print(f"    raise via --session-cap N if intentional.")
        return []
    if not confirm:
        print(f"  --confirm not set; not running. Re-run with --confirm to launch.")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    session_calls = 0
    for arm in arms:
        for seed in seeds:
            run_path = out_dir / f"{arm}_seed{seed}.jsonl"
            print(f"[harness] starting arm={arm} seed={seed} -> {run_path}")
            try:
                summary = run_one(
                    arm=arm, seed=seed, out_path=run_path,
                    n_steps=n_steps, snapshot_every=snapshot_every,
                    n_creatures=n_creatures, grid_size=grid_size,
                    refresh_every=refresh_every, model_id=model_id,
                    per_run_cap=per_run_cap,
                )
            except Exception as e:
                print(f"[harness] FAILED arm={arm} seed={seed}: {e!r}")
                summary = {
                    "kind": "summary", "arm": arm, "seed": seed,
                    "error": repr(e), "llm_calls": 0, "llm_cost_usd": 0.0,
                }
            session_calls += summary.get("llm_api_calls", 0)
            summary["session_calls_so_far"] = session_calls
            summaries.append(summary)
            print(
                f"[harness] done arm={arm} seed={seed} ticks={summary.get('ticks_completed')} "
                f"wall={summary.get('wall_seconds')}s "
                f"api_calls={summary.get('llm_api_calls')} "
                f"emits={summary.get('llm_emits_with_wave')} "
                f"$={summary.get('llm_cost_usd')}"
            )
            if session_calls >= session_cap:
                print(
                    f"[harness] SESSION CALL CAP REACHED ({session_calls}/{session_cap}). Stopping."
                )
                break
        else:
            continue
        break
    # Aggregate manifest.
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")
    print(f"[harness] manifest at {manifest_path}")
    return summaries


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="no_free_signal.experiments.harness")
    p.add_argument("--arm", choices=ARMS + [a.lower() for a in ARMS],
                   help="single-arm run; omit with --grid for full grid")
    p.add_argument("--seed", type=int, help="single-seed run; omit for full grid")
    p.add_argument("--grid", action="store_true",
                   help="run all arms for all --seeds (comma-separated)")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="comma-separated seeds for --grid (default: 0,1,2,3,4)")
    p.add_argument("--out", type=str, default=None,
                   help="output JSONL path (single run) or directory (grid)")
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--snapshot-every", type=int, default=DEFAULT_SNAPSHOT_EVERY)
    p.add_argument("--n-creatures", type=int, default=DEFAULT_N_CREATURES)
    p.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE)
    p.add_argument("--refresh-every", type=int, default=DEFAULT_REFRESH_EVERY)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--session-cap", type=int, default=SESSION_CALL_CAP,
                   help=f"abort grid if high-end call estimate exceeds this "
                        f"(default {SESSION_CALL_CAP}). Tighten for cost control.")
    p.add_argument("--per-run-cap", type=int, default=PER_RUN_CALL_CAP,
                   help=f"per-run LLM call cap; run terminates early if hit "
                        f"(default {PER_RUN_CALL_CAP}). Sized for ~15k ticks × 25 "
                        f"creatures / 15-tick refresh.")
    p.add_argument("--estimate", action="store_true",
                   help="print cost estimate and exit (no API calls)")
    p.add_argument("--confirm", action="store_true",
                   help="required to actually launch a grid run")
    p.add_argument("--log-emit-events", action="store_true",
                   help="dump per-emission records (kind=emit) at end of "
                        "run for downstream behavioral analysis. Off by "
                        "default to keep JSONL files small.")
    p.add_argument("--log-behavioral", action="store_true",
                   help="dump per-emit-event records (with receivers + "
                        "outcome window) and per-creature-tick state "
                        "records for receiver-response analysis. Off "
                        "by default; ~1-2 GB per 15k-tick run when on.")
    p.add_argument("--behavioral-state-log-every", type=int, default=10,
                   help="cadence (in ticks) for creature_tick records "
                        "when --log-behavioral is set (default 10). "
                        "Lower values give finer-grained matched-control "
                        "sampling at analysis time, at the cost of "
                        "~3× more disk and runtime.")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if args.estimate:
        est = estimate_cost(
            ARMS if args.grid else ([args.arm.upper()] if args.arm else ARMS),
            seeds if args.grid else ([args.seed] if args.seed is not None else seeds),
            args.n_steps, args.n_creatures, args.refresh_every, args.model,
        )
        print(json.dumps(est, indent=2))
        return 0

    if args.grid:
        out_dir = Path(args.out or "results")
        run_grid(
            out_dir=out_dir,
            arms=ARMS,
            seeds=seeds,
            n_steps=args.n_steps,
            snapshot_every=args.snapshot_every,
            n_creatures=args.n_creatures,
            grid_size=args.grid_size,
            refresh_every=args.refresh_every,
            model_id=args.model,
            confirm=args.confirm,
            session_cap=args.session_cap,
            per_run_cap=args.per_run_cap,
        )
        return 0

    if args.arm is None or args.seed is None:
        print("error: single run requires --arm and --seed (or use --grid)", file=sys.stderr)
        return 2

    out_path = Path(args.out or f"results/{args.arm.upper()}_seed{args.seed}.jsonl")
    summary = run_one(
        arm=args.arm,
        seed=args.seed,
        out_path=out_path,
        n_steps=args.n_steps,
        snapshot_every=args.snapshot_every,
        n_creatures=args.n_creatures,
        grid_size=args.grid_size,
        refresh_every=args.refresh_every,
        model_id=args.model,
        per_run_cap=args.per_run_cap,
        log_emit_events=args.log_emit_events,
        log_behavioral=args.log_behavioral,
        behavioral_state_log_every=args.behavioral_state_log_every,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
