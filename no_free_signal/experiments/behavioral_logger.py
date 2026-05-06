"""Per-emit + per-creature-tick instrumentation for behavioral receiver-
response analysis.

Two streams of records, written to the run JSONL when ``--log-behavioral``
is set on the harness:

- ``kind="emit_event"``: one record per VOCALIZE event. Captures emitter
  context (pos, driver, energy, fear, distance to nearest predator/food/
  creature, prior-tick audio) and a list of *receivers* — every other
  creature within Chebyshev radius 6 of the emitter at emit time, with
  raw + attention-weighted heard vectors, three threshold flags
  (``heard_threshold_005`` etc.), and per-receiver receiver-state at
  emit time. Outcome fields (action_t1..t3, predator_dist_t1..t3,
  energy_delta_next_10, survived_next_25, reproduced_next_50) are
  filled in over the next 50 ticks before the record is finalized.

- ``kind="creature_tick"``: per-creature state every ``state_log_every``
  ticks. Used by the analysis script to sample matched no-heard
  controls — creatures in similar arm/seed/tick/energy/predator-distance
  buckets who *didn't* hear anything that tick.

Design priorities:

1. **Pure-Python, no shared state with the controllers.** The logger
   is owned by the no_free_signal ``World`` and only reads the env's creatures
   dict; it never blocks the action-selection loop.
2. **Outcome windows are tracked in a deque.** Each emit appends a
   record marked ``"_pending_outcomes"`` until tick = emit_tick + 50,
   at which point all outcome fields are filled and the record is
   finalized.
3. **Driver tagging comes from ``c.pending_driver``** (set in
   ``World._resolve_vocal_drivers`` and read inside
   ``unified_world._try_vocalize`` before the field is cleared).
4. **Per-creature last-state cache** is maintained inside the logger
   so each emit_event can include ``previous_action`` and
   ``predator_dist_delta_tminus1_to_t`` for the receivers — necessary
   to detect "already fleeing" confounds at analysis time.

Pseudoreplication is avoided downstream: the analysis script reports
seed-level bootstrap CIs, not event-level p-values."""
from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from typing import Any

import numpy as np

# Audio physics constants (mirror foresight.envs.audio without importing it
# to keep this module loadable in standalone analysis contexts).
ATTENUATION_RADIUS = 6
ATTENUATION_SCALE = 2.5
NUM_AUDIO_BINS = 8

# Inclusion threshold: receivers below this raw_heard_strength are not
# logged (they're effectively silent). The three downstream "heard"
# threshold flags (0.05/0.10/0.25) all sit above this floor.
RECEIVER_INCLUSION_FLOOR = 0.001


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _action_name(idx: int | None) -> str | None:
    if idx is None:
        return None
    names = (
        "NORTH", "SOUTH", "EAST", "WEST", "EAT", "REST", "GATHER",
        "BUILD", "VOCALIZE", "NOOP",
    )
    if 0 <= idx < len(names):
        return names[idx]
    return f"UNKNOWN_{idx}"


class BehavioralLogger:
    """Owns the behavioral event log for a single run.

    The ``arm`` and ``seed`` fields are stamped onto every record so
    cross-run analyses can join freely.

    Memory model: finalized records stream straight to disk via the
    ``write_callback`` (one JSON-encodable dict per call). The only
    in-memory state is the pending-outcome window — at most
    ``outcome_horizon`` ticks of in-flight emit records, which is
    bounded by emit-rate × outcome_horizon (typically <100 records,
    <1 MB). This is the difference between a previous design that
    accumulated ALL events in a list and OOM-crashed a 128 GB EC2
    instance under 32 concurrent workers, and the current design
    that runs in tens of MB per worker."""

    def __init__(self, *, arm: str, seed: int, state_log_every: int = 10,
                 outcome_horizon: int = 50,
                 write_callback: Any = None):
        self.arm = arm
        self.seed = int(seed)
        self.state_log_every = int(state_log_every)
        self.outcome_horizon = int(outcome_horizon)
        # If write_callback is None, fall back to in-memory accumulation
        # (legacy mode for backwards compatibility / tests). Production
        # runs ALWAYS pass a callback so finalized records flush
        # immediately.
        self._write_callback = write_callback

        # Legacy in-memory storage. Only populated when no callback is
        # supplied. Kept for backwards compatibility with the harness
        # before it was updated to use streaming.
        self.events: list[dict[str, Any]] = []
        self.creature_ticks: list[dict[str, Any]] = []

        # Pending emit_event records waiting for outcome windows to close.
        # Keyed by emit_tick → list of records.
        self._pending: dict[int, list[dict[str, Any]]] = defaultdict(list)

        # Per-creature short-term state cache for prev-tick features.
        # cid → {"action": int|None, "pos": (r,c), "energy": float,
        #        "predator_dist": int}.
        self._prev_state: dict[int, dict[str, Any]] = {}

        # Per-creature ambient-audio rolling history (last 3 ticks of
        # raw heard strength sampled at the receiver's tile).
        self._audio_history: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=3)
        )

        # Track per-creature cumulative food eaten so we can compute
        # food_eaten_next_10 by differencing.
        self._food_count: dict[int, int] = {}

        # Action snapshot for the current tick — set by `set_actions`.
        self._actions_this_tick: dict[int, int] = {}

        # Per-tick caches: rebuilt at the start of each tick to avoid
        # repeating the same O(grid²) and O(N) scans across multiple
        # emissions in the same tick.
        self._cache_tick: int = -1
        self._food_coords_cache: list[tuple[int, int]] = []
        self._predator_dist_cache: dict[int, int] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Per-tick driver: no_free_signal World calls these in step() order.
    # ------------------------------------------------------------------

    def _emit_finalized_event(self, record: dict[str, Any]) -> None:
        """Stream OR buffer one finalized emit_event record."""
        if self._write_callback is not None:
            try:
                self._write_callback(record)
            except Exception:
                # Don't let logging failures break the simulation.
                pass
        else:
            self.events.append(record)

    def _emit_creature_tick(self, record: dict[str, Any]) -> None:
        """Stream OR buffer one creature_tick record."""
        if self._write_callback is not None:
            try:
                self._write_callback(record)
            except Exception:
                pass
        else:
            self.creature_ticks.append(record)

    def set_actions(self, actions: dict[int, int]) -> None:
        """Called before env.step(). Captures the actions dict so
        ``log_emission`` can stamp the emitter's chosen action on the
        emit record (LLM and reflex emits both override action to
        VOCALIZE; the brain's pre-override action would also be useful
        but is not currently exposed)."""
        self._actions_this_tick = dict(actions)

    def _refresh_cache_if_needed(self, *, env: Any, tick: int) -> None:
        """Rebuild per-tick caches once per tick. Saves the O(grid²)
        food scan and O(N) predator scan from running per receiver per
        emit per tick — typically a 5-20× speedup on the logger."""
        if tick == self._cache_tick:
            return
        self._cache_tick = tick
        # Food coords: one grid scan per tick.
        from foresight.envs.unified_world import TILE_FOOD
        grid = getattr(env, "_grid", None)
        coords: list[tuple[int, int]] = []
        if grid is not None:
            rows, cols = grid.shape
            for r in range(rows):
                for c in range(cols):
                    if int(grid[r, c]) == TILE_FOOD:
                        coords.append((r, c))
        self._food_coords_cache = coords
        # Predator distances: one O(N²) pass per tick.
        creatures = env.creatures
        # Identify predators once, then for each creature compute min
        # Chebyshev distance to any predator (excluding self).
        predators: list[tuple[int, tuple[int, int]]] = []
        for cid, c in creatures.items():
            try:
                pred = float(c.genome.traits().get("predate_drive", 0.0))
            except Exception:
                pred = 0.0
            if pred > 0.3:
                predators.append((cid, c.pos))
        pred_dist: dict[int, int] = {}
        for cid, c in creatures.items():
            best = 9_999
            for pcid, ppos in predators:
                if pcid == cid:
                    continue
                d = max(abs(ppos[0] - c.pos[0]), abs(ppos[1] - c.pos[1]))
                if d < best:
                    best = d
            pred_dist[cid] = best
        self._predator_dist_cache = pred_dist

    def log_emission(self, *, world: Any, env: Any, tick: int,
                      emitter_id: int, wave: np.ndarray, amp: float,
                      driver: str | None) -> None:
        """Hook called from inside _try_vocalize after a successful
        audio.emit. Computes receiver list immediately (positions are
        current; the field has just been written). Records are stored
        as pending; outcome fields fill in over the next ``outcome_horizon``
        ticks."""
        creatures = env.creatures
        emitter = creatures.get(emitter_id)
        if emitter is None:
            return

        # Refresh per-tick caches so all receiver lookups within this
        # tick share one scan.
        self._refresh_cache_if_needed(env=env, tick=tick)
        food_coords = self._food_coords_cache
        pred_cache = self._predator_dist_cache

        # Predator + food distances for the emitter at emit time
        # (cache-hit for predator; food via cached coord list).
        emitter_dist_predator = pred_cache.get(emitter_id, 9_999)
        emitter_dist_food = _nearest_food_dist_cached(emitter.pos, food_coords)
        emitter_dist_creature = _nearest_other_creature_dist(emitter, creatures)

        wave_list = [float(x) for x in np.asarray(wave, dtype=np.float64).tolist()]
        amp_f = float(amp)

        # Build receiver records for every other creature within radius.
        receivers: list[dict[str, Any]] = []
        for rid, rec in creatures.items():
            d = _chebyshev(emitter.pos, rec.pos)
            if d > ATTENUATION_RADIUS:
                continue
            falloff = math.exp(-d / ATTENUATION_SCALE) if d > 0 else 1.0
            # Contribution from THIS emission only (analytic, not from
            # the field — the field may contain other emitters' signal).
            raw_heard = np.array(wave_list, dtype=np.float64) * (amp_f * falloff)
            raw_heard_strength = float(np.max(raw_heard))
            if raw_heard_strength < RECEIVER_INCLUSION_FLOOR:
                continue
            attention = np.asarray(rec.genome.audio_attention(), dtype=np.float64)
            attn_weighted = raw_heard * attention
            attn_strength = float(np.max(attn_weighted))

            prev = self._prev_state.get(rid, {})
            prev_pred_dist = prev.get("predator_dist")
            cur_pred_dist = pred_cache.get(rid, 9_999)
            pred_delta_tm1_to_t = (
                None if prev_pred_dist is None
                else int(cur_pred_dist - prev_pred_dist)
            )
            prev_action = prev.get("action")

            audio_hist = list(self._audio_history.get(rid, deque()))
            heard_audio_last_3 = max(audio_hist) if audio_hist else 0.0

            receivers.append({
                "receiver_id": int(rid),
                "self_hearing": bool(rid == emitter_id),
                "receiver_pos": [int(rec.pos[0]), int(rec.pos[1])],
                "distance": int(d),
                "raw_heard_vector": raw_heard.round(4).tolist(),
                "attention_weighted_heard": attn_weighted.round(4).tolist(),
                "raw_heard_strength": round(raw_heard_strength, 4),
                "attention_weighted_strength": round(attn_strength, 4),
                "max_bin_raw": round(float(np.max(raw_heard)), 4),
                "max_bin_attention_weighted": round(float(np.max(attn_weighted)), 4),
                "heard_threshold_005": bool(attn_strength > 0.05),
                "heard_threshold_010": bool(attn_strength > 0.10),
                "heard_threshold_025": bool(attn_strength > 0.25),
                "receiver_energy": round(float(rec.energy), 3),
                "receiver_health": round(float(rec.health), 3),
                "receiver_fatigue": round(float(rec.fatigue), 3),
                "receiver_dist_predator_t": int(cur_pred_dist),
                "receiver_dist_food_t": int(_nearest_food_dist_cached(rec.pos, food_coords)),
                "receiver_predator_dist_delta_tminus1_to_t": pred_delta_tm1_to_t,
                "receiver_previous_action": _action_name(prev_action),
                "receiver_heard_audio_last_3_ticks": round(heard_audio_last_3, 4),
                # Outcome fields, filled later.
                "action_t1": None,
                "action_t2": None,
                "action_t3": None,
                "predator_dist_t1": None,
                "predator_dist_t2": None,
                "predator_dist_t3": None,
                "energy_t": round(float(rec.energy), 3),
                "energy_t10": None,
                "energy_delta_next_10": None,
                "food_eaten_next_10": None,
                "survived_next_10": None,
                "survived_next_25": None,
                "survived_next_50": None,
                "reproduced_next_50": None,
                "_food_count_at_t": int(self._food_count.get(rid, 0)),
            })

        emit_record = {
            "kind": "emit_event",
            "arm": self.arm,
            "seed": self.seed,
            "tick": int(tick),
            "emitter_id": int(emitter_id),
            "emitter_pos": [int(emitter.pos[0]), int(emitter.pos[1])],
            "driver": driver or "unknown",
            "wave": [round(x, 4) for x in wave_list],
            "amplitude": round(amp_f, 4),
            "emitter_energy": round(float(emitter.energy), 3),
            "emitter_health": round(float(emitter.health), 3),
            "emitter_fear": round(
                float(emitter.social_signal.get("danger", 0.0))
                + max(0.0, float(emitter.genome.fear_baseline)), 3),
            "emitter_dist_predator": int(emitter_dist_predator),
            "emitter_dist_food": int(emitter_dist_food),
            "emitter_dist_creature": int(emitter_dist_creature),
            "receivers": receivers,
        }

        with self._lock:
            self._pending[int(tick)].append(emit_record)

    def advance_tick(self, *, world: Any, env: Any, tick: int,
                      events: dict[str, Any]) -> None:
        """Called once after each world.step. Updates outcome fields on
        pending emit records, finalizes records whose horizon has
        closed, and updates the per-creature state cache used for
        prev-tick features on future emits."""
        # Refresh per-tick caches once per tick.
        self._refresh_cache_if_needed(env=env, tick=tick)
        food_coords = self._food_coords_cache
        pred_cache = self._predator_dist_cache

        creatures = env.creatures
        live_ids = set(creatures.keys())

        # Track births so reproduced_next_50 can be filled. The no_free_signal
        # World step result has events; harness passes them through.
        births_this_tick: set[int] = set()
        deaths_this_tick: set[int] = set()
        food_eaters_this_tick: dict[int, int] = {}
        for ev in events.get("events", []) if isinstance(events, dict) else []:
            kind = ev.get("kind")
            if kind == "birth":
                # Track parent ids (parents reproduced).
                for parent_field in ("parent_a", "parent_b", "parent"):
                    p = ev.get(parent_field)
                    if isinstance(p, int):
                        births_this_tick.add(p)
            elif kind == "death":
                cid = ev.get("creature_id")
                if isinstance(cid, int):
                    deaths_this_tick.add(cid)
            elif kind == "ate_food":
                cid = ev.get("creature_id")
                if isinstance(cid, int):
                    food_eaters_this_tick[cid] = food_eaters_this_tick.get(cid, 0) + 1

        # Update cumulative food count.
        for cid, n in food_eaters_this_tick.items():
            self._food_count[cid] = self._food_count.get(cid, 0) + n

        # Update each pending record's outcome fields based on the
        # offset (tick - emit_tick). Skip the inner receiver pass on
        # ticks where no outcome field is due AND no births happened
        # — that's most of the 50-tick horizon.
        relevant_offsets = {1, 2, 3, 10, 25, 50}
        has_births = bool(births_this_tick)
        with self._lock:
            now_finalized: list[dict[str, Any]] = []
            for emit_tick, pending_list in list(self._pending.items()):
                offset = tick - emit_tick
                if offset <= 0:
                    continue
                if offset > self.outcome_horizon:
                    continue
                # Decide whether the inner receiver loop needs to run.
                needs_inner_pass = (
                    offset in relevant_offsets
                    or (offset <= 50 and has_births)
                )
                if needs_inner_pass:
                    for record in pending_list:
                        for r in record["receivers"]:
                            rid = r["receiver_id"]
                            rec = creatures.get(rid)
                            # action / predator_dist at t+1, t+2, t+3
                            if offset in (1, 2, 3):
                                r[f"action_t{offset}"] = _action_name(
                                    self._actions_this_tick.get(rid)
                                )
                                if rec is not None:
                                    r[f"predator_dist_t{offset}"] = int(
                                        pred_cache.get(rid, 9_999)
                                    )
                            # energy delta at t+10
                            if offset == 10 and rec is not None:
                                r["energy_t10"] = round(float(rec.energy), 3)
                                r["energy_delta_next_10"] = round(
                                    float(rec.energy) - float(r["energy_t"]), 3
                                )
                                r["food_eaten_next_10"] = int(
                                    self._food_count.get(rid, 0)
                                    - r["_food_count_at_t"]
                                )
                            # survived flags
                            if offset == 10:
                                r["survived_next_10"] = rid in live_ids
                            if offset == 25:
                                r["survived_next_25"] = rid in live_ids
                            if offset == 50:
                                r["survived_next_50"] = rid in live_ids
                            # reproduced flag — accumulates over window
                            if offset <= 50 and rid in births_this_tick:
                                r["reproduced_next_50"] = True
                # Finalize when offset reaches the horizon.
                if offset >= self.outcome_horizon:
                    for record in pending_list:
                        for r in record["receivers"]:
                            for k in ("survived_next_10", "survived_next_25",
                                      "survived_next_50"):
                                if r.get(k) is None:
                                    r[k] = r["receiver_id"] in live_ids
                            if r.get("reproduced_next_50") is None:
                                r["reproduced_next_50"] = False
                            r.pop("_food_count_at_t", None)
                        now_finalized.append(record)
                    del self._pending[emit_tick]
            # Stream/buffer finalized records.
            for record in now_finalized:
                self._emit_finalized_event(record)

        # Snapshot per-creature state for the prev-tick cache and the
        # ambient-audio rolling history. Done AFTER outcome updates so
        # the cache reflects post-step state. Uses the per-tick caches
        # built at the top of advance_tick.
        log_state_this_tick = (
            self.state_log_every > 0 and tick % self.state_log_every == 0
        )
        for cid, rec in creatures.items():
            ambient = float(np.max(env.audio.sample(rec.pos))) if hasattr(env, "audio") else 0.0
            self._audio_history[cid].append(ambient)
            self._prev_state[cid] = {
                "action": self._actions_this_tick.get(cid),
                "pos": (int(rec.pos[0]), int(rec.pos[1])),
                "energy": float(rec.energy),
                "predator_dist": int(pred_cache.get(cid, 9_999)),
            }
            # Cadence-gated creature_tick logging — fold into the same
            # creature pass to avoid a second iteration.
            if log_state_this_tick:
                self._emit_creature_tick({
                    "kind": "creature_tick",
                    "arm": self.arm,
                    "seed": self.seed,
                    "tick": int(tick),
                    "creature_id": int(cid),
                    "pos": [int(rec.pos[0]), int(rec.pos[1])],
                    "energy": round(float(rec.energy), 3),
                    "health": round(float(rec.health), 3),
                    "fatigue": round(float(rec.fatigue), 3),
                    "fear": round(
                        float(rec.social_signal.get("danger", 0.0))
                        + max(0.0, float(rec.genome.fear_baseline)), 3),
                    "dist_predator": int(pred_cache.get(cid, 9_999)),
                    "dist_food": int(_nearest_food_dist_cached(rec.pos, food_coords)),
                    "ambient_audio_strength": round(ambient, 4),
                    "last_action": _action_name(self._actions_this_tick.get(cid)),
                })

    def finalize_remaining(self, *, env: Any, tick: int) -> None:
        """At end-of-run, fill outcome fields for any still-pending
        records using the current state (not all 50-tick windows will
        have closed). Streams via the write callback (or appends to
        self.events in legacy mode)."""
        creatures = env.creatures
        live_ids = set(creatures.keys())
        with self._lock:
            for emit_tick, pending_list in list(self._pending.items()):
                for record in pending_list:
                    for r in record["receivers"]:
                        for k in ("survived_next_10", "survived_next_25",
                                  "survived_next_50"):
                            if r.get(k) is None:
                                r[k] = r["receiver_id"] in live_ids
                        if r.get("reproduced_next_50") is None:
                            r["reproduced_next_50"] = False
                        r.pop("_food_count_at_t", None)
                    self._emit_finalized_event(record)
            self._pending.clear()


def _nearest_predator_dist(target: Any, creatures: dict[int, Any]) -> int:
    best = 9_999
    tid = target.individual_id
    for cid, c in creatures.items():
        if cid == tid:
            continue
        try:
            pred = float(c.genome.traits().get("predate_drive", 0.0))
        except Exception:
            pred = 0.0
        if pred <= 0.3:
            continue
        d = _chebyshev(target.pos, c.pos)
        if d < best:
            best = d
    return best


def _nearest_other_creature_dist(target: Any, creatures: dict[int, Any]) -> int:
    best = 9_999
    tid = target.individual_id
    for cid, c in creatures.items():
        if cid == tid:
            continue
        d = _chebyshev(target.pos, c.pos)
        if d < best:
            best = d
    return best


def _nearest_food_dist(target: Any, env: Any) -> int:
    """Linear-scan grid for the nearest food tile. O(N²) but N≤24.
    Slow path; the per-tick cache in BehavioralLogger uses
    ``_nearest_food_dist_cached`` to amortize the grid scan."""
    grid = getattr(env, "_grid", None)
    if grid is None:
        return 9_999
    # TILE_FOOD constant: import here to avoid module-level cycles.
    from foresight.envs.unified_world import TILE_FOOD
    rows, cols = grid.shape
    tr, tc = target.pos
    best = 9_999
    for r in range(rows):
        for c in range(cols):
            if int(grid[r, c]) == TILE_FOOD:
                d = max(abs(r - tr), abs(c - tc))
                if d < best:
                    best = d
    return best


def _nearest_food_dist_cached(pos: tuple[int, int],
                                food_coords: list[tuple[int, int]]) -> int:
    """Fast variant: takes a precomputed list of food coords. Caller
    builds the list once per tick and shares it across all queries."""
    if not food_coords:
        return 9_999
    tr, tc = pos
    best = 9_999
    for r, c in food_coords:
        d = max(abs(r - tr), abs(c - tc))
        if d < best:
            best = d
    return best
