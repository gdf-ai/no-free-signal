"""World — Lamdis facade over the unified-creature env.

Three grounding levels, multi-mode API surface, per-creature brains,
intervention API, snapshot/restore. Single Creature type — niche emerges from
genome traits, not hardcoded species.
"""
from __future__ import annotations

import copy
import threading
import traceback
from collections import deque
from typing import Any

import numpy as np

from foresight.envs.audio import NUM_AUDIO_BINS
from foresight.envs.unified_world import (
    ACTION_VOCALIZE,
    Creature,
    StepEvents,
    TILE_EMPTY,
    TILE_FOOD,
    TILE_SHELTER,
    UnifiedWorld,
    WorldConfig,
)
from foresight.evolution.genome import Genome
from no_free_signal.brains import CreatureBrainManager
from no_free_signal.controller import ControllerFn, ControllerHandle, ControllerRegistry
from no_free_signal.llm_controller import (
    LLMController,
    USD_PER_CALL_ESTIMATE,
    _today_count,
    _daily_limit,
    get_last_disable_reason,
    has_aws_credentials,
    is_llm_enabled,
    set_llm_enabled,
)
from no_free_signal.observation import (
    RawObservation,
    build_observation_for_creature,
    render_world_array,
    render_world_png,
)
from no_free_signal.serialize import creature_to_dict, genome_to_dict, world_to_dict

MAX_EVENT_HISTORY = 500


class World:
    def __init__(
        self,
        seed: int = 42,
        n_creatures: int = 16,
        grid_size: int = 40,
        device: str = "cpu",
        enable_brains: bool = True,
        llm_emit_logger: Any = None,
        llm_wave_transform_factory: Any = None,
    ):
        cfg = WorldConfig(
            seed=seed,
            initial_creatures=n_creatures,
            grid_size=grid_size,
        )
        self._cfg = cfg
        self._env = UnifiedWorld(cfg)
        self._event_history: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._enable_brains = enable_brains
        self._device = device
        self._action_rng = np.random.default_rng(seed + 1)

        sample = next(iter(self._env.creatures.values()))
        sample_obs = build_observation_for_creature(sample, self._env)
        self._obs_dim = int(sample_obs.flat.shape[0])
        # 10 actions: NSEW, eat, rest, gather, build, vocalize, noop
        self._n_actions = 10
        # Stats for the build-puzzle attempts surface
        self._build_attempts: int = 0
        self._build_successes: int = 0
        # Auto-attach LLM controllers to every creature on birth (Phase 9.N).
        # Set via World.enable_auto_attach_llm() after server start.
        self._llm_auto_attach: bool = False
        self._llm_default_personality: str = (
            "You are a creature exploring this world. Your innate biology and "
            "drives shape your behaviour; speak naturally about what you "
            "sense, what you intend to do, and what other creatures you see "
            "or hear."
        )
        self._llm_target_cps: float = 1.0  # population-wide target calls/sec
        # Dialogue ring buffer: utterances spoken by any creature, with the
        # speaker's position and vision_range at the moment they spoke (so
        # spatial broadcast is consistent even after the speaker moves).
        self._utterances: deque[dict[str, Any]] = deque(maxlen=400)
        self._utterances_lock = threading.Lock()

        # Experiment hooks. ``_llm_emit_logger`` is invoked once per validated
        # LLM emission (post-transform). ``_llm_wave_transform_factory`` returns
        # a per-creature wave-transform callable used by arms E and F. Set
        # before any LLMController is spawned so newborns inherit them too.
        self._llm_emit_logger: Any = llm_emit_logger
        self._llm_wave_transform_factory: Any = llm_wave_transform_factory
        # Hooks called for every newborn creature with (world, creature). Used
        # by the mute ablation to clamp vocal_amplitude to 0 on births.
        self._post_birth_hooks: list[Any] = []
        # Behavioral instrumentation logger (off by default). Attached by the
        # harness when --log-behavioral is set; receives per-emit and
        # per-tick callbacks during the step loop.
        self._behavioral_logger: Any = None

        self.brain_manager: CreatureBrainManager | None = None
        if enable_brains:
            self.brain_manager = CreatureBrainManager(
                obs_dim=self._obs_dim, n_actions=self._n_actions, device=device, seed=seed,
            )
            for cid, c in self._env.creatures.items():
                self._safe_alloc_brain(cid, c.genome)

        self.controllers = ControllerRegistry()

        self._genome_history: list[dict[str, Any]] = []
        self._genome_history_every: int = 10
        self._last_genome_log_step: int = -10**9
        self._record_genome_history(force=True)

        self._log_event(kind="world_init", details={
            "seed": seed,
            "n_creatures": len(self._env.creatures),
            "grid_size": grid_size,
            "brains_enabled": enable_brains,
            "device": device,
        })

    # ----- safe brain allocation (Phase 9.A bug fix) -----
    def _safe_alloc_brain(self, cid: int, genome: Genome) -> bool:
        if self.brain_manager is None:
            return False
        try:
            self.brain_manager.on_birth(cid, genome)
        except Exception as e:
            self._log_event("brain_allocation_failed", {
                "creature_id": cid, "error": repr(e),
                "trace": traceback.format_exc(limit=3),
            })
            return False
        # Auto-attach LLM controller if enabled, AWS creds available, and
        # the creature isn't already controlled.
        if self._llm_auto_attach and has_aws_credentials() and cid in self._env.creatures:
            if cid not in self.controllers:
                try:
                    refresh = self._compute_refresh_every_ticks()
                    c = self._env.creatures[cid]
                    ctrl = LLMController(
                        creature=c,
                        personality=self._llm_default_personality,
                        lambda_advisory=1.0,
                        refresh_every_ticks=refresh,
                        world_ref=self,
                        emit_logger=self._llm_emit_logger,
                        wave_transform=(
                            self._llm_wave_transform_factory(c)
                            if self._llm_wave_transform_factory else None
                        ),
                    )
                    self.controllers.attach(cid, ctrl)
                except Exception as e:
                    self._log_event("llm_auto_attach_failed", {
                        "creature_id": cid, "error": repr(e),
                    })
        # Post-birth hooks (e.g. mute ablation clamping vocal_amplitude=0).
        if cid in self._env.creatures:
            c = self._env.creatures[cid]
            for hook in self._post_birth_hooks:
                try:
                    hook(self, c)
                except Exception:
                    pass
        return True

    # ------------------------------------------------------------------
    # Behavioral instrumentation (off by default; harness opt-in)
    # ------------------------------------------------------------------
    def attach_behavioral_logger(self, logger: Any) -> None:
        """Install a BehavioralLogger and wire the env's per-emit
        observer hook so each successful audio.emit fires
        ``logger.log_emission(...)`` with full context. The per-tick
        ``logger.advance_tick(...)`` is invoked from inside step()
        when this attachment is non-None."""
        self._behavioral_logger = logger
        # Install the per-emit observer on the env. The callback is
        # invoked with kwargs (creature, wave, amp, driver, tick).
        def _on_emit(*, creature, wave, amp, driver, tick):
            logger.log_emission(
                world=self, env=self._env, tick=int(tick),
                emitter_id=int(creature.individual_id),
                wave=wave, amp=float(amp), driver=driver,
            )
        self._env._emit_observer = _on_emit

    # ------------------------------------------------------------------
    # Auto-attach + auto-tune (Phase 9.N)
    # ------------------------------------------------------------------
    def enable_auto_attach_llm(
        self,
        personality: str | None = None,
        target_cps: float = 1.0,
    ) -> dict[str, Any]:
        """Turn on auto-attach so every existing creature gets an LLM
        controller and every newborn gets one too. Idempotent."""
        with self._lock:
            self._llm_auto_attach = True
            if personality:
                self._llm_default_personality = personality
            self._llm_target_cps = max(0.05, min(10.0, float(target_cps)))
            attached: list[int] = []
            if has_aws_credentials():
                refresh = self._compute_refresh_every_ticks()
                for cid, c in self._env.creatures.items():
                    if cid in self.controllers:
                        continue
                    try:
                        ctrl = LLMController(
                            creature=c,
                            personality=self._llm_default_personality,
                            lambda_advisory=1.0,
                            refresh_every_ticks=refresh,
                            world_ref=self,
                            emit_logger=self._llm_emit_logger,
                            wave_transform=(
                                self._llm_wave_transform_factory(c)
                                if self._llm_wave_transform_factory else None
                            ),
                        )
                        self.controllers.attach(cid, ctrl)
                        attached.append(cid)
                    except Exception as e:
                        self._log_event("llm_auto_attach_failed", {
                            "creature_id": cid, "error": repr(e),
                        })
            return {
                "ok": True,
                "auto_attach": True,
                "credentials_available": has_aws_credentials(),
                "attached": attached,
                "refresh_every_ticks": self._compute_refresh_every_ticks(),
                "target_cps": self._llm_target_cps,
            }

    def _compute_refresh_every_ticks(self) -> int:
        """Pick a per-creature refresh rate so that the population-wide LLM
        call rate is roughly `target_cps` calls/sec. Assumes ~5 ticks/sec at
        full speed. Clamped to [3, 600]."""
        n = max(1, len(self._env.creatures))
        ticks_per_sec = 5.0
        ticks_per_call = ticks_per_sec * n / max(0.05, self._llm_target_cps)
        return int(max(3, min(600, ticks_per_call)))

    # ------------------------------------------------------------------
    # Observation — symbolic
    # ------------------------------------------------------------------
    def observe(self, n_recent_events: int = 20) -> dict[str, Any]:
        with self._lock:
            llm_state = self.get_all_llm_thoughts()
            brain_state: dict[int, dict[str, Any]] = {}
            if self.brain_manager is not None:
                for cid, brain in self.brain_manager.brains.items():
                    s = brain.stats()
                    brain_state[cid] = {
                        "age": s["age"],
                        "loss_recent": s["loss_recent"],
                    }
            base = world_to_dict(
                self._env,
                recent_events=self._event_history[-n_recent_events:],
                llm_state=llm_state,
                brain_state=brain_state,
            )
            base["brains_enabled"] = self._enable_brains
            base["n_brains"] = len(self.brain_manager) if self.brain_manager else 0
            base["n_controllers"] = len(self.controllers)
            base["build_attempts"] = self._build_attempts
            base["build_successes"] = self._build_successes
            base["llm_creature_ids"] = list(llm_state.keys())
            base["llm_auto_attach"] = self._llm_auto_attach
            base["recent_utterances"] = self.recent_utterances(limit=30)
            return base

    def observe_creature(self, creature_id: int) -> dict[str, Any]:
        with self._lock:
            if creature_id not in self._env.creatures:
                return {"error": f"no living creature with id {creature_id}"}
            c = self._env.creatures[creature_id]
            d = creature_to_dict(c, verbose=True, env=self._env)
            if self.brain_manager and creature_id in self.brain_manager:
                brain = self.brain_manager.get(creature_id)
                d["brain"] = brain.stats()
                d["brain"]["drives_now"] = brain.current_drives()
            return d

    def list_creatures(self) -> dict[str, Any]:
        with self._lock:
            return {
                "creatures": [creature_to_dict(c) for c in self._env.creatures.values()],
            }

    def lineage(self, creature_id: int, max_generations: int = 30) -> dict[str, Any]:
        with self._lock:
            chain: list[dict[str, Any]] = []
            cur = self._env.creatures.get(creature_id)
            if cur is None:
                return {"error": f"no living creature with id {creature_id}"}
            for _ in range(max_generations):
                chain.append({
                    "id": cur.individual_id,
                    "phenotype": cur.genome.phenotype(),
                    "generation": cur.generation,
                    "age": cur.age,
                    "n_offspring": cur.n_offspring,
                    "genome": genome_to_dict(cur.genome),
                })
                if cur.parent_a_id is None:
                    break
                parent = self._env.creatures.get(cur.parent_a_id)
                if parent is None:
                    chain.append({"id": cur.parent_a_id, "note": "ancestor no longer alive"})
                    break
                cur = parent
            return {"creature_id": creature_id, "chain": chain}

    # ------------------------------------------------------------------
    # Observation — sensory (raw)
    # ------------------------------------------------------------------
    def obs_for(self, creature_id: int):
        with self._lock:
            if creature_id not in self._env.creatures:
                return {"error": f"no living creature with id {creature_id}"}
            return build_observation_for_creature(self._env.creatures[creature_id], self._env)

    def obs_for_dict(self, creature_id: int) -> dict[str, Any]:
        obs = self.obs_for(creature_id)
        if isinstance(obs, dict):
            return obs
        return {
            "obs_window": obs.obs_window.tolist(),
            "drives": obs.drives,
            "extras": obs.extras,
            "flat_shape": list(obs.flat.shape),
        }

    # ------------------------------------------------------------------
    # Observation — perceptual
    # ------------------------------------------------------------------
    def render_png_bytes(self, scale: int = 16) -> bytes:
        with self._lock:
            return render_world_png(self._env, scale=scale)

    def render_array(self, scale: int = 16) -> np.ndarray:
        with self._lock:
            return render_world_array(self._env, scale=scale)

    # ------------------------------------------------------------------
    # Controllers
    # ------------------------------------------------------------------
    def attach_controller(self, creature_id: int, fn: ControllerFn) -> ControllerHandle:
        with self._lock:
            if creature_id not in self._env.creatures:
                raise ValueError(f"no living creature with id {creature_id}")
            return self.controllers.attach(creature_id, fn)

    # ------------------------------------------------------------------
    # LLM controllers (Phase 9.L)
    # ------------------------------------------------------------------
    def attach_llm_controller(
        self,
        creature_id: int,
        personality: str,
        lambda_advisory: float = 1.0,
        refresh_every_ticks: int = 5,
    ) -> dict[str, Any]:
        with self._lock:
            if creature_id not in self._env.creatures:
                return {"error": f"no living creature with id {creature_id}"}
            if not has_aws_credentials():
                return {"error": "AWS credentials not set (need AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)"}
            c = self._env.creatures[creature_id]
            ctrl = LLMController(
                creature=c,
                personality=personality,
                lambda_advisory=lambda_advisory,
                refresh_every_ticks=refresh_every_ticks,
                world_ref=self,
                emit_logger=self._llm_emit_logger,
                wave_transform=(
                    self._llm_wave_transform_factory(c)
                    if self._llm_wave_transform_factory else None
                ),
            )
            self.controllers.attach(creature_id, ctrl)
            self._log_event("llm_attached", {
                "creature_id": creature_id,
                "personality": personality[:120],
                "lambda_advisory": lambda_advisory,
                "refresh_every_ticks": refresh_every_ticks,
            })
            return {"ok": True, "creature_id": creature_id, "stats": ctrl.stats()}

    def detach_controller(self, creature_id: int) -> dict[str, Any]:
        with self._lock:
            fn = self.controllers.get(creature_id)
            if fn is None:
                return {"error": f"no controller attached to creature {creature_id}"}
            self.controllers._detach(creature_id)
            self._log_event("controller_detached", {"creature_id": creature_id})
            return {"ok": True, "creature_id": creature_id}

    def get_llm_log(self, creature_id: int, limit: int = 20) -> dict[str, Any]:
        with self._lock:
            fn = self.controllers.get(creature_id)
            if fn is None or not isinstance(fn, LLMController):
                return {"error": f"no LLM controller on creature {creature_id}"}
            return {"creature_id": creature_id, "stats": fn.stats(),
                    "log": fn.get_log(limit=limit)}

    def list_llm_controllers(self) -> list[int]:
        with self._lock:
            return [
                cid for cid, fn in self.controllers._controllers.items()
                if isinstance(fn, LLMController)
            ]

    # ------------------------------------------------------------------
    # Bulk thought collector (Phase 9.N)
    # ------------------------------------------------------------------
    def get_all_llm_thoughts(self) -> dict[int, dict[str, Any]]:
        """Latest cached thought per LLM-attached creature in one call.
        Used by `world_to_dict` to populate the at-a-glance roster."""
        out: dict[int, dict[str, Any]] = {}
        for cid, fn in self.controllers._controllers.items():
            if isinstance(fn, LLMController):
                try:
                    out[cid] = fn.current_thought()
                except Exception:
                    pass
        return out

    # ------------------------------------------------------------------
    # Dialogue (Phase 9.N)
    # ------------------------------------------------------------------
    def broadcast_utterance(self, speaker_id: int, text: str) -> None:
        """Called from an LLMController's refresh thread when it emits a
        non-empty `say`. Captures the speaker's pos + vision_range *now* so
        the audience set is consistent even after movement."""
        if not text:
            return
        c = self._env.creatures.get(speaker_id)
        if c is None:
            return
        vision = float(c.genome.traits().get("vision_range", 7.0))
        utt = {
            "speaker_id": int(speaker_id),
            "text": str(text)[:80],
            "step": int(self._env.steps),
            "pos": list(c.pos),
            "vision": vision,
        }
        with self._utterances_lock:
            self._utterances.append(utt)

    def utterances_heard_by(
        self, listener_id: int, since_step: int = 0,
    ) -> list[dict[str, Any]]:
        """Return utterances whose speaker was within their `vision` distance
        (chebyshev) of the listener at the moment of speaking, since the given
        step. Used by LLMControllers when assembling their next prompt."""
        listener = self._env.creatures.get(listener_id)
        if listener is None:
            return []
        lr, lc = listener.pos
        with self._utterances_lock:
            snapshot = list(self._utterances)
        out: list[dict[str, Any]] = []
        for u in snapshot:
            if u["step"] <= since_step:
                continue
            if u["speaker_id"] == listener_id:
                continue
            sr, sc = u["pos"]
            d = max(abs(sr - lr), abs(sc - lc))
            if d <= u["vision"]:
                out.append(u)
        return out

    def recent_utterances(self, limit: int = 30) -> list[dict[str, Any]]:
        """For the world snapshot — used by the frontend to render speech
        bubbles above speakers."""
        with self._utterances_lock:
            return list(self._utterances)[-limit:]

    # ------------------------------------------------------------------
    # Pain / discomfort / pleasure feedback into the LLM prompt context
    # ------------------------------------------------------------------
    def _distribute_personal_events(self, ev: StepEvents) -> None:
        """After an env step, translate relevant events into vivid sensations
        and push them to each affected creature's LLMController prompt
        buffer. Only creatures with attached LLM controllers are touched —
        plain-brain creatures don't have a language layer to feel into."""
        # Local helper: push a sensation onto a creature's controller, if it
        # has one and is still alive.
        def push(cid: int, kind: str, sentence: str) -> None:
            ctrl = self.controllers.get(cid)
            if ctrl is None or not isinstance(ctrl, LLMController):
                return
            ctrl._personal_events.append({
                "step": int(self._env.steps),
                "kind": kind,
                "sentence": sentence[:160],
            })

        for cid in ev.eats_food:
            push(cid, "ate_food",
                 "You ate food. It tasted right and you feel less hungry.")
        for eater, victim in ev.eats_creature:
            push(eater, "ate_creature",
                 f"You bit c{victim}. Their flesh was warm and gamey — energy returns "
                 f"but their struggle was unsettling.")
        for defender, attacker, dmg in ev.counter_attacks:
            # The eater (attacker) FELT the bite — pain is the strongest signal.
            push(attacker, "counter_hit",
                 f"You tried to eat c{defender} but they fought back. Sharp pain — "
                 f"you took {dmg:.0f} damage and the meal felt wrong.")
            push(defender, "fought_off",
                 f"c{attacker} attacked you. You bit/clawed back hard and they yelped.")
        for cid, matter in (getattr(ev, "poisonings", []) or []):
            push(cid, "poisoned",
                 f"You ingested raw {matter}. Your stomach burns. This was NOT food. "
                 f"Health dropping. Don't try that again.")
        for cid in ev.rests:
            push(cid, "rested",
                 "You rested under shelter. Fatigue easing.")
        for cid in ev.builds:
            push(cid, "built",
                 "You finished building a shelter. The puzzle in your head fell into "
                 "place — you feel proud and a little tired.")
        for cid in (getattr(ev, "build_fails", []) or []):
            push(cid, "build_failed",
                 "You tried to build but the construction puzzle in your head wouldn't "
                 "resolve. Wasted some materials. Frustrating.")
        for cid, kind in ev.gathers:
            push(cid, "gathered",
                 f"You picked up some raw {kind}. Useful for building, NOT for eating.")
        for parent_a, parent_b, child in ev.matings:
            push(parent_a, "mated",
                 f"You mated with c{parent_b}. A new creature c{child} was born from you.")
            push(parent_b, "mated",
                 f"You mated with c{parent_a}. A new creature c{child} was born from you.")

    def detach_all_llm_controllers(self) -> dict[str, Any]:
        with self._lock:
            ids = self.list_llm_controllers()
            for cid in ids:
                self.controllers._detach(cid)
            self._log_event("llm_detach_all", {"count": len(ids)})
            return {"ok": True, "detached": ids}

    def llm_runtime_state(self) -> dict[str, Any]:
        with self._lock:
            attached = self.list_llm_controllers()
            daily = _today_count()
            limit = _daily_limit()
            return {
                "enabled": is_llm_enabled(),
                "credentials_available": has_aws_credentials(),
                "attached_count": len(attached),
                "attached_creature_ids": attached,
                "daily_calls": daily,
                "daily_limit": limit,
                "estimated_usd_today": round(daily * USD_PER_CALL_ESTIMATE, 4),
                "usd_per_call_estimate": USD_PER_CALL_ESTIMATE,
                "last_disable_reason": get_last_disable_reason(),
            }

    def set_llm_runtime_enabled(self, enabled: bool) -> dict[str, Any]:
        set_llm_enabled(enabled)
        self._log_event("llm_runtime_toggle", {"enabled": enabled})
        return self.llm_runtime_state()

    def load_scenario(self, scenario: dict[str, Any]) -> dict[str, Any]:
        """Reset the world to the scenario's initial state, then introduce
        creatures with the prescribed genome overrides and attach LLM
        controllers with the prescribed personalities. Caller has already
        validated the scenario shape."""
        with self._lock:
            seed = int(scenario.get("world_seed", 0))
            n_others = int(scenario.get("world_creatures", 4))
            # Reset with the scenario seed and a small population — the LLM
            # creatures are introduced on top.
            self._cfg = WorldConfig(
                seed=seed,
                initial_creatures=max(1, n_others),
                grid_size=int(scenario.get("world_size", 30)),
            )
            self._env = UnifiedWorld(self._cfg)
            self._event_history = []
            # Reset the controller registry FIRST, then allocate brains.
            # _safe_alloc_brain has the LLM-auto-attach hook which adds to
            # `self.controllers`; if we wiped controllers afterward we'd
            # erase the auto-attached LLMs (regression caught 2026-04-29).
            self.controllers = ControllerRegistry()
            if self.brain_manager is not None:
                self.brain_manager = CreatureBrainManager(
                    obs_dim=self._obs_dim, n_actions=self._n_actions,
                    device=self._device, seed=seed,
                )
                for cid, c in self._env.creatures.items():
                    self._safe_alloc_brain(cid, c.genome)
            self._utterances.clear()
            self._genome_history = []
            self._last_genome_log_step = -10**9
            self._record_genome_history(force=True)

            attached: list[dict[str, Any]] = []
            issues: list[str] = []
            if not has_aws_credentials():
                issues.append(
                    "AWS credentials not set — scenario world loaded but no LLM "
                    "controllers were attached."
                )
            for spec in scenario.get("creatures", []):
                # Build a Genome from the random pool, then override fields.
                base = Genome.random(self._env._rng).traits()
                for k, v in (spec.get("genome_overrides") or {}).items():
                    if k in base:
                        base[k] = float(v)
                try:
                    g = Genome(**base)
                except TypeError as e:
                    issues.append(f"genome construction failed: {e}")
                    continue
                new = self._env._spawn_creature(
                    genome=g, parent_ids=(None, None), generation=0,
                )
                if self.brain_manager:
                    self._safe_alloc_brain(new.individual_id, g)
                if has_aws_credentials():
                    ctrl = LLMController(
                        creature=new,
                        personality=str(spec.get("personality", "")),
                        lambda_advisory=float(spec.get("lambda_advisory", 1.0)),
                        refresh_every_ticks=int(spec.get("refresh_every_ticks", 5)),
                        world_ref=self,  # required so this creature can speak + hear
                        emit_logger=self._llm_emit_logger,
                        wave_transform=(
                            self._llm_wave_transform_factory(new)
                            if self._llm_wave_transform_factory else None
                        ),
                    )
                    self.controllers.attach(new.individual_id, ctrl)
                attached.append({
                    "creature_id": new.individual_id,
                    "personality": str(spec.get("personality", ""))[:200],
                    "phenotype": g.phenotype(),
                })
            self._log_event("scenario_loaded", {
                "scenario_id": scenario.get("id"),
                "name": scenario.get("name"),
                "attached_creatures": [a["creature_id"] for a in attached],
                "issues": issues,
            })
            return {
                "ok": True,
                "scenario_id": scenario.get("id"),
                "attached": attached,
                "issues": issues,
            }

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------
    def step(self, n: int = 1) -> dict[str, Any]:
        emitted: list[dict[str, Any]] = []
        steps_taken = 0
        with self._lock:
            n = max(1, int(n))
            # Fast-forward: when more than one step is requested per tick, run
            # the leading n-1 steps with the env's heuristic (no brain
            # perceive / choose_action / training) and only the final step
            # with full brain pipeline. Brains catch up perceiving at the end.
            # This is what makes the speed slider produce real speedup —
            # without it, each step pays the per-creature RSSM-forward cost
            # (~180 ms with 16 brains on CPU), so step(n) ≈ n × step(1).
            if n > 1:
                fast_n = n - 1
                for _ in range(fast_n):
                    pre_ids = set(self._env.creatures.keys())
                    ev = self._env.step(actions=None)  # heuristic per creature
                    steps_taken += 1
                    self._distribute_personal_events(ev)
                    for e in self._convert_events(ev):
                        self._log_event_full(e)
                        emitted.append(e)
                    if self.brain_manager:
                        # Allocate brains for newborns and reap dead ones so
                        # the invariant survives the fast-forward window.
                        for cid in self._env.creatures:
                            if cid not in self.brain_manager and cid not in pre_ids:
                                self._safe_alloc_brain(
                                    cid, self._env.creatures[cid].genome
                                )
                        self.brain_manager.reap(set(self._env.creatures.keys()))
                    self.controllers.reap(set(self._env.creatures.keys()))
                    self._record_genome_history()
                    if not self._env.creatures:
                        self._log_event(kind="total_extinction")
                        break

            # Final step: full brain pipeline (same as the n=1 path). Skip
            # if extinction wiped everyone during fast-forward.
            for _ in range(1 if self._env.creatures else 0):
                actions: dict[int, int] = {}
                for cid, c in self._env.creatures.items():
                    raw = build_observation_for_creature(c, self._env)
                    fn = self.controllers.get(cid)
                    is_llm = isinstance(fn, LLMController)
                    if fn is not None and not is_llm:
                        # External non-LLM controller: authoritative override.
                        try:
                            a = int(fn(raw))
                            if 0 <= a < self._n_actions:
                                actions[cid] = a
                        except Exception as e:
                            self._log_event("controller_error", {
                                "creature_id": cid, "error": repr(e),
                            })
                            continue
                        if self.brain_manager and cid in self.brain_manager:
                            self.brain_manager.get(cid).perceive(raw)
                    elif is_llm:
                        # LLM controller: advisory blend with the brain.
                        # Tick the controller (kicks off async refresh, returns
                        # cached action immediately) and feed its boost into
                        # brain.choose_action.
                        try:
                            fn(raw)  # pulse the controller's tick counter
                            advisory = fn.current_advice()
                        except Exception as e:
                            self._log_event("controller_error", {
                                "creature_id": cid, "error": repr(e),
                            })
                            advisory = None
                        if self.brain_manager and cid in self.brain_manager:
                            brain = self.brain_manager.get(cid)
                            brain.perceive(raw)
                            actions[cid] = brain.choose_action(
                                raw, self._action_rng, advisory=advisory,
                            )
                        elif advisory is not None:
                            actions[cid] = int(advisory[0])
                    elif self.brain_manager and cid in self.brain_manager:
                        brain = self.brain_manager.get(cid)
                        brain.perceive(raw)
                        actions[cid] = brain.choose_action(raw, self._action_rng)

                pre_ids = set(self._env.creatures.keys())
                self._resolve_vocal_drivers(actions)
                # Behavioral logger: stamp the actions dict so the per-emit
                # hook (fired inside _try_vocalize) can read prev-action
                # state and the post-step advance can fill action_t1..t3.
                if self._behavioral_logger is not None:
                    try:
                        self._behavioral_logger.set_actions(actions)
                    except Exception:
                        pass
                ev = self._env.step(actions=actions if actions else None)
                steps_taken += 1
                self._distribute_personal_events(ev)

                converted_events = list(self._convert_events(ev))
                for e in converted_events:
                    self._log_event_full(e)
                    emitted.append(e)

                # Behavioral logger: per-tick advance fills outcome
                # fields on pending records and snapshots creature_tick
                # state. Done after env.step completes so positions and
                # creature populations reflect post-step state.
                if self._behavioral_logger is not None:
                    try:
                        self._behavioral_logger.advance_tick(
                            world=self, env=self._env,
                            tick=int(self._env.steps),
                            events={"events": converted_events},
                        )
                    except Exception as exc:
                        self._log_event("behavioral_logger_error",
                                        {"error": repr(exc)})

                if self.brain_manager:
                    rewards = self._approx_rewards_from_events(ev)
                    for cid, c in self._env.creatures.items():
                        if cid in pre_ids and cid in self.brain_manager:
                            new_obs = build_observation_for_creature(c, self._env)
                            self.brain_manager.get(cid).add_step(
                                new_obs, actions.get(cid, 9), rewards.get(cid, 0.0), False
                            )
                    # New births → allocate brains (with safe wrapper)
                    for cid in self._env.creatures:
                        if cid not in self.brain_manager and cid not in pre_ids:
                            self._safe_alloc_brain(cid, self._env.creatures[cid].genome)
                    self.brain_manager.reap(set(self._env.creatures.keys()))

                    # Phase 9.A invariant check
                    living = set(self._env.creatures.keys())
                    brain_ids = set(self.brain_manager.brains.keys())
                    orphans = brain_ids - living
                    if orphans:
                        self._log_event("brain_invariant_orphans", {"ids": list(orphans)})

                    train_result = self.brain_manager.round_robin_train()
                    if train_result is not None:
                        cid, loss = train_result
                        self._log_event_full({
                            "step": self._env.steps, "kind": "brain_trained",
                            "creature_id": cid, "loss": loss["loss"], "recon": loss["recon"],
                        })

                self.controllers.reap(set(self._env.creatures.keys()))
                self._record_genome_history()

                if not self._env.creatures:
                    self._log_event(kind="total_extinction")
                    break

            return {
                "steps_advanced": steps_taken,
                "current_step": self._env.steps,
                "events": emitted,
                "summary": self._brief_summary_locked(),
            }

    def _resolve_vocal_drivers(self, actions: dict[int, int]) -> None:
        """Decide who emits an audio wave this tick and write the pending
        slot on each Creature. Three drivers compete; only one fires:

        1. Reflex — high fear + RNG vs vocal_reflex_fear, suppressed if the
           creature already vocalized last tick (refractory; breaks echo
           loops). Wave shape comes from the genome's vocal_freq_bias.
        2. LLM — if an LLM controller produced an 8-bin wave intent on its
           last refresh, emit that. Amp is clipped by the genome's
           vocal_amplitude so loud bodies still set the ceiling.
        3. Brain — if the trained policy picked VOCALIZE on its own, emit
           the genome's vocal_freq_bias.

        For 1 and 2, this method also overrides ``actions[cid] = VOCALIZE``
        so the env.step action handler runs the emission path."""
        rng = self._action_rng
        for cid, c in self._env.creatures.items():
            # Reset driver tag from any prior tick.
            c.pending_driver = None
            # 1. Reflex
            if not c.vocalized_last_tick:
                fear_baseline = max(0.0, float(c.genome.fear_baseline))
                witnessed_danger = float(c.social_signal.get("danger", 0.0))
                effective_fear = fear_baseline + witnessed_danger
                if (
                    effective_fear > 0.7
                    and rng.random() < float(c.genome.vocal_reflex_fear)
                ):
                    c.pending_wave = c.genome.vocal_freq_bias()
                    c.pending_amp = float(c.genome.vocal_amplitude)
                    c.pending_driver = "reflex"
                    actions[cid] = ACTION_VOCALIZE
                    continue
            # 2. LLM intent
            ctrl = self.controllers.get(cid)
            if isinstance(ctrl, LLMController):
                intent = ctrl.current_vocal_intent()
                if intent is not None:
                    wave_list, amp = intent
                    if len(wave_list) == NUM_AUDIO_BINS:
                        c.pending_wave = np.asarray(wave_list, dtype=np.float32)
                        # Genome's vocal_amplitude caps how loud this body can
                        # speak — LLM intent doesn't get to override anatomy.
                        c.pending_amp = min(amp, float(c.genome.vocal_amplitude))
                        # Distinguish "llm" (Bedrock) from "random" (arm G's
                        # RandomEmitterController) via the controller's
                        # driver_kind tag.
                        c.pending_driver = getattr(ctrl, "driver_kind", "llm")
                        actions[cid] = ACTION_VOCALIZE
                        continue
            # 3. Brain chose VOCALIZE on its own
            if actions.get(cid) == ACTION_VOCALIZE:
                c.pending_wave = c.genome.vocal_freq_bias()
                c.pending_amp = float(c.genome.vocal_amplitude)
                c.pending_driver = "brain"

    def _approx_rewards_from_events(self, ev: StepEvents) -> dict[int, float]:
        r: dict[int, float] = {}
        for cid in ev.eats_food:
            r[cid] = r.get(cid, 0.0) + 1.0
        for eater, eaten in ev.eats_creature:
            r[eater] = r.get(eater, 0.0) + 1.5
            r[eaten] = r.get(eaten, 0.0) - 2.0
        for cid in ev.rests:
            r[cid] = r.get(cid, 0.0) + 0.05
        for prey, pred, dmg in ev.counter_attacks:
            r[prey] = r.get(prey, 0.0) + 0.3
        for cid in ev.builds:
            r[cid] = r.get(cid, 0.0) + 1.5
        for cid, _ in ev.gathers:
            r[cid] = r.get(cid, 0.0) + 0.1
        for cid, _matter in getattr(ev, "poisonings", []) or []:
            # Strong negative — selection should favour brains that learn to
            # avoid matter and DNA that discriminates innately.
            r[cid] = r.get(cid, 0.0) - 2.0
        return r

    def _brief_summary_locked(self) -> dict[str, Any]:
        return {
            "n_creatures": len(self._env.creatures),
            "n_food": int((self._env.grid == TILE_FOOD).sum()),
            "n_shelter": int((self._env.grid == TILE_SHELTER).sum()),
            "max_generation": max((c.generation for c in self._env.creatures.values()), default=0),
            "phenotype_counts": dict({
                ph: count
                for ph, count in self._env.population_summary().get("phenotype_counts", {}).items()
            }),
        }

    def _convert_events(self, ev: StepEvents) -> list[dict[str, Any]]:
        s = self._env.steps
        out: list[dict[str, Any]] = []
        for cid in ev.eats_food:
            out.append({"step": s, "kind": "ate_food", "creature_id": cid})
        for eater, eaten in ev.eats_creature:
            out.append({"step": s, "kind": "ate_creature", "eater_id": eater, "victim_id": eaten})
        for cid in ev.rests:
            out.append({"step": s, "kind": "rested", "creature_id": cid})
        for cid, kind in ev.gathers:
            out.append({"step": s, "kind": "gathered", "creature_id": cid, "resource": kind})
        for cid in ev.builds:
            out.append({"step": s, "kind": "built_shelter", "creature_id": cid})
        for cid in getattr(ev, "build_fails", []) or []:
            out.append({"step": s, "kind": "build_failed", "creature_id": cid})
        for cid, matter in getattr(ev, "poisonings", []) or []:
            out.append({"step": s, "kind": "poisoned",
                        "creature_id": cid, "matter": matter})
        for prey_id, pred_id, dmg in ev.counter_attacks:
            out.append({"step": s, "kind": "counter_attack",
                        "defender_id": prey_id, "attacker_id": pred_id,
                        "damage": round(float(dmg), 1)})
        for cid, cause in ev.deaths:
            out.append({"step": s, "kind": "death", "creature_id": cid, "cause": cause})
        for cid in ev.births:
            c = self._env.creatures.get(cid)
            out.append({
                "step": s, "kind": "birth",
                "creature_id": cid,
                "parent_a_id": c.parent_a_id if c else None,
                "parent_b_id": c.parent_b_id if c else None,
                "generation": c.generation if c else None,
                "phenotype": c.genome.phenotype() if c else None,
            })
        for pa, pb, child in ev.matings:
            out.append({"step": s, "kind": "mating",
                        "parent_a_id": pa, "parent_b_id": pb, "child_id": child})
        return out

    def _log_event(self, kind: str, details: dict[str, Any] | None = None) -> None:
        self._log_event_full({"step": self._env.steps, "kind": kind, **(details or {})})

    def _log_event_full(self, event: dict[str, Any]) -> None:
        self._event_history.append(event)
        if len(self._event_history) > MAX_EVENT_HISTORY:
            self._event_history = self._event_history[-MAX_EVENT_HISTORY:]

    def _record_genome_history(self, force: bool = False) -> None:
        if not force and (
            self._env.steps - self._last_genome_log_step < self._genome_history_every
        ):
            return
        from foresight.evolution.genome import TRAIT_NAMES
        s = self._env.population_summary()
        mg = s.get("mean_genome")
        mean = (
            {n: round(float(v), 4) for n, v in zip(TRAIT_NAMES, mg)}
            if mg is not None else None
        )
        recent_window = self._genome_history_every
        deaths = sum(1 for e in self._event_history[-200:]
                     if e.get("kind") == "death" and e.get("step", 0) >= self._env.steps - recent_window)
        kills = sum(1 for e in self._event_history[-200:]
                    if e.get("kind") == "ate_creature" and e.get("step", 0) >= self._env.steps - recent_window)
        births = sum(1 for e in self._event_history[-200:]
                     if e.get("kind") == "birth" and e.get("step", 0) >= self._env.steps - recent_window)
        entry = {
            "step": int(self._env.steps),
            "n_creatures": len(self._env.creatures),
            "n_food": int((self._env.grid == TILE_FOOD).sum()),
            "max_generation": s.get("max_generation", 0),
            "phenotype_counts": dict(s.get("phenotype_counts", {})),
            "recent_deaths": deaths,
            "recent_kills": kills,
            "recent_births": births,
            "mean_genome": mean,
        }
        self._genome_history.append(entry)
        if len(self._genome_history) > 400:
            self._genome_history = self._genome_history[-400:]
        self._last_genome_log_step = self._env.steps

    def get_genome_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._genome_history)

    # ------------------------------------------------------------------
    # Interventions
    # ------------------------------------------------------------------
    def add_food(self, x: int, y: int) -> dict[str, Any]:
        with self._lock:
            H, W = self._env.grid.shape
            if not (0 <= y < H and 0 <= x < W):
                return {"error": f"({x},{y}) out of bounds"}
            if int(self._env.grid[y, x]) != TILE_EMPTY:
                return {"error": f"cell ({x},{y}) not empty"}
            self._env.grid[y, x] = TILE_FOOD
            self._log_event("food_added", {"pos": [x, y]})
            return {"ok": True, "pos": [x, y]}

    def kill_creature(self, creature_id: int) -> dict[str, Any]:
        with self._lock:
            if creature_id not in self._env.creatures:
                return {"error": f"no living creature with id {creature_id}"}
            del self._env.creatures[creature_id]
            if self.brain_manager:
                self.brain_manager.on_death(creature_id)
            self.controllers._detach(creature_id)
            self._log_event("creature_killed_externally", {"creature_id": creature_id})
            return {"ok": True, "creature_id": creature_id}

    def introduce_creature(
        self, genome_overrides: dict[str, float] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            rng = np.random.default_rng(seed) if seed is not None else self._env._rng
            genome = Genome.random(rng)
            if genome_overrides:
                base = genome.traits()
                base.update({k: float(v) for k, v in genome_overrides.items() if k in base})
                genome = Genome(**base)
            new = self._env._spawn_creature(
                genome=genome, parent_ids=(None, None), generation=0
            )
            if self.brain_manager:
                self._safe_alloc_brain(new.individual_id, genome)
            self._log_event("creature_introduced", {
                "creature_id": new.individual_id,
                "pos": list(new.pos),
                "phenotype": genome.phenotype(),
                "genome": genome_to_dict(genome),
            })
            return {
                "ok": True,
                "creature_id": new.individual_id,
                "pos": list(new.pos),
                "phenotype": genome.phenotype(),
                "genome": genome_to_dict(genome),
            }

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        with self._lock:
            if seed is None:
                seed = int(np.random.default_rng().integers(0, 2**31 - 1))
            self._env = UnifiedWorld(WorldConfig(
                seed=seed,
                initial_creatures=self._cfg.initial_creatures,
                grid_size=self._cfg.grid_size,
            ))
            self._event_history = []
            if self.brain_manager is not None:
                self.brain_manager = CreatureBrainManager(
                    obs_dim=self._obs_dim, n_actions=self._n_actions, device=self._device, seed=seed,
                )
                for cid, c in self._env.creatures.items():
                    self._safe_alloc_brain(cid, c.genome)
            self.controllers = ControllerRegistry()
            self._genome_history = []
            self._last_genome_log_step = -10**9
            self._record_genome_history(force=True)
            self._log_event("world_reset", {"seed": seed})
            return {"ok": True, "seed": seed}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"env": copy.deepcopy(self._env), "events": list(self._event_history)}

    def restore(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._env = copy.deepcopy(snapshot["env"])
            self._event_history = list(snapshot["events"])
            if self.brain_manager is not None:
                self.brain_manager = CreatureBrainManager(
                    obs_dim=self._obs_dim, n_actions=self._n_actions, device=self._device, seed=0,
                )
                for cid, c in self._env.creatures.items():
                    self._safe_alloc_brain(cid, c.genome)
            self._log_event("restored_from_snapshot")
            return {"ok": True, "current_step": self._env.steps}

    # ------------------------------------------------------------------
    @property
    def step_count(self) -> int: return self._env.steps
    @property
    def n_creatures(self) -> int: return len(self._env.creatures)
    @property
    def alive(self) -> bool: return self.n_creatures > 0
    @property
    def obs_dim(self) -> int: return self._obs_dim
    @property
    def n_actions(self) -> int: return self._n_actions
    @property
    def grid_shape(self) -> tuple[int, int]: return tuple(self._env.grid.shape)
    # Back-compat aliases used by other no_free_signal modules
    @property
    def n_prey(self) -> int: return self.n_creatures
    @property
    def n_predators(self) -> int: return 0
