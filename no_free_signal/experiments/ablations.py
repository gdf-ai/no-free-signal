"""Ablation hooks for the seven experiment arms.

Each ablation is a function that takes a constructed World and mutates it
in place. All are idempotent and survive birth events that re-spawn brains
or auto-attach LLM controllers.

Hierarchy of hooks consulted by the World when spawning an LLM controller
(see ``no_free_signal.world``): ``world._llm_emit_logger`` (callable) and
``world._llm_wave_transform_factory`` (callable taking a creature, returning
a wave-transform callable or None). Set these once at the start of a run.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np

from foresight.evolution.genome import freeze_substrate_traits
from no_free_signal.experiments.llm_emit_logger import EmitLog
from no_free_signal.llm_controller import LLMController


# ---------------------------------------------------------------------------
# A: mute
# ---------------------------------------------------------------------------
def apply_mute(world: Any) -> None:
    """Clamp ``vocal_amplitude`` to 0 on every living creature and every
    future birth. Communication is mechanically impossible — the audio
    field stays empty regardless of what any driver tries to emit."""
    for c in world._env.creatures.values():
        _clamp_amplitude_zero(c.genome)
    world._post_birth_hooks = list(getattr(world, "_post_birth_hooks", []))
    world._post_birth_hooks.append(_clamp_amplitude_zero_on_birth)


def _clamp_amplitude_zero(genome: Any) -> None:
    # Genome is a frozen dataclass; bypass via object.__setattr__.
    object.__setattr__(genome, "vocal_amplitude", 0.0)


def _clamp_amplitude_zero_on_birth(world: Any, creature: Any) -> None:
    _clamp_amplitude_zero(creature.genome)


# ---------------------------------------------------------------------------
# B: fixed substrate
# ---------------------------------------------------------------------------
def apply_fixed_substrate(world: Any) -> None:
    """Freeze ``vocal_freq_*`` and ``audio_att_*`` so they don't drift
    under reproduction. Encoding and perception are stuck at whatever the
    initial population happened to start with."""
    freeze_substrate_traits()


# ---------------------------------------------------------------------------
# C: no LLM
# ---------------------------------------------------------------------------
def apply_no_llm(world: Any) -> None:
    """Disable LLM globally and prevent auto-attach so reflex + brain are
    the only drivers of vocalization."""
    from no_free_signal.llm_controller import set_llm_enabled
    set_llm_enabled(False)
    world._llm_auto_attach = False


# ---------------------------------------------------------------------------
# D: full stack — no ablation hook needed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# E: scrambled LLM
# ---------------------------------------------------------------------------
def apply_scrambled_llm(world: Any, seed: int) -> None:
    """Install a per-creature fixed wave-bin permutation. Each LLM emission
    is reordered by that permutation before caching. Reflex and brain
    emissions untouched. The permutation is fixed per individual so the
    speaker hears a self-consistent (if 'wrong') wave."""
    rng = random.Random(seed)
    permutations: dict[int, list[int]] = {}

    def factory(creature: Any):
        cid = int(creature.individual_id)
        if cid not in permutations:
            perm = list(range(8))
            rng.shuffle(perm)
            permutations[cid] = perm
        perm = permutations[cid]

        def transform(wave: list[float]) -> list[float]:
            return [float(wave[i]) for i in perm]

        return transform

    world._llm_wave_transform_factory = factory


# ---------------------------------------------------------------------------
# F: context-randomized LLM
# ---------------------------------------------------------------------------
def apply_context_randomized_llm(world: Any, seed: int, emit_log: EmitLog) -> None:
    """At emission time, replace the LLM's wave with a random previous
    emission from the same run. Same LLM call rate as D — only the routing
    changes. Tests whether substrate alignment depends on context-sensitive
    LLM contribution or just on a consistent LLM-shaped wave source."""
    rng = random.Random(seed)

    def factory(creature: Any):
        def transform(wave: list[float]) -> list[float]:
            recent = emit_log.recent_waves(n=200)
            if not recent:
                # First emission of the run — fall through unchanged so the
                # buffer has something to seed from. After that, every
                # emission gets a random replacement.
                return wave
            return list(rng.choice(recent))

        return transform

    world._llm_wave_transform_factory = factory


# ---------------------------------------------------------------------------
# G: random emitter (no LLM)
# ---------------------------------------------------------------------------
class RandomEmitterController(LLMController):
    """Drop-in replacement for LLMController in arm G. Same surface so
    ``World._resolve_vocal_drivers`` recognises it via ``isinstance``,
    but every refresh synthesises a uniform-random 8-bin wave and a
    Gaussian-sampled amplitude instead of calling Bedrock. No API cost.

    The action returned each refresh is uniform-random across the 10
    actions — arm G has no LLM-driven action either, so D vs G bounds
    *both* the LLM's emission contribution and its action contribution.
    """

    driver_kind = "random"

    def __init__(self, *args: Any, rng_seed: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rng = random.Random(rng_seed ^ self.creature_id)

    def _maybe_kick_refresh(self, obs: Any) -> None:
        """No I/O, so synthesize inline rather than spawning a thread per
        refresh window. Avoids a thread storm with a 25-creature population."""
        with self._refresh_lock:
            if self._closed.is_set():
                return
            self._last_refresh_tick = self._tick_count
        self._do_refresh({"step": self._tick_count})

    def _do_refresh(self, snap: dict) -> None:
        rng = self._rng
        wave = [rng.uniform(0.0, 1.0) for _ in range(8)]
        # Amplitude matched to D's empirical distribution (mean ~0.5,
        # narrow spread). Could be re-fit from a held-out D emit log if
        # we wanted exact distributional matching.
        amp = max(0.1, min(0.9, rng.gauss(0.5, 0.15)))
        if self._wave_transform is not None:
            try:
                wave = list(self._wave_transform(list(wave)))
            except Exception:
                pass
        if self._emit_logger is not None:
            try:
                self._emit_logger(snap["step"], self.creature_id, list(wave), amp)
            except Exception:
                pass
        action = rng.randrange(10)
        with self._cache_lock:
            self._cached_action = action
            self._cached_confidence = 0.5
            self._cached_thought = "(arm G: random emitter, no LLM)"
            self._cached_say = ""
            self._cached_vocal_wave = wave
            self._cached_vocal_amp = amp
        self._record_log({
            "step": snap["step"], "kind": "random_emit",
            "thought": "(arm G synthetic)", "action": "random",
            "confidence": 0.5, "say": "", "ms": 0,
        })


def apply_random_emitter(
    world: Any, seed: int, refresh_every_ticks: int,
    emit_log: EmitLog,
) -> None:
    """Arm G. Disable LLM auto-attach (no API calls), then attach a
    :class:`RandomEmitterController` to every creature. Newborns get one
    via a post-birth hook so the emitter survives reproduction.

    Cadence (``refresh_every_ticks``) and the emit logger are matched to
    arm D so that comparing D vs G isolates the LLM's contribution
    (action + emission shape together) against frequency- and amplitude-
    matched noise."""
    world._llm_auto_attach = False
    personality = "(random emitter — no LLM)"

    def _attach(creature: Any) -> None:
        cid = int(creature.individual_id)
        if cid in world.controllers:
            return
        ctrl = RandomEmitterController(
            creature=creature,
            personality=personality,
            lambda_advisory=1.0,
            refresh_every_ticks=refresh_every_ticks,
            world_ref=world,
            emit_logger=emit_log,
            wave_transform=None,
            rng_seed=seed,
        )
        world.controllers.attach(cid, ctrl)

    for c in list(world._env.creatures.values()):
        _attach(c)

    def _on_birth(_world: Any, creature: Any) -> None:
        _attach(creature)

    world._post_birth_hooks = list(getattr(world, "_post_birth_hooks", []))
    world._post_birth_hooks.append(_on_birth)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def apply_arm(
    world: Any, arm: str, seed: int, emit_log: EmitLog,
    *, refresh_every_ticks: int = 15,
) -> None:
    """Apply the ablation hooks for the named arm. Must be called *after*
    the world is constructed and *before* any LLM controllers are spawned
    so the hooks are in place when ``_safe_alloc_brain`` reads them."""
    arm = arm.upper()
    if arm == "A":
        apply_mute(world)
        apply_no_llm(world)
    elif arm == "B":
        apply_fixed_substrate(world)
    elif arm == "C":
        apply_no_llm(world)
    elif arm == "D":
        pass  # full stack
    elif arm == "E":
        apply_scrambled_llm(world, seed)
    elif arm == "F":
        apply_context_randomized_llm(world, seed, emit_log)
    elif arm == "G":
        apply_random_emitter(world, seed, refresh_every_ticks, emit_log)
    else:
        raise ValueError(f"unknown arm: {arm!r}")
