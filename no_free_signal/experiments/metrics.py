"""Per-snapshot metrics for the experimental harness.

Each call to :func:`compute_snapshot` returns a flat dict ready for
JSONL serialization. Heavy / population-level statistics (bootstrap CIs,
functional effect with windows) are computed offline by ``plot.py`` from
the snapshot stream and the event history; this module focuses on the
running state at a moment in time.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from no_free_signal.experiments.llm_emit_logger import EmitLog


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in bits over a vector treated as a distribution.
    Renormalizes to sum 1; clamps zeros before log."""
    s = float(p.sum())
    if s <= 0:
        return 0.0
    q = np.clip(p / s, 1e-9, 1.0)
    return float(-(q * np.log2(q)).sum())


def compute_snapshot(world: Any, emit_log: EmitLog, tick: int) -> dict[str, Any]:
    creatures = list(world._env.creatures.values())
    n = len(creatures)
    if n == 0:
        return {
            "tick": int(tick),
            "n_creatures": 0,
            "extinction": True,
        }

    # Population-level mean trait vectors.
    audio_atts = np.stack([c.genome.audio_attention() for c in creatures])
    freq_biases = np.stack([c.genome.vocal_freq_bias() for c in creatures])
    mean_attention = audio_atts.mean(axis=0).astype(np.float64)
    mean_reflex = freq_biases.mean(axis=0).astype(np.float64)

    # LLM emission mean (post-transform, including any scramble or replay).
    llm_mean = emit_log.mean_emission()

    # Audio field statistics — saturation indicator.
    field_tiles = list(world._env.audio._tiles.values())
    if field_tiles:
        stacked = np.stack(field_tiles)
        field_active = int(stacked.shape[0])
        field_max = float(stacked.max())
        field_total_per_bin = stacked.sum(axis=0)
        field_entropy_bits = _entropy(field_total_per_bin)
    else:
        field_active = 0
        field_max = 0.0
        field_entropy_bits = 0.0

    # Lifespan / fitness — read directly from living creatures (mean age).
    ages = np.array([c.age for c in creatures], dtype=np.float64)
    energies = np.array([c.energy for c in creatures], dtype=np.float64)
    healths = np.array([c.health for c in creatures], dtype=np.float64)

    # Death log — count deaths in the recent event window and average their
    # ages-at-death if available. The event history records cause and id;
    # we cross-ref a separate `_age_at_death` map maintained by the harness
    # (or fall back to a zero count if not).
    age_at_death = getattr(world, "_age_at_death", {})
    recent_window = 200
    recent_death_ages = [
        age for (recorded_tick, age) in age_at_death.values()
        if tick - recorded_tick <= recent_window
    ]
    mean_age_at_death_recent = (
        float(np.mean(recent_death_ages)) if recent_death_ages else 0.0
    )

    # Counters maintained by the harness in `world._counters` if present:
    #   "vocalize_actions": int  total VOCALIZE actions taken across all creatures
    #   "vocalize_emit_with_wave": int  subset that actually emitted (had wave + amp + energy)
    counters = getattr(world, "_counters", {})

    # Specialization — entropy of which creatures have called recently.
    # If the harness tracks per-creature call counts, use them; otherwise 0.
    per_cid_calls = counters.get("per_cid_call_counts", {})
    if per_cid_calls:
        call_counts = np.array(list(per_cid_calls.values()), dtype=np.float64)
        specialization_entropy_bits = _entropy(call_counts)
    else:
        specialization_entropy_bits = 0.0

    snap: dict[str, Any] = {
        "tick": int(tick),
        "n_creatures": int(n),
        "mean_age": float(ages.mean()),
        "max_age": float(ages.max()),
        "mean_energy": float(energies.mean()),
        "mean_health": float(healths.mean()),
        "mean_age_at_death_recent": mean_age_at_death_recent,
        "deaths_recent": int(len(recent_death_ages)),
        "mean_audio_attention": mean_attention.tolist(),
        "mean_vocal_freq_bias": mean_reflex.tolist(),
        "mean_vocal_amplitude": float(np.mean([c.genome.vocal_amplitude for c in creatures])),
        "llm_emit_count": int(emit_log.total_calls()),
        "llm_emit_mean": llm_mean.tolist(),
        "cos_attention_vs_llm": _cos(mean_attention, llm_mean),
        "cos_attention_vs_reflex": _cos(mean_attention, mean_reflex),
        "audio_field_active_tiles": field_active,
        "audio_field_max_amp": field_max,
        "audio_field_entropy_bits": field_entropy_bits,
        "vocalize_actions_total": int(counters.get("vocalize_actions", 0)),
        "vocalize_emits_total": int(counters.get("vocalize_emits", 0)),
        "specialization_entropy_bits": specialization_entropy_bits,
    }
    return snap
