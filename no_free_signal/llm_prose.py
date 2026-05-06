"""Genome and observation -> natural-language prose for the LLM controller.

Pure transformations; no I/O, no model calls. Used by `LLMController` to build
a prompt that tells Claude what kind of creature it's inhabiting and what the
creature is currently sensing.
"""
from __future__ import annotations

import json
import re
from typing import Any

from foresight.envs.unified_world import (
    Creature,
    TILE_FOOD,
    TILE_RAW_STONE,
    TILE_RAW_WOOD,
    TILE_SHELTER,
    TILE_WALL,
)
from foresight.evolution.genome import Genome
from no_free_signal.observation import RawObservation


# Action enum the LLM is asked to choose. Order MUST match unified_world ACTION_*
ACTION_NAMES: tuple[str, ...] = (
    "NORTH", "SOUTH", "EAST", "WEST", "EAT", "REST", "GATHER", "BUILD",
    "VOCALIZE", "NOOP",
)
ACTION_BY_NAME: dict[str, int] = {n: i for i, n in enumerate(ACTION_NAMES)}


def _bucket(v: float, low: float, high: float) -> str:
    if v < low: return "low"
    if v > high: return "high"
    return "moderate"


def genome_to_prose(g: Genome) -> str:
    """Translate the 27-trait genome into a short paragraph an LLM can use as
    a personality grounding. We focus on traits that produce *behavioural*
    signal: drives, reward weights, niche, and a few biology-shaping traits."""
    t = g.traits()
    parts: list[str] = []
    parts.append(f"You are a {g.phenotype()} creature.")

    # Niche / diet
    if t["predate_drive"] > 0.55 and t["forage_drive"] < 0.4:
        parts.append("You hunt other creatures to survive.")
    elif t["predate_drive"] < 0.2 and t["forage_drive"] > 0.55:
        parts.append("You forage for plant food and avoid combat.")
    elif t["predate_drive"] > 0.4 and t["forage_drive"] > 0.4:
        parts.append("You are an opportunistic omnivore — you eat both plants and other creatures.")
    else:
        parts.append("Your diet preferences are weak.")

    # Affective drives
    drive_phrases = []
    if t["fear_baseline"] > 0.5: drive_phrases.append("naturally fearful")
    elif t["fear_baseline"] < -0.5: drive_phrases.append("unusually bold")
    if t["seeking_baseline"] > 0.5: drive_phrases.append("curious and exploratory")
    elif t["seeking_baseline"] < -0.5: drive_phrases.append("conservative, sticking close to known places")
    if t["play_baseline"] > 0.5: drive_phrases.append("playful")
    if t["rage_baseline"] > 0.5: drive_phrases.append("quick to anger")
    if drive_phrases:
        parts.append("Temperament: " + ", ".join(drive_phrases) + ".")

    # Reward centres — what the creature *wants*
    rewards = {
        "safety": t["reward_safety"],
        "reproduction": t["reward_reproduction"],
        "comfort": t["reward_comfort"],
        "satiation": t["reward_hunger"],
        "novelty": t["reward_happiness"],
    }
    top = sorted(rewards.items(), key=lambda kv: -kv[1])[:2]
    parts.append(
        f"What you care about most: {top[0][0]} (weight {top[0][1]:.2f}) and "
        f"{top[1][0]} (weight {top[1][1]:.2f})."
    )

    # Biology
    bio: list[str] = []
    if t["armor"] > 0.3: bio.append("thick armoured hide")
    if t["camouflage"] > 0.3: bio.append("good camouflage")
    if t["attack_strength"] > 10: bio.append("powerful bite/claws")
    elif t["attack_strength"] > 4: bio.append("moderate weapons")
    if t["vision_range"] > 11: bio.append("sharp long-range vision")
    elif t["vision_range"] < 5: bio.append("poor eyesight")
    if t["chase_speed"] > 0.7: bio.append("fast")
    if t["food_discrimination"] > 0.7: bio.append("can tell food from poisonous matter")
    elif t["food_discrimination"] < 0.3: bio.append("often mistakes wood/stone for food")
    if bio:
        parts.append("Body: " + ", ".join(bio) + ".")

    parts.append(
        f"Aversions — you instinctively shy away from danger (creature-aversion {t['creature_aversion']:.2f}, "
        f"hunger-aversion {t['hunger_aversion']:.2f})."
    )
    return " ".join(parts)


_TILE_NAMES = {
    TILE_FOOD: "food",
    TILE_SHELTER: "shelter",
    TILE_RAW_WOOD: "raw wood",
    TILE_RAW_STONE: "raw stone",
    TILE_WALL: "wall",
}


def obs_to_prose(
    obs: RawObservation,
    creature: Creature,
    env: Any,
    recent_personal_events: list[dict] | None = None,
) -> str:
    """Build a one-paragraph situation report for the prompt."""
    parts: list[str] = []
    e = obs.drives.get("energy", 1.0)
    h = obs.drives.get("health", 1.0)
    f = obs.drives.get("fatigue", 0.0)
    parts.append(
        f"You are at grid position ({creature.pos[1]},{creature.pos[0]}). "
        f"Energy {e*100:.0f}/100, fatigue {f*100:.0f}/100, health {h*100:.0f}/100. "
        f"Inventory: {creature.inventory_wood} wood, {creature.inventory_stone} stone."
    )

    # Tiles in the 7x7 obs window — count by type and direction
    spatial = obs.obs_window  # (W, W, n_channels)
    half = spatial.shape[0] // 2
    here_tile_layer = spatial[half, half]
    nearby_counts: dict[str, int] = {}
    for tile_id, name in _TILE_NAMES.items():
        c = int(spatial[..., tile_id].sum())
        if c > 0:
            nearby_counts[name] = c
    if nearby_counts:
        bits = ", ".join(f"{n}× {name}" for name, n in sorted(nearby_counts.items()))
        parts.append(f"Around you ({spatial.shape[0]}-tile window): {bits}.")

    # Other creatures with concrete IDs / phenotypes / distances so the LLM
    # has someone specific to address in its `say` field.
    if env is not None:
        nearby = []
        my_pos = creature.pos
        try:
            vision = float(creature.genome.traits().get("vision_range", 7.0))
        except Exception:
            vision = 7.0
        for o in env.creatures.values():
            if o.individual_id == creature.individual_id:
                continue
            d = max(abs(o.pos[0] - my_pos[0]), abs(o.pos[1] - my_pos[1]))
            if d <= vision:
                nearby.append((d, o))
        nearby.sort(key=lambda t: t[0])
        if nearby:
            who = ", ".join(
                f"c{o.individual_id} ({o.genome.phenotype()}, {d} away)"
                for d, o in nearby[:6]
            )
            parts.append(f"Other creatures within sight: {who}.")

    # Standing on?
    if creature.pos and env is not None:
        cur = int(env.grid[creature.pos])
        if cur in _TILE_NAMES:
            parts.append(f"You are standing on {_TILE_NAMES[cur]}.")

    # Vicarious / social signal
    soc = creature.social_signal
    soc_active = sorted(((k, v) for k, v in soc.items() if v > 0.15), key=lambda kv: -kv[1])
    if soc_active:
        labels = {
            "food": "saw another eat food",
            "predation": "saw a kill",
            "build": "saw a successful build",
            "gather": "saw a gather",
            "mate": "saw a mating",
            "rest": "saw another rest",
            "danger": "sensed danger",
        }
        parts.append("Recent observation: " + ", ".join(
            f"{labels.get(k, k)} ({v:.2f})" for k, v in soc_active
        ) + ".")

    # Heard audio — present the 8-bin field at this tile (post-attention). We
    # do NOT label the bins. Their meaning is whatever the population
    # converges on under selection pressure.
    audio_vec = [obs.extras.get(f"audio_{i}", 0.0) for i in range(8)]
    if any(v > 0.02 for v in audio_vec):
        bins_str = ", ".join(f"{v:.2f}" for v in audio_vec)
        parts.append(f"Audio at your tile (8 bins, no fixed meaning): [{bins_str}].")

    # Recent personal sensations (pain / pleasure / discovery from prior turns).
    # Sentences come from `World._distribute_personal_events` and read like
    # "Your stomach burns. This was NOT food." — the LLM should let these
    # condition its next action choice and what it chooses to say.
    if recent_personal_events:
        recent = recent_personal_events[-3:]
        sentences = [
            e.get("sentence", e.get("kind", "?")) for e in recent
        ]
        parts.append("Recent: " + " ".join(sentences))

    return " ".join(parts)


def parse_llm_response(text: str) -> dict:
    """Robustly extract {action, thought, confidence} from the LLM output.
    Falls back to NOOP / 0.0 confidence if parsing fails."""
    fallback = {"action": "NOOP", "action_int": ACTION_BY_NAME["NOOP"],
                "thought": "(parse error)", "confidence": 0.0,
                "raw": text[:300]}
    if not text:
        return fallback
    # Try to find a JSON object in the response.
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return fallback
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    action = str(data.get("action", "NOOP")).upper().strip()
    if action not in ACTION_BY_NAME:
        # Tolerate common aliases the LLM might emit.
        action_aliases = {
            "MOVE_NORTH": "NORTH", "UP": "NORTH",
            "MOVE_SOUTH": "SOUTH", "DOWN": "SOUTH",
            "MOVE_EAST":  "EAST",  "RIGHT": "EAST",
            "MOVE_WEST":  "WEST",  "LEFT": "WEST",
            "WAIT": "NOOP", "IDLE": "NOOP", "PASS": "NOOP",
            "EAT_FOOD": "EAT", "ATTACK": "EAT",
            "SLEEP": "REST",
            "HARVEST": "GATHER", "PICK": "GATHER",
            "CONSTRUCT": "BUILD",
        }
        action = action_aliases.get(action, "NOOP")
    confidence = float(data.get("confidence", 0.6))
    confidence = max(0.0, min(1.0, confidence))
    say = str(data.get("say", "") or "").strip()[:80]
    # Audio: 8-bin frequency vector + amplitude. The model is asked to
    # always emit both. If the wave is missing or malformed but the
    # amplitude is non-zero, fall back to a zero-vector wave (the
    # downstream vocalize handler treats zero-amp / zero-wave as no-op
    # anyway, so this is a graceful degradation rather than a silent drop).
    vocal_wave_raw = data.get("vocalize_wave")
    vocal_amp_raw = data.get("vocalize_amp", 0.0)
    vocal_wave: list[float] | None = None
    vocal_amp: float = 0.0
    if isinstance(vocal_wave_raw, list) and len(vocal_wave_raw) == 8:
        try:
            vocal_wave = [max(0.0, min(1.0, float(x))) for x in vocal_wave_raw]
        except (TypeError, ValueError):
            vocal_wave = None
    try:
        vocal_amp = max(0.0, min(1.0, float(vocal_amp_raw)))
    except (TypeError, ValueError):
        vocal_amp = 0.0
    # If amplitude is meaningful but the wave came back null/malformed,
    # use zeros so we don't silently drop the call. amp <= 0 still means
    # no emission downstream.
    if vocal_wave is None and vocal_amp > 0.0:
        vocal_wave = [0.0] * 8
    return {
        "action": action,
        "action_int": ACTION_BY_NAME[action],
        "thought": str(data.get("thought", "")).strip()[:400],
        "confidence": confidence,
        "say": say,
        "vocal_wave": vocal_wave,
        "vocal_amp": vocal_amp,
        "raw": text[:300],
    }
