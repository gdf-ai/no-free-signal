"""JSON-friendly serialisers for the unified-Creature world."""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from foresight.envs.unified_world import (
    NUM_TILE_TYPES,
    Creature,
    TILE_FOOD,
    TILE_RAW_STONE,
    TILE_RAW_WOOD,
    TILE_SHELTER,
    TILE_WALL,
)
from foresight.evolution.genome import TRAIT_NAMES


def genome_to_dict(genome) -> dict[str, float]:
    return {k: round(float(v), 4) for k, v in genome.traits().items()}


def creature_to_dict(
    c: Creature, verbose: bool = False, env=None,
    brain_stats: dict[str, Any] | None = None,
    llm_thought: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": c.individual_id,
        "pos": list(c.pos),
        "phenotype": c.genome.phenotype(),
        "energy": round(float(c.energy), 1),
        "fatigue": round(float(c.fatigue), 1),
        "health": round(float(c.health), 1),
        "age": c.age,
        "generation": c.generation,
        "parent_a_id": c.parent_a_id,
        "parent_b_id": c.parent_b_id,
        "n_offspring": c.n_offspring,
        "food_eaten_total": c.food_eaten_total,
        "creatures_eaten_total": c.creatures_eaten_total,
        "inventory": {"wood": c.inventory_wood, "stone": c.inventory_stone},
        "social_signal": {k: round(float(v), 3) for k, v in c.social_signal.items()},
    }
    if verbose:
        base["genome"] = genome_to_dict(c.genome)
        if env is not None:
            base["nearby"] = _nearby_info(c, env)
    if brain_stats is not None:
        base["brain"] = brain_stats
    # llm_thought may carry {action, thought, confidence, say, calls_made}.
    # Always include the field (with attached=False) when caller passed the
    # llm_state map at all, so the UI can distinguish "no LLM" from "LLM
    # attached but hasn't thought yet".
    if llm_thought is not None:
        base["llm"] = {"attached": True, **llm_thought}
    return base


def _nearby_info(c: Creature, env) -> dict[str, Any]:
    pos = c.pos
    out: dict[str, Any] = {}
    others = []
    for o in env.creatures.values():
        if o.individual_id == c.individual_id:
            continue
        d = max(abs(o.pos[0] - pos[0]), abs(o.pos[1] - pos[1]))
        if d <= 6:
            others.append({
                "id": o.individual_id,
                "distance": d,
                "phenotype": o.genome.phenotype(),
            })
    out["other_creatures"] = sorted(others, key=lambda t: t["distance"])

    rs, cs = np.where(env.grid == TILE_FOOD)
    food = []
    for r, cc in zip(rs.tolist(), cs.tolist()):
        d = max(abs(r - pos[0]), abs(cc - pos[1]))
        if d <= 6:
            food.append({"pos": [int(cc), int(r)], "distance": d})
    out["food_in_range"] = sorted(food, key=lambda t: t["distance"])[:5]
    out["on_shelter"] = bool(env.grid[pos] == TILE_SHELTER)
    return out


def world_to_dict(
    env,
    recent_events: list | None = None,
    llm_state: dict[int, dict] | None = None,
    brain_state: dict[int, dict] | None = None,
) -> dict[str, Any]:
    grid = env.grid
    n_food = int((grid == TILE_FOOD).sum())
    n_shelter = int((grid == TILE_SHELTER).sum())
    n_wood = int((grid == TILE_RAW_WOOD).sum())
    n_stone = int((grid == TILE_RAW_STONE).sum())
    n_walls = int((grid == TILE_WALL).sum())

    creatures = list(env.creatures.values())
    creature_list = [
        creature_to_dict(
            c,
            brain_stats=(brain_state or {}).get(c.individual_id),
            llm_thought=(llm_state or {}).get(c.individual_id),
        )
        for c in creatures
    ]
    summary = env.population_summary()

    mean_genome = (
        {n: round(float(v), 4) for n, v in zip(TRAIT_NAMES, summary["mean_genome"])}
        if summary.get("mean_genome") is not None else None
    )

    # Niche-based aggregate genomes (the "predator gene pool" replacement)
    by_phenotype: dict[str, list[np.ndarray]] = {}
    for c in creatures:
        by_phenotype.setdefault(c.genome.phenotype(), []).append(c.genome.to_array())
    niche_means: dict[str, dict[str, float]] = {}
    for ph, arrs in by_phenotype.items():
        m = np.stack(arrs).mean(axis=0)
        niche_means[ph] = {n: round(float(v), 4) for n, v in zip(TRAIT_NAMES, m)}

    return {
        "step": env.steps,
        "grid": {
            "size": list(grid.shape),
            "cells": grid.astype(int).tolist(),
            "n_food": n_food,
            "n_shelter": n_shelter,
            "n_wood": n_wood,
            "n_stone": n_stone,
            "n_walls": n_walls,
        },
        "populations": {
            "n_creatures": len(creatures),
            "phenotype_counts": dict(summary.get("phenotype_counts", {})),
            "max_generation": summary.get("max_generation", 0),
            "max_age": summary.get("max_age", 0),
        },
        "mean_genome": mean_genome,
        "niche_means": niche_means,
        "creatures": creature_list,
        "recent_events": recent_events or [],
    }
