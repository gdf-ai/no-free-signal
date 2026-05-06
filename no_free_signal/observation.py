"""Raw sensorimotor observations for creatures + server-side rendering."""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

from foresight.envs.unified_world import (
    NUM_TILE_TYPES,
    Creature,
    TILE_EMPTY,
    TILE_FOOD,
    TILE_RAW_STONE,
    TILE_RAW_WOOD,
    TILE_SHELTER,
    TILE_WALL,
)

TILE_RGB = {
    TILE_EMPTY:    (232, 232, 232),
    TILE_WALL:     (34, 34, 34),
    TILE_FOOD:     (76, 175, 80),
    TILE_SHELTER:  (33, 150, 243),
    TILE_RAW_WOOD: (139, 90, 43),
    TILE_RAW_STONE:(120, 120, 130),
}


@dataclass
class RawObservation:
    obs_window: np.ndarray
    drives: dict[str, float]
    extras: dict[str, float]
    flat: np.ndarray


def build_observation_for_creature(c: Creature, env) -> RawObservation:
    cfg = env.config
    w = cfg.obs_window
    half = w // 2
    ar, ac = c.pos
    H, W = env.grid.shape

    # NOTE: extra channel for "creature in this cell" so brains can perceive
    # other creatures.
    n_channels = NUM_TILE_TYPES + 1
    spatial = np.zeros((w, w, n_channels), dtype=np.float32)
    for dr in range(-half, half + 1):
        for dc in range(-half, half + 1):
            rr, cc = ar + dr, ac + dc
            wr, wc = dr + half, dc + half
            if 0 <= rr < H and 0 <= cc < W:
                spatial[wr, wc, int(env.grid[rr, cc])] = 1.0
            else:
                spatial[wr, wc, TILE_WALL] = 1.0

    n_others_visible = 0
    for other in env.creatures.values():
        if other.individual_id == c.individual_id:
            continue
        dr, dc = other.pos[0] - ar, other.pos[1] - ac
        if -half <= dr <= half and -half <= dc <= half:
            spatial[dr + half, dc + half, NUM_TILE_TYPES] = 1.0  # creature channel
            n_others_visible += 1

    drives = {
        "energy": float(c.energy) / 100.0,
        "fatigue": float(c.fatigue) / 100.0,
        "health": float(c.health) / 100.0,
    }
    # Is non-food matter adjacent? Used by the brain's choose_action to apply
    # a poison-risk penalty on EAT, gated by genome food_discrimination.
    matter_adjacent = 0.0
    for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = ar + dr, ac + dc
        if 0 <= rr < H and 0 <= cc < W:
            t = int(env.grid[rr, cc])
            if t == TILE_RAW_WOOD or t == TILE_RAW_STONE:
                matter_adjacent = 1.0
                break

    extras = {
        "n_others_visible": float(n_others_visible) / max(1, len(env.creatures) or 1),
        "x_norm": float(ac) / cfg.grid_size,
        "y_norm": float(ar) / cfg.grid_size,
        "inventory_wood": float(c.inventory_wood) / 10.0,
        "inventory_stone": float(c.inventory_stone) / 10.0,
        "matter_adjacent": matter_adjacent,
    }
    # Surface the vicarious-reward signal so policies can read it. We do NOT
    # add these to `flat` (would change obs_dim and break existing brains);
    # the brain reads them as scalar context in choose_action.
    for k, v in c.social_signal.items():
        extras[f"social_{k}"] = float(v)
    # Audio: sample the per-tile field and gate by per-bin attention. Each
    # creature's attention vector co-evolves with the population's vocal
    # frequency biases under selection pressure.
    audio_field = getattr(env, "audio", None)
    if audio_field is None:
        audio_bins = np.zeros(8, dtype=np.float32)
    else:
        raw_audio = audio_field.sample(c.pos)
        audio_bins = (raw_audio * c.genome.audio_attention()).astype(np.float32)
    for i, v in enumerate(audio_bins):
        extras[f"audio_{i}"] = float(v)
    flat = np.concatenate([
        spatial.reshape(-1),
        np.array([
            drives["energy"], drives["fatigue"], drives["health"],
            extras["n_others_visible"], extras["x_norm"], extras["y_norm"],
            extras["inventory_wood"], extras["inventory_stone"],
        ], dtype=np.float32),
        audio_bins,
    ]).astype(np.float32)
    return RawObservation(obs_window=spatial, drives=drives, extras=extras, flat=flat)


def render_world_png(env, scale: int = 16) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise RuntimeError("Pillow required") from e

    grid = env.grid
    H, W = grid.shape
    img = Image.new("RGB", (W * scale, H * scale))
    pixels = img.load()
    for r in range(H):
        for c in range(W):
            color = TILE_RGB.get(int(grid[r, c]), (255, 0, 255))
            x0, y0 = c * scale, r * scale
            for dy in range(scale):
                for dx in range(scale):
                    pixels[x0 + dx, y0 + dy] = color

    draw = ImageDraw.Draw(img)
    for cr in env.creatures.values():
        cy, cx = cr.pos
        x0, y0 = cx * scale, cy * scale
        # Phenotype-driven colour
        ph = cr.genome.phenotype()
        if ph == "predator-like":
            colour = (244, 63, 94)
        elif ph == "forager":
            colour = (255, 235, 59)
        elif ph == "omnivore":
            colour = (251, 146, 60)
        else:
            colour = (180, 180, 200)
        draw.ellipse(
            [(x0 + 2, y0 + 2), (x0 + scale - 2, y0 + scale - 2)],
            fill=colour, outline=(0, 0, 0), width=1,
        )

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_world_array(env, scale: int = 16) -> np.ndarray:
    from PIL import Image
    png = render_world_png(env, scale=scale)
    img = Image.open(io.BytesIO(png))
    return np.asarray(img.convert("RGB"))
