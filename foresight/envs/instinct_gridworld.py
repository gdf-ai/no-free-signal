"""Instinct Gridworld: a biology-grounded foraging environment.

A 32x32 grid with food, shelter, and predators. The agent has hunger, fatigue,
and health that evolve over time. Designed to engage Panksepp's primary affects
(SEEKING via food, FEAR via predators, PLAY when FEAR is low, RAGE when
cornered) and homeostasis (hunger/fatigue regulation).

Pure Python + Gymnasium 1.0 API; no rendering required for training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Tile encoding (also channel index in the observation tensor)
TILE_EMPTY = 0
TILE_WALL = 1
TILE_FOOD = 2
TILE_SHELTER = 3
TILE_PREDATOR = 4
NUM_TILE_TYPES = 5

# Actions
ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_EAT = 4
ACTION_REST = 5
ACTION_NOOP = 6
NUM_ACTIONS = 7

_MOVE_DELTAS = {
    ACTION_NORTH: (-1, 0),
    ACTION_SOUTH: (1, 0),
    ACTION_EAST: (0, 1),
    ACTION_WEST: (0, -1),
}


@dataclass
class InstinctGridworldConfig:
    grid_size: int = 32
    obs_window: int = 7  # must be odd
    n_predators: int = 1
    n_food_initial: int = 8
    n_shelter: int = 4
    food_respawn_prob: float = 0.005  # per checked cell per step
    max_food: int = 12
    hunger_rate: float = 1.0
    fatigue_rate_move: float = 2.0
    fatigue_rate_rest: float = -5.0  # negative => restores
    health_loss_starving: float = 1.0
    health_loss_predator: float = 25.0
    food_hunger_restore: float = 40.0
    step_penalty: float = -0.01
    food_reward: float = 1.0
    damage_reward: float = -2.0
    death_reward: float = -10.0
    max_steps: int = 1000
    seed: int | None = None


class InstinctGridworldEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(self, config: InstinctGridworldConfig | None = None):
        super().__init__()
        self.config = config or InstinctGridworldConfig()
        cfg = self.config
        if cfg.obs_window % 2 == 0:
            raise ValueError("obs_window must be odd so it has a center cell")

        self.action_space = spaces.Discrete(NUM_ACTIONS)
        spatial_dim = cfg.obs_window * cfg.obs_window * NUM_TILE_TYPES
        # extras: hunger, fatigue, health, predator_visible_count, x_norm, y_norm
        self._extra_dim = 6
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(spatial_dim + self._extra_dim,),
            dtype=np.float32,
        )

        self._rng: np.random.Generator | None = None
        self._grid: np.ndarray
        self._predators: list[tuple[int, int]] = []
        self._agent: tuple[int, int] = (0, 0)
        self._hunger: float = 0.0
        self._fatigue: float = 0.0
        self._health: float = 100.0
        self._steps: int = 0
        self._terminated: bool = False
        self._truncated: bool = False

        self.reset(seed=cfg.seed)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        cfg = self.config
        self._grid = np.zeros((cfg.grid_size, cfg.grid_size), dtype=np.int32)
        self._grid[0, :] = TILE_WALL
        self._grid[-1, :] = TILE_WALL
        self._grid[:, 0] = TILE_WALL
        self._grid[:, -1] = TILE_WALL

        self._place_shelter_clusters(cfg.n_shelter)
        self._place_food(cfg.n_food_initial)

        self._agent = self._random_empty_cell()
        self._predators = self._place_predators(cfg.n_predators, min_dist_from_agent=6)

        self._hunger = 0.0
        self._fatigue = 0.0
        self._health = 100.0
        self._steps = 0
        self._terminated = False
        self._truncated = False

        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("step() on a finished episode; call reset() first")

        cfg = self.config
        reward = cfg.step_penalty
        ate = False
        rested = False
        damaged = False

        # 1. agent action
        if action in _MOVE_DELTAS:
            self._fatigue += cfg.fatigue_rate_move
            dr, dc = _MOVE_DELTAS[action]
            nr, nc = self._agent[0] + dr, self._agent[1] + dc
            if self._is_passable(nr, nc):
                self._agent = (nr, nc)
        elif action == ACTION_EAT:
            ate = self._try_eat()
            if ate:
                reward += cfg.food_reward
        elif action == ACTION_REST:
            ar, ac = self._agent
            if self._grid[ar, ac] == TILE_SHELTER:
                self._fatigue += cfg.fatigue_rate_rest  # negative restores
                rested = True
        # ACTION_NOOP: do nothing

        # 2. predators advance (Manhattan greedy, blocked by shelter LOS)
        self._move_predators()

        # 3. predator contact
        if self._agent in self._predators:
            self._health -= cfg.health_loss_predator
            damaged = True
            reward += cfg.damage_reward

        # 4. clip homeostatic vars; apply starvation
        self._fatigue = float(np.clip(self._fatigue, 0.0, 100.0))
        self._hunger = float(np.clip(self._hunger + cfg.hunger_rate, 0.0, 100.0))
        if self._hunger >= 100.0:
            self._health -= cfg.health_loss_starving
        self._health = float(np.clip(self._health, 0.0, 100.0))

        # 5. food respawn (checks ~grid_size random cells per step)
        if int(self._food_count()) < cfg.max_food:
            for _ in range(cfg.grid_size):
                r = int(self._rng.integers(1, cfg.grid_size - 1))
                c = int(self._rng.integers(1, cfg.grid_size - 1))
                if (
                    self._grid[r, c] == TILE_EMPTY
                    and self._rng.random() < cfg.food_respawn_prob
                ):
                    self._grid[r, c] = TILE_FOOD

        # 6. termination
        self._steps += 1
        if self._health <= 0.0:
            self._terminated = True
            reward += cfg.death_reward
        if self._steps >= cfg.max_steps:
            self._truncated = True

        info = self._info()
        info.update({"ate": ate, "rested": rested, "damaged": damaged, "reward": reward})
        return self._observation(), reward, self._terminated, self._truncated, info

    def render(self) -> str:
        H, W = self._grid.shape
        char_map = {
            TILE_EMPTY: ".",
            TILE_WALL: "#",
            TILE_FOOD: "f",
            TILE_SHELTER: "s",
        }
        lines = []
        for r in range(H):
            row = []
            for c in range(W):
                if (r, c) == self._agent:
                    row.append("A")
                elif (r, c) in self._predators:
                    row.append("P")
                else:
                    row.append(char_map.get(int(self._grid[r, c]), "?"))
            lines.append("".join(row))
        status = (
            f"step={self._steps} hunger={self._hunger:.0f} "
            f"fatigue={self._fatigue:.0f} health={self._health:.0f} "
            f"food={self._food_count()} pred={len(self._predators)}"
        )
        return "\n".join(lines) + "\n" + status

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _is_passable(self, r: int, c: int) -> bool:
        if r < 0 or c < 0 or r >= self._grid.shape[0] or c >= self._grid.shape[1]:
            return False
        return int(self._grid[r, c]) != TILE_WALL

    def _empty_cells(self) -> list[tuple[int, int]]:
        rs, cs = np.where(self._grid == TILE_EMPTY)
        return list(zip(rs.tolist(), cs.tolist()))

    def _random_empty_cell(self) -> tuple[int, int]:
        cells = self._empty_cells()
        idx = int(self._rng.integers(0, len(cells)))
        return cells[idx]

    def _place_shelter_clusters(self, n: int) -> None:
        # 2x2 clusters; gives shelter strategic value (block LOS from a row of predators)
        cfg = self.config
        placed = 0
        attempts = 0
        while placed < n and attempts < 200:
            r = int(self._rng.integers(2, cfg.grid_size - 3))
            c = int(self._rng.integers(2, cfg.grid_size - 3))
            block = self._grid[r : r + 2, c : c + 2]
            if np.all(block == TILE_EMPTY):
                self._grid[r : r + 2, c : c + 2] = TILE_SHELTER
                placed += 1
            attempts += 1

    def _place_food(self, n: int) -> None:
        cells = self._empty_cells()
        n = min(n, len(cells))
        idx = self._rng.choice(len(cells), size=n, replace=False)
        for i in idx:
            r, c = cells[int(i)]
            self._grid[r, c] = TILE_FOOD

    def _place_predators(
        self, n: int, min_dist_from_agent: int
    ) -> list[tuple[int, int]]:
        ar, ac = self._agent
        candidates = [
            cell
            for cell in self._empty_cells()
            if max(abs(cell[0] - ar), abs(cell[1] - ac)) >= min_dist_from_agent
        ]
        if not candidates:
            candidates = self._empty_cells()
        chosen = []
        for _ in range(n):
            if not candidates:
                break
            i = int(self._rng.integers(0, len(candidates)))
            chosen.append(candidates.pop(i))
        return chosen

    def _try_eat(self) -> bool:
        ar, ac = self._agent
        for r, c in [(ar, ac), (ar - 1, ac), (ar + 1, ac), (ar, ac - 1), (ar, ac + 1)]:
            if 0 <= r < self._grid.shape[0] and 0 <= c < self._grid.shape[1]:
                if int(self._grid[r, c]) == TILE_FOOD:
                    self._grid[r, c] = TILE_EMPTY
                    self._hunger = max(
                        0.0, self._hunger - self.config.food_hunger_restore
                    )
                    return True
        return False

    def _move_predators(self) -> None:
        new_positions = []
        ar, ac = self._agent
        for pr, pc in self._predators:
            blocked = self._line_blocked_by_shelter((pr, pc), (ar, ac))
            if blocked:
                dr = int(self._rng.integers(-1, 2))
                dc = int(self._rng.integers(-1, 2))
            else:
                dr = int(np.sign(ar - pr))
                dc = int(np.sign(ac - pc))
                # one axis per step, prefer the larger gap so they don't dance diagonally
                if dr != 0 and dc != 0:
                    if abs(ar - pr) >= abs(ac - pc):
                        dc = 0
                    else:
                        dr = 0
            nr, nc = pr + dr, pc + dc
            if self._is_passable(nr, nc) and int(self._grid[nr, nc]) != TILE_SHELTER:
                new_positions.append((nr, nc))
            else:
                new_positions.append((pr, pc))
        self._predators = new_positions

    def _line_blocked_by_shelter(
        self, a: tuple[int, int], b: tuple[int, int]
    ) -> bool:
        ar, ac = a
        br, bc = b
        steps = max(abs(br - ar), abs(bc - ac))
        if steps == 0:
            return False
        for i in range(1, steps):
            r = ar + (br - ar) * i // steps
            c = ac + (bc - ac) * i // steps
            if int(self._grid[r, c]) == TILE_SHELTER:
                return True
        return False

    def _food_count(self) -> int:
        return int((self._grid == TILE_FOOD).sum())

    # ------------------------------------------------------------------
    # Observation & info
    # ------------------------------------------------------------------
    def _observation(self) -> np.ndarray:
        cfg = self.config
        w = cfg.obs_window
        half = w // 2
        ar, ac = self._agent
        H, W = self._grid.shape

        spatial = np.zeros((w, w, NUM_TILE_TYPES), dtype=np.float32)
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                rr, cc = ar + dr, ac + dc
                wr, wc = dr + half, dc + half
                if 0 <= rr < H and 0 <= cc < W:
                    spatial[wr, wc, int(self._grid[rr, cc])] = 1.0
                else:
                    # treat out-of-bounds as wall — agent can "see" the edge
                    spatial[wr, wc, TILE_WALL] = 1.0

        n_pred_visible = 0
        for pr, pc in self._predators:
            if abs(pr - ar) <= half and abs(pc - ac) <= half:
                wr, wc = (pr - ar) + half, (pc - ac) + half
                spatial[wr, wc, TILE_PREDATOR] = 1.0
                n_pred_visible += 1

        extras = np.array(
            [
                self._hunger / 100.0,
                self._fatigue / 100.0,
                self._health / 100.0,
                n_pred_visible / max(1, cfg.n_predators),
                ar / cfg.grid_size,
                ac / cfg.grid_size,
            ],
            dtype=np.float32,
        )

        return np.concatenate([spatial.reshape(-1), extras]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "hunger": self._hunger,
            "fatigue": self._fatigue,
            "health": self._health,
            "n_food": int(self._food_count()),
            "n_predators": len(self._predators),
            "agent": self._agent,
            "predators": list(self._predators),
            "steps": self._steps,
        }


# ----------------------------------------------------------------------
# Smoke test (run as `python -m foresight.envs.instinct_gridworld`)
# ----------------------------------------------------------------------
def _smoke_test(seed: int = 0, n_steps: int = 1000, verbose: bool = True) -> dict:
    env = InstinctGridworldEnv(InstinctGridworldConfig(seed=seed))
    obs, info = env.reset(seed=seed)

    rng = np.random.default_rng(seed)
    rewards: list[float] = []
    hunger_trace: list[float] = []
    fatigue_trace: list[float] = []
    health_trace: list[float] = []
    food_eaten = 0
    damages = 0
    rested_count = 0
    last_step = 0

    for step in range(n_steps):
        action = int(rng.integers(0, NUM_ACTIONS))
        obs, reward, term, trunc, info = env.step(action)
        rewards.append(reward)
        hunger_trace.append(info["hunger"])
        fatigue_trace.append(info["fatigue"])
        health_trace.append(info["health"])
        if info["ate"]:
            food_eaten += 1
        if info["damaged"]:
            damages += 1
        if info["rested"]:
            rested_count += 1
        last_step = step + 1
        if term or trunc:
            break

    summary = {
        "obs_shape": tuple(obs.shape),
        "obs_min": float(obs.min()),
        "obs_max": float(obs.max()),
        "steps": last_step,
        "terminated_or_truncated": term or trunc,
        "total_reward": float(sum(rewards)),
        "food_eaten": food_eaten,
        "damages": damages,
        "rested_count": rested_count,
        "final_hunger": hunger_trace[-1] if hunger_trace else None,
        "final_health": health_trace[-1] if health_trace else None,
        "avg_hunger": float(np.mean(hunger_trace)) if hunger_trace else None,
        "max_hunger": float(np.max(hunger_trace)) if hunger_trace else None,
        "avg_fatigue": float(np.mean(fatigue_trace)) if fatigue_trace else None,
    }
    if verbose:
        for k, v in summary.items():
            print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    print("Instinct Gridworld smoke test (random policy, seed=0)")
    _smoke_test(seed=0, n_steps=1000, verbose=True)
