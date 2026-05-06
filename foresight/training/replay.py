"""Rollout replay buffer with optional surprise-prioritized sampling.

The wake phase samples uniformly. The sleep phase samples with weight
∝ priority^α (priority is per-sample prediction error stored alongside the
transition). This is the biological analogue of hippocampal sharp-wave
ripples preferentially replaying high-surprise sequences (Pfeiffer & Foster
2013; Stickgold 2005 — see notes/biology.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    """A single contiguous rollout, stored as parallel arrays."""

    observations: np.ndarray  # (T+1, obs_dim) — includes terminal observation
    actions: np.ndarray  # (T,) int64
    rewards: np.ndarray  # (T,) float32
    terminals: np.ndarray  # (T,) bool — True at step that ended episode
    priorities: np.ndarray  # (T,) float32 — per-step priority (default 1.0)

    def __len__(self) -> int:
        return int(self.actions.shape[0])


@dataclass
class ReplayBuffer:
    """In-memory rollout buffer.

    Stores complete episodes. `sample` returns batched fixed-length sequences
    suitable for RSSM training (sequences allow the recurrent state to warm
    up before the loss is computed).
    """

    obs_dim: int
    capacity_steps: int = 100_000
    episodes: list[Episode] = field(default_factory=list)
    _total_steps: int = 0

    def add_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)
        self._total_steps += len(episode)
        # Drop oldest episodes until we are under capacity.
        while self._total_steps > self.capacity_steps and len(self.episodes) > 1:
            dropped = self.episodes.pop(0)
            self._total_steps -= len(dropped)

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def update_priorities(
        self, episode_idx: int, step_idxs: np.ndarray, new_priorities: np.ndarray
    ) -> None:
        ep = self.episodes[episode_idx]
        ep.priorities[step_idxs] = new_priorities.astype(np.float32)

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator,
        prioritized: bool = False,
        alpha: float = 0.6,
    ) -> dict[str, np.ndarray]:
        """Sample batch_size sequences of length seq_len.

        Returns dict with keys obs (B, T+1, D), action (B, T), reward (B, T),
        terminal (B, T), weight (B, T) — for prioritized sampling, weights
        are the importance-sampling correction; uniform sampling sets all
        weights to 1.
        """
        if not self.episodes:
            raise RuntimeError("replay buffer is empty")

        # Build the list of valid (episode_idx, start_step) pairs. A start is
        # valid if there are ≥ seq_len transitions starting from it within the
        # same episode (we don't cross episode boundaries — terminals reset
        # the recurrent state).
        starts: list[tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            n = len(ep)
            if n < seq_len:
                continue
            for s in range(n - seq_len + 1):
                starts.append((ep_idx, s))
        if not starts:
            raise RuntimeError(
                f"no episode in buffer is long enough for seq_len={seq_len}"
            )

        if prioritized:
            # priority of a sequence = mean priority of its transitions
            seq_pri = np.fromiter(
                (
                    self.episodes[ep_idx].priorities[s : s + seq_len].mean()
                    for ep_idx, s in starts
                ),
                dtype=np.float32,
                count=len(starts),
            )
            # add a small floor to avoid zero-probability sequences after
            # initialization (everything starts at priority 1.0 so this only
            # matters once priorities have been updated)
            seq_pri = np.maximum(seq_pri, 1e-6)
            probs = seq_pri**alpha
            probs = probs / probs.sum()
            idx = rng.choice(len(starts), size=batch_size, replace=True, p=probs)
            # importance-sampling correction (β=1 for simplicity in v1)
            weights = (1.0 / (len(starts) * probs[idx])).astype(np.float32)
            weights = weights / weights.max()  # normalize to [0, 1]
        else:
            idx = rng.integers(0, len(starts), size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)

        D = self.obs_dim
        obs_batch = np.empty((batch_size, seq_len + 1, D), dtype=np.float32)
        act_batch = np.empty((batch_size, seq_len), dtype=np.int64)
        rew_batch = np.empty((batch_size, seq_len), dtype=np.float32)
        term_batch = np.empty((batch_size, seq_len), dtype=bool)
        w_batch = np.empty((batch_size, seq_len), dtype=np.float32)

        for i, k in enumerate(idx):
            ep_idx, s = starts[int(k)]
            ep = self.episodes[ep_idx]
            obs_batch[i] = ep.observations[s : s + seq_len + 1]
            act_batch[i] = ep.actions[s : s + seq_len]
            rew_batch[i] = ep.rewards[s : s + seq_len]
            term_batch[i] = ep.terminals[s : s + seq_len]
            w_batch[i] = weights[i]

        return {
            "obs": obs_batch,
            "action": act_batch,
            "reward": rew_batch,
            "terminal": term_batch,
            "weight": w_batch,
        }


def collect_random_episodes(
    env,
    n_episodes: int,
    seed: int,
    max_steps_per_episode: int | None = None,
) -> list[Episode]:
    """Run a random policy for n_episodes; return fully-formed Episode objects.

    Used for v1 bootstrap data and as the smoke test for the buffer.
    """
    rng = np.random.default_rng(seed)
    episodes: list[Episode] = []

    for ep_i in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep_i)
        observations = [np.asarray(obs, dtype=np.float32)]
        actions: list[int] = []
        rewards: list[float] = []
        terminals: list[bool] = []
        n_actions = env.action_space.n
        steps = 0
        while True:
            a = int(rng.integers(0, n_actions))
            obs, r, term, trunc, _ = env.step(a)
            observations.append(np.asarray(obs, dtype=np.float32))
            actions.append(a)
            rewards.append(float(r))
            terminals.append(bool(term))
            steps += 1
            if term or trunc:
                break
            if max_steps_per_episode is not None and steps >= max_steps_per_episode:
                break

        T = len(actions)
        episodes.append(
            Episode(
                observations=np.stack(observations, axis=0),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=np.asarray(rewards, dtype=np.float32),
                terminals=np.asarray(terminals, dtype=bool),
                priorities=np.ones(T, dtype=np.float32),
            )
        )
    return episodes


# ----------------------------------------------------------------------
# Smoke test (run as `python -m foresight.training.replay`)
# ----------------------------------------------------------------------
def _smoke_test() -> None:
    from foresight.envs.instinct_gridworld import (
        InstinctGridworldConfig,
        InstinctGridworldEnv,
    )

    print("collecting 10 random episodes...")
    env = InstinctGridworldEnv(InstinctGridworldConfig(seed=0))
    episodes = collect_random_episodes(env, n_episodes=10, seed=0)
    obs_dim = int(env.observation_space.shape[0])
    buffer = ReplayBuffer(obs_dim=obs_dim, capacity_steps=100_000)
    for ep in episodes:
        buffer.add_episode(ep)
    print(
        f"  episodes: {buffer.n_episodes}  "
        f"total transitions: {buffer.total_steps}  "
        f"obs_dim: {obs_dim}"
    )
    ep_lens = [len(e) for e in episodes]
    print(
        f"  episode lengths: min={min(ep_lens)}  max={max(ep_lens)}  "
        f"mean={np.mean(ep_lens):.1f}"
    )

    rng = np.random.default_rng(0)
    seq_len = 16
    print(f"\nsampling 4 uniform sequences of length {seq_len}...")
    batch = buffer.sample(batch_size=4, seq_len=seq_len, rng=rng, prioritized=False)
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape}  dtype={v.dtype}")

    print("\nsetting some priorities and sampling prioritized...")
    # boost priority of the first half of episode 0 to test prioritization
    if len(buffer.episodes[0]) >= 10:
        buffer.update_priorities(
            0,
            np.arange(min(10, len(buffer.episodes[0]))),
            np.full(min(10, len(buffer.episodes[0])), 100.0, dtype=np.float32),
        )
    batch = buffer.sample(
        batch_size=4, seq_len=seq_len, rng=rng, prioritized=True, alpha=0.6
    )
    print("  prioritized sample shapes match? ", all(
        batch[k].shape[0] == 4 for k in ("obs", "action", "reward")
    ))
    print(f"  importance weight range: [{batch['weight'].min():.3f}, {batch['weight'].max():.3f}]")

    print("\nbuffer smoke test OK.")


if __name__ == "__main__":
    _smoke_test()
