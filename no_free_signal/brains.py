"""Per-individual neural brains for prey.

Each living prey has its own encoder + RSSM + decoder + drive heads + Adam
optimizer + replay buffer. On birth, a fresh brain is allocated; on death,
discarded. Round-robin training keeps total compute bounded — only one
creature trains per env step.

This turns each prey into a real biologically-grounded sensorimotor learner:
genome controls innate drive biases; the brain learns the world model and
policy through lived experience.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from foresight.evolution.genome import Genome
from foresight.models.drives import DriveHeads
from foresight.models.encoder import SensoryEncoder
from foresight.models.rssm import RSSM, Decoder, RSSMState, kl_balanced_loss
from foresight.training.replay import Episode, ReplayBuffer
from no_free_signal.observation import RawObservation


class CreatureBrain:
    """One prey's brain. Owns its own model + optimizer + buffer."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        genome: Genome,
        device: str = "cpu",
        feature_dim: int = 128,
        det_dim: int = 128,
        stoch_dim: int = 16,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.genome = genome
        self.feature_dim = feature_dim
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim

        self.encoder = SensoryEncoder(obs_dim=obs_dim, feature_dim=feature_dim).to(device)
        self.rssm = RSSM(
            feature_dim=feature_dim, action_dim=n_actions,
            det_dim=det_dim, stoch_dim=stoch_dim,
        ).to(device)
        self.decoder = Decoder(
            det_dim=det_dim, stoch_dim=stoch_dim, obs_dim=obs_dim
        ).to(device)
        self.drive_heads = DriveHeads(det_dim=det_dim, stoch_dim=stoch_dim).to(device)

        # Drive heads are innate biology — frozen weights + genome bias.
        for p in self.drive_heads.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            for head in self.drive_heads.heads.values():
                head.net[-1].weight.mul_(0.05)
                head.net[-1].bias.zero_()
        self.drive_heads.set_genome_biases(genome.traits())

        self._world_params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
        )
        self.optimizer = torch.optim.Adam(self._world_params, lr=3e-4, eps=1e-5)
        self.buffer = ReplayBuffer(obs_dim=obs_dim, capacity_steps=8_000)

        self.rssm_state: RSSMState = self.rssm.initial_state(1, device)
        self.age = 0
        self.recent_losses: deque[float] = deque(maxlen=20)
        self.recent_recons: deque[float] = deque(maxlen=20)
        self._episode_obs: list[np.ndarray] = []
        self._episode_actions: list[int] = []
        self._episode_rewards: list[float] = []
        self._episode_terminals: list[bool] = []
        self._last_action: int = 6  # noop

    # ----- inference -----
    @torch.no_grad()
    def perceive(self, obs: RawObservation) -> None:
        """Update internal RSSM state from a new observation."""
        torch_obs = torch.from_numpy(obs.flat).float().unsqueeze(0).to(self.device)
        action_t = torch.tensor([self._last_action], dtype=torch.long, device=self.device)
        e = self.encoder(torch_obs)
        _, post = self.rssm.step(self.rssm_state, action_t, e)
        self.rssm_state = post

    @torch.no_grad()
    def choose_action_scores(self, obs: RawObservation) -> torch.Tensor:
        """Compute the full per-action score vector — the same scoring used
        by `choose_action`, but exposed so the LLM-controller advisory blend
        can add a boost on top before the argmax is taken.
        Returns a 1-D tensor of length `n_actions` on `self.device`."""
        cs = self.rssm_state
        h_b = cs.h.repeat(self.n_actions, 1)
        z_b = cs.z.repeat(self.n_actions, 1)
        mu_b = cs.mu.repeat(self.n_actions, 1)
        sigma_b = cs.sigma.repeat(self.n_actions, 1)
        batched = RSSMState(h=h_b, z=z_b, mu=mu_b, sigma=sigma_b)
        actions_t = torch.arange(self.n_actions, dtype=torch.long, device=self.device)
        prior, _ = self.rssm.step(batched, actions_t, observation_feature=None)
        drives = self.drive_heads(prior.h, prior.mu)
        inhibited = self.drive_heads.cross_inhibited(drives)

        traits = self.genome.traits()
        energy_now = float(obs.drives.get("energy", 1.0))
        hunger_pressure = max(0.0, 1.0 - energy_now)
        creature_aversion = traits.get(
            "creature_aversion", traits.get("predator_aversion", 1.0)
        )
        score = (
            inhibited["seeking"]
            - creature_aversion * inhibited["fear"]
            - traits.get("hunger_aversion", 1.0) * hunger_pressure
        )

        soc_food = float(obs.extras.get("social_food", 0.0))
        soc_pred = float(obs.extras.get("social_predation", 0.0))
        soc_build = float(obs.extras.get("social_build", 0.0))
        soc_gather = float(obs.extras.get("social_gather", 0.0))
        soc_mate = float(obs.extras.get("social_mate", 0.0))
        soc_rest = float(obs.extras.get("social_rest", 0.0))
        soc_danger = float(obs.extras.get("social_danger", 0.0))
        bias = torch.zeros(self.n_actions, device=self.device, dtype=score.dtype)
        bias[4] = bias[4] + 0.6 * (soc_food + soc_pred)
        matter_adj = float(obs.extras.get("matter_adjacent", 0.0))
        food_disc = float(traits.get("food_discrimination", 0.5))
        bias[4] = bias[4] - 0.6 * matter_adj * (1.0 - food_disc)
        bias[5] = bias[5] + 0.5 * soc_rest * traits.get("reward_comfort", 0.5)
        bias[6] = bias[6] + 0.5 * soc_gather * traits.get("reward_happiness", 0.5)
        bias[7] = bias[7] + 0.7 * soc_build * traits.get("reward_safety", 0.5)
        bias[6] = bias[6] + 0.3 * soc_build * traits.get("reward_safety", 0.5)
        if soc_mate > 0.0:
            mate_b = 0.3 * soc_mate * traits.get("reward_reproduction", 0.5)
            bias[0:4] = bias[0:4] + mate_b
        if soc_danger > 0.0:
            d_intensity = soc_danger * (1.0 + max(0.0, traits.get("fear_baseline", 0.0)))
            bias[4] = bias[4] - 0.6 * d_intensity
            bias[7] = bias[7] - 0.6 * d_intensity
            bias[5] = bias[5] - 0.4 * d_intensity
            bias[8] = bias[8] - 0.4 * d_intensity
        score = score + bias
        score[-1] = score[-1] - 0.05  # NOOP penalty
        return score

    @torch.no_grad()
    def choose_action(
        self, obs: RawObservation, rng: np.random.Generator, epsilon: float = 0.15,
        advisory: tuple[int, float] | None = None,
    ) -> int:
        """Genome + brain heuristic policy with optional advisory blend.
        When `advisory=(action, boost)` is provided (e.g. from an attached
        LLMController), `boost` is added to that action's score before
        argmax — supports the user-confirmed advisory-blend semantic."""
        if rng.random() < epsilon:
            return int(rng.integers(0, self.n_actions))
        score = self.choose_action_scores(obs)
        if advisory is not None:
            a, boost = advisory
            if 0 <= a < self.n_actions and boost > 0.0:
                score[a] = score[a] + float(boost)
        return int(score.argmax().item())

    @torch.no_grad()
    def current_drives(self) -> dict[str, float]:
        """Drive activations at the current internal state — for the UI."""
        d = self.drive_heads(self.rssm_state.h, self.rssm_state.mu)
        d = self.drive_heads.cross_inhibited(d)
        return {k: float(v.item()) for k, v in d.items()}

    @torch.no_grad()
    def latent_snapshot(self) -> np.ndarray:
        """Flat (det_dim+stoch_dim) latent for visualization."""
        h = self.rssm_state.h.squeeze(0).cpu().numpy()
        z = self.rssm_state.mu.squeeze(0).cpu().numpy()
        return np.concatenate([h, z]).astype(np.float32)

    @torch.no_grad()
    def predict_trajectory(self, prefix_obs: np.ndarray, n_predict: int) -> np.ndarray:
        """Run the world model forward to predict the next `n_predict` steps
        of the observation, given a prefix of `prefix_obs.shape[0]` true
        observations. Used by the construction-puzzle mechanic — a creature
        whose RSSM has learned good prediction can solve the puzzle and BUILD
        successfully.

        Returns predicted next-step decoded observations of shape
        (n_predict, obs_dim).
        """
        torch_local = torch
        prefix = torch_local.from_numpy(prefix_obs).float().to(self.device)
        if prefix.dim() == 1:
            prefix = prefix.unsqueeze(0)
        T_in = prefix.shape[0]
        # Encode prefix and ground RSSM state through it.
        state = self.rssm.initial_state(1, self.device)
        for t in range(T_in):
            e = self.encoder(prefix[t : t + 1])
            # Use a noop action surrogate for unrolling the puzzle prefix.
            a = torch_local.zeros((1,), dtype=torch_local.long, device=self.device)
            _, post = self.rssm.step(state, a, e)
            state = post
        # Now imagine forward without observations.
        out = []
        for _ in range(n_predict):
            a = torch_local.zeros((1,), dtype=torch_local.long, device=self.device)
            prior, _ = self.rssm.step(state, a, None)
            decoded = self.decoder(prior.h, prior.mu).squeeze(0).cpu().numpy()
            out.append(decoded)
            state = prior
        return np.stack(out, axis=0).astype(np.float32)

    # ----- learning -----
    def add_step(
        self,
        obs: RawObservation,
        action: int,
        reward: float,
        terminal: bool,
    ) -> None:
        if not self._episode_obs:
            self._episode_obs.append(obs.flat.copy())
        self._episode_actions.append(int(action))
        self._episode_rewards.append(float(reward))
        self._episode_terminals.append(bool(terminal))
        self._episode_obs.append(obs.flat.copy())
        self._last_action = int(action)
        self.age += 1
        if terminal:
            self._commit_episode()

    def _commit_episode(self) -> None:
        if not self._episode_actions:
            return
        ep = Episode(
            observations=np.stack(self._episode_obs, axis=0),
            actions=np.asarray(self._episode_actions, dtype=np.int64),
            rewards=np.asarray(self._episode_rewards, dtype=np.float32),
            terminals=np.asarray(self._episode_terminals, dtype=bool),
            priorities=np.ones(len(self._episode_actions), dtype=np.float32),
        )
        self.buffer.add_episode(ep)
        self._episode_obs.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_terminals.clear()

    def train_step(
        self,
        rng: np.random.Generator,
        batch_size: int = 4,
        seq_len: int = 8,
        kl_weight: float = 1.0,
        alpha: float = 0.8,
        free_nats: float = 1.0,
    ) -> Optional[dict[str, float]]:
        """One world-model gradient step. Returns None if buffer too small."""
        # Need at least one episode of length >= seq_len.
        if not any(len(e) >= seq_len for e in self.buffer.episodes):
            return None
        batch = self.buffer.sample(
            batch_size=batch_size, seq_len=seq_len, rng=rng, prioritized=False
        )
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        action = torch.from_numpy(batch["action"]).to(self.device)
        B, Tp1, D = obs.shape
        T = Tp1 - 1

        e_all = self.encoder(obs.reshape(B * Tp1, D)).view(B, Tp1, -1)
        e_post = e_all[:, 1:]
        init = self.rssm.initial_state(B, self.device)
        priors, posts = self.rssm.rollout(init, action, e_post)
        h_stack = torch.stack([s.h for s in posts], dim=1)
        z_stack = torch.stack([s.z for s in posts], dim=1)
        obs_pred = self.decoder(
            h_stack.reshape(B * T, -1), z_stack.reshape(B * T, -1)
        ).view(B, T, D)
        recon = F.mse_loss(obs_pred, obs[:, 1:])
        kl_terms = [
            kl_balanced_loss(p, q, alpha=alpha, free_nats=free_nats)
            for q, p in zip(priors, posts)
        ]
        kl = torch.stack(kl_terms).mean()
        loss = recon + kl_weight * kl

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._world_params, max_norm=100.0)
        self.optimizer.step()

        l = float(loss.item())
        r = float(recon.item())
        self.recent_losses.append(l)
        self.recent_recons.append(r)
        return {"loss": l, "recon": r, "kl": float(kl.item())}

    # ----- introspection -----
    def stats(self) -> dict[str, float | int]:
        return {
            "age": int(self.age),
            "buffer_episodes": int(self.buffer.n_episodes),
            "buffer_steps": int(self.buffer.total_steps),
            "loss_recent": float(np.mean(self.recent_losses)) if self.recent_losses else None,
            "recon_recent": float(np.mean(self.recent_recons)) if self.recent_recons else None,
        }


class CreatureBrainManager:
    """Lifecycle + round-robin trainer for per-prey brains."""

    def __init__(
        self, obs_dim: int, n_actions: int, device: str = "cpu", seed: int = 0,
        feature_dim: int = 128, det_dim: int = 128, stoch_dim: int = 16,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self.feature_dim = feature_dim
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.brains: dict[int, CreatureBrain] = {}
        self._train_idx: int = 0
        self._rng = np.random.default_rng(seed)

    def on_birth(self, creature_id: int, genome: Genome) -> CreatureBrain:
        brain = CreatureBrain(
            obs_dim=self.obs_dim,
            n_actions=self.n_actions,
            genome=genome,
            device=self.device,
            feature_dim=self.feature_dim,
            det_dim=self.det_dim,
            stoch_dim=self.stoch_dim,
        )
        self.brains[creature_id] = brain
        return brain

    def on_death(self, creature_id: int) -> None:
        self.brains.pop(creature_id, None)

    def reap(self, alive_ids: set[int]) -> list[int]:
        dead = [cid for cid in self.brains if cid not in alive_ids]
        for cid in dead:
            self.on_death(cid)
        return dead

    def get(self, creature_id: int) -> Optional[CreatureBrain]:
        return self.brains.get(creature_id)

    def round_robin_train(self) -> Optional[tuple[int, dict[str, float]]]:
        """Train one brain. Returns (creature_id, loss_dict) or None."""
        if not self.brains:
            return None
        ids = sorted(self.brains.keys())
        cid = ids[self._train_idx % len(ids)]
        self._train_idx += 1
        brain = self.brains[cid]
        result = brain.train_step(self._rng)
        if result is None:
            return None
        return cid, result

    def __len__(self) -> int:
        return len(self.brains)

    def __contains__(self, creature_id: int) -> bool:
        return creature_id in self.brains
