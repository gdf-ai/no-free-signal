"""Recurrent State-Space Model (V2-style) + decoder.

Two-stream latent:
  h_t : deterministic GRU state (carries context across time)
  z_t : Gaussian stochastic latent (carries momentary uncertainty)

This mirrors predictive coding's two-component cortical representation
(deterministic context + stochastic uncertainty about momentary state) per
notes/biology.md. KL-balancing trick from Hafner 2021 prevents posterior
collapse: the prior and posterior are aligned asymmetrically, with α=0.8
weighting the prior side so it adapts faster.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSMState(NamedTuple):
    h: torch.Tensor      # deterministic state, (B, det_dim)
    z: torch.Tensor      # sampled stochastic state, (B, stoch_dim)
    mu: torch.Tensor     # mean of z's distribution
    sigma: torch.Tensor  # std of z's distribution


class RSSM(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        det_dim: int = 200,
        stoch_dim: int = 32,
        hidden_dim: int = 200,
        min_std: float = 0.1,
    ):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.min_std = min_std

        self.gru_input = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.ELU(inplace=True),
        )
        self.gru = nn.GRUCell(hidden_dim, det_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(det_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> RSSMState:
        h = torch.zeros(batch_size, self.det_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_dim, device=device)
        mu = torch.zeros_like(z)
        sigma = torch.ones_like(z)
        return RSSMState(h=h, z=z, mu=mu, sigma=sigma)

    def _action_one_hot(self, action: torch.Tensor) -> torch.Tensor:
        if action.dtype in (torch.long, torch.int32, torch.int64):
            return F.one_hot(action.long(), num_classes=self.action_dim).float()
        return action.float()

    def _gauss(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, raw_sigma = torch.chunk(params, 2, dim=-1)
        sigma = F.softplus(raw_sigma) + self.min_std
        return mu, sigma

    def step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        observation_feature: torch.Tensor | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        """One RSSM step. Returns (prior, posterior). If observation is None
        (imagined rollout), posterior == prior."""
        a = self._action_one_hot(prev_action)
        gru_in = self.gru_input(torch.cat([prev_state.z, a], dim=-1))
        h = self.gru(gru_in, prev_state.h)

        prior_mu, prior_sigma = self._gauss(self.prior_net(h))
        prior_z = prior_mu + prior_sigma * torch.randn_like(prior_mu)
        prior_state = RSSMState(h=h, z=prior_z, mu=prior_mu, sigma=prior_sigma)

        if observation_feature is None:
            return prior_state, prior_state

        post_mu, post_sigma = self._gauss(
            self.posterior_net(torch.cat([h, observation_feature], dim=-1))
        )
        post_z = post_mu + post_sigma * torch.randn_like(post_mu)
        posterior_state = RSSMState(h=h, z=post_z, mu=post_mu, sigma=post_sigma)
        return prior_state, posterior_state

    def rollout(
        self,
        initial_state: RSSMState,
        actions: torch.Tensor,                            # (B, T)
        observation_features: torch.Tensor | None = None,  # (B, T, feature_dim)
    ) -> tuple[list[RSSMState], list[RSSMState]]:
        """Run a sequence forward.

        With observation_features (training): posterior trajectory is grounded
        in real observations, prior trajectory is what the model would have
        predicted without seeing them. KL between them is the prediction error.
        Without observation_features (imagination): pure rollout from priors.
        """
        T = actions.shape[1]
        priors: list[RSSMState] = []
        posteriors: list[RSSMState] = []
        state = initial_state
        for t in range(T):
            obs_feat = (
                observation_features[:, t] if observation_features is not None else None
            )
            prior_state, post_state = self.step(state, actions[:, t], obs_feat)
            priors.append(prior_state)
            posteriors.append(post_state)
            # During training we walk the posterior trajectory (the true path);
            # during imagination posterior == prior.
            state = post_state
        return priors, posteriors


def kl_balanced_loss(
    posterior: RSSMState,
    prior: RSSMState,
    alpha: float = 0.8,
    free_nats: float = 1.0,
) -> torch.Tensor:
    """KL balancing trick from Hafner 2021.

    L_KL = (1-alpha) * KL(q || stop_grad(p)) + alpha * KL(stop_grad(q) || p)

    alpha=0.8 means the prior gets 80% of the gradient pressure on the KL,
    so it 'chases' the posterior faster than the other way around. This
    prevents posterior collapse (q -> p with both shrinking to a delta).
    """
    def kl(q_mu, q_sigma, p_mu, p_sigma):
        var_ratio = (q_sigma / p_sigma).pow(2)
        mean_diff_sq = ((q_mu - p_mu) / p_sigma).pow(2)
        kl_per_dim = 0.5 * (var_ratio + mean_diff_sq - 1.0 - var_ratio.log())
        return kl_per_dim.sum(dim=-1)

    kl_q_to_p_sg = kl(
        posterior.mu, posterior.sigma,
        prior.mu.detach(), prior.sigma.detach(),
    )
    kl_q_sg_to_p = kl(
        posterior.mu.detach(), posterior.sigma.detach(),
        prior.mu, prior.sigma,
    )
    kl_loss = (1 - alpha) * kl_q_to_p_sg + alpha * kl_q_sg_to_p
    return torch.clamp(kl_loss, min=free_nats).mean()


class Decoder(nn.Module):
    """Decode (h, z) -> predicted observation. The 'imagination' branch
    of the world model.
    """

    def __init__(
        self,
        det_dim: int,
        stoch_dim: int,
        obs_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1))
