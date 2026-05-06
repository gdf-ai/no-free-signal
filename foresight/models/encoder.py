"""Sensory encoder: observation -> learned feature vector.

Maps the flat observation (7x7x6 tile one-hot incl. creature channel + 8
scalar internals + 8 audio bins = 259 dims at the time of writing; obs_dim
is wired through dynamically) to a feature_dim-dim feature embedding. This
is the "primary sensory cortex" module in the predictive-brain framing
(Rao & Ballard).

For symbolic gridworld observations a small MLP is sufficient; no CNN needed
because the input is already a flat one-hot. When we move to pixel-based envs
later (Phase 5), this module is the one that grows into a CNN/ResNet.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SensoryEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, feature_dim))
        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
