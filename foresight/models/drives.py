"""Drive heads: Panksepp-inspired affective scalars over the latent state.

Each drive is a small MLP `(h, z) -> scalar`. The genome contributes an
additive bias to each head's output, so individual differences in innate
temperament are observable from generation to generation.

Phase 6 implements four drives that have natural single-agent expression:
  FEAR     — predicted aversive outcome (predator proximity, damage)
  SEEKING  — curiosity / novelty drive
  PLAY     — low-stakes exploration (gated by FEAR via cross-system inhibition)
  RAGE     — cornered aggression (fires when FEAR high AND escape options low)

CARE / PANIC / LUST need conspecifics; they are reserved for Phase 7+ multi-agent.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

DRIVE_NAMES: tuple[str, ...] = ("fear", "seeking", "play", "rage")


def _drive_bias_keys() -> dict[str, str]:
    """Map drive name -> matching genome trait name (the additive bias)."""
    return {
        "fear": "fear_baseline",
        "seeking": "seeking_baseline",
        "play": "play_baseline",
        "rage": "rage_baseline",
    }


class _DriveHead(nn.Module):
    """A single drive scalar over (h, z). Tiny MLP."""

    def __init__(self, det_dim: int, stoch_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(det_dim + stoch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class DriveHeads(nn.Module):
    """The four-drive head bundle, with genome-controlled additive biases.

    Output of each head = `mlp(h, z) + bias[drive]`, where `bias` is set per
    individual from the genome via `set_genome_biases()`. The bias is a
    non-trainable buffer (it's an *innate* prior, not learned).

    With genome biases zero and untrained MLPs, drives are roughly N(0, 1).
    Genome biases shift these baselines per-individual, which is what selection
    operates on.
    """

    def __init__(self, det_dim: int, stoch_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: _DriveHead(det_dim, stoch_dim, hidden_dim) for name in DRIVE_NAMES}
        )
        # Innate biases live as buffers — they move with the model but receive
        # no gradient. They are set fresh per individual at birth.
        for name in DRIVE_NAMES:
            self.register_buffer(f"bias_{name}", torch.zeros(()))

    def set_genome_biases(self, genome_traits: dict[str, float]) -> None:
        """Update the innate biases from a Genome.traits() dict."""
        for drive, trait_name in _drive_bias_keys().items():
            if trait_name in genome_traits:
                buf = getattr(self, f"bias_{drive}")
                buf.copy_(torch.tensor(float(genome_traits[trait_name]), device=buf.device))

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in DRIVE_NAMES:
            head_out = self.heads[name](h, z)
            bias = getattr(self, f"bias_{name}")
            out[name] = head_out + bias
        return out

    def cross_inhibited(
        self, drives: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply Panksepp cross-system inhibition.

        - PLAY suppressed by FEAR: play_eff = play * sigmoid(-fear)
        - RAGE gated by FEAR (rises when FEAR is high — cornered): rage_eff = rage * sigmoid(fear)
        - FEAR and SEEKING pass through.
        """
        fear = drives["fear"]
        play_eff = drives["play"] * torch.sigmoid(-fear)
        rage_eff = drives["rage"] * torch.sigmoid(fear)
        return {
            "fear": fear,
            "seeking": drives["seeking"],
            "play": play_eff,
            "rage": rage_eff,
        }


# ----------------------------------------------------------------------
# Smoke test — `python -m foresight.models.drives`
# ----------------------------------------------------------------------
def _smoke() -> None:
    torch.manual_seed(0)
    det_dim, stoch_dim = 200, 32
    drives = DriveHeads(det_dim=det_dim, stoch_dim=stoch_dim)
    h = torch.randn(4, det_dim)
    z = torch.randn(4, stoch_dim)

    print("=== untrained drives, zero genome biases ===")
    out = drives(h, z)
    for name, val in out.items():
        print(f"  {name:8s}: shape={tuple(val.shape)}  "
              f"mean={val.mean().item():+.3f}  std={val.std().item():.3f}")

    print("\n=== set strong fear bias from genome ===")
    drives.set_genome_biases({
        "fear_baseline": 1.5,
        "seeking_baseline": 0.0,
        "play_baseline": 0.5,
        "rage_baseline": -0.2,
    })
    out2 = drives(h, z)
    for name, val in out2.items():
        delta = (val - out[name]).mean().item()
        print(f"  {name:8s}: mean={val.mean().item():+.3f}  (delta from baseline: {delta:+.3f})")

    print("\n=== cross-system inhibition (PLAY suppressed by FEAR, RAGE amplified) ===")
    inhibited = drives.cross_inhibited(out2)
    for name in DRIVE_NAMES:
        before = out2[name].mean().item()
        after = inhibited[name].mean().item()
        print(f"  {name:8s}: before={before:+.3f}  after={after:+.3f}  delta={after-before:+.3f}")

    print("\nOK")


if __name__ == "__main__":
    _smoke()
