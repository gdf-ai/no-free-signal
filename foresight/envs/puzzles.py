"""Construction puzzles — small dynamical systems whose future state must be
predicted by a creature's brain to successfully BUILD shelter.

Each puzzle: a hidden 8-dim oscillator coupled in a deterministic way. The
brain receives the first `prefix_steps` rolled out as 'observations' (padded
to obs_dim with zeros), and must predict the next `n_predict` steps. MSE
between predicted and true determines pass/fail; threshold is modulated by
the creature's `puzzle_solver_strength` genome trait.

Untrained brains will fail almost always. After a few hundred training
steps on real-world experience, the RSSM learns enough to occasionally
succeed — and that success then cascades into reproductive advantage
(shelter → safety → less predation → more children).
"""
from __future__ import annotations

import numpy as np


def generate_puzzle(
    rng: np.random.Generator,
    prefix_steps: int = 8,
    n_predict: int = 8,
    obs_dim: int = 351,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll a small coupled-oscillator system, return (prefix, target_future).

    The system has 8 hidden dims `x`. Update rule:
        x[i] += dt * (sin(x[i-1]) - 0.3 * x[i] + 0.5 * sin(0.7 * x[(i+3) % 8]))
    With a random initial condition. The full trajectory has length
    prefix_steps + n_predict. We embed each step into an obs-shaped vector by
    placing the 8 dims in the leading positions, zeroing the rest.
    """
    H = 8
    x = rng.normal(0.0, 1.0, size=H)
    dt = 0.4
    full = np.zeros((prefix_steps + n_predict, obs_dim), dtype=np.float32)
    for t in range(prefix_steps + n_predict):
        x = x + dt * (
            np.sin(np.roll(x, 1)) - 0.3 * x + 0.5 * np.sin(0.7 * np.roll(x, -3))
        )
        full[t, :H] = x
    prefix = full[:prefix_steps]
    future = full[prefix_steps:]
    return prefix, future


def evaluate_puzzle(
    predicted: np.ndarray, truth: np.ndarray, threshold: float
) -> tuple[bool, float]:
    """Pass/fail + the actual MSE on the relevant dims (first 8)."""
    pred = predicted[:, :8]
    real = truth[:, :8]
    mse = float(((pred - real) ** 2).mean())
    return mse < threshold, mse


def threshold_for(puzzle_solver_strength: float, sigma_baseline: float = 1.0) -> float:
    """Genome-modulated MSE threshold. Stronger solvers need less accuracy
    to count as a 'success' — they're more confident, even at higher MSE.
    But for v1 we keep this simple: lower sigma_baseline = harder."""
    return sigma_baseline * (0.4 + 0.6 * puzzle_solver_strength)
