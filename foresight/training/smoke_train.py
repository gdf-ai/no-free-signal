"""Smoke-train the encoder + RSSM + decoder on random rollouts.

Verifies the training stack: data flows, gradients propagate, loss decreases.
Saves a checkpoint to checkpoints/smoke.pt that play_visual loads to show the
model's internal state and predictions live.

Run:  python -m foresight.training.smoke_train
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from foresight.envs.instinct_gridworld import (
    InstinctGridworldConfig,
    InstinctGridworldEnv,
)
from foresight.models.encoder import SensoryEncoder
from foresight.models.rssm import RSSM, Decoder, kl_balanced_loss
from foresight.training.replay import ReplayBuffer, collect_random_episodes


def smoke_train(
    n_episodes: int = 80,
    batch_size: int = 16,
    seq_len: int = 8,
    n_train_steps: int = 300,
    feature_dim: int = 256,
    det_dim: int = 200,
    stoch_dim: int = 32,
    lr: float = 3e-4,
    kl_weight: float = 1.0,
    alpha: float = 0.8,
    free_nats: float = 1.0,
    seed: int = 0,
    checkpoint_path: str = "checkpoints/smoke.pt",
    device: str | None = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"device: {device}")

    print(f"collecting {n_episodes} random episodes...")
    env = InstinctGridworldEnv(InstinctGridworldConfig(seed=seed))
    obs_dim = int(env.observation_space.shape[0])
    n_actions = int(env.action_space.n)
    eps = collect_random_episodes(env, n_episodes=n_episodes, seed=seed)
    buf = ReplayBuffer(obs_dim=obs_dim)
    for e in eps:
        buf.add_episode(e)
    print(
        f"  buffer: {buf.n_episodes} episodes, {buf.total_steps} transitions, "
        f"obs_dim={obs_dim}, n_actions={n_actions}"
    )

    encoder = SensoryEncoder(obs_dim=obs_dim, feature_dim=feature_dim).to(device)
    rssm = RSSM(
        feature_dim=feature_dim,
        action_dim=n_actions,
        det_dim=det_dim,
        stoch_dim=stoch_dim,
    ).to(device)
    decoder = Decoder(det_dim=det_dim, stoch_dim=stoch_dim, obs_dim=obs_dim).to(device)
    n_params = sum(p.numel() for m in [encoder, rssm, decoder] for p in m.parameters())
    print(f"  model: {n_params/1e6:.2f}M params")

    params = (
        list(encoder.parameters())
        + list(rssm.parameters())
        + list(decoder.parameters())
    )
    opt = torch.optim.Adam(params, lr=lr, eps=1e-5)

    rng = np.random.default_rng(seed)
    losses: list[float] = []
    recons: list[float] = []
    kls: list[float] = []

    print(
        f"\ntraining for {n_train_steps} steps "
        f"(batch={batch_size}, seq_len={seq_len})..."
    )
    for step in range(n_train_steps):
        batch = buf.sample(
            batch_size=batch_size, seq_len=seq_len, rng=rng, prioritized=False
        )
        obs = torch.from_numpy(batch["obs"]).to(device)        # (B, T+1, D)
        action = torch.from_numpy(batch["action"]).to(device)  # (B, T)

        B, Tp1, D = obs.shape
        T = Tp1 - 1

        # encode every step (we use feature at t+1 to ground the posterior at t)
        e_all = encoder(obs.reshape(B * Tp1, D)).view(B, Tp1, -1)
        e_post = e_all[:, 1:]  # (B, T, feature_dim)

        init = rssm.initial_state(B, device)
        priors, posteriors = rssm.rollout(init, action, e_post)

        # decode posterior at each step -> reconstruct obs[t+1]
        h_stack = torch.stack([s.h for s in posteriors], dim=1)  # (B, T, det)
        z_stack = torch.stack([s.z for s in posteriors], dim=1)  # (B, T, stoch)
        obs_pred = decoder(
            h_stack.reshape(B * T, -1), z_stack.reshape(B * T, -1)
        ).view(B, T, D)
        recon = F.mse_loss(obs_pred, obs[:, 1:])

        kl_terms = [
            kl_balanced_loss(post, prior, alpha=alpha, free_nats=free_nats)
            for prior, post in zip(priors, posteriors)
        ]
        kl = torch.stack(kl_terms).mean()

        loss = recon + kl_weight * kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
        opt.step()

        losses.append(float(loss.item()))
        recons.append(float(recon.item()))
        kls.append(float(kl.item()))

        if step % 30 == 0 or step == n_train_steps - 1:
            print(
                f"  step {step:4d}: loss={loss.item():.4f}  "
                f"recon={recon.item():.4f}  kl={kl.item():.4f}"
            )

    start_loss = float(np.mean(losses[:10]))
    end_loss = float(np.mean(losses[-10:]))
    start_recon = float(np.mean(recons[:10]))
    end_recon = float(np.mean(recons[-10:]))

    print(
        f"\nstart loss (mean of first 10):  {start_loss:.4f}"
        f"  recon={start_recon:.4f}"
    )
    print(
        f"end   loss (mean of last 10):   {end_loss:.4f}"
        f"  recon={end_recon:.4f}"
    )
    print(
        f"recon improvement: {((start_recon - end_recon) / start_recon * 100):+.1f}%"
    )

    out_path = Path(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "rssm": rssm.state_dict(),
            "decoder": decoder.state_dict(),
            "config": {
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "feature_dim": feature_dim,
                "det_dim": det_dim,
                "stoch_dim": stoch_dim,
            },
            "training_summary": {
                "n_episodes": n_episodes,
                "n_train_steps": n_train_steps,
                "start_loss": start_loss,
                "end_loss": end_loss,
                "start_recon": start_recon,
                "end_recon": end_recon,
            },
        },
        out_path,
    )
    print(f"\ncheckpoint saved to {out_path.resolve()}")

    return {
        "start_loss": start_loss,
        "end_loss": end_loss,
        "start_recon": start_recon,
        "end_recon": end_recon,
        "n_params": n_params,
        "checkpoint_path": str(out_path.resolve()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=80)
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoints/smoke.pt")
    args = p.parse_args()
    smoke_train(
        n_episodes=args.n_episodes,
        n_train_steps=args.n_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
