# v1 Decision: Architecture and Proxy Environment

**Date:** 2026-04-26 (revised after user review)
**Status:** SIGNED OFF — proceeding to Phase 2.

This synthesis combines the three Phase-1 surveys (`biology.md`, `ml-architectures.md`, `proxy-options.md`) into a single concrete v1 plan. Every architectural choice cites the biological pillar that motivates it (per the plan's hard guardrail).

---

## TL;DR

- **Trunk:** Dreamer V2-style RSSM (Gaussian stochastic latents, KL balancing). Adapt patterns from `danijar/dreamer` and good PyTorch ports, but write a **minimal in-house RSSM** we fully understand — pulling whole reference impls risks importing complexity we can't debug. V3 is an explicit upgrade path for a later iteration.
- **Proxy environment:** **Custom Instinct Gridworld** (primary). After the model code is wired, a 30-minute sanity check on **MiniGrid** confirms the model isn't quietly broken / overfit to the custom env.
- **Biology augmentations:** all 7 Panksepp drives with **active vs reserved** partition (5 active in single-agent v1, 3 reserved for v2-multiagent), plus a competence/self-actualization head; cross-system inhibition (`PLAY ← PLAY × (1 − λ·FEAR)`); successor-feature head; wake / sleep training loop with surprise-prioritized replay + multiplicative weight downscaling.
- **VRAM budget:** ~3–4 GB used; ~4–5 GB free on the 8 GB RTX 4060.
- **Sleep validation:** single end-of-training **A/B** (sleep-on vs sleep-off, two complete runs ~4h each), not interleaved cadence.

---

## Architecture: Dreamer V2-style RSSM with biology heads

```
                    ┌────────────────────────────────┐
                    │    Symbolic obs (gridworld)    │
                    │    (7x7x5 tile one-hot +       │
                    │     hunger/fatigue/health/etc.)│
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │  Sensory encoder (MLP, 3-layer)│  PILLAR: predictive brain
                    │    obs → e_t                   │
                    └──────────────┬─────────────────┘
                                   │
                    ┌──────────────▼─────────────────┐
                    │  RSSM (V2-style)               │  PILLAR: predictive brain +
                    │    h_t = GRU(h_{t-1}, z_{t-1}, │  embodiment
                    │              a_{t-1})          │
                    │    prior     p(z_t | h_t)      │  KL-balancing trick:
                    │    posterior q(z_t | h_t, e_t) │  L_KL = (1-α)KL(q||p)+
                    │    z ~ N(μ, σ), 32 dims        │        α KL(p||q)
                    └──┬─────┬───────┬───────────┬───┘
                       │     │       │           │
            ┌──────────▼─┐ ┌─▼───┐ ┌─▼───────┐ ┌─▼───────────┐
            │ obs        │ │drive│ │successor│ │ continue    │
            │ decoder    │ │heads│ │ feat ψ  │ │ head        │
            │ + reward   │ │×8   │ │         │ │ (ep end?)   │
            └────────────┘ └──┬──┘ └─────────┘ └─────────────┘
              PILLAR: pred.    │     PILLAR:
              brain (recon =   │     cognitive maps
              error signal)    │
                          ┌────┴───────────────────┐
                          │  PANKSEPP DRIVE HEADS  │  PILLAR: drives
                          │  + competence          │
                          │                         │
                          │  ACTIVE in v1:          │
                          │    SEEKING  (RND-style)│
                          │    FEAR     (pred loss)│
                          │    PLAY     (info-gain │
                          │              × ¬FEAR)  │
                          │    RAGE     (FEAR ×    │
                          │              ¬escape)  │
                          │    competence          │
                          │      (empowerment-     │
                          │       style, info on   │
                          │       own future state)│
                          │                         │
                          │  RESERVED for v2-multi: │
                          │    CARE   (→0 stub)    │
                          │    PANIC  (→0 stub)    │
                          │    LUST   (→0 stub)    │
                          └─────────────────────────┘
```

### Pillar → Module Mapping (the required guardrail table)

| Biological pillar | Source theory | v1 module | Reference / pattern source |
|---|---|---|---|
| Predictive brain | Rao & Ballard 1999; Friston FEP | RSSM dynamics with prior/posterior KL = "prediction error" | Dreamer V2 (danijar/dreamer + paper pseudocode) |
| Dreaming / replay | Buzsáki sharp-wave ripples; Stickgold | Sleep phase: surprise-prioritized replay, imagined rollouts at compressed rate | Custom; informed by Dreamer's imagination loop + Pfeiffer & Foster priority |
| Synaptic homeostasis (SHY) | Tononi & Cirelli 2003/2014 | Multiplicative weight downscaling (~15% cumulative per "day") at end of each sleep phase | Custom — small hook on the optimizer |
| Drives — SEEKING | Panksepp; Schultz 2002 | Curiosity head trained as RND-style predictor error | RND (openai/random-network-distillation) |
| Drives — FEAR | Panksepp; LeDoux | Predicted-loss / aversive-outcome scalar over imagined rollouts | Custom — MLP head over (z, h) |
| Drives — PLAY | Panksepp | Information-gain × (1 − λ·FEAR); only fires when FEAR low | Custom |
| Drives — RAGE | Panksepp | FEAR × (1 − escape_options); fires when cornered | Custom |
| Drives — competence / self-actualization | Schmidhuber "fun"; empowerment (Klyubin) | Empowerment-style head: predicted mutual info between own actions and own future state | Variational empowerment (Mohamed & Rezende 2015) |
| Drives — CARE / PANIC / LUST | Panksepp | Heads present, output ≈ 0 in v1; activate in v2-multiagent | Architectural reservation |
| Somatic markers | Damasio 1996 | Drive-head trajectories used as fast heuristic during planning | Implicit — falls out of imagined drive trajectories |
| Cognitive maps | Tolman; O'Keefe; Stachenfeld 2017 | Successor-feature head ψ(z, h) trained via TD on φ(z) | Successor features (Barreto 2016) |
| Embodiment / Umwelt | Varela; von Uexküll | Closed sensorimotor loop; obs encodes body state | Built into env design |

### Concrete sizes (v1, fits 8 GB VRAM with ~4 GB headroom)

- Sensory encoder: 3-layer MLP, hidden 256.
- RSSM: GRU h ∈ ℝ^200; stochastic z ∈ ℝ^32 (Gaussian, learned σ).
- Decoder: 3-layer MLP back to obs space.
- Drive heads: 2-layer MLPs (256 → 128 → 1) per drive, 8 heads.
- Successor-feature head: 2-layer MLP, output dim = feature dim (64).
- Total params: ~3–5 M. Training batch 32 × seq-len 64 fits easily.

---

## Proxy Environment: Custom Instinct Gridworld + MiniGrid sanity check

### v1 spec (Instinct Gridworld)

- 32×32 grid; partial observation: 7×7 ego-centric one-hot tile window.
- Tile types: empty, wall, food, shelter, predator (predator is rendered as a channel in the obs window when within view).
- Internal scalars exposed in obs: `[hunger, fatigue, health, predator_visible_count, x_norm, y_norm]`.
- 1–2 predators. Manhattan-distance pursuit; line-of-sight blocked by shelter (predators wander randomly when blocked); predators cannot enter shelter.
- 7 discrete actions: N / S / E / W move, eat, rest, no-op.
- Hunger ↑ over time → health ↓ when starved; eating restores hunger.
- Fatigue ↑ on movement, restored by `rest` in shelter.
- Predator contact → health damage.
- Reward: small step penalty; +food on eat; -damage on contact; -death on health=0.
- Episode terminates on death; truncates at step limit (1000).
- Pure Python + Gymnasium 1.0 API; no rendering required for training.

### MiniGrid sanity check

After the Dreamer-V2-style RSSM is wired, point it at `MiniGrid-Empty-8x8-v0` for a 30-minute training run. We're not optimizing for MiniGrid performance — we just want the loss curves to look reasonable on a known env to confirm the model code isn't subtly broken. **MiniGrid is a debugging tool, not a target.**

### What the Instinct Gridworld exercises (per pillar)

| Pillar | How the env exercises it |
|---|---|
| SEEKING | Food spawns at unknown locations → curiosity-driven exploration is rewarded |
| FEAR | Predators move toward agent; shelter provides safe space → aversive prediction matters |
| PLAY | Low-stakes exploration possible only when FEAR is low — natural test of cross-system inhibition |
| RAGE | Walls + predator = cornering scenarios → tests `RAGE = FEAR × (1 − escape)` |
| Homeostasis | Hunger and fatigue are scalar internal states the agent must regulate |
| Embodiment | The 7×7 obs window is the agent's Umwelt; predator-distance info is body-relative |
| Cognitive maps | 32×32 + partial obs → agent must build a latent spatial map to find food efficiently |

---

## Wake / Sleep Training Loop

```
loop forever:
  WAKE phase (τ_wake env steps, e.g. 5_000):
    collect rollouts using current policy + drive-shaped intrinsic reward
    update world model + heads + actor + critic on real experience
    record per-sample prediction error → priority

  SLEEP phase (τ_sleep gradient steps, e.g. 1_000):
    sample replay batches with priority ∝ prediction-error^α (α≈0.6, Pfeiffer & Foster)
    1) DREAM rollouts: imagine forward N steps using RSSM prior, no env interaction
       (temperature T_dream > 1 on z prior → broader exploration of latent futures,
        Hobson AIM analogue)
    2) update model on imagined rollouts (Dreamer-style imagination)
    3) Tononi-Cirelli downscaling: w ← w × (1 − ε), ε small per sleep phase
       (calibrate so cumulative downscaling per "day" ≈ 15%)
```

### Sleep validation (simple A/B)

After v1 is training stably, run **two complete training jobs** with identical config except the sleep phase: `sleep_on` and `sleep_off`. Compare final N-step prediction error and training stability. Target: a measurable foresight gap. If we can't show one, the biological-grounding claim is empty and we re-examine.

(No interleaved "every K cycles" eval — overengineered. Just two runs at the end.)

---

## Phase 2 file plan

```
foresight/
├── envs/
│   ├── __init__.py
│   └── instinct_gridworld.py        # custom Gymnasium env
├── models/
│   ├── __init__.py
│   ├── encoder.py                   # symbolic-obs MLP encoder
│   ├── rssm.py                      # V2-style RSSM (Gaussian z, KL balancing)
│   ├── heads.py                     # decoder, reward, continue, drive heads, ψ head
│   └── foresight_model.py           # composed model
├── training/
│   ├── __init__.py
│   ├── replay.py                    # surprise-prioritized replay buffer
│   ├── wake.py                      # wake phase update
│   ├── sleep.py                     # sleep: dream rollouts + downscaling
│   └── train.py                     # main loop, alternates wake/sleep
├── eval/
│   ├── __init__.py
│   ├── horizon.py                   # N-step prediction error, calibration
│   └── biology.py                   # drive-provoke tests, place-cell rate maps,
│                                    # sleep A/B, cross-system inhibition probe
└── foresight_api.py                 # ForesightModel.encode/imagine/consequence
```

---

## Verification gates between v1 milestones

1. **Env smoke test:** random policy for 1k steps → non-degenerate hunger/fatigue/predator dynamics; episode termination on death.
2. **MiniGrid sanity check:** model trains stably for 30 min on `MiniGrid-Empty-8x8-v0`; loss curves look reasonable (no NaN explosions, KL doesn't collapse).
3. **Model smoke test:** ≥1 wake + 1 sleep cycle on Instinct Gridworld → checkpoint produced, no OOM.
4. **Foresight beats persistence baseline:** at horizon 5, prediction error meaningfully below "predict last observation."
5. **Drive plausibility:** FEAR scalar spikes >2σ above baseline in the 5 steps before predator contact; SEEKING correlates with novel-tile visitation; PLAY suppressed when FEAR > threshold.
6. **Sleep A/B:** sleep-on N-step prediction error at horizon 5 measurably lower than sleep-off (matched gradient steps).
7. **Phase 4 API:** `ForesightModel.consequence(latent, action_seq)` returns predicted obs, drive trajectories, top-K high-uncertainty latent dims, and any drive crossing threshold.

---

## Decisions made (recording the user's review on 2026-04-26)

1. **Trunk:** Dreamer V2 (Gaussian, KL balancing), minimal in-house RSSM. V3 is an upgrade path, not v1.
2. **Proxy env:** Instinct Gridworld primary; MiniGrid as a 30-minute sanity check on the model code.
3. **Drive vocab:** all 7 Panksepp + competence (8 heads). 5 active in v1 (SEEKING / FEAR / PLAY / RAGE / competence); 3 reserved (CARE / PANIC / LUST → output ≈ 0 in single-agent v1, activate in v2-multiagent).
4. **Sleep validation:** single end-of-training A/B, no interleaved eval.

---

## Explicitly deferred to v2

- **Multi-agent extensions** that activate CARE / PANIC / LUST (conspecifics, mating, separation, offspring care).
- **Reproduction dynamics** (offspring-bearing as a real env event, not a stub).
- **Cross-embodiment transfer** — testing that a model trained in one body can transfer to another (von Uexküll Umwelt experiment).
- **Richer environments** — Crafter for generalization validation, more complex 3D envs.
- **Hierarchical timescales** in the RSSM — multiple predictors at 100 ms / 500 ms / 2 s / 10 s timescales.
- **Dreamer V3 categorical latents** — only if V2 hits a clear ceiling.

---

## Risk register

- **Drive heads can collapse to constant outputs** if not regularized. Mitigation: small entropy bonus on drive scalars during early training; cite-check ICM/RND tricks.
- **Place-cell-like units appear by chance** in any spatial encoder. Mitigation: shuffle controls in `eval/biology.py` (compare rate-map sparsity to a permuted-trajectory baseline).
- **Biological grounding can drift into LLM-style reasoning.** Hard rule: this trained model has no language inputs, no pretrained LM weights. Reviews catch this.
- **Reserved drive heads may degenerate to noise** if they receive zero gradient. Mitigation: detach them from the loss in v1 (no gradient flows), so they remain randomly initialized and ready for v2 activation.
