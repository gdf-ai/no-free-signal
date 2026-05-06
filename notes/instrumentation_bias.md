# Instrumentation bias — methods note

## Determinism sanity check

We tested whether the `--log-behavioral` instrumentation alters
simulation outcomes by running 6 seeds (arms A and C × seeds 0, 1, 2)
twice under current code on the same machine: once with the logger
attached (canonical behavioral dataset) and once without it.

| run | with logger (ticks) | without logger (ticks) | drift |
|---|---:|---:|---:|
| A_seed0 | 570 | 718 | -21% |
| A_seed1 | 15000 | 15000 | identical |
| A_seed2 | 795 | 828 | -4% |
| C_seed0 | 683 | 718 | -5% |
| C_seed1 | 15000 | 15000 | identical |
| C_seed2 | 713 | 755 | -6% |

A and C are no-LLM arms, so any divergence cannot come from LLM
non-determinism. The behavioral logger contains no random-number
generation at the Python level (verified by source inspection). The
most plausible mechanism for the observed drift is allocation-pattern
sensitivity: the logger creates many small Python objects per tick,
which shifts garbage-collection timing and therefore the order of
floating-point arithmetic in compound numpy expressions during
low-population steps. In near-extinction states, a creature crossing
an energy threshold one tick earlier or later changes the trajectory
non-trivially.

## Direction of the bias

- **Healthy populations (full 15000 ticks):** unaffected.
- **Stressed / extinction-bound populations:** the logger
  accelerates extinction by ~5-21% on the four sanity-check seeds
  that went extinct.

## Implication for the paper

Because the bias acts only on extinction-bound trajectories, arms
with higher baseline extinction rates absorb proportionally more of
it. In the canonical run, the LLM-active arms (B, D, E) had ~60%
extinction% versus ~40% for the controls (A, G), so the logger
biases against the LLM-active arms when fitness is measured on
instrumented runs.

We therefore use a **strict dataset separation** for this paper:

| claim | dataset |
|---|---|
| population fitness, extinction, AUC, LLM cost | `results_15k_fitness/` (logger OFF) |
| receiver-response behavior (per-emit events) | `results_15k_behavioral/` (logger ON) |

Behavioral receiver-response analyses depend only on per-emit event
records, which the logger captures correctly during whichever
simulation trajectory the run produces. The trajectory bias does not
contaminate per-event conditional analyses ("given a heard event,
what is the conditional fleeing probability?"). It only contaminates
trajectory-level summaries, which is why we use the uninstrumented
grid for those.

The two datasets share code, flags, seeds, and run parameters
(7 arms × 20 seeds × 15000 ticks; `--workers 8`, `--timeout 0`,
`--per-run-cap 50000`, `--n-creatures 25`, `--grid-size 24`,
`--refresh-every 15`, `--snapshot-every 50`, model `nova-lite`); the
sole difference is the presence of `--log-behavioral`.

## What we are not claiming

We do not stitch a causal story between the two datasets. A
statement like "behavior X explains fitness Y" mixes evidence from
two different simulation trajectories — even though produced by
identical code on identical seeds, they diverge in the
extinction-bound regime. Where behavioral and fitness findings
agree, we report them as independent corroborating signals; where
they disagree, we take the fitness signal from the uninstrumented
grid as authoritative for trajectory-level claims.
