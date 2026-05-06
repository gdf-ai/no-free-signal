# No Free Signal

Code, raw data, and reproducibility artifacts for the paper:

> **No Free Signal: A Negative Result for Substrate-Evolution Around Fixed LLMs in an Embodied Multi-Agent Population**
> Sterling Morrison — Independent Researcher, Generative Development Framework (GDF), GDF.ai
> 2026

The paper itself is at [`results_15k_fitness/REPORT.pdf`](results_15k_fitness/REPORT.pdf) (also `.docx` and `.md`).

## What this repository contains

A 7-arm × 20-seed × 15,000-tick controlled experiment testing whether a frozen LLM (Amazon Nova Lite) becomes adaptively useful in an embodied multi-agent population when selection pressure acts on a heritable communication substrate around the model. The study tests against matched controls (mute, no-emitter, scrambled LLM, replay-randomized LLM, random emitter at matched cadence) and reports a **negative fitness result** with a small but non-zero behavioral receiver-response signal.

### Headline finding

Under matched controls, fixed LLM emissions did not produce a population-level fitness advantage over scrambled, replayed, or random-emission controls. The cleanest internal contrast — F (replay-randomized LLM) vs G (uniform random emitter) — gave a coin flip. Population fitness was indistinguishable across all evolvable emission-bearing arms (D, E, F, G); the substrate-only no-emitter arm (C) was descriptively lowest. Receiver-response analysis showed small but real behavioral differences hidden by population fitness alone.

See `REPORT.md` / `REPORT.pdf` for the full paper.

## Layout

```
no_free_signal/         # experiment harness + bootstrap stats + report generator
  experiments/
    harness.py          # single-arm, single-seed runner
    parallel.py         # parallel grid orchestrator with cadence-protection abort
    stats.py            # paired-seed bootstrap + null-shuffle alignment
    metrics.py          # snapshot computation
    plot.py             # matplotlib figures (grayscale, print-friendly)
    report.py           # markdown / DOCX / PDF report assembler
    behavior_analysis.py
    behavioral_logger.py
    ablations.py        # per-arm wave transforms + RandomEmitterController
  llm_controller.py     # Bedrock-backed LLM controller with throttle protections
  llm_prose.py
  brains.py             # frozen torch policy net (held constant across arms)
  controller.py
  observation.py
  world.py
foresight/              # simulation engine (env, evolution, agent body)
  envs/unified_world.py
  evolution/genome.py
  models/
  training/
notes/
  instrumentation_bias.md   # logger on/off determinism check (cited in Methods)
results_15k_fitness/    # CANONICAL FITNESS DATA (logger off)
  *.jsonl               # 140 raw run files
  REPORT.md             # paper, markdown source
  REPORT.docx
  REPORT.pdf
  figures/              # auto-generated print-friendly figures
results_15k_behavioral/ # behavioral analysis
  behavior_results.json # analyzed receiver-response output (raw .jsonl files
                        # are 22 GB; available on request — see "Raw data" below)
validate_canonical.py   # cadence audit + completeness gate (exit-code-driven)
compare_determinism.py  # logger on/off sanity check (used to derive the
                        # instrumentation bias note)
```

## Reproducing the paper

```bash
# install
uv sync   # (or pip install -e .)

# regenerate the report from the existing JSONL data
python -m no_free_signal.experiments.report \
    --in results_15k_fitness \
    --behavior-in results_15k_behavioral \
    --out results_15k_fitness/REPORT.md

# audit gate (must exit 0 before trusting a fresh run)
python validate_canonical.py results_15k_fitness
```

To re-run the full grid from scratch:

```bash
# requires AWS Bedrock access for Nova Lite via cross-region inference profile
python -m no_free_signal.experiments.parallel \
    --confirm \
    --arms A,B,C,D,E,F,G \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --workers 4 \
    --n-steps 15000 \
    --snapshot-every 50 \
    --n-creatures 25 \
    --grid-size 24 \
    --refresh-every 15 \
    --model nova-lite \
    --out-dir results_15k_fitness \
    --timeout 0 \
    --per-run-cap 50000

# expect ~$80-100 Bedrock + ~14h wall on a c7i.8xlarge equivalent
# set NFS_TICK_RATE=10 and NFS_BEDROCK_RPS=99 if running on
# fast hardware where the simulation outruns Bedrock latency
```

For the **behavioral grid** (logger on, used only for receiver-response analysis), append `--log-behavioral` and use `--out-dir results_15k_behavioral`. The 22 GB of raw behavioral JSONLs can be re-derived this way — they are not committed because they exceed GitHub's per-repo limits.

## Raw data

- `results_15k_fitness/*.jsonl` — 140 fitness run files, ~24 MB total, included.
- `results_15k_behavioral/behavior_results.json` — analyzed receiver-response output, ~50 KB, included.
- `results_15k_behavioral/*.jsonl` — 140 raw behavioral run files, ~22 GB total, **not included** due to GitHub size limits. Re-derivable from the bootstrap above; available on request from the author.

## Statistical claims and reproducibility

All bootstrap claims are deterministic (RNG seeded at 42 in `no_free_signal/experiments/stats.py`). The paired-seed bootstrap uses 10,000 resamples; null-shuffle baselines for the alignment metric use 100 iterations per run. The behavioral receiver-response analysis bootstraps at the seed level, not the event level, to avoid pseudoreplication.

## Generative-AI use

This work uses Amazon Nova Lite as a study subject (the LLM under test in arms B, D, E, F). Anthropic Claude was used as a coding and writing assistant during development. Full disclosure is in the paper's "Generative-AI use disclosure" section. The author retains full responsibility for methodology, data integrity, and conclusions.

## License

MIT — see `LICENSE`.

## Citation

If this work is useful in your research, please cite:

```bibtex
@misc{morrison2026nofreesignal,
  title  = {No Free Signal: A Negative Result for Substrate-Evolution Around Fixed LLMs in an Embodied Multi-Agent Population},
  author = {Morrison, Sterling},
  year   = {2026},
  note   = {Independent Researcher, Generative Development Framework (GDF), GDF.ai},
  url    = {https://github.com/gdf-ai/no-free-signal}
}
```
