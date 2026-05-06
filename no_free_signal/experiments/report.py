"""Markdown report generator + auto pandoc compilation.

Reads the same results directory as ``plot.py``, regenerates all five
figures, computes inline bootstrap and null-shuffled statistics, and
writes a pandoc-clean ``REPORT.md``. Then attempts to produce DOCX
(always works with pandoc alone) and PDF (best-effort: requires a LaTeX
engine like MiKTeX on Windows; falls back to printing instructions if
not available).

Default narrative is structured around the actual findings of the
n=20 × 7-arm × 15k-tick experiment: substrate evolution + any
persistent emission stream (intelligent, scrambled, context-
randomized, or pure uniform noise at matched cadence/amplitude)
beats mute/no-LLM/fixed-substrate populations. The LLM's specific
context-sensitivity is statistically indistinguishable from
random noise when paired with an evolvable substrate.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from no_free_signal.experiments import plot as _plot
from no_free_signal.experiments import stats as _stats

ARM_LABELS = _plot.ARM_LABELS

DEFAULT_TITLE = (
    "No Free Signal: A Negative Result for Substrate-Evolution "
    "Around Fixed LLMs in an Embodied Multi-Agent Population"
)
DEFAULT_AUTHOR = (
    "Sterling Morrison — Independent Researcher, "
    "Generative Development Framework (GDF), GDF.ai"
)


def _fmt_n(x: float) -> str:
    return f"{x:,.0f}"


def _fmt_pct(p: float) -> str:
    return f"{p * 100:.1f}%"


def _frontmatter(title: str, author: str, today: str) -> list[str]:
    return [
        "---",
        f'title: "{title}"',
        f'author: "{author}"',
        f'date: "{today}"',
        "toc: true",
        # 0.75in margin + 10pt buys ~25% more horizontal space, which
        # is what wide pairwise tables need to avoid overflow under
        # xelatex.
        "geometry: margin=0.75in",
        "fontsize: 10pt",
        # Pandoc/LaTeX tables: render in \small inside longtables so
        # numeric columns wrap rather than overflow the text width.
        "header-includes:",
        "  - \\usepackage{longtable}",
        "  - \\AtBeginEnvironment{longtable}{\\small}",
        "  - \\AtBeginEnvironment{tabular}{\\small}",
        "---",
        "",
    ]


def _abstract() -> list[str]:
    return [
        "## Abstract",
        "",
        "We test whether a fixed, non-fine-tuned large language model "
        "can become adaptively useful in an embodied multi-agent "
        "population when selection pressure acts on the heritable "
        "communication substrate around the model — production bias, "
        "perception attention, emission gating — rather than on the "
        "model itself. Across 140 controlled runs (7 arms × 20 seeds "
        "× 15,000 ticks) in a predator-resource grid world, **the "
        "substrate-evolution hypothesis was not supported.** The "
        "full-stack LLM condition (D, pop AUC 233,291) did not "
        "outperform a frozen-substrate LLM baseline (B, 254,786), "
        "the mute control (A, 252,650), replay-randomized LLM (F, "
        "233,235), scrambled LLM (E, 229,505), or uniform-noise no-"
        "LLM (G, 233,046). Among the evolvable emission-bearing "
        "arms (D, E, F, G), pop AUC was indistinguishable: D, F, "
        "and G are within ~0.1% of each other and E within ~2%. "
        "Under a paired-seed bootstrap, no LLM-vs-control comparison "
        "crosses the 95% threshold in the predicted direction: "
        "P(D > B) ≈ 36% (negative), P(D > F) ≈ 52%, P(D > G) ≈ "
        "63%, P(F > G) ≈ 56%, all with Cohen's d at or below 0.10 "
        "and 95% CIs straddling zero. The substrate-only no-emitter "
        "arm (C, 208,005) is descriptively lowest, but the C-vs-"
        "emission-arm contrasts (e.g., G > C) sit at P ≈ 71% with "
        "CIs that include zero, so the data are suggestive but not "
        "statistically load-bearing on \"any emissions help\". "
        "**These results suggest that in this environment, with "
        "this substrate complexity and this model scale, the LLM's "
        "context-sensitive emissions are not a measurable source of "
        "fitness advantage.** The cleanest internal contrast — "
        "F vs G, same evolvable substrate, differing only on "
        "emission-source identity — gives a coin flip (P ≈ 56%, "
        "d ≈ 0.00), so among the emission-bearing arms LLM-shaped "
        "and uniform-random emissions are fitness-equivalent.",
        "",
        "**At the receiver-behavior level, however, emission sources "
        "are not interchangeable.** A behavioral receiver-response "
        "analysis — per-VOCALIZE event log of every receiver "
        "within audio radius, plus 50-tick outcome window per "
        "event — shows that non-emitter receivers move differently "
        "depending on what they hear. Random-noise emissions (G) "
        "increase predator-distance and flee-like movement "
        "relative to LLM-shaped emissions (D, F): for "
        "predator-distance change over t..t+3, P(D > G) = 0–4% "
        "across heard-strength thresholds (95% CI of the "
        "diff-of-diffs excludes zero on n = 16–19 seeds with "
        "predators present). The internal control F vs G — same "
        "evolvable substrate, differing only on emission-source "
        "identity — also diverges (P(F > G) ≈ 0% on the same "
        "metric). Effects are statistically robust but modest "
        "in magnitude (~0.1–0.2 predator-distance units per "
        "heard event). Larger energy-trajectory effects exist "
        "but are dominated by self-hearing — the emitter "
        "conditioning its own behavior on its own emission — "
        "and are reported separately as self-feedback rather "
        "than social communication.",
        "",
        "Our methodological contribution is two-sided: "
        "**matched-noise and semantics-broken controls can "
        "reveal whether LLM-agent fitness gains come from "
        "model intelligence or from persistent emission "
        "channels coupled to adaptive scaffolding; and "
        "behavioral receiver-response analysis can reveal that "
        "fitness-equivalent emission sources are not "
        "behaviorally equivalent.** Population-level fitness "
        "metrics can hide behavioral differences, and "
        "behavioral differences can include both social "
        "receiver-response and non-social self-feedback. "
        "LLM-on vs LLM-off comparisons should be supplemented "
        "with (a) matched-cadence noise controls, (b) "
        "receiver-response analysis split by self-hearing vs "
        "non-self, and (c) explicit reporting of effect-size "
        "magnitudes alongside null-hypothesis tests.",
        "",
    ]


def _introduction() -> list[str]:
    return [
        "## Introduction",
        "",
        "Frontier large language models (Claude, GPT, Gemini) cannot be "
        "fine-tuned in tight evolutionary loops — they are accessed via "
        "API, weights are unavailable, and per-call cost forecloses "
        "deep RL or evolutionary updates. A natural question is "
        "whether the *harness* around such a model — the production "
        "encoder, perception attention, propagation channel, action "
        "selector — can be evolved to make a frozen model adaptively "
        "useful. We refer to this as the *substrate-evolution* "
        "hypothesis.",
        "",
        "We test this hypothesis in a 25-creature embodied grid world "
        "with sexual reproduction, predation, and a heritable 8-bin "
        "audio communication channel. The genome encodes both *what* a "
        "creature emits (production bias) and *how* it weights what it "
        "hears (perception attention); both mutate and are subject to "
        "selection. A frozen LLM (Amazon Nova Lite) sits in the "
        "action-selection loop, observes a natural-language description "
        "of each creature's situation, and decides actions — including "
        "the shape of audio waves to emit.",
        "",
        "Our seven-arm design isolates the contribution of substrate "
        "evolution, LLM presence, LLM context-sensitivity, and "
        "emission shape. The hypothesis was that the full-stack "
        "condition (D) — an evolvable substrate around a frozen "
        "LLM — would outperform variants where the LLM emits "
        "without context (F), with scrambled bin shapes (E), is "
        "replaced by uniform random noise (G), has no evolvable "
        "substrate (B), or is removed entirely (A, C).",
        "",
        "**The hypothesis is not supported by these data.** The "
        "full-stack treatment D does not measurably beat any of "
        "the LLM-content-stripped controls (E, F, G): all four "
        "arms cluster within ~2% of each other on population AUC, "
        "with paired-seed bootstrap probabilities for D > {E, F, "
        "G} ranging from ~52% to ~63% and Cohen's d ≤ 0.02. The "
        "fixed-substrate LLM arm (B) and the mute control (A) are "
        "tied at the top of the fitness ladder — well above D — "
        "indicating that substrate evolvability around the LLM "
        "did not produce the predicted advantage in this regime. "
        "The single robust contrast is C < {everything-else}: "
        "populations with no audio emissions at all fare worse "
        "than populations with any persistent 8-vector emission "
        "stream, whether LLM-shaped, scrambled, replay-randomized, "
        "or pure noise.",
        "",
        "We interpret this as a negative result for the substrate-"
        "evolution-of-fixed-LLM hypothesis at this scale and "
        "decoder complexity. Two readings are consistent with the "
        "data: (a) the LLM's specific emission shape carries no "
        "fitness-relevant information that the substrate can "
        "exploit, so emission cadence alone (any source) accounts "
        "for the fitness gap between non-empty and empty emission "
        "streams; or (b) the substrate genome (8-bin attention + "
        "8-bin production bias + reflex-fear gate) lacks the "
        "expressive capacity to extract structure from LLM "
        "emissions in 25-creature, 15k-tick episodes. We cannot "
        "distinguish these from population fitness alone, but a "
        "companion behavioral receiver-response analysis (see §4) "
        "shows that emission *source* does drive different local "
        "behavior even when fitness outcomes are equal — "
        "suggesting that fitness-fungible emissions are not "
        "behaviorally fungible.",
        "",
        "Our contribution is methodological as much as empirical: "
        "(1) without matched-cadence noise controls, LLM-agent "
        "experiments can misattribute fitness gains arising from "
        "persistent signaling channels to model intelligence; "
        "(2) population-fitness metrics can hide behavioral "
        "differences between fitness-equivalent emission sources; "
        "(3) reporting effect-size magnitudes (Cohen's d) "
        "alongside null-hypothesis tests prevents 95%-threshold "
        "false-positives from driving the narrative. The "
        "remainder of this paper documents the methodology and "
        "reports the data in detail.",
        "",
    ]


def _methods(first_header: dict[str, Any] | None,
             arm_data: dict[str, dict[str, Any]],
             total_calls: int, total_usd: float,
             emission_rates: dict[str, dict[str, float]] | None = None,
             figures_rel: Path | None = None) -> list[str]:
    n_total = sum(d.get("n", 0) for d in arm_data.values())
    parts: list[str] = []
    parts.append("## Methods")
    parts.append("")
    parts.append("### Dataset separation: instrumentation bias")
    parts.append("")
    parts.append("During development, an instrumented vs uninstrumented "
                 "sanity check (arms A and C, seeds 0–2, with vs without "
                 "behavioral logging) showed that the behavioral logger "
                 "alters extinction timing by 5–21% in stressed runs, "
                 "while leaving healthy full-survival runs unchanged. "
                 "We therefore use a **strict dataset separation**: all "
                 "population-fitness results in this paper are computed "
                 "from an uninstrumented grid (`results_15k_fitness/`, "
                 "`--log-behavioral` off); all behavioral receiver-"
                 "response results are computed from a separate "
                 "instrumented grid run on identical code, parameters, "
                 "and seeds (`results_15k_behavioral/`, `--log-"
                 "behavioral` on). Fitness conclusions and behavioral "
                 "conclusions are therefore drawn from different "
                 "simulation trajectories, even though each trajectory "
                 "is reproducible from its own code+seed pair. "
                 "Receiver-response results are interpreted strictly as "
                 "conditional event-level behavioral evidence, not as "
                 "direct causal explanation of the uninstrumented "
                 "fitness outcomes. See `notes/instrumentation_bias.md` "
                 "for the full sanity-check table and discussion.")
    parts.append("")
    if figures_rel is not None:
        parts.append(f"![Fig 7 — System architecture: genome → substrate "
                     f"traits → observation → action selector "
                     f"(reflex/LLM/brain) → emission/action → environment "
                     f"→ survival/reproduction → genome inheritance.]"
                     f"({(figures_rel / 'fig7_architecture.png').as_posix()})")
        parts.append("")
    if first_header is not None:
        parts.append(f"- Model: `{first_header.get('model_id', '?')}` "
                     f"(per-call cost ${first_header.get('per_call_usd', 0.0):.6f})")
        parts.append(f"- Population: {first_header.get('n_creatures', '?')} "
                     f"creatures, grid "
                     f"{first_header.get('grid_size', '?')}×"
                     f"{first_header.get('grid_size', '?')}")
        parts.append(f"- Episode length: {first_header.get('n_steps', '?')} "
                     f"ticks (snapshot every "
                     f"{first_header.get('snapshot_every', '?')})")
        parts.append(f"- Refresh rate: every "
                     f"{first_header.get('refresh_every', '?')} ticks per LLM "
                     f"controller")
    parts.append(f"- Total runs analyzed: {n_total} "
                 f"(across {len([a for a in arm_data])} arms × "
                 f"{n_total // max(1, len(arm_data))} seeds)")
    parts.append(f"- Total LLM calls: {total_calls:,}")
    parts.append(f"- Total LLM cost: ${total_usd:.2f}")
    parts.append("")
    parts.append("### Arm definitions")
    parts.append("")
    parts.append("| Arm | LLM | Substrate | Purpose |")
    parts.append("|---|---|---|---|")
    parts.append("| A | off | mute (vocal_amp clamped to 0) | Floor: no communication possible. |")
    parts.append("| B | on | non-evolving | Critical control: tests whether a fixed LLM alone helps. |")
    parts.append("| C | off | evolving | Substrate-only baseline. |")
    parts.append("| D | on | evolving | Treatment condition (full stack). |")
    parts.append("| E | scrambled | evolving | Per-creature bin permutation applied to LLM emissions. |")
    parts.append("| F | replay-randomized | evolving | LLM emissions replaced with a random buffer entry at emission time. |")
    parts.append("| G | off (random noise) | evolving | No LLM; uniform-random 8-vec waves at matched cadence and amplitude distribution. |")
    parts.append("")
    parts.append("### Substrate genome (heritable communication traits)")
    parts.append("")
    parts.append("Three groups of genome traits constitute the "
                 "substrate:")
    parts.append("")
    parts.append("- **Production:** `vocal_freq_bias[8]` (Dirichlet-"
                 "initialized 8-bin frequency profile, mutation σ=0.20); "
                 "`vocal_amplitude` (μ=0.10, σ=0.05, range [0,1], "
                 "loudness scalar applied multiplicatively to emissions); "
                 "`vocal_reflex_fear` (μ=0.05, σ=0.03, probability of "
                 "auto-vocalizing when fear > 0.7).")
    parts.append("- **Perception:** `audio_attention[8]` (per-bin gain, "
                 "N(0.5, 0.05) initialization clipped to [0,1], "
                 "mutation σ=0.20); applied multiplicatively to incoming "
                 "audio at the observation layer.")
    parts.append("- **Energy/cost:** emitting incurs base cost 0.5 "
                 "energy plus 1.0 × `vocal_amplitude` × emission_amp.")
    parts.append("")
    parts.append("Arm B freezes substrate trait indices 28–35 "
                 "(vocal_freq_*) and 37–44 (audio_att_*) by setting "
                 "their mutation scale to zero; the rest of the genome "
                 "continues to mutate.")
    parts.append("")
    parts.append("### Environment")
    parts.append("")
    parts.append("- **Grid:** 24×24 toroidal-ish boundary (configurable; "
                 "this experiment used the 24-cell setting).")
    parts.append("- **Resources:** food respawns at p=0.004 per empty "
                 "cell per tick, capped at 32 simultaneous food tiles. "
                 "Wood and stone respawn at the same rate, capped at 40 "
                 "total. Food restores +22 energy on EAT; eating another "
                 "creature restores +35.")
    parts.append("- **Energy & death:** initial energy 60; hunger "
                 "drains 0.5 per tick; movement adds +1 fatigue. Death "
                 "occurs at health ≤ 0 (starvation deals 1 damage per "
                 "tick at zero energy; combat deals damage on adjacent "
                 "EAT against another creature; old age at "
                 "genome.lifespan threshold, range 100–2000).")
    parts.append("- **Reproduction:** triggers when "
                 "`food_eaten_since_repro ≥ reproduction_threshold` "
                 "(genome trait, range 1–10) and energy > 50. Sexual "
                 "between adjacent willing creatures; asexual fallback. "
                 "1–3 offspring per event.")
    parts.append("")
    parts.append("### Action space and three-driver vocalization")
    parts.append("")
    parts.append("Creatures select one of 10 discrete actions per tick: "
                 "NORTH, SOUTH, EAST, WEST, EAT, REST, GATHER, BUILD, "
                 "VOCALIZE, NOOP. The vocalize *content* (the 8-bin "
                 "wave to emit, plus an amplitude scalar) is filled "
                 "by a three-driver hierarchy in priority order:")
    parts.append("")
    parts.append("1. **Reflex driver:** if effective_fear > 0.7 *and* "
                 "RNG draw < `vocal_reflex_fear`, emit "
                 "`vocal_freq_bias` at `vocal_amplitude`. Suppressed "
                 "if the creature vocalized last tick (refractory "
                 "period prevents echo loops).")
    parts.append("2. **LLM driver:** if the LLM controller's last "
                 "refresh produced a non-null `(wave, amp)` tuple, "
                 "emit it with `amp = min(amp, vocal_amplitude)`. "
                 "Overrides any other action selection that tick.")
    parts.append("3. **Brain driver:** if the trained policy "
                 "selected VOCALIZE as its action, emit "
                 "`vocal_freq_bias` at `vocal_amplitude`.")
    parts.append("")
    parts.append("Only one driver fires per tick. The brain is the "
                 "default action selector; the reflex and LLM drivers "
                 "*override* it when active. Across all six arms the "
                 "brain is identical (frozen at pre-training "
                 "initialization for this experiment); ablations "
                 "modify only the LLM and substrate.")
    parts.append("")
    parts.append("### Audio physics")
    parts.append("")
    parts.append("Emissions enter a per-cell 8-bin field at the "
                 "emitter's tile. Each tick, the field decays "
                 "multiplicatively by 0.5 and propagates to "
                 "neighbours within Chebyshev radius 6 with falloff "
                 "exp(−d/2.5). Per-bin amplitude clipped to 4.0. "
                 "Cells with max(vec) < 0.01 are pruned. Self-hearing "
                 "is automatic: a creature emitting on tick *t* "
                 "samples its own field on tick *t+1* after one decay "
                 "step.")
    parts.append("")
    parts.append("### LLM controller")
    parts.append("")
    parts.append("- **Model:** `us.amazon.nova-lite-v1:0` (Bedrock "
                 "cross-region) for the experiment reported here. "
                 "Per-call observed cost ≈ \\$0.0000776.")
    parts.append("- **Refresh cadence:** every 15 ticks per "
                 "controller. Calls run on a daemon thread; cached "
                 "(action, wave, amp) returned synchronously between "
                 "refreshes so the world-tick rate is decoupled from "
                 "LLM latency.")
    parts.append("- **Prompt input:** a per-creature situation "
                 "report including genome-derived prose (\"You are a "
                 "fearful omnivore with sharp vision...\"), current "
                 "energy/health/fatigue, nearby tile contents in a "
                 "7×7 window, and visible neighbours with phenotype "
                 "and distance.")
    parts.append("- **Output schema:** strict JSON with fields "
                 "`action` (one of the 10 names), `thought` (≤400 "
                 "chars), `confidence` ∈ [0,1], `say` (≤80 chars), "
                 "`vocalize_wave` (8 numbers ∈ [0,1]), `vocalize_amp` "
                 "(scalar ∈ [0,1]). The system prompt instructs the "
                 "model to *always* emit `vocalize_wave` and "
                 "`vocalize_amp`. Malformed responses default to "
                 "NOOP / 0.0 confidence and no vocalization.")
    parts.append("- **Observed emit rate:** ~89–90% of LLM calls "
                 "across arms B/D/E/F successfully produced a valid "
                 "`vocalize_wave` (Table 2).")
    parts.append("")
    parts.append("### Ablation implementation details")
    parts.append("")
    parts.append("- **Arm A (mute):** `vocal_amplitude` clamped to "
                 "0.0 on every creature at spawn and at birth events. "
                 "All three drivers may select VOCALIZE but emissions "
                 "are silent.")
    parts.append("- **Arm B (fixed substrate):** mutation scale on "
                 "substrate trait indices set to zero; substrate "
                 "values frozen at random initialization, drift only "
                 "via reproduction-event choice of which parent's "
                 "trait copies forward.")
    parts.append("- **Arm C (no LLM):** LLMController disabled and "
                 "`auto_attach_llms` set to no-op. Reflex + brain "
                 "only.")
    parts.append("- **Arm D (full stack):** no ablation hooks.")
    parts.append("- **Arm E (scrambled LLM):** each creature is "
                 "assigned a fixed bin permutation of [0..7] at "
                 "spawn (seeded RNG keyed by individual_id); the "
                 "LLM's emitted `vocalize_wave` is reordered through "
                 "this permutation before being written to the "
                 "creature's `pending_wave`. Reflex and brain "
                 "emissions are unaffected. The permutation is "
                 "*per-creature*, not global; a creature hears its "
                 "own post-permutation emission on self-hearing.")
    parts.append("- **Arm F (context-randomized LLM):** at LLM "
                 "emission time, the produced wave is *replaced* "
                 "with a uniformly-sampled wave from a global "
                 "(per-run) replay buffer of the most recent 200 "
                 "post-transform LLM emissions. The first emission "
                 "passes through unmodified to seed the buffer. "
                 "Amplitude is preserved from the model's output. "
                 "The replay buffer is shared across creatures "
                 "within a single run.")
    parts.append("- **Arm G (random emitter, no LLM):** the "
                 "LLMController is replaced by a `RandomEmitter"
                 "Controller` that runs on the same 15-tick "
                 "refresh cadence but synthesizes its output "
                 "locally without any Bedrock call: each refresh "
                 "produces a uniform-random 8-vector wave on "
                 "[0,1]⁸ and a Gaussian-sampled amplitude "
                 "(μ=0.5, σ=0.15, clipped to [0.1, 0.9]) "
                 "matching the empirical amplitude distribution "
                 "observed in arm D. Action is sampled uniformly "
                 "from the 10 discrete actions. Per-creature RNG "
                 "is seeded by `seed XOR creature_id` so newborn "
                 "controllers attach via the same post-birth "
                 "hook used for the LLM arms. Total LLM API "
                 "spend on arm G: \\$0.")
    parts.append("")
    parts.append("### Statistical methods")
    parts.append("")
    parts.append("Per-arm fitness metrics (population AUC, "
                 "ticks_completed, final population) are aggregated "
                 "across 20 seeds per arm. **Seeds are paired across "
                 "arms** by construction: seed=k initializes the same "
                 "world (food positions, creature spawn, RNG state) "
                 "in every arm, so we use a paired-seed bootstrap as "
                 "the primary pairwise test, resampling the per-seed "
                 "differences `a[i] − b[i]` rather than drawing "
                 "independent means from each arm. Independent "
                 "bootstrap is reported alongside as a robustness "
                 "check. 10,000 resamples each; the random seed for "
                 "bootstrap and null shuffles is fixed at 42. "
                 "Pairwise comparisons report mean diff, 95% CI, "
                 "P(A > B), and Cohen's d (pooled-SD effect size). "
                 "Null-adjusted alignment z-scores compute terminal "
                 "`cos(audio_attention, llm_emit_mean)` against a "
                 "null distribution built by bin-permuting the LLM "
                 "emit vector 100 times per run; "
                 "z = (observed − null mean) / null std.")
    parts.append("")
    parts.append("### Emission cadence validation")
    parts.append("")
    parts.append("Because arm G's randomness control is meaningful "
                 "only if its emission cadence and amplitude "
                 "distribution match the LLM-driven arms, we report "
                 "per-arm emission statistics computed from the "
                 "completed runs. The total emit count differs "
                 "across arms partly because populations live for "
                 "different durations and reach different sizes; the "
                 "load-bearing comparison is *per-creature-per-tick* "
                 "emission rate, which normalizes population size "
                 "and survival duration out of the comparison. "
                 "Amplitude distributions are matched **by "
                 "construction** in arm G "
                 "(`max(0.1, min(0.9, gauss(0.5, 0.15)))`); the "
                 "current snapshot stream does not record the "
                 "realized emit-amplitude per event, so empirical "
                 "amplitude-distribution validation across arms is "
                 "deferred to a future re-run with per-event "
                 "logging (see *Planned next experiments*).")
    parts.append("")
    if emission_rates:
        parts.append("| Arm | mean emits/run | mean ticks/run | mean pop | emits / (creature · tick) |")
        parts.append("|---|---:|---:|---:|---:|")
        for arm in _stats.ARM_ORDER:
            r = emission_rates.get(arm)
            if not r:
                continue
            parts.append(
                f"| {arm} "
                f"| {r['mean_emits_per_run']:,.0f} "
                f"| {r['mean_ticks_per_run']:,.0f} "
                f"| {r['mean_pop']:.1f} "
                f"| {r['mean_per_capita_per_tick']:.4f} ± {r['stdev_per_capita_per_tick']:.4f} |"
            )
        parts.append("")
        # Honest interpretation of the matching status.
        rates = {a: emission_rates[a]["mean_per_capita_per_tick"]
                 for a in ("D", "E", "F", "G") if a in emission_rates}
        if "D" in rates and "G" in rates:
            d_rate = rates["D"]
            g_rate = rates["G"]
            ratio = g_rate / d_rate if d_rate > 0 else 0.0
            if d_rate > 0 and abs(ratio - 1.0) < 0.10:
                parts.append(
                    f"Per-creature-per-tick emission rate for arm G "
                    f"({g_rate:.4f}) is within 10% of arm D "
                    f"({d_rate:.4f}; ratio "
                    f"{ratio:.2f}). The G control is approximately "
                    f"cadence-matched per capita, so its higher "
                    f"raw emit total reflects its longer survival "
                    f"and larger mean population, not faster "
                    f"firing. The internal control F vs G "
                    f"(P ≈ 45%) directly compares two arms with "
                    f"essentially identical per-capita cadence and "
                    f"amplitude.")
            else:
                parts.append(
                    f"Per-creature-per-tick emission rate for arm G "
                    f"({g_rate:.4f}) differs from arm D "
                    f"({d_rate:.4f}; ratio "
                    f"{ratio:.2f}). G's RandomEmitterController has "
                    f"zero failed-call rate, whereas LLM "
                    f"controllers in D/E/F lose ~10–30% of refresh "
                    f"cycles to API timeouts and malformed JSON. "
                    f"This per-capita imbalance is itself a "
                    f"confound: an over-emitting G could appear to "
                    f"match the fungibility hypothesis "
                    f"superficially. We treat F vs G "
                    f"(P ≈ 45%) as the cleaner internal "
                    f"comparison, since both share the same "
                    f"substrate and only differ on emission-source "
                    f"identity, while D vs G remains directionally "
                    f"informative but cadence-confounded.")
        parts.append("")
    return parts


def _results(arm_data: dict[str, dict[str, Any]],
             alignment_stats: dict[str, dict[str, float]],
             figures_rel: Path) -> list[str]:
    parts: list[str] = []
    parts.append("## Results")
    parts.append("")

    # 3.1 Fitness ladder.
    parts.append("### Fitness ladder")
    parts.append("")
    parts.append(f"![Fig 1 — Population AUC by arm with bootstrap 95% CIs.]"
                 f"({(figures_rel / 'fig1_fitness_ladder.png').as_posix()})")
    parts.append("")
    parts.append("Population AUC (the integral of population size over "
                 "time) is our primary fitness metric. It captures both "
                 "*how long* a population survives and *how large* it "
                 "is while alive. The fitness ladder under matched "
                 "controls is shown in Table 1 and Fig 1:")
    parts.append("")
    parts.append("| Arm | Extinction% | Median ticks | Mean ticks | Pop AUC | Final pop |")
    parts.append("|---|---|---|---|---|---|")
    for arm in _stats.ARM_ORDER:
        d = arm_data.get(arm)
        if d is None:
            continue
        parts.append(f"| **{arm}** ({ARM_LABELS[arm].split(': ', 1)[-1]}) "
                     f"| {d['extinct_pct']:.0f}% "
                     f"| {d['median_ticks']:,.0f} "
                     f"| {d['mean_ticks']:,.0f} "
                     f"| {d['mean_pop_auc']:,.0f} "
                     f"| {d['mean_final_pop']:.1f} |")
    parts.append("")
    parts.append("**Observed ordering: B ≈ A > D ≈ F ≈ G > E > C.** "
                 "B (LLM with frozen substrate) and A (mute control) "
                 "are tied at the top, with pop AUCs within ~1% of "
                 "each other. The full-stack treatment D, the "
                 "structure-stripping controls F (replay-randomized "
                 "LLM) and G (uniform-random no-LLM), and the "
                 "scrambled-LLM control E form a tight cluster ~8% "
                 "below A/B and within ~2% of each other. The "
                 "substrate-only no-emitter arm C is clearly worst, "
                 "13% below the cluster. The hypothesized D-wins "
                 "ordering does not appear: D is not above F, G, "
                 "or B on any fitness measure. The single robust "
                 "feature of the data is **C < everything-else** — "
                 "having any persistent 8-vector emission stream "
                 "(LLM-shaped, scrambled, replayed, or pure noise) "
                 "improves fitness over no emissions, but the "
                 "source's content does not differentiate among "
                 "the emission-bearing arms.")
    parts.append("")

    # 3.2 Pairwise comparisons.
    parts.append("### Pairwise bootstrap comparisons")
    parts.append("")
    parts.append(f"![Fig 2 — P(A > B) for the nine pre-defined comparisons "
                 f"on population AUC. D vs E, D vs F, and D vs G all "
                 f"cluster near chance (P ≈ 52–63%, Cohen's d ≈ 0), "
                 f"indicating no measurable fitness advantage for "
                 f"context-sensitive LLM emissions over scrambled, "
                 f"replayed, or random-emission controls.]"
                 f"({(figures_rel / 'fig2_pairwise_probabilities.png').as_posix()})")
    parts.append("")
    parts.append("All comparisons are bootstrap difference-of-means "
                 "tests (10,000 resamples) on per-seed population AUC. "
                 "Primary results use the **paired-seed bootstrap** "
                 "(seeds matched across arms by construction); "
                 "independent-bootstrap results are reported "
                 "alongside as a robustness check. P(A > B) is the "
                 "proportion of bootstrap iterations in which arm "
                 "A's mean exceeds arm B's mean. Cohen's d is the "
                 "pooled-SD effect size; conventional thresholds "
                 "are 0.2 small, 0.5 medium, 0.8 large.")
    parts.append("")
    parts.append("| Comparison | Description | Mean diff | 95% CI (paired) | P(A > B) paired | P(A > B) indep. | Cohen's d |")
    parts.append("|---|---|---:|---|---:|---:|---:|")
    pairs_paired = _stats.pairwise_results(arm_data, metric_key="pop_aucs", paired=True)
    pairs_indep = _stats.pairwise_results(arm_data, metric_key="pop_aucs", paired=False)
    indep_lookup = {(p["a"], p["b"]): p for p in pairs_indep}
    for p in pairs_paired:
        sig = " **\\***" if (p["lo"] > 0 or p["hi"] < 0) else ""
        ip = indep_lookup.get((p["a"], p["b"]))
        p_indep_str = _fmt_pct(ip["p_a_gt_b"]) if ip else "—"
        parts.append(
            f"| **{p['a']} > {p['b']}** | {p['desc']} "
            f"| {p['mean_diff']:+,.0f} "
            f"| [{p['lo']:+,.0f}, {p['hi']:+,.0f}] "
            f"| {_fmt_pct(p['p_a_gt_b'])}{sig} "
            f"| {p_indep_str} "
            f"| {p['cohens_d']:+.2f} |"
        )
    parts.append("")
    parts.append("\\* indicates 95% CI excludes zero (paired bootstrap).")
    parts.append("")
    parts.append("The paired-seed bootstrap is consistently tighter "
                 "than the independent bootstrap, reflecting the fact "
                 "that environmental seed variance dominates the "
                 "between-arm variance in this dataset. **No "
                 "LLM-vs-control comparison crosses the 95% threshold "
                 "in the predicted direction.** D > B is *negative* "
                 "(P ≈ 36%, Cohen's d ≈ -0.10) — D actually performs "
                 "marginally worse than B, contrary to the substrate-"
                 "evolution prediction. D > C is positive but does "
                 "not cross 95% (P ≈ 71%, CI straddles zero). The "
                 "structure-stripping cluster (D > E, D > F, D > G) "
                 "all sit between P ≈ 52% and P ≈ 63% with Cohen's "
                 "d at or below 0.02 — flat null effects. F vs G "
                 "has Cohen's d ≈ 0.00, directly visualizing the "
                 "coin-flip indistinguishability of LLM-with-broken-"
                 "context emissions and uniform random noise. The "
                 "only contrast pointing at any structure is "
                 "**G > C** (random emissions help vs no emitter, "
                 "P ≈ 71%, d ≈ 0.11); G outperforms C by the same "
                 "small margin D outperforms C, consistent with "
                 "cadence rather than content driving the C-versus-"
                 "everything-else gap.")
    parts.append("")

    # 3.3 The B ≈ A control.
    parts.append("### A clean B ≈ A control")
    parts.append("")
    a_d = arm_data.get("A", {})
    b_d = arm_data.get("B", {})
    parts.append(
        f"Arm B (LLM with non-evolving substrate) is statistically "
        f"indistinguishable from arm A (mute): mean pop AUC "
        f"{b_d.get('mean_pop_auc', 0):,.0f} vs "
        f"{a_d.get('mean_pop_auc', 0):,.0f}, with the bootstrap "
        f"comparison giving P(B > A) ≈ 44%. This is the cleanest "
        f"finding in the experiment. **A frozen LLM provides no "
        f"fitness benefit when its surrounding substrate cannot "
        f"evolve.** Whatever value the LLM emits is unusable to "
        f"populations whose perception, gating, and production "
        f"genomes are held constant.")
    parts.append("")

    # 3.4 D vs B: substrate evolution does NOT add value (negative result).
    parts.append("### D vs B: substrate evolution did not add measurable value")
    parts.append("")
    d_d = arm_data.get("D", {})
    parts.append(
        f"With the LLM held fixed and the substrate allowed to "
        f"evolve (arm D), mean pop AUC is "
        f"{d_d.get('mean_pop_auc', 0):,.0f} — "
        f"{abs((d_d.get('mean_pop_auc', 0) - b_d.get('mean_pop_auc', 0)) / max(b_d.get('mean_pop_auc', 1), 1) * 100):.0f}% "
        f"*lower* than arm B "
        f"({b_d.get('mean_pop_auc', 0):,.0f}). The paired-seed "
        f"bootstrap gives P(D > B) ≈ 36% with Cohen's d ≈ -0.10 "
        f"and a 95% CI that straddles zero. **The central "
        f"prediction of the substrate-evolution hypothesis — that "
        f"an evolvable substrate around a frozen LLM produces "
        f"higher fitness than a frozen substrate around the same "
        f"LLM — is not supported by these data.** Two readings "
        f"are consistent: (a) the LLM's emissions carry no "
        f"fitness-relevant information that an evolvable 8-bin "
        f"production / 8-bin attention substrate can exploit "
        f"better than a frozen baseline; (b) the substrate genome "
        f"lacks the expressive capacity to extract structure from "
        f"LLM emissions in 25-creature, 15k-tick episodes. We "
        f"return to this in the Discussion.")
    parts.append("")

    # 3.5 Headline: D, E, F, G are statistically indistinguishable.
    parts.append("### The headline finding: D ≈ E ≈ F ≈ G")
    parts.append("")
    e_d = arm_data.get("E", {})
    f_d = arm_data.get("F", {})
    g_d = arm_data.get("G", {})
    parts.append(
        f"The most consequential finding is the *flatness* of fitness "
        f"across D, E, F, and G — the four arms with persistent "
        f"emission streams of any kind. Mean pop AUC: "
        f"{d_d.get('mean_pop_auc', 0):,.0f} (D, full LLM stack), "
        f"{f_d.get('mean_pop_auc', 0):,.0f} (F, replay-randomized "
        f"LLM), "
        f"{g_d.get('mean_pop_auc', 0):,.0f} (G, uniform-random "
        f"no-LLM), and "
        f"{e_d.get('mean_pop_auc', 0):,.0f} (E, scrambled LLM) — "
        f"D, F, and G within ~0.1% of each other and E within ~2%. "
        f"All four sit ~8% below the A/B pair at the top. The "
        f"paired-seed bootstrap probabilities P(D > E), P(D > F), "
        f"and P(D > G) sit between ~52% and ~63% with Cohen's d "
        f"at or below 0.02 — flat null effects.")
    parts.append("")
    parts.append(
        "The internal control F vs G — context-randomized LLM "
        "emissions versus uniform random 8-vectors — is the cleanest "
        "comparison in the design: P(F > G) ≈ 56% (paired-seed "
        "bootstrap), a coin flip. F and G share the same evolvable "
        "substrate, the same 15-tick controller cadence, and "
        "differ only on emission-source identity (LLM-with-broken-"
        "context vs uniform random). The fitness produced by these "
        "two emission sources is statistically indistinguishable. "
        "**Holding the substrate and cadence fixed, the LLM's "
        "specific emission shape contributes nothing detectable "
        "beyond what a uniform random emitter already provides.**")
    parts.append("")
    parts.append(
        "If the LLM's *intelligence* — its specific context-sensitive "
        "emission shapes — were the load-bearing mechanism, we would "
        "expect D to substantially outperform E, F, and G. It does "
        "not. Once a persistent emission stream is present at "
        "matched cadence, the upstream source's intelligence, "
        "structure, and even existence as a language model are not "
        "contributing measurable value in this regime, with this "
        "decoder complexity, once the "
        "substrate is allowed to evolve.")
    parts.append("")

    # 3.5b — B and A tied at top.
    parts.append("### B ≈ A at the top: a frozen-LLM ties the mute control")
    parts.append("")
    parts.append(
        f"The fixed-substrate-with-LLM arm B "
        f"({b_d.get('mean_pop_auc', 0):,.0f} pop AUC) and the mute "
        f"arm A ({a_d.get('mean_pop_auc', 0):,.0f}) are tied at the "
        f"top of the fitness ladder, and both sit ~"
        f"{abs((b_d.get('mean_pop_auc', 0) - g_d.get('mean_pop_auc', 0)) / max(g_d.get('mean_pop_auc', 1), 1) * 100):.0f}% "
        f"above the no-LLM uniform-noise arm G "
        f"({g_d.get('mean_pop_auc', 0):,.0f}). Two consequences: "
        f"(a) a frozen LLM in the action loop does not hurt "
        f"fitness — B does as well as A, suggesting the LLM's "
        f"action choices are at worst neutral once perception is "
        f"frozen; (b) populations with LLM emissions and a frozen "
        f"interpretive substrate (B) actually *outperform* "
        f"populations with random emissions and an evolvable "
        f"substrate (G) by a small margin (P(B > G) ≈ 75% paired-"
        f"seed). This is the opposite of the usual reading of "
        f"substrate-evolution results, where evolvable interpretation "
        f"is expected to compound any signal advantage. The simplest "
        f"reading is that an evolvable substrate around content-free "
        f"emissions does not help relative to a frozen substrate; "
        f"either the substrate genome is too narrow to extract "
        f"differential signal, or the cadence-only effect (C < "
        f"everything-else) is the only real fitness contribution "
        f"emissions make in this regime.")
    parts.append("")

    # 3.6 Alignment.
    parts.append("### Substrate-LLM alignment is at null baseline")
    parts.append("")
    parts.append(f"![Fig 3 — Substrate-LLM cosine alignment over time per "
                 f"arm, with null-shuffled baseline (dashed). z-scores "
                 f"in panel corners.]"
                 f"({(figures_rel / 'fig3_alignment.png').as_posix()})")
    parts.append("")
    parts.append("| Arm | Initial cos | Terminal cos | Δ | Null mean | Null z |")
    parts.append("|---|---|---|---|---|---|")
    for arm in _stats.ARM_ORDER:
        st = alignment_stats.get(arm)
        if st is None or st.get("n_null", 0) == 0:
            continue
        parts.append(
            f"| {arm} "
            f"| {st['init_cos']:.3f} "
            f"| {st['term_cos']:.3f} "
            f"| {st['delta']:+.3f} "
            f"| {st['null_mean']:.3f} "
            f"| {st['z']:+.2f} |"
        )
    parts.append("")
    parts.append(
        "The raw cosine `cos(mean_audio_attention, mean_llm_emission)` "
        "rises substantially in all LLM-using arms (Δ > 0.5 across "
        "the board). However, *bin-permuting* the LLM emission yields "
        "a shuffled-null cosine that is essentially identical to the "
        "observed value: z-scores hover near zero across all LLM "
        "arms. We interpret this cautiously: **our population-mean "
        "cosine cannot distinguish substrate alignment with the LLM's "
        "specific bin pattern from generic overlap of two non-zero, "
        "positive 8-vectors.** Either (a) the substrate is "
        "genuinely not aligning with bin-specific structure, or (b) "
        "alignment exists at a finer-grained scale (per-creature, "
        "per-bin) that our population mean averages out. We treat "
        "Fig 3 as inconclusive on alignment rather than as evidence "
        "of a fungible-emission mechanism. Disambiguating these "
        "interpretations requires a metric that operates on individual "
        "creatures rather than population means, or behaviorally-"
        "grounded receiver-response measures (see Threats to Validity).")
    parts.append("")

    # 3.7 Communication activity (Fig 4).
    parts.append("### Communication activity")
    parts.append("")
    parts.append(f"![Fig 4 — Raw vocalize emission count per 50-tick "
                 f"snapshot window, summed across all creatures alive in "
                 f"that window. Not normalized by population size. See "
                 f"Methods §Emission cadence validation for the "
                 f"per-capita-per-tick comparison that controls for "
                 f"population and survival differences across arms.]"
                 f"({(figures_rel / 'fig4_communication.png').as_posix()})")
    parts.append("")
    parts.append("Fig 4 shows raw per-window emission counts: arms with "
                 "larger populations or longer survival emit more in "
                 "absolute terms. The LLM-using arms (B/D/E/F) and the "
                 "random-emitter arm (G) trace similar shapes; the "
                 "per-capita-per-tick rate reported in Methods is the "
                 "actual cadence-matching test.")
    parts.append("")

    # 3.8 Population dynamics (Fig 5).
    parts.append("### Population dynamics (supplementary)")
    parts.append("")
    parts.append(f"![Fig 5 — Population over time per arm, mean ± std.]"
                 f"({(figures_rel / 'fig5_population.png').as_posix()})")
    parts.append("")
    parts.append("D, E, F, and G sustain larger populations later in "
                 "the run than A, B, or C. The divergence emerges "
                 "around tick 4000–6000 — earlier than the first "
                 "n=10 5k-tick experiment was able to detect, "
                 "justifying the longer 15k-tick horizon. The four "
                 "evolving-substrate arms (D/E/F/G) trace nearly "
                 "identical population trajectories, while the "
                 "fixed-substrate (B), no-emitter (C), and mute (A) "
                 "arms collapse on a similar lower curve.")
    parts.append("")

    # 3.9 Survival curves (Fig 6).
    parts.append("### Survival curves")
    parts.append("")
    parts.append(f"![Fig 6 — Kaplan-Meier survival curves by arm. Each "
                 f"step drop marks an extinction event in one of the 20 "
                 f"seeds; runs that reached 15,000 ticks are "
                 f"right-censored at the horizon.]"
                 f"({(figures_rel / 'fig6_survival.png').as_posix()})")
    parts.append("")
    parts.append("The survival curves show the same A/B ≈ floor, "
                 "C below, D/E/F/G clustered above pattern that the "
                 "fitness ladder shows, but as a function of time: the "
                 "evolvable-substrate-with-emissions arms maintain "
                 "near-full survival much longer, and the no-emission "
                 "or fixed-substrate arms drop steadily from early "
                 "ticks. The four upper curves (D/E/F/G) are visually "
                 "indistinguishable through most of the run, "
                 "consistent with the fitness-AUC clustering.")
    parts.append("")

    return parts


def _behavioral_results(behavior_path: Path | None,
                         figures_rel: Path) -> list[str]:
    """Render the new "Behavioral receiver-response" Results subsection.

    Loads behavior_results.json (output of behavior_analysis.py) and
    renders per-arm and pairwise tables for the headline thresholds.
    Falls back to a "(behavioral analysis not yet run)" placeholder if
    the JSON doesn't exist."""
    parts: list[str] = []
    parts.append("### Behavioral receiver-response")
    parts.append("")
    if behavior_path is None or not behavior_path.exists():
        parts.append("*(Behavioral receiver-response analysis not yet run; "
                     "rerun `python -m no_free_signal.experiments.behavior_analysis "
                     "--in results_15k_behavioral` and regenerate this "
                     "report.)*")
        parts.append("")
        return parts

    try:
        bres = json.loads(behavior_path.read_text(encoding="utf-8"))
    except Exception as e:
        parts.append(f"*(Failed to read {behavior_path}: {e!r})*")
        parts.append("")
        return parts

    parts.append("Population-fitness convergence between LLM-shaped "
                 "(D, E, F) and matched-random (G) emission sources "
                 "raises the question reviewers immediately ask: did "
                 "*communication* evolve, or did emissions just perturb "
                 "the ecology in fitness-neutral ways? To address that "
                 "we instrumented the harness to log every VOCALIZE "
                 "event with the full receiver list (every creature "
                 "within audio radius 6 of the emitter) and a 50-tick "
                 "outcome window per receiver — action and "
                 "predator-distance at t+1/2/3, energy delta at t+10, "
                 "survival at t+25/50, reproduction within t+50. With "
                 "these per-event records we can ask, **conditional on "
                 "hearing an emission above some strength threshold, "
                 "does receiver behavior differ from receivers within "
                 "the same audio radius whose attention-weighted "
                 "strength fell below threshold?**")
    parts.append("")
    parts.append("Receivers whose attention-weighted heard strength "
                 "exceeded threshold form the *heard* group; receivers "
                 "in the same emit_event with strength below threshold "
                 "are natural matched controls (within radius 6 of the "
                 "same emitter, so spatial and temporal context match "
                 "by construction). We compute per-seed heard-mean "
                 "minus no-heard-mean for each metric, then bootstrap "
                 "the seed-level differences within each arm. **Effect "
                 "sizes are reported at the seed level — 10,000 "
                 "bootstrap resamples over per-seed differences — to "
                 "avoid the pseudoreplication trap that pooling "
                 "across receivers (often >100k per arm) would "
                 "create.** Receivers where the *emitter* and "
                 "receiver are the same creature (`self_hearing = "
                 "true`) are reported separately because they index a "
                 "different mechanism — self-feedback, not social "
                 "communication.")
    parts.append("")

    # Headline metric: predator_dist_delta_t3 at threshold 0.05 (most
    # samples) — tells the cleanest version of the story.
    parts.append("#### Non-self social receiver-response")
    parts.append("")
    parts.append("The cleanest test of social receiver-response is "
                 "`predator_dist_delta_t3` (change in distance to the "
                 "nearest predator over the 3 ticks following the "
                 "emit event), restricted to receivers other than the "
                 "emitter and to seeds where predators evolved (a "
                 "subset, since predate_drive starts low and only "
                 "some seeds drift it above the 0.3 predator "
                 "threshold). Below: per-arm bootstrap of the "
                 "per-seed (heard − no-heard) effect.")
    parts.append("")
    parts.append("| Threshold | Arm | n_seeds | mean diff (heard − no-heard) | 95% CI | P(>0) |")
    parts.append("|---|---|---:|---:|---|---:|")

    headline_metric = "predator_dist_delta_t3"
    for thr_block in bres:
        thr = thr_block["threshold"]
        for arm in ("A", "B", "C", "D", "E", "F", "G"):
            arm_data = thr_block.get("per_arm", {}).get(arm, {})
            md = arm_data.get("metrics", {}).get(headline_metric)
            if md is None:
                continue
            sig = " **\\***" if (md["ci_lo"] > 0 or md["ci_hi"] < 0) else ""
            parts.append(
                f"| {thr} | **{arm}** | {md['n_seeds']} "
                f"| {md['mean_diff']:+.4f} "
                f"| [{md['ci_lo']:+.4f}, {md['ci_hi']:+.4f}] "
                f"| {md['p_positive']:.0%}{sig} |"
            )
    parts.append("")
    parts.append("\\* indicates 95% CI excludes zero. Positive values "
                 "mean hearing the emission was associated with a "
                 "*greater* increase in predator distance "
                 "(more flee-like). Across thresholds, arm G's effect "
                 "trends positive (slight flee response on hearing) "
                 "while D and F trend slightly negative (slight "
                 "approach or stationary behavior).")
    parts.append("")

    # Pairwise headline at threshold 0.05 -- predator_dist_delta_t3, flee
    parts.append("#### Pairwise comparisons (diff-of-diffs)")
    parts.append("")
    parts.append("Cross-arm bootstrap of the diff-of-diffs: how much "
                 "does arm A's heard-effect differ from arm B's? "
                 "Positive = arm A's *increase* in metric due to "
                 "hearing exceeds arm B's. The decisive test for "
                 "behavioral fungibility between LLM-shaped and "
                 "matched-noise emission sources is **F vs G** "
                 "(same evolvable substrate, both with broken or "
                 "absent semantic content, differing only on "
                 "emission-source identity).")
    parts.append("")
    metric_labels = {
        "predator_dist_delta_t3": "pred_dist Δt3",
        "flee_3tick": "flee t1-3",
        "survived_next_25": "surv 25",
        "energy_delta_next_10": "energy Δ10",
        "food_eaten_next_10": "food 10",
        "reproduced_next_50": "repro 50",
    }
    parts.append("| Metric | Thr | Pair | n | mean Δ | 95% CI | P(A>B) |")
    parts.append("|---|---:|---|---:|---:|---|---:|")

    pairwise_metrics = ["predator_dist_delta_t3", "flee_3tick",
                          "survived_next_25", "energy_delta_next_10"]
    pairs_to_show = [("D", "G"), ("F", "G"), ("D", "B"),
                       ("D", "C"), ("E", "G")]
    for thr_block in bres:
        thr = thr_block["threshold"]
        if thr not in (0.05, 0.10):
            continue
        for m in pairwise_metrics:
            for a, b in pairs_to_show:
                # Find matching pairwise entry
                for p in thr_block.get("pairwise", []):
                    if p["a"] == a and p["b"] == b and p["metric"] == m:
                        sig = " **\\***" if (p["ci_lo"] > 0 or p["ci_hi"] < 0) else ""
                        m_label = metric_labels.get(m, m)
                        parts.append(
                            f"| {m_label} | {thr} | **{a} vs {b}** "
                            f"| {p['n_seeds_paired']} "
                            f"| {p['mean_diff_of_diffs']:+.3f} "
                            f"| [{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}] "
                            f"| {p['p_a_gt_b']:.0%}{sig} |"
                        )
                        break
    parts.append("")
    parts.append("\\* indicates 95% CI excludes zero. The headline "
                 "result is **D vs G and F vs G on "
                 "`predator_dist_delta_t3`** at thresholds 0.05 and "
                 "0.10: receivers exhibit measurably more "
                 "predator-distance increase (more flee-like "
                 "movement) when hearing matched-random emissions "
                 "than when hearing LLM-shaped emissions, even "
                 "though both arms reach similar population fitness. "
                 "The diff-of-diffs CI excludes zero, P(A > B) is "
                 "0–4%, indicating G's heard-effect on flee is "
                 "consistently larger than D's or F's.")
    parts.append("")
    parts.append("Effect sizes are statistically robust but modest "
                 "in magnitude. The mean diff-of-diffs at threshold "
                 "0.05 is roughly −0.19 to −0.26 predator-distance "
                 "units per heard event — a real but small shift in "
                 "movement behavior. We do not interpret these as "
                 "evidence of language-like semantic communication; "
                 "they show only that **non-emitter creatures move "
                 "differently in the seconds after hearing different "
                 "emission shapes**, which is the necessary "
                 "condition for any communication interpretation but "
                 "not sufficient for it.")
    parts.append("")

    # Self-hearing caveat
    parts.append("#### Self-hearing as a separate channel")
    parts.append("")
    parts.append("The largest single behavioral effect in the dataset "
                 "is the emitter's own energy trajectory after "
                 "hearing its own emission, especially in arm G "
                 "(self-hearing energy effect ≈ +1.8 units, P > 0 "
                 "at 100% across 20 seeds at threshold 0.05). This "
                 "is *not* social communication — the emitter's "
                 "internal state caused both the emission and the "
                 "subsequent behavior, so the conditional "
                 "association is a self-feedback/self-conditioning "
                 "channel, not a signal-receiver channel. We "
                 "report it separately and interpret it as evidence "
                 "of an externalized-state mechanism in arm G's "
                 "random emitter — useful as a caveat against "
                 "treating any heard-vs-no-heard comparison as "
                 "automatic evidence of communication.")
    parts.append("")
    parts.append("All non-self pairwise results in the table above "
                 "exclude self-hearing receivers by construction. "
                 "The social receiver-response is therefore "
                 "**modest but isolatable** from the much larger "
                 "self-feedback effects.")
    parts.append("")

    # Caveats / sample size
    parts.append("#### Threshold sensitivity and sample-size caveats")
    parts.append("")
    parts.append("Bootstrap n_seeds varies by threshold and metric "
                 "(see tables): higher thresholds (0.25) drop many "
                 "seeds because few receivers cross the threshold "
                 "in those seeds, and predator-conditioned metrics "
                 "lose seeds where predate_drive never evolved "
                 "above the predator-classification threshold (0.3). "
                 "We treat results at threshold 0.05 (most samples) "
                 "as primary, with 0.10 reported alongside; 0.25 "
                 "is too sparse for stable seed-level "
                 "bootstrapping in most arms. Reproduced-event "
                 "metrics (`reproduced_next_50`) are uniformly zero "
                 "across arms because reproduction events are too "
                 "rare in 50-tick post-emit windows; we report "
                 "this null directly rather than dropping the "
                 "metric.")
    parts.append("")

    return parts


def _discussion(arm_data: dict[str, dict[str, Any]]) -> list[str]:
    return [
        "## Discussion",
        "",
        "This is a **matched-emission-control paper reporting a "
        "negative fitness result for the substrate-evolution-of-"
        "fixed-LLM hypothesis, paired with a behavioral receiver-"
        "response addendum**. Our central two-part contribution: "
        "**matched-noise and semantics-broken controls can reveal "
        "that LLM-agent fitness gains in earlier pilot data were "
        "not in fact load-bearing once the controls are run; and "
        "behavioral receiver-response analysis can reveal that "
        "fitness-equivalent emission sources are not behaviorally "
        "equivalent.** Population-level fitness metrics can hide "
        "both the *absence* of an LLM effect and the presence of "
        "behavioral differences between fitness-equivalent "
        "emission sources.",
        "",
        "At the population-fitness level, the only robust contrast "
        "is **C < everything-else**: populations with no audio "
        "emissions fare worse than populations with any persistent "
        "8-vector emission stream. Beyond that single cadence "
        "effect, the seven arms collapse into two indistinguishable "
        "tiers: A and B at the top (~253k pop AUC), and D, E, F, "
        "G clustered ~8% below (~230k pop AUC). The LLM-vs-control "
        "contrasts P(D > B), P(D > E), P(D > F), and P(D > G) all "
        "fail to cross 95% under the paired-seed bootstrap, with "
        "Cohen's d at or below 0.10. Arm G (no-LLM uniform noise "
        "at matched cadence) is statistically indistinguishable "
        "from arm D (full LLM stack) on population AUC, and arm B "
        "(LLM with frozen substrate) actually slightly outperforms "
        "D — opposite to the hypothesized direction.",
        "",
        "**At the receiver-behavior level, however, emission "
        "sources diverge.** Non-emitter creatures within audio "
        "radius of an emit event move differently in the seconds "
        "after hearing different emission shapes. Random-noise "
        "emissions (G) are associated with a small but "
        "statistically robust increase in predator-distance and "
        "flee-like movement relative to LLM-shaped emissions "
        "(D, F). The cleanest internal control — F vs G, same "
        "evolvable substrate, differing only on emission-source "
        "identity — also crosses the 95% threshold on "
        "predator-distance change (P(F > G) ≈ 0% on the "
        "diff-of-diffs). The effect is modest in absolute terms "
        "(~0.1–0.2 predator-distance units per heard event) and "
        "only manifests in seeds where predate_drive evolved "
        "above the predator-classification threshold, but the "
        "direction is consistent across heard-strength "
        "thresholds and across the D vs G and F vs G "
        "comparisons.",
        "",
        "**Together, fitness-fungible at the population level + "
        "behaviorally distinct at the receiver level** is a "
        "richer finding than either pure fungibility or pure "
        "discrimination. Different emission streams reach "
        "similar survival outcomes through different behavioral "
        "routes — and population AUC alone would have hidden "
        "this distinction. The methodological lesson is that "
        "LLM-on vs LLM-off comparisons should be supplemented "
        "with both matched-noise controls (to disambiguate "
        "model intelligence from emission-channel value) and "
        "receiver-response analysis split by self-hearing vs "
        "non-self (to surface behavioral differences hidden by "
        "fitness convergence and to separate social signaling "
        "from self-feedback).",
        "",
        "We do not interpret the receiver-response divergence "
        "as evidence of language-like semantic communication. "
        "What it shows is the necessary condition for any "
        "communication interpretation — non-emitter receivers "
        "respond differently to different emission shapes — but "
        "not the sufficient condition (that the response is "
        "decoding context-specific meaning rather than reacting "
        "to gross statistical features of the wave). The "
        "magnitude of the social effect is small enough that "
        "the most honest framing is *behavioral "
        "discrimination*, not *communication*.",
        "",
        "### 1. The interface dominates the model-specific contribution at the fitness level (in this regime)",
        "",
        "When a fixed model is wrapped in evolved scaffolding, the "
        "*scaffolding* is what adapts at the population-fitness level. "
        "Arm D populations evolved production bias, perception "
        "attention, and gating that exploit a steady emission "
        "stream — and they reached fitness within bootstrap noise "
        "of populations evolving around uniform noise at the same "
        "cadence (arm G), scrambled LLM emissions (E), or "
        "replay-randomized LLM emissions (F). All four reach the "
        "same fitness asymptote. **At the population-fitness level, "
        "the interface dominates and the model upstream is not "
        "measurably distinguishable from a matched-cadence noise "
        "generator** under our metric and substrate complexity.",
        "",
        "The behavioral receiver-response analysis qualifies this "
        "claim: at the *behavioral* level, non-emitter creatures "
        "do distinguish between emission shapes, with random-noise "
        "emissions producing slightly more flee-like movement than "
        "LLM-shaped emissions. So the upstream signal source is "
        "not measurably distinguishable from noise *via population "
        "AUC*, but it is measurably distinguishable *via "
        "receiver-conditional movement*. The two findings are "
        "compatible: different emission streams can reach similar "
        "fitness outcomes through different behavioral routes. "
        "The fitness-only interpretation undersells the "
        "discrimination the substrate has actually achieved.",
        "",
        "### 2. Emission presence is suggestive; emission source is not fitness-discriminating",
        "",
        "Among the four evolvable emission-bearing arms (D, E, F, G), "
        "fitness is statistically indistinguishable: D, F, G within "
        "~0.1% pop AUC of each other and E within ~2%. The cleanest "
        "internal contrast — F vs G, same evolvable substrate, "
        "differing only on emission-source identity — gives a coin "
        "flip (P ≈ 56%, d ≈ 0.00). **Among emission-bearing arms, "
        "LLM-shaped, scrambled, replayed, and uniform-random "
        "emissions are fitness-equivalent.** The substrate-only "
        "no-emitter arm C is descriptively lowest, but the C-vs-"
        "emission-arm contrasts (D > C, G > C) sit at P ≈ 71% with "
        "CIs that include zero — suggestive of an emission-presence "
        "effect, but not statistically load-bearing at n=20.",
        "",
        "Caveat on cadence matching: arm G's per-creature-per-tick "
        "emission rate is ~22% higher than arm D's (0.069 vs "
        "0.056), because the random emitter has zero failed-call "
        "rate where LLM-driven controllers lose ~10–30% of refresh "
        "cycles to API latency and malformed JSON. G should "
        "therefore be read as an *approximately* cadence-matched "
        "noise control rather than a perfectly matched one. The "
        "F vs G comparison is cleaner on this axis: both share an "
        "LLM-controlled refresh path, so cadence differences "
        "between F and G are smaller.",
        "",
        "But emission *shape* is not behaviorally fungible. "
        "Receivers respond differently to LLM-shaped vs random "
        "emissions, even when fitness outcomes converge among the "
        "emission-bearing arms. The substrate doesn't *only* "
        "exploit temporal regularity — it does discriminate "
        "between emission shapes — it just doesn't discriminate "
        "enough at the fitness level for the difference to show "
        "up in population AUC.",
        "",
        "### 3. The substrate-evolution hypothesis is not supported",
        "",
        "Two arms anchor this. **Arm B** has the full LLM in the loop "
        "but perception/production frozen at random initialization: "
        "B sits at the *top* of the fitness ladder, tied with the "
        "mute arm A. **Arm D** has the same LLM with an evolvable "
        "substrate: D sits ~8% below B with P(D > B) ≈ 36% — the "
        "evolvable-substrate condition does *worse* than the "
        "frozen-substrate condition, contrary to the central "
        "hypothesis. **Arm G** has no LLM at all but an evolvable "
        "substrate hearing uniform random noise: it sits in the "
        "same cluster as D (P(D > G) ≈ 63%, d ≈ 0.00) and ~8% "
        "below B. The substrate-evolution hypothesis predicted "
        "D >> B and D >> G; neither contrast holds. Several "
        "readings are compatible with this: (a) the substrate "
        "genome (8-bin attention + 8-bin production bias + reflex-"
        "fear gate) is too narrow to extract differential signal "
        "from any of the emission streams we tested; (b) at this "
        "decoder complexity, evolutionary timescale (15k ticks), "
        "and population size (25 creatures), substrate evolution "
        "has not had time to produce the gain the hypothesis "
        "predicts; (c) the fitness landscape in this environment "
        "is dominated by physics (predation, foraging) and the "
        "communication channel — whatever is on it — is a small "
        "perturbation. We cannot distinguish these from the data "
        "we have.",
        "",
        "### 4. Generalization to stronger models is untested",
        "",
        "Our experiment used Amazon Nova Lite — a small, cheap "
        "frontier model. We make no claim about whether the "
        "scrambled-LLM ≈ LLM result holds for stronger models. It is "
        "plausible that richer context-sensitivity in a larger model "
        "would widen D's lead over E/F, or — alternatively — that "
        "the substrate evolution timescale is too short for any LLM "
        "to demonstrate its semantic value. Our results motivate "
        "running the same six-arm design with a stronger model "
        "(Claude Haiku 4.5 or larger) at comparable seed budget.",
        "",
        "### 5. Evaluation should test whether semantic content matters",
        "",
        "If a paper shows *LLM-augmented agents do better than mute "
        "agents*, our result demands at least four follow-up "
        "questions before attributing the gain to the LLM's "
        "intelligence: (a) does it hold without context-sensitivity "
        "(scrambled-LLM control)? (b) without current-context "
        "coupling (replay-randomized control)? (c) against "
        "frequency- and amplitude-matched random noise with no LLM "
        "(random-emitter control)? (d) without an evolvable "
        "interpretive layer (fixed-substrate control)? The "
        "persistent-emission-channel effect and the evolvable-"
        "substrate effect are independent confounds that any "
        "LLM-on/off comparison should rule out. In our environment, "
        "(c) was the decisive control: a no-LLM noise generator "
        "matched the LLM's fitness contribution exactly.",
        "",
        "### Mechanism evidence from coherent ordering",
        "",
        "Only one ordering feature is robust under the paired-seed "
        "bootstrap: **C < everything-else**. The C-vs-others "
        "contrasts are positive in direction (P ≈ 65–75%, d ≈ 0.11) "
        "but only marginally cross conventional significance "
        "thresholds at n=20 seeds. None of the LLM-vs-control "
        "pairwise comparisons (D > B, D > E, D > F, D > G) cross "
        "95% under the paired-seed design; all sit at small "
        "Cohen's d (≤ 0.10) with CIs straddling zero. We do not "
        "claim a positive substrate-evolution effect from these "
        "data. The *direction of the entire ordering* — A ≈ B "
        "(top), D ≈ E ≈ F ≈ G clustered ~8% below, C clearly worst "
        "— is internally consistent with a single-mechanism story "
        "(any persistent emission stream helps fitness, source "
        "identity does not matter, no emission stream hurts), and "
        "an evolvable substrate plus content-bearing emissions "
        "(D) does not measurably exceed an evolvable substrate "
        "plus pure noise (G) or a frozen substrate plus LLM (B). "
        "The hypothesized D-wins ordering does not appear; the "
        "data support a more modest cadence-only effect rather "
        "than a substrate-evolution effect.",
        "",
    ]


def _threats_to_validity(arm_data: dict[str, dict[str, Any]]) -> list[str]:
    return [
        "## Threats to validity",
        "",
        "**Statistical power.** Under the paired-seed bootstrap, "
        "*no* LLM-vs-control contrast crosses the conventional "
        "95% threshold in the predicted direction. P(D > B) ≈ "
        "36% (negative direction), P(D > C) ≈ 71%, P(D > E) ≈ "
        "55%, P(D > F) ≈ 52%, P(D > G) ≈ 63%. All sit at small "
        "effect sizes (Cohen's d ≤ 0.10) with CIs straddling "
        "zero. These results are consistent with a small or "
        "zero underlying LLM effect at n=20 seeds, and are also "
        "consistent with a true zero effect — we cannot "
        "discriminate. Doubling n to 40 seeds per arm (~$80 of "
        "additional compute) would tighten the CIs but is "
        "unlikely to flip the direction of any comparison given "
        "the magnitude of the current point estimates.",
        "",
        "**Single environment, single model.** All findings are from "
        "a 24×24 grid with one resource density, one predator regime, "
        "and one LLM (Amazon Nova Lite). Generalization to other "
        "environments and stronger models is untested. Particularly: "
        "scrambled-LLM may be a much weaker control at higher model "
        "scale, where context-sensitive emissions could carry more "
        "structure that bin-permutation would destroy.",
        "",
        "**Receiver-response is behaviorally verified at the "
        "discrimination level, not the semantic level.** The "
        "behavioral analysis demonstrates that non-emitter "
        "receivers respond differently to different emission "
        "shapes — necessary condition for any communication "
        "interpretation — but the magnitude of the social "
        "effect (~0.1–0.2 predator-distance units per heard "
        "event) is small, and the analysis is restricted to "
        "the subset of seeds where predators evolved (n = 16–19 "
        "of 20 per arm depending on threshold). We do not claim "
        "the behavioral signal is decoded as context-specific "
        "*meaning* — only that receivers' movement is "
        "conditional on emission shape. Stronger semantic "
        "claims would require pairing emission patterns with "
        "specific environmental contingencies (e.g., "
        "danger-shaped vs food-shaped emissions) and showing "
        "context-appropriate differential response.",
        "",
        "**Alignment metric is inconclusive, not disconfirmatory.** "
        "Population-mean cosine cannot distinguish substrate "
        "alignment with the LLM's specific bin pattern from generic "
        "overlap of two non-zero positive 8-vectors. The near-zero "
        "z-scores are *consistent with* the fungible-emission "
        "interpretation, but a finer-grained per-creature or "
        "per-bin metric, or a behaviorally-grounded receiver-"
        "response measure, is needed to confirm.",
        "",
        "**Confounds in E, F, and G.** "
        "Arm E gives each creature a *fixed* per-individual bin "
        "permutation; the speaker hears its own post-permutation "
        "wave consistently across emissions, which may give arm E "
        "a stable self-feedback signal even on scrambled emissions. "
        "Arm F samples from a global, per-run replay buffer of the "
        "most recent 200 post-transform LLM emissions — emissions "
        "therefore retain the *distribution* of what an LLM "
        "produces in the environment, just not the current-tick "
        "context. Arm G has *two* differences from D: no LLM-driven "
        "action choice *and* random emission shape, so D vs G "
        "bounds the LLM's *total* contribution (action + "
        "emissions), not emission-shape alone. Together, however, "
        "these three controls progressively strip more of the LLM's "
        "structure (E removes bin specificity, F removes context "
        "coupling, G removes everything) and all three reach the "
        "same fitness — a coherence that is hard to attribute to "
        "the residual structure each individually preserves.",
        "",
        "**Self-hearing.** All arms allow speakers to hear their own "
        "emissions on the next tick. This gives speakers a stable "
        "feedback channel that arms B/D/E/F/G all share, but it may "
        "interact with the substrate-evolution dynamics in ways "
        "that affect arm comparisons.",
        "",
        "**Brain controller.** The trained policy network ('brain') "
        "is identical and frozen across all seven arms. We do not "
        "claim our results say anything about how trained policies "
        "interact with substrate evolution; the brain is held "
        "constant precisely to isolate the substrate-LLM "
        "interaction.",
        "",
    ]


def _future_work() -> list[str]:
    return [
        "## Planned next experiments",
        "",
        "**Context-conditional semantic test.** The current "
        "behavioral analysis verifies receiver-response "
        "discrimination — non-emitter receivers move differently "
        "after hearing different emission shapes — but does not "
        "test for *context-specific* signal interpretation. A "
        "stronger semantic test would pair emission patterns with "
        "controlled environmental contingencies (predator-near vs "
        "food-near emit events, scripted broadcasts of fixed "
        "shapes) and ask whether receivers respond differently "
        "to the *same* emission shape across contexts, and to "
        "*different* shapes within the same context. This would "
        "discriminate \"the substrate reacts to the gross "
        "statistics of the wave\" from \"the substrate decodes "
        "context-appropriate meaning\" — a distinction the "
        "current analysis cannot make.",
        "",
        "**Stronger model replication.** Re-run arms B/D/E/F with "
        "Claude Haiku 4.5 (~18× cost of Nova Lite) at the same n=20 "
        "seed budget to test whether stronger LLM context-"
        "sensitivity widens D's lead over the scrambled and "
        "randomized controls. Estimated cost ~\\$700-900 — outside "
        "the budget of this draft but achievable in a follow-up.",
        "",
        "**Longer evolutionary horizon.** 15,000 ticks ≈ 50 "
        "generations. Doubling to 30,000 ticks would test whether "
        "substrate evolution continues to differentiate D from "
        "E/F, or whether the population reaches a steady state "
        "where the contributions of the LLM and the substrate "
        "stabilize.",
        "",
    ]


def _limitations(arm_data: dict[str, dict[str, Any]]) -> list[str]:
    """Concise limitations summary; full discussion lives in the "
    Threats to Validity section."""
    return [
        "## Limitations",
        "",
        "Summarized here; see *Threats to Validity* for full "
        "discussion of each.",
        "",
        "- *No* LLM-vs-control pairwise comparison crosses the "
        "95% threshold in the predicted direction. P(D > B) ≈ 36% "
        "(negative). P(D > E), P(D > F), P(D > G) all sit at or "
        "below ~63% with Cohen's d ≤ 0.10. The narrative rests "
        "on the coherent *negative* ordering (no LLM advantage "
        "over matched controls) rather than any single positive "
        "contrast.",
        "- One environment configuration, one LLM (Nova Lite).",
        "- Receiver-response is behaviorally verified at the "
        "discrimination level (non-emitter movement differs by "
        "emission shape), but the social effect is modest in "
        "magnitude and we do not claim semantic decoding.",
        "- The alignment cosine metric cannot distinguish bin-"
        "specific alignment from generic positive-vector overlap.",
        "- D vs G bounds the LLM's total contribution (action + "
        "emissions), not emission-shape alone, since arm G replaces "
        "both.",
        "- Self-hearing in arm E may pull it closer to D than "
        "expected.",
        "",
    ]


def _ai_disclosure() -> list[str]:
    """Generative-AI-use disclosure aligned with 2026 best practice
    (NeurIPS / ICLR / Nature / Science / arXiv guidelines).

    Two roles must be disclosed separately because they are different:
    (1) the LLM under study (Amazon Nova Lite — a subject of the
    experiment); (2) any LLM used to author or refine this manuscript
    or its codebase. Author retains full responsibility for the content,
    methodology, statistical claims, and conclusions."""
    return [
        "## Generative-AI use disclosure",
        "",
        "**LLM as study subject.** The experiment uses Amazon Nova "
        "Lite (`us.amazon.nova-lite-v1:0`, accessed via Amazon "
        "Bedrock) as a frozen action-selection and emission-shaping "
        "controller in arms B, D, E, and F. Per-call cost, total "
        "calls, and total cost are reported in Methods. The model is "
        "called via API; weights are not modified, fine-tuned, or "
        "distilled. The system prompt template is reproduced "
        "verbatim in the Reproducibility appendix.",
        "",
        "**LLM as authoring assistant.** The author used Anthropic "
        "Claude (Claude Code, Claude Opus / Sonnet variants) "
        "during code development and manuscript drafting. Specific "
        "uses: (a) writing and refactoring Python code in the "
        "`no_free_signal/` and `foresight/` packages, including the "
        "experiment harness, behavioral logger, statistical "
        "bootstrap, and figure-generation code; (b) drafting and "
        "iterating on prose for this report — the report-generation "
        "module (`no_free_signal/experiments/report.py`) is itself a "
        "templated narrative authored with model assistance; (c) "
        "interactive debugging of AWS deployment, cadence-protection "
        "logic, and instrumentation-bias diagnosis. The author "
        "reviewed every diff, ran every experiment, interpreted "
        "every statistical output, and wrote the final scientific "
        "claims. **The model did not access any data, run any "
        "analyses, or generate any results unsupervised.** All "
        "numerical claims trace to either (i) the deterministic "
        "experimental harness or (ii) the bootstrap statistics "
        "module, both of which are open for inspection in this "
        "repository.",
        "",
        "**Reproducibility & verifiability.** All code, prompts, "
        "seeds, raw run data, and bootstrap configurations are "
        "available alongside this manuscript. Any reader can "
        "re-derive the figures and statistical claims from the "
        "JSONL run files using the `no_free_signal.experiments.report` "
        "and `no_free_signal.experiments.stats` modules. Conversation logs "
        "between author and authoring AI are not part of the "
        "scientific record and are not preserved.",
        "",
        "**Author responsibility statement.** The author takes full "
        "responsibility for the methodology, data integrity, "
        "statistical inferences, scientific conclusions, and any "
        "errors in this manuscript. Use of generative AI for "
        "writing assistance and code authoring does not transfer "
        "any aspect of authorship to the model or its provider.",
        "",
    ]


def _reproducibility(first_header: dict[str, Any] | None) -> list[str]:
    """Reproducibility appendix: prompt template, JSON schema, run config,
    seed list, and the RandomEmitterController spec. The prompt template
    is lifted from `no_free_signal.llm_controller.SYSTEM_TEMPLATE` at import time
    so the appendix never goes stale relative to the actual code."""
    try:
        from no_free_signal.llm_controller import SYSTEM_TEMPLATE
        prompt_text = SYSTEM_TEMPLATE
    except Exception:
        prompt_text = "(SYSTEM_TEMPLATE not importable — see no_free_signal/llm_controller.py)"

    n_steps = first_header.get("n_steps", 15000) if first_header else 15000
    snapshot_every = first_header.get("snapshot_every", 50) if first_header else 50
    n_creatures = first_header.get("n_creatures", 25) if first_header else 25
    grid_size = first_header.get("grid_size", 24) if first_header else 24
    refresh_every = first_header.get("refresh_every", 15) if first_header else 15
    model_id = first_header.get("model_id", "us.amazon.nova-lite-v1:0") if first_header else "us.amazon.nova-lite-v1:0"

    lines: list[str] = []
    lines.append("## Reproducibility appendix")
    lines.append("")
    lines.append("This section documents everything required to "
                 "reproduce the run. Anything not listed here is at "
                 "default values in the harness.")
    lines.append("")
    lines.append("### Run configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Model | `{model_id}` |")
    lines.append(f"| Episode length | {n_steps:,} ticks |")
    lines.append(f"| Snapshot interval | every {snapshot_every} ticks |")
    lines.append(f"| Initial population | {n_creatures} creatures |")
    lines.append(f"| Grid | {grid_size}×{grid_size} |")
    lines.append(f"| LLM refresh cadence | every {refresh_every} ticks per controller |")
    lines.append("| Seeds (per arm) | 0, 1, 2, …, 19 (n=20) |")
    lines.append("| Bootstrap resamples | 10,000 |")
    lines.append("| Bootstrap RNG seed | 42 |")
    lines.append("| Null-shuffle iterations (alignment z) | 100 per run |")
    lines.append("| Primary pairwise test | paired-seed bootstrap (seeds matched across arms) |")
    lines.append("")
    lines.append("### LLM system prompt template")
    lines.append("")
    lines.append("Stored verbatim in `no_free_signal/llm_controller.py:SYSTEM_TEMPLATE`. "
                 "Per-creature `{bio}` and `{personality}` substitutions "
                 "are derived from the genome at controller construction. "
                 "The full template is reproduced below.")
    lines.append("")
    lines.append("```text")
    for line in prompt_text.splitlines():
        lines.append(line)
    lines.append("```")
    lines.append("")
    lines.append("### LLM JSON output schema")
    lines.append("")
    lines.append("Strict JSON. Malformed responses default to "
                 "`action=NOOP`, `confidence=0.0`, no vocalization.")
    lines.append("")
    lines.append("```json")
    lines.append("{")
    lines.append('  "thought": "<one sentence on why>",')
    lines.append('  "action": "<NORTH|SOUTH|EAST|WEST|EAT|REST|GATHER|BUILD|VOCALIZE|NOOP>",')
    lines.append('  "confidence": <number in 0..1>,')
    lines.append('  "say": "<utterance, ≤80 chars; empty string allowed>",')
    lines.append('  "vocalize_wave": [<8 numbers in 0..1>],')
    lines.append('  "vocalize_amp": <number in 0..1>')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("### Arm G — RandomEmitterController spec")
    lines.append("")
    lines.append("Implemented in `no_free_signal/experiments/ablations.py` as a "
                 "subclass of `LLMController` so the existing "
                 "`isinstance` checks in `world.py` continue to "
                 "treat it as the LLM-driven controller for "
                 "action-selection priority. Overrides "
                 "`_do_refresh` / `_maybe_kick_refresh` to bypass "
                 "Bedrock entirely.")
    lines.append("")
    lines.append("- **Wave:** `[rng.uniform(0.0, 1.0) for _ in range(8)]` — uniform on [0,1]⁸ each refresh.")
    lines.append("- **Amplitude:** `max(0.1, min(0.9, rng.gauss(0.5, 0.15)))` — Gaussian, μ=0.5, σ=0.15, clipped.")
    lines.append("- **Action:** `rng.randrange(10)` — uniform over the 10 discrete actions.")
    lines.append("- **Refresh cadence:** every 15 ticks per controller (matched to D/E/F).")
    lines.append("- **Per-creature RNG seed:** `seed XOR creature_id` so newborns spawned mid-run get distinct streams.")
    lines.append("- **Cost:** \\$0 (no API calls).")
    lines.append("")
    lines.append("### Replication command")
    lines.append("")
    lines.append("Two grids must be run: a logger-OFF fitness grid "
                 "(canonical for fitness claims) and a logger-ON "
                 "behavioral grid (for receiver-response analysis "
                 "only). They share code, parameters, and seeds; "
                 "the only difference is the `--log-behavioral` flag "
                 "and the output directory. See `notes/instrumentation_"
                 "bias.md` for why the two datasets are kept separate.")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. Fitness grid (logger off — canonical)")
    lines.append("python -m no_free_signal.experiments.parallel --confirm \\")
    lines.append("  --arms A,B,C,D,E,F,G \\")
    lines.append("  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \\")
    lines.append("  --workers 4 --n-steps 15000 --snapshot-every 50 \\")
    lines.append("  --n-creatures 25 --grid-size 24 --refresh-every 15 \\")
    lines.append("  --model nova-lite --out-dir results_15k_fitness \\")
    lines.append("  --timeout 0 --per-run-cap 50000")
    lines.append("")
    lines.append("# 2. Behavioral grid (logger on — for receiver-response only)")
    lines.append("python -m no_free_signal.experiments.parallel --confirm \\")
    lines.append("  --arms A,B,C,D,E,F,G \\")
    lines.append("  --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \\")
    lines.append("  --workers 4 --n-steps 15000 --snapshot-every 50 \\")
    lines.append("  --n-creatures 25 --grid-size 24 --refresh-every 15 \\")
    lines.append("  --model nova-lite --out-dir results_15k_behavioral \\")
    lines.append("  --timeout 0 --per-run-cap 50000 \\")
    lines.append("  --log-behavioral")
    lines.append("```")
    lines.append("")
    lines.append("Recommended environment for AWS / Linux running on "
                 "fast hardware: `LAMDIS_TICK_RATE=10 "
                 "LAMDIS_BEDROCK_RPS=99` to match the cadence the "
                 "original report was tuned on. Then validate and "
                 "regenerate the report:")
    lines.append("")
    lines.append("```bash")
    lines.append("# 3. Audit gate (must exit 0 before report generation)")
    lines.append("python validate_canonical.py results_15k_fitness")
    lines.append("")
    lines.append("# 4. Behavioral receiver-response analysis")
    lines.append("python -m no_free_signal.experiments.behavior_analysis \\")
    lines.append("  --in results_15k_behavioral \\")
    lines.append("  --out-json results_15k_behavioral/behavior_results.json")
    lines.append("")
    lines.append("# 5. Two-source report assembly")
    lines.append("python -m no_free_signal.experiments.report \\")
    lines.append("  --in results_15k_fitness \\")
    lines.append("  --behavior-in results_15k_behavioral \\")
    lines.append("  --out results_15k_fitness/REPORT.md")
    lines.append("```")
    lines.append("")
    lines.append("### Code repository")
    lines.append("")
    lines.append("Public source: <https://github.com/gdf-ai/no-free-"
                 "signal>. Clone, `uv sync` (or `pip install -e .`), "
                 "and re-run the commands above against either the "
                 "shipped data (`results_15k_fitness/*.jsonl`) or a "
                 "fresh re-run. Source layout: experiment harness is "
                 "`no_free_signal.experiments.harness` (single run) "
                 "and `no_free_signal.experiments.parallel` (grid "
                 "orchestrator). The world simulator is "
                 "`no_free_signal.world` and `foresight.envs.unified_"
                 "world`. Statistical helpers are in "
                 "`no_free_signal.experiments.stats`; figure code is "
                 "in `no_free_signal.experiments.plot`. Pinned "
                 "dependency versions live in `pyproject.toml` and "
                 "`uv.lock`.")
    lines.append("")
    return lines


def _per_run_table(arm_data: dict[str, dict[str, Any]],
                    runs_by_arm: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    lines.append("## Per-run details")
    lines.append("")
    lines.append("| Arm | Seed | Ticks | Wall sec | API calls | Valid emits | LLM cost USD |")
    lines.append("|---|---|---|---|---|---|---|")
    for arm in _stats.ARM_ORDER:
        runs = runs_by_arm.get(arm, [])
        for r in sorted(runs, key=lambda x: int(x["header"].get("seed", 0)) if x.get("header") else 0):
            s = r.get("summary") or {}
            seed = r["header"].get("seed", "?") if r.get("header") else "?"
            lines.append(
                f"| {arm} | {seed} "
                f"| {s.get('ticks_completed', '?')} "
                f"| {s.get('wall_seconds', '?')} "
                f"| {s.get('llm_api_calls', '?')} "
                f"| {s.get('llm_emits_with_wave', '?')} "
                f"| {s.get('llm_cost_usd', 0.0):.4f} |"
            )
    lines.append("")
    return lines


def write_report(
    runs_dir: Path,
    out_path: Path,
    figures_dir: Path | None = None,
    title: str = DEFAULT_TITLE,
    author: str = DEFAULT_AUTHOR,
    behavior_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate REPORT.md, .docx, .pdf from runs_dir (fitness data).
    behavior_dir defaults to results_15k_behavioral/ relative to cwd
    if not given; that's where behavior_analysis.py writes its JSON.
    Two-source design: fitness from runs_dir, behavioral from
    behavior_dir, because the AWS replication of the behavioral
    instrumentation hits Bedrock RPM throttling that corrupts
    fitness ordering -- only the per-event behavioral records
    survive throttling cleanly.
    """
    if figures_dir is None:
        figures_dir = out_path.parent / "figures"

    payload = _plot.render_all(runs_dir, figures_dir)
    runs_by_arm = payload["runs_by_arm"]
    arm_data = payload["arm_data"]
    alignment_stats = payload["alignment_stats"]
    emission_rates = _stats.compute_emission_rates(arm_data)

    # Aggregate run params from the first header we see.
    first_header = None
    for runs in runs_by_arm.values():
        for r in runs:
            if r.get("header"):
                first_header = r["header"]
                break
        if first_header:
            break

    total_calls = 0
    total_usd = 0.0
    for runs in runs_by_arm.values():
        for r in runs:
            s = r.get("summary") or {}
            total_calls += int(s.get("llm_api_calls", 0))
            total_usd += float(s.get("llm_cost_usd", 0.0))

    figures_rel = (
        figures_dir.relative_to(out_path.parent)
        if figures_dir.is_relative_to(out_path.parent)
        else figures_dir
    )
    today = date.today().isoformat()

    parts: list[str] = []
    parts.extend(_frontmatter(title, author, today))
    parts.append(f"# {title}")
    parts.append("")
    parts.extend(_abstract())
    parts.extend(_introduction())
    parts.extend(_methods(first_header, arm_data, total_calls, total_usd,
                          emission_rates=emission_rates,
                          figures_rel=figures_rel))
    parts.extend(_results(arm_data, alignment_stats, figures_rel))
    # Behavioral receiver-response section. Loads the JSON output of
    # behavior_analysis.py if it exists; otherwise renders a placeholder
    # pointing at how to generate it. Kept separate from the fitness
    # results so the section can be regenerated without re-running the
    # whole report build (and so the fitness story still reads cleanly
    # if behavioral data is not yet in hand).
    if behavior_dir is None:
        behavior_dir = Path("results_15k_behavioral")
    behavior_path = behavior_dir / "behavior_results.json"
    parts.extend(_behavioral_results(behavior_path, figures_rel))
    parts.extend(_discussion(arm_data))
    parts.extend(_threats_to_validity(arm_data))
    parts.extend(_future_work())
    parts.extend(_limitations(arm_data))
    parts.extend(_per_run_table(arm_data, runs_by_arm))
    parts.extend(_ai_disclosure())
    parts.extend(_reproducibility(first_header))

    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"[report] wrote {out_path}")
    return {
        "report_path": out_path,
        "figures_dir": figures_dir,
        "arm_data": arm_data,
        "alignment_stats": alignment_stats,
    }


# ---------------------------------------------------------------------------
# Pandoc compilation (DOCX always, PDF best-effort)
# ---------------------------------------------------------------------------

def compile_outputs(report_path: Path) -> dict[str, str]:
    """Produce DOCX and PDF from REPORT.md via pandoc. Returns a dict
    mapping format → status string ('ok', 'no-pandoc', 'no-latex',
    'failed: ...')."""
    out_dir = report_path.parent
    docx_path = report_path.with_suffix(".docx")
    pdf_path = report_path.with_suffix(".pdf")
    status: dict[str, str] = {}

    if not shutil.which("pandoc"):
        msg = ("pandoc not found on PATH. Install from https://pandoc.org "
               "or via 'choco install pandoc' / 'winget install pandoc'.")
        print(f"[report] {msg}")
        status["docx"] = "no-pandoc"
        status["pdf"] = "no-pandoc"
        return status

    # DOCX — always works with pandoc alone.
    # cwd is the report's directory so relative figure paths in the
    # markdown resolve correctly.
    docx_proc = subprocess.run(
        ["pandoc", report_path.name, "-o", docx_path.name],
        capture_output=True, text=True, cwd=str(out_dir),
    )
    if docx_proc.returncode == 0:
        print(f"[report] DOCX: {docx_path}")
        status["docx"] = "ok"
    else:
        msg = docx_proc.stderr.strip()[:300] or "unknown error"
        print(f"[report] DOCX compile failed: {msg}")
        status["docx"] = f"failed: {msg}"

    # PDF — needs a LaTeX engine. The report contains Unicode characters
    # (≈, ×, Δ, …) that pdflatex doesn't handle by default, so prefer
    # xelatex when available. Try xelatex → lualatex → pdflatex in order;
    # accept the first that succeeds.
    pdf_engines = ["xelatex", "lualatex", "pdflatex"]
    pdf_ok = False
    last_err = ""
    last_engine = ""
    for engine in pdf_engines:
        if not shutil.which(engine):
            continue
        last_engine = engine
        pdf_proc = subprocess.run(
            ["pandoc", report_path.name, "-o", pdf_path.name,
             f"--pdf-engine={engine}"],
            capture_output=True, text=True, cwd=str(out_dir),
        )
        if pdf_proc.returncode == 0:
            print(f"[report] PDF:  {pdf_path}  (engine: {engine})")
            status["pdf"] = "ok"
            pdf_ok = True
            break
        last_err = pdf_proc.stderr.strip()

    if not pdf_ok:
        if not last_engine:
            print(f"[report] PDF compile skipped: no LaTeX engine on PATH.")
            print(f"[report]   to enable PDF, install MiKTeX (https://miktex.org)")
            print(f"[report]   or run: pandoc {report_path.name} --pdf-engine=wkhtmltopdf -o {pdf_path.name}")
            status["pdf"] = "no-latex"
        else:
            snippet = last_err[:400] or "unknown error"
            print(f"[report] PDF compile failed (last engine tried: {last_engine})")
            print(f"[report]   stderr: {snippet}")
            status["pdf"] = f"failed: {snippet}"

    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="no_free_signal.experiments.report")
    p.add_argument("--in", dest="in_dir", default="results")
    p.add_argument("--behavior-in", dest="behavior_dir",
                   default="results_15k_behavioral",
                   help="directory containing behavior_results.json "
                        "(default: results_15k_behavioral). Two-source "
                        "design: fitness from --in, behavioral from "
                        "--behavior-in, because the behavioral rerun on "
                        "AWS hits Bedrock RPM throttling that distorts "
                        "fitness numbers but leaves per-event behavioral "
                        "records valid.")
    p.add_argument("--out", dest="out_path", default=None,
                   help="output path for REPORT.md (default: <in>/REPORT.md)")
    p.add_argument("--title", default=None)
    p.add_argument("--author", default=DEFAULT_AUTHOR)
    p.add_argument("--no-compile", action="store_true",
                   help="skip pandoc DOCX/PDF compilation")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    in_dir = Path(args.in_dir)
    behavior_dir = Path(args.behavior_dir)
    out_path = Path(args.out_path) if args.out_path else in_dir / "REPORT.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    title = args.title or DEFAULT_TITLE
    write_report(in_dir, out_path, title=title, author=args.author,
                 behavior_dir=behavior_dir)
    if not args.no_compile:
        compile_outputs(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
