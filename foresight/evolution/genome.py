"""Unified Genome — one species, traits drive niche emergence.

Replaces the old prey-Genome / PredatorGenome split. A single creature has
genome traits that include drives, defences, reproduction, locomotion,
sensorimotor, reward weights, and cognitive abilities. Behaviour (predator-
like vs forager-like vs cannibal vs omnivore) emerges from the values, not
from a hardcoded type.

References:
- Drives: Panksepp's primary affective systems (Affective Neuroscience, 2004)
- Reward centres: Maslow + Damasio somatic markers
- Life history: Stearns 1992
- Defences: anti-predator adaptations across taxa
- Cognition: predictive-coding + active inference (Friston 2010)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# Order matters for to_array / from_array / mean-genome reporting.
TRAIT_NAMES: tuple[str, ...] = (
    # ----- Affective drives (Panksepp) -----
    "fear_baseline",
    "seeking_baseline",
    "play_baseline",
    "rage_baseline",
    # ----- Sensory-driven REWARD CENTERS (Phase 9 v0.3) -----
    # How rewarding each kind of outcome is to this individual. Different
    # individuals literally weight their senses/motivations differently.
    "reward_safety",
    "reward_reproduction",
    "reward_comfort",
    "reward_hunger",
    "reward_happiness",
    # ----- Diet / niche emergence -----
    # Replaces hardcoded predator/prey: high predate_drive + low forage_drive
    # behaves like a predator; the inverse like a herbivore; both high =
    # omnivore; predate_drive > creature_aversion targeting same species =
    # cannibal.
    "predate_drive",
    "forage_drive",
    # ----- Aversions (Damasio somatic markers) -----
    "hunger_aversion",
    "creature_aversion",
    # ----- Defences -----
    "armor",
    "camouflage",
    "attack_strength",
    "attack_chance",
    # ----- Sensorimotor (was predator-only, now universal) -----
    "vision_range",
    "metabolism",
    "chase_speed",
    "aggression",
    # ----- Life history / reproduction -----
    "reproduction_threshold",
    "offspring_count",
    "lifespan",
    "reproductive_cost",
    # ----- Cognitive / construction -----
    "puzzle_solver_strength",
    # ----- Discrimination / feeding -----
    # Innate accuracy at distinguishing food from non-food matter (wood/stone)
    # before attempting to eat. 0 = always mistakes matter for food, 1 = never.
    # Brains additionally learn discrimination through poisoning experience.
    "food_discrimination",
    # ----- Vocal / audio (Phase 9.O) -----
    # Loudness scalar applied on top of the per-bin freq bias. Loud calls
    # propagate further but cost more energy.
    "vocal_amplitude",
    # 8-bin frequency bias — the wave shape this individual emits when it
    # vocalizes. Co-evolves with conspecifics' audio_att_* to converge on a
    # shared dialect under selection pressure.
    "vocal_freq_0", "vocal_freq_1", "vocal_freq_2", "vocal_freq_3",
    "vocal_freq_4", "vocal_freq_5", "vocal_freq_6", "vocal_freq_7",
    # Probability per tick of an auto-vocalize when fear is high (cry-of-
    # fright reflex; no learning required).
    "vocal_reflex_fear",
    # Per-bin gain on heard waves before they enter the brain's observation.
    # Co-evolves with vocal_freq_* so that a population can develop selective
    # attention to the bins that carry meaningful signal.
    "audio_att_0", "audio_att_1", "audio_att_2", "audio_att_3",
    "audio_att_4", "audio_att_5", "audio_att_6", "audio_att_7",
)


# Module-level mutation freeze set, consulted by ``mutate``. Used by the
# experiment harness's "fixed substrate" ablation to hold specific trait
# indices constant across reproduction. Empty by default — normal evolution
# proceeds unchanged.
_FROZEN_TRAIT_INDICES: set[int] = set()


def freeze_substrate_traits() -> None:
    """Freeze ``vocal_freq_0..7`` and ``audio_att_0..7`` against mutation —
    the production/perception substrate stops evolving across reproduction.
    Used by the experiment harness arm B (``apply_fixed_substrate``)."""
    _FROZEN_TRAIT_INDICES.update(range(28, 36))   # vocal_freq_0..7
    _FROZEN_TRAIT_INDICES.update(range(37, 45))   # audio_att_0..7


def unfreeze_traits() -> None:
    _FROZEN_TRAIT_INDICES.clear()


@dataclass(frozen=True)
class Genome:
    fear_baseline: float
    seeking_baseline: float
    play_baseline: float
    rage_baseline: float
    reward_safety: float
    reward_reproduction: float
    reward_comfort: float
    reward_hunger: float
    reward_happiness: float
    predate_drive: float
    forage_drive: float
    hunger_aversion: float
    creature_aversion: float
    armor: float
    camouflage: float
    attack_strength: float
    attack_chance: float
    vision_range: float
    metabolism: float
    chase_speed: float
    aggression: float
    reproduction_threshold: float
    offspring_count: float
    lifespan: float
    reproductive_cost: float
    puzzle_solver_strength: float
    food_discrimination: float
    vocal_amplitude: float
    vocal_freq_0: float
    vocal_freq_1: float
    vocal_freq_2: float
    vocal_freq_3: float
    vocal_freq_4: float
    vocal_freq_5: float
    vocal_freq_6: float
    vocal_freq_7: float
    vocal_reflex_fear: float
    audio_att_0: float
    audio_att_1: float
    audio_att_2: float
    audio_att_3: float
    audio_att_4: float
    audio_att_5: float
    audio_att_6: float
    audio_att_7: float

    BASELINE_RANGE: ClassVar[tuple[float, float]] = (-2.0, 2.0)
    UNIT_RANGE: ClassVar[tuple[float, float]] = (0.0, 1.0)
    AVERSION_RANGE: ClassVar[tuple[float, float]] = (0.1, 3.0)
    REPRO_THRESHOLD_RANGE: ClassVar[tuple[float, float]] = (1.0, 10.0)
    OFFSPRING_RANGE: ClassVar[tuple[float, float]] = (1.0, 3.0)
    LIFESPAN_RANGE: ClassVar[tuple[float, float]] = (100.0, 2000.0)
    REPRO_COST_RANGE: ClassVar[tuple[float, float]] = (5.0, 60.0)
    ARMOR_RANGE: ClassVar[tuple[float, float]] = (0.0, 0.7)
    CAMOUFLAGE_RANGE: ClassVar[tuple[float, float]] = (0.0, 0.6)
    ATTACK_STRENGTH_RANGE: ClassVar[tuple[float, float]] = (0.0, 30.0)
    VISION_RANGE: ClassVar[tuple[float, float]] = (3.0, 15.0)
    METABOLISM_RANGE: ClassVar[tuple[float, float]] = (0.05, 1.0)

    @classmethod
    def random(cls, rng: np.random.Generator | None = None) -> "Genome":
        rng = rng or np.random.default_rng()
        # Each individual starts with an idiosyncratic 8-bin wave shape via a
        # symmetric Dirichlet — small perturbations from uniform that give
        # different individuals different "voices" to begin with.
        freq = rng.dirichlet([1.0] * 8).astype(np.float64)
        att = np.full(8, 0.5, dtype=np.float64) + rng.normal(0.0, 0.05, size=8)
        att = np.clip(att, 0.0, 1.0)
        return cls(
            fear_baseline=float(rng.normal(0.0, 0.5)),
            seeking_baseline=float(rng.normal(0.0, 0.5)),
            play_baseline=float(rng.normal(0.0, 0.5)),
            rage_baseline=float(rng.normal(0.0, 0.5)),
            reward_safety=float(np.clip(rng.normal(0.5, 0.2), 0.0, 1.0)),
            reward_reproduction=float(np.clip(rng.normal(0.5, 0.2), 0.0, 1.0)),
            reward_comfort=float(np.clip(rng.normal(0.5, 0.2), 0.0, 1.0)),
            reward_hunger=float(np.clip(rng.normal(0.5, 0.2), 0.0, 1.0)),
            reward_happiness=float(np.clip(rng.normal(0.5, 0.2), 0.0, 1.0)),
            # Niche traits start near zero — pure forager by default; predator
            # behaviour must evolve.
            predate_drive=float(np.clip(rng.normal(0.05, 0.20), 0.0, 1.0)),
            forage_drive=float(np.clip(rng.normal(0.6, 0.20), 0.0, 1.0)),
            hunger_aversion=float(np.clip(rng.normal(1.0, 0.3), *cls.AVERSION_RANGE)),
            creature_aversion=float(np.clip(rng.normal(1.0, 0.3), *cls.AVERSION_RANGE)),
            armor=float(np.clip(rng.normal(0.05, 0.10), *cls.ARMOR_RANGE)),
            camouflage=float(np.clip(rng.normal(0.05, 0.10), *cls.CAMOUFLAGE_RANGE)),
            attack_strength=float(np.clip(rng.normal(2.0, 3.0), *cls.ATTACK_STRENGTH_RANGE)),
            attack_chance=float(np.clip(rng.normal(0.10, 0.10), 0.0, 1.0)),
            vision_range=float(np.clip(rng.normal(7.0, 2.0), *cls.VISION_RANGE)),
            metabolism=float(np.clip(rng.normal(0.20, 0.08), *cls.METABOLISM_RANGE)),
            chase_speed=float(np.clip(rng.normal(0.4, 0.2), 0.0, 1.0)),
            aggression=float(np.clip(rng.normal(0.3, 0.2), 0.0, 1.0)),
            reproduction_threshold=float(np.clip(rng.normal(3.0, 1.0), *cls.REPRO_THRESHOLD_RANGE)),
            offspring_count=float(np.clip(rng.normal(1.2, 0.3), *cls.OFFSPRING_RANGE)),
            lifespan=float(np.clip(rng.normal(500.0, 120.0), *cls.LIFESPAN_RANGE)),
            reproductive_cost=float(np.clip(rng.normal(30.0, 8.0), *cls.REPRO_COST_RANGE)),
            puzzle_solver_strength=float(np.clip(rng.normal(0.5, 0.2), 0.0, 1.0)),
            food_discrimination=float(np.clip(rng.normal(0.5, 0.2), 0.0, 1.0)),
            vocal_amplitude=float(np.clip(rng.normal(0.10, 0.05), 0.0, 1.0)),
            vocal_freq_0=float(freq[0]), vocal_freq_1=float(freq[1]),
            vocal_freq_2=float(freq[2]), vocal_freq_3=float(freq[3]),
            vocal_freq_4=float(freq[4]), vocal_freq_5=float(freq[5]),
            vocal_freq_6=float(freq[6]), vocal_freq_7=float(freq[7]),
            vocal_reflex_fear=float(np.clip(rng.normal(0.05, 0.03), 0.0, 1.0)),
            audio_att_0=float(att[0]), audio_att_1=float(att[1]),
            audio_att_2=float(att[2]), audio_att_3=float(att[3]),
            audio_att_4=float(att[4]), audio_att_5=float(att[5]),
            audio_att_6=float(att[6]), audio_att_7=float(att[7]),
        )

    def mutate(self, sigma: float, rng: np.random.Generator | None = None) -> "Genome":
        rng = rng or np.random.default_rng()
        v = self.to_array()
        # Per-trait scales
        scales = np.array([
            0.5, 0.5, 0.5, 0.5,                          # baselines
            0.20, 0.20, 0.20, 0.20, 0.20,                # rewards
            0.20, 0.20,                                   # diet
            0.30, 0.30,                                   # aversions
            0.10, 0.10, 3.0, 0.10,                        # defences
            2.0, 0.08, 0.20, 0.20,                        # sensorimotor
            1.0, 0.30, 120.0, 8.0,                        # life history
            0.20,                                          # cognition
            0.20,                                          # food_discrimination
            0.20,                                          # vocal_amplitude
            0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,  # vocal_freq_0..7
            0.15,                                          # vocal_reflex_fear
            0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20,  # audio_att_0..7
        ], dtype=np.float64)
        scales_eff = scales.copy()
        if _FROZEN_TRAIT_INDICES:
            for idx in _FROZEN_TRAIT_INDICES:
                if 0 <= idx < scales_eff.shape[0]:
                    scales_eff[idx] = 0.0
        noise = rng.normal(0.0, sigma * scales_eff, size=len(scales_eff))
        new = v + noise
        # Apply clamps
        new[0:4]   = np.clip(new[0:4], *self.BASELINE_RANGE)
        new[4:9]   = np.clip(new[4:9], *self.UNIT_RANGE)
        new[9:11]  = np.clip(new[9:11], *self.UNIT_RANGE)
        new[11:13] = np.clip(new[11:13], *self.AVERSION_RANGE)
        new[13]    = np.clip(new[13], *self.ARMOR_RANGE)
        new[14]    = np.clip(new[14], *self.CAMOUFLAGE_RANGE)
        new[15]    = np.clip(new[15], *self.ATTACK_STRENGTH_RANGE)
        new[16]    = np.clip(new[16], *self.UNIT_RANGE)
        new[17]    = np.clip(new[17], *self.VISION_RANGE)
        new[18]    = np.clip(new[18], *self.METABOLISM_RANGE)
        new[19:21] = np.clip(new[19:21], *self.UNIT_RANGE)
        new[21]    = np.clip(new[21], *self.REPRO_THRESHOLD_RANGE)
        new[22]    = np.clip(new[22], *self.OFFSPRING_RANGE)
        new[23]    = np.clip(new[23], *self.LIFESPAN_RANGE)
        new[24]    = np.clip(new[24], *self.REPRO_COST_RANGE)
        new[25]    = np.clip(new[25], *self.UNIT_RANGE)
        new[26]    = np.clip(new[26], *self.UNIT_RANGE)
        new[27:45] = np.clip(new[27:45], *self.UNIT_RANGE)
        return type(self)(*[float(x) for x in new])

    @classmethod
    def crossover(
        cls, a: "Genome", b: "Genome", rng: np.random.Generator | None = None
    ) -> "Genome":
        """Sexual reproduction — per-trait random pick from either parent + small
        gaussian noise."""
        rng = rng or np.random.default_rng()
        va = a.to_array()
        vb = b.to_array()
        mask = rng.random(va.shape[0]) < 0.5
        new = np.where(mask, va, vb)
        # mild post-crossover mutation — biology equivalent of recombination errors
        return cls.from_array(new).mutate(sigma=0.05, rng=rng)

    def to_array(self) -> np.ndarray:
        return np.asarray([getattr(self, n) for n in TRAIT_NAMES], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Genome":
        if arr.shape != (len(TRAIT_NAMES),):
            raise ValueError(f"expected shape ({len(TRAIT_NAMES)},), got {arr.shape}")
        return cls(*[float(x) for x in arr])

    def traits(self) -> dict[str, float]:
        return {n: float(getattr(self, n)) for n in TRAIT_NAMES}

    def vocal_freq_bias(self) -> np.ndarray:
        return np.array([
            self.vocal_freq_0, self.vocal_freq_1, self.vocal_freq_2, self.vocal_freq_3,
            self.vocal_freq_4, self.vocal_freq_5, self.vocal_freq_6, self.vocal_freq_7,
        ], dtype=np.float32)

    def audio_attention(self) -> np.ndarray:
        return np.array([
            self.audio_att_0, self.audio_att_1, self.audio_att_2, self.audio_att_3,
            self.audio_att_4, self.audio_att_5, self.audio_att_6, self.audio_att_7,
        ], dtype=np.float32)

    def phenotype(self) -> str:
        """Compute an emergent species-label from current trait values."""
        if self.predate_drive > 0.55 and self.forage_drive < 0.4:
            return "predator-like"
        if self.predate_drive < 0.2 and self.forage_drive > 0.55:
            return "forager"
        if self.predate_drive > 0.4 and self.forage_drive > 0.4:
            return "omnivore"
        return "drifter"

    def short_repr(self) -> str:
        return (
            f"[{self.phenotype()}] "
            f"F={self.fear_baseline:+.2f} S={self.seeking_baseline:+.2f} "
            f"hunt={self.predate_drive:.2f} forage={self.forage_drive:.2f} "
            f"|arm={self.armor:.2f} cam={self.camouflage:.2f} "
            f"|R(safe={self.reward_safety:.2f},rep={self.reward_reproduction:.2f},"
            f"hung={self.reward_hunger:.2f})"
        )


def _smoke() -> None:
    rng = np.random.default_rng(0)
    g = Genome.random(rng)
    print(g.short_repr())
    print(f"phenotype: {g.phenotype()}")
    print(f"trait count: {len(g.traits())}")
    g2 = g.mutate(sigma=0.2, rng=rng)
    print("after mutate:", g2.short_repr())
    g3 = Genome.crossover(g, g2, rng=rng)
    print("crossover:   ", g3.short_repr())
    print("\n40 random genomes — phenotype distribution:")
    from collections import Counter
    pts = Counter(Genome.random(rng).phenotype() for _ in range(40))
    print(f"  {dict(pts)}")
    print("OK")


if __name__ == "__main__":
    _smoke()
