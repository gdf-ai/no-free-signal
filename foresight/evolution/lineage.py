"""Lineage: the graveyard of past individuals plus parent-selection logic.

When an agent dies, its (genome, fitness, lifetime, ...) is appended here. New
agents are born by roulette-wheel sampling a parent from the lineage weighted
by fitness — so lines that died young contribute less DNA. A small fraction of
births use a fresh-random genome (diversity injection).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foresight.evolution.genome import TRAIT_NAMES, Genome


@dataclass(frozen=True)
class Ancestor:
    genome: Genome
    fitness: float                  # higher is better
    lifetime: int                   # number of env steps survived
    generation: int                 # generation index (0-based)
    food_eaten: int
    predator_hits: int
    individual_id: int              # unique sequential ID across the run
    parent_id: int | None = None    # individual_id of parent (None if random / first)
    mutation_sigma: float = 0.0     # sigma used when mutating parent's genome
    is_random: bool = False         # True if genome was a fresh random (not mutation of parent)


@dataclass
class Lineage:
    ancestors: list[Ancestor] = field(default_factory=list)
    fresh_random_prob: float = 0.05  # chance of a brand-new random genome on birth

    @property
    def n_ancestors(self) -> int:
        return len(self.ancestors)

    @property
    def n_generations(self) -> int:
        return self.ancestors[-1].generation + 1 if self.ancestors else 0

    def add(self, ancestor: Ancestor) -> None:
        self.ancestors.append(ancestor)

    def best(self) -> Ancestor | None:
        if not self.ancestors:
            return None
        return max(self.ancestors, key=lambda a: a.fitness)

    def sample_parent(self, rng: np.random.Generator) -> Ancestor:
        """Roulette-wheel sample weighted by fitness (shifted to be non-negative)."""
        if not self.ancestors:
            raise RuntimeError("cannot sample parent from empty lineage")
        fitnesses = np.array([a.fitness for a in self.ancestors], dtype=np.float64)
        # shift so the worst ancestor has weight ~0, best has weight = fitness range
        shifted = fitnesses - fitnesses.min() + 1.0
        probs = shifted / shifted.sum()
        idx = int(rng.choice(len(self.ancestors), p=probs))
        return self.ancestors[idx]

    def next_genome(
        self, sigma: float, rng: np.random.Generator
    ) -> tuple[Genome, Ancestor | None]:
        """Produce the genome for the next individual.

        Returns (new_genome, parent_or_None). With small probability returns a
        fresh-random genome (diversity injection); parent is None in that case.
        """
        if (not self.ancestors) or rng.random() < self.fresh_random_prob:
            return Genome.random(rng), None
        parent = self.sample_parent(rng)
        return parent.genome.mutate(sigma=sigma, rng=rng), parent

    def lifetime_curve(self) -> np.ndarray:
        """Per-generation lifetime — useful for the survivorship chart in the UI."""
        if not self.ancestors:
            return np.empty(0, dtype=np.float32)
        return np.asarray([a.lifetime for a in self.ancestors], dtype=np.float32)

    def fitness_curve(self) -> np.ndarray:
        if not self.ancestors:
            return np.empty(0, dtype=np.float32)
        return np.asarray([a.fitness for a in self.ancestors], dtype=np.float32)

    def trait_history(self) -> np.ndarray:
        """Per-generation trait values — shape (n_ancestors, len(TRAIT_NAMES))."""
        if not self.ancestors:
            return np.empty((0, len(TRAIT_NAMES)), dtype=np.float32)
        return np.stack([a.genome.to_array() for a in self.ancestors], axis=0)


# ----------------------------------------------------------------------
# Smoke test — `python -m foresight.evolution.lineage`
# ----------------------------------------------------------------------
def _smoke() -> None:
    rng = np.random.default_rng(0)
    lineage = Lineage()

    print("=== empty lineage ===")
    g, parent = lineage.next_genome(sigma=0.1, rng=rng)
    assert parent is None
    print(f"  first individual (no parent): {g.short_repr()}")

    print("\n=== simulate 30 generations: fitness scales with predator_aversion ===")
    # synthetic experiment: high predator_aversion -> longer life. Selection
    # should drive predator_aversion up over generations.
    for gen in range(30):
        if gen == 0:
            individual_id = 0
            current_genome = g
            parent_a = None
        else:
            current_genome, parent_a = lineage.next_genome(sigma=0.15, rng=rng)
            individual_id = gen
        # synthetic fitness: more predator_aversion + a bit of noise
        lifetime = int(50 * current_genome.predator_aversion + rng.normal(0, 5))
        lifetime = max(5, lifetime)
        fitness = lifetime + 5 * rng.integers(0, 3)
        lineage.add(
            Ancestor(
                genome=current_genome,
                fitness=float(fitness),
                lifetime=lifetime,
                generation=gen,
                food_eaten=int(rng.integers(0, 3)),
                predator_hits=int(rng.integers(0, 4)),
                individual_id=individual_id,
                parent_id=parent_a.individual_id if parent_a else None,
                mutation_sigma=0.15 if parent_a else 0.0,
                is_random=parent_a is None and gen > 0,
            )
        )

    print(f"  total ancestors: {lineage.n_ancestors}")
    best = lineage.best()
    print(f"  best ancestor:    fitness={best.fitness:.1f}  lifetime={best.lifetime}  "
          f"genome: {best.genome.short_repr()}")
    print(f"  fitness curve last 5: {lineage.fitness_curve()[-5:].tolist()}")

    # mean predator_aversion of first 10 vs last 10
    history = lineage.trait_history()
    pred_idx = TRAIT_NAMES.index("predator_aversion")
    print(
        f"  predator_aversion: first 10 mean = {history[:10, pred_idx].mean():.3f}  "
        f"last 10 mean = {history[-10:, pred_idx].mean():.3f}  "
        f"({'evolution working' if history[-10:, pred_idx].mean() > history[:10, pred_idx].mean() else 'no clear trend'})"
    )

    print("\nOK")


if __name__ == "__main__":
    _smoke()
