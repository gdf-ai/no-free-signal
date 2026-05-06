"""Unified Creature world — one species that evolves into ecological niches.

Replaces multi_instinct_gridworld.py's prey/predator dichotomy with a single
Creature type whose behaviour emerges from genome traits (predate_drive vs
forage_drive vs etc.). Predator-like, forager, omnivore, and cannibal
phenotypes all coexist in the same population.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from foresight.envs.audio import NUM_AUDIO_BINS, AudioField
from foresight.evolution.genome import Genome

# Tile encoding (matches the grid renderer's color map)
TILE_EMPTY = 0
TILE_WALL = 1
TILE_FOOD = 2
TILE_SHELTER = 3
TILE_RAW_WOOD = 4
TILE_RAW_STONE = 5
NUM_TILE_TYPES = 6

# Actions
ACTION_NORTH = 0
ACTION_SOUTH = 1
ACTION_EAST = 2
ACTION_WEST = 3
ACTION_EAT = 4         # eat adjacent food OR adjacent creature (if predate_drive high)
ACTION_REST = 5        # rest in shelter
ACTION_GATHER = 6      # gather raw resource into inventory
ACTION_BUILD = 7       # build shelter from inventory
ACTION_VOCALIZE = 8    # emit an audio wave; cost in energy; speaker self-hears
ACTION_NOOP = 9
NUM_ACTIONS = 10

VOCAL_BASE_COST = 0.5
VOCAL_AMP_COST = 1.0   # multiplier on (pending_amp * vocal_amplitude_genome)

_MOVE_DELTAS = {
    ACTION_NORTH: (-1, 0),
    ACTION_SOUTH: (1, 0),
    ACTION_EAST: (0, 1),
    ACTION_WEST: (0, -1),
}


SOCIAL_CHANNELS: tuple[str, ...] = (
    "food",       # witnessed another creature eat food
    "predation",  # witnessed a creature eat another creature
    "build",      # witnessed a successful BUILD
    "gather",     # witnessed a gather
    "mate",       # witnessed a mating
    "rest",       # witnessed a creature rest in shelter
    "danger",     # witnessed a death or counter-attack
)


@dataclass
class Creature:
    """One unified organism. Genome decides whether it acts predator-like,
    forager-like, omnivore, or cannibal."""
    individual_id: int
    pos: tuple[int, int]
    genome: Genome
    energy: float = 60.0
    fatigue: float = 0.0
    health: float = 100.0
    age: int = 0
    inventory_wood: int = 0
    inventory_stone: int = 0
    food_eaten_total: int = 0
    creatures_eaten_total: int = 0
    food_eaten_since_repro: int = 0
    parent_a_id: int | None = None
    parent_b_id: int | None = None  # second parent for sexual reproduction
    n_offspring: int = 0
    generation: int = 0
    # Vicarious-reward signal: accumulated when this creature WITNESSES another
    # creature performing rewarded actions within its vision_range. Each step
    # the signal decays. Heuristic + brain policies read these to bias action
    # choice — biology: mirror-neuron / observational-learning analogue. The
    # actual gain per channel is gated by this creature's own reward_* genes,
    # so two observers of the same event can react completely differently.
    social_signal: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in SOCIAL_CHANNELS})
    # Vocalization slot, written by an external driver (genome reflex, LLM
    # intent, or brain policy) before env.step(). On VOCALIZE, the wave is
    # emitted into world.audio at this position with this amplitude and
    # cleared. ``vocalized_last_tick`` is set on success and consumed next
    # tick to suppress the fear reflex (echo-loop refractory).
    pending_wave: np.ndarray | None = None
    pending_amp: float = 0.0
    # Tag set by the vocal-driver resolver indicating which driver wrote
    # `pending_wave` this tick: "reflex", "llm", "brain", "random", or None.
    # Read at the audio.emit call site for behavioral-instrumentation tagging.
    pending_driver: str | None = None
    vocalized_last_tick: bool = False


@dataclass
class WorldConfig:
    grid_size: int = 40
    obs_window: int = 7
    n_shelter_initial: int = 5
    n_food_initial: int = 22
    n_resource_initial: int = 12
    # Tuned so food is *scarce enough* to make hunger a real selection
    # pressure. Previously food respawned ~10× faster than 50 creatures
    # could eat it, and `reward_hunger` drifted to 0.16 because there was
    # no fitness gradient. New cap ≈ ~0.6 food per creature at max pop.
    food_respawn_prob: float = 0.004
    resource_respawn_prob: float = 0.004
    max_food: int = 32
    max_resources: int = 40
    initial_creatures: int = 16
    max_creatures: int = 50

    hunger_rate: float = 0.5
    fatigue_rate_move: float = 1.0
    food_energy_restore: float = 22.0
    creature_energy_restore: float = 35.0
    energy_loss_attacked: float = 25.0
    health_loss_starving: float = 1.0
    poison_damage: float = 8.0   # health lost when a creature mistakenly eats matter

    seed: int | None = None


@dataclass
class StepEvents:
    eats_food: list[int] = field(default_factory=list)
    eats_creature: list[tuple[int, int]] = field(default_factory=list)  # (eater, eaten)
    rests: list[int] = field(default_factory=list)
    gathers: list[tuple[int, str]] = field(default_factory=list)        # (cid, "wood"/"stone")
    builds: list[int] = field(default_factory=list)
    counter_attacks: list[tuple[int, int, float]] = field(default_factory=list)
    deaths: list[tuple[int, str]] = field(default_factory=list)
    births: list[int] = field(default_factory=list)
    matings: list[tuple[int, int, int]] = field(default_factory=list)   # (parent_a, parent_b, child)
    build_fails: list[int] = field(default_factory=list)
    poisonings: list[tuple[int, str]] = field(default_factory=list)     # (cid, "wood"/"stone")


class UnifiedWorld:
    """Single-species, multi-agent world with genome-driven niche emergence."""

    def __init__(self, config: WorldConfig | None = None):
        self.config = config or WorldConfig()
        self._rng: np.random.Generator | None = None
        self._grid: np.ndarray
        self._creatures: dict[int, Creature] = {}
        self._next_id: int = 0
        self._steps: int = 0
        self.audio: AudioField = AudioField()
        self.reset(seed=self.config.seed)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng()

        cfg = self.config
        N = cfg.grid_size
        self._grid = np.zeros((N, N), dtype=np.int32)
        self._grid[0, :] = TILE_WALL
        self._grid[-1, :] = TILE_WALL
        self._grid[:, 0] = TILE_WALL
        self._grid[:, -1] = TILE_WALL

        self._place_clusters(cfg.n_shelter_initial, TILE_SHELTER)
        self._place_singles(cfg.n_food_initial, TILE_FOOD)
        self._place_singles(cfg.n_resource_initial // 2, TILE_RAW_WOOD)
        self._place_singles(cfg.n_resource_initial - cfg.n_resource_initial // 2,
                             TILE_RAW_STONE)

        self._creatures.clear()
        self._next_id = 0
        self._steps = 0
        self.audio.clear()
        for _ in range(cfg.initial_creatures):
            self._spawn_creature(genome=Genome.random(self._rng), parent_ids=(None, None),
                                  generation=0)

    # ------------------------------------------------------------------
    @property
    def grid(self) -> np.ndarray:
        return self._grid

    @property
    def creatures(self) -> dict[int, Creature]:
        return self._creatures

    @property
    def steps(self) -> int:
        return self._steps

    # ------------------------------------------------------------------
    def step(self, actions: dict[int, int] | None = None) -> StepEvents:
        if actions is None:
            actions = {}
        ev = StepEvents()
        cfg = self.config
        rng = self._rng

        # 0. Decay vicarious-reward signal carried over from last tick. Also
        # roll over the vocalized-last-tick flag to a transient one so the
        # reflex driver can read it (refractory) before the action handler
        # potentially overwrites it.
        for c in self._creatures.values():
            for k in c.social_signal:
                c.social_signal[k] *= 0.85

        # 1. Apply actions per creature. Reset the vocalized flag now so
        # _apply_action's VOCALIZE branch can set it fresh this tick.
        for c in self._creatures.values():
            c.vocalized_last_tick = False
        for cid, c in list(self._creatures.items()):
            action = actions.get(cid)
            if action is None:
                action = self.heuristic(c)
            self._apply_action(c, action, ev)

        # 2. Update homeostatic vars
        for c in self._creatures.values():
            c.energy = float(np.clip(c.energy - c.genome.metabolism, 0.0, 100.0))
            c.fatigue = float(np.clip(c.fatigue, 0.0, 100.0))
            if c.energy <= 0:
                c.health -= cfg.health_loss_starving
            c.health = float(np.clip(c.health, 0.0, 100.0))
            c.age += 1

        # 3. Resource respawn
        n_food = int((self._grid == TILE_FOOD).sum())
        if n_food < cfg.max_food:
            for _ in range(cfg.grid_size):
                r = int(rng.integers(1, cfg.grid_size - 1))
                cc = int(rng.integers(1, cfg.grid_size - 1))
                if self._grid[r, cc] == TILE_EMPTY and rng.random() < cfg.food_respawn_prob:
                    self._grid[r, cc] = TILE_FOOD
        n_res = int(((self._grid == TILE_RAW_WOOD) | (self._grid == TILE_RAW_STONE)).sum())
        if n_res < cfg.max_resources:
            for _ in range(cfg.grid_size // 2):
                r = int(rng.integers(1, cfg.grid_size - 1))
                cc = int(rng.integers(1, cfg.grid_size - 1))
                if self._grid[r, cc] == TILE_EMPTY and rng.random() < cfg.resource_respawn_prob:
                    self._grid[r, cc] = TILE_RAW_WOOD if rng.random() < 0.5 else TILE_RAW_STONE

        # 4. Apply observational/social reward — every event the creature
        # witnessed within its vision range contributes to its social_signal,
        # weighted by its own reward-center / drive genes.
        self._apply_observation(ev)

        # 5. Deaths
        for cid in list(self._creatures.keys()):
            c = self._creatures[cid]
            if c.health <= 0:
                cause = "starved" if c.energy <= 0 else "killed"
                ev.deaths.append((cid, cause))
                del self._creatures[cid]
            elif c.age >= c.genome.lifespan:
                ev.deaths.append((cid, "old_age"))
                del self._creatures[cid]

        # 6. Births — sexual reproduction between adjacent willing creatures
        if len(self._creatures) < cfg.max_creatures:
            self._resolve_matings(ev)

        # 7. Asexual fallback — high reward_reproduction + lots of food eaten can
        # also produce a child without a partner (parthenogenesis-style, keeps
        # populations from collapsing on isolated individuals).
        if len(self._creatures) < cfg.max_creatures:
            self._resolve_asexual(ev)

        # 8. Decay the audio field. Emissions that happened this tick are
        # halved before any observer (including the speaker, who self-hears
        # at distance 0) reads the field on the next obs build.
        self.audio.tick()

        self._steps += 1
        return ev

    # ------------------------------------------------------------------
    # Heuristic policy — used when no external action is supplied.
    # ------------------------------------------------------------------
    def heuristic(self, c: Creature) -> int:
        rng = self._rng
        traits = c.genome.traits()
        soc = c.social_signal
        # If we just witnessed danger nearby, lean toward fleeing instead of
        # foraging.
        spooked = soc.get("danger", 0.0) > 0.3 + (-traits["fear_baseline"]) * 0.3
        # If we just watched something that hits our reward centres, lean
        # toward the matching action when possible.
        watched_food = soc.get("food", 0.0) > 0.25
        watched_predation = soc.get("predation", 0.0) > 0.3
        watched_build = soc.get("build", 0.0) > 0.25
        watched_gather = soc.get("gather", 0.0) > 0.25
        watched_rest = soc.get("rest", 0.0) > 0.2

        # If standing on shelter and tired (or socially primed), rest.
        if int(self._grid[c.pos]) == TILE_SHELTER and (c.fatigue > 50 or watched_rest) and traits["reward_comfort"] > 0.4:
            return ACTION_REST

        # ---- 4-way feeding decision ----
        # Score each candidate target a creature can act on at this position:
        # food, predation, matter (poison risk), shelter (rest). Highest score
        # wins; if none are positive, fall through to navigation.
        adj_food = self._adjacent_to(c.pos, TILE_FOOD)
        adj_creature = self._adjacent_creature(c)
        adj_matter = any(
            int(self._grid[r, cc]) in (TILE_RAW_WOOD, TILE_RAW_STONE)
            for r, cc in self._adjacent_cells_with(c.pos)
        )
        on_shelter = int(self._grid[c.pos]) == TILE_SHELTER
        hunger_pressure = max(0.0, (100.0 - c.energy) / 100.0)
        fatigue_pressure = c.fatigue / 100.0
        discrimination = float(np.clip(traits.get("food_discrimination", 0.5), 0.0, 1.0))

        score_food = (
            traits["forage_drive"] * (0.5 + hunger_pressure)
            + 0.6 * soc.get("food", 0.0)
        ) if adj_food else float("-inf")
        score_creature = (
            traits["predate_drive"] * (0.4 + hunger_pressure)
            + 0.5 * soc.get("predation", 0.0)
            - 0.3 * traits["creature_aversion"]
        ) if (adj_creature is not None and traits["creature_aversion"] < 1.5) else float("-inf")
        # Matter score: positive only when discrimination is poor AND hunger is
        # high. Even then the poison cost dominates, so this is almost always
        # negative — but it lets clueless creatures occasionally test matter.
        score_matter = (
            (1.0 - discrimination) * (0.2 + 0.5 * hunger_pressure) - 0.6
        ) if adj_matter else float("-inf")
        score_rest = (
            traits["reward_comfort"] * fatigue_pressure
            + 0.5 * soc.get("rest", 0.0)
        ) if on_shelter else float("-inf")

        candidates = {
            ACTION_EAT: max(score_food, score_creature, score_matter),
            ACTION_REST: score_rest,
        }
        best_action, best_score = max(candidates.items(), key=lambda kv: kv[1])
        if best_score > 0.05:
            return best_action

        # Standing on resource with capacity — gather (vicarious gather raises
        # priority).
        tile = int(self._grid[c.pos])
        if tile in (TILE_RAW_WOOD, TILE_RAW_STONE) and (c.inventory_wood + c.inventory_stone) < 8:
            return ACTION_GATHER
        # Build if we have enough resources and stable energy. Witnessed-build
        # boosts the build probability sharply.
        build_p = 0.05 + (0.3 if watched_build else 0.0)
        if c.inventory_wood >= 2 and c.inventory_stone >= 2 and c.energy > 40 and rng.random() < build_p:
            return ACTION_BUILD

        # Spooked → flee threats more aggressively
        if spooked:
            nt = self._nearest_threatening_creature(c)
            if nt is not None:
                return self._move_away(c.pos, nt.pos, rng)
            shelter_pos = self._nearest_tile_pos(c.pos, TILE_SHELTER,
                                                  max_dist=int(traits["vision_range"]))
            if shelter_pos is not None:
                return self._move_toward(c.pos, shelter_pos, rng)

        # Threat-aware navigation
        nearest_threat = self._nearest_threatening_creature(c)
        if nearest_threat is not None:
            d = self._dist(c.pos, nearest_threat.pos)
            fear = traits["fear_baseline"] + (4 - d) / 4 * traits["creature_aversion"]
            if d <= 1 and rng.random() < traits["attack_chance"] * 0.5:
                return self._move_toward(c.pos, nearest_threat.pos, rng)
            if fear > 0.5:
                return self._move_away(c.pos, nearest_threat.pos, rng)

        # Hunt: move toward nearest weaker creature
        if traits["predate_drive"] > 0.4 and c.energy < 80:
            target = self._nearest_huntable(c)
            if target is not None:
                return self._move_toward(c.pos, target.pos, rng)

        # Forage: move toward nearest food
        if traits["forage_drive"] > 0.3 and c.energy < 90:
            food_pos = self._nearest_tile_pos(c.pos, TILE_FOOD,
                                              max_dist=int(traits["vision_range"]))
            if food_pos is not None:
                return self._move_toward(c.pos, food_pos, rng)

        # Drift toward shelter when reward_safety high and threats around
        if traits["reward_safety"] > 0.5:
            shelter_pos = self._nearest_tile_pos(c.pos, TILE_SHELTER,
                                                  max_dist=int(traits["vision_range"]))
            if shelter_pos is not None and rng.random() < 0.4:
                return self._move_toward(c.pos, shelter_pos, rng)

        return int(rng.integers(0, 4))  # random walk N/S/E/W

    # ------------------------------------------------------------------
    def _apply_action(self, c: Creature, action: int, ev: StepEvents) -> None:
        cfg = self.config
        if action in _MOVE_DELTAS:
            c.fatigue += cfg.fatigue_rate_move
            dr, dc = _MOVE_DELTAS[action]
            nr, nnc = c.pos[0] + dr, c.pos[1] + dc
            if self._is_passable(nr, nnc):
                c.pos = (nr, nnc)
        elif action == ACTION_EAT:
            self._try_eat(c, ev)
        elif action == ACTION_REST:
            if int(self._grid[c.pos]) == TILE_SHELTER:
                c.fatigue = max(0.0, c.fatigue - 5.0)
                ev.rests.append(c.individual_id)
        elif action == ACTION_GATHER:
            tile = int(self._grid[c.pos])
            if tile == TILE_RAW_WOOD:
                c.inventory_wood += 1
                self._grid[c.pos] = TILE_EMPTY
                ev.gathers.append((c.individual_id, "wood"))
            elif tile == TILE_RAW_STONE:
                c.inventory_stone += 1
                self._grid[c.pos] = TILE_EMPTY
                ev.gathers.append((c.individual_id, "stone"))
        elif action == ACTION_VOCALIZE:
            self._try_vocalize(c)
        elif action == ACTION_BUILD:
            if (c.inventory_wood >= 2 and c.inventory_stone >= 2
                and int(self._grid[c.pos]) == TILE_EMPTY):
                # If a puzzle gate is attached (no_free_signal attaches one when brains
                # are enabled), call it. Otherwise build always succeeds.
                gate = getattr(self, "_build_puzzle_gate", None)
                ok = True
                if gate is not None:
                    try:
                        ok = bool(gate(c))
                    except Exception:
                        ok = True  # don't penalise creatures for our bugs
                if ok:
                    c.inventory_wood -= 2
                    c.inventory_stone -= 2
                    self._grid[c.pos] = TILE_SHELTER
                    ev.builds.append(c.individual_id)
                else:
                    # Failed puzzle: lose 1 unit of each resource and tag the
                    # event for the UI.
                    c.inventory_wood = max(0, c.inventory_wood - 1)
                    c.inventory_stone = max(0, c.inventory_stone - 1)
                    fails = getattr(ev, "build_fails", None)
                    if fails is None:
                        ev.build_fails = []  # patched on the fly
                        fails = ev.build_fails
                    fails.append(c.individual_id)
        # NOOP: nothing

    def _try_vocalize(self, c: Creature) -> None:
        """Emit a wave into the audio field if the driver wrote one. If
        ``pending_wave`` is None or amplitude is zero, collapse to NOOP at no
        cost — keeps the brain from being punished out of ever choosing
        VOCALIZE while exploring. Wave + amp are cleared after consumption."""
        wave = c.pending_wave
        amp = float(c.pending_amp)
        driver = c.pending_driver
        c.pending_wave = None
        c.pending_amp = 0.0
        c.pending_driver = None
        if wave is None or amp <= 0.0:
            return
        if wave.shape != (NUM_AUDIO_BINS,):
            return
        cost = VOCAL_BASE_COST + VOCAL_AMP_COST * amp * float(c.genome.vocal_amplitude)
        if c.energy < cost:
            return  # not enough energy to vocalize this tick
        self.audio.emit(c.pos, wave, amplitude=amp)
        c.energy = float(np.clip(c.energy - cost, 0.0, 100.0))
        c.vocalized_last_tick = True
        # Behavioral instrumentation hook: optional callback fired
        # immediately after a successful emit, before the driver tag
        # is lost. Off by default (None); set externally by the harness
        # when --log-behavioral is on.
        cb = getattr(self, "_emit_observer", None)
        if cb is not None:
            try:
                cb(creature=c, wave=wave, amp=amp, driver=driver,
                   tick=self.steps)
            except Exception:
                # Never let instrumentation break the simulation.
                pass

    def _try_eat(self, c: Creature, ev: StepEvents) -> None:
        cfg = self.config
        traits = c.genome.traits()
        # First try food tile (adjacent or under)
        if traits["forage_drive"] > 0.1:
            for r, cc in self._adjacent_cells_with(c.pos):
                if int(self._grid[r, cc]) == TILE_FOOD:
                    self._grid[r, cc] = TILE_EMPTY
                    c.energy = min(100.0, c.energy + cfg.food_energy_restore)
                    c.food_eaten_total += 1
                    c.food_eaten_since_repro += 1
                    ev.eats_food.append(c.individual_id)
                    return
        # Then try predation on adjacent creature
        if traits["predate_drive"] > 0.3:
            target = self._adjacent_creature(c)
            if target is not None and target.individual_id != c.individual_id:
                # Apply armor reduction; counter-attack risk
                damage = cfg.energy_loss_attacked * (1.0 - target.genome.armor)
                target.health -= damage
                c.energy = min(100.0, c.energy + cfg.creature_energy_restore)
                c.creatures_eaten_total += 1
                c.food_eaten_since_repro += 1  # creature counts as food for repro
                ev.eats_creature.append((c.individual_id, target.individual_id))
                if self._rng.random() < target.genome.attack_chance:
                    counter = target.genome.attack_strength * (1.0 - c.genome.armor)
                    c.energy -= counter
                    if counter > 0:
                        ev.counter_attacks.append((target.individual_id, c.individual_id, float(counter)))
                return
        # Fallthrough: no food, no huntable target. If matter (raw_wood / raw_stone)
        # is adjacent and food_discrimination is imperfect, the creature may
        # mistakenly ingest it and be poisoned. The matter tile itself stays
        # intact so other creatures can independently encounter it (and so the
        # selection signal is durable).
        discrimination = float(np.clip(traits.get("food_discrimination", 0.5), 0.0, 1.0))
        for r, cc in self._adjacent_cells_with(c.pos):
            tile = int(self._grid[r, cc])
            if tile == TILE_RAW_WOOD or tile == TILE_RAW_STONE:
                if self._rng.random() >= discrimination:
                    c.health = max(0.0, c.health - cfg.poison_damage)
                    matter_kind = "wood" if tile == TILE_RAW_WOOD else "stone"
                    ev.poisonings.append((c.individual_id, matter_kind))
                return

    # ------------------------------------------------------------------
    def _resolve_matings(self, ev: StepEvents) -> None:
        cfg = self.config
        rng = self._rng
        attempted: set[int] = set()
        for cid, c in list(self._creatures.items()):
            if cid in attempted: continue
            if len(self._creatures) >= cfg.max_creatures: break
            ct = c.genome.traits()
            if ct["reward_reproduction"] < 0.45 or c.energy < 35: continue
            partner = self._adjacent_creature(c)
            if partner is None: continue
            if partner.individual_id in attempted: continue
            pt = partner.genome.traits()
            if pt["reward_reproduction"] < 0.45 or partner.energy < 35: continue
            # genome compatibility — euclidean distance over key reproductive
            # traits; lower is more compatible (more likely to produce viable child)
            compat = float(np.linalg.norm(c.genome.to_array() - partner.genome.to_array()))
            if compat > 8.0 and rng.random() > 0.2: continue  # too distant
            # mate
            child_pos = self._adjacent_empty_cell(c.pos)
            if child_pos is None: continue
            child_genome = Genome.crossover(c.genome, partner.genome, rng=rng)
            child = self._spawn_creature(
                genome=child_genome,
                parent_ids=(c.individual_id, partner.individual_id),
                generation=max(c.generation, partner.generation) + 1,
                pos=child_pos,
            )
            ev.births.append(child.individual_id)
            ev.matings.append((c.individual_id, partner.individual_id, child.individual_id))
            c.n_offspring += 1
            partner.n_offspring += 1
            c.energy *= 0.6
            partner.energy *= 0.6
            c.food_eaten_since_repro = 0
            partner.food_eaten_since_repro = 0
            attempted.add(cid)
            attempted.add(partner.individual_id)

    def _resolve_asexual(self, ev: StepEvents) -> None:
        """Parthenogenesis-style fallback so isolated creatures can still propagate
        DNA forward. Triggered only by very high reward_reproduction or when the
        creature has eaten a lot."""
        cfg = self.config
        for cid, c in list(self._creatures.items()):
            if len(self._creatures) >= cfg.max_creatures: break
            t = c.genome.traits()
            ready = (
                c.food_eaten_since_repro >= t["reproduction_threshold"]
                and c.energy > 50
                and t["reward_reproduction"] > 0.55
            )
            if not ready: continue
            n_off = int(round(t["offspring_count"]))
            for _ in range(n_off):
                if len(self._creatures) >= cfg.max_creatures: break
                child_pos = self._adjacent_empty_cell(c.pos)
                if child_pos is None: break
                child_genome = c.genome.mutate(sigma=0.10, rng=self._rng)
                child = self._spawn_creature(
                    genome=child_genome,
                    parent_ids=(c.individual_id, None),
                    generation=c.generation + 1,
                    pos=child_pos,
                )
                ev.births.append(child.individual_id)
                c.n_offspring += 1
            c.energy = max(0.0, c.energy - t["reproductive_cost"])
            c.food_eaten_since_repro = 0

    # ------------------------------------------------------------------
    # Observational / vicarious reward (Phase 9.G)
    # ------------------------------------------------------------------
    def _apply_observation(self, ev: StepEvents) -> None:
        """For every event with a known actor position, find living creatures
        within the actor's neighbourhood whose vision_range covers it, and add
        a genome-weighted contribution to their `social_signal`. Each observer
        weights events through its own reward-centre genes — two witnesses of
        the same kill can react completely differently."""
        # Build a list of (actor_id, actor_pos, channel, intensity) entries.
        observations: list[tuple[int, tuple[int, int], str, float]] = []

        def add(actor_id: int, channel: str, intensity: float = 1.0) -> None:
            actor = self._creatures.get(actor_id)
            if actor is None:
                return
            observations.append((actor_id, actor.pos, channel, intensity))

        for cid in ev.eats_food:
            add(cid, "food", 1.0)
        for eater, victim in ev.eats_creature:
            add(eater, "predation", 1.0)
            # Victims often die instantly: log it as a separate danger event so
            # nearby creatures get a fear spike too.
            add(eater, "danger", 0.6)
        for cid in ev.builds:
            add(cid, "build", 1.0)
        for cid, _ in ev.gathers:
            add(cid, "gather", 0.7)
        for parent_a, parent_b, _child in ev.matings:
            add(parent_a, "mate", 1.0)
            add(parent_b, "mate", 1.0)
        for cid in ev.rests:
            add(cid, "rest", 0.6)
        for defender, attacker, _dmg in ev.counter_attacks:
            add(attacker, "danger", 1.0)
            add(defender, "danger", 0.5)
        for cid, _matter in ev.poisonings:
            # Watching another creature poison itself is informative — nearby
            # observers get a danger spike (they should learn not to do that).
            add(cid, "danger", 0.7)

        if not observations:
            return

        for observer in self._creatures.values():
            ot = observer.genome.traits()
            vision = float(ot.get("vision_range", 5.0))
            for actor_id, actor_pos, channel, intensity in observations:
                if actor_id == observer.individual_id:
                    continue
                d = max(
                    abs(actor_pos[0] - observer.pos[0]),
                    abs(actor_pos[1] - observer.pos[1]),
                )
                if d > vision:
                    continue
                # Falloff with distance (closer = more salient).
                falloff = max(0.0, 1.0 - d / max(1.0, vision))
                gain = self._observer_gain(channel, ot)
                observer.social_signal[channel] = float(
                    np.clip(
                        observer.social_signal[channel] + gain * intensity * falloff,
                        0.0,
                        2.0,
                    )
                )

    @staticmethod
    def _observer_gain(channel: str, traits: dict[str, float]) -> float:
        """Reward-centre weighting that turns an observed event into a
        vicarious-reward gain for this specific observer's genome."""
        rh = float(traits.get("reward_hunger", 0.5))
        rs = float(traits.get("reward_safety", 0.5))
        rr = float(traits.get("reward_reproduction", 0.5))
        rc = float(traits.get("reward_comfort", 0.5))
        rhap = float(traits.get("reward_happiness", 0.5))
        pred = float(traits.get("predate_drive", 0.0))
        forg = float(traits.get("forage_drive", 0.0))
        fear = float(traits.get("fear_baseline", 0.0))
        cav = float(traits.get("creature_aversion", 1.0))
        if channel == "food":
            return 0.6 * rh * (0.5 + 0.5 * forg)
        if channel == "predation":
            return 0.5 * rh * (0.4 + 0.6 * pred)
        if channel == "build":
            return 0.7 * rs
        if channel == "gather":
            return 0.4 * rhap + 0.2 * rs
        if channel == "mate":
            return 0.7 * rr
        if channel == "rest":
            return 0.5 * rc
        if channel == "danger":
            # Danger gain is genome-bounded: bold creatures (negative
            # fear_baseline, low creature_aversion) feel less of the spike.
            base = 0.5 * (1.0 + fear)  # fear in roughly [-2,2]
            return float(np.clip(base * (cav / 2.0), 0.0, 1.5))
        return 0.0

    # ------------------------------------------------------------------
    # Spawn / geometry helpers
    # ------------------------------------------------------------------
    def _spawn_creature(
        self,
        genome: Genome,
        parent_ids: tuple[int | None, int | None],
        generation: int,
        pos: tuple[int, int] | None = None,
    ) -> Creature:
        if pos is None:
            pos = self._random_empty_cell()
        c = Creature(
            individual_id=self._next_id,
            pos=pos,
            genome=genome,
            parent_a_id=parent_ids[0],
            parent_b_id=parent_ids[1],
            generation=generation,
        )
        self._creatures[self._next_id] = c
        self._next_id += 1
        return c

    def _is_passable(self, r: int, c: int) -> bool:
        if r < 0 or c < 0 or r >= self._grid.shape[0] or c >= self._grid.shape[1]:
            return False
        return int(self._grid[r, c]) != TILE_WALL

    def _random_empty_cell(self) -> tuple[int, int]:
        rs, cs = np.where(self._grid == TILE_EMPTY)
        cells = list(zip(rs.tolist(), cs.tolist()))
        occupied = {p.pos for p in self._creatures.values()}
        cells = [c for c in cells if c not in occupied]
        if not cells:
            return (1, 1)
        return cells[int(self._rng.integers(0, len(cells)))]

    def _adjacent_empty_cell(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = pos[0] + dr, pos[1] + dc
            if not (0 <= r < self._grid.shape[0] and 0 <= c < self._grid.shape[1]):
                continue
            if int(self._grid[r, c]) == TILE_WALL:
                continue
            occupied = {p.pos for p in self._creatures.values()}
            if (r, c) not in occupied:
                return (r, c)
        return None

    def _adjacent_to(self, pos: tuple[int, int], tile: int) -> bool:
        for r, c in self._adjacent_cells_with(pos):
            if int(self._grid[r, c]) == tile:
                return True
        return False

    def _adjacent_cells_with(self, pos: tuple[int, int]):
        H, W = self._grid.shape
        for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = pos[0] + dr, pos[1] + dc
            if 0 <= r < H and 0 <= c < W:
                yield (r, c)

    def _adjacent_creature(self, me: Creature) -> Creature | None:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = me.pos[0] + dr, me.pos[1] + dc
            for other in self._creatures.values():
                if other.pos == (r, c) and other.individual_id != me.individual_id:
                    return other
        return None

    def _nearest_threatening_creature(self, me: Creature) -> Creature | None:
        best, best_d = None, 6
        my_id = me.individual_id
        for other in self._creatures.values():
            if other.individual_id == my_id: continue
            ot = other.genome.traits()
            if ot["predate_drive"] < 0.3: continue
            d = max(abs(other.pos[0] - me.pos[0]), abs(other.pos[1] - me.pos[1]))
            if d < best_d:
                best, best_d = other, d
        return best

    def _nearest_huntable(self, me: Creature) -> Creature | None:
        my = me.genome.traits()
        best, best_d = None, int(my["vision_range"]) + 1
        for other in self._creatures.values():
            if other.individual_id == me.individual_id: continue
            # Avoid hunting kin if cooperative (low predate_drive against
            # similar genome). Simple heuristic: skip very-close-genome targets
            # when creature_aversion is high.
            d = max(abs(other.pos[0] - me.pos[0]), abs(other.pos[1] - me.pos[1]))
            if d < best_d:
                best, best_d = other, d
        return best

    def _nearest_tile_pos(self, pos: tuple[int, int], tile: int,
                          max_dist: int) -> tuple[int, int] | None:
        rs, cs = np.where(self._grid == tile)
        best, best_d = None, max_dist + 1
        for r, c in zip(rs.tolist(), cs.tolist()):
            d = max(abs(r - pos[0]), abs(c - pos[1]))
            if d < best_d:
                best, best_d = (r, c), d
        return best

    def _dist(self, a, b) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _move_toward(self, src, dst, rng) -> int:
        dr = dst[0] - src[0]
        dc = dst[1] - src[1]
        if dr == 0 and dc == 0:
            return int(rng.integers(0, 4))
        if abs(dr) >= abs(dc):
            return ACTION_NORTH if dr < 0 else ACTION_SOUTH
        return ACTION_WEST if dc < 0 else ACTION_EAST

    def _move_away(self, src, danger, rng) -> int:
        dr = src[0] - danger[0]
        dc = src[1] - danger[1]
        if dr == 0 and dc == 0:
            return int(rng.integers(0, 4))
        if abs(dr) >= abs(dc):
            return ACTION_NORTH if dr < 0 else ACTION_SOUTH
        return ACTION_WEST if dc < 0 else ACTION_EAST

    def _place_clusters(self, n: int, tile: int) -> None:
        cfg = self.config
        placed = 0
        attempts = 0
        while placed < n and attempts < 200:
            r = int(self._rng.integers(2, cfg.grid_size - 3))
            c = int(self._rng.integers(2, cfg.grid_size - 3))
            block = self._grid[r:r + 2, c:c + 2]
            if np.all(block == TILE_EMPTY):
                self._grid[r:r + 2, c:c + 2] = tile
                placed += 1
            attempts += 1

    def _place_singles(self, n: int, tile: int) -> None:
        cells = list(zip(*np.where(self._grid == TILE_EMPTY)))
        n = min(n, len(cells))
        idx = self._rng.choice(len(cells), size=n, replace=False)
        for i in idx:
            r, c = cells[int(i)]
            self._grid[r, c] = tile

    # ------------------------------------------------------------------
    def population_summary(self) -> dict:
        if not self._creatures:
            return {"n_creatures": 0, "mean_genome": None, "phenotype_counts": {}}
        ages = [c.age for c in self._creatures.values()]
        gens = [c.generation for c in self._creatures.values()]
        arr = np.stack([c.genome.to_array() for c in self._creatures.values()])
        from collections import Counter
        pheno_counts = Counter(c.genome.phenotype() for c in self._creatures.values())
        return {
            "n_creatures": len(self._creatures),
            "mean_genome": arr.mean(axis=0),
            "ages": ages,
            "max_age": int(max(ages)),
            "max_generation": int(max(gens)),
            "phenotype_counts": dict(pheno_counts),
        }


def _smoke() -> None:
    cfg = WorldConfig(seed=0)
    w = UnifiedWorld(cfg)
    print(f"init: {len(w.creatures)} creatures")
    n_births = n_deaths = n_eats = n_predations = n_builds = 0
    for s in range(2000):
        ev = w.step()
        n_births += len(ev.births)
        n_deaths += len(ev.deaths)
        n_eats += len(ev.eats_food)
        n_predations += len(ev.eats_creature)
        n_builds += len(ev.builds)
        if s % 200 == 199:
            from collections import Counter
            phen = Counter(c.genome.phenotype() for c in w.creatures.values())
            print(f"step {s+1}: n={len(w.creatures)} births={n_births} deaths={n_deaths} "
                  f"food_eats={n_eats} predations={n_predations} builds={n_builds} "
                  f"phenotypes={dict(phen)}")
        if not w.creatures:
            print(">>> extinction at step", s+1)
            break
    print("done")


if __name__ == "__main__":
    _smoke()
