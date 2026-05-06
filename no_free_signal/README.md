# Lamdis

**A biologically-grounded sensory substrate for AI agents.**

Lamdis runs a tiny living ecosystem — multi-species prey/predator co-evolution with DNA-driven behaviour, real reproduction, real death — and exposes it at **multiple grounding levels** so different kinds of AI agent (symbolic, multimodal, embodied, hybrid) can attach to the same world at the abstraction level their architecture supports.

The point: **ground reasoning about drives, consequences, and ecological dynamics in a real simulation, not in word associations** — and be honest about the fact that for true sensory grounding you need raw observations, not just JSON.

---

## What it is

- A persistent, evolving multi-agent world running locally.
- DNA encoding 14 prey traits (drives, defenses, reproduction params) + 7 predator traits.
- Real co-evolutionary arms race (prey evolve armor / camouflage / counter-attack; predators evolve speed / vision / armor).
- All state exposed as plain JSON.
- An MCP server so any tool-using LLM can drive the world: observe, step time, zoom into creatures, intervene.

## Why it exists

LLMs reason about "fear", "hunger", "consequences" through text association. They've never been hungry. Their reasoning about biological drives is shallow because it's purely linguistic.

The honest answer to "can you fix that with a tool API?" is **partially**. JSON tool calls give an LLM *causal* grounding (its actions cause real consequences it sees back), which is a step beyond pure language modelling — but it isn't *sensory* grounding. For genuine sensory grounding you need raw observation streams, not JSON.

Lamdis exposes both, plus everything in between. Same world, multiple attachment surfaces:

| Layer | What the agent gets | Grounding strength |
|---|---|---|
| **Symbolic / JSON** | Structured world snapshots + tool calls (the MCP server) | Causal — actions produce real consequences |
| **Perceptual** | Rendered grid frames (PNG/numpy) | Visual structure available |
| **Sensory / embodied** | Raw 7×7×5 observation tensor + drive scalars — the same input the creature's own brain sees | True sensorimotor loop |
| **Hybrid** | LLM reasoning over JSON, controlling an embodied creature that's getting raw obs | Symbolic reasoning anchored to a real sensory loop |

The JSON / MCP layer is shipped today; the raw-obs and render APIs are next. Together they make Lamdis a substrate that supports the full grounding spectrum, not a glorified text dump.

## Quick start

```python
from no_free_signal import World

world = World(seed=42, n_prey=8, n_predators=2)

# Snapshot the world
state = world.observe()
print(f"step {state['step']}: "
      f"{state['populations']['n_prey']} prey, "
      f"{state['populations']['n_predators']} predators")

# Advance time
result = world.step(n=100)
print(f"events: {len(result['events'])}")

# Zoom in on a creature
if state['prey']:
    creature = world.observe_creature(state['prey'][0]['id'])
    print(f"creature {creature['id']}: {creature['genome_summary']}")
    print(f"  threats: {creature['nearby']['threats']}")

# Intervene
world.add_food(x=15, y=15)
world.introduce_creature(
    species='prey',
    genome_overrides={'armor': 0.6, 'camouflage': 0.5},
)
```

## MCP server

Lamdis ships with a Model Context Protocol server. Once installed, any
MCP-aware LLM (Claude Desktop, Claude Code, etc.) can call into a Lamdis
world as tools.

```bash
# Run the MCP server (stdio transport, for Claude Desktop / Code)
python -m no_free_signal.mcp_server
```

Tools exposed:

| Tool | Purpose |
|---|---|
| `observe` | Snapshot population state, recent events, genome aggregates |
| `observe_creature(id)` | Full detail on one individual: genome, drives, nearby threats/food |
| `list_creatures` | Compact roster of every living creature |
| `step(n)` | Advance time, return events |
| `lineage(id)` | Ancestry chain |
| `add_food(x, y)` | Drop a food tile |
| `add_shelter(x, y)` | Drop a shelter tile |
| `kill_creature(id)` | God action |
| `introduce_creature(species, genome_overrides?)` | Spawn a new individual |
| `reset(seed?)` | Reset to a fresh world |
| `world_status` | Quick health check |

In Claude Desktop's `mcp.json`:

```json
{
  "mcpServers": {
    "no_free_signal": {
      "command": "python",
      "args": ["-m", "no_free_signal.mcp_server"]
    }
  }
}
```

Then ask Claude: *"Observe the Lamdis ecosystem, run 50 steps, and tell me which species is winning the arms race."*

## What's deployable today

- **Python package** — pip-installable, importable as `no_free_signal`.
- **MCP server** — any tool-using LLM can drive the simulation.
- **JSON observation surface** — plain dicts, easy to log, replay, or feed into datasets.
- **Snapshot / restore** — rudimentary counterfactual support (`world.snapshot()` + `world.restore(snapshot)`).

## What's coming next

Roadmap (in dependency order):

1. **Per-individual neural brains** (Phase 8 in the plan). Each creature gets its own small encoder + recurrent dynamics + drive heads. Today they use a DNA-driven heuristic.
2. **Counterfactual rollout API** — `simulate_outcome(creature_id, action, n_steps)` that branches without committing.
3. **Dataset generator** — produce `(state, action, consequence)` corpora at scale, exportable as JSONL or HuggingFace dataset.
4. **Eval suite** — biological-reasoning benchmark for LLMs, with the simulator as ground truth.
5. **Cloud / federation layer** — distributed instances exchanging genomes, optional.
6. **Polished standalone game UX** — the public-facing Tamagotchi-style window, sharing the same engine.

## Architecture

Lamdis sits on top of the existing `foresight` research engine:

```
+----------------------------------------+
|              no_free_signal (this layer)       |
|  World facade  +  serializers  +  MCP  |
+----------------------------------------+
|              foresight (engine)        |
|  multi-agent env  +  evolution  +      |
|  (eventually) per-agent world models   |
+----------------------------------------+
```

`foresight/` is the research engine — implementation detail. `no_free_signal/` is the deployable AI-facing layer. They will diverge cleanly: research lives in `foresight/`, anything an external user / LLM touches lives in `no_free_signal/`.

## License

(Pick one — MIT and Apache-2.0 are both fine for an open-source AI substrate.)

## Citations / references

The architecture draws on:

- Ha & Schmidhuber 2018, *World Models*
- Hafner et al. 2021/2023, Dreamer V2 / V3
- Panksepp 2004, *Affective Neuroscience*
- Damasio 1996, somatic-marker hypothesis
- Cannon 1932, homeostatic regulation
- MacArthur & Wilson 1967, r/K selection
- Friston 2010, free-energy principle

See `notes/biology.md` and `notes/ml-architectures.md` in the repo for the full reading.
