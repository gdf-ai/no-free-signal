"""Lamdis — a biologically-grounded sensory substrate for AI agents.

Three grounding levels:
  - Symbolic / JSON      : World.observe() / observe_creature()
  - Perceptual / image   : World.render()
  - Sensory / raw        : World.obs_for(), World.attach_controller()

Plus per-individual neural brains and the experiment harness in
``no_free_signal.experiments``.
"""
from foresight.evolution.genome import Genome

from no_free_signal.brains import CreatureBrain, CreatureBrainManager
from no_free_signal.controller import ControllerHandle, ControllerRegistry
from no_free_signal.observation import RawObservation, render_world_array, render_world_png
from no_free_signal.world import World

__version__ = "0.3.0"

__all__ = [
    "World",
    "Genome",
    "RawObservation",
    "ControllerHandle",
    "ControllerRegistry",
    "CreatureBrain",
    "CreatureBrainManager",
    "render_world_array",
    "render_world_png",
    "__version__",
]
