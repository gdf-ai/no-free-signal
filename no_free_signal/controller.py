"""External-controller attachment — let outside code drive a creature directly.

A `controller` is any callable that takes a RawObservation and returns an int
in [0, NUM_ACTIONS). When attached, the World feeds the controller raw obs on
every step and uses the returned action for that creature, bypassing both the
heuristic and the brain. The controller detaches automatically when the
creature dies.

This is the **embodied** grounding surface. RL policies, hand-coded scripts,
and LLM-as-controller architectures all attach here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from no_free_signal.observation import RawObservation

ControllerFn = Callable[[RawObservation], int]


@dataclass
class ControllerHandle:
    creature_id: int
    _registry: "ControllerRegistry"
    _attached: bool = True

    def detach(self) -> None:
        if self._attached:
            self._registry._detach(self.creature_id)
            self._attached = False

    @property
    def attached(self) -> bool:
        return self._attached


class ControllerRegistry:
    """Maps creature_id -> ControllerFn. Cleared automatically as creatures die."""

    def __init__(self) -> None:
        self._controllers: dict[int, ControllerFn] = {}
        self._handles: dict[int, ControllerHandle] = {}

    def attach(self, creature_id: int, fn: ControllerFn) -> ControllerHandle:
        if creature_id in self._controllers:
            self._detach(creature_id)
        self._controllers[creature_id] = fn
        handle = ControllerHandle(creature_id=creature_id, _registry=self)
        self._handles[creature_id] = handle
        return handle

    def _detach(self, creature_id: int) -> None:
        fn = self._controllers.pop(creature_id, None)
        # If the controller exposes a close()/cleanup hook (e.g. LLM
        # controllers shutting down a background thread), call it.
        close = getattr(fn, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        h = self._handles.pop(creature_id, None)
        if h is not None:
            h._attached = False

    def get(self, creature_id: int) -> ControllerFn | None:
        return self._controllers.get(creature_id)

    def reap(self, alive_ids: set[int]) -> list[int]:
        """Detach controllers for creatures that no longer exist. Returns
        the ids that were reaped — useful for logging."""
        dead = [cid for cid in self._controllers if cid not in alive_ids]
        for cid in dead:
            self._detach(cid)
        return dead

    def __len__(self) -> int:
        return len(self._controllers)

    def __contains__(self, creature_id: int) -> bool:
        return creature_id in self._controllers
