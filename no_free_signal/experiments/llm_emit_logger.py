"""Run-scoped buffer of LLM-emitted (post-transform) audio waves.

The harness wires a single :class:`EmitLog` per run into the World; the
World hands the same instance to every LLMController it spawns; controllers
append on every validated emission. Read paths produce a thread-safe
snapshot.

The buffer's mean emission vector is the operationalization of "what the
LLM is contributing into the substrate" for the convergence metric
``cos(mean_audio_attention, mean_llm_emission)``. For arm E (scrambled)
and arm F (context-randomized), the wave is logged *post-transform*, which
is the right thing to measure: we want to know whether the population
aligns with what actually entered the field, not what the LLM intended.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Iterable

import numpy as np


class EmitLog:
    def __init__(self, max_entries: int = 100_000):
        self._buf: deque[tuple[int, int, list[float], float]] = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._sum = np.zeros(8, dtype=np.float64)
        self._count = 0

    def __call__(self, tick: int, creature_id: int, wave: list[float], amp: float) -> None:
        if len(wave) != 8:
            return
        with self._lock:
            self._buf.append((int(tick), int(creature_id), list(wave), float(amp)))
            self._sum += np.asarray(wave, dtype=np.float64)
            self._count += 1

    def snapshot(self) -> list[tuple[int, int, list[float], float]]:
        with self._lock:
            return list(self._buf)

    def recent_waves(self, n: int = 200) -> list[list[float]]:
        with self._lock:
            return [w for _, _, w, _ in list(self._buf)[-n:]]

    def mean_emission(self) -> np.ndarray:
        with self._lock:
            if self._count == 0:
                return np.zeros(8, dtype=np.float64)
            return (self._sum / self._count).copy()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def total_calls(self) -> int:
        with self._lock:
            return self._count
