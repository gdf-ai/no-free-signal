"""Audio wave field for inter-creature communication.

Pure NumPy. The world holds one ``AudioField``; creatures emit waves into it
when they take the VOCALIZE action and sample it through their per-tile
observation. Self-hearing falls out automatically: the emitted wave is written
to the field on the same tick, and the speaker samples from the same field on
the next tick (post-decay), so utterances enter the speaker's own observation
just like any other creature's.
"""
from __future__ import annotations

import math

import numpy as np

NUM_AUDIO_BINS = 8
RING_DECAY = 0.5
ATTENUATION_RADIUS = 6
ATTENUATION_SCALE = 2.5
MAX_PER_BIN = 4.0
PRUNE_THRESHOLD = 0.01


class AudioField:
    """Sparse per-tile accumulator of 8-bin frequency vectors.

    Emit-side: ``emit(pos, wave_vec, amplitude)`` stamps an attenuated copy of
    ``wave_vec * amplitude`` onto every tile within Chebyshev distance
    ``ATTENUATION_RADIUS``. Read-side: ``sample(pos)`` returns the current
    8-vector at that tile (zeros if absent). ``tick()`` decays the whole
    field by ``RING_DECAY`` and prunes entries that have rung out.
    """

    def __init__(self) -> None:
        self._tiles: dict[tuple[int, int], np.ndarray] = {}

    def emit(
        self,
        pos: tuple[int, int],
        wave_vec: np.ndarray,
        amplitude: float,
        attenuation_radius: int = ATTENUATION_RADIUS,
    ) -> None:
        if amplitude <= 0.0:
            return
        wave_vec = np.asarray(wave_vec, dtype=np.float32)
        if wave_vec.shape != (NUM_AUDIO_BINS,):
            raise ValueError(
                f"wave_vec shape must be ({NUM_AUDIO_BINS},), got {wave_vec.shape}"
            )
        r0, c0 = pos
        for dr in range(-attenuation_radius, attenuation_radius + 1):
            for dc in range(-attenuation_radius, attenuation_radius + 1):
                d = max(abs(dr), abs(dc))
                if d > attenuation_radius:
                    continue
                falloff = math.exp(-d / ATTENUATION_SCALE)
                contrib = wave_vec * (amplitude * falloff)
                key = (r0 + dr, c0 + dc)
                cur = self._tiles.get(key)
                if cur is None:
                    self._tiles[key] = contrib.copy()
                else:
                    cur += contrib
                np.minimum(self._tiles[key], MAX_PER_BIN, out=self._tiles[key])

    def sample(self, pos: tuple[int, int]) -> np.ndarray:
        cur = self._tiles.get(tuple(pos))
        if cur is None:
            return np.zeros(NUM_AUDIO_BINS, dtype=np.float32)
        return cur.copy()

    def tick(self) -> None:
        if not self._tiles:
            return
        dead: list[tuple[int, int]] = []
        for key, vec in self._tiles.items():
            vec *= RING_DECAY
            if float(vec.max()) < PRUNE_THRESHOLD:
                dead.append(key)
        for key in dead:
            del self._tiles[key]

    def clear(self) -> None:
        self._tiles.clear()

    def __len__(self) -> int:
        return len(self._tiles)


def _smoke() -> None:
    field = AudioField()
    wave = np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    field.emit((10, 10), wave, amplitude=1.0)
    here = field.sample((10, 10))
    far = field.sample((10, 16))
    edge = field.sample((10, 17))
    print(f"distance 0:  bin0={here[0]:.3f} bin2={here[2]:.3f}")
    print(f"distance 6:  bin0={far[0]:.3f}  expected~{math.exp(-6/2.5):.3f}")
    print(f"distance 7:  bin0={edge[0]:.3f}  expected=0.0")
    field.tick()
    decayed = field.sample((10, 10))
    print(f"after tick:  bin0={decayed[0]:.3f}  expected~0.5")
    print(f"tiles:       {len(field)}")


if __name__ == "__main__":
    _smoke()
