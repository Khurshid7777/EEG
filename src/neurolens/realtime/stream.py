"""Stream sources for real-time cEEG (TZ §4).

A :class:`StreamSource` yields :class:`Chunk` s of samples in stream order.
``EdfReplaySource`` replays a recording file chunk-by-chunk (optionally paced to
wall-clock) — the offline/deterministic source. The live source over Lab Streaming
Layer is :class:`neurolens.realtime.lsl.LslStreamSource` (D82).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..layer1_ingest.registry import ingest


@dataclass
class Chunk:
    """A block of consecutive samples starting at ``t_start`` seconds."""

    data: np.ndarray          # [channels][samples]
    t_start: float            # stream time of the first sample (seconds)


class StreamSource(ABC):
    channel_names: list[str]
    sampling_rate_hz: float
    reference: str = "unknown"

    @abstractmethod
    def chunks(self) -> Iterator[Chunk]:
        """Yield chunks of samples in stream order."""


class EdfReplaySource(StreamSource):
    """Replays an EDF as a stream of fixed-size chunks."""

    def __init__(self, path: str | Path, chunk_s: float = 5.0, realtime: bool = False):
        sig = ingest(path)
        # keep float32 (as ingested); the window buffer is float64 and the monitor
        # re-casts windows to float32, so values are identical at half the memory
        self._data = sig.signal
        self.channel_names = list(sig.channel_names)
        self.sampling_rate_hz = float(sig.sampling_rate_hz)
        self.reference = sig.reference
        self.chunk_samples = max(1, int(round(chunk_s * self.sampling_rate_hz)))
        self.realtime = realtime
        self.source_name = Path(path).name

    def chunks(self) -> Iterator[Chunk]:
        fs = self.sampling_rate_hz
        n = self._data.shape[1]
        for start in range(0, n, self.chunk_samples):
            stop = min(start + self.chunk_samples, n)
            if self.realtime:
                time.sleep((stop - start) / fs)  # pace to wall-clock
            yield Chunk(data=self._data[:, start:stop], t_start=start / fs)
