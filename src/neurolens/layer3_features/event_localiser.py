"""Signal-based localisation of an event to electrode / region / hemisphere (increment 77,
docs/preregistration_increment77.md). The event time comes from outside (a detector or an expert); only the signal
decides *where*. Scalp level (19 electrodes of the 10-20 system), not source localisation.

Variants (fixed in the pre-registration): ``A`` peak |x|, ``B`` most negative peak, ``C`` second-derivative energy,
all after common-average reference and a 2-40 Hz band-pass, normalised by the electrode's robust sigma over the record."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .regional import REGIONS, STANDARD_1020


def region_of(ch: str) -> str | None:
    return next((r for r, chans in REGIONS.items() if ch in chans), None)


def side_of(ch: str) -> str:
    return "mid" if ch.endswith("z") else ("L" if int(ch[-1]) % 2 else "R")


class EventLocaliser:
    def __init__(self, signal: np.ndarray, channels: Sequence[str], fs: float):
        idx = [i for i, c in enumerate(channels) if c in STANDARD_1020]
        if len(idx) < 8:
            raise ValueError("fewer than 8 standard 10-20 electrodes")
        self.names = [channels[i] for i in idx]
        self.fs = fs
        x = np.asarray(signal)[idx].astype(float)
        x = x - x.mean(axis=0, keepdims=True)
        x = sosfiltfilt(butter(4, [2.0, 40.0], btype="band", fs=fs, output="sos"), x, axis=1)
        self.x = x
        self.sigma = 1.4826 * np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1) + 1e-9
        self.d2 = np.diff(x, n=2, axis=1)
        self.sigma_d2 = 1.4826 * np.median(np.abs(self.d2 - np.median(self.d2, axis=1, keepdims=True)), axis=1) + 1e-9

    def scores(self, t0: float, t1: float, variant: str = "A") -> np.ndarray:
        if t1 - t0 < 1e-6:                       # point event: +-0.15 s around the onset
            a, b = t0 - 0.15, t0 + 0.15
        else:
            a, b = t0 - 0.1, t1 + 0.1
        i0, i1 = max(0, int(a * self.fs)), min(self.x.shape[1], int(b * self.fs) + 1)
        if i1 - i0 < 3:
            raise ValueError("event outside the recording")
        seg = self.x[:, i0:i1]
        if variant == "A":
            return np.abs(seg).max(axis=1) / self.sigma
        if variant == "B":
            return -seg.min(axis=1) / self.sigma
        if variant == "C":
            d = self.d2[:, max(0, i0 - 1):max(i0 + 1, i1 - 1)]
            return (d ** 2).sum(axis=1) / self.sigma_d2 ** 2
        raise ValueError(variant)

    def locate(self, t0: float, t1: float, variant: str = "A", k: int = 3) -> dict:
        s = self.scores(t0, t1, variant)
        order = np.argsort(-s)
        top = [self.names[i] for i in order[:k]]
        return {"top": top, "scores": [round(float(s[i]), 2) for i in order[:k]],
                "region": region_of(top[0]), "side": side_of(top[0])}
