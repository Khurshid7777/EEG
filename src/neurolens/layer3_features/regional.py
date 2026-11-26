"""Regional (lobe / hemisphere) scalp EEG summaries (increment 73, docs/preregistration_increment73.md).

Scalp mapping from the 19 electrodes of the 10-20 system, **not** source localisation. Reference: common average
over the available 10-20 electrodes (so the result does not depend on the instrument's reference). Windowed Welch
power, windows with peak-to-peak > 200 uV rejected. Region definitions and metrics are fixed by the pre-registration.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.signal import welch

REGIONS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"),
    "central": ("C3", "C4", "Cz"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "parietal": ("P3", "P4", "Pz"),
    "occipital": ("O1", "O2"),
}
HOMOLOGOUS_PAIRS: tuple[tuple[str, str], ...] = (
    ("F3", "F4"), ("C3", "C4"), ("T3", "T4"), ("P3", "P4"), ("O1", "O2"), ("F7", "F8"), ("T5", "T6"),
)
BANDS = {"alpha": (8.0, 12.0), "theta": (4.0, 8.0)}
TOTAL_BAND = (1.0, 30.0)
WINDOW_S, MAX_PTP_UV = 4.0, 200.0
STANDARD_1020 = tuple(sorted({c for r in REGIONS.values() for c in r}))


def _windows(x: np.ndarray, fs: float) -> np.ndarray:
    n = int(WINDOW_S * fs)
    step = n // 2
    starts = range(0, x.shape[1] - n + 1, step)
    return np.stack([x[:, s:s + n] for s in starts], axis=0) if len(starts) else np.empty((0, x.shape[0], n))


def channel_band_power(signal: np.ndarray, channels: Sequence[str], fs: float) -> dict[str, dict[str, float]]:
    """Per-channel {band: relative power, "abs_alpha": absolute alpha power} after common-average referencing.

    ``signal``: (channels, samples) in uV; only channels in the standard 10-20 set are used."""
    idx = [i for i, c in enumerate(channels) if c in STANDARD_1020]
    if len(idx) < 8:
        raise ValueError("fewer than 8 standard 10-20 electrodes")
    names = [channels[i] for i in idx]
    x = np.asarray(signal, dtype=float)[idx]
    x = x - x.mean(axis=0, keepdims=True)
    w = _windows(x, fs)
    if len(w) == 0:
        raise ValueError("recording shorter than one window")
    keep = (np.ptp(w, axis=2).max(axis=1) <= MAX_PTP_UV)
    if keep.sum() < 3:
        raise ValueError("too few clean windows")
    w = w[keep]
    f, p = welch(w, fs=fs, nperseg=int(WINDOW_S * fs), axis=-1)       # (win, ch, freq)
    p = p.mean(axis=0)                                                  # (ch, freq)
    tot = p[:, (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])].sum(axis=1) + 1e-12
    out: dict[str, dict[str, float]] = {n: {} for n in names}
    for band, (lo, hi) in BANDS.items():
        bp = p[:, (f >= lo) & (f < hi)].sum(axis=1)
        for n, v, t in zip(names, bp, tot):
            out[n][band] = float(v / t)
            if band == "alpha":
                out[n]["abs_alpha"] = float(v)
    return out


def region_values(per_channel: Mapping[str, Mapping[str, float]], key: str) -> dict[str, float]:
    """Mean of ``key`` over the channels of each region (regions with no electrode are omitted)."""
    out: dict[str, float] = {}
    for region, chans in REGIONS.items():
        vals = [per_channel[c][key] for c in chans if c in per_channel]
        if vals:
            out[region] = float(np.mean(vals))
    return out


def asymmetry_index(per_channel: Mapping[str, Mapping[str, float]], left: str, right: str, key: str = "abs_alpha") -> float:
    """ln(P_right / P_left); positive = right hemisphere stronger."""
    return float(np.log(per_channel[right][key] / per_channel[left][key]))


def posterior_alpha(per_channel: Mapping[str, Mapping[str, float]]) -> float:
    """Mean relative alpha over occipital + parietal electrodes (pre-registered H2b measure)."""
    ch = [c for c in REGIONS["occipital"] + REGIONS["parietal"] if c in per_channel]
    return float(np.mean([per_channel[c]["alpha"] for c in ch]))
