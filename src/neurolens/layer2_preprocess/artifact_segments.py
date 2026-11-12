"""Time-resolved artifact segmentation (increment 61, docs/preregistration_increment61.md).

One-second windows per EEG channel; a window is flagged by any of six types. Thresholds are fixed in the
pre-registration (physiological and record-relative, not fitted to any real data set):

* ``muscle``   P(20..min(40, 0.45 fs) Hz) / P(1..20 Hz) > 0.25 and P_hf > 3 x record median
* ``high_amp`` std(0.5-35 Hz) > max(5 x record median, 100 uV)
* ``drift``    peak-to-peak of the < 0.5 Hz component in a 4-s window around the second > max(6 x record median, 100 uV)
* ``pop``      max |x[t+k] - x[t]|, k = ceil(0.02 fs), > max(15 x record median, 150 uV)
* ``flat_sat`` >= 0.2 s of identical consecutive samples
* ``mains``    P(f0 +- 1 Hz) / P(0.5..0.45 fs) > 0.4 for f0 in {50, 60} Hz when fs > 2 f0 + 10

Flags are evidence only: nothing is removed from the signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

TYPES = ("muscle", "high_amp", "drift", "pop", "flat_sat", "mains")


@dataclass
class ArtifactSegments:
    """Per-channel boolean masks per one-second window and their union."""

    window_s: float
    n_windows: int
    channels: list[str]
    masks: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)   # channel -> type -> bool[n_windows]

    def union(self, channel: str) -> np.ndarray:
        m = self.masks.get(channel, {})
        return np.logical_or.reduce(list(m.values())) if m else np.zeros(self.n_windows, bool)

    def fraction(self, channel: str, kind: str | None = None) -> float:
        m = self.union(channel) if kind is None else self.masks.get(channel, {}).get(kind, np.zeros(self.n_windows, bool))
        return float(m.mean()) if m.size else 0.0


def _windows(x: np.ndarray, fs: float, n: int) -> np.ndarray:
    w = int(round(fs))
    return x[: n * w].reshape(n, w)


def _bandpower(win: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    f, p = welch(win, fs=fs, nperseg=win.shape[1], axis=1)
    sel = (f >= lo) & (f <= hi)
    if not sel.any():
        return np.zeros(win.shape[0])
    return np.trapezoid(p[:, sel], f[sel], axis=1)


def _safe_sos(fs: float, lo: float | None, hi: float | None, order: int = 3):
    nyq = fs / 2.0
    if lo is not None and hi is not None:
        hi = min(hi, 0.95 * nyq)
        return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    if hi is not None:
        return butter(order, min(hi, 0.95 * nyq) / nyq, btype="low", output="sos")
    return butter(order, lo / nyq, btype="high", output="sos")


def segment_channel(x: np.ndarray, fs: float, mains: tuple[float, ...] = (50.0, 60.0)) -> dict[str, np.ndarray]:
    """Masks for one channel (µV). Returns {type: bool[n_windows]}; all types always present."""
    x = np.asarray(x, dtype=np.float64)
    n = int(len(x) // round(fs))
    out = {t: np.zeros(max(n, 0), bool) for t in TYPES}
    if n < 1:
        return out
    w = int(round(fs))
    raw = _windows(x, fs, n)

    # muscle: relative high-frequency power (band shortened for low sampling rates)
    hi_edge = min(40.0, 0.45 * fs)
    if hi_edge - 20.0 >= 3.0:
        p_hf = _bandpower(raw, fs, 20.0, hi_edge)
        p_lf = _bandpower(raw, fs, 1.0, 20.0)
        ratio = p_hf / np.maximum(p_lf, 1e-12)
        out["muscle"] = (ratio > 0.25) & (p_hf > 3.0 * np.median(p_hf))

    # high amplitude: std of the 0.5-35 Hz signal
    if fs > 2 * 36:
        band = sosfiltfilt(_safe_sos(fs, 0.5, 35.0), x)
    else:
        band = sosfiltfilt(_safe_sos(fs, 0.5, 0.9 * fs / 2.0), x)
    std = _windows(band, fs, n).std(axis=1)
    out["high_amp"] = std > np.maximum(5.0 * np.median(std), 100.0)

    # drift: peak-to-peak of the < 0.5 Hz component over a centred 4-s window
    slow = sosfiltfilt(_safe_sos(fs, None, 0.5, order=2), x)
    sw = _windows(slow, fs, n)
    p2p = np.empty(n)
    for i in range(n):
        a, b = max(0, i - 1), min(n, i + 3)
        seg = sw[a:b]
        p2p[i] = seg.max() - seg.min()
    out["drift"] = p2p > np.maximum(6.0 * np.median(p2p), 100.0)

    # electrode pop: largest 20-ms jump
    k = max(1, int(np.ceil(0.02 * fs)))
    d = np.abs(x[k:] - x[:-k])
    d = np.concatenate([d, np.zeros(k)])
    jump = _windows(d, fs, n).max(axis=1)
    out["pop"] = jump > np.maximum(15.0 * np.median(jump), 150.0)

    # flat line / saturation: >= 0.2 s of identical consecutive samples
    need = max(2, int(np.ceil(0.2 * fs)))
    same = np.concatenate([[False], np.diff(x) == 0.0])
    run = np.zeros(len(x), np.int32)
    c = 0
    for i, s in enumerate(same):
        c = c + 1 if s else 0
        run[i] = c
    out["flat_sat"] = _windows(run, fs, n).max(axis=1) >= need - 1

    # mains interference
    tot = _bandpower(raw, fs, 0.5, 0.45 * fs)
    m = np.zeros(n, bool)
    for f0 in mains:
        if fs > 2 * f0 + 10:
            m |= (_bandpower(raw, fs, f0 - 1.0, f0 + 1.0) / np.maximum(tot, 1e-12)) > 0.4
    out["mains"] = m
    return out


def segment_artifacts(sig, mains: tuple[float, ...] = (50.0, 60.0)) -> ArtifactSegments:
    """Segment every EEG channel of a UnifiedSignal (values in µV)."""
    from ..layer1_ingest.electrodes import is_eeg_channel

    fs = float(sig.sampling_rate_hz)
    names = list(sig.channel_names)
    keep = [i for i, nme in enumerate(names) if is_eeg_channel(nme)]
    n = int(sig.signal.shape[1] // round(fs))
    seg = ArtifactSegments(window_s=1.0, n_windows=n, channels=[names[i] for i in keep])
    for i in keep:
        seg.masks[names[i]] = segment_channel(sig.signal[i], fs, mains)
    return seg
