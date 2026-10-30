"""Filtering: highpass, lowpass, and mains notch (TZ §5).

Zero-phase (filtfilt) IIR filters on numpy arrays. Mains notch defaults to
50 Hz (Uzbekistan grid), switchable to 60 Hz via config.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from ..contracts.config import FilterConfig
from ..contracts.signal import UnifiedSignal


def _sos_filtfilt_bandpass(
    data: np.ndarray, fs: float, hp: float, lp: float, order: int
) -> np.ndarray:
    nyq = fs / 2.0
    lp_eff = min(lp, nyq * 0.99)
    b, a = butter(order, [hp / nyq, lp_eff / nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=-1)


def _filter_segments(raw: np.ndarray, finite: np.ndarray, run, min_len: int) -> np.ndarray:
    """Signal gaps (NaN): each channel's finite stretches are filtered separately and the gaps stay
    zero (increment 53). Filtering through a zero-filled gap smeared a step over the neighbouring
    seconds, which shifted the detectors' running baselines. Stretches shorter than ``min_len`` are
    zeroed (too short to filter)."""
    out = np.zeros_like(raw)
    for c in range(raw.shape[0]):
        m = np.concatenate([[False], finite[c], [False]])
        edges = np.flatnonzero(m[1:] != m[:-1])
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= min_len:
                out[c, a:b] = run(raw[c:c + 1, a:b])[0]
    return out


def apply_filters(sig: UnifiedSignal, cfg: FilterConfig) -> UnifiedSignal:
    """Return a copy of ``sig`` with band-pass + optional notch applied.

    Non-finite samples are zeroed before filtering to keep filtfilt stable.
    """
    fs = sig.sampling_rate_hz
    raw = sig.signal.astype(np.float64)

    def run(x: np.ndarray) -> np.ndarray:
        x = _sos_filtfilt_bandpass(x, fs, cfg.highpass_hz, cfg.lowpass_hz, cfg.filter_order)
        if cfg.notch_enabled and cfg.notch_hz < fs / 2.0:
            b, a = iirnotch(cfg.notch_hz / (fs / 2.0), Q=30.0)
            x = filtfilt(b, a, x, axis=-1)
        return x

    finite = np.isfinite(raw)
    if finite.all():
        data = run(raw)
    else:
        data = _filter_segments(raw, finite, run, min_len=int(2 * fs))

    flags = list(sig.quality.flags)
    flags.append(
        f"filtered:hp={cfg.highpass_hz}Hz,lp={cfg.lowpass_hz}Hz,"
        f"notch={cfg.notch_hz if cfg.notch_enabled else 'off'}Hz"
    )
    new = sig.with_signal(data.astype(np.float32))
    new.quality.flags = flags
    return new
