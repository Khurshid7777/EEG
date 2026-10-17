"""Epoch rhythmicity for neonatal seizure detection (increment 27).

Neonatal seizures are sustained repetitive discharges. For each epoch and channel: the
maximum autocorrelation of the 0.5-8 Hz signal at lags 0.15-2 s over the WIN_S seconds
ending at the epoch's end (causal: no later samples). Aggregated over channels:
maximum, mean of the top three, fraction of channels >= 0.6, and the period of the most
rhythmic channel. Epochs without WIN_S seconds of history in the signal get NaN (the
trees route missing values; in the monitor these are the first seconds of a window).
"""

from __future__ import annotations

import numpy as np

WIN_S = 8.0
FS_LOW = 64.0
LAG_MIN_S, LAG_MAX_S = 0.15, 2.0
RHYTHM_FIELDS = ("rhy_max", "rhy_top3", "rhy_frac_ge_0.6", "rhy_period_s")


def epoch_rhythm(data: np.ndarray, fs: float, epoch_times: np.ndarray, epoch_s: float = 2.0) -> np.ndarray:
    """``data``: [channels, samples] EEG (already re-referenced). Returns [n_epochs, 4]."""
    from scipy.signal import butter, resample_poly, sosfiltfilt

    out = np.full((len(epoch_times), len(RHYTHM_FIELDS)), np.nan)
    if data.size == 0 or data.shape[1] < int(WIN_S * fs):
        return out
    x = sosfiltfilt(butter(4, [0.5, 8.0], "band", fs=fs, output="sos"), data.astype(np.float64), axis=1)
    up, down = int(FS_LOW), int(round(fs))
    g = np.gcd(up, down)
    x = resample_poly(x, up // g, down // g, axis=1)
    w = int(WIN_S * FS_LOW)
    lo, hi = int(LAG_MIN_S * FS_LOW), int(LAG_MAX_S * FS_LOW)
    for k, t in enumerate(np.asarray(epoch_times, float)):
        e = int(round((t + epoch_s / 2) * FS_LOW))
        s = e - w
        if s < 0 or e > x.shape[1]:
            continue
        seg = x[:, s:e] - x[:, s:e].mean(axis=1, keepdims=True)
        f = np.fft.rfft(seg, 2 * w, axis=1)
        ac = np.fft.irfft(f * np.conj(f), 2 * w, axis=1)[:, :hi + 1]
        ac = ac / np.maximum(ac[:, :1], 1e-12)
        r = ac[:, lo:hi + 1]
        rho = r.max(axis=1)
        top = np.sort(rho)[::-1]
        out[k] = [top[0], top[:3].mean(), (rho >= 0.6).mean(), (lo + r[int(np.argmax(rho))].argmax()) / FS_LOW]
    return out
