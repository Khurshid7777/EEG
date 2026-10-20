"""Background characterization: posterior dominant rhythm, continuity (TZ §6)."""

from __future__ import annotations

import numpy as np

from .peak_utils import dominant_frequency, spectral_peak
from .suppression import suppression_ratio


PDR_EPOCH_S = 10.0
PDR_MIN_EPOCHS = 6      # >= 1 min of epochs with a genuine posterior peak
PDR_ALERT_QUANTILE = 75


def posterior_dominant_rhythm(
    o1: np.ndarray, o2: np.ndarray, fs: float, method: str = "whole"
) -> float:
    """PDR = frequency of a genuine 4-13 Hz spectral peak over the occipital channels
    (>= 2x the 1/f background); 0.0 = not identified (see ``spectral_peak``).

    ``method="whole"``: the peak of the whole-record spectrum. ``method="alert"``
    (increment 20): the PDR is measured when it is best developed - in the most alert
    state, since drowsiness slows it (ACNS) - as the upper quartile of the genuine
    peaks of 10-s epochs; records with < 1 min of such epochs use the whole-record peak.
    (On routine EEG the whole-record spectrum reported 5.75 Hz for an adult whose
    epochs peaked at 9.5 Hz - drowsy theta outweighed the alpha peak.)"""
    occ = (np.asarray(o1, dtype=np.float64) + np.asarray(o2, dtype=np.float64)) / 2.0
    whole = spectral_peak(occ, fs, fmin=4.0, fmax=13.0)
    if method == "whole":
        return whole
    if method != "alert":
        raise ValueError(f"unknown PDR method {method!r}")
    n = int(PDR_EPOCH_S * fs)
    peaks = [spectral_peak(occ[k:k + n], fs, fmin=4.0, fmax=13.0) for k in range(0, occ.size - n + 1, n)]
    genuine = [p for p in peaks if p > 0]
    if len(genuine) < PDR_MIN_EPOCHS:
        return whole
    return float(np.percentile(genuine, PDR_ALERT_QUANTILE, method="nearest"))


def posterior_dominance(occ: np.ndarray, front: np.ndarray, fs: float, f0: float,
                        half_width_hz: float = 1.0) -> float:
    """Power of the rhythm (``f0`` +/- ``half_width_hz``) over the occipital average divided
    by that over the frontal average (increment 37). A posterior dominant rhythm is by
    definition posterior (ACNS 2021); a 4-13 Hz peak as strong or stronger frontally - alpha/
    theta coma, sedation - is not one."""
    from scipy.signal import welch

    nper = int(min(len(occ), max(64, fs * 4)))
    f, po = welch(np.asarray(occ, dtype=np.float64), fs=fs, nperseg=nper)
    _, pf = welch(np.asarray(front, dtype=np.float64), fs=fs, nperseg=nper)
    band = (f >= f0 - half_width_hz) & (f <= f0 + half_width_hz)
    den = float(pf[band].sum())
    return float(po[band].sum() / den) if den > 0 else float("inf")


def continuity_index(x: np.ndarray, fs: float, amp_thresh_uv: float = 10.0) -> float:
    """1 - suppression_ratio: fraction of time the trace is continuous (0..1)."""
    return float(1.0 - suppression_ratio(x, fs, amp_thresh_uv=amp_thresh_uv))
