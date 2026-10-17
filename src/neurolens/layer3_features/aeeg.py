"""Amplitude-integrated EEG (aEEG / CFM) — TZ §6, critical in neonatology/ICU.

Classic aEEG pipeline: asymmetric band-pass (~2-15 Hz) -> rectify -> smooth ->
semi-logarithmic amplitude compression -> per-epoch upper/lower margins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt

from .epochs import epoch_indices


@dataclass
class AEEG:
    times_s: np.ndarray      # epoch centers
    upper_uv: np.ndarray     # upper margin (compressed uV scale, for display)
    lower_uv: np.ndarray     # lower margin (compressed)
    channel: str
    upper_raw: np.ndarray = None  # raw (uncompressed) uV margins for classification
    lower_raw: np.ndarray = None

    def bandwidth(self) -> np.ndarray:
        return self.upper_uv - self.lower_uv

    def median_upper_raw(self) -> float:
        return float(np.median(self.upper_raw)) if self.upper_raw is not None and self.upper_raw.size else 0.0

    def median_lower_raw(self) -> float:
        return float(np.median(self.lower_raw)) if self.lower_raw is not None and self.lower_raw.size else 0.0


def _semilog_compress(uv: np.ndarray) -> np.ndarray:
    """aEEG display compression: linear 0-10 uV, logarithmic above 10 uV."""
    uv = np.asarray(uv, dtype=np.float64)
    out = np.where(uv <= 10.0, uv, 10.0 + 15.0 * np.log10(np.maximum(uv, 10.0) / 10.0))
    return out


def aeeg_envelope(
    x: np.ndarray, fs: float, channel: str = "", epoch_s: float = 15.0, segment_s: float = 1.0
) -> AEEG:
    """Margins from the peak-to-peak amplitude of the 2-15 Hz signal in ``segment_s``
    segments (10th / 90th percentile per epoch), the scale of the Hellström-Westas
    margins. Increment 29: percentiles of the rectified samples put the lower margin at
    zero crossings (~0.5 µV in every recording), so a continuous background was never
    classified as continuous."""
    nyq = fs / 2.0
    b, a = butter(2, [2.0 / nyq, min(15.0, nyq * 0.99) / nyq], btype="bandpass")
    filt = filtfilt(b, a, x)
    seg_n = max(1, int(round(segment_s * fs)))

    centers, uppers, lowers = [], [], []
    for s0, s1 in epoch_indices(len(filt), fs, epoch_s):
        n = (s1 - s0) // seg_n * seg_n
        if n == 0:
            continue
        seg = filt[s0:s0 + n].reshape(-1, seg_n)
        pp = seg.max(axis=1) - seg.min(axis=1)
        centers.append((s0 + s1) / 2.0 / fs)
        uppers.append(np.percentile(pp, 90))
        lowers.append(np.percentile(pp, 10))
    uppers = np.asarray(uppers)
    lowers = np.asarray(lowers)
    return AEEG(
        times_s=np.asarray(centers),
        upper_uv=_semilog_compress(uppers),
        lower_uv=_semilog_compress(lowers),
        channel=channel,
        upper_raw=uppers,
        lower_raw=lowers,
    )
