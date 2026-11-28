"""Regional (lobe / hemisphere) scalp EEG summaries (increment 73, docs/preregistration_increment73.md).

Scalp mapping from the 19 electrodes of the 10-20 system, **not** source localisation. Reference: common average
over the available 10-20 electrodes (so the result does not depend on the instrument's reference). Windowed Welch
power, windows with peak-to-peak > 200 uV rejected. Region definitions and metrics are fixed by the pre-registration.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
MAX_PTP_UV_SPARSE = 500.0
MIN_GOOD_CHANNELS, MIN_GOOD_WINDOWS = 12, 3
STANDARD_1020 = tuple(sorted({c for r in REGIONS.values() for c in r}))
_ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}


def canonical(ch: str) -> str:
    """10-10 names of the temporal electrodes -> the 10-20 names used here (T7->T3, T8->T4, P7->T5, P8->T6)."""
    return _ALIAS.get(ch, ch)


def usable_channels(signal: np.ndarray, channels: Sequence[str]) -> list[int]:
    """Indices of standard electrodes that are neither flat nor mostly missing (NaN/inf share <= 20%)."""
    sig = np.asarray(signal)
    out, seen = [], set()
    for i, c in enumerate(channels):
        c = canonical(c)
        if c not in STANDARD_1020 or c in seen:
            continue
        x = sig[i]
        bad = float(np.mean(~np.isfinite(x)))
        if bad > 0.2 or np.nanstd(np.where(np.isfinite(x), x, np.nan)) <= 1e-9:
            continue
        out.append(i)
        seen.add(c)
    return out


def _windows(x: np.ndarray, fs: float) -> np.ndarray:
    n = int(WINDOW_S * fs)
    step = n // 2
    starts = range(0, x.shape[1] - n + 1, step)
    return np.stack([x[:, s:s + n] for s in starts], axis=0) if len(starts) else np.empty((0, x.shape[0], n))


def channel_band_power(signal: np.ndarray, channels: Sequence[str], fs: float,
                       highpass_hz: float | None = None, robust: bool = True) -> dict[str, dict[str, float]]:
    """Per-channel {band: relative power, "abs_alpha": absolute alpha power} after common-average referencing.

    ``signal``: (channels, samples) in uV; only usable channels of the standard 10-20 set are used.
    ``robust`` (default since increment 82, D122): artefact rejection per channel and window - a channel is good in a window when its
    peak-to-peak is <= 200 uV; a window is usable when at least 12 channels are good; the average reference is taken over
    the good channels of the window; a channel's power is averaged over its good windows (needs >= 3). Without it
    (``robust=False``, the increment 73 rule) a window is dropped when any channel exceeds 200 uV."""
    idx = usable_channels(signal, channels)
    if len(idx) < 8:
        raise ValueError("fewer than 8 usable standard 10-20 electrodes")
    names = [canonical(channels[i]) for i in idx]
    x = np.asarray(signal, dtype=float)[idx]
    if highpass_hz:   # sensitivity option (not part of the pre-registered path): removes drift before the artefact rule
        from scipy.signal import butter, sosfiltfilt
        x = sosfiltfilt(butter(2, highpass_hz, btype="high", fs=fs, output="sos"), x, axis=1)
    if robust:
        return _robust_band_power(x, names, fs)
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




def _robust_band_power(x: np.ndarray, names: list[str], fs: float) -> dict[str, dict[str, float]]:
    n = int(WINDOW_S * fs)
    step = n // 2
    starts = list(range(0, x.shape[1] - n + 1, step))
    if not starts:
        raise ValueError("recording shorter than one window")
    w = np.stack([x[:, s:s + n] for s in starts], axis=0)                 # (win, ch, samples)
    good = np.ptp(w, axis=2) <= MAX_PTP_UV                                # (win, ch)
    usable = good.sum(axis=1) >= min(MIN_GOOD_CHANNELS, len(names))
    if usable.sum() < MIN_GOOD_WINDOWS:
        raise ValueError("too few usable windows")
    w, good = w[usable], good[usable]
    ref = (w * good[:, :, None]).sum(axis=1, keepdims=True) / good.sum(axis=1)[:, None, None]
    w = w - ref
    f, p = welch(w, fs=fs, nperseg=n, axis=-1)                            # (win, ch, freq)
    tot_band = (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])
    enough = good.sum(axis=0) >= MIN_GOOD_WINDOWS
    if enough.sum() < min(MIN_GOOD_CHANNELS, len(names)):
        raise ValueError("too few usable channels")
    gw = good.astype(float)
    pm = (p * gw[:, :, None]).sum(axis=0) / np.maximum(gw.sum(axis=0), 1)[:, None]    # (ch, freq), mean over the channel's good windows
    tot = pm[:, tot_band].sum(axis=1) + 1e-12
    out: dict[str, dict[str, float]] = {}
    for i, name in enumerate(names):
        if not enough[i]:
            continue
        out[name] = {}
    for band, (lo, hi) in BANDS.items():
        bp = pm[:, (f >= lo) & (f < hi)].sum(axis=1)
        for i, name in enumerate(names):
            if enough[i]:
                out[name][band] = float(bp[i] / tot[i])
                if band == "alpha":
                    out[name]["abs_alpha"] = float(bp[i])
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


def fit_asymmetry_norms(ai_by_record: Sequence[Mapping[str, float]], lo: float = 2.5, hi: float = 97.5) -> dict[str, tuple[float, float]]:
    """Per-pair normative limits (empirical percentiles of AI over healthy training records; increment 74)."""
    pairs = {p for r in ai_by_record for p in r}
    return {p: (float(np.percentile([r[p] for r in ai_by_record if p in r], lo)),
                float(np.percentile([r[p] for r in ai_by_record if p in r], hi))) for p in sorted(pairs)}


def flag_asymmetry(ai: Mapping[str, float], norms: Mapping[str, tuple[float, float]]) -> dict[str, str]:
    """{pair: 'right>left' | 'left>right'} for pairs outside the normative limits."""
    out: dict[str, str] = {}
    for pair, v in ai.items():
        if pair in norms:
            lo, hi = norms[pair]
            if v > hi:
                out[pair] = "right>left"
            elif v < lo:
                out[pair] = "left>right"
    return out


# ---------------------------------------------------------------------------
# windowed series for the interactive view (increment 76)
# ---------------------------------------------------------------------------

SERIES_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (12.0, 30.0), "sigma": (12.0, 15.0)}
LEFT_OF = {"Fp1": "Fp2", "F3": "F4", "F7": "F8", "C3": "C4", "T3": "T4", "T5": "T6", "P3": "P4", "O1": "O2"}


def regional_series(signal: np.ndarray, channels: Sequence[str], fs: float, win_s: float = 10.0,
                    max_windows: int = 720, max_sample_s: float = 30.0) -> dict:
    """Non-overlapping windows over the whole recording (window lengthened so that at most ``max_windows`` remain).

    Returns relative band power per channel and window (delta/theta/alpha/beta, share of 1-30 Hz), log10 absolute alpha
    power, and an artefact flag per window (fewer than 12 good channels; a channel is good when its peak-to-peak is <= 200 uV, 500 uV for sparse PSG montages; increment 82). Common-average reference
    over the standard 10-20 electrodes when at least 8 are present (otherwise the recorded reference is kept, e.g. PSG with mastoid references), computed per window (memory stays small for 24-hour recordings). When a window is
    longer than ``max_sample_s`` seconds only its first ``max_sample_s`` seconds are analysed (a representative sample).
    Scalp mapping only (no source localisation)."""
    idx = usable_channels(signal, channels)
    if len(idx) < 3:
        raise ValueError("fewer than 3 usable standard 10-20 electrodes")
    car = len(idx) >= 8            # sparse (PSG-type) montages keep the recorded reference: no common average
    names = [canonical(channels[i]) for i in idx]
    sig = np.asarray(signal)
    total_s = sig.shape[1] / fs
    win_s = max(4.0, float(win_s), float(np.ceil(total_s / max_windows)))
    n = int(win_s * fs)
    n_win = sig.shape[1] // n
    if n_win < 1:
        raise ValueError("recording shorter than one window")
    m = int(min(win_s, max_sample_s) * fs)
    max_ptp = MAX_PTP_UV if car else MAX_PTP_UV_SPARSE          # sleep PSG montages: slow waves reach 200-300 uV (increments 79-80 used 500)
    need = min(MIN_GOOD_CHANNELS, len(idx))
    rel = {b: [] for b in SERIES_BANDS}
    lab = {b: [] for b in SERIES_BANDS}
    abs_alpha, artefact, bad = [], [], []
    for w in range(n_win):
        x = sig[idx, w * n:w * n + m].astype(float)
        nan_win = not np.isfinite(x).all()
        if nan_win:
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        good = np.ptp(x, axis=1) <= max_ptp                       # increment 82: rejection per channel and window
        if not good.any():
            good = np.ones(len(idx), bool)
        if car:
            x = x - x[good].mean(axis=0, keepdims=True)
        f, p = welch(x, fs=fs, nperseg=int(min(m, 4 * fs)), axis=-1)               # (ch, freq)
        tot = p[:, (f >= TOTAL_BAND[0]) & (f < TOTAL_BAND[1])].sum(axis=1) + 1e-12
        vals = {}
        for b, (lo, hi) in SERIES_BANDS.items():
            bp = p[:, (f >= lo) & (f < hi)].sum(axis=1)
            vals[b] = (bp / tot, np.log10(bp + 1e-12))
        for b in SERIES_BANDS:
            r, l = vals[b]
            r, l = r.copy(), l.copy()
            if (~good).any() and good.any():                       # neutral fill for the display: median of the window's good channels
                r[~good], l[~good] = np.median(r[good]), np.median(l[good])
            rel[b].append(r)
            lab[b].append(l)
        a = vals["alpha"][1].copy()
        if (~good).any():
            a[~good] = np.median(a[good])
        abs_alpha.append(a)
        artefact.append(bool(nan_win or good.sum() < need))
        bad.append(~good)
    bad_by_channel = {names[c]: [w for w in range(n_win) if bad[w][c]] for c in range(len(idx))}
    return {"channels": names, "reference": "car" if car else "recorded", "win_s": win_s, "n_windows": int(n_win), "duration_s": float(total_s),
            "rel": {b: np.array(v).T.round(4).tolist() for b, v in rel.items()},
            "log10_abs": {b: np.array(v).T.round(3).tolist() for b, v in lab.items()},
            "log10_abs_alpha": np.array(abs_alpha).T.round(3).tolist(), "artefact": artefact,
            "bad_windows_by_channel": {c: v for c, v in bad_by_channel.items() if v}}


# ---------------------------------------------------------------------------
# electrode-label plausibility (increment 84)
# ---------------------------------------------------------------------------

POSTERIOR = ("O1", "O2", "P3", "P4", "Pz")
ANTERIOR = ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "T3", "T4")


def label_plausibility(per_channel: Mapping[str, Mapping[str, float]]) -> dict:
    """Rule of increment 84 (docs/preregistration_increment84.md) on per-channel relative alpha.

    ``suspect`` when (a) none of the three channels with the highest relative alpha is occipital/parietal and (b) the mean
    posterior alpha is lower than the mean anterior (frontal + temporal) alpha. Valid only for an awake, eyes-closed
    recording; fewer than 8 channels -> ``available`` False."""
    if len(per_channel) < 8:
        return {"available": False, "suspect": None, "reason": "fewer than 8 usable electrodes"}
    alpha = {c: v["alpha"] for c, v in per_channel.items()}
    top3 = sorted(alpha, key=alpha.get, reverse=True)[:3]
    post = [alpha[c] for c in POSTERIOR if c in alpha]
    ant = [alpha[c] for c in ANTERIOR if c in alpha]
    if not post or not ant:
        return {"available": False, "suspect": None, "reason": "no posterior or anterior electrodes"}
    a = not any(c in POSTERIOR for c in top3)
    b = float(np.mean(post)) < float(np.mean(ant))
    return {"available": True, "suspect": bool(a and b), "top3": top3, "posterior_alpha": round(float(np.mean(post)), 4),
            "anterior_alpha": round(float(np.mean(ant)), 4)}


OCCIPITAL = ("O1", "O2")
FRONTAL = ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz")


def label_plausibility_v2(per_channel: Mapping[str, Mapping[str, float]]) -> dict:
    """Version 2 (increment 86, docs/preregistration_increment86.md): ``suspect`` when the mean relative alpha of the
    occipital electrodes is lower than that of the frontal electrodes. Valid for an awake, eyes-closed recording only."""
    if len(per_channel) < 8:
        return {"available": False, "suspect": None, "reason": "fewer than 8 usable electrodes"}
    occ = [per_channel[c]["alpha"] for c in OCCIPITAL if c in per_channel]
    fro = [per_channel[c]["alpha"] for c in FRONTAL if c in per_channel]
    if not occ or not fro:
        return {"available": False, "suspect": None, "reason": "no occipital or frontal electrodes"}
    o, f = float(np.mean(occ)), float(np.mean(fro))
    return {"available": True, "suspect": bool(o < f), "occipital_alpha": round(o, 4), "frontal_alpha": round(f, 4)}


def montage_sanity(signal: np.ndarray, channels: Sequence[str], fs: float, version: int = 2) -> dict:
    """Electrode-label plausibility of an awake eyes-closed recording (``version`` 2 = increment 86, 1 = increment 84)."""
    try:
        pc = channel_band_power(signal, channels, fs, robust=True)
    except ValueError as exc:
        return {"available": False, "suspect": None, "reason": str(exc)}
    return label_plausibility_v2(pc) if version == 2 else label_plausibility(pc)
