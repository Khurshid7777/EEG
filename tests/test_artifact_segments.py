"""Increment 61: synthetic checks of the time-resolved artifact segmentation (thresholds fixed in the pre-registration)."""

import numpy as np

from neurolens.layer2_preprocess.artifact_segments import TYPES, segment_channel

FS = 200.0


def _eeg(seconds=60, seed=0, fs=FS):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    return 20 * np.sin(2 * np.pi * 10 * t) + 8 * rng.standard_normal(t.size)   # ~alpha, ~20 uV


def test_clean_eeg_is_not_flagged():
    m = segment_channel(_eeg(), FS)
    assert set(m) == set(TYPES)
    assert all(v.mean() < 0.05 for v in m.values()), {k: v.mean() for k, v in m.items()}


def test_muscle_burst_is_flagged_in_its_seconds():
    x = _eeg(60)
    rng = np.random.default_rng(1)
    s, e = int(20 * FS), int(25 * FS)
    x[s:e] += 60 * rng.standard_normal(e - s) * np.sin(np.linspace(0, 3, e - s)) ** 2 + 30 * rng.standard_normal(e - s)
    m = segment_channel(x, FS)["muscle"]
    assert m[20:25].sum() >= 3 and m[:15].sum() == 0 and m[30:].sum() == 0


def test_electrode_pop_and_flat_line():
    x = _eeg(40)
    x[int(10 * FS)] += 3000.0                     # electrode pop: a jump far above the record's own 20-ms jumps
    x[int(25 * FS):int(26 * FS)] = 5.0            # one second flat
    m = segment_channel(x, FS)
    assert m["pop"][10] and not m["pop"][5]
    assert m["flat_sat"][25] and not m["flat_sat"][5]


def test_slow_drift_and_high_amplitude():
    x = _eeg(60)
    t = np.arange(x.size) / FS
    x += 400 * np.exp(-((t - 30) / 3.0) ** 2) * np.sin(2 * np.pi * 0.2 * t)    # slow wave artifact around 30 s
    m = segment_channel(x, FS)
    assert m["drift"][28:33].any() and not m["drift"][:15].any()
    y = _eeg(60)
    y[int(40 * FS):int(42 * FS)] *= 12                                          # sudden amplitude increase
    assert segment_channel(y, FS)["high_amp"][40:42].any()


def test_mains_and_low_sampling_rate_do_not_crash():
    x = _eeg(30)
    t = np.arange(x.size) / FS
    z = x + 80 * np.sin(2 * np.pi * 50 * t)
    assert segment_channel(z, FS)["mains"].mean() > 0.9
    low = segment_channel(_eeg(30, fs=50.0, seed=2), 50.0)
    assert all(len(v) == 30 for v in low.values()) and not low["mains"].any()
