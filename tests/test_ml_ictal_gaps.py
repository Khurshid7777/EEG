"""Increment 53: signal gaps (zeros after NaN filling, disconnections) must not pull the ictal baseline down."""

import numpy as np

from neurolens.layer4_detect.ml_ictal import EMA_TAU_EPOCHS, causal_baseline, ema_next


def _plain(x, init):
    a = 1.0 / EMA_TAU_EPOCHS
    out, y = np.empty_like(x), init.copy()
    for t in range(len(x)):
        out[t] = y
        y = (1 - a) * y + a * x[t]
    return out


def test_baseline_unchanged_without_gaps():
    x = np.random.default_rng(0).normal(3.0, 0.3, (500, 4))
    assert np.allclose(causal_baseline(x), _plain(x, np.median(x[:30], axis=0)))


def test_gap_epochs_hold_the_baseline():
    x = np.random.default_rng(1).normal(3.0, 0.3, (400, 2))
    g = x.copy()
    g[200:210, 0] = np.log(1e-6)            # 10-s gap filled with zeros on channel 0
    b, bg = causal_baseline(x), causal_baseline(g)
    assert np.allclose(bg[:, 1], b[:, 1])    # the other channel is untouched
    assert np.allclose(bg[200:211, 0], bg[200, 0])  # held through the gap
    # afterwards it tracks the same signal from the same level: the difference to the gap-free run is tiny
    assert np.abs(bg[211:, 0] - b[211:, 0]).max() < 0.05
    # streaming state: a gap epoch at the end of a chunk does not move the EMA
    assert np.allclose(ema_next(np.array([3.0]), np.array([np.log(1e-6)])), [3.0])


def test_filters_keep_gaps_zero_and_do_not_smear_them():
    from neurolens.contracts.config import FilterConfig
    from neurolens.contracts.signal import UnifiedSignal
    from neurolens.layer2_preprocess.filters import apply_filters

    fs = 200.0
    x = np.random.default_rng(2).normal(0, 20, (2, int(120 * fs)))
    g = x.copy()
    g[0, int(60 * fs):int(70 * fs)] = np.nan
    mk = lambda d: UnifiedSignal(signal=d.astype(np.float32), sampling_rate_hz=fs, channel_names=["C3", "C4"])
    cfg = FilterConfig()
    fx, fg = apply_filters(mk(x), cfg).signal, apply_filters(mk(g), cfg).signal
    assert np.allclose(fg[1], fx[1], atol=1e-3)                    # untouched channel: as before
    assert np.all(fg[0, int(60 * fs):int(70 * fs)] == 0)           # the gap stays empty
    # no step smeared into the signal a few seconds before the gap
    seg = slice(int(50 * fs), int(57 * fs))
    assert np.abs(fg[0, seg] - fx[0, seg]).max() < 0.05 * np.abs(fx[0, seg]).max()
