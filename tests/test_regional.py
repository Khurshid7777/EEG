import numpy as np

from neurolens.layer3_features.regional import (STANDARD_1020, asymmetry_index, channel_band_power, posterior_alpha,
                                                region_values)


def _synthetic(fs=200.0, secs=60, strong=("O1", "O2"), left_boost=1.0):
    rng = np.random.default_rng(0)
    t = np.arange(int(fs * secs)) / fs
    chans = list(STANDARD_1020)
    x = np.array([5 * rng.standard_normal(t.size) for _ in chans])
    for c in strong:
        x[chans.index(c)] += 20 * np.sin(2 * np.pi * 10 * t) * (left_boost if c.endswith("1") else 1.0)
    return x, chans, fs


def test_occipital_alpha_is_the_maximum_region():
    x, ch, fs = _synthetic()
    pc = channel_band_power(x, ch, fs)
    rv = region_values(pc, "alpha")
    assert max(rv, key=rv.get) == "occipital"
    assert posterior_alpha(pc) > np.mean([v for k, v in rv.items() if k in ("frontal", "central")])


def test_asymmetry_sign_and_reference_independence():
    x, ch, fs = _synthetic(left_boost=2.0)                 # O1 (left) has 4x the alpha power
    ai = asymmetry_index(channel_band_power(x, ch, fs), "O1", "O2")
    assert ai < -0.5                                       # left stronger -> negative
    shifted = x + 100 * np.sin(2 * np.pi * 10 * np.arange(x.shape[1]) / fs)   # common signal on every channel
    ai2 = asymmetry_index(channel_band_power(shifted, ch, fs), "O1", "O2")
    assert abs(ai2 - ai) < 0.2                             # common-average reference removes the shared component


def test_artifact_windows_are_rejected_and_short_input_fails():
    x, ch, fs = _synthetic(secs=40)
    x[:, 1000:1100] += 400                                 # one huge artefact burst
    assert "alpha" in channel_band_power(x, ch, fs)["O1"]
    import pytest
    with pytest.raises(ValueError):
        channel_band_power(x[:, :300], ch, fs)


def test_asymmetry_norms_flag_only_outside_limits():
    from neurolens.layer3_features.regional import fit_asymmetry_norms, flag_asymmetry

    rng = np.random.default_rng(1)
    train = [{"O1-O2": float(v)} for v in rng.normal(0, 0.3, 500)]
    norms = fit_asymmetry_norms(train)
    lo, hi = norms["O1-O2"]
    assert -0.7 < lo < -0.5 and 0.5 < hi < 0.7
    assert flag_asymmetry({"O1-O2": 0.0}, norms) == {}
    assert flag_asymmetry({"O1-O2": 1.5}, norms) == {"O1-O2": "right>left"}
    assert flag_asymmetry({"O1-O2": -1.5}, norms) == {"O1-O2": "left>right"}
