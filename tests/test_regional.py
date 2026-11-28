import numpy as np

from neurolens.layer3_features.regional import (
    STANDARD_1020,
    asymmetry_index,
    channel_band_power,
    posterior_alpha,
    region_values,
)


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
    shifted = x + 50 * np.sin(2 * np.pi * 10 * np.arange(x.shape[1]) / fs)   # common signal on every channel
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


def test_regional_series_shapes_and_alpha_peak():
    from neurolens.layer3_features.regional import regional_series

    x, ch, fs = _synthetic(secs=60)
    s = regional_series(x, ch, fs, win_s=10)
    assert s["n_windows"] == 6 and s["win_s"] == 10 and len(s["channels"]) == 19
    i = s["channels"].index("O1")
    j = s["channels"].index("Fz")
    assert np.mean(s["rel"]["alpha"][i]) > 2 * np.mean(s["rel"]["alpha"][j])
    assert len(s["artefact"]) == 6 and not any(s["artefact"])
    # long recording: window lengthened to respect max_windows
    s2 = regional_series(np.tile(x, (1, 20)), ch, fs, win_s=10, max_windows=50)
    assert s2["n_windows"] <= 50 and s2["win_s"] > 10


def test_robust_rule_survives_one_bad_electrode():
    import pytest

    x, ch, fs = _synthetic(secs=60)
    x[ch.index("T4")] += 400 * np.sin(2 * np.pi * 0.5 * np.arange(x.shape[1]) / fs)     # one electrode with huge drift
    with pytest.raises(ValueError):
        channel_band_power(x, ch, fs, robust=False)                     # old rule: every window rejected
    pc = channel_band_power(x, ch, fs, robust=True)
    assert "T4" not in pc and len(pc) == len(ch) - 1
    rv = region_values(pc, "alpha")
    assert max(rv, key=rv.get) == "occipital"
    clean, ch2, fs2 = _synthetic(secs=60)
    a, b = channel_band_power(clean, ch2, fs2), channel_band_power(clean, ch2, fs2, robust=True)
    assert abs(a["O1"]["alpha"] - b["O1"]["alpha"]) < 0.02             # same result when nothing is bad


def test_label_plausibility_flags_swapped_posterior_and_anterior_labels():
    from neurolens.layer3_features.regional import label_plausibility, label_plausibility_v2, montage_sanity

    x, ch, fs = _synthetic(secs=60)
    swap0 = {"O1": "Fp1", "Fp1": "O1", "O2": "Fp2", "Fp2": "O2"}
    assert montage_sanity(x, ch, fs, version=2)["suspect"] is False
    assert montage_sanity(x, [swap0.get(c, c) for c in ch], fs, version=2)["suspect"] is True
    assert label_plausibility_v2({"O1": {"alpha": 0.5}})["available"] is False
    ok = montage_sanity(x, ch, fs, version=1)
    assert ok["available"] and ok["suspect"] is False and {"O1", "O2"} <= set(ok["top3"])
    swap = {"O1": "Fp1", "Fp1": "O1", "O2": "Fp2", "Fp2": "O2"}
    ch2 = [swap.get(c, c) for c in ch]                       # the signal stays, the labels are exchanged
    bad = montage_sanity(x, ch2, fs, version=1)
    assert bad["suspect"] is True and not (set(bad["top3"]) & {"O1", "O2", "P3", "P4", "Pz"})
    lr = {"O1": "O2", "O2": "O1"}
    assert montage_sanity(x, [lr.get(c, c) for c in ch], fs, version=1)["suspect"] is False        # left/right swap is not detectable
    assert label_plausibility({"O1": {"alpha": 0.5}})["available"] is False
