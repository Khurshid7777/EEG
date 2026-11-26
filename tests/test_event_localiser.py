import numpy as np

from neurolens.layer3_features.event_localiser import EventLocaliser, region_of, side_of
from neurolens.layer3_features.regional import STANDARD_1020


def _rec(spike_ch="T3", fs=200.0, secs=40):
    rng = np.random.default_rng(5)
    ch = list(STANDARD_1020)
    x = 5 * rng.standard_normal((len(ch), int(secs * fs)))
    i = int(20 * fs)
    x[ch.index(spike_ch), i:i + 12] += 60 * np.hanning(12)          # a sharp transient on one electrode
    return x, ch, fs


def test_locator_finds_the_spiking_electrode_in_all_variants():
    x, ch, fs = _rec("T3")
    loc = EventLocaliser(x, ch, fs)
    for v in ("A", "B", "C"):
        r = loc.locate(20.0, 20.06, variant=v)
        assert r["top"][0] == "T3" and r["region"] == "temporal" and r["side"] == "L", v


def test_point_event_and_edges():
    x, ch, fs = _rec("F4")
    loc = EventLocaliser(x, ch, fs)
    assert loc.locate(20.03, 20.03)["top"][0] == "F4"          # t1 == t0: +-0.15 s window
    import pytest
    with pytest.raises(ValueError):
        loc.locate(500.0, 500.1)
    assert (region_of("Pz"), side_of("Pz"), side_of("O2")) == ("parietal", "mid", "R")
