import numpy as np

from neurolens.contracts.signal import Marker, UnifiedSignal
from neurolens.layer3_features.procedures import (
    analyze_procedures,
    classify,
    intervals,
    procedures_text,
    tag_markers,
)

FS = 200.0
CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]


def test_classify_ru_en():
    assert classify("Eyes Closed")[0] == "eyes_closed"
    assert classify("Глаза закрыты")[0] == "eyes_closed"
    assert classify("открыл глаза")[0] == "eyes_open"
    assert classify("EO")[0] == "eyes_open"
    assert classify("HV start")[0] == "hv_start"
    assert classify("ГВ конец")[0] == "hv_end"
    assert classify("После ГВ")[0] == "hv_end"
    assert classify("Photic 6 Hz") == ("photic", 6.0)
    assert classify("ФС 12 Гц") == ("photic", 12.0)
    assert classify("14 Hz") == ("photic", 14.0)
    assert classify("Comment: patient moved") is None


def test_intervals():
    tagged, unknown = tag_markers([Marker(onset_s=10, text="eyes closed"), Marker(onset_s=40, text="eyes open"),
                                   Marker(onset_s=200, text="HV"), Marker(onset_s=380, text="HV end"),
                                   Marker(onset_s=500, text="Photic 6 Hz"), Marker(onset_s=505, text="x")])
    iv = intervals(tagged, 900.0)
    assert iv["eyes_closed"] == [(11.0, 40.0)]
    assert iv["eyes_open"] == [(41.0, 100.0)]          # capped at 60 s after the marker
    assert iv["hv"] == [(200.0, 380.0)]
    assert iv["photic"] == [(500.0, 510.0, 6.0)]
    assert unknown == ["x"]


def _signal(markers):
    rng = np.random.default_rng(0)
    n = int(600 * FS)
    t = np.arange(n) / FS
    x = rng.normal(0, 5, (len(CH), n))
    alpha = np.where((t >= 0) & (t < 60), 40.0, 5.0) * np.sin(2 * np.pi * 10 * t)   # EC 0-60, EO 60-120
    slow = np.where((t >= 240) & (t < 420), 60.0, 3.0) * np.sin(2 * np.pi * 3 * t)  # HV 240-420
    drive = np.where((t >= 500) & (t < 510), 20.0, 0.0) * np.sin(2 * np.pi * 6 * t)
    for i, c in enumerate(CH):
        x[i] += slow
        if c in ("O1", "O2", "P3", "P4"):
            x[i] += alpha + drive
    return UnifiedSignal(signal=x.astype(np.float32), sampling_rate_hz=FS, channel_names=CH, markers=markers)


def test_analyze_procedures():
    sig = _signal([Marker(onset_s=0, text="Eyes closed"), Marker(onset_s=60, text="Eyes open"),
                   Marker(onset_s=240, text="HV start"), Marker(onset_s=420, text="HV end"),
                   Marker(onset_s=500, text="Photic 6 Hz")])
    p = analyze_procedures(sig)
    assert abs(p["ec_pdr_hz"] - 10.0) <= 0.5
    assert p["reactive"] and p["alpha_attenuation"] > 0.8
    assert p["hv"]["slowing"] and not p["hv"]["persists"]
    assert p["photic"][0]["driving"]
    txt = procedures_text(p)
    assert "Гц" in txt.ru and "реактивен" in txt.ru and "в заключение не входит" in txt.ru


def test_no_markers_and_unrecognized():
    assert analyze_procedures(_signal([])) is None
    p = analyze_procedures(_signal([Marker(onset_s=5, text="Impedance")]))
    assert "не распознаны" in procedures_text(p).ru


def test_markers_survive_with_signal():
    sig = _signal([Marker(onset_s=1, text="EC")])
    assert sig.with_signal(np.asarray(sig.signal)).markers[0].text == "EC"
