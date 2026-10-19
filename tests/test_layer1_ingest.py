"""Layer 1 ingestion tests."""

from __future__ import annotations

from neurolens.contracts.signal import ElectrodeSystem
from neurolens.layer1_ingest.electrodes import (
    detect_electrode_system,
    is_eeg_channel,
    normalize_channel_name,
)
from neurolens.layer1_ingest.registry import ingest


def test_normalize_channel_names():
    assert normalize_channel_name("EEG Fp1-Ref") == "Fp1"
    assert normalize_channel_name("FP1") == "Fp1"
    assert normalize_channel_name("T7") == "T3"  # 10-10 -> classic alias
    assert normalize_channel_name("P8") == "T6"
    assert normalize_channel_name("EKG") == "ECG"


def test_detect_system_10_20():
    ch = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
    assert detect_electrode_system(ch) == ElectrodeSystem.TEN_TWENTY


def test_is_eeg_channel():
    assert is_eeg_channel("C3")
    assert is_eeg_channel("Fz")
    assert not is_eeg_channel("ECG")
    assert not is_eeg_channel("EOG")


def test_ingest_synthetic_edf(demo_edf):
    sig = ingest(demo_edf)
    assert sig.sampling_rate_hz == 256
    assert "Fp1" in sig.channel_names
    assert "ECG" in sig.channel_names
    assert sig.electrode_system == ElectrodeSystem.TEN_TWENTY
    assert sig.n_samples == 360 * 256
    assert sig.units == "uV"


def test_file_check_flags_scale_and_short_recording(demo_edf, tmp_path):
    import numpy as np
    import pyedflib

    from neurolens.layer1_ingest.file_check import check_recording

    ok = {c.name: c.status for c in check_recording(demo_edf).checks}
    assert ok["format"] == "ok" and ok["electrodes_1020"] == "ok" and ok["patient_age"] == "warning"

    # 2 minutes, 128 Hz, signals in volts labelled as uV, 12 of 19 electrodes
    labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "Cz", "Pz"]
    fs, n = 128, 128 * 120
    rng = np.random.default_rng(0)
    path = tmp_path / "bad.edf"
    w = pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_EDFPLUS)
    w.setSignalHeaders([{"label": c, "dimension": "uV", "sample_frequency": fs, "physical_max": 1.0,
                         "physical_min": -1.0, "digital_max": 32767, "digital_min": -32768,
                         "prefilter": "HP:0.5Hz LP:35Hz", "transducer": ""} for c in labels])
    w.writeSamples([rng.normal(0, 2e-5, n) for _ in labels])
    w.close()
    rep = check_recording(path)
    st = {c.name: c.status for c in rep.checks}
    assert not rep.accepted
    assert st["duration"] == "reject" and st["amplitude_units"] == "reject" and st["electrodes_1020"] == "reject"
    assert st["sampling_rate"] == "warning" and st["export_filters"] == "warning"


def test_file_check_unreadable(tmp_path):
    from neurolens.layer1_ingest.file_check import check_recording

    p = tmp_path / "x.edf"
    p.write_bytes(b"not an edf")
    rep = check_recording(p)
    assert not rep.accepted and rep.checks[0].name == "format"
