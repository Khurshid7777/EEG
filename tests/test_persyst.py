"""Persyst .lay/.dat reader (D80); also checked against MNE-Python on the MNE testing file
(exact agreement)."""

import numpy as np

from neurolens.layer1_ingest.registry import ingest


def _write(tmp_path, data, cal=0.5, datatype=0):
    lay = tmp_path / "rec.lay"
    names = ["Fp1-Ref", "O1-Ref", "POL EKG1-Ref"]
    lay.write_text(
        "[FileInfo]\nFile=C:\\\\export\\\\rec.dat\nFileType=Interleaved\nSamplingRate=256\nHeaderLength=0\n"
        f"Calibration={cal}\nWaveformCount=3\nDataType={datatype}\n\n[ChannelMap]\n"
        + "".join(f"{n}={i + 1}\n" for i, n in enumerate(names))
        + "\n[Comments]\n2.500,0,0,65536,Eyes closed\n10.0,180.0,0,65536,HV start\n\n[SampleTimes]\n0=0.0\n")
    data.T.astype("<i2" if datatype == 0 else "<i4").tofile(tmp_path / "rec.dat")
    return lay


def test_persyst_int16(tmp_path):
    d = np.array([[10, -20, 30, 40], [1, 2, 3, 4], [-100, 0, 100, 200]])
    s = ingest(_write(tmp_path, d))
    assert s.sampling_rate_hz == 256 and s.channel_names == ["Fp1", "O1", "ECG"]
    assert np.allclose(s.signal, d * 0.5)
    assert [(m.onset_s, m.duration_s, m.text) for m in s.markers] == [(2.5, 0.0, "Eyes closed"),
                                                                       (10.0, 180.0, "HV start")]


def test_persyst_int32(tmp_path):
    d = np.array([[100000, -1], [2, 3], [4, 5]])
    s = ingest(_write(tmp_path, d, cal=0.01, datatype=7))
    assert np.allclose(s.signal[0], [1000.0, -0.01])
