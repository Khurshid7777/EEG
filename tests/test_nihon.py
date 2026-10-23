"""Nihon Kohden .EEG reader (D79) on a synthetic file built to the documented layout; the reader was
also checked against MNE-Python on the MNE testing file MB0400FU.EEG (max relative difference 6e-8)."""

import struct

import numpy as np
import pytest

from neurolens.layer1_ingest.nihon import DC_RANGE_MV, EEG_RANGE_UV, NihonKohdenIngestor, parse_header
from neurolens.layer1_ingest.registry import ingest

VERSION = b"EEG-1100C V01.00"


def _digital(v, lo, hi):
    step = (hi - lo) / 65535.0
    return np.round((np.asarray(v) - lo) / step).astype(np.int64) - 32768


def write_nk(path, signals_uv, chan_idx, fs=200, events=()):
    n, ns = len(chan_idx), signals_uv.shape[1]
    b = bytearray(0x2000)
    b[0:16] = VERSION
    b[0x81:0x91] = VERSION
    b[0x17FE] = 1
    b[0x91] = 1
    ctl = 0x1000
    struct.pack_into("<I", b, 0x92, ctl)
    b[ctl + 17] = 1
    data = 0x1900
    struct.pack_into("<I", b, ctl + 18, data)
    struct.pack_into("<H", b, data + 0x1A, fs | 0x4000)           # high bits are flags
    struct.pack_into("<I", b, data + 0x1C, ns * 10 // fs)        # duration in 0.1 s
    b[data + 0x26] = n
    for k, i in enumerate(chan_idx):
        b[data + 0x27 + k * 10] = i
    start = data + 0x27 + n * 10
    words = np.zeros((ns, n + 1), np.int64)
    for k, i in enumerate(chan_idx):
        lo, hi = (DC_RANGE_MV[0] * 1e3, DC_RANGE_MV[1] * 1e3) if 42 <= i <= 73 else EEG_RANGE_UV
        words[:, k] = _digital(signals_uv[k], lo, hi)
    raw = ((words + 0x8000) & 0xFFFF).astype("<u2").tobytes()
    b = b[:start] + raw
    path.write_bytes(bytes(b))
    path.with_suffix(".21E").write_text("[ELECTRODE]\n8=O1\n9=O2\n42=DC01\n[SETTING]\nx=1\n")
    log = bytearray(0x800)
    log[0:16] = VERSION
    log[0x91] = 1
    struct.pack_into("<I", log, 0x92, 0x300)
    struct.pack_into("<I", log, 0x92 + 22 * 20, 0x500)
    log[0x300 + 0x12] = log[0x500 + 0x12] = len(events)
    for k, (t, txt) in enumerate(events):
        s = int(t)
        rec = txt.encode().ljust(20, b"\0") + f"{s // 3600:02d}{s % 3600 // 60:02d}{s % 60:02d}".encode()
        log[0x300 + 0x14 + k * 45: 0x300 + 0x14 + k * 45 + len(rec)] = rec
        sub = b"\0" * 24 + f"{round((t - s) * 1e6):06d}".encode()
        log[0x500 + 0x14 + k * 45: 0x500 + 0x14 + k * 45 + len(sub)] = sub
    path.with_suffix(".LOG").write_bytes(bytes(log))


def test_nihon_roundtrip(tmp_path):
    t = np.arange(400) / 200
    sig = np.stack([50 * np.sin(2 * np.pi * 10 * t), -30 * np.cos(2 * np.pi * 3 * t), 1000 * t])
    p = tmp_path / "REC.EEG"
    write_nk(p, sig, [8, 9, 42], events=[(0.0, "REC START"), (1.25, "Eyes closed")])
    h = parse_header(p)
    assert h["fs"] == 200 and h["channels"] == ["O1", "O2", "DC01"]
    s = ingest(p)
    assert s.channel_names == ["O1", "O2", "Dc01"] and s.signal.shape == (3, 400)
    assert np.max(np.abs(s.signal[:2] - sig[:2])) < 0.06          # EEG step 0.098 µV
    assert np.max(np.abs(s.signal[2] - sig[2])) < 400               # DC step 0.37 mV
    assert [(m.onset_s, m.text) for m in s.markers] == [(0.0, "REC START"), (1.25, "Eyes closed")]


def test_brainvision_eeg_is_not_nihon(tmp_path):
    p = tmp_path / "x.eeg"
    p.write_bytes(np.zeros(4096, "<i2").tobytes())
    with pytest.raises(ValueError, match="vhdr"):
        NihonKohdenIngestor().read(p)


def test_nihon_zip_upload(tmp_path):
    import io
    import zipfile

    from neurolens.api.uploads import materialize

    p = tmp_path / "REC.EEG"
    write_nk(p, np.zeros((3, 200)), [8, 9, 42], events=[(0.5, "HV")])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for ext in (".EEG", ".21E", ".LOG"):
            z.write(p.with_suffix(ext), f"dir/REC{ext}")
    path, d = materialize(buf.getvalue(), "rec.zip")
    try:
        assert path.name == "REC.EEG"
        assert [m.text for m in ingest(path).markers] == ["HV"]
    finally:
        import shutil

        shutil.rmtree(d)
