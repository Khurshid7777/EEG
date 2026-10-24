"""Micromed .TRC reader (D81) on a synthetic header-v4 file; also checked against python-neo on
neo's test files (signals identical)."""

import struct

import numpy as np
import pytest

from neurolens.layer1_ingest.micromed import ZONES, MicromedIngestor
from neurolens.layer1_ingest.registry import ingest


def write_trc(path, raw, names, fs=256, notes=(), segments=(), version=4):
    n, ns = raw.shape
    b = bytearray(4096)
    b[0:32] = b"* MICROMED  Brain-Quick file *  "
    b[175] = version
    zpos = {"ORDER": 1024, "LABCOD": 1200, "NOTE": 2800, "TRONCA": 3300, "EVENT A": 3500, "EVENT B": 3600}
    for i, z in enumerate(ZONES):
        struct.pack_into("<8sII", b, 176 + i * 16, z.encode().ljust(8), zpos.get(z, 3900), 44 * 4 if z == "NOTE" else 12)
    data_start = 4096
    struct.pack_into("<IHHHH", b, 138, data_start, n, 0, 1, 2)
    for k in range(n):
        struct.pack_into("<H", b, 1024 + 2 * k, k)
        o = 1200 + k * 128 + 2
        b[o:o + 6] = names[k].encode().ljust(6, b"\0")
        struct.pack_into("<iiiii", b, o + 12, 0, 65535, 32768, -3200, 3200)   # 65536 steps over 6400 µV
        struct.pack_into("<h", b, o + 32, 0)                                   # µV
        struct.pack_into("<H", b, o + 42, fs)
    for k, (t, txt) in enumerate(notes):
        struct.pack_into("<I", b, 2800 + 44 * k, int(t * fs))
        b[2804 + 44 * k: 2804 + 44 * k + len(txt)] = txt.encode()
    for k, (start, off) in enumerate(segments):
        struct.pack_into("<II", b, 3300 + 8 * k, start, off)
    path.write_bytes(bytes(b) + raw.T.astype("<u2").tobytes())


def test_micromed_roundtrip(tmp_path):
    raw = np.tile(np.array([[32768, 32778, 32758, 40000], [0, 65535, 32768, 32769]]), (1, 64))  # 1 s
    p = tmp_path / "rec.TRC"
    write_trc(p, raw, ["FP1", "O2"], notes=[(0.5, "Eyes closed")])
    s = ingest(p)
    assert s.channel_names == ["Fp1", "O2"] and s.sampling_rate_hz == 256
    assert np.allclose(s.signal, (raw - 32768) * (6400 / 65536), atol=1e-3)
    assert [(m.onset_s, m.text) for m in s.markers] == [(0.5, "Eyes closed")]


def test_micromed_segments_and_version(tmp_path):
    raw = np.full((1, 600), 32768)
    p = tmp_path / "cut.TRC"
    write_trc(p, raw, ["CZ"], segments=[(5, 0), (900, 300)])
    assert [(m.onset_s, m.text.split()[0]) for m in ingest(p).markers] == [(300 / 256, "boundary")]
    write_trc(p, raw, ["CZ"], version=3)
    with pytest.raises(ValueError, match="version 3"):
        MicromedIngestor().read(p)
