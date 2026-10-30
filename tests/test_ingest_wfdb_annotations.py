"""WFDB MIT-format annotation reader (increment 52)."""

import struct

from neurolens.layer1_ingest.wfdb import read_annotations


def _word(code, val):
    return struct.pack("<H", (code << 10) | val)


def _aux(s):
    raw = s.encode()
    return _word(63, len(raw)) + raw + (b"\0" if len(raw) % 2 else b"")


def test_read_annotations_notes_skip_and_aux(tmp_path):
    p = tmp_path / "x.arousal"
    data = (_word(22, 0) + _aux("W")                       # note at sample 0
            + _word(59, 0) + struct.pack("<HH", 0, 6000)   # skip 6000 samples
            + _word(22, 0) + _aux("N1")                    # note at 6000
            + _word(22, 1000) + _aux("(arousal_rera")      # note at 7000
            + _word(22, 1023) + _aux("R")                  # note at 8023
            + _word(0, 0))
    p.write_bytes(data)
    assert read_annotations(p) == [(0, 22, "W"), (6000, 22, "N1"), (7000, 22, "(arousal_rera"),
                                   (8023, 22, "R")]
