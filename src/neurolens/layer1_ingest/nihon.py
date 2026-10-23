"""Nihon Kohden ingestor (EEG-1100/2100, QI-403A, DAE-2100D) - the data file ``.EEG`` with optional
companions in the same folder: ``.21E`` (electrode names), ``.LOG`` (events), ``.PNT`` (patient /
start time). No MNE dependency; the layout follows the format description in MNE-Python's
``mne/io/nihon`` (BSD-3-Clause) and the result is checked against it (tests, D79).

Layout: a device header; a control block whose data blocks each hold sampling rate, duration
(0.1 s units), channel indices and multiplexed little-endian 16-bit samples stored with a 0x8000
offset, one extra (event-mark) word per sample. EEG channels span -3200..3199.902 µV over 65535
steps; DC channels and names outside the electrode table -12002.9..12002.56 mV. Several data blocks (recording paused and resumed) are
concatenated with a "boundary" marker at each join.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..contracts.signal import ClinicalContext, Marker, PatientInfo, Provenance, UnifiedSignal
from .base import Ingestor
from .electrodes import detect_electrode_system, normalize_channel_name

VERSIONS = {"EEG-1100A V01.00", "EEG-1100B V01.00", "EEG-1100C V01.00", "QI-403A   V01.00",
            "QI-403A   V02.00", "EEG-2100  V01.00", "EEG-2100  V02.00", "DAE-2100D V01.30",
            "DAE-2100D V02.00"}

# default electrode table (index -> name); the .21E file overrides it
DEFAULT_LABELS = (["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4",
                   "T5", "T6", "FZ", "CZ", "PZ", "E", "PG1", "PG2", "A1", "A2", "T1", "T2"]
                  + [f"X{i}" for i in range(1, 12)] + [f"NA{i}" for i in range(1, 6)]
                  + [f"DC{i:02}" for i in range(1, 33)] + ["BN1", "BN2", "Mark1", "Mark2"]
                  + [f"NA{i}" for i in range(6, 28)] + ["X12/BP1", "X13/BP2", "X14/BP3", "X15/BP4"]
                  + [f"X{i}" for i in range(16, 166)] + ["NA28", "Z"])
EEG_RANGE_UV = (-3200.0, 3199.902)
DC_RANGE_MV = (-12002.9, 12002.56)


def _u8(b: bytes, o: int) -> int:
    return b[o]


def _u16(b: bytes, o: int) -> int:
    return struct.unpack_from("<H", b, o)[0]


def _u32(b: bytes, o: int) -> int:
    return struct.unpack_from("<I", b, o)[0]


def _text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _companion(path: Path, ext: str) -> Path | None:
    for cand in (path.with_suffix(ext), path.with_suffix(ext.lower())):
        if cand.exists():
            return cand
    return None


def electrode_labels(path: Path) -> list[str]:
    labels = list(DEFAULT_LABELS)
    e21 = _companion(path, ".21E")
    if e21 is None:
        return labels
    keep = False
    for line in _text(e21).splitlines():
        if line.startswith("["):
            keep = "ELECTRODE" in line or "REFERENCE" in line
        elif keep and "=" in line:
            k, v = line.split("=", 1)
            if k.strip().isdigit():
                i = int(k)
                labels.extend(["UNK"] * (i + 1 - len(labels)))
                labels[i] = v.strip()
    return labels


def parse_header(path: str | Path) -> dict:
    path = Path(path)
    b = path.read_bytes()
    version = b[:16].decode("ascii", errors="replace")
    if version not in VERSIONS:
        raise ValueError(f"not a supported Nihon Kohden .EEG file (device header {version!r}); "
                         "BrainVision data is read from its .vhdr")
    if b[0x81:0x91].decode("ascii", errors="replace") not in VERSIONS or b[0x17FE] != 1:
        raise ValueError("not a valid Nihon Kohden .EEG file (control/waveform block)")
    if _u8(b, 0x91) != 1:
        raise ValueError(f"Nihon Kohden: {_u8(b, 0x91)} control blocks; only one is supported")
    labels = electrode_labels(path)
    ctl = _u32(b, 0x92)
    blocks = []
    for i in range(_u8(b, ctl + 17)):
        a = _u32(b, ctl + i * 20 + 18)
        n = _u8(b, a + 0x26)
        blocks.append({"address": a, "n_channels": n, "fs": _u16(b, a + 0x1A) & 0x3FFF,
                       "n_samples": _u32(b, a + 0x1C) * (_u16(b, a + 0x1A) & 0x3FFF) // 10,
                       "channels": [labels[_u8(b, a + 0x27 + k * 10)] for k in range(n)]})
    if not blocks:
        raise ValueError("Nihon Kohden: no data blocks")
    if any(bl["channels"] != blocks[0]["channels"] or bl["fs"] != blocks[0]["fs"] for bl in blocks):
        raise ValueError("Nihon Kohden: data blocks differ in channels or sampling rate")
    return {"version": version, "fs": float(blocks[0]["fs"]), "channels": blocks[0]["channels"], "blocks": blocks}


def _scale_uv(name: str, labels_upper: list[str]) -> tuple[float, float]:
    """(physical minimum, step) in µV for a channel, by its position in the electrode table:
    DC inputs (positions 42-73, 76, 77) and names not in the table span ±12 V (mV range),
    the rest ±3.2 mV."""
    up = name.upper()
    idx = labels_upper.index(up) if up in labels_upper else None
    dc = idx is None or 42 <= idx <= 73 or idx in (76, 77)
    lo, hi = (DC_RANGE_MV[0] * 1e3, DC_RANGE_MV[1] * 1e3) if dc else EEG_RANGE_UV
    return lo, (hi - lo) / 65535.0


def _log_block(b: bytes, i: int) -> list[bytes]:
    if 0x92 + i * 20 + 4 > len(b):
        return []
    a = _u32(b, 0x92 + i * 20)
    if a == 0 or a + 0x14 > len(b):
        return []
    n = _u8(b, a + 0x12)
    return [b[a + 0x14 + k * 45: a + 0x14 + (k + 1) * 45] for k in range(n)]


def read_log(path: Path) -> list[tuple[float, str]]:
    """(seconds from the start, text) events of the .LOG file. Files with up to 21 log blocks
    keep sub-second parts (microseconds) in a parallel block 22 positions further."""
    log = _companion(path, ".LOG")
    if log is None:
        return []
    b = log.read_bytes()
    if b[:16].decode("ascii", errors="replace") not in VERSIONS:
        return []
    n_blocks = _u8(b, 0x91)
    out = []
    for i in range(n_blocks):
        logs = _log_block(b, i)
        subs = _log_block(b, i + 22) if n_blocks <= 21 else []
        for k, rec in enumerate(logs):
            hms = rec[20:26].decode("ascii", errors="replace")
            if not hms.isdigit():
                continue
            t = int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])
            desc = rec[:20]
            if len(subs) == len(logs):
                frac = subs[k][24:30].decode("ascii", errors="replace")
                t += int(frac) / 1e6 if frac.isdigit() else 0.0
                desc += subs[k][:20]
            desc = desc.rstrip(b"\x00")
            try:
                txt = desc.decode("utf-8")
            except UnicodeDecodeError:
                txt = desc.decode("latin-1")
            out.append((float(t), txt.replace("\x00", "").strip()))
    return out


class NihonKohdenIngestor(Ingestor):
    extensions = (".eeg",)
    source_name = "nihon_kohden"

    def read(self, path: str | Path, *, patient: PatientInfo | None = None,
             context: ClinicalContext | None = None) -> UnifiedSignal:
        path = Path(path)
        h = parse_header(path)
        b = path.read_bytes()
        n = len(h["channels"])
        labels_upper = [x.upper() for x in electrode_labels(path)]
        lo, step = np.array([_scale_uv(c, labels_upper) for c in h["channels"]], float).T
        parts, markers, t0 = [], [], 0.0
        for bl in h["blocks"]:
            start = bl["address"] + 0x27 + n * 10
            raw = np.frombuffer(b, "<u2", count=bl["n_samples"] * (n + 1), offset=start)
            d = (raw + np.uint16(0x8000)).astype(np.uint16).view(np.int16).reshape(-1, n + 1).T[:n]
            parts.append(((d.astype(np.float64) + 32768.0) * step[:, None] + lo[:, None]).astype(np.float32))
            if t0 > 0:
                markers.append(Marker(onset_s=t0, text="boundary (recording paused)"))
            t0 += bl["n_samples"] / h["fs"]
        markers += [Marker(onset_s=float(o), text=t) for o, t in read_log(path)]
        names = [normalize_channel_name(c) for c in h["channels"]]
        return UnifiedSignal(
            signal=np.concatenate(parts, axis=1), sampling_rate_hz=h["fs"], channel_names=names,
            electrode_system=detect_electrode_system(names), reference="unknown", source=self.source_name,
            patient=patient or PatientInfo(), context=context or ClinicalContext(),
            provenance=Provenance(digitized=False, confidence=1.0, source_file=path.name, pipeline_version="0.1.0"),
            markers=sorted(markers, key=lambda m: m.onset_s),
        )
