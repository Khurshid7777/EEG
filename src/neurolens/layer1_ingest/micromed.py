"""Micromed ingestor (Brain-Quick / SystemPLUS ``.TRC``, header version 4). No MNE/neo dependency;
the layout follows the format description in python-neo's ``micromedrawio`` (BSD-3-Clause) and the
result is checked against it (tests, D81).

Layout: a fixed header (data offset, channel count, base rate, bytes per sample at 138; header
version at 175) and 15 named zones from 176 (8-byte name, offset, length). ORDER lists the
electrode codes of the stored channels; LABCOD holds 128-byte electrode records (name, logical
min/max/ground, physical min/max, unit, rate factor); data are multiplexed unsigned integers,
physical = (raw - logical ground) * (physical range / (logical range + 1)). TRONCA lists the
segments of a trimmed recording (start time, offset in the data); NOTE (text comments) and
EVENT A/B (labelled intervals) become markers; TRIGGER codes are not carried. Marker times are
positions in the stored data (the signal's own time axis); neo reports them on the original clock
instead, i.e. shifted by the start of the first segment of a trimmed file.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..contracts.signal import ClinicalContext, Marker, PatientInfo, Provenance, UnifiedSignal
from .base import Ingestor
from .electrodes import detect_electrode_system, normalize_channel_name

ZONES = ("ORDER", "LABCOD", "NOTE", "FLAGS", "TRONCA", "IMPED_B", "IMPED_E", "MONTAGE", "COMPRESS",
         "AVERAGE", "HISTORY", "DVIDEO", "EVENT A", "EVENT B", "TRIGGER")
_UNIT_TO_UV = {-1: 1e-3, 0: 1.0, 1: 1e3, 2: 1e6}       # nV, µV, mV, V; other codes are not voltages


def parse_header(path: str | Path) -> dict:
    b = Path(path).read_bytes()
    if b[175] != 4:
        raise ValueError(f"Micromed .TRC header version {b[175]} not supported (version 4 only)")
    data_start, n_ch, _mux, rate_min, nbytes = struct.unpack_from("<IHHHH", b, 138)
    if nbytes not in (1, 2, 4):
        raise ValueError(f"Micromed: {nbytes} bytes per sample not supported")
    zones = {}
    for i, name in enumerate(ZONES):
        z, pos, length = struct.unpack_from("<8sII", b, 176 + i * 16)
        if z.decode("ascii", errors="replace").strip() != name:
            raise ValueError(f"Micromed: zone {i} is {z!r}, expected {name!r}")
        zones[name] = (pos, length)
    order = np.frombuffer(b, "<u2", count=n_ch, offset=zones["ORDER"][0])
    chans = []
    for code in order:
        o = zones["LABCOD"][0] + int(code) * 128 + 2
        name = b[o:o + 6].split(b"\x00")[0].decode("ascii", errors="replace").strip()
        lmin, lmax, lground, pmin, pmax = struct.unpack_from("<iiiii", b, o + 12)
        (unit,) = struct.unpack_from("<h", b, o + 32)
        (rate,) = struct.unpack_from("<H", b, o + 42)
        chans.append({"name": name, "gain": (pmax - pmin) / (lmax - lmin + 1), "ground": lground,
                      "unit": unit, "fs": rate * rate_min})
    if len({c["fs"] for c in chans}) != 1:
        raise ValueError("Micromed: channels differ in sampling rate")
    segs = []
    pos = zones["TRONCA"][0]
    for k in range(100):
        start, offset = struct.unpack_from("<II", b, pos + 8 * k)
        if start == 0 and offset == 0:
            break
        segs.append((start, offset))
    return {"data_start": data_start, "n_channels": n_ch, "bytes": nbytes, "fs": float(chans[0]["fs"]),
            "channels": chans, "zones": zones, "segments": segs or [(0, 0)]}


def _events(b: bytes, zones: dict, n_samples: int, fs: float) -> list[Marker]:
    out = []
    pos, length = zones["NOTE"]
    for k in range(length // 44):
        start, = struct.unpack_from("<I", b, pos + 44 * k)
        text = b[pos + 44 * k + 4: pos + 44 * k + 44].split(b"\x00")[0].decode("latin-1").strip()
        if start and start < n_samples and text:
            out.append(Marker(onset_s=start / fs, text=text))
    for zone in ("EVENT A", "EVENT B"):
        pos, length = zones[zone]
        for k in range(length // 12):
            label, start, stop = struct.unpack_from("<III", b, pos + 12 * k)
            if start and start < n_samples:
                out.append(Marker(onset_s=start / fs, duration_s=max(0, int(stop) - int(start)) / fs,
                                  text=f"{zone.title()} {label}"))
    return out


class MicromedIngestor(Ingestor):
    extensions = (".trc",)
    source_name = "micromed"

    def read(self, path: str | Path, *, patient: PatientInfo | None = None,
             context: ClinicalContext | None = None) -> UnifiedSignal:
        path = Path(path)
        h = parse_header(path)
        b = path.read_bytes()
        n, fs = h["n_channels"], h["fs"]
        raw = np.frombuffer(b, f"<u{h['bytes']}", offset=h["data_start"])
        ns = raw.size // n
        x = raw[: ns * n].reshape(ns, n).T.astype(np.float64)
        gain = np.array([c["gain"] * _UNIT_TO_UV.get(c["unit"], 1.0) for c in h["channels"]])[:, None]
        ground = np.array([c["ground"] for c in h["channels"]], float)[:, None]
        markers = _events(b, h["zones"], ns, fs)
        markers += [Marker(onset_s=off / fs, text="boundary (segment of a trimmed recording)")
                    for _start, off in h["segments"][1:]]
        names = [normalize_channel_name(c["name"]) for c in h["channels"]]
        return UnifiedSignal(
            signal=((x - ground) * gain).astype(np.float32), sampling_rate_hz=fs, channel_names=names,
            electrode_system=detect_electrode_system(names), reference="unknown", source=self.source_name,
            patient=patient or PatientInfo(), context=context or ClinicalContext(),
            provenance=Provenance(digitized=False, confidence=1.0, source_file=path.name, pipeline_version="0.1.0"),
            markers=sorted(markers, key=lambda m: m.onset_s),
        )
