"""BrainVision ingestor (Brain Products; also written by EEGLAB, FieldTrip, MNE) - a text header
``.vhdr``, a binary data file and an optional marker file ``.vmrk``. Supported: DataFormat=BINARY,
orientation MULTIPLEXED or VECTORIZED, INT_16 / UINT_16 / INT_32 / IEEE_FLOAT_32. Physical value =
raw x resolution, in the channel's unit (µV unless given: nV, µV/uV, mV, V). No MNE dependency.
Markers (:func:`read_markers`) are carried as :attr:`UnifiedSignal.markers`.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import numpy as np

from ..contracts.signal import ClinicalContext, Marker, PatientInfo, Provenance, UnifiedSignal
from .base import Ingestor
from .electrodes import detect_electrode_system, normalize_channel_name

_DTYPES = {"INT_16": "<i2", "UINT_16": "<u2", "INT_32": "<i4", "IEEE_FLOAT_32": "<f4"}
_UNIT_TO_UV = {"nv": 1e-3, "µv": 1.0, "μv": 1.0, "uv": 1.0, "mv": 1e3, "v": 1e6}


def _ini(path: Path) -> configparser.ConfigParser:
    text = path.read_bytes().decode("utf-8", errors="replace")
    body = text[text.index("["):] if "[" in text else text     # drop the identification line
    cp = configparser.ConfigParser(interpolation=None, comment_prefixes=(";",), strict=False)
    cp.optionxform = str
    cp.read_string(body)
    return cp


def parse_header(path: str | Path) -> dict:
    path = Path(path)
    cp = _ini(path)
    ci = cp["Common Infos"]
    n = int(ci["NumberOfChannels"])
    names, res, units = [], [], []
    for i in range(1, n + 1):
        f = [v.replace(r"\1", ",") for v in cp["Channel Infos"][f"Ch{i}"].split(",")]
        names.append(f[0].strip())
        res.append(float(f[2]) if len(f) > 2 and f[2].strip() else 1.0)
        units.append(f[3].strip() if len(f) > 3 and f[3].strip() else "µV")
    return {"data_file": path.with_name(ci["DataFile"].strip()),
            "marker_file": path.with_name(ci["MarkerFile"].strip()) if ci.get("MarkerFile") else None,
            "format": ci.get("DataFormat", "BINARY").strip().upper(),
            "orientation": ci.get("DataOrientation", "MULTIPLEXED").strip().upper(),
            "n_channels": n, "fs": 1e6 / float(ci["SamplingInterval"]),
            "binary_format": cp["Binary Infos"].get("BinaryFormat", "INT_16").strip().upper()
            if cp.has_section("Binary Infos") else "INT_16",
            "names": names, "resolution": res, "units": units}


def read_markers(path: str | Path, fs: float) -> list[tuple[float, float, str]]:
    """(onset s, duration s, "type: description") from a .vmrk; positions are 1-based samples."""
    cp = _ini(Path(path))
    out = []
    if cp.has_section("Marker Infos"):
        for v in cp["Marker Infos"].values():
            f = [x.replace(r"\1", ",") for x in v.split(",")]
            if len(f) >= 3 and f[2].strip().lstrip("-").isdigit():
                size = int(f[3]) if len(f) > 3 and f[3].strip().isdigit() else 1
                out.append(((int(f[2]) - 1) / fs, size / fs, f"{f[0].strip()}: {f[1].strip()}".strip(": ")))
    return sorted(out)


class BrainVisionIngestor(Ingestor):
    extensions = (".vhdr",)
    source_name = "brainvision"

    def read(self, path: str | Path, *, patient: PatientInfo | None = None,
             context: ClinicalContext | None = None) -> UnifiedSignal:
        path = Path(path)
        h = parse_header(path)
        if h["format"] != "BINARY":
            raise ValueError(f"BrainVision: DataFormat={h['format']} not supported (BINARY only)")
        if h["binary_format"] not in _DTYPES:
            raise ValueError(f"BrainVision: BinaryFormat={h['binary_format']} not supported")
        n = h["n_channels"]
        raw = np.fromfile(h["data_file"], dtype=_DTYPES[h["binary_format"]])
        n_s = raw.size // n
        raw = raw[: n_s * n]
        data = raw.reshape(n_s, n).T if h["orientation"] == "MULTIPLEXED" else raw.reshape(n, n_s)
        scale = np.array([r * _UNIT_TO_UV.get(u.lower(), 1.0) for r, u in zip(h["resolution"], h["units"])],
                         dtype=np.float64)[:, None]
        names = [normalize_channel_name(s) for s in h["names"]]
        mk = h["marker_file"]
        markers = ([Marker(onset_s=o, duration_s=d, text=t) for o, d, t in read_markers(mk, h["fs"])]
                   if mk is not None and mk.exists() else [])
        return UnifiedSignal(
            signal=(data.astype(np.float64) * scale).astype(np.float32), sampling_rate_hz=h["fs"],
            channel_names=names, electrode_system=detect_electrode_system(names), reference="unknown",
            source=self.source_name, patient=patient or PatientInfo(), context=context or ClinicalContext(),
            provenance=Provenance(digitized=False, confidence=1.0, source_file=path.name, pipeline_version="0.1.0"),
            markers=markers,
        )
