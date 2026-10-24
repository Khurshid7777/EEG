"""Persyst ingestor - the text layout file ``.lay`` (INI-like: [FileInfo], [ChannelMap],
[Comments], ...) next to its binary ``.dat``. Data are interleaved 16-bit (DataType=0) or 32-bit
(DataType=7) little-endian integers; physical value (µV) = raw x Calibration. [Comments] lines
``onset,duration,state,type,text`` become markers. No MNE dependency; checked against MNE-Python's
``read_raw_persyst`` (D80).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.signal import ClinicalContext, Marker, PatientInfo, Provenance, UnifiedSignal
from .base import Ingestor
from .electrodes import detect_electrode_system, normalize_channel_name

_DTYPES = {0: "<i2", 7: "<i4"}


def parse_lay(path: str | Path) -> dict:
    """Sections of a .lay file: {section: {key: value}}, with [Comments] as a list of
    (onset s, duration s, text)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    out: dict = {"comments": []}
    sec = None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip().lower()
            out.setdefault(sec, {} if sec != "comments" else [])
            continue
        if sec == "comments":
            parts = line.split(",", 4)
            if len(parts) == 5:
                try:
                    out["comments"].append((float(parts[0]), float(parts[1]), parts[4].strip()))
                except ValueError:
                    pass
        elif sec is not None and "=" in line:
            k, v = line.split("=", 1)
            out[sec][k.strip() if sec == "channelmap" else k.strip().lower()] = v.strip()
    return out


class PersystIngestor(Ingestor):
    extensions = (".lay",)
    source_name = "persyst"

    def read(self, path: str | Path, *, patient: PatientInfo | None = None,
             context: ClinicalContext | None = None) -> UnifiedSignal:
        path = Path(path)
        lay = parse_lay(path)
        fi = lay.get("fileinfo", {})
        if fi.get("filetype", "Interleaved").lower() != "interleaved":
            raise ValueError(f"Persyst: FileType={fi.get('filetype')} not supported (Interleaved only)")
        dtype = _DTYPES.get(int(fi.get("datatype", 0)))
        if dtype is None:
            raise ValueError(f"Persyst: DataType={fi.get('datatype')} not supported (0: int16, 7: int32)")
        n = int(fi["waveformcount"])
        cmap = sorted(lay.get("channelmap", {}).items(), key=lambda kv: int(kv[1]))
        if len(cmap) != n:
            raise ValueError(f"Persyst: {len(cmap)} channels in [ChannelMap], WaveformCount={n}")
        dat = path.with_name(Path(fi.get("file", path.with_suffix(".dat").name)).name)
        if not dat.exists():
            dat = path.with_suffix(".dat")
        raw = np.fromfile(dat, dtype=dtype, offset=int(fi.get("headerlength", 0) or 0))
        ns = raw.size // n
        data = raw[: ns * n].reshape(ns, n).T.astype(np.float64) * float(fi.get("calibration", 1.0))
        names = [normalize_channel_name(k) for k, _ in cmap]
        markers = sorted((Marker(onset_s=o, duration_s=max(0.0, d), text=t) for o, d, t in lay["comments"] if t),
                         key=lambda m: m.onset_s)
        return UnifiedSignal(
            signal=data.astype(np.float32), sampling_rate_hz=float(fi["samplingrate"]), channel_names=names,
            electrode_system=detect_electrode_system(names), reference="unknown", source=self.source_name,
            patient=patient or PatientInfo(), context=context or ClinicalContext(),
            provenance=Provenance(digitized=False, confidence=1.0, source_file=path.name, pipeline_version="0.1.0"),
            markers=markers,
        )
