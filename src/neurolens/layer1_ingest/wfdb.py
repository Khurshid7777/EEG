"""WFDB ingestor (increment 35) - a plain-text ``.hea`` header and its signal file; formats 16
(int16, interleaved, optional ``+offset`` bytes, e.g. MATLAB v4 ``.mat`` written by WFDB) and
212 are not mixed: only format 16 is supported. Physical value = (digital - baseline) / gain.
A signal file shorter than the header says (a partial download) is read as far as it goes.
No wfdb dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.signal import ClinicalContext, PatientInfo, Provenance, UnifiedSignal
from .base import Ingestor
from .electrodes import detect_electrode_system, normalize_channel_name


def parse_header(path: str | Path) -> dict:
    lines = [ln.strip() for ln in Path(path).read_text().splitlines()]
    body = [ln for ln in lines if ln and not ln.startswith("#")]
    rec = body[0].split()
    n_sig, fs = int(rec[1]), float(rec[2].split("/")[0])
    n_samp = int(rec[3]) if len(rec) > 3 else None
    sigs = []
    for ln in body[1:1 + n_sig]:
        f = ln.split()
        fmt, _, off = f[1].partition("+")
        g = f[2].split("/")[0]
        gain, _, base = g.partition("(")
        sigs.append({"file": f[0], "format": int(fmt.split("x")[0].split(":")[0]), "offset": int(off or 0),
                     "gain": float(gain) or 200.0, "baseline": float(base.rstrip(")")) if base else 0.0,
                     "units": f[2].split("/")[1] if "/" in f[2] else "", "name": f[-1] if len(f) > 8 else f"Ch{len(sigs) + 1}"})
    comments = {k.strip(): v.strip() for k, _, v in (ln.lstrip("#").partition(":") for ln in lines if ln.startswith("#"))}
    return {"n_sig": n_sig, "fs": fs, "n_samp": n_samp, "signals": sigs, "comments": comments}


class WfdbIngestor(Ingestor):
    extensions = (".hea",)
    source_name = "wfdb"

    def read(self, path: str | Path, *, patient: PatientInfo | None = None,
             context: ClinicalContext | None = None) -> UnifiedSignal:
        path = Path(path)
        h = parse_header(path)
        sigs = h["signals"]
        if {s["file"] for s in sigs} != {sigs[0]["file"]} or {s["format"] for s in sigs} != {16}:
            raise ValueError("WFDB: only a single signal file in format 16 is supported")
        n = h["n_sig"]
        raw = np.fromfile(path.with_name(sigs[0]["file"]), dtype="<i2", offset=sigs[0]["offset"])
        n_s = len(raw) // n if h["n_samp"] is None else min(h["n_samp"], len(raw) // n)
        d = raw[: n_s * n].reshape(n_s, n).T.astype(np.float32)
        gain = np.array([s["gain"] for s in sigs], dtype=np.float32)[:, None]
        base = np.array([s["baseline"] for s in sigs], dtype=np.float32)[:, None]
        names = [normalize_channel_name(s["name"]) for s in sigs]
        return UnifiedSignal(
            signal=(d - base) / gain, sampling_rate_hz=h["fs"], channel_names=names,
            electrode_system=detect_electrode_system(names), reference="unknown",
            source=self.source_name, patient=patient or PatientInfo(),
            context=context or ClinicalContext(),
            provenance=Provenance(digitized=False, confidence=1.0, source_file=path.name,
                                  pipeline_version="0.1.0"),
        )
