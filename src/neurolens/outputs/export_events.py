"""Export of detector events for the tools a reader already uses (increment 85).

* ``events.tsv`` - BIDS-style table (onset, duration, trial_type, channels, regions, confidence, learned_probability),
  tab-separated, UTF-8, ``n/a`` for missing values.
* ``events.edf`` - an EDF+ file whose annotation channel carries the events (``onset``, ``duration``, text
  ``"<code>; ch=<channels>; conf=<x>"``) next to a one-channel flat signal of the recording's length, so that any EDF+ viewer
  (read back with pyedflib/MNE in the tests) can show them on the time axis. Compatibility with specific vendor viewers
  is not verified here.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..layer3_features.regional import REGIONS, canonical

FIELDS = ("onset", "duration", "trial_type", "channels", "regions", "confidence", "learned_probability")


def _regions(channels: Sequence[str]) -> str:
    out = []
    for c in (canonical(x) for x in channels):
        for region, chans in REGIONS.items():
            if c in chans and region not in out:
                out.append(region)
    return ",".join(out)


def event_rows(result_json: Mapping) -> list[dict]:
    rows = []
    for e in sorted(result_json.get("events") or [], key=lambda x: x["t_start"]):
        ev = e.get("evidence") or {}
        ch = e.get("channels") or []
        rows.append({"onset": f"{float(e['t_start']):.3f}", "duration": f"{max(0.0, float(e['t_end']) - float(e['t_start'])):.3f}",
                     "trial_type": e["code"], "channels": ",".join(ch) if ch else "n/a", "regions": _regions(ch) or "n/a",
                     "confidence": f"{float(e.get('confidence') or 0):.3f}",
                     "learned_probability": f"{float(ev['ied_probability_learned']):.3f}" if "ied_probability_learned" in ev else "n/a"})
    return rows


def write_events_tsv(result_json: Mapping, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(event_rows(result_json))
    return p


def write_events_edf(result_json: Mapping, duration_s: float, path: str | Path, fs: float = 1.0) -> Path:
    """EDF+ with the events as annotations (one flat 1 Hz channel spans the recording)."""
    import pyedflib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, int(np.ceil(duration_s)))
    w = pyedflib.EdfWriter(str(p), 1, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        w.setSignalHeader(0, {"label": "events", "dimension": "uV", "sample_frequency": fs, "physical_min": -1.0,
                              "physical_max": 1.0, "digital_min": -32768, "digital_max": 32767, "transducer": "", "prefilter": ""})
        w.writeSamples([np.zeros(int(n * fs))])
        for r in event_rows(result_json):
            w.writeAnnotation(float(r["onset"]), float(r["duration"]),
                              f"{r['trial_type']}; ch={r['channels']}; conf={r['confidence']}")
    finally:
        w.close()
    return p
