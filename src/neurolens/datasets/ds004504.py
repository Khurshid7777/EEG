"""OpenNeuro ds004504 (Miltiadous et al., 2023; CC0): resting eyes-closed EEG of 36 people
with Alzheimer's disease (A), 23 with frontotemporal dementia (F) and 29 cognitively normal
controls (C), University of Ioannina, Greece; Nihon Kohden, 19 channels 10-20, A1-A2
reference, 500 Hz, ~10-13 min. Raw BIDS EEGLAB files from the public OpenNeuro S3 bucket.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

BASE = "https://s3.amazonaws.com/openneuro.org/ds004504"


@dataclass
class DsRecord:
    record_id: int
    subject: str
    group: str            # "A" | "F" | "C"
    age_years: Optional[float]
    mmse: Optional[float]

    @property
    def abnormal(self) -> bool:  # dementia (A/F); not an EEG label
        return self.group != "C"

    @property
    def key(self) -> str:
        return f"{self.subject}/eeg/{self.subject}_task-eyesclosed_eeg.set"


class Ds004504Client:
    def __init__(self, cache: str | Path = "data/openneuro/ds004504"):
        self.cache = Path(cache)

    def _get(self, key: str) -> Path:
        p = self.cache / key
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            part = p.with_suffix(p.suffix + ".part")
            with urllib.request.urlopen(f"{BASE}/{key}", timeout=120) as r, open(part, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            part.replace(p)
        return p

    def records(self) -> list[DsRecord]:
        rows = csv.DictReader(io.StringIO(self._get("participants.tsv").read_text()), delimiter="\t")
        out = []
        for i, r in enumerate(rows):
            num = lambda v: float(v) if v not in ("", "n/a", None) else None  # noqa: E731
            out.append(DsRecord(i, r["participant_id"], r["Group"], num(r.get("Age")), num(r.get("MMSE"))))
        return out

    def fetch(self, rec: DsRecord) -> Path:
        return self._get(rec.key)
