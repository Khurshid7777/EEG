"""TUH Abnormal EEG Corpus (TUAB) - local copy, read-only (increment 32, prepared before access).

Access is granted per researcher by the Neural Engineering Data Consortium (Temple University);
the corpus is downloaded by the owner and pointed to with ``root``. Layout (v2/v3):
``<root>/edf/{train,eval}/{normal,abnormal}/01_tcp_ar/<subject>_<session>_<token>.edf``.
Label and split come from the path; age, when present, from the EDF header ("Age:57").
Amplitudes are in µV (scale 1, unlike NMT's ×100 unit error).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_AGE = re.compile(rb"Age:\s*(\d{1,3})")


@dataclass
class TUABRecord:
    record_id: int          # position in the sorted file list of the whole corpus
    name: str               # file stem, e.g. "aaaaaaav_s004_t000"
    path: Path
    split: str              # "train" | "eval"
    abnormal: bool
    age_years: Optional[float] = None

    @property
    def subject(self) -> str:
        return self.name.split("_")[0]


def header_age(path: Path) -> Optional[float]:
    """Patient age from the EDF header's patient field (first 256 bytes), if given."""
    with open(path, "rb") as f:
        m = _AGE.search(f.read(256))
    if not m:
        return None
    age = float(m.group(1))
    return age if 0 <= age <= 120 else None


class TUABClient:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def records(self, split: Optional[str] = None) -> list[TUABRecord]:
        out = []
        for p in sorted(self.root.rglob("*.edf")):
            parts = [s.lower() for s in p.parts]
            sp = next((s for s in ("train", "eval") if s in parts), None)
            lab = next((s for s in ("normal", "abnormal") if s in parts), None)
            if sp is None or lab is None:
                continue
            out.append(TUABRecord(len(out), p.stem, p, sp, lab == "abnormal", header_age(p)))
        return [r for r in out if split is None or r.split == split]
