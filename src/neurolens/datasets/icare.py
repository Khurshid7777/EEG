"""I-CARE (Amorim et al., 2023; PhysioNet, CC BY-NC-SA 4.0): EEG of 607 comatose adults after
cardiac arrest, 7 hospitals (Netherlands, Belgium, USA) coded A-F, hourly WFDB segments named
``PPPP_SSS_HHH_EEG`` (HHH = hour after return of circulation), per-patient ``PPPP.txt`` with age,
sex, hospital and outcome (CPC 1-2 good, 3-5 poor).

Increment 35 analyses one segment per patient (``select_segment``) and downloads only its
first ``minutes`` with an HTTP range request (format 16 is sample-interleaved).
"""

from __future__ import annotations

import http.client
import math
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BASE = "https://physionet.org/files/i-care/2.1"
TARGET_HOUR = 24
MIN_MINUTES = 20


@dataclass
class IcareRecord:
    record_id: int
    patient: str
    meta: dict

    @property
    def age_years(self) -> float | None:
        try:
            age = float(self.meta.get("Age", ""))
        except ValueError:
            return None
        return None if math.isnan(age) else age  # "Age: nan" = not recorded

    @property
    def hospital(self) -> str:
        return self.meta.get("Hospital", "")

    @property
    def poor_outcome(self) -> bool | None:
        o = self.meta.get("Outcome", "")
        return {"Poor": True, "Good": False}.get(o)


def _parse_meta(text: str) -> dict:
    return {k.strip(): v.strip() for k, _, v in (ln.partition(":") for ln in text.splitlines() if ":" in ln)}


def segments(listing_html: str) -> list[tuple[str, int]]:
    """EEG segment stems and their hour after ROSC from a PhysioNet directory listing."""
    stems = sorted(set(re.findall(r'href="(\d+_\d+_(\d+)_EEG)\.hea"', listing_html)))
    return [(s, int(h)) for s, h in stems]


def candidate_order(segs: list[tuple[str, int]], target: int = TARGET_HOUR) -> list[str]:
    """Segments by distance of their hour from ``target`` (ties: the earlier hour first)."""
    return [s for s, h in sorted(segs, key=lambda x: (abs(x[1] - target), x[1], x[0]))]


class IcareClient:
    base = BASE

    def __init__(self, cache: str | Path = "data/icare"):
        self.cache = Path(cache)

    def _url(self, key: str, *, byte_range: int | None = None, retries: int = 5) -> bytes:
        req = urllib.request.Request(f"{self.base}/{key}")
        if byte_range is not None:
            req.add_header("Range", f"bytes=0-{byte_range - 1}")
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = r.read()
                if byte_range is not None and len(data) < byte_range and r.status == 206:
                    raise OSError(f"truncated range transfer: {len(data)} of {byte_range} bytes")
                return data
            except (OSError, http.client.HTTPException):  # incl. IncompleteRead mid-transfer
                if attempt == retries:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")

    def _get(self, key: str, *, byte_range: int | None = None) -> Path:
        p = self.cache / key
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            part = p.with_suffix(p.suffix + ".part")
            part.write_bytes(self._url(key, byte_range=byte_range))
            part.replace(p)
        return p

    def records(self) -> list[IcareRecord]:
        ids = [ln.strip().rstrip("/").split("/")[-1] for ln in self._get("RECORDS").read_text().splitlines() if ln.strip()]
        return [IcareRecord(i, pid, {}) for i, pid in enumerate(ids)]

    def with_meta(self, rec: IcareRecord) -> IcareRecord:
        text = self._get(f"training/{rec.patient}/{rec.patient}.txt").read_text()
        return IcareRecord(rec.record_id, rec.patient, _parse_meta(text))

    def select_segment(self, rec: IcareRecord, minutes: int = MIN_MINUTES) -> tuple[str, int, dict] | None:
        """The segment closest to hour 24 with at least ``minutes`` of EEG (header only)."""
        from ..layer1_ingest.wfdb import parse_header

        html = self._url(f"training/{rec.patient}/").decode("utf-8", "replace")
        segs = dict(segments(html))
        for stem in candidate_order(list(segs.items())):
            h = parse_header(self._get(f"training/{rec.patient}/{stem}.hea"))
            if h["n_samp"] and h["n_samp"] >= minutes * 60 * h["fs"]:
                return stem, segs[stem], h
        return None

    def fetch(self, rec: IcareRecord, stem: str, header: dict, minutes: int = MIN_MINUTES) -> Path:
        """Header plus the first ``minutes`` of the signal file; returns the ``.hea`` path."""
        s0 = header["signals"][0]
        n_bytes = s0["offset"] + 2 * header["n_sig"] * int(minutes * 60 * header["fs"])
        self._get(f"training/{rec.patient}/{s0['file']}", byte_range=n_bytes)
        return self.cache / "training" / rec.patient / f"{stem}.hea"

    def drop(self, rec: IcareRecord) -> None:
        """Remove the patient's signal files (headers and metadata stay cached)."""
        for p in (self.cache / "training" / rec.patient).glob("*.mat"):
            p.unlink()
