"""vepiset (Lin et al., Scientific Data 2025; figshare 10.6084/m9.figshare.28069568.v2, CC BY):
20 min of routine EEG from each of 84 patients with interictal epileptiform discharges (IEDs),
Peking Union Medical College Hospital (China); 500 Hz, 19 10-20 electrodes + PG1/PG2, A1/A2,
ECG and EMG pairs; MAT files with ``eeg_data`` (29 x samples) and ``events`` (time, -, text):
``!`` a discharge at that moment, ``!start`` / ``!end`` a run of discharges, ``Waking`` /
``Sleeping`` the state until the next state label.

The archive is one 17 GB zip; members are read with HTTP range requests (the zip central
directory, then each member) so that only the file being analysed is on disk.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/28069568"
CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5",
            "T6", "Fz", "Cz", "Pz", "PG1", "PG2", "A1", "A2", "ECG1", "ECG2", "EMG1", "EMG2", "EMG3", "EMG4"]
FS = 500.0


def _download_url() -> str:
    with urllib.request.urlopen(ARTICLE, timeout=60) as r:
        return json.load(r)["files"][0]["download_url"]


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (re-resolving short-lived redirects)."""

    def __init__(self, url: str, size: int | None = None, retries: int = 5):
        self.url, self.pos, self.retries = url, 0, retries
        self.size = size if size is not None else self._size()

    def _get(self, start: int, end: int) -> bytes:
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise OSError(f"short range read: {len(data)} of {end - start + 1}")
                return data
            except Exception:  # network: retry, then raise
                if attempt == self.retries:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise AssertionError("unreachable")

    def _size(self) -> int:
        req = urllib.request.Request(self.url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers["Content-Range"].split("/")[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.size - 1, self.pos + n - 1)
        data = self._get(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


@dataclass
class VepiRecord:
    record_id: int
    name: str          # zip member name


@dataclass
class VepiAnnotation:
    discharges: list[tuple[float, float]]   # reference IED events (s)
    states: list[tuple[float, str]]         # (time, "Waking" | "Sleeping")


def parse_events(events) -> VepiAnnotation:
    """Points ``!`` -> [t, t]; ``!start``..``!end`` -> [start, end] (a start without an end is
    a point). Times in seconds from the first sample."""
    import numpy as np

    rows = []
    for e in np.atleast_1d(events) if not isinstance(events, list) else events:
        e = list(np.atleast_1d(e)) if not isinstance(e, (list, tuple)) else list(e)
        t = float(np.asarray(e[0]).ravel()[0])
        txt = str(np.asarray(e[-1]).ravel()[0]).strip()
        rows.append((t, txt))
    rows.sort()
    out, states, open_start = [], [], None
    for t, txt in rows:
        low = txt.lower()
        if low in ("waking", "sleeping"):
            states.append((t, txt.capitalize()))
        elif low.startswith("!start"):
            if open_start is not None:
                out.append((open_start, open_start))
            open_start = t
        elif low.startswith("!end"):
            if open_start is not None:
                out.append((open_start, t))
                open_start = None
        elif low.startswith("!"):
            out.append((t, t))
    if open_start is not None:
        out.append((open_start, open_start))
    return VepiAnnotation(sorted(out), states)


class VepisetClient:
    def __init__(self, cache: str | Path = "data/vepiset"):
        self.cache = Path(cache)
        self._zip = None

    def _zipfile(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(_download_url()), buffer_size=1 << 20))
        return self._zip

    def records(self) -> list[VepiRecord]:
        idx = self.cache / "members.json"
        if idx.exists():
            names = json.loads(idx.read_text())
        else:
            names = sorted(n for n in self._zipfile().namelist() if n.lower().endswith(".mat"))
            self.cache.mkdir(parents=True, exist_ok=True)
            idx.write_text(json.dumps(names))
        return [VepiRecord(i, n) for i, n in enumerate(names)]

    def fetch(self, rec: VepiRecord) -> Path:
        p = self.cache / "mat" / Path(rec.name).name
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            part = p.with_suffix(".part")
            # a fresh signed URL per member: figshare redirects expire within seconds
            self._zip = None
            with self._zipfile().open(rec.name) as src, open(part, "wb") as dst:
                while chunk := src.read(1 << 22):
                    dst.write(chunk)
            part.replace(p)
        return p

    def drop(self, rec: VepiRecord) -> None:
        (self.cache / "mat" / Path(rec.name).name).unlink(missing_ok=True)

    @staticmethod
    def load(path: Path):
        """(signal µV? [29 x n], annotation) - units are checked by the caller."""
        from scipy.io import loadmat

        m = loadmat(str(path))
        return m["eeg_data"], parse_events(m["events"])
