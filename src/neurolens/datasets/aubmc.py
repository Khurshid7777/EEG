"""AUBMC epileptic EEG (Nasreddine, Mendeley Data 10.17632/5pc2j46cbc.1, CC BY 4.0): 20 long-term
video-EEG recordings (~3 h each) of 6 children (4.5-9 years) with focal epilepsy, American
University of Beirut Medical Center, 2014-2015; 19-21 scalp electrodes 10-20 (+ ECG), 500 Hz,
EDF; band-passed 1/1.6-70 Hz with a 50 Hz notch by the authors.

Seizures: the authors' ``Seizure_times.py`` (clock time of onset, duration in s) converted to
seconds from the EDF start time. Not part of the reference (``ambiguous``, neither required nor
counted as false alarms): seizures marked "no change over surface EEG" (patient 13, records 1-3)
and a seizure that the authors list once but that also lies in a neighbouring file.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .annotations import RecordAnnotation, SeizureInterval

API = "https://data.mendeley.com/public-api/datasets/5pc2j46cbc"

# Seizure_times.py (authors): {patient: {record: [(h, m, s, duration_s)]}}
SEIZURES = {
    15: {1: [(17, 18, 8, 50)], 2: [(22, 49, 24, 47)], 3: [(2, 57, 4, 13)], 4: [(5, 3, 26, 56), (6, 23, 29, 20)]},
    14: {1: [(14, 32, 2, 28), (15, 34, 32, 134)], 2: [(16, 20, 58, 32), (17, 50, 56, 10)],
         3: [(20, 20, 46, 31), (21, 2, 4, 26), (21, 27, 49, 40), (21, 50, 24, 40)]},
    13: {1: [(2, 31, 9, 52)], 2: [(3, 33, 4, 25), (4, 38, 59, 16)], 3: [(6, 45, 51, 18)],
         4: [(10, 51, 41, 30), (12, 18, 22, 24)]},
    12: {1: [(1, 56, 14, 76), (2, 20, 12, 104)], 2: [(5, 50, 55, 118)],
         3: [(6, 40, 30, 82), (7, 21, 36, 164), (8, 37, 48, 113)]},
    11: {1: [(15, 8, 55, 64)], 2: [(18, 11, 38, 51)],
         3: [(19, 18, 38, 91), (20, 6, 38, 83), (20, 53, 22, 76), (21, 27, 24, 73)], 4: [(23, 55, 35, 1358)]},
    10: {1: [(7, 36, 38, 445)], 2: [(6, 29, 14, 305)]},
}
NO_EEG_CHANGE = {(13, 1), (13, 2), (13, 3)}           # Seizures_Information.xlsx: "no change over surface EEG"
ALSO_IN = {(12, 2): [(6, 40, 30, 82)]}                 # listed under record 3, also inside record 2
AGES = {10: 5.0, 11: 4.5, 12: 8.0, 13: 9.0, 14: 7.0, 15: 8.0}


def _resolve(url: str) -> str:
    """Mendeley answers a ranged request with its redirect page: resolve the S3 URL first."""
    r = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{redirect_url}", url],
                       check=True, capture_output=True, text=True)
    return r.stdout.strip() or url


def _curl(url: str, out: Path | None = None, byte_range: str | None = None) -> bytes:
    cmd = ["curl", "-sSLf", "--retry", "5", "--retry-delay", "4"]
    if byte_range:
        url = _resolve(url)
        cmd += ["-r", byte_range]
    if out is not None:
        cmd += ["-o", str(out)]
    cmd.append(url)
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def edf_start_seconds(header: bytes) -> float:
    """Start time of day (s) from the EDF header ('hh.mm.ss' at byte 176)."""
    h, m, s = (int(v) for v in header[176:184].decode("ascii").split("."))
    return h * 3600 + m * 60 + s


def to_offset(clock: tuple[int, int, int], start_s: float) -> float:
    t = clock[0] * 3600 + clock[1] * 60 + clock[2] - start_s
    return t + 86400 if t < -3600 else t   # recordings crossing midnight


class AubmcClient:
    def __init__(self, cache: str | Path = "data/aubmc"):
        self.cache = Path(cache)

    def _meta(self) -> dict:
        p = self.cache / "meta.json"
        if not p.exists():
            self.cache.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_curl(API))
        return json.loads(p.read_text())

    def files(self) -> dict[str, dict]:
        return {f["filename"]: f for f in self._meta()["files"] if f["filename"].endswith(".edf")}

    def header(self, name: str) -> bytes:
        p = self.cache / "headers" / (name + ".hdr")
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(_curl(self.files()[name]["content_details"]["download_url"], byte_range="0-255"))
        return p.read_bytes()

    def annotations(self) -> list[RecordAnnotation]:
        out = []
        for name in sorted(self.files()):
            pat, rec = int(name[1:3]), int(name.split("Record")[1].split(".")[0])
            start = edf_start_seconds(self.header(name))
            iv = [SeizureInterval(to_offset((h, m, s), start), to_offset((h, m, s), start) + d)
                  for h, m, s, d in SEIZURES[pat].get(rec, [])]
            amb = [SeizureInterval(to_offset((h, m, s), start), to_offset((h, m, s), start) + d)
                   for h, m, s, d in ALSO_IN.get((pat, rec), [])]
            if (pat, rec) in NO_EEG_CHANGE:
                amb, iv = amb + iv, []
            out.append(RecordAnnotation(database="aubmc", subject=f"p{pat}", file=f"p{pat}/{name}", seizures=iv,
                                        ambiguous=amb))
        return out

    def local_path(self, file: str) -> Path:
        return self.cache / "edf" / Path(file).name

    def fetch(self, file: str) -> Path:
        """``file``: an annotation's ``p<NN>/<name>.edf`` (or the bare name)."""
        name = Path(file).name
        p = self.local_path(name)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            part = p.with_suffix(".part")
            _curl(self.files()[name]["content_details"]["download_url"], out=part)
            if part.stat().st_size != self.files()[name]["size"]:
                raise OSError(f"{name}: size mismatch")
            part.replace(p)
        return p
