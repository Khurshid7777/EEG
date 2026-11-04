"""Increment 56 (pre-registered, docs/preregistration_increment56.md): the AASM-montage sleep stager (v4 +
transition smoothing) on COVISLEEP (Zenodo 10527276, CC BY 4.0): 93 overnight polysomnograms of adults with
chronic insomnia, 31 of them after long COVID (IDs 'S*'), 62 without ('T*'); stages scored by trained sleep
physicians following AASM; Nox A1 recorder, F4-M1 / C4-M1 / O2-M1 / E1-M2 / E2-M1 among 80 channels.
The EDF files are read from the 24.8 GB zip by HTTP range requests, one record at a time, and deleted after use
(disk is small). Product path (Pipeline._sleep_staging); one run gives the smoothed and the plain hypnogram.
Resumable: one JSON line per record.

    PYTHONPATH=src python scripts/run_covisleep_sleep.py check T1_29            # excluded record, sanity only
    PYTHONPATH=src python scripts/run_covisleep_sleep.py run OUT.jsonl
    PYTHONPATH=src python scripts/run_covisleep_sleep.py assess OUT.jsonl OUT.json
"""

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hmc_sleep as hmc  # noqa: E402  (scoring helpers)

API = "https://zenodo.org/api/records/10527276"
MODEL = "configs/models/sleep_stager_aasm_v4.json"
MODEL_SHA256 = "3affd3b5bb9c996f56259fe85817065ddc6ee4fbe29a57fd3a81f985e7b102d7"
SMOOTHING = "configs/models/sleep_smoothing_aasm_v1.json"
SMOOTHING_SHA256 = "be614d1c8c95709734df5bb14d396c08fbd88eff98713f853f4f2fb7841a851f"
EXCLUDED = {"T1_29"}      # opened before the pre-registration (reader and units check)
SEED, N_BOOT, WORKERS = 56, 2000, 4
MAX_OFFSET_S, MIN_INSIDE = 45.0, 0.90
STAGE = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
STAGES = ["W", "N1", "N2", "N3", "REM"]
KEEP = {"F3", "F4", "C3", "C4", "O1", "O2", "E1", "E2", "M1", "M2"}   # normalised names; results identical to the full read (checked on T1_29)


def _files() -> dict:
    d = json.load(urllib.request.urlopen(API, timeout=60))
    return {f["key"]: (f["links"]["self"], f["size"]) for f in d["files"]}


def _http(url: str) -> bytes:
    return urllib.request.urlopen(url, timeout=300).read()


def _zip(url: str, size: int) -> zipfile.ZipFile:
    from neurolens.datasets.nmt_events import HttpRangeFile

    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(url, size), buffer_size=32 << 20))


def records() -> list[str]:
    files = _files()
    z = _zip(*files["raw_edfs.zip"])
    names = sorted(Path(i.filename).name.removesuffix("_PSG.edf") for i in z.infolist() if i.filename.lower().endswith(".edf"))
    return [n for n in names if n not in EXCLUDED]


def _clock(text: str) -> dt.datetime:
    """'YYYY-MM-DD HH:MM:SS' (60 files) or 'HH:MM:SS' (39 files; the date is irrelevant, only the time of day is used)."""
    return dt.datetime.strptime(text.strip(), "%Y-%m-%d %H:%M:%S" if len(text.strip()) > 10 else "%H:%M:%S")


def _stage(name: str):
    n = name.strip()
    if n in STAGE:
        return STAGE[n]
    return 0 if n.lower().endswith("veil") else None   # 'Éveil' (French 'wake'), encoding of the accent varies


def parse_hypnogram(text: str) -> tuple[list[tuple[int, int]], dt.datetime]:
    """[(0-based epoch, stage)] and the wall-clock time of epoch 0. Columns are read by header name. Some files
    start at a later epoch ID (the first rows have no stage), one file has no 'Epoch ID' column (row order is used):
    epoch 0 is therefore the time of any row minus (epoch number - 1) x 30 s."""
    rows = list(csv.reader(text.splitlines()))
    col = {h.strip(): i for i, h in enumerate(rows[0])}
    lab, t0 = [], None
    for n, r in enumerate(rows[1:], start=1):
        if len(r) <= col["Time"]:
            continue
        k = int(r[col["Epoch ID"]]) if "Epoch ID" in col else n
        if t0 is None:
            t0 = _clock(r[col["Time"]]) - dt.timedelta(seconds=30 * (k - 1))
        st = _stage(r[col["Sleep Stage"]]) if len(r) > col["Sleep Stage"] else None
        if st is not None:
            lab.append((k - 1, st))
    return lab, t0


def edf_start(path: Path) -> dt.time:
    h = path.open("rb").read(256)
    return dt.datetime.strptime(h[176:184].decode().strip(), "%H.%M.%S").time()


def offset_seconds(first: dt.datetime, start: dt.time) -> float:
    a = first.hour * 3600 + first.minute * 60 + first.second
    b = start.hour * 3600 + start.minute * 60 + start.second
    return float(((a - b + 43200) % 86400) - 43200)


def _one(args) -> dict:
    rec, files, ann_bytes = args
    from neurolens.layer1_ingest.edf import EdfIngestor
    from neurolens.pipeline.pipeline import Pipeline

    try:
        ann = zipfile.ZipFile(io.BytesIO(ann_bytes))
        member = next(n for n in ann.namelist() if n.endswith(f"/{rec}_Hypnogramme.csv") and "checkpoint" not in n)
        lab, first = parse_hypnogram(ann.read(member).decode("utf-8", "replace"))
        d = tempfile.mkdtemp(prefix="covisleep_")
        try:
            z = _zip(*files["raw_edfs.zip"])
            p = Path(d) / f"{rec}.edf"
            with z.open(f"edfs/{rec}_PSG.edf") as src, p.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            off = offset_seconds(first, edf_start(p))
            sig = EdfIngestor().read(p, include=lambda n: n in KEEP)   # EEG/EOG only: the 80-channel files need ~10x the memory
        finally:
            shutil.rmtree(d, ignore_errors=True)
        dur_ep = sig.duration_s / 30.0
        shift = 0 if abs(off) <= MAX_OFFSET_S else int(round(off / 30.0))
        lab = [(k + shift, c) for k, c in lab]
        inside = float(np.mean([0 <= k < int(dur_ep) for k, _ in lab])) if lab else 0.0
        if inside < MIN_INSIDE:
            return {"record": rec, "error": f"only {inside:.0%} of labelled epochs fall inside the recording (shift {shift})"}
        pipe = Pipeline(provider_pref="template", run_ica=False)
        pipe.cfg.sleep = pipe.cfg.sleep.model_copy(update={"aasm_model": MODEL, "aasm_smoothing": SMOOTHING})
        st = pipe._sleep_staging(sig)
        if not st or st.get("skipped") or "hypnogram_unsmoothed" not in st:
            return {"record": rec, "error": f"not staged / not smoothed: {str(st)[:200]}"}
        return {"record": rec, "group": rec[0], "derivations": st["derivations"], "offset_s": off, "shift_epochs": shift,
                "n_ann": len(lab), "duration_epochs": round(dur_ep, 1),
                "hyp": [STAGES.index(s) for s in st["hypnogram"]], "hyp_raw": [STAGES.index(s) for s in st["hypnogram_unsmoothed"]],
                "human": lab}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"record": rec, "error": repr(exc)}


def _score(rows, key) -> dict:
    return hmc._score([{**r, "hyp": r[key]} for r in rows])


def assess(path: Path) -> dict:
    allrows = hmc._load(path)
    rows = [r for r in allrows if "error" not in r]
    sm, raw = _score(rows, "hyp"), _score(rows, "hyp_raw")
    rng = np.random.default_rng(SEED)
    k_ci, g_ci = [], []
    for _ in range(N_BOOT):
        b = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        s, r = _score(b, "hyp")["kappa"], _score(b, "hyp_raw")["kappa"]
        k_ci.append(s)
        g_ci.append(s - r)
    ci = [round(float(q), 3) for q in np.percentile(k_ci, [2.5, 97.5])]
    gci = [round(float(q), 4) for q in np.percentile(g_ci, [2.5, 97.5])]
    gain = sm["kappa"] - raw["kappa"]
    per = [_score([r], "hyp")["kappa"] for r in rows]
    by_group = {g: round(_score([r for r in rows if r["group"] == g], "hyp")["kappa"], 3) for g in ("S", "T")}
    kappa = sm["kappa"]
    status = "confirmed" if (kappa >= 0.60 and ci[0] >= 0.50) else ("weakened" if kappa >= 0.50 else "failed")
    smoothing = "confirmed" if (gain >= 0.005 and gci[0] > 0) else ("harmful" if gci[1] < 0 else "neutral")
    return {"n_recordings": len(rows), "errors": [(r["record"], r["error"]) for r in allrows if "error" in r],
            "kappa_smoothed": kappa, "kappa_ci95_by_recording": ci, "kappa_unsmoothed": raw["kappa"],
            "gain": round(gain, 4), "gain_ci95": gci, "per_recording_kappa_median": round(float(np.nanmedian(per)), 3),
            "kappa_by_group": by_group, "accuracy": sm["accuracy"], "macro_f1": sm["macro_f1"],
            "f1_smoothed": sm["f1"], "f1_unsmoothed": raw["f1"], "confusion": sm["confusion"],
            "n_epochs": sm["n_epochs"], "aasm_path": status, "smoothing": smoothing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "run", "assess"])
    ap.add_argument("args", nargs="+")
    a = ap.parse_args()
    if a.cmd == "assess":
        res = assess(Path(a.args[0]))
        Path(a.args[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k not in ("confusion",)}, indent=1, default=float))
        return 0
    for p, h in ((MODEL, MODEL_SHA256), (SMOOTHING, SMOOTHING_SHA256)):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise RuntimeError(f"{p} does not match the frozen sha256")
    files = _files()
    ann = _http(files["annotations.zip"][0])
    if a.cmd == "check":
        r = _one((a.args[0], files, ann))
        if "error" in r:
            print(r)
            return 1
        for key in ("hyp", "hyp_raw"):
            print(key, "kappa", round(_score([r], key)["kappa"], 3))
        print({k: r[k] for k in ("offset_s", "shift_epochs", "n_ann", "duration_epochs", "derivations")})
        return 0
    out = Path(a.args[0])
    workers = int(a.args[1]) if len(a.args) > 1 else WORKERS   # a 800 MB EDF needs ~6.5 GB: four workers ran out of memory
    recs = records()
    done = {r["record"] for r in hmc._load(out)} if out.exists() else set()
    todo = [r for r in recs if r not in done]
    print(f"{len(recs)} recordings, {len(todo)} to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, [(t, files, ann) for t in todo]):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("record", "offset_s", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
