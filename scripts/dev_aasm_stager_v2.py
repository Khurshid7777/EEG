"""Increment 50, development only: an AASM-montage sleep stager trained on AASM scoring.
Increment 49 failed on BOAS (kappa 0.49): the CAP-trained stager (R&K S3+S4 -> N3) staged a third
of N2 as N3. Development data: OpenNeuro ds007181 (66 adults, Chongqing; full 10-20 PSG against the
contralateral mastoid; majority vote of three scorers, AASM stages). CAP nights (data/cap/aasm,
increment 49) may be pooled. No test data here (EESM19 ds005185, BOAS ds005555).

    PYTHONPATH=src python scripts/dev_aasm_stager_v2.py build data/ds007181 [--workers 2]
    PYTHONPATH=src python scripts/dev_aasm_stager_v2.py cv data/ds007181
    PYTHONPATH=src python scripts/dev_aasm_stager_v2.py train data/ds007181 CANDIDATE
"""

import argparse
import csv
import io
import json
import pickle
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://s3.amazonaws.com/openneuro.org/ds007181"
FS = 100.0
STAGE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}


def _curl(url, out=None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def subjects() -> list[str]:
    return [r["participant_id"] for r in csv.DictReader(io.StringIO(_curl(f"{BASE}/participants.tsv").decode()),
                                                        delimiter="\t")]


def _read(edf, names):
    """Selected channels at FS (only these are read: a night at 1024 Hz x 26 channels is ~3 GB)."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    from neurolens.layer1_ingest.edf import open_edf

    r = open_edf(edf)
    try:
        labels = [lb.strip().upper() for lb in r.getSignalLabels()]
        out = {}
        for n in names:
            i = labels.index(n.upper())
            fr = Fraction(FS / r.getSampleFrequency(i)).limit_denominator(1000)
            out[n] = resample_poly(r.readSignal(i), fr.numerator, fr.denominator).astype(np.float32)
    finally:
        r.close()
    return out


def _one(item):
    sub, out = item
    from neurolens.datasets.sleepedf import analysis_window
    from neurolens.evaluation.sleep_eval import FEATURE_VERSION, Night
    from neurolens.layer4_detect.sleep_staging import epoch_features

    dst = Path(out) / f"ds7181_{sub}_fv{FEATURE_VERSION}.pkl"
    if dst.exists():
        return f"cached {sub}"
    try:
        ev = list(csv.DictReader(io.StringIO(_curl(f"{BASE}/{sub}/eeg/{sub}_task-sleep_acq-PSG_events.tsv").decode()),
                                 delimiter="\t"))
        with tempfile.TemporaryDirectory(dir=out) as d:
            p = Path(d) / f"{sub}.edf"
            _curl(f"{BASE}/{sub}/eeg/{sub}_task-sleep_acq-PSG_eeg.edf", p)
            x = _read(p, ["F4-M1", "C4-M1", "O2-M1", "E1-M2", "E2-M2"])
        n = min(len(v) for v in x.values())
        frontal = x["F4-M1"][:n] - x["C4-M1"][:n]
        parietal = x["C4-M1"][:n] - x["O2-M1"][:n]
        eog = x["E1-M2"][:n] - x["E2-M2"][:n]
        n_ep = int(n / FS // 30)
        stages = [None] * n_ep
        for e in ev:
            k = int(round(float(e["onset"]) / 30))
            if 0 <= k < n_ep:
                stages[k] = STAGE.get(e["stage"].strip())
        a, b = analysis_window(stages)
        F, names = epoch_features(frontal, parietal, eog, FS, a, b)
        y = np.array([-1 if s is None else s for s in stages[a:b]], dtype=int)
        dst.write_bytes(pickle.dumps(Night(f"ds7181_{sub}", zlib.crc32(sub.encode()), F, names, y)))
        return f"ok {sub} {n_ep} epochs"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {sub}: {exc!r}"


def _nights(path: Path, prefix: str):
    # local feature caches written by this project (build() here; data/cap/aasm by increment 49)
    return [pickle.loads(p.read_bytes()) for p in sorted(path.glob(f"{prefix}_*.pkl"))]


def _cv_on(target, extra, folds=5):
    """Subject-grouped CV scored on ``target`` nights; ``extra`` nights are added to every training fold."""
    from neurolens.evaluation.sleep_eval import predict_night, score, training_matrix
    from neurolens.layer4_detect.sleep_staging import fit_stager

    preds = {}
    for k in range(folds):
        train = [n for n in target if n.subject % folds != k] + extra
        X, y, names = training_matrix(train)
        m = fit_stager(X, y, names)
        for n in target:
            if n.subject % folds == k:
                preds[n.key] = predict_night(m, n)
    s = score(target, [preds[n.key] for n in target])
    return {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "per_night_kappa_median", "f1")}


def cv(out: Path) -> dict:
    from neurolens.evaluation.sleep_eval import predict_night, score
    from neurolens.layer4_detect.sleep_staging import MulticlassTreeModel

    d7 = _nights(out, "ds7181")
    cap = _nights(Path("data/cap/aasm"), "aasm")
    m = MulticlassTreeModel.load("configs/models/sleep_stager_aasm_v1.json")
    s = score(d7, [predict_night(m, n) for n in d7])
    res = {"n_ds7181": len(d7), "n_cap": len(cap),
           "aasm_v1_on_ds7181": {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "f1")},
           "A_ds7181_only_cv": _cv_on(d7, []),
           "B_ds7181_plus_cap_cv": _cv_on(d7, cap)}
    return res


def train(out: Path, cand: str) -> Path:
    from neurolens.evaluation.sleep_eval import training_matrix
    from neurolens.layer4_detect.sleep_staging import fit_stager

    d7 = _nights(out, "ds7181")
    nights = d7 + (_nights(Path("data/cap/aasm"), "aasm") if cand == "B" else [])
    X, y, names = training_matrix(nights)
    m = fit_stager(X, y, names, {"name": "sleep_stager_aasm_v2", "candidate": cand,
                                 "training": f"OpenNeuro ds007181 ({len(d7)} nights, AASM majority vote)"
                                             + (f" + CAP ({len(nights) - len(d7)} nights)" if cand == "B" else ""),
                                 "derivations": "F4-C4 / C4-O2 / E1-E2"})
    return m.save("configs/models/sleep_stager_aasm_v2.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "cv", "train"])
    ap.add_argument("out")
    ap.add_argument("candidate", nargs="?")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        subs = subjects()
        print(f"{len(subs)} subjects", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for msg in ex.map(_one, [(s, str(out)) for s in subs]):
                print(msg, flush=True)
        return 0
    if a.cmd == "train":
        print(train(out, a.candidate))
        return 0
    res = cv(out)
    (out / "dev_cv.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
