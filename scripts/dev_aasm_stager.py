"""Increment 49, development only: sleep staging on the standard AASM PSG montage (F4/C4/O2 against
a mastoid; no Fp, Cz, Pz, P4), which the product does not stage today. CAP Sleep Database records
with a full parasagittal chain (96) - all CAP data counts as development; the test set is separate.

Derivations: frontal = F4-C4, parietal = C4-O2 (= (C4-P4) + (P4-O2) in CAP's bipolar chain),
EOG = ROC-LOC; left chain if the right one is missing.

    PYTHONPATH=src python scripts/dev_aasm_stager.py build data/cap/aasm [--workers 2]
    PYTHONPATH=src python scripts/dev_aasm_stager.py cv data/cap/aasm
    PYTHONPATH=src python scripts/dev_aasm_stager.py train data/cap/aasm   # final model, all 96 nights
"""

import argparse
import json
import pickle
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RIGHT = ("F4-C4", "C4-P4", "P4-O2")
LEFT = ("F3-C3", "C3-P3", "P3-O1")
EOG = "ROC-LOC"
FS = 100.0


def usable(labels) -> bool:
    up = {lb.strip().upper() for lb in labels}
    return (set(RIGHT) <= up or set(LEFT) <= up) and EOG in up


def read_aasm(edf_path, txt):
    from fractions import Fraction

    from scipy.signal import resample_poly

    from neurolens.datasets.capsleep import parse_scoring
    from neurolens.datasets.sleepedf import EPOCH_S, analysis_window
    from neurolens.layer1_ingest.edf import open_edf

    r = open_edf(edf_path)
    try:
        labels = [lb.strip().upper() for lb in r.getSignalLabels()]
        chain = RIGHT if set(RIGHT) <= set(labels) else LEFT

        def get(name):
            i = labels.index(name)
            x = r.readSignal(i)
            fr = Fraction(FS / r.getSampleFrequency(i)).limit_denominator(1000)
            return resample_poly(x, fr.numerator, fr.denominator).astype(np.float32)

        fc, cp, po = (get(n) for n in chain)
        eog = get(EOG)
        start = r.getStartdatetime().strftime("%H:%M:%S")
    finally:
        r.close()
    n = min(len(fc), len(cp), len(po), len(eog))
    stages = parse_scoring(txt, start, int(n / FS // EPOCH_S))
    a, b = analysis_window(stages)
    return {"frontal": fc[:n], "parietal": cp[:n] + po[:n], "eog": eog[:n]}, stages, a, b


def _one(item):
    name, out = item
    from neurolens.datasets.physionet import PhysioNetClient
    from neurolens.evaluation.sleep_eval import FEATURE_VERSION, Night
    from neurolens.layer4_detect.sleep_staging import epoch_features

    dst = Path(out) / f"aasm_{name.removesuffix('.edf')}_fv{FEATURE_VERSION}.pkl"
    if dst.exists():
        return f"cached {name}"
    c = PhysioNetClient("capsleep", "data/cap")
    try:
        edf = c.fetch(name)
        sig, stages, a, b = read_aasm(edf, c.text(name.removesuffix(".edf") + ".txt"))
        F, names = epoch_features(sig["frontal"], sig["parietal"], sig["eog"], FS, a, b)
        y = np.array([-1 if s is None else s for s in stages[a:b]], dtype=int)
        night = Night(name.removesuffix(".edf"), zlib.crc32(name.encode()), F, names, y)
        dst.write_bytes(pickle.dumps(night))
        edf.unlink(missing_ok=True)
        return f"ok {name}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {name}: {exc!r}"


def records() -> list[str]:
    from neurolens.datasets.physionet import PhysioNetClient
    from neurolens.evaluation.sleep_eval import cap_headers

    h = cap_headers(PhysioNetClient("capsleep", "data/cap"))
    return sorted((r for r, lb in h.items() if usable(lb)), key=lambda r: (re.match(r"[a-z]+", r).group(0),
                                                                            int(re.search(r"\d+", r).group(0))))


def _nights(out: Path):
    # local feature caches written by build() above in this session
    return [pickle.loads(p.read_bytes()) for p in sorted(out.glob("aasm_*.pkl"))]


def cv(out: Path) -> dict:
    from neurolens.evaluation.sleep_eval import grouped_cv, predict_night, score
    from neurolens.layer4_detect.sleep_staging import MulticlassTreeModel

    nights = _nights(out)
    res = {"n_nights": len(nights)}
    for name, path in (("v1_zero_shot", "configs/models/sleep_stager_v1.json"),
                       ("clinical_v1_zero_shot", "configs/models/sleep_stager_clinical_v1.json")):
        m = MulticlassTreeModel.load(path)
        s = score(nights, [predict_night(m, n) for n in nights])
        res[name] = {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "per_night_kappa_median")}
    s = grouped_cv(nights, folds=5, progress=print)
    res["aasm_grouped_cv"] = {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "per_night_kappa_median")}
    return res


def train(out: Path) -> Path:
    from neurolens.evaluation.sleep_eval import training_matrix
    from neurolens.layer4_detect.sleep_staging import fit_stager

    nights = _nights(out)
    X, y, names = training_matrix(nights)
    m = fit_stager(X, y, names, {"name": "sleep_stager_aasm_v1", "training": f"CAP, {len(nights)} nights, "
                                 "F4-C4 / C4-O2 / ROC-LOC (left chain if the right is missing)",
                                 "records": [n.key for n in nights]})
    return m.save("configs/models/sleep_stager_aasm_v1.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "cv", "train"])
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build":
        recs = records()
        print(f"{len(recs)} usable records", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for msg in ex.map(_one, [(r, str(out)) for r in recs]):
                print(msg, flush=True)
        return 0
    if a.cmd == "train":
        print(train(out))
        return 0
    res = cv(out)
    (out / "dev_cv.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
