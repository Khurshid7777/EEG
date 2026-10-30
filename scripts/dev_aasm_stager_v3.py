"""Increment 51, development only: a multi-centre AASM-montage stager. Increments 49-50 showed that a
stager trained on one centre does not transfer (kappa 0.49-0.56). Training centres: OpenNeuro
ds007181 (Chongqing, 59 nights; features from scripts/dev_aasm_stager_v2.py), ds005555 BOAS
(Zaragoza, 128) and ds005185 EESM19 (Aarhus, 78) - the last two were test sets of increments 49-50
and are now development data. Features come from the product derivations (F4-C4 / C4-O2 / EOG).
The test centre (PhysioNet HMC) is not touched here.

    PYTHONPATH=src python scripts/dev_aasm_stager_v3.py build-boas data/aasm3
    PYTHONPATH=src python scripts/dev_aasm_stager_v3.py build-eesm19 data/aasm3
    PYTHONPATH=src python scripts/dev_aasm_stager_v3.py loco data/aasm3
    PYTHONPATH=src python scripts/dev_aasm_stager_v3.py train data/aasm3
"""

import argparse
import json
import pickle
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_boas_sleep as boas      # noqa: E402
import run_eesm19_sleep as eesm   # noqa: E402


def night_from_signal(sig, labels: list[tuple[int, int]], key: str, subject: str):
    """Product derivations (AASM montage) -> epoch features over the scored window."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    from neurolens.datasets.sleepedf import analysis_window
    from neurolens.evaluation.sleep_eval import Night
    from neurolens.layer4_detect.sleep_staging import EPOCH_S, MODEL_FS, derivations, epoch_features

    der = derivations(sig.channel_names, sig.signal, aasm=True)
    if der is None or not der.pop("aasm", False):
        raise ValueError("no AASM derivations")
    fr = Fraction(MODEL_FS / sig.sampling_rate_hz).limit_denominator(1000)
    der = {k: None if v is None else resample_poly(v, fr.numerator, fr.denominator) for k, v in der.items()}
    n_ep = int(len(der["frontal"]) / MODEL_FS // EPOCH_S)
    stages = [None] * n_ep
    for k, c in labels:
        if 0 <= k < n_ep and c >= 0:
            stages[k] = c
    a, b = analysis_window(stages)
    F, names = epoch_features(der["frontal"], der["parietal"], der["eog"], MODEL_FS, a, b)
    y = np.array([-1 if s is None else s for s in stages[a:b]], dtype=int)
    return Night(key, zlib.crc32(subject.encode()), F, names, y)


def build_boas(out: Path) -> None:
    from neurolens.layer1_ingest.registry import ingest

    for p in boas.participants():
        sub, pid = p["participant_id"], p["pid"]
        dst = out / f"boas_{sub}.pkl"
        if dst.exists():
            continue
        try:
            labels = [(int(round(float(e["onset"]) / 30)), boas.HUMAN.get(int(float(e["stage_hum"])), -1))
                      for e in boas._events(sub) if e.get("stage_hum") not in (None, "", "n/a")]
            with tempfile.TemporaryDirectory() as d:
                f = Path(d) / f"{sub}.edf"
                boas._curl(f"{boas.BASE}/{sub}/eeg/{sub}_task-Sleep_acq-psg_eeg.edf", f)
                sig = ingest(f)
            dst.write_bytes(pickle.dumps(night_from_signal(sig, labels, f"boas_{sub}", f"boas_pid{pid}")))
            print("ok", sub, flush=True)
        except Exception as exc:  # noqa: BLE001
            print("ERROR", sub, repr(exc), flush=True)


def build_eesm19(out: Path) -> None:
    from neurolens.layer1_ingest.registry import ingest

    for sub, ses, _age in eesm.nights():
        dst = out / f"eesm19_{sub}_{ses}.pkl"
        if dst.exists():
            continue
        try:
            labels = eesm._scoring(sub, ses, 1)
            stem = f"{sub}_{ses}_task-sleep_acq-PSG_eeg"
            with tempfile.TemporaryDirectory(dir="data") as d:
                for ext in (".set", ".fdt"):
                    eesm._curl(f"{eesm.BASE}/{sub}/{ses}/eeg/{stem}{ext}", Path(d) / f"{stem}{ext}")
                sig = ingest(Path(d) / f"{stem}.set")
            dst.write_bytes(pickle.dumps(night_from_signal(sig, labels, f"eesm19_{sub}_{ses}", f"eesm19_{sub}")))
            print("ok", sub, ses, flush=True)
        except Exception as exc:  # noqa: BLE001
            print("ERROR", sub, ses, repr(exc), flush=True)


def _load(paths):
    # local feature caches written by this project (build_* here; data/ds007181 by dev_aasm_stager_v2.py)
    return [pickle.loads(p.read_bytes()) for p in paths]


def centres(out: Path) -> dict:
    return {"ds007181": _load(sorted(Path("data/ds007181").glob("ds7181_*.pkl"))),
            "boas": _load(sorted(out.glob("boas_*.pkl"))),
            "eesm19": _load(sorted(out.glob("eesm19_*.pkl")))}


def loco(out: Path) -> dict:
    """Leave-one-centre-out: train on two centres, score the third."""
    from neurolens.evaluation.sleep_eval import predict_night, score, training_matrix
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = centres(out)
    res = {"n_nights": {k: len(v) for k, v in c.items()}}
    for held in c:
        train = [n for k, v in c.items() if k != held for n in v]
        X, y, names = training_matrix(train)
        m = fit_stager(X, y, names)
        s = score(c[held], [predict_night(m, n) for n in c[held]])
        res[f"held_out_{held}"] = {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "f1")}
        print(held, res[f"held_out_{held}"]["kappa"], flush=True)
    return res


def train(out: Path) -> Path:
    from neurolens.evaluation.sleep_eval import training_matrix
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = centres(out)
    X, y, names = training_matrix([n for v in c.values() for n in v])
    m = fit_stager(X, y, names, {"name": "sleep_stager_aasm_v3",
                                 "training": {k: len(v) for k, v in c.items()},
                                 "derivations": "F4-C4 / C4-O2 / horizontal EOG (product derivations)"})
    return m.save("configs/models/sleep_stager_aasm_v3.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-boas", "build-eesm19", "loco", "train"])
    ap.add_argument("out")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build-boas":
        build_boas(out)
    elif a.cmd == "build-eesm19":
        build_eesm19(out)
    elif a.cmd == "train":
        print(train(out))
    else:
        res = loco(out)
        (out / "loco.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({k: v.get("kappa", v) if isinstance(v, dict) else v for k, v in res.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
