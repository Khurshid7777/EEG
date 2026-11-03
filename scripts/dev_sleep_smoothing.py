"""Increment 55, development only: does Viterbi smoothing of the stager's stage probabilities along
the night (transition matrix learned from the training centres) raise agreement with experts?
Leave-one-centre-out on the four development centres (data/aasm3, written by dev_aasm_stager_v3/v4);
the fold models and their probabilities are cached, so the smoothing weight can be varied cheaply.
The transition matrix of a fold comes from that fold's training centres only.

    PYTHONPATH=src python scripts/dev_sleep_smoothing.py fit data/aasm3     # fold models -> probabilities
    PYTHONPATH=src python scripts/dev_sleep_smoothing.py smooth data/aasm3  # weights, per-centre kappa
    PYTHONPATH=src python scripts/dev_sleep_smoothing.py freeze data/aasm3  # weight 0.5 + transitions of all nights
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_aasm_stager_v4 as v4  # noqa: E402

BETAS = (0.0, 0.25, 0.5, 1.0, 1.5)


def transitions(nights) -> np.ndarray:
    """Row-stochastic 5x5 matrix of consecutive scored epochs."""
    A = np.ones((5, 5))  # add-one smoothing
    for n in nights:
        y = n.y
        ok = (y[:-1] >= 0) & (y[1:] >= 0)
        np.add.at(A, (y[:-1][ok], y[1:][ok]), 1)
    return A / A.sum(axis=1, keepdims=True)


def fit(out: Path) -> None:
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = v4.centres(out)
    folds = {}
    for held in c:
        X, y, names = v4._matrix([n for k, v in c.items() if k != held for n in v], v4.design_b)
        m = fit_stager(X, y, names)
        folds[held] = {"A": transitions([n for k, v in c.items() if k != held for n in v]),
                       "proba": [m.predict_proba(v4.design_b(n)[0]) for n in c[held]]}
        print("fold", held, flush=True)
    (out / "smoothing_folds.pkl").write_bytes(pickle.dumps(folds))


def smooth(out: Path) -> dict:
    from neurolens.evaluation.sleep_eval import score
    from neurolens.layer4_detect.sleep_staging import viterbi_smooth

    c = v4.centres(out)
    # local cache written by fit() above
    folds = pickle.loads((out / "smoothing_folds.pkl").read_bytes())
    res = {}
    for beta in BETAS:
        for held, f in folds.items():
            preds = [p.argmax(axis=1) if beta == 0 else viterbi_smooth(p, f["A"], beta) for p in f["proba"]]
            s = score(c[held], preds)
            res[f"beta{beta}_{held}"] = {"kappa": s["kappa"], "f1": s["f1"]}
        res[f"beta{beta}_mean"] = round(float(np.mean([res[f"beta{beta}_{h}"]["kappa"] for h in folds])), 4)
        print(beta, {h: res[f"beta{beta}_{h}"]["kappa"] for h in folds}, res[f"beta{beta}_mean"], flush=True)
    return res


def freeze(out: Path, weight: float = 0.5) -> Path:
    c = v4.centres(out)
    A = transitions([n for v in c.values() for n in v])
    path = Path("configs/models/sleep_smoothing_aasm_v1.json")
    path.write_text(json.dumps({
        "weight": weight, "stages": ["W", "N1", "N2", "N3", "REM"], "transitions": A.round(6).tolist(),
        "meta": {"name": "sleep_smoothing_aasm_v1", "nights": {k: len(v) for k, v in c.items()},
                 "selection": "docs/increment55_dev_smoothing.json: weight 0.5 = best mean leave-one-centre-out kappa "
                              "(0.610 -> 0.627), gain on every centre"}}, indent=1) + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "smooth", "freeze"])
    ap.add_argument("out")
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "fit":
        fit(out)
    elif a.cmd == "freeze":
        print(freeze(out))
    else:
        res = smooth(out)
        (out / "smoothing.json").write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
