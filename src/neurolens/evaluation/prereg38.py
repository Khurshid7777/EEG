"""Pre-registered analysis for increment 38 (docs/preregistration_increment38.md): frozen IED
detectors on vepiset (84 patients, PUMCH, China). Written before any detector was run on it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .ied_eval import aggregate

AUC_MIN = 0.80            # H1: epoch-level AUC of the learned probability
AUC_KEEP_EVIDENCE = 0.70  # decision: below -> the probability is not shown as evidence
N_BOOT = 2000


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else float("nan")


def _epochs(rows: list[dict], state: str | None = None):
    y, lp, hz, rec = [], [], [], []
    for i, r in enumerate(rows):
        for e in r["epochs"]:
            if state is None or e["state"] == state:
                y.append(e["label"]); lp.append(e["learned"]); hz.append(e["heuristic_z"]); rec.append(i)
    return np.array(y), np.array(lp), np.array(hz), np.array(rec)


def assess(rows: list[dict], seed: int = 0) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    y, lp, hz, rec = _epochs(ok)
    auc_l, auc_h = _auc(y, lp), _auc(y, hz)
    rng = np.random.default_rng(seed)
    by = [np.where(rec == i)[0] for i in range(len(ok))]
    bl, bd = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[i] for i in rng.integers(0, len(ok), len(ok))])
        a, b = _auc(y[idx], lp[idx]), _auc(y[idx], hz[idx])
        bl.append(a); bd.append(a - b)
    ci = [float(v) for v in np.nanpercentile(bl, [2.5, 97.5])]
    ci_d = [float(v) for v in np.nanpercentile(bd, [2.5, 97.5])]
    learned = aggregate([r["learned"] for r in ok])
    zs = sorted(ok[0]["heuristic"], key=float)
    curve = [{"sharpness_z": float(z), **aggregate([r["heuristic"][z] for r in ok])} for z in zs]
    by_state = {s: {"auc_learned": _auc(*_epochs(ok, s)[:2]), "n_epochs": int(_epochs(ok, s)[0].size),
                    "n_positive": int(_epochs(ok, s)[0].sum())} for s in ("Waking", "Sleeping")}
    h = {"H1_learned_epoch_auc_at_least_0.80": auc_l >= AUC_MIN,
         "H2_learned_better_than_heuristic": ci_d[0] > 0}
    return {"n_rows": len(rows), "n_scored": len(ok), "n_excluded": len(rows) - len(ok),
            "n_with_discharges": sum(r["n_ref"] > 0 for r in ok), "n_without": sum(r["n_ref"] == 0 for r in ok),
            "n_epochs": int(y.size), "n_positive_epochs": int(y.sum()),
            "auc_learned": auc_l, "auc_learned_ci95": ci, "auc_heuristic": auc_h,
            "auc_difference_ci95": ci_d, "by_state": by_state,
            "learned_at_frozen_threshold": learned,
            "heuristic_default_z6": next(c for c in curve if c["sharpness_z"] == 6.0),
            "heuristic_curve": curve, "hypotheses": h,
            "decision": {"keep_learned_probability_as_evidence": auc_l >= AUC_KEEP_EVIDENCE}}
