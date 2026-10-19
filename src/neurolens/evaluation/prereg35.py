"""Pre-registered analysis for increment 35 (docs/preregistration_increment35.md): the frozen
routine-EEG verdict on I-CARE (comatose adults after cardiac arrest, 7 hospitals) - sensitivity
on recordings expected to be abnormal, and the record score against neurological outcome.
Written and committed before any EEG signal was downloaded.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .prereg import clopper_pearson

SENSITIVITY_MIN = 0.80   # share of "abnormal" verdicts, amplitude-plausible hospitals
AUC_OUTCOME_MIN = 0.60   # record score, poor (CPC 3-5) vs good (CPC 1-2) outcome
RULE_GAP = 0.10          # rules alone more sensitive by more than this -> severe-finding gate
AMPLITUDE_UV = (1.0, 300.0)  # hospital median of per-record median channel SD; outside -> scale unknown


def _auc(pos, neg) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg)))


def _sens(rows: list[dict]) -> dict:
    tp = sum(bool(r["impression_abnormal"]) for r in rows)
    n = len(rows)
    return {"n": n, "abnormal_verdicts": tp, "sensitivity": tp / n if n else float("nan"),
            "sensitivity_ci95": list(clopper_pearson(tp, n)) if n else None,
            "rule_sensitivity": sum(bool(r["rule_impression_abnormal"]) for r in rows) / n if n else float("nan"),
            "borderline": sum(bool(r.get("record_borderline")) for r in rows)}


def assess(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r and r.get("record_probability") is not None]
    hospitals = sorted({r["hospital"] for r in ok})
    amp = {h: float(np.median([r["amplitude_sd_uv"] for r in ok if r["hospital"] == h])) for h in hospitals}
    plausible = {h for h, a in amp.items() if AMPLITUDE_UV[0] <= a <= AMPLITUDE_UV[1]}
    main = [r for r in ok if r["hospital"] in plausible]
    S = _sens(main)
    poor = [r["record_probability"] for r in main if r.get("poor_outcome") is True]
    good = [r["record_probability"] for r in main if r.get("poor_outcome") is False]
    auc = _auc(poor, good) if poor and good else float("nan")
    h = {"H1_sensitivity_at_least_0.80": S["sensitivity"] >= SENSITIVITY_MIN,
         "H2_auc_poor_vs_good_outcome_at_least_0.60": auc >= AUC_OUTCOME_MIN}
    return {
        "n_rows": len(rows), "n_scored": len(ok), "n_excluded": len(rows) - len(ok),
        "hospital_amplitude_sd_uv": amp, "amplitude_plausible_hospitals": sorted(plausible),
        "main": S, "all_hospitals": _sens(ok),
        "by_hospital": {h: _sens([r for r in ok if r["hospital"] == h]) for h in hospitals},
        "by_outcome": {k: _sens([r for r in main if r.get("poor_outcome") is v]) for k, v in (("poor", True), ("good", False))},
        "auc_poor_vs_good": auc, "n_poor": len(poor), "n_good": len(good),
        "hypotheses": h,
        "decision": {"severe_finding_gate": S["rule_sensitivity"] - S["sensitivity"] > RULE_GAP},
    }
