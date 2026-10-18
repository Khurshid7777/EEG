"""Pre-registered analysis for increment 33 (docs/preregistration_increment33.md): the frozen
routine-EEG verdict on OpenNeuro ds004504 (Greece; healthy elderly vs Alzheimer's disease /
frontotemporal dementia). Written and committed before any EEG file was downloaded.
"""

from __future__ import annotations

from typing import Any

from .prereg import clopper_pearson

SPECIFICITY_MIN = 0.65  # controls without "abnormal"
AUC_AD_MIN = 0.70       # record-level score, Alzheimer's disease vs controls


def _auc(pos, neg) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg)))


def assess(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r and r.get("record_probability") is not None]
    by = {g: [r for r in ok if r["group"] == g] for g in ("C", "A", "F")}
    if not by["C"] or not by["A"]:
        raise ValueError("need scored controls and Alzheimer's recordings")
    C = by["C"]
    tn = sum(not r["impression_abnormal"] for r in C)
    spec = tn / len(C)
    auc_a = _auc([r["record_probability"] for r in by["A"]], [r["record_probability"] for r in C])
    groups = {g: {"n": len(R), "abnormal_verdicts": sum(bool(r["impression_abnormal"]) for r in R),
                  "rule_abnormal": sum(bool(r["rule_impression_abnormal"]) for r in R),
                  "borderline": sum(bool(r.get("record_borderline")) for r in R)} for g, R in by.items()}
    h = {"H1_control_specificity_at_least_0.65": spec >= SPECIFICITY_MIN,
         "H2_auc_alzheimer_vs_controls_at_least_0.70": auc_a >= AUC_AD_MIN}
    return {
        "n_rows": len(rows), "n_scored": len(ok), "n_excluded": len(rows) - len(ok),
        "control_specificity": spec, "control_specificity_ci95": list(clopper_pearson(tn, len(C))),
        "rule_control_specificity": sum(not r["rule_impression_abnormal"] for r in C) / len(C),
        "auc_alzheimer_vs_controls": auc_a,
        "auc_ftd_vs_controls": _auc([r["record_probability"] for r in by["F"]],
                                    [r["record_probability"] for r in C]) if by["F"] else None,
        "groups": groups, "hypotheses": h,
    }
