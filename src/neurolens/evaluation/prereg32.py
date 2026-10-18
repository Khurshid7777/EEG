"""Pre-registered analysis for increment 32 (docs/preregistration_increment32.md): the frozen
record-level verdict on the TUH Abnormal EEG Corpus evaluation split (another country,
hospital, device and population), never seen by this project.

Written and committed before the corpus was downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .prereg import clopper_pearson

AUC_MIN = 0.80
SPECIFICITY_MIN = 0.65


def _auc(pos: list[float], neg: list[float]) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg))


def _rates(rows: list[dict], key: str) -> dict:
    nm = [r for r in rows if not r["abnormal"]]
    ab = [r for r in rows if r["abnormal"]]
    tn = sum(not r[key] for r in nm)
    tp = sum(bool(r[key]) for r in ab)
    spec, sens = tn / len(nm), tp / len(ab)
    return {"n_normal": len(nm), "n_abnormal": len(ab), "specificity": spec,
            "specificity_ci95": list(clopper_pearson(tn, len(nm))), "sensitivity": sens,
            "sensitivity_ci95": list(clopper_pearson(tp, len(ab))), "balanced_accuracy": (spec + sens) / 2}


def assess(rows: list[dict]) -> dict[str, Any]:
    """``rows``: harness rows (``scripts/run_routine_jsonl.py``) of the TUAB evaluation split.
    Rows without a record-level score (not a routine recording, or unreadable) are excluded
    and counted, as pre-registered."""
    scored = [r for r in rows if r.get("record_probability") is not None and "error" not in r]
    if not any(r["abnormal"] for r in scored) or all(r["abnormal"] for r in scored):
        raise ValueError("need normal and abnormal recordings with a record-level score")
    M = _rates(scored, "impression_abnormal")
    R = _rates(scored, "rule_impression_abnormal")
    auc = _auc([r["record_probability"] for r in scored if r["abnormal"]],
               [r["record_probability"] for r in scored if not r["abnormal"]])
    h = {
        "H1_auc_at_least_0.80": auc >= AUC_MIN,
        "H2_model_balanced_accuracy_not_below_rules": M["balanced_accuracy"] >= R["balanced_accuracy"],
        "H3_specificity_at_least_0.65": M["specificity"] >= SPECIFICITY_MIN,
    }
    return {
        "n_rows": len(rows), "n_scored": len(scored), "n_excluded": len(rows) - len(scored),
        "auc": auc, "model": M, "rules": R,
        "borderline_fraction": sum(bool(r.get("record_borderline")) for r in scored) / len(scored),
        "without_age": sum(r.get("age_years") is None for r in scored),
        "hypotheses": h,
        "decision": {"record_model_decides": h["H2_model_balanced_accuracy_not_below_rules"]
                     and h["H3_specificity_at_least_0.65"]},
    }


def apply_decision(a: dict, impression_yaml: str = "configs/impression.yaml") -> bool:
    """Set ``record_model_decides`` (text edit keeps comments)."""
    p = Path(impression_yaml)
    v = a["decision"]["record_model_decides"]
    lines = [ln if not ln.startswith("record_model_decides:") else
             f"record_model_decides: {str(v).lower()}  # increment 32 (docs/preregistration_increment32.md)"
             for ln in p.read_text(encoding="utf-8").splitlines()]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return v
