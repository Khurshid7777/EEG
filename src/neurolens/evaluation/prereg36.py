"""Pre-registered regression check for increment 36 (docs/preregistration_increment36.md): the
record model may no longer clear burst-suppression (D68). NMT evaluation split, current
configuration, verdict with the rule (``impression_abnormal``) against the record model alone
(``model_abnormal``). Written before the run.
"""

from __future__ import annotations

from typing import Any

from .prereg import clopper_pearson

MAX_SPECIFICITY_LOSS = 0.02


def _rates(rows: list[dict], key: str) -> dict:
    nm = [r for r in rows if not r["abnormal"]]
    ab = [r for r in rows if r["abnormal"]]
    tn = sum(not r[key] for r in nm)
    tp = sum(bool(r[key]) for r in ab)
    return {"specificity": tn / len(nm), "specificity_ci95": list(clopper_pearson(tn, len(nm))),
            "sensitivity": tp / len(ab), "sensitivity_ci95": list(clopper_pearson(tp, len(ab)))}


def assess(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r and r.get("record_probability") is not None]
    gate, model = _rates(ok, "impression_abnormal"), _rates(ok, "model_abnormal")
    kept = [r for r in ok if r.get("kept_rule_findings")]
    loss = model["specificity"] - gate["specificity"]
    return {"n": len(ok), "n_normal": sum(not r["abnormal"] for r in ok), "n_abnormal": sum(r["abnormal"] for r in ok),
            "with_rule": gate, "model_alone": model,
            "kept": {"normal": sum(not r["abnormal"] for r in kept), "abnormal": sum(r["abnormal"] for r in kept)},
            "specificity_loss": loss,
            "decision": {"keep_record_model_cannot_clear": round(loss, 9) <= MAX_SPECIFICITY_LOSS}}
