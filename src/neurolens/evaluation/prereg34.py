"""Pre-registered analysis for increment 34 (docs/preregistration_increment34.md): the frozen
routine-EEG verdict on 111 healthy adults of OpenNeuro ds003775 (Oslo, BioSemi 64 ch),
as delivered and reduced to the 19 10-20 electrodes. Written before any EEG was downloaded.
"""

from __future__ import annotations

from typing import Any

from .prereg import clopper_pearson

SPECIFICITY_MIN = 0.65
MONTAGE_GAP = 0.10  # reduced-montage specificity better by more than this -> montage gate


def _spec(rows: list[dict]) -> dict:
    ok = [r for r in rows if "error" not in r and r.get("record_probability") is not None]
    tn = sum(not r["impression_abnormal"] for r in ok)
    return {"n": len(ok), "n_excluded": len(rows) - len(ok), "specificity": tn / len(ok) if ok else float("nan"),
            "specificity_ci95": list(clopper_pearson(tn, len(ok))) if ok else None,
            "rule_specificity": sum(not r["rule_impression_abnormal"] for r in ok) / len(ok) if ok else float("nan"),
            "borderline": sum(bool(r.get("record_borderline")) for r in ok)}


def assess(delivered: list[dict], reduced: list[dict]) -> dict[str, Any]:
    D, R = _spec(delivered), _spec(reduced)
    h = {"H1_specificity_as_delivered_at_least_0.65": D["specificity"] >= SPECIFICITY_MIN}
    return {"as_delivered": D, "reduced_1020": R, "hypotheses": h,
            "decision": {"montage_gate_for_routine_verdict": R["specificity"] - D["specificity"] > MONTAGE_GAP}}
