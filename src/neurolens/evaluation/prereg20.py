"""Pre-registered analysis for increment 20 (docs/preregistration_increment20.md):
age-specific cardiac rates and physiologic heart-rate variability for the ECG mimic of
periodic discharges (F14), learned seizure events on routine EEG as review candidates
(F15) and lateralized patterns that need a field (F17; F16, the "alert" PDR, was
dropped during development). Fresh seeded NMT train-split
subset (seed 20): 300 normal recordings never run before and the 45 remaining abnormal
ones, against the increment-19 pipeline (commit 8251619) on the same recordings.

Written and committed before either pipeline was run on the fresh subset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .prereg17 import _counts

SPECIFICITY_GAIN_MIN = 0.04
SENSITIVITY_MARGIN = 0.07          # <= 3 of 45 abnormal recordings
SPECIFICITY_MIN = 0.65
SEIZURE_CLAIM_MAX = 0.02           # normal reports whose critical list claims seizures
SEIZURE_CLAIMS = ("ongoing_seizure", "status_epilepticus_suspected", "high_seizure_burden")


def seizure_claim_rate(rows: list[dict]) -> float:
    nm = [r for r in rows if not r["abnormal"]]
    return sum(bool(set(r.get("critical_codes", ())) & set(SEIZURE_CLAIMS)) for r in nm) / max(1, len(nm))


def assess(new: dict, old: dict) -> dict[str, Any]:
    N, O = _counts(new["records"]), _counts(old["records"])
    if sorted(str(r.get("record_id")) for r in new["records"]) != sorted(
            str(r.get("record_id")) for r in old["records"]):
        raise ValueError("the two runs did not score the same recordings")
    claims = seizure_claim_rate(new["records"])
    # the increment-19 harness did not save critical findings: every learned seizure
    # event there produced "seizure activity registered" (old proxy, reported only)
    old_proxy = sum("ictal_ml" in r["detectors_fired"] for r in old["records"] if not r["abnormal"]) / max(
        1, O["n_normal"])
    h = {
        "H1_specificity_gain_at_least_0.04": N["specificity"] - O["specificity"] >= SPECIFICITY_GAIN_MIN,
        "H2_sensitivity_not_lower_by_more_than_0.07": N["sensitivity"] >= O["sensitivity"] - SENSITIVITY_MARGIN,
        "H3_specificity_at_least_0.65": N["specificity"] >= SPECIFICITY_MIN,
        "H4_seizure_claims_in_normal_reports_at_most_0.02": claims <= SEIZURE_CLAIM_MAX,
    }
    return {"new": N, "old": O, "hypotheses": h,
            "seizure_claims_normal": {"new": claims, "old_proxy": old_proxy},
            "decision": {"keep_increment20_rules": all(h.values())}}


def markdown_report(a: dict) -> str:
    from .prereg18 import markdown_report as _md

    text = _md(a).replace("инкремент 18 |", "инкремент 20 |", 1).replace(
        "инкремент 17 (286bb2a)", "инкремент 19 (8251619)").replace("H1 ∧ H2 ∧ H3", "H1 ∧ H2 ∧ H3 ∧ H4")
    c = a["seizure_claims_normal"]
    return text + (f"\nДоля нормальных записей с критической находкой о приступах: {c['new']:.3f} "
                   f"(до: {c['old_proxy']:.3f} — любое событие обученной модели приступов).\n")


def apply_decision(a: dict, impression_yaml: str = "configs/impression.yaml",
                   thresholds_yaml: str = "configs/thresholds.yaml") -> dict:
    keep = a["decision"]["keep_increment20_rules"]
    if keep:
        N = a["new"]
        lines = [
            "# What the report may claim about its own normal/abnormal conclusion on routine EEG.",
            "# Written by evaluation/prereg20.apply_decision (docs/preregistration_increment20.md).",
            f"# Fresh NMT subset 20 ({N['n_normal']} normal, {N['n_abnormal']} abnormal): specificity "
            f"{N['specificity']:.3f}, sensitivity {N['sensitivity']:.3f} (docs/increment20_report.md);",
            "# earlier: increment 18 0.640 / 0.780 (repeated in 19: 0.640 / 0.750), increment 17 0.589 / 0.844.",
            "routine_validated: true",
            f"specificity: {N['specificity']:.3f}",
            f"sensitivity: {N['sensitivity']:.3f}",
            "provenance:",
            "  preregistration: docs/preregistration_increment20.md",
            "  report: docs/increment20_report.md",
            "  previous: docs/increment18_report.md",
        ]
        Path(impression_yaml).write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        p = Path(thresholds_yaml)
        text = p.read_text(encoding="utf-8")
        text = re.sub(r"(?m)^pd_ecg_isi_cv_max:.*$", "pd_ecg_isi_cv_max: 0.05", text)
        text = re.sub(r"(?m)^pd_ecg_freq_hz_by_age:.*\n(?:\s+- .*\n)*", "pd_ecg_freq_hz_by_age: []\n", text)
        text = re.sub(r"(?m)^ictal_ml_routine_uncertain:.*$", "ictal_ml_routine_uncertain: false", text)
        text = re.sub(r"(?m)^pdr_method:.*$", "pdr_method: whole", text)
        text = re.sub(r"(?m)^field_required:.*$", "field_required: false", text)
        p.write_text(text, encoding="utf-8")
    return {"keep_increment20_rules": keep}
