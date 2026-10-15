"""Pre-registered analysis for increment 22 (docs/preregistration_increment22.md): the
record-level abnormality model (configs/models/record_gbm_v1.json, threshold fixed by
cross-validation) against the rule-based impression, both from the same run of this
commit's pipeline on the NMT authors' evaluation split - used once before (increment 17,
earlier pipeline), never for development since.

Written and committed before the pipeline was run on the evaluation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .prereg import clopper_pearson

BALANCED_ACCURACY_GAIN_MIN = 0.03
SPECIFICITY_MIN = 0.65
SENSITIVITY_MARGIN = 0.03


def _rates(rows: list[dict], key: str) -> dict:
    nm = [r for r in rows if not r["abnormal"]]
    ab = [r for r in rows if r["abnormal"]]
    tn = sum(not r[key] for r in nm)
    tp = sum(bool(r[key]) for r in ab)
    spec, sens = tn / len(nm), tp / len(ab)
    return {"n_normal": len(nm), "n_abnormal": len(ab), "tn": tn, "tp": tp,
            "specificity": spec, "specificity_ci95": list(clopper_pearson(tn, len(nm))),
            "sensitivity": sens, "sensitivity_ci95": list(clopper_pearson(tp, len(ab))),
            "balanced_accuracy": (spec + sens) / 2}


def assess(run: dict, threshold: float) -> dict[str, Any]:
    rows = run["records"]
    for r in rows:
        if r.get("record_probability") is None:
            raise ValueError(f"record {r.get('record_id')} has no record-level score")
        r["model_abnormal"] = r["record_probability"] >= threshold
    M, R = _rates(rows, "model_abnormal"), _rates(rows, "rule_impression_abnormal")
    h = {
        "H1_balanced_accuracy_gain_at_least_0.03": M["balanced_accuracy"] - R["balanced_accuracy"]
        >= BALANCED_ACCURACY_GAIN_MIN,
        "H2_specificity_at_least_0.65": M["specificity"] >= SPECIFICITY_MIN,
        "H3_sensitivity_not_lower_by_more_than_0.03": M["sensitivity"] >= R["sensitivity"] - SENSITIVITY_MARGIN,
    }
    return {"model": M, "rules": R, "threshold": threshold, "hypotheses": h,
            "decision": {"record_model_decides": all(h.values())}}


def markdown_report(a: dict) -> str:
    def row(name, t):
        ci = lambda v: f"{v[0]:.2f}–{v[1]:.2f}"  # noqa: E731
        return (f"| {name} | {t['specificity']:.3f} ({ci(t['specificity_ci95'])}) | "
                f"{t['sensitivity']:.3f} ({ci(t['sensitivity_ci95'])}) | {t['balanced_accuracy']:.3f} |")

    lines = ["| Заключение | Специфичность (95% ДИ) | Чувствительность (95% ДИ) | Сбаланс. точность |",
             "|---|---|---|---|", row(f"модель уровня записи (порог {a['threshold']:.3f})", a["model"]),
             row("правила (тот же прогон)", a["rules"]), "",
             f"{a['model']['n_normal']} нормальных и {a['model']['n_abnormal']} аномальных записей.", "",
             "Гипотезы:", ""]
    lines += [f"* **{k}** — {'подтверждена' if v else 'отвергнута'}" for k, v in a["hypotheses"].items()]
    lines += ["", "Решение (H1 ∧ H2 ∧ H3):", ""]
    lines += [f"* {k}: **{'да' if v else 'нет'}**" for k, v in a["decision"].items()]
    return "\n".join(lines) + "\n"


def apply_decision(a: dict, impression_yaml: str = "configs/impression.yaml") -> dict:
    import yaml

    p = Path(impression_yaml)
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    decides = a["decision"]["record_model_decides"]
    doc["record_model_decides"] = decides
    if decides:
        M = a["model"]
        doc["specificity"] = round(M["specificity"], 3)
        doc["sensitivity"] = round(M["sensitivity"], 3)
        doc["provenance"] = {"preregistration": "docs/preregistration_increment22.md",
                             "report": "docs/increment22_report.md",
                             "previous": "docs/increment20_report.md"}
    header = ("# What the report may claim about its own normal/abnormal conclusion on routine EEG.\n"
              "# Written by evaluation/prereg22.apply_decision (docs/preregistration_increment22.md).\n"
              + (f"# NMT evaluation split ({a['model']['n_normal']} normal, {a['model']['n_abnormal']} abnormal): "
                 f"record-level model decides; specificity {a['model']['specificity']:.3f}, sensitivity "
                 f"{a['model']['sensitivity']:.3f} (docs/increment22_report.md).\n" if decides else
                 "# Record-level model not adopted (docs/increment22_report.md): its score is reported "
                 "only; the rule-based impression keeps the increment-20 numbers.\n"))
    p.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return {"record_model_decides": decides}
