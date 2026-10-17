"""Pre-registered analysis for increment 28 (docs/preregistration_increment28.md): neonatal
background alarms of the monitor against the consensus background grade of the Cork HIE
dataset (Zenodo 7477575), never used by this project before.

Written and committed before the monitor was run on any Cork recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .prereg import clopper_pearson

SENSITIVITY_MIN = 0.70  # grade 3-4 (severely abnormal, inactive) hours with a background alarm
SPECIFICITY_MIN = 0.80  # grade 1 (normal / mildly abnormal) hours without one
SPEARMAN_MIN = 0.40     # grade vs background alarms per hour
N_BOOT, SEED = 2000, 0


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).correlation)


def _boot_ci(rows: list[dict], stat) -> list[float]:
    """95% interval resampling babies (hours of one baby are not independent)."""
    babies = sorted({r["baby_id"] for r in rows})
    by = {b: [r for r in rows if r["baby_id"] == b] for b in babies}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(N_BOOT):
        sample = [r for b in rng.choice(babies, len(babies)) for r in by[b]]
        v = stat(sample)
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _sens(rows):
    s = [r for r in rows if r["grade"] >= 3]
    return np.mean([r["background_alarms"] > 0 for r in s]) if s else None


def _spec(rows):
    n = [r for r in rows if r["grade"] == 1]
    return np.mean([r["background_alarms"] == 0 for r in n]) if n else None


def _rho(rows):
    return _spearman([r["grade"] for r in rows], [r["background_alarms"] / max(r["hours"], 1e-9) for r in rows])


def evaluate(rows: list[dict]) -> dict[str, Any]:
    severe = [r for r in rows if r["grade"] >= 3]
    normal = [r for r in rows if r["grade"] == 1]
    tp = sum(r["background_alarms"] > 0 for r in severe)
    tn = sum(r["background_alarms"] == 0 for r in normal)
    sens, spec, rho = tp / len(severe), tn / len(normal), _rho(rows)
    by_grade = {}
    for g in (1, 2, 3, 4):
        G = [r for r in rows if r["grade"] == g]
        if G:
            by_grade[g] = {"n": len(G), "with_background_alarm": sum(r["background_alarms"] > 0 for r in G),
                           "background_alarms_per_hour": sum(r["background_alarms"] for r in G)
                           / sum(r["hours"] for r in G)}
    h = {
        "H1_sensitivity_grade3_4_at_least_0.70": sens >= SENSITIVITY_MIN,
        "H2_specificity_grade1_at_least_0.80": spec >= SPECIFICITY_MIN,
        "H3_spearman_grade_vs_alarm_rate_at_least_0.40": rho >= SPEARMAN_MIN,
    }
    return {
        "n_hours": len(rows), "n_babies": len({r["baby_id"] for r in rows}),
        "sensitivity": sens, "sensitivity_ci95_exact": list(clopper_pearson(tp, len(severe))),
        "sensitivity_ci95_baby_bootstrap": _boot_ci(rows, _sens),
        "specificity": spec, "specificity_ci95_exact": list(clopper_pearson(tn, len(normal))),
        "specificity_ci95_baby_bootstrap": _boot_ci(rows, _spec),
        "spearman": rho, "spearman_ci95_baby_bootstrap": _boot_ci(rows, _rho),
        "by_grade": by_grade, "secondary": secondary(rows), "hypotheses": h,
        "decision": {"neonatal_background_alarms": "critical" if h["H2_specificity_grade1_at_least_0.80"]
                     else "review"},
    }


def secondary(rows: list[dict]) -> dict[str, Any]:
    """Descriptive only (no hypotheses): offline report vs grade, seizure alarms vs seizures."""
    out: dict[str, Any] = {"offline_aeeg_category_by_grade": {}, "offline_abnormal_for_age_by_grade": {}}
    for g in (1, 2, 3, 4):
        G = [r for r in rows if r["grade"] == g]
        cats: dict[str, int] = {}
        for r in G:
            cats[str(r.get("offline_aeeg_category"))] = cats.get(str(r.get("offline_aeeg_category")), 0) + 1
        out["offline_aeeg_category_by_grade"][g] = cats
        out["offline_abnormal_for_age_by_grade"][g] = sum(bool(r.get("offline_abnormal_for_age")) for r in G)
    for yn in ("Y", "N"):
        G = [r for r in rows if r.get("seizures_yn") == yn]
        out[f"seizure_alarm_hours_when_seizures_{yn}"] = [sum(r["alarms_by_type"].get("seizure", 0) > 0 for r in G),
                                                          len(G)]
    return out


def apply_decision(a: dict, realtime_yaml: str = "configs/realtime.yaml") -> str:
    """Set ``neonatal_background_alarms`` in the monitor configuration (text edit keeps comments)."""
    p = Path(realtime_yaml)
    text = p.read_text(encoding="utf-8")
    value = a["decision"]["neonatal_background_alarms"]
    lines = [ln for ln in text.splitlines() if not ln.startswith("neonatal_background_alarms:")]
    lines.append(f"neonatal_background_alarms: {value}  # increment 28 (docs/preregistration_increment28.md)")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return value
