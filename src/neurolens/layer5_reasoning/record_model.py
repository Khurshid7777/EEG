"""Record-level abnormality score for routine EEG (increment 22).

The rule-based impression decides "abnormal" from single findings (a spike candidate the
learned IED model rates >= 0.95, slowing with a slow PDR, ...). On 1,045 NMT routine EEGs
the evidence those rules see - learned-model scores, background, PDR, which findings
fired - separates neurologist-normal from abnormal recordings better when weighed
together (cross-validated AUC 0.87). This module turns a record summary
(``pipeline.record_summary.summarize``) into a fixed feature vector and scores it with a
frozen gradient-boosted model (JSON, numpy inference).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

CODES = ("spike", "diffuse_slowing", "ictal_ml", "bipds", "lpds", "lrda", "grda", "gpds",
         "focal_slowing", "burst_suppression", "triphasic_waves", "extreme_delta_brush", "firda")

FEATURE_NAMES = (
    ["age_years", "pdr_hz", "rel_delta", "rel_theta", "rel_alpha", "robust_amplitude_uv",
     "ied_p_max", "ied_p_mean", "ied_n_ge_095", "ied_n_ge_080",
     "ictal_n", "ictal_conf_max", "ictal_fraction", "slowing_delta_theta_fraction"]
    + [f"patho_{c}" for c in CODES] + [f"fired_{c}" for c in CODES]
)


def record_features(row: dict) -> list[float]:
    sp = [float(e["evidence"].get("ied_probability_learned") or 0.0)
          for e in row["events"] if e["code"] == "spike"]
    ic = [e for e in row["events"] if e["code"] == "ictal_ml"]
    ds = [e for e in row["events"] if e["code"] == "diffuse_slowing"]
    dur = max(float(row.get("duration_s") or 0.0), 1.0)
    age = row.get("age_years")
    f = [age if age is not None else -1.0, row["pdr_hz"], row["global_rel_delta"],
         row["global_rel_theta"], row["global_rel_alpha"], row["robust_amplitude_uv"],
         max(sp, default=0.0), float(np.mean(sp)) if sp else 0.0,
         float(sum(p >= 0.95 for p in sp)), float(sum(p >= 0.8 for p in sp)),
         float(len(ic)), max((float(e["confidence"]) for e in ic), default=0.0),
         sum(float(e["t_end"]) - float(e["t_start"]) for e in ic) / dur,
         float(ds[0]["evidence"].get("rel_delta_theta_fraction") or 0.0) if ds else 0.0]
    f += [float(c in row["pathologic_nodes"]) for c in CODES]
    f += [float(c in row["detectors_fired"]) for c in CODES]
    return f


# what the report calls each feature (RU, UZ)
LABELS = {
    "age_years": ("возраст", "yosh"), "pdr_hz": ("частота ЗДР", "ZDR chastotasi"),
    "rel_delta": ("доля дельта-активности", "delta faolligi ulushi"),
    "rel_theta": ("доля тета-активности", "teta faolligi ulushi"),
    "rel_alpha": ("доля альфа-активности", "alfa faolligi ulushi"),
    "robust_amplitude_uv": ("амплитуда фона", "fon amplitudasi"),
    "ied_p_max": ("максимальная оценка разряда обученной моделью", "o‘qitilgan model razryad bahosi (maks.)"),
    "ied_p_mean": ("средняя оценка кандидатов-разрядов", "nomzod razryadlar o‘rtacha bahosi"),
    "ied_n_ge_095": ("число кандидатов-разрядов с оценкой ≥ 0.95", "≥ 0.95 baholi nomzodlar soni"),
    "ied_n_ge_080": ("число кандидатов-разрядов с оценкой ≥ 0.80", "≥ 0.80 baholi nomzodlar soni"),
    "ictal_n": ("число кандидатов в приступ", "tutqanoq nomzodlari soni"),
    "ictal_conf_max": ("оценка кандидата в приступ", "tutqanoq nomzodi bahosi"),
    "ictal_fraction": ("доля записи с кандидатами в приступ", "tutqanoq nomzodli yozuv ulushi"),
    "slowing_delta_theta_fraction": ("выраженность замедления", "sekinlashuv darajasi"),
}


def _label(name: str) -> tuple[str, str]:
    if name in LABELS:
        return LABELS[name]
    from ..layer4_detect.base import LABELS as EVENT_LABELS

    kind, code = name.split("_", 1)
    ru, uz = EVENT_LABELS.get(code, (code, code))
    return ((f"{ru} (патологическая находка)" if kind == "patho" else f"{ru} (кандидат)"),
            (f"{uz} (patologik topilma)" if kind == "patho" else f"{uz} (nomzod)"))


class RecordModel:
    """A frozen tree model over FEATURE_NAMES or an ordered subset of them (the age-free
    variant, increment 24, for recordings whose patient age is unknown)."""

    def __init__(self, model):
        names = list(model.feature_names)
        if any(n not in FEATURE_NAMES for n in names):
            raise ValueError("record model was trained on a different feature set")
        self.model = model
        self.idx = [FEATURE_NAMES.index(n) for n in names]

    def _x(self, row: dict) -> np.ndarray:
        return np.array(record_features(row), float)[self.idx]

    @classmethod
    def load(cls, path: str | Path) -> "RecordModel":
        from ..layer4_detect.ml_ictal import load_model

        return cls(load_model(path))

    def probability(self, row: dict) -> float:
        return float(self.model.predict_proba(self._x(row)[None])[0])

    def reasons(self, row: dict, k: int = 3, min_drop: float = 0.02) -> list[dict]:
        """Features that raise the score most: the drop when one feature is set to the
        median of neurologist-normal recordings (training data). Empty without medians."""
        med = self.model.meta.get("normal_medians")
        if not med:
            return []
        x = self._x(row)
        base = float(self.model.predict_proba(x[None])[0])
        Xs = np.repeat(x[None], len(x), axis=0)
        Xs[np.arange(len(x)), np.arange(len(x))] = med
        drop = base - self.model.predict_proba(Xs)
        out = []
        for i in np.argsort(-drop)[:k]:
            if drop[i] < min_drop:
                break
            name = self.model.feature_names[i]
            ru, uz = _label(name)
            out.append({"feature": name, "value": round(float(x[i]), 3),
                        "typical_normal": round(float(med[i]), 3), "score_drop": round(float(drop[i]), 3),
                        "label_ru": ru, "label_uz": uz})
        return out


def load_optional(path: Optional[str | Path]) -> Optional[RecordModel]:
    return RecordModel.load(path) if path and Path(path).exists() else None
