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


class RecordModel:
    def __init__(self, model):
        if list(model.feature_names) != FEATURE_NAMES:
            raise ValueError("record model was trained on a different feature set")
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> "RecordModel":
        from ..layer4_detect.ml_ictal import load_model

        return cls(load_model(path))

    def probability(self, row: dict) -> float:
        return float(self.model.predict_proba(np.array([record_features(row)], float))[0])


def load_optional(path: Optional[str | Path]) -> Optional[RecordModel]:
    return RecordModel.load(path) if path and Path(path).exists() else None
