"""Train the record-level abnormality model (increment 22) on routine-EEG harness outputs.

    PYTHONPATH=src python scripts/train_record_model.py OUT_MODEL.json OUT_CV.json METRICS.json [METRICS2.json ...]

Five-fold stratified cross-validation gives out-of-fold probabilities; the threshold is the
one at which the out-of-fold specificity is TARGET_SPECIFICITY (fixed in advance). The
final model is fitted on all recordings with the same hyper-parameters and exported to
JSON (numpy inference, no pickles).
"""

import json
import sys
from pathlib import Path

import numpy as np

TARGET_SPECIFICITY = 0.75
# increment 22b: the target specificity is met in each age band, not only overall (the single
# threshold gave specificity 0.41 in normal children of 2-11 years, out of fold)
AGE_BANDS_BELOW = (2.0, 12.0, 18.0, 60.0, 200.0)
PARAMS = dict(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0)


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from neurolens.layer4_detect.ml_ictal import export_hist_gbm
    from neurolens.layer5_reasoning.record_model import FEATURE_NAMES, record_features

    out_model, out_cv = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [r for f in sys.argv[3:] for r in json.loads(Path(f).read_text())["records"]]
    X = np.array([record_features(r) for r in rows], float)
    y = np.array([bool(r["abnormal"]) for r in rows], int)
    rule = np.array([bool(r["impression_abnormal"]) for r in rows], int)
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    p = cross_val_predict(HistGradientBoostingClassifier(**PARAMS), X, y, cv=cv, method="predict_proba")[:, 1]
    thr = float(np.quantile(p[y == 0], TARGET_SPECIFICITY))
    ages = [r.get("age_years") for r in rows]
    band_thr, lo = [], 0.0
    for hi in AGE_BANDS_BELOW:
        m = np.array([a is not None and lo <= a < hi for a in ages]) & (y == 0)
        band_thr.append([hi, round(float(np.quantile(p[m], TARGET_SPECIFICITY)), 4)])
        lo = hi
    t_rec = np.array([next((t for hi, t in band_thr if a is not None and a < hi), thr) for a in ages])
    pred_b = p >= t_rec
    pred = p >= thr
    cv_report = {
        "n_normal": int((y == 0).sum()), "n_abnormal": int((y == 1).sum()),
        "auc_oof": float(roc_auc_score(y, p)), "threshold": thr,
        "target_specificity": TARGET_SPECIFICITY,
        "specificity_oof": float((~pred[y == 0]).mean()), "sensitivity_oof": float(pred[y == 1].mean()),
        "rule_specificity": float(1 - rule[y == 0].mean()), "rule_sensitivity": float(rule[y == 1].mean()),
        "threshold_by_age": band_thr,
        "specificity_oof_by_age_thresholds": float((~pred_b[y == 0]).mean()),
        "sensitivity_oof_by_age_thresholds": float(pred_b[y == 1].mean()),
        "params": PARAMS, "sources": sys.argv[3:],
    }
    clf = HistGradientBoostingClassifier(**PARAMS).fit(X, y)
    model = export_hist_gbm(clf, list(FEATURE_NAMES), meta={
        "name": "record_gbm_v1", "task": "routine EEG normal vs abnormal (NMT, neurologist labels)",
        "threshold": thr, "cv": {k: cv_report[k] for k in ("auc_oof", "specificity_oof", "sensitivity_oof")},
        # typical values of neurologist-normal recordings, for the report's "main reasons"
        "normal_medians": [float(v) for v in np.median(X[y == 0], axis=0)]})
    model.save(out_model)
    # exported model must reproduce scikit-learn
    assert np.allclose(model.predict_proba(X), clf.predict_proba(X)[:, 1], atol=1e-6)
    out_cv.write_text(json.dumps(cv_report, indent=1) + "\n")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in cv_report.items()
                      if k not in ("params", "sources")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
