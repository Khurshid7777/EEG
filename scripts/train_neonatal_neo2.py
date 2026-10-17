"""Increment 27: neonatal monitor model with rhythmicity (ictal_gbm_realtime_neo2).

On monitor traces of all 79 Helsinki recordings (scripts/neonatal_rt_traces.py):
1. grouped 10-fold CV (by patient) of the same gradient-boosted model on feature version 1
   (as ictal_gbm_realtime_neo1) and version 5 (+ rhythmicity), causal event scoring;
2. comparison at the deployed neo1 operating point (threshold 0.5, >= 20 epochs) and at
   the v5 point with no more false alarms per hour than v1 there (most sensitive such);
3. final v5 model on all recordings + an event-level Platt calibration from the
   out-of-fold detections.

    PYTHONPATH=src python scripts/train_neonatal_neo2.py OUT_MODEL.json OUT_CAL.json OUT_REPORT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

NEO1_POINT = {"threshold": 0.5, "min_epochs": 20}


def main() -> int:
    from sklearn.model_selection import GroupKFold

    from neurolens.calibration.calibrator import ConfidenceCalibrator
    from neurolens.calibration.temperature import PlattScaler
    from neurolens.evaluation.matching import aggregate
    from neurolens.evaluation.ml import (
        POSTPROC_GRID, Dataset, detections_from_probs, fit_gbm, realtime_records, score_probs,
    )
    from neurolens.evaluation.realtime_replay import build_traces
    from neurolens.evaluation.runner import default_config, load_items

    out_model, out_cal, out_rep = (Path(a) for a in sys.argv[1:4])
    items = load_items("helsinki", ["dev"]) + load_items("helsinki", ["test"])
    traces = build_traces(items, default_config(), "data/physionet/trace_cache", workers=1, rhythm=True)
    recs = realtime_records(traces)
    groups = np.array([r.annotation.subject for r in recs])
    report = {"n_records": len(recs), "n_seizures": int(sum(len(r.annotation.seizures) for r in recs)),
              "variants": {}}
    oof = {}
    for version in (1, 5):
        ds = Dataset.build(recs, version=version)
        probs = [None] * len(recs)
        for tr, te in GroupKFold(10).split(np.arange(len(recs)), groups=groups):
            m = fit_gbm(*ds.stacked(list(tr)))
            for i in te:
                probs[i] = m.predict_proba(ds.X[i])
        oof[version] = (ds, probs)
        grid = {}
        for th in POSTPROC_GRID["threshold"]:
            for me in POSTPROC_GRID["min_epochs"]:
                s = aggregate(score_probs(ds, probs, th, me, causal=True))
                grid[f"{th}/{me}"] = {"sensitivity": s.sensitivity, "fa_per_hour": s.fa_per_hour,
                                      "latency_median_s": s.latency_median_s}
        report["variants"][f"v{version}"] = {"grid": grid}
        print(f"v{version} at neo1 point", grid[f"{NEO1_POINT['threshold']}/{NEO1_POINT['min_epochs']}"], flush=True)

    ref = report["variants"]["v1"]["grid"][f"{NEO1_POINT['threshold']}/{NEO1_POINT['min_epochs']}"]
    feasible = {k: v for k, v in report["variants"]["v5"]["grid"].items() if v["fa_per_hour"] <= ref["fa_per_hour"]}
    best_key = max(feasible, key=lambda k: (feasible[k]["sensitivity"], -feasible[k]["fa_per_hour"]))
    th, me = (float(x) for x in best_key.split("/"))
    report["comparison"] = {"v1_at_neo1_point": ref, "v5_point": {"threshold": th, "min_epochs": int(me)},
                            "v5_at_point": feasible[best_key]}
    print("comparison", json.dumps(report["comparison"]), flush=True)

    # final model on all recordings
    ds5, probs5 = oof[5]
    model = fit_gbm(*ds5.stacked(list(range(len(recs)))))
    model.meta.update({"name": "ictal_gbm_realtime_neo2", "feature_version": 5,
                       "trained_on": "all 79 Helsinki neonatal recordings (monitor traces)",
                       "operating_point": {"threshold": th, "min_epochs": int(me)},
                       "cv": report["comparison"]})
    model.save(out_model)
    # event-level Platt calibration from out-of-fold detections at the chosen point
    conf, lab = [], []
    for rec, t, p in zip(ds5.records, ds5.times, probs5):
        for d in detections_from_probs(t, p, th, int(me), causal=True):
            conf.append(d.confidence)
            lab.append(float(any(d.t_start <= s.offset_s and d.t_end >= s.onset_s for s in rec.annotation.seizures)))
    sc = PlattScaler().fit(np.clip(conf, 0.5, 0.99), lab)
    cal = ConfidenceCalibrator.identity()
    cal.platt["ictal_ml"] = [round(sc.a, 4), round(sc.b, 4)]
    cal.metrics["ictal_ml"] = {"n_events": len(conf), "true_events": int(sum(lab)),
                               "precision": round(float(np.mean(lab)), 4) if lab else None,
                               "source": "out-of-fold detections, grouped 10-fold CV, 79 Helsinki recordings"}
    cal.save(out_cal)
    out_rep.write_text(json.dumps(report, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
