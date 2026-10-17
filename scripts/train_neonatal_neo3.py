"""Increment 30: neonatal monitor model on a fixed 9-electrode set (ictal_gbm_realtime_neo3).

neo2 (19 electrodes) gave 10.9 seizure alarms/h on Helsinki seizure-free recordings
reduced to Cork's 9 electrodes (1.7/h on all 19; scripts/neonatal_device_perturb.py),
i.e. it learned the montage. The monitor now reduces every neonatal recording to
CHANNELS; the model is trained on monitor traces of the same reduction.

1. grouped 10-fold CV (by patient), feature version 5, on traces of all 79 Helsinki
   recordings with all 19 electrodes ("19ch", = neo2) and reduced to CHANNELS ("9ch");
2. both at the neo2 operating point (threshold 0.5, >= 20 epochs);
3. final 9ch model on all recordings + event-level Platt calibration (out-of-fold).

    PYTHONPATH=src python scripts/train_neonatal_neo3.py OUT_MODEL.json OUT_CAL.json OUT_REPORT.json [--traces-only]
"""

CHANNELS = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2", "Cz"]

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
    cache = "data/physionet/trace_cache"
    traces9 = build_traces(items, default_config(), cache, workers=4, rhythm=True, channels=CHANNELS)
    if "--traces-only" in sys.argv:
        return 0
    traces19 = build_traces(items, default_config(), cache, workers=1, rhythm=True)
    report = {"channels": CHANNELS, "variants": {}}
    oof = {}
    key = f"{NEO1_POINT['threshold']}/{NEO1_POINT['min_epochs']}"
    for name, traces in (("19ch", traces19), ("9ch", traces9)):
        recs = realtime_records(traces)
        groups = np.array([r.annotation.subject for r in recs])
        ds = Dataset.build(recs, version=5)
        probs = [None] * len(recs)
        for tr, te in GroupKFold(10).split(np.arange(len(recs)), groups=groups):
            m = fit_gbm(*ds.stacked(list(tr)))
            for i in te:
                probs[i] = m.predict_proba(ds.X[i])
        oof[name] = (recs, ds, probs)
        grid = {}
        for th in POSTPROC_GRID["threshold"]:
            for me in POSTPROC_GRID["min_epochs"]:
                s = aggregate(score_probs(ds, probs, th, me, causal=True))
                grid[f"{th}/{me}"] = {"sensitivity": s.sensitivity, "fa_per_hour": s.fa_per_hour,
                                      "latency_median_s": s.latency_median_s}
        report["variants"][name] = {"grid": grid, "at_neo2_point": grid[key]}
        print(name, "at neo2 point", grid[key], flush=True)
    report["n_records"] = len(oof["9ch"][0])
    report["n_seizures"] = int(sum(len(r.annotation.seizures) for r in oof["9ch"][0]))
    th, me = float(NEO1_POINT["threshold"]), int(NEO1_POINT["min_epochs"])

    # final model on all recordings
    recs, ds5, probs5 = oof["9ch"]
    model = fit_gbm(*ds5.stacked(list(range(len(recs)))))
    model.meta.update({"name": "ictal_gbm_realtime_neo3", "feature_version": 5, "channels": CHANNELS,
                       "trained_on": "all 79 Helsinki neonatal recordings (monitor traces, 9 electrodes)",
                       "operating_point": {"threshold": th, "min_epochs": me},
                       "cv": {k: v["at_neo2_point"] for k, v in report["variants"].items()}})
    model.save(out_model)
    conf, lab = [], []
    for rec, t, p in zip(ds5.records, ds5.times, probs5):
        for d in detections_from_probs(t, p, th, me, causal=True):
            conf.append(d.confidence)
            lab.append(float(any(d.t_start <= s.offset_s and d.t_end >= s.onset_s for s in rec.annotation.seizures)))
    sc = PlattScaler().fit(np.clip(conf, 0.5, 0.99), lab)
    cal = ConfidenceCalibrator.identity()
    cal.platt["ictal_ml"] = [round(sc.a, 4), round(sc.b, 4)]
    cal.metrics["ictal_ml"] = {"n_events": len(conf), "true_events": int(sum(lab)),
                               "precision": round(float(np.mean(lab)), 4) if lab else None,
                               "source": "out-of-fold detections, grouped 10-fold CV, 79 Helsinki recordings, 9 electrodes"}
    cal.save(out_cal)
    out_rep.write_text(json.dumps(report, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
