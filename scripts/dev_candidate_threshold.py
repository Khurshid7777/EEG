"""Increment 63 (development, docs/preregistration_increment63.md): sensitivity / false-alarm curve of the frozen
offline seizure model (increment 11, ``offline_gbm_v2``) as a function of the detection threshold, per recording.
Saves per-recording detections (start, end, peak probability) for each threshold so later steps need no recomputation.

    PYTHONPATH=src python scripts/dev_candidate_threshold.py aubmc OUT.json [--workers 3]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]   # applied to the event's mean probability (ml_probability evidence)
FROZEN = "docs/preregistration_increment11_frozen.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from neurolens.evaluation.matching import aggregate, score_record
    from neurolens.evaluation.ml import Dataset, detections_from_probs
    from neurolens.evaluation.runner import ScoringRules, prepare_records
    from neurolens.layer4_detect.ml_ictal import load_model
    from neurolens.pipeline.config_loader import load_configs

    frozen = json.loads(Path(FROZEN).read_text())
    cfg = load_configs()
    if a.set == "aubmc":
        from neurolens.datasets.aubmc import AubmcClient

        c = AubmcClient()
        items = [(ann, c.fetch(ann.file)) for ann in c.annotations()]
    elif a.set == "inc11":   # the held-out CHB-MIT chb06 + Siena PN01/03/06/10/14 records of increment 11
        from neurolens.datasets.physionet import client_for

        files = set(json.loads(Path("docs/increment11_metrics.json").read_text())["test_records"])
        items = []
        for db, subs in (("chbmit", ["chb06"]), ("siena", ["PN01", "PN03", "PN06", "PN10", "PN14"])):
            c = client_for(db)
            for sub in subs:
                for ann in c.annotations(sub):
                    if ann.file in files:
                        items.append((ann, c.fetch(ann.file)))
                        print("fetched", ann.file, flush=True)
    else:
        raise SystemExit("unknown set")
    rules = ScoringRules()
    recs = prepare_records(items, cfg, None, workers=a.workers)
    ds = Dataset.build(recs, rules)
    model = load_model(frozen["offline"]["model"]["path"])
    mins = frozen["offline"]["A_replacement"]["params"]["min_epochs"]
    probs = [model.predict_proba(X) if len(X) else X[:, 0] for X in ds.X]
    res = {}
    for thr in THRESHOLDS:
        scores = []
        for rec, p, t in zip(ds.records, probs, ds.times):
            dets = [d for d in detections_from_probs(t, p, 0.3, mins, False) if d.confidence >= thr]
            scores.append(score_record(rec.annotation.file, rec.annotation.seizures, dets, rec.duration_s,
                                       rules.pre_s, rules.post_s, rules.merge_gap_s))
        agg = aggregate(scores)
        res[str(thr)] = {k: agg[k] for k in ("n_records", "hours", "n_seizures", "tp", "fn", "fp", "sensitivity", "fa_per_hour")}
        print(thr, res[str(thr)], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
