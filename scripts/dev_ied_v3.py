"""Increment 40 development (training split of NMT-Events only): cross-validated comparison of
the IED model on the v1 features and on v3 (+ waveform morphology and field), same folds, same
FA budget. Writes docs/increment40_dev.json.

    PYTHONPATH=src python scripts/dev_ied_v3.py [--workers 4]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _prep(item):
    rid, normal, member = item
    from neurolens.datasets.nmt_events import NMTEventsClient
    from neurolens.evaluation.ied_eval import prepare_record
    from neurolens.pipeline.config_loader import load_configs

    return prepare_record(NMTEventsClient("data/nmt_events").local_path(member), rid, normal, load_configs(),
                          Path("data/nmt_events/.features"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    import numpy as np
    from sklearn.metrics import roc_auc_score

    from neurolens.datasets.nmt_events import NMTEventsClient
    from neurolens.evaluation import ied_eval as E
    from neurolens.evaluation.prereg15 import FA_BUDGET_PER_H
    from neurolens.layer4_detect import ied_ml

    c = NMTEventsClient("data/nmt_events")
    anns = c.annotations()
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(_prep, c.split_members("train")))
    morph = tuple(n for n in ied_ml.feature_names() if any(m in n for m in ied_ml.MORPHOLOGY))
    out = {}
    for name, drop in (("v1_features", morph), ("v3", ())):
        probs = E.cross_val_probabilities(recs, anns, drop=drop)
        curve = E.learned_curve(recs, anns, probs)
        y, s = [], []
        for r, p in zip(recs, probs):
            ep, _ = E.reference_events(None if r.normal else anns.get(r.record_id))
            lab = ied_ml.window_labels(r.t0, ep)
            keep = lab >= 0
            y.append(lab[keep]); s.append(p[keep])
        y, s = np.concatenate(y), np.concatenate(s)
        out[name] = {"window_auc_oof": float(roc_auc_score(y, s)),
                     "operating_point_15_per_h": E.operating_point(curve, FA_BUDGET_PER_H, "threshold"),
                     "curve": curve}
        print(name, out[name]["window_auc_oof"], out[name]["operating_point_15_per_h"], flush=True)
    Path("docs/increment40_dev.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
