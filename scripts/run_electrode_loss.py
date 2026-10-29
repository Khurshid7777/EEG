"""Increment 48 (pre-registered, docs/preregistration_increment48.md): the learned monitor
(ictal_gbm_realtime_v2 at 0.7 x 5) on AUBMC when electrodes are lost.

    PYTHONPATH=src python scripts/run_electrode_loss.py docs/increment48_assess.json
"""

import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

STREAMS = Path("data/aubmc/stream")
FROZEN = Path("docs/preregistration_increment11_frozen.json")
KS = (0, 1, 3, 6, 9, 12)
N_SETS = 20


def drop_channels(f, lost_names: list[str]):
    keep = np.array([i for i, c in enumerate(f.eeg_channels) if c not in lost_names], int)
    g = SimpleNamespace(**{k: v for k, v in vars(f).items()})
    g.epoch_rms = np.asarray(f.epoch_rms)[:, keep]
    g.epoch_band_conc = np.asarray(f.epoch_band_conc)[:, keep]
    g.epoch_domfreq = np.asarray(f.epoch_domfreq)[:, keep]
    g.epoch_relpow = {k: np.asarray(v)[:, keep] for k, v in f.epoch_relpow.items()}
    g.eeg_channels = [f.eeg_channels[i] for i in keep]
    return g


def main(out: Path) -> int:
    from neurolens.evaluation.matching import score_record
    from neurolens.evaluation.ml import detections_from_probs
    from neurolens.evaluation.prereg12 import _overlaps
    from neurolens.evaluation.runner import ScoringRules
    from neurolens.layer4_detect.ml_ictal import featurize, load_model
    from neurolens.pipeline.config_loader import load_configs

    fz = json.loads(FROZEN.read_text(encoding="utf-8"))["realtime"]["model"]
    if hashlib.sha256(Path(fz["path"]).read_bytes()).hexdigest() != fz["sha256"]:
        raise RuntimeError("realtime model does not match the frozen sha256")
    model = load_model(fz["path"])
    mode = load_configs().ml.realtime
    rules = ScoringRules()
    # local monitor-stream caches written by this project in increment 39 (scripts/run_aubmc_eval.py)
    recs = [pickle.loads(p.read_bytes())[0] for p in sorted(STREAMS.glob("stream_*.pkl"))]
    common = sorted(set.intersection(*(set(r.features.eeg_channels) for r in recs)))
    n_ch = len(common)
    print(f"{len(recs)} records, {n_ch} channels common to all", flush=True)
    rows = []
    for k in KS:
        per_set = []
        for seed in range(N_SETS if k else 1):
            lost = [common[i] for i in np.random.default_rng(seed).choice(n_ch, k, replace=False)] if k else []
            tp = n = fp = 0
            hours = 0.0
            for rec in recs:
                f = drop_channels(rec.features, lost)
                X = featurize(f, 1)
                p = model.predict_proba(X) if len(X) else np.zeros(0)
                t = np.asarray(f.epoch_times, float)
                dets = detections_from_probs(t, p, mode.threshold, mode.min_epochs, True)
                sc = score_record(rec.annotation.file, rec.annotation.seizures, dets, rec.duration_s,
                                  rules.pre_s, rules.post_s, rules.merge_gap_s)
                amb = [(s.onset_s, s.offset_s) for s in getattr(rec.annotation, "ambiguous", [])]
                tp += sc.tp
                n += sc.n_seizures
                fp += sum(1 for d in sc.false_positives if not _overlaps(d.t_start, d.t_end, amb))
                hours += sc.hours
            per_set.append((tp / n, fp / hours))
        s = np.array([x[0] for x in per_set])
        a = np.array([x[1] for x in per_set])
        rows.append({"k_lost": k, "n_sets": len(per_set), "common_channels_left": n_ch - k,
                     "sensitivity_mean": round(float(s.mean()), 3), "sensitivity_range": [round(float(s.min()), 3), round(float(s.max()), 3)],
                     "fa_per_h_mean": round(float(a.mean()), 2), "fa_per_h_range": [round(float(a.min()), 2), round(float(a.max()), 2)]})
        print(json.dumps(rows[-1]), flush=True)
    base = rows[0]
    ok = [r["k_lost"] for r in rows if r["sensitivity_mean"] >= base["sensitivity_mean"] - 0.05
          and r["fa_per_h_mean"] <= 1.5 * base["fa_per_h_mean"]]
    res = {"rows": rows, "H1_k1_ok": 1 in ok,
           "H2_largest_k_ok_contiguous": max((k for k in KS if all(j in ok for j in KS[: KS.index(k) + 1])), default=0)}
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
