"""Increment 47 (pre-registered, docs/preregistration_increment47.md): the real-time monitor's
'status epilepticus' alarm at its operating point (learned v2, 0.7 x 5 epochs).

    PYTHONPATH=src python scripts/run_monitor_status_eval.py dev data/dev46        # development proxy
    PYTHONPATH=src python scripts/run_monitor_status_eval.py assess docs/increment47_assess.json
"""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

AUBMC_STREAMS = Path("data/aubmc/stream")
FROZEN = Path("docs/preregistration_increment11_frozen.json")
MIN_EPOCHS_T = 15


def _intervals(t, p, thr, min_epochs):
    from neurolens.layer4_detect.ml_ictal import probability_runs

    return [(float(t[a]), float(t[b - 1])) for a, b, _ in probability_runs(p, thr, min_epochs)]


def dev(out: Path) -> dict:
    """Training patients, out-of-fold probabilities of increment 46 (offline features: a proxy)."""
    import dev_seizure_end as d
    from neurolens.layer4_detect.status_criteria import status_epilepticus

    ds = d._dataset(out)
    z = np.load(out / "loso_probs.npz")
    probs = [z[f"arr_{i}"] for i in range(len(z.files))]
    rows = []
    for off in (0.7, 0.5, 0.4, 0.3, 0.2, 0.1):
        for bridge in (0, 5, 15, 30):
            fp = tp = 0
            longest = []
            for rec, t, p in zip(ds.records, ds.times, probs):
                truth = status_epilepticus([(s.onset_s, s.offset_s) for s in rec.annotation.seizures])["suspected"]
                iv = [(float(t[a]), float(t[b - 1])) for a, b in d.hysteresis_runs(p, 0.7, off, 5, bridge)]
                st = status_epilepticus(iv)
                if truth:
                    tp += st["suspected"]
                    longest.append(st["longest_s"])
                else:
                    fp += st["suspected"]
            rows.append({"thr_off": off, "bridge_epochs": bridge, "true_found": tp,
                         "true_longest_s": longest, "false_status": fp})
    return {"n_records": len(ds.records), "variants": rows}


def assess() -> dict:
    from neurolens.layer4_detect.ml_ictal import load_model
    from neurolens.layer4_detect.status_criteria import status_epilepticus
    from neurolens.pipeline.config_loader import load_configs

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))["realtime"]["model"]
    if hashlib.sha256(Path(frozen["path"]).read_bytes()).hexdigest() != frozen["sha256"]:
        raise RuntimeError("realtime model does not match the frozen sha256")
    mode = load_configs().ml.realtime
    model = load_model(frozen["path"])
    from neurolens.layer4_detect.ml_ictal import featurize

    rows = []
    # local monitor-stream caches written by this project in increment 39 (scripts/run_aubmc_eval.py)
    for path in sorted(AUBMC_STREAMS.glob("stream_*.pkl")):
        rec, _ = pickle.loads(path.read_bytes())
        X = featurize(rec.features, 1)
        t = np.asarray(rec.features.epoch_times, float)
        p = model.predict_proba(X) if len(X) else np.zeros(0)
        amb = [(s.onset_s, s.offset_s) for s in getattr(rec.annotation, "ambiguous", [])]
        truth = status_epilepticus([(s.onset_s, s.offset_s) for s in rec.annotation.seizures])
        cur = status_epilepticus(_intervals(t, p, mode.threshold, mode.min_epochs))
        t15 = status_epilepticus(_intervals(t, p, mode.threshold, MIN_EPOCHS_T))
        rows.append({"record": rec.annotation.file, "hours": round(rec.duration_s / 3600, 2),
                     "n_seizures": len(rec.annotation.seizures), "n_ambiguous": len(amb),
                     "truth_ese": truth["suspected"], "truth_longest_s": truth["longest_s"],
                     "monitor_status": cur["suspected"], "monitor_longest_s": cur["longest_s"],
                     "monitor_max_s_per_hour": cur["max_seizure_s_per_hour"],
                     "min15_status": t15["suspected"]})
    neg = [r for r in rows if not r["truth_ese"]]
    pos = [r for r in rows if r["truth_ese"]]
    fp = sum(r["monitor_status"] for r in neg)
    h1 = fp / len(neg) <= 0.05 if neg else False
    return {"operating_point": {"threshold": mode.threshold, "min_epochs": mode.min_epochs},
            "n_records": len(rows), "hours": round(sum(r["hours"] for r in rows), 1),
            "non_ese": {"n": len(neg), "monitor_false_status": fp,
                        "min15_false_status": sum(r["min15_status"] for r in neg)},
            "ese": [{k: r[k] for k in ("record", "truth_longest_s", "monitor_status", "monitor_longest_s",
                                       "monitor_max_s_per_hour")} for r in pos],
            "H1": bool(h1), "H2_found": [r["monitor_status"] for r in pos],
            "decision": {"status_rule": "unchanged" if h1 else "min_epochs_15_then_retest",
                         "summary_note_and_R1": not all(r["monitor_status"] for r in pos) if pos else False},
            "records": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "assess"])
    ap.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "dev":
        res = dev(Path(a.path))
        (Path(a.path) / "monitor_status_dev.json").write_text(json.dumps(res, indent=1) + "\n")
        for r in res["variants"]:
            print(json.dumps(r))
        return 0
    res = assess()
    Path(a.path).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
