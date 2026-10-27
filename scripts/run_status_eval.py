"""Increment 46 (pre-registered, docs/preregistration_increment46.md): the critical 'status
epilepticus' finding under the current rule and rule T, on the increment-11 test patients and AUBMC.

    PYTHONPATH=src python scripts/run_status_eval.py build data/test46     # increment-11 test features (EDFs deleted)
    PYTHONPATH=src python scripts/run_status_eval.py assess data/test46 docs/increment46_assess.json
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_seizure_end as dev  # noqa: E402

TEST = [("chbmit", "chb06")] + [("siena", s) for s in ("PN01", "PN03", "PN06", "PN10", "PN14")]
AUBMC_FEATURES = Path("data/aubmc/features")
STATUS_THRESHOLD, STATUS_MIN_EPOCHS = 0.7, 15     # rule T, fixed by the pre-registration


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    items = [(db, a, str(out)) for db, s in TEST for a in dev.files_for(db, s)]
    print(f"{len(items)} records", flush=True)
    for it in items:                                  # one at a time: Siena files are large
        print(dev._one(it), flush=True)


def _records(out: Path) -> list:
    # local derived feature caches written by this project (build() above, increment 39 for AUBMC)
    paths = sorted(out.glob("*.pkl")) + sorted(AUBMC_FEATURES.glob("*.pkl"))
    return [pickle.loads(p.read_bytes()) for p in paths]


def assess(out: Path) -> dict:
    from neurolens.contracts.events import DetectionResult
    from neurolens.layer4_detect.ml_ictal import MLDetectorConfig, MLIctalDetector, load_model
    from neurolens.layer4_detect.seizure_burden import compute_seizure_burden
    from neurolens.layer4_detect.status_criteria import status_epilepticus
    from neurolens.pipeline.config_loader import load_configs

    cfg = load_configs()
    mode = cfg.ml.offline
    model = load_model(cfg.ml_model_path(mode.model))
    det_t = MLIctalDetector(model, MLDetectorConfig(threshold=mode.threshold, min_epochs=mode.min_epochs,
                                                    status_threshold=STATUS_THRESHOLD,
                                                    status_min_epochs=STATUS_MIN_EPOCHS))
    rows = []
    for rec in _records(out):
        events = det_t.detect(None, rec.features, None)
        sb = compute_seizure_burden(DetectionResult(events=events), rec.duration_s)
        current = status_epilepticus([(e.t_start, e.t_end) for e in events])["suspected"]
        truth = status_epilepticus([(s.onset_s, s.offset_s) for s in rec.annotation.seizures])["suspected"]
        rows.append({"record": f"{rec.annotation.database}:{rec.annotation.file}",
                     "hours": round(rec.duration_s / 3600, 2), "n_seizures": len(rec.annotation.seizures),
                     "truth_ese": truth, "current_critical": current,
                     "t_critical": sb.status_epilepticus_suspected, "t_review": sb.status_criteria_by_candidates})
    neg = [r for r in rows if not r["truth_ese"]]
    pos = [r for r in rows if r["truth_ese"]]
    cur_fp = sum(r["current_critical"] for r in neg)
    t_fp = sum(r["t_critical"] for r in neg)
    h1 = (t_fp / len(neg) <= 0.05) and (t_fp < cur_fp or cur_fp == 0) if neg else False
    h2 = all(r["t_critical"] or r["t_review"] for r in pos)
    return {"n_records": len(rows), "hours": round(sum(r["hours"] for r in rows), 1),
            "n_truth_ese": len(pos), "truth_ese_records": [r["record"] for r in pos],
            "non_ese": {"n": len(neg), "current_false_critical": cur_fp, "t_false_critical": t_fp,
                        "t_review_only": sum(r["t_review"] for r in neg),
                        "seizure_free_n": sum(1 for r in neg if r["n_seizures"] == 0),
                        "seizure_free_current_false": sum(r["current_critical"] for r in neg if r["n_seizures"] == 0),
                        "seizure_free_t_false": sum(r["t_critical"] for r in neg if r["n_seizures"] == 0)},
            "ese": [{k: r[k] for k in ("record", "current_critical", "t_critical", "t_review")} for r in pos],
            "H1": bool(h1), "H2": bool(h2) if pos else None,
            "decision": "enable_rule_t" if h1 and (h2 or not pos) else "keep_current",
            "records": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "assess"])
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    out = Path(a.paths[0])
    if a.cmd == "build":
        build(out)
        return 0
    res = assess(out)
    Path(a.paths[1]).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "records"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
