"""Increment 70 (pre-registered, docs/preregistration_increment70.md): the frozen routine-EEG verdict on
the 36 healthy adults of PhysioNet eegmat (rest = suffix 1, mental arithmetic = suffix 2).

    PYTHONPATH=src python scripts/run_eegmat_eval.py OUT.jsonl --suffix 1
    PYTHONPATH=src python scripts/run_eegmat_eval.py --assess REST.jsonl TASK.jsonl OUT.json
"""

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

BASE = "https://physionet-open.s3.amazonaws.com/eegmat/1.0.0"
CACHE = Path("data/eegmat")
STANDARD_1020 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
                 "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]


def _get(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", str(p), f"{BASE}/{name}"], check=True)
    return p


def subjects() -> list[dict]:
    rows = csv.DictReader(io.StringIO(_get("subject-info.csv").read_text()))
    return [{"subject": r["Subject"], "age": float(r["Age"])} for r in rows]


def _one(item):
    s, suffix = item
    import numpy as np

    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    base = {"subject": s["subject"], "age_years": s["age"], "suffix": suffix}
    try:
        sig = ingest(_get(f"{s['subject']}_{suffix}.edf"), patient=PatientInfo(age_years=s["age"]))
        idx = [sig.channel_names.index(c) for c in STANDARD_1020 if c in sig.channel_names]
        sig = sig.model_copy(update={"signal": np.ascontiguousarray(sig.signal[idx]),
                                     "channel_names": [sig.channel_names[i] for i in idx]})
        out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(sig)
        row = dict(out.result_json["record_summary"])
        rs = out.result_json.get("record_score") or {}
        return {**base, "n_channels": len(idx), "duration_s": row.get("duration_s"),
                "decides": bool(rs.get("decides")),
                "impression_abnormal": bool(rs["abnormal"]) if rs.get("decides") else row["impression_abnormal"],
                "rule_impression_abnormal": row["impression_abnormal"], "record_probability": rs.get("probability"),
                "record_borderline": rs.get("borderline"), "detectors_fired": row.get("detectors_fired"),
                "pathologic_nodes": row.get("pathologic_nodes")}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--suffix", default="1", choices=["1", "2"])
    ap.add_argument("--assess", action="store_true")
    a = ap.parse_args()
    if a.assess:
        from neurolens.evaluation.prereg34 import SPECIFICITY_MIN, _spec

        load = lambda p: [json.loads(ln) for ln in Path(p).read_text().splitlines()]  # noqa: E731
        rest, task = load(a.paths[0]), load(a.paths[1])
        res = {"rest": _spec(rest), "task": _spec(task),
               "rest_not_decided": sum(1 for r in rest if "error" not in r and not r.get("decides")),
               "task_not_decided": sum(1 for r in task if "error" not in r and not r.get("decides")),
               "hypotheses": {"H1_rest_specificity_at_least_0.65": _spec(rest)["specificity"] >= SPECIFICITY_MIN}}
        Path(a.paths[2]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps(res, indent=1, default=float))
        return 0
    out = Path(a.paths[0])
    with out.open("w") as f:
        for s in subjects():
            f.write(json.dumps(_one((s, a.suffix)), default=float) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
