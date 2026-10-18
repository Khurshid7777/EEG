"""Increment 34 (pre-registered, docs/preregistration_increment34.md): the frozen pipeline on
the 111 healthy adults of OpenNeuro ds003775, as delivered (64 electrodes) or reduced to the
19 10-20 electrodes (--reduce1020). Resumable: one JSON line per participant.

    PYTHONPATH=src python scripts/run_ds003775_eval.py OUT.jsonl [--reduce1020] [--workers 4]
    PYTHONPATH=src python scripts/run_ds003775_eval.py --assess DELIVERED.jsonl REDUCED.jsonl OUT.json
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

STANDARD_1020 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
                 "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]


def _one(item):
    rec, reduce = item
    import numpy as np

    from neurolens.contracts.signal import PatientInfo
    from neurolens.datasets.ds004504 import Ds003775Client
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    base = {"record_id": rec.record_id, "subject": rec.subject, "age_years": rec.age_years, "reduced": reduce}
    try:
        sig = ingest(Ds003775Client().fetch(rec), patient=PatientInfo(age_years=rec.age_years))
        if reduce:
            idx = [sig.channel_names.index(c) for c in STANDARD_1020 if c in sig.channel_names]
            sig = sig.model_copy(update={"signal": np.ascontiguousarray(sig.signal[idx]),
                                         "channel_names": [sig.channel_names[i] for i in idx]})
        out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(sig)
        row = dict(out.result_json["record_summary"])
        rs = out.result_json.get("record_score") or {}
        return {**base, "n_channels": len(sig.channel_names), "duration_s": row.get("duration_s"),
                "impression_abnormal": bool(rs["abnormal"]) if rs.get("decides") else row["impression_abnormal"],
                "rule_impression_abnormal": row["impression_abnormal"], "record_probability": rs.get("probability"),
                "record_borderline": rs.get("borderline"), "pdr_hz": row.get("pdr_hz"),
                "detectors_fired": row.get("detectors_fired"), "pathologic_nodes": row.get("pathologic_nodes")}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--reduce1020", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--assess", action="store_true")
    a = ap.parse_args()
    if a.assess:
        from neurolens.evaluation.prereg34 import assess

        load = lambda p: [json.loads(ln) for ln in Path(p).read_text().splitlines()]  # noqa: E731
        res = assess(load(a.paths[0]), load(a.paths[1]))
        Path(a.paths[2]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps(res, indent=1, default=float))
        return 0
    from neurolens.datasets.ds004504 import Ds003775Client

    out = Path(a.paths[0])
    recs = Ds003775Client().records()
    done = {json.loads(ln)["subject"] for ln in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in recs if r.subject not in done]
    print(f"{len(recs)} participants, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, [(r, a.reduce1020) for r in todo]):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
