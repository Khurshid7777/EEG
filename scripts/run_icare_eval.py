"""Increment 35 (pre-registered, docs/preregistration_increment35.md): the frozen routine-EEG
verdict on I-CARE - one segment per patient (closest to hour 24 after return of circulation with
>= 20 min of EEG), its first 20 minutes. Resumable: one JSON line per patient; signal files are
deleted after each patient.

    PYTHONPATH=src python scripts/run_icare_eval.py OUT.jsonl [--workers 4]
    PYTHONPATH=src python scripts/run_icare_eval.py --assess IN.jsonl OUT.json
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(rec):
    import numpy as np

    from neurolens.contracts.signal import PatientInfo
    from neurolens.datasets.icare import IcareClient
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    cl = IcareClient()
    base = {"record_id": rec.record_id, "patient": rec.patient}
    try:
        rec = cl.with_meta(rec)
        base.update(hospital=rec.hospital, age_years=rec.age_years, poor_outcome=rec.poor_outcome,
                    cpc=rec.meta.get("CPC"))
        seg = cl.select_segment(rec)
        if seg is None:
            return {**base, "error": "no segment with >= 20 min of EEG"}
        stem, hour, hdr = seg
        sig = ingest(cl.fetch(rec, stem, hdr), patient=PatientInfo(age_years=rec.age_years))
        amp = float(np.median(np.std(sig.signal, axis=1)))
        out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(sig)
        row = dict(out.result_json["record_summary"])
        rs = out.result_json.get("record_score") or {}
        return {**base, "segment": stem, "hour": hour, "n_channels": len(sig.channel_names),
                "duration_s": row.get("duration_s"), "amplitude_sd_uv": amp,
                "impression_abnormal": bool(rs["abnormal"]) if rs.get("decides") else row["impression_abnormal"],
                "rule_impression_abnormal": row["impression_abnormal"], "record_probability": rs.get("probability"),
                "record_borderline": rs.get("borderline"), "pdr_hz": row.get("pdr_hz"),
                "detectors_fired": row.get("detectors_fired"), "pathologic_nodes": row.get("pathologic_nodes")}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}
    finally:
        cl.drop(rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--assess", action="store_true")
    a = ap.parse_args()
    if a.assess:
        from neurolens.evaluation.prereg35 import assess

        res = assess([json.loads(ln) for ln in Path(a.paths[0]).read_text().splitlines()])
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps(res, indent=1, default=float))
        return 0
    from neurolens.datasets.icare import IcareClient

    out = Path(a.paths[0])
    recs = IcareClient().records()
    done = {json.loads(ln)["patient"] for ln in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in recs if r.patient not in done]
    print(f"{len(recs)} patients, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
