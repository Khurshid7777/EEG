"""Test-retest reliability of the routine-EEG conclusion (increment 23): the pipeline is run
separately on the first and the second half of each recording.

    PYTHONPATH=src python scripts/run_routine_halves.py IDS.json OUT.json CACHE [WORKERS]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(item):
    rid, cache = item
    import numpy as np

    from neurolens.datasets.nmt import AMPLITUDE_SCALE, NMTClient
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    c = NMTClient(cache)
    rec = next(r for r in c.labels() if r.record_id == rid)
    sig = ingest(c.fetch(c.member_of(rec)))
    sig = sig.with_signal((sig.signal * AMPLITUDE_SCALE).astype(np.float32))
    if rec.age_years is not None:
        sig = sig.model_copy(update={"patient": sig.patient.model_copy(update={"age_years": rec.age_years})})
    n = sig.n_samples // 2
    out = {"record_id": rid, "abnormal": rec.abnormal, "age_years": rec.age_years}
    for name, sl in (("first", slice(0, n)), ("second", slice(n, 2 * n))):
        x = sig.signal[:, sl].copy()
        part = sig.with_signal(x, duration_s=x.shape[1] / sig.sampling_rate_hz)
        try:
            o = Pipeline(None, provider_pref="template", run_ica=False).analyze_signal(part)
            rs = o.result_json.get("record_score") or {}
            out[name] = {"duration_s": part.duration_s, "rule_abnormal": o.result_json["record_summary"]["impression_abnormal"],
                         "probability": rs.get("probability"), "model_abnormal": rs.get("abnormal"),
                         "borderline": rs.get("borderline"), "summary": o.result_json["record_summary"]}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": repr(exc)}
    return out


def main() -> int:
    ids = json.loads(Path(sys.argv[1]).read_text())["record_ids"]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, [(i, sys.argv[3]) for i in ids]))
    Path(sys.argv[2]).write_text(json.dumps({"records": rows}, indent=1, default=float) + "\n")
    print(len(rows), "errors", sum("error" in r.get("first", {}) or "error" in r.get("second", {}) for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
