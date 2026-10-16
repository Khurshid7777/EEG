"""Robustness of the routine-EEG conclusion to acquisition differences (increment 25):
the same recordings resampled to 256 Hz, without ear electrodes (A1/A2), or without
midline electrodes (Fz/Cz/Pz).

    PYTHONPATH=src python scripts/run_routine_perturb.py IDS.json OUT.json CACHE [WORKERS]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PERTURBATIONS = ("resample_256", "no_ears", "no_midline")


def _perturb(sig, name):
    import numpy as np
    from scipy.signal import resample_poly

    if name == "resample_256":
        x = resample_poly(sig.signal.astype(np.float64), 32, 25, axis=1).astype(np.float32)  # 200 -> 256 Hz
        return sig.with_signal(x, sampling_rate_hz=256.0, duration_s=x.shape[1] / 256.0)
    drop = {"no_ears": {"A1", "A2"}, "no_midline": {"FZ", "CZ", "PZ"}}[name]
    keep = [i for i, n in enumerate(sig.channel_names) if n.upper() not in drop]
    return sig.with_signal(sig.signal[keep].copy(), channel_names=[sig.channel_names[i] for i in keep])


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
    out = {"record_id": rid, "abnormal": rec.abnormal, "age_years": rec.age_years}
    for name in PERTURBATIONS:
        try:
            o = Pipeline(None, provider_pref="template", run_ica=False).analyze_signal(_perturb(sig, name))
            rs = o.result_json.get("record_score") or {}
            out[name] = {"probability": rs.get("probability"), "model_abnormal": rs.get("abnormal"),
                         "rule_abnormal": o.result_json["record_summary"]["impression_abnormal"]}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": repr(exc)}
    return out


def main() -> int:
    ids = json.loads(Path(sys.argv[1]).read_text())["record_ids"]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, [(i, sys.argv[3]) for i in ids]))
    Path(sys.argv[2]).write_text(json.dumps({"records": rows}, indent=1, default=float) + "\n")
    print(len(rows), "errors", sum(any("error" in r.get(p, {}) for p in PERTURBATIONS) for r in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
