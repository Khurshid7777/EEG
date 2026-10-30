"""Increment 53: robustness of the offline pipeline to signal gaps (NaN samples, e.g. EEGLAB exports of
home or ICU recordings with disconnections). Healthy recordings (NMT normal, OpenNeuro ds003775):
any seizure candidate is false. Each recording is analysed as is and with gaps inserted - two 10-s
gaps on all channels (30% and 60% of the recording) and a 60-s gap on one channel (45%). Run with the
code under test on PYTHONPATH (old and new code give two JSON files to compare).

    PYTHONPATH=src python scripts/run_gap_robustness.py OUT.json
"""

import glob
import json
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
RECORDS = sorted(glob.glob("data/nmt/normal/*.edf"))[:10] + \
    sorted(glob.glob("data/openneuro/ds003775/sub-*/ses-t1/eeg/*resteyesc_eeg.edf"))[:10]


def with_gaps(sig):
    g = sig.model_copy(deep=True)
    fs, n = g.sampling_rate_hz, g.signal.shape[1]
    for f in (0.3, 0.6):
        a = int(n * f)
        g.signal[:, a:a + int(10 * fs)] = np.nan
    a = int(n * 0.45)
    g.signal[0, a:a + int(60 * fs)] = np.nan
    return g


def summarise(out) -> dict:
    ev = out.result_json["events"]
    return {"ictal": [(e["t_start"], e["t_end"]) for e in ev if e["code"] == "ictal_ml"],
            "codes": sorted({e["code"] for e in ev}),
            "record_probability": (out.result_json.get("record_score") or {}).get("probability"),
            "abnormal": (out.result_json.get("record_summary") or {}).get("impression_abnormal")}


def main() -> int:
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    pipe = Pipeline(provider_pref="template", run_ica=False)
    res = {}
    for p in RECORDS:
        sig = ingest(p)
        row = {"duration_s": round(sig.duration_s, 1)}
        for tag, s in (("as_is", sig), ("gaps", with_gaps(sig))):
            try:
                row[tag] = summarise(pipe.analyze_signal(s))
            except Exception as exc:  # noqa: BLE001
                row[tag] = {"error": repr(exc)}
        res[p] = row
        print(p, {k: (v.get("ictal"), v.get("abnormal")) if isinstance(v, dict) else v for k, v in row.items()},
              flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
