"""Increment 41 (pre-registered, docs/preregistration_increment41.md): eyes-closed PDR and
alpha reactivity on PhysioNet EEGMMIDB baselines (R02 eyes closed + R01 eyes open, joined with
markers). Resumable: one JSON line per subject.

    PYTHONPATH=src python scripts/run_eegmmidb_reactivity.py OUT.jsonl [--workers 6] [--assess OUT.json]
"""

import argparse
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE = "https://physionet.org/files/eegmmidb/1.0.0"
SUBJECTS = [f"S{i:03d}" for i in range(1, 110)]


def _prep(path):
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer2_preprocess.filters import apply_filters
    from neurolens.layer2_preprocess.reref import rereference
    from neurolens.pipeline.config_loader import load_configs

    return rereference(apply_filters(ingest(path), load_configs().filters), scheme="average")


def _one(subject: str) -> dict:
    import numpy as np

    from neurolens.contracts.signal import Marker
    from neurolens.layer3_features.procedures import analyze_procedures

    try:
        with tempfile.TemporaryDirectory() as d:
            parts = {}
            for run in ("R02", "R01"):
                p = Path(d) / f"{subject}{run}.edf"
                subprocess.run(["curl", "-sSfL", "--retry", "5", "-o", str(p), f"{BASE}/{subject}/{subject}{run}.edf"],
                               check=True, capture_output=True)
                parts[run] = _prep(p)
            ec, eo = parts["R02"], parts["R01"]
            if ec.channel_names != eo.channel_names or ec.sampling_rate_hz != eo.sampling_rate_hz:
                raise ValueError("runs differ in channels or rate")
            t_open = ec.n_samples / ec.sampling_rate_hz
            joined = ec.with_signal(np.concatenate([np.asarray(ec.signal), np.asarray(eo.signal)], axis=1),
                                    duration_s=None,
                                    markers=[Marker(onset_s=0.0, text="Eyes closed"),
                                             Marker(onset_s=t_open, text="Eyes open")])
            p = analyze_procedures(joined)
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"subject": subject, "error": repr(exc)}
    return {"subject": subject, "ec_pdr_hz": p["ec_pdr_hz"], "alpha_attenuation": p["alpha_attenuation"],
            "reactive": p["reactive"], "eyes_closed_s": p["eyes_closed_s"], "eyes_open_s": p["eyes_open_s"]}


def assess(rows: list[dict]) -> dict:
    import numpy as np

    ok = [r for r in rows if "error" not in r and r["alpha_attenuation"] is not None]
    att = np.array([r["alpha_attenuation"] for r in ok])
    pdr = [r["ec_pdr_hz"] for r in ok]
    found = [x for x in pdr if x]
    reactive = float(np.mean([r["reactive"] for r in ok])) if ok else float("nan")
    h1 = bool(reactive >= 0.70 and np.median(att) >= 0.5)
    h2_found = len(found) / len(ok) if ok else float("nan")
    h2_alpha = float(np.mean([8 <= x <= 13 for x in found])) if found else float("nan")
    return {"n": len(rows), "n_analysed": len(ok), "errors": [r["subject"] for r in rows if "error" in r],
            "reactive_fraction": round(reactive, 3), "attenuation_median": round(float(np.median(att)), 3),
            "attenuation_iqr": [round(float(q), 3) for q in np.percentile(att, [25, 75])],
            "ec_pdr_found_fraction": round(h2_found, 3), "ec_pdr_in_8_13_fraction": round(h2_alpha, 3),
            "ec_pdr_median_hz": float(np.median(found)) if found else None,
            "H1": h1, "H2": bool(h2_found >= 0.8 and h2_alpha >= 0.9),
            "decision": ("include" if h1 and h2_found >= 0.8 and h2_alpha >= 0.9 else
                         "revise" if reactive < 0.5 else "descriptive_with_limitation")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--assess")
    a = ap.parse_args()
    out = Path(a.out)
    done = {json.loads(ln)["subject"] for ln in out.read_text().splitlines()} if out.exists() else set()
    todo = [s for s in SUBJECTS if s not in done]
    print(f"{len(SUBJECTS)} subjects, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for row in ex.map(_one, todo):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(row, flush=True)
    if a.assess:
        res = assess([json.loads(ln) for ln in out.read_text().splitlines()])
        Path(a.assess).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
