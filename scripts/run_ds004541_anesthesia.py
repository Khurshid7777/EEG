"""Increment 67 (pre-registered, docs/preregistration_increment67.md): replication of the sedation effect (increment 66) on OpenNeuro
ds004541 (CC0): 8 patients under general anesthesia, 58-channel EEG at 1000 Hz. AWAKE = before the drug marker 'start', UNCONSCIOUS = loc+60 s .. roc.

    PYTHONPATH=src python scripts/run_ds004541_anesthesia.py check            # reader/units check on sub-11 (excluded)
    PYTHONPATH=src python scripts/run_ds004541_anesthesia.py run OUT.json
"""

import csv
import io
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

BASE = "https://s3.amazonaws.com/openneuro.org/ds004541"
SUBJECTS = ["sub-02", "sub-03", "sub-04", "sub-07", "sub-08", "sub-09", "sub-10"]
CHECK = "sub-11"
MIN_S, SEG_S, LOC_SKIP_S = 120.0, 300.0, 60.0
KEEP = {"FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T7", "T3", "C3", "CZ", "C4", "T8", "T4", "P7", "T5", "P3", "PZ", "P4", "P8", "T6", "O1", "O2"}


def events(sub: str):
    t = urllib.request.urlopen(f"{BASE}/{sub}/ses-01/eeg/{sub}_ses-01_task-anesthesia_events.tsv", timeout=60).read().decode("utf-8-sig")
    return [(float(r["onset"]), r["trial_type"]) for r in csv.DictReader(io.StringIO(t), delimiter="\t")]


def windows(ev):
    start = min([t for t, k in ev if k == "start"] or [None], key=lambda x: x if x is not None else 1e9)
    loc = min([t for t, k in ev if k == "loc"] or [None], key=lambda x: x if x is not None else 1e9)
    roc = [t for t, k in ev if k == "roc"]
    awake = (0.0, start) if start else None
    unc = None
    if loc is not None and roc:
        a = loc + LOC_SKIP_S
        b = min(a + SEG_S, [r for r in roc if r > loc][0]) if [r for r in roc if r > loc] else None
        if b and b - a >= MIN_S:
            unc = (a, b)
    return awake, unc


def analyse(sig, win):
    from neurolens.contracts.signal import UnifiedSignal
    from neurolens.layer3_features.feature_set import compute_features
    from neurolens.layer4_detect.background_abn import DiffuseSlowingDetector
    from neurolens.layer4_detect.suppression_events import BurstSuppressionDetector
    from neurolens.pipeline.config_loader import load_configs

    fs = sig.sampling_rate_hz
    a, b = win
    x = sig.signal[:, int(a * fs):int(b * fs)]
    if x.shape[1] / fs < MIN_S:
        return None
    s = UnifiedSignal(signal=x.astype(np.float32), sampling_rate_hz=fs, channel_names=list(sig.channel_names),
                      duration_s=x.shape[1] / fs)
    cfg = load_configs()
    feats = compute_features(s)
    dtf = np.median(feats.epoch_relpow["delta"] + feats.epoch_relpow["theta"], axis=1)
    return {"seconds": round(x.shape[1] / fs, 1),
            "diffuse_slowing": bool(DiffuseSlowingDetector().detect(s, feats, cfg.thresholds)),
            "burst_suppression": bool(BurstSuppressionDetector().detect(s, feats, cfg.thresholds)),
            "slow_epoch_fraction": round(float((dtf >= cfg.thresholds.diffuse_slowing_delta_theta_ratio).mean()), 3)}


def one(sub: str) -> dict:
    from neurolens.layer1_ingest.edf import EdfIngestor

    ev = events(sub)
    awake, unc = windows(ev)
    with tempfile.TemporaryDirectory(prefix="ds4541_") as d:
        p = Path(d) / f"{sub}.edf"
        p.write_bytes(urllib.request.urlopen(f"{BASE}/{sub}/ses-01/eeg/{sub}_ses-01_task-anesthesia_eeg.edf", timeout=900).read())
        sig = EdfIngestor().read(p, include=lambda n: n in KEEP)
    from neurolens.layer2_preprocess.filters import apply_filters
    from neurolens.pipeline.config_loader import load_configs

    sig = apply_filters(sig, load_configs().filters)   # the product's own band-pass/notch (raw DC-coupled EDF has huge offsets)
    out = {"subject": sub, "channels": list(sig.channel_names), "fs": sig.sampling_rate_hz, "duration_s": round(sig.duration_s, 1),
           "std_uv": [round(float(v), 1) for v in sig.signal.std(axis=1)][:5], "awake": None, "unconscious": None}
    if awake and awake[1] - awake[0] >= MIN_S:
        out["awake"] = analyse(sig, awake)
    if unc:
        out["unconscious"] = analyse(sig, unc)
    return out


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "check":
        print(events(CHECK))
        print(windows(events(CHECK)))
        r = one(CHECK)
        print(r)
        return 0
    rows = []
    for s in SUBJECTS:
        try:
            rows.append(one(s))
        except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
            rows.append({"subject": s, "error": repr(exc)})
        print(rows[-1], flush=True)
    ok = [r for r in rows if "error" not in r]
    res = {"rows": rows,
           "unconscious_slowing": f"{sum(bool(r['unconscious'] and r['unconscious']['diffuse_slowing']) for r in ok)}/{sum(bool(r['unconscious']) for r in ok)}",
           "awake_slowing": f"{sum(bool(r['awake'] and r['awake']['diffuse_slowing']) for r in ok)}/{sum(bool(r['awake']) for r in ok)}",
           "unconscious_burst_suppression": f"{sum(bool(r['unconscious'] and r['unconscious']['burst_suppression']) for r in ok)}/{sum(bool(r['unconscious']) for r in ok)}"}
    Path(sys.argv[2]).write_text(json.dumps(res, indent=1) + "\n")
    print({k: v for k, v in res.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
