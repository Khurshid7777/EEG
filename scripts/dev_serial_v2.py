"""Increment 43, development only (OpenNeuro ds004148 sub-01..30 - the development part of
increment 42): epoch-level band shares and the posterior alpha per epoch, saved so that a
state-aware serial comparison (alert epochs only) can be designed before its pre-registration.

    PYTHONPATH=src python scripts/dev_serial_v2.py run OUT.jsonl [--workers 6]
    PYTHONPATH=src python scripts/dev_serial_v2.py explore OUT.jsonl
"""

import argparse
import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ds004148_serial import BASE, STANDARD_1020, _curl, _load, ages  # noqa: E402

ITEMS = [(1, "eyesclosed"), (3, "eyesclosed"), (1, "eyesopen")]


def _one(item) -> dict:
    sub, ses, task, age = item
    import numpy as np

    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer3_features.background import posterior_dominant_rhythm
    from neurolens.pipeline.pipeline import Pipeline

    base = {"subject": sub, "session": ses, "task": task}
    stem = f"{sub}_ses-session{ses}_task-{task}_eeg"
    try:
        with tempfile.TemporaryDirectory() as d:
            for ext in (".vhdr", ".vmrk", ".eeg"):
                _curl(f"{BASE}/{sub}/ses-session{ses}/eeg/{stem}{ext}", Path(d) / f"{stem}{ext}")
            sig = ingest(Path(d) / f"{stem}.vhdr", patient=PatientInfo(age_years=age))
            idx = [sig.channel_names.index(c) for c in STANDARD_1020 if c in sig.channel_names]
            sig = sig.model_copy(update={"signal": np.ascontiguousarray(sig.signal[idx]),
                                         "channel_names": [sig.channel_names[i] for i in idx]})
            out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(sig)
        f, a = out.features, out.signal
        post = [f.eeg_channels.index(c) for c in ("O1", "O2") if c in f.eeg_channels]
        o = [a.channel_names.index(c) for c in ("O1", "O2")]
        x = np.asarray(a.signal, float)
        row = out.result_json["record_summary"]
        return {**base, **{k: row.get(k) for k in ("pdr_hz", "global_rel_delta", "global_rel_theta",
                                                    "global_rel_alpha", "robust_amplitude_uv")},
                "pdr_alert_hz": posterior_dominant_rhythm(x[o[0]], x[o[1]], a.sampling_rate_hz, method="alert"),
                "ep_delta": np.round(f.epoch_relpow["delta"].mean(axis=1), 4).tolist(),
                "ep_theta": np.round(f.epoch_relpow["theta"].mean(axis=1), 4).tolist(),
                "ep_alpha": np.round(f.epoch_relpow["alpha"].mean(axis=1), 4).tolist(),
                "ep_post_alpha": np.round(f.epoch_relpow["alpha"][:, post].mean(axis=1), 4).tolist()}
    except Exception as exc:  # noqa: BLE001
        return {**base, "error": repr(exc)}


def alert_shares(r: dict, q: float) -> dict:
    """Mean band shares over epochs whose posterior alpha share is at or above the record's
    q-quantile (q = 0: all epochs)."""
    import numpy as np

    pa = np.asarray(r["ep_post_alpha"])
    keep = pa >= np.quantile(pa, q) if q > 0 else np.ones(pa.size, bool)
    return {f"global_rel_{b}": float(np.asarray(r[f"ep_{b}"])[keep].mean()) for b in ("delta", "theta", "alpha")}


def explore(path: Path) -> None:
    import math

    import numpy as np

    rows = [r for r in _load(path) if "error" not in r]
    by = {(r["subject"], r["session"], r["task"]): r for r in rows}
    subs = sorted({r["subject"] for r in rows})
    for q in (0.0, 0.5, 0.75, 0.9):
        s = {k: {**v, **alert_shares(v, q)} for k, v in by.items()}
        for b in ("delta", "theta", "alpha"):
            k = f"global_rel_{b}"
            stab = np.array([math.log(s[(u, 3, "eyesclosed")][k] / s[(u, 1, "eyesclosed")][k]) for u in subs])
            eo = np.array([math.log(s[(u, 1, "eyesopen")][k] / s[(u, 1, "eyesclosed")][k]) for u in subs
                           if (u, 1, "eyesopen") in s])
            lim = 2.576 * np.sqrt(np.mean(stab ** 2))
            print(f"q={q:.2f} {b:5s} limit {lim:.2f}  EO median {np.median(eo):+.2f}  "
                  f"EO beyond limit {np.mean(np.abs(eo) > lim):.2f}")
    for k in ("pdr_hz", "pdr_alert_hz"):
        d = np.array([s[(u, 3, "eyesclosed")][k] - s[(u, 1, "eyesclosed")][k] for u in subs
                      if s[(u, 3, "eyesclosed")][k] > 0 and s[(u, 1, "eyesclosed")][k] > 0])
        print(f"{k}: limit {2.576 * np.sqrt(np.mean(d ** 2)):.2f} Hz (n={d.size})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "explore"])
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "explore":
        explore(out)
        return 0
    ag = ages()
    items = [(f"sub-{i:02d}", ses, task, ag.get(f"sub-{i:02d}")) for i in range(1, 31) for ses, task in ITEMS]
    done = {(r["subject"], r["session"], r["task"]) for r in _load(out)} if out.exists() else set()
    todo = [it for it in items if it[:3] not in done]
    print(f"{len(items)} recordings, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("subject", "session", "task", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
