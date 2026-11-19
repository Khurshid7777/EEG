"""Increment 66 (pre-registered, docs/preregistration_increment66.md): do the product's diffuse-slowing and burst-suppression
detectors fire under propofol sedation? DOSE-I (Zenodo 22145442, CC BY 4.0). Per recording: AWAKE (<= 5 min before the first
propofol dose, MOAA/S >= 4) vs DEEP (<= 5 min from LOC+60 s, MOAA/S <= 2), 2 channels at 125 Hz named F7/F8 (assumption).

    PYTHONPATH=src python scripts/run_dosei_sedation.py check            # reader/units check on record 10-154 (excluded)
    PYTHONPATH=src python scripts/run_dosei_sedation.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_dosei_sedation.py assess OUT.jsonl OUT.json
"""

import csv
import datetime as dt
import io
import json
import sys
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

API = "https://zenodo.org/api/records/22145442"
CHECK = "10-154"
FS = 125.0
SEG_S, MIN_S, MIN_AWAKE_S, LOC_SKIP_S = 300.0, 120.0, 60.0, 60.0   # amendment 1: awake = after ROC, min 60 s
SEED, N_BOOT = 66, 2000
ART_ZIP = Path("data/inc66/dosei/artifacts.zip")


def _zip():
    from neurolens.datasets.nmt_events import HttpRangeFile

    d = json.load(urllib.request.urlopen(API, timeout=60))
    f = next(x for x in d["files"] if x["key"] == "data.zip")
    return zipfile.ZipFile(io.BufferedReader(HttpRangeFile(f["links"]["self"], f["size"]), buffer_size=8 << 20))


def _t(s: str) -> float:
    s = s.strip()
    if "." not in s:
        s += ".000"
    return dt.datetime.strptime(s[:23], "%Y-%m-%d %H:%M:%S.%f").timestamp()


def read_record(z, rid: str):
    """(eeg float32 [2, n] at 125 Hz, sample times (s from start), events [(t, kind, value)])."""
    t, e1, e2, ev = [], [], [], []
    with z.open(f"data/{rid}.csv") as fh:
        for row in csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8")):
            a, b = row.get("Intellivue/EEG_1"), row.get("Intellivue/EEG_2")
            if a and b:
                t.append(row["Time"]); e1.append(float(a)); e2.append(float(b))
            k = row.get("EVENT")
            if k:
                ev.append((row["Time"], k))
    t0 = _t(t[0])
    return (np.array([e1, e2], dtype=np.float32), np.array([_t(x) - t0 for x in t]), [(_t(x) - t0, k) for x, k in ev])


def artifact_epochs(rid: str, t0_abs: float, n_s: float) -> np.ndarray:
    """Boolean per second: True when a DOSE-I (algorithmic) artifact flag covers it."""
    out = np.zeros(int(n_s) + 1, bool)
    with zipfile.ZipFile(ART_ZIP) as az:
        name = f"artifacts/artifacts/{rid}_artifacts.csv"
        if name not in az.namelist():
            return out
        for row in csv.DictReader(io.TextIOWrapper(az.open(name), encoding="utf-8")):
            if row["is_artifact"] == "True":
                a, b = _t(row["begin_idx"]) - t0_abs, _t(row["end_idx"]) - t0_abs
                out[max(0, int(a)):min(len(out), int(np.ceil(b)))] = True
    return out


def segments(ev):
    """AWAKE and DEEP time windows [(a, b)] from the events (seconds from the first EEG sample)."""
    moaas = sorted((t, int(k[5:])) for t, k in ev if k.startswith("MOAAS") and k[5:].isdigit())
    props = sorted(t for t, k in ev if k == "PROP")
    locs = sorted(t for t, k in ev if k == "LOC")
    rocs = sorted(t for t, k in ev if k == "ROC")

    def last_moaas(t):
        v = [m for tt, m in moaas if tt <= t]
        return v[-1] if v else None

    # amendment 1 (docs/preregistration_increment66.md): AWAKE = after the last ROC (+30 s) and >= 120 s after the last dose, MOAA/S >= 4
    awake = None
    if rocs:
        a = rocs[-1] + 30.0
        a = max(a, max([p for p in props if p < a] or [0.0]) + 120.0)
        end = a + SEG_S
        pts = [m for tt, m in moaas if a <= tt <= end]
        if not pts or min(pts) >= 4:
            awake = (a, end)
    deep = None
    if locs:
        a = locs[0] + LOC_SKIP_S
        nxt = [r for r in rocs if r > locs[0]]
        b = min(a + SEG_S, nxt[0] if nxt else a + SEG_S)
        low = [tt for tt, m in moaas if a <= tt <= b and m <= 2]
        if b - a >= MIN_S and (low or (last_moaas(a) is not None and last_moaas(a) <= 2)):
            deep = (a, b)
    return awake, deep


def analyse(sig_arr, times, win, art, min_s=MIN_S):
    from neurolens.contracts.signal import UnifiedSignal
    from neurolens.layer3_features.feature_set import compute_features
    from neurolens.layer4_detect.background_abn import DiffuseSlowingDetector
    from neurolens.layer4_detect.suppression_events import BurstSuppressionDetector
    from neurolens.pipeline.config_loader import load_configs

    a, b = win
    m = (times >= a) & (times < b)
    sec = np.clip(times[m].astype(int), 0, len(art) - 1)
    ok = ~art[sec]
    x = sig_arr[:, m][:, ok]
    if x.shape[1] / FS < min_s:
        return None
    sig = UnifiedSignal(signal=x, sampling_rate_hz=FS, channel_names=["F7", "F8"], duration_s=x.shape[1] / FS)
    cfg = load_configs()
    feats = compute_features(sig)
    dt_frac = feats.epoch_relpow["delta"] + feats.epoch_relpow["theta"]
    slow_epoch_frac = float((np.median(dt_frac, axis=1) >= cfg.thresholds.diffuse_slowing_delta_theta_ratio).mean())
    slowing = bool(DiffuseSlowingDetector().detect(sig, feats, cfg.thresholds))
    bs = bool(BurstSuppressionDetector().detect(sig, feats, cfg.thresholds))
    return {"seconds": round(x.shape[1] / FS, 1), "diffuse_slowing": slowing, "burst_suppression": bs,
            "slow_epoch_fraction": round(slow_epoch_frac, 3)}


def _one(rid: str) -> dict:
    try:
        z = _zip()
        x, t, ev = read_record(z, rid)
        awake, deep = segments(ev)
        t0_abs = _t(_first_time(z, rid))
        art = artifact_epochs(rid, t0_abs, t[-1])
        return {"record": rid, "awake": analyse(x, t, awake, art, MIN_AWAKE_S) if awake else None,
                "deep": analyse(x, t, deep, art) if deep else None,
                "amp_uv_std": [round(float(v), 1) for v in x.std(axis=1)]}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"record": rid, "error": repr(exc)}


def _first_time(z, rid):
    with z.open(f"data/{rid}.csv") as fh:
        r = csv.DictReader(io.TextIOWrapper(fh, encoding="utf-8"))
        for row in r:
            if row.get("Intellivue/EEG_1") and row.get("Intellivue/EEG_2"):
                return row["Time"]


def assess(path: Path) -> dict:
    from scipy.stats import binomtest

    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    ok = [r for r in rows if "error" not in r]
    pair = [r for r in ok if r["awake"] and r["deep"]]
    n = len(pair)
    a = np.array([r["awake"]["diffuse_slowing"] for r in pair], int)
    d = np.array([r["deep"]["diffuse_slowing"] for r in pair], int)
    diff = float(d.mean() - a.mean())
    rng = np.random.default_rng(SEED)
    bs = [float(d[i].mean() - a[i].mean()) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    b01, b10 = int(((a == 0) & (d == 1)).sum()), int(((a == 1) & (d == 0)).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    ci = [round(float(q), 3) for q in np.percentile(bs, [2.5, 97.5])]
    bsa = np.array([r["awake"]["burst_suppression"] for r in pair], int)
    bsd = np.array([r["deep"]["burst_suppression"] for r in pair], int)
    return {"n_records": len(ok), "errors": [(r["record"], r["error"]) for r in rows if "error" in r], "n_paired": n,
            "diffuse_slowing_awake": round(float(a.mean()), 3), "diffuse_slowing_deep": round(float(d.mean()), 3),
            "difference": round(diff, 3), "difference_ci95": ci, "mcnemar_p": round(float(p), 6),
            "slow_epoch_fraction_median": {"awake": round(float(np.median([r["awake"]["slow_epoch_fraction"] for r in pair])), 3),
                                           "deep": round(float(np.median([r["deep"]["slow_epoch_fraction"] for r in pair])), 3)},
            "burst_suppression_awake": round(float(bsa.mean()), 3), "burst_suppression_deep": round(float(bsd.mean()), 3),
            "H1": "confirmed" if (diff >= 0.30 and ci[0] > 0) else "not_confirmed"}


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "assess":
        res = assess(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 0
    z = _zip()
    if cmd == "check":
        x, t, ev = read_record(z, CHECK)
        print("samples", x.shape, "duration_s", round(float(t[-1]), 1), "median dt", round(float(np.median(np.diff(t))) * 1000, 2), "ms")
        print("std uV", x.std(axis=1).round(1).tolist(), "events", [k for _, k in ev][:6], "...")
        awake, deep = segments(ev)
        print("awake", awake, "deep", deep)
        r = _one(CHECK)
        print({k: v for k, v in r.items() if k != "record"})
        return 0
    recs = sorted(i.filename[5:-4] for i in z.infolist() if i.filename.endswith(".csv") and i.filename[5:-4] != CHECK)
    out, workers = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3
    done = {json.loads(ln)["record"] for ln in out.read_text().splitlines() if ln.strip()} if out.exists() else set()
    todo = [r for r in recs if r not in done]
    print(f"{len(recs)} records, {len(todo)} to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r.get("record"), r.get("error"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
