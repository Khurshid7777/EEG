"""Increment 46, development only: the end of long seizures (model card §5: detections cover 0.46 of
the last third). Training patients of the working seizure model (31: CHB-MIT and Siena, increment 11)
- no held-out data.

    PYTHONPATH=src python scripts/dev_seizure_end.py build data/dev46 [--workers 3]   # features, EDFs deleted
    PYTHONPATH=src python scripts/dev_seizure_end.py loso data/dev46                  # out-of-fold probabilities
    PYTHONPATH=src python scripts/dev_seizure_end.py explore data/dev46               # hysteresis variants
"""

import argparse
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CHB = [f"chb{i:02d}" for i in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24)]
SIENA = ["PN00", "PN05", "PN07", "PN09", "PN11", "PN12", "PN13", "PN16", "PN17"]
N_FREE = 3
THR_ON, MIN_EPOCHS = 0.3, 15        # offline operating point A of ictal_gbm_offline_v2 (increment 11)


def files_for(db: str, sub: str):
    from neurolens.datasets.physionet import client_for

    anns = client_for(db, None).annotations(sub)
    return [a for a in anns if a.has_seizure] + [a for a in anns if not a.has_seizure][:N_FREE]


def _one(item):
    db, ann, out = item
    from neurolens.datasets.physionet import client_for
    from neurolens.evaluation.runner import default_config, prepare_records

    dst = Path(out) / f"{db}__{ann.file.replace('/', '__')}.pkl"
    if dst.exists():
        return f"cached {dst.name}"
    client = client_for(db, None)
    try:
        path = client.fetch(ann.file)
        rec = prepare_records([(ann, path)], default_config(), None)[0]
        dst.write_bytes(pickle.dumps(rec))
        path.unlink(missing_ok=True)
        return f"ok {dst.name}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {db}:{ann.file}: {exc!r}"


def build(out: Path, workers: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    items = [("chbmit", a, str(out)) for s in CHB for a in files_for("chbmit", s)]
    items += [("siena", a, str(out)) for s in SIENA for a in files_for("siena", s)]
    print(f"{len(items)} records", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for msg in ex.map(_one, items):
            print(msg, flush=True)


def _dataset(out: Path):
    from neurolens.evaluation.ml import Dataset

    # local derived feature caches written by build() above in this session (same scheme as the
    # runner's own pickle cache); not data received from outside
    recs = [pickle.loads(p.read_bytes()) for p in sorted(out.glob("*.pkl"))]
    return Dataset.build(recs, version=1)


def loso(out: Path) -> None:
    from neurolens.evaluation.ml import loso_probabilities

    ds = _dataset(out)
    probs = loso_probabilities(ds, kind="gbm", progress=lambda m: print(m, flush=True))
    np.savez_compressed(out / "loso_probs.npz", *probs)
    print(f"saved {len(probs)} records", flush=True)


def hysteresis_runs(p, thr_on, thr_off, min_epochs, bridge):
    """Runs started by >= ``min_epochs`` epochs at >= thr_on (as the working detector), then extended
    both ways while the probability stays >= thr_off; runs closer than ``bridge`` epochs are joined."""
    from neurolens.layer4_detect.ml_ictal import probability_runs

    base = probability_runs(p, thr_on, min_epochs)
    if thr_off >= thr_on and bridge == 0:
        return [(a, b) for a, b, _ in base]
    ps = np.asarray(p, float)
    runs = []
    for a, b, _ in base:
        while b < len(ps) and ps[b] >= thr_off:
            b += 1
        while a > 0 and ps[a - 1] >= thr_off:
            a -= 1
        runs.append([a, b])
    runs.sort()
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= bridge:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(r) for r in merged]


def evaluate(ds, probs, thr_off, bridge) -> dict:
    from neurolens.evaluation.matching import Detection, score_record
    from neurolens.evaluation.runner import ScoringRules

    rules = ScoringRules()
    tp = fn = fp = 0
    hours = 0.0
    thirds, whole, fa_dur = [[], [], []], [], 0.0
    for rec, t, p in zip(ds.records, ds.times, probs):
        runs = hysteresis_runs(p, THR_ON, thr_off, MIN_EPOCHS, bridge)
        det = [Detection(float(t[a]), float(t[b - 1]), float(np.mean(p[a:b]))) for a, b in runs]
        sc = score_record(rec.annotation.file, rec.annotation.seizures, det, rec.duration_s,
                          rules.pre_s, rules.post_s, rules.merge_gap_s)
        tp += sc.tp
        fn += sc.fn
        fp += sc.fp
        hours += rec.duration_s / 3600
        iv = [(d.t_start, d.t_end) for d in det]
        for s in rec.annotation.seizures:
            dur = s.offset_s - s.onset_s
            cov = lambda a, b: sum(max(0.0, min(b, y) - max(a, x)) for x, y in iv) / max(b - a, 1e-9)  # noqa: E731
            whole.append(cov(s.onset_s, s.offset_s))
            if dur > 120:
                for k in range(3):
                    thirds[k].append(cov(s.onset_s + k * dur / 3, s.onset_s + (k + 1) * dur / 3))
        seiz = [(s.onset_s - rules.pre_s, s.offset_s + rules.post_s) for s in rec.annotation.seizures]
        for x, y in iv:   # detected time outside any (padded) seizure
            inside = sum(max(0.0, min(y, b) - max(x, a)) for a, b in seiz)
            fa_dur += max(0.0, (y - x) - inside)
    return {"thr_off": thr_off, "bridge_epochs": bridge, "sensitivity": round(tp / max(1, tp + fn), 3),
            "fa_per_hour": round(fp / hours, 2), "coverage_all": round(float(np.mean(whole)), 3),
            "coverage_thirds_long": [round(float(np.mean(c)), 3) for c in thirds], "n_long": len(thirds[0]),
            "false_detected_min_per_h": round(fa_dur / 60 / hours, 2)}


def explore(out: Path) -> None:
    import json

    ds = _dataset(out)
    z = np.load(out / "loso_probs.npz")
    probs = [z[f"arr_{i}"] for i in range(len(z.files))]
    print(f"{len(ds.records)} records, {sum(len(r.annotation.seizures) for r in ds.records)} seizures, "
          f"{sum(r.duration_s for r in ds.records) / 3600:.1f} h", flush=True)
    rows = [evaluate(ds, probs, thr_off, bridge) for thr_off in (THR_ON, 0.2, 0.15, 0.1, 0.05)
            for bridge in (0, 5, 15, 30)]
    for r in rows:
        print(json.dumps(r), flush=True)
    (out / "explore.json").write_text(json.dumps(rows, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "loso", "explore"])
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    {"build": lambda: build(out, a.workers), "loso": lambda: loso(out), "explore": lambda: explore(out)}[a.cmd]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
