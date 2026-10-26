"""Increment 45 (pre-registered, docs/preregistration_increment45.md): sleep staging in children,
OpenNeuro ds003555 (Zurich, 30 children, interval-level stage labels). Resumable: one JSON line per
child.

    PYTHONPATH=src python scripts/run_ds003555_sleep.py run OUT.jsonl [--workers 2]
    PYTHONPATH=src python scripts/run_ds003555_sleep.py assess OUT.jsonl OUT.json
"""

import argparse
import csv
import io
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BASE = "https://s3.amazonaws.com/openneuro.org/ds003555"
CLASSES = ("W", "N2", "N3", "REM")
LABEL = {"Awake": "W", "N2": "N2", "N3": "N3", "REM": "REM"}
SEED, N_BOOT = 45, 2000


def _curl(url: str, out: Path | None = None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def participants() -> list[dict]:
    return list(csv.DictReader(io.StringIO(_curl(f"{BASE}/participants.tsv").decode()), delimiter="\t"))


def intervals(sub: str) -> list[dict]:
    t = _curl(f"{BASE}/derivatives/{sub}/ses-01/eeg/DataIntervals.tsv").decode()
    return list(csv.DictReader(io.StringIO(t), delimiter="\t"))


def _one(item) -> dict:
    sub, age = item
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    try:
        ivs = intervals(sub)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"{sub}.edf"
            _curl(f"{BASE}/{sub}/ses-01/eeg/{sub}_ses-01_task-hfo_eeg.edf", p)
            sig = ingest(p)                                  # no age: the pre-registered test skips the gate
        fs = sig.sampling_rate_hz
        st = Pipeline(provider_pref="template", run_ica=False)._sleep_staging(sig)
        if not st or st.get("skipped"):
            return {"subject": sub, "age": age, "error": f"not staged: {st}"}
        hyp, ep = st["hypnogram"], st["epoch_s"]
        rows = []
        for iv in ivs:
            a, b = int(iv["StartInd"]) / fs, int(iv["EndInd"]) / fs
            k0, k1 = int(-(-a // ep)), int(b // ep)          # epochs entirely inside the interval
            pred = [hyp[k] for k in range(k0, min(k1, len(hyp)))]
            rows.append({"start_s": round(a, 1), "end_s": round(b, 1), "label": iv["SleepStage"].strip(),
                         "pred": max(set(pred), key=pred.count) if pred else None, "n_epochs": len(pred)})
        return {"subject": sub, "age": age, "fs": fs, "duration_h": round(sig.duration_s / 3600, 2),
                "derivations": st["derivations"], "n_hyp_epochs": len(hyp), "intervals": rows}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"subject": sub, "age": age, "error": repr(exc)}


def _load(p) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]


def kappa(pairs) -> float:
    import numpy as np

    if not pairs:
        return float("nan")
    idx = {c: i for i, c in enumerate(CLASSES + ("N1",))}
    m = np.zeros((5, 5))
    for y, p in pairs:
        m[idx[y], idx[p]] += 1
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def assess(path: Path) -> dict:
    import numpy as np

    rows = _load(path)
    ok = [r for r in rows if "error" not in r]

    def pairs(rs):
        return [(LABEL[i["label"]], i["pred"]) for r in rs for i in r["intervals"]
                if i["label"] in LABEL and i["pred"] is not None]

    def summary(rs):
        pr = pairs(rs)
        rng = np.random.default_rng(SEED)
        boot = []
        for _ in range(N_BOOT):
            sample = [rs[i] for i in rng.integers(0, len(rs), len(rs))]
            k = kappa(pairs(sample))
            if k == k:
                boot.append(k)
        per = {c: {"n": sum(y == c for y, _ in pr),
                   "agree": round(sum(y == c and p == c for y, p in pr) / max(1, sum(y == c for y, _ in pr)), 3)}
               for c in CLASSES}
        k = kappa(pr)
        return {"n_children": len(rs), "n_intervals": len(pr), "kappa": round(k, 3),
                "ci95": [round(float(q), 3) for q in np.percentile(boot, [2.5, 97.5])] if boot else None,
                "accuracy": round(sum(y == p for y, p in pr) / len(pr), 3) if pr else None,
                "per_class": per, "pred_n1": sum(p == "N1" for _, p in pr)}

    young = [r for r in ok if float(r["age"]) < 6]
    older = [r for r in ok if float(r["age"]) >= 6]
    res = {"errors": [(r["subject"], r["error"]) for r in rows if "error" in r],
           "all": summary(ok), "age_lt6": summary(young), "age_ge6": summary(older),
           "derivations": sorted({r["derivations"] for r in ok})}
    passes = lambda s: s["kappa"] >= 0.60 and s["ci95"] is not None and s["ci95"][0] >= 0.40  # noqa: E731
    res["H1"] = res["all"]["kappa"] >= 0.60
    res["decision"] = ("min_age_years=0.7" if passes(res["age_lt6"]) and passes(res["age_ge6"]) else
                       "min_age_years=6" if passes(res["age_ge6"]) else "keep_18")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assess"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.paths[0])
    if a.cmd == "assess":
        res = assess(out)
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps(res, indent=1, default=float))
        return 0
    items = [(p["participant_id"], float(p["age"])) for p in participants()]
    done = {r["subject"] for r in _load(out)} if out.exists() else set()
    todo = [it for it in items if it[0] not in done]
    print(f"{len(items)} children, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("subject", "age", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
