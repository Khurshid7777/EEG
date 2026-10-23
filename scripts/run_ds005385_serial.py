"""Increment 43 (pre-registered, docs/preregistration_increment43.md): serial comparison v2
(alert-epoch band shares) against v1 on OpenNeuro ds005385 (Dortmund Vital Study). Resumable:
one JSON line per recording.

    PYTHONPATH=src python scripts/run_ds005385_serial.py run DEV.jsonl --part dev
    PYTHONPATH=src python scripts/run_ds005385_serial.py derive DEV.jsonl OUT_LIMITS.json
    PYTHONPATH=src python scripts/run_ds005385_serial.py run TEST.jsonl --part test     # after the limits commit
    PYTHONPATH=src python scripts/run_ds005385_serial.py assess TEST.jsonl LIMITS.json OUT.json
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

BASE = "https://s3.amazonaws.com/openneuro.org/ds005385"
STANDARD_1020 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
                 "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
SEED, N_PART = 43, 100
KEEP = ("pdr_hz", "global_rel_delta", "global_rel_theta", "global_rel_alpha", "alert_rel_delta",
        "alert_rel_theta", "alert_rel_alpha", "robust_amplitude_uv", "detectors_fired", "duration_s")
STABLE = (("1", "EyesClosed", "pre"), ("1", "EyesClosed", "post"))
EYES = (("1", "EyesClosed", "pre"), ("1", "EyesOpen", "pre"))
YEARS5 = (("1", "EyesClosed", "pre"), ("2", "EyesClosed", "pre"))


def _curl(url: str, out: Path | None = None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def participants() -> list[dict]:
    return list(csv.DictReader(io.StringIO(_curl(f"{BASE}/participants.tsv").decode()), delimiter="\t"))


def split() -> dict[str, list[dict]]:
    import numpy as np

    ps = sorted((p for p in participants() if p["session1"] == "yes"), key=lambda p: p["participant_id"])
    order = np.random.default_rng(SEED).permutation(len(ps))
    return {"dev": [ps[i] for i in order[:N_PART]], "test": [ps[i] for i in order[N_PART:2 * N_PART]]}


def items(part: str) -> list[tuple]:
    out = []
    for p in split()[part]:
        age = float(p["age"]) if p["age"] not in ("", "n/a") else None
        recs = list(STABLE) + ([EYES[1]] if part == "test" else []) + \
            ([YEARS5[1]] if part == "test" and p["session2"] == "yes" else [])
        out += [(p["participant_id"], ses, task, acq, age) for ses, task, acq in recs]
    return out


def _one(item) -> dict:
    sub, ses, task, acq, age = item
    import numpy as np

    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    base = {"subject": sub, "session": ses, "task": task, "acq": acq, "age_years": age}
    name = f"{sub}_ses-{ses}_task-{task}_acq-{acq}_eeg.edf"
    try:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / name
            _curl(f"{BASE}/{sub}/ses-{ses}/eeg/{name}", p)
            sig = ingest(p, patient=PatientInfo(age_years=age))
        idx = [sig.channel_names.index(c) for c in STANDARD_1020 if c in sig.channel_names]
        sig = sig.model_copy(update={"signal": np.ascontiguousarray(sig.signal[idx]),
                                     "channel_names": [sig.channel_names[i] for i in idx]})
        out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(sig)
        row = out.result_json["record_summary"]
        return {**base, "n_channels": len(idx), **{k: row.get(k) for k in KEEP}}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}


def _load(p) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]


def _key(r):
    return (r["subject"], r["session"], r["task"], r["acq"])


def pairs(rows, a, b):
    by = {_key(r): r for r in rows if "error" not in r}
    return [(by[(s, *a)], by[(s, *b)]) for s in sorted({r["subject"] for r in rows})
            if (s, *a) in by and (s, *b) in by]


def run(out: Path, part: str, workers: int) -> None:
    todo_all = items(part)
    done = {_key(r) for r in _load(out)} if out.exists() else set()
    todo = [it for it in todo_all if it[:4] not in done]
    print(f"{len(todo_all)} recordings, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("subject", "session", "task", "acq", "error")}, flush=True)


def derive(dev: Path) -> dict:
    from neurolens.pipeline.serial import QUANTITIES, QUANTITIES_V2, Z, derive_limits

    rows = _load(dev)
    ps = pairs(rows, *STABLE)
    src = ("OpenNeuro ds005385 (Dortmund Vital Study) development sample (seed 43, 100 people), "
           "session 1 eyes closed before vs after 2 h of tasks, 19 10-20 electrodes (increment 43)")
    return {"n_subjects": len(ps), "errors": [_key(r) for r in rows if "error" in r],
            "v1": {"source": src, "version": 1, "z": Z, "limits": derive_limits(ps, quantities=QUANTITIES)},
            "v2": {"source": src, "version": 2, "z": Z, "alert_quantile": 0.75,
                   "limits": derive_limits(ps, quantities=QUANTITIES_V2)}}


def assess(test: Path, limits_path: Path) -> dict:
    from scipy.stats import binomtest

    from neurolens.evaluation.prereg import clopper_pearson
    from neurolens.pipeline.serial import compare, load_limits

    rows, L = _load(test), json.loads(limits_path.read_text())
    v1_ds4148 = load_limits(Path(__file__).resolve().parents[1] / "configs" / "serial_limits_v1_ds004148.yaml")

    def run_pairs(pp, lim):
        return [compare(a, b, lim) for a, b in pp]

    def qrow(c, name):
        return next(r for r in c["quantities"] if r["quantity"] == name)

    def frac(k, n):
        return {"k": k, "n": n, "fraction": round(k / n, 3) if n else None,
                "ci95": [round(x, 3) for x in clopper_pearson(k, n)] if n else None}

    def per_q(cs):
        return {r["quantity"]: round(sum(qrow(c, r["quantity"])["exceeds"] for c in cs) / len(cs), 3)
                for r in (cs[0]["quantities"] if cs else [])}

    st, ey, y5 = pairs(rows, *STABLE), pairs(rows, *EYES), pairs(rows, *YEARS5)
    s2, e2, e1 = run_pairs(st, L["v2"]), run_pairs(ey, L["v2"]), run_pairs(ey, L["v1"])
    h1 = frac(sum(c["any_exceeds"] for c in s2), len(s2))
    det2 = [qrow(c, "alert_rel_alpha")["exceeds"] and qrow(c, "alert_rel_alpha")["change"] < 0 for c in e2]
    det1 = [qrow(c, "global_rel_alpha")["exceeds"] and qrow(c, "global_rel_alpha")["change"] < 0 for c in e1]
    h2 = frac(sum(det2), len(det2))
    b = sum(x and not y for x, y in zip(det2, det1))      # v2 only
    c = sum(y and not x for x, y in zip(det2, det1))      # v1 only
    p3 = binomtest(b, b + c, 0.5, alternative="greater").pvalue if b + c else 1.0
    s1 = run_pairs(st, L["v1"])
    transfer = run_pairs(st, v1_ds4148) if v1_ds4148 else []
    y2 = run_pairs(y5, L["v2"])
    import numpy as np

    def mean_change(cs, q):
        v = [qrow(c, q)["change"] for c in cs if qrow(c, q)["change"] is not None]
        return round(float(np.mean(v)), 3) if v else None

    res = {"errors": [list(_key(r)) for r in rows if "error" in r],
           "H1_v2_stable_any_exceeds": {**h1, "per_quantity": per_q(s2), "pass": h1["fraction"] <= 0.15},
           "H2_v2_eyes_open_detected": {**h2, "pass": h2["fraction"] >= 0.40},
           "H3_v2_vs_v1": {"v2_detected": sum(det2), "v1_detected": sum(det1), "v2_only": b, "v1_only": c,
                           "mcnemar_one_sided_p": round(float(p3), 5), "pass": bool(p3 < 0.05)},
           "v1_stable_any_exceeds": {**frac(sum(c["any_exceeds"] for c in s1), len(s1)), "per_quantity": per_q(s1)},
           "secondary_transfer_ds004148_v1_limits": (
               {**frac(sum(c["any_exceeds"] for c in transfer), len(transfer)), "per_quantity": per_q(transfer)}
               if transfer else None),
           "secondary_5_years_v2": {**frac(sum(c["any_exceeds"] for c in y2), len(y2)), "per_quantity": per_q(y2),
                                    "mean_change": {q: mean_change(y2, q) for q in
                                                    ("pdr_hz", "alert_rel_delta", "alert_rel_theta",
                                                     "alert_rel_alpha", "robust_amplitude_uv")}}}
    ok1, ok2, ok3 = res["H1_v2_stable_any_exceeds"]["pass"], res["H2_v2_eyes_open_detected"]["pass"], \
        res["H3_v2_vs_v1"]["pass"]
    res["decision"] = ("v2_replaces_v1" if ok1 and ok3 else "keep_v1")
    res["keep_large_changes_note"] = not (ok1 and ok3 and h2["fraction"] >= 0.70)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "derive", "assess"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--part", choices=["dev", "test"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "run":
        run(Path(a.paths[0]), a.part, a.workers)
        return 0
    if a.cmd == "derive":
        res = derive(Path(a.paths[0]))
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
    else:
        res = assess(Path(a.paths[0]), Path(a.paths[1]))
        Path(a.paths[2]).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
