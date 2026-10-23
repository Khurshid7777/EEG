"""Increment 42 (pre-registered, docs/preregistration_increment42.md): day-to-day variability
limits for serial comparison, OpenNeuro ds004148 (60 healthy, 3 sessions). Resumable: one JSON
line per recording.

    # development (sub-01..30): summaries, then limits -> configs/serial_limits.yaml
    PYTHONPATH=src python scripts/run_ds004148_serial.py run DEV.jsonl --part dev
    PYTHONPATH=src python scripts/run_ds004148_serial.py derive DEV.jsonl
    # test (sub-31..60), only after the limits are committed
    PYTHONPATH=src python scripts/run_ds004148_serial.py run TEST.jsonl --part test
    PYTHONPATH=src python scripts/run_ds004148_serial.py assess TEST.jsonl OUT.json
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

BASE = "https://s3.amazonaws.com/openneuro.org/ds004148"
STANDARD_1020 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
                 "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
PARTS = {"dev": range(1, 31), "test": range(31, 61)}
# (session, task) per part: development needs only the eyes-closed pairs it derives limits from
ITEMS = {"dev": [(1, "eyesclosed"), (3, "eyesclosed")],
         "test": [(1, "eyesclosed"), (3, "eyesclosed"), (2, "eyesclosed"), (1, "eyesopen")]}
KEEP = ("pdr_hz", "global_rel_delta", "global_rel_theta", "global_rel_alpha", "robust_amplitude_uv",
        "detectors_fired", "duration_s")


def _curl(url: str, out: Path | None = None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def ages() -> dict[str, float]:
    rows = csv.DictReader(io.StringIO(_curl(f"{BASE}/participants.tsv").decode()), delimiter="\t")
    return {r["participant_id"]: float(r["age"]) for r in rows if r["age"] not in ("", "n/a")}


def _one(item) -> dict:
    sub, ses, task, age = item
    import numpy as np

    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    base = {"subject": sub, "session": ses, "task": task, "age_years": age}
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
        row = out.result_json["record_summary"]
        return {**base, "n_channels": len(idx), **{k: row.get(k) for k in KEEP}}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}


def _load(p) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]


def _pairs(rows, a, b):
    by = {(r["subject"], r["session"], r["task"]): r for r in rows if "error" not in r}
    return [(by[(s, *a)], by[(s, *b)]) for s in sorted({r["subject"] for r in rows})
            if (s, *a) in by and (s, *b) in by]


def run(out: Path, part: str, workers: int) -> None:
    ag = ages()
    items = [(f"sub-{i:02d}", ses, task, ag.get(f"sub-{i:02d}")) for i in PARTS[part] for ses, task in ITEMS[part]]
    done = {(r["subject"], r["session"], r["task"]) for r in _load(out)} if out.exists() else set()
    todo = [it for it in items if it[:3] not in done]
    print(f"{len(items)} recordings, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("subject", "session", "task", "error")}, flush=True)


def derive(dev: Path) -> dict:
    import yaml

    from neurolens.pipeline.serial import LIMITS_PATH, Z, derive_limits

    rows = _load(dev)
    pairs = _pairs(rows, (1, "eyesclosed"), (3, "eyesclosed"))
    res = {"source": "OpenNeuro ds004148 sub-01..30, session 1 vs 3, eyes closed, 19 10-20 electrodes "
                     "(increment 42, docs/preregistration_increment42.md)",
           "z": Z, "n_subjects": len(pairs), "errors": [f"{r['subject']} s{r['session']} {r['task']}"
                                                      for r in rows if "error" in r],
           "limits": derive_limits(pairs)}
    LIMITS_PATH.write_text(yaml.safe_dump(res, allow_unicode=True, sort_keys=False))
    return res


def assess(test: Path) -> dict:
    from neurolens.evaluation.prereg import clopper_pearson
    from neurolens.pipeline.serial import compare, load_limits

    rows, lim = _load(test), load_limits()
    stable = [compare(a, b, lim) for a, b in _pairs(rows, (1, "eyesclosed"), (3, "eyesclosed"))]
    same_day = [compare(a, b, lim) for a, b in _pairs(rows, (1, "eyesclosed"), (2, "eyesclosed"))]
    eo = [compare(a, b, lim) for a, b in _pairs(rows, (1, "eyesclosed"), (1, "eyesopen"))]

    def q(c, name):
        return next(r for r in c["quantities"] if r["quantity"] == name)

    def per_quantity(cs):
        return {r["quantity"]: round(sum(q(c, r["quantity"])["exceeds"] for c in cs) / max(1, len(cs)), 3)
                for r in (cs[0]["quantities"] if cs else [])}

    k1 = sum(c["any_exceeds"] for c in stable)
    k2 = sum(q(c, "global_rel_alpha")["exceeds"] and q(c, "global_rel_alpha")["change"] < 0 for c in eo)
    h1 = k1 / len(stable) if stable else float("nan")
    h2 = k2 / len(eo) if eo else float("nan")
    return {"errors": [f"{r['subject']} s{r['session']} {r['task']}" for r in rows if "error" in r],
            "H1": {"n": len(stable), "any_exceeds": k1, "fraction": round(h1, 3),
                   "ci95": [round(x, 3) for x in clopper_pearson(k1, len(stable))],
                   "per_quantity": per_quantity(stable), "pass": h1 <= 0.15},
            "H2": {"n": len(eo), "alpha_decrease_exceeds": k2, "fraction": round(h2, 3),
                   "ci95": [round(x, 3) for x in clopper_pearson(k2, len(eo))], "pass": h2 >= 0.70},
            "secondary_session1_vs_2": {"n": len(same_day), "any_exceeds": sum(c["any_exceeds"] for c in same_day),
                                        "per_quantity": per_quantity(same_day)},
            "decision": ("freeze" if h1 <= 0.15 and h2 >= 0.70 else
                         "descriptive_only" if h1 > 0.15 else "freeze_insensitive")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "derive", "assess"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--part", choices=list(PARTS))
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "run":
        run(Path(a.paths[0]), a.part, a.workers)
        return 0
    res = derive(Path(a.paths[0])) if a.cmd == "derive" else assess(Path(a.paths[0]))
    if a.cmd == "assess":
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
