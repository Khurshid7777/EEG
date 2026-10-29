"""Increment 49 (pre-registered, docs/preregistration_increment49.md): sleep staging on the standard
AASM PSG montage, OpenNeuro ds005555 (BOAS: 128 nights, 100 adults, Micromed PSG F3/F4/C3/C4/O1/O2
against M1, consensus of three scorers). Product path (Pipeline._sleep_staging) with the frozen
AASM stager enabled for the test. Resumable: one JSON line per night.

    PYTHONPATH=src python scripts/run_boas_sleep.py run OUT.jsonl [--workers 2]
    PYTHONPATH=src python scripts/run_boas_sleep.py assess OUT.jsonl OUT.json
    # increment 50 (second use of BOAS): --model v2
    PYTHONPATH=src python scripts/run_boas_sleep.py run OUT.jsonl --model v2
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://s3.amazonaws.com/openneuro.org/ds005555"
MODEL = "configs/models/sleep_stager_aasm_v1.json"
MODEL_SHA256 = "fab0f2b3dea88586b75141e3f0c9d2e1725d460cf5bf53e7a932356ea509022e"
HUMAN = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}      # BOAS code -> STAGES index (W N1 N2 N3 REM); 8 = disconnection
SEED, N_BOOT = 49, 2000
MODELS = {"v1": (MODEL, MODEL_SHA256),
          "v2": ("configs/models/sleep_stager_aasm_v2.json",
                 "97d986b1c6a3e2f57e127591a05b8564aaf02872943ee7f280b5d4654c771d5a")}


def _curl(url: str, out: Path | None = None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def participants() -> list[dict]:
    return list(csv.DictReader(io.StringIO(_curl(f"{BASE}/participants.tsv").decode()), delimiter="\t"))


def _events(sub: str) -> list[dict]:
    t = _curl(f"{BASE}/{sub}/eeg/{sub}_task-Sleep_acq-psg_events.tsv").decode()
    return list(csv.DictReader(io.StringIO(t), delimiter="\t"))


def _one(item) -> dict:
    sub, age, pid, model = item
    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    try:
        ev = _events(sub)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"{sub}.edf"
            _curl(f"{BASE}/{sub}/eeg/{sub}_task-Sleep_acq-psg_eeg.edf", p)
            sig = ingest(p)
        age_y = None if age in (None, "", "n/a") else float(age)   # sub-16: age "n/a" in participants.tsv
        sig = sig.model_copy(update={"patient": PatientInfo(age_years=age_y)})
        pipe = Pipeline(provider_pref="template", run_ica=False)
        pipe.cfg.sleep = pipe.cfg.sleep.model_copy(update={"aasm_model": model})
        st = pipe._sleep_staging(sig)
        if not st or st.get("skipped"):
            return {"subject": sub, "pid": pid, "error": f"not staged: {st}"}
        hyp = [["W", "N1", "N2", "N3", "REM"].index(s) for s in st["hypnogram"]]
        human, ai = [], []
        for e in ev:
            k = int(round(float(e["onset"]) / 30.0))
            h = int(float(e["stage_hum"])) if e.get("stage_hum") not in (None, "", "n/a") else None
            a = int(float(e["stage_ai"])) if e.get("stage_ai") not in (None, "", "n/a") else None
            human.append([k, HUMAN.get(h, -1) if h is not None else -1])
            ai.append([k, HUMAN.get(a, -1) if a is not None else -1])
        return {"subject": sub, "pid": pid, "age": age_y, "duration_h": round(sig.duration_s / 3600, 2),
                "derivations": st["derivations"], "eog_source": st.get("eog_source"), "model": st.get("model"),
                "hyp": hyp, "human": human, "authors_ai": ai}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"subject": sub, "pid": pid, "error": repr(exc)}


def _load(p) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]


def _pairs(r, key="hyp"):
    """(human, predicted) stage indices for epochs with a human label (code 8 / missing excluded)."""
    if key == "hyp":
        pred = r["hyp"]
        return [(h, pred[k]) for k, h in r["human"] if h >= 0 and 0 <= k < len(pred)]
    ai = dict((k, v) for k, v in r["authors_ai"])
    return [(h, ai[k]) for k, h in r["human"] if h >= 0 and ai.get(k, -1) >= 0]


def _score(rows, key="hyp") -> dict:
    from neurolens.layer4_detect.sleep_staging import confusion, scores_from_confusion

    m = np.zeros((5, 5), int)
    for r in rows:
        pr = _pairs(r, key)
        if pr:
            y, p = zip(*pr)
            m += confusion(np.array(y), np.array(p))
    return scores_from_confusion(m)


def assess(path: Path) -> dict:
    from neurolens.layer4_detect.sleep_staging import confusion, scores_from_confusion

    rows = _load(path)
    ok = [r for r in rows if "error" not in r]
    pooled = _score(ok)
    by_pid: dict[str, list] = {}
    for r in ok:
        by_pid.setdefault(r["pid"], []).append(r)
    pids = sorted(by_pid)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        sample = [r for i in rng.integers(0, len(pids), len(pids)) for r in by_pid[pids[i]]]
        boot.append(_score(sample)["kappa"])
    per_night = []
    for r in ok:
        y, p = zip(*_pairs(r))
        per_night.append(scores_from_confusion(confusion(np.array(y), np.array(p)))["kappa"])
    ci = [round(float(q), 3) for q in np.percentile(boot, [2.5, 97.5])]
    h1 = pooled["kappa"] >= 0.60 and ci[0] >= 0.50
    return {"n_nights": len(ok), "n_people": len(pids), "errors": [(r["subject"], r["error"]) for r in rows if "error" in r],
            "derivations": sorted({r["derivations"] for r in ok}), "eog_source": sorted({str(r["eog_source"]) for r in ok}),
            "pooled": {k: pooled[k] for k in ("n_epochs", "kappa", "accuracy", "macro_f1", "f1", "confusion")},
            "kappa_ci95_by_person": ci, "per_night_kappa_median": round(float(np.nanmedian(per_night)), 3),
            "authors_ai_vs_human": {k: v for k, v in _score(ok, "ai").items() if k in ("kappa", "accuracy", "macro_f1")},
            "H1": bool(h1), "decision": "enable_aasm_model" if h1 else "keep_off"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assess"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--model", choices=sorted(MODELS), default="v1")
    a = ap.parse_args()
    out = Path(a.paths[0])
    if a.cmd == "assess":
        res = assess(out)
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "pooled"} | {"pooled_kappa": res["pooled"]["kappa"]},
                         indent=1, default=float))
        return 0
    model, sha = MODELS[a.model]
    if hashlib.sha256(Path(model).read_bytes()).hexdigest() != sha:
        raise RuntimeError(f"{model} does not match the frozen sha256")
    items = [(p["participant_id"], p["age"], p["pid"], model) for p in participants()]
    done = {r["subject"] for r in _load(out)} if out.exists() else set()
    todo = [it for it in items if it[0] not in done]
    print(f"{len(items)} nights, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("subject", "derivations", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
