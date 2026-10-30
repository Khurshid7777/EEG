"""Increment 51 (pre-registered, docs/preregistration_increment51.md): the multi-centre AASM-montage
stager on PhysioNet HMC Sleep Staging 1.1 (Haaglanden Medisch Centrum, The Hague: 151 clinical PSG,
AASM scoring; F4-M1, C4-M1, O2-M1, C3-M2, E1-M2, E2-M2). Product path (Pipeline._sleep_staging).
Resumable: one JSON line per recording.

    PYTHONPATH=src python scripts/run_hmc_sleep.py run OUT.jsonl
    PYTHONPATH=src python scripts/run_hmc_sleep.py assess OUT.jsonl OUT.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = "https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1"
MODEL = "configs/models/sleep_stager_aasm_v3.json"
MODEL_SHA256 = None          # set in the pre-registration commit
STAGE = {"Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2, "Sleep stage N3": 3, "Sleep stage R": 4}
SEED, N_BOOT = 51, 2000


def _curl(url, out=None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def records() -> list[str]:
    return [ln.strip().split("/")[-1].removesuffix(".edf") for ln in _curl(f"{BASE}/RECORDS").decode().splitlines()
            if ln.strip()]


def _scoring(rec) -> list[tuple[int, int]]:
    out = []
    for ln in _curl(f"{BASE}/recordings/{rec}_sleepscoring.txt").decode().splitlines()[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 5 and parts[4] in STAGE and float(parts[3]) == 30:
            out.append((int(round(float(parts[2]) / 30.0)), STAGE[parts[4]]))
    return out


def _one(rec: str) -> dict:
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    try:
        lab = _scoring(rec)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"{rec}.edf"
            _curl(f"{BASE}/recordings/{rec}.edf", p)
            sig = ingest(p)
        pipe = Pipeline(provider_pref="template", run_ica=False)
        pipe.cfg.sleep = pipe.cfg.sleep.model_copy(update={"aasm_model": MODEL})
        st = pipe._sleep_staging(sig)
        if not st or st.get("skipped"):
            return {"record": rec, "error": f"not staged: {st}"}
        hyp = [["W", "N1", "N2", "N3", "REM"].index(s) for s in st["hypnogram"]]
        return {"record": rec, "duration_h": round(sig.duration_s / 3600, 2), "derivations": st["derivations"],
                "eog_source": st.get("eog_source"), "hyp": hyp, "human": lab}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"record": rec, "error": repr(exc)}


def _load(p) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]


def _score(rows) -> dict:
    from neurolens.layer4_detect.sleep_staging import confusion, scores_from_confusion

    m = np.zeros((5, 5), int)
    for r in rows:
        pr = [(h, r["hyp"][k]) for k, h in r["human"] if 0 <= k < len(r["hyp"])]
        if pr:
            y, p = zip(*pr)
            m += confusion(np.array(y), np.array(p))
    return scores_from_confusion(m)


def assess(path: Path) -> dict:
    rows = _load(path)
    ok = [r for r in rows if "error" not in r]
    pooled = _score(ok)
    rng = np.random.default_rng(SEED)
    boot = [_score([ok[i] for i in rng.integers(0, len(ok), len(ok))])["kappa"] for _ in range(N_BOOT)]
    ci = [round(float(q), 3) for q in np.percentile(boot, [2.5, 97.5])]
    per = [_score([r])["kappa"] for r in ok]
    h1 = pooled["kappa"] >= 0.60 and ci[0] >= 0.50
    return {"n_recordings": len(ok), "errors": [(r["record"], r["error"]) for r in rows if "error" in r],
            "derivations": sorted({r["derivations"] for r in ok}), "eog_source": sorted({str(r["eog_source"]) for r in ok}),
            "pooled": {k: pooled[k] for k in ("n_epochs", "kappa", "accuracy", "macro_f1", "f1", "confusion")},
            "kappa_ci95_by_recording": ci, "per_recording_kappa_median": round(float(np.nanmedian(per)), 3),
            "H1": bool(h1), "decision": "enable_aasm_model_v3" if h1 else "keep_off"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assess"])
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    out = Path(a.paths[0])
    if a.cmd == "assess":
        res = assess(out)
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "pooled"} | {"kappa": res["pooled"]["kappa"]},
                         indent=1, default=float))
        return 0
    if MODEL_SHA256 is None or hashlib.sha256(Path(MODEL).read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError(f"{MODEL} does not match the frozen sha256")
    recs = records()
    done = {r["record"] for r in _load(out)} if out.exists() else set()
    todo = [r for r in recs if r not in done]
    print(f"{len(recs)} recordings, {len(todo)} to run", flush=True)
    with out.open("a") as f:
        for rec in todo:
            r = _one(rec)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("record", "derivations", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
