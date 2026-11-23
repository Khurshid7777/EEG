"""Increment 72 stage 2 (docs/preregistration_increment72.md): frozen sleep_stager_aasm_v5 vs v4 on COVISLEEP
(decisive) and the 149 test records of increment 68 (control). Both models see the same loaded signal.

    PYTHONPATH=src python scripts/run_v5_external.py covisleep OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_v5_external.py cinc OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_v5_external.py assess OUT_COVI.jsonl OUT_CINC.jsonl OUT.json
"""

import hashlib
import io
import json
import shutil
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cinc2018_sleep as cinc          # noqa: E402
import run_cinc_n1 as inc68                # noqa: E402
import run_covisleep_sleep as cov          # noqa: E402
import run_hmc_sleep as hmc                # noqa: E402

V4, V4_SHA = cov.MODEL, cov.MODEL_SHA256
V5 = "configs/models/sleep_stager_aasm_v5.json"
SMOOTHING = cov.SMOOTHING
SEED, N_BOOT = 72, 2000
STAGES = ["W", "N1", "N2", "N3", "REM"]


def _stage_both(pipe_cls, sig):
    out = {}
    for tag, model in (("v4", V4), ("v5", V5)):
        pipe = pipe_cls(provider_pref="template", run_ica=False)
        pipe.cfg.sleep = pipe.cfg.sleep.model_copy(update={"aasm_model": model, "aasm_smoothing": SMOOTHING})
        st = pipe._sleep_staging(sig)
        if not st or st.get("skipped") or "hypnogram_unsmoothed" not in st:
            raise RuntimeError(f"{tag}: not staged: {str(st)[:150]}")
        out[tag] = [STAGES.index(s) for s in st["hypnogram"]]
    return out


def _covi(args) -> dict:
    rec, files, ann_bytes = args
    from neurolens.layer1_ingest.edf import EdfIngestor
    from neurolens.pipeline.pipeline import Pipeline

    try:
        ann = zipfile.ZipFile(io.BytesIO(ann_bytes))
        member = next(n for n in ann.namelist() if n.endswith(f"/{rec}_Hypnogramme.csv") and "checkpoint" not in n)
        lab, first = cov.parse_hypnogram(ann.read(member).decode("utf-8", "replace"))
        d = tempfile.mkdtemp(prefix="covisleep_")
        try:
            z = cov._zip(*files["raw_edfs.zip"])
            p = Path(d) / f"{rec}.edf"
            with z.open(f"edfs/{rec}_PSG.edf") as src, p.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            off = cov.offset_seconds(first, cov.edf_start(p))
            sig = EdfIngestor().read(p, include=lambda n: n in cov.KEEP)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        dur_ep = sig.duration_s / 30.0
        shift = 0 if abs(off) <= cov.MAX_OFFSET_S else int(round(off / 30.0))
        lab = [(k + shift, c) for k, c in lab]
        inside = float(np.mean([0 <= k < int(dur_ep) for k, _ in lab])) if lab else 0.0
        if inside < cov.MIN_INSIDE:
            return {"record": rec, "error": f"only {inside:.0%} of labelled epochs fall inside the recording"}
        return {"record": rec, "group": rec[0], **{f"hyp_{k}": v for k, v in _stage_both(Pipeline, sig).items()}, "human": lab}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"record": rec, "error": repr(exc)}


def _cinc(args) -> dict:
    rec, split = args
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer1_ingest.wfdb import read_annotations
    from neurolens.pipeline.pipeline import Pipeline

    try:
        with tempfile.TemporaryDirectory() as d:
            for ext in ("hea", "mat", "arousal"):
                cinc._curl(f"{cinc.BASE}/training/{rec}/{rec}.{ext}", Path(d) / f"{rec}.{ext}")
            sig = ingest(Path(d) / f"{rec}.hea")
            lab = cinc.stages_from_notes(read_annotations(Path(d) / f"{rec}.arousal"), sig.signal.shape[1])
        return {"record": rec, "split": split, **{f"hyp_{k}": v for k, v in _stage_both(Pipeline, sig).items()}, "human": lab}
    except Exception as exc:  # noqa: BLE001
        return {"record": rec, "split": split, "error": repr(exc)}


def _sc(rows, key):
    return hmc._score([{**r, "hyp": r[key]} for r in rows])


def _paired(rows) -> dict:
    s4, s5 = _sc(rows, "hyp_v4"), _sc(rows, "hyp_v5")
    rng = np.random.default_rng(SEED)
    g = []
    for _ in range(N_BOOT):
        b = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        g.append(_sc(b, "hyp_v5")["kappa"] - _sc(b, "hyp_v4")["kappa"])
    ci = [round(float(q), 4) for q in np.percentile(g, [2.5, 97.5])]
    f1d = {k: round(s5["f1"][k] - s4["f1"][k], 3) for k in s5["f1"]}
    guard = f1d["N1"] >= 0 and all(f1d[k] >= -0.03 for k in f1d if k != "N1")
    return {"n": len(rows), "kappa_v4": round(s4["kappa"], 4), "kappa_v5": round(s5["kappa"], 4),
            "gain": round(s5["kappa"] - s4["kappa"], 4), "gain_ci95": ci, "f1_v4": {k: round(v, 3) for k, v in s4["f1"].items()},
            "f1_v5": {k: round(v, 3) for k, v in s5["f1"].items()}, "f1_difference": f1d, "guard_ok": bool(guard)}


def assess(covi_path, cinc_path) -> dict:
    cv = [r for r in hmc._load(covi_path) if "error" not in r]
    ci = [r for r in hmc._load(cinc_path) if "error" not in r and r["split"] == "test"]
    a, b = _paired(cv), _paired(ci)
    return {"covisleep": a, "cinc_test": b,
            "errors_covisleep": [(r["record"], r["error"]) for r in hmc._load(covi_path) if "error" in r],
            "errors_cinc": [(r["record"], r["error"]) for r in hmc._load(cinc_path) if "error" in r],
            "H_covisleep_decisive": bool(a["gain"] >= 0.005 and a["gain_ci95"][0] > 0 and a["guard_ok"]),
            "H_cinc_no_regression": bool(b["gain_ci95"][0] > -0.005)}


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "assess":
        res = assess(sys.argv[2], sys.argv[3])
        res["adopt_v5"] = res["H_covisleep_decisive"] and res["H_cinc_no_regression"]
        Path(sys.argv[4]).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 0
    for p, h in ((V4, V4_SHA), (SMOOTHING, cov.SMOOTHING_SHA256)):
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise RuntimeError(f"{p} does not match the frozen sha256")
    out, workers = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3
    done = {r["record"] for r in hmc._load(out)} if out.exists() else set()
    if cmd == "covisleep":
        files = cov._files()
        ann = cov._http(files["annotations.zip"][0])
        jobs = [(r, files, ann) for r in cov.records() if r not in done]
        fn = _covi
    else:
        jobs = [(r, s) for r, s in inc68.fresh_records() if s == "test" and r not in done]
        fn = _cinc
    print(f"{len(jobs)} to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(fn, jobs):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
            print(r["record"], r.get("error"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
