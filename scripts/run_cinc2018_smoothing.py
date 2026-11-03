"""Increment 55 (pre-registered, docs/preregistration_increment55.md): Viterbi smoothing of the AASM-montage
hypnogram on a fresh random sample of the PhysioNet/CinC Challenge 2018 training set (MGH, AASM scoring).
The 150 records of increment 52 and tr03-0005 are excluded; seed 53. One pipeline run per record gives both
hypnograms (``hypnogram`` smoothed, ``hypnogram_unsmoothed`` plain argmax of the same probabilities), so the
comparison is paired. Resumable: one JSON line per record.

    PYTHONPATH=src python scripts/run_cinc2018_smoothing.py run OUT.jsonl
    PYTHONPATH=src python scripts/run_cinc2018_smoothing.py assess OUT.jsonl OUT.json
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cinc2018_sleep as cinc  # noqa: E402
import run_hmc_sleep as hmc        # noqa: E402

MODEL, MODEL_SHA256 = cinc.MODEL, cinc.MODEL_SHA256
SMOOTHING = "configs/models/sleep_smoothing_aasm_v1.json"
SMOOTHING_SHA256 = "be614d1c8c95709734df5bb14d396c08fbd88eff98713f853f4f2fb7841a851f"
N_SAMPLE, SEED, N_BOOT = 150, 53, 2000
MIN_GAIN, GUARD = 0.005, 0.03      # H1: paired kappa gain; guard: no stage F1 may fall by more than this
STAGES = ["W", "N1", "N2", "N3", "REM"]


def records() -> list[str]:
    used = set(cinc.records()) | cinc.EXCLUDED
    tr = sorted(ln.strip().split("/")[1] for ln in cinc._curl(f"{cinc.BASE}/RECORDS").decode().splitlines()
                if ln.startswith("training/"))
    tr = [r for r in tr if r not in used]
    return sorted(np.random.default_rng(SEED).choice(tr, N_SAMPLE, replace=False).tolist())


def _one(rec: str) -> dict:
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer1_ingest.wfdb import read_annotations
    from neurolens.pipeline.pipeline import Pipeline

    try:
        with tempfile.TemporaryDirectory() as d:
            for ext in ("hea", "mat", "arousal"):
                cinc._curl(f"{cinc.BASE}/training/{rec}/{rec}.{ext}", Path(d) / f"{rec}.{ext}")
            sig = ingest(Path(d) / f"{rec}.hea")
            lab = cinc.stages_from_notes(read_annotations(Path(d) / f"{rec}.arousal"), sig.signal.shape[1])
        pipe = Pipeline(provider_pref="template", run_ica=False)
        pipe.cfg.sleep = pipe.cfg.sleep.model_copy(update={"aasm_model": MODEL, "aasm_smoothing": SMOOTHING})
        st = pipe._sleep_staging(sig)
        if not st or st.get("skipped") or "hypnogram_unsmoothed" not in st:
            return {"record": rec, "error": f"not staged / not smoothed: {str(st)[:200]}"}
        idx = STAGES.index
        return {"record": rec, "derivations": st["derivations"],
                "hyp": [idx(s) for s in st["hypnogram"]], "hyp_raw": [idx(s) for s in st["hypnogram_unsmoothed"]],
                "human": lab}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"record": rec, "error": repr(exc)}


def _score(rows, key) -> dict:
    return hmc._score([{**r, "hyp": r[key]} for r in rows])


def assess(path: Path) -> dict:
    rows = [r for r in hmc._load(path) if "error" not in r]
    sm, raw = _score(rows, "hyp"), _score(rows, "hyp_raw")
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(N_BOOT):
        b = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        gains.append(_score(b, "hyp")["kappa"] - _score(b, "hyp_raw")["kappa"])
    ci = [round(float(q), 4) for q in np.percentile(gains, [2.5, 97.5])]
    gain = sm["kappa"] - raw["kappa"]
    f1_drop = {s: round(raw["f1"][s] - sm["f1"][s], 4) for s in STAGES}
    h1 = gain >= MIN_GAIN and ci[0] > 0
    guard = max(f1_drop.values()) <= GUARD
    flips = lambda k: float(np.mean([np.mean(np.diff(r[k]) != 0) for r in rows]))
    return {"n_recordings": len(rows), "errors": [(r["record"], r["error"]) for r in hmc._load(path) if "error" in r],
            "kappa_unsmoothed": raw["kappa"], "kappa_smoothed": sm["kappa"], "gain": round(gain, 4),
            "gain_ci95_by_recording": ci, "f1_unsmoothed": raw["f1"], "f1_smoothed": sm["f1"], "f1_drop": f1_drop,
            "stage_changes_per_epoch": {"unsmoothed": round(flips("hyp_raw"), 4), "smoothed": round(flips("hyp"), 4)},
            "H1": bool(h1), "guard_ok": bool(guard),
            "decision": "enable_aasm_smoothing" if (h1 and guard) else "keep_off"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assess"])
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    out = Path(a.paths[0])
    if a.cmd == "assess":
        res = assess(out)
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps(res, indent=1, default=float))
        return 0
    for p, h in ((MODEL, MODEL_SHA256), (SMOOTHING, SMOOTHING_SHA256)):
        if h is None or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h:
            raise RuntimeError(f"{p} does not match the frozen sha256")
    recs = records()
    done = {r["record"] for r in hmc._load(out)} if out.exists() else set()
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
