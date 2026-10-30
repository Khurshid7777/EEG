"""Increment 52 (pre-registered, docs/preregistration_increment52.md): the AASM-montage stager on the
PhysioNet/CinC Challenge 2018 training set (Massachusetts General Hospital: 994 clinical PSG, AASM
scoring; F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, one eye channel E1-M2; WFDB format 16, 200 Hz).
A pre-registered random sample of records (seed 52; tr03-0005 excluded - its scoring was opened to
check the annotation reader). Stage notes in ``.arousal`` mark stage changes; a stage lasts until the
next stage note, the last one to the end of the record. Product path (Pipeline._sleep_staging).
Resumable: one JSON line per recording.

    PYTHONPATH=src python scripts/run_cinc2018_sleep.py run OUT.jsonl
    PYTHONPATH=src python scripts/run_cinc2018_sleep.py assess OUT.jsonl OUT.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hmc_sleep as hmc  # noqa: E402  (scoring and bootstrap are shared)

BASE = "https://physionet-open.s3.amazonaws.com/challenge-2018/1.0.0"
MODEL = "configs/models/sleep_stager_aasm_v4.json"
MODEL_SHA256 = "3affd3b5bb9c996f56259fe85817065ddc6ee4fbe29a57fd3a81f985e7b102d7"
N_SAMPLE, SEED, N_BOOT = 150, 52, 2000
EXCLUDED = {"tr03-0005"}
STAGE = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}
FS, EPOCH = 200, 30


def _curl(url, out=None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def records() -> list[str]:
    tr = sorted(ln.strip().split("/")[1] for ln in _curl(f"{BASE}/RECORDS").decode().splitlines()
                if ln.startswith("training/"))
    tr = [r for r in tr if r not in EXCLUDED]
    return sorted(np.random.default_rng(SEED).choice(tr, N_SAMPLE, replace=False).tolist())


def stages_from_notes(notes: list[tuple[int, int, str]], n_samples: int) -> list[tuple[int, int]]:
    st = [(t, STAGE[a]) for t, _, a in notes if a in STAGE]
    out = []
    for (t, c), nxt in zip(st, [s[0] for s in st[1:]] + [n_samples]):
        out += [(k, c) for k in range(t // (FS * EPOCH), nxt // (FS * EPOCH))]
    return out


def _one(rec: str) -> dict:
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer1_ingest.wfdb import read_annotations
    from neurolens.pipeline.pipeline import Pipeline

    try:
        with tempfile.TemporaryDirectory() as d:
            for ext in ("hea", "mat", "arousal"):
                _curl(f"{BASE}/training/{rec}/{rec}.{ext}", Path(d) / f"{rec}.{ext}")
            sig = ingest(Path(d) / f"{rec}.hea")
            lab = stages_from_notes(read_annotations(Path(d) / f"{rec}.arousal"), sig.signal.shape[1])
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


def assess(path: Path) -> dict:
    res = hmc.assess(path)
    rows = [r for r in hmc._load(path) if "error" not in r]
    rng = np.random.default_rng(SEED)
    boot = [hmc._score([rows[i] for i in rng.integers(0, len(rows), len(rows))])["kappa"] for _ in range(N_BOOT)]
    res["kappa_ci95_by_recording"] = [round(float(q), 3) for q in np.percentile(boot, [2.5, 97.5])]
    res["H1"] = bool(res["pooled"]["kappa"] >= 0.60 and res["kappa_ci95_by_recording"][0] >= 0.50)
    res["decision"] = "enable_aasm_model" if res["H1"] else "keep_off"
    return res


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
    done = {r["record"] for r in hmc._load(out)} if out.exists() else set()
    todo = [r for r in recs if r not in done]
    print(f"{len(recs)} recordings, {len(todo)} to run", flush=True)
    with out.open("a") as f:
        for rec in todo:
            r = _one(rec)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("record", "derivations", "eog_source", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
