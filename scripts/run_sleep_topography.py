"""Increment 79 (pre-registered, docs/preregistration_increment79.md): sleep rhythm topography on HMC.
Resumable: one JSON line per recording.

    PYTHONPATH=src python scripts/run_sleep_topography.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_sleep_topography.py assess OUT.jsonl OUT.json
"""

import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_hmc_sleep as hmc  # noqa: E402

CH = ("F4", "C4", "C3", "O2")
BANDS = {"sigma": (12.0, 15.0), "delta": (0.5, 4.0), "alpha": (8.0, 12.0)}
STAGE_OF = {"W": 0, "N2": 2, "N3": 3}
MIN_EPOCHS, MAX_PTP = 20, 500.0


def _one(rec):
    from scipy.signal import welch

    from neurolens.layer1_ingest.registry import ingest

    try:
        lab = hmc._scoring(rec)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / f"{rec}.edf"
            hmc._curl(f"{hmc.BASE}/recordings/{rec}.edf", p)
            sig = ingest(p)
        idx = [sig.channel_names.index(c) for c in CH]
        fs = sig.sampling_rate_hz
        n = int(30 * fs)
        x = sig.signal[idx]
        out = {"record": rec, "stages": {}}
        for sname, code in STAGE_OF.items():
            vals = {c: {b: [] for b in BANDS} for c in CH}
            absv = {c: {b: [] for b in BANDS} for c in CH}
            used = 0
            for k, c in lab:
                if c != code or (k + 1) * n > x.shape[1]:
                    continue
                seg = x[:, k * n:(k + 1) * n]
                if np.ptp(seg, axis=1).max() > MAX_PTP:
                    continue
                f, pw = welch(seg, fs=fs, nperseg=int(4 * fs), axis=-1)
                tot = pw[:, (f >= 0.5) & (f < 30)].sum(axis=1)
                for j, ch in enumerate(CH):
                    for b, (lo, hi) in BANDS.items():
                        bp = pw[j, (f >= lo) & (f < hi)].sum()
                        vals[ch][b].append(bp / tot[j])
                        absv[ch][b].append(np.log10(bp + 1e-12))
                used += 1
            out["stages"][sname] = {"n_epochs": used,
                                    "median": {ch: {b: float(np.median(v)) for b, v in vals[ch].items()} for ch in CH} if used else None,
                                    "median_log10_abs": {ch: {b: float(np.median(v)) for b, v in absv[ch].items()} for ch in CH} if used else None}
        return out
    except Exception as exc:  # noqa: BLE001 - excluded and counted, as pre-registered
        return {"record": rec, "error": repr(exc)}


def run(out_path, workers):
    out = Path(out_path)
    done = {json.loads(l)["record"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in hmc.records() if r not in done]
    print(len(todo), "to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["record"], r.get("error"), flush=True)


def _ci(k, n):
    from neurolens.evaluation.prereg import clopper_pearson
    return [round(float(x), 3) for x in clopper_pearson(k, n)]


def assess(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    res = {"n_records": len(rows), "n_errors": len(rows) - len(ok)}

    def frac(stage, fn):
        use = [r for r in ok if r["stages"][stage]["n_epochs"] >= MIN_EPOCHS]
        hits = sum(bool(fn(r["stages"][stage]["median"])) for r in use)
        return {"n": len(use), "excluded": len(ok) - len(use), "frac": hits / len(use) if use else None, "ci95": _ci(hits, len(use)) if use else None}

    cen = lambda m, b: (m["C3"][b] + m["C4"][b]) / 2  # noqa: E731
    res["H1_N2_sigma_central_gt_occipital"] = frac("N2", lambda m: cen(m, "sigma") > m["O2"]["sigma"])
    res["H2_N3_delta_frontal_gt_occipital"] = frac("N3", lambda m: m["F4"]["delta"] > m["O2"]["delta"])
    res["H3_W_alpha_occipital_gt_frontal"] = frac("W", lambda m: m["O2"]["alpha"] > m["F4"]["alpha"])
    res["descriptive_N2_sigma_C4_gt_C3"] = frac("N2", lambda m: m["C4"]["sigma"] > m["C3"]["sigma"])
    res["descriptive_N2_sigma_frontal_gt_central"] = frac("N2", lambda m: m["F4"]["sigma"] > cen(m, "sigma"))
    for k in ("H1_N2_sigma_central_gt_occipital", "H2_N3_delta_frontal_gt_occipital", "H3_W_alpha_occipital_gt_frontal"):
        res[k]["ok"] = bool(res[k]["frac"] is not None and res[k]["frac"] >= 0.90)
    res["decision"] = {k[:2]: res[k]["ok"] for k in res if k.startswith("H")}
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        r = assess(sys.argv[2])
        Path(sys.argv[3]).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r, indent=1))
