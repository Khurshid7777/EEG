"""Increment 68 (pre-registered, docs/preregistration_increment68.md): N1 prior-shift correction of the AASM stager.
Fresh CinC 2018 records (150 development + 150 test, seed 68). The run stores each record's raw stage probabilities so the
offset grid is evaluated offline (identical to the product path: offset -> argmax / Viterbi with the frozen transitions).

    PYTHONPATH=src python scripts/run_cinc_n1.py check                  # reader check on tr03-0005 (excluded)
    PYTHONPATH=src python scripts/run_cinc_n1.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_cinc_n1.py assess OUT.jsonl OUT.json
"""

import json
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cinc2018_sleep as cinc  # noqa: E402
import run_cinc2018_smoothing as smo  # noqa: E402
import run_hmc_sleep as hmc  # noqa: E402

SEED, N_DEV, N_TEST, N_BOOT = 68, 150, 150, 2000
GRID = [0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -2.0]
SMOOTHING = "configs/models/sleep_smoothing_aasm_v1.json"
STAGES = ["W", "N1", "N2", "N3", "REM"]


def fresh_records() -> list[tuple[str, str]]:
    used = set(cinc.records()) | set(smo.records()) | cinc.EXCLUDED
    tr = sorted(ln.strip().split("/")[1] for ln in cinc._curl(f"{cinc.BASE}/RECORDS").decode().splitlines()
                if ln.startswith("training/"))
    rest = [r for r in tr if r not in used]
    pick = np.random.default_rng(SEED).permutation(rest)[: N_DEV + N_TEST].tolist()
    return [(r, "dev" if i < N_DEV else "test") for i, r in enumerate(pick)]


def _one(args) -> dict:
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer1_ingest.wfdb import read_annotations
    from neurolens.layer4_detect.sleep_staging import MulticlassTreeModel, load_smoothing, stage_signal

    rec, split = args
    try:
        with tempfile.TemporaryDirectory() as d:
            for ext in ("hea", "mat", "arousal"):
                cinc._curl(f"{cinc.BASE}/training/{rec}/{rec}.{ext}", Path(d) / f"{rec}.{ext}")
            sig = ingest(Path(d) / f"{rec}.hea")
            lab = cinc.stages_from_notes(read_annotations(Path(d) / f"{rec}.arousal"), sig.signal.shape[1])
        model = MulticlassTreeModel.load(cinc.MODEL)
        st = stage_signal(sig.channel_names, sig.signal, sig.sampling_rate_hz, model, aasm_model=model,
                          aasm_smoothing=load_smoothing(SMOOTHING), keep_proba=True)
        if not st:
            return {"record": rec, "split": split, "error": "not staged"}
        return {"record": rec, "split": split, "duration_h": round(sig.duration_s / 3600, 2),
                "hyp0": [STAGES.index(s) for s in st["hypnogram"]], "proba": np.round(st["_proba"], 4).tolist(), "human": lab}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"record": rec, "split": split, "error": repr(exc)}


def hyp_for(row, b: float, sm) -> list[int]:
    from neurolens.layer4_detect.sleep_staging import apply_n1_offset, viterbi_smooth

    p = apply_n1_offset(np.array(row["proba"]), b)
    return [int(x) for x in viterbi_smooth(p, sm["transitions"], sm["weight"])]


def kappa_of(rows, b, sm) -> float:
    return hmc._score([{**r, "hyp": hyp_for(r, b, sm)} for r in rows])["kappa"]


def assess(path: Path) -> dict:
    from neurolens.layer4_detect.sleep_staging import load_smoothing

    sm = load_smoothing(SMOOTHING)
    rows = [r for r in hmc._load(path) if "error" not in r]
    dev, test = [r for r in rows if r["split"] == "dev"], [r for r in rows if r["split"] == "test"]
    # sanity: b = 0 reproduces the product hypnogram
    same = float(np.mean([hyp_for(r, 0.0, sm) == r["hyp0"] for r in rows]))
    dev_k = {b: kappa_of(dev, b, sm) for b in GRID}
    best = max(GRID, key=lambda b: (round(dev_k[b], 3), -abs(b)))
    # tie rule: |b| smaller within 0.001
    cands = [b for b in GRID if dev_k[best] - dev_k[b] < 0.001]
    b_star = min(cands, key=abs)
    hyp0 = [{**r, "hyp": hyp_for(r, 0.0, sm)} for r in test]
    hypb = [{**r, "hyp": hyp_for(r, b_star, sm)} for r in test]
    s0, sb = hmc._score(hyp0), hmc._score(hypb)
    rng = np.random.default_rng(SEED)
    gains = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(test), len(test))
        gains.append(hmc._score([hypb[j] for j in i])["kappa"] - hmc._score([hyp0[j] for j in i])["kappa"])
    ci = [round(float(q), 4) for q in np.percentile(gains, [2.5, 97.5])]
    f1d = {k: round(sb["f1"][k] - s0["f1"][k], 3) for k in sb["f1"]}
    guard = f1d.get("N1", 0) >= 0 and all(f1d[k] >= -0.03 for k in f1d if k != "N1")
    gain = sb["kappa"] - s0["kappa"]
    return {"n_dev": len(dev), "n_test": len(test), "errors": [(r["record"], r["error"]) for r in hmc._load(path) if "error" in r],
            "b0_reproduces_product": round(same, 3), "dev_kappa_by_b": {str(b): round(v, 4) for b, v in dev_k.items()}, "b_star": b_star,
            "test_kappa_b0": round(s0["kappa"], 4), "test_kappa_bstar": round(sb["kappa"], 4), "test_gain": round(gain, 4), "test_gain_ci95": ci,
            "f1_b0": {k: round(v, 3) for k, v in s0["f1"].items()}, "f1_bstar": {k: round(v, 3) for k, v in sb["f1"].items()},
            "f1_difference": f1d, "guard_ok": bool(guard),
            "H_cinc": bool(gain >= 0.005 and ci[0] > 0 and guard)}


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "assess":
        res = assess(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 0
    if cmd == "check":
        r = _one(("tr03-0005", "check"))
        print({k: (v if k in ("record", "error", "duration_h") else len(v)) for k, v in r.items()})
        if "error" not in r:
            from neurolens.layer4_detect.sleep_staging import load_smoothing
            sm = load_smoothing(SMOOTHING); print("b=0 reproduces:", hyp_for(r, 0.0, sm) == r["hyp0"], "kappa b0", round(kappa_of([r], 0.0, sm), 3), "b-1", round(kappa_of([r], -1.0, sm), 3))
        return 0
    out, workers = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 3
    recs = fresh_records()
    done = {r["record"] for r in hmc._load(out)} if out.exists() else set()
    todo = [x for x in recs if x[0] not in done]
    print(f"{len(recs)} records ({N_DEV} dev + {N_TEST} test), {len(todo)} to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r) + "\n"); f.flush()
            print(r["record"], r["split"], r.get("error"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
