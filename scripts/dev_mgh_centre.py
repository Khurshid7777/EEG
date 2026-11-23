"""Increment 71, development only (docs/preregistration_increment71.md): MGH (CinC 2018) as a fifth
training centre of the AASM sleep stager.

    PYTHONPATH=src python scripts/dev_mgh_centre.py build data/inc71 [WORKERS]
    PYTHONPATH=src python scripts/dev_mgh_centre.py loco data/inc71 data/inc69
    MGH_N=393 PYTHONPATH=src python scripts/dev_mgh_centre.py build data/inc71 [WORKERS]   # increment 72
    PYTHONPATH=src python scripts/dev_mgh_centre.py loco-boot data/inc71 data/inc69         # increment 72
    PYTHONPATH=src python scripts/dev_mgh_centre.py train data/inc71 data/inc69
"""

import json
import pickle
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_aasm_stager_v3 as v3          # noqa: E402
import dev_aasm_stager_v4 as v4          # noqa: E402
import run_cinc2018_sleep as cinc        # noqa: E402
import run_cinc2018_smoothing as smo     # noqa: E402
import run_cinc_n1 as inc68              # noqa: E402

import os

SEED, N_TRAIN = 71, int(os.environ.get("MGH_N", 250))


def selection() -> list[str]:
    used = set(cinc.records()) | set(smo.records()) | cinc.EXCLUDED | {r for r, _ in inc68.fresh_records()}
    tr = sorted(ln.strip().split("/")[1] for ln in cinc._curl(f"{cinc.BASE}/RECORDS").decode().splitlines()
                if ln.startswith("training/"))
    rest = [r for r in tr if r not in used]
    return np.random.default_rng(SEED).permutation(rest)[:N_TRAIN].tolist()


def _one(item):
    rec, out = item
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer1_ingest.wfdb import read_annotations

    dst = Path(out) / f"mgh_{rec}.pkl"
    if dst.exists():
        return f"cached {rec}"
    try:
        with tempfile.TemporaryDirectory() as d:
            for ext in ("hea", "mat", "arousal"):
                cinc._curl(f"{cinc.BASE}/training/{rec}/{rec}.{ext}", Path(d) / f"{rec}.{ext}")
            sig = ingest(Path(d) / f"{rec}.hea")
            lab = cinc.stages_from_notes(read_annotations(Path(d) / f"{rec}.arousal"), sig.signal.shape[1])
        dst.write_bytes(pickle.dumps(v3.night_from_signal(sig, lab, f"mgh_{rec}", f"mgh_{rec}")))
        return f"ok {rec}"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {rec}: {exc!r}"


def _centres(out: Path, old: Path) -> dict:
    import dev_n1_features as n1

    c = n1.centres(old)
    design = n1._design(False)   # v4 features only (design B)
    c["_design"] = design
    c["mgh"] = [pickle.loads(p.read_bytes()) for p in sorted(out.glob("mgh_*.pkl"))]
    return c


def loco(out: Path, old: Path) -> dict:
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = _centres(out, old)
    design, mgh = c.pop("_design"), c.pop("mgh")
    b = json.loads(Path("docs/increment69_loco.json").read_text())
    res = {"n_nights": {**{k: len(v) for k, v in c.items()}, "mgh": len(mgh)}}
    for held in c:
        train = [n for k, v in c.items() if k != held for n in v] + mgh
        X, y, names = v4._matrix(train, design)
        s = v4._eval(fit_stager(X, y, names), c[held], design)
        res[f"D_held_out_{held}"] = {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "f1")}
        print(held, s["kappa"], s["f1"]["N1"], flush=True)
    d = {h: res[f"D_held_out_{h}"]["kappa"] - b[f"B_held_out_{h}"]["kappa"] for h in c}
    n1 = {h: res[f"D_held_out_{h}"]["f1"]["N1"] - b[f"B_held_out_{h}"]["f1"]["N1"] for h in c}
    res["baseline_B_kappa"] = {h: b[f"B_held_out_{h}"]["kappa"] for h in c}
    res["delta_kappa"] = {h: round(v, 4) for h, v in d.items()}
    res["delta_f1_n1"] = {h: round(v, 4) for h, v in n1.items()}
    res["mean_delta_kappa"] = round(float(np.mean(list(d.values()))), 4)
    res["criteria"] = {"mean_gain_ge_0.005": bool(res["mean_delta_kappa"] >= 0.005),
                       "n1_f1_up_in_ge_3": bool(sum(v > 0 for v in n1.values()) >= 3),
                       "no_centre_drop_gt_0.01": bool(min(d.values()) >= -0.01)}
    res["accepted"] = all(res["criteria"].values())
    return res


def loco_boot(out: Path, old: Path) -> dict:
    """Increment 72: with / without MGH, paired bootstrap over the held-out centre's nights."""
    from neurolens.evaluation.sleep_eval import score
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = _centres(out, old)
    design, mgh = c.pop("_design"), c.pop("mgh")
    rng = np.random.default_rng(72)
    res = {"n_nights": {**{k: len(v) for k, v in c.items()}, "mgh": len(mgh)}}
    preds, boot = {}, {}
    for held in c:
        base = [n for k, v in c.items() if k != held for n in v]
        for tag, train in (("B", base), ("D", base + mgh)):
            X, y, names = v4._matrix(train, design)
            m = fit_stager(X, y, names)
            preds[held, tag] = [m.predict_proba(design(n)[0]).argmax(axis=1) for n in c[held]]
        sb, sd = (score(c[held], preds[held, t]) for t in ("B", "D"))
        res[f"B_{held}"], res[f"D_{held}"] = ({k: s[k] for k in ("kappa", "f1")} for s in (sb, sd))
        gains = []
        for _ in range(1000):
            i = rng.integers(0, len(c[held]), len(c[held]))
            ns = [c[held][j] for j in i]
            gains.append(score(ns, [preds[held, "D"][j] for j in i])["kappa"]
                         - score(ns, [preds[held, "B"][j] for j in i])["kappa"])
        boot[held] = np.array(gains)
        print(held, sb["kappa"], sd["kappa"], flush=True)
    d = {h: res[f"D_{h}"]["kappa"] - res[f"B_{h}"]["kappa"] for h in c}
    n1 = {h: res[f"D_{h}"]["f1"]["N1"] - res[f"B_{h}"]["f1"]["N1"] for h in c}
    mean_boot = np.mean([boot[h] for h in c], axis=0)
    res["delta_kappa"] = {h: round(v, 4) for h, v in d.items()}
    res["delta_f1_n1"] = {h: round(v, 4) for h, v in n1.items()}
    res["mean_delta_kappa"] = round(float(np.mean(list(d.values()))), 4)
    res["mean_delta_ci95"] = [round(float(q), 4) for q in np.percentile(mean_boot, [2.5, 97.5])]
    res["criteria"] = {"mean_gain_ge_0.008_and_ci_above_0": bool(res["mean_delta_kappa"] >= 0.008 and res["mean_delta_ci95"][0] > 0),
                       "n1_f1_up_in_ge_3": bool(sum(v > 0 for v in n1.values()) >= 3),
                       "no_centre_drop_gt_0.015": bool(min(d.values()) >= -0.015)}
    res["accepted"] = all(res["criteria"].values())
    return res


def train(out: Path, old: Path) -> Path:
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = _centres(out, old)
    design, mgh = c.pop("_design"), c.pop("mgh")
    X, y, names = v4._matrix([n for v in c.values() for n in v] + mgh, design)
    m = fit_stager(X, y, names, {"name": "sleep_stager_aasm_v5",
                                 "training": {**{k: len(v) for k, v in c.items()}, "mgh": len(mgh)},
                                 "derivations": "F4-C4 / C4-O2 / horizontal EOG (product derivations)",
                                 "absolute_features": list(v4.ABS)})
    return m.save("configs/models/sleep_stager_aasm_v5.json")


def main() -> int:
    cmd, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if cmd == "build":
        recs = selection()
        print(f"{len(recs)} records", flush=True)
        with ProcessPoolExecutor(int(sys.argv[3]) if len(sys.argv) > 3 else 3) as ex:
            for msg in ex.map(_one, [(r, str(out)) for r in recs]):
                print(msg, flush=True)
    elif cmd == "loco-boot":
        res = loco_boot(out, Path(sys.argv[3]))
        (out / "loco_boot.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
    elif cmd == "loco":
        res = loco(out, Path(sys.argv[3]))
        (out / "loco5.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
    else:
        print(train(out, Path(sys.argv[3])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
