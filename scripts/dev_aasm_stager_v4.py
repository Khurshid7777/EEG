"""Increment 52, development only. Increment 51 (D93): the multi-centre stager v3 called 54% of N3
epochs N2 on HMC. Its features are z-scored per night (median / IQR), which removes absolute
amplitude - the quantity AASM defines N3 by (slow waves >= 75 uV). HMC (PhysioNet, the test of
increment 51) is now a fourth development centre.

  * variant A: the v3 design (per-night z-score + context);
  * variant B: A + the un-standardised log band powers (total, 0.5-2 Hz, 2-4 Hz) of both
    derivations - absolute amplitude;
  * EOG fallback for a recording with one eye electrode (E1-M2 only - the planned test centre):
    neutral EOG (current product) vs the single referential channel, on HMC held out.

    PYTHONPATH=src python scripts/dev_aasm_stager_v4.py build-hmc data/aasm3
    PYTHONPATH=src python scripts/dev_aasm_stager_v4.py loco data/aasm3
    PYTHONPATH=src python scripts/dev_aasm_stager_v4.py eog data/aasm3
"""

import argparse
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_aasm_stager_v3 as v3   # noqa: E402
import run_hmc_sleep as hmc       # noqa: E402

ABS = ("fc_logpow", "fc_log_sdelta", "fc_log_delta", "po_logpow", "po_log_sdelta", "po_log_delta")
EOG_NAMES = ("eog_log_slow", "eog_log_fast", "eog_logvar")


def build_hmc(out: Path) -> None:
    from fractions import Fraction

    from scipy.signal import resample_poly

    from neurolens.datasets.sleepedf import analysis_window
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer4_detect.sleep_staging import MODEL_FS, _eog_features, epoch_matrix

    for rec in hmc.records():
        dst = out / f"hmc_{rec}.pkl"
        if dst.exists():
            continue
        try:
            lab = hmc._scoring(rec)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / f"{rec}.edf"
                hmc._curl(f"{hmc.BASE}/recordings/{rec}.edf", p)
                sig = ingest(p)
            night = v3.night_from_signal(sig, lab, f"hmc_{rec}", f"hmc_{rec}")
            # the single referential eye channel E1-M2, over the same analysis window
            e1 = sig.signal[[n.upper() for n in sig.channel_names].index("EOG E1")]
            fr = Fraction(MODEL_FS / sig.sampling_rate_hz).limit_denominator(1000)
            e1 = resample_poly(e1, fr.numerator, fr.denominator)
            n_ep = int(len(e1) / MODEL_FS // 30)
            stages = [None] * n_ep
            for k, c in lab:
                if 0 <= k < n_ep:
                    stages[k] = c
            a, b = analysis_window(stages)
            single, _ = _eog_features(epoch_matrix(e1, MODEL_FS, a, b), MODEL_FS)
            dst.write_bytes(pickle.dumps({"night": night, "eog_e1": single}))
            print("ok", rec, flush=True)
        except Exception as exc:  # noqa: BLE001
            print("ERROR", rec, repr(exc), flush=True)


def centres(out: Path) -> dict:
    # local feature caches written by this project (build_hmc here; the rest by dev_aasm_stager_v3.py)
    c = v3.centres(out)
    c["hmc"] = [pickle.loads(p.read_bytes())["night"] for p in sorted(out.glob("hmc_*.pkl"))]
    return c


def design_b(night):
    from neurolens.layer4_detect.sleep_staging import standardise_and_context

    return standardise_and_context(night.F, night.names, absolute=ABS)


def _matrix(nights, design):
    Xs, ys, names = [], [], None
    for n in nights:
        X, names = design(n)
        keep = n.y >= 0
        Xs.append(X[keep])
        ys.append(n.y[keep])
    return np.vstack(Xs), np.concatenate(ys), names


def _eval(model, nights, design):
    from neurolens.evaluation.sleep_eval import score

    return score(nights, [model.predict_proba(design(n)[0]).argmax(axis=1) for n in nights])


def loco(out: Path) -> dict:
    from neurolens.evaluation.sleep_eval import design as design_a
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = centres(out)
    res = {"n_nights": {k: len(v) for k, v in c.items()},
           "abs_median_by_centre": {k: {n: round(float(np.median(np.concatenate(
               [x.F[:, x.names.index(n)] for x in v]))), 2) for n in ABS} for k, v in c.items()}}
    for tag, design in (("A", design_a), ("B", design_b)):
        for held in c:
            X, y, names = _matrix([n for k, v in c.items() if k != held for n in v], design)
            s = _eval(fit_stager(X, y, names), c[held], design)
            res[f"{tag}_held_out_{held}"] = {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "f1")}
            print(tag, held, s["kappa"], s["f1"]["N3"], flush=True)
        res[f"{tag}_mean_kappa"] = round(float(np.mean([res[f"{tag}_held_out_{h}"]["kappa"] for h in c])), 4)
    return res


def eog(out: Path, design_tag: str) -> dict:
    """HMC held out: bipolar E1-E2 (reference) vs neutral EOG vs the single E1-M2 channel."""
    from neurolens.evaluation.sleep_eval import Night, design as design_a
    from neurolens.layer4_detect.sleep_staging import fit_stager

    design = design_a if design_tag == "A" else design_b
    c = v3.centres(out)
    X, y, names = _matrix([n for v in c.values() for n in v], design)
    m = fit_stager(X, y, names)
    raw = [pickle.loads(p.read_bytes()) for p in sorted(out.glob("hmc_*.pkl"))]
    cols = [raw[0]["night"].names.index(n) for n in EOG_NAMES]

    def swap(night, F):
        return Night(night.key, night.subject, F, night.names, night.y)

    variants = {"bipolar": [r["night"] for r in raw], "neutral": [], "single_e1": []}
    for r in raw:
        n = r["night"]
        F = n.F.copy()
        F[:, cols] = 0.0  # the product's neutral EOG: constant columns -> z-score 0
        variants["neutral"].append(swap(n, F))
        F = n.F.copy()
        F[:, cols] = r["eog_e1"]
        variants["single_e1"].append(swap(n, F))
    res = {"design": design_tag}
    for k, v in variants.items():
        s = _eval(m, v, design)
        res[k] = {kk: s[kk] for kk in ("kappa", "f1")}
        print(k, s["kappa"], flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-hmc", "loco", "eog"])
    ap.add_argument("out")
    ap.add_argument("--design", default="A", choices=["A", "B"])
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "build-hmc":
        build_hmc(out)
        return 0
    res = loco(out) if a.cmd == "loco" else eog(out, a.design)
    name = "loco_v4.json" if a.cmd == "loco" else f"eog_{a.design}.json"
    (out / name).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if "mean" in k or k in ("design",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
