"""Increment 69, development only (docs/preregistration_increment69.md).

Rebuilds the four development centres' per-night feature caches with the 13 new N1 features appended
to the v4 features (same analysis window, same labels), then runs leave-one-centre-out: design B (v4
features) vs design C (v4 + N1 features), on the same nights and folds.

    PYTHONPATH=src python scripts/dev_n1_features.py build-ds7181 data/inc69 [--workers 2]
    PYTHONPATH=src python scripts/dev_n1_features.py build-boas data/inc69
    PYTHONPATH=src python scripts/dev_n1_features.py build-hmc data/inc69
    PYTHONPATH=src python scripts/dev_n1_features.py build-eesm19 data/inc69      # ~3.7 GB per night
    PYTHONPATH=src python scripts/dev_n1_features.py loco data/inc69
"""

import argparse
import csv
import io
import json
import pickle
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev_aasm_stager_v2 as v2   # noqa: E402
import dev_aasm_stager_v3 as v3   # noqa: E402
import dev_aasm_stager_v4 as v4   # noqa: E402


def _extend(night, frontal, parietal, eog, fs, a, b):
    from neurolens.evaluation.sleep_eval import Night
    from neurolens.layer4_detect.sleep_staging import n1_features

    G, gn = n1_features(frontal, parietal, eog, fs, a, b)
    return Night(night.key, night.subject, np.column_stack([night.F, G]), night.names + gn, night.y)


def night_from_signal(sig, labels, key, subject):
    """v3's night (product derivations) + the N1 features over the same epochs."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    from neurolens.datasets.sleepedf import analysis_window
    from neurolens.evaluation.sleep_eval import Night
    from neurolens.layer4_detect.sleep_staging import EPOCH_S, MODEL_FS, derivations, epoch_features, n1_features

    der = derivations(sig.channel_names, sig.signal, aasm=True)
    if der is None or not der.pop("aasm", False):
        raise ValueError("no AASM derivations")
    fr = Fraction(MODEL_FS / sig.sampling_rate_hz).limit_denominator(1000)
    der = {k: None if v is None else resample_poly(v, fr.numerator, fr.denominator) for k, v in der.items()}
    n_ep = int(len(der["frontal"]) / MODEL_FS // EPOCH_S)
    stages = [None] * n_ep
    for k, c in labels:
        if 0 <= k < n_ep and c >= 0:
            stages[k] = c
    a, b = analysis_window(stages)
    F, names = epoch_features(der["frontal"], der["parietal"], der["eog"], MODEL_FS, a, b)
    G, gn = n1_features(der["frontal"], der["parietal"], der["eog"], MODEL_FS, a, b)
    y = np.array([-1 if s is None else s for s in stages[a:b]], dtype=int)
    return Night(key, zlib.crc32(subject.encode()), np.column_stack([F, G]), names + gn, y)


v3.night_from_signal = night_from_signal  # used by v3.build_boas / build_eesm19 and v4.build_hmc


def _one7181(item):
    sub, out = item
    from neurolens.datasets.sleepedf import analysis_window
    from neurolens.evaluation.sleep_eval import Night
    from neurolens.layer4_detect.sleep_staging import epoch_features, n1_features

    dst = Path(out) / f"ds7181_{sub}.pkl"
    if dst.exists():
        return f"cached {sub}"
    try:
        ev = list(csv.DictReader(io.StringIO(v2._curl(
            f"{v2.BASE}/{sub}/eeg/{sub}_task-sleep_acq-PSG_events.tsv").decode()), delimiter="\t"))
        with tempfile.TemporaryDirectory(dir=out) as d:
            p = Path(d) / f"{sub}.edf"
            v2._curl(f"{v2.BASE}/{sub}/eeg/{sub}_task-sleep_acq-PSG_eeg.edf", p)
            x = v2._read(p, ["F4-M1", "C4-M1", "O2-M1", "E1-M2", "E2-M2"])
        n = min(len(v) for v in x.values())
        frontal = x["F4-M1"][:n] - x["C4-M1"][:n]
        parietal = x["C4-M1"][:n] - x["O2-M1"][:n]
        eog = x["E1-M2"][:n] - x["E2-M2"][:n]
        n_ep = int(n / v2.FS // 30)
        stages = [None] * n_ep
        for e in ev:
            k = int(round(float(e["onset"]) / 30))
            if 0 <= k < n_ep:
                stages[k] = v2.STAGE.get(e["stage"].strip())
        a, b = analysis_window(stages)
        F, names = epoch_features(frontal, parietal, eog, v2.FS, a, b)
        G, gn = n1_features(frontal, parietal, eog, v2.FS, a, b)
        y = np.array([-1 if s is None else s for s in stages[a:b]], dtype=int)
        dst.write_bytes(pickle.dumps(Night(f"ds7181_{sub}", zlib.crc32(sub.encode()),
                                           np.column_stack([F, G]), names + gn, y)))
        return f"ok {sub} {n_ep} epochs"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {sub}: {exc!r}"


def centres(out: Path) -> dict:
    ld = lambda pat: [pickle.loads(p.read_bytes()) for p in sorted(out.glob(pat))]  # noqa: E731
    return {"ds007181": ld("ds7181_*.pkl"), "boas": ld("boas_*.pkl"), "eesm19": ld("eesm19_*.pkl"),
            "hmc": [d["night"] for d in ld("hmc_*.pkl")]}


def _design(with_n1: bool):
    from neurolens.layer4_detect.sleep_staging import N1_FEATURE_NAMES, standardise_and_context

    def fn(n):
        F, names = n.F, n.names
        if not with_n1:
            keep = [i for i, nm in enumerate(names) if nm not in N1_FEATURE_NAMES]
            F, names = F[:, keep], [names[i] for i in keep]
        return standardise_and_context(F, names, absolute=v4.ABS)
    return fn


def loco(out: Path) -> dict:
    from neurolens.layer4_detect.sleep_staging import fit_stager

    c = centres(out)
    res = {"n_nights": {k: len(v) for k, v in c.items()}}
    for tag, with_n1 in (("B", False), ("C", True)):
        design = _design(with_n1)
        for held in c:
            X, y, names = v4._matrix([n for k, v in c.items() if k != held for n in v], design)
            s = v4._eval(fit_stager(X, y, names), c[held], design)
            res[f"{tag}_held_out_{held}"] = {k: s[k] for k in ("kappa", "accuracy", "macro_f1", "f1")}
            print(tag, held, s["kappa"], s["f1"]["N1"], flush=True)
        res[f"{tag}_mean_kappa"] = round(float(np.mean([res[f"{tag}_held_out_{h}"]["kappa"] for h in c])), 4)
    d = {h: res[f"C_held_out_{h}"]["kappa"] - res[f"B_held_out_{h}"]["kappa"] for h in c}
    n1 = {h: res[f"C_held_out_{h}"]["f1"]["N1"] - res[f"B_held_out_{h}"]["f1"]["N1"] for h in c}
    res["delta_kappa"] = {h: round(v, 4) for h, v in d.items()}
    res["delta_f1_n1"] = {h: round(v, 4) for h, v in n1.items()}
    res["mean_delta_kappa"] = round(float(np.mean(list(d.values()))), 4)
    res["criteria"] = {"mean_gain_ge_0.008": bool(res["mean_delta_kappa"] >= 0.008),
                       "n1_f1_up_in_ge_3": bool(sum(v > 0 for v in n1.values()) >= 3),
                       "no_centre_drop_gt_0.01": bool(min(d.values()) >= -0.01)}
    res["accepted"] = all(res["criteria"].values())
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-ds7181", "build-boas", "build-hmc", "build-eesm19", "loco"])
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "build-ds7181":
        subs = v2.subjects()
        print(f"{len(subs)} subjects", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for msg in ex.map(_one7181, [(s, str(out)) for s in subs]):
                print(msg, flush=True)
    elif a.cmd == "build-boas":
        v3.build_boas(out)
    elif a.cmd == "build-hmc":
        v4.build_hmc(out)
    elif a.cmd == "build-eesm19":
        v3.build_eesm19(out)
    else:
        res = loco(out)
        (out / "loco.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
