"""Increment 84 (pre-registered, docs/preregistration_increment84.md): electrode-label plausibility check.
Per-channel relative alpha is computed once per record (robust rule, increment 82); label swaps are applied to the names only.

    PYTHONPATH=src python scripts/run_montage_sanity.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_montage_sanity.py assess OUT.jsonl OUT.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SWAPS = {
    "S1": [("O1", "Fp1"), ("O2", "Fp2")],
    "S2": [("O1", "F7"), ("O2", "F8")],
    "S3": [("Pz", "Fz"), ("P3", "F3"), ("P4", "F4")],
    "S4": [("T5", "F7"), ("T6", "F8"), ("O1", "Fp1"), ("O2", "Fp2")],
    "S5": [("O1", "O2")],
}


def _perm(pairs):
    m = {}
    for a, b in pairs:
        m[a], m[b] = b, a
    return m


def _one(item):
    which, key = item
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer3_features.regional import channel_band_power, label_plausibility

    try:
        if which == "ds003775":
            from neurolens.datasets.ds004504 import Ds003775Client
            c = Ds003775Client(); sig = ingest(c.fetch(next(r for r in c.records() if r.subject == key)))
        else:
            from neurolens.datasets.ds004504 import Ds004504Client
            c = Ds004504Client(); sig = ingest(c.fetch(next(r for r in c.records() if r.subject == key)))
        pc = channel_band_power(sig.signal, list(sig.channel_names), sig.sampling_rate_hz, robust=True)
        out = {"set": which, "subject": key, "none": label_plausibility(pc)["suspect"]}
        for name, pairs in SWAPS.items():
            m = _perm(pairs)
            out[name] = label_plausibility({m.get(k, k): v for k, v in pc.items()})["suspect"]
        return out
    except Exception as exc:  # noqa: BLE001 - excluded and counted
        return {"set": which, "subject": key, "error": repr(exc)}


def _items():
    from neurolens.datasets.ds004504 import Ds003775Client, Ds004504Client
    return [("ds003775", r.subject) for r in Ds003775Client().records()] + \
           [("ds004504", r.subject) for r in Ds004504Client().records() if r.group == "C"]


def run(out, workers):
    out = Path(out)
    done = {(json.loads(l)["set"], json.loads(l)["subject"]) for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, [i for i in _items() if i not in done]):
            f.write(json.dumps(r) + "\n"); f.flush()


def _ci(k, n):
    from neurolens.evaluation.prereg import clopper_pearson
    return [round(float(x), 3) for x in clopper_pearson(k, n)]


def assess(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    ok = [r for r in rows if "error" not in r and r["none"] is not None]
    res = {"n": len(rows), "n_usable": len(ok), "errors": len(rows) - len([r for r in rows if "error" not in r])}
    for s in ("ds003775", "ds004504"):
        sub = [r for r in ok if r["set"] == s]
        k = sum(bool(r["none"]) for r in sub)
        res[f"C1_false_flag_{s}"] = {"n": len(sub), "k": k, "rate": round(k / max(1, len(sub)), 3), "ci95": _ci(k, len(sub)), "ok": bool(k / max(1, len(sub)) <= 0.08)}
    for name in SWAPS:
        k = sum(bool(r[name]) for r in ok)
        res[f"detect_{name}"] = {"k": k, "n": len(ok), "rate": round(k / len(ok), 3), "ci95": _ci(k, len(ok))}
    res["C2_ok"] = bool(all(res[f"detect_{n}"]["rate"] >= 0.85 for n in ("S1", "S2", "S4")))
    res["C3_S5_le_0.10"] = bool(res["detect_S5"]["rate"] <= 0.10)
    res["connect_to_pipeline"] = bool(res["C1_false_flag_ds003775"]["ok"] and res["C1_false_flag_ds004504"]["ok"] and res["C2_ok"])
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        r = assess(sys.argv[2]); Path(sys.argv[3]).write_text(json.dumps(r, indent=1) + "\n"); print(json.dumps(r, indent=1))
