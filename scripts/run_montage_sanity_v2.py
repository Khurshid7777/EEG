"""Increment 86 (pre-registered, docs/preregistration_increment86.md): electrode-label plausibility v2 on new data
(eegmat rest; ds004504 A and F groups). Downloaded patient files are deleted after use.

    PYTHONPATH=src python scripts/run_montage_sanity_v2.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_montage_sanity_v2.py assess OUT.jsonl OUT.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_montage_sanity import SWAPS, _ci, _perm


def _one(item):
    which, key = item
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer3_features.regional import channel_band_power, label_plausibility_v2

    path = None
    try:
        if which == "eegmat":
            import run_eegmat_eval as eg
            path = None
            sig = ingest(eg._get(f"{key}_1.edf"))
        else:
            from neurolens.datasets.ds004504 import Ds004504Client
            c = Ds004504Client()
            rec = next(r for r in c.records() if r.subject == key)
            path = c.fetch(rec)
            sig = ingest(path)
        pc = channel_band_power(sig.signal, list(sig.channel_names), sig.sampling_rate_hz, robust=True)
        out = {"set": which, "subject": key, "none": label_plausibility_v2(pc)["suspect"]}
        for name, pairs in SWAPS.items():
            m = _perm(pairs)
            out[name] = label_plausibility_v2({m.get(k, k): v for k, v in pc.items()})["suspect"]
        return out
    except Exception as exc:  # noqa: BLE001 - excluded and counted
        return {"set": which, "subject": key, "error": repr(exc)}
    finally:
        if path is not None:
            try:
                Path(path).unlink(missing_ok=True)
                fdt = Path(path).with_suffix(".fdt")
                fdt.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def _items():
    import run_eegmat_eval as eg

    from neurolens.datasets.ds004504 import Ds004504Client
    return [("eegmat", s["subject"]) for s in eg.subjects()] + \
           [("ds004504", r.subject) for r in Ds004504Client().records() if r.group in ("A", "F")]


def run(out, workers):
    out = Path(out)
    done = {(json.loads(l)["set"], json.loads(l)["subject"]) for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, [i for i in _items() if i not in done]):
            f.write(json.dumps(r) + "\n"); f.flush()


def assess(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    ok = [r for r in rows if "error" not in r and r["none"] is not None]
    res = {"n": len(rows), "n_usable": len(ok), "errors": [(r["set"], r["subject"], r["error"][:60]) for r in rows if "error" in r][:5]}
    for s, lim, tag in (("eegmat", 0.10, "C1_eegmat"), ("ds004504", 0.15, "C1_ds004504_AF")):
        sub = [r for r in ok if r["set"] == s]
        k = sum(bool(r["none"]) for r in sub)
        res[tag] = {"n": len(sub), "k": k, "rate": round(k / max(1, len(sub)), 3), "ci95": _ci(k, len(sub)), "limit": lim, "ok": bool(k / max(1, len(sub)) <= lim)}
    for name in SWAPS:
        k = sum(bool(r[name]) for r in ok)
        res[f"detect_{name}"] = {"k": k, "n": len(ok), "rate": round(k / len(ok), 3), "ci95": _ci(k, len(ok))}
    res["C2_ok"] = bool(all(res[f"detect_{n}"]["rate"] >= 0.80 for n in ("S1", "S2", "S4")))
    res["connect_to_pipeline"] = bool(res["C1_eegmat"]["ok"] and res["C1_ds004504_AF"]["ok"] and res["C2_ok"])
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        r = assess(sys.argv[2]); Path(sys.argv[3]).write_text(json.dumps(r, indent=1) + "\n"); print(json.dumps(r, indent=1))
