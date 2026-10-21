"""Increment 38 (pre-registered, docs/preregistration_increment38.md): the frozen IED detectors on
vepiset (84 patients, PUMCH). Resumable: one JSON line per recording; each MAT file is streamed
from the remote zip and deleted after use.

    PYTHONPATH=src python scripts/run_vepiset_eval.py OUT.jsonl [--workers 3]
    PYTHONPATH=src python scripts/run_vepiset_eval.py --assess IN.jsonl OUT.json
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(rec):
    import numpy as np

    from neurolens.datasets.vepiset import VepisetClient
    from neurolens.evaluation import vepiset_eval as V
    from neurolens.evaluation.ied_eval import _sha256
    from neurolens.layer4_detect.ml_ictal import load_model
    from neurolens.pipeline.config_loader import load_configs

    frozen = json.loads(Path("docs/preregistration_increment15_frozen.json").read_text())
    if _sha256(Path(frozen["model"]["path"])) != frozen["model"]["sha256"]:
        raise RuntimeError("IED model does not match the frozen sha256")
    cl = VepisetClient()
    base = {"record_id": rec.record_id, "name": Path(rec.name).stem}
    try:
        x, ann = VepisetClient.load(cl.fetch(rec))
        sig = V.to_signal(x)
        normal = not ann.discharges
        r = V.prepare(sig, rec.record_id, normal, load_configs())
        p = V.ied_ml.model_probability(load_model(Path(frozen["model"]["path"])), r.X) if r.X.shape[0] else np.zeros(0)
        return {**base, "n_marks": len(ann.discharges), "n_ref": len(V.reference_events(ann)),
                "duration_s": r.duration_s, "amplitude_sd_uv": float(np.median(np.std(sig.signal[:19], axis=1))),
                "states": ann.states, **V.score(r, ann, p, frozen["threshold"]),
                "epochs": V.epoch_table(r, ann, p)}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}
    finally:
        cl.drop(rec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--assess", action="store_true")
    a = ap.parse_args()
    if a.assess:
        from neurolens.evaluation.prereg38 import assess

        res = assess([json.loads(ln) for ln in Path(a.paths[0]).read_text().splitlines()])
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "heuristic_curve"}, indent=1, default=float))
        return 0
    from neurolens.datasets.vepiset import VepisetClient

    out = Path(a.paths[0])
    recs = VepisetClient().records()
    done = {json.loads(ln)["name"] for ln in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in recs if Path(r.name).stem not in done]
    print(f"{len(recs)} recordings, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
