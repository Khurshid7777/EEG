"""Increment 32 (pre-registered, docs/preregistration_increment32.md): the frozen pipeline
on the TUAB evaluation split. Resumable: one JSON line per recording.

    PYTHONPATH=src python scripts/run_tuab_eval.py TUAB_ROOT OUT.jsonl [--split eval] [--workers 4] [--assess OUT.json]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(rec):
    from neurolens.evaluation.routine_eval import run_record

    base = {"record_id": rec.record_id, "name": rec.name, "abnormal": rec.abnormal}
    try:
        row = run_record(rec.path, rec, scale=1.0)  # TUH amplitudes are in µV
    except Exception as exc:  # noqa: BLE001 - unreadable files are listed and excluded
        return {**base, "error": repr(exc)}
    return {**row, **base, "age_years": rec.age_years}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--assess")
    a = ap.parse_args()
    from neurolens.datasets.tuab import TUABClient

    out = Path(a.out)
    recs = TUABClient(a.root).records(a.split)
    done = {json.loads(ln)["name"] for ln in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in recs if r.name not in done]
    print(f"{len(recs)} recordings, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    if a.assess:
        from neurolens.evaluation.prereg32 import assess

        res = assess([json.loads(ln) for ln in out.read_text().splitlines()])
        Path(a.assess).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps({k: res[k] for k in ("n_scored", "n_excluded", "auc", "hypotheses", "decision")},
                         indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
