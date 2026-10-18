"""Increment 33 (pre-registered, docs/preregistration_increment33.md): the frozen pipeline on
OpenNeuro ds004504. Resumable: one JSON line per participant.

    PYTHONPATH=src python scripts/run_ds004504_eval.py OUT.jsonl [--workers 4] [--assess OUT.json]
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(rec):
    from neurolens.datasets.ds004504 import Ds004504Client
    from neurolens.evaluation.routine_eval import run_record

    base = {"record_id": rec.record_id, "subject": rec.subject, "group": rec.group,
            "abnormal": rec.abnormal, "age_years": rec.age_years, "mmse": rec.mmse}
    try:
        row = run_record(Ds004504Client().fetch(rec), rec, scale=1.0)  # EEGLAB data in µV
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}
    return {**row, **base}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--assess")
    a = ap.parse_args()
    from neurolens.datasets.ds004504 import Ds004504Client

    out = Path(a.out)
    recs = Ds004504Client().records()
    done = {json.loads(ln)["subject"] for ln in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in recs if r.subject not in done]
    print(f"{len(recs)} participants, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
    if a.assess:
        from neurolens.evaluation.prereg33 import assess

        res = assess([json.loads(ln) for ln in out.read_text().splitlines()])
        Path(a.assess).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
