"""Resumable routine-EEG harness (increment 31): one JSON line per NMT record, written as
it finishes; rerunning skips records already in OUT.jsonl. ``--finish OUT.json`` turns the
lines into the usual metrics file (`routine_eval.save`).

    PYTHONPATH=src python scripts/run_routine_jsonl.py IDS.json [IDS2.json ...] --out OUT.jsonl [--cache data/nmt] [--workers 4] [--finish OUT.json]

IDS files: ``{"record_ids": [...]}`` or harness metrics files (``{"records": [...]}``).
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_routine_ids import _one  # noqa: E402


def _ids(path: str) -> list[int]:
    d = json.loads(Path(path).read_text())
    return list(d["record_ids"]) if "record_ids" in d else [r["record_id"] for r in d["records"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="data/nmt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--finish")
    a = ap.parse_args()
    out = Path(a.out)
    rows = [json.loads(ln) for ln in out.read_text().splitlines()] if out.exists() else []
    done = {r["record_id"] for r in rows if "error" not in r}
    todo = [i for i in dict.fromkeys(i for f in a.ids for i in _ids(f)) if i not in done]
    print(f"{len(done)} done, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, out.open("a") as f:
        for n, r in enumerate(ex.map(_one, [(i, a.cache) for i in todo]), 1):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    if a.finish:
        from neurolens.evaluation.routine_eval import save

        want = set(i for f in a.ids for i in _ids(f))
        last = {}
        for r in map(json.loads, out.read_text().splitlines()):
            if r["record_id"] in want:
                last[r["record_id"]] = r
        rows = list(last.values())
        errors = [r for r in rows if "error" in r]
        s = save([r for r in rows if "error" not in r], Path(a.finish), {"errors": errors, "ids_files": a.ids})
        print(json.dumps({k: s[k] for k in ("n_normal", "n_abnormal", "impression_specificity",
                                              "impression_sensitivity")}), "errors", len(errors))
    return 0


if __name__ == "__main__":
    sys.exit(main())
