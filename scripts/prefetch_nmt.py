"""Download NMT records in parallel before a harness run (increment 20).

Each record is fetched with direct HTTP range reads of its zip member (offsets in
``index.json``), so threads do not contend for one ``zipfile`` handle:

    PYTHONPATH=src python scripts/prefetch_nmt.py IDS.json [IDS2.json ...] --cache data/nmt --threads 8

``IDS.json`` is either an ids file (``{"record_ids": [...]}``) or a metrics file
written by the harness (``{"records": [{"record_id": ...}, ...]}``).
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _ids(path: str) -> list[int]:
    d = json.loads(Path(path).read_text())
    if "record_ids" in d:
        return list(d["record_ids"])
    return [r["record_id"] for r in d["records"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cache", default="data/nmt")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    from neurolens.datasets.nmt import NMTClient

    c = NMTClient(a.cache)
    if any(len(v) < 5 for v in c.index().values()):
        c._write_index()  # member offsets first: the zipfile fallback is not thread-safe
    recs = {r.record_id: r for r in c.labels()}
    ids = sorted({i for f in a.files for i in _ids(f)})
    members = [c.member_of(recs[i]) for i in ids]
    t0, done, failed = time.time(), 0, []

    def one(m):
        for attempt in range(3):
            try:
                return c.fetch(m)
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    raise exc
                time.sleep(10 * (attempt + 1))

    with ThreadPoolExecutor(a.threads) as ex:
        futs = {ex.submit(one, m): m for m in members}
        for f in as_completed(futs):
            try:
                f.result()
                done += 1
            except Exception as exc:  # noqa: BLE001
                failed.append((futs[f], repr(exc)))
            if (done + len(failed)) % 25 == 0:
                print(f"{done + len(failed)}/{len(members)} in {time.time() - t0:.0f} s", flush=True)
    print(json.dumps({"fetched": done, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
