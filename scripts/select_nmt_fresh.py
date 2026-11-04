"""Increment 56: the NMT records that no earlier increment has opened (the pre-registered fresh test of the
routine-EEG verdict). Excluded: the 1,045 development records (docs/increment21_dev_ids.json), the authors'
evaluation split (`loc == "eval"`, 185 records, docs/increment22_eval_ids.json), records with conflicting labels,
and every record of the NMT-Events train and test splits (NMT-Events is a subset of NMT with the same numbering;
its IED model and tests opened those records). The list is written once, before any of these records is run.

    PYTHONPATH=src python scripts/select_nmt_fresh.py docs/increment56_nmt_fresh_ids.json
"""

import json
import sys
from pathlib import Path


def main() -> int:
    from neurolens.datasets.nmt import NMTClient
    from neurolens.datasets.nmt_events import NMTEventsClient

    c = NMTClient("data/nmt")
    recs = c.labels()
    dev = {int(x) for x in json.loads(Path("docs/increment21_dev_ids.json").read_text())["record_ids"]} \
        if isinstance(json.loads(Path("docs/increment21_dev_ids.json").read_text()), dict) \
        else {int(x) for x in json.loads(Path("docs/increment21_dev_ids.json").read_text())}
    ev = NMTEventsClient("data/nmt_events")
    events = {rid for s in ("train", "test") for rid, _, _ in ev.split_members(s)}
    pool = [r for r in recs if r.loc == "train" and not r.label_conflict]
    fresh = sorted((r for r in pool if r.record_id not in dev and r.record_id not in events), key=lambda r: r.record_id)
    out = {"record_ids": [r.record_id for r in fresh],
           "n_normal": sum(not r.abnormal for r in fresh), "n_abnormal": sum(r.abnormal for r in fresh),
           "excluded": {"all_records": len(recs), "authors_eval_split": sum(r.loc == "eval" for r in recs),
                        "label_conflicts": sum(r.label_conflict for r in recs), "development": len(dev),
                        "nmt_events_train_test": len(events),
                        "nmt_events_not_already_development": len(events - dev)}}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "record_ids"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
