"""Increment 39 (pre-registered, docs/preregistration_increment39.md): frozen seizure detectors
(increment 11 defaults: learned v2 at point A, and the threshold detectors) on AUBMC.

    PYTHONPATH=src python scripts/run_aubmc_eval.py [--workers 3]
"""

import argparse
import json
import sys
from pathlib import Path

OUT = Path("docs/increment39_aubmc")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    from neurolens.datasets.aubmc import AubmcClient
    from neurolens.evaluation.ml_eval import run_and_save
    from neurolens.evaluation.prereg39 import assess
    from neurolens.pipeline.config_loader import load_configs

    c = AubmcClient()
    anns = c.annotations()
    items = []
    for ann in anns:
        items.append((ann, c.fetch(ann.file)))
        print(f"fetched {ann.file}", flush=True)
    out = run_and_save(items, load_configs(), "docs/preregistration_increment11_frozen.json", OUT,
                       feature_cache="data/aubmc/features", workers=a.workers, progress=print,
                       stream_cache="data/aubmc/stream")
    metrics = json.loads(out["paths"]["json"].read_text(encoding="utf-8"))
    amb = {x.file: [(s.onset_s, s.offset_s) for s in x.ambiguous] for x in anns}
    res = assess(metrics, amb)
    Path("docs/increment39_assessment.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: res[k] for k in ("hypotheses", "decision", "fa_per_hour_excluding_ambiguous")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
