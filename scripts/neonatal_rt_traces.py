"""Build monitor traces with epoch rhythmicity for all Helsinki recordings (increment 27).

    PYTHONPATH=src python scripts/neonatal_rt_traces.py [WORKERS]
"""

import sys

from neurolens.evaluation.realtime_replay import build_traces
from neurolens.evaluation.runner import default_config, load_items


def main() -> int:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    items = load_items("helsinki", ["dev"]) + load_items("helsinki", ["test"])
    tr = build_traces(items, default_config(), "data/physionet/trace_cache", progress=print,
                      workers=workers, rhythm=True)
    print("traces", len(tr), "with rhythm", sum("epoch_rhythm" in (w[2].extra or {}) for t in tr[:1] for w in t.windows[:1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
