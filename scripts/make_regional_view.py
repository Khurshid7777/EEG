"""Build the interactive regional view (HTML) for one EDF file.

    PYTHONPATH=src python scripts/make_regional_view.py INPUT.edf OUT.html [--title TEXT]
"""

import argparse

from neurolens.layer1_ingest.registry import ingest
from neurolens.outputs.regional_view import write_regional_html
from neurolens.pipeline.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("edf")
    ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--win", type=float, default=10.0)
    a = ap.parse_args()
    sig = ingest(a.edf)
    out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(sig)
    p = write_regional_html(a.out, sig.signal, sig.channel_names, sig.sampling_rate_hz, out.result_json,
                            win_s=a.win, title=a.title)
    print(p, p.stat().st_size // 1024, "KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
