"""Demo pages of the interactive regional view built from open data (docs/demo/).

    PYTHONPATH=src python scripts/make_regional_demo.py
Pages: eegmat rest and task of one healthy volunteer; one Siena seizure (clinically right temporal focus; the view
does not state a side - increment 75). Only derived power summaries are stored, not the raw signal.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_siena_laterality as sl  # noqa: E402

from neurolens.contracts.signal import UnifiedSignal  # noqa: E402
from neurolens.layer1_ingest.registry import ingest  # noqa: E402
from neurolens.layer3_features.regional import STANDARD_1020  # noqa: E402
from neurolens.outputs.regional_view import write_regional_html  # noqa: E402
from neurolens.pipeline.pipeline import Pipeline  # noqa: E402

OUT = Path("docs/demo")


def _analyze(sig):
    return Pipeline(provider_pref="template", run_ica=False).analyze_signal(sig)


def eegmat(sub: str, suffix: str, title: str, name: str) -> None:
    import run_eegmat_eval as eg
    sig = ingest(eg._get(f"{sub}_{suffix}.edf"))
    out = _analyze(sig)
    write_regional_html(OUT / name, sig.signal, sig.channel_names, sig.sampling_rate_hz, out.result_json, title=title)


def siena(subject: str, fname: str, onset: float, title: str, name: str) -> None:
    t0 = onset - 110
    names, fs, x = sl.read_edf_segment(f"{sl.BASE}/{subject}/{fname}", t0, onset + 70)
    keep = [i for i, n in enumerate(names) if n in STANDARD_1020]
    sig = UnifiedSignal(signal=np.ascontiguousarray(x[keep]), sampling_rate_hz=fs, channel_names=[names[i] for i in keep],
                        duration_s=x.shape[1] / fs, source="edf")
    out = _analyze(sig)
    write_regional_html(OUT / name, sig.signal, sig.channel_names, fs, out.result_json, title=title, win_s=5.0)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    eegmat("Subject00", "1", "eegmat Subject00 — покой (здоровый доброволец)", "regional_eegmat_rest.html")
    eegmat("Subject00", "2", "eegmat Subject00 — устный счёт (здоровый доброволец)", "regional_eegmat_task.html")
    siena("PN00", "PN00-1.edf", 1143.0, "Siena PN00 — фрагмент вокруг приступа (110 с до, 70 с после)", "regional_siena_seizure.html")
    print(sorted(p.name for p in OUT.iterdir()))
