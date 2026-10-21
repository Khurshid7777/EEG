"""Pre-registered analysis for increment 39 (docs/preregistration_increment39.md): the frozen
seizure detectors (increment 11 defaults) on AUBMC (Beirut; 6 children, 20 long-term video-EEG
recordings, 30 seizures with an EEG correlate). Written before any EEG signal was downloaded.

False alarms exclude detections overlapping the ``ambiguous`` intervals (seizures without a
surface-EEG change, and a seizure listed under the neighbouring file).
"""

from __future__ import annotations

from typing import Any

from .prereg import _subject, summarize
from .prereg10 import macro_sensitivity
from .prereg12 import fa_excluding_ambiguous

MONITOR_SENS_MIN = 0.50     # increment 11 test: 0.70
MONITOR_FA_MAX = 5.0        # increment 11 test: 3.30 / h
OFFLINE_SENS_MIN = 0.70     # increment 11 test: 0.88
OFFLINE_FA_MAX = 12.0       # increment 11 test: 7.93 / h
DOMINANCE_MARGIN = 0.05


def assess(metrics: dict, ambiguous: dict[str, list[tuple[float, float]]]) -> dict[str, Any]:
    res = {r["mode"]: r for r in metrics["results"]}
    tab = {m: summarize(r["records"])["all"] for m, r in res.items()}
    fa = {m: fa_excluding_ambiguous(r["records"], ambiguous) for m, r in res.items()}
    rt, off = "realtime-ml-A", "offline-ml-A"
    h = {"H1_monitor_sensitivity_at_least_0.50": tab[rt]["sensitivity"] >= MONITOR_SENS_MIN,
         "H2_monitor_fa_per_h_at_most_5": fa[rt] <= MONITOR_FA_MAX,
         "H3_offline_sensitivity_0.70_at_fa_12": tab[off]["sensitivity"] >= OFFLINE_SENS_MIN and fa[off] <= OFFLINE_FA_MAX}

    def dominated(path: str) -> bool:
        ml, thr = f"{path}-ml-A", f"{path}-threshold-default"
        return (tab[thr]["sensitivity"] > tab[ml]["sensitivity"] + DOMINANCE_MARGIN) and fa[thr] < fa[ml]

    return {"overall": tab, "fa_per_hour_excluding_ambiguous": fa,
            "macro_sensitivity": {m: macro_sensitivity(r["records"]) for m, r in res.items()},
            "by_subject": {m: summarize(r["records"], _subject) for m, r in res.items()},
            "hypotheses": h,
            "decision": {f"{p}_default": "threshold" if dominated(p) else "learned_v2" for p in ("offline", "realtime")}}
