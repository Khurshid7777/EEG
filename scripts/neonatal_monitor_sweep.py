"""Neonatal monitor over all Helsinki recordings (as `neurolens monitor --pma 40`).

Per recording: alarms per hour and by type, consensus seizures with a seizure alarm
(alarm window overlapping the seizure or starting within 60 s after it), seizure alarms
outside seizures, status alarm vs experts' status (ACNS neonatal criteria).

    PYTHONPATH=src python scripts/neonatal_monitor_sweep.py OUT.json [WORKERS]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _one(item):
    subject, path, seizures, split = item
    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer4_detect.status_criteria import status_epilepticus
    from neurolens.realtime.monitor import RealtimeMonitor
    from neurolens.realtime.stream import EdfReplaySource

    src = EdfReplaySource(Path(path))
    mon = RealtimeMonitor(patient=PatientInfo(postmenstrual_age_weeks=40))
    s = mon.run(src)
    by_type = {}
    for a in s.alarms:
        by_type[a.type.value] = by_type.get(a.type.value, 0) + 1
    sz_alarms = [(a.t_start, a.t_end) for a in s.alarms if a.type.value == "seizure"]
    hit = [any(t0 <= off + 60 and t1 >= on for t0, t1 in sz_alarms) for on, off in seizures]
    outside = [w for w in sz_alarms if not any(w[0] <= off + 60 and w[1] >= on for on, off in seizures)]
    return {"subject": subject, "split": split, "hours": s.duration_s / 3600, "n_alarms": len(s.alarms),
            "alarms_per_hour": s.alarms_per_hour, "by_type": by_type, "seizures": len(seizures),
            "seizures_alarmed": int(sum(hit)), "seizure_alarms_outside": len(outside),
            "status_alarm": by_type.get("status_epilepticus", 0) > 0,
            "expert_status": bool(status_epilepticus(seizures, True)["suspected"]),
            "within_budget": s.within_budget, "detector": s.seizure_detector}


def main() -> int:
    from neurolens.datasets.helsinki import HelsinkiClient

    c = HelsinkiClient()
    split = c.split()
    side = {s: k for k, v in split.items() for s in v}
    items = [(a.subject, str(c.local_path(a.file)), [(z.onset_s, z.offset_s) for z in a.seizures], side.get(a.subject))
             for a in c.all_annotations()]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, items))
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1) + "\n")
    for sp in ("dev", "test"):
        R = [r for r in rows if r["split"] == sp]
        print(sp, "seizures alarmed", sum(r["seizures_alarmed"] for r in R), "/", sum(r["seizures"] for r in R),
              "outside/h", round(sum(r["seizure_alarms_outside"] for r in R) / sum(r["hours"] for r in R), 2),
              "bs alarms", sum(r["by_type"].get("burst_suppression", 0) for r in R),
              "over budget", sum(not r["within_budget"] for r in R))
    return 0


if __name__ == "__main__":
    sys.exit(main())
