"""Increment 30 (development): is the neonatal seizure model's false-alarm rate on Cork
(increment 28: 10.8 seizure alarms/h in seizure-free hours, vs ~2.5/h on Helsinki) a
property of the recording set-up? Helsinki seizure-free recordings replayed through the
monitor with Cork's montage (9 electrodes: F3 F4 C3 C4 T3 T4 O1 O2 Cz) and/or Cork's
200 Hz sampling rate.

    PYTHONPATH=src python scripts/neonatal_device_perturb.py OUT.json [WORKERS]
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

CORK_CHANNELS = ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2", "Cz"]
CONDITIONS = {"original": (False, None), "cork9": (True, None), "fs200": (False, 200.0), "cork9_fs200": (True, 200.0)}


def _one(item):
    subject, path, cond = item
    import numpy as np
    from scipy.signal import resample_poly

    from neurolens.contracts.signal import PatientInfo
    from neurolens.realtime.monitor import RealtimeMonitor
    from neurolens.realtime.stream import EdfReplaySource

    subset, fs_new = CONDITIONS[cond]
    src = EdfReplaySource(Path(path))
    if subset:
        idx = [src.channel_names.index(c) for c in CORK_CHANNELS]
        src._data = src._data[idx]
        src.channel_names = list(CORK_CHANNELS)
    if fs_new:
        up, down = int(fs_new), int(round(src.sampling_rate_hz))
        g = np.gcd(up, down)
        src._data = resample_poly(src._data, up // g, down // g, axis=1).astype(np.float32)
        src.sampling_rate_hz = fs_new
        src.chunk_samples = int(round(5.0 * fs_new))
    s = RealtimeMonitor(patient=PatientInfo(postmenstrual_age_weeks=40)).run(src)
    n_sz = sum(a.type.value == "seizure" for a in s.alarms)
    return {"subject": subject, "condition": cond, "hours": s.duration_s / 3600, "seizure_alarms": n_sz,
            "status": any(a.type.value == "status_epilepticus" for a in s.alarms)}


def main() -> int:
    from neurolens.datasets.helsinki import HelsinkiClient

    c = HelsinkiClient()
    free = [(a.subject, str(c.local_path(a.file))) for a in c.all_annotations() if not a.seizures]
    items = [(s, p, k) for k in CONDITIONS for s, p in free]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, items))
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=1) + "\n")
    for k in CONDITIONS:
        R = [r for r in rows if r["condition"] == k]
        print(k, len(R), "seizure alarms/h", round(sum(r["seizure_alarms"] for r in R) / sum(r["hours"] for r in R), 2),
              "records with alarms", sum(r["seizure_alarms"] > 0 for r in R), "status", sum(r["status"] for r in R))
    return 0


if __name__ == "__main__":
    sys.exit(main())
