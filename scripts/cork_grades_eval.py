"""Increment 28 (pre-registered, docs/preregistration_increment28.md): neonatal background
alarms on the Cork HIE grading dataset (Zenodo 7477575; 169 one-hour EEGs of 53 term
neonates with hypoxic-ischaemic encephalopathy, consensus grade 1-4 per hour).

Per one-hour file:
* monitor (as ``neurolens monitor --pma 40``): alarms by type; a *background alarm* is a
  ``burst_suppression`` or ``suppression`` alarm;
* offline pipeline (as the routine harness: template provider, no ICA): the suppression
  group codes and the neonatal background event (aEEG category, abnormal for age).

    PYTHONPATH=src python scripts/cork_grades_eval.py DATA_DIR OUT.json [WORKERS]

DATA_DIR holds eeg_grades.csv, metadata.csv and edf/**/IDxx_epochy.edf.
"""

import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PMA_WEEKS = 40.0  # all neonates term; per-baby age not in the dataset
BACKGROUND_ALARMS = ("burst_suppression", "suppression")


def _one(item):
    try:
        return _run(item)
    except Exception as exc:  # noqa: BLE001 - unreadable files are listed and excluded (pre-registration §3)
        return {"file_id": item[0], "error": repr(exc)}


def _run(item):
    file_id, path = item
    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline
    from neurolens.realtime.monitor import RealtimeMonitor
    from neurolens.realtime.stream import EdfReplaySource

    patient = PatientInfo(postmenstrual_age_weeks=PMA_WEEKS)
    s = RealtimeMonitor(patient=patient).run(EdfReplaySource(Path(path)))
    by_type = {}
    for a in s.alarms:
        by_type[a.type.value] = by_type.get(a.type.value, 0) + 1
    out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(ingest(Path(path), patient=patient))
    ev = out.detection.events
    bg = next((e for e in ev if e.code == "neonatal_background"), None)
    return {
        "file_id": file_id, "hours": s.duration_s / 3600, "alarms_by_type": by_type,
        "background_alarms": sum(by_type.get(k, 0) for k in BACKGROUND_ALARMS),
        "detector": s.seizure_detector,
        "offline_suppression_codes": sorted({e.code for e in ev if e.group == "suppression"}),
        "offline_aeeg_category": next((x.note for x in bg.evidence if x.feature == "aeeg_category"), None) if bg else None,
        "offline_abnormal_for_age": bool(bg.metadata.get("abnormal_for_age")) if bg else None,
        "offline_ictal_events": sum(1 for e in ev if e.group == "ictal"),
    }


def main() -> int:
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    meta = {r["file_ID"]: r for r in csv.DictReader(open(root / "metadata.csv"))}
    paths = {p.stem: p for p in root.glob("edf/**/*.edf")}
    items = [(f, str(paths[f])) for f in sorted(meta) if f in paths]
    missing = sorted(set(meta) - set(paths))
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, items))
    failed = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    for r in rows:
        m = meta[r["file_id"]]
        r.update(baby_id=m["baby_ID"], grade=int(m["grade"]), seizures_yn=m["seizures_YN"])
    out.write_text(json.dumps({"missing": missing, "failed": failed, "records": rows}, indent=1) + "\n")
    from neurolens.evaluation.prereg28 import evaluate

    print(json.dumps(evaluate(rows), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
