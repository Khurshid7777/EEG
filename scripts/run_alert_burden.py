"""Increment 83 (descriptive): burden of critical / candidate findings and events in healthy adults at the default settings.
Datasets: ds003775 (111), ds004504 controls (29), eegmat rest (36). No tuning; documents the current default behaviour.

    PYTHONPATH=src python scripts/run_alert_burden.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_alert_burden.py assess OUT.jsonl OUT.json
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _items():
    import run_eegmat_eval as eg
    from neurolens.datasets.ds004504 import Ds003775Client, Ds004504Client
    out = [("ds003775", r.subject) for r in Ds003775Client().records()]
    out += [("ds004504", r.subject) for r in Ds004504Client().records() if r.group == "C"]
    out += [("eegmat", s["subject"]) for s in eg.subjects()]
    return out


def _one(item):
    which, key = item
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline
    try:
        if which == "ds003775":
            from neurolens.datasets.ds004504 import Ds003775Client
            c = Ds003775Client(); sig = ingest(c.fetch(next(r for r in c.records() if r.subject == key)))
        elif which == "ds004504":
            from neurolens.datasets.ds004504 import Ds004504Client
            c = Ds004504Client(); sig = ingest(c.fetch(next(r for r in c.records() if r.subject == key)))
        else:
            import run_eegmat_eval as eg
            sig = ingest(eg._get(f"{key}_1.edf"))
        out = Pipeline(provider_pref="deterministic", run_ica=False).analyze_signal(sig)
        r = out.result_json
        ev = r.get("events") or []
        codes = {}
        for e in ev:
            codes[e["code"]] = codes.get(e["code"], 0) + 1
        crit = [str(c.get("code", c)) if isinstance(c, dict) else str(c) for c in (r.get("critical_findings") or [])]
        rs = r.get("record_score") or {}
        return {"set": which, "subject": key, "duration_s": float(sig.duration_s), "events_by_code": codes, "critical": crit,
                "n_events": len(ev), "max_conf_by_code": {c: max(e["confidence"] for e in ev if e["code"] == c) for c in codes},
                "record_abnormal": rs.get("abnormal"), "record_prob": rs.get("probability"), "borderline": rs.get("borderline"),
                "regional_ok": bool(r.get("regional", {}).get("version"))}
    except Exception as exc:  # noqa: BLE001
        return {"set": which, "subject": key, "error": repr(exc)}


def run(out, workers):
    out = Path(out)
    done = {(json.loads(l)["set"], json.loads(l)["subject"]) for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [i for i in _items() if i not in done]
    print(len(todo), "to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r) + "\n"); f.flush()


def assess(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    res = {"n": len(rows), "errors": len(rows) - len(ok), "by_set": {}}
    for s in sorted({r["set"] for r in ok}) + ["all"]:
        sub = [r for r in ok if s == "all" or r["set"] == s]
        hours = sum(r["duration_s"] for r in sub) / 3600
        codes = sorted({c for r in sub for c in r["events_by_code"]})
        crit_codes = sorted({c for r in sub for c in r["critical"]})
        res["by_set"][s] = {
            "n": len(sub), "total_hours": round(hours, 2), "median_minutes": round(float(np.median([r["duration_s"] for r in sub])) / 60, 1),
            "records_with_any_critical": round(float(np.mean([bool(r["critical"]) for r in sub])), 3),
            "records_with_critical_by_code": {c: round(float(np.mean([c in r["critical"] for r in sub])), 3) for c in crit_codes},
            "records_with_event_by_code": {c: round(float(np.mean([c in r["events_by_code"] for r in sub])), 3) for c in codes},
            "events_per_hour_by_code": {c: round(sum(r["events_by_code"].get(c, 0) for r in sub) / hours, 1) for c in codes},
            "record_abnormal_fraction": round(float(np.mean([bool(r["record_abnormal"]) for r in sub])), 3),
            "borderline_fraction": round(float(np.mean([bool(r["borderline"]) for r in sub])), 3),
            "median_max_ictal_ml_conf": round(float(np.median([r["max_conf_by_code"].get("ictal_ml", 0) for r in sub])), 3)}
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
    else:
        r = assess(sys.argv[2]); Path(sys.argv[3]).write_text(json.dumps(r, indent=1) + "\n"); print(json.dumps(r, indent=1))
