"""Increment 56 (pre-registered, docs/preregistration_increment56.md): specificity of the routine-EEG verdict on
the 997 NMT normal records that no earlier increment has opened (docs/increment56_nmt_fresh_ids.json, frozen
before the first run). The current production configuration is run unchanged (record model + age-band
thresholds + rules, `run_record` of the earlier routine-EEG increments). Each EDF is fetched from the NMT
zip by HTTP range requests, analysed and deleted (small disk). Resumable: one JSON line per record.

    PYTHONPATH=src python scripts/run_nmt_fresh.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_nmt_fresh.py assess OUT.jsonl OUT.json
"""

import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

IDS = Path("docs/increment56_nmt_fresh_ids.json")
SPEC_PASS, SPEC_FAIL = 0.72, 0.65
AGE_BANDS = ((0.0, 2.0), (2.0, 12.0), (12.0, 18.0), (18.0, 60.0), (60.0, 200.0))


_STATE: dict = {}


def _client():
    """One NMT client (and its zip directory) per worker process; only the EDF is deleted after each record."""
    import atexit

    if "c" not in _STATE:
        from neurolens.datasets.nmt import NMTClient

        _STATE["dir"] = tempfile.mkdtemp(prefix="nmtfresh_")
        atexit.register(shutil.rmtree, _STATE["dir"], True)
        c = NMTClient(_STATE["dir"])
        _STATE["c"], _STATE["recs"] = c, {r.record_id: r for r in c.labels()}
    return _STATE["c"], _STATE["recs"]


def _one(rid: int) -> dict:
    from neurolens.evaluation.routine_eval import run_record

    path = None
    try:
        c, recs = _client()
        rec = recs[rid]
        if rec.abnormal:
            return {"record_id": rid, "error": "abnormal record in the fresh list"}
        path = c.fetch(c.member_of(rec))
        row = run_record(path, rec)
        row["age_years"] = rec.age_years
        return row
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"record_id": rid, "error": repr(exc)}
    finally:
        if path is not None:
            Path(path).unlink(missing_ok=True)


def _cp(k: int, n: int, alpha: float = 0.05) -> list[float]:
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def assess(path: Path) -> dict:
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    tn = sum(not r["impression_abnormal"] for r in ok)
    spec = tn / n
    by_age = {}
    for lo, hi in AGE_BANDS:
        g = [r for r in ok if r.get("age_years") is not None and lo <= r["age_years"] < hi]
        if g:
            k = sum(not r["impression_abnormal"] for r in g)
            by_age[f"{lo:g}-{hi:g}"] = {"n": len(g), "specificity": round(k / len(g), 4), "ci95": _cp(k, len(g))}
    rule_tn = sum(not r["rule_impression_abnormal"] for r in ok)
    model_tn = sum(not r["model_abnormal"] for r in ok if r.get("model_abnormal") is not None)
    ci = _cp(tn, n)
    decision = "validated" if spec >= SPEC_PASS else ("below_target_kept" if spec >= SPEC_FAIL else "unvalidate")
    return {"n_records": n, "errors": [(r["record_id"], r["error"]) for r in rows if "error" in r],
            "specificity": round(spec, 4), "ci95": ci, "false_abnormal": n - tn,
            "specificity_rules_only": round(rule_tn / n, 4), "specificity_model_only": round(model_tn / n, 4),
            "borderline_rate": round(float(np.mean([bool(r.get("record_borderline")) for r in ok])), 4),
            "by_age_band": by_age, "decision": decision}


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "assess":
        res = assess(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 0
    out, workers = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ids = json.loads(IDS.read_text())["record_ids"]
    done = {json.loads(ln)["record_id"] for ln in out.read_text().splitlines() if ln.strip()} if out.exists() else set()
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} records, {len(todo)} to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, todo, chunksize=2):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("record_id", "impression_abnormal", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
