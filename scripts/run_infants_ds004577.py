"""Increment 58 (pre-registered, docs/preregistration_increment58.md): the routine-EEG verdict on healthy infants,
OpenNeuro ds004577 (CC0, 103 infants, first year of life). One recording per child (session 1); the frozen
production configuration is run unchanged. Each EDF is fetched, analysed and deleted. Resumable: one JSON line per record.

    PYTHONPATH=src python scripts/run_infants_ds004577.py check            # excluded record, reader/units only
    PYTHONPATH=src python scripts/run_infants_ds004577.py run OUT.jsonl [WORKERS]
    PYTHONPATH=src python scripts/run_infants_ds004577.py assess OUT.jsonl OUT.json
"""

import json
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BASE = "https://s3.amazonaws.com/openneuro.org/"
CHECK_SUBJECT = "sub-NORB00001"
SPEC_PASS, SPEC_FAIL = 0.72, 0.50
AGE_BANDS = ((0.0, 0.25), (0.25, 0.5), (0.5, 2.0))   # < 3 months, 3-6 months, >= 6 months


def manifest() -> list[dict]:
    keys, tok = [], None
    while True:
        u = BASE + "?list-type=2&prefix=ds004577/sub-&max-keys=1000" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
        t = urllib.request.urlopen(u, timeout=60).read().decode()
        keys += re.findall(r"<Key>([^<]*_eeg\.edf)</Key>", t)
        m = re.search(r"<NextContinuationToken>([^<]*)", t)
        if not m:
            break
        tok = m.group(1)
    rows = []
    for k in sorted(keys):
        m = re.match(r"ds004577/(sub-\w+)/(ses-\d+)/eeg/", k)
        sc = f"{BASE}ds004577/{m.group(1)}/{m.group(2)}/{m.group(1)}_{m.group(2)}_scans.tsv"
        age = float(urllib.request.urlopen(sc, timeout=60).read().decode().splitlines()[1].split("\t")[1])
        rows.append({"subject": m.group(1), "session": m.group(2), "key": k, "age_years": age})
    return rows


def test_set() -> list[dict]:
    return [r for r in manifest() if r["session"] == "ses-1" and r["subject"] != CHECK_SUBJECT]


def _one(r: dict) -> dict:
    from neurolens.evaluation.routine_eval import run_record

    base = {"record_id": r["subject"], "subject": r["subject"], "age_years": r["age_years"], "abnormal": False}
    try:
        with tempfile.TemporaryDirectory(prefix="infant_") as d:
            p = Path(d) / "rec.edf"
            for attempt in (0, 1):
                try:
                    p.write_bytes(urllib.request.urlopen(BASE + r["key"], timeout=300).read())
                    break
                except Exception:  # noqa: BLE001 - one retry, as pre-registered
                    if attempt:
                        raise
            row = run_record(p, SimpleNamespace(record_id=r["subject"], abnormal=False, age_years=r["age_years"]), scale=1.0)
        return {**row, **base}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {**base, "error": repr(exc)}


def _cp(k: int, n: int, alpha: float = 0.05) -> list[float]:
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def assess(path: Path) -> dict:
    from collections import Counter

    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    tn = sum(not r["impression_abnormal"] for r in ok)
    by_age = {}
    for lo, hi in AGE_BANDS:
        g = [r for r in ok if lo <= r["age_years"] < hi]
        if g:
            k = sum(not r["impression_abnormal"] for r in g)
            by_age[f"{lo:g}-{hi:g}"] = {"n": len(g), "specificity": round(k / len(g), 4), "ci95": _cp(k, len(g))}
    spec = tn / n
    det = Counter(c for r in ok if r["impression_abnormal"] for c in r.get("detectors_fired", []))
    return {"n_records": n, "errors": [(r["record_id"], r["error"]) for r in rows if "error" in r],
            "specificity": round(spec, 4), "ci95": _cp(tn, n), "false_abnormal": n - tn,
            "specificity_rules_only": round(sum(not r["rule_impression_abnormal"] for r in ok) / n, 4),
            "specificity_model_only": round(sum(not r["model_abnormal"] for r in ok if r.get("model_abnormal") is not None) / n, 4),
            "borderline_rate": round(float(np.mean([bool(r.get("record_borderline")) for r in ok])), 4),
            "detectors_in_false_abnormal": dict(det.most_common(8)), "by_age": by_age,
            "decision": "validated" if spec >= SPEC_PASS else ("caveat_under_1y" if spec >= SPEC_FAIL else "unvalidated_under_1y")}


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "assess":
        res = assess(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 0
    if cmd == "check":
        r = next(x for x in manifest() if x["subject"] == CHECK_SUBJECT and x["session"] == "ses-1")
        from neurolens.layer1_ingest.registry import ingest

        with tempfile.TemporaryDirectory(prefix="infant_") as d:
            p = Path(d) / "rec.edf"
            p.write_bytes(urllib.request.urlopen(BASE + r["key"], timeout=300).read())
            sig = ingest(p)
        print("channels", sig.channel_names, "fs", sig.sampling_rate_hz, "dur_s", round(sig.duration_s, 1))
        print("std uV per channel", np.round(sig.signal.std(axis=1), 1).tolist())
        res = _one(r)
        print({k: res.get(k) for k in ("impression_abnormal", "rule_impression_abnormal", "model_abnormal", "record_probability",
                                        "record_borderline", "detectors_fired", "error")})
        return 0
    out, workers = Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 4
    recs = test_set()
    done = {json.loads(ln)["subject"] for ln in out.read_text().splitlines() if ln.strip()} if out.exists() else set()
    todo = [r for r in recs if r["subject"] not in done]
    print(f"{len(recs)} records, {len(todo)} to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_one, todo, chunksize=2):
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("subject", "impression_abnormal", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
