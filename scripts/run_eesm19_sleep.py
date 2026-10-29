"""Increment 50 (pre-registered, docs/preregistration_increment50.md): the AASM-montage stager on
OpenNeuro ds005185 (EESM19, Aarhus: 20 young adults x 4 home PSG nights; F3/F4/C3/C4/O1/O2, M1/M2,
EOG; two scorings per night). Product path (Pipeline._sleep_staging) with the frozen stager.
Resumable: one JSON line per night.

    PYTHONPATH=src python scripts/run_eesm19_sleep.py run OUT.jsonl
    PYTHONPATH=src python scripts/run_eesm19_sleep.py assess OUT.jsonl OUT.json
"""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BASE = "https://s3.amazonaws.com/openneuro.org/ds005185"
MODEL = "configs/models/sleep_stager_aasm_v2.json"
MODEL_SHA256 = "97d986b1c6a3e2f57e127591a05b8564aaf02872943ee7f280b5d4654c771d5a"
# scoring codes (code/BIDS/createBIDSs1.m): s1 Wake, s2 REM, s3 N1, s4 N2, s5 N3, s6-s8 A / artefact / unscored
CODE = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}                 # -> STAGES index (W N1 N2 N3 REM)
SEED, N_BOOT = 50, 2000


def _curl(url, out=None) -> bytes:
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "4"] + (["-o", str(out)] if out else []) + [url]
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def nights() -> list[tuple[str, str, str]]:
    rows = list(csv.DictReader(io.StringIO(_curl(f"{BASE}/participants.tsv").decode()), delimiter="\t"))
    return [(r["participant_id"], f"ses-00{k}", r["age"]) for r in rows for k in (1, 2, 3, 4)]


def _scoring(sub, ses, which) -> list[tuple[int, int]]:
    t = _curl(f"{BASE}/{sub}/{ses}/eeg/{sub}_{ses}_task-sleep_acq-scoring{which}_events.tsv").decode()
    out = []
    for r in csv.DictReader(io.StringIO(t), delimiter="\t"):
        col = next(c for c in r if c.lower().startswith("scoring"))
        v = r[col].strip()
        code = CODE.get(int(float(v))) if v not in ("", "n/a") else None
        out.append((int(np.floor(float(r["onset"]) / 30.0 + 0.5)), -1 if code is None else code))
    return out


def _one(item) -> dict:
    sub, ses, age = item
    from neurolens.contracts.signal import PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    key = f"{sub}_{ses}"
    try:
        s1, s2 = _scoring(sub, ses, 1), _scoring(sub, ses, 2)
        stem = f"{sub}_{ses}_task-sleep_acq-PSG_eeg"
        with tempfile.TemporaryDirectory(dir="data") as d:
            for ext in (".set", ".fdt"):
                _curl(f"{BASE}/{sub}/{ses}/eeg/{stem}{ext}", Path(d) / f"{stem}{ext}")
            sig = ingest(Path(d) / f"{stem}.set")
        age_y = None if age in (None, "", "n/a") else float(age)
        sig = sig.model_copy(update={"patient": PatientInfo(age_years=age_y)})
        pipe = Pipeline(provider_pref="template", run_ica=False)
        pipe.cfg.sleep = pipe.cfg.sleep.model_copy(update={"aasm_model": MODEL})
        st = pipe._sleep_staging(sig)
        if not st or st.get("skipped"):
            return {"night": key, "subject": sub, "error": f"not staged: {st}"}
        hyp = [["W", "N1", "N2", "N3", "REM"].index(s) for s in st["hypnogram"]]
        return {"night": key, "subject": sub, "age": age_y, "duration_h": round(sig.duration_s / 3600, 2),
                "derivations": st["derivations"], "eog_source": st.get("eog_source"), "hyp": hyp,
                "scoring1": s1, "scoring2": s2}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"night": key, "subject": sub, "error": repr(exc)}


def _load(p) -> list[dict]:
    return [json.loads(ln) for ln in Path(p).read_text().splitlines() if ln.strip()]


def _pairs(r, ref="scoring1", other=None):
    pred = r["hyp"] if other is None else dict(r[other])
    out = []
    for k, h in r[ref]:
        if h < 0:
            continue
        p = (pred[k] if 0 <= k < len(pred) else None) if other is None else pred.get(k, -1)
        if p is not None and p >= 0:
            out.append((h, p))
    return out


def _score(rows, ref="scoring1", other=None) -> dict:
    from neurolens.layer4_detect.sleep_staging import confusion, scores_from_confusion

    m = np.zeros((5, 5), int)
    for r in rows:
        pr = _pairs(r, ref, other)
        if pr:
            y, p = zip(*pr)
            m += confusion(np.array(y), np.array(p))
    return scores_from_confusion(m)


def assess(path: Path) -> dict:
    rows = _load(path)
    ok = [r for r in rows if "error" not in r]
    pooled = _score(ok)
    by_sub: dict[str, list] = {}
    for r in ok:
        by_sub.setdefault(r["subject"], []).append(r)
    subs = sorted(by_sub)
    rng = np.random.default_rng(SEED)
    boot = [_score([r for i in rng.integers(0, len(subs), len(subs)) for r in by_sub[subs[i]]])["kappa"]
            for _ in range(N_BOOT)]
    ci = [round(float(q), 3) for q in np.percentile(boot, [2.5, 97.5])]
    h1 = pooled["kappa"] >= 0.60 and ci[0] >= 0.50
    return {"n_nights": len(ok), "n_people": len(subs),
            "errors": [(r["night"], r["error"]) for r in rows if "error" in r],
            "derivations": sorted({r["derivations"] for r in ok}), "eog_source": sorted({str(r["eog_source"]) for r in ok}),
            "vs_scoring1": {k: pooled[k] for k in ("n_epochs", "kappa", "accuracy", "macro_f1", "f1", "confusion")},
            "kappa_ci95_by_person": ci,
            "vs_scoring2": {k: v for k, v in _score(ok, "scoring2").items() if k in ("kappa", "accuracy", "macro_f1")},
            "scorer2_vs_scorer1": {k: v for k, v in _score(ok, "scoring1", "scoring2").items()
                                   if k in ("kappa", "accuracy", "macro_f1")},
            "H1": bool(h1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assess"])
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    out = Path(a.paths[0])
    if a.cmd == "assess":
        res = assess(out)
        Path(a.paths[1]).write_text(json.dumps(res, indent=1, default=float) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "vs_scoring1"} | {"kappa": res["vs_scoring1"]["kappa"]},
                         indent=1, default=float))
        return 0
    if MODEL_SHA256 is None or hashlib.sha256(Path(MODEL).read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError(f"{MODEL} does not match the frozen sha256")
    items = nights()
    done = {r["night"] for r in _load(out)} if out.exists() else set()
    todo = [it for it in items if f"{it[0]}_{it[1]}" not in done]
    print(f"{len(items)} nights, {len(todo)} to run", flush=True)
    with out.open("a") as f:
        for it in todo:                                   # one at a time: a night is ~1.8 GB on disk
            r = _one(it)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print({k: r.get(k) for k in ("night", "derivations", "error")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
