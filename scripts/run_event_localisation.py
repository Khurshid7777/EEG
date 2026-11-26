"""Increment 77 (pre-registered, docs/preregistration_increment77.md): signal-based localisation of expert-marked events
on the NMT-Events test split. Resumable: one JSON line per event; each record's EDF is deleted after processing.

    PYTHONPATH=src python scripts/run_event_localisation.py run OUT.jsonl
    PYTHONPATH=src python scripts/run_event_localisation.py assess OUT.jsonl OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

CACHE = "data/nmt_events"


def run(out: Path) -> None:
    from neurolens.datasets.nmt_events import EPILEPTIFORM, SLOWING, NMTEventsClient
    from neurolens.layer1_ingest.electrodes import normalize_channel_name
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.layer3_features.event_localiser import EventLocaliser
    from neurolens.layer3_features.regional import STANDARD_1020

    c = NMTEventsClient(CACHE)
    anns = c.annotations()
    done = {json.loads(l)["record"] for l in out.read_text().splitlines()} if out.exists() else set()
    with out.open("a") as f:
        for rid, normal, member in c.split_members("test"):
            if normal or rid in done:
                continue
            ev = [e for e in anns[rid].events if e.category in (EPILEPTIFORM, SLOWING)]
            if not ev:
                continue
            try:
                p = c.fetch(member)
                sig = ingest(p)
                loc = EventLocaliser(sig.signal, list(sig.channel_names), sig.sampling_rate_hz)
                rows = []
                for e in ev:
                    exp = sorted({normalize_channel_name(ch) for ch in e.channels} & set(STANDARD_1020))
                    row = {"record": rid, "onset": e.onset_s, "offset": e.offset_s, "category": e.category,
                           "comment": e.comment, "expert": exp}
                    if not exp:
                        row["error"] = "no standard expert channel"
                    else:
                        try:
                            for v in ("A", "B", "C"):
                                row[v] = loc.locate(e.onset_s, e.offset_s, v)
                        except Exception as exc:  # noqa: BLE001 - excluded and counted, as pre-registered
                            row["error"] = repr(exc)
                    rows.append(row)
                f.write("".join(json.dumps(r) + "\n" for r in rows) + json.dumps({"record": rid, "marker": "done", "n": len(rows)}) + "\n")
            except Exception as exc:  # noqa: BLE001
                f.write(json.dumps({"record": rid, "marker": "error", "error": repr(exc)}) + "\n")
            f.flush()
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass
            print(rid, flush=True)


def _hemi(expert):
    from neurolens.layer3_features.event_localiser import side_of
    s = {side_of(c) for c in expert if side_of(c) != "mid"}
    return next(iter(s)) if len(s) == 1 else None


def assess(path: Path) -> dict:
    from neurolens.layer3_features.event_localiser import region_of, side_of

    rows = [json.loads(l) for l in path.read_text().splitlines()]
    ev = [r for r in rows if "marker" not in r]
    ok = [r for r in ev if "error" not in r]
    res = {"n_events": len(ev), "n_excluded": len(ev) - len(ok), "records": sorted({r["record"] for r in ok}),
           "record_errors": [r for r in rows if r.get("marker") == "error"]}
    rng = np.random.default_rng(77)
    recs = res["records"]

    def metrics(subset, v):
        out = {"n": len(subset)}
        if not subset:
            return out
        out["M1_top1_in_expert"] = float(np.mean([r[v]["top"][0] in r["expert"] for r in subset]))
        out["M1_top3_hit"] = float(np.mean([bool(set(r[v]["top"]) & set(r["expert"])) for r in subset]))
        out["baseline_random_channel"] = float(np.mean([len(r["expert"]) / 19 for r in subset]))
        reg = lambda r: {region_of(c) for c in r["expert"]}  # noqa: E731
        out["M2_region_in_expert"] = float(np.mean([r[v]["region"] in reg(r) for r in subset]))
        out["baseline_random_region"] = float(np.mean([np.mean([region_of(c) in reg(r) for c in
                                                                 ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz", "C3", "C4", "Cz", "T3", "T4", "T5", "T6", "P3", "P4", "Pz", "O1", "O2"]])
                                                      for r in subset]))
        allreg = [region_of(c) for r in subset for c in r["expert"]]
        top = max(set(allreg), key=allreg.count)
        out["baseline_most_frequent_region"] = float(np.mean([top in reg(r) for r in subset]))
        uni = [r for r in subset if _hemi(r["expert"])]
        out["n_unilateral"] = len(uni)
        out["M3_hemisphere_unilateral"] = float(np.mean([side_of(r[v]["top"][0]) == _hemi(r["expert"]) for r in uni])) if uni else None
        return out

    def boot(subset, v, key):
        # per-event indicators computed once; bootstrap resamples records
        ind = {"M1_top1_in_expert": lambda r: r[v]["top"][0] in r["expert"],
               "M2_region_in_expert": lambda r: r[v]["region"] in {region_of(c) for c in r["expert"]},
               "M3_hemisphere_unilateral": lambda r: (side_of(r[v]["top"][0]) == _hemi(r["expert"])) if _hemi(r["expert"]) else None}[key]
        by = {rid: [x for x in (ind(r) for r in subset if r["record"] == rid) if x is not None] for rid in recs}
        vals = []
        for _ in range(2000):
            pick = rng.choice(recs, len(recs))
            s = [x for rid in pick for x in by[rid]]
            if s:
                vals.append(float(np.mean(s)))
        return [round(float(np.percentile(vals, q)), 3) for q in (2.5, 97.5)]

    for cat in ("epileptiform", "slowing"):
        sub = [r for r in ok if r["category"] == cat]
        res[cat] = {v: metrics(sub, v) for v in ("A", "B", "C")}
        if cat == "epileptiform":
            for key in ("M1_top1_in_expert", "M2_region_in_expert", "M3_hemisphere_unilateral"):
                res[cat]["A"][key + "_ci95_by_record"] = boot(sub, "A", key)
    a = res["epileptiform"]["A"]
    res["H1"] = bool(a["M3_hemisphere_unilateral"] >= 0.75 and a["M3_hemisphere_unilateral_ci95_by_record"][0] > 0.60)
    res["H2"] = bool(a["M2_region_in_expert"] >= 0.60 and a["M2_region_in_expert"] - a["baseline_random_region"] >= 0.15)
    res["H3"] = bool(a["M1_top1_in_expert"] >= 0.40 and a["M1_top1_in_expert"] >= 2 * a["baseline_random_channel"])
    return res


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]))
    else:
        r = assess(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(r, indent=1) + "\n")
        print(json.dumps(r, indent=1))
