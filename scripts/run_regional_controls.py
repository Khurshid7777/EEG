"""Increment 73 (pre-registered, docs/preregistration_increment73.md): regional scalp map, physiological controls.

    PYTHONPATH=src python scripts/run_regional_controls.py run SET OUT.jsonl [--workers 3]   # SET: ds003775 | ds004504 | eegmat_1 | eegmat_2
    PYTHONPATH=src python scripts/run_regional_controls.py assess OUT.json ds003775.jsonl ds004504.jsonl eegmat_1.jsonl eegmat_2.jsonl
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


HP = None   # set by --highpass (post-hoc sensitivity run only)


def _summarise(sig) -> dict:
    from neurolens.layer3_features import regional as R

    pc = R.channel_band_power(sig.signal, sig.channel_names, sig.sampling_rate_hz, highpass_hz=HP)
    return {"n_channels": len(pc), "duration_s": float(sig.duration_s),
            "alpha": R.region_values(pc, "alpha"), "theta": R.region_values(pc, "theta"),
            "posterior_alpha": R.posterior_alpha(pc),
            "ai_alpha": {f"{l}-{r}": R.asymmetry_index(pc, l, r) for l, r in R.HOMOLOGOUS_PAIRS if l in pc and r in pc}}


def _one(item):
    which, key = item
    from neurolens.layer1_ingest.registry import ingest

    try:
        if which == "ds003775":
            from neurolens.datasets.ds004504 import Ds003775Client
            c = Ds003775Client()
            rec = next(r for r in c.records() if r.subject == key)
            return {"subject": key, **_summarise(ingest(c.fetch(rec)))}
        if which == "ds004504":
            from neurolens.datasets.ds004504 import Ds004504Client
            c = Ds004504Client()
            rec = next(r for r in c.records() if r.subject == key)
            return {"subject": key, **_summarise(ingest(c.fetch(rec)))}
        import run_eegmat_eval as eg
        s = next(x for x in eg.subjects() if x["subject"] == key)
        return {"subject": key, **_summarise(ingest(eg._get(f"{key}_{which[-1]}.edf")))}
    except Exception as exc:  # noqa: BLE001 - listed and excluded, as pre-registered
        return {"subject": key, "error": repr(exc)}


def _keys(which: str) -> list[str]:
    if which == "ds003775":
        from neurolens.datasets.ds004504 import Ds003775Client
        return [r.subject for r in Ds003775Client().records()]
    if which == "ds004504":
        from neurolens.datasets.ds004504 import Ds004504Client
        return [r.subject for r in Ds004504Client().records() if r.group == "C"]
    import run_eegmat_eval as eg
    return [s["subject"] for s in eg.subjects()]


def _ci(k: int, n: int) -> list[float]:
    from neurolens.evaluation.prereg import clopper_pearson
    return [round(float(x), 3) for x in clopper_pearson(k, n)]


def assess(paths) -> dict:
    from scipy.stats import binomtest

    load = lambda p: [r for r in (json.loads(l) for l in Path(p).read_text().splitlines()) if "error" not in r]  # noqa: E731
    d3, d4, r1, r2 = (load(p) for p in paths)
    res = {"errors_excluded": {p: sum("error" in json.loads(l) for l in Path(p).read_text().splitlines()) for p in paths}}
    res["H1"] = {}
    for tag, rows in (("ds003775", d3), ("ds004504_controls", d4)):
        top = [max(r["alpha"], key=r["alpha"].get) for r in rows]
        post = sum(t in ("occipital", "parietal") for t in top)
        res["H1"][tag] = {"n": len(rows), "posterior_top": post / len(rows), "ci95": _ci(post, len(rows)),
                          "occipital_top": sum(t == "occipital" for t in top) / len(rows),
                          "ok": bool(post / len(rows) >= 0.90)}
    # H2: paired by subject
    s1, s2 = {r["subject"]: r for r in r1}, {r["subject"]: r for r in r2}
    both = sorted(set(s1) & set(s2))
    th = [s2[s]["theta"]["frontal"] - s1[s]["theta"]["frontal"] for s in both]
    al = [s2[s]["posterior_alpha"] - s1[s]["posterior_alpha"] for s in both]
    up, down = sum(x > 0 for x in th), sum(x < 0 for x in al)
    res["H2"] = {"n_pairs": len(both),
                 "frontal_theta_up": {"k": up, "frac": up / len(both), "ci95": _ci(up, len(both)),
                                      "p": float(binomtest(up, len(both), 0.5, alternative="greater").pvalue)},
                 "posterior_alpha_down": {"k": down, "frac": down / len(both), "ci95": _ci(down, len(both)),
                                          "p": float(binomtest(down, len(both), 0.5, alternative="greater").pvalue)}}
    for k in ("frontal_theta_up", "posterior_alpha_down"):
        h = res["H2"][k]
        h["ok"] = bool(h["frac"] > 0.5 and h["p"] < 0.05)
    res["H3"] = {}
    for tag, rows in (("ds003775", d3), ("ds004504_controls", d4)):
        med = {pair: float(np.median([abs(r["ai_alpha"][pair]) for r in rows if pair in r["ai_alpha"]]))
               for pair in rows[0]["ai_alpha"]}
        res["H3"][tag] = {"median_abs_ai": {k: round(v, 3) for k, v in med.items()}, "O1-O2_ok": bool(med["O1-O2"] < 0.20)}
    res["decision"] = {"H1": all(v["ok"] for v in res["H1"].values()), "H2": all(res["H2"][k]["ok"] for k in
                       ("frontal_theta_up", "posterior_alpha_down")), "H3": all(v["O1-O2_ok"] for v in res["H3"].values())}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "assess"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--highpass", type=float, default=None)
    a = ap.parse_args()
    global HP
    HP = a.highpass
    if a.cmd == "assess":
        res = assess(a.args[1:])
        Path(a.args[0]).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        return 0
    which, out = a.args[0], Path(a.args[1])
    done = {json.loads(l)["subject"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [(which, k) for k in _keys("eegmat" if which.startswith("eegmat") else which) if k not in done]
    print(f"{len(todo)} to run", flush=True)
    with out.open("a") as f, ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_one, todo):
            f.write(json.dumps(r, default=float) + "\n"); f.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
