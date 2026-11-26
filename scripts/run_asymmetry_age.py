"""Increment 81 (pre-registered, docs/preregistration_increment81.md): are age-dependent asymmetry norms needed?

    PYTHONPATH=src python scripts/run_asymmetry_age.py OUT.json
"""

import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import QuantileRegressor

from neurolens.layer3_features.regional import fit_asymmetry_norms


def _rows(name):
    return [r for r in (json.loads(l) for l in Path(f"docs/increment73_{name}.jsonl").read_text().splitlines()) if "error" not in r]


def main() -> int:
    from neurolens.datasets.ds004504 import Ds003775Client, Ds004504Client

    age75 = {r.subject: r.age_years for r in Ds003775Client().records()}
    age04 = {r.subject: r.age_years for r in Ds004504Client().records()}
    import urllib.request
    base = "https://physionet-open.s3.amazonaws.com/eegmat/1.0.0"
    ageeg = {r["Subject"]: float(r["Age"]) for r in csv.DictReader(io.StringIO(urllib.request.urlopen(f"{base}/subject-info.csv", timeout=60).read().decode()))}
    train = [(age75[r["subject"]], r["ai_alpha"]) for r in _rows("ds003775") if age75.get(r["subject"]) is not None]
    held = [(age04[r["subject"]], r["ai_alpha"], "ds004504") for r in _rows("ds004504_controls") if age04.get(r["subject"]) is not None] + \
           [(ageeg[r["subject"]], r["ai_alpha"], "eegmat") for r in _rows("eegmat_rest")]
    pairs = sorted(train[0][1])
    ages = np.array([a for a, _ in train])
    res = {"n_train": len(train), "train_age_range": [float(ages.min()), float(ages.max())],
           "held_age_range": {k: [float(min(a for a, _, s in held if s == k)), float(max(a for a, _, s in held if s == k))] for k in ("ds004504", "eegmat")}}
    tests = []
    for p in pairs:
        ai = np.array([r[p] for _, r in train])
        for kind, y in (("signed", ai), ("abs", np.abs(ai))):
            rho, pv = spearmanr(ages, y)
            tests.append({"pair": p, "kind": kind, "rho": round(float(rho), 3), "p": float(pv)})
    # Holm correction
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    m = len(tests)
    sig_prev = True
    for rank, i in enumerate(order):
        thr = 0.05 / (m - rank)
        sig_prev = sig_prev and tests[i]["p"] < thr
        tests[i]["holm_threshold"] = round(thr, 5)
        tests[i]["significant"] = bool(sig_prev)
    res["step1"] = tests
    sigpairs = sorted({t["pair"] for t in tests if t["significant"]})
    res["step1_significant_pairs"] = sigpairs
    res["age_norms_needed"] = bool(sigpairs)
    norms = fit_asymmetry_norms([r for _, r in train])
    pooled = {p: float(np.mean([(r[p] < norms[p][0] or r[p] > norms[p][1]) for _, r, _ in held])) for p in pairs}
    res["pooled_limits_out_fraction_held_out"] = {p: round(v, 3) for p, v in pooled.items()}
    res["pooled_out_fraction_by_set"] = {k: {p: round(float(np.mean([(r[p] < norms[p][0] or r[p] > norms[p][1]) for _, r, s in held if s == k])), 3) for p in pairs} for k in ("ds004504", "eegmat")}
    if sigpairs:
        res["step2"] = {}
        for p in sigpairs:
            X = ages.reshape(-1, 1)
            y = np.array([r[p] for _, r in train])
            lo = QuantileRegressor(quantile=0.025, alpha=0.0, solver="highs").fit(X, y)
            hi = QuantileRegressor(quantile=0.975, alpha=0.0, solver="highs").fit(X, y)
            out = np.mean([(r[p] < lo.predict([[a]])[0] or r[p] > hi.predict([[a]])[0]) for a, r, _ in held])
            res["step2"][p] = {"age_out_fraction": round(float(out), 3), "pooled_out_fraction": round(pooled[p], 3),
                               "adopt": bool(abs(out - 0.05) < abs(pooled[p] - 0.05) - 0.02 and out <= 0.12)}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "step1"}, indent=1))
    print("min p:", sorted(tests, key=lambda t: t["p"])[:3])
    return 0


if __name__ == "__main__":
    sys.exit(main())
