"""Increment 74 (pre-registered, docs/preregistration_increment74.md): normative asymmetry limits from ds003775,
checked on held-out healthy sets. Uses the per-record AI values of increment 73.

    PYTHONPATH=src python scripts/run_asymmetry_norms.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from neurolens.layer3_features.regional import fit_asymmetry_norms, flag_asymmetry


def _load(name):
    return [r for r in (json.loads(l) for l in Path(f"docs/increment73_{name}.jsonl").read_text().splitlines()) if "error" not in r]


def main() -> int:
    train = _load("ds003775")
    norms = fit_asymmetry_norms([r["ai_alpha"] for r in train])
    held = {"ds004504_controls": _load("ds004504_controls"), "eegmat_rest": _load("eegmat_rest")}
    pooled = [r for rows in held.values() for r in rows]
    pairs = list(norms)

    def rates(rows):
        flags = [flag_asymmetry(r["ai_alpha"], norms) for r in rows]
        return ({p: sum(p in f for f in flags) / len(rows) for p in pairs}, sum(bool(f) for f in flags) / len(rows))

    per_pair, any_flag = rates(pooled)
    res = {"n_train": len(train), "limits": {p: [round(a, 3), round(b, 3)] for p, (a, b) in norms.items()},
           "held_out": {k: {"n": len(v), "per_pair_out_fraction": {p: round(x, 3) for p, x in rates(v)[0].items()},
                            "any_pair_flagged": round(rates(v)[1], 3)} for k, v in held.items()},
           "pooled": {"n": len(pooled), "per_pair_out_fraction": {p: round(x, 3) for p, x in per_pair.items()},
                      "mean_out_fraction": round(float(np.mean(list(per_pair.values()))), 4), "any_pair_flagged": round(any_flag, 3)}}
    ages = [(r.get("age_years")) for r in train]
    res["H1"] = bool(res["pooled"]["mean_out_fraction"] <= 0.10 and max(per_pair.values()) <= 0.20)
    res["H2_any_flag_le_0.30"] = bool(any_flag <= 0.30)
    # descriptive: age dependence (ds003775 ages from participants.tsv)
    try:
        from neurolens.datasets.ds004504 import Ds003775Client
        age = {r.subject: r.age_years for r in Ds003775Client().records()}
        res["age_spearman_abs_ai"] = {p: round(float(spearmanr([age[r["subject"]] for r in train if age.get(r["subject"]) is not None and p in r["ai_alpha"]],
                                                              [abs(r["ai_alpha"][p]) for r in train if age.get(r["subject"]) is not None and p in r["ai_alpha"]])[0]), 3) for p in pairs}
    except Exception as exc:  # noqa: BLE001
        res["age_spearman_abs_ai"] = f"not computed: {exc!r}"
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
