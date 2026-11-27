"""Increment 82 (pre-registered, docs/preregistration_increment82.md): criteria C1-C4 for the per-channel robust rule.

    PYTHONPATH=src python scripts/run_regional_robust_assess.py OUT.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_regional_controls as rc  # noqa: E402


def main() -> int:
    d = "data/inc82"
    paths = [f"{d}/ds003775.jsonl", f"{d}/ds004504.jsonl", f"{d}/eegmat_1.jsonl", f"{d}/eegmat_2.jsonl"]
    res = rc.assess(paths)
    rows = lambda p: [r for r in (json.loads(l) for l in Path(p).read_text().splitlines()) if "error" not in r]  # noqa: E731
    n75 = len(rows(paths[0]))
    limits = json.loads(Path("configs/regional_norms_v1.json").read_text())["limits"]
    held = rows(paths[1]) + rows(paths[2])
    frac = {p: float(np.mean([(r["ai_alpha"][p] < lo or r["ai_alpha"][p] > hi) for r in held if p in r["ai_alpha"]])) for p, (lo, hi) in limits.items()}
    out = {"n_usable_ds003775": n75, "n_usable_ds004504": len(rows(paths[1])), "errors": res["errors_excluded"],
           "C1_coverage": {"value": n75, "criterion": ">= 105", "ok": bool(n75 >= 105)},
           "C2_H1": {"ds003775": res["H1"]["ds003775"], "ds004504_controls": res["H1"]["ds004504_controls"],
                     "ok": bool(all(v["ok"] for v in res["H1"].values()))},
           "C3_H2": {"frontal_theta_up": res["H2"]["frontal_theta_up"], "posterior_alpha_down": res["H2"]["posterior_alpha_down"],
                     "n_pairs": res["H2"]["n_pairs"], "ok": bool(all(res["H2"][k]["ok"] for k in ("frontal_theta_up", "posterior_alpha_down")))},
           "C4_asymmetry_limits": {"n_held_out": len(held), "out_fraction_by_pair": {k: round(v, 3) for k, v in frac.items()},
                                   "mean": round(float(np.mean(list(frac.values()))), 4), "max": round(max(frac.values()), 3),
                                   "ok": bool(np.mean(list(frac.values())) <= 0.10 and max(frac.values()) <= 0.20)},
           "H3_descriptive_median_abs_AI": res["H3"]}
    out["adopt_robust_rule"] = bool(out["C1_coverage"]["ok"] and out["C2_H1"]["ok"] and out["C3_H2"]["ok"] and out["C4_asymmetry_limits"]["ok"])
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
