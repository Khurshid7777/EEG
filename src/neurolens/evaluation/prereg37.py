"""Pre-registered check for increment 37 (docs/preregistration_increment37.md): the PDR must be
posterior dominant. Regression on the NMT evaluation split and on 111 healthy adults
(ds003775), each against the stored run of the previous configuration. Written before the runs.
"""

from __future__ import annotations

from typing import Any

MAX_LOSS = 0.02


def _nmt(rows: list[dict]) -> dict:
    nm = [r for r in rows if not r["abnormal"]]
    ab = [r for r in rows if r["abnormal"]]
    return {"specificity": sum(not r["impression_abnormal"] for r in nm) / len(nm),
            "sensitivity": sum(bool(r["impression_abnormal"]) for r in ab) / len(ab),
            "pdr_found_normal": sum((r.get("pdr_hz") or 0) > 0 for r in nm) / len(nm)}


def _spec(rows: list[dict]) -> float:
    ok = [r for r in rows if "error" not in r and r.get("record_probability") is not None]
    return sum(not r["impression_abnormal"] for r in ok) / len(ok)


def assess(nmt_old: list[dict], nmt_new: list[dict], ds_old: list[dict], ds_new: list[dict]) -> dict[str, Any]:
    a, b = _nmt(nmt_old), _nmt(nmt_new)
    so, sn = _spec(ds_old), _spec(ds_new)
    lost = sum(1 for r in nmt_new if not r["abnormal"] and (r.get("pdr_hz_unchecked") or 0) > 0 and not r["pdr_hz"])
    checks = {"nmt_specificity_loss": a["specificity"] - b["specificity"],
              "nmt_sensitivity_loss": a["sensitivity"] - b["sensitivity"],
              "ds003775_specificity_loss": so - sn}
    return {"nmt_before": a, "nmt_after": b, "nmt_normals_losing_pdr": lost,
            "ds003775_specificity_before": so, "ds003775_specificity_after": sn, "losses": checks,
            "decision": {"keep_pdr_posterior_dominance": all(round(v, 9) <= MAX_LOSS for v in checks.values())}}
