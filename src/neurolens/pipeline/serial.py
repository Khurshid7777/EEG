"""Serial comparison of two recordings of the same patient (D76): what changed since the prior EEG.

Both recordings are summarised by the same pipeline (:func:`record_summary.summarize`). Five
quantities are compared - posterior dominant rhythm (difference, Hz), relative delta / theta /
alpha power and robust amplitude (log ratio). A change is flagged when it exceeds the
day-to-day variability of healthy people: ``z * RMS`` of the same quantity between repeat
recordings (``configs/serial_limits.yaml``, increment 42, OpenNeuro ds004148). Detectors that
fired in only one of the recordings are listed without a limit.

v2 (increment 43): the band shares are measured on the most alert epochs only - the quarter of
2-s epochs with the highest posterior alpha share (``alert_rel_*`` in the record summary) - since
drowsiness at a repeat recording changed the whole-record shares as much as eye opening did
(increment 42). The quantities compared are those listed in the limits file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

from ..contracts.report import Bilingual

QUANTITIES = ("pdr_hz", "global_rel_delta", "global_rel_theta", "global_rel_alpha", "robust_amplitude_uv")
QUANTITIES_V2 = ("pdr_hz", "alert_rel_delta", "alert_rel_theta", "alert_rel_alpha", "robust_amplitude_uv")
Z = 2.576
LIMITS_PATH = Path(__file__).resolve().parents[3] / "configs" / "serial_limits.yaml"

_NAMES = {
    "pdr_hz": ("основной ритм", "asosiy ritm"),
    "global_rel_delta": ("доля дельта", "delta ulushi"),
    "global_rel_theta": ("доля тета", "teta ulushi"),
    "global_rel_alpha": ("доля альфа", "alfa ulushi"),
    "robust_amplitude_uv": ("амплитуда", "amplituda"),
    "alert_rel_delta": ("доля дельта (бодрствование)", "delta ulushi (bedorlik)"),
    "alert_rel_theta": ("доля тета (бодрствование)", "teta ulushi (bedorlik)"),
    "alert_rel_alpha": ("доля альфа (бодрствование)", "alfa ulushi (bedorlik)"),
}


def change(q: str, a, b) -> float | None:
    """Change from A (prior) to B (current): Hz for the PDR, ln(B/A) otherwise; None when not
    comparable (PDR not found in either recording, non-positive value)."""
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if q == "pdr_hz":
        return b - a if a > 0 and b > 0 else None
    return math.log(b / a) if a > 0 and b > 0 else None


def derive_limits(pairs: list[tuple[dict, dict]], z: float = Z, quantities=QUANTITIES) -> dict:
    out = {}
    for q in quantities:
        d = np.array([c for a, b in pairs if (c := change(q, a.get(q), b.get(q))) is not None])
        out[q] = {"limit": round(float(z * np.sqrt(np.mean(d ** 2))), 4) if d.size else None,
                  "n_pairs": int(d.size), "mean": round(float(d.mean()), 4) if d.size else None,
                  "sd": round(float(d.std(ddof=1)), 4) if d.size > 1 else None}
    return out


def load_limits(path: str | Path = LIMITS_PATH) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text())


def compare(prior: dict, current: dict, limits: dict | None = None) -> dict:
    """``prior`` / ``current``: record summaries. ``limits``: ``serial_limits.yaml`` content
    (its ``limits`` mapping is used); None -> values side by side, no flags."""
    lim = (limits or {}).get("limits", {})
    rows = []
    for q in (tuple(lim) or QUANTITIES):
        a, b = prior.get(q), current.get(q)
        c = change(q, a, b)
        l_ = (lim.get(q) or {}).get("limit")
        rows.append({"quantity": q, "prior": a, "current": b,
                     "change": round(c, 4) if c is not None else None,
                     "unit": "Hz" if q == "pdr_hz" else "ln_ratio", "limit": l_,
                     "exceeds": bool(c is not None and l_ is not None and abs(c) > l_)})
    pa, pc = set(prior.get("detectors_fired") or []), set(current.get("detectors_fired") or [])
    return {"quantities": rows, "detectors_new": sorted(pc - pa), "detectors_gone": sorted(pa - pc),
            "limits_source": (limits or {}).get("source"),
            "any_exceeds": any(r["exceeds"] for r in rows)}


def compare_text(cmp: dict) -> Bilingual:
    ru, uz = [], []
    for r in cmp["quantities"]:
        n_ru, n_uz = _NAMES[r["quantity"]]
        if r["change"] is None:
            continue
        if r["quantity"] == "pdr_hz":
            v_ru = f"{n_ru} {r['prior']:g} → {r['current']:g} Гц"
            v_uz = f"{n_uz} {r['prior']:g} → {r['current']:g} Gs"
        else:
            pct = round(100 * (math.exp(r["change"]) - 1))
            v_ru = f"{n_ru} {pct:+d}%"
            v_uz = f"{n_uz} {pct:+d}%"
        if r["exceeds"]:
            v_ru += " (больше повседневной изменчивости)"
            v_uz += " (kundalik o‘zgaruvchanlikdan katta)"
        ru.append(v_ru)
        uz.append(v_uz)
    if cmp["detectors_new"]:
        ru.append("появились: " + ", ".join(cmp["detectors_new"]))
        uz.append("paydo bo‘ldi: " + ", ".join(cmp["detectors_new"]))
    if cmp["detectors_gone"]:
        ru.append("исчезли: " + ", ".join(cmp["detectors_gone"]))
        uz.append("yo‘qoldi: " + ", ".join(cmp["detectors_gone"]))
    head_ru = "По сравнению с прошлой записью: " if ru else "Сравнимых величин нет."
    head_uz = "Oldingi yozuvga nisbatan: " if uz else "Taqqoslanadigan qiymatlar yo‘q."
    # increment 42: the limits held on stable healthy people (3% flagged) but caught eye opening as
    # an alpha decrease in only 10%: they flag large changes only - say so in every comparison
    note_ru = (" Пределы — изменчивость здоровых между повторными записями; у пациентов она может быть больше."
               " Пределы широкие: отмечаются только крупные изменения, отсутствие отметки не исключает"
               " умеренного изменения — сравните записи визуально."
               if cmp.get("limits_source") else " Пределы изменчивости не заданы: только значения.")
    note_uz = (" Chegaralar — sog‘lomlarda takroriy yozuvlar orasidagi o‘zgaruvchanlik; bemorlarda kattaroq bo‘lishi mumkin."
               " Chegaralar keng: faqat katta o‘zgarishlar belgilanadi, belgi yo‘qligi o‘rtacha o‘zgarishni"
               " istisno qilmaydi — yozuvlarni vizual solishtiring."
               if cmp.get("limits_source") else " O‘zgaruvchanlik chegaralari berilmagan: faqat qiymatlar.")
    return Bilingual(ru=head_ru + "; ".join(ru) + ("." if ru else "") + note_ru,
                     uz=head_uz + "; ".join(uz) + ("." if uz else "") + note_uz)
