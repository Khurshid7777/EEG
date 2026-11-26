"""Interactive regional view: diagnostic process and regional cortical map on one synchronised screen (increment 76).

Two linked layers: (1) the diagnostic timeline (detector events, sleep stages if available, artefacts, regional band-power
strips) and (2) a scalp map of the selected window (lobes x hemispheres, electrode interpolation) with hemispheric
asymmetry against normative limits. Scalp mapping from the 19 electrodes of the 10-20 system - **not** source
localisation, **not** a diagnosis, and it never states the side of a seizure focus (docs/increment75_report.md).
The output is one self-contained HTML file (no external libraries or network access).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import yaml

from ..layer3_features.regional import (HOMOLOGOUS_PAIRS, REGIONS, regional_series)

_ROOT = Path(__file__).resolve().parents[3]
CAVEAT = {
    "ru": "Скальповое картирование по электродам 10-20 (доли и полушария), не локализация источников и не диагноз. "
          "Сторона очага не определяется (инкремент 75). Нормативный коридор асимметрии применим к записи целиком.",
    "uz": "10-20 elektrodlari bo‘yicha bosh terisi xaritasi (bo‘laklar va yarimsharlar), manba lokalizatsiyasi va tashxis emas. "
          "O‘choq tomoni aniqlanmaydi (75-bosqich). Assimetriya me’yor koridori butun yozuvga tegishli.",
}
EVENT_LABELS_RU = {
    "spike": "Спайк / острая волна", "ictal_ml": "Иктальный паттерн (ML)", "ictal": "Иктальный паттерн",
    "focal_slowing": "Фокальное замедление", "diffuse_slowing": "Диффузное замедление", "wicket": "Wicket-ритм",
    "burst_suppression": "Вспышка-подавление", "gped": "ГПД", "lpd": "ЛПД", "periodic": "Периодические разряды",
}


def _coords() -> dict[str, list[float]]:
    return {k: list(v) for k, v in yaml.safe_load((_ROOT / "configs/electrodes/coords_10_20.yaml").read_text())["positions"].items()
            if k in {c for r in REGIONS.values() for c in r}}


def load_norms() -> dict:
    return json.loads((_ROOT / "configs/regional_norms_v1.json").read_text())


def _record_asymmetry(series: dict, norms: Mapping) -> dict:
    """Record-level AI = ln(mean P_right / mean P_left) over artefact-free windows, with the normative flag."""
    names = series["channels"]
    ok = [i for i, a in enumerate(series["artefact"]) if not a]
    out = {}
    if len(ok) < 3:
        return out
    for left, right in HOMOLOGOUS_PAIRS:
        if left not in names or right not in names:
            continue
        pl = np.mean([10 ** series["log10_abs_alpha"][names.index(left)][i] for i in ok])
        pr = np.mean([10 ** series["log10_abs_alpha"][names.index(right)][i] for i in ok])
        ai = float(np.log(pr / pl))
        lo, hi = norms["limits"].get(f"{left}-{right}", [None, None])
        flag = None if lo is None else ("right>left" if ai > hi else ("left>right" if ai < lo else None))
        out[f"{left}-{right}"] = {"ai": round(ai, 3), "low": lo, "high": hi, "flag": flag}
    return out


def _event_regions(channels: list[str]) -> list[dict]:
    out = []
    for c in channels:
        for region, chans in REGIONS.items():
            if c in chans:
                side = "mid" if c.endswith("z") else ("L" if int(c[-1]) % 2 else "R")
                out.append({"ch": c, "region": region, "side": side})
    return out


def build_view(signal: np.ndarray, channels: list[str], fs: float, result_json: Optional[Mapping] = None,
               win_s: float = 10.0, norms: Optional[Mapping] = None, title: str = "") -> dict:
    result_json = result_json or {}
    norms = norms or load_norms()
    series = regional_series(signal, channels, fs, win_s=win_s)
    labels = {}
    for n in (result_json.get("causal_graph") or {}).get("nodes", []):
        if n.get("event_code") and n.get("label_ru"):
            labels.setdefault(n["event_code"], n["label_ru"])
    events = []
    for e in (result_json.get("events") or [])[:3000]:
        events.append({"code": e["code"], "label": labels.get(e["code"], EVENT_LABELS_RU.get(e["code"], e["code"])),
                       "t0": round(float(e["t_start"]), 2), "t1": round(float(e["t_end"]), 2),
                       "conf": round(float(e.get("confidence") or 0), 2), "channels": e.get("channels") or [],
                       "where": _event_regions(e.get("channels") or [])})
    sleep = result_json.get("sleep_staging") or {}
    rep = result_json.get("report") or {}
    imp = (rep.get("impression") or {})
    return {
        "version": "regional_view_v1", "title": title or "Региональная карта коры", "caveat": CAVEAT,
        "record": {"duration_s": series["duration_s"], "n_channels": len(series["channels"]), "win_s": series["win_s"],
                   "impression_ru": imp.get("ru"), "impression_uz": imp.get("uz"),
                   "critical": [str(c.get("code", c)) if isinstance(c, dict) else str(c) for c in (result_json.get("critical_findings") or [])]},
        "series": series, "coords": _coords(), "regions": {k: list(v) for k, v in REGIONS.items()},
        "pairs": [f"{l}-{r}" for l, r in HOMOLOGOUS_PAIRS], "asymmetry_record": _record_asymmetry(series, norms),
        "norms": {"version": norms["version"], "fit": norms["fit"], "checked_on": norms["checked_on"]},
        "events": events,
        "hypnogram": sleep.get("hypnogram") if isinstance(sleep, dict) else None,
    }


def render_html(view: Mapping) -> str:
    tpl = (Path(__file__).with_name("regional_view.html")).read_text(encoding="utf-8")
    data = json.dumps(view, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return tpl.replace("/*__DATA__*/null", data)


def write_regional_html(path: str | Path, signal: np.ndarray, channels: list[str], fs: float,
                        result_json: Optional[Mapping] = None, **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(build_view(signal, channels, fs, result_json, **kw)), encoding="utf-8")
    return p
