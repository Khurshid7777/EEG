"""Interactive regional view: diagnostic process and regional cortical map on one synchronised screen (increment 76).

Two linked layers: (1) the diagnostic timeline (detector events, sleep stages if available, artefacts, regional band-power
strips) and (2) a scalp map of the selected window (lobes x hemispheres, electrode interpolation) with hemispheric
asymmetry against normative limits. Scalp mapping from the 19 electrodes of the 10-20 system - **not** source
localisation, **not** a diagnosis, and it never states the side of a seizure focus (docs/increment75_report.md).
The output is one self-contained HTML file (no external libraries or network access).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from ..layer3_features.regional import HOMOLOGOUS_PAIRS, REGIONS, canonical, regional_series

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
    car = series.get("reference") == "car"          # the normative corridor was fitted on common-average data only
    for left, right in HOMOLOGOUS_PAIRS:
        if left not in names or right not in names:
            continue
        pl = np.mean([10 ** series["log10_abs_alpha"][names.index(left)][i] for i in ok])
        pr = np.mean([10 ** series["log10_abs_alpha"][names.index(right)][i] for i in ok])
        ai = float(np.log(pr / pl))
        lo, hi = norms["limits"].get(f"{left}-{right}", [None, None]) if car else [None, None]
        flag = None if lo is None else ("right>left" if ai > hi else ("left>right" if ai < lo else None))
        out[f"{left}-{right}"] = {"ai": round(ai, 3), "low": lo, "high": hi, "flag": flag}
    return out


def _event_regions(channels: list[str]) -> list[dict]:
    out = []
    for c in (canonical(x) for x in channels):
        for region, chans in REGIONS.items():
            if c in chans:
                side = "mid" if c.endswith("z") else ("L" if int(c[-1]) % 2 else "R")
                out.append({"ch": c, "region": region, "side": side})
    return out


def build_view(signal: np.ndarray, channels: list[str], fs: float, result_json: Mapping | None = None,
               win_s: float = 10.0, norms: Mapping | None = None, title: str = "") -> dict:
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
                   "reference": series["reference"],
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
                        result_json: Mapping | None = None, **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(build_view(signal, channels, fs, result_json, **kw)), encoding="utf-8")
    return p


def regional_summary(view: Mapping) -> dict:
    """Compact record-level summary for ``result_json['regional']`` (no time series)."""
    s = view["series"]
    names = s["channels"]
    ok = [i for i, a in enumerate(s["artefact"]) if not a] or list(range(s["n_windows"]))
    bands = {}
    for b, rows in s["rel"].items():
        chm = {c: float(np.mean([rows[names.index(c)][i] for i in ok])) for c in names}
        bands[b] = {r: round(float(np.mean([chm[c] for c in chans if c in chm])), 4)
                    for r, chans in REGIONS.items() if any(c in chm for c in chans)}
    counts: dict[str, int] = {}
    for e in view["events"]:
        for w in e["where"]:
            key = f"{w['region']}_{w['side']}"
            counts[key] = counts.get(key, 0) + 1
    return {"version": view["version"], "level": "scalp regions from the available 10-20 electrodes; not source localisation",
            "reference": s["reference"], "n_electrodes": len(names), "window_s": s["win_s"], "n_windows": s["n_windows"], "artefact_window_fraction": round(float(np.mean(s["artefact"])), 3),
            "relative_power_by_region": bands, "asymmetry_alpha": view["asymmetry_record"], "norms": view["norms"]["version"],
            "events_by_region": counts, "events_without_channels": sum(1 for e in view["events"] if not e["where"]),
            "limitations": CAVEAT["ru"]}


def plot_regional_static(view: Mapping, out_path: str | Path) -> Path:
    """Static figure for the PDF: record-mean band maps, asymmetry with normative corridor, timeline with events."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .topomap import plot_topomap

    s = view["series"]
    names = s["channels"]
    ok = [i for i, a in enumerate(s["artefact"]) if not a] or list(range(s["n_windows"]))
    coords = {k: tuple(v) for k, v in view["coords"].items()}
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.15, 1.0], hspace=0.45, wspace=0.15)
    for k, (b, ru) in enumerate((("delta", "дельта"), ("theta", "тета"), ("alpha", "альфа"), ("beta", "бета"))):
        ax = fig.add_subplot(gs[0, k])
        vals = {c: float(np.mean([s["rel"][b][names.index(c)][i] for i in ok])) for c in names}
        plot_topomap(ax, vals, coords, title=f"{ru}, отн. мощность (вся запись)", cmap="viridis")
    # asymmetry
    ax = fig.add_subplot(gs[1, :])
    pairs = view["pairs"]
    for j, p in enumerate(pairs):
        rec = view["asymmetry_record"].get(p)
        if not rec:
            continue
        ax.barh(j, rec["high"] - rec["low"], left=rec["low"], color="#b9c8dc", height=0.6, zorder=1)
        ax.plot(rec["ai"], j, marker="D", color="#c2410c" if rec["flag"] else "#1b5fd1", zorder=3)
    ax.axvline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-2, 2)
    ax.set_xlabel("ln(P правое / P левое), альфа;  ← левое сильнее | правое сильнее →;  коридор — нормативные пределы здоровых", fontsize=8)
    ax.set_title("Межполушарная асимметрия альфа-ритма (вся запись; ромб — значение, полоса — норма 2.5–97.5%)", fontsize=9)
    # timeline: events + regional alpha strip
    ax = fig.add_subplot(gs[2, :])
    cells = [(r, side) for r in REGIONS for side in ("L", "R")]
    mat = []
    for r, side in cells:
        ch = [c for c in REGIONS[r] if c in names and (c.endswith("z") or (int(c[-1]) % 2 == (1 if side == "L" else 0)))]
        mat.append(np.mean([s["rel"]["alpha"][names.index(c)] for c in ch], axis=0) if ch else np.zeros(s["n_windows"]))
    ax.imshow(np.array(mat), aspect="auto", cmap="viridis", extent=[0, s["duration_s"], len(cells), 0], interpolation="nearest")
    ax.set_yticks(np.arange(len(cells)) + 0.5)
    ru = {"frontal": "лоб.", "central": "центр.", "temporal": "вис.", "parietal": "тем.", "occipital": "зат."}
    ax.set_yticklabels([f"{ru[r]} {sd}" for r, sd in cells], fontsize=7)
    for e in view["events"][:400]:
        ax.plot([e["t0"], e["t0"]], [-0.2, -0.9], color="#d1342f", lw=1.2, clip_on=False)
    for k, a in enumerate(s["artefact"]):
        if a:
            ax.axvspan(k * s["win_s"], (k + 1) * s["win_s"], color="grey", alpha=0.45, lw=0)
    ax.set_xlabel("время, с (красные метки сверху — события детекторов; серое — окна с артефактами)", fontsize=8)
    ax.set_title("Процесс по времени: относительная альфа по областям (строки: лобная…затылочная × левое/правое)", fontsize=9)
    fig.suptitle("Региональная карта коры (скальповая, 19 электродов; не диагноз, сторона очага не определяется)", fontsize=11)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
