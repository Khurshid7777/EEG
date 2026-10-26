"""Intake check of a scanned paper EEG page against the scanning regulation
(docs/scanning_regulation.md) - before any attempt to digitize it.

The page image and a filled-in form (:class:`ScanForm`: paper speed, sensitivity,
channel order, calibration pulse, filters, page order) are checked; every requirement
gives a result (ok / warning / reject) with the measured value and the limit. Derived
quantities say what the scan can resolve at all: time and amplitude per pixel and the
highest frequency the trace can carry. A page with any "reject" must be rescanned.

The project does not decode traces from images yet; this module fixes what an image must
satisfy for a decoder to be trustworthy, and :func:`gate_reasons` closes autonomous mode
for any digitized signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

MM_PER_INCH = 25.4
LOSSLESS = {"PNG", "TIFF"}
# Untrusted uploads: only these decoders are tried, and an image larger than an A3 page at
# 1200 dpi is refused before decoding (decompression bombs; D85)
DECODERS = ("PNG", "TIFF", "JPEG")
MAX_PIXELS = 300_000_000
# regulation limits (docs/scanning_regulation.md §3)
MIN_DPI = 300
RECOMMENDED_DPI = 600
MAX_MS_PER_PX = 4.0          # >= 250 effective samples/s along the time axis
MAX_UV_PER_PX = 2.0          # amplitude resolution
MAX_SKEW_DEG = 0.5
DARK_FRACTION = (0.003, 0.40)
MAX_SATURATED_FRACTION = 0.02


class ScanForm(BaseModel):
    """What the technician records for every scanned page (regulation §2)."""

    model_config = ConfigDict(extra="forbid")

    paper_speed_mm_s: float = Field(gt=0, description="Paper speed, typically 15 / 30 / 60 mm/s.")
    sensitivity_uv_per_mm: float = Field(gt=0, description="Gain, e.g. 7 or 10 µV/mm.")
    channel_labels: list[str] = Field(min_length=1, description="Traces top to bottom, as printed.")
    montage: str = Field(description="As printed, e.g. 'longitudinal bipolar', 'referential A1A2'.")
    calibration_pulse_uv: Optional[float] = Field(
        default=None, gt=0, description="Square calibration pulse printed on the page (µV).")
    highpass_hz: Optional[float] = None
    lowpass_hz: Optional[float] = None
    notch_hz: Optional[float] = None
    page_index: int = Field(ge=1)
    n_pages: int = Field(ge=1)
    page_overlap_s: Optional[float] = Field(default=None, ge=0)
    scanner: Literal["flatbed", "sheetfed", "camera"] = "flatbed"


@dataclass
class Check:
    name: str
    status: Literal["ok", "warning", "reject"]
    value: object
    limit: str
    message_ru: str


@dataclass
class ScanReport:
    path: str
    checks: list[Check] = field(default_factory=list)
    derived: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return not any(c.status == "reject" for c in self.checks)

    def to_dict(self) -> dict:
        return {"path": self.path, "accepted": self.accepted, "derived": self.derived,
                "checks": [c.__dict__ for c in self.checks]}


def estimate_skew_deg(gray: np.ndarray, max_deg: float = 3.0, step: float = 0.1) -> float:
    """Skew of the horizontal structure (grid lines, baselines) by the projection-profile
    method: the angle whose row profile of dark pixels is sharpest."""
    from scipy.ndimage import rotate, zoom

    g = gray.astype(np.float32)
    if max(g.shape) > 1200:  # speed: the angle does not depend on resolution
        f = 1200.0 / max(g.shape)
        g = zoom(g, f, order=1)
    dark = (g < np.percentile(g, 20)).astype(np.float32)
    best, best_score = 0.0, -1.0
    for a in np.arange(-max_deg, max_deg + 1e-9, step):
        prof = rotate(dark, a, reshape=False, order=0, mode="constant").sum(axis=1)
        score = float(np.var(prof))
        if score > best_score:
            best, best_score = float(a), score
    return -best


def check_scan(path: str | Path, form: ScanForm) -> ScanReport:
    from PIL import Image

    from .electrodes import normalize_channel_name

    path = Path(path)
    rep = ScanReport(str(path))
    add = lambda *a: rep.checks.append(Check(*a))  # noqa: E731
    try:
        with Image.open(path, formats=DECODERS) as im:
            fmt, mode, (w, h) = im.format, im.mode, im.size
            if w * h > MAX_PIXELS:
                raise ValueError(f"{w}x{h} px exceeds {MAX_PIXELS} px")
            dpi = im.info.get("dpi")
            gray = np.asarray(im.convert("L"))
    except Exception as exc:  # noqa: BLE001 - an unreadable or unsafe image is the finding
        add("format", "reject", path.suffix or "-", "PNG или TIFF (также принимается JPEG для отказа с объяснением)",
            f"изображение не читается или слишком велико ({type(exc).__name__}): сохраните скан как PNG/TIFF "
            "по регламенту сканирования")
        return rep

    # 1. format and colour depth
    add("format", "ok" if fmt in LOSSLESS else "reject", fmt, "PNG или TIFF без сжатия с потерями",
        "формат без потерь" if fmt in LOSSLESS else "сжатие с потерями (JPEG и т. п.) искажает тонкие линии кривых")
    bad_mode = mode in ("1", "P")
    add("colour_depth", "reject" if bad_mode else "ok", mode, "оттенки серого 8 бит или цвет 24 бит",
        "1-битное/палитровое изображение теряет тонкие участки кривых" if bad_mode else "глубина цвета достаточна")

    # 2. resolution
    dpi_x = float(dpi[0]) if dpi else None
    if dpi_x is None:
        add("dpi", "reject", None, f">= {MIN_DPI} dpi в метаданных", "разрешение не записано в файл — масштаб неизвестен")
    else:
        st = "ok" if dpi_x >= RECOMMENDED_DPI else ("warning" if dpi_x >= MIN_DPI else "reject")
        add("dpi", st, round(dpi_x), f">= {MIN_DPI} (рекомендуется {RECOMMENDED_DPI})",
            {"ok": "разрешение достаточное", "warning": "минимально допустимое разрешение",
             "reject": "разрешение ниже минимума"}[st])

    # 3. what the scan can resolve (needs dpi)
    if dpi_x:
        px_per_mm = dpi_x / MM_PER_INCH
        ms_per_px = 1000.0 / (form.paper_speed_mm_s * px_per_mm)
        uv_per_px = form.sensitivity_uv_per_mm / px_per_mm
        rep.derived = {"px_per_mm": round(px_per_mm, 2), "ms_per_px": round(ms_per_px, 3),
                       "uv_per_px": round(uv_per_px, 3), "max_frequency_hz": round(500.0 / ms_per_px, 1),
                       "page_seconds": round(w / px_per_mm / form.paper_speed_mm_s, 2)}
        need = MM_PER_INCH * 1000.0 / (MAX_MS_PER_PX * form.paper_speed_mm_s)
        add("time_resolution", "ok" if ms_per_px <= MAX_MS_PER_PX else "reject", round(ms_per_px, 2),
            f"<= {MAX_MS_PER_PX} мс/пиксель (при {form.paper_speed_mm_s:g} мм/с — >= {need:.0f} dpi)",
            "временное разрешение достаточно" if ms_per_px <= MAX_MS_PER_PX
            else "при этой скорости бумаги разрешения не хватает для спайков и бета-активности")
        add("amplitude_resolution", "ok" if uv_per_px <= MAX_UV_PER_PX else "reject", round(uv_per_px, 2),
            f"<= {MAX_UV_PER_PX} мкВ/пиксель", "амплитудное разрешение достаточно" if uv_per_px <= MAX_UV_PER_PX
            else "амплитудное разрешение недостаточно")

    # 4. geometry and exposure
    skew = estimate_skew_deg(gray)
    add("skew", "ok" if abs(skew) <= MAX_SKEW_DEG else "reject", round(skew, 2), f"|наклон| <= {MAX_SKEW_DEG}°",
        "лист ровный" if abs(skew) <= MAX_SKEW_DEG else "лист перекошен — временная ось искажена; пересканировать ровно")
    dark = float((gray < 96).mean())
    st = "ok" if DARK_FRACTION[0] <= dark <= DARK_FRACTION[1] else "reject"
    add("exposure", st, round(dark, 4), f"доля тёмных пикселей {DARK_FRACTION[0]}–{DARK_FRACTION[1]}",
        "экспозиция нормальная" if st == "ok" else "кривые не видны (пересвет) или фон залит (недосвет)")
    sat = float((gray <= 2).mean())
    add("black_clipping", "ok" if sat <= MAX_SATURATED_FRACTION else "warning", round(sat, 4),
        f"<= {MAX_SATURATED_FRACTION}", "нет залитых чёрных областей" if sat <= MAX_SATURATED_FRACTION
        else "крупные залитые области (штамп, тень, сгиб?) — проверить")
    border = np.concatenate([gray[:3].ravel(), gray[-3:].ravel(), gray[:, :3].ravel(), gray[:, -3:].ravel()])
    cut = float((border < 96).mean())
    add("margins", "ok" if cut <= 0.05 else "warning", round(cut, 3), "кривые не касаются краёв",
        "поля сохранены" if cut <= 0.05 else "кривые касаются края изображения — лист мог быть обрезан")

    # 5. the form
    labels = [normalize_channel_name(x.split("-")[0]) for x in form.channel_labels]
    add("channel_labels", "ok" if len(labels) >= 4 else "reject", len(labels), ">= 4 подписанных отведений",
        "подписи отведений указаны" if len(labels) >= 4 else "меньше 4 отведений — анализ невозможен")
    add("calibration_pulse", "ok" if form.calibration_pulse_uv else "reject", form.calibration_pulse_uv,
        "калибровочный сигнал на листе", "калибровка есть — масштаб амплитуды проверяем"
        if form.calibration_pulse_uv else "нет калибровочного сигнала — амплитуда не проверяема")
    add("scanner", "ok" if form.scanner != "camera" else "reject", form.scanner, "планшетный или протяжный сканер",
        "сканер" if form.scanner != "camera" else "фото камерой даёт перспективные искажения — не допускается")
    add("filters", "ok" if form.highpass_hz is not None and form.lowpass_hz is not None else "warning",
        [form.highpass_hz, form.lowpass_hz, form.notch_hz], "фильтры записи указаны",
        "фильтры указаны" if form.highpass_hz is not None and form.lowpass_hz is not None
        else "фильтры не указаны — частотные выводы ограничены")
    if form.n_pages > 1:
        add("page_overlap", "ok" if form.page_overlap_s is not None else "warning", form.page_overlap_s,
            "перекрытие соседних листов указано", "склейка страниц возможна" if form.page_overlap_s is not None
            else "не указано перекрытие листов — стыки страниц не проверить")
    return rep


def gate_reasons(digitized: bool) -> list[tuple[str, str]]:
    """Mode-B gate: a signal digitized from an image never gets an autonomous impression."""
    if not digitized:
        return []
    return [("сигнал оцифрован с изображения (бумажная ЭЭГ): точность расшифровки не проверена — только просмотр врачом",
             "signal tasvirdan raqamlashtirilgan (qog‘oz EEG): aniqlik tekshirilmagan — faqat shifokor ko‘rigi")]
