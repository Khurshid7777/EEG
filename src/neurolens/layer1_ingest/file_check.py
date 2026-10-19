"""Intake check of a digital EEG recording against the recording/export regulation
(docs/recording_regulation.md) - before it is analysed.

Every requirement gives a result (ok / warning / reject) with the measured value and the limit,
and a Russian message saying what to change at the recording or export step. A file with any
"reject" is not analysed; warnings mean the analysis runs with a known limitation (stated in the
report). The limits are those of the pipeline as validated: the record-level model and the IED
detector were trained on 19-electrode 10-20 referential recordings at 200 Hz (NMT), the verdict
uses age-band thresholds, and amplitude-based detectors need microvolts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..contracts.signal import PatientInfo, UnifiedSignal

STANDARD_1020 = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4",
                 "T5", "P3", "Pz", "P4", "T6", "O1", "O2")
# regulation limits (docs/recording_regulation.md §2)
MIN_FS_HZ, RECOMMENDED_FS_HZ = 100.0, 200.0
MIN_1020, RECOMMENDED_1020 = 16, 19
MIN_DURATION_MIN, RECOMMENDED_DURATION_MIN = 3.0, 20.0
AMPLITUDE_SD_UV = (2.0, 300.0)   # median channel SD; outside -> units/scale wrong
FLAT_SD_UV = 0.5                 # a channel quieter than this is disconnected
FLAT_RELATIVE = 0.05             # ... or quieter than 5% of the median channel
MAX_FLAT = 3
MIN_LOWPASS_HZ = 60.0            # a hardware/export low-pass below this cuts spikes and beta
LINE_RATIO_WARN = 0.5            # 50/60 Hz power over 1-40 Hz power

Status = Literal["ok", "warning", "reject"]


@dataclass
class Check:
    name: str
    status: Status
    value: str
    limit: str
    message_ru: str


@dataclass
class FileReport:
    path: str
    checks: list[Check] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return all(c.status != "reject" for c in self.checks)

    def to_dict(self) -> dict:
        return {"path": self.path, "accepted": self.accepted, "summary": self.summary,
                "checks": [c.__dict__ for c in self.checks]}


def _edf_header(path: Path) -> dict:
    """Physical units, prefilter strings, annotations and patient field of an EDF/BDF."""
    from .edf import open_edf

    r = open_edf(path)
    try:
        n = r.signals_in_file
        get = lambda name, i: getattr(r, name)(i) if hasattr(r, name) else ""
        ann = r.readAnnotations() if hasattr(r, "readAnnotations") else ([], [], [])
        return {"labels": list(r.getSignalLabels()),
                "dims": [get("getPhysicalDimension", i) for i in range(n)],
                "prefilter": [get("getPrefilter", i) for i in range(n)],
                "annotations": [str(t) for t in (ann[2] if len(ann) > 2 else [])],
                "birthdate": getattr(r, "getBirthdate", lambda: "")() or ""}
    finally:
        r.close()


def _lowpass_hz(prefilter: str) -> float | None:
    m = re.search(r"LP\s*:?\s*([\d.]+)\s*(k?)Hz", prefilter or "", re.IGNORECASE)
    return float(m.group(1)) * (1000 if m.group(2) else 1) if m else None


def _line_ratio(sig: UnifiedSignal) -> float:
    from scipy.signal import welch

    x = np.asarray(sig.signal, dtype=np.float64)
    seg = x[:, : int(min(x.shape[1], 120 * sig.sampling_rate_hz))]
    f, p = welch(seg, fs=sig.sampling_rate_hz, nperseg=int(2 * sig.sampling_rate_hz), axis=-1)
    p = np.median(p, axis=0)
    band = p[(f >= 1) & (f <= 40)].sum()
    line = max(p[(f >= lo - 1) & (f <= lo + 1)].sum() for lo in (50.0, 60.0) if lo < sig.sampling_rate_hz / 2)
    return float(line / band) if band > 0 else float("inf")


def check_recording(path: str | Path, patient: PatientInfo | None = None) -> FileReport:
    from .registry import get_ingestor

    path = Path(path)
    rep = FileReport(str(path))
    add = lambda *a: rep.checks.append(Check(*a))
    try:
        sig = get_ingestor(path).read(path, patient=patient)
    except Exception as exc:  # noqa: BLE001 - unreadable is the finding
        add("format", "reject", path.suffix or "-", "EDF/EDF+/BDF, EEGLAB .set, WFDB .hea",
            f"файл не читается ({type(exc).__name__}): экспортируйте запись в EDF+ (см. регламент §3)")
        return rep
    hdr = _edf_header(path) if path.suffix.lower() in (".edf", ".bdf") else {}
    add("format", "ok", path.suffix.lower(), "EDF/EDF+/BDF, EEGLAB .set, WFDB .hea", "формат читается")

    fs = sig.sampling_rate_hz
    st = "reject" if fs < MIN_FS_HZ else "warning" if fs < RECOMMENDED_FS_HZ else "ok"
    add("sampling_rate", st, f"{fs:g} Гц", f"≥ {RECOMMENDED_FS_HZ:g} Гц (минимум {MIN_FS_HZ:g})",
        {"ok": "частота дискретизации достаточна",
         "warning": "частота ниже 200 Гц: спайки и бета-активность анализируются с потерями",
         "reject": "частота слишком низкая для анализа: записывайте с 200–500 Гц"}[st])

    have = [c for c in STANDARD_1020 if c in sig.channel_names]
    missing = [c for c in STANDARD_1020 if c not in have]
    st = "reject" if len(have) < MIN_1020 else "warning" if len(have) < RECOMMENDED_1020 else "ok"
    add("electrodes_1020", st, f"{len(have)}/19", f"все 19 (минимум {MIN_1020})",
        "все 19 электродов 10-20 на месте" if st == "ok" else
        f"нет электродов: {', '.join(missing)} — проверьте названия каналов при экспорте "
        "(Fp1…O2, T3/T4/T5/T6 или T7/T8/P7/P8) и наложение электродов")

    ref = sig.reference
    st = "warning" if ref == "average_from_bipolar" else "ok"
    add("reference", st, ref, "референтная запись (A1/A2, общий референт)",
        "экспортирован биполярный монтаж: исходные отведения восстановлены приближённо — "
        "экспортируйте исходную референтную запись, а не монтаж просмотра" if st == "warning"
        else "референтная запись")

    dur_min = sig.signal.shape[1] / fs / 60
    st = "reject" if dur_min < MIN_DURATION_MIN else "warning" if dur_min < RECOMMENDED_DURATION_MIN else "ok"
    add("duration", st, f"{dur_min:.1f} мин", f"≥ {RECOMMENDED_DURATION_MIN:g} мин (минимум {MIN_DURATION_MIN:g})",
        {"ok": "длительность достаточна",
         "warning": "запись короче 20 мин: редкие разряды могут не попасть в запись",
         "reject": "запись слишком короткая: экспортируйте запись целиком, не фрагмент экрана"}[st])

    eeg = [i for i, c in enumerate(sig.channel_names) if c in STANDARD_1020]
    sd = np.std(np.asarray(sig.signal, dtype=np.float64)[eeg], axis=1) if eeg else np.array([])
    med = float(np.median(sd)) if sd.size else float("nan")
    dims = {d.strip() for d in hdr.get("dims", []) if d.strip()}
    st = "ok" if AMPLITUDE_SD_UV[0] <= med <= AMPLITUDE_SD_UV[1] else "reject"
    add("amplitude_units", st, f"{med:.1f} мкВ" + (f" ({'/'.join(sorted(dims))})" if dims else ""),
        f"медиана СКО {AMPLITUDE_SD_UV[0]:g}–{AMPLITUDE_SD_UV[1]:g} мкВ",
        "амплитуда в микровольтах правдоподобна" if st == "ok" else
        "амплитуда неправдоподобна для мкВ: проверьте единицы (uV/mV) и калибровку при экспорте")

    # absolute floor only when the scale is plausible; relative to the other channels always
    floor = max(FLAT_SD_UV if st == "ok" else 0.0, FLAT_RELATIVE * med) if sd.size else 0.0
    flat = [sig.channel_names[i] for i, s in zip(eeg, sd) if s < floor]
    st = "reject" if len(flat) > MAX_FLAT else "warning" if flat else "ok"
    add("flat_channels", st, str(len(flat)), f"0 (не более {MAX_FLAT})",
        "плоских каналов нет" if not flat else
        f"плоские (отключённые) каналы: {', '.join(flat)} — проверьте электроды и сопротивление")

    lps = [v for v in (_lowpass_hz(p) for p in hdr.get("prefilter", [])) if v]
    lp = min(lps) if lps else None
    st = "warning" if lp is not None and lp < MIN_LOWPASS_HZ else "ok"
    add("export_filters", st, f"LP {lp:g} Гц" if lp else "не указаны",
        f"верхняя граница полосы (LP) ≥ {MIN_LOWPASS_HZ:g} Гц или фильтр выключен",
        "экспорт с LP ниже 60 Гц срезает спайки и бета-ритм: экспортируйте без фильтров "
        "просмотра" if st == "warning" else "фильтры экспорта не мешают анализу")

    lr = _line_ratio(sig)
    st = "warning" if lr > LINE_RATIO_WARN else "ok"
    add("line_noise", st, f"{lr:.2f}", f"≤ {LINE_RATIO_WARN:g} мощности 1–40 Гц",
        "сильная сетевая наводка 50 Гц: проверьте заземление и сопротивление электродов"
        if st == "warning" else "сетевая наводка в норме")

    age = (patient.age_years if patient else None) or sig.patient.age_years
    st = "ok" if age is not None or (patient and patient.postmenstrual_age_weeks) else "warning"
    add("patient_age", st, f"{age:g} лет" if age is not None else "нет", "возраст указан",
        "возраст указан" if st == "ok" else
        "возраст не указан: пороги по возрасту не применяются — передайте возраст (--age или поле запроса)")

    has_ecg = any(c == "ECG" for c in sig.channel_names)
    add("ecg_channel", "ok" if has_ecg else "warning", "есть" if has_ecg else "нет", "канал ЭКГ",
        "канал ЭКГ есть" if has_ecg else
        "нет канала ЭКГ: кардиоартефакт хуже отличается от периодических разрядов — записывайте ЭКГ")

    if hdr:
        ann = hdr["annotations"]
        st = "ok" if ann else "warning"
        add("annotations", st, str(len(ann)), "метки проб (ГВ, ФС, открыл/закрыл глаза)",
            "метки событий есть" if ann else
            "нет меток: экспортируйте EDF+ с аннотациями (гипервентиляция, фотостимуляция, глаза)")

    rep.summary = {"sampling_rate_hz": fs, "duration_min": round(dur_min, 2), "n_channels": len(sig.channel_names),
                   "electrodes_1020": len(have), "reference": ref, "age_years": age}
    return rep
