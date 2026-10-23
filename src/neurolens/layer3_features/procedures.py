"""Activation procedures from the recording's own markers (D75): eyes closed / open,
hyperventilation (HV), intermittent photic stimulation (IPS).

The markers are free text written by the recording software or the technician (EDF+
annotations, BrainVision .vmrk); they are recognised by keywords (Russian and English) and
turned into intervals. From these intervals the module measures what a clinician reads off
the procedures:

* the posterior dominant rhythm on eyes-closed segments only (ACNS: the PDR is assessed
  with eyes closed, in the most alert state), next to the whole-record PDR of the
  record-level model;
* reactivity: attenuation of posterior alpha on eye opening (1 - P_open / P_closed);
* HV: slow (1-8 Hz) power in the last minute of HV and in the minute after it, relative to
  the minute before HV;
* IPS: occipital photic driving - power at the flash frequency (or its 2nd harmonic)
  relative to the neighbouring frequencies.

Descriptive only: the values go to the report and the result JSON. They do not enter the
record-level model (trained on NMT, which has no markers) or the impression.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..contracts.report import Bilingual
from .peak_utils import spectral_peak

STATE_SEG_MAX_S = 60.0      # use at most 60 s after an eyes-closed/-open marker
STATE_SKIP_S = 1.0          # skip the eye-movement transition after the marker
MIN_STATE_S = 10.0          # >= 10 s of each state to measure the PDR / reactivity
HV_DEFAULT_S = 180.0        # HV without an end marker: 3 min (regulation §4)
HV_WINDOW_S = 60.0          # baseline / last-minute / recovery windows
PHOTIC_DEFAULT_S = 10.0     # flash train without an end or duration
REACTIVE_MIN = 0.5          # alpha attenuation >= 50% on eye opening
HV_SLOWING_MIN = 2.0        # slow power x2 vs baseline
PHOTIC_DRIVING_SNR = 3.0

_CLOSED = re.compile(r"eyes?\s*clos|\bec\b|глаза\s*закр|закр\w*\s*глаз|\bзакрыл|\bзг\b", re.I)
_OPEN = re.compile(r"eyes?\s*open|\beo\b|глаза\s*откр|откр\w*\s*глаз|\bоткрыл|\bог\b", re.I)
_HV = re.compile(r"hyperv|\bhv\b|гипервент|\bгв\b", re.I)
_PHOTIC = re.compile(r"photic|\bi?ps\b|фотостим|\bфс\b|flash|вспышк", re.I)
_END = re.compile(r"\b(end|stop|off|post|after)\b|конец|оконч|после|стоп", re.I)
_HZ = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:hz|гц)", re.I)

OCCIPITAL = ("O1", "O2")
POSTERIOR_FALLBACK = ("P3", "P4", "Pz", "T5", "T6")


@dataclass
class Tagged:
    onset_s: float
    duration_s: float
    kind: str               # eyes_closed | eyes_open | hv_start | hv_end | photic | photic_end
    hz: float | None = None


def classify(text: str) -> tuple[str, float | None] | None:
    t = text.strip()
    hz = _HZ.search(t)
    f = float(hz.group(1).replace(",", ".")) if hz else None
    if _HV.search(t):
        return ("hv_end" if _END.search(t) else "hv_start"), None
    if _PHOTIC.search(t) or (hz and _HZ.fullmatch(t)):
        return ("photic_end" if _END.search(t) and f is None else "photic"), f
    if _CLOSED.search(t):
        return "eyes_closed", None
    if _OPEN.search(t):
        return "eyes_open", None
    return None


def tag_markers(markers) -> tuple[list[Tagged], list[str]]:
    tagged, unknown = [], []
    for m in markers:
        c = classify(m.text)
        if c is None:
            unknown.append(m.text)
        else:
            tagged.append(Tagged(float(m.onset_s), float(m.duration_s), c[0], c[1]))
    return sorted(tagged, key=lambda x: x.onset_s), unknown


def intervals(tagged: list[Tagged], duration_s: float) -> dict[str, list]:
    """{"eyes_closed": [(a, b)], "eyes_open": [...], "hv": [(a, b)], "photic": [(a, b, hz)]}."""
    out: dict[str, list] = {"eyes_closed": [], "eyes_open": [], "hv": [], "photic": []}
    starts = [t.onset_s for t in tagged]

    def next_after(i: int) -> float:
        return starts[i + 1] if i + 1 < len(starts) else duration_s

    for i, t in enumerate(tagged):
        if t.kind in ("eyes_closed", "eyes_open"):
            a = t.onset_s + STATE_SKIP_S
            b = min(next_after(i), t.onset_s + STATE_SEG_MAX_S, duration_s)
            if b > a:
                out[t.kind].append((a, b))
        elif t.kind == "hv_start":
            end = next((u.onset_s for u in tagged[i + 1:] if u.kind == "hv_end"), None)
            if t.duration_s > 0:
                end = t.onset_s + t.duration_s
            b = min(end if end is not None else t.onset_s + HV_DEFAULT_S, t.onset_s + 2 * HV_DEFAULT_S, duration_s)
            if b > t.onset_s:
                out["hv"].append((t.onset_s, b))
        elif t.kind == "photic":
            b = t.onset_s + (t.duration_s if t.duration_s > 0 else min(next_after(i) - t.onset_s, PHOTIC_DEFAULT_S))
            b = min(b, duration_s)
            if b > t.onset_s:
                out["photic"].append((t.onset_s, b, t.hz))
    return out


def _cut(x: np.ndarray, fs: float, segs) -> np.ndarray:
    parts = [x[..., int(a * fs):int(b * fs)] for a, b, *_ in segs]
    parts = [p - p.mean(axis=-1, keepdims=True) for p in parts if p.shape[-1] > 0]
    return np.concatenate(parts, axis=-1) if parts else x[..., :0]


def _band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    from scipy.signal import welch

    if x.shape[-1] < int(2 * fs):
        return float("nan")
    f, p = welch(x, fs=fs, nperseg=int(min(x.shape[-1], 4 * fs)), axis=-1)
    p = p.mean(axis=0) if p.ndim > 1 else p
    return float(p[(f >= lo) & (f <= hi)].sum())


def _photic_snr(x: np.ndarray, fs: float, hz: float) -> float:
    """Best SNR at the flash frequency or its 2nd harmonic: power within +/-0.25 Hz over the
    mean power 0.75-2 Hz away on both sides."""
    from scipy.signal import welch

    if x.shape[-1] < int(2 * fs):
        return float("nan")
    f, p = welch(x, fs=fs, nperseg=int(min(x.shape[-1], 4 * fs)), axis=-1)
    p = p.mean(axis=0) if p.ndim > 1 else p
    best = 0.0
    for h in (hz, 2 * hz):
        if h + 2 >= fs / 2:
            continue
        sig = p[np.abs(f - h) <= 0.25].mean()
        side = p[(np.abs(f - h) >= 0.75) & (np.abs(f - h) <= 2.0)]
        if side.size and side.mean() > 0:
            best = max(best, float(sig / side.mean()))
    return best


def _total(segs) -> float:
    return float(sum(b - a for a, b, *_ in segs))


def analyze_procedures(signal) -> dict | None:
    """``signal``: the analysed (filtered, re-referenced) UnifiedSignal. None when the
    recording has no markers."""
    if not signal.markers:
        return None
    fs = signal.sampling_rate_hz
    data = np.asarray(signal.signal, dtype=np.float64)
    names = list(signal.channel_names)
    dur = data.shape[1] / fs
    tagged, unknown = tag_markers(signal.markers)
    iv = intervals(tagged, dur)
    counts = {k: sum(t.kind == k for t in tagged)
              for k in ("eyes_closed", "eyes_open", "hv_start", "hv_end", "photic")}
    out: dict = {"markers_total": len(signal.markers), "recognized": counts,
                 "unrecognized_examples": sorted(set(unknown))[:8],
                 "eyes_closed_s": round(_total(iv["eyes_closed"]), 1),
                 "eyes_open_s": round(_total(iv["eyes_open"]), 1),
                 "ec_pdr_hz": None, "alpha_attenuation": None, "reactive": None, "hv": None, "photic": []}

    post = [names.index(c) for c in OCCIPITAL if c in names] or \
           [names.index(c) for c in POSTERIOR_FALLBACK if c in names]
    eeg = [i for i, c in enumerate(names) if c in OCCIPITAL + POSTERIOR_FALLBACK
           or c in ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4")]
    if not post:
        return out
    occ = data[post]

    if out["eyes_closed_s"] >= MIN_STATE_S:
        ec = _cut(occ, fs, iv["eyes_closed"])
        pdr = spectral_peak(ec.mean(axis=0), fs, fmin=4.0, fmax=13.0)
        out["ec_pdr_hz"] = round(pdr, 2) if pdr > 0 else 0.0
        if out["eyes_open_s"] >= MIN_STATE_S:
            lo, hi = (pdr - 1.5, pdr + 1.5) if pdr >= 6.0 else (8.0, 13.0)
            pc = _band_power(ec, fs, lo, hi)
            po = _band_power(_cut(occ, fs, iv["eyes_open"]), fs, lo, hi)
            if pc == pc and po == po and pc > 0:
                att = 1.0 - po / pc
                out["alpha_attenuation"] = round(att, 3)
                out["reactive"] = bool(att >= REACTIVE_MIN)

    if iv["hv"] and eeg:
        a, b = iv["hv"][0]
        x = data[eeg]
        base = _band_power(_cut(x, fs, [(max(0.0, a - HV_WINDOW_S), a)]), fs, 1.0, 8.0)
        last = _band_power(_cut(x, fs, [(max(a, b - HV_WINDOW_S), b)]), fs, 1.0, 8.0)
        rec = _band_power(_cut(x, fs, [(b, min(dur, b + HV_WINDOW_S))]), fs, 1.0, 8.0)
        hv = {"start_s": round(a, 1), "duration_s": round(b - a, 1),
              "slow_ratio_end": round(last / base, 2) if base == base and base > 0 and last == last else None,
              "slow_ratio_recovery": round(rec / base, 2) if base == base and base > 0 and rec == rec else None}
        hv["slowing"] = hv["slow_ratio_end"] is not None and hv["slow_ratio_end"] >= HV_SLOWING_MIN
        hv["persists"] = hv["slow_ratio_recovery"] is not None and hv["slow_ratio_recovery"] >= HV_SLOWING_MIN
        out["hv"] = hv

    for a, b, hz in iv["photic"]:
        if hz is None or hz <= 0:
            continue
        snr = _photic_snr(_cut(occ, fs, [(a, b)]).mean(axis=0, keepdims=True), fs, hz)
        out["photic"].append({"hz": hz, "start_s": round(a, 1), "duration_s": round(b - a, 1),
                              "snr": round(snr, 2) if snr == snr else None,
                              "driving": bool(snr == snr and snr >= PHOTIC_DRIVING_SNR)})
    return out


def procedures_text(p: dict | None) -> Bilingual | None:
    if not p:
        return None
    ru, uz = [], []
    c = p["recognized"]
    if not any(c.values()):
        return Bilingual(ru=f"Метки в записи ({p['markers_total']}) не распознаны как пробы "
                            "(глаза, ГВ, ФС) — пробы не оценены.",
                         uz=f"Yozuvdagi belgilar ({p['markers_total']}) sinamalar sifatida tanilmadi "
                            "(ko‘z, GV, FS) — sinamalar baholanmadi.")
    if p["ec_pdr_hz"] is not None:
        if p["ec_pdr_hz"]:
            ru.append(f"основной ритм с закрытыми глазами {p['ec_pdr_hz']:g} Гц ({p['eyes_closed_s']:g} с)")
            uz.append(f"ko‘z yumilganda asosiy ritm {p['ec_pdr_hz']:g} Gs ({p['eyes_closed_s']:g} s)")
        else:
            ru.append(f"с закрытыми глазами выраженный пик 4–13 Гц не найден ({p['eyes_closed_s']:g} с)")
            uz.append(f"ko‘z yumilganda 4–13 Gs aniq cho‘qqi topilmadi ({p['eyes_closed_s']:g} s)")
    if p["alpha_attenuation"] is not None:
        pct = round(100 * p["alpha_attenuation"])
        ru.append(f"реакция на открывание глаз: подавление ритма на {pct}% "
                  + ("(реактивен)" if p["reactive"] else "(снижена)"))
        uz.append(f"ko‘z ochilishiga javob: ritm {pct}% ga susaydi "
                  + ("(reaktiv)" if p["reactive"] else "(pasaygan)"))
    hv = p["hv"]
    if hv and hv["slow_ratio_end"] is not None:
        ru.append(f"ГВ {hv['duration_s']:g} с: медленная активность ×{hv['slow_ratio_end']:g} к фону"
                  + (", сохраняется через минуту после ГВ" if hv["persists"] else "")
                  + (" (нарастание)" if hv["slowing"] else ""))
        uz.append(f"GV {hv['duration_s']:g} s: sekin faollik fonga nisbatan ×{hv['slow_ratio_end']:g}"
                  + (", GVdan bir daqiqa keyin ham saqlanadi" if hv["persists"] else ""))
    if p["photic"]:
        drv = sorted({x["hz"] for x in p["photic"] if x["driving"]})
        tested = sorted({x["hz"] for x in p["photic"]})
        ru.append(f"ФС {', '.join(f'{h:g}' for h in tested)} Гц: усвоение ритма "
                  + (f"на {', '.join(f'{h:g}' for h in drv)} Гц" if drv else "не выявлено"))
        uz.append(f"FS {', '.join(f'{h:g}' for h in tested)} Gs: ritm o‘zlashtirish "
                  + (f"{', '.join(f'{h:g}' for h in drv)} Gs da" if drv else "aniqlanmadi"))
    if not ru:
        ru.append("распознанных проб недостаточно для оценки")
        uz.append("baholash uchun tanilgan sinamalar yetarli emas")
    note_ru = " Описательно, в заключение не входит."
    note_uz = " Tavsifiy, xulosaga kirmaydi."
    cap = lambda t: t[:1].upper() + t[1:]
    return Bilingual(ru=cap("; ".join(ru)) + "." + note_ru, uz=cap("; ".join(uz)) + "." + note_uz)
