"""Lab Streaming Layer transport for the real-time monitor (TZ §4; D82). Requires ``pylsl``, which
bundles the native ``liblsl``.

:class:`LslStreamSource` reads an EEG stream (found by name or type): channel labels and units
from the stream description (volts / millivolts / microvolts -> µV), samples in chunks, and the
LSL timestamps of every sample - a jump larger than 1.5 sample periods is recorded as a gap (lost
samples), because the monitor's windows assume contiguous samples. The stream ends after
``idle_timeout_s`` without data or after ``max_duration_s``.

:func:`stream_recording` publishes a recording file as an LSL stream (paced to wall-clock), to
connect acquisition software or to test the monitor end to end.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np

from ..layer1_ingest.electrodes import normalize_channel_name
from .stream import Chunk, StreamSource

_UNIT_TO_UV = {"microvolts": 1.0, "microvolt": 1.0, "uv": 1.0, "µv": 1.0, "μv": 1.0,
               "millivolts": 1e3, "millivolt": 1e3, "mv": 1e3, "volts": 1e6, "volt": 1e6, "v": 1e6}
GAP_FACTOR = 1.5


def _pylsl():
    try:
        import pylsl
    except Exception as exc:  # noqa: BLE001 - pylsl or its native library missing
        raise RuntimeError("LSL needs the pylsl package (pip install pylsl; it bundles liblsl)") from exc
    return pylsl


def channel_info(info) -> tuple[list[str], list[str]]:
    """(labels, units) from an LSL StreamInfo description; ch<i> / '' when absent."""
    labels, units = [], []
    ch = info.desc().child("channels").child("channel")
    for i in range(info.channel_count()):
        labels.append(ch.child_value("label") or f"ch{i}")
        units.append(ch.child_value("unit") or "")
        ch = ch.next_sibling()
    return labels, units


class LslStreamSource(StreamSource):
    def __init__(self, stream_name: str | None = None, stream_type: str = "EEG", chunk_s: float = 1.0,
                 resolve_timeout_s: float = 10.0, idle_timeout_s: float = 5.0,
                 max_duration_s: float | None = None):
        pylsl = _pylsl()
        prop, value = ("name", stream_name) if stream_name else ("type", stream_type)
        found = pylsl.resolve_byprop(prop, value, timeout=resolve_timeout_s)
        if not found:
            raise RuntimeError(f"no LSL stream with {prop}={value!r} within {resolve_timeout_s:g} s")
        self._inlet = pylsl.StreamInlet(found[0], max_buflen=360)
        info = self._inlet.info()
        if info.nominal_srate() <= 0:
            raise RuntimeError("LSL stream has an irregular sampling rate; the monitor needs a fixed rate")
        labels, units = channel_info(info)
        self.sampling_rate_hz = float(info.nominal_srate())
        self.channel_names = [normalize_channel_name(c) for c in labels]
        self.reference = "unknown"
        self.source_name = f"lsl:{info.name()}"
        self._scale = np.array([_UNIT_TO_UV.get(u.strip().lower(), 1.0) for u in units])[:, None]
        self.units_assumed = [labels[i] for i, u in enumerate(units) if u.strip().lower() not in _UNIT_TO_UV]
        self.chunk_s, self.idle_timeout_s, self.max_duration_s = chunk_s, idle_timeout_s, max_duration_s
        self.gaps: list[tuple[float, float]] = []      # (stream time, lost seconds)
        self.n_samples = 0

    def chunks(self) -> Iterator[Chunk]:
        fs = self.sampling_rate_hz
        want = max(1, int(round(self.chunk_s * fs)))
        first_ts = last_ts = None
        idle_since = time.monotonic()
        while self.max_duration_s is None or self.n_samples < self.max_duration_s * fs:
            samples, stamps = self._inlet.pull_chunk(timeout=min(self.chunk_s, 0.5), max_samples=want)
            if not samples:
                if time.monotonic() - idle_since > self.idle_timeout_s:
                    return
                continue
            idle_since = time.monotonic()
            ts = np.asarray(stamps, float)
            if first_ts is None:
                first_ts = ts[0]
            seq = np.concatenate([[last_ts], ts]) if last_ts is not None else ts
            for k in np.nonzero(np.diff(seq) > GAP_FACTOR / fs)[0]:
                self.gaps.append((round(float(seq[k] - first_ts), 3), round(float(seq[k + 1] - seq[k] - 1 / fs), 3)))
            last_ts = ts[-1]
            data = np.asarray(samples, dtype=np.float64).T * self._scale
            if self.max_duration_s is not None:
                data = data[:, : int(self.max_duration_s * fs) - self.n_samples]
            yield Chunk(data=data, t_start=self.n_samples / fs)
            self.n_samples += data.shape[1]

    def warnings(self) -> list[str]:
        out = []
        if self.units_assumed:
            out.append(f"lsl_units_assumed_uV:{','.join(self.units_assumed[:8])}: the stream gives no unit "
                       "for these channels - check the amplitude scale")
        if self.gaps:
            lost = sum(g for _, g in self.gaps)
            out.append(f"lsl_gaps:{len(self.gaps)} ({lost:.1f} s lost): windows spanning a gap join "
                       "non-contiguous samples")
        return out


def stream_recording(path: str | Path, name: str | None = None, chunk_s: float = 0.1,
                     speed: float = 1.0, start_delay_s: float = 0.0, linger_s: float = 3.0) -> None:
    """Publish a recording (any supported format) as an LSL EEG stream in µV, paced to wall-clock
    (``speed`` > 1 plays faster). The outlet stays open ``linger_s`` after the last sample so that
    data still in transit reaches the inlets (destroying it at once lost the tail of the stream)."""
    from ..layer1_ingest.registry import ingest

    pylsl = _pylsl()
    sig = ingest(path)
    fs, n_ch = float(sig.sampling_rate_hz), sig.n_channels
    info = pylsl.StreamInfo(name or Path(path).stem, "EEG", n_ch, fs, "float32", f"neurolens-{Path(path).name}")
    chans = info.desc().append_child("channels")
    for c in sig.channel_names:
        ch = chans.append_child("channel")
        ch.append_child_value("label", c)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    outlet = pylsl.StreamOutlet(info, chunk_size=max(1, int(chunk_s * fs)))
    time.sleep(start_delay_s)
    step = max(1, int(round(chunk_s * fs)))
    x = np.asarray(sig.signal, np.float32)
    t0 = time.monotonic()
    for start in range(0, x.shape[1], step):
        due = t0 + start / fs / speed
        if (wait := due - time.monotonic()) > 0:
            time.sleep(wait)
        outlet.push_chunk(np.ascontiguousarray(x[:, start:start + step].T))
    time.sleep(linger_s)
