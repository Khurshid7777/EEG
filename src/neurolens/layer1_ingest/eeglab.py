"""EEGLAB ``.set`` ingestor (increment 33) - MATLAB v5 files, data in the ``.set`` or in a
companion ``.fdt`` (float32, channels x samples, column-major). No MNE dependency.
EEGLAB stores EEG in µV. MATLAB v7.3 (HDF5) ``.set`` files are not supported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.signal import ClinicalContext, PatientInfo, Provenance, UnifiedSignal
from .base import Ingestor
from .electrodes import detect_electrode_system, normalize_channel_name


def _eeg_struct(path: Path):
    from scipy.io import loadmat

    m = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    return m["EEG"] if "EEG" in m else _Flat(m)


class _Flat:
    """EEGLAB 'pop_saveset' without the EEG wrapper: fields at the top level."""

    def __init__(self, m: dict):
        self.__dict__.update({k: v for k, v in m.items() if not k.startswith("__")})


def _labels(chanlocs, n: int) -> list[str]:
    if chanlocs is None or (hasattr(chanlocs, "size") and chanlocs.size == 0):
        return [f"Ch{i + 1}" for i in range(n)]
    locs = np.atleast_1d(chanlocs)
    return [str(getattr(c, "labels", f"Ch{i + 1}")) for i, c in enumerate(locs)]


class EeglabIngestor(Ingestor):
    extensions = (".set",)
    source_name = "eeglab"

    def read(self, path: str | Path, *, patient: PatientInfo | None = None,
             context: ClinicalContext | None = None) -> UnifiedSignal:
        path = Path(path)
        eeg = _eeg_struct(path)
        n_ch, n_s, fs = int(eeg.nbchan), int(eeg.pnts), float(eeg.srate)
        data = eeg.data
        if isinstance(data, str):  # samples in a companion .fdt file
            raw = np.fromfile(path.with_name(data), dtype="<f4")
            data = raw[: n_ch * n_s].reshape((n_ch, n_s), order="F")
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 3:  # epoched: concatenate trials
            data = data.reshape(n_ch, -1, order="F")
        if data.ndim == 1:
            data = data[None, :]
        names = [normalize_channel_name(s) for s in _labels(getattr(eeg, "chanlocs", None), n_ch)]
        ref = str(getattr(eeg, "ref", "") or "unknown")
        return UnifiedSignal(
            signal=data, sampling_rate_hz=fs, channel_names=names,
            electrode_system=detect_electrode_system(names), reference=ref.lower(),
            source=self.source_name, patient=patient or PatientInfo(),
            context=context or ClinicalContext(),
            provenance=Provenance(digitized=False, confidence=1.0, source_file=path.name,
                                  pipeline_version="0.1.0"),
        )
