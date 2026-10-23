"""Ingestor registry — dispatch by file extension (TZ §4).

EDF/BDF, EEGLAB, WFDB, BrainVision and Nihon Kohden are registered. Further instrument formats
(Natus/XLTEK, Micromed, Persyst) and streaming (LSL) register here in v2 behind the same
``Ingestor`` interface without touching downstream layers.
"""

from __future__ import annotations

from pathlib import Path

from ..contracts.signal import ClinicalContext, PatientInfo, UnifiedSignal
from .base import Ingestor
from .brainvision import BrainVisionIngestor
from .edf import EdfIngestor
from .eeglab import EeglabIngestor
from .nihon import NihonKohdenIngestor
from .wfdb import WfdbIngestor

_REGISTRY: list[Ingestor] = [EdfIngestor(), EeglabIngestor(), WfdbIngestor(), BrainVisionIngestor(),
                             NihonKohdenIngestor()]


def register_ingestor(ingestor: Ingestor) -> None:
    _REGISTRY.insert(0, ingestor)


def get_ingestor(path: str | Path) -> Ingestor:
    for ing in _REGISTRY:
        if ing.can_read(path):
            return ing
    raise ValueError(
        f"no ingestor for {Path(path).suffix!r}; supported: "
        f"{sorted({e for ing in _REGISTRY for e in ing.extensions})}"
    )


def ingest(
    path: str | Path,
    *,
    patient: PatientInfo | None = None,
    context: ClinicalContext | None = None,
) -> UnifiedSignal:
    """Read any supported file into a :class:`UnifiedSignal`."""
    return get_ingestor(path).read(path, patient=patient, context=context)
