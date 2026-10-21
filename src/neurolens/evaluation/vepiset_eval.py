"""External validation of the IED detectors on vepiset (increment 38,
docs/preregistration_increment38.md): 84 patients, Peking Union Medical College Hospital.

Same pipeline and matching as increment 15 (:mod:`ied_eval`): filters, average reference,
window features of the frozen ``ied_gbm_v1``, heuristic ``SpikeDetector`` grid. Reference
discharge events: a point mark ``!`` at t -> [t - REF_HALF_S, t + REF_HALF_S]; a ``!start``..
``!end`` run -> its interval; merged as in increment 15. A detection matches within ``TOL_S``.
False detections per hour are counted in recordings without any marked discharge.
Epoch-level discrimination uses the dataset's own 4-s epochs (positive: a mark more than 0.1 s
inside the epoch, or overlap with a run).
"""

from __future__ import annotations

import numpy as np

from ..contracts.signal import UnifiedSignal
from ..datasets.vepiset import CHANNELS, FS, VepiAnnotation
from ..layer4_detect import ied_ml
from .ied_eval import (
    HEURISTIC_Z_GRID,
    IEDRecord,
    heuristic_detections,
    merge_intervals,
    score_record,
)

REF_HALF_S = 0.15
EPOCH_S = 4.0
EPOCH_BOUNDARY_S = 0.1
VOLT_TO_UV = 1e6


def to_signal(eeg_data: np.ndarray) -> UnifiedSignal:
    """19 10-20 electrodes + ECG (ECG1 - ECG2), volts -> µV."""
    x = np.asarray(eeg_data, dtype=np.float64) * VOLT_TO_UV
    eeg = x[:19]
    ecg = x[CHANNELS.index("ECG1")] - x[CHANNELS.index("ECG2")]
    return UnifiedSignal(signal=np.vstack([eeg, ecg[None]]).astype(np.float32), sampling_rate_hz=FS,
                         channel_names=CHANNELS[:19] + ["ECG"], source="vepiset")


def reference_events(ann: VepiAnnotation) -> list[tuple[float, float]]:
    return merge_intervals([(a - REF_HALF_S, b + REF_HALF_S) if a == b else (a, b) for a, b in ann.discharges])


def prepare(sig: UnifiedSignal, record_id: int, normal: bool, cfg) -> IEDRecord:
    from ..layer2_preprocess.filters import apply_filters
    from ..layer2_preprocess.reref import rereference

    analysis = rereference(apply_filters(sig, cfg.filters), "average")
    X, t0 = ied_ml.window_features(analysis.signal, analysis.sampling_rate_hz, analysis.channel_names)
    heur = {z: heuristic_detections(analysis, z) for z in HEURISTIC_Z_GRID}
    return IEDRecord(record_id, normal, float(analysis.duration_s), t0, X.astype(np.float32), heur,
                     list(sig.quality.flags))


def epoch_table(rec: IEDRecord, ann: VepiAnnotation, p: np.ndarray) -> list[dict]:
    """Dataset-style 4-s epochs: label, learned score (max window p overlapping the epoch),
    heuristic score (highest grid z with a detection in the epoch, 0 if none), state."""
    runs = [(a, b) for a, b in ann.discharges if b > a]
    points = [a for a, b in ann.discharges if a == b]
    states = sorted(ann.states)
    out = []
    n = int(rec.duration_s // EPOCH_S)
    for k in range(n):
        s, e = k * EPOCH_S, (k + 1) * EPOCH_S
        pos = any(s + EPOCH_BOUNDARY_S < t < e - EPOCH_BOUNDARY_S for t in points) or \
            any(a < e and b > s for a, b in runs)
        w = (rec.t0 < e) & (rec.t0 + ied_ml.WIN_S > s)
        lp = float(p[w].max()) if p.size and w.any() else 0.0
        hz = max([z for z in HEURISTIC_Z_GRID if any(a < e and b > s for a, b in rec.heuristic[z])], default=0.0)
        st = next((lab for t, lab in reversed(states) if t <= s), states[0][1] if states else "")
        out.append({"label": int(pos), "learned": lp, "heuristic_z": float(hz), "state": st})
    return out


def score(rec: IEDRecord, ann: VepiAnnotation, p: np.ndarray, threshold: float) -> dict:
    ref = reference_events(ann)
    learned = [(a, b) for a, b, _ in ied_ml.probability_events(rec.t0, p, threshold)]
    return {"learned": score_record(learned, ref, [], rec.duration_s, rec.normal),
            "heuristic": {str(z): score_record(rec.heuristic[z], ref, [], rec.duration_s, rec.normal)
                          for z in HEURISTIC_Z_GRID}}
