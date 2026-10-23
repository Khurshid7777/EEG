"""One-record summary of a pipeline run (increment 22): what the routine-EEG harness saves
per recording and what the record-level model reads. One function serves both, so the
model sees in the pipeline exactly the features it was trained on."""

from __future__ import annotations

META_KEYS = ("artifact_hypothesis", "review_candidate")
ALERT_QUANTILE = 0.75   # increment 43: top quarter of epochs by posterior alpha
POSTERIOR = ("O1", "O2")


def alert_band_shares(features, q: float = ALERT_QUANTILE) -> dict:
    """Delta / theta / alpha shares averaged over the most alert epochs - those whose posterior
    (O1/O2, else all-channel) alpha share is at or above the record's ``q``-quantile - for the
    serial comparison (D77). Empty when the recording has no epochs."""
    import numpy as np

    rp = features.epoch_relpow
    if not rp or rp["alpha"].size == 0:
        return {}
    post = [features.eeg_channels.index(c) for c in POSTERIOR if c in features.eeg_channels]
    pa = rp["alpha"][:, post].mean(axis=1) if post else rp["alpha"].mean(axis=1)
    keep = pa >= np.quantile(pa, q)
    return {f"alert_rel_{b}": float(rp[b][keep].mean()) for b in ("delta", "theta", "alpha")}


def summarize(analysis, features, detection, graph) -> dict:
    from ..layer2_preprocess.sqi import robust_amplitude_uv

    patho = sorted({n.event_code or n.id for n in graph.nodes
                    if getattr(n.physiology, "value", n.physiology) == "pathologic"})
    return {
        "age_years": analysis.patient.age_years,
        "detectors_fired": sorted({e.code for e in detection.events}),
        "pathologic_nodes": patho,
        "impression_abnormal": bool(patho),
        "pdr_hz": float(features.pdr_hz),
        "pdr_hz_unchecked": features.extra.get("pdr_hz_unchecked"),
        "pdr_posterior_ratio": features.extra.get("pdr_posterior_ratio"),
        "global_rel_delta": float(features.global_relpow["delta"]),
        "global_rel_theta": float(features.global_relpow["theta"]),
        "global_rel_alpha": float(features.global_relpow["alpha"]),
        **alert_band_shares(features),
        "robust_amplitude_uv": robust_amplitude_uv(analysis),
        "flags": list(analysis.quality.flags),
        "duration_s": float(features.duration_s),
        "events": [{"code": e.code, "t_start": e.t_start, "t_end": e.t_end,
                    "confidence": round(float(e.confidence), 3),
                    "channels": list(e.localization.channels)[:20],
                    "evidence": {ev.feature: ev.value for ev in e.evidence},
                    **({"metadata": {k: e.metadata[k] for k in META_KEYS if k in e.metadata}}
                       if any(k in e.metadata for k in META_KEYS) else {})}
                   for e in detection.events],
    }
