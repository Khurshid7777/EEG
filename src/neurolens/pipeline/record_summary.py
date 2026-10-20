"""One-record summary of a pipeline run (increment 22): what the routine-EEG harness saves
per recording and what the record-level model reads. One function serves both, so the
model sees in the pipeline exactly the features it was trained on."""

from __future__ import annotations

META_KEYS = ("artifact_hypothesis", "review_candidate")


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
