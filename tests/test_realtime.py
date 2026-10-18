"""Real-time cEEG monitoring tests (TZ §2 sub-mode, §10, §16)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import jsonschema

from neurolens.contracts.alarms import AlarmSeverity, AlarmType
from neurolens.contracts.config import AlarmRule, RealtimeConfig
from neurolens.contracts.report import Bilingual
from neurolens.realtime.alarms import AlarmCandidate, AlarmManager
from neurolens.realtime.buffer import WindowBuffer
from neurolens.realtime.monitor import RealtimeMonitor
from neurolens.realtime.stream import EdfReplaySource

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


# --- buffer ---------------------------------------------------------------
def test_window_buffer_overlap_count():
    fs = 10.0
    buf = WindowBuffer(n_channels=2, fs=fs, window_s=3.0, step_s=1.0)  # win=30, step=10
    emits = []
    for _ in range(6):  # six 1s chunks = 60 samples
        emits += buf.push(np.ones((2, 10)))
    assert len(emits) == 4  # windows ready after samples 30,40,50,60
    first_window, start_sample = emits[0]
    assert first_window.shape == (2, 30)
    assert start_sample == 0


# --- stream source --------------------------------------------------------
def test_edf_replay_covers_signal(demo_edf):
    src = EdfReplaySource(demo_edf, chunk_s=5.0)
    total = sum(c.data.shape[1] for c in src.chunks())
    assert total == 360 * 256
    assert "Fp1" in src.channel_names


# --- alarm manager gating -------------------------------------------------
def _cfg(**rule_kw):
    rule = AlarmRule(**rule_kw)
    return RealtimeConfig(alarms={"seizure": rule})


def _cand(conf=0.9, critical=True):
    return AlarmCandidate(
        type=AlarmType.SEIZURE, severity=AlarmSeverity.CRITICAL,
        message=Bilingual(ru="s", uz="s"), confidence=conf, is_critical=critical,
    )


def test_persistence_blocks_single_window():
    mgr = AlarmManager(_cfg(persistence_windows=2, refractory_s=0, min_confidence=0.5))
    assert mgr.process([_cand()], 0, 0.0, 30.0, 1.0) == []      # streak 1 -> no alarm
    fired = mgr.process([_cand()], 1, 5.0, 35.0, 1.0)           # streak 2 -> alarm
    assert len(fired) == 1 and fired[0].type == AlarmType.SEIZURE


def test_refractory_prevents_duplicate():
    mgr = AlarmManager(_cfg(persistence_windows=1, refractory_s=60.0, min_confidence=0.5))
    assert len(mgr.process([_cand()], 0, 0.0, 30.0, 1.0)) == 1
    assert mgr.process([_cand()], 1, 5.0, 35.0, 1.0) == []       # within refractory
    assert len(mgr.process([_cand()], 2, 70.0, 100.0, 1.0)) == 1  # refractory passed


def test_per_hour_cap_suppresses_noncritical():
    mgr = AlarmManager(_cfg(persistence_windows=1, refractory_s=0, max_per_hour=2, min_confidence=0.5))
    for i in range(2):
        assert len(mgr.process([_cand(critical=False)], i, float(i), i + 30.0, 1.0)) == 1
    # third within the hour -> suppressed
    assert mgr.process([_cand(critical=False)], 2, 2.0, 32.0, 1.0) == []
    assert mgr.n_suppressed == 1


def test_critical_bypasses_per_hour_cap():
    mgr = AlarmManager(_cfg(persistence_windows=1, refractory_s=0, max_per_hour=1, min_confidence=0.5))
    assert len(mgr.process([_cand(critical=True)], 0, 0.0, 30.0, 1.0)) == 1
    assert len(mgr.process([_cand(critical=True)], 1, 1.0, 31.0, 1.0)) == 1  # not suppressed


def test_low_confidence_resets_streak():
    mgr = AlarmManager(_cfg(persistence_windows=2, refractory_s=0, min_confidence=0.6))
    assert mgr.process([_cand(conf=0.9)], 0, 0.0, 30.0, 1.0) == []
    assert mgr.process([_cand(conf=0.3)], 1, 5.0, 35.0, 1.0) == []   # below min -> reset
    assert mgr.process([_cand(conf=0.9)], 2, 10.0, 40.0, 1.0) == []  # streak restarts at 1


# --- end-to-end monitor ---------------------------------------------------
def test_monitor_raises_alarms_with_fragments(demo_edf, configs, tmp_path):
    src = EdfReplaySource(demo_edf, chunk_s=configs.realtime.step_s)
    summary = RealtimeMonitor(config=configs, out_dir=tmp_path / "rt").run(src)

    types = {a.type for a in summary.alarms}
    assert AlarmType.SEIZURE in types or AlarmType.STATUS_EPILEPTICUS in types
    assert AlarmType.BURST_SUPPRESSION in types
    # every alarm carries an EEG fragment for verification (TZ §10)
    for a in summary.alarms:
        assert a.fragment_path and Path(a.fragment_path).exists()
    # latency recorded and real-time feasible (compute << step)
    assert summary.latency_ms_mean > 0
    assert summary.realtime_feasible is True
    assert summary.n_windows > 0


def test_monitor_summary_schema_valid(demo_edf, configs, tmp_path):
    src = EdfReplaySource(demo_edf, chunk_s=configs.realtime.step_s)
    summary = RealtimeMonitor(config=configs, out_dir=tmp_path / "rt").run(src)
    schema = json.loads((SCHEMAS / "monitor_summary.schema.json").read_text())
    jsonschema.validate(summary.model_dump(mode="json"), schema)


def test_neonatal_background_alarm_review_mode(configs):
    """Increment 28: 'review' turns neonatal burst-suppression alarms into capped warnings."""
    import dataclasses

    from neurolens.contracts.events import DetectionResult, Event, EventEvidence, Localization
    from neurolens.contracts.signal import PatientInfo

    ev = Event(code="burst_suppression", label_ru="ВП", label_uz="BS", group="suppression",
               localization=Localization(region="generalized"), t_start=0.0, t_end=30.0, confidence=0.9,
               evidence=[EventEvidence(feature="mean_ibi_s", value=20.0, unit="s")])
    det = DetectionResult(events=[ev])
    for mode, sev in (("critical", AlarmSeverity.CRITICAL), ("review", AlarmSeverity.WARNING)):
        cfg = dataclasses.replace(configs, realtime=configs.realtime.model_copy(
            update={"neonatal_background_alarms": mode}))
        mon = RealtimeMonitor(config=cfg, patient=PatientInfo(postmenstrual_age_weeks=40))
        (c,) = [c for c in mon._candidates(det, 0.0) if c.type == AlarmType.BURST_SUPPRESSION]
        assert c.severity == sev and c.is_critical == (mode == "critical")


def test_prereg28_evaluate_and_decision(tmp_path):
    from neurolens.evaluation.prereg28 import apply_decision, evaluate

    rows = [{"baby_id": f"b{i % 6}", "grade": g, "hours": 1.0, "background_alarms": a, "alarms_by_type": {}}
            for i, (g, a) in enumerate([(1, 0)] * 8 + [(1, 2)] * 2 + [(2, 1)] * 3 + [(3, 3)] * 4 + [(4, 5)])]
    a = evaluate(rows)
    assert a["specificity"] == 0.8 and a["sensitivity"] == 1.0
    assert a["hypotheses"]["H2_specificity_grade1_at_least_0.80"]
    assert a["decision"]["neonatal_background_alarms"] == "critical"
    y = tmp_path / "rt.yaml"
    y.write_text("window_s: 30.0\nneonatal_background_alarms: critical\n")
    a["decision"]["neonatal_background_alarms"] = "review"
    apply_decision(a, str(y))
    assert y.read_text().count("neonatal_background_alarms") == 1 and "review" in y.read_text()


def test_monitor_reduces_to_model_electrodes(demo_edf, configs):
    """Increment 30: the neonatal monitor analyses only the model's electrode set."""
    from neurolens.contracts.signal import PatientInfo

    mon = RealtimeMonitor(config=configs, patient=PatientInfo(postmenstrual_age_weeks=40))
    assert mon.channels == ["F3", "F4", "C3", "C4", "T3", "T4", "O1", "O2", "Cz"]
    mon.run(EdfReplaySource(demo_edf, chunk_s=configs.realtime.step_s))
    eeg = [n for n in mon._names if n.upper() in {c.upper() for c in mon.channels}]
    assert eeg and set(eeg) <= set(mon.channels)
    assert RealtimeMonitor(config=configs).channels is None  # adults: all electrodes


def test_monitor_incomplete_electrode_set_warns(demo_edf, configs):
    """Increment 30 review: a recording without the model's electrode set is analysed as is,
    with a warning (not reduced to too few channels, not a crash)."""
    mon = RealtimeMonitor(config=configs, channels=["Fp1", "Fp2", "P3"])
    s = mon.run(EdfReplaySource(demo_edf, chunk_s=configs.realtime.step_s))
    assert any(w.startswith("model_electrode_set_incomplete") for w in s.warnings)
    assert mon._names == list(EdfReplaySource(demo_edf).channel_names)
