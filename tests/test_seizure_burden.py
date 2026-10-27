"""Seizure-burden aggregation tests (TZ §7.3)."""

from __future__ import annotations

from neurolens.contracts.events import DetectionResult, Event, Localization
from neurolens.layer4_detect.seizure_burden import compute_seizure_burden


def _ictal(t0, t1):
    return Event(code="ictal_rhythm", label_ru="i", label_uz="i", group="ictal",
                 localization=Localization(), t_start=t0, t_end=t1, confidence=0.9)


def test_burden_aggregates_multiple_seizures():
    det = DetectionResult(events=[_ictal(10, 25), _ictal(100, 110)])
    sb = compute_seizure_burden(det, duration_s=600.0)
    assert sb.n_seizures == 2
    assert sb.total_seizure_time_s == 25.0
    assert sb.longest_seizure_s == 15.0
    assert abs(sb.seizure_fraction - 25.0 / 600.0) < 1e-3
    assert abs(sb.seizures_per_hour - 12.0) < 1e-6
    assert sb.status_epilepticus_suspected is False


def test_burden_flags_status():
    # ACNS: >= 10 continuous minutes (a 45-s seizure is not status epilepticus)
    assert compute_seizure_burden(DetectionResult(events=[_ictal(0, 45)]), 300.0).status_epilepticus_suspected is False
    sb = compute_seizure_burden(DetectionResult(events=[_ictal(0, 700)]), 3600.0)
    assert sb.status_epilepticus_suspected is True


def test_burden_empty():
    sb = compute_seizure_burden(DetectionResult(events=[]), duration_s=300.0)
    assert sb.n_seizures == 0
    assert sb.total_seizure_time_s == 0.0


def _learned(t0, t1, high):
    """A learned offline event with its high-confidence parts (increment 46, rule T)."""
    return Event(code="ictal_ml", label_ru="i", label_uz="i", group="ictal", localization=Localization(),
                 t_start=t0, t_end=t1, confidence=0.6, metadata={"status_intervals": high})


def test_rule_t_two_tiers():
    from neurolens.critical.safety import scan_critical_findings

    # 15 min of low-confidence candidates in an hour, no high-confidence part: review finding only
    low = DetectionResult(events=[_learned(k * 300, k * 300 + 180, []) for k in range(5)])
    low.seizure_burden = compute_seizure_burden(low, 3600.0)
    assert not low.seizure_burden.status_epilepticus_suspected
    assert low.seizure_burden.status_criteria_by_candidates
    codes = {f.code for f in scan_critical_findings(low)}
    assert "status_criteria_by_candidates" in codes and "status_epilepticus_suspected" not in codes
    # a 12-min high-confidence run: critical status
    high = DetectionResult(events=[_learned(0, 800, [[10, 730]])])
    high.seizure_burden = compute_seizure_burden(high, 3600.0)
    assert high.seizure_burden.status_epilepticus_suspected
    codes = {f.code for f in scan_critical_findings(high)}
    assert "status_epilepticus_suspected" in codes and "status_criteria_by_candidates" not in codes
    # events without the rule (other detectors, or rule off) behave as before
    assert compute_seizure_burden(DetectionResult(events=[_ictal(0, 700)]), 3600.0).status_epilepticus_suspected
