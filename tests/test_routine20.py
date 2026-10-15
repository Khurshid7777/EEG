"""Increment 20: age-specific cardiac rates for the ECG mimic of periodic discharges, and
learned seizure events on routine-length EEG as review candidates."""

import pytest

from neurolens.contracts.events import DetectionResult, Event, Localization
from neurolens.contracts.signal import ClinicalContext, PatientInfo
from neurolens.critical.safety import scan_critical_findings
from neurolens.layer3_features.norms import NormsEngine
from neurolens.layer4_detect.periodic import cardiac_band
from neurolens.layer4_detect.seizure_burden import compute_seizure_burden
from neurolens.layer5_reasoning.physiology import PhysiologyEngine
from neurolens.pipeline.pipeline import mark_routine_ictal_candidates


def _ictal(a, b, conf=0.8):
    return Event(code="ictal_ml", label_ru="приступ", label_uz="tutqanoq", group="ictal",
                 localization=Localization(), t_start=a, t_end=b, confidence=conf)


def test_cardiac_band_by_age(configs):
    thr = configs.thresholds
    assert cardiac_band(thr, 0.5) == (1.55, 3.0)     # infant 93-181 /min
    assert cardiac_band(thr, 5.0) == (1.0, 2.2)
    assert cardiac_band(thr, 40.0) == (0.75, 2.0)    # adult
    assert cardiac_band(thr, None) == (0.75, 2.0)
    assert thr.pd_ecg_isi_cv_max == pytest.approx(0.10)


def test_routine_ictal_events_are_review_candidates(configs):
    thr = configs.thresholds
    det = DetectionResult(events=[_ictal(10, 50), _ictal(100, 400)])
    adult = PatientInfo(age_years=30)
    assert mark_routine_ictal_candidates(det, 900.0, adult, thr) == 2
    eng = PhysiologyEngine(NormsEngine(configs.norms))
    assert all(eng.label_event(e, adult).value == "uncertain" for e in det.events)
    # not seizures for the burden, but still surfaced - worded as candidates
    assert compute_seizure_burden(det, 900.0).n_seizures == 0
    codes = [f.code for f in scan_critical_findings(det)]
    assert codes == ["seizure_candidate"]
    assert "не подтверждена" in scan_critical_findings(det)[0].text.ru


def test_long_term_and_neonatal_ictal_events_unchanged(configs):
    thr = configs.thresholds
    long_rec = DetectionResult(events=[_ictal(10, 50)])
    assert mark_routine_ictal_candidates(long_rec, 3600.0, PatientInfo(age_years=30), thr) == 0
    explicit = DetectionResult(events=[_ictal(10, 50)])
    assert mark_routine_ictal_candidates(explicit, 900.0, PatientInfo(age_years=30), thr,
                                         setting="long_term") == 0
    neo = DetectionResult(events=[_ictal(10, 50)])
    assert mark_routine_ictal_candidates(neo, 900.0, PatientInfo(postmenstrual_age_weeks=40), thr) == 0
    assert [f.code for f in scan_critical_findings(long_rec)] == ["ongoing_seizure"]
    eng = PhysiologyEngine(NormsEngine(configs.norms))
    assert eng.label_event(long_rec.events[0], PatientInfo(age_years=30)).value == "pathologic"


def test_switch_off_restores_previous_behaviour(configs):
    thr = configs.thresholds.model_copy(update={"ictal_ml_routine_uncertain": False})
    det = DetectionResult(events=[_ictal(10, 50)])
    assert mark_routine_ictal_candidates(det, 900.0, PatientInfo(age_years=30), thr) == 0


def test_context_accepts_recording_setting():
    assert ClinicalContext(recording_setting="routine").recording_setting == "routine"
    with pytest.raises(Exception):
        ClinicalContext(recording_setting="weekly")


def test_pdr_alert_estimate_ignores_drowsy_theta():
    import numpy as np

    from neurolens.layer3_features.background import posterior_dominant_rhythm

    fs, rng = 200.0, np.random.default_rng(3)
    t = np.arange(int(10 * fs)) / fs
    epochs = []
    for k in range(30):  # 5 min: alert alpha 9.5 Hz, drowsy minutes with strong 5.5 Hz theta
        drowsy = k % 3 == 0
        x = (40 * np.sin(2 * np.pi * 5.5 * t) if drowsy else 15 * np.sin(2 * np.pi * 9.5 * t))
        epochs.append(x + 5 * rng.standard_normal(t.size))
    occ = np.concatenate(epochs)
    assert posterior_dominant_rhythm(occ, occ, fs, method="alert") == pytest.approx(9.5, abs=0.3)
    assert posterior_dominant_rhythm(occ, occ, fs, method="whole") == pytest.approx(5.5, abs=0.3)
    short = occ[: int(30 * fs)]  # < 1 min of epochs: whole-record peak
    assert posterior_dominant_rhythm(short, short, fs, method="alert") == \
        posterior_dominant_rhythm(short, short, fs, method="whole")
    with pytest.raises(ValueError):
        posterior_dominant_rhythm(occ, occ, fs, method="median")


def test_prereg20_rule_and_apply(tmp_path):
    import shutil

    from neurolens.contracts.config import Thresholds
    from neurolens.evaluation import prereg20

    import yaml

    def rows(tn, tp, claims=0, n_n=300, n_a=45):
        out = []
        for i in range(n_n):
            out.append({"record_id": i, "abnormal": False, "impression_abnormal": i >= tn,
                        "detectors_fired": ["ictal_ml"], "critical_codes": ["ongoing_seizure"] if i < claims else []})
        for i in range(n_a):
            out.append({"record_id": 1000 + i, "abnormal": True, "impression_abnormal": i < tp,
                        "detectors_fired": [], "critical_codes": []})
        return {"records": out}

    imp, thr = tmp_path / "impression.yaml", tmp_path / "thresholds.yaml"
    shutil.copy("configs/thresholds.yaml", thr)
    ok = prereg20.assess(rows(216, 34), rows(192, 35))  # 0.72 vs 0.64
    assert ok["decision"]["keep_increment20_rules"] and ok["seizure_claims_normal"]["old_proxy"] == 1.0
    assert "подтверждена" in prereg20.markdown_report(ok)
    prereg20.apply_decision(ok, str(imp), str(thr))
    assert "specificity: 0.720" in imp.read_text()
    loud = prereg20.assess(rows(216, 34, claims=30), rows(192, 35))
    assert not loud["hypotheses"]["H4_seizure_claims_in_normal_reports_at_most_0.02"]
    bad = prereg20.assess(rows(216, 30), rows(192, 35))  # 4 more abnormal recordings missed
    assert not bad["decision"]["keep_increment20_rules"]
    prereg20.apply_decision(bad, str(imp), str(thr))
    t = Thresholds(**yaml.safe_load(thr.read_text()))
    assert (t.pd_ecg_isi_cv_max, t.pd_ecg_freq_hz_by_age, t.ictal_ml_routine_uncertain, t.pdr_method,
            t.field_required) == (0.05, [], False, "whole", False)


def test_pattern_without_field_is_uncertain(configs):
    from neurolens.layer4_detect.periodic import has_field

    assert has_field(["F7", "T3"]) and has_field(["C4", "F4", "O2"])
    assert not has_field(["F7", "P3"]) and not has_field(["A1", "T5"]) and not has_field(["O1"])
    assert configs.thresholds.field_required
    eng = PhysiologyEngine(NormsEngine(configs.norms))
    ev = Event(code="lrda", label_ru="LRDA", label_uz="LRDA", group="periodic",
               localization=Localization(channels=["F7", "P3"]), t_start=0, t_end=12, confidence=0.53,
               metadata={"artifact_hypothesis": "no_field"})
    assert eng.label_event(ev, PatientInfo(age_years=30)).value == "uncertain"


def test_spike_candidates_ranked_by_sharpness_not_channel_order(configs):
    import numpy as np

    from neurolens.contracts.signal import UnifiedSignal
    from neurolens.layer4_detect.ied import SpikeDetector

    fs, n = 200.0, 200 * 60
    rng = np.random.default_rng(0)
    names = ["Fp1", "Fp2", "F3", "C3", "O1"]
    x = rng.standard_normal((len(names), n)) * 5
    for k, t in enumerate(range(5, 55, 5)):       # moderately sharp peaks on Fp1
        x[0, int(t * fs)] += 120
    for t in (7.3, 23.1, 41.7):                    # much sharper peaks on O1
        x[4, int(t * fs)] += 600
    sig = UnifiedSignal(signal=x.astype(np.float32), sampling_rate_hz=fs, channel_names=names)
    thr = configs.thresholds.model_copy(update={"spike_candidate_rank": "sharpness"})
    ev = SpikeDetector(max_events=3).detect(sig, None, thr)
    assert [e.localization.channels[0] for e in ev] == ["O1"] * 3
    old = SpikeDetector(max_events=3).detect(sig, None, thr.model_copy(update={"spike_candidate_rank": "confidence"}))
    assert [e.localization.channels[0] for e in old] == ["Fp1"] * 3
