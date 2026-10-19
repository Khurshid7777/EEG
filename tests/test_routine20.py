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


def test_record_model_features_and_scoring(configs):
    from neurolens.layer5_reasoning.record_model import FEATURE_NAMES, RecordModel, record_features

    row = {"age_years": 30.0, "pdr_hz": 9.5, "global_rel_delta": 0.4, "global_rel_theta": 0.2,
           "global_rel_alpha": 0.1, "robust_amplitude_uv": 20.0, "duration_s": 900.0,
           "pathologic_nodes": ["spike"], "detectors_fired": ["spike", "ictal_ml"],
           "events": [{"code": "spike", "t_start": 1, "t_end": 1.1, "confidence": 0.8,
                       "evidence": {"ied_probability_learned": 0.97}},
                      {"code": "ictal_ml", "t_start": 10, "t_end": 100, "confidence": 0.6, "evidence": {}}]}
    f = record_features(row)
    assert len(f) == len(FEATURE_NAMES) and f[FEATURE_NAMES.index("ied_n_ge_095")] == 1.0
    assert f[FEATURE_NAMES.index("ictal_fraction")] == pytest.approx(0.1)
    m = RecordModel.load(configs.ml_model_path(configs.impression.record_model))
    assert 0.0 <= m.probability(row) <= 1.0


def test_record_score_decides_only_when_enabled(configs):
    from neurolens.contracts.report import Bilingual
    from neurolens.pipeline.pipeline import apply_record_score

    class Rep:
        impression = Bilingual(ru="Заключение: явных патологических паттернов не выделено.", uz="Xulosa: x.")

    off = configs.impression.model_copy(update={"record_model_decides": False, "record_threshold": 0.3})
    r = Rep()
    s = apply_record_score(r, {"impression_abnormal": False}, 0.8, off)
    assert s["abnormal"] and not s["decides"] and "явных" in r.impression.ru[:40]
    on = off.model_copy(update={"record_model_decides": True})
    r = Rep()
    s = apply_record_score(r, {"impression_abnormal": False}, 0.8, on)
    assert s["decides"] and r.impression.ru.startswith("Заключение: признаки отклонений по совокупной")


def test_prereg22_rule_and_apply(tmp_path):
    import shutil

    import yaml

    from neurolens.contracts.config import ImpressionConfig
    from neurolens.evaluation import prereg22

    def run(p_n, p_a, rule_tn, rule_tp, n=95, a=90):
        rows = [{"record_id": i, "abnormal": False, "record_probability": p_n(i),
                 "rule_impression_abnormal": i >= rule_tn} for i in range(n)]
        rows += [{"record_id": 1000 + i, "abnormal": True, "record_probability": p_a(i),
                  "rule_impression_abnormal": i < rule_tp} for i in range(a)]
        return {"records": rows}

    imp = tmp_path / "impression.yaml"
    shutil.copy("configs/impression.yaml", imp)
    good = prereg22.assess(run(lambda i: 0.1 if i < 75 else 0.9, lambda i: 0.9 if i < 75 else 0.1, 66, 63), 0.229)
    assert good["decision"]["record_model_decides"]
    assert "подтверждена" in prereg22.markdown_report(good)
    prereg22.apply_decision(good, str(imp))
    cfg = ImpressionConfig(**yaml.safe_load(imp.read_text()))
    assert cfg.record_model_decides and cfg.specificity == pytest.approx(75 / 95, abs=1e-3)
    bad = prereg22.assess(run(lambda i: 0.1 if i < 60 else 0.9, lambda i: 0.9 if i < 75 else 0.1, 66, 63), 0.229)
    assert not bad["hypotheses"]["H2_specificity_at_least_0.65"]
    prereg22.apply_decision(bad, str(imp))
    assert ImpressionConfig(**yaml.safe_load(imp.read_text())).record_model_decides is False


def test_record_threshold_by_age(configs):
    icfg = configs.impression
    assert icfg.threshold_for(5.0) > icfg.threshold_for(30.0)  # children need a higher threshold
    assert icfg.model_copy(update={"record_model_noage": None}).threshold_for(None) == pytest.approx(
        icfg.record_threshold)
    assert icfg.model_copy(update={"record_threshold_by_age": []}).threshold_for(5.0) == pytest.approx(
        icfg.record_threshold)


def test_record_model_only_on_routine_recordings(demo_edf, configs):
    from neurolens.pipeline.pipeline import Pipeline, is_routine_recording

    thr = configs.thresholds
    assert is_routine_recording(900, thr) and not is_routine_recording(3600, thr)
    assert not is_routine_recording(900, thr, "long_term") and is_routine_recording(7200, thr, "routine")
    pipe = Pipeline(config=configs, provider_pref="deterministic")
    ltm = pipe.analyze_file(demo_edf, patient=PatientInfo(age_years=55),
                            context=ClinicalContext(recording_setting="long_term"))
    assert ltm.result_json["record_score"] is None
    routine = pipe.analyze_file(demo_edf, patient=PatientInfo(age_years=55),
                                context=ClinicalContext(recording_setting="routine"))
    assert routine.result_json["record_score"] is not None


def test_record_model_reasons(configs):
    import json

    from neurolens.layer5_reasoning.record_model import RecordModel

    m = RecordModel.load(configs.ml_model_path(configs.impression.record_model))
    row = next(r for r in json.load(open("docs/increment22_eval.json"))["records"]
               if r["abnormal"] and r["record_probability"] > 0.8)
    why = m.reasons(row)
    assert 1 <= len(why) <= 3 and all(w["score_drop"] >= 0.02 for w in why)
    assert all(w["label_ru"] and "_" not in w["label_ru"] for w in why)


def test_borderline_record_score(configs):
    from neurolens.contracts.report import Bilingual
    from neurolens.pipeline.pipeline import apply_record_score

    class Rep:
        impression = Bilingual(ru="Заключение: x.", uz="Xulosa: x.")

    icfg = configs.impression.model_copy(update={"record_model_decides": True, "record_threshold_by_age": [],
                                                 "record_threshold": 0.3, "record_borderline_logit": 0.85})
    r = Rep()
    s = apply_record_score(r, {"impression_abnormal": False, "age_years": 30}, 0.35, icfg)
    assert s["borderline"] and r.impression.ru.startswith("Заключение: пограничная")
    r = Rep()
    s = apply_record_score(r, {"impression_abnormal": False, "age_years": 30}, 0.9, icfg)
    assert not s["borderline"] and s["abnormal"]
    off = icfg.model_copy(update={"record_borderline_logit": 0.0})
    assert not apply_record_score(Rep(), {"impression_abnormal": False, "age_years": 30}, 0.31, off)["borderline"]


def test_age_free_record_model_for_unknown_age(configs, demo_edf):
    from neurolens.layer5_reasoning.record_model import FEATURE_NAMES, RecordModel
    from neurolens.pipeline.pipeline import Pipeline

    icfg = configs.impression
    m0 = RecordModel.load(configs.ml_model_path(icfg.record_model_noage))
    assert "age_years" not in m0.model.feature_names and len(m0.idx) < len(FEATURE_NAMES)
    assert icfg.threshold_for(None) == pytest.approx(icfg.record_threshold_noage)
    out = Pipeline(config=configs, provider_pref="deterministic").analyze_file(
        demo_edf, context=ClinicalContext(recording_setting="routine"))
    assert out.result_json["record_score"]["model"].endswith("_noage")


def test_record_model_v2_has_no_absolute_amplitude(configs):
    from neurolens.layer5_reasoning.record_model import RecordModel

    for path in (configs.impression.record_model, configs.impression.record_model_noage):
        m = RecordModel.load(configs.ml_model_path(path))
        assert "robust_amplitude_uv" not in m.model.feature_names


def test_rule_numbers_when_rules_decide(configs):
    from neurolens.contracts.report import Bilingual
    from neurolens.pipeline.pipeline import _impression_with_validation

    imp = Bilingual(ru="Заключение: x.", uz="Xulosa: x.")
    icfg = configs.impression
    by_model = _impression_with_validation(imp, icfg, decided_by_record_model=True)
    by_rules = _impression_with_validation(imp, icfg, decided_by_record_model=False)
    assert f"{icfg.specificity:.2f}" in by_model.ru
    assert "правилами" in by_rules.ru and f"{icfg.rule_specificity:.2f}" in by_rules.ru


def test_bipds_need_a_field_on_each_side():
    from neurolens.layer4_detect.periodic import _side, has_field

    chans = ["F3", "C3", "T6", "Fp2"]
    left = [c for c in chans if _side(c) != "right"]
    right = [c for c in chans if _side(c) != "left"]
    assert has_field(chans) and has_field(left) and not has_field(right)


def test_noage_threshold_only_when_noage_model_used(configs):
    icfg = configs.impression
    assert icfg.threshold_for(None, noage_model=False) == pytest.approx(icfg.record_threshold)
    assert icfg.threshold_for(None, noage_model=True) == pytest.approx(icfg.record_threshold_noage)


def test_rhythmicity_offline_equals_streaming():
    import numpy as np

    from neurolens.layer3_features.rhythm_ac import epoch_rhythm
    from neurolens.layer4_detect.ml_ictal import (
        FEATURE_NAMES_V5, StreamingFeaturizerV5, _featurize_v1, rhythm_columns,
    )
    from types import SimpleNamespace

    fs, rng = 200.0, np.random.default_rng(1)
    t = np.arange(int(120 * fs)) / fs
    x = rng.standard_normal((4, t.size)) * 10
    x[:, int(60 * fs):int(80 * fs)] += 60 * np.sin(2 * np.pi * 2.0 * t[int(60 * fs):int(80 * fs)])
    times = np.arange(1.0, 119.0, 1.0)
    R = epoch_rhythm(x, fs, times)
    assert np.isnan(R[0]).all() and np.isfinite(R[20]).all()
    assert np.nanmean(R[62:78, 1]) > np.nanmean(R[20:50, 1]) + 0.3  # 2-Hz discharges are rhythmic
    n = len(times)
    feats = SimpleNamespace(epoch_rms=rng.uniform(5, 15, (n, 4)), epoch_band_conc=rng.uniform(0, 1, (n, 4)),
                            epoch_domfreq=rng.uniform(1, 10, (n, 4)),
                            epoch_relpow={b: rng.uniform(0, 1, (n, 4)) for b in ("delta", "theta", "alpha", "beta", "gamma")})
    off = np.column_stack([_featurize_v1(feats), rhythm_columns(R)])
    sf, rows = StreamingFeaturizerV5(), []
    for a in range(0, n, 7):
        b = min(n, a + 7)
        out = sf.push(feats.epoch_rms[a:b], feats.epoch_band_conc[a:b], feats.epoch_domfreq[a:b],
                      {k: v[a:b] for k, v in feats.epoch_relpow.items()}, R[a:b])
        rows.append(out)
    on = np.vstack([r for r in rows if len(r)])
    assert on.shape == off.shape and off.shape[1] == len(FEATURE_NAMES_V5)
    assert np.allclose(np.nan_to_num(on, nan=-9), np.nan_to_num(off, nan=-9))


def test_tuab_client_layout(tmp_path):
    """Increment 32: TUAB labels/split from the path, age from the EDF header."""
    from neurolens.datasets.tuab import TUABClient

    def edf(rel, patient=b"X M 01-JAN-1960 X Age:57"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"0       " + patient.ljust(80) + b" " * 168)
        return p

    edf("edf/eval/normal/01_tcp_ar/aaaaaaaa_s001_t000.edf")
    edf("edf/eval/abnormal/01_tcp_ar/aaaaaaab_s002_t001.edf", b"X F X")
    edf("edf/train/normal/01_tcp_ar/aaaaaaac_s001_t000.edf")
    (tmp_path / "README.edf").write_bytes(b"x")  # outside the layout: ignored
    ev = TUABClient(tmp_path).records("eval")
    assert [(r.name, r.abnormal, r.age_years) for r in ev] == [
        ("aaaaaaab_s002_t001", True, None), ("aaaaaaaa_s001_t000", False, 57.0)]
    assert len(TUABClient(tmp_path).records()) == 3 and ev[0].subject == "aaaaaaab"


def test_prereg32_assess_and_decision(tmp_path):
    from neurolens.evaluation.prereg32 import apply_decision, assess

    def row(ab, p, model, rule):
        return {"abnormal": ab, "record_probability": p, "impression_abnormal": model,
                "rule_impression_abnormal": rule, "record_borderline": False, "age_years": 40.0}
    rows = ([row(False, 0.1, False, False)] * 8 + [row(False, 0.6, True, True)] * 2
            + [row(True, 0.9, True, False)] * 8 + [row(True, 0.2, False, False)] * 2
            + [{"abnormal": True, "error": "x"}])
    a = assess(rows)
    assert a["n_scored"] == 20 and a["n_excluded"] == 1
    assert a["model"]["specificity"] == 0.8 and a["model"]["sensitivity"] == 0.8
    assert all(a["hypotheses"].values()) and a["decision"]["record_model_decides"]
    y = tmp_path / "imp.yaml"
    y.write_text("record_threshold: 0.25\nrecord_model_decides: true\n")
    a["decision"]["record_model_decides"] = False
    apply_decision(a, str(y))
    assert "record_model_decides: false" in y.read_text() and "record_threshold: 0.25" in y.read_text()


def test_eeglab_set_ingest(tmp_path):
    """Increment 33: EEGLAB .set (data inside, and in a companion .fdt)."""
    import numpy as np
    from scipy.io import savemat

    from neurolens.layer1_ingest.registry import ingest

    rng = np.random.default_rng(0)
    x = rng.normal(0, 20, (3, 1000)).astype(np.float32)
    locs = np.array([("Fp1",), ("C3",), ("O2",)], dtype=[("labels", "O")])
    eeg = {"nbchan": 3, "pnts": 1000, "srate": 250.0, "data": x, "chanlocs": locs, "ref": "A1 A2"}
    savemat(tmp_path / "a.set", {"EEG": eeg})
    s = ingest(tmp_path / "a.set")
    assert s.channel_names == ["Fp1", "C3", "O2"] and s.sampling_rate_hz == 250.0
    assert np.allclose(s.signal, x)
    x.T.tofile(tmp_path / "b.fdt")  # column-major float32
    savemat(tmp_path / "b.set", {"EEG": {**eeg, "data": "b.fdt"}})
    assert np.allclose(ingest(tmp_path / "b.set").signal, x)


def test_prereg33_assess():
    from neurolens.evaluation.prereg33 import assess

    def row(g, p, v):
        return {"group": g, "record_probability": p, "impression_abnormal": v,
                "rule_impression_abnormal": v, "record_borderline": False}
    rows = ([row("C", 0.1, False)] * 8 + [row("C", 0.5, True)] * 2 + [row("A", 0.8, True)] * 9
            + [row("A", 0.05, False)] + [row("F", 0.3, False)] * 3 + [{"group": "A", "error": "x"}])
    a = assess(rows)
    assert a["control_specificity"] == 0.8 and a["n_excluded"] == 1
    assert abs(a["auc_alzheimer_vs_controls"] - 0.9) < 1e-9 and all(a["hypotheses"].values())


def test_prereg34_assess():
    from neurolens.evaluation.prereg34 import assess

    def r(v, p=0.1):
        return {"impression_abnormal": v, "rule_impression_abnormal": v, "record_probability": p}
    a = assess([r(False)] * 7 + [r(True)] * 3, [r(False)] * 9 + [r(True)] + [{"error": "x"}])
    assert a["as_delivered"]["specificity"] == 0.7 and a["reduced_1020"]["n_excluded"] == 1
    assert a["hypotheses"]["H1_specificity_as_delivered_at_least_0.65"]
    assert a["decision"]["montage_gate_for_routine_verdict"]  # 0.9 - 0.7 > 0.10


def test_prereg35_assess():
    from neurolens.evaluation.prereg35 import assess

    def r(h, v, p, poor, amp=20.0, rule=None):
        return {"hospital": h, "impression_abnormal": v, "rule_impression_abnormal": v if rule is None else rule,
                "record_probability": p, "poor_outcome": poor, "amplitude_sd_uv": amp}
    rows = ([r("A", True, 0.9, True)] * 8 + [r("A", False, 0.1, False, rule=True)] * 2
            + [r("B", False, 0.1, True, amp=0.01)] * 3 + [{"patient": "x", "error": "e"}])
    a = assess(rows)
    assert a["amplitude_plausible_hospitals"] == ["A"] and a["n_excluded"] == 1
    assert a["main"]["sensitivity"] == 0.8 and a["auc_poor_vs_good"] == 1.0
    assert all(a["hypotheses"].values()) and a["decision"]["severe_finding_gate"]  # rules 1.0 vs 0.8


def test_wfdb_ingest_partial_and_segment_order(tmp_path):
    import numpy as np

    from neurolens.datasets.icare import candidate_order, segments
    from neurolens.layer1_ingest.registry import ingest

    d = np.array([[100, -50], [110, -40], [120, -30]], dtype="<i2")  # 3 samples x 2 channels
    (tmp_path / "r.mat").write_bytes(b"\0" * 24 + d.tobytes())
    (tmp_path / "r.hea").write_text("r 2 500 10\n"
                                    "r.mat 16+24 10.0(100)/nu 16 0 100 0 0 Fp1\n"
                                    "r.mat 16+24 2.0/uV 16 0 -50 0 0 Fp2\n#Utility frequency: 50\n")
    s = ingest(tmp_path / "r.hea")
    assert s.signal.shape == (2, 3) and s.channel_names == ["Fp1", "Fp2"]
    assert np.allclose(s.signal[0], [0, 1, 2]) and np.allclose(s.signal[1], [-25, -20, -15])
    html = 'href="0001_001_004_EEG.hea" href="0001_002_023_EEG.hea" href="0001_003_025_EEG.hea" href="0001_003_025_ECG.hea"'
    assert candidate_order(segments(html)) == ["0001_002_023_EEG", "0001_003_025_EEG", "0001_001_004_EEG"]


def test_icare_missing_age_is_none():
    from neurolens.datasets.icare import IcareRecord

    assert IcareRecord(0, "0592", {"Age": "nan"}).age_years is None
    assert IcareRecord(0, "0284", {"Age": "53"}).age_years == 53.0
