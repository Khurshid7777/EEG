"""Layer 5 causal-reasoning tests."""

from __future__ import annotations

from neurolens.contracts.causal import PhysiologyLabel
from neurolens.contracts.events import DetectionResult, Event, EventEvidence, Localization
from neurolens.contracts.signal import ClinicalContext, PatientInfo
from neurolens.layer2_preprocess.artifacts import ArtifactReport
from neurolens.layer3_features.norms import NormsEngine
from neurolens.layer5_reasoning.physiology import PhysiologyEngine
from neurolens.layer5_reasoning.rules_engine import build_causal_graph


def _engine(configs):
    return PhysiologyEngine(NormsEngine(configs.norms))


def _event(code, group, **kw):
    return Event(code=code, label_ru=code, label_uz=code, group=group,
                 localization=Localization(), t_start=0.0, t_end=10.0,
                 confidence=kw.get("confidence", 0.8),
                 evidence=kw.get("evidence", []),
                 is_artifact_hypothesis=kw.get("artifact", False))


def test_diffuse_slowing_creates_encephalopathy_edge(configs):
    det = DetectionResult(events=[_event("diffuse_slowing", "background",
                                         evidence=[EventEvidence(feature="pdr_hz", value=5.0)])])
    graph = build_causal_graph(det, ClinicalContext(), PatientInfo(age_years=50),
                               configs.rule_base, _engine(configs))
    rule_ids = {e.rule_id for e in graph.edges}
    assert "diffuse_slowing_encephalopathy" in rule_ids


def test_ecg_artifact_resolves_spikes(configs):
    spike = _event("spike", "ied", confidence=0.5)
    spike.localization = Localization(channels=["T3"])
    det = DetectionResult(events=[_event("ecg_artifact", "artifact", artifact=True), spike])
    art = ArtifactReport(ecg_present=True, heart_rate_hz=1.0,
                         ecg_qrs_times_s=[5.0], ecg_contaminated_channels=["T3"])
    graph = build_causal_graph(det, ClinicalContext(), PatientInfo(age_years=50),
                               configs.rule_base, _engine(configs), art)
    # spike node re-labeled as artifact
    spike_nodes = [n for n in graph.nodes if n.event_code == "spike"]
    assert spike_nodes and spike_nodes[0].physiology == PhysiologyLabel.ARTIFACT
    # pseudo-IED mechanism node exists
    assert any(n.id == "mech:pseudo_ied" for n in graph.nodes)


def test_burst_suppression_is_critical(configs):
    det = DetectionResult(events=[_event("burst_suppression", "suppression")])
    graph = build_causal_graph(det, ClinicalContext(sedatives=["propofol"]),
                               PatientInfo(age_years=50), configs.rule_base, _engine(configs))
    assert graph.has_critical()


# --- increment 64: context conditions and hedged conclusions (docs/rule_base_audit_2026-11-19.md) -------------------
def _rules(configs, code, group, **ctx):
    det = DetectionResult(events=[_event(code, group)])
    g = build_causal_graph(det, ClinicalContext(**ctx), PatientInfo(age_years=60), configs.rule_base, _engine(configs))
    return g, {e.rule_id for e in g.edges}


def test_triphasic_waves_conclusion_depends_on_hepatic_or_renal_context(configs):
    _, none = _rules(configs, "triphasic_waves", "special")
    assert "triphasic_nonspecific" in none and not {"triphasic_metabolic_hepatic", "triphasic_metabolic_renal"} & none
    _, hep = _rules(configs, "triphasic_waves", "special", symptoms=["печёночная недостаточность"])
    assert "triphasic_metabolic_hepatic" in hep and "triphasic_nonspecific" not in hep
    _, ren = _rules(configs, "triphasic_waves", "special", symptoms=["chronic kidney disease, uremia"])
    assert "triphasic_metabolic_renal" in ren and "triphasic_nonspecific" not in ren
    _, both = _rules(configs, "triphasic_waves", "special", symptoms=["liver failure", "renal failure"])
    assert {"triphasic_metabolic_hepatic", "triphasic_metabolic_renal"} <= both and "triphasic_nonspecific" not in both


def test_burst_suppression_without_context_is_not_called_severe_injury(configs):
    g, _ = _rules(configs, "burst_suppression", "suppression")
    text = " ".join(n.label_ru for n in g.nodes if n.id == "mech:burst_suppression_context")
    assert "контекст не указан" in text and "приоритет тяжёлого повреждения" not in text
    g2, _ = _rules(configs, "burst_suppression", "suppression", sedatives=["propofol"])
    assert "седация/гипотермия" in " ".join(n.label_ru for n in g2.nodes if n.id == "mech:burst_suppression_context")


def test_no_rule_claims_raised_icp_as_a_conclusion(configs):
    for chain in configs.rule_base.chains:
        assert "повышение ВЧД" not in chain.effect_ru.replace("не доказана", "") or "не доказана" in chain.effect_ru
        assert "ВЧД" not in chain.effect_ru, chain.id


def test_extreme_delta_brush_is_gated_by_postmenstrual_age(configs):
    """Increment 65: delta brush in a preterm neonate is maturation, not pathology; adults/children keep the old label."""
    eng = _engine(configs)
    ev = _event("extreme_delta_brush", "special")
    assert eng.label_event(ev, PatientInfo(postmenstrual_age_weeks=32)) == PhysiologyLabel.PHYSIOLOGIC
    assert eng.label_event(ev, PatientInfo(postmenstrual_age_weeks=40)) == PhysiologyLabel.UNCERTAIN
    assert eng.label_event(ev, PatientInfo(age_years=25)) == PhysiologyLabel.PATHOLOGIC
    assert eng.label_event(ev, PatientInfo(age_years=0.5)) == PhysiologyLabel.PATHOLOGIC
    det = DetectionResult(events=[ev])
    g = build_causal_graph(det, ClinicalContext(), PatientInfo(postmenstrual_age_weeks=32), configs.rule_base, eng)
    mech = [n for n in g.nodes if n.id == "mech:edb_specific"]
    assert mech and mech[0].physiology == PhysiologyLabel.PHYSIOLOGIC and "нормальный признак созревания" in mech[0].label_ru
    assert not g.has_critical()
    adult = build_causal_graph(det, ClinicalContext(), PatientInfo(age_years=25), configs.rule_base, eng)
    assert [n for n in adult.nodes if n.id == "mech:edb_specific"][0].physiology == PhysiologyLabel.PATHOLOGIC


def test_every_rule_trigger_code_is_emitted_by_a_detector(configs):
    """Increment 64: a rule whose trigger event no detector emits can never fire (audit finding: drowsiness_slowing)."""
    import glob
    import re

    src = "".join(open(f, encoding="utf-8").read() for f in glob.glob("src/neurolens/layer4_detect/*.py")
                 if not f.endswith("base.py"))
    for chain in configs.rule_base.chains:
        for code in list(chain.when.any_event) + list(chain.when.all_events):
            assert re.search(r'"%s"' % re.escape(code), src), f"{chain.id}: no detector emits {code}"
