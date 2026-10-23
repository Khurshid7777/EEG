import math

from neurolens.layer6_interpret.protocol import render_protocol
from neurolens.pipeline.serial import change, compare, compare_text, derive_limits


def _s(pdr=10.0, d=0.2, t=0.2, a=0.4, amp=20.0, det=()):
    return {"pdr_hz": pdr, "global_rel_delta": d, "global_rel_theta": t, "global_rel_alpha": a,
            "robust_amplitude_uv": amp, "detectors_fired": list(det)}


def test_change():
    assert change("pdr_hz", 10.0, 9.0) == -1.0
    assert change("pdr_hz", 0.0, 9.0) is None                  # PDR not found in the prior
    assert math.isclose(change("global_rel_alpha", 0.4, 0.2), math.log(0.5))
    assert change("global_rel_alpha", None, 0.2) is None


def test_derive_and_compare():
    pairs = [(_s(), _s(pdr=10 + e, a=0.4 * math.exp(e / 10))) for e in (-0.5, 0.5, -0.3, 0.3)]
    lim = {"limits": derive_limits(pairs), "source": "test"}
    assert math.isclose(lim["limits"]["pdr_hz"]["limit"], 2.576 * math.sqrt((0.25 * 2 + 0.09 * 2) / 4), rel_tol=1e-3)
    same = compare(_s(), _s(pdr=10.2), lim)
    assert not same["any_exceeds"]
    slowed = compare(_s(), _s(pdr=8.0, a=0.2, det=["spike"]), lim)
    rows = {r["quantity"]: r for r in slowed["quantities"]}
    assert rows["pdr_hz"]["exceeds"] and rows["global_rel_alpha"]["exceeds"]
    assert slowed["detectors_new"] == ["spike"]
    txt = compare_text(slowed)
    assert "10 → 8 Гц" in txt.ru and "больше повседневной изменчивости" in txt.ru and "spike" in txt.ru


def test_no_limits_no_flags():
    c = compare(_s(), _s(pdr=6.0), None)
    assert not c["any_exceeds"] and "не заданы" in compare_text(c).ru


def test_protocol_section():
    from neurolens.contracts.report import Bilingual, Confidence, LLMReport, OperatingMode

    rep = LLMReport(mode=OperatingMode.A_DECISION_SUPPORT, impression=Bilingual(ru="x", uz="x"),
                    overall_confidence=Confidence(value=0.5, label_ru="с", label_uz="o"))
    assert "Сравнение с прошлой ЭЭГ" in render_protocol(rep, serial=compare(_s(), _s(pdr=9.0), None))
