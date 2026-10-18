"""Scanning regulation intake check (docs/scanning_regulation.md)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from neurolens.layer1_ingest.scan_check import ScanForm, check_scan, estimate_skew_deg, gate_reasons


def _page(w=2400, h=1600, n=8, angle=0.0):
    """White page, n sinusoidal traces with a baseline grid, optional rotation."""
    img = np.full((h, w), 255, np.uint8)
    x = np.arange(w)
    for k in range(n):
        y0 = int((k + 0.5) * h / n)
        img[y0, :] = 150  # grid / baseline
        y = (y0 + 25 * np.sin(2 * np.pi * x / 120.0)).astype(int)
        for d in (-1, 0, 1):
            img[np.clip(y + d, 0, h - 1), x] = 0
    im = Image.fromarray(img)
    return im.rotate(angle, fillcolor=255, resample=Image.BILINEAR) if angle else im


FORM = dict(paper_speed_mm_s=30, sensitivity_uv_per_mm=7, channel_labels=["Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
            "Fp2-F4", "F4-C4", "C4-P4", "P4-O2"], montage="longitudinal bipolar", calibration_pulse_uv=50,
            highpass_hz=0.5, lowpass_hz=70, notch_hz=50, page_index=1, n_pages=1)


def test_good_scan_accepted(tmp_path):
    p = tmp_path / "p.png"
    _page().save(p, dpi=(600, 600))
    rep = check_scan(p, ScanForm(**FORM))
    assert rep.accepted, [c for c in rep.checks if c.status == "reject"]
    assert rep.derived["ms_per_px"] == pytest.approx(1000 / (30 * 600 / 25.4), rel=1e-3)
    assert rep.derived["max_frequency_hz"] > 100


@pytest.mark.parametrize("kind", ["jpeg", "lowdpi", "nodpi", "skew", "camera", "nocal"])
def test_bad_scans_rejected(tmp_path, kind):
    form = dict(FORM)
    im, ext, kw = _page(), "png", {"dpi": (600, 600)}
    if kind == "jpeg":
        ext, kw = "jpg", {"dpi": (600, 600), "quality": 90}
    elif kind == "lowdpi":
        kw = {"dpi": (150, 150)}
    elif kind == "nodpi":
        kw = {}
    elif kind == "skew":
        im = _page(angle=1.5)
    elif kind == "camera":
        form["scanner"] = "camera"
    elif kind == "nocal":
        form["calibration_pulse_uv"] = None
    p = tmp_path / f"p.{ext}"
    im.save(p, **kw)
    rep = check_scan(p, ScanForm(**form))
    assert not rep.accepted
    names = {c.name for c in rep.checks if c.status == "reject"}
    expected = {"jpeg": "format", "lowdpi": "dpi", "nodpi": "dpi", "skew": "skew", "camera": "scanner",
                "nocal": "calibration_pulse"}[kind]
    assert expected in names


def test_slow_paper_needs_more_dpi(tmp_path):
    """15 mm/s at 300 dpi: 5.6 ms per pixel > 4 ms -> rejected on time resolution."""
    p = tmp_path / "p.png"
    _page().save(p, dpi=(300, 300))
    rep = check_scan(p, ScanForm(**{**FORM, "paper_speed_mm_s": 15}))
    assert any(c.name == "time_resolution" and c.status == "reject" for c in rep.checks)


def test_skew_estimate():
    assert abs(estimate_skew_deg(np.asarray(_page(angle=1.0)))) == pytest.approx(1.0, abs=0.25)
    assert abs(estimate_skew_deg(np.asarray(_page()))) <= 0.15


def test_digitized_signal_closes_mode_b():
    assert gate_reasons(False) == [] and len(gate_reasons(True)) == 1


def test_cli_scan_check(tmp_path):
    from neurolens.cli import main

    p, f = tmp_path / "p.png", tmp_path / "f.json"
    _page().save(p, dpi=(600, 600))
    f.write_text(json.dumps(FORM))
    assert main(["scan-check", str(p), str(f), "--out", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["accepted"] is True


def test_pipeline_digitized_signal_gate(demo_edf, configs):
    from neurolens.contracts.report import OperatingMode
    from neurolens.contracts.signal import ClinicalContext, PatientInfo
    from neurolens.layer1_ingest.registry import ingest
    from neurolens.pipeline.pipeline import Pipeline

    sig = ingest(demo_edf, patient=PatientInfo(age_years=55),
                 context=ClinicalContext(recording_setting="long_term"))
    sig = sig.model_copy(update={"provenance": sig.provenance.model_copy(update={"digitized": True})})
    out = Pipeline(config=configs, provider_pref="deterministic").analyze_signal(sig, mode=OperatingMode.B_AUTONOMOUS)
    assert out.gate.passed is False
    assert any("оцифрован с изображения" in r for r in out.gate.reasons_ru)
