import json
import re

import numpy as np

from neurolens.layer3_features.regional import STANDARD_1020
from neurolens.outputs.regional_view import (
    build_view,
    load_norms,
    plot_regional_static,
    regional_summary,
    render_html,
)


def _signal(secs=60, fs=200.0):
    rng = np.random.default_rng(3)
    t = np.arange(int(secs * fs)) / fs
    ch = list(STANDARD_1020)
    x = 5 * rng.standard_normal((len(ch), t.size))
    for c in ("O1", "O2"):
        x[ch.index(c)] += 20 * np.sin(2 * np.pi * 10 * t)
    return x, ch, fs


RESULT = {"events": [{"code": "spike", "t_start": 12.0, "t_end": 12.1, "confidence": 0.8, "channels": ["T3", "Fz"]},
                     {"code": "spike", "t_start": 30.0, "t_end": 30.1, "confidence": 0.6, "channels": []}],
          "causal_graph": {"nodes": [{"event_code": "spike", "label_ru": "Спайк"}]},
          "report": {"impression": {"ru": "Заключение RU", "uz": "Xulosa UZ"}}, "critical_findings": []}


def test_view_regions_events_and_asymmetry_flags():
    x, ch, fs = _signal()
    v = build_view(x, ch, fs, RESULT)
    assert v["series"]["n_windows"] == 6 and len(v["pairs"]) == 7
    where = v["events"][0]["where"]
    assert {"ch": "T3", "region": "temporal", "side": "L"} in where and {"ch": "Fz", "region": "frontal", "side": "mid"} in where
    assert v["events"][1]["where"] == [] and v["events"][0]["label"] == "Спайк"
    assert set(v["asymmetry_record"]) == set(v["pairs"])
    assert all(r["flag"] in (None, "right>left", "left>right") for r in v["asymmetry_record"].values())
    s = regional_summary(v)
    assert s["relative_power_by_region"]["alpha"]["occipital"] > s["relative_power_by_region"]["alpha"]["frontal"]
    assert s["events_without_channels"] == 1 and s["events_by_region"]["temporal_L"] == 1
    assert "not source localisation" in s["level"]


def test_asymmetry_flag_outside_normative_limits():
    x, ch, fs = _signal()
    x[ch.index("O1")] *= 2                                   # left occipital much stronger (still < 200 uV) -> left>right
    v = build_view(x, ch, fs, RESULT)
    assert v["asymmetry_record"]["O1-O2"]["flag"] == "left>right"
    assert load_norms()["version"] == "regional_norms_v1"


def test_html_is_self_contained_and_data_roundtrips(tmp_path):
    x, ch, fs = _signal()
    v = build_view(x, ch, fs, RESULT, title="t</script>x")
    html = render_html(v)
    assert "/*__DATA__*/null" not in html and "<script src" not in html and "http://" not in html.replace("http://www.w3.org", "")
    m = re.search(r"const V=(\{.*?\});\n\(\(\)", html, re.DOTALL)
    back = json.loads(m.group(1))
    assert back["title"] == "t</script>x" and back["series"]["n_windows"] == 6
    assert html.count("</script>") == 1                      # the data cannot close the script tag
    png = plot_regional_static(v, tmp_path / "s.png")
    assert png.stat().st_size > 10_000


def test_pipeline_adds_regional_block():
    from neurolens.contracts.signal import UnifiedSignal
    from neurolens.pipeline.pipeline import Pipeline

    x, ch, fs = _signal(secs=40)
    out = Pipeline(provider_pref="template", run_ica=False).analyze_signal(
        UnifiedSignal(signal=x, sampling_rate_hz=fs, channel_names=ch, duration_s=40.0, source="edf"))
    r = out.result_json["regional"]
    assert r["version"] == "regional_view_v1" and set(r["asymmetry_alpha"]) >= {"O1-O2", "T3-T4"}


def test_sparse_psg_montage_keeps_recorded_reference_and_has_no_corridor():
    rng = np.random.default_rng(9)
    fs, secs = 200.0, 90
    ch = ["F3", "F4", "C3", "C4", "O1", "O2"]
    x = 10 * rng.standard_normal((6, int(fs * secs)))
    v = build_view(x, ch, fs, {})
    assert v["series"]["reference"] == "recorded" and v["record"]["reference"] == "recorded"
    rec = v["asymmetry_record"]["C3-C4"]
    assert rec["low"] is None and rec["flag"] is None
    assert "log10_abs" in v["series"] and "sigma" in v["series"]["rel"]
    import pytest

    from neurolens.layer3_features.regional import regional_series
    with pytest.raises(ValueError):
        regional_series(x[:2], ch[:2], fs)


def test_ten_ten_names_flat_and_nan_channels():
    from neurolens.layer3_features.regional import canonical, regional_series

    x, ch, fs = _signal(secs=60)
    ch2 = [{"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}.get(c, c) for c in ch]
    s = regional_series(x, ch2, fs)
    assert {"T3", "T4", "T5", "T6"} <= set(s["channels"]) and canonical("P8") == "T6"
    flat = x.copy()
    flat[ch.index("Cz")] = 0.0                                   # flat electrode is dropped, not used as a reference leak
    s2 = regional_series(flat, ch, fs)
    assert "Cz" not in s2["channels"] and len(s2["channels"]) == len(ch) - 1
    nan = x.copy()
    nan[ch.index("F3"), 2000:2400] = np.nan                       # short gap: kept, the window is flagged
    s3 = regional_series(nan, ch, fs)
    assert "F3" in s3["channels"] and any(s3["artefact"]) and np.isfinite(np.array(s3["rel"]["alpha"])).all()
    nan[ch.index("F4"), :] = np.nan                               # mostly missing: dropped
    assert "F4" not in regional_series(nan, ch, fs)["channels"]


def test_page_script_parses_with_node(tmp_path):
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node not available")
    x, ch, fs = _signal(secs=40)
    html = render_html(build_view(x, ch, fs, RESULT))
    js = re.search(r"<script>(.*)</script>", html, re.DOTALL).group(1)
    f = tmp_path / "page.js"
    f.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr


def test_cli_regional_writes_html_and_json(tmp_path):
    from neurolens.cli import _generate_demo_edf, main

    edf = _generate_demo_edf(tmp_path / "demo.edf")
    html, js = tmp_path / "v.html", tmp_path / "v.json"
    assert main(["regional", str(edf), "--html", str(html), "--json", str(js)]) == 0
    assert "regional_view_v1" in html.read_text(encoding="utf-8")
    assert json.loads(js.read_text())["version"] == "regional_view_v1"


def test_page_escapes_labels_channels_and_title():
    x, ch, fs = _signal(secs=40)
    res = {"events": [{"code": "spike", "t_start": 12.0, "t_end": 12.1, "confidence": 0.8,
                       "channels": ["<img src=x onerror=1>", "T3"]}],
           "causal_graph": {"nodes": [{"event_code": "spike", "label_ru": "<img src=x onerror=1>"}]}}
    html = render_html(build_view(x, ch, fs, res, title="<script>1</script>"))
    assert "const esc=" in html and "esc(evLbl(e))" in html and "esc(e.channels.join" in html
    assert html.count("<script>") == 1 and html.count("</script>") == 1      # title/label text cannot open or close a script
