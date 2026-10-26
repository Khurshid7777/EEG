"""Live LSL input of the real-time monitor (D82): a recording published over LSL on this machine
must give the monitor exactly what a file replay gives."""

import threading
import time
import uuid

import numpy as np
import pytest

pylsl = pytest.importorskip("pylsl")

from neurolens.realtime.lsl import LslStreamSource, stream_recording  # noqa: E402
from neurolens.realtime.monitor import RealtimeMonitor  # noqa: E402
from neurolens.realtime.stream import EdfReplaySource  # noqa: E402


def _outlet(name, labels, units, fs=100.0):
    info = pylsl.StreamInfo(name, "EEG", len(labels), fs, "float32", name)
    chans = info.desc().append_child("channels")
    for lab, u in zip(labels, units):
        c = chans.append_child("channel")
        c.append_child_value("label", lab)
        if u:
            c.append_child_value("unit", u)
    return pylsl.StreamOutlet(info)


def test_units_names_and_gaps():
    name = f"t-{uuid.uuid4().hex[:8]}"
    out = _outlet(name, ["EEG FP1-REF", "O2", "X"], ["volts", "microvolts", ""])

    def push():
        time.sleep(1.0)
        t = pylsl.local_clock()
        for k in range(10):                      # 10 chunks x 10 samples at 100 Hz; skip 0.5 s after chunk 4
            ts0 = t + k * 0.1 + (0.5 if k >= 5 else 0.0)
            out.push_chunk(np.ones((10, 3), np.float32).tolist(), [ts0 + i / 100 for i in range(10)])
            time.sleep(0.05)

    threading.Thread(target=push, daemon=True).start()
    src = LslStreamSource(stream_name=name, chunk_s=0.1, idle_timeout_s=2.0)
    assert src.channel_names == ["Fp1", "O2", "X"]
    data = np.concatenate([c.data for c in src.chunks()], axis=1)
    assert data.shape == (3, 100)
    assert np.allclose(data[:, 0], [1e6, 1.0, 1.0])
    assert len(src.gaps) == 1 and abs(src.gaps[0][1] - 0.5) < 0.02
    w = " ".join(src.warnings())
    assert "lsl_units_assumed_uV:X" in w and "lsl_gaps:1" in w


def test_monitor_over_lsl_matches_file_replay(demo_edf):
    name = f"demo-{uuid.uuid4().hex[:8]}"
    th = threading.Thread(target=stream_recording, args=(demo_edf,),
                          kwargs={"name": name, "speed": 40.0, "start_delay_s": 0.5}, daemon=True)
    th.start()
    src = LslStreamSource(stream_name=name, chunk_s=0.5, idle_timeout_s=3.0)
    live = RealtimeMonitor().run(src)
    ref = RealtimeMonitor().run(EdfReplaySource(demo_edf))
    th.join(timeout=5)
    assert live.n_windows == ref.n_windows
    assert [(a.type, round(a.t_start, 1)) for a in live.alarms] == [(a.type, round(a.t_start, 1)) for a in ref.alarms]
    assert src.gaps == [] and src.source_name == f"lsl:{name}"
