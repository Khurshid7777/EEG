"""Real-time cEEG monitoring with alarms (TZ §2 sub-mode, §4, §10, §16).

Streaming pipeline: a :class:`StreamSource` yields buffered, overlapping windows;
the :class:`RealtimeMonitor` runs a lightweight subset of Layers 2-4 per window
and raises :class:`~neurolens.contracts.alarms.Alarm` s via the
:class:`AlarmManager`, which controls false-alarms/hour (persistence, refractory,
per-hour caps) and attaches an EEG fragment to each alarm.

Sources: a recording-file replay (offline, deterministic) and a live Lab Streaming Layer inlet
(:mod:`neurolens.realtime.lsl`, needs ``pylsl``, which bundles liblsl; D82).
"""

from __future__ import annotations

from .stream import Chunk, StreamSource, EdfReplaySource
from .buffer import WindowBuffer
from .alarms import AlarmManager
from .monitor import RealtimeMonitor

__all__ = [
    "Chunk",
    "StreamSource",
    "EdfReplaySource",
    "WindowBuffer",
    "AlarmManager",
    "RealtimeMonitor",
]
