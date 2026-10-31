"""Frame timing of Record3D clips, read through HTTP ranges, without touching the network."""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("r3d_timing", ROOT / "tools" / "r3d_timing.py")
rt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rt)


def test_timing_tells_a_steady_rate_from_a_change_and_from_drops():
    t = np.arange(300) / 30.0
    r = rt.timing(t.tolist(), 30.0)
    assert r["dropped_frames"] == 0 and r["rate_changes"] == 0 and r["recorded_fps"] == 30

    every_other = t[::2]  # metadata says 30 fps, the phone records a steady 15: not a drop
    r = rt.timing(every_other.tolist(), 30.0)
    assert r["recorded_fps"] == 15 and r["rate_changes"] == 0 and r["dropped_frames"] == 0

    slows = np.r_[t[:150], 5.0 + np.arange(75) * 2 / 30]  # 30 fps for 5 s, then 15
    r = rt.timing(slows.tolist(), 30.0)
    assert r["rate_changes"] == 1 and r["rates"] == "30@0s > 15@5s" and r["dropped_frames"] == 0

    stall = np.r_[t[:100], t[160:]]  # a 2-second stall
    r = rt.timing(stall.tolist(), 30.0)
    assert r["dropped_frames"] == 60 and r["gaps_over_0.5s"] == 1 and r["longest_gap_s"] == 2.033


def test_clip_timing_reads_only_the_metadata_through_ranges(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("rgbd/0.jpg", b"\xff" * 200_000)  # stands in for hundreds of MB of frames
        z.writestr("metadata", json.dumps({"fps": 60, "frameTimestamps": (np.arange(120) / 60).tolist()}))
    blob = buf.getvalue()
    fetched = []

    class Response:
        def __init__(self, data, headers=None):
            self.data, self.headers = data, headers or {}

        def read(self):
            return self.data

    def urlopen(req, timeout=None):
        if req.get_method() == "HEAD":
            return Response(b"", {"Content-Length": str(len(blob))})
        a, b = req.headers["Range"].removeprefix("bytes=").split("-")
        fetched.append(int(b) - int(a) + 1)
        return Response(blob[int(a): int(b) + 1])

    monkeypatch.setattr(rt.urllib.request, "urlopen", urlopen)
    r = rt.clip_timing("acme/raw", "main", "clips/a.r3d")
    assert r["frames"] == 120 and r["recorded_fps"] == 60 and r["dropped_frames"] == 0
    assert sum(fetched) < len(blob) / 10  # the frames were never downloaded
