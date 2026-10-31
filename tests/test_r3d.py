"""Record3D .r3d clips (iPhone LiDAR RGB-D): frames, per-frame timing and depth confidence."""

from __future__ import annotations

import json
import zipfile

import cv2
import numpy as np

from handoff import video
from handoff.run import run
from handoff.spec import load_spec

SPEC = """version: 1
name: r3d-test
sampling: {fps: 2, max_frames: 100}
checks:
  frame_timing: {max_dropped_fraction: 0.05, max_gap_s: 0.25}
  depth: {min_high_confidence: 0.3}
"""


def _r3d(path, stamps, conf_level=2, fps=30):
    """A Record3D clip: textured JPEG frames, a raw confidence map per frame, and metadata with the claimed fps."""
    rng = np.random.default_rng(0)
    dw, dh = 16, 12
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("metadata", json.dumps({"fps": fps, "w": 64, "h": 48, "dw": dw, "dh": dh,
                                           "frameTimestamps": [1000.0 + s for s in stamps]}))
        for i in range(len(stamps)):
            ok, jpg = cv2.imencode(".jpg", rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
            z.writestr(f"rgbd/{i}.jpg", jpg.tobytes())
            z.writestr(f"rgbd/{i}.conf", np.full(dw * dh, conf_level, np.uint8).tobytes())


def _clips(tmp_path):
    t = np.arange(150) / 30.0
    (tmp_path / "clips").mkdir()
    _r3d(tmp_path / "clips" / "good.r3d", t.tolist())
    _r3d(tmp_path / "clips" / "half_rate.r3d", t[::2].tolist())             # claims 30 fps, records a steady 15
    _r3d(tmp_path / "clips" / "slows_down.r3d", np.r_[t[:75], 2.5 + np.arange(38) / 15].tolist())  # 30, then 15
    _r3d(tmp_path / "clips" / "stall.r3d", np.r_[t[:60], t[90:]].tolist())  # a 1-second stall
    _r3d(tmp_path / "clips" / "no_depth.r3d", t.tolist(), conf_level=0)     # the sensor trusts none of it
    spec = tmp_path / "spec.yaml"
    spec.write_text(SPEC, encoding="utf-8")
    return load_spec(spec), tmp_path / "clips"


def _checks(result):
    return {v.clip: {c.name: c for c in v.checks} for v in result.verdicts}


def test_frame_timing_catches_rate_changes_and_stalls(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "_lzfse", lambda: (lambda b: b))  # the maps here are stored raw
    spec, clips = _clips(tmp_path)
    result = run(spec, clips)
    c = _checks(result)
    assert c["good.r3d"]["frame_timing"].passed and c["good.r3d"]["depth"].passed

    half = c["half_rate.r3d"]["frame_timing"]  # steady, just not the metadata's rate: reported, and fps sees 15
    assert half.passed and "steady 15 fps, 0.0% dropped (the file's metadata says 30 fps)" in half.reason
    assert {v.clip: v.info["fps"] for v in result.verdicts}["half_rate.r3d"] == 15

    slows = c["slows_down.r3d"]["frame_timing"]
    assert slows.passed is False and "frame rate changes mid-clip: 30 fps until 0:03, then 15 fps" in slows.reason
    assert slows.evidence == [[3.0, 3.0]]

    stall = c["stall.r3d"]["frame_timing"]
    assert stall.passed is False and "1 stalls, longest 1.03s at 0:02" in stall.reason and stall.evidence == [[2.0, 3.0]]


def test_depth_check_flags_depth_the_sensor_doesnt_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "_lzfse", lambda: (lambda b: b))
    spec, clips = _clips(tmp_path)
    depth = _checks(run(spec, clips))["no_depth.r3d"]["depth"]
    assert depth.passed is False and depth.value == 0.0 and "check the LiDAR" in depth.reason


def test_depth_check_says_what_to_install_when_it_cant_read_the_maps(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "_lzfse", lambda: None)
    spec, clips = _clips(tmp_path)
    c = _checks(run(spec, clips))
    depth = c["no_depth.r3d"]["depth"]
    assert depth.passed is None and "install pyliblzfse" in depth.reason
    assert c["stall.r3d"]["frame_timing"].passed is False  # timing needs no extra library


def test_r3d_frames_feed_the_ordinary_checks(tmp_path, monkeypatch):
    clip = tmp_path / "a.r3d"
    _r3d(clip, (np.arange(90) / 30.0).tolist())
    s = video.sample_clip(clip, target_fps=2, max_frames=100)
    assert s.info.readable and s.info.fps == 30 and s.info.frame_count == 90 and abs(s.info.duration_s - 2.967) < 0.01
    assert len(s.frames) == 6 and s.frames[0].shape == (48, 64, 3) and len(s.bursts) == 6

    def corrupt(_):
        raise ValueError("bad block")

    monkeypatch.setattr(video, "_lzfse", lambda: corrupt)
    s = video.sample_clip(clip, target_fps=2, max_frames=100)
    assert len(s.frames) == 6 and s.depth_conf is None and "couldn't be decoded" in s.depth_note
    bad = tmp_path / "broken.r3d"
    bad.write_bytes(b"not a zip")
    assert not video.sample_clip(bad, 2, 100).info.readable
