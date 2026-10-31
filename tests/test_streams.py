from __future__ import annotations

import json

import numpy as np
import pytest

from handoff.checks import check_streams, streams_file
from handoff.video import ClipInfo, Sample

PARAMS = {"required": ["video", "imu", "actions"], "max_start_offset_s": 0.05, "max_gap_s": 0.25, "min_coverage": 0.98}


def _sample(tmp_path, streams: dict | None):
    clip = tmp_path / "clip.mp4"
    if streams is not None:
        streams_file(clip).write_text(json.dumps({"streams": {k: list(v) for k, v in streams.items()}}), encoding="utf-8")
    return Sample(ClipInfo(clip, True), [], [])


def _good():
    return {
        "video": np.arange(0, 10, 1 / 30),
        "imu": np.arange(0.01, 10, 1 / 200),
        "actions": np.arange(0.02, 10, 1 / 10),
    }


def test_streams_in_sync_pass(tmp_path):
    r = check_streams(_sample(tmp_path, _good()), PARAMS)
    assert r.passed is True and "3 streams in sync" in r.reason


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda s: s.pop("actions"), "missing streams: actions"),
        (lambda s: s.update(imu=np.concatenate([s["imu"][:400], s["imu"][600:]])), "imu drops data"),
        (lambda s: s.update(actions=s["actions"] + 0.5), "streams start 0.52s apart (actions starts last)"),
        (lambda s: s.update(video=s["video"][: len(s["video"]) // 2]), "video stops early"),
        (lambda s: s.update(imu=s["imu"][::-1]), "imu timestamps go backward"),
    ],
)
def test_stream_problems_are_named(tmp_path, mutate, expect):
    streams = _good()
    mutate(streams)
    r = check_streams(_sample(tmp_path, streams), PARAMS)
    assert r.passed is False and expect in r.reason, r.reason


def test_gap_has_timestamps(tmp_path):
    streams = _good()
    streams["imu"] = np.concatenate([streams["imu"][:400], streams["imu"][600:]])  # ~1 s hole around 2 s
    r = check_streams(_sample(tmp_path, streams), PARAMS)
    assert r.evidence == [[2.0, 3.0]]


def test_missing_file_fails_only_when_streams_are_required(tmp_path):
    assert check_streams(_sample(tmp_path, None), PARAMS).passed is False
    assert check_streams(_sample(tmp_path, None), {}).passed is None
