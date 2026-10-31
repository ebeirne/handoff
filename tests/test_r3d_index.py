"""A corrected index for a folder of Record3D clips, read from the clips rather than the vendor's metadata."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np

from handoff import video
from tests.test_r3d import _r3d

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("r3d_index", ROOT / "tools" / "r3d_index.py")
ri = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ri)


def test_index_measures_each_clip_and_catches_a_wrong_vendor_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(video, "_lzfse", lambda: (lambda b: b))  # the test maps are stored raw
    t = np.arange(150) / 30.0
    clips = tmp_path / "clips" / "day1"
    clips.mkdir(parents=True)
    _r3d(clips / "good.r3d", t.tolist())
    _r3d(clips / "half_rate.r3d", t[::2].tolist())                                  # claims 30, records 15
    _r3d(clips / "slows_down.r3d", np.r_[t[:75], 2.5 + np.arange(38) / 15].tolist())  # 30, then 15
    _r3d(clips / "stall.r3d", np.r_[t[:60], t[90:]].tolist())                      # a 1-second stall
    _r3d(clips / "no_depth.r3d", t.tolist(), conf_level=0)
    (clips / "broken.r3d").write_bytes(b"not a zip")
    vendor = tmp_path / "vendor.csv"  # the vendor's index says good.r3d has no usable depth
    vendor.write_text("clip_id,depth_conf_high\ngood,0\nno_depth,0\nstall,0.9\n", encoding="utf-8")

    out = tmp_path / "index.csv"
    assert ri.main([str(tmp_path / "clips"), "--out", str(out), "--compare", str(vendor), "--workers", "1"]) == 0
    rows = {r["clip_id"]: r for r in csv.DictReader(open(out, encoding="utf-8"))}

    assert rows["good"]["flags"] == "" and float(rows["good"]["depth_high_share"]) == 1.0
    assert rows["good"]["file"] == "day1/good.r3d"
    assert rows["half_rate"]["flags"] == "" and float(rows["half_rate"]["recorded_fps"]) == 15
    assert "rate_change" in rows["slows_down"]["flags"]
    assert "stall" in rows["stall"]["flags"]
    assert rows["no_depth"]["flags"] == "low_depth"
    assert rows["broken"]["flags"] == "unreadable" and rows["broken"]["error"]

    summary = capsys.readouterr().out
    assert "6 clips" in summary and "1 unreadable" in summary
    assert "steady rate unlike the metadata's (reported, not flagged): 1 clips" in summary
    # good.r3d: the vendor says 0, the maps say 100% high confidence. no_depth agrees, and stall isn't a depth issue.
    assert "vendor index disagrees on usable depth: 1 clips" in summary


def test_a_map_that_wont_decompress_costs_the_depth_reading_not_the_clip(tmp_path, monkeypatch):
    def broken(_):
        raise RuntimeError("lzfse: bad block")

    monkeypatch.setattr(video, "_lzfse", lambda: broken)
    _r3d(tmp_path / "clip.r3d", (np.arange(90) / 30.0).tolist())
    row = ri.clip_row(tmp_path / "clip.r3d", tmp_path)
    assert "error" not in row and row["frames"] == 90
    assert "depth_high_share" not in row and row["depth_note"] == "maps couldn't be read"
