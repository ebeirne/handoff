"""Capture-time watch: clips are checked once they stop growing, exactly once, with reasons."""

from __future__ import annotations

import json

from handoff.spec import load_spec
from handoff.synthetic import write_clip
from handoff.watch import Watcher, describe


def test_watch_checks_new_clips_once_they_stop_growing(tmp_path, spec_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    log = tmp_path / "watch.jsonl"
    w = Watcher(inbox, load_spec(spec_path), log=log)

    write_clip(inbox / "good.avi", "good")
    assert w.poll() == []                      # first sight: wait until the size holds still
    first = w.poll()
    assert [r["clip"] for r in first] == ["good.avi"] and first[0]["verdict"] == "pass"
    assert w.poll() == []                      # never checked twice

    write_clip(inbox / "black.avi", "black")
    w.poll()
    black = w.poll()
    assert black[0]["verdict"] == "flag"
    assert any(f["check"] == "black_frames" for f in black[0]["failed"])
    assert "black_frames" in describe(black[0])

    lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert [x["clip"] for x in lines] == ["good.avi", "black.avi"]


def test_watch_waits_while_a_file_is_still_being_written(tmp_path, spec_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    w = Watcher(inbox, load_spec(spec_path))
    partial = inbox / "upload.avi"
    partial.write_bytes(b"\x00" * 100)
    assert w.poll() == []
    partial.write_bytes(b"\x00" * 200)         # still growing
    assert w.poll() == []
    out = w.poll()                             # stable now; unreadable, so it's rejected with a reason
    assert out[0]["verdict"] == "reject" and out[0]["failed"][0]["check"] == "readable"
