"""The console page pulls every output together."""

from __future__ import annotations

import json

from handoff.cli import main


def test_console_combines_every_kind_of_output(tmp_path):
    check = tmp_path / "results.json"
    check.write_text(json.dumps({"spec": {"name": "egocentric-v1"}, "complete": True, "fingerprint": "ab" * 32,
                                 "summary": {"clips": 20, "pass": 18, "downgrade": 0, "flag": 2, "reject": 0}}), encoding="utf-8")
    fleet = tmp_path / "fleet.json"
    fleet.write_text(json.dumps({"summary": {"sessions": 9, "keep": 4, "review": 3, "drop": 2,
                                             "hours": {"keep": 0.5, "review": 0.3, "drop": 0.1}},
                                 "issues": {"dropout": 2}, "by_robot": {}}), encoding="utf-8")
    repair = tmp_path / "REPAIR_LOG.json"
    repair.write_text(json.dumps({"repaired_episodes": [{"episode": 1}, {"episode": 3}], "offsets_from": {"tool": "rdq"}}), encoding="utf-8")
    passport = tmp_path / "passport.json"
    passport.write_text(json.dumps({"name": "batch-7", "totals": {"files": 21, "bytes": 1, "clips": 20}, "license": "CC BY 4.0",
                                    "consent": {"clips": 20, "covered": 19, "missing": ["x"], "expired": []},
                                    "passport_sha256": "cd" * 32}), encoding="utf-8")
    watch = tmp_path / "watch.jsonl"
    watch.write_text('{"clip": "a.mp4", "verdict": "pass", "failed": []}\n{"clip": "b.mp4", "verdict": "flag", "failed": []}\n', encoding="utf-8")

    out = tmp_path / "console.html"
    assert main(["console", "--check", str(check), "--fleet", str(fleet), "--repair", str(repair),
                 "--passport", str(passport), "--watch", str(watch), "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    for text in ("18 of 20", "4 of 9", "0.50 of 0.90", "batch-7", "19 of 20", "rdq", "Capture-time checks"):
        assert text in page, text
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["repairs"][0]["repaired"] == 2


def test_console_needs_at_least_one_input(tmp_path):
    assert main(["console", "--out", str(tmp_path / "c.html")]) == 2
