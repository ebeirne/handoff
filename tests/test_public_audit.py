"""The public-audit report card, badge, fingerprint and index, without touching the network."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("public_audit", ROOT / "tools" / "public_audit.py")
pa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pa)

SOURCE = {"repo": "acme/pick_place", "revision": "a" * 40, "license": "apache-2.0", "tables": 1, "tables_total": 3,
          "mb": 12.5, "files": [{"path": "data/chunk-000/file-000.parquet", "size": 12_500_000, "sha256": "b" * 64}]}


def _report(verdicts: dict[str, list[str]]) -> dict:
    sessions = [{"session": name, "verdict": "review" if checks else "keep",
                 "issues": [{"level": "review", "check": c, "message": f"{c} happened", "at_s": []} for c in checks]}
                for name, checks in verdicts.items()]
    counts: dict[str, int] = {}
    for s in sessions:
        for c in {i["check"] for i in s["issues"]}:
            counts[c] = counts.get(c, 0) + 1
    keep = sum(s["verdict"] == "keep" for s in sessions)
    return {"sessions": sessions, "issues": counts,
            "summary": {"sessions": len(sessions), "keep": keep, "review": len(sessions) - keep, "drop": 0,
                        "hours": {"keep": 0.5, "review": 0.1, "drop": 0.0}, "median_lag_frames": 1}}


def test_fingerprint_depends_on_verdicts_and_revision_only():
    a = _report({"episode_000000": [], "episode_000001": ["action_lag"]})
    b = json.loads(json.dumps(a))
    b["sessions"][1]["issues"][0]["message"] = "worded differently"  # wording isn't part of the verdict
    assert pa.fingerprint(SOURCE, a) == pa.fingerprint(SOURCE, b)
    assert pa.fingerprint({**SOURCE, "revision": "c" * 40}, a) != pa.fingerprint(SOURCE, a)
    c = _report({"episode_000000": [], "episode_000001": []})
    assert pa.fingerprint(SOURCE, c) != pa.fingerprint(SOURCE, a)


def test_badge_escapes_the_slash_and_colors_by_share_flagged():
    clean = pa.badge(SOURCE, _report({f"e{i}": [] for i in range(20)}))
    assert "20%2F20_episodes_clean-brightgreen" in clean and "{report_url}" in clean
    mixed = pa.badge(SOURCE, _report({**{f"e{i}": [] for i in range(7)}, **{f"f{i}": ["untracked"] for i in range(3)}}))
    assert "7%2F10_episodes_clean-orange" in mixed


def test_report_card_explains_each_issue_and_how_to_reproduce():
    html = pa.report_card(SOURCE, _report({"episode_000000": [], "episode_000001": ["untracked"]}), "f" * 64, "")
    assert "the dataset marks the state as not measured" in html
    assert "episode_000001" in html and "a" * 40 in html and "f" * 64 in html
    assert "first 1 of 3 data files" in html  # a sampled audit says so
    assert "1 frame here" in html and "Contact" not in html


def test_report_card_says_which_checks_dont_apply_to_a_dataset_without_actions():
    report = _report({"episode_000000": []})
    report["spec"] = {"required": ["observation.state"], "absent_streams": ["action"]}
    html = pa.report_card(SOURCE, report, "f" * 64, "")
    assert "doesn't record action" in html and "lag and copied-state checks don't apply" in html
    assert "marks as not measured" not in html and "tracking or visibility flags" in html  # none were read here

    report["sessions"][0]["facts"] = {"untracked": {"stream": "observation.state", "fraction": 0.0}}
    html = pa.report_card(SOURCE, report, "f" * 64, "")
    assert "marks as not measured" in html and "tracking or visibility flags" not in html


def test_index_counts_datasets_and_lists_failures(tmp_path):
    ok = tmp_path / "acme__pick_place"
    ok.mkdir()
    rep = _report({"episode_000000": [], "episode_000001": ["action_lag"]})
    (ok / "audit.json").write_text(json.dumps({"repo": "acme/pick_place", "source": SOURCE, "summary": rep["summary"],
                                               "issues": rep["issues"], "fingerprint": "f" * 64}), encoding="utf-8")
    bad = tmp_path / "acme__gated"
    bad.mkdir()
    (bad / "audit.json").write_text(json.dumps({"repo": "acme/gated", "error": "PermissionError: gated"}), encoding="utf-8")
    index = pa.write_index(tmp_path).read_text(encoding="utf-8")
    assert "acme/pick_place" in index and "PermissionError: gated" in index
    rows = (tmp_path / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 3 and "50.0" in rows[1] or "50.0" in rows[2]


def test_fetch_reads_only_the_tables_it_lists(tmp_path, monkeypatch):
    """A cached table from an earlier, larger run would be read by the audit but missing from its listing."""
    tables = [{"path": f"data/chunk-000/file-00{i}.parquet", "size": 60_000_000} for i in range(3)]
    monkeypatch.setattr(pa, "_json", lambda url: {"sha": "a" * 40, "cardData": {"license": "mit"}})
    monkeypatch.setattr(pa, "_tree", lambda repo, rev, path: tables)
    monkeypatch.setattr(pa, "_get", lambda url: b"x" * 60_000_000 if url.endswith(".parquet") else b"{}")

    local = tmp_path / "acme__pick_place"
    pa.fetch("acme/pick_place", tmp_path, max_mb=200)
    assert len(list(local.rglob("*.parquet"))) == 3

    source = pa.fetch("acme/pick_place", tmp_path, max_mb=100)
    assert source["tables"] == 1
    assert [p.name for p in local.rglob("*.parquet")] == ["file-000.parquet"]
