"""Delivery passports: built, sealed, and verified against tampering."""

from __future__ import annotations

import json
from datetime import date

from handoff.cli import main
from handoff.passport import build, verify


def _delivery(tmp_path):
    d = tmp_path / "delivery"
    d.mkdir()
    (d / "a.mp4").write_bytes(b"clip a")
    (d / "b.mp4").write_bytes(b"clip b")
    (d / "c.mp4").write_bytes(b"clip c")
    (d / "notes.txt").write_text("hello", encoding="utf-8")
    consent = tmp_path / "consent.csv"
    consent.write_text("clip,consent_id,expires\na.mp4,C-1,\nb.mp4,C-2,2026-01-01\n", encoding="utf-8")
    return d, consent


def test_passport_records_files_consent_and_licence(tmp_path):
    d, consent = _delivery(tmp_path)
    p = build(d, consent=consent, license="CC BY 4.0", today=date(2026, 10, 30))
    assert p["totals"] == {"files": 4, "bytes": 23, "clips": 3}  # three 6-byte clips and a 5-byte note
    assert p["license"] == "CC BY 4.0"
    assert p["consent"]["covered"] == 1
    assert p["consent"]["missing"] == ["c.mp4"] and p["consent"]["expired"] == ["b.mp4"]
    assert verify(p, d)["ok"]


def test_verify_catches_changed_missing_added_files_and_an_edited_passport(tmp_path):
    d, _ = _delivery(tmp_path)
    p = build(d)
    (d / "a.mp4").write_bytes(b"swapped")
    (d / "b.mp4").unlink()
    (d / "extra.mp4").write_bytes(b"new")
    r = verify(p, d)
    assert (r["changed"], r["missing"], r["added"]) == (["a.mp4"], ["b.mp4"], ["extra.mp4"])
    assert r["passport_intact"] and not r["ok"]

    p["license"] = "edited later"
    assert not verify(p, d)["passport_intact"]


def test_passport_cli_round_trip(tmp_path, capsys):
    d, consent = _delivery(tmp_path)
    out = d / "passport.json"  # written inside the delivery, and left out of its own file list
    assert main(["passport", str(d), "--license", "CC BY 4.0", "--out", str(out)]) == 0
    p = json.loads(out.read_text(encoding="utf-8"))
    assert "passport.json" not in {f["path"] for f in p["files"]}
    assert main(["verify-passport", str(out), str(d)]) == 0
    (d / "c.mp4").write_bytes(b"tampered")
    assert main(["verify-passport", str(out), str(d)]) == 1
    assert "changed: c.mp4" in capsys.readouterr().out
