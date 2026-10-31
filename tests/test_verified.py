"""Handoff-verified cards: signed by Handoff's key, checkable offline, broken by any change."""

from __future__ import annotations

import base64
import json

import pytest

from handoff import verified
from handoff.cli import main

crypto = pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

CARD = {"card": "handoff-verified/1", "repo": "acme/sample", "revision": "a285cd1249fe", "tool_version": "0.1.0",
        "scope": "public sample plus one raw shoot day", "summary": {"sessions": 100, "keep": 97, "review": 3, "drop": 0},
        "fingerprint": "f" * 64}


def _key(tmp_path, name="k.pem"):
    key = Ed25519PrivateKey.generate()
    path = tmp_path / name
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    pub = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return path, base64.b64encode(pub).decode()


def test_a_signed_card_verifies_and_any_change_breaks_it(tmp_path):
    pem, pub = _key(tmp_path)
    keys = {"test-key": pub}
    card = verified.sign_card(CARD, pem, "test-key")
    ok, message = verified.verify_card(card, keys)
    assert ok and "acme/sample" in message and "97 of 100" in message

    for field, value in [("summary", {**CARD["summary"], "keep": 100}), ("revision", "0000000"), ("scope", "everything")]:
        assert not verified.verify_card({**card, field: value}, keys)[0]
    assert not verified.verify_card({**card, "extra": "added later"}, keys)[0]


def test_a_card_not_signed_by_a_published_key_does_not_verify(tmp_path):
    pem, _ = _key(tmp_path)
    _, other_pub = _key(tmp_path, "other.pem")
    card = verified.sign_card(CARD, pem, "test-key")
    ok, message = verified.verify_card(card, {"test-key": other_pub})  # same key id, someone else's key
    assert not ok and "doesn't match" in message
    assert "isn't one of Handoff's published keys" in verified.verify_card(card, {})[1]
    assert "self-check" in verified.verify_card(CARD, {"test-key": other_pub})[1]  # never signed at all


def test_verify_card_command(tmp_path, monkeypatch, capsys):
    pem, pub = _key(tmp_path)
    monkeypatch.setattr(verified, "PUBLIC_KEYS", {"test-key": pub})
    good = tmp_path / "card.json"
    good.write_text(json.dumps(verified.sign_card(CARD, pem, "test-key")), encoding="utf-8")
    assert main(["verify-card", str(good)]) == 0
    assert capsys.readouterr().out.startswith("VERIFIED")

    forged = tmp_path / "forged.json"
    forged.write_text(json.dumps({**json.loads(good.read_text()), "summary": {"sessions": 100, "keep": 100}}),
                      encoding="utf-8")
    assert main(["verify-card", str(forged)]) == 1
    assert capsys.readouterr().out.startswith("NOT VERIFIED")
