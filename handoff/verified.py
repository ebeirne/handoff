"""Handoff-verified report cards: a check result signed with Handoff's private key, so anyone can confirm offline that
Handoff issued it and that nothing in it has changed since.

Anyone can run the checks and publish what they find; that's a self-check. A verified card is different: Handoff ran
the checks (and, for a vendor's sample, looked at the raw data behind it) and signs what it found. The signature
covers every field of the card. Changing a number, the data revision or the verdict breaks it. A card signed with any
key other than Handoff's published ones doesn't verify.

Signing needs Handoff's private key, which never leaves Handoff. Verifying needs only the public keys below and the
`cryptography` package (`pip install -e .[verify]`).
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

ISSUER = "Handoff (Ethan Beirne, github.com/ebeirne/handoff)"
# Handoff's public signing keys, by key id. A key is never removed, so cards it signed keep verifying.
PUBLIC_KEYS = {"handoff-2026-11": "2xSlXFJPz/w4sy7EiWD2ABWW1yIe1qL3vBKesHF2oFk="}


def _canonical(card: dict) -> bytes:
    """The bytes the signature covers: every field except the signature itself, keys sorted, no whitespace."""
    body = {k: v for k, v in card.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sign_card(card: dict, private_key_pem: str | Path, key_id: str) -> dict:
    """A copy of `card` stamped with the issuer, time and key id, and signed. Used by Handoff only."""
    from cryptography.hazmat.primitives import serialization

    key = serialization.load_pem_private_key(Path(private_key_pem).read_bytes(), password=None)
    signed = {**{k: v for k, v in card.items() if k != "signature"}, "issuer": ISSUER, "key_id": key_id,
              "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    signed["signature"] = base64.b64encode(key.sign(_canonical(signed))).decode()
    return signed


def verify_card(card: dict, keys: dict[str, str] | None = None) -> tuple[bool, str]:
    """(True, summary) when Handoff signed this card and nothing in it changed; (False, why) otherwise."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return False, "can't check signatures: install the verify extra (pip install -e .[verify])"
    keys = PUBLIC_KEYS if keys is None else keys
    if not card.get("signature"):
        return False, "not signed: this is a self-check, not a Handoff-verified card"
    key_b64 = keys.get(card.get("key_id", ""))
    if key_b64 is None:
        return False, f"signed with key {card.get('key_id')!r}, which isn't one of Handoff's published keys"
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(key_b64)).verify(
            base64.b64decode(card["signature"]), _canonical(card))
    except (InvalidSignature, ValueError):
        return False, "the signature doesn't match: the card was changed after Handoff signed it, or isn't Handoff's"
    s = card.get("summary", {})
    what = card.get("repo") or card.get("subject") or "this data"
    counts = f", {s.get('keep')} of {s.get('sessions')} episodes kept" if s.get("sessions") is not None else ""
    return True, (f"Handoff signed this card on {card.get('issued_at', '?')[:10]} for {what}"
                  f"{' at revision ' + card['revision'][:7] if card.get('revision') else ''}{counts}"
                  f"{' (scope: ' + card['scope'] + ')' if card.get('scope') else ''}")
