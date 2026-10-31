"""A delivery passport: what was delivered, under what consent and licence, and whether it passed, in a form anyone
holding the files can verify.

It lists every file with its SHA-256 and size, ties in the acceptance results (fingerprint and verdict counts) when
given, checks every clip against a consent sheet, and records the licence. The passport is hashed as well, so an edit
to it shows. `handoff verify-passport` re-hashes the files and reports anything changed, missing or added.

    handoff passport delivery/ --results reports/x/results.json --consent consent.csv --license "CC BY 4.0"
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from . import __version__
from .pages import cards, esc, page
from .video import VIDEO_EXTENSIONS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seal(passport: dict) -> str:
    body = {k: v for k, v in passport.items() if k != "passport_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _files(folder: Path, skip: set[Path]) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.resolve() not in skip)


def _consent(path: Path, clips: list[str], today: date) -> dict:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = {r["clip"].strip(): r for r in csv.DictReader(fh) if r.get("clip")}
    missing, expired = [], []
    for clip in clips:
        row = rows.get(clip) or rows.get(Path(clip).name)
        if not row or not (row.get("consent_id") or "").strip():
            missing.append(clip)
        elif (row.get("expires") or "").strip() and date.fromisoformat(row["expires"].strip()) < today:
            expired.append(clip)
    return {"sheet": path.name, "sheet_sha256": _sha256(path), "clips": len(clips),
            "covered": len(clips) - len(missing) - len(expired), "missing": missing, "expired": expired}


def build(folder: str | Path, results: str | Path | None = None, consent: str | Path | None = None,
          license: str | None = None, name: str | None = None, skip: list[str | Path] = (), today: date | None = None) -> dict:
    folder = Path(folder)
    files = _files(folder, {Path(p).resolve() for p in skip})
    entries = [{"path": p.relative_to(folder).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files]
    clips = [e["path"] for e in entries if Path(e["path"]).suffix.lower() in VIDEO_EXTENSIONS]
    passport = {
        "passport_version": 1,
        "tool": "handoff passport", "tool_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": name or folder.name,
        "license": license,
        "files": entries,
        "totals": {"files": len(entries), "bytes": sum(e["bytes"] for e in entries), "clips": len(clips)},
    }
    if results:
        r = json.loads(Path(results).read_text(encoding="utf-8"))
        passport["acceptance"] = {"fingerprint": r.get("fingerprint"), "spec": r.get("spec", {}).get("name") if isinstance(r.get("spec"), dict) else r.get("spec"),
                                  "complete": r.get("complete"), "summary": r.get("summary"),
                                  "results_sha256": _sha256(Path(results))}
    if consent:
        passport["consent"] = _consent(Path(consent), clips, today or date.today())
    passport["passport_sha256"] = _seal(passport)
    return passport


def verify(passport: dict, folder: str | Path, skip: list[str | Path] = ()) -> dict:
    folder = Path(folder)
    intact = passport.get("passport_sha256") == _seal(passport)
    recorded = {e["path"]: e for e in passport["files"]}
    present = {p.relative_to(folder).as_posix(): p for p in _files(folder, {Path(s).resolve() for s in skip})}
    changed = sorted(k for k in recorded.keys() & present.keys() if _sha256(present[k]) != recorded[k]["sha256"])
    missing = sorted(recorded.keys() - present.keys())
    added = sorted(present.keys() - recorded.keys())
    return {"passport_intact": intact, "changed": changed, "missing": missing, "added": added,
            "ok": intact and not (changed or missing or added)}


def to_html(p: dict) -> str:
    items = [("files", p["totals"]["files"]), ("clips", p["totals"]["clips"]), ("GB", f"{p['totals']['bytes'] / 1e9:.2f}")]
    body = ""
    if "consent" in p:
        c = p["consent"]
        items.append(("clips with consent", f"{c['covered']} of {c['clips']}"))
    body += cards(items)
    body += f'<p>Licence: <b>{esc(p.get("license") or "not stated")}</b></p>'
    if "acceptance" in p:
        a = p["acceptance"]
        body += (f'<h2>Acceptance</h2><p>Spec <b>{esc(a.get("spec"))}</b>, complete: {esc(a.get("complete"))}, '
                 f'fingerprint <span class="mono">{esc(a.get("fingerprint"))}</span></p>'
                 f'<p>{esc(json.dumps(a.get("summary")))}</p>')
    if "consent" in p and (p["consent"]["missing"] or p["consent"]["expired"]):
        body += "<h2>Consent gaps</h2><ul>" + "".join(f'<li class="drop">{esc(x)}: no consent on file</li>' for x in p["consent"]["missing"]) \
            + "".join(f'<li class="review">{esc(x)}: consent expired</li>' for x in p["consent"]["expired"]) + "</ul>"
    body += f'<h2>Seal</h2><p class="mono">{esc(p["passport_sha256"])}</p>'
    body += "<h2>Files</h2><table><tr><th>Path</th><th>Bytes</th><th>SHA-256</th></tr>" + "".join(
        f'<tr><td class="mono">{esc(e["path"])}</td><td>{e["bytes"]}</td><td class="mono">{esc(e["sha256"])}</td></tr>'
        for e in p["files"]) + "</table>"
    return page(f"Delivery passport: {p['name']}", f"tool v{p['tool_version']} · {p['created_at']}", body)
