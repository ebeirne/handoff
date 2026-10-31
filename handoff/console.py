"""One page over everything Handoff produced: deliveries checked, fleet sessions triaged, episodes repaired,
deliveries sealed with passports, and clips checked at capture time.

    handoff console --check reports/x/results.json --fleet reports/fleet/fleet.json \
        --repair data/fixed/REPAIR_LOG.json --passport passport.json --watch watch_log.jsonl --out console.html
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .pages import cards, esc, page


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _watch(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build(checks=(), fleets=(), repairs=(), passports=(), watches=()) -> dict:
    data = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool_version": __version__,
            "checks": [], "fleets": [], "repairs": [], "passports": [], "watches": []}
    for p in checks:
        r = _load(p)
        data["checks"].append({"source": str(p), "spec": (r.get("spec") or {}).get("name") if isinstance(r.get("spec"), dict) else r.get("spec"),
                               "summary": r.get("summary", {}), "complete": r.get("complete"), "fingerprint": r.get("fingerprint")})
    for p in fleets:
        r = _load(p)
        data["fleets"].append({"source": str(p), "summary": r["summary"], "issues": r.get("issues", {}), "by_robot": r.get("by_robot", {})})
    for p in repairs:
        r = _load(p)
        data["repairs"].append({"source": str(p), "repaired": len(r.get("repaired_episodes", [])),
                                "offsets_from": (r.get("offsets_from") or {}).get("tool")})
    for p in passports:
        r = _load(p)
        data["passports"].append({"source": str(p), "name": r.get("name"), "totals": r.get("totals"), "license": r.get("license"),
                                  "consent": {k: r["consent"][k] for k in ("clips", "covered")} if "consent" in r else None,
                                  "seal": r.get("passport_sha256")})
    for p in watches:
        rows = _watch(p)
        data["watches"].append({"source": str(p), "checked": len(rows),
                                "flagged": sum(r["verdict"] in ("flag", "reject") for r in rows)})
    return data


def to_html(d: dict) -> str:
    clips = sum(c["summary"].get("clips", 0) for c in d["checks"])
    passed = sum(c["summary"].get("pass", 0) for c in d["checks"])
    sessions = sum(f["summary"]["sessions"] for f in d["fleets"])
    keep = sum(f["summary"]["keep"] for f in d["fleets"])
    usable_h = sum(f["summary"]["hours"]["keep"] for f in d["fleets"])
    total_h = sum(sum(f["summary"]["hours"].values()) for f in d["fleets"])
    body = cards([("delivered clips passed", f"{passed} of {clips}"), ("fleet sessions kept", f"{keep} of {sessions}"),
                  ("usable fleet hours", f"{usable_h:.2f} of {total_h:.2f}"),
                  ("episodes repaired", sum(r["repaired"] for r in d["repairs"])),
                  ("deliveries sealed", len(d["passports"])),
                  ("clips checked at capture", sum(w["checked"] for w in d["watches"]))])

    if d["checks"]:
        body += "<h2>Deliveries</h2><table><tr><th>Results</th><th>Spec</th><th>Pass</th><th>Downgrade</th><th>Flag</th><th>Reject</th><th>Complete</th></tr>"
        for c in d["checks"]:
            s = c["summary"]
            body += (f'<tr><td class="mono">{esc(c["source"])}</td><td>{esc(c["spec"])}</td><td class="keep">{s.get("pass", 0)}</td>'
                     f'<td>{s.get("downgrade", 0)}</td><td class="review">{s.get("flag", 0)}</td><td class="drop">{s.get("reject", 0)}</td>'
                     f'<td>{esc(c["complete"])}</td></tr>')
        body += "</table>"
    if d["fleets"]:
        body += "<h2>Fleet</h2><table><tr><th>Report</th><th>Keep</th><th>Review</th><th>Drop</th><th>Top issues</th></tr>"
        for f in d["fleets"]:
            s = f["summary"]
            top = ", ".join(f"{k} ({v})" for k, v in list(f["issues"].items())[:4]) or "none"
            body += (f'<tr><td class="mono">{esc(f["source"])}</td><td class="keep">{s["keep"]}</td><td class="review">{s["review"]}</td>'
                     f'<td class="drop">{s["drop"]}</td><td>{esc(top)}</td></tr>')
        body += "</table>"
    if d["repairs"]:
        body += "<h2>Repairs</h2><table><tr><th>Log</th><th>Episodes repaired</th><th>Offsets from</th></tr>" + "".join(
            f'<tr><td class="mono">{esc(r["source"])}</td><td>{r["repaired"]}</td><td>{esc(r["offsets_from"] or "handoff scan")}</td></tr>'
            for r in d["repairs"]) + "</table>"
    if d["passports"]:
        body += "<h2>Passports</h2><table><tr><th>Delivery</th><th>Files</th><th>Licence</th><th>Consent</th><th>Seal</th></tr>"
        for p in d["passports"]:
            consent = f'{p["consent"]["covered"]} of {p["consent"]["clips"]}' if p["consent"] else "not checked"
            body += (f'<tr><td>{esc(p["name"])}</td><td>{p["totals"]["files"]}</td><td>{esc(p["license"] or "not stated")}</td>'
                     f'<td>{esc(consent)}</td><td class="mono">{esc((p["seal"] or "")[:16])}</td></tr>')
        body += "</table>"
    if d["watches"]:
        body += "<h2>Capture-time checks</h2><table><tr><th>Log</th><th>Clips checked</th><th>Flagged or rejected</th></tr>" + "".join(
            f'<tr><td class="mono">{esc(w["source"])}</td><td>{w["checked"]}</td><td class="review">{w["flagged"]}</td></tr>'
            for w in d["watches"]) + "</table>"
    return page("Handoff console", f"robot data operations · tool v{d['tool_version']} · {d['created_at']}", body)
