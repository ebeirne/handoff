"""Audit public LeRobot datasets on Hugging Face and write a one-page report card for each.

Downloads only what fleet triage reads (meta/info.json and the data/ parquet tables, never the videos), runs
`handoff fleet` on them, and writes, per dataset, the triage outputs plus:

- audit.html: a report card a dataset owner can read in two minutes and share
- audit.json: the same facts, with the data revision, every file's SHA-256 and a verdict fingerprint
- badge.md: a shields.io badge snippet for the dataset card

and, across every dataset audited so far, index.html and summary.csv.

This script talks to the network (the Hugging Face Hub). The `handoff` package itself never does.

    python tools/public_audit.py lerobot/svla_so100_pickplace lerobot/svla_so100_stacking
    python tools/public_audit.py --top 25          # the most-downloaded LeRobot datasets published by organizations
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from handoff import __version__  # noqa: E402
from handoff.fleet import lerobot_defaults, lerobot_fit, load_fleet_spec, load_logs, triage, write_outputs  # noqa: E402
from handoff.pages import cards, esc, page  # noqa: E402

HUB = "https://huggingface.co"
UA = {"User-Agent": f"handoff-public-audit/{__version__}"}

# What each check means, for someone who has never seen Handoff.
PLAIN = {
    "unreadable": "the episode couldn't be read at all",
    "missing_stream": "the state or action stream is missing",
    "too_short": "too short to use (under 3 s)",
    "no_motion": "the state never moves: a frozen recording",
    "untracked": "the dataset marks the state as not measured (e.g. hand not tracked) in most frames",
    "mostly_idle": "the robot is idle for most of the episode",
    "action_lag": "actions are out of step with the measured state, compared with the rest of the dataset (repairable)",
    "state_copies_action": "the state is a copy of the earlier command, not a measurement",
    "dropout": "gaps in the recording",
    "low_rate": "logged below its expected rate",
    "late_start": "a stream starts late",
    "stops_early": "a stream stops early",
    "truncated": "the recording ends abruptly",
}


def _open(url: str, tries: int = 5) -> tuple[bytes, dict]:
    """GET with retries on rate limits, server errors and dropped connections. Other HTTP errors (404, 401) raise."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt * 3)
    raise RuntimeError("unreachable")


def _get(url: str) -> bytes:
    return _open(url)[0]


def _json(url: str):
    return json.loads(_get(url))


def _tree(repo: str, rev: str, folder: str) -> list[dict]:
    """Every file under `folder`, following the Hub's paginated tree listing."""
    url = f"{HUB}/api/datasets/{repo}/tree/{rev}/{folder}?recursive=true"
    files = []
    while url:
        body, headers = _open(url)
        files += [f for f in json.loads(body) if f.get("type") == "file"]
        link = headers.get("Link") or headers.get("link") or ""
        url = next((p.split(";")[0].strip(" <>") for p in link.split(",") if 'rel="next"' in p), None)
    return files


def top_datasets(n: int, include_lerobot: bool = False) -> list[str]:
    """The most-downloaded LeRobot-format datasets whose author is an organization, not an individual."""
    found, orgs = [], {}
    listing = _json(f"{HUB}/api/datasets?filter=LeRobot&sort=downloads&direction=-1&limit=400")
    for d in listing:
        author = d["id"].split("/")[0]
        if author == "lerobot" and not include_lerobot:
            continue
        if author not in orgs:
            try:
                _get(f"{HUB}/api/organizations/{urllib.parse.quote(author)}/overview")
                orgs[author] = True
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
                orgs[author] = False  # an individual (404), or the Hub wouldn't say: leave it out
            time.sleep(0.3)
        if orgs[author]:
            found.append(d["id"])
        if len(found) >= n:
            break
    return found


def author_datasets(author: str, n: int = 1) -> list[str]:
    """An organization's most-downloaded LeRobot datasets (falling back to any dataset if none is tagged LeRobot)."""
    q = urllib.parse.quote(author)
    for extra in ("&filter=LeRobot", ""):
        try:
            found = [d["id"] for d in _json(f"{HUB}/api/datasets?author={q}{extra}&sort=downloads&direction=-1&limit={n}")]
        except (urllib.error.URLError, http.client.HTTPException, ConnectionError, TimeoutError):
            found = []
        if found:
            return found
    return []


def fetch(repo: str, data_root: Path, max_mb: float) -> dict:
    """Download meta/info.json and the data tables (up to `max_mb`). Returns what was fetched and from which revision."""
    info = _json(f"{HUB}/api/datasets/{repo}")
    if info.get("gated") or info.get("private"):
        raise PermissionError("gated or private: needs the owner's permission")
    rev = info["sha"]
    files = sorted(_tree(repo, rev, "data"), key=lambda f: f["path"])
    tables = [f for f in files if f["path"].endswith(".parquet")]
    if not tables:
        raise ValueError("no parquet tables under data/ (not a LeRobot v2/v3 dataset?)")
    if tables[0]["size"] > 2 * max_mb * 1e6:
        raise ValueError(f"first data table is {tables[0]['size'] / 1e6:.0f} MB, too large for a quick audit (raise --max-mb)")
    picked, total = [], 0
    for f in tables:
        if picked and total + f["size"] > max_mb * 1e6:
            break
        picked.append(f)
        total += f["size"]
    local = data_root / repo.replace("/", "__")
    keep = {f["path"] for f in picked}
    for old in (local / "data").rglob("*.parquet") if (local / "data").is_dir() else ():
        if old.relative_to(local).as_posix() not in keep:  # left by an earlier run: the audit reads only what it lists
            old.unlink()
    for path in ["meta/info.json"] + [f["path"] for f in picked]:
        dst = local / path
        size = next((f["size"] for f in picked if f["path"] == path), None)
        if dst.exists() and (size is None or dst.stat().st_size == size):
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(_get(f"{HUB}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"))
    card = info.get("cardData") or {}
    return {"repo": repo, "revision": rev, "local": str(local), "license": card.get("license") or "not stated",
            "downloads": info.get("downloads"), "last_modified": info.get("lastModified"),
            "tables": len(picked), "tables_total": len(tables), "mb": round(total / 1e6, 1),
            "files": [{"path": f["path"], "size": f["size"], "sha256": (f.get("lfs") or {}).get("oid")} for f in picked]}


def fingerprint(source: dict, report: dict) -> str:
    """Same data revision, same tool version, same verdicts: same fingerprint, on any machine."""
    body = {"repo": source["repo"], "revision": source["revision"], "tool_version": __version__,
            "sessions": [[s["session"], s["verdict"], sorted({i["check"] for i in s["issues"]})] for s in report["sessions"]]}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def badge(source: dict, report: dict) -> str:
    s = report["summary"]
    flagged = (s["review"] + s["drop"]) / max(s["sessions"], 1)
    color = "brightgreen" if flagged <= 0.05 else "yellow" if flagged <= 0.2 else "orange"
    label = urllib.parse.quote(f"{s['keep']}/{s['sessions']} episodes clean", safe="").replace("%20", "_")
    return (f"[![Handoff audit: {s['keep']} of {s['sessions']} episodes clean]"
            f"(https://img.shields.io/badge/Handoff_audit-{label}-{color})]({{report_url}})\n")


def report_card(source: dict, report: dict, fp: str, contact: str) -> str:
    s = report["summary"]
    total_h = sum(s["hours"].values())
    body = cards([("episodes", s["sessions"]), ("keep", s["keep"]), ("review", s["review"]), ("drop", s["drop"]),
                  ("usable hours", f"{s['hours']['keep']:.2f} of {total_h:.2f}")])
    sampled = "" if source["tables"] == source["tables_total"] else \
        f" This audit read the first {source['tables']} of {source['tables_total']} data files."
    if report["issues"]:
        rows = []
        for check, n in report["issues"].items():
            examples = [(r["session"], next(i["message"] for i in r["issues"] if i["check"] == check))
                        for r in report["sessions"] if any(i["check"] == check for i in r["issues"])][:3]
            ex = "".join(f'<li><span class="mono">{esc(e)}</span>: {esc(m)}</li>' for e, m in examples)
            rows.append(f"<tr><td><b>{esc(check)}</b><br><span class='muted'>{esc(PLAIN.get(check, ''))}</span></td>"
                        f"<td>{n}</td><td><ul>{ex}</ul></td></tr>")
        body += ("<h2>What we found</h2><table><tr><th>Issue</th><th>Episodes</th><th>Examples</th></tr>"
                 + "".join(rows) + "</table>")
    else:
        body += "<h2>What we found</h2><p>No issues in any episode.</p>"
    lag = s.get("median_lag_frames")
    absent = report.get("spec", {}).get("absent_streams") or []
    flagged = any("untracked" in r.get("facts", {}) for r in report["sessions"])  # validity flags were found and read
    marked = ", and frames the dataset marks as not measured" if flagged else ""
    if absent:
        covered = (f"every episode's {esc(' and '.join(report['spec']['required']))} stream: gaps, rates, frozen or idle "
                   f"recordings{marked}. The dataset doesn't record {esc(' or '.join(absent))}, so the action/state lag "
                   f"and copied-state checks don't apply")
    else:
        lag_note = "" if lag is None else f" ({lag} frame{'' if lag == 1 else 's'} here)"
        covered = (f"every episode's state and action streams: missing streams, gaps, rates, frozen or idle recordings, "
                   f"action/state lag against the dataset's own usual lag{lag_note}, states copied from commands{marked}")
    unread = "" if flagged else " Per-frame tracking or visibility flags, if the dataset has them."
    body += f"""<h2>What this audit covers</h2>
<ul><li>Covered: {covered}.</li>
<li>Not covered: video quality, labels, whether the task succeeded. Those need the video checks or a person.{unread}</li>
<li>Flags mean "look here", not "bad data". Every reason names the episode, so you can check it yourself.{esc(sampled)}</li></ul>
<h2>Reproduce it</h2>
<p>Data: <span class="mono">{esc(source['repo'])}</span> at revision <span class="mono">{esc(source['revision'])}</span>
({source['tables']} table(s), {source['mb']} MB; SHA-256 of each file in audit.json).<br>
Tool: Handoff v{esc(__version__)}, <span class="mono">handoff fleet &lt;dataset folder&gt;</span>.<br>
Verdict fingerprint: <span class="mono">{fp}</span>. The same data and tool give the same fingerprint on any machine.</p>
<p class="muted">License of the dataset: {esc(source['license'])}. Handoff is an independent acceptance check for robot
training data that runs where the data lives.{f' Contact: {esc(contact)}.' if contact else ''}</p>"""
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return page(f"Handoff audit: {source['repo']}", f"{when} · data revision {source['revision'][:7]} · tool v{__version__}", body)


def audit(repo: str, out_root: Path, data_root: Path, max_mb: float, contact: str) -> dict:
    out = out_root / repo.replace("/", "__")
    try:
        source = fetch(repo, data_root, max_mb)
        spec = lerobot_defaults(load_fleet_spec(None))
        sessions = load_logs(source["local"], spec)
        spec = lerobot_fit(spec, sessions, fit_duration=True)
        report = triage(sessions, spec)
    except Exception as exc:  # one bad dataset is a line in the summary, not a failed run
        out.mkdir(parents=True, exist_ok=True)
        result = {"repo": repo, "error": f"{type(exc).__name__}: {exc}"[:300]}
        (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    write_outputs(report, out)
    fp = fingerprint(source, report)
    result = {"repo": repo, "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "tool_version": __version__, "source": source, "summary": report["summary"], "issues": report["issues"],
              "fingerprint": fp}
    (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "audit.html").write_text(report_card(source, report, fp, contact), encoding="utf-8")
    (out / "badge.md").write_text(badge(source, report), encoding="utf-8")
    return result


def write_index(out_root: Path) -> Path:
    results = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_root.glob("*/audit.json"))]
    rows, html_rows = [], []
    for r in results:
        if "error" in r:
            rows.append({"dataset": r["repo"], "error": r["error"]})
            html_rows.append(f"<tr><td class='mono'>{esc(r['repo'])}</td><td colspan='5' class='muted'>{esc(r['error'])}</td></tr>")
            continue
        s = r["summary"]
        flagged = s["review"] + s["drop"]
        top = ", ".join(f"{k} ({v})" for k, v in list(r["issues"].items())[:3]) or "none"
        rows.append({"dataset": r["repo"], "license": r["source"]["license"], "episodes": s["sessions"],
                     "hours": round(sum(s["hours"].values()), 2), "keep": s["keep"], "review": s["review"],
                     "drop": s["drop"], "flagged_pct": round(100 * flagged / max(s["sessions"], 1), 1),
                     "top_issues": top, "fingerprint": r["fingerprint"]})
        link = f"{r['repo'].replace('/', '__')}/audit.html"
        html_rows.append(f"<tr><td class='mono'><a href='{esc(link)}'>{esc(r['repo'])}</a></td><td>{s['sessions']}</td>"
                         f"<td>{sum(s['hours'].values()):.1f}</td><td>{s['keep']}</td><td>{flagged}</td><td>{esc(top)}</td></tr>")
    fields = ["dataset", "license", "episodes", "hours", "keep", "review", "drop", "flagged_pct", "top_issues",
              "fingerprint", "error"]
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok = [r for r in rows if "error" not in r]
    eps = sum(r["episodes"] for r in ok)
    flagged = sum(r["review"] + r["drop"] for r in ok)
    body = cards([("datasets", len(ok)), ("episodes", eps), ("hours", f"{sum(r['hours'] for r in ok):.1f}"),
                  ("flagged", f"{flagged} ({100 * flagged / max(eps, 1):.1f}%)")])
    body += ("<table><tr><th>Dataset</th><th>Episodes</th><th>Hours</th><th>Keep</th><th>Flagged</th><th>Top issues</th></tr>"
             + "".join(html_rows) + "</table>")
    path = out_root / "index.html"
    path.write_text(page("Handoff audits of public robot datasets", f"tool v{__version__}", body), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("repos", nargs="*", help="Hugging Face dataset ids, e.g. org/name")
    p.add_argument("--top", type=int, default=0, help="also audit the N most-downloaded org-published LeRobot datasets")
    p.add_argument("--include-lerobot", action="store_true", help="include the lerobot org's own example datasets")
    p.add_argument("--authors", nargs="*", default=[], help="organizations: audit each one's most-downloaded dataset(s)")
    p.add_argument("--per-author", type=int, default=1, help="datasets per organization with --authors")
    p.add_argument("--out", default=str(ROOT / "reports" / "audits"))
    p.add_argument("--data", default=str(ROOT / "data" / "public"))
    p.add_argument("--max-mb", type=float, default=250.0, help="data tables to download per dataset, at most")
    p.add_argument("--skip-done", action="store_true", help="skip datasets that already have an audit.json")
    p.add_argument("--contact", default="", help="contact line printed on each report card")
    args = p.parse_args(argv)
    repos = list(args.repos) + (top_datasets(args.top, args.include_lerobot) if args.top else [])
    for author in args.authors:
        picked = author_datasets(author, args.per_author)
        print(f"{author}: {', '.join(picked) or 'no public datasets found'}", flush=True)
        repos += picked
    out_root, data_root = Path(args.out), Path(args.data)
    out_root.mkdir(parents=True, exist_ok=True)
    for repo in dict.fromkeys(repos):
        if args.skip_done and (out_root / repo.replace("/", "__") / "audit.json").exists():
            print(f"skip {repo} (done)")
            continue
        r = audit(repo, out_root, data_root, args.max_mb, args.contact)
        if "error" in r:
            print(f"FAIL {repo}: {r['error']}", flush=True)
        else:
            s = r["summary"]
            print(f"ok   {repo}: {s['sessions']} episodes, {s['keep']} keep, {s['review']} review, {s['drop']} drop", flush=True)
    print(f"index: {write_index(out_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
