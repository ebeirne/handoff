"""Write results: JSON for machines, HTML for people, and the clip lists that go with them."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from .run import VERDICTS, RunResult

VERDICT_COLORS = {"pass": "#1a7f37", "downgrade": "#0969da", "flag": "#9a6700", "reject": "#cf222e"}
VERDICT_LABELS = {"pass": "pass", "downgrade": "downgraded", "flag": "flagged", "reject": "rejected"}
MAX_EVIDENCE_SHOWN = 6


def fingerprint(result: RunResult) -> str:
    """Same spec + same tool version + same data gives the same fingerprint, so both sides can re-run and compare."""
    h = hashlib.sha256()
    h.update(result.spec.sha256.encode())
    h.update(result.tool_version.encode())
    h.update(f"not_run:{','.join(sorted(result.not_run))}\n".encode())  # an incomplete run never matches a complete one
    for v in sorted(result.verdicts, key=lambda v: v.clip):
        failed = ",".join(sorted(c.name for c in v.failed))
        h.update(f"{v.clip}|{v.verdict}|{failed}\n".encode())
    return h.hexdigest()


def spot_check(result: RunResult, fp: str) -> dict:
    """Which passed clips a person should review, and how many, for the spec's confidence target.

    Zero-failure acceptance sampling: if none of n randomly chosen passed clips turns out bad, you can be
    `confidence` sure that fewer than `max_defect_rate` of passed clips are bad. n = ln(1 - C) / ln(1 - p).
    The choice is seeded by the fingerprint, so anyone re-running the same spec on the same data draws the same clips.
    """
    passed = sorted(v.clip for v in result.verdicts if v.verdict == "pass")
    c, p = result.spec.confidence, result.spec.max_defect_rate
    needed = math.ceil(math.log(1 - c) / math.log(1 - p))
    n = min(needed, len(passed))
    clips = sorted(random.Random(fp).sample(passed, n)) if n else []
    return {
        "confidence": c,
        "max_defect_rate": p,
        "needed": needed,
        "sample_size": n,
        "review_all": n == len(passed) and n > 0,
        "clips": clips,
    }


def cost(result: RunResult) -> dict:
    hours = result.video_seconds / 3600
    return {
        "video_hours": round(hours, 4),
        "workers": result.workers,
        "wall_seconds": round(result.compute_seconds, 2),
        "wall_seconds_per_video_hour": round(result.compute_seconds / hours, 1) if hours else None,
    }


def to_json(result: RunResult) -> dict:
    fp = fingerprint(result)
    counts = {k: sum(v.verdict == k for v in result.verdicts) for k in VERDICTS}
    return {
        "tool": "handoff-check",
        "tool_version": result.tool_version,
        "spec": {"name": result.spec.name, "version": result.spec.version, "sha256": result.spec.sha256},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fingerprint": fp,
        "complete": result.complete,
        "not_run": result.not_run,
        "summary": {"clips": len(result.verdicts), **counts},
        "cost": cost(result),
        "spot_check": spot_check(result, fp),
        "coverage": result.coverage,
        "warnings": result.warnings,
        "clips": [
            {"clip": v.clip, "verdict": v.verdict, "info": v.info, "checks": [c.to_dict() for c in v.checks]}
            for v in result.verdicts
        ],
    }


def _clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"


def _evidence(c) -> str:
    if not c.evidence:
        return ""
    parts = [_clock(a) if a == b else f"{_clock(a)}–{_clock(b)}" for a, b in c.evidence[:MAX_EVIDENCE_SHOWN]]
    more = len(c.evidence) - MAX_EVIDENCE_SHOWN
    tail = f" (+{more} more)" if more > 0 else ""
    return f' <span class="at">at {html.escape(", ".join(parts))}{tail}</span>'


def _check_cell(c) -> str:
    if c.passed is None:
        return f'<li class="skip">{html.escape(c.name)}: {html.escape(c.reason)}</li>'
    if c.passed:
        return f'<li class="ok">{html.escape(c.name)}: {html.escape(c.reason)}</li>'
    return f'<li class="bad">{html.escape(c.name)}: {html.escape(c.reason)}{_evidence(c)}</li>'


def to_html(result: RunResult, data: dict) -> str:
    s = data["summary"]
    order = {"reject": 0, "flag": 1, "downgrade": 2, "pass": 3}
    rows = []
    for v in sorted(result.verdicts, key=lambda v: (order[v.verdict], v.clip)):
        thumb = ""
        if v.thumbnail:
            thumb = f'<img alt="" src="data:image/jpeg;base64,{base64.b64encode(v.thumbnail).decode()}">'
        failed = [c for c in v.checks if c.passed is False]
        shown = failed or [c for c in v.checks if c.passed is not False]
        rows.append(
            "<tr>"
            f"<td>{thumb}</td>"
            f'<td><span class="verdict" style="background:{VERDICT_COLORS[v.verdict]}">{VERDICT_LABELS[v.verdict]}</span></td>'
            f"<td class=\"clip\">{html.escape(v.clip)}</td>"
            f"<td><ul>{''.join(_check_cell(c) for c in shown)}</ul></td>"
            "</tr>"
        )

    coverage_html = ""
    if result.coverage:
        crow = "".join(
            f"<tr><td>{html.escape(r['value'])}</td><td>{r['passed']}</td><td>{r['downgraded']}</td><td>{r['flagged']}</td>"
            f"<td>{r['rejected']}</td><td>{'' if r['target'] is None else r['target']}</td>"
            f"<td>{'' if r['shortfall'] is None else r['shortfall']}</td></tr>"
            for r in result.coverage["rows"]
        )
        coverage_html = (
            f"<h2>Coverage by {html.escape(result.coverage['field'])}</h2>"
            "<table class=\"cov\"><tr><th>Value</th><th>Passed</th><th>Downgraded</th><th>Flagged</th><th>Rejected</th>"
            f"<th>Target</th><th>Short by</th></tr>{crow}</table>"
        )

    sc = data["spot_check"]
    if sc["sample_size"]:
        intro = (
            f"Review all {sc['sample_size']} passed clips: this batch is smaller than the {sc['needed']} needed"
            if sc["review_all"]
            else f"Review these {sc['sample_size']} randomly chosen passed clips"
        )
        spot_html = (
            f"<h2>Spot-check</h2><p>{intro}. If none of them is bad, you can be {sc['confidence']:.0%} confident "
            f"that fewer than {sc['max_defect_rate']:.0%} of passed clips are bad. Anyone re-running this spec on this "
            "data gets the same list.</p>"
            f"<p class=\"clip\">{'<br>'.join(html.escape(c) for c in sc['clips'])}</p>"
        )
    else:
        spot_html = "<h2>Spot-check</h2><p>No clips passed, so there is nothing to spot-check.</p>"

    warnings_html = "".join(f"<p class=\"warn\">{html.escape(w)}</p>" for w in result.warnings)
    if not result.complete:
        warnings_html = (
            f'<p class="incomplete"><b>Incomplete run.</b> The spec asks for {html.escape(", ".join(result.not_run))}, '
            "which didn't run, so a pass below doesn't mean those checks passed. Fix the warning and re-run.</p>"
            + warnings_html
        )
    c = data["cost"]
    per_hour = c["wall_seconds_per_video_hour"]
    cards = "".join(
        f'<div class="card" style="color:{VERDICT_COLORS[k]}"><b>{s[k]}</b>{VERDICT_LABELS[k]}</div>' for k in VERDICTS
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Handoff Check report</title>
<style>
:root {{ --bg:#fff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#3d444d; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }}
h1 {{ margin:0 0 4px; }} .muted {{ color:var(--muted); }}
.cards {{ display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:10px 14px; min-width:110px; }}
.card b {{ display:block; font-size:24px; }}
table {{ border-collapse:collapse; width:100%; }} td, th {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }}
img {{ width:160px; border-radius:4px; display:block; }}
.verdict {{ color:#fff; border-radius:12px; padding:2px 10px; font-weight:600; white-space:nowrap; }}
.clip {{ word-break:break-all; font-family:ui-monospace, monospace; font-size:13px; }}
ul {{ margin:0; padding-left:18px; }} li.bad {{ color:#cf222e; }} li.skip {{ color:var(--muted); }}
.at {{ color:var(--muted); }}
.warn {{ border-left:3px solid #9a6700; padding-left:8px; }}
.incomplete {{ border:2px solid #cf222e; color:#cf222e; border-radius:6px; padding:10px 12px; }}
.cov {{ max-width:720px; margin-bottom:24px; }}
code {{ font-size:12px; word-break:break-all; }}
</style></head><body>
<h1>Handoff Check report</h1>
<p class="muted">Spec <b>{html.escape(result.spec.name)}</b> (v{result.spec.version}) · tool v{result.tool_version} · {data['created_at']}</p>
<div class="cards">
<div class="card"><b>{s['clips']}</b>clips</div>
{cards}
<div class="card"><b>{c['video_hours']:.2f}</b>hours of video</div>
<div class="card"><b>{'n/a' if per_hour is None else f'{per_hour / 60:.1f} min'}</b>per video hour ({c['workers']} worker{'s' if c['workers'] > 1 else ''})</div>
</div>
<p class="muted">Flagged clips need a person to look. Downgraded clips are usable at a lower grade. Only checks listed
under <code>rules.reject_on</code> reject a clip.</p>
{warnings_html}
{coverage_html}
{spot_html}
<h2>Clips</h2>
<table><tr><th></th><th>Verdict</th><th>Clip</th><th>Why</th></tr>
{''.join(rows)}
</table>
<p class="muted">Fingerprint <code>{data['fingerprint']}</code>. Re-running the same spec and tool version on the same data gives the same fingerprint.</p>
</body></html>
"""


def write_outputs(result: RunResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = to_json(result)
    paths = {
        "json": out / "results.json",
        "html": out / "report.html",
        "accepted": out / "accepted.txt",
        "downgraded": out / "downgraded.txt",
        "spot_check": out / "spot_check.txt",
    }
    paths["json"].write_text(json.dumps(data, indent=2), encoding="utf-8")
    paths["html"].write_text(to_html(result, data), encoding="utf-8")

    def lines(items: list[str]) -> str:
        return "\n".join(items) + ("\n" if items else "")

    paths["accepted"].write_text(lines([v.clip for v in result.verdicts if v.verdict == "pass"]), encoding="utf-8")
    paths["downgraded"].write_text(lines([v.clip for v in result.verdicts if v.verdict == "downgrade"]), encoding="utf-8")
    paths["spot_check"].write_text(lines(data["spot_check"]["clips"]), encoding="utf-8")
    return paths
