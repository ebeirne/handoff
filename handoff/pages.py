"""Shared HTML shell for the fleet, passport and console pages, styled like the check report."""

from __future__ import annotations

import html

CSS = """
:root { --bg:#fff; --fg:#1f2328; --muted:#59636e; --line:#d1d9e0; --keep:#1a7f37; --review:#9a6700; --drop:#cf222e; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --muted:#9198a1; --line:#3d444d;
  --keep:#3fb950; --review:#d29922; --drop:#f85149; } }
body { background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; margin:0 auto; max-width:1100px; padding:24px 16px; }
h1 { margin:0 0 4px; } h2 { margin:28px 0 8px; } .muted { color:var(--muted); }
.cards { display:flex; gap:12px; flex-wrap:wrap; margin:16px 0; }
.card { border:1px solid var(--line); border-radius:8px; padding:10px 14px; min-width:110px; }
.card b { display:block; font-size:24px; }
table { border-collapse:collapse; width:100%; } td, th { border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:top; }
.tag { border-radius:12px; padding:2px 10px; font-weight:600; white-space:nowrap; border:1px solid currentColor; }
.keep, .ok { color:var(--keep); } .review { color:var(--review); } .drop, .bad { color:var(--drop); }
.mono { font-family:ui-monospace, monospace; font-size:13px; word-break:break-all; }
ul { margin:0; padding-left:18px; }
"""


def esc(value) -> str:
    return html.escape(str(value))


def page(title: str, subtitle: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<h1>{esc(title)}</h1>
<p class="muted">{esc(subtitle)}</p>
{body}
</body></html>
"""


def cards(items: list[tuple[str, object]]) -> str:
    return '<div class="cards">' + "".join(f'<div class="card"><b>{esc(v)}</b>{esc(k)}</div>' for k, v in items) + "</div>"
