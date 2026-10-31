"""Scan the most-used public LeRobot datasets on Hugging Face for action/observation desync.

Picks datasets in LeRobot's v3 format with low-dimensional `observation.state` and `action`, 20-2000 episodes, and
state/action tables under 60 MB. Downloads only those tables (no video) and runs the same scan as `value.scan`.

A flagged episode is a candidate to inspect, not a confirmed bug: an outlier offset can also come from an unusual
demonstration. Inspect before claiming anything, and contact a dataset's owner before naming it publicly.

    .venv-value\\Scripts\\python -m value.community_scan --max 15
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data import load_lerobot
from .sync import confirm, estimate_lags, reliable, screen

API = "https://huggingface.co/api/datasets"
RESOLVE = "https://huggingface.co/datasets/{id}/resolve/main/{path}"
MAX_TABLE_BYTES = 60 * 1024 * 1024


def _get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def candidates(limit: int) -> list[dict]:
    return _get_json(f"{API}?filter=LeRobot&sort=downloads&direction=-1&limit={limit}")


def qualify(ds: dict) -> dict | None:
    try:
        info = _get_json(RESOLVE.format(id=ds["id"], path="meta/info.json"))
    except Exception:
        return None
    f = info.get("features", {})
    st, ac = f.get("observation.state"), f.get("action")
    if not str(info.get("codebase_version", "")).startswith("v3") or not st or not ac:
        return None
    if st.get("dtype") not in ("float32", "float64") or ac.get("dtype") not in ("float32", "float64"):
        return None
    if max(st.get("shape", [99])) > 32 or max(ac.get("shape", [99])) > 32:
        return None
    if not 20 <= int(info.get("total_episodes", 0)) <= 2000:
        return None
    try:
        tree = _get_json(f"{API}/{ds['id']}/tree/main/data?recursive=true")
    except Exception:
        return None
    files = [t for t in tree if t.get("type") == "file" and t["path"].endswith(".parquet")]
    total = sum(t.get("size", 0) for t in files)
    if not files or total > MAX_TABLE_BYTES:
        return None
    lic = next((t.split(":", 1)[1] for t in ds.get("tags", []) if t.startswith("license:")), "none listed")
    return {"id": ds["id"], "fps": float(info["fps"]), "episodes": int(info["total_episodes"]),
            "frames": int(info.get("total_frames", 0)), "robot": info.get("robot_type"), "license": lic,
            "files": [t["path"] for t in files], "bytes": total, "downloads": ds.get("downloads")}


def download(ds: dict, root: Path) -> Path:
    dest = root / ds["id"].replace("/", "__")
    for p in ds["files"]:
        out = dest / p
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RESOLVE.format(id=ds["id"], path=p), out)
    (dest / "SOURCE.md").write_text(f"Source: https://huggingface.co/datasets/{ds['id']} (license: {ds['license']}). "
                                    "State/action tables only.\n", encoding="utf-8")
    return dest


def scan(pool, folder: Path, fps: float) -> dict:
    all_eps = load_lerobot(folder)
    window = max(3, round(0.5 * fps))
    min_shift = max(3, round(0.1 * fps))
    status = screen(all_eps, window)
    idle = sorted((e.index, status[e.index], len(e.obs)) for e in all_eps if status[e.index] != "ok")
    eps = [e for e in all_eps if status[e.index] == "ok"]
    est = estimate_lags(pool, eps, window=window)
    off = np.array([est[e.index]["lag"] for e in eps])
    median = int(np.median(off))
    rel = off - median
    ok = reliable(rel, min_shift)
    candidates = {int(e.index): int(r) for e, r in zip(eps, rel) if abs(r) >= min_shift} if ok else {}
    second = confirm(eps, candidates, window)
    flagged = [(i, r) for i, r in candidates.items() if second[i] is not False]  # confirmed, or no second method
    unconfirmed = [(i, r) for i, r in candidates.items() if second[i] is False]
    return {
        "second_opinion": "cross-correlation" if any(v is not None for v in second.values()) or not candidates else "none",
        "unconfirmed": len(unconfirmed),
        "unconfirmed_episodes": unconfirmed[:25],
        "measured_episodes": len(eps),
        "idle_or_short": len(idle),
        "idle_or_short_episodes": [{"episode": i, "reason": s, "frames": n} for i, s, n in idle][:25],
        "verdict": "measured" if ok else "inconclusive: offsets too spread for the median rule",
        "median_offset_frames": median,
        "median_offset_seconds": round(median / fps, 3),
        "spread_mad_frames": float(np.median(np.abs(rel))),
        "min_shift_frames": min_shift,
        "flagged": len(flagged),
        "flagged_share": round(len(flagged) / max(1, len(eps)), 3),
        "flagged_episodes": sorted(flagged, key=lambda x: -abs(x[1]))[:25],
        "at_window_edge": int((np.abs(off) >= window).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=15, help="datasets to scan")
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--root", default="data/community")
    ap.add_argument("--out", default="reports/community_scan/summary.json")
    args = ap.parse_args()
    started = time.perf_counter()
    root = Path(args.root)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for ds in candidates(args.candidates):
            if len(results) >= args.max:
                break
            q = qualify(ds)
            if not q:
                continue
            try:
                folder = download(q, root)
                q.update(scan(pool, folder, q["fps"]))
                q.pop("files")
            except Exception as exc:  # keep going: one odd dataset shouldn't stop the survey
                print(f"skip {ds['id']}: {exc}", flush=True)
                continue
            results.append(q)
            print(f"{q['id']}: {q['episodes']} eps @ {q['fps']:g} fps | idle/short {q['idle_or_short']} | {q['verdict']}"
                  f" | typical offset {q['median_offset_seconds']:+.3f} s | sync flags {q['flagged']}"
                  f" (unconfirmed {q['unconfirmed']})", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"datasets": results, "minutes": round((time.perf_counter() - started) / 60, 1)}, indent=2),
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
