"""Scan a LeRobot dataset for action/observation desync, episode by episode.

For each episode, a policy trained on the other episodes predicts its actions from its observations at every time
offset in a window. The best-fitting offset is compared with the dataset's median. Episodes at least `--min-shift`
frames away are flagged, with the offset that would re-align them.

Assumes most episodes are aligned. On real SO-100 teleoperation data with offsets planted in 20% of episodes, this
found every offset of 0.17 s or more with 0-2 false flags out of 40 (docs/value_demo_2026-10-30.md).

    python -m value.scan --root data/so100_pickplace
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data import load_lerobot
from .sync import confirm, estimate_lags, reliable, screen


def dataset_fps(root) -> float | None:
    """The dataset's frame rate from LeRobot's meta/info.json, or None if it isn't recorded."""
    info = Path(root) / "meta" / "info.json"
    fps = json.loads(info.read_text(encoding="utf-8")).get("fps") if info.exists() else None
    return float(fps) if fps else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Find episodes whose actions and observations are offset in time.")
    ap.add_argument("--root", required=True, help="local LeRobot v3 dataset folder (with data/ and meta/)")
    ap.add_argument("--obs-key", default="observation.state")
    ap.add_argument("--act-key", default="action")
    ap.add_argument("--fps", type=float, help="defaults to the fps in meta/info.json")
    ap.add_argument("--window", type=float, default=0.5, help="offsets tried, in seconds either way")
    ap.add_argument("--min-shift", type=int, default=3, help="frames from the median offset that count as desynced")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="reports/sync_scan")
    ap.add_argument("--repair", metavar="FOLDER", help="also write a re-aligned copy of the dataset here (confirmed flags only)")
    args = ap.parse_args()
    args.fps = args.fps or dataset_fps(args.root)
    if not args.fps:
        ap.error("no fps in meta/info.json; pass --fps")

    all_episodes = load_lerobot(args.root, args.obs_key, args.act_key)
    window = max(1, round(args.window * args.fps))
    status = screen(all_episodes, window)
    episodes = [e for e in all_episodes if status[e.index] == "ok"]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        est = estimate_lags(pool, episodes, window=window)
    offsets = np.array([est[e.index]["lag"] for e in episodes])
    median = int(np.median(offsets))
    measured = reliable(offsets - median, args.min_shift)
    candidates = {e.index: int(o - median) for e, o in zip(episodes, offsets) if measured and abs(o - median) >= args.min_shift}
    second = confirm(episodes, candidates, window)

    rows = []
    for e in all_episodes:
        if status[e.index] != "ok":
            rows.append({"episode": e.index, "frames": len(e.obs), "status": status[e.index], "offset_frames": "",
                         "offset_seconds": "", "flagged": False, "second_opinion": "", "fit_gain_if_realigned": ""})
            continue
        rel = int(est[e.index]["lag"] - median)
        rows.append({
            "episode": e.index,
            "frames": len(e.obs),
            "status": "measured" if measured else "inconclusive",
            "offset_frames": rel,
            "offset_seconds": round(rel / args.fps, 3),
            # A flag needs the model-free cross-check to agree, where it applies (same joints for state and action).
            "flagged": e.index in candidates and second.get(e.index) is not False,
            "second_opinion": {True: "agrees", False: "disagrees", None: "n/a"}[second[e.index]] if e.index in candidates else "",
            "fit_gain_if_realigned": round(est[e.index]["improvement"], 3),
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sync_scan.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    flagged = [r for r in rows if r["flagged"]]
    idle = [r for r in rows if r["status"] in ("too short", "no motion")]
    summary = {
        "dataset": str(args.root),
        "episodes": len(rows),
        "fps": args.fps,
        "verdict": "measured" if measured else "inconclusive: offsets too spread for the median rule",
        "median_offset_frames": median,
        "median_offset_seconds": round(median / args.fps, 3),
        "min_shift_frames": args.min_shift,
        "flagged": len(flagged),
        "episodes_flagged": flagged,
        "idle_or_short": len(idle),
        "episodes_idle_or_short": idle,
        "needs_review": [r for r in rows if r.get("second_opinion") == "disagrees"],
    }
    (out / "sync_scan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{len(rows)} episodes; typical action-to-observation offset {median:+d} frames ({median / args.fps:+.3f} s)")
    for r in idle:
        print(f"  episode {r['episode']}: {r['status']} ({r['frames']} frames), can't measure sync; check this episode")
    if not measured:
        print("inconclusive: offsets are too spread for the median rule on this dataset, so nothing is flagged")
    elif flagged:
        print(f"{len(flagged)} flagged as desynced:")
        for r in flagged:
            print(f"  episode {r['episode']}: {r['offset_frames']:+d} frames ({r['offset_seconds']:+.3f} s) from the rest")
    else:
        print(f"none flagged (no episode {args.min_shift}+ frames from the median)")
    review = [r for r in rows if r.get("second_opinion") == "disagrees"]
    if review:
        print(f"{len(review)} possible, needing a look (one method sees an offset, the other doesn't; not repaired):")
        for r in review:
            print(f"  episode {r['episode']}: {r['offset_frames']:+d} frames ({r['offset_seconds']:+.3f} s) by the scan alone")
    print(f"wrote {out / 'sync_scan.csv'} and {out / 'sync_scan.json'}")

    if args.repair:
        from .repair import repair_dataset

        # The scan's offset is the shift that re-aligns the episode, so it's applied as is.
        fixes = {r["episode"]: r["offset_frames"] for r in flagged if second.get(r["episode"]) is True}
        skipped = len(flagged) - len(fixes)
        log = repair_dataset(args.root, args.repair, fixes, args.act_key)
        print(f"repaired {len(log['repaired_episodes'])} episodes into {args.repair} (see REPAIR_LOG.json)")
        if skipped:
            print(f"left {skipped} flagged episodes as they are: no second method confirmed their offset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
