"""Sync check on real teleoperation data: LeRobot's SO-100 pick-and-place (50 episodes, 30 fps).

Real data has no simulator here, so this measures detection only: plant offsets between the follower arm's state
and the leader arm's actions in 20% of episodes, and see whether the check finds them and recovers their size. A
run with no planted offsets measures the natural leader-to-follower lag and how much clean episodes spread.

The flagging rule is fixed before running: an episode is flagged when its best offset is at least 3 frames (0.1 s)
from the batch median.

    .venv-value\\Scripts\\python -m value.real_sync --workers 3
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data import corrupt_desync, load_lerobot
from .sync import confirm, estimate_lags

ROOT = Path("data/so100_pickplace")
FPS = 30
WINDOW = 15  # ±0.5 s
MIN_SHIFT = 3  # frames (0.1 s), fixed before running
CASES = [  # (lag in frames, which episodes)
    (0, 101),
    (-5, 103),
    (5, 107),
    (-10, 109),
    (3, 113),
]


def run_case(pool, lag: int, seed: int) -> dict:
    eps = load_lerobot(ROOT)
    if lag:
        eps = corrupt_desync(eps, fraction=0.2, lag=lag, seed=seed)
    bad = np.array([e.corrupted for e in eps])
    est = estimate_lags(pool, eps, window=WINDOW)
    found = np.array([est[e.index]["lag"] for e in eps])
    median = int(np.median(found))
    rel = found - median
    flagged = np.abs(rel) >= MIN_SHIFT
    second = confirm(eps, {int(e.index): int(r) for e, r, f in zip(eps, rel, flagged) if f}, WINDOW)
    confirmed = np.array([bool(second.get(int(e.index))) for e in eps])
    fix = -lag
    hist = lambda m: {int(k): int(v) for k, v in zip(*np.unique(rel[m], return_counts=True))}
    return {
        "lag_frames": lag,
        "lag_seconds": round(lag / FPS, 3),
        "median_offset_frames": median,
        "bad_episodes": int(bad.sum()),
        "bad_flagged": int((flagged & bad).sum()),
        "clean_flagged": int((flagged & ~bad).sum()),
        "bad_confirmed": int((confirmed & bad).sum()),
        "clean_confirmed": int((confirmed & ~bad).sum()),
        "clean_episodes": int((~bad).sum()),
        "bad_offset_within_1_frame": int(((np.abs(rel - fix) <= 1) & bad).sum()) if lag else None,
        "relative_offsets_bad": hist(bad) if lag else {},
        "relative_offsets_clean": hist(~bad),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="reports/value/real_sync_so100.json")
    args = ap.parse_args()
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for lag, seed in CASES:
            r = run_case(pool, lag, seed)
            results.append(r)
            if lag:
                print(f"offset {lag:+d} frames ({r['lag_seconds']:+.2f} s): found {r['bad_flagged']} of {r['bad_episodes']}, "
                      f"{r['clean_flagged']} of {r['clean_episodes']} clean flagged, size right within 1 frame for "
                      f"{r['bad_offset_within_1_frame']} | both methods agree: {r['bad_confirmed']} bad, "
                      f"{r['clean_confirmed']} clean", flush=True)
            else:
                print(f"no planted offsets: median leader-to-follower offset {r['median_offset_frames']:+d} frames "
                      f"({r['median_offset_frames'] / FPS:+.2f} s); {r['clean_flagged']} of {r['clean_episodes']} "
                      f"flagged; spread {r['relative_offsets_clean']}", flush=True)
    report = {"dataset": "lerobot/svla_so100_pickplace", "fps": FPS, "min_shift_frames": MIN_SHIFT,
              "window_frames": WINDOW, "cases": results, "minutes": round((time.perf_counter() - started) / 60, 1)}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out} in {report['minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
