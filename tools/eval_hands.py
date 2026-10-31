"""Measure the hands_visible detector against labeled egocentric frames.

Uses Build AI's Egocentric-10K evaluation set (Apache-2.0): 10,000 frames sampled from Egocentric-10K, each labeled
by gemini-2.5-flash with how many of the camera wearer's hands are visible (0, 1 or 2). Those labels come from a
model, not people, so the numbers below are agreement with Gemini, not ground-truth accuracy.

    python tools/eval_hands.py data/egocentric10k_eval/egocentric_10k.parquet --model models/hand_landmarker.task
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff.checks import HandDetector  # noqa: E402
from handoff.video import WORK_WIDTH  # noqa: E402

_detector: HandDetector | None = None


def _init(model: str, confidence: float) -> None:
    global _detector
    _detector = HandDetector(model, confidence)


def _decode(data: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    if w > WORK_WIDTH:  # same preprocessing as `handoff check`
        frame = cv2.resize(frame, (WORK_WIDTH, round(h * WORK_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return frame


def _run_group(args: tuple[str, int, int | None]) -> list[tuple[str, int, str, int]]:
    path, group, limit = args
    table = pq.ParquetFile(path).read_row_group(group, columns=["frame_id", "image", "hand_count", "active_labor"])
    rows = table.to_pylist()[:limit]
    out = []
    for r in rows:
        n, _ = _detector.hands_in(_decode(r["image"]["bytes"]))
        out.append((r["frame_id"], int(r["hand_count"]), str(r["active_labor"]), int(min(n, 2))))
    return out


def evaluate(path: str, model: str, confidence: float, workers: int, limit: int | None) -> list[tuple]:
    f = pq.ParquetFile(path)
    jobs, remaining = [], limit
    for g in range(f.metadata.num_row_groups):
        if remaining is not None and remaining <= 0:
            break
        n = f.metadata.row_group(g).num_rows
        take = n if remaining is None else min(n, remaining)
        jobs.append((path, g, take))
        if remaining is not None:
            remaining -= take
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(model, confidence)) as pool:
        return [row for part in pool.map(_run_group, jobs) for row in part]


def summarize(rows: list[tuple], confidence: float) -> dict:
    label = np.array([r[1] for r in rows])
    pred = np.array([r[3] for r in rows])
    active = np.array([r[2] == "yes" for r in rows])
    any_l, any_p = label >= 1, pred >= 1

    def rate(num, den):
        return round(float(num.sum() / den.sum()), 4) if den.sum() else None

    confusion = {f"label_{a}": {f"detected_{b}": int(((label == a) & (pred == b)).sum()) for b in (0, 1, 2)} for a in (0, 1, 2)}
    return {
        "confidence": confidence,
        "frames": len(rows),
        "label_any_hand_rate": round(float(any_l.mean()), 4),
        "detected_any_hand_rate": round(float(any_p.mean()), 4),
        "any_hand_recall": rate(any_l & any_p, any_l),  # of frames Gemini says show a hand, share we detect
        "any_hand_precision": rate(any_l & any_p, any_p),  # of frames we say show a hand, share Gemini agrees
        "two_hand_recall": rate((label == 2) & (pred == 2), label == 2),
        "exact_count_agreement": round(float((label == pred).mean()), 4),
        "any_hand_recall_during_active_work": rate(any_l & any_p & active, any_l & active),
        "confusion": confusion,
    }


def miss_sheet(path: str, rows: list[tuple], out: Path, n: int = 12) -> None:
    """Contact sheet of frames Gemini says show two hands where the detector found none."""
    misses = {r[0] for r in rows if r[1] == 2 and r[3] == 0}
    if not misses:
        return
    picked = sorted(misses)[:n]
    f = pq.ParquetFile(path)
    tiles = []
    for g in range(f.metadata.num_row_groups):
        for r in f.read_row_group(g, columns=["frame_id", "image"]).to_pylist():
            if r["frame_id"] in picked:
                tile = cv2.resize(_decode(r["image"]["bytes"]), (320, 180))
                tiles.append(tile)
        if len(tiles) >= len(picked):
            break
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(out), sheet)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("--model", required=True, help="MediaPipe hand_landmarker.task")
    ap.add_argument("--confidence", type=float, nargs="+", default=[0.5, 0.3, 0.1])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, help="only the first N frames")
    ap.add_argument("--out", default="reports/hands_eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for conf in args.confidence:
        rows = evaluate(args.parquet, args.model, conf, args.workers, args.limit)
        s = summarize(rows, conf)
        summaries.append(s)
        print(
            f"confidence {conf}: frames {s['frames']}, Gemini says a hand is visible in {s['label_any_hand_rate']:.1%}, "
            f"detector says {s['detected_any_hand_rate']:.1%} | recall {s['any_hand_recall']:.1%}, "
            f"precision {s['any_hand_precision']:.1%}, both-hands recall {s['two_hand_recall']:.1%}, "
            f"exact count {s['exact_count_agreement']:.1%}"
        )
        if conf == 0.3:
            miss_sheet(args.parquet, rows, out / "misses_two_hands_detected_none.jpg")
    (out / "results.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"wrote {out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
