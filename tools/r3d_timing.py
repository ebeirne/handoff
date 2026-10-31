"""Frame timing of Record3D (.r3d) clips on Hugging Face, read without downloading them.

An .r3d is a ZIP whose `metadata` member holds `frameTimestamps` and `fps`. HTTP range requests fetch only the ZIP
directory and that member (tens of KB of a clip that is hundreds of MB), so a whole raw dataset's timing can be
checked from a laptop: dropped frames, stalls, and clips whose real rate is far from what they claim.

    python tools/r3d_timing.py org/raw_r3d_dataset --index index.csv --clips CLIP_A CLIP_B
    python tools/r3d_timing.py org/raw_r3d_dataset --index index.csv --sample 40 --out timing.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from handoff.video import rate_runs  # noqa: E402  (the same reading of frame rate as `handoff check`)

HUB = "https://huggingface.co"


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, fetched in HTTP ranges as zipfile asks for them."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        head = self._open("HEAD")
        # The Hub redirects each file to a signed CDN address. Ranges go there directly: the Hub's resolver is
        # rate-limited (3,000 requests per 5 minutes), and a ZIP read takes several ranges per clip.
        self.url = head.geturl() if hasattr(head, "geturl") else url
        self.size = int(head.headers["Content-Length"])

    def _open(self, method: str = "GET", headers: dict | None = None):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, method=method, headers=headers or {})
                return urllib.request.urlopen(req, timeout=60)
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500 or attempt == 5:
                    raise
                wait = re.search(r"t=(\d+)", exc.headers.get("RateLimit", "") or "")  # seconds until the window resets
                time.sleep(int(wait.group(1)) + 1 if wait else 2 ** attempt)
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        data = self._open(headers={"Range": f"bytes={self.pos}-{end - 1}"}).read()
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def timing(stamps: list[float], fps: float) -> dict:
    """Frame rate as recorded against the metadata's `fps`: stretches of steady rate (a change partway through is a
    defect; a steady rate that differs from the metadata isn't), frames dropped within them, and stalls."""
    t = np.asarray(stamps, dtype=float) - stamps[0]
    dt = np.diff(t)
    runs, usual = rate_runs(t, fps)
    missing = np.clip(np.round(dt * fps / usual) - 1, 0, None)  # frames that belong between two recorded ones
    span = float(t[-1]) if len(t) > 1 else 0.0
    return {"frames": len(t), "seconds": round(span, 2), "metadata_fps": fps,
            "recorded_fps": round(max(runs, key=lambda r: r[1] - r[0])[2], 2),
            "average_fps": round((len(t) - 1) / span, 2) if span > 0 else 0.0,
            "rate_changes": len(runs) - 1, "rates": " > ".join(f"{r[2]:g}@{r[0]:.0f}s" for r in runs),
            "dropped_frames": int(missing.sum()),
            "dropped_fraction": round(float(missing.sum() / (missing.sum() + len(t))), 4) if len(t) else 0.0,
            "gaps_over_0.5s": int((dt > 0.5).sum()), "longest_gap_s": round(float(dt.max()), 3) if len(dt) else 0.0}


def clip_timing(repo: str, revision: str, path: str) -> dict:
    url = f"{HUB}/datasets/{repo}/resolve/{revision}/{path}"
    with zipfile.ZipFile(RangeFile(url)) as z:
        meta = json.loads(z.read("metadata"))
    return timing(meta["frameTimestamps"], float(meta["fps"]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo")
    ap.add_argument("--index", required=True, help="the dataset's index.csv (columns clip_id, file)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--clips", nargs="*", default=[], help="clip ids to check")
    ap.add_argument("--sample", type=int, default=0, help="also check this many other clips, chosen at random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="check every clip in the index")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", help="a previous run's output (one JSON line per clip): keep its results, redo errors")
    ap.add_argument("--out", help="write one row per clip to this CSV")
    args = ap.parse_args(argv)

    rows = {r["clip_id"]: r for r in csv.DictReader(open(args.index, encoding="utf-8"))}
    picked = list(rows) if args.all else [c for c in args.clips if c in rows]
    rest = sorted(set(rows) - set(picked))
    picked += random.Random(args.seed).sample(rest, min(args.sample, len(rest)))
    done: dict[str, dict] = {}
    if args.resume:
        for line in open(args.resume, encoding="utf-8"):
            if line.startswith("{"):
                r = json.loads(line)
                if "error" not in r:
                    done[r["clip_id"]] = r
    picked = [c for c in picked if c not in done]

    def one(cid: str) -> dict:
        try:
            return {"clip_id": cid, **clip_timing(args.repo, args.revision, rows[cid]["file"])}
        except Exception as exc:  # one unreadable clip shouldn't stop the run; it's reported
            return {"clip_id": cid, "error": f"{type(exc).__name__}: {exc}"[:200]}

    results = list(done.values())
    with ThreadPoolExecutor(max(1, args.workers)) as pool:  # network-bound: threads are enough
        for r in pool.map(one, picked):
            results.append(r)
            print(json.dumps(r), flush=True)
    if args.out:
        keys = sorted({k for r in results for k in r}, key=lambda k: (k != "clip_id", k))
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
