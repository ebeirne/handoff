"""A corrected index for a folder of Record3D (.r3d) clips, read from the clips themselves instead of a vendor's
metadata: the frame rate each clip actually recorded and where it changes, frames dropped, stalls, and how much of
the depth the sensor trusted. Only each clip's `metadata` and a dozen confidence maps are read, never the video
frames, so a whole catalog goes quickly. Depth needs `pip install -e .[depth]`.

    python tools/r3d_index.py path/to/clips --out index.csv
    python tools/r3d_index.py path/to/clips --out index.csv --compare vendor_index.csv --id-column clip_id

With --compare, each row also carries the vendor index's depth value, and the summary lists the clips where it
disagrees with the measured one (for example a session marked as having no usable depth whose maps say otherwise).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff import video  # noqa: E402
from r3d_timing import timing  # noqa: E402

DEPTH_MAPS = 12  # confidence maps read per clip, evenly spaced


def clip_row(path: Path, root: Path) -> dict:
    """One clip's index row. An unreadable clip gets an `error` instead of stopping the run."""
    row = {"file": path.relative_to(root).as_posix(), "clip_id": path.stem}
    try:
        with zipfile.ZipFile(path) as z:
            meta = json.loads(z.read("metadata"))
            row |= timing(meta["frameTimestamps"], float(meta["fps"]))
            confs = sorted((k for k in z.namelist() if k.startswith("rgbd/") and k.endswith(".conf")),
                           key=lambda k: int(k[5:-5]))
            decompress = video._lzfse()
            size = int(meta.get("dw") or 0) * int(meta.get("dh") or 0)
            if confs and decompress and size:
                picks = [confs[i] for i in sorted({int(i) for i in np.linspace(0, len(confs) - 1, DEPTH_MAPS)})]
                shares = []
                for key in picks:
                    try:
                        c = np.frombuffer(decompress(z.read(key)), np.uint8)
                    except Exception:  # a corrupt map costs that reading, not the clip or the run
                        continue
                    if c.size == size:
                        shares.append(float((c == 2).mean()))
                if shares:
                    row |= {"depth_high_share": round(float(np.mean(shares)), 4),
                            "depth_min_share": round(min(shares), 4), "depth_maps_read": len(shares)}
            if "depth_high_share" not in row:
                row["depth_note"] = ("no depth confidence maps" if not confs else
                                     "install pyliblzfse to read depth" if not decompress else "maps couldn't be read")
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return row


def flags(row: dict, max_dropped: float, max_gap: float, min_depth: float) -> list[str]:
    out = []
    if "error" in row:
        return ["unreadable"]
    if row["rate_changes"]:
        out.append("rate_change")
    if row["dropped_fraction"] > max_dropped:
        out.append("dropped_frames")
    if row["longest_gap_s"] > max_gap:
        out.append("stall")
    if row.get("depth_high_share", 1.0) < min_depth:
        out.append("low_depth")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder of .r3d clips (searched recursively)")
    ap.add_argument("--out", required=True, help="CSV to write, one row per clip")
    ap.add_argument("--compare", help="the vendor's own index CSV, to list clips where it disagrees")
    ap.add_argument("--id-column", default="clip_id", help="column in --compare matching each clip's file name stem")
    ap.add_argument("--depth-column", default="depth_conf_high", help="column in --compare with the depth share")
    ap.add_argument("--max-dropped", type=float, default=0.05)
    ap.add_argument("--max-gap", type=float, default=0.25, help="seconds")
    ap.add_argument("--min-depth", type=float, default=0.3, help="share of high-confidence depth pixels")
    ap.add_argument("--workers", type=int, default=0, help="processes (0 = all cores but one; 1 = no subprocesses)")
    args = ap.parse_args(argv)

    root = Path(args.folder)
    clips = sorted(p for p in root.rglob("*.r3d") if p.is_file())
    if not clips:
        print(f"error: no .r3d files in {root}", file=sys.stderr)
        return 2
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        rows = [clip_row(p, root) for p in clips]
    else:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(clip_row, clips, [root] * len(clips), chunksize=8))

    vendor = {}
    if args.compare:
        vendor = {r[args.id_column]: r for r in csv.DictReader(open(args.compare, encoding="utf-8-sig"))}
    disagree = []
    for r in rows:
        r["flags"] = " ".join(flags(r, args.max_dropped, args.max_gap, args.min_depth))
        v = vendor.get(r["clip_id"])
        if v is not None and v.get(args.depth_column, "") != "":
            r["vendor_depth"] = float(v[args.depth_column])
            measured = r.get("depth_high_share")
            if measured is not None and (r["vendor_depth"] < args.min_depth) != (measured < args.min_depth):
                disagree.append(r)

    keys = ["clip_id", "file", "flags"] + sorted({k for r in rows for k in r} - {"clip_id", "file", "flags"})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def hours(rs: list[dict]) -> float:
        return sum(r["seconds"] for r in rs) / 3600

    ok = [r for r in rows if "error" not in r]
    by_flag = {name: [r for r in ok if name in r["flags"].split()] for name in
               ("rate_change", "dropped_frames", "stall", "low_depth")}
    print(f"{len(rows)} clips, {hours(ok):.2f} h readable, {len(rows) - len(ok)} unreadable -> {args.out}")
    for name, rs in by_flag.items():
        print(f"  {name}: {len(rs)} clips, {hours(rs):.2f} h")
    steady_off = [r for r in ok if not r["rate_changes"] and abs(r["recorded_fps"] - r["metadata_fps"]) > 0.5]
    print(f"  steady rate unlike the metadata's (reported, not flagged): {len(steady_off)} clips")
    if args.compare:
        print(f"  vendor index disagrees on usable depth: {len(disagree)} clips, {hours(disagree):.2f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
