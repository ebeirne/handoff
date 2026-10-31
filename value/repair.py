"""Write a re-aligned copy of a LeRobot dataset. The original is never modified.

Each repaired episode's actions are shifted by the offset the sync scan measured. The ends are padded by repeating
the first or last action, so a repaired episode loses |offset| frames of real action data at one end. Everything
else (other columns, other episodes, file layout, meta/) is copied unchanged, and REPAIR_LOG.json records exactly
what was shifted and by how much.

The offsets can come from Handoff's own scan (`value.scan --repair`) or from another tool's report. With rdq's,
which needs no training and so gives the same offsets on every machine:

    rdq check data/my_dataset --json rdq.json
    python -m value.repair --root data/my_dataset --out data/my_dataset_fixed --from-rdq rdq.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import shift_actions


def shifts_from_rdq(report: str | Path) -> tuple[dict[int, int], dict]:
    """Re-aligning shifts from an rdq JSON report's sync finding, plus a record of where they came from."""
    raw = Path(report).read_bytes()
    r = json.loads(raw)
    tool = r.get("tool", {})
    source = {"tool": tool.get("name", "rdq"), "version": tool.get("version"), "report": str(report),
              "report_sha256": hashlib.sha256(raw).hexdigest(), "dataset": r.get("dataset", {})}
    found = [f for f in r.get("findings", []) if f.get("check") == "sync.action_state_lag"]
    if not found:
        return {}, source
    d = found[0]["details"]
    # rdq gives each flagged episode's lag next to the dataset's typical lag. Shifting the actions by the difference,
    # the other way, re-aligns the episode.
    return {int(e): int(d["typical_lag"]) - int(lag) for e, lag in d["lags"].items()}, source


def _check_same_dataset(root: Path, reported: dict) -> None:
    # Copies of a dataset share its counts, so a report on one copy must not be applied to another.
    path = reported.get("path")
    if path and Path(path).exists() and Path(path).resolve() != Path(root).resolve():
        raise ValueError(f"the report is for {path}, not {root}")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    for ours, theirs in (("total_episodes", "episodes"), ("total_frames", "frames")):
        if theirs in reported and info.get(ours) != reported[theirs]:
            raise ValueError(f"the report describes a different dataset: {theirs} {reported[theirs]} there, "
                             f"{info.get(ours)} in {root}")


def repair_dataset(root: str | Path, out: str | Path, shifts: dict[int, int], act_key: str = "action",
                   offsets_from: dict | None = None) -> dict:
    root, out = Path(root), Path(out)
    if out.resolve() == root.resolve():
        raise ValueError("write the repaired copy to a different folder; the original stays untouched")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    for item in root.iterdir():
        if item.name != "data":
            (shutil.copytree if item.is_dir() else shutil.copy2)(item, out / item.name)

    done: dict[int, dict] = {}
    for src in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(src)
        ep = table.column("episode_index").to_numpy()
        frame = table.column("frame_index").to_numpy()
        actions = table.column(act_key)
        values = np.array(actions.to_pylist(), dtype=np.float32)
        changed = False
        for idx, lag in shifts.items():
            rows = np.where(ep == idx)[0]
            if not len(rows) or lag == 0:
                continue
            if idx in done:
                raise ValueError(f"episode {idx} spans several data files; repair it by hand")
            rows = rows[np.argsort(frame[rows])]
            values[rows] = shift_actions(values[rows], int(lag))
            done[idx] = {"episode": int(idx), "shift_frames": int(lag), "frames": int(len(rows)), "file": str(src.relative_to(root))}
            changed = True
        dest = out / src.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if changed:
            new = pa.array(values.tolist(), type=actions.type)
            table = table.set_column(table.schema.get_field_index(act_key), act_key, new)
            pq.write_table(table, dest)
        else:
            shutil.copy2(src, dest)

    log = {
        "source": str(root),
        "offsets_from": offsets_from or {"tool": "handoff value.scan"},
        "repaired_episodes": sorted(done.values(), key=lambda d: d["episode"]),
        "note": "Actions shifted by shift_frames (positive: actions moved earlier). Ends padded by repeating the "
                "nearest action. meta/stats.json was copied unchanged and may differ slightly from the repaired data.",
    }
    missing = sorted(set(shifts) - set(done) - {i for i, s in shifts.items() if s == 0})
    if missing:
        log["not_found"] = missing
    (out / "REPAIR_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a re-aligned copy of a LeRobot dataset from rdq's sync findings.")
    ap.add_argument("--root", required=True, help="local LeRobot dataset that rdq checked")
    ap.add_argument("--out", required=True, help="new or empty folder for the repaired copy")
    ap.add_argument("--from-rdq", required=True, metavar="REPORT", help="JSON written by `rdq check --json`")
    ap.add_argument("--act-key", default="action")
    args = ap.parse_args()

    shifts, source = shifts_from_rdq(args.from_rdq)
    _check_same_dataset(Path(args.root), source["dataset"])
    if not shifts:
        print("the report has no action/state lag finding; nothing to repair")
        return 0
    log = repair_dataset(args.root, args.out, shifts, args.act_key, offsets_from=source)
    for r in log["repaired_episodes"]:
        print(f"  episode {r['episode']}: actions shifted {r['shift_frames']:+d} frames")
    print(f"repaired {len(log['repaired_episodes'])} episodes into {args.out} (see REPAIR_LOG.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
