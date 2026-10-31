"""Measure how often the checks agree with a person.

1. `handoff label-sheet results.json` writes a CSV with one row per clip.
2. A person fills in `human_verdict` (ok or bad) and, for bad clips, `human_bad_checks`
   (the check names that should have failed, separated by semicolons).
3. `handoff accuracy results.json labels.csv` scores the tool against those labels.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

GOOD = {"ok", "good", "yes", "y", "pass", "1"}
BAD = {"bad", "no", "n", "fail", "flag", "reject", "0"}
FIELDS = ["clip", "tool_verdict", "tool_failed_checks", "human_verdict", "human_bad_checks", "notes"]


def _load_results(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tool_failed(clip: dict) -> set[str]:
    return {c["name"] for c in clip["checks"] if c["passed"] is False}


def write_label_sheet(results_path: str | Path, out_path: str | Path) -> int:
    data = _load_results(results_path)
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for clip in data["clips"]:
            w.writerow({
                "clip": clip["clip"],
                "tool_verdict": clip["verdict"],
                "tool_failed_checks": ";".join(sorted(_tool_failed(clip))),
                "human_verdict": "",
                "human_bad_checks": "",
                "notes": "",
            })
    return len(data["clips"])


def _scores(tp: int, fp: int, fn: int, tn: int) -> dict:
    total = tp + fp + fn + tn
    return {
        "labeled": total,
        "true_flags": tp,
        "false_flags": fp,
        "missed": fn,
        "true_passes": tn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,  # of what we flagged, how much was really bad
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,  # of what was really bad, how much we caught
        "agreement": round((tp + tn) / total, 3) if total else None,
    }


def accuracy(results_path: str | Path, labels_path: str | Path) -> dict:
    data = _load_results(results_path)
    tool = {c["clip"]: c for c in data["clips"]}
    check_names = sorted({c["name"] for clip in data["clips"] for c in clip["checks"] if c["passed"] is not None})

    clip_counts = [0, 0, 0, 0]  # tp, fp, fn, tn
    per_check = {name: [0, 0, 0, 0] for name in check_names}
    unknown, unlabeled = [], 0
    with Path(labels_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            human = (row.get("human_verdict") or "").strip().lower()
            if human not in GOOD | BAD:
                unlabeled += 1
                continue
            clip = tool.get(row["clip"])
            if clip is None:
                unknown.append(row["clip"])
                continue
            human_bad = human in BAD
            tool_bad = clip["verdict"] != "pass"
            clip_counts[(0 if tool_bad else 2) if human_bad else (1 if tool_bad else 3)] += 1

            human_checks = {c.strip() for c in (row.get("human_bad_checks") or "").split(";") if c.strip()}
            # Score individual checks only when the labeler said which checks should fail (or that the clip is fine).
            if human_bad and not human_checks:
                continue
            failed = _tool_failed(clip)
            ran = {c["name"] for c in clip["checks"] if c["passed"] is not None}
            for name in ran:
                should, did = name in human_checks, name in failed
                per_check[name][(0 if did else 2) if should else (1 if did else 3)] += 1

    return {
        "clips": _scores(*clip_counts),
        "checks": {name: _scores(*counts) for name, counts in per_check.items() if sum(counts)},
        "unlabeled_rows": unlabeled,
        "clips_not_in_results": unknown,
    }


def format_accuracy(report: dict) -> str:
    def pct(x):
        return "  n/a" if x is None else f"{x:5.0%}"

    lines = ["", f"{'':18}{'labeled':>8}{'precision':>11}{'recall':>8}{'agree':>7}  false flags / missed"]
    rows = [("all clips", report["clips"])] + sorted(report["checks"].items())
    for name, s in rows:
        lines.append(
            f"{name:18}{s['labeled']:>8}{pct(s['precision']):>11}{pct(s['recall']):>8}{pct(s['agreement']):>7}"
            f"  {s['false_flags']} / {s['missed']}"
        )
    if report["unlabeled_rows"]:
        lines.append(f"\n{report['unlabeled_rows']} rows without a human_verdict were skipped.")
    if report["clips_not_in_results"]:
        lines.append(f"Not in results: {', '.join(report['clips_not_in_results'])}")
    lines.append("\nPrecision: of the clips the tool flagged, how many were really bad.")
    lines.append("Recall: of the clips that were really bad, how many the tool caught.")
    return "\n".join(lines)
