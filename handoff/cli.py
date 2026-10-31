"""Command line: `handoff check`, `init-spec`, `demo-clips`, `label-sheet`, `accuracy`, `fleet`, `repair`,
`init-fleet-spec`, `watch`, `passport`, `verify-passport` and `console`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import resources
from pathlib import Path

import yaml

from .report import write_outputs
from .run import VERDICTS, run
from .spec import SpecError, load_spec


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        spec = load_spec(args.spec)
    except (OSError, SpecError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    clips = Path(args.clips)
    if not clips.is_dir():
        print(f"error: {clips} is not a folder", file=sys.stderr)
        return 2

    def progress(i: int, n: int, name: str) -> None:
        if not args.quiet:
            print(f"[{i}/{n}] {name}", file=sys.stderr)

    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    result = run(spec, clips, hand_model=args.hand_model, progress=progress, workers=workers)
    paths = write_outputs(result, args.out)
    counts = {k: sum(v.verdict == k for v in result.verdicts) for k in VERDICTS}
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(
        f"{len(result.verdicts)} clips: {counts['pass']} pass, {counts['downgrade']} downgraded, "
        f"{counts['flag']} flagged, {counts['reject']} rejected"
    )
    print(f"report: {paths['html']}")
    if not result.complete:
        print(f"INCOMPLETE: {', '.join(result.not_run)} did not run; passes don't cover those checks", file=sys.stderr)
        return 3
    return 0


def _cmd_init_spec(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"error: {target} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    text = resources.files("handoff").joinpath("example_spec.yaml").read_text(encoding="utf-8")
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}")
    return 0


def _cmd_demo_clips(args: argparse.Namespace) -> int:
    from .synthetic import make_demo_set

    folder = make_demo_set(args.path)
    print(f"wrote synthetic demo clips to {folder}")
    return 0


def _cmd_label_sheet(args: argparse.Namespace) -> int:
    from .labels import write_label_sheet

    target = Path(args.out)
    if target.exists() and not args.force:
        print(f"error: {target} exists (use --force to overwrite; it may hold your labels)", file=sys.stderr)
        return 2
    n = write_label_sheet(args.results, target)
    print(f"wrote {n} rows to {target}. Fill in human_verdict (ok/bad) and, for bad clips, human_bad_checks.")
    return 0


def _cmd_accuracy(args: argparse.Namespace) -> int:
    from .labels import accuracy, format_accuracy

    report = accuracy(args.results, args.labels)
    print(format_accuracy(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _cmd_fleet(args: argparse.Namespace) -> int:
    from .fleet import lerobot_defaults, lerobot_fit, load_fleet_spec, load_logs, triage, write_outputs

    logs = Path(args.logs)
    if not logs.is_dir():
        print(f"error: {logs} is not a folder", file=sys.stderr)
        return 2
    try:
        spec = load_fleet_spec(args.spec)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    lerobot = (logs / "meta" / "info.json").exists()
    required_by_user = bool(spec["required"])
    if lerobot:
        spec = lerobot_defaults(spec)
    workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    try:
        sessions = load_logs(logs, spec, workers=workers)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not sessions:
        print(f"error: no .mcap files or LeRobot episodes in {logs}", file=sys.stderr)
        return 2
    if lerobot and not required_by_user:  # a spec's own `required` list is kept as written
        spec = lerobot_fit(spec, sessions)
    report = triage(sessions, spec)
    report["source"] = str(logs.resolve())
    paths = write_outputs(report, args.out)
    s = report["summary"]
    print(f"{s['sessions']} sessions: {s['keep']} keep, {s['review']} review, {s['drop']} drop "
          f"({s['hours']['keep']:.2f} usable hours of {sum(s['hours'].values()):.2f})")
    print(f"report: {paths['html']}")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    import hashlib

    from value.repair import repair_dataset

    from .fleet import repair_shifts

    root = Path(args.dataset)
    if not (root / "meta" / "info.json").exists():
        print(f"error: {root} is not a LeRobot dataset (no meta/info.json)", file=sys.stderr)
        return 2
    raw = Path(args.fleet).read_bytes()
    report = json.loads(raw)
    if report.get("source") and Path(report["source"]).resolve() != root.resolve():
        print(f"error: the fleet report is for {report['source']}, not {root}", file=sys.stderr)
        return 2
    shifts = repair_shifts(report)
    if not shifts:
        print("the fleet report has no lagged episodes; nothing to repair")
        return 0
    source = {"tool": "handoff fleet", "version": report.get("tool_version"), "report": str(args.fleet),
              "report_sha256": hashlib.sha256(raw).hexdigest()}
    try:
        log = repair_dataset(root, args.out, shifts, offsets_from=source)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for r in log["repaired_episodes"]:
        print(f"  episode {r['episode']}: actions shifted {r['shift_frames']:+d} frames")
    print(f"repaired {len(log['repaired_episodes'])} episodes into {args.out} (see REPAIR_LOG.json)")
    return 0


def _cmd_init_fleet_spec(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if target.exists() and not args.force:
        print(f"error: {target} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    target.write_text(resources.files("handoff").joinpath("example_fleet_spec.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"wrote {target}")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    from .watch import Watcher, run_forever

    try:
        spec = load_spec(args.spec)
    except (OSError, SpecError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: {folder} is not a folder", file=sys.stderr)
        return 2
    watcher = Watcher(folder, spec, hand_model=args.hand_model, log=args.log)
    print(f"watching {folder} (Ctrl+C to stop); results also go to {args.log}", file=sys.stderr)
    run_forever(watcher, interval=args.interval, printer=lambda line: print(line, flush=True))
    return 0


def _cmd_passport(args: argparse.Namespace) -> int:
    from .passport import build, to_html

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"error: {folder} is not a folder", file=sys.stderr)
        return 2
    out = Path(args.out)
    p = build(folder, results=args.results, consent=args.consent, license=args.license, name=args.name,
              skip=[out, out.with_suffix(".html")])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=2), encoding="utf-8")
    out.with_suffix(".html").write_text(to_html(p), encoding="utf-8")
    line = f"passport for {p['totals']['files']} files ({p['totals']['clips']} clips), seal {p['passport_sha256'][:16]}"
    if "consent" in p:
        c = p["consent"]
        line += f"; consent covers {c['covered']} of {c['clips']} clips"
    print(line)
    print(f"wrote {out} and {out.with_suffix('.html')}")
    return 1 if "consent" in p and (p["consent"]["missing"] or p["consent"]["expired"]) else 0


def _cmd_verify_passport(args: argparse.Namespace) -> int:
    from .passport import verify

    passport_path = Path(args.passport)
    p = json.loads(passport_path.read_text(encoding="utf-8"))
    r = verify(p, args.folder, skip=[passport_path, passport_path.with_suffix(".html")])
    if r["ok"]:
        print(f"OK: all {len(p['files'])} files match the passport, and the passport is unedited")
        return 0
    if not r["passport_intact"]:
        print("the passport itself was edited after it was sealed")
    for key in ("changed", "missing", "added"):
        for path in r[key]:
            print(f"{key}: {path}")
    return 1


def _cmd_console(args: argparse.Namespace) -> int:
    from .console import build, to_html

    if not any([args.check, args.fleet, args.repair, args.passport, args.watch]):
        print("error: give at least one of --check, --fleet, --repair, --passport, --watch", file=sys.stderr)
        return 2
    data = build(args.check or [], args.fleet or [], args.repair or [], args.passport or [], args.watch or [])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(to_html(data), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="handoff", description="Acceptance checks for robot training video.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="check a folder of clips against a spec")
    check.add_argument("clips", help="folder of video clips (searched recursively)")
    check.add_argument("--spec", required=True, help="YAML delivery spec")
    check.add_argument("--out", default="reports/latest", help="output folder (default: reports/latest)")
    check.add_argument("--hand-model", help="path to MediaPipe hand_landmarker.task, enables hands_visible")
    check.add_argument("--workers", type=int, default=1, help="clips processed in parallel (0 = all cores but one)")
    check.add_argument("--quiet", action="store_true", help="no per-clip progress")
    check.set_defaults(func=_cmd_check)

    init = sub.add_parser("init-spec", help="write an example spec to start from")
    init.add_argument("path", nargs="?", default="spec.yaml")
    init.add_argument("--force", action="store_true")
    init.set_defaults(func=_cmd_init_spec)

    demo = sub.add_parser("demo-clips", help="write synthetic clips with known defects, to try the tool")
    demo.add_argument("path", nargs="?", default="data/demo")
    demo.set_defaults(func=_cmd_demo_clips)

    sheet = sub.add_parser("label-sheet", help="write a CSV for a person to label clips good or bad")
    sheet.add_argument("results", help="results.json from a check run")
    sheet.add_argument("--out", default="labels.csv")
    sheet.add_argument("--force", action="store_true")
    sheet.set_defaults(func=_cmd_label_sheet)

    acc = sub.add_parser("accuracy", help="score a check run against a person's labels")
    acc.add_argument("results", help="results.json from a check run")
    acc.add_argument("labels", help="the filled-in label sheet")
    acc.add_argument("--out", help="also write the scores as JSON")
    acc.set_defaults(func=_cmd_accuracy)

    fleet = sub.add_parser("fleet", help="triage robot logs: which sessions to keep, review or drop, and why")
    fleet.add_argument("logs", help="folder of .mcap files (subfolders = robots), or a LeRobot dataset folder")
    fleet.add_argument("--spec", help="fleet spec YAML (see init-fleet-spec); optional")
    fleet.add_argument("--out", default="reports/fleet", help="output folder (default: reports/fleet)")
    fleet.add_argument("--workers", type=int, default=1, help="log files read in parallel (0 = all cores but one)")
    fleet.set_defaults(func=_cmd_fleet)

    repair = sub.add_parser("repair", help="write a re-aligned copy of a LeRobot dataset from a fleet report's lag findings")
    repair.add_argument("dataset", help="the LeRobot dataset folder the fleet report was made on")
    repair.add_argument("--fleet", required=True, help="fleet.json from `handoff fleet` on that dataset")
    repair.add_argument("--out", required=True, help="new or empty folder for the repaired copy")
    repair.set_defaults(func=_cmd_repair)

    watch = sub.add_parser("watch", help="check clips as they land in a folder, for feedback while recording")
    watch.add_argument("folder", help="upload or recording folder to watch")
    watch.add_argument("--spec", required=True, help="YAML delivery spec")
    watch.add_argument("--hand-model", help="path to MediaPipe hand_landmarker.task, enables hands_visible")
    watch.add_argument("--interval", type=float, default=2.0, help="seconds between polls (default: 2)")
    watch.add_argument("--log", default="watch_log.jsonl", help="JSON-lines log of every result")
    watch.set_defaults(func=_cmd_watch)

    passport = sub.add_parser("passport", help="seal a delivery: file hashes, consent coverage, licence, acceptance")
    passport.add_argument("folder", help="the delivery folder")
    passport.add_argument("--results", help="results.json from `handoff check` on this delivery")
    passport.add_argument("--consent", help="CSV with columns clip, consent_id and optional expires (YYYY-MM-DD)")
    passport.add_argument("--license", help="licence the data is delivered under, e.g. 'CC BY 4.0'")
    passport.add_argument("--name", help="delivery name (default: folder name)")
    passport.add_argument("--out", default="passport.json", help="where to write the passport (default: passport.json)")
    passport.set_defaults(func=_cmd_passport)

    verify_p = sub.add_parser("verify-passport", help="check a delivery still matches its passport")
    verify_p.add_argument("passport", help="passport.json")
    verify_p.add_argument("folder", help="the delivery folder")
    verify_p.set_defaults(func=_cmd_verify_passport)

    console = sub.add_parser("console", help="one page over checks, fleet triage, repairs, passports and watch logs")
    console.add_argument("--check", action="append", help="results.json from `handoff check` (repeatable)")
    console.add_argument("--fleet", action="append", help="fleet.json from `handoff fleet` (repeatable)")
    console.add_argument("--repair", action="append", help="REPAIR_LOG.json from a repair (repeatable)")
    console.add_argument("--passport", action="append", help="passport.json (repeatable)")
    console.add_argument("--watch", action="append", help="watch log, JSON lines (repeatable)")
    console.add_argument("--out", default="reports/console.html")
    console.set_defaults(func=_cmd_console)

    init_fleet = sub.add_parser("init-fleet-spec", help="write an example fleet spec to start from")
    init_fleet.add_argument("path", nargs="?", default="fleet.yaml")
    init_fleet.add_argument("--force", action="store_true")
    init_fleet.set_defaults(func=_cmd_init_fleet_spec)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
