"""Compare a Handoff run on another machine with the laptop reference in tests/fixtures/repro_demo.json.

The two-party promise only holds if a second organization's hardware reaches the same verdicts. This confirms both
machines started from identical bytes, compares the fingerprint, and reports how far every check's value moved, so
we can see how close a verdict came to flipping even when the fingerprints match.

    python tools/check_repro.py out/results.json tests/fixtures/repro_demo.json --clips delivery --hand-model models/hand_landmarker.task

Exit codes: 0 same fingerprint, 1 different fingerprint, 2 the inputs weren't the same bytes (the test didn't run).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results")
    ap.add_argument("fixture")
    ap.add_argument("--clips", required=True)
    ap.add_argument("--hand-model", required=True)
    args = ap.parse_args()

    ref = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    run = json.loads(Path(args.results).read_text(encoding="utf-8"))

    bad = [name for name, digest in ref["clips"].items() if _sha256(Path(args.clips) / name) != digest]
    if _sha256(Path(args.hand_model)) != ref["hand_model_sha256"]:
        bad.append("hand model")
    if bad:
        print(f"Inputs differ from the reference ({', '.join(bad)}), so this isn't a fair comparison.")
        return 2

    same = run["fingerprint"] == ref["fingerprint"] and run["complete"]
    print("## Cross-machine fingerprint\n")
    print(f"- This machine: {platform.system()} {platform.release()}, {platform.machine()}, Python {platform.python_version()}")
    print(f"- Fingerprint here:   `{run['fingerprint'][:16]}` (complete: {run['complete']})")
    print(f"- Laptop reference:   `{ref['fingerprint'][:16]}`")
    print(f"- **{'SAME' if same else 'DIFFERENT'}**\n")

    print("| Clip | Verdict here | Laptop | Largest value change |")
    print("|---|---|---|---|")
    worst: dict[str, float] = {}
    for clip in run["clips"]:
        name = clip["clip"]
        here = {c["name"]: c for c in clip["checks"]}
        moves = []
        for check, expected in ref["checks"][name].items():
            a, b = here.get(check, {}).get("value"), expected["value"]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                rel = abs(a - b) / max(abs(b), 1e-9)
                worst[check] = max(worst.get(check, 0.0), rel)
                moves.append((rel, check, a, b))
            if here.get(check, {}).get("passed") != expected["passed"]:
                moves.append((float("inf"), check + " FLIPPED", a, b))
        top = max(moves) if moves else None
        change = f"{top[1]}: {top[3]:.4g} → {top[2]:.4g}" if top and top[0] > 0 else "none"
        print(f"| {name} | {clip['verdict']} | {ref['verdicts'][name]} | {change} |")

    print("\nLargest relative change per check, across all clips: "
          + ", ".join(f"{k} {v:.2%}" for k, v in sorted(worst.items(), key=lambda kv: -kv[1])))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
