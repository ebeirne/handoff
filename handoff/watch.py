"""Check clips the moment they land: a capture-time loop for collection sites and upload folders.

Polls a folder. Once a new clip has stopped growing (same size on two polls), it runs the spec's checks on it and
prints the verdict with the reasons and times, so the collector can re-shoot while still on site. Every result is
appended to a JSON-lines log. Catching a bad clip where it was recorded is the cheapest place to catch it.

    handoff watch uploads/ --spec spec.yaml
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .checks import HandDetector
from .run import _check_clip, _verdict
from .spec import Spec
from .video import find_clips


class Watcher:
    def __init__(self, folder: str | Path, spec: Spec, hand_model: str | None = None, log: str | Path | None = None):
        self.folder = Path(folder)
        self.spec = spec
        self.detector = HandDetector(hand_model, spec.checks["hands_visible"].get("detection_confidence", 0.1)) \
            if hand_model and "hands_visible" in spec.checks else None
        self.log = Path(log) if log else None
        self._seen: dict[Path, tuple[int, int]] = {}   # last size and mtime observed
        self._done: dict[Path, tuple[int, int]] = {}   # size and mtime when checked

    def poll(self) -> list[dict]:
        """Check every clip that is new or changed and has stopped growing since the last poll."""
        results = []
        for path in find_clips(self.folder):
            try:
                st = path.stat()
            except OSError:
                continue
            key = (st.st_size, st.st_mtime_ns)
            if self._done.get(path) == key:
                continue
            if self._seen.get(path) != key:
                self._seen[path] = key  # first sight, or still being written
                continue
            rel = path.relative_to(self.folder).as_posix()
            verdict, _sig, _duration = _check_clip(path, rel, self.spec, self.detector)
            verdict.verdict = _verdict(verdict.checks, self.spec.reject_on, self.spec.downgrade_on)
            record = {
                "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "clip": rel,
                "verdict": verdict.verdict,
                "failed": [{"check": c.name, "reason": c.reason, "evidence": c.evidence} for c in verdict.failed],
            }
            self._done[path] = key
            results.append(record)
            if self.log:
                with self.log.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
        return results

    def close(self) -> None:
        if self.detector is not None:
            self.detector.close()


def describe(record: dict) -> str:
    """One line a collector can act on."""
    head = f"{record['verdict'].upper():9} {record['clip']}"
    if not record["failed"]:
        return head
    return head + "  <- " + "; ".join(f"{f['check']}: {f['reason']}" for f in record["failed"])


def run_forever(watcher: Watcher, interval: float = 2.0, printer=print) -> None:
    try:
        while True:
            for record in watcher.poll():
                printer(describe(record))
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        watcher.close()
