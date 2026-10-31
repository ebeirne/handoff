"""Run a spec over a folder of clips and produce verdicts."""

from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import util as mp_util
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .checks import PER_CLIP, CheckResult, HandDetector, check_hands_visible, check_readable, find_duplicates, signature
from .spec import Spec
from .video import find_clips, sample_clip

VERDICTS = ("pass", "downgrade", "flag", "reject")


@dataclass
class ClipVerdict:
    clip: str
    verdict: str  # pass | downgrade | flag | reject
    checks: list[CheckResult]
    info: dict
    thumbnail: bytes | None = None

    @property
    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if c.passed is False]


@dataclass
class RunResult:
    spec: Spec
    clips_dir: Path
    verdicts: list[ClipVerdict]
    coverage: dict | None
    video_seconds: float
    compute_seconds: float
    workers: int = 1
    tool_version: str = __version__
    warnings: list[str] = field(default_factory=list)
    # Checks the spec asked for that couldn't run (missing model, missing library). A run with any is incomplete,
    # and its passes must not be read as "checked and clean".
    not_run: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.not_run


def _verdict(checks: list[CheckResult], reject_on: frozenset, downgrade_on: frozenset) -> str:
    failed = {c.name for c in checks if c.passed is False}
    if failed & reject_on:
        return "reject"
    if failed - downgrade_on:
        return "flag"
    return "downgrade" if failed else "pass"


def _thumbnail(frames) -> bytes | None:
    import cv2

    if not frames:
        return None
    frame = frames[len(frames) // 2]
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (160, max(1, round(h * 160 / w))))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None


def _check_clip(path: Path, rel: str, spec: Spec, detector: HandDetector | None):
    """Everything per clip. Returns (verdict-in-progress, duplicate signature, seconds of video)."""
    sample = sample_clip(path, spec.sampling_fps, spec.max_frames)
    checks = [check_readable(sample, {})]
    sig = None
    if sample.info.readable:
        for name, params in spec.checks.items():
            if name in PER_CLIP:
                checks.append(PER_CLIP[name](sample, params))
            elif name == "hands_visible":
                checks.append(check_hands_visible(sample, params, detector))
        sig = signature(sample)
    info = {
        "fps": round(sample.info.fps, 3),
        "width": sample.info.width,
        "height": sample.info.height,
        "duration_s": round(sample.info.duration_s, 3),
        "frames_sampled": len(sample.frames),
    }
    duration = sample.info.duration_s if sample.info.readable else 0.0
    return ClipVerdict(rel, "pass", checks, info, _thumbnail(sample.frames)), sig, duration


# One hand detector per worker process, created once rather than per clip.
_worker_detector: HandDetector | None = None


def _init_worker(hand_model: str | None, confidence: float) -> None:
    global _worker_detector
    _worker_detector = HandDetector(hand_model, confidence) if hand_model else None
    if _worker_detector is not None:
        # Runs when the worker exits, before the interpreter tears down the modules MediaPipe needs to close.
        mp_util.Finalize(_worker_detector, _worker_detector.close, exitpriority=10)


def _check_clip_in_worker(path: str, rel: str, spec: Spec):
    return _check_clip(Path(path), rel, spec, _worker_detector)


def _coverage(spec: Spec, clips_dir: Path, verdicts: list[ClipVerdict], warnings: list[str]) -> dict | None:
    cov = spec.coverage
    if not cov:
        return None
    meta_path = Path(cov["metadata"])
    if not meta_path.is_absolute():
        candidates = [clips_dir / meta_path, spec.base_dir / meta_path]
        meta_path = next((p for p in candidates if p.exists()), candidates[0])
    if not meta_path.exists():
        warnings.append(f"coverage metadata not found: {meta_path}")
        return None

    key, fieldname = cov.get("key", "clip"), cov["field"]
    labels: dict[str, str] = {}
    with meta_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels[row[key]] = row.get(fieldname, "")

    targets: dict = cov.get("targets") or {}
    empty = dict.fromkeys(VERDICTS, 0)
    counts: dict[str, dict[str, int]] = {}
    for v in verdicts:
        label = labels.get(v.clip) or labels.get(Path(v.clip).name) or "(unlabeled)"
        counts.setdefault(label, dict(empty))[v.verdict] += 1

    rows = []
    for label in sorted(set(counts) | set(targets)):
        c = counts.get(label, empty)
        target = targets.get(label)
        rows.append({
            "value": label,
            "passed": c["pass"],
            "downgraded": c["downgrade"],
            "flagged": c["flag"],
            "rejected": c["reject"],
            "target": target,
            "shortfall": max(0, target - c["pass"]) if target is not None else None,
        })
    return {"field": fieldname, "rows": rows}


def run(
    spec: Spec,
    clips_dir: str | Path,
    hand_model: str | Path | None = None,
    progress=None,
    workers: int = 1,
) -> RunResult:
    clips_dir = Path(clips_dir)
    started = time.perf_counter()
    warnings: list[str] = []

    detector = None
    not_run: list[str] = []
    confidence = spec.checks.get("hands_visible", {}).get("detection_confidence", 0.1)
    if "hands_visible" in spec.checks:
        if hand_model:
            try:
                detector = HandDetector(hand_model, confidence)
            except Exception as exc:  # missing mediapipe, bad model path, missing system library
                warnings.append(f"hands_visible did not run: {exc}")
                hand_model = None
        else:
            warnings.append("hands_visible did not run: pass --hand-model (see README)")
        if detector is None:
            not_run.append("hands_visible")
    else:
        hand_model = None

    clips = [(p, p.relative_to(clips_dir).as_posix()) for p in find_clips(clips_dir)]
    results: dict[str, tuple] = {}
    if workers > 1 and len(clips) > 1:
        if detector is not None:
            detector.close()  # it only proved the model loads; each worker builds its own
        detector = None
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker, initargs=(str(hand_model) if hand_model else None, confidence)
        ) as pool:
            futures = {pool.submit(_check_clip_in_worker, str(p), rel, spec): rel for p, rel in clips}
            for i, fut in enumerate(as_completed(futures), 1):
                rel = futures[fut]
                results[rel] = fut.result()
                if progress:
                    progress(i, len(clips), rel)
    else:
        try:
            for i, (p, rel) in enumerate(clips, 1):
                if progress:
                    progress(i, len(clips), rel)
                results[rel] = _check_clip(p, rel, spec, detector)
        finally:
            if detector is not None:
                detector.close()

    verdicts = [results[rel][0] for _, rel in clips]
    signatures = {rel: results[rel][1] for _, rel in clips if results[rel][1] is not None}
    video_seconds = sum(results[rel][2] for _, rel in clips)

    if "duplicates" in spec.checks:
        max_hamming = spec.checks["duplicates"].get("max_hamming", 6)
        dupes = find_duplicates(signatures, max_hamming)
        for v in verdicts:
            if v.clip not in signatures:
                continue
            if v.clip in dupes:
                original, dist = dupes[v.clip]
                v.checks.append(CheckResult("duplicates", False, dist, f"> {max_hamming}", f"near-duplicate of {original}"))
            else:
                v.checks.append(CheckResult("duplicates", True, None, f"> {max_hamming}", "no near-duplicate found"))

    for v in verdicts:
        v.verdict = _verdict(v.checks, spec.reject_on, spec.downgrade_on)

    coverage = _coverage(spec, clips_dir, verdicts, warnings)
    elapsed = time.perf_counter() - started
    return RunResult(
        spec, clips_dir, verdicts, coverage, video_seconds, elapsed, workers=max(1, workers), warnings=warnings, not_run=not_run
    )
