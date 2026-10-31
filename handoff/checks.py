"""Per-clip checks. Each returns a CheckResult with a reason a person can act on."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .video import WORK_WIDTH, Sample, rate_runs


@dataclass
class CheckResult:
    name: str
    passed: bool | None  # None means skipped
    value: float | None = None
    threshold: str = ""
    reason: str = ""
    evidence: list[list[float]] | None = None  # [start_s, end_s] stretches where the problem shows

    def __post_init__(self) -> None:
        # NumPy scalars break `passed is False` and JSON output; store plain Python values.
        if self.passed is not None:
            self.passed = bool(self.passed)
        if self.value is not None:
            self.value = float(self.value)

    def to_dict(self) -> dict:
        return asdict(self)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _ranges(times: list[float], mask: list[bool]) -> list[list[float]]:
    """Merge flagged sample times into [start, end] stretches, so a reviewer can jump to them."""
    step = (times[1] - times[0]) if len(times) > 1 else 1.0
    out: list[list[float]] = []
    for t, m in zip(times, mask):
        if not m:
            continue
        if out and t - out[-1][1] <= step * 1.5:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 1), round(b, 1)] for a, b in out]


def check_readable(sample: Sample, params: dict) -> CheckResult:
    info = sample.info
    if info.readable:
        return CheckResult("readable", True, reason="decodes")
    return CheckResult("readable", False, reason=f"unreadable: {info.error}")


def check_duration(sample: Sample, params: dict) -> CheckResult:
    d = sample.info.duration_s
    lo, hi = params.get("min_s", 0.0), params.get("max_s", float("inf"))
    ok = lo <= d <= hi
    reason = f"{d:.1f}s long" if ok else f"{d:.1f}s long, outside {lo}-{hi}s"
    return CheckResult("duration", ok, d, f"{lo}-{hi}s", reason)


def check_fps(sample: Sample, params: dict) -> CheckResult:
    fps = sample.info.fps
    lo = params.get("min", 0.0)
    ok = fps >= lo
    return CheckResult("fps", ok, fps, f">= {lo}", f"{fps:.1f} fps" + ("" if ok else f", below {lo}"))


def check_resolution(sample: Sample, params: dict) -> CheckResult:
    w, h = sample.info.width, sample.info.height
    min_w, min_h = params.get("min_width", 0), params.get("min_height", 0)
    ok = w >= min_w and h >= min_h
    reason = f"{w}x{h}" + ("" if ok else f", below {min_w}x{min_h}")
    return CheckResult("resolution", ok, float(w * h), f">= {min_w}x{min_h}", reason)


def check_bitrate(sample: Sample, params: dict) -> CheckResult:
    """Bits per second against what the frame size needs. Phone cameras record 1080p at roughly 8-20 Mbps; a 1080p file
    at 1-2 Mbps has usually been re-encoded by a messaging app on the way in, which smears fast hand motion."""
    min_mbps = params.get("min_mbps_1080p", 3.0)
    info = sample.info
    if info.duration_s <= 0 or not info.width or info.path.suffix.lower() == ".r3d":  # .r3d stores JPEG frames
        return CheckResult("bitrate", None, reason="skipped: no video stream to measure")
    mbps = info.path.stat().st_size * 8 / info.duration_s / 1e6
    need = min_mbps * (info.width * info.height) / (1920 * 1080)
    ok = mbps >= need
    reason = f"{mbps:.1f} Mbps at {info.width}x{info.height}"
    if not ok:
        reason += f", below {need:.1f}: probably re-encoded (a messaging app or editor) after capture"
    return CheckResult("bitrate", ok, mbps, f">= {min_mbps} Mbps at 1080p, scaled by frame size", reason)


def check_black_frames(sample: Sample, params: dict) -> CheckResult:
    level = params.get("level", 16)
    limit = params.get("max_fraction", 0.05)
    mask = [float(_gray(f).mean()) < level for f in sample.frames]
    frac = float(np.mean(mask)) if mask else 1.0
    ok = frac <= limit
    reason = f"{frac:.0%} of sampled frames are black" + ("" if ok else f" (limit {limit:.0%})")
    return CheckResult("black_frames", ok, frac, f"<= {limit:.0%}", reason, _ranges(sample.times, mask))


def check_frozen(sample: Sample, params: dict) -> CheckResult:
    limit = params.get("max_fraction", 0.2)
    min_diff = params.get("min_diff", 1.0)
    grays = [cv2.resize(_gray(f), (160, 120)).astype(np.float32) for f in sample.frames]
    mask = [float(np.abs(a - b).mean()) < min_diff for a, b in zip(grays, grays[1:])]
    frac = float(np.mean(mask)) if mask else 1.0
    ok = frac <= limit
    reason = f"{frac:.0%} of sampled intervals show no change" + ("" if ok else f" (limit {limit:.0%})")
    return CheckResult("frozen", ok, frac, f"<= {limit:.0%}", reason, _ranges(sample.times[1:], mask))


def check_blur(sample: Sample, params: dict) -> CheckResult:
    minimum = params.get("min_sharpness", 60.0)
    scores = [float(cv2.Laplacian(_gray(f), cv2.CV_64F).var()) for f in sample.frames]
    sharp = float(np.median(scores)) if scores else 0.0
    ok = sharp >= minimum
    reason = f"sharpness {sharp:.0f}" + ("" if ok else f", below {minimum:.0f} (blurry)")
    evidence = _ranges(sample.times, [s < minimum for s in scores])
    return CheckResult("blur", ok, sharp, f">= {minimum}", reason, evidence)


def _image_area(grays: list[np.ndarray]) -> np.ndarray | None:
    """Pixels that belong to the picture: a fisheye lens's black surround or a letterbox stays black in every frame,
    and counting it would call every frame of a fisheye clip too dark. None when there's no such border."""
    if len(grays) < 3:
        return None
    border = np.mean([g <= 5 for g in grays], axis=0) >= 0.95
    if not border.any() or border.mean() > 0.8:  # nothing to drop, or the clip is simply black
        return None
    return ~border


def check_exposure(sample: Sample, params: dict) -> CheckResult:
    """A frame is badly exposed when a large share of it is blown out or crushed; glare patches are fine. A border
    that's black in every frame (fisheye surround, letterbox) isn't part of the picture and isn't counted."""
    max_over = params.get("max_overexposed_area", 0.35)
    max_under = params.get("max_underexposed_area", 0.35)
    limit = params.get("max_bad_frames", 0.2)
    grays = [_gray(f) for f in sample.frames]
    area = _image_area(grays)
    bad_over = bad_under = 0
    mask = []
    for g in grays:
        px = g[area] if area is not None else g
        over, under = (px >= 250).mean() > max_over, (px <= 5).mean() > max_under
        bad_over += int(over)
        bad_under += int(under)
        mask.append(bool(over or under))
    n = len(sample.frames) or 1
    frac_over, frac_under = bad_over / n, bad_under / n
    frac = (bad_over + bad_under) / n
    ok = frac <= limit
    if ok:
        reason = f"{frac:.0%} of sampled frames badly exposed"
    else:
        parts = []
        if bad_over:
            parts.append(f"{frac_over:.0%} of frames blown out")
        if bad_under:
            parts.append(f"{frac_under:.0%} of frames too dark")
        reason = ", ".join(parts) + f" (limit {limit:.0%})"
    return CheckResult(
        "exposure",
        ok,
        frac,
        f"<= {limit:.0%} of frames with over {max_over:.0%} or under {max_under:.0%} clipped",
        reason,
        _ranges(sample.times, mask),
    )


def check_shake(sample: Sample, params: dict) -> CheckResult:
    """Shake is motion that changes direction frame to frame. Smooth head turns and pans move at a steady rate."""
    limit = params.get("max_jitter", 3.0)
    if not sample.bursts:
        return CheckResult("shake", None, reason="skipped: sampling too dense to measure frame-to-frame motion")
    jerks = []
    for a, b, c in sample.bursts:
        (dx1, dy1), _ = cv2.phaseCorrelate(a, b)
        (dx2, dy2), _ = cv2.phaseCorrelate(b, c)
        jerks.append(float(np.hypot(dx2 - dx1, dy2 - dy1)))
    scale = WORK_WIDTH / sample.bursts[0][0].shape[1]  # report in pixels at 640 wide
    jitter = float(np.median(jerks)) * scale
    ok = jitter <= limit
    reason = f"frame-to-frame jitter {jitter:.1f}px" + ("" if ok else f", above {limit} (shaky camera)")
    return CheckResult("shake", ok, jitter, f"<= {limit}px", reason)


def streams_file(clip: Path) -> Path:
    return clip.with_name(clip.stem + ".streams.json")


def check_streams(sample: Sample, params: dict) -> CheckResult:
    """Dropped data and sync across a clip's recorded streams (video, IMU, actions, gloves...).

    Reads `<clip>.streams.json`: {"streams": {"video": [t0, t1, ...], "imu": [...]}} with timestamps in seconds.
    This catches gaps, missing streams, misaligned starts and streams that stop early. It can't catch a logger that
    stamps the wrong time; that kind of offset only shows up in the data itself (see Handoff Value's consistency score).
    """
    import json

    required = list(params.get("required", []))
    max_offset = params.get("max_start_offset_s", 0.05)
    max_gap = params.get("max_gap_s", 0.25)
    min_cov = params.get("min_coverage", 0.98)
    threshold = f"start offset <= {max_offset}s, gaps <= {max_gap}s, coverage >= {min_cov:.0%}"

    path = streams_file(sample.info.path)
    if not path.exists():
        if required:
            return CheckResult("streams", False, None, threshold, f"no {path.name} for required streams {', '.join(required)}")
        return CheckResult("streams", None, reason=f"skipped: no {path.name}")
    try:
        streams = {k: np.asarray(v, dtype=float) for k, v in json.loads(path.read_text(encoding="utf-8"))["streams"].items()}
    except (ValueError, KeyError, TypeError) as exc:
        return CheckResult("streams", False, None, threshold, f"unreadable {path.name}: {exc}")

    problems: list[str] = []
    evidence: list[list[float]] = []
    missing = [s for s in required if s not in streams or len(streams[s]) < 2]
    if missing:
        problems.append(f"missing streams: {', '.join(missing)}")
    present = {k: v for k, v in streams.items() if len(v) >= 2}
    for name, ts in present.items():
        d = np.diff(ts)
        if (d <= 0).any():
            problems.append(f"{name} timestamps go backward or repeat")
            continue
        big = np.where(d > max_gap)[0]
        if len(big):
            worst = int(big[np.argmax(d[big])])
            problems.append(f"{name} drops data: {len(big)} gaps, longest {d[worst]:.2f}s at {_mmss(ts[worst])}")
            evidence += [[round(float(ts[i]), 1), round(float(ts[i + 1]), 1)] for i in big]
    if len(present) > 1:
        starts = {k: float(v[0]) for k, v in present.items()}
        offset = max(starts.values()) - min(starts.values())
        if offset > max_offset:
            late = max(starts, key=starts.get)
            problems.append(f"streams start {offset:.2f}s apart ({late} starts last)")
        spans = {k: float(v[-1] - v[0]) for k, v in present.items()}
        longest = max(spans.values()) or 1.0
        for name, span in spans.items():
            if span / longest < min_cov:
                problems.append(f"{name} stops early: covers {span / longest:.0%} of the episode")

    ok = not problems
    reason = "; ".join(problems) if problems else f"{len(present)} streams in sync, no gaps over {max_gap}s"
    return CheckResult("streams", ok, float(len(problems)), threshold, reason, sorted(evidence) or None)


def check_frame_timing(sample: Sample, params: dict) -> CheckResult:
    """Frame timing from the timestamp kept for every frame (Record3D .r3d): frames dropped, stalls, and a frame rate
    that changes partway through, as when a phone that heats up halves its rate. The clip still plays and looks fine,
    but a converter that assumes one rate per episode (LeRobot does) gets the timing wrong from that point on. A steady
    rate that differs from the metadata's is reported, not flagged: the fps check judges the rate actually recorded."""
    max_dropped = params.get("max_dropped_fraction", 0.05)
    max_gap = params.get("max_gap_s", 0.25)
    allow_change = params.get("allow_rate_change", False)
    threshold = f"dropped <= {max_dropped:.0%}, gaps <= {max_gap}s" + ("" if allow_change else ", one frame rate")
    t = sample.frame_times
    if t is None or len(t) < 2:
        return CheckResult("frame_timing", None, reason="skipped: this format has no per-frame timestamps")
    dt = np.diff(t)
    claimed = sample.claimed_fps or sample.info.fps or 1 / float(np.median(dt))
    runs, usual = rate_runs(t, claimed)
    missing = np.clip(np.round(dt * claimed / usual) - 1, 0, None)  # frames that belong between two recorded ones
    dropped = float(missing.sum() / (missing.sum() + len(t)))
    problems = []
    if len(runs) > 1 and not allow_change:
        steps = ", then ".join(f"{fps:g} fps" + ("" if i == len(runs) - 1 else f" until {_mmss(end)}")
                               for i, (_, end, fps) in enumerate(runs[:4]))
        problems.append(f"frame rate changes mid-clip: {steps}")
    if dropped > max_dropped:
        problems.append(f"{dropped:.0%} of frames dropped")
    big = np.where(dt > max_gap)[0]
    if len(big):
        worst = int(big[np.argmax(dt[big])])
        problems.append(f"{len(big)} stalls, longest {dt[worst]:.2f}s at {_mmss(t[worst])}")
    if problems:
        reason = "; ".join(problems)
    else:
        rates = " and ".join(f"{fps:g}" for _, _, fps in runs)
        reason = f"steady {rates} fps, {dropped:.1%} dropped"
        if len(runs) == 1 and abs(runs[0][2] - claimed) > 0.03 * claimed:
            reason += f" (the file's metadata says {claimed:g} fps)"
    evidence = [[round(end, 1), round(end, 1)] for _, end, _ in runs[:-1]] + \
        [[round(float(t[i]), 1), round(float(t[i + 1]), 1)] for i in big]
    return CheckResult("frame_timing", not problems, dropped, threshold, reason, evidence or None)


def check_depth(sample: Sample, params: dict) -> CheckResult:
    """How much of each depth map the sensor trusts (ARKit confidence 2, high). Water, glass, glossy surfaces and
    sunlight lower it; a whole session near zero points at the capture setup, worth fixing before the next clip."""
    min_high = params.get("min_high_confidence", 0.3)
    threshold = f"high-confidence depth >= {min_high:.0%} of pixels"
    if not sample.depth_conf:
        return CheckResult("depth", None, reason=f"skipped: {sample.depth_note or 'no depth in this clip'}")
    times = [ti for ti, _ in sample.depth_conf]
    share = [float((c == 2).mean()) for _, c in sample.depth_conf]
    value = float(np.median(share))
    ok = value >= min_high
    reason = f"high-confidence depth on {value:.0%} of pixels (median of {len(share)} sampled frames)"
    if not ok:
        reason += "; check the LiDAR, the surfaces in view (water, glass, gloss) and the capture settings"
    return CheckResult("depth", ok, value, threshold, reason, None if ok else _ranges(times, [s < min_high for s in share]))


def _mmss(t: float) -> str:
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


class HandDetector:
    """MediaPipe Hand Landmarker (Apache-2.0). The model file is downloaded separately."""

    # 0.1 rather than MediaPipe's 0.5: on 10k labeled Egocentric-10K frames it finds 85% of visible hands instead of
    # 73%, and stays right 99.5% of the time it reports one (docs/hands_eval_2026-10-30.md).
    def __init__(self, model_path: str | Path, confidence: float = 0.1):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def hands_in(self, frame: np.ndarray) -> tuple[int, float]:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        scores = [h[0].score for h in result.handedness] if result.handedness else []
        return len(scores), (max(scores) if scores else 0.0)

    def close(self) -> None:
        # Close explicitly. Left to interpreter shutdown, MediaPipe's own cleanup prints a harmless but alarming
        # "Exception ignored ... TypeError" after the results are written. Closing twice is safe.
        self._landmarker.close()


def check_hands_visible(sample: Sample, params: dict, detector: HandDetector | None) -> CheckResult:
    limit = params.get("min_fraction", 0.6)
    if detector is None:
        return CheckResult("hands_visible", None, reason="skipped: no hand model (see README)")
    counts, confidences = [], []
    for f in sample.frames:
        n, conf = detector.hands_in(f)
        counts.append(n)
        if n:
            confidences.append(conf)
    frac = float(np.mean([c > 0 for c in counts])) if counts else 0.0
    conf = float(np.mean(confidences)) if confidences else 0.0
    ok = frac >= limit
    reason = f"hands visible in {frac:.0%} of sampled frames (mean confidence {conf:.2f})"
    if not ok:
        reason += f", below {limit:.0%}"
    evidence = _ranges(sample.times, [c == 0 for c in counts])
    return CheckResult("hands_visible", ok, frac, f">= {limit:.0%}", reason, evidence)


def signature(sample: Sample, n: int = 8) -> np.ndarray | None:
    """Difference hashes of n evenly spaced frames, for near-duplicate detection."""
    if not sample.frames:
        return None
    idx = np.linspace(0, len(sample.frames) - 1, num=min(n, len(sample.frames))).round().astype(int)
    hashes = []
    for i in idx:
        small = cv2.resize(_gray(sample.frames[i]), (9, 8), interpolation=cv2.INTER_AREA)
        hashes.append((small[:, 1:] > small[:, :-1]).flatten())
    return np.array(hashes)


def find_duplicates(signatures: dict[str, np.ndarray | None], max_hamming: float) -> dict[str, tuple[str, float]]:
    """Map each later clip to the earlier clip it nearly duplicates."""
    # Flat frames (all black, all white) hash to all-zero bits and would "match" each other; other checks catch them.
    names = [k for k, v in signatures.items() if v is not None and v.any()]
    dupes: dict[str, tuple[str, float]] = {}
    for j, b in enumerate(names):
        for a in names[:j]:
            if a in dupes:
                continue
            sa, sb = signatures[a], signatures[b]
            k = min(len(sa), len(sb))
            dist = float(np.mean(np.sum(sa[:k] != sb[:k], axis=1)))
            if dist <= max_hamming:
                dupes[b] = (a, dist)
                break
    return dupes


PER_CLIP = {
    "duration": check_duration,
    "fps": check_fps,
    "resolution": check_resolution,
    "bitrate": check_bitrate,
    "black_frames": check_black_frames,
    "frozen": check_frozen,
    "blur": check_blur,
    "exposure": check_exposure,
    "shake": check_shake,
    "streams": check_streams,
    "frame_timing": check_frame_timing,
    "depth": check_depth,
}
