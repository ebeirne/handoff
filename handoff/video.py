"""Read clip metadata and a sampled set of frames with OpenCV."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
WORK_WIDTH = 640  # frames are downscaled to at most this width before checks
BURST = 3  # consecutive frames kept at each sample point, for frame-to-frame motion
BURST_WIDTH = 320


@dataclass
class ClipInfo:
    path: Path
    readable: bool
    fps: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    error: str = ""


@dataclass
class Sample:
    info: ClipInfo
    times: list[float]
    frames: list[np.ndarray]  # BGR, downscaled to WORK_WIDTH
    # Runs of BURST consecutive frames (grayscale float32, BURST_WIDTH wide) starting at sample points.
    bursts: list[list[np.ndarray]] = field(default_factory=list)


def find_clips(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def _downscale(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= WORK_WIDTH:
        return frame
    scale = WORK_WIDTH / w
    return cv2.resize(frame, (WORK_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)


def _burst_gray(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (BURST_WIDTH, round(h * BURST_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sample_clip(path: Path, target_fps: float, max_frames: int) -> Sample:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return Sample(ClipInfo(path, readable=False, error="could not open file"), [], [])

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    info = ClipInfo(
        path=path,
        readable=True,
        fps=float(fps),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    step = max(1, round(fps / target_fps)) if fps > 0 else 1

    times: list[float] = []
    frames: list[np.ndarray] = []
    bursts: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    decoded = 0
    index = 0
    while len(frames) < max_frames:
        # grab() advances without converting the frame; retrieve() only for frames we keep.
        if not cap.grab():
            break
        decoded += 1
        if index % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                times.append(index / fps if fps > 0 else float(index))
                frames.append(_downscale(frame))
                current = [_burst_gray(frame)] if step >= BURST else []
        elif current:
            ok, frame = cap.retrieve()
            current = current + [_burst_gray(frame)] if ok else []
            if len(current) == BURST:
                bursts.append(current)
                current = []
        index += 1
    cap.release()

    if decoded == 0:
        info.readable = False
        info.error = "no decodable frames"
    # Container frame counts can be wrong; trust what we decoded when we read to the end.
    if len(frames) < max_frames:
        info.frame_count = decoded
    info.duration_s = info.frame_count / fps if fps > 0 else 0.0
    return Sample(info, times, frames, bursts)
