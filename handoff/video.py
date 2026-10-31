"""Read clip metadata and a sampled set of frames with OpenCV, or from Record3D's .r3d (iPhone LiDAR RGB-D)."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".r3d"}
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
    # Seconds of every recorded frame, for formats that keep one per frame (.r3d); None for ordinary video.
    frame_times: np.ndarray | None = None
    claimed_fps: float = 0.0  # the rate the file's metadata states, when it differs from what was recorded
    # (time, per-pixel sensor confidence: 0 low, 1 medium, 2 high) at sample points, for clips with depth.
    depth_conf: list[tuple[float, np.ndarray]] | None = None
    depth_note: str = ""  # why depth_conf is None


def rate_runs(t: np.ndarray, claimed: float) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Stretches of steady frame rate from per-frame timestamps, as [(start_s, end_s, fps)], plus each step's usual
    spacing at that point (in frames of the claimed rate). A phone that drops from 60 to 30 fps halfway gives two
    stretches; one that records every other frame all along gives one stretch at half the claimed rate."""
    k = np.maximum(np.round(np.diff(t) * claimed), 1).astype(int)
    window = np.floor(t[:-1] - t[0]).astype(int)  # one-second windows
    usual = {int(w): int(np.bincount(k[window == w]).argmax()) for w in np.unique(window)}
    def joined(runs):
        out = [runs[0]]
        for r in runs[1:]:
            if r[2] == out[-1][2]:
                out[-1][1] = r[1]
            else:
                out.append(r)
        return out

    runs = joined([[w, w, step] for w, step in usual.items()])  # [first window, last window, step]
    while len(runs) > 1:  # a blip under two seconds belongs to the stretch beside it
        short = next((i for i, r in enumerate(runs) if r[1] - r[0] < 1), None)
        if short is None:
            break
        r = runs.pop(short)
        if short:
            runs[short - 1][1] = r[1]
        else:
            runs[0][0] = r[0]
        runs = joined(runs)
    local = np.empty(len(k))
    for a, b, step in runs:
        local[(window >= a) & (window <= b)] = step
    span = float(t[-1] - t[0])
    return [(float(a), min(float(b + 1), span), claimed / step) for a, b, step in runs], local


def _lzfse():
    """Record3D compresses depth and confidence maps with Apple's LZFSE; reading them needs `pip install pyliblzfse`."""
    try:
        import liblzfse
    except ImportError:
        return None
    return liblzfse.decompress


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


def sample_r3d(path: Path, target_fps: float, max_frames: int) -> Sample:
    """Record3D's .r3d: a ZIP of `rgbd/<i>.jpg` frames, LiDAR depth and confidence maps, and a `metadata` JSON with
    the claimed fps and a timestamp for every frame. The timestamps show frames the phone dropped, which the claimed
    fps hides; the confidence maps show depth the sensor itself didn't trust."""
    try:
        z = zipfile.ZipFile(path)
        meta = json.loads(z.read("metadata"))
    except (OSError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        return Sample(ClipInfo(path, readable=False, error=f"not a readable Record3D clip: {exc}"), [], [])
    with z:
        names = set(z.namelist())
        n = sum(1 for k in names if k.startswith("rgbd/") and k.endswith(".jpg"))
        claimed = float(meta.get("fps") or 0.0)
        stamps = np.asarray(meta.get("frameTimestamps") or [], dtype=float)
        t = stamps - stamps[0] if len(stamps) else np.arange(n) / (claimed or 1.0)
        fps = claimed
        if len(stamps) > 2 and claimed > 0:  # the rate actually recorded for most of the clip, not the metadata's
            runs, _ = rate_runs(t, claimed)
            fps = max(runs, key=lambda r: r[1] - r[0])[2]
        info = ClipInfo(path, readable=n > 0, fps=fps, frame_count=n, width=int(meta.get("w") or 0),
                        height=int(meta.get("h") or 0), duration_s=float(t[-1]) if len(t) else 0.0,
                        error="" if n else "no frames")
        dw, dh = int(meta.get("dw") or 0), int(meta.get("dh") or 0)
        has_conf = any(k.endswith(".conf") for k in names)
        decompress = _lzfse() if has_conf else None
        note = ("" if decompress else "depth confidence not read: install pyliblzfse") if has_conf else \
            "no depth confidence maps in this clip"

        def jpg(i: int) -> np.ndarray | None:
            key = f"rgbd/{i}.jpg"
            return cv2.imdecode(np.frombuffer(z.read(key), np.uint8), cv2.IMREAD_COLOR) if key in names else None

        step = max(1, round(fps / target_fps)) if fps > 0 else 1
        times, frames, bursts, conf = [], [], [], []
        for i in range(0, n, step):
            if len(frames) >= max_frames:
                break
            frame = jpg(i)
            if frame is None:
                continue
            ti = float(t[i]) if i < len(t) else i / (fps or 1.0)
            times.append(ti)
            frames.append(_downscale(frame))
            if step >= BURST:
                burst = [frame] + [jpg(i + k) for k in range(1, BURST)]
                if all(f is not None for f in burst):
                    bursts.append([_burst_gray(f) for f in burst])
            key = f"rgbd/{i}.conf"
            if decompress and key in names and dw * dh:
                try:
                    c = np.frombuffer(decompress(z.read(key)), np.uint8)
                except Exception:  # a corrupt map costs the depth check, not the clip
                    continue
                if c.size == dw * dh:
                    conf.append((ti, c.reshape(dh, dw)))
    if decompress and not conf:
        note = "depth confidence maps couldn't be decoded"
    return Sample(info, times, frames, bursts, frame_times=t if len(stamps) else None, claimed_fps=claimed,
                  depth_conf=conf or None, depth_note=note)


def sample_clip(path: Path, target_fps: float, max_frames: int) -> Sample:
    if path.suffix.lower() == ".r3d":
        return sample_r3d(path, target_fps, max_frames)
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
