"""Synthetic clips with known defects: used by the tests and to demo the tool without real data."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 320, 240, 24
KINDS = ("good", "black", "white", "frozen", "shaky", "blurry")


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, (H * 3, W * 3, 3), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (3, 3), 0)


def write_clip(path: Path, kind: str, seconds: float = 3.0, seed: int = 0) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open a video writer")
    canvas = _texture(seed)
    rng = np.random.default_rng(seed + 1)
    for i in range(int(seconds * FPS)):
        if kind == "black":
            frame = np.zeros((H, W, 3), np.uint8)
        elif kind == "white":
            frame = np.full((H, W, 3), 255, np.uint8)
        elif kind == "frozen":
            frame = canvas[H:2 * H, W:2 * W].copy()
        elif kind == "shaky":
            x, y = W + int(rng.integers(-15, 16)), H + int(rng.integers(-15, 16))
            frame = canvas[y:y + H, x:x + W].copy()
        else:  # good / blurry: a smooth pan across the texture
            x = 2 * i % (W * 2)
            frame = canvas[H:2 * H, x:x + W].copy()
            if kind == "blurry":
                frame = cv2.GaussianBlur(frame, (31, 31), 0)
        writer.write(frame)
    writer.release()
    return path


def make_demo_set(folder: str | Path) -> Path:
    """One clip per defect, an exact duplicate, a broken file, and a metadata.csv for coverage."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for kind in KINDS:
        write_clip(folder / f"{kind}.avi", kind)
    (folder / "zz_copy_of_good.avi").write_bytes((folder / "good.avi").read_bytes())
    (folder / "broken.mp4").write_bytes(b"not a video")
    (folder / "metadata.csv").write_text(
        "clip,task\ngood.avi,pour\nblack.avi,pour\nwhite.avi,wipe\nfrozen.avi,wipe\nshaky.avi,pour\n",
        encoding="utf-8",
    )
    return folder
