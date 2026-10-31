"""Fixtures: a folder of synthetic clips with known defects, and a spec that checks them."""

from __future__ import annotations

from pathlib import Path

import pytest

from handoff.synthetic import make_demo_set


@pytest.fixture
def clips_dir(tmp_path: Path) -> Path:
    return make_demo_set(tmp_path / "clips")


@pytest.fixture
def spec_path(tmp_path: Path, clips_dir: Path) -> Path:
    p = tmp_path / "spec.yaml"
    p.write_text(
        """
version: 1
name: test-spec
sampling: {fps: 4}
checks:
  duration: {min_s: 1, max_s: 60}
  fps: {min: 20}
  resolution: {min_width: 320, min_height: 240}
  black_frames: {max_fraction: 0.05}
  frozen: {max_fraction: 0.2}
  blur: {min_sharpness: 60}
  exposure: {max_overexposed_area: 0.35, max_underexposed_area: 0.35, max_bad_frames: 0.2}
  shake: {max_jitter: 3.0}
  duplicates: {max_hamming: 6}
rules:
  default_action: flag
  reject_on: [readable]
coverage:
  metadata: metadata.csv
  field: task
  targets: {pour: 3, wipe: 2}
""",
        encoding="utf-8",
    )
    return p
