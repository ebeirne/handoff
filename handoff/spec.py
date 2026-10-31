"""Load and validate a delivery spec: the buyer's written definition of "usable"."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

KNOWN_CHECKS = {
    "readable",
    "duration",
    "fps",
    "resolution",
    "bitrate",
    "black_frames",
    "frozen",
    "blur",
    "exposure",
    "shake",
    "hands_visible",
    "duplicates",
    "streams",
    "frame_timing",
    "depth",
}


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class Spec:
    name: str
    version: int
    sampling_fps: float
    max_frames: int
    checks: dict
    reject_on: frozenset
    downgrade_on: frozenset
    coverage: dict | None
    confidence: float
    max_defect_rate: float
    sha256: str
    base_dir: Path


def load_spec(path: str | Path) -> Spec:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise SpecError("spec must be a YAML mapping")

    checks = raw.get("checks") or {}
    unknown = set(checks) - KNOWN_CHECKS
    if unknown:
        raise SpecError(f"unknown checks: {', '.join(sorted(unknown))}")
    for name, params in checks.items():
        if params is not None and not isinstance(params, dict):
            raise SpecError(f"check '{name}' must map to a set of parameters")

    rules = raw.get("rules") or {}
    # Flag by default; a hard reject needs an explicit rule the buyer wrote (guardrail W1).
    default_action = rules.get("default_action", "flag")
    if default_action != "flag":
        raise SpecError("rules.default_action must be 'flag'; list hard rejects under rules.reject_on")
    reject_on = frozenset(rules.get("reject_on") or ["readable"])
    downgrade_on = frozenset(rules.get("downgrade_on") or [])
    for key, names in (("reject_on", reject_on), ("downgrade_on", downgrade_on)):
        unknown = names - KNOWN_CHECKS
        if unknown:
            raise SpecError(f"{key} names unknown checks: {', '.join(sorted(unknown))}")
    if reject_on & downgrade_on:
        raise SpecError(f"checks can't both reject and downgrade: {', '.join(sorted(reject_on & downgrade_on))}")

    acceptance = raw.get("acceptance") or {}
    confidence = float(acceptance.get("confidence", 0.95))
    max_defect_rate = float(acceptance.get("max_defect_rate", 0.05))
    if not (0 < confidence < 1 and 0 < max_defect_rate < 1):
        raise SpecError("acceptance.confidence and acceptance.max_defect_rate must be between 0 and 1")

    sampling = raw.get("sampling") or {}
    coverage = raw.get("coverage")
    if coverage is not None:
        for key in ("metadata", "field"):
            if key not in coverage:
                raise SpecError(f"coverage.{key} is required when coverage is set")

    return Spec(
        name=str(raw.get("name", path.stem)),
        version=int(raw.get("version", 1)),
        sampling_fps=float(sampling.get("fps", 2.0)),
        max_frames=int(sampling.get("max_frames", 600)),
        checks={name: dict(params or {}) for name, params in checks.items()},
        reject_on=reject_on,
        downgrade_on=downgrade_on,
        coverage=coverage,
        confidence=confidence,
        max_defect_rate=max_defect_rate,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        base_dir=path.parent,
    )
