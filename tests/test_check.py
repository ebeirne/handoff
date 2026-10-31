from __future__ import annotations

import json

import pytest

from handoff.cli import main
from handoff.report import fingerprint, write_outputs
from handoff.run import run
from handoff.spec import SpecError, load_spec


def _by_clip(result):
    return {v.clip: v for v in result.verdicts}


def _failed(verdict) -> set[str]:
    return {c.name for c in verdict.failed}


@pytest.fixture
def result(clips_dir, spec_path):
    return run(load_spec(spec_path), clips_dir)


def test_good_clip_passes(result):
    good = _by_clip(result)["good.avi"]
    assert good.verdict == "pass", [c.reason for c in good.failed]


@pytest.mark.parametrize(
    "clip, check",
    [
        ("black.avi", "black_frames"),
        ("white.avi", "exposure"),
        ("frozen.avi", "frozen"),
        ("blurry.avi", "blur"),
        ("shaky.avi", "shake"),
        ("zz_copy_of_good.avi", "duplicates"),
    ],
)
def test_defects_are_flagged_with_the_right_reason(result, clip, check):
    v = _by_clip(result)[clip]
    assert v.verdict == "flag"
    assert check in _failed(v)


def test_check_results_hold_plain_python_values(result):
    for v in result.verdicts:
        for c in v.checks:
            assert c.passed is None or type(c.passed) is bool, c
            assert c.value is None or type(c.value) is float, c


def test_glare_patch_does_not_fail_exposure(tmp_path):
    import numpy as np

    from handoff.checks import check_exposure
    from handoff.video import ClipInfo, Sample

    frame = np.full((240, 320, 3), 128, np.uint8)
    frame[:40, :] = 255  # bright strip covering ~17% of the frame, like glare off a bench
    sample = Sample(ClipInfo(tmp_path / "x", True), [0.0] * 10, [frame] * 10)
    assert check_exposure(sample, {}).passed is True


def test_fisheye_border_is_not_counted_as_dark(tmp_path):
    """A fisheye picture is a circle inside a black frame (some vendors' preview clips, Egocentric-10K). The surround is
    black in every frame; only the picture itself counts. A dark picture inside the circle is still caught."""
    import numpy as np

    from handoff.checks import check_exposure
    from handoff.video import ClipInfo, Sample

    yy, xx = np.mgrid[:240, :320]
    circle = (yy - 120) ** 2 + (xx - 160) ** 2 < 100 ** 2  # the picture covers ~41% of the frame
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(10):
        f = np.zeros((240, 320, 3), np.uint8)
        f[circle] = rng.integers(60, 200, (int(circle.sum()), 3))
        frames.append(f)
    sample = Sample(ClipInfo(tmp_path / "x", True), [float(i) for i in range(10)], frames)
    assert check_exposure(sample, {}).passed is True

    dark = [np.where(circle[..., None], f // 30, 0).astype(np.uint8) for f in frames]  # the scene itself is too dark
    assert check_exposure(Sample(ClipInfo(tmp_path / "x", True), [float(i) for i in range(10)], dark), {}).passed is False


def test_flat_clips_are_not_duplicates_of_each_other(result):
    assert "duplicates" not in _failed(_by_clip(result)["white.avi"])


def test_unreadable_file_is_rejected(result):
    broken = _by_clip(result)["broken.mp4"]
    assert broken.verdict == "reject"
    assert _failed(broken) == {"readable"}


def test_flag_not_reject_by_default(result):
    # Only checks listed under rules.reject_on may reject (guardrail W1).
    rejected = [v.clip for v in result.verdicts if v.verdict == "reject"]
    assert rejected == ["broken.mp4"]


def test_coverage_counts_passing_clips_against_targets(result):
    rows = {r["value"]: r for r in result.coverage["rows"]}
    assert rows["pour"]["passed"] == 1 and rows["pour"]["shortfall"] == 2
    assert rows["wipe"]["passed"] == 0 and rows["wipe"]["shortfall"] == 2


def test_fingerprint_is_reproducible(clips_dir, spec_path, result):
    again = run(load_spec(spec_path), clips_dir)
    assert fingerprint(result) == fingerprint(again)


def test_outputs_written(result, tmp_path):
    paths = write_outputs(result, tmp_path / "out")
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["summary"]["clips"] == len(result.verdicts)
    assert paths["accepted"].read_text(encoding="utf-8").split() == ["good.avi"]
    assert paths["spot_check"].read_text(encoding="utf-8").split() == ["good.avi"]
    page = paths["html"].read_text(encoding="utf-8")
    assert "Handoff Check report" in page and "Spot-check" in page and "at 0:00" in page


def test_hands_check_is_skipped_without_model(clips_dir, tmp_path):
    spec = tmp_path / "hands.yaml"
    spec.write_text("checks:\n  hands_visible: {min_fraction: 0.5}\n", encoding="utf-8")
    res = run(load_spec(spec), clips_dir)
    good = _by_clip(res)["good.avi"]
    hands = next(c for c in good.checks if c.name == "hands_visible")
    assert hands.passed is None and good.verdict == "pass"
    assert any("hands_visible did not run" in w for w in res.warnings)
    # A requested check that didn't run makes the whole run incomplete, and the fingerprint says so.
    assert not res.complete and res.not_run == ["hands_visible"]
    data = write_outputs(res, tmp_path / "incomplete")
    page = data["html"].read_text(encoding="utf-8")
    assert "Incomplete run" in page
    assert json.loads(data["json"].read_text(encoding="utf-8"))["complete"] is False


def test_cli_exits_nonzero_on_incomplete_run(clips_dir, tmp_path, capsys):
    spec = tmp_path / "hands.yaml"
    spec.write_text("checks:\n  hands_visible: {min_fraction: 0.5}\n", encoding="utf-8")
    assert main(["check", str(clips_dir), "--spec", str(spec), "--out", str(tmp_path / "o"), "--quiet"]) == 3
    assert "INCOMPLETE" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text",
    [
        "checks:\n  sparkle: {}\n",
        "rules:\n  default_action: reject\n",
        "rules:\n  reject_on: [sparkle]\n",
        "coverage:\n  field: task\n",
    ],
)
def test_bad_specs_are_refused(tmp_path, text):
    p = tmp_path / "bad.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec(p)


def test_downgrade_rule_keeps_clip_at_lower_grade(clips_dir, spec_path, tmp_path):
    text = spec_path.read_text(encoding="utf-8").replace("reject_on: [readable]", "reject_on: [readable]\n  downgrade_on: [blur]")
    spec = tmp_path / "downgrade.yaml"
    spec.write_text(text, encoding="utf-8")
    res = _by_clip(run(load_spec(spec), clips_dir))
    assert res["blurry.avi"].verdict == "downgrade"  # only blur failed
    assert res["black.avi"].verdict == "flag"  # blur failed, but so did checks that aren't downgrade-only


def test_reject_and_downgrade_cannot_overlap(tmp_path):
    p = tmp_path / "overlap.yaml"
    p.write_text("rules:\n  reject_on: [blur]\n  downgrade_on: [blur]\n", encoding="utf-8")
    with pytest.raises(SpecError):
        load_spec(p)


def test_failed_checks_say_when(result):
    black = next(c for c in _by_clip(result)["black.avi"].checks if c.name == "black_frames")
    assert black.evidence and black.evidence[0][0] == 0.0
    assert black.evidence[-1][1] >= 2.5  # the whole 3-second clip, as one stretch
    assert len(black.evidence) == 1


def test_spot_check_is_reproducible_and_sized(clips_dir, spec_path, result):
    from handoff.report import spot_check

    sc = spot_check(result, fingerprint(result))
    assert sc["needed"] == 59  # ln(0.05) / ln(0.95), rounded up
    assert sc["sample_size"] == 1 and sc["review_all"] and sc["clips"] == ["good.avi"]
    again = run(load_spec(spec_path), clips_dir)
    assert spot_check(again, fingerprint(again)) == sc


def test_parallel_run_matches_serial(clips_dir, spec_path, result):
    parallel = run(load_spec(spec_path), clips_dir, workers=2)
    assert fingerprint(parallel) == fingerprint(result)
    assert [v.clip for v in parallel.verdicts] == [v.clip for v in result.verdicts]


def test_label_sheet_and_accuracy(result, tmp_path):
    import csv

    from handoff.labels import accuracy, write_label_sheet

    paths = write_outputs(result, tmp_path / "out")
    sheet = tmp_path / "labels.csv"
    assert write_label_sheet(paths["json"], sheet) == len(result.verdicts)

    rows = list(csv.DictReader(sheet.open(encoding="utf-8")))
    human = {
        "good.avi": ("ok", ""),
        "blurry.avi": ("bad", "blur"),
        "zz_copy_of_good.avi": ("ok", ""),  # say the duplicate was fine: one false flag
        "frozen.avi": ("bad", "frozen;shake"),  # say it was also shaky: one miss for shake
    }
    for r in rows:
        r["human_verdict"], r["human_bad_checks"] = human.get(r["clip"], ("", ""))
    with sheet.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    acc = accuracy(paths["json"], sheet)
    assert acc["clips"]["labeled"] == 4 and acc["unlabeled_rows"] == len(rows) - 4
    assert acc["clips"]["true_flags"] == 2 and acc["clips"]["false_flags"] == 1 and acc["clips"]["true_passes"] == 1
    assert acc["checks"]["duplicates"]["false_flags"] == 1
    assert acc["checks"]["shake"]["missed"] == 1
    assert acc["checks"]["blur"]["true_flags"] == 1


def test_cli_end_to_end(clips_dir, spec_path, tmp_path, capsys):
    code = main(["check", str(clips_dir), "--spec", str(spec_path), "--out", str(tmp_path / "cli"), "--quiet"])
    assert code == 0
    assert "1 pass" in capsys.readouterr().out
    assert (tmp_path / "cli" / "report.html").exists()


def test_init_spec_writes_a_loadable_spec(tmp_path):
    target = tmp_path / "spec.yaml"
    assert main(["init-spec", str(target)]) == 0
    spec = load_spec(target)
    assert "hands_visible" in spec.checks


def test_bitrate_flags_a_file_too_small_for_its_frame_size(tmp_path):
    from handoff.checks import check_bitrate
    from handoff.video import ClipInfo, Sample

    def sample(mb, w=1920, h=1080, seconds=60.0):
        p = tmp_path / f"{mb}_{w}.mp4"
        p.write_bytes(b"\0" * int(mb * 1e6))
        return Sample(ClipInfo(p, True, 30.0, int(seconds * 30), w, h, seconds), [], [])

    assert check_bitrate(sample(60), {}).passed                       # 8 Mbps at 1080p: a phone's own recording
    low = check_bitrate(sample(10), {})                                # 1.3 Mbps at 1080p
    assert low.passed is False and "1.3 Mbps at 1920x1080" in low.reason and "re-encoded" in low.reason
    assert check_bitrate(sample(10, 640, 360), {}).passed             # the same bits are plenty at 360p
