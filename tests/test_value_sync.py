"""Tests for Handoff Value's sync tools that don't need a GPU or a trained policy."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from value.data import Episode, corrupt_desync, shift_actions  # noqa: E402
from value.sync import confirm, reliable, screen, xcorr_lag  # noqa: E402


def _arm(seed: int, n: int = 300, delay: int = 3) -> Episode:
    """A smooth joint trajectory where the measured state trails the commanded action by `delay` frames."""
    rng = np.random.default_rng(seed)
    t = np.arange(n + delay)
    act = np.stack([np.sin(t / (7 + k) + rng.uniform(0, 6)) * 40 + 256 for k in range(6)], axis=1).astype(np.float32)
    obs = np.concatenate([np.repeat(act[:1], delay, axis=0), act[:-delay]])[:n]
    return Episode(seed, obs, act[:n])


def test_shift_actions_moves_both_ways_and_keeps_length():
    a = np.arange(10, dtype=np.float32)[:, None]
    assert shift_actions(a, 3)[:, 0].tolist() == [3, 4, 5, 6, 7, 8, 9, 9, 9, 9]
    assert shift_actions(a, -3)[:, 0].tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    assert shift_actions(a, 0) is a


def test_corrupt_desync_marks_the_right_share():
    eps = [_arm(i) for i in range(20)]
    bad = corrupt_desync(eps, fraction=0.25, lag=-5, seed=1)
    assert sum(e.corrupted for e in bad) == 5
    for clean, maybe in zip(eps, bad):
        assert np.array_equal(clean.obs, maybe.obs)  # observations never change, only action timing


def test_xcorr_lag_recovers_the_tracking_delay():
    assert xcorr_lag(_arm(0, delay=3).obs, _arm(0, delay=3).action, window=10) == 3


def test_screen_separates_idle_and_short_episodes():
    moving = [_arm(i) for i in range(5)]
    still = Episode(99, np.full((300, 6), 256, np.float32), np.full((300, 6), 256, np.float32))
    short = Episode(98, moving[0].obs[:10], moving[0].action[:10])
    status = screen(moving + [still, short], window=10)
    assert status[99] == "no motion" and status[98] == "too short"
    assert all(status[i] == "ok" for i in range(5))


def test_reliable_rejects_widely_spread_offsets():
    assert reliable(np.array([0, 0, 1, -1, 0, 0, 5]), min_shift=3)
    assert not reliable(np.array([0, 6, -6, 5, -5, 4, -4]), min_shift=3)


def test_confirm_agrees_only_when_both_methods_see_the_same_offset():
    eps = [_arm(i) for i in range(10)]
    bad = corrupt_desync(eps, fraction=0.2, lag=-5, seed=2)
    bad_ids = [e.index for e in bad if e.corrupted]
    clean_id = next(e.index for e in bad if not e.corrupted)
    # The model-based search reports the re-aligning shift (+5 undoes a -5 corruption).
    votes = confirm(bad, {bad_ids[0]: 5, clean_id: 5}, window=10)
    assert votes[bad_ids[0]] is True
    assert votes[clean_id] is False


def _write_dataset(root, episodes):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {"episode_index": [], "frame_index": [], "observation.state": [], "action": [], "timestamp": []}
    for e in episodes:
        for f, (o, a) in enumerate(zip(e.obs, e.action)):
            rows["episode_index"].append(e.index)
            rows["frame_index"].append(f)
            rows["observation.state"].append(o.tolist())
            rows["action"].append(a.tolist())
            rows["timestamp"].append(f / 30)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text('{"fps": 30}', encoding="utf-8")
    pq.write_table(pa.table(rows), root / "data" / "chunk-000" / "file-000.parquet")


def _raw_actions(root) -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq

    t = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet")
    ep, frame = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
    act = np.array(t.column("action").to_pylist(), np.float32)
    out = {}
    for i in np.unique(ep):
        rows = np.where(ep == i)[0]
        out[int(i)] = act[rows[np.argsort(frame[rows])]]
    return out


def test_repair_realigns_only_the_named_episodes_and_keeps_the_original(tmp_path):
    from value.repair import repair_dataset

    clean = [_arm(i) for i in range(6)]
    broken = [Episode(e.index, e.obs, shift_actions(e.action, -5)) if e.index == 2 else e for e in clean]
    src, dst = tmp_path / "src", tmp_path / "fixed"
    _write_dataset(src, broken)
    before = (src / "data" / "chunk-000" / "file-000.parquet").read_bytes()

    log = repair_dataset(src, dst, {2: 5})
    assert [r["episode"] for r in log["repaired_episodes"]] == [2]
    assert (src / "data" / "chunk-000" / "file-000.parquet").read_bytes() == before  # original untouched
    assert (dst / "meta" / "info.json").exists() and (dst / "REPAIR_LOG.json").exists()

    fixed = _raw_actions(dst)
    # Episode 2 lines up with its clean actions again, apart from the 5 padded frames at the end.
    assert np.allclose(fixed[2][:-5], clean[2].action[:-5], atol=1e-4)
    for i in (0, 1, 3, 4, 5):
        assert np.allclose(fixed[i], clean[i].action, atol=1e-4)


def test_repair_refuses_to_overwrite_the_original(tmp_path):
    from value.repair import repair_dataset

    src = tmp_path / "src"
    _write_dataset(src, [_arm(0)])
    with pytest.raises(ValueError):
        repair_dataset(src, src, {0: 1})


def test_repair_takes_offsets_from_an_rdq_report(tmp_path):
    import json

    from value.repair import _check_same_dataset, repair_dataset, shifts_from_rdq

    clean = [_arm(i) for i in range(6)]
    broken = [Episode(e.index, e.obs, shift_actions(e.action, -5)) if e.index == 2 else e for e in clean]
    src = tmp_path / "src"
    _write_dataset(src, broken)
    (src / "meta" / "info.json").write_text('{"fps": 30, "total_episodes": 6, "total_frames": 1800}', encoding="utf-8")
    report = tmp_path / "rdq.json"
    report.write_text(json.dumps({
        "tool": {"name": "rdq", "version": "0.3.4"},
        "dataset": {"episodes": 6, "frames": 1800},
        # rdq's typical lag here is 3 frames, and episode 2 sits 5 frames below it.
        "findings": [{"check": "sync.action_state_lag", "details": {"typical_lag": 3, "lags": {"2": -2}}}],
    }), encoding="utf-8")

    shifts, source = shifts_from_rdq(report)
    assert shifts == {2: 5}
    assert source["tool"] == "rdq" and source["version"] == "0.3.4" and len(source["report_sha256"]) == 64
    _check_same_dataset(src, source["dataset"])
    log = repair_dataset(src, tmp_path / "fixed", shifts, offsets_from=source)
    assert log["offsets_from"]["report_sha256"] == source["report_sha256"]
    assert np.allclose(_raw_actions(tmp_path / "fixed")[2][:-5], clean[2].action[:-5], atol=1e-4)

    with pytest.raises(ValueError):
        _check_same_dataset(src, {"episodes": 7, "frames": 1800})
    with pytest.raises(ValueError):  # same counts, but the report was written for another copy
        _check_same_dataset(src, {"path": str(tmp_path / "fixed"), "episodes": 6, "frames": 1800})


def test_scan_reads_fps_from_the_dataset_metadata(tmp_path):
    from value.scan import dataset_fps

    _write_dataset(tmp_path, [_arm(0)])
    assert dataset_fps(tmp_path) == 30.0
    assert dataset_fps(tmp_path / "missing") is None


def test_confirm_steps_aside_when_actions_are_not_joint_positions():
    eps = [Episode(i, _arm(i).obs, _arm(i).action[:, :2]) for i in range(4)]
    assert confirm(eps, {0: 4}, window=10) == {0: None}
