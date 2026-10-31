"""Fleet triage on MCAP logs and LeRobot datasets with known faults planted."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mcap")

from handoff.fleet import lerobot_defaults, load_fleet_spec, load_logs, triage  # noqa: E402

RATE = 30.0


def _joints(n: int, seed: int, delay: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A smooth 6-joint trajectory: the measured state trails the commanded action by `delay` frames, with the
    small sensor noise a real joint reading has (so it never equals the command exactly)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n + delay)
    act = np.stack([np.sin(t / (9 + k) + rng.uniform(0, 6)) for k in range(6)], axis=1)
    state = np.concatenate([np.repeat(act[:1], delay, axis=0), act[:-delay]])[:n] + rng.normal(0, 1e-3, (n, 6))
    return state, act[:n]


def _write_mcap(path, streams: dict[str, list[tuple[float, dict]]], robot: str | None = None, chunk_size: int = 1 << 20) -> None:
    from mcap.writer import Writer

    with open(path, "wb") as f:
        w = Writer(f, chunk_size=chunk_size)
        w.start()
        schema = w.register_schema(name="json", encoding="jsonschema", data=b"{}")
        chans = {topic: w.register_channel(topic=topic, message_encoding="json", schema_id=schema) for topic in streams}
        if robot:
            w.add_metadata("session", {"robot": robot})
        events = sorted(((t, topic, p) for topic, msgs in streams.items() for t, p in msgs), key=lambda e: (e[0], e[1]))
        for t, topic, payload in events:
            ns = int(t * 1e9)
            w.add_message(channel_id=chans[topic], log_time=ns, publish_time=ns, data=json.dumps(payload).encode())
        w.finish()


def _session(path, seed: int, n: int = 300, delay: int = 3, drop_camera=None, camera_start: float = 0.0,
             no_action: bool = False, frozen: bool = False, robot: str | None = None, chunk_size: int = 1 << 20) -> None:
    state, act = _joints(n, seed, delay)
    if frozen:
        state = np.repeat(state[:1], n, axis=0)
        act = np.repeat(act[:1], n, axis=0)
    t = np.arange(n) / RATE
    streams = {"/joint_states": [(float(t[i]), {"position": state[i].tolist()}) for i in range(n)]}
    if not no_action:
        streams["/action"] = [(float(t[i]), {"position": act[i].tolist()}) for i in range(n)]
    cam = [(float(x), {"frame": int(k)}) for k, x in enumerate(np.arange(camera_start, n / RATE, 1 / 15))]
    if drop_camera:
        lo, hi = drop_camera
        cam = [c for c in cam if not lo <= c[0] < hi]
    streams["/camera/front"] = cam
    _write_mcap(path, streams, robot, chunk_size)


@pytest.fixture
def spec(tmp_path):
    p = tmp_path / "fleet.yaml"
    p.write_text("""fleet:
  required: [/joint_states, /action, /camera/front]
  expected_rate_hz: {/joint_states: 30, /camera/front: 15}
  idle: {stream: /joint_states, field: position, max_idle_fraction: 0.5}
  sync: {state: /joint_states, state_field: position, action: /action, action_field: position, max_offset_frames: 3}
""", encoding="utf-8")
    return load_fleet_spec(p)


def _by_name(report):
    return {r["session"]: r for r in report["sessions"]}


def test_fleet_finds_each_planted_fault_and_keeps_clean_sessions(tmp_path, spec):
    logs = tmp_path / "logs"
    (logs / "arm_a").mkdir(parents=True)
    (logs / "arm_b").mkdir(parents=True)
    for i in range(4):
        _session(logs / "arm_a" / f"clean_{i}.mcap", seed=i)
    _session(logs / "arm_b" / "dropout.mcap", seed=10, drop_camera=(4.0, 5.5))
    _session(logs / "arm_b" / "late_camera.mcap", seed=11, camera_start=2.5)
    _session(logs / "arm_b" / "no_action.mcap", seed=12, no_action=True)
    _session(logs / "arm_b" / "frozen.mcap", seed=13, frozen=True)
    _session(logs / "arm_b" / "lagged.mcap", seed=14, delay=9)

    report = triage(load_logs(logs, spec), spec)
    r = _by_name(report)

    for i in range(4):
        assert r[f"clean_{i}"]["verdict"] == "keep", r[f"clean_{i}"]["issues"]
        assert r[f"clean_{i}"]["robot"] == "arm_a"
    drop = [i for i in r["dropout"]["issues"] if i["check"] == "dropout"]
    assert r["dropout"]["verdict"] == "review" and drop and abs(drop[0]["at_s"][0] - 3.93) < 0.1
    assert {i["check"] for i in r["late_camera"]["issues"]} == {"late_start"}
    assert r["no_action"]["verdict"] == "drop" and "missing_stream" in {i["check"] for i in r["no_action"]["issues"]}
    assert r["frozen"]["verdict"] == "drop" and "no_motion" in {i["check"] for i in r["frozen"]["issues"]}
    assert r["lagged"]["verdict"] == "review" and {i["check"] for i in r["lagged"]["issues"]} == {"action_lag"}
    assert report["summary"]["median_lag_frames"] == 3
    assert report["by_robot"]["arm_a"] == {"keep": 4}


def _lerobot(root, delays: list[int], frozen: tuple[int, ...] = (), copied: tuple[int, ...] = (), n: int = 200) -> None:
    """A LeRobot v3 dataset, one episode per entry in `delays` (frames by which the state trails the action)."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "robot_type": "so100", "total_episodes": len(delays),
                                                         "total_frames": n * len(delays)}), encoding="utf-8")
    eps, frames, ts, states, acts = [], [], [], [], []
    for e, delay in enumerate(delays):
        state, act = _joints(n, seed=e, delay=delay)
        if e in frozen:
            state, act = np.repeat(state[:1], n, axis=0), np.repeat(act[:1], n, axis=0)
        if e in copied:  # state filled in from the previous command instead of read from the joints
            state = np.concatenate([act[:1], act[:-1]])
        eps += [e] * n
        frames += list(range(n))
        ts += (np.arange(n) / RATE).tolist()
        states += state.tolist()
        acts += act.tolist()
    pq.write_table(pa.table({"episode_index": eps, "frame_index": frames, "timestamp": ts, "observation.state": states,
                             "action": acts}), root / "data" / "chunk-000" / "file-000.parquet")


def test_fleet_reads_a_lerobot_dataset(tmp_path):
    root = tmp_path / "ds"
    _lerobot(root, delays=[3, 3, 3, 3, 9], frozen=(3,))

    spec = lerobot_defaults(load_fleet_spec(None))
    r = _by_name(triage(load_logs(root, spec), spec))
    assert [r[f"episode_{i:06d}"]["verdict"] for i in range(3)] == ["keep"] * 3
    assert r["episode_000003"]["verdict"] == "drop"
    assert {i["check"] for i in r["episode_000004"]["issues"]} == {"action_lag"}
    assert r["episode_000000"]["robot"] == "so100"


def test_fleet_reads_state_and_action_split_by_body_part(tmp_path):
    """Unitree's G1 datasets store observation.state.left_arm beside action.left_arm, and so on, with no single
    observation.state column. The parts both sides have are joined, so the checks still run."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "g1"
    _lerobot(root, delays=[3, 3, 3, 3, 9])
    path = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(path)
    state, act = np.asarray(t.column("observation.state").to_pylist()), np.asarray(t.column("action").to_pylist())
    split = {k: t.column(k) for k in ("episode_index", "frame_index", "timestamp")}
    split.update({"observation.state.left_arm": state[:, :4].tolist(), "action.left_arm": act[:, :4].tolist(),
                  "observation.state.left_gripper": state[:, 4].tolist(), "action.left_gripper": act[:, 4].tolist(),
                  "observation.state.waist_state_joint": state[:, 5:].tolist()})  # no action twin: left out
    legs = np.repeat(state[:, 5:6], 6, axis=1).tolist()  # balance-controlled legs: the "action" is the measured state
    split.update({"observation.state.left_leg": legs, "action.left_leg": legs})
    pq.write_table(pa.table(split), path)

    spec = lerobot_defaults(load_fleet_spec(None))
    report = triage(load_logs(root, spec), spec)
    r = _by_name(report)
    assert [r[f"episode_{i:06d}"]["verdict"] for i in range(4)] == ["keep"] * 4
    assert {i["check"] for i in r["episode_000004"]["issues"]} == {"action_lag"}
    assert report["summary"]["median_lag_frames"] == 3  # the arms' lag, not the legs' 0
    assert r["episode_000000"]["facts"]["action_equals_state_dims"] == 6


def test_fleet_stops_on_a_dataset_without_state_or_action(tmp_path):
    from handoff.cli import main

    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "odd"
    _lerobot(root, delays=[3, 3])
    path = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(path)
    pq.write_table(t.rename_columns(["episode_index", "frame_index", "timestamp", "obs", "act"]), path)
    assert main(["fleet", str(root), "--out", str(tmp_path / "out")]) == 2  # an error, not 2 dropped episodes


def test_fleet_reads_tracked_hands_with_actions_as_per_frame_changes(tmp_path):
    """Some hand-tracking datasets have no observation.state: `observation.left_hand` beside `action.left_hand_delta`,
    the change per frame. They load, the changes are summed back into a path, and a lagged episode is still found."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "hands"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 30, "robot_type": "human"}), encoding="utf-8")
    n, cols = 200, {k: [] for k in ("episode_index", "frame_index", "timestamp", "observation.left_hand",
                                    "observation.right_hand", "observation.camera_pose", "action.left_hand_delta",
                                    "action.right_hand_delta", "action.camera_delta")}
    for e, delay in enumerate([3, 3, 3, 3, 3, 12]):
        state, act = _joints(n, seed=e, delay=delay)
        step = np.diff(act, axis=0, prepend=act[:1])
        half = state.shape[1] // 2
        cols["episode_index"] += [e] * n
        cols["frame_index"] += list(range(n))
        cols["timestamp"] += (np.arange(n) / RATE).tolist()
        cols["observation.left_hand"] += state[:, :half].tolist()
        cols["observation.right_hand"] += state[:, half:].tolist()
        cols["observation.camera_pose"] += np.zeros((n, 6)).tolist()  # no matching action part: left out
        cols["action.left_hand_delta"] += step[:, :half].tolist()
        cols["action.right_hand_delta"] += step[:, half:].tolist()
        cols["action.camera_delta"] += np.zeros((n, 6)).tolist()
    pq.write_table(pa.table(cols), root / "data" / "chunk-000" / "file-000.parquet")

    empty = pa.table({k: pa.array([], type=pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").schema.field(k).type)
                      for k in cols})
    pq.write_table(empty, root / "data" / "chunk-000" / "episode_000006.parquet")  # an episode file with no rows

    spec = lerobot_defaults(load_fleet_spec(None))
    r = _by_name(triage(load_logs(root, spec), spec))
    assert [r[f"episode_{i:06d}"]["verdict"] for i in range(5)] == ["keep"] * 5
    assert {i["check"] for i in r["episode_000005"]["issues"]} == {"action_lag"}
    assert r["episode_000006"]["verdict"] == "drop" and "has no rows" in r["episode_000006"]["issues"][0]["message"]


def test_lag_limit_widens_where_the_lag_estimate_itself_is_noisy(tmp_path):
    """Where every episode gives the same lag, 3 frames off it is flagged. Where the estimate spreads from episode to
    episode (DROID's does), ordinary spread isn't flagged; a real outlier still is."""
    root = tmp_path / "noisy"
    _lerobot(root, delays=[5, 6, 7, 8, 9, 10, 11] * 3 + [30], n=200)
    spec = lerobot_defaults(load_fleet_spec(None))
    spec["sync"] = {**spec["sync"], "window_frames": 40}  # the default search reaches 15 frames; the outlier is 30
    report = triage(load_logs(root, spec), spec)
    flagged = [s["session"] for s in report["sessions"] if any(i["check"] == "action_lag" for i in s["issues"])]
    assert flagged == ["episode_000021"]  # the 30-frame one; the 5s and 11s sit inside the spread
    assert report["summary"]["lag_limit_frames"] > 3

    root = tmp_path / "steady"
    _lerobot(root, delays=[8] * 20 + [11])
    report = triage(load_logs(root, spec), spec)
    assert report["summary"]["lag_limit_frames"] == 3
    assert [s["session"] for s in report["sessions"] if s["verdict"] == "review"] == ["episode_000020"]


def test_short_episodes_are_judged_against_the_dataset_not_a_fixed_floor(tmp_path):
    """Language Table's episodes are a couple of seconds long by design. Without a spec, "too short" means short for
    this dataset; a spec's own min_duration_s is kept as written."""
    from handoff.cli import main

    root = tmp_path / "short"
    _lerobot(root, delays=[3] * 6, n=40)  # 1.3 s episodes
    out = tmp_path / "out"
    assert main(["fleet", str(root), "--out", str(out)]) == 0
    report = json.loads((out / "fleet.json").read_text(encoding="utf-8"))
    assert not [i for s in report["sessions"] for i in s["issues"] if i["check"] == "too_short"]
    assert report["spec"]["min_duration_s"] < 1.0

    spec = tmp_path / "spec.yaml"
    spec.write_text("min_duration_s: 3\n", encoding="utf-8")
    assert main(["fleet", str(root), "--spec", str(spec), "--out", str(tmp_path / "out2")]) == 0
    report = json.loads((tmp_path / "out2" / "fleet.json").read_text(encoding="utf-8"))
    assert report["summary"]["drop"] == 6


def test_fleet_requires_only_the_streams_a_lerobot_dataset_records(tmp_path):
    """Hand-pose datasets from human video often have an action stream and no robot state. That's the format, so the
    episodes aren't dropped as missing a stream; the report says what isn't recorded. A spec's own list still holds."""
    from handoff.cli import main

    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "video"
    _lerobot(root, delays=[3, 3, 3])
    path = root / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pq.read_table(path).drop_columns(["observation.state"]), path)

    out = tmp_path / "out"
    assert main(["fleet", str(root), "--out", str(out)]) == 0
    report = json.loads((out / "fleet.json").read_text(encoding="utf-8"))
    assert report["summary"]["drop"] == 0
    assert report["spec"]["absent_streams"] == ["observation.state"]
    assert "Not recorded in this dataset: observation.state" in (out / "fleet.html").read_text(encoding="utf-8")

    t = pq.read_table(path)  # per-hand flags, as Arete's ego2lerobot writes them: [left found, right found]
    ep, frame = np.asarray(t.column("episode_index").to_pylist()), np.asarray(t.column("frame_index").to_pylist())
    hands = np.ones((len(ep), 2))
    hands[frame % 10 == 0, 1] = 0                  # the right hand drops out now and then: still measured
    hands[(ep == 2) & (frame >= 60), :] = 0        # episode 2 loses both hands for most of it
    pq.write_table(t.append_column("observation.hands.valid", pa.array(hands.tolist())), path)
    out2 = tmp_path / "out2"
    assert main(["fleet", str(root), "--out", str(out2)]) == 0
    r = {s["session"]: s for s in json.loads((out2 / "fleet.json").read_text(encoding="utf-8"))["sessions"]}
    assert r["episode_000000"]["facts"]["untracked"]["fraction"] == 0.0
    assert {i["check"] for i in r["episode_000002"]["issues"]} == {"untracked"}

    p = tmp_path / "strict.yaml"
    p.write_text("fleet:\n  required: [observation.state, action]\n", encoding="utf-8")
    assert main(["fleet", str(root), "--spec", str(p), "--out", str(tmp_path / "strict")]) == 0
    strict = json.loads((tmp_path / "strict" / "fleet.json").read_text(encoding="utf-8"))
    assert strict["summary"]["drop"] == 3


def test_fleet_flags_state_copied_from_actions(tmp_path):
    root = tmp_path / "ds"
    _lerobot(root, delays=[3, 3, 3, 3], copied=(2,))
    spec = lerobot_defaults(load_fleet_spec(None))
    r = _by_name(triage(load_logs(root, spec), spec))
    copied = [i for i in r["episode_000002"]["issues"] if i["check"] == "state_copies_action"]
    assert copied and "from 1 frame(s) earlier" in copied[0]["message"]
    assert all(r[f"episode_{i:06d}"]["verdict"] == "keep" for i in (0, 1, 3))  # measured state, with noise: not flagged

    p = tmp_path / "declared.yaml"  # retargeted demos have no feedback, so the spec can say so
    p.write_text("fleet:\n  sync: {state: observation.state, action: action, state_from_commands: true}\n", encoding="utf-8")
    spec = lerobot_defaults(load_fleet_spec(p))
    assert _by_name(triage(load_logs(root, spec), spec))["episode_000002"]["verdict"] == "keep"


def test_fleet_reports_untracked_frames_and_keeps_them_out_of_idle_and_lag(tmp_path):
    """Human-demo datasets built from video mark frames with no tracked pose in observation.state_valid and zero the
    row. Those frames are reported as untracked, not as idle, and they don't distort the lag measurement."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "hands"
    _lerobot(root, delays=[3, 3, 3, 3, 9], n=320)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(path)
    ep, frame = np.asarray(t.column("episode_index").to_pylist()), np.asarray(t.column("frame_index").to_pylist())
    state, act = np.asarray(t.column("observation.state").to_pylist()), np.asarray(t.column("action").to_pylist())
    valid = np.ones(len(ep))
    valid[frame % 40 >= 30] = 0                   # short tracking gaps everywhere: 25% of every episode
    valid[(ep == 1) & (frame >= 120)] = 0         # episode 1 loses the hand for most of the episode
    state[valid == 0], act[valid == 0] = 0.0, 0.0
    cols = {k: t.column(k) for k in ("episode_index", "frame_index", "timestamp")}
    cols.update({"observation.state": state.tolist(), "action": act.tolist(),
                 "observation.state_valid": valid.reshape(-1, 1).tolist()})
    pq.write_table(pa.table(cols), path)

    spec = lerobot_defaults(load_fleet_spec(None))
    report = triage(load_logs(root, spec), spec)
    r = _by_name(report)
    assert [r[f"episode_{i:06d}"]["verdict"] for i in (0, 2, 3)] == ["keep"] * 3  # gaps alone: no idle, no lag flag
    assert r["episode_000000"]["facts"]["untracked"]["fraction"] == 0.25
    checks = {i["check"] for i in r["episode_000001"]["issues"]}
    assert "untracked" in checks and "mostly_idle" not in checks
    assert "not tracked in 72% of frames" in next(i["message"] for i in r["episode_000001"]["issues"]
                                                  if i["check"] == "untracked")
    assert {i["check"] for i in r["episode_000004"]["issues"]} == {"action_lag"}  # the real lag is still found
    assert report["summary"]["median_lag_frames"] == 3


def test_fleet_ignores_the_jump_when_tracking_switches_hands(tmp_path):
    """Datasets that track one hand per frame (observation.state_hand: 0 left, 1 right) jump when they switch. The
    jump isn't motion: it mustn't hide the real lag or invent one."""
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root = tmp_path / "two_hands"
    _lerobot(root, delays=[3, 3, 3, 3, 9], n=320)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(path)
    frame = np.asarray(t.column("frame_index").to_pylist())
    state, act = np.asarray(t.column("observation.state").to_pylist()), np.asarray(t.column("action").to_pylist())
    hand = (frame // 25) % 2                  # switch hands every 25 frames, in every episode
    state[hand == 1] += 5.0                   # the other hand sits somewhere else
    act[hand == 1] += 5.0
    cols = {k: t.column(k) for k in ("episode_index", "frame_index", "timestamp")}
    cols.update({"observation.state": state.tolist(), "action": act.tolist(),
                 "observation.state_hand": hand.reshape(-1, 1).astype(float).tolist()})
    pq.write_table(pa.table(cols), path)

    spec = lerobot_defaults(load_fleet_spec(None))
    report = triage(load_logs(root, spec), spec)
    r = _by_name(report)
    assert [r[f"episode_{i:06d}"]["verdict"] for i in range(4)] == ["keep"] * 4
    assert {i["check"] for i in r["episode_000004"]["issues"]} == {"action_lag"}
    assert report["summary"]["median_lag_frames"] == 3
    assert r["episode_000000"]["facts"]["track_switches"] == 12


def test_repair_brings_lagged_episodes_back_in_line(tmp_path):
    from handoff.cli import main

    root, fixed = tmp_path / "ds", tmp_path / "fixed"
    _lerobot(root, delays=[3, 3, 3, 3, 9, 1])
    assert main(["fleet", str(root), "--out", str(tmp_path / "before")]) == 0
    before = json.loads((tmp_path / "before" / "fleet.json").read_text(encoding="utf-8"))
    assert before["summary"]["review"] == 1  # 9 frames is 6 off the usual 3; 1 frame is only 2 off, within limits

    assert main(["repair", str(root), "--fleet", str(tmp_path / "before" / "fleet.json"), "--out", str(fixed)]) == 0
    log = json.loads((fixed / "REPAIR_LOG.json").read_text(encoding="utf-8"))
    assert [(e["episode"], e["shift_frames"]) for e in log["repaired_episodes"]] == [(4, -6)]
    assert log["offsets_from"]["tool"] == "handoff fleet"

    assert main(["fleet", str(fixed), "--out", str(tmp_path / "after")]) == 0
    after = json.loads((tmp_path / "after" / "fleet.json").read_text(encoding="utf-8"))
    assert after["summary"]["keep"] == 6
    assert _by_name(after)["episode_000004"]["facts"]["lag_frames"] == 3


def test_repair_refuses_a_report_from_another_dataset(tmp_path):
    from handoff.cli import main

    a, b = tmp_path / "a", tmp_path / "b"
    _lerobot(a, delays=[3, 3, 3, 9])
    _lerobot(b, delays=[3, 3, 3, 9])
    assert main(["fleet", str(a), "--out", str(tmp_path / "rep")]) == 0
    assert main(["repair", str(b), "--fleet", str(tmp_path / "rep" / "fleet.json"), "--out", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x").exists()


JOINT_STATE = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def _write_ros2(path, seed: int, delay: int, n: int = 300) -> None:
    """A ROS 2 log as `ros2 bag record -s mcap` writes it: CDR-encoded sensor_msgs/JointState."""
    from mcap_ros2.writer import Writer

    state, act = _joints(n, seed, delay)
    names = [f"joint_{k}" for k in range(6)]
    with open(path, "wb") as f:
        w = Writer(f)
        schema = w.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE)
        for i in range(n):
            ns = int(i / RATE * 1e9)
            for topic, row in (("/joint_states", state[i]), ("/joint_commands", act[i])):
                msg = {"header": {"stamp": {"sec": ns // 10**9, "nanosec": ns % 10**9}, "frame_id": ""},
                       "name": names, "position": row.tolist(), "velocity": [], "effort": []}
                w.write_message(topic, schema, msg, log_time=ns, publish_time=ns)
        w.finish()


def test_fleet_reads_ros2_cdr_logs_and_survives_a_corrupt_file(tmp_path):
    pytest.importorskip("mcap_ros2")
    logs = tmp_path / "ros2"
    logs.mkdir()
    for i in range(4):
        _write_ros2(logs / f"run_{i}.mcap", seed=i, delay=3)
    _write_ros2(logs / "run_lagged.mcap", seed=9, delay=9)
    (logs / "run_corrupt.mcap").write_bytes(b"\x89MCAP0\r\n" + b"\x00" * 50)  # a truncated log

    p = tmp_path / "ros2.yaml"
    p.write_text("""fleet:
  required: [/joint_states, /joint_commands]
  idle: {stream: /joint_states, field: position}
  sync: {state: /joint_states, state_field: position, action: /joint_commands, action_field: position}
""", encoding="utf-8")
    spec = load_fleet_spec(p)
    r = _by_name(triage(load_logs(logs, spec), spec))

    assert [r[f"run_{i}"]["verdict"] for i in range(4)] == ["keep"] * 4
    assert r["run_0"]["facts"]["lag_frames"] == 3                     # values decoded from CDR
    assert {i["check"] for i in r["run_lagged"]["issues"]} == {"action_lag"}
    assert r["run_corrupt"]["verdict"] == "drop" and r["run_corrupt"]["issues"][0]["check"] == "unreadable"


def test_fleet_recovers_a_log_cut_off_mid_recording(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _session(logs / "full.mcap", seed=2, chunk_size=4096)  # many chunks, as long ROS 2 recordings have
    data = (logs / "full.mcap").read_bytes()
    (logs / "cut.mcap").write_bytes(data[: len(data) // 2])   # power lost halfway through recording
    spec = load_fleet_spec(None)
    r = _by_name(triage(load_logs(logs, spec), spec))
    assert r["full"]["verdict"] == "keep"
    cut = r["cut"]
    assert cut["verdict"] == "review" and {i["check"] for i in cut["issues"]} >= {"truncated"}
    assert 0 < cut["facts"]["duration_s"] < r["full"]["facts"]["duration_s"]


def test_fleet_parallel_run_matches_single_process(tmp_path, spec):
    logs = tmp_path / "logs"
    logs.mkdir()
    for i in range(3):
        _session(logs / f"ok_{i}.mcap", seed=i)
    _session(logs / "late.mcap", seed=7, camera_start=2.5)
    _session(logs / "lag.mcap", seed=8, delay=9)

    def verdicts(workers):
        return [(s["session"], s["verdict"], [i["check"] for i in s["issues"]])
                for s in triage(load_logs(logs, spec, workers=workers), spec)["sessions"]]

    assert verdicts(1) == verdicts(3)


def test_fleet_says_when_it_cannot_read_values(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _session(logs / "s.mcap", seed=1)
    p = tmp_path / "f.yaml"
    p.write_text("fleet:\n  idle: {stream: /joint_states, field: effort}\n", encoding="utf-8")  # wrong field name
    spec = load_fleet_spec(p)
    r = _by_name(triage(load_logs(logs, spec), spec))
    assert "no numbers at field 'effort'" in r["s"]["facts"]["values_not_read"]["/joint_states"]


def test_fleet_spec_rejects_unknown_settings(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("fleet:\n  max_gap: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_fleet_spec(p)
