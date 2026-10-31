"""Build a fleet-triage demo from real robot data: SO-100 teleoperation episodes written out as MCAP logs.

The joint states and commanded actions are real (lerobot/svla_so100_pickplace). The camera stream is timestamps
only, and the faults are planted, one kind per session on the second robot, and listed in PLANTED.json:
a camera dropout, a camera that starts late, a missing action stream, a frozen arm, actions lagging the state, a
session too short to use, and joint states logged at 20 Hz instead of 30.

    python tools/make_fleet_demo.py data/so100_pickplace data/fleet_demo            # JSON-encoded messages
    python tools/make_fleet_demo.py data/so100_pickplace data/fleet_demo_ros2 --ros2  # as `ros2 bag record` writes it
    handoff fleet data/fleet_demo --spec data/fleet_demo/fleet.yaml --out reports/fleet_demo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from mcap.writer import Writer

FAULTS = ["camera_dropout", "late_camera", "no_action", "frozen", "action_lag", "too_short", "low_rate"]
SPEC = """fleet:
  required: [/joint_states, /action, /camera/front]
  expected_rate_hz: {/joint_states: 30, /action: 30, /camera/front: 30}
  max_gap_s: 0.5
  max_start_offset_s: 1.0
  max_end_offset_s: 1.0
  min_duration_s: 3.0
  idle: {stream: /joint_states, field: position, max_idle_fraction: 0.5}
  sync: {state: /joint_states, state_field: position, action: /action, action_field: position, max_offset_frames: 3}
"""


def episodes(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    t = pq.read_table(next((root / "data").rglob("*.parquet")))
    ep, ts = t.column("episode_index").to_numpy(), t.column("timestamp").to_numpy()
    state = np.asarray(t.column("observation.state").to_pylist(), dtype=float)
    action = np.asarray(t.column("action").to_pylist(), dtype=float)
    out = {}
    for i in np.unique(ep):
        idx = np.where(ep == i)[0]
        idx = idx[np.argsort(ts[idx])]
        out[int(i)] = (ts[idx] - ts[idx][0], state[idx], action[idx])
    return out


def write(path: Path, robot: str, t, state, action, camera_t, write_action: bool = True) -> None:
    with path.open("wb") as f:
        w = Writer(f)
        w.start()
        schema = w.register_schema(name="json", encoding="jsonschema", data=b"{}")
        js = w.register_channel(topic="/joint_states", message_encoding="json", schema_id=schema)
        ac = w.register_channel(topic="/action", message_encoding="json", schema_id=schema)
        cam = w.register_channel(topic="/camera/front", message_encoding="json", schema_id=schema)
        w.add_metadata("session", {"robot": robot})
        events = [(x, js, {"position": s.tolist()}) for x, s in zip(t, state)]
        if write_action:
            events += [(x, ac, {"position": a.tolist()}) for x, a in zip(t, action)]
        events += [(x, cam, {"frame": k}) for k, x in enumerate(camera_t)]
        for x, ch, payload in sorted(events, key=lambda e: (e[0], e[1])):
            ns = int(round(x * 1e9))
            w.add_message(channel_id=ch, log_time=ns, publish_time=ns, data=json.dumps(payload).encode())
        w.finish()


HEADER = """================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""
JOINT_STATE = "std_msgs/Header header\nstring[] name\nfloat64[] position\nfloat64[] velocity\nfloat64[] effort\n" + HEADER
COMPRESSED_IMAGE = "std_msgs/Header header\nstring format\nuint8[] data\n" + HEADER
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def write_ros2(path: Path, robot: str, t, state, action, camera_t, write_action: bool = True) -> None:
    """The same session as a ROS 2 bag: CDR sensor_msgs/JointState and sensor_msgs/CompressedImage."""
    from mcap_ros2.writer import Writer as Ros2Writer

    def header(x):
        ns = int(round(x * 1e9))
        return {"stamp": {"sec": ns // 10**9, "nanosec": ns % 10**9}, "frame_id": robot}

    with path.open("wb") as f:
        w = Ros2Writer(f)
        js = w.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE)
        img = w.register_msgdef("sensor_msgs/msg/CompressedImage", COMPRESSED_IMAGE)
        events = [(x, "/joint_states", js, {"header": header(x), "name": NAMES, "position": s.tolist(), "velocity": [], "effort": []})
                  for x, s in zip(t, state)]
        if write_action:
            events += [(x, "/action", js, {"header": header(x), "name": NAMES, "position": a.tolist(), "velocity": [], "effort": []})
                       for x, a in zip(t, action)]
        events += [(x, "/camera/front", img, {"header": header(x), "format": "jpeg", "data": []}) for x in camera_t]
        for x, topic, schema, msg in sorted(events, key=lambda e: (e[0], e[1])):
            ns = int(round(x * 1e9))
            w.write_message(topic, schema, msg, log_time=ns, publish_time=ns)
        w.finish()


def main(src: str, dst: str, ros2: bool = False) -> None:
    src, dst = Path(src), Path(dst)
    writer = write_ros2 if ros2 else write
    (dst / "arm_a").mkdir(parents=True, exist_ok=True)
    (dst / "arm_b").mkdir(parents=True, exist_ok=True)
    eps = episodes(src)
    planted = {}
    for n, i in enumerate(sorted(eps)[:20]):
        t, state, action = eps[i]
        robot = "arm_a" if n < 13 else "arm_b"
        fault = FAULTS[n - 13] if n >= 13 else None
        cam_t = t.copy()
        write_action = True
        if fault == "camera_dropout":
            mid = t[len(t) // 2]
            cam_t = cam_t[(cam_t < mid) | (cam_t >= mid + 2.0)]
        elif fault == "late_camera":
            cam_t = cam_t[cam_t >= 3.0]
        elif fault == "no_action":
            write_action = False
        elif fault == "frozen":
            state = np.repeat(state[:1], len(state), axis=0)
            action = np.repeat(action[:1], len(action), axis=0)
        elif fault == "action_lag":
            action = np.concatenate([np.repeat(action[:1], 8, axis=0), action[:-8]])  # actions logged 8 frames late
        elif fault == "too_short":
            keep = t < 2.0
            t, state, action, cam_t = t[keep], state[keep], action[keep], cam_t[cam_t < 2.0]
        elif fault == "low_rate":
            keep = np.arange(len(t)) % 3 != 2  # joint states and actions at 20 Hz
            t, state, action = t[keep], state[keep], action[keep]
        name = f"session_{i:03d}"
        writer(dst / robot / f"{name}.mcap", robot, t, state, action, cam_t, write_action)
        if fault:
            planted[name] = fault
    (dst / "PLANTED.json").write_text(json.dumps(planted, indent=2), encoding="utf-8")
    (dst / "fleet.yaml").write_text(SPEC, encoding="utf-8")
    (dst / "SOURCE.md").write_text("Joint states and actions: lerobot/svla_so100_pickplace (Apache-2.0), episodes 0-19. "
                                   "Camera stream: timestamps only. Faults: planted, see PLANTED.json.\n", encoding="utf-8")
    print(f"wrote 20 sessions to {dst} ({len(planted)} with planted faults)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], ros2="--ros2" in sys.argv[3:])
