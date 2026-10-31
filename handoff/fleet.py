"""Fleet log triage: which robot sessions are worth keeping for training, and why the others aren't.

Reads one session per MCAP file (the ROS 2 default log format) or one per episode of a LeRobot dataset, and checks
every stream's timing: required streams that are missing, start late or stop early, dropouts, and rates below what
the robot should log. With joint signals it also measures idle time and the lag between commanded actions and the
measured state. Each session gets keep, review or drop, with the reasons and the times they happened, and the fleet
summary counts usable hours per robot.

    handoff fleet logs/ --spec fleet.yaml --out reports/fleet
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .pages import cards, esc, page

DEFAULTS = {
    "required": [],              # streams every session must have
    "expected_rate_hz": {},      # stream -> rate it should log at
    "min_rate_fraction": 0.9,    # flag a stream logging below this share of its expected rate
    "max_gap_s": 0.5,            # flag a dropout longer than this
    "max_start_offset_s": 1.0,   # flag a stream that starts this much later than the first one
    "max_end_offset_s": 1.0,     # flag a stream that stops this much earlier than the last one
    "min_duration_s": 3.0,       # drop sessions shorter than this
    "max_untracked_fraction": 0.5,  # flag a state the dataset itself marks invalid (`<column>_valid`) this often
    "idle": None,                # {stream, max_idle_fraction}: flag sessions where the robot mostly doesn't move
    "sync": None,                # {state, action, max_offset_frames}: flag action/state lag off the fleet's median;
                                 # state_from_commands: true says the state is the command on purpose (no feedback)
}
VERDICTS = ("keep", "review", "drop")


@dataclass
class Stream:
    topic: str
    t: np.ndarray                      # seconds, sorted
    values: np.ndarray | None = None   # (n, d) numeric signal, when known
    valid: np.ndarray | None = None    # (n,) bool: False where the dataset says the value wasn't measured


@dataclass
class Session:
    name: str
    robot: str
    streams: dict[str, Stream]
    error: str | None = None                            # set when the log couldn't be read at all
    truncated: bool = False                             # the log ends abruptly; what came before was recovered
    unread: dict[str, str] = field(default_factory=dict)  # signal stream -> why its values weren't read

    @property
    def start(self) -> float:
        return min(float(s.t[0]) for s in self.streams.values() if len(s.t))

    @property
    def end(self) -> float:
        return max(float(s.t[-1]) for s in self.streams.values() if len(s.t))

    @property
    def duration(self) -> float:
        return self.end - self.start if self.streams else 0.0


@dataclass
class Issue:
    level: str          # "review" or "drop"
    check: str
    message: str
    at_s: list[float] = field(default_factory=list)


def load_fleet_spec(path: str | Path | None) -> dict:
    spec = dict(DEFAULTS)
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(data.get("fleet", {})) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown fleet settings: {', '.join(sorted(unknown))}")
        spec.update(data.get("fleet", {}))
    return spec


# --- loading ---------------------------------------------------------------------------------------------------

def _numeric(value, key: str | None) -> list[float] | None:
    """Numbers at `key` (dotted path) in a decoded message: a JSON dict or a decoded ROS 2 object."""
    if key:
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else getattr(value, part, None)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) and all(isinstance(v, (int, float, np.number)) for v in value):
        return [float(v) for v in value]
    return None


def _ros2_factory():
    try:
        from mcap_ros2.decoder import DecoderFactory
    except ImportError:  # the fleet extra without ROS 2 support: timing still works, ROS 2 values aren't read
        return None
    return DecoderFactory()


_NO_DECODER = object()


class _Collector:
    """Message times per stream, plus numeric values for the streams named in `signals`."""

    def __init__(self, signals: dict[str, str | None]):
        self.signals = signals
        self.times: dict[str, list[float]] = defaultdict(list)
        self.values: dict[str, list[tuple[float, list[float]]]] = defaultdict(list)
        self.unread: dict[str, str] = {}
        self.decoders: dict[int, object] = {}
        self.ros2 = _ros2_factory() if signals else None
        self.count = 0

    def add(self, schema, channel, message) -> None:
        topic = channel.topic
        t = message.log_time / 1e9
        self.times[topic].append(t)
        self.count += 1
        if topic not in self.signals:
            return
        obj = None
        if channel.message_encoding == "json":
            obj = json.loads(message.data)
        else:
            dec = self.decoders.get(channel.id)
            if dec is None:
                dec = (self.ros2.decoder_for(channel.message_encoding, schema) if self.ros2 and schema else None) or _NO_DECODER
                self.decoders[channel.id] = dec
                if dec is _NO_DECODER:
                    self.unread[topic] = f"{channel.message_encoding} messages can't be decoded here"
            if dec is not _NO_DECODER:
                obj = dec(message.data)
        vals = _numeric(obj, self.signals[topic]) if obj is not None else None
        if vals is not None:
            self.values[topic].append((t, vals))
        elif obj is not None and topic not in self.unread:
            self.unread[topic] = f"no numbers at field '{self.signals[topic]}'"

    def session(self, name: str, robot: str, truncated: bool = False) -> Session:
        streams = {}
        for topic, times in self.times.items():
            t = np.sort(np.asarray(times, dtype=float))
            v = None
            pairs = sorted(self.values.get(topic, []), key=lambda tv: tv[0])
            if pairs and len(pairs) == len(t) and len({len(x) for _, x in pairs}) == 1:
                v = np.asarray([x for _, x in pairs], dtype=float)
            streams[topic] = Stream(topic, t, v)
        return Session(name, robot, streams, unread=self.unread, truncated=truncated)


def load_mcap(path: str | Path, signals: dict[str, str | None] | None = None, robot: str | None = None) -> Session:
    """One session from an MCAP file. Message times come from log_time. For streams named in `signals`
    (stream -> field holding a number list, e.g. {"/joint_states": "position"}), the values are read too, from
    JSON-encoded messages or from ROS 2 messages (CDR, as `ros2 bag record` writes them)."""
    from mcap.reader import make_reader

    col = _Collector(signals or {})
    meta_robot = None
    with open(path, "rb") as f:
        reader = make_reader(f)
        for record in reader.iter_metadata():
            meta_robot = meta_robot or record.metadata.get("robot")
        f.seek(0)
        reader = make_reader(f)
        for schema, channel, message in reader.iter_messages(log_time_order=True):
            col.add(schema, channel, message)
    return col.session(Path(path).stem, robot or meta_robot or "unknown")


def salvage_mcap(path: str | Path, signals: dict[str, str | None] | None = None, robot: str | None = None) -> Session | None:
    """Read a log that ends abruptly (power loss, crash mid-recording) up to where it stops. None if nothing's there."""
    from mcap.records import Channel, Message, Metadata, Schema
    from mcap.stream_reader import StreamReader

    col = _Collector(signals or {})
    schemas, channels, meta_robot = {}, {}, None
    with open(path, "rb") as f:
        try:
            for rec in StreamReader(f).records:
                if isinstance(rec, Schema):
                    schemas[rec.id] = rec
                elif isinstance(rec, Channel):
                    channels[rec.id] = rec
                elif isinstance(rec, Message) and rec.channel_id in channels:
                    ch = channels[rec.channel_id]
                    col.add(schemas.get(ch.schema_id), ch, rec)
                elif isinstance(rec, Metadata):
                    meta_robot = meta_robot or rec.metadata.get("robot")
        except Exception:
            pass  # the file stops here; keep what came before
    if col.count == 0:
        return None
    return col.session(Path(path).stem, robot or meta_robot or "unknown", truncated=True)


def _lerobot_columns(cols: set[str]) -> dict[str, list[str]]:
    """The parquet columns that make up the state and the action. Most datasets have one `observation.state` and one
    `action` column. Some (Unitree's G1 sets) split both by body part, `observation.state.left_arm` beside
    `action.left_arm`; then the parts both sides have are joined, in the same order on each side."""
    parts = sorted(c[len("observation.state."):] for c in cols
                   if c.startswith("observation.state.") and f"action.{c[len('observation.state.'):]}" in cols)
    return {"observation.state": ["observation.state"] if "observation.state" in cols else [f"observation.state.{p}" for p in parts],
            "action": ["action"] if "action" in cols else [f"action.{p}" for p in parts]}


def _joined(table, names: list[str]) -> np.ndarray:
    arrays = [np.asarray(table.column(n).to_pylist(), dtype=float) for n in names]
    return np.concatenate([a.reshape(len(a), -1) for a in arrays], axis=1)


def _validity(table, cols: set[str], names: list[str]) -> np.ndarray | None:
    """Per-row validity of the state, from flag columns beside it. Datasets built from video (hand tracking, pose
    estimation) mark frames with no measured pose, e.g. `observation.state_valid` = 0, and fill the row with zeros."""
    flags = [f"{n}_valid" for n in names if f"{n}_valid" in cols]
    if not flags:
        return None
    return np.all(_joined(table, flags) >= 0.5, axis=1)


def load_lerobot(root: str | Path) -> list[Session]:
    """One session per episode of a LeRobot v3 dataset: the state and action streams, timed by `timestamp`, and the
    state's validity flags when the dataset has them."""
    import pyarrow.parquet as pq

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    robot = info.get("robot_type") or root.name
    rows: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for f in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        cols = set(table.column_names)
        layout = _lerobot_columns(cols)
        if not layout["observation.state"] and not layout["action"]:
            raise ValueError(f"{f.name} has no observation.state or action columns (it has: {', '.join(sorted(cols))[:300]})")
        ep = table.column("episode_index").to_numpy()
        ts = table.column("timestamp").to_numpy()
        series = {k: _joined(table, names) for k, names in layout.items() if names}
        valid = _validity(table, cols, layout["observation.state"])
        for i in np.unique(ep):
            idx = np.where(ep == i)[0]
            rows[int(i)]["t"].extend(ts[idx].tolist())
            for k, arr in series.items():
                rows[int(i)][k].extend(arr[idx].tolist())
            if valid is not None:
                rows[int(i)]["valid"].extend(valid[idx].tolist())
    sessions = []
    for i in sorted(rows):
        t = np.asarray(rows[i]["t"], dtype=float)
        order = np.argsort(t)
        streams = {}
        for k in ("observation.state", "action"):
            if rows[i].get(k):
                vals = np.asarray(rows[i][k], dtype=float)
                streams[k] = Stream(k, t[order], (vals.reshape(len(t), -1))[order])
        if rows[i].get("valid") and "observation.state" in streams:
            streams["observation.state"].valid = np.asarray(rows[i]["valid"], dtype=bool)[order]
        sessions.append(Session(f"episode_{i:06d}", robot, streams))
    return sessions


def lerobot_defaults(spec: dict) -> dict:
    """LeRobot's stream names are fixed, so the idle and lag checks can switch on without a spec."""
    spec = dict(spec)
    spec["idle"] = spec.get("idle") or {"stream": "observation.state", "max_idle_fraction": 0.5}
    spec["sync"] = spec.get("sync") or {"state": "observation.state", "action": "action", "max_offset_frames": 3}
    spec["required"] = spec["required"] or ["observation.state", "action"]
    return spec


def _load_one(f: Path, signals: dict, robot: str | None) -> Session:
    try:
        return load_mcap(f, signals, robot)
    except Exception as exc:  # a broken log is one session, not a failed run
        recovered = salvage_mcap(f, signals, robot)
        return recovered or Session(f.stem, robot or "unknown", {}, error=f"{type(exc).__name__}: {exc}"[:300])


def load_logs(path: str | Path, spec: dict, workers: int = 1) -> list[Session]:
    """A LeRobot dataset folder, or a folder of .mcap files (one session each; a subfolder name is the robot).
    With workers > 1, files are read in parallel processes; the result is the same, in the same order."""
    path = Path(path)
    if (path / "meta" / "info.json").exists():
        return load_lerobot(path)
    signals = {}
    if spec.get("idle"):
        signals[spec["idle"]["stream"]] = spec["idle"].get("field")
    if spec.get("sync"):
        signals[spec["sync"]["state"]] = spec["sync"].get("state_field")
        signals[spec["sync"]["action"]] = spec["sync"].get("action_field")
    files = sorted(path.rglob("*.mcap"))
    jobs = [(f, signals, f.parent.name if f.parent != path else None) for f in files]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_load_one, *zip(*jobs)))
    return [_load_one(*job) for job in jobs]


# --- checks ----------------------------------------------------------------------------------------------------

def _gaps(t: np.ndarray, max_gap: float) -> list[tuple[float, float]]:
    if len(t) < 2:
        return []
    d = np.diff(t)
    idx = np.where(d > max_gap)[0]
    return [(float(t[i]), float(d[i])) for i in idx]


def _xcorr_lag(state: np.ndarray, action: np.ndarray, window: int, ok: np.ndarray | None = None) -> int | None:
    """Frames by which state changes best trail action changes. Same idea as value.sync.xcorr_lag. `ok` marks the
    frame-to-frame changes to use; changes into or out of an untracked frame are jumps to a placeholder, not motion."""
    dims = min(state.shape[1], action.shape[1])
    ds, da = np.diff(state[:, :dims], axis=0), np.diff(action[:, :dims], axis=0)
    ok = np.ones(len(da), dtype=bool) if ok is None else ok
    if ok.sum() < 2 * window + 10:
        return None
    best, best_score = 0, -np.inf
    for lag in range(-window, window + 1):
        a = da[: len(da) - lag] if lag >= 0 else da[-lag:]
        s = ds[lag:] if lag >= 0 else ds[: len(ds) + lag]
        m = (ok[: len(ok) - lag] & ok[lag:]) if lag >= 0 else (ok[-lag:] & ok[: len(ok) + lag])
        if m.sum() < 10:
            continue
        a, s = a[m], s[m]
        score = float(np.mean((a * s).sum(axis=0) / (np.sqrt((a * a).sum(axis=0) * (s * s).sum(axis=0)) + 1e-9)))
        if score > best_score:
            best, best_score = lag, score
    return best


def _on_grid(src: Stream, grid: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(grid, src.t, src.values[:, k]) for k in range(src.values.shape[1])], axis=1)


def check_session(s: Session, spec: dict) -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    facts: dict = {"duration_s": round(s.duration, 3) if s.streams else 0.0, "streams": {}}
    if s.error:
        return [Issue("drop", "unreadable", f"the log couldn't be read ({s.error})")], facts
    if s.unread:
        facts["values_not_read"] = s.unread
    if s.truncated:
        issues.append(Issue("review", "truncated", f"the log ends abruptly at {s.duration:.1f}s; "
                            f"{sum(len(x.t) for x in s.streams.values())} messages before the cut were recovered"))
    rel = s.start if s.streams else 0.0

    for name in spec["required"]:
        if name not in s.streams or len(s.streams[name].t) == 0:
            issues.append(Issue("drop", "missing_stream", f"{name} was never logged"))
    if s.duration < spec["min_duration_s"]:
        issues.append(Issue("drop", "too_short", f"{s.duration:.1f}s long, under {spec['min_duration_s']}s"))

    end = s.end if s.streams else 0.0
    for name, st in s.streams.items():
        if len(st.t) < 2:
            continue
        # Rate over the stretches where the stream is actually logging, so a dropout isn't counted twice.
        steps = np.diff(st.t)
        live = steps[steps <= spec["max_gap_s"]]
        rate = len(live) / float(live.sum()) if live.sum() > 0 else 0.0
        facts["streams"][name] = {"messages": len(st.t), "rate_hz": round(rate, 2),
                                  "start_s": round(float(st.t[0]) - rel, 3), "end_s": round(float(st.t[-1]) - rel, 3)}
        gaps = _gaps(st.t, spec["max_gap_s"])
        if gaps:
            longest = max(g for _, g in gaps)
            issues.append(Issue("review", "dropout", f"{name}: {len(gaps)} gap(s), longest {longest:.2f}s",
                                [round(t0 - rel, 2) for t0, _ in gaps[:10]]))
        expected = spec["expected_rate_hz"].get(name)
        if expected and rate < expected * spec["min_rate_fraction"]:
            issues.append(Issue("review", "low_rate", f"{name}: {rate:.1f} Hz, expected {expected} Hz"))
        if float(st.t[0]) - rel > spec["max_start_offset_s"]:
            issues.append(Issue("review", "late_start", f"{name} starts {float(st.t[0]) - rel:.2f}s after the session",
                                [round(float(st.t[0]) - rel, 2)]))
        if end - float(st.t[-1]) > spec["max_end_offset_s"]:
            issues.append(Issue("review", "stops_early", f"{name} stops {end - float(st.t[-1]):.2f}s before the session ends",
                                [round(float(st.t[-1]) - rel, 2)]))

    if s.duration < spec["min_duration_s"]:
        return issues, facts  # too little to judge motion or lag; it's dropped anyway

    # A state the dataset marks invalid wasn't measured (the hand left the frame, tracking failed). Report it as such,
    # and keep those frames out of the idle and lag checks: a zero placeholder looks like a robot standing still.
    for name, st in s.streams.items():
        if st.valid is None or len(st.valid) < 2:
            continue
        bad = ~st.valid
        frac = float(bad.mean())
        edges = np.diff(np.r_[0, bad.astype(int), 0])
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        spans = [(float(st.t[a]), float(st.t[min(b, len(st.t) - 1)] - st.t[a])) for a, b in zip(starts, ends)]
        longest = max((d for _, d in spans), default=0.0)
        facts["untracked"] = {"stream": name, "fraction": round(frac, 3), "gaps": len(spans), "longest_s": round(longest, 2)}
        if frac >= 0.95:
            issues.append(Issue("drop", "untracked", f"{name} is marked invalid in {frac:.0%} of frames: almost nothing "
                                                     f"was measured"))
        elif frac > spec.get("max_untracked_fraction", 0.5):
            issues.append(Issue("review", "untracked", f"{name} not tracked in {frac:.0%} of frames (the dataset marks "
                                                       f"them invalid); {len(spans)} gaps, longest {longest:.1f}s",
                                [round(t0 - rel, 2) for t0, d in spans if d >= 1.0][:10]))

    idle = spec.get("idle")
    if idle and idle["stream"] in s.streams and s.streams[idle["stream"]].values is not None:
        st = s.streams[idle["stream"]]
        speed = np.linalg.norm(np.diff(st.values, axis=0), axis=1) / np.maximum(np.diff(st.t), 1e-6)
        if st.valid is not None:
            speed = speed[st.valid[:-1] & st.valid[1:]]  # only steps between two measured frames
        if len(speed) > 2:
            moving = float(np.percentile(speed, 90))
            frac = float(np.mean(speed < max(0.05 * moving, 1e-9))) if moving > 1e-9 else 1.0
            facts["idle_fraction"] = round(frac, 3)
            if moving <= 1e-9 or frac > 0.95:
                issues.append(Issue("drop", "no_motion", f"{idle['stream']} barely moves in this session"))
            elif frac > idle.get("max_idle_fraction", 0.5):
                issues.append(Issue("review", "mostly_idle", f"idle {frac:.0%} of the time"))

    sync = spec.get("sync")
    still = any(i.check == "no_motion" for i in issues)  # no motion, no lag to measure
    if sync and not still and sync["state"] in s.streams and sync["action"] in s.streams:
        st, ac = s.streams[sync["state"]], s.streams[sync["action"]]
        if st.values is not None and ac.values is not None and len(st.t) > 20:
            grid = st.t
            window = int(sync.get("window_frames", 15))
            state, action = _on_grid(st, grid), _on_grid(ac, grid)
            valid = st.valid  # on the state's own timestamps, which are the grid
            ok = None
            if valid is not None:
                # A change counts only between measured frames, and delta-style actions are computed from the next
                # frame too, so that one must be measured as well.
                ok = valid[:-1] & valid[1:] & np.r_[valid[2:], False]
            copied = None if sync.get("state_from_commands") else _copied_from_action(state, action, window, valid)
            if copied:
                k, frac = copied
                issues.append(Issue("review", "state_copies_action",
                                    f"{sync['state']} equals {sync['action']} from {k} frame(s) earlier in {frac:.0%} "
                                    f"of frames: it looks copied from the commands, not measured"))
            else:
                # Some joints can carry no command at all: a humanoid's legs under its own balance controller log the
                # measured position as the "action". Those dimensions sit at lag 0 and would drown out the arms.
                dims = min(state.shape[1], action.shape[1])
                rows = valid if valid is not None and valid.any() else slice(None)  # placeholder zeros match too
                close = np.isclose(state[:, :dims], action[:, :dims], rtol=1e-6, atol=1e-6)[rows]
                same = np.mean(close, axis=0) >= 0.95
                if same.any() and not same.all():
                    facts["action_equals_state_dims"] = int(same.sum())
                    state, action = state[:, :dims][:, ~same], action[:, :dims][:, ~same]
            lag = _xcorr_lag(state, action, window, ok)
            if lag is not None:
                facts["lag_frames"] = lag
                facts["frame_s"] = round(float(np.median(np.diff(grid))), 5)
    return issues, facts


def _copied_from_action(state: np.ndarray, action: np.ndarray, window: int,
                        valid: np.ndarray | None = None) -> tuple[int, float] | None:
    """(k, fraction) when the state is, value for value, the action from k frames earlier. A measured joint reading
    never matches the command exactly, so a match in nearly every frame means the state was filled in from commands.
    Untracked frames are left out: their zero placeholders match a zero action without meaning anything."""
    dims = min(state.shape[1], action.shape[1])
    s, a = state[:, :dims], action[:, :dims]
    for k in range(0, min(window, len(s) - 2) + 1):
        same = np.all(np.isclose(s[k:], a[: len(a) - k], rtol=1e-6, atol=1e-6), axis=1)
        if valid is not None:
            keep = valid[k:] & valid[: len(valid) - k]
            if keep.sum() < 20:
                continue
            same = same[keep]
        frac = float(np.mean(same))
        if frac >= 0.95:
            return k, frac
    return None


def triage(sessions: list[Session], spec: dict) -> dict:
    results = []
    for s in sessions:
        issues, facts = check_session(s, spec)
        results.append({"session": s.name, "robot": s.robot, "facts": facts, "issues": issues})

    sync = spec.get("sync")
    lags = [r["facts"]["lag_frames"] for r in results if "lag_frames" in r["facts"]]
    median_lag = int(np.median(lags)) if lags else None
    if sync and median_lag is not None:
        limit = int(sync.get("max_offset_frames", 3))
        for r in results:
            lag = r["facts"].get("lag_frames")
            if lag is not None and abs(lag - median_lag) >= limit:
                off = lag - median_lag
                r["issues"].append(Issue("review", "action_lag", f"actions {off:+d} frames ({off * r['facts']['frame_s']:+.3f}s) "
                                         f"off the fleet's usual lag; repairable"))

    for r in results:
        levels = {i.level for i in r["issues"]}
        r["verdict"] = "drop" if "drop" in levels else "review" if "review" in levels else "keep"

    hours = {v: round(sum(r["facts"]["duration_s"] for r in results if r["verdict"] == v) / 3600, 4) for v in VERDICTS}
    by_robot: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        by_robot[r["robot"]][r["verdict"]] += 1
    issue_counts = Counter(check for r in results for check in {i.check for i in r["issues"]})  # sessions, not entries
    return {
        "tool": "handoff fleet", "tool_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": {k: v for k, v in spec.items()},
        "summary": {"sessions": len(results), **{v: sum(r["verdict"] == v for r in results) for v in VERDICTS},
                    "hours": hours, "median_lag_frames": median_lag},
        "by_robot": {k: dict(v) for k, v in sorted(by_robot.items())},
        "issues": dict(issue_counts.most_common()),
        "sessions": [{**r, "issues": [vars(i) for i in r["issues"]]} for r in results],
    }


# --- outputs ---------------------------------------------------------------------------------------------------

def repair_shifts(report: dict) -> dict[int, int]:
    """Per-episode action shifts that bring each lagged LeRobot episode back to the fleet's usual lag, in
    value.repair's convention (positive: actions move earlier)."""
    median = report["summary"]["median_lag_frames"]
    shifts = {}
    for s in report["sessions"]:
        if s["session"].startswith("episode_") and any(i["check"] == "action_lag" for i in s["issues"]):
            shifts[int(s["session"].split("_")[1])] = median - s["facts"]["lag_frames"]
    return shifts


def to_html(report: dict) -> str:
    s = report["summary"]
    total_h = sum(s["hours"].values())
    body = cards([("sessions", s["sessions"]), ("keep", s["keep"]), ("review", s["review"]), ("drop", s["drop"]),
                  ("usable hours", f"{s['hours']['keep']:.2f} of {total_h:.2f}")])
    if report["issues"]:
        body += "<h2>What goes wrong most</h2><table><tr><th>Issue</th><th>Sessions</th></tr>" + "".join(
            f"<tr><td>{esc(k)}</td><td>{v}</td></tr>" for k, v in report["issues"].items()) + "</table>"
    body += "<h2>By robot</h2><table><tr><th>Robot</th><th>Keep</th><th>Review</th><th>Drop</th></tr>" + "".join(
        f"<tr><td>{esc(k)}</td><td>{v.get('keep', 0)}</td><td>{v.get('review', 0)}</td><td>{v.get('drop', 0)}</td></tr>"
        for k, v in report["by_robot"].items()) + "</table>"
    rows = []
    for r in report["sessions"]:
        reasons = "".join(f'<li class="{i["level"]}">{esc(i["message"])}'
                          + (f' <span class="muted">at {esc(", ".join(f"{t:g}s" for t in i["at_s"]))}</span>' if i["at_s"] else "")
                          + "</li>" for i in r["issues"]) or '<li class="keep">no issues</li>'
        reasons += "".join(f'<li class="muted">values not read on {esc(k)}: {esc(v)}</li>'
                           for k, v in r["facts"].get("values_not_read", {}).items())
        rows.append(f'<tr><td class="mono">{esc(r["session"])}</td><td>{esc(r["robot"])}</td>'
                    f'<td>{r["facts"]["duration_s"]:.1f}s</td><td><span class="tag {r["verdict"]}">{r["verdict"]}</span></td>'
                    f"<td><ul>{reasons}</ul></td></tr>")
    body += ("<h2>Sessions</h2><table><tr><th>Session</th><th>Robot</th><th>Length</th><th>Verdict</th><th>Why</th></tr>"
             + "".join(rows) + "</table>")
    return page("Handoff fleet triage", f"tool v{report['tool_version']} · {report['created_at']}", body)


def write_outputs(report: dict, out: str | Path) -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "fleet.json", "html": out / "fleet.html", "keep": out / "keep.txt"}
    paths["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["html"].write_text(to_html(report), encoding="utf-8")
    paths["keep"].write_text("".join(r["session"] + "\n" for r in report["sessions"] if r["verdict"] == "keep"), encoding="utf-8")
    return paths
