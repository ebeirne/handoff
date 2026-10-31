"""Load LeRobot PushT (keypoints) episodes, and make controlled corruptions of them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DEFAULT_ROOT = Path("data/pusht_keypoints")


@dataclass
class Episode:
    index: int
    obs: np.ndarray  # (T, 18): agent position (2) + 8 block keypoints (16), pixel coordinates 0..512
    action: np.ndarray  # (T, 2): target agent position, pixel coordinates
    corrupted: bool = False


def load_episodes(root: str | Path = DEFAULT_ROOT) -> list[Episode]:
    files = sorted(Path(root, "data").rglob("*.parquet"))
    table = pq.read_table(files, columns=["observation.state", "observation.environment_state", "action", "episode_index", "frame_index"])
    ep = table.column("episode_index").to_numpy()
    frame = table.column("frame_index").to_numpy()
    state = np.array(table.column("observation.state").to_pylist(), np.float32)
    env = np.array(table.column("observation.environment_state").to_pylist(), np.float32)
    act = np.array(table.column("action").to_pylist(), np.float32)
    obs = np.concatenate([state, env], axis=1)

    episodes = []
    for i in np.unique(ep):
        rows = np.where(ep == i)[0]
        rows = rows[np.argsort(frame[rows])]
        episodes.append(Episode(int(i), obs[rows], act[rows]))
    return episodes


def load_lerobot(root: str | Path, obs_key: str = "observation.state", act_key: str = "action") -> list[Episode]:
    """Any LeRobot v3 dataset's low-dimensional observations and actions, rescaled per dimension.

    Values are z-scored with dataset-wide statistics and mapped to 256 + 48z, the 0..512 range the policy expects
    (its fixed normalization was written for PushT's pixel coordinates).
    """
    files = sorted(Path(root, "data").rglob("*.parquet"))
    table = pq.read_table(files, columns=[obs_key, act_key, "episode_index", "frame_index"])
    ep = table.column("episode_index").to_numpy()
    frame = table.column("frame_index").to_numpy()
    obs = np.array(table.column(obs_key).to_pylist(), np.float32)
    act = np.array(table.column(act_key).to_pylist(), np.float32)
    both = np.concatenate([obs, act])  # same joints, so share one scale and keep them comparable
    mean, std = both.mean(axis=0), both.std(axis=0) + 1e-6

    def rescale(x):
        return (256 + 48 * (x - mean) / std).astype(np.float32)

    episodes = []
    for i in np.unique(ep):
        rows = np.where(ep == i)[0]
        rows = rows[np.argsort(frame[rows])]
        episodes.append(Episode(int(i), rescale(obs[rows]), rescale(act[rows])))
    return episodes


def shift_actions(action: np.ndarray, lag: int) -> np.ndarray:
    """lag > 0: the action recorded at t really belongs to t + lag. lag < 0: it belongs to t - |lag| (stale)."""
    if lag > 0:
        return np.concatenate([action[lag:], np.repeat(action[-1:], lag, axis=0)])
    if lag < 0:
        k = -lag
        return np.concatenate([np.repeat(action[:1], k, axis=0), action[:-k]])
    return action


def corrupt_desync(episodes: list[Episode], fraction: float, lag: int, seed: int) -> list[Episode]:
    """Misalign actions and observations by `lag` steps in a random `fraction` of episodes.

    A stand-in for a capture-sync bug: every frame is valid and the clip looks fine, but the action log and the
    observations are offset in time. No video check can see it.
    """
    rng = np.random.default_rng(seed)
    n = round(len(episodes) * fraction)
    bad = set(rng.choice(len(episodes), size=n, replace=False).tolist())
    out = []
    for k, e in enumerate(episodes):
        if k in bad:
            out.append(Episode(e.index, e.obs, shift_actions(e.action, lag), corrupted=True))
        else:
            out.append(Episode(e.index, e.obs, e.action))
    return out
