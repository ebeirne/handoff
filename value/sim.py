"""Evaluate a policy in the PushT simulator on a fixed set of starting positions."""

from __future__ import annotations

import gymnasium as gym
import gym_pusht  # noqa: F401  (registers the environment)
import numpy as np

from .policy import CHUNK, Policy

EXECUTE = 4  # actions executed from each predicted chunk before re-planning
MAX_STEPS = 300


def _obs(o: dict) -> np.ndarray:
    return np.concatenate([o["agent_pos"], o["environment_state"]]).astype(np.float32)


def evaluate(policy: Policy, seeds: list[int]) -> dict:
    """Max coverage of the target reached per episode (0..1), and the success rate (coverage > 95%)."""
    env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos", max_episode_steps=MAX_STEPS)
    coverage, success = [], []
    for s in seeds:
        o, _ = env.reset(seed=s)
        obs = prev = _obs(o)
        best, done, ok = 0.0, False, False
        while not done:
            for a in policy.act(obs, prev)[: min(EXECUTE, CHUNK)]:
                o, r, term, trunc, info = env.step(np.clip(a, 0, 512))
                prev, obs = obs, _obs(o)
                best = max(best, float(r))
                ok = ok or bool(info.get("is_success", False))
                done = term or trunc
                if done:
                    break
        coverage.append(best)
        success.append(ok)
    env.close()
    return {"coverage": float(np.mean(coverage)), "success": float(np.mean(success)), "per_seed": coverage}
