"""A small behavior-cloning policy: two observations in, a chunk of future actions out.

Deliberately simple and fast (seconds to train on a laptop GPU), because the experiment trains many of them.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .data import Episode

HISTORY = 2  # observations fed to the policy
CHUNK = 8  # actions predicted per call
SCALE = 256.0  # pixel coordinates 0..512 -> -1..1


def _norm(x: np.ndarray) -> np.ndarray:
    return x / SCALE - 1.0


def _denorm(x: np.ndarray) -> np.ndarray:
    return (x + 1.0) * SCALE


def make_dataset(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for e in episodes:
        T = len(e.obs)
        prev = np.concatenate([e.obs[:1], e.obs[:-1]])
        x = np.concatenate([e.obs, prev], axis=1)
        idx = np.minimum(np.arange(T)[:, None] + np.arange(CHUNK)[None, :], T - 1)  # pad with the final action
        y = e.action[idx].reshape(T, -1)
        xs.append(x)
        ys.append(y)
    return _norm(np.concatenate(xs)).astype(np.float32), _norm(np.concatenate(ys)).astype(np.float32)


class Policy(nn.Module):
    def __init__(self, obs_dim: int = 18, act_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim * HISTORY, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim * CHUNK),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def act(self, obs: np.ndarray, prev: np.ndarray) -> np.ndarray:
        """Returns the next CHUNK actions in pixel coordinates."""
        x = torch.from_numpy(_norm(np.concatenate([obs, prev]))[None].astype(np.float32))
        x = x.to(next(self.parameters()).device)
        return _denorm(self(x)[0].cpu().numpy().reshape(CHUNK, self.act_dim))


def train(episodes: list[Episode], seed: int, steps: int = 3000, batch: int = 512, device: str | None = None) -> Policy:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, y = make_dataset(episodes)
    X, Y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    model = Policy(obs_dim=episodes[0].obs.shape[1], act_dim=episodes[0].action.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        i = torch.from_numpy(rng.integers(0, len(X), batch)).to(device)
        loss = nn.functional.mse_loss(model(X[i]), Y[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    return model.eval()
