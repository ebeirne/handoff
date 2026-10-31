"""Second opinion on flagged episodes: model-free cross-correlation, plus a picture.

For each episode, find the delay at which the observed state best tracks the commanded action (the lag that
maximizes the correlation of their frame-to-frame changes, averaged over joints). If the sync scan and this
independent estimate agree that an episode sits apart from the rest by the same amount, the flag is real evidence.
If they disagree, it's probably noise.

    .venv-value\\Scripts\\python -m value.inspect --root data/community/<dataset> --fps 30 --episodes 3 17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import load_lerobot  # noqa: E402
from .sync import xcorr_lag  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=float, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=[], help="episodes to draw next to a typical one")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    eps = load_lerobot(args.root)
    window = max(3, round(0.5 * args.fps))
    lags = {e.index: xcorr_lag(e.obs, e.action, window) for e in eps}
    median = int(np.median(list(lags.values())))
    print(f"cross-correlation: typical delay {median:+d} frames ({median / args.fps:+.3f} s) across {len(eps)} episodes")
    for i in args.episodes:
        print(f"  episode {i}: {lags[i] - median:+d} frames from typical ({(lags[i] - median) / args.fps:+.3f} s)")

    if args.episodes:
        typical = min(eps, key=lambda e: abs(lags[e.index] - median)).index
        show = [typical] + list(args.episodes)[:3]
        by_id = {e.index: e for e in eps}
        fig, axes = plt.subplots(len(show), 1, figsize=(9, 2.2 * len(show)), sharex=False)
        axes = np.atleast_1d(axes)
        for ax, i in zip(axes, show):
            e = by_id[i]
            j = int(np.argmax(np.std(e.action, axis=0)))  # the joint that moves most
            t = np.arange(len(e.obs)) / args.fps
            ax.plot(t, e.action[:, j], label="commanded (action)", lw=1.4)
            ax.plot(t, e.obs[:, j], label="measured (state)", lw=1.4)
            tag = "typical episode" if i == typical else "flagged episode"
            ax.set_title(f"episode {i}: {tag}, cross-correlation delay {lags[i] - median:+d} frames vs typical", loc="left", fontsize=10)
            ax.set_ylabel(f"joint {j}")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].legend(loc="upper right", fontsize=8, frameon=False)
        axes[-1].set_xlabel("seconds")
        fig.tight_layout()
        out = Path(args.out or f"reports/community_scan/{Path(args.root).name}_inspect.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=130)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
