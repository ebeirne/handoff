"""Find and repair action/observation desync, using only the data itself.

A sync bug leaves every frame valid, so no video or timestamp check can see it. But a policy trained on other
episodes predicts an episode's actions best at the offset where they really line up. For each held-out episode we
try every offset in a window and keep the one with the lowest prediction error.

The estimator has a built-in bias (on PushT, clean episodes come out around -2 steps), so offsets are judged against
the batch's median offset, assuming most of a delivery is aligned. An episode whose offset differs from the median by
at least `min_shift` steps is flagged and can be re-aligned by that difference instead of thrown away.

The relative rule was chosen after the first run (lag -5, seed 11). The --lag and --seed options exist to test it
on corruptions it has never seen.

    .venv-value\\Scripts\\python -m value.sync --workers 3
    .venv-value\\Scripts\\python -m value.sync --lag 4 --seed 29 --out reports/value/sync_heldout_plus4.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data import Episode, corrupt_desync, load_episodes, shift_actions
from .experiment import FINAL_TRAIN_SEEDS, TEST_SEEDS

WINDOW = 10  # offsets tried: -10..+10 steps (±1 s at 10 fps)


def _lag_job(job: tuple) -> dict[int, list[float]]:
    import torch

    from .policy import make_dataset, train

    train_eps, test_eps, seed, window = job
    model = train(train_eps, seed=seed)
    device = next(model.parameters()).device
    out = {}
    with torch.no_grad():
        for e in test_eps:
            losses = []
            for lag in range(-window, window + 1):
                x, y = make_dataset([Episode(e.index, e.obs, shift_actions(e.action, lag))])
                pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
                losses.append(float(((pred - y) ** 2).mean()))
            out[e.index] = losses
    return out


def _train_eval(job: tuple) -> tuple[float, float]:
    from .policy import train
    from .sim import evaluate

    episodes, seed = job
    r = evaluate(train(episodes, seed=seed), TEST_SEEDS)
    return r["coverage"], r["success"]


def screen(episodes: list[Episode], window: int, min_motion: float = 0.1) -> dict[int, str]:
    """Episodes a time offset can't be measured on: too short, or the arm barely moves.

    With no motion there is no delay to find, and any method returns an arbitrary offset. These episodes are a data
    problem in their own right (they carry no training signal), so they're reported separately rather than as desync.
    """
    motion = {e.index: float(np.abs(np.diff(e.action, axis=0)).sum(axis=1).mean()) if len(e.action) > 1 else 0.0
              for e in episodes}
    typical = float(np.median([m for m in motion.values() if m > 0] or [0.0]))
    status = {}
    for e in episodes:
        if len(e.obs) < 3 * window:
            status[e.index] = "too short"
        elif typical == 0 or motion[e.index] < min_motion * typical:
            status[e.index] = "no motion"
        else:
            status[e.index] = "ok"
    return status


def xcorr_lag(obs: np.ndarray, act: np.ndarray, window: int) -> int:
    """Model-free delay (frames) at which state changes best follow action changes. Positive: state trails action.

    Needs actions and states in the same joint space (same dimensions, same order).
    """
    dims = min(obs.shape[1], act.shape[1])
    do, da = np.diff(obs[:, :dims], axis=0), np.diff(act[:, :dims], axis=0)
    if len(da) < 2 * window + 10:
        return 0  # too short to measure; screen() reports these separately
    best, best_score = 0, -np.inf
    for lag in range(-window, window + 1):
        a = da[: len(da) - lag] if lag >= 0 else da[-lag:]
        o = do[lag:] if lag >= 0 else do[: len(do) + lag]
        num = (a * o).sum(axis=0)
        den = np.sqrt((a * a).sum(axis=0) * (o * o).sum(axis=0)) + 1e-9
        score = float(np.mean(num / den))
        if score > best_score:
            best, best_score = lag, score
    return best


def confirm(episodes: list[Episode], model_rel: dict[int, int], window: int, tolerance: int = 1) -> dict[int, bool | None]:
    """Second opinion on flagged episodes from the model-free cross-correlation.

    The two methods measure the offset with opposite signs (re-aligning shift vs observed delay), so they agree when
    model_rel + xcorr_rel is about 0. Returns None for every episode when actions and states aren't the same joints,
    since cross-correlation doesn't apply there.
    """
    if not episodes or episodes[0].obs.shape[1] != episodes[0].action.shape[1]:
        return {i: None for i in model_rel}
    delays = {e.index: xcorr_lag(e.obs, e.action, window) for e in episodes}
    typical = int(np.median(list(delays.values())))
    return {i: abs(r + (delays[i] - typical)) <= tolerance for i, r in model_rel.items()}


def reliable(rel: np.ndarray, min_shift: int, max_flagged_share: float = 0.3) -> bool:
    """The median rule assumes most episodes are aligned. If the spread is wide or a large share sits far from the
    median, that assumption fails for this dataset and the result is inconclusive rather than a list of flags."""
    if len(rel) == 0:
        return False
    mad = float(np.median(np.abs(rel)))
    share = float(np.mean(np.abs(rel) >= min_shift))
    return mad < min_shift / 2 and share <= max_flagged_share


def estimate_lags(pool, episodes: list[Episode], folds: int = 5, seed: int = 23, window: int = WINDOW) -> dict[int, dict]:
    perm = np.random.default_rng(seed).permutation(len(episodes))
    parts = [set(p.tolist()) for p in np.array_split(perm, folds)]
    jobs = [
        ([e for i, e in enumerate(episodes) if i not in p], [episodes[i] for i in sorted(p)], 700 + f, window)
        for f, p in enumerate(parts)
    ]
    curves: dict[int, list[float]] = {}
    for part in pool.map(_lag_job, jobs):
        curves.update(part)
    lags = list(range(-window, window + 1))
    out = {}
    for idx, losses in curves.items():
        best = int(np.argmin(losses))
        out[idx] = {
            "lag": lags[best],
            "loss_at_best": losses[best],
            "loss_at_zero": losses[window],
            "improvement": 1 - losses[best] / losses[window] if losses[window] else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--lag", type=int, default=-5, help="corruption: steps of desync (10 per second)")
    ap.add_argument("--seed", type=int, default=11, help="corruption: which episodes get it")
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--min-shift", type=int, default=3, help="flag when offset differs from the batch median by this many steps")
    ap.add_argument("--out", default="reports/value/sync_results.json")
    args = ap.parse_args()
    started = time.perf_counter()

    corrupt = {"fraction": args.fraction, "lag": args.lag, "seed": args.seed}
    episodes = corrupt_desync(load_episodes(), **corrupt)
    bad = np.array([e.corrupted for e in episodes])
    true_fix = -args.lag  # shifting by +5 undoes a -5 corruption

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        est = estimate_lags(pool, episodes)
        lag = np.array([est[e.index]["lag"] for e in episodes])
        baseline = int(np.median(lag))
        rel = lag - baseline
        flagged = np.abs(rel) >= args.min_shift

        repaired = [
            Episode(e.index, e.obs, shift_actions(e.action, int(rel[i])), e.corrupted) if flagged[i] else e
            for i, e in enumerate(episodes)
        ]
        conditions = {
            "all episodes (with bad ones)": episodes,
            "drop flagged": [e for i, e in enumerate(episodes) if not flagged[i]],
            "repair flagged (re-align)": repaired,
            "drop the actual bad ones (oracle)": [e for e in episodes if not e.corrupted],
        }
        jobs, names = [], []
        for name, eps in conditions.items():
            for s in FINAL_TRAIN_SEEDS:
                jobs.append((eps, 1000 + s))
                names.append(name)
        results: dict[str, list] = {n: [] for n in conditions}
        for name, (cov, ok) in zip(names, pool.map(_train_eval, jobs)):
            results[name].append({"coverage": cov, "success": ok})

    tp, fp = int((flagged & bad).sum()), int((flagged & ~bad).sum())
    report = {
        "corruption": corrupt,
        "flag_rule": {"relative_to_batch_median": True, "median_offset": baseline, "min_shift": args.min_shift, "window_steps": WINDOW},
        "detection": {
            "bad_episodes": int(bad.sum()),
            "flagged": int(flagged.sum()),
            "bad_flagged": tp,
            "clean_flagged": fp,
            "recall": tp / int(bad.sum()),
            "precision": tp / int(flagged.sum()) if flagged.sum() else None,
            "bad_with_exact_fix": int(((rel == true_fix) & bad).sum()),
            "bad_within_one_step": int(((np.abs(rel - true_fix) <= 1) & bad).sum()),
        },
        "estimated_lag_bad": np.bincount(lag[bad] + WINDOW, minlength=2 * WINDOW + 1).tolist(),
        "estimated_lag_clean": np.bincount(lag[~bad] + WINDOW, minlength=2 * WINDOW + 1).tolist(),
        "comparison": {
            n: {
                "episodes": len(conditions[n]),
                "coverage_mean": float(np.mean([r["coverage"] for r in rs])),
                "coverage_std": float(np.std([r["coverage"] for r in rs])),
                "success_mean": float(np.mean([r["success"] for r in rs])),
                "runs": rs,
            }
            for n, rs in results.items()
        },
        "minutes": round((time.perf_counter() - started) / 60, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    d = report["detection"]
    print(f"lag {args.lag:+d}, seed {args.seed}: median offset {baseline:+d}; flagged {d['flagged']}: "
          f"{d['bad_flagged']} of {d['bad_episodes']} bad, {d['clean_flagged']} clean; "
          f"fix exact for {d['bad_with_exact_fix']}, within 1 step for {d['bad_within_one_step']}")
    for n, c in report["comparison"].items():
        print(f"  {n:36} n={c['episodes']:3}  coverage {c['coverage_mean']:.3f} ± {c['coverage_std']:.3f}")
    print(f"wrote {args.out} in {report['minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
