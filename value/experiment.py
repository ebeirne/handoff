"""Demo 2: score every training episode by its effect on task success, then test whether the scores help.

Scoring is the datamodels approach (as in DataMIL): train many small policies on random halves of the episodes,
test each in the simulator, and give each episode the gap in success between policies that saw it and policies that
didn't. Scores use one set of simulator starting positions; every comparison afterward uses a different, held-out
set, so the result can't come from tuning to the test.

Two experiments:
- clean:     the public PushT demonstrations as they are. Does training on the top-scored half beat a random half?
- corrupted: 20% of episodes get a hidden sync bug (actions recorded 0.5 s stale). Do the scores find them, and does
             dropping the lowest-scored 20% recover performance?

    .venv-value\\Scripts\\python -m value.experiment --k 64 --workers 3
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .data import corrupt_desync, load_episodes

SCORE_SEEDS = list(range(100_000, 100_050))  # used only to score episodes
TEST_SEEDS = list(range(200_000, 200_100))  # held out, used only for the final comparisons
FINAL_TRAIN_SEEDS = (0, 1, 2)

_cache: dict = {}


def _episodes(corrupt: dict | None):
    key = json.dumps(corrupt, sort_keys=True)
    if key not in _cache:
        eps = load_episodes()
        _cache[key] = corrupt_desync(eps, **corrupt) if corrupt else eps
    return _cache[key]


def _job(job: tuple) -> tuple[float, float]:
    from .policy import train
    from .sim import evaluate

    subset, corrupt, train_seed, eval_seeds = job
    eps = _episodes(corrupt)
    r = evaluate(train([eps[i] for i in subset], seed=train_seed), eval_seeds)
    return r["coverage"], r["success"]


def _loss_job(job: tuple) -> dict[int, float]:
    """Train without some episodes, then measure how well the policy predicts their actions (lower = consistent)."""
    import torch

    from .policy import make_dataset, train

    train_idx, test_idx, corrupt, seed = job
    eps = _episodes(corrupt)
    model = train([eps[i] for i in train_idx], seed=seed)
    device = next(model.parameters()).device
    losses = {}
    with torch.no_grad():
        for i in test_idx:
            x, y = make_dataset([eps[i]])
            pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
            losses[i] = float(((pred - y) ** 2).mean())
    return losses


def consistency_scores(pool, n: int, corrupt: dict | None, folds: int = 5, seed: int = 17) -> np.ndarray:
    """Cross-fitted action-prediction error per episode, negated so higher means more consistent."""
    perm = np.random.default_rng(seed).permutation(n)
    parts = np.array_split(perm, folds)
    jobs = [(sorted(set(range(n)) - set(p.tolist())), p.tolist(), corrupt, 500 + f) for f, p in enumerate(parts)]
    losses: dict[int, float] = {}
    for part in pool.map(_loss_job, jobs):
        losses.update(part)
    return -np.array([losses[i] for i in range(n)])


def datamodel_scores(pool, n: int, k: int, corrupt: dict | None, seed: int) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    subsets = [sorted(rng.choice(n, n // 2, replace=False).tolist()) for _ in range(k)]
    results = list(pool.map(_job, [(s, corrupt, i, SCORE_SEEDS) for i, s in enumerate(subsets)]))
    perf = np.array([c for c, _ in results])
    member = np.zeros((k, n), bool)
    for i, s in enumerate(subsets):
        member[i, s] = True
    scores = np.array([
        perf[member[:, j]].mean() - perf[~member[:, j]].mean() if 0 < member[:, j].sum() < k else 0.0
        for j in range(n)
    ])
    runs = [{"subset_size": len(s), "coverage": c, "success": ok} for s, (c, ok) in zip(subsets, results)]
    return scores, runs


def compare(pool, conditions: dict[str, callable], corrupt: dict | None) -> dict:
    jobs, keys = [], []
    for name, pick in conditions.items():
        for s in FINAL_TRAIN_SEEDS:
            jobs.append((pick(s), corrupt, 1000 + s, TEST_SEEDS))
            keys.append(name)
    out: dict[str, list] = {name: [] for name in conditions}
    for name, (cov, ok) in zip(keys, pool.map(_job, jobs)):
        out[name].append({"coverage": cov, "success": ok})
    return {
        name: {
            "coverage_mean": float(np.mean([r["coverage"] for r in runs])),
            "coverage_std": float(np.std([r["coverage"] for r in runs])),
            "success_mean": float(np.mean([r["success"] for r in runs])),
            "episodes": len(conditions[name](0)),
            "runs": runs,
        }
        for name, runs in out.items()
    }


def _auc(scores: np.ndarray, bad: np.ndarray) -> float:
    """Probability a random clean episode scores higher than a random corrupted one (0.5 = chance)."""
    good, worse = scores[~bad], scores[bad]
    return float((good[:, None] > worse[None, :]).mean() + 0.5 * (good[:, None] == worse[None, :]).mean())


def run_clean(pool, n: int, k: int) -> dict:
    value, runs = datamodel_scores(pool, n, k, None, seed=7)
    consistency = consistency_scores(pool, n, None)
    by_value, by_consistency = np.argsort(-value), np.argsort(-consistency)
    half = n // 2
    conditions = {
        "all episodes": lambda s: list(range(n)),
        "top half by value": lambda s: sorted(by_value[:half].tolist()),
        "top half by consistency": lambda s: sorted(by_consistency[:half].tolist()),
        "random half": lambda s: sorted(np.random.default_rng(50 + s).choice(n, half, replace=False).tolist()),
        "bottom half by value": lambda s: sorted(by_value[-half:].tolist()),
    }
    return {
        "value_scores": value.tolist(),
        "consistency_scores": consistency.tolist(),
        "scoring_runs": runs,
        "comparison": compare(pool, conditions, None),
    }


def _detection(scores: np.ndarray, bad: np.ndarray) -> dict:
    n_bad = int(bad.sum())
    return {"auc": _auc(scores, bad), "bad_in_lowest_scored": int(bad[np.argsort(scores)[:n_bad]].sum()), "of": n_bad}


def run_corrupted(pool, n: int, k: int, fraction: float, lag: int) -> dict:
    corrupt = {"fraction": fraction, "lag": lag, "seed": 11}
    bad = np.array([e.corrupted for e in _episodes(corrupt)])
    n_bad = int(bad.sum())
    value, runs = datamodel_scores(pool, n, k, corrupt, seed=13)
    consistency = consistency_scores(pool, n, corrupt)
    keep_value = sorted(np.argsort(-value)[: n - n_bad].tolist())
    keep_consistency = sorted(np.argsort(-consistency)[: n - n_bad].tolist())
    conditions = {
        "all episodes (with bad ones)": lambda s: list(range(n)),
        f"drop lowest {fraction:.0%} by value": lambda s: keep_value,
        f"drop lowest {fraction:.0%} by consistency": lambda s: keep_consistency,
        f"drop random {fraction:.0%}": lambda s: sorted(np.random.default_rng(90 + s).choice(n, n - n_bad, replace=False).tolist()),
        "drop the actual bad ones (oracle)": lambda s: sorted(np.where(~bad)[0].tolist()),
    }
    return {
        "corruption": {"type": "action/observation desync", "lag_steps": lag, "lag_seconds": lag / 10, **corrupt, "episodes": n_bad},
        "value_scores": value.tolist(),
        "consistency_scores": consistency.tolist(),
        "corrupted": bad.tolist(),
        "detection": {"value": _detection(value, bad), "consistency": _detection(consistency, bad)},
        "scoring_runs": runs,
        "comparison": compare(pool, conditions, corrupt),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64, help="policies trained on random halves, per experiment")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--lag", type=int, default=-5, help="sync offset in steps (10 per second); negative = stale actions")
    ap.add_argument("--only", choices=["clean", "corrupted"])
    ap.add_argument("--out", default="reports/value")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = len(load_episodes())
    results = {"episodes": n, "k": args.k, "score_seeds": len(SCORE_SEEDS), "test_seeds": len(TEST_SEEDS)}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.only in (None, "clean"):
            results["clean"] = run_clean(pool, n, args.k)
            print("clean done", flush=True)
        if args.only in (None, "corrupted"):
            results["corrupted"] = run_corrupted(pool, n, args.k, args.fraction, args.lag)
            print("corrupted done", flush=True)
    results["minutes"] = round((time.perf_counter() - started) / 60, 1)
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out / 'results.json'} in {results['minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
