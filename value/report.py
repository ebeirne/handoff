"""Turn reports/value/results.json into a chart and a markdown write-up.

    .venv-value\\Scripts\\python -m value.report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _bars(ax, comparison: dict, title: str, xmax: float | None = None) -> None:
    names = list(comparison)
    means = [comparison[n]["coverage_mean"] * 100 for n in names]
    stds = [comparison[n]["coverage_std"] * 100 for n in names]
    colors = ["#59636e"] + ["#0969da"] * (len(names) - 1)
    for i, n in enumerate(names):
        if "random" in n:
            colors[i] = "#9a6700"
        if "oracle" in n or "bottom" in n:
            colors[i] = "#8c959f"
    ax.barh(range(len(names)), means, xerr=stds, color=colors, capsize=3)
    ax.set_yticks(range(len(names)), [f"{n} ({comparison[n]['episodes']})" for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("average target coverage on 100 held-out starts (%)")
    ax.set_title(title, loc="left", fontsize=11)
    for i, (m, sd) in enumerate(zip(means, stds)):
        ax.text(m + sd + 1.5, i, f"{m:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, xmax or max(m + sd for m, sd in zip(means, stds)) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)


SYNC_RUNS = [
    ("sync_dev_minus5.json", "0.5 s stale (rule tuned here)"),
    ("sync_heldout_plus4.json", "0.4 s ahead (held-out)"),
    ("sync_heldout_minus3.json", "0.3 s stale (held-out)"),
]


def load_sync(folder: Path) -> list[tuple[str, dict]]:
    return [(label, json.loads((folder / f).read_text(encoding="utf-8"))) for f, label in SYNC_RUNS if (folder / f).exists()]


def chart(results: dict, path: Path, sync: list[tuple[str, dict]] | None = None) -> None:
    panels = [(results[k]["comparison"], t) for k, t in (("clean", "Clean demonstrations"), ("corrupted", "20% with a hidden 0.5 s sync bug: dropping low scorers")) if k in results]
    for label, s in sync or []:
        panels.append((s["comparison"], f"20% desynced, {label}: find and repair"))
    bars = sum(len(c) for c, _ in panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 0.9 * len(panels) + 0.42 * bars))
    axes = [axes] if len(panels) == 1 else axes
    xmax = 1.2 * max((r["coverage_mean"] + r["coverage_std"]) * 100 for c, _ in panels for r in c.values())
    for ax, (comparison, title) in zip(axes, panels):
        _bars(ax, comparison, title, xmax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def _sync_section(sync: list[tuple[str, dict]]) -> list[str]:
    if not sync:
        return []
    out = [
        "## Finding and repairing hidden sync bugs",
        "",
        "For each episode, a policy trained on the other episodes predicts its actions at every time offset within "
        "±1 s, and the offset with the lowest error wins. Offsets are judged against the batch's median, assuming most "
        "of a delivery is aligned. An episode at least 3 steps (0.3 s) off the median is flagged, then re-aligned "
        "by that amount rather than dropped. No timestamps and no knowledge of which episodes are bad are used.",
        "",
        "| Hidden bug in 20% of episodes | Bad found | Clean falsely flagged | Offset fixed within 0.1 s | "
        "With bugs | Drop flagged | **Repair flagged** | Drop true bad ones (oracle) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, s in sync:
        d, c = s["detection"], s["comparison"]

        def cov(name: str) -> str:
            return f"{c[name]['coverage_mean']:.1%} ± {c[name]['coverage_std']:.1%}"

        out.append(
            f"| {label} | {d['bad_flagged']} of {d['bad_episodes']} | {d['clean_flagged']} of "
            f"{206 - d['bad_episodes']} | {d['bad_within_one_step']} of {d['bad_episodes']} | "
            f"{cov('all episodes (with bad ones)')} | {cov('drop flagged')} | **{cov('repair flagged (re-align)')}** | "
            f"{cov('drop the actual bad ones (oracle)')} |"
        )
    out += [
        "",
        "The flagging rule (relative to the batch median, 3-step threshold) was chosen after the first run. The two "
        "held-out rows use different episodes and different offsets the rule never saw.",
        "",
    ]
    return out


def _table(comparison: dict) -> str:
    rows = ["| Training set | Episodes | Coverage (mean ± sd, 3 seeds) | Full successes |", "|---|---|---|---|"]
    for name, r in comparison.items():
        rows.append(
            f"| {name} | {r['episodes']} | {r['coverage_mean']:.1%} ± {r['coverage_std']:.1%} | {r['success_mean']:.1%} |"
        )
    return "\n".join(rows)


def _real_section(real: dict | None) -> list[str]:
    if not real:
        return []
    out = [
        "## Real robot data: SO-100 teleoperation",
        "",
        f"`{real['dataset']}`: a real SO-100 arm, 50 human-teleoperated pick-and-place episodes at {real['fps']} fps, "
        "using only joint states and actions. We planted offsets between the follower arm's state and the leader "
        "arm's actions in 10 of the 50 episodes. There's no simulator here, so this measures detection only. The rule "
        f"was fixed before running: flag an episode whose best offset is at least {real['min_shift_frames']} frames "
        f"({real['min_shift_frames'] / real['fps']:.1f} s) from the batch median.",
        "",
        "Flags are shown two ways. The model-based scan alone, and **confirmed**: the model-free cross-correlation "
        "of state and action changes agrees on the offset within one frame. Confirmed flags are what `value.scan` "
        "reports.",
        "",
        "| Planted offset | Bad found (scan alone) | Bad found (confirmed) | Clean flagged (scan alone) | Clean flagged (confirmed) | Offset measured within 1 frame |",
        "|---|---|---|---|---|---|",
    ]
    for c in real["cases"]:
        if c["lag_frames"] == 0:
            out.append(
                f"| none (natural data) | n/a | n/a | {c['clean_flagged']} of {c['clean_episodes']} | "
                f"{c.get('clean_confirmed', 0)} of {c['clean_episodes']} | natural offset "
                f"{c['median_offset_frames']:+d} frame ({c['median_offset_frames'] / real['fps']:+.3f} s) |"
            )
        else:
            out.append(
                f"| {c['lag_seconds']:+.2f} s ({c['lag_frames']:+d} frames) | {c['bad_flagged']} of {c['bad_episodes']} | "
                f"**{c.get('bad_confirmed', '?')} of {c['bad_episodes']}** | {c['clean_flagged']} of {c['clean_episodes']} | "
                f"**{c.get('clean_confirmed', '?')} of {c['clean_episodes']}** | {c['bad_offset_within_1_frame']} of {c['bad_episodes']} |"
            )
    out += [
        "",
        "Run it on any LeRobot dataset: `python -m value.scan --root <dataset folder> --fps 30`.",
        "",
    ]
    return out


def markdown(results: dict, chart_rel: str, sync: list[tuple[str, dict]] | None = None, real: dict | None = None) -> str:
    out = [
        "# Handoff Value, Demo 2: which training episodes make a robot policy better?",
        "",
        f"Task: LeRobot **PushT** (keypoint observations), {results['episodes']} human demonstrations. "
        f"Each score comes from {results['k']} small policies trained on random halves of the data. "
        f"Every comparison below uses {results['test_seeds']} simulator starting positions that scoring never saw.",
        "",
        f"![results]({chart_rel})",
        "",
    ]
    if "clean" in results:
        out += ["## Clean demonstrations", "", _table(results["clean"]["comparison"]), ""]
    if "corrupted" in results:
        c = results["corrupted"]
        d = c["detection"]
        out += [
            "## 20% of episodes with a hidden sync bug",
            "",
            f"{c['corruption']['episodes']} episodes had their actions recorded {abs(c['corruption']['lag_seconds'])} s "
            "stale relative to the observations. Every frame is valid, so no video check can see it.",
            "",
            "| Score | Finds bad episodes (AUC, 0.5 = chance) | Bad episodes among the lowest-scored "
            f"{d['value']['of']} |",
            "|---|---|---|",
            f"| Value (task success) | {d['value']['auc']:.2f} | {d['value']['bad_in_lowest_scored']} of {d['value']['of']} |",
            f"| Consistency (action prediction) | {d['consistency']['auc']:.2f} | "
            f"{d['consistency']['bad_in_lowest_scored']} of {d['consistency']['of']} |",
            "",
            _table(c["comparison"]),
            "",
        ]
    out += _sync_section(sync or [])
    out += _real_section(real)
    out += [
        "## Limits",
        "",
        "- Task performance was measured on one simulated task (PushT) with low-dimensional observations and a "
        "deliberately simple policy (an MLP with action chunks). Full successes are rare, so coverage is the main measure.",
        "- On the real SO-100 data we measured detection, not task performance: there's no simulator for it.",
        "- All sync bugs were planted constant offsets. Real desync can drift within an episode.",
        "- 3 training seeds per condition, so differences of a few points are within noise.",
        "",
        f"Runtime: {results['minutes']} minutes for the main experiment, about 3 minutes per sync run, on a laptop "
        "(RTX 4050, Ryzen 9 7940HS).",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="reports/value/results.json")
    ap.add_argument("--doc", default="docs/value_demo_2026-10-30.md")
    args = ap.parse_args()
    results_path = Path(args.results)
    results = json.loads(results_path.read_text(encoding="utf-8"))
    sync = load_sync(results_path.parent)
    doc = Path(args.doc)
    doc.parent.mkdir(parents=True, exist_ok=True)
    chart_path = doc.parent / "value_demo_2026-10-30.png"
    real_path = results_path.parent / "real_sync_so100.json"
    real = json.loads(real_path.read_text(encoding="utf-8")) if real_path.exists() else None
    chart(results, chart_path, sync)
    doc.write_text(markdown(results, chart_path.name, sync, real), encoding="utf-8")
    print(f"wrote {doc} and {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
