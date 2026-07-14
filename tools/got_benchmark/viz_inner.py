"""Visualize the equal-budget two-level-toy tuning.

Panels: per-param effect plots (param value -> final@budget, mean over toys + per-toy)
at the tuned base, plus a budget-scan panel (final vs max_evals, tuned vs default).
Output: dev_plan/got-equalbudget-viz.png

Run: uv --directory tasks/hard-interactions run python <ROOT>/tools/got_benchmark/viz_inner.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "got_benchmark"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import bench_inner as bi  # noqa: E402
from got_select import DEFAULT_CFG  # noqa: E402

OUT = ROOT / "dev_plan" / "got-equalbudget-viz.png"
SEEDS = tuple(range(8))
MAX_EVALS = 200
TOYS = list(bi.it.TOY_SPECS)
INNER_KEYS = {"K", "patience", "top_percentile", "n_trials"}

# base = the LANDED config (DEFAULT_CFG already = n_seed=5,S=10,B=2,c_pucb=0.8,…);
# each panel varies one param with the others held at the landed config.
TBASE_CFG = dict(DEFAULT_CFG)
TBASE_INNER = dict(K=5, patience=12, top_percentile=80, n_trials=40)

EFFECTS = [  # (param, values) — all 11 swept meta-params
    ("B", [1, 2, 3, 4]), ("C", [1.0, 1.5, 2.5]), ("alpha", [0.3, 0.5, 0.7]),
    ("c_pucb", [0.1, 0.2, 0.4, 0.8, 1.5]), ("c_leaf", [0.1, 0.2, 0.4, 0.8, 1.5]),
    ("tau", [0.1, 0.3, 0.6]), ("n_seed", [2, 3, 5, 8]), ("S", [5, 10, 20, 40]),
    ("gamma", [0.3, 0.6, 0.8, 0.95]), ("K", [2, 3, 5, 8]), ("patience", [6, 12, 20, 40]),
    ("top_percentile", [70, 80, 90]),
]


def eval_param(param, val):
    cfg = dict(TBASE_CFG); inner = dict(TBASE_INNER)
    (inner if param in INNER_KEYS else cfg)[param] = val
    pt = bi.run_cfg_inner(cfg, max_evals=MAX_EVALS, seeds=SEEDS, **inner)
    return {t: pt[t]["final"] for t in TOYS}


def main():
    fig, axes = plt.subplots(5, 3, figsize=(16, 20))
    axes = axes.flatten()
    colors = {"base": "C0", "manydir": "C1", "rugged": "C2"}
    for ax, (param, vals) in zip(axes, EFFECTS):
        per = {t: [] for t in TOYS}; meanv = []
        for v in vals:
            r = eval_param(param, v)
            for t in TOYS:
                per[t].append(r[t])
            meanv.append(mean(r[t] for t in TOYS))
        xs = [str(v) for v in vals]
        for t in TOYS:
            ax.plot(xs, per[t], marker=".", lw=1, alpha=.6, color=colors[t], label=t)
        ax.plot(xs, meanv, marker="o", lw=2.4, color="crimson", label="mean", zorder=5)
        bestv = vals[meanv.index(min(meanv))]
        ax.set_title(f"{param}  (best@{MAX_EVALS}: {bestv})", fontsize=11)
        ax.set_xlabel(param); ax.set_ylabel(f"final@{MAX_EVALS} (lower=better)")
        ax.grid(alpha=.3); ax.legend(fontsize=7)

    # budget scan: tuned (S=10 landed) vs default, final vs max_evals
    ax = axes[len(EFFECTS)]
    budgets = [50, 100, 200, 400, 800]
    tuned = dict(TBASE_CFG); tuned["S"] = 10  # landed S
    for name, cfg, inner, col in [
        ("tuned(S=10)", tuned, TBASE_INNER, "crimson"),
        ("default", dict(DEFAULT_CFG), dict(K=5, patience=20, top_percentile=80, n_trials=40), "grey"),
    ]:
        ys = []
        for me in budgets:
            pt = bi.run_cfg_inner(cfg, max_evals=me, seeds=SEEDS, **inner)
            ys.append(mean(pt[t]["final"] for t in TOYS))
        ax.plot([str(b) for b in budgets], ys, marker="o", lw=2.2, color=col, label=name)
    ax.set_title("budget scan: final vs max_evals (3-toy mean)", fontsize=11)
    ax.set_xlabel("max_evals"); ax.set_ylabel("final (lower=better)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    for ax in axes[len(EFFECTS) + 1:]:
        ax.axis("off")
    plt.suptitle(f"Equal-budget two-level-toy tuning (max_evals={MAX_EVALS}, 8 seeds, 3 toys)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT, dpi=120)
    print("saved", OUT)


if __name__ == "__main__":
    main()
