"""Visualize the HPO benchmark (runs/hard-interactions/hpo-bench/results/*.json).

Produces a 2x2 panel PNG: (1) patience effect, (2) optimizer × dimensionality,
(3) improvement heatmap, (4) running-best curves.

Run: uv --directory tasks/hard-interactions run python <ROOT>/tools/hpo_benchmark/viz.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESDIR = ROOT / "runs" / "hard-interactions" / "hpo-bench" / "results"
OUT = ROOT / "runs" / "hard-interactions" / "hpo-bench"
PATIENCES = [6, 12, 20, 40, 10**9]
EPS = 1e-9
CANDS = ["L1-svm", "L2-rf", "M1-gbdt", "M2-stack", "M3-mlp", "M4-xgb", "H1-bigstack", "H2-deepmlp"]
OPTS = ["random", "cmaes", "cmaes+", "smac", "tpe", "tpe+"]


def derive(warm, trials, p):
    best, since = warm, 0
    for v in trials:
        if v < best - EPS:
            best, since = v, 0
        else:
            since += 1
            if since >= p:
                break
    return best


def main():
    rows = []
    for f in sorted(RESDIR.glob("*.json")):
        d = json.loads(f.read_text())
        if "error" in d or d.get("warm_best") is None or not d.get("trial_values"):
            continue
        rows.append(d)
    R = {(r["candidate"], r["optimizer"]): r for r in rows}
    NDIMS = {r["candidate"]: r["n_dims"] for r in rows}

    def best_imp(c, o):
        r = R.get((c, o))
        if not r:
            return np.nan
        return r["warm_best"] - min(derive(r["warm_best"], r["trial_values"], p) for p in PATIENCES)

    def bucket(n):
        return "low(≤3)" if n <= 3 else ("mid(4-12)" if n <= 12 else "high(≥13)")

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # (1) patience effect — TPE+ ONLY (the landed optimizer), per-dim-bucket curves.
    # x-axis auto-extends to the shortest tpe+ cell's trial count (1..100 once run finishes).
    ax = axes[0, 0]
    tpe_rows = [r for r in rows if r["optimizer"] == "tpe+"]
    maxp = min((len(r["trial_values"]) for r in tpe_rows), default=40)
    ps = list(range(1, maxp + 1))

    def mean_imp(subset, p):
        return float(np.mean([r["warm_best"] - derive(r["warm_best"], r["trial_values"], p)
                              for r in subset])) if subset else np.nan

    ax.plot(ps, [mean_imp(tpe_rows, p) for p in ps], lw=2.6, color="crimson",
            label=f"tpe+ all ({len(tpe_rows)} cand)", zorder=5)
    for b, col in [("low(≤3)", "C0"), ("mid(4-12)", "C1"), ("high(≥13)", "C2")]:
        sub = [r for r in tpe_rows if bucket(r["n_dims"]) == b]
        if sub:
            ax.plot(ps, [mean_imp(sub, p) for p in ps], lw=1.4, color=col, alpha=.85, label=b)
    ax.axvline(6, color="grey", ls=":", alpha=.7); ax.axvline(20, color="grey", ls="--", alpha=.7)
    ymax = ax.get_ylim()[1]
    ax.annotate("old default 6", (6, ymax * 0.30), fontsize=7, color="grey", rotation=90, va="bottom")
    ax.annotate("new default 20", (20, ymax * 0.30), fontsize=7, color="grey", rotation=90, va="bottom")
    ax.set_title(f"(1) TPE+ only: patience swept 1-{maxp}: mean improvement over warm\n"
                 f"knee shifts RIGHT with dimensionality (high-dim needs more patience)", fontsize=11)
    ax.set_xlabel("patience"); ax.set_ylabel("mean improvement (higher=better)")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    # (2) optimizer × dim bucket grouped bars
    ax = axes[0, 1]
    buckets = ["low(≤3)", "mid(4-12)", "high(≥13)"]
    x = np.arange(len(buckets)); w = 0.13
    for i, o in enumerate(OPTS):
        vals = []
        for b in buckets:
            cs = [c for c in CANDS if bucket(NDIMS.get(c, 0)) == b]
            v = [best_imp(c, o) for c in cs]; v = [z for z in v if not np.isnan(z)]
            vals.append(np.mean(v) if v else 0)
        ax.bar(x + (i - 2.5) * w, vals, w, label=o)
    ax.set_xticks(x); ax.set_xticklabels(buckets)
    ax.set_title("(2) optimizer x dimensionality: mean improvement\n-> rewrite select-method (high-dim: tpe+ wins, cmaes worst)", fontsize=11)
    ax.set_ylabel("mean improvement"); ax.legend(fontsize=8, ncol=3); ax.grid(alpha=.3, axis="y")

    # (3) heatmap
    ax = axes[1, 0]
    M = np.array([[best_imp(c, o) for c in CANDS] for o in OPTS])
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(CANDS)))
    ax.set_xticklabels([f"{c}\n({NDIMS.get(c,'?')}d)" for c in CANDS], fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(OPTS))); ax.set_yticklabels(OPTS)
    for i in range(len(OPTS)):
        for j in range(len(CANDS)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                        color="white" if M[i, j] < (np.nanmax(M) * .6) else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="improvement")
    ax.set_title("(3) improvement heatmap (optimizer x candidate)", fontsize=11)

    # (4) running-best curves
    ax = axes[1, 1]
    colors = {"tpe+": "C0", "smac": "C1", "cmaes": "C2", "random": "C3"}
    for cand, style in [("M4-xgb", "-"), ("H2-deepmlp", "--")]:
        for o in ["tpe+", "smac", "cmaes", "random"]:
            r = R.get((cand, o))
            if not r:
                continue
            rb, b = [], r["warm_best"]
            for v in r["trial_values"]:
                b = min(b, v); rb.append(b)
            ax.plot(range(1, len(rb) + 1), rb, style, color=colors[o], alpha=.85,
                    label=f"{cand}/{o}")
    for p in (6, 20):
        ax.axvline(p, color="grey", ls=":", alpha=.6)
        ax.annotate(f"p={p}", (p, ax.get_ylim()[1]), fontsize=7, color="grey")
    ax.set_title("(4) running-best vs trial (solid=M4-xgb, dashed=H2-deepmlp)\ndotted=patience 6/20", fontsize=11)
    ax.set_xlabel("trial"); ax.set_ylabel("running-best (lower=better)")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(OUT / "viz.png", dpi=130)
    print("saved", OUT / "viz.png")


if __name__ == "__main__":
    main()
