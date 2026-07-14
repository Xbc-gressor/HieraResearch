"""Collect + visualize the tabular-blind method comparison (framework vs hillclimb
× haiku/sonnet/opus). Reads the 6 run dirs under runs/tabular-blind/cmp-*.nosync,
extracts best neg_mean_test_accuracy and the best-vs-evaluations curve for each,
and writes a markdown report + a PNG (best-vs-evals curves + a 6-cell bar chart).

Handles incomplete runs gracefully (uses whatever has been recorded so far).

Run (needs matplotlib — use a task env that has it):
  uv --directory tasks/hard-interactions run python tools/got_benchmark/collect_compare.py
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "tabular-blind"
OUT_PNG = ROOT / "dev_plan" / "tabular-blind-compare.png"
OUT_MD = ROOT / "dev_plan" / "tabular-blind-compare.md"
TIERS = ["haiku", "sonnet", "opus"]
METHODS = [("hc", "hillclimb"), ("exp", "framework")]
INF = float("inf")


def _runningmin(scores):
    """(eval_index, running_best) curve from a sequence of per-eval scores."""
    best = INF
    xs, ys = [], []
    for i, s in enumerate(scores, 1):
        if s is not None and s == s and s < best:  # not None, not NaN, improves
            best = s
        xs.append(i)
        ys.append(best)
    return xs, ys


def load_hc(d: Path):
    """hillclimb: results.tsv rows = evals; col 2 = score. Returns (evals, best, curve)."""
    f = d / "results.tsv"
    if not f.exists():
        return 0, INF, ([], [])
    scores = []
    for line in f.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            v = float(parts[1])
        except ValueError:
            continue
        scores.append(v if v < 0 or v == INF else v)  # keep as-is
    # treat 0.0/inf crash rows as non-improving (very large)
    clean = [(s if (s == s and s != INF and s != 0.0) else INF) for s in scores]
    xs, ys = _runningmin(clean)
    best = min([s for s in clean if s != INF], default=INF)
    return len(scores), best, (xs, ys)


def load_exp(d: Path):
    """framework: ledger records; evals = Σ(warm_start_K+trials_completed) cumulative;
    score = final_best_score (fallback best_warm_score). Returns (evals, best, curve)."""
    f = d / "ledger.json"
    if not f.exists():
        return 0, INF, ([], [])
    try:
        recs = json.loads(f.read_text()).get("records", [])
    except (ValueError, OSError):
        return 0, INF, ([], [])
    cum, xs, ys, best = 0, [], [], INF
    for r in recs:
        ev = (r.get("warm_start_K") or 0) + (r.get("trials_completed") or 0)
        cum += ev
        s = r.get("final_best_score")
        if s is None:
            s = r.get("best_warm_score")
        if isinstance(s, (int, float)) and s == s and s != INF and s < best:
            best = s
        xs.append(cum)
        ys.append(best)
    return cum, best, (xs, ys)


def main():
    data = {}
    for mk, mname in METHODS:
        for tier in TIERS:
            d = RUNS / f"cmp-{mk}-{tier}.nosync"
            evals, best, curve = (load_hc if mk == "hc" else load_exp)(d)
            data[(mk, tier)] = dict(evals=evals, best=best, curve=curve, name=f"{mname}-{tier}")

    # ---- report ----
    lines = ["# tabular-blind: 框架 vs 单线程爬山 对比（haiku/sonnet/opus）\n",
             "> 盲化任务（生成过程隐藏）。指标 neg_mean_test_accuracy，**越低（越负）越好**。\n",
             "| 方法 | 档 | best | 总评估数 |", "|---|---|---|---|"]
    for mk, mname in METHODS:
        for tier in TIERS:
            x = data[(mk, tier)]
            b = f"{x['best']:.4f}" if x["best"] != INF else "—"
            lines.append(f"| {mname} | {tier} | {b} | {x['evals']} |")
    # equal-budget compare: best within the min common eval budget per tier
    lines.append("\n## 等评估对比（每档取两法共同的最小评估数处的 best）")
    lines.append("| 档 | 共同预算 | 框架 best@预算 | 爬山 best@预算 | 胜者 |")
    lines.append("|---|---|---|---|---|")
    for tier in TIERS:
        hc, ex = data[("hc", tier)], data[("exp", tier)]
        budget = min(hc["evals"], ex["evals"])
        def at(curve, b):
            xs, ys = curve
            v = INF
            for x, y in zip(xs, ys):
                if x <= b:
                    v = y
            return v
        eb, hb = at(ex["curve"], budget), at(hc["curve"], budget)
        win = "框架" if eb < hb else ("爬山" if hb < eb else "平")
        lines.append(f"| {tier} | {budget} | {eb:.4f} | {hb:.4f} | {win} |"
                     if budget > 0 else f"| {tier} | 0 | — | — | — |")
    OUT_MD.write_text("\n".join(lines) + "\n")

    # ---- viz ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    colors = {"haiku": "C0", "sonnet": "C1", "opus": "C2"}
    for mk, mname in METHODS:
        for tier in TIERS:
            xs, ys = data[(mk, tier)]["curve"]
            if not xs:
                continue
            ax.plot(xs, ys, color=colors[tier], lw=2,
                    ls="-" if mk == "exp" else "--",
                    label=f"{mname}-{tier}")
    ax.set_xlabel("evaluations"); ax.set_ylabel("best neg_mean_test_accuracy (lower=better)")
    ax.set_title("best-vs-evaluations (solid=framework, dashed=hillclimb)")
    ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = axes[1]
    labels, vals, bars_c = [], [], []
    for tier in TIERS:
        for mk, mname in METHODS:
            x = data[(mk, tier)]
            labels.append(f"{mname[:4]}\n{tier}")
            vals.append(x["best"] if x["best"] != INF else 0)
            bars_c.append(colors[tier])
    ax.bar(range(len(vals)), vals, color=bars_c,
           hatch=["", "//"] * len(TIERS))
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("best (lower=better)"); ax.set_title("final best by method × tier")
    ax.grid(alpha=.3, axis="y")
    plt.suptitle("tabular-blind: framework vs hillclimb × {haiku,sonnet,opus}")
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=120)

    print("=== summary ===")
    for k, x in data.items():
        b = f"{x['best']:.4f}" if x["best"] != INF else "—"
        print(f"  {x['name']}: best={b} evals={x['evals']}")
    print("wrote", OUT_MD, "+", OUT_PNG)


if __name__ == "__main__":
    main()
