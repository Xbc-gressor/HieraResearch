"""Offline outer-S-GoT benchmark using the production calculation primitives.

Run the deterministic benchmark loop on synthetic toys, vary the framework
cfg, and measure search efficiency.

Metrics per run: best@K-candidates (running-best after K nodes), AUC (mean of the
best-so-far curve; lower = reached good faster), final best, crash%, crossover
synergy% (crossover children strictly better than BOTH parents — the c_dag payoff).

Library: run_cfg(cfg, toys, seeds) -> per-toy mean metrics; sweep(...) for grids.
CLI (smoke): prints DEFAULT_CFG metrics across the 3 toys.
"""
from __future__ import annotations
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "got_benchmark"))

from got_select import DEFAULT_CFG  # noqa: E402
from sgot_runner import run_sgot  # noqa: E402
import toys as toysmod  # noqa: E402

KS = [10, 20, 40, 80]


def metrics(g, info, n_gens):
    nodes = list(g.nodes.items())
    running = float("inf")
    curve = []
    for _, n in nodes:
        if n.status != "crash" and n.score < running:
            running = n.score
        curve.append(running)
    n_c = len(curve)

    def best_at(k):
        return curve[min(k, n_c) - 1] if n_c else float("inf")

    # forward-fill leading inf (candidates before the first non-crash) with the
    # first finite best, so AUC has a stable denominator and isn't inf-polluted.
    first_finite = next((c for c in curve if c != float("inf")), float("inf"))
    filled = [c if c != float("inf") else first_finite for c in curve]

    cross = [(nid, n) for nid, n in nodes if n.op == "crossover" and n.status != "crash"]
    syn = sum(1 for nid, n in cross
              if g.parents(nid)
              and n.score < min(g.nodes[p].score for p in g.parents(nid)))
    crashes = sum(1 for _, n in nodes if n.status == "crash")
    out = {f"best@{k}": best_at(k) for k in KS}
    out.update({
        "final": info["best"],
        "auc": mean(filled) if filled else float("inf"),
        "n_cand": n_c,
        "crash_pct": round(100.0 * crashes / n_c, 1) if n_c else 0.0,
        "synergy_pct": round(100.0 * syn / len(cross), 1) if cross else 0.0,
        "Nop": dict(info["Nop"]),
    })
    return out


def run_one(toy_name, cfg, seed):
    spec = toysmod.TOY_SPECS[toy_name]
    oracle, ops, dir_priority, _ = toysmod.make_toy(seed=seed, **spec)
    g, _hist, info = run_sgot(oracle, ops, dir_priority, cfg, toysmod.N_GENS)
    return metrics(g, info, toysmod.N_GENS)


def run_cfg(cfg, toy_names=None, seeds=(0, 1, 2)):
    """Mean metrics per toy (averaged over seeds)."""
    toy_names = toy_names or list(toysmod.TOY_SPECS)
    per_toy = {}
    for t in toy_names:
        runs = [run_one(t, cfg, s) for s in seeds]
        keys = [f"best@{k}" for k in KS] + ["final", "auc", "crash_pct", "synergy_pct", "n_cand"]
        per_toy[t] = {k: mean(r[k] for r in runs) for k in keys}
    return per_toy


def main():
    print(f"DEFAULT_CFG smoke — {len(toysmod.TOY_SPECS)} toys × 3 seeds, n_gens={toysmod.N_GENS}")
    per_toy = run_cfg(dict(DEFAULT_CFG))
    hdr = ["toy", "best@10", "best@20", "best@40", "final", "auc", "crash%", "syn%", "n_cand"]
    print("  " + "  ".join(f"{h:>8}" for h in hdr))
    for t, m in per_toy.items():
        row = [t, m["best@10"], m["best@20"], m["best@40"], m["final"], m["auc"],
               m["crash_pct"], m["synergy_pct"], m["n_cand"]]
        print("  " + "  ".join(f"{v:>8.4f}" if isinstance(v, float) else f"{str(v):>8}" for v in row))


if __name__ == "__main__":
    main()
