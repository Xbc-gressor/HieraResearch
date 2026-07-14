"""Integrated outer+inner toy loop: the real S-GoT outer search (decide_gen /
pick_pucb / select_leaf on the Graph) + per-candidate warm-best-of-K scoring +
the REAL once-per-round decoupled deep-tune gate (tune_tools.select_candidate).

Lets the INNER knobs (K, patience, n_trials, top_percentile, n_min) be tuned
offline together with the outer cfg. Metrics weigh quality against inner compute.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "got_benchmark"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

from got_graph import Graph, CRASH  # noqa: E402
from got_select import decide_gen, pick_pucb, ucb_node, DEFAULT_CFG  # noqa: E402
from tune_tools import select_candidate  # noqa: E402  (the REAL deep-tune gate)
import inner_toy as it  # noqa: E402

N_GENS = 60


def run_inner(toy, cfg, *, K=5, top_percentile=80.0, patience=20, n_trials=40,
              max_evals=200, n_gens=10**6):
    """toy = dict from make_inner_toy. Runs until the TOTAL validation budget
    `max_evals` is spent (each warm HP sample + each deep-tune trial = 1 validation),
    so cfgs are compared at EQUAL budget. n_min derives from top_percentile."""
    ops, dir_priority = toy["ops"], toy["dir_priority"]
    warm_eval, deep_tune = toy["warm_eval"], toy["deep_tune"]
    n_min = math.ceil(100.0 / (100.0 - top_percentile)) if top_percentile < 100 else 10

    g = Graph(C=cfg["C"], alpha=cfg["alpha"])
    gbar = {"improve": 0.0, "crossover": 0.0}
    gcnt = {"improve": 0, "crossover": 0}
    Nop = {"improve": 0, "crossover": 0}
    best, stall = float("inf"), 0
    consumed, history, tuned = set(), [], set()
    warm_cost = deep_cost = n_deep = n_evals = 0

    for gen in range(n_gens):
        if n_evals >= max_evals:
            break
        roots = [r for r in g.roots() if g.nodes[r].status != "crash"]
        kind, k = decide_gen(len(roots), stall, cfg)
        L = []
        if kind == "pucb":
            key = lambda nid: ucb_node(g, nid, cfg["c_leaf"])
            for s in g.alive_roots():
                leaf = g.select_leaf(s, key)
                if leaf is not None:
                    L.append(leaf)
            L = list(dict.fromkeys(L))
            if not L:
                kind, k = "fresh", 1
        best_before = best
        produced = []
        if kind == "fresh":
            avail = [d for d in dir_priority if d not in consumed] or list(dir_priority)
            for d in avail[:k]:
                genome = ops["fresh"](d)
                score, crashed, nu = warm_eval(genome, K); warm_cost += nu; n_evals += nu
                status = "crash" if crashed else ("kept" if score < best_before else "discard")
                nid = g.add("fresh", [d], genome, CRASH if crashed else score, status)
                consumed.add(d); produced.append((nid, "fresh", []))
        else:
            for op, args in pick_pucb(g, L, gbar, Nop, cfg):
                if op == "improve":
                    genome, parents = ops["improve"](g.nodes[args[0]].genome), [args[0]]
                else:
                    genome = ops["crossover"](g.nodes[args[0]].genome, g.nodes[args[1]].genome)
                    parents = list(args)
                score, crashed, nu = warm_eval(genome, K); warm_cost += nu; n_evals += nu
                status = "crash" if crashed else ("kept" if score < best_before else "discard")
                nid = g.add(op, [str(p) for p in parents], genome, CRASH if crashed else score, status)
                Nop[op] += 1; produced.append((nid, op, parents))

        r = g.r_map()
        for nid, op, parents in produced:
            n = g.nodes[nid]
            if n.status != "crash" and n.score < best:
                best = n.score
            if op in ("improve", "crossover") and n.status != "crash" and parents:
                gval = max(0.0, r[nid] - max(r[p] for p in parents))
                gcnt[op] += 1; gbar[op] += (gval - gbar[op]) / gcnt[op]
        new_best = best < best_before
        stall = 0 if (kind == "fresh" or new_best) else stall + 1

        # ---- decoupled deep-tune gate (real select_candidate) ----
        ledger = {"records": [{"run_id": nid, "best_warm_score": n.score,
                               "tune": nid in tuned, "status": n.status}
                              for nid, n in g.nodes.items()]}
        sel = select_candidate(ledger, n_min=n_min, top_percentile=top_percentile)
        if sel.get("run_id") is not None and n_evals < max_evals:
            nid = sel["run_id"]; node = g.nodes[nid]
            budget = min(n_trials, max_evals - n_evals)  # cap deep trials at remaining budget
            new, nu = deep_tune(node.genome, node.score, budget, patience)
            deep_cost += nu; n_evals += nu
            node.score = new; tuned.add(nid); n_deep += 1
            if new < best:
                best = new

        history.append(dict(gen=gen, kind=kind, best=best))
    return g, history, dict(best=best, n_deep=n_deep, warm_cost=warm_cost,
                            deep_cost=deep_cost, Nop=Nop)


def run_cfg_inner(cfg, *, K=5, top_percentile=80.0, patience=20, n_trials=40,
                  max_evals=200, toy_names=None, seeds=(0, 1, 2)):
    toy_names = toy_names or list(it.TOY_SPECS)
    per_toy = {}
    for t in toy_names:
        finals, deeps, warmc, deepc = [], [], [], []
        for s in seeds:
            toy = it.make_inner_toy(seed=s, **it.TOY_SPECS[t])
            _, _, info = run_inner(toy, cfg, K=K, top_percentile=top_percentile,
                                   patience=patience, n_trials=n_trials, max_evals=max_evals)
            finals.append(info["best"]); deeps.append(info["n_deep"])
            warmc.append(info["warm_cost"]); deepc.append(info["deep_cost"])
        per_toy[t] = dict(final=mean(finals), n_deep=mean(deeps),
                          inner_cost=mean(w + d for w, d in zip(warmc, deepc)))
    return per_toy


def main():
    print(f"integrated outer+inner smoke — DEFAULT_CFG, K=5/p=20/nt=40/topP=80, {len(it.TOY_SPECS)} toys × 3 seeds")
    pt = run_cfg_inner(dict(DEFAULT_CFG))
    print(f"  {'toy':8} {'final':>8} {'n_deep':>7} {'inner_cost':>11}")
    for t, m in pt.items():
        print(f"  {t:8} {m['final']:>8.4f} {m['n_deep']:>7.1f} {m['inner_cost']:>11.0f}")


if __name__ == "__main__":
    main()
