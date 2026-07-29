"""Deterministic outer-S-GoT loop used by the synthetic benchmark.

This is benchmark infrastructure, not a validator. It composes the production
graph and selection primitives with synthetic genomes and oracles.
"""
from __future__ import annotations

from got_graph import CRASH, Graph
from got_select import decide_gen, pick_pucb, ucb_node


def run_sgot(oracle, ops, region_priority, cfg, n_gens):
    g = Graph(C=cfg["C"], alpha=cfg["alpha"])
    gbar = {"improve": 0.0, "crossover": 0.0}
    gcnt = {"improve": 0, "crossover": 0}
    Nop = {"improve": 0, "crossover": 0}
    best, stall = float("inf"), 0
    visited_regions, history = set(), []

    for gen in range(n_gens):
        roots = [r for r in g.roots() if g.nodes[r].status != "crash"]
        kind, k = decide_gen(len(roots), stall, cfg)

        leaves = []
        if kind == "pucb":
            key = lambda nid: ucb_node(g, nid, cfg["c_leaf"])
            for root in g.alive_roots():
                leaf = g.select_leaf(root, key)
                if leaf is not None:
                    leaves.append(leaf)
            leaves = list(dict.fromkeys(leaves))
            if not leaves:
                kind, k = "fresh", 1

        best_before = best
        produced = []

        if kind == "fresh":
            available = [
                region for region in region_priority
                if region not in visited_regions
            ] or list(region_priority)
            for region in available[:k]:
                genome = ops["fresh"](region)
                score, crashed = oracle(genome)
                status = "crash" if crashed else (
                    "kept" if score < best_before else "discard"
                )
                nid = g.add(
                    "fresh", [], genome, CRASH if crashed else score, status
                )
                visited_regions.add(region)
                produced.append((nid, "fresh", []))
        else:
            for op, args in pick_pucb(g, leaves, gbar, Nop, cfg):
                if op == "improve":
                    genome = ops["improve"](g.nodes[args[0]].genome)
                    parents = [args[0]]
                else:
                    genome = ops["crossover"](
                        g.nodes[args[0]].genome,
                        g.nodes[args[1]].genome,
                    )
                    parents = list(args)
                score, crashed = oracle(genome)
                status = "crash" if crashed else (
                    "kept" if score < best_before else "discard"
                )
                nid = g.add(
                    op, [str(parent) for parent in parents], genome,
                    CRASH if crashed else score, status,
                )
                Nop[op] += 1
                produced.append((nid, op, parents))

        rewards = g.r_map()
        for nid, op, parents in produced:
            node = g.nodes[nid]
            if node.status != "crash" and node.score < best:
                best = node.score
            if op in gbar and node.status != "crash" and parents:
                gain = max(
                    0.0,
                    rewards[nid] - max(rewards[parent] for parent in parents),
                )
                gcnt[op] += 1
                gbar[op] += (gain - gbar[op]) / gcnt[op]

        new_best = best < best_before
        stall = 0 if (kind == "fresh" or new_best) else stall + 1
        history.append({"gen": gen, "kind": kind, "best": best})

    info = {
        "best": best,
        "visited_regions": visited_regions,
        "Nop": Nop,
    }
    return g, history, info
