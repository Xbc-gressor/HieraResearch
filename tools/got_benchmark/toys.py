"""Toy oracles for the offline outer-S-GoT benchmark.

Each toy is a synthetic "candidate quality" function exercising the production
calculation primitives through ``sgot_runner`` without real LLM-written
candidates. Structure: a D-dim target; each `tf-*` direction owns a subset of
dims; a `fresh` from that direction gets its owned dims ~right and the rest
random; `crossover` merges lineages (so combining complementary directions is
genuinely valuable — the c_dag/gamma payoff); `improve` is a local Gaussian
walk; evals crash with prob p_crash. score = squared distance to target (lower
is better, like the real neg-accuracy). Everything uses a LOCAL seeded RNG so
the benchmark can average over seeds.
"""
from __future__ import annotations
import random


def make_toy(seed, D, dir_dims, p_crash=0.07, fresh_noise=0.02, improve_noise=0.04):
    """Return (oracle, ops, dir_priority, target) with a local RNG(seed)."""
    rng = random.Random(seed)
    target = [rng.uniform(0.0, 1.0) for _ in range(D)]
    dir_ids = [f"tf-{i+1:02d}" for i in range(len(dir_dims))]
    dirs = dict(zip(dir_ids, dir_dims))

    def score_of(gn):
        return sum((gn[d] - target[d]) ** 2 for d in range(D))

    def oracle(gn):
        if rng.random() < p_crash:
            return (float("inf"), True)
        return (score_of(gn), False)

    def make_fresh(dir_id):
        owned = set(dirs[dir_id])
        return [(target[d] + rng.gauss(0, fresh_noise)) if d in owned
                else rng.uniform(0.0, 1.0) for d in range(D)]

    def make_improve(gn):
        return [min(1.0, max(0.0, gn[d] + rng.gauss(0, improve_noise))) for d in range(D)]

    def make_crossover(a, b):
        return [a[d] if rng.random() < 0.5 else b[d] for d in range(D)]

    ops = {"fresh": make_fresh, "improve": make_improve, "crossover": make_crossover}
    return oracle, ops, dir_ids, target


# 3 structurally-different toys (so param conclusions aren't toy-specific):
TOY_SPECS = {
    # baseline: 4 lineages each own 2 of 8 dims, moderate crash — the proto toy.
    "base": dict(D=8, dir_dims=[[0, 1], [2, 3], [4, 5], [6, 7]],
                 p_crash=0.07, improve_noise=0.04),
    # many lineages, each owns few dims → crossover across complements matters MORE.
    "manydir": dict(D=12, dir_dims=[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
                    p_crash=0.07, improve_noise=0.04),
    # rugged: higher crash + bigger improve noise + uneven dim ownership → harder.
    "rugged": dict(D=10, dir_dims=[[0, 1, 2], [3, 4], [5, 6], [7, 8, 9]],
                   p_crash=0.15, improve_noise=0.08),
}

N_GENS = 60  # deep enough for outer params (PUCB rounds, stall) to actually matter
