"""Deterministic grid arm — no LLM, no surrogate. Control for small spaces.

Mirrors the semantics of production ``tools/tuners/grid_search.py`` (the
legacy ``select_method`` choice for n_dims <= 2): expand every dimension to a
discrete set of levels, take the Cartesian product, evaluate the combos in a
seeded shuffled order.

Level construction (all from the contract, no tuning knobs):

- discrete dimensions — categorical, or int with at most ``MAX_ENUM`` values —
  are enumerated in full;
- numeric dimensions — float, or wider int — get ``R`` levels with inclusive
  endpoints, equispaced in log space when the dimension is log-scaled, else
  linear; ``R = max(2, ceil((B / D) ** (1 / n_numeric)))`` where ``B`` is the
  cell budget and ``D`` the product of the discrete cardinalities, so the
  first grid has roughly ``B`` combos (spooky: 8 x 3 = 24; a log-float x
  binary space: 12 x 2 = 24);
- FIXED (degenerate) dimensions keep their base value.

When the grid is consumed before the budget is, the numeric levels are
refined by inserting midpoints (in the dimension's own scale) and the new
combos are emitted, again shuffled; an int dimension stops refining once
every integer in its range is a level. A space whose every dimension is
discrete can therefore run out — the generator then returns, which the runner
records as an early exhaustion (a real outcome for this arm, not a bug).

Combos duplicating executed history (frozen rows + this cell's own) are
skipped before proposing; a runner-side rejection of any kind is likewise
skipped, since the next combo is independent of the outcome.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import arm_api  # noqa: E402

MAX_ENUM = 12  # int dimensions this narrow are enumerated, not gridded


def _numeric_levels(dim, count: int) -> list:
    lo, hi = float(dim.lo), float(dim.hi)
    if dim.log:
        raw = [math.exp(x) for x in _linspace(math.log(lo), math.log(hi), count)]
    else:
        raw = _linspace(lo, hi, count)
    if dim.kind == "int":
        raw = [int(round(v)) for v in raw]
    # Preserve order, drop collisions (int rounding, or count > range).
    seen, levels = set(), []
    for value in raw:
        if value not in seen:
            seen.add(value)
            levels.append(value)
    return levels


def _linspace(lo: float, hi: float, count: int) -> list[float]:
    if count <= 1:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + step * index for index in range(count)]


def _refine(dim, levels: list) -> list:
    """Insert midpoints between consecutive levels in the dimension's scale."""
    if len(levels) < 2:
        return levels
    out = [levels[0]]
    for a, b in zip(levels, levels[1:]):
        if dim.log:
            mid = math.exp((math.log(float(a)) + math.log(float(b))) / 2)
        else:
            mid = (float(a) + float(b)) / 2
        if dim.kind == "int":
            mid = int(round(mid))
            if mid in (a, b):
                out.append(b)
                continue
        out.extend([mid, b])
    return out


class Grid:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "grid"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        contract = ctx.contract
        varying = contract.varying_dimensions
        if not varying:
            raise arm_api.Unsupported("grid arm: no varying dimensions")

        discrete: dict[str, list] = {}
        numeric: dict[str, object] = {}
        for dim in varying:
            if dim.kind == "categorical":
                discrete[dim.name] = list(dim.options)
            elif dim.kind == "int" and int(dim.hi) - int(dim.lo) + 1 <= MAX_ENUM:
                discrete[dim.name] = list(range(int(dim.lo), int(dim.hi) + 1))
            else:
                numeric[dim.name] = dim
        fixed = {
            dim.name: contract.base_params[dim.name]
            for dim in contract.dimensions
            if dim.is_degenerate
        }

        d_card = math.prod(len(v) for v in discrete.values()) if discrete else 1
        if numeric:
            per_dim = (max(ctx.budget, 1) / d_card) ** (1.0 / len(numeric))
            resolution = max(2, math.ceil(per_dim - 1e-9))
        else:
            resolution = 0
        levels = {name: _numeric_levels(dim, resolution) for name, dim in numeric.items()}

        names = list(contract.search_space)  # declaration order
        proposed: set[str] = set()
        grid_level = 0
        while True:
            axes = []
            for name in names:
                if name in fixed:
                    axes.append([fixed[name]])
                elif name in discrete:
                    axes.append(discrete[name])
                else:
                    axes.append(levels[name])
            combos = [dict(zip(names, values)) for values in itertools.product(*axes)]
            fresh = []
            for combo in combos:
                identity = contract.params_identity(combo)
                if identity in proposed:
                    continue
                proposed.add(identity)
                fresh.append(combo)
            ctx.rng.shuffle(fresh)
            for params in fresh:
                executed = [
                    trial.config
                    for trial in ctx.state.trials
                    if trial.status in ("ok", "crash")
                ]
                if contract.is_duplicate(params, executed):
                    continue
                yield arm_api.Proposal(
                    params=params,
                    source="grid",
                    arm_state={
                        "grid_level": grid_level,
                        "grid_size": len(combos),
                        "grid_resolution": {k: len(v) for k, v in levels.items()},
                    },
                )
            # Refine numeric levels; stop when nothing can move any further.
            refined = {name: _refine(numeric[name], lv) for name, lv in levels.items()}
            if all(len(refined[name]) == len(levels[name]) for name in levels):
                return
            levels = refined
            grid_level += 1


ARM = Grid()
