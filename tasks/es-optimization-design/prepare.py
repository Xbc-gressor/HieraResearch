"""Fixed benchmark suite and the single config->score evaluation for
`es-optimization-design` — a black-box continuous OPTIMIZER DESIGN task.

Do not modify during normal experiments.

Why this task exists (contract-shape diversity): the tabular tasks all reduce
to "build an sklearn-style estimator, fit on x_train/y_train, score on a hidden
split". Here the candidate is not an estimator at all: it is a stateful,
iterative *optimization algorithm* that must find a good point of an unknown
function within a strict evaluation budget. There is no training data, no
fit/predict phase split, and no accuracy — the feedback signal is the best
function value found within the budget.

Design (why it has headroom):
- Four benchmark families with different failure modes: rastrigin/ackley (many
  local optima), rosenbrock (a narrow curved valley), schwefel (deceptive — its
  global optimum sits far from the origin, punishing origin-centered
  initialization).
- Blind/random search stays far from the optimum (score ~1-2); a well-designed
  modern ES (step-size/covariance adaptation + restarts) approaches 0 on most
  problems. Algorithm structure AND hyperparameter tuning both pay off, so the
  outer idea search and the inner tuner both have room to matter.

Scores follow the framework convention **lower is better**: every benchmark's
global minimum is 0, so each problem contributes log10(1 + best_fitness) and
the task reports `mean_log10_1p_best_fitness` over all problems (0 = every
problem solved to near-exact). `evaluate_config` is the ONE evaluation surface
(warm-start eval + Phase C tuning both call it); there is no separate official
run.
"""

from __future__ import annotations

import math

import numpy as np


METRIC = "mean_log10_1p_best_fitness"  # mean of log10(1 + best fitness): lower is better

DIMS = (2, 10, 30)
SEEDS = (11, 23, 37)
BUDGET_PER_DIM = 2000  # evaluate() calls allowed per problem = BUDGET_PER_DIM * dim


# ---------------------------------------------------------------------------
# Benchmark functions (all have global minimum f = 0)
# ---------------------------------------------------------------------------

def _rastrigin(x: np.ndarray) -> float:
    d = x.size
    return float(10.0 * d + np.sum(x * x - 10.0 * np.cos(2.0 * math.pi * x)))


def _rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def _ackley(x: np.ndarray) -> float:
    d = x.size
    return float(
        -20.0 * math.exp(-0.2 * math.sqrt(float(np.sum(x * x)) / d))
        - math.exp(float(np.sum(np.cos(2.0 * math.pi * x))) / d)
        + 20.0
        + math.e
    )


def _schwefel(x: np.ndarray) -> float:
    d = x.size
    return float(418.9828872724339 * d - np.sum(x * np.sin(np.sqrt(np.abs(x)))))


_BENCHMARKS = {
    # name -> (function, lower bound, upper bound); bounds are per-dimension
    "rastrigin": (_rastrigin, -5.12, 5.12),
    "rosenbrock": (_rosenbrock, -5.0, 10.0),
    "ackley": (_ackley, -32.768, 32.768),
    "schwefel": (_schwefel, -500.0, 500.0),
}


# ---------------------------------------------------------------------------
# Problem instance handed to the candidate's optimizer
# ---------------------------------------------------------------------------

class Problem:
    """One benchmark instance: an opaque function the optimizer may only probe
    through `evaluate`, within `budget` calls.

    Attributes the candidate may read: `name`, `dim`, `seed`, `bounds`
    (a `(lo, hi)` pair), `budget`, and `evals_used`. All randomness inside the
    candidate must derive from `seed` (e.g. `np.random.default_rng(problem.seed)`)
    so that scores are reproducible."""

    __slots__ = ("name", "dim", "seed", "bounds", "budget", "evals_used", "_fn")

    def __init__(self, name, fn, dim, seed, bounds, budget):
        self.name = name
        self._fn = fn
        self.dim = dim
        self.seed = seed
        self.bounds = bounds
        self.budget = budget
        self.evals_used = 0

    def evaluate(self, x) -> float:
        """Probe the function at point `x` (sequence of length `dim`). Raises
        RuntimeError once the budget is exhausted — plan batch sizes so that
        `evals_used` never exceeds `budget`."""
        if self.evals_used >= self.budget:
            raise RuntimeError(
                f"evaluation budget exhausted ({self.budget} calls) on "
                f"{self.name}/dim={self.dim}/seed={self.seed}"
            )
        self.evals_used += 1
        return float(self._fn(np.asarray(x, dtype=float)))


def build_problems() -> list[Problem]:
    """The fixed suite: 4 benchmarks x dims (2, 10, 30) x 3 seeds = 36 problems.
    Fresh instances per call (each carries its own mutable eval counter)."""
    return [
        Problem(name, fn, dim, seed, bounds, BUDGET_PER_DIM * dim)
        for name, (fn, *bounds) in _BENCHMARKS.items()
        for dim in DIMS
        for seed in SEEDS
    ]


# ---------------------------------------------------------------------------
# The single config -> score evaluation
# ---------------------------------------------------------------------------

def evaluate_config(make_model, params: dict) -> float:
    """The single `config -> score` evaluation (lower is better). For every
    problem, builds an optimizer via `make_model(problem, params)` and calls its
    `run()`, which must return the best fitness found using only
    `problem.evaluate` within `problem.budget`. Each problem contributes
    log10(1 + best_fitness); returns the mean over all problems. No separate
    official run — this value IS the candidate's score."""
    scores = []
    for problem in build_problems():
        optimizer = make_model(problem, params)
        best = float(optimizer.run())
        if not math.isfinite(best):
            raise ValueError(
                f"optimizer returned non-finite fitness on "
                f"{problem.name}/dim={problem.dim}/seed={problem.seed}: {best!r}"
            )
        scores.append(math.log10(1.0 + max(best, 0.0)))
    return float(sum(scores) / len(scores))
