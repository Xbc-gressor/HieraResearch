"""LLM pool + TPE density-ratio ranker arm (PLAN §6.4, ``pool_tpe``).

Each step the proposer generates POOL=5 configs (arms/pool.py driver); an
explicit, independent TPE-style scorer then ranks the filtered pool and only
the argmax config is executed. This is a diagnostic scorer — it answers "can
TPE-style good/bad signal rank LLM proposals better than the LLM's own
order" — and is NOT an Optuna multivariate TPE reproduction (PLAN §6.4, TPE
density ratio; Bergstra et al. 2011).

Scorer (scores are always lower-is-better):

- fit data: ``ctx.state.finite_unique_history()`` — deliberately the LIVE
  surface (arm_api checklist #4): it includes this cell's own outcomes and
  grows within the bout; crash (+inf) rows never enter it (PLAN §5.1);
- good set: the best ``max(2, ceil(0.25 * n))`` observations; the rest are bad;
- numeric dimensions: independent Gaussian Parzen densities in the codec's
  canonical normalized z space, Silverman bandwidth
  ``1.06 * std * n**(-1/5)`` floored at 0.05, log-densities via logsumexp
  (pure numpy, no scipy);
- categorical dimensions: Laplace-smoothed (+1) frequency models over the
  option index;
- score: ``sum_j [log l_j(x_j) - log g_j(x_j)]`` — higher is better; ties
  break toward the earlier pool member (numpy argmax);
- only varying dimensions are scored: a degenerate dimension cannot move, so
  its log-ratio is identically 0 and excluding it is cleaner (D14).

WARMUP gate (PLAN §5.1/§6.4): while the LIVE finite/unique/executed
observation count is below WARMUP=8 the arm executes the proposer's rank-1
config (pool[0] after duplicate filtering) with ``ranker_fallback=true`` and
counts it under ``ranker_fallback_count`` — an engineering fallback that
continuation/deep checkpoints are not expected to trigger.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import arm_api  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402

GOOD_FRACTION = 0.25
GOOD_MIN = 2
BANDWIDTH_FLOOR = 0.05

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _logsumexp(values) -> float:
    peak = float(np.max(values))
    return peak + math.log(float(np.exp(values - peak).sum()))


def _log_parzen(x: float, points: np.ndarray) -> float:
    """Log density at x of an independent Gaussian KDE over ``points``
    (Silverman bandwidth floored at BANDWIDTH_FLOOR)."""
    n = points.shape[0]
    bandwidth = max(BANDWIDTH_FLOOR, 1.06 * float(np.std(points)) * n ** (-0.2))
    log_kernels = (
        -0.5 * ((x - points) / bandwidth) ** 2
        - math.log(bandwidth)
        - _HALF_LOG_2PI
    )
    return _logsumexp(log_kernels) - math.log(n)


def tpe_pool_scores(pool, history, contract, codec) -> list[float]:
    """``sum_j [log l_j(x_j) - log g_j(x_j)]`` for each pool config.

    ``history``: (config, finite score) pairs, lower score is better. Both
    good and bad sets must be non-empty — the arm's WARMUP gate guarantees
    n >= 8 before this is called. Only varying dimensions contribute.
    """
    ordered = sorted(history, key=lambda item: item[1])
    n = len(ordered)
    n_good = max(GOOD_MIN, math.ceil(GOOD_FRACTION * n))
    good, bad = ordered[:n_good], ordered[n_good:]

    numeric = [
        index
        for index, dim in enumerate(codec.numeric_dimensions)
        if not dim.is_degenerate
    ]
    categoricals = [
        dim for dim in codec.categorical_dimensions if not dim.is_degenerate
    ]

    def encode(configs):
        zs, cat_options = [], []
        for config in configs:
            z, labels = codec.encode(contract.cast(config))
            zs.append(z)
            cat_options.append(
                {
                    dim.name: codec.encode_categorical(dim.name, labels[dim.name])
                    for dim in categoricals
                }
            )
        return np.array(zs, dtype=float), cat_options

    good_z, good_cats = encode([config for config, _ in good])
    bad_z, bad_cats = encode([config for config, _ in bad])
    n_bad = len(bad)

    scores = []
    for config in pool:
        z, labels = codec.encode(contract.cast(config))
        total = 0.0
        for index in numeric:
            total += _log_parzen(z[index], good_z[:, index]) - _log_parzen(
                z[index], bad_z[:, index]
            )
        for dim in categoricals:
            option = codec.encode_categorical(dim.name, labels[dim.name])
            k_options = len(dim.options)
            good_hits = sum(1 for entry in good_cats if entry[dim.name] == option)
            bad_hits = sum(1 for entry in bad_cats if entry[dim.name] == option)
            total += math.log((good_hits + 1) / (n_good + k_options)) - math.log(
                (bad_hits + 1) / (n_bad + k_options)
            )
        scores.append(float(total))
    return scores


class PoolTpe:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "pool_tpe"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        if not ctx.contract.varying_dimensions:
            raise arm_api.Unsupported(
                "pool_tpe: checkpoint has no varying dimension to move"
            )
        driver = PoolDriver(ctx)
        ranker_fallback_count = 0
        try:
            while True:
                result = driver.ask_pool()
                pool = result["pool"]
                observations = ctx.state.finite_unique_history()
                fallback = len(observations) < arm_api.WARMUP
                if fallback:
                    ranker_fallback_count += 1
                    chosen_index = 0
                    scores = None
                else:
                    scores = tpe_pool_scores(
                        pool, observations, ctx.contract, ctx.codec
                    )
                    chosen_index = int(np.argmax(scores))
                chosen = pool[chosen_index]
                # Capture before yield: the runner advances the incumbent
                # before the feedback returns; the verdict needs the score
                # the proposal had to beat.
                incumbent_before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(
                    params=chosen,
                    source="pool_tpe",
                    rationale=result["rationale"],
                    arm_state={
                        "pool_size": len(pool),
                        "pool_attempts": result["attempts"],
                        **pool_persistence_state(result),
                        "ranker_fallback": bool(fallback),
                        "tpe_scores": None
                        if scores is None
                        else [float(score) for score in scores],
                        "tpe_chosen_index": int(chosen_index),
                        "tpe_n_observations": len(observations),
                    },
                )
                driver.report_outcome(chosen, feedback, incumbent_before=incumbent_before)
        finally:
            ctx.emit(
                {**driver.totals(), "ranker_fallback_count": ranker_fallback_count}
            )


ARM = PoolTpe()
