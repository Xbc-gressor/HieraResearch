"""LLM pool + GP-EI arm (PLAN §6.4, ``pool_gp_ei``).

Each step the proposer generates POOL=5 configs ranked by its own judgment
(arms/pool.py driver); this arm re-ranks the filtered pool with a
Gaussian-process Expected-Improvement scorer and executes the argmax. The
ranker sees only configs and the cell's fittable history — never the LLM's
rationale (PLAN §四). Continuation/deep checkpoints only (PLAN §一).

Ranker (PLAN §6.4 GP-EI variant; scores are always lower-is-better):

- Features: canonical normalized z of every VARYING numeric dimension
  (codec.encode), one-hot of every VARYING categorical dimension
  (codec.encode_categorical). Degenerate dimensions are excluded (D14):
  constant features carry no information. Receipt values are cast before
  encoding so the model sees what would actually run (production int cast
  truncates: a proposed 3.7 runs as 3).
- Objective: the LIVE finite unique history (ctx.state.finite_unique_history —
  checkpoint history plus the incumbent plus this cell's own ok outcomes),
  standardized before the fit: subtract the mean, divide by the std; a zero
  std is treated as 1 (a constant objective then fits a zero-mean GP and EI
  still computes, driven by sigma alone).
- GP: sklearn GaussianProcessRegressor, ARD Matérn-5/2 kernel (one independent
  lengthscale per feature — never isotropic), alpha=1e-8 fixed jitter,
  n_restarts_optimizer=4 MLE restarts, random_state=ctx.seed,
  normalize_y=False (standardization is done above). PLAN leaves the
  lengthscale bounds open; this arm fixes them at (1e-2, 1e2) — features live
  in [0,1] (z) / {0,1} (one-hot), so 1e-2 is "very local" and 1e2 is
  "dimension irrelevant" for ARD purposes.
- Acquisition: closed-form Expected Improvement for minimization against the
  CURRENT incumbent (ctx.state.incumbent_score, mapped into standardized
  units): imp = best - mu; Z = imp/sigma; EI = imp*Phi(Z) + sigma*phi(Z);
  EI = 0 where sigma = 0. Ties break toward the better proposer rank (lower
  pool index — first argmax wins).
- WARMUP gate (PLAN §5.1/§6.4): while the LIVE count
  len(ctx.state.finite_unique_history()) is below WARMUP=8, the arm executes
  the proposer's rank-1 config (pool[0] after duplicate filtering) and
  records ranker_fallback / ranker_fallback_count (arm_api checklist #4: the
  gate is the live count, which grows with this cell's own outcomes). Pure
  engineering insurance — continuation/deep checkpoints always carry >=
  WARMUP frozen history, so it is expected never to fire.
- A failed GP fit/predict (numpy linalg errors etc.) ends the cell as
  ArmError. No kernel/optimizer swap, no degradation to another strategy
  (PLAN §6.4).

A checkpoint with zero varying dimensions gives the GP an empty feature
vector; the arm marks itself Unsupported rather than fit noise.

A preflight rejection of the chosen config is reported to the proposer like
any outcome (PoolDriver.report_outcome) and the arm simply asks for a fresh
pool; the runner's 5-consecutive-reject tripwire bounds that path.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import Matern  # noqa: E402

import arm_api  # noqa: E402
from arms.pool import PoolDriver, pool_persistence_state  # noqa: E402

# PLAN §6.4 leaves the ARD lengthscale search bounds open; fixed here so every
# cell fits the same kernel family (see module docstring).
LENGTH_SCALE_BOUNDS = (1e-2, 1e2)
ALPHA_JITTER = 1e-8  # fixed numerical jitter (PLAN §6.4)
N_RESTARTS = 4  # MLE restarts (PLAN §6.4: 多次 restart)


class GpEiRanker:
    """ARD Matérn-5/2 GP + closed-form EI over pool configs (PLAN §6.4).

    One instance per cell (created in run()); ``gp_`` holds the last fitted
    regressor for inspection (e.g. per-dimension lengthscales).
    """

    def __init__(self, contract, codec, *, seed: int) -> None:
        self._contract = contract
        self._codec = codec
        self._seed = seed
        self._varying_numeric = [
            index
            for index, dim in enumerate(codec.numeric_dimensions)
            if not dim.is_degenerate
        ]
        self._varying_categoricals = [
            dim for dim in codec.categorical_dimensions if not dim.is_degenerate
        ]
        self.n_features = len(self._varying_numeric) + sum(
            len(dim.options) for dim in self._varying_categoricals
        )
        self.gp_: GaussianProcessRegressor | None = None

    def features(self, params: dict) -> list[float]:
        """params -> feature vector (varying numeric z, then varying
        categorical one-hots, declaration order)."""
        z, cat_labels = self._codec.encode(self._contract.cast(params))
        vector = [float(z[index]) for index in self._varying_numeric]
        for dim in self._varying_categoricals:
            one_hot = [0.0] * len(dim.options)
            one_hot[self._codec.encode_categorical(dim.name, cat_labels[dim.name])] = 1.0
            vector.extend(one_hot)
        return vector

    def rank(self, history, pool, incumbent_score: float) -> list[float]:
        """EI of every pool member (pool order), fit on ``history``
        (finite unique (config, score) pairs). Raises ArmError on fit failure."""
        X = np.array([self.features(params) for params, _ in history], dtype=float)
        y = np.array([score for _, score in history], dtype=float)
        y_mean = float(y.mean())
        y_std = float(y.std())
        scale = y_std if y_std > 0.0 else 1.0  # std 0 -> treat as 1 (PLAN §6.4)
        ys = (y - y_mean) / scale
        best = (float(incumbent_score) - y_mean) / scale
        gp = GaussianProcessRegressor(
            kernel=Matern(
                length_scale=np.ones(self.n_features),
                length_scale_bounds=LENGTH_SCALE_BOUNDS,
                nu=2.5,
            ),
            alpha=ALPHA_JITTER,
            n_restarts_optimizer=N_RESTARTS,
            random_state=self._seed,
            normalize_y=False,
        )
        try:
            gp.fit(X, ys)
            mu, sigma = gp.predict(
                np.array([self.features(params) for params in pool], dtype=float),
                return_std=True,
            )
        except Exception as exc:
            raise arm_api.ArmError(
                f"pool_gp_ei: GP fit/predict failed: {exc}"
            ) from exc
        self.gp_ = gp
        eis = np.zeros(len(pool), dtype=float)
        positive = sigma > 0.0
        imp = best - mu[positive]
        Z = imp / sigma[positive]
        eis[positive] = imp * norm.cdf(Z) + sigma[positive] * norm.pdf(Z)
        return [float(ei) for ei in eis]


class LlmPoolGpEi:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "pool_gp_ei"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        # PoolDriver raises Unsupported on zero varying dimensions, before
        # any session exists (uniform across the four pool arms).
        driver = PoolDriver(ctx)
        ranker = GpEiRanker(ctx.contract, ctx.codec, seed=ctx.seed)
        fallback_count = 0
        try:
            while True:
                result = driver.ask_pool()
                pool = result["pool"]
                history = ctx.state.finite_unique_history()
                if len(history) < arm_api.WARMUP:
                    # WARMUP gate (live count): execute proposer rank 1.
                    fallback_count += 1
                    fell_back = True
                    chosen_index = 0
                    eis = None
                else:
                    fell_back = False
                    eis = ranker.rank(history, pool, ctx.state.incumbent_score)
                    # First argmax wins ties -> the better proposer rank.
                    chosen_index = max(range(len(pool)), key=lambda index: eis[index])
                chosen = pool[chosen_index]
                # Capture before yield: the runner advances the incumbent
                # before the feedback returns; the verdict needs the score
                # the proposal had to beat.
                incumbent_before = ctx.state.incumbent_score
                feedback = yield arm_api.Proposal(
                    params=chosen,
                    source="pool_gp_ei_warmup_fallback" if fell_back else "pool_gp_ei",
                    rationale=result["rationale"],
                    arm_state={
                        "pool_size": len(pool),
                        "pool_attempts": result["attempts"],
                        **pool_persistence_state(result),
                        "selected_index": int(chosen_index),
                        "ranker_fallback": bool(fell_back),
                        "pool_eis": None if eis is None else [float(ei) for ei in eis],
                    },
                )
                driver.report_outcome(chosen, feedback, incumbent_before=incumbent_before)
        finally:
            ctx.emit({**driver.totals(), "ranker_fallback_count": fallback_count})


ARM = LlmPoolGpEi()
