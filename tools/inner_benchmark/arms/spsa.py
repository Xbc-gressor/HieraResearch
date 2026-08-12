"""SPSA arm (PLAN §6.3).

First-order two-sided simultaneous-perturbation stochastic approximation over
the VARYING continuous dimensions only. Degenerate float dimensions cannot
move (the codec maps every z to the one legal value, so a perturbation along
them is a no-op that still pollutes the gradient estimate) and are excluded
exactly like integer / categorical dimensions, which hold the checkpoint
incumbent's values for the whole cell. NOT paired hillclimb: the internal
iterate theta is moved by the two-sided gradient estimate, never replaced by
the better side, and the iterate center itself never consumes budget — the
benchmark incumbent stays best-so-far over actually-executed configs
(runner-owned).

Per completed pair index k (0-based):

- c_k = 0.10 / (k+1)^0.101, a_k = a_0 / (k+1)^0.602 (classic exponents, fixed
  schedule, no stability constant A);
- delta_k iid Rademacher (+-1), drawn from ctx.np_rng;
- theta_k+- = Pi(theta_k +- c_k delta_k), Pi the codec's box projection;
- g_hat_j = (f(theta_k+) - f(theta_k-)) / (2 c_k delta_k,j);
- theta_(k+1) = Pi(theta_k - a_k g_hat).

a_0 is calibrated per checkpoint (PLAN §6.3, Spall 1998): s is the robust
scale of the FROZEN finite unique history scores
(``checkpoint.finite_unique_history(contract)``) — IQR at >= 4 observations,
else the best-worst range, and a zero scale from either rule falls back to
the 0.05 global default; a_0 = 2 c_0 target_step / s with target_step = 0.05.
(PLAN's ``a_k = 0.05/(k+1)^0.602`` is the uncalibrated nominal, i.e. s = 0.2.)
The frozen history is deliberately the calibration surface, not the live one:
a_0 is a per-checkpoint constant and must not drift with the cell's own
outcomes. s and a_0 reach the manifest via calibration_report.

Pair legality: a member duplicating ANY executed history row (crashes and the
incumbent included) would bounce at the runner's deterministic preflight, and
a pair whose two sides collapse to the same config (boundary projection)
carries no gradient signal — so the arm resamples delta_k in-arm until both
sides are legal and mutually distinct; 32 consecutive failed draws ->
ArmError, never a silent one-sided search. Drops/resamples are counted under
internal_duplicate_count / internal_resample_count, emitted in the finally.

Outcome semantics: the gradient update applies only when BOTH sides return ok
with finite scores. A crash on either side skips the update (iterate
unchanged, straight to k+1; the spent budget is not refunded). A preflight
rejection is scored the same way — no score pair, no update (only the task
preflight can still fire: the in-arm legality check already covers the
deterministic stage). Each pair consumes 2 evaluations; B=10 buys 5 gradient
estimates.

Reference: Spall (1992); Spall (1998).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import numpy as np  # noqa: E402

import arm_api  # noqa: E402

C0 = 0.10
C_EXPONENT = 0.101
A_EXPONENT = 0.602
TARGET_STEP = 0.05
DEFAULT_SCORE_SCALE = 0.05
IQR_MIN_OBSERVATIONS = 4
MAX_CONSECUTIVE_PAIR_DRAWS = 32

_EXECUTED = ("ok", "crash")


def _moving_continuous_indices(contract) -> list[int]:
    """z-vector indices SPSA moves: non-degenerate float dimensions."""
    return [
        index
        for index, dim in enumerate(contract.numeric_dimensions)
        if dim.kind == "float" and not dim.is_degenerate
    ]


def _calibrate(checkpoint, contract) -> tuple[float, float]:
    """(s, a_0) from the frozen finite unique history scores (PLAN §6.3).

    The incumbent is an executed finite config and counts. Scores are never
    empty (a checkpoint always carries a finite-score incumbent), so the
    only fallback that can fire is a zero scale.
    """
    scores = [score for _, score in checkpoint.finite_unique_history(contract)]
    if len(scores) >= IQR_MIN_OBSERVATIONS:
        scale = float(np.percentile(scores, 75) - np.percentile(scores, 25))
    else:
        scale = float(max(scores) - min(scores))
    if scale <= 0.0:
        scale = DEFAULT_SCORE_SCALE
    return scale, 2.0 * C0 * TARGET_STEP / scale


class Spsa:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "spsa"

    def active_dimensions(self, contract) -> int:
        return len(_moving_continuous_indices(contract))

    def calibration_report(self, ctx) -> dict:
        s, a_0 = _calibrate(ctx.checkpoint, ctx.contract)
        return {"s": s, "a_0": a_0}

    def run(self, ctx):
        contract = ctx.contract
        codec = ctx.codec
        moving = _moving_continuous_indices(contract)
        if not moving:
            raise arm_api.Unsupported(
                "spsa: checkpoint has no varying continuous dimension"
            )
        _, a_0 = _calibrate(ctx.checkpoint, ctx.contract)
        np_rng = ctx.np_rng
        internal_duplicates = 0
        internal_resamples = 0

        def executed_configs() -> list[dict]:
            return [
                trial.config
                for trial in ctx.state.trials
                if trial.status in _EXECUTED
            ]

        # theta_0 = the checkpoint incumbent's continuous coordinates; integer
        # and categorical dimensions stay at the incumbent's values throughout.
        theta, cats = codec.encode(ctx.checkpoint.incumbent.params)
        k = 0
        try:
            while True:
                c_k = C0 / (k + 1) ** C_EXPONENT
                a_k = a_0 / (k + 1) ** A_EXPONENT
                consecutive_failed_draws = 0
                while True:
                    delta = np_rng.integers(0, 2, size=len(moving)) * 2 - 1
                    plus = theta.copy()
                    plus[moving] += c_k * delta
                    minus = theta.copy()
                    minus[moving] -= c_k * delta
                    params_plus = codec.decode(codec.project(plus), cats)
                    params_minus = codec.decode(codec.project(minus), cats)
                    if (
                        contract.params_identity(params_plus)
                        != contract.params_identity(params_minus)
                        and not contract.is_duplicate(params_plus, executed_configs())
                        and not contract.is_duplicate(params_minus, executed_configs())
                    ):
                        break
                    internal_duplicates += 1
                    internal_resamples += 1
                    consecutive_failed_draws += 1
                    if consecutive_failed_draws >= MAX_CONSECUTIVE_PAIR_DRAWS:
                        raise arm_api.ArmError(
                            f"spsa: {MAX_CONSECUTIVE_PAIR_DRAWS} consecutive "
                            "pair draws hit duplicates/boundary collapse"
                        )
                pair_state = {
                    "spsa_k": k,
                    "c_k": float(c_k),
                    "a_k": float(a_k),
                }
                feedback_plus = yield arm_api.Proposal(
                    params=params_plus,
                    source="spsa_plus",
                    arm_state={**pair_state, "spsa_side": "plus"},
                )
                feedback_minus = yield arm_api.Proposal(
                    params=params_minus,
                    source="spsa_minus",
                    arm_state={**pair_state, "spsa_side": "minus"},
                )
                both_ok = (
                    feedback_plus.kind == "outcome"
                    and feedback_plus.status == "ok"
                    and feedback_minus.kind == "outcome"
                    and feedback_minus.status == "ok"
                )
                if both_ok:
                    g_hat = (feedback_plus.score - feedback_minus.score) / (
                        2.0 * c_k * delta
                    )
                    theta[moving] -= a_k * g_hat
                    theta = codec.project(theta)
                # else: crash or task-preflight rejection on either side — no
                # score pair exists, so skip the update (iterate unchanged)
                # and go straight to k+1; the spent budget is not refunded.
                k += 1
        finally:
            ctx.emit(
                {
                    "internal_duplicate_count": internal_duplicates,
                    "internal_resample_count": internal_resamples,
                }
            )


ARM = Spsa()
