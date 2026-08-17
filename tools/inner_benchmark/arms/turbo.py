"""Hot-start TuRBO-1 local GP Thompson-sampling arm for DEEP20.

The arm is intended for a DEEP policy made of two scheduler-visible 10-eval
bouts.  Those bouts must pass ``turbo_final_state`` from the first cell back as
``ctx.extras["turbo_state"]`` for the second cell: the trust-region state and
proposal sequence then form one continuous 20-eval trajectory.

This is a task-adapted TuRBO-1 policy, not an exact reproduction of either the
original Uber code or the current BoTorch tutorial:

* one incumbent-centred, ARD-lengthscale-weighted trust region in [0, 1]^d;
* a Matérn-5/2 exact GP and Thompson sampling over perturbed Sobol candidates;
* length=0.8 initially, success_tolerance=3, and
  failure_tolerance=max(4, d) for sequential (batch-size one) evaluations;
* significant improvement means a decrease greater than 1e-3 * |best|.
* the GP uses the modern lengthscale bounds and candidate-pool size, while
  Thompson sampling draws from the latent posterior rather than observation
  noise.

Only varying numeric dimensions participate.  Categoricals are frozen at the
incumbent, and GP history is restricted to rows with those categorical values.
Integer coordinates are canonicalized through decode -> production cast ->
encode before filtering, fitting, or Thompson sampling, so the GP sees the
location that was actually executable.

The benchmark runner closes a generator without returning the last feedback
when the budget is exhausted.  The finally block reconciles that factual last
trial from ``ctx.state`` before emitting ``turbo_final_state``; otherwise a
10+10 continuation would always lose the outcome that can trigger a shrink.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import gpytorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import arm_api  # noqa: E402
import tune_tools  # noqa: E402

LENGTH_INIT = 0.8
LENGTH_MIN = 0.5**7
LENGTH_MAX = 1.6
SUCCESS_TOLERANCE = 3
SIGNIFICANT_RELATIVE_GAIN = 1e-3
NOISE_BOUNDS = (5e-4, 0.2)
LENGTHSCALE_BOUNDS = (0.005, 4.0)
OUTPUTSCALE_BOUNDS = (0.05, 20.0)
DEFAULT_FIT_STEPS = 50
DEFAULT_N_CANDIDATES_MIN = 2000
DEFAULT_N_CANDIDATES_MAX = 5000
MAX_CONSECUTIVE_EMPTY_POOLS = 3
MAX_CONSECUTIVE_WARMUP_DRAWS = 32
STATE_VERSION = 1
POLICY_ID = "hotstart-turbo1-deep20-v1"

_EXECUTED = ("ok", "crash")


@dataclass
class TurboState:
    dimension: int
    rng_seed: int
    length: float
    success_counter: int
    failure_counter: int
    success_tolerance: int
    failure_tolerance: int
    best_score: float
    proposal_index: int = 0
    outcomes_seen: int = 0
    restart_triggered: bool = False

    def to_json(self) -> dict:
        return {
            "version": STATE_VERSION,
            "policy_id": POLICY_ID,
            "dimension": int(self.dimension),
            "rng_seed": int(self.rng_seed),
            "length": float(self.length),
            "success_counter": int(self.success_counter),
            "failure_counter": int(self.failure_counter),
            "success_tolerance": int(self.success_tolerance),
            "failure_tolerance": int(self.failure_tolerance),
            "best_score": float(self.best_score),
            "proposal_index": int(self.proposal_index),
            "outcomes_seen": int(self.outcomes_seen),
            "restart_triggered": bool(self.restart_triggered),
        }


class _ExactTurboGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, dimension: int) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=dimension,
                lengthscale_constraint=gpytorch.constraints.Interval(
                    *LENGTHSCALE_BOUNDS
                ),
            ),
            outputscale_constraint=gpytorch.constraints.Interval(
                *OUTPUTSCALE_BOUNDS
            ),
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def _moving_numeric_indices(contract) -> list[int]:
    return [
        index
        for index, dim in enumerate(contract.numeric_dimensions)
        if not dim.is_degenerate
    ]


def _failure_tolerance(dimension: int) -> int:
    # Current TuRBO batch formula ceil(max(4 / batch, d / batch)), batch=1.
    return int(math.ceil(max(4.0, float(dimension))))


def _new_state(ctx, dimension: int) -> TurboState:
    return TurboState(
        dimension=dimension,
        rng_seed=int(ctx.seed),
        length=LENGTH_INIT,
        success_counter=0,
        failure_counter=0,
        success_tolerance=SUCCESS_TOLERANCE,
        failure_tolerance=_failure_tolerance(dimension),
        best_score=float(ctx.state.incumbent_score),
    )


def _load_state(ctx, dimension: int) -> TurboState:
    raw = (getattr(ctx, "extras", None) or {}).get("turbo_state")
    if raw is None:
        return _new_state(ctx, dimension)
    if not isinstance(raw, dict):
        raise arm_api.ArmError("turbo: extras.turbo_state must be an object")
    try:
        if int(raw.get("version")) != STATE_VERSION:
            raise ValueError(f"unsupported version {raw.get('version')!r}")
        if raw.get("policy_id") != POLICY_ID:
            raise ValueError(f"unexpected policy_id {raw.get('policy_id')!r}")
        if int(raw["dimension"]) != dimension:
            raise ValueError(
                f"state dimension {raw['dimension']} != current {dimension}"
            )
        state = TurboState(
            dimension=dimension,
            rng_seed=int(raw["rng_seed"]),
            length=float(raw["length"]),
            success_counter=int(raw["success_counter"]),
            failure_counter=int(raw["failure_counter"]),
            success_tolerance=int(raw["success_tolerance"]),
            failure_tolerance=int(raw["failure_tolerance"]),
            best_score=float(raw["best_score"]),
            proposal_index=int(raw.get("proposal_index", 0)),
            outcomes_seen=int(raw.get("outcomes_seen", 0)),
            restart_triggered=bool(raw.get("restart_triggered", False)),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise arm_api.ArmError(f"turbo: invalid persisted state: {exc}") from exc
    if not (
        LENGTH_MIN <= state.length <= LENGTH_MAX
        and state.success_tolerance == SUCCESS_TOLERANCE
        and state.failure_tolerance == _failure_tolerance(dimension)
        and 0 <= state.success_counter < state.success_tolerance
        and 0 <= state.failure_counter < state.failure_tolerance
        and math.isfinite(state.best_score)
        and state.proposal_index >= 0
        and state.outcomes_seen >= 0
    ):
        raise arm_api.ArmError("turbo: persisted state violates policy bounds")
    return state


def _recover_interrupted_outcomes(ctx, state: TurboState) -> None:
    """Advance a persisted pre-proposal state across its durable outcome.

    A hard process exit can leave the production adapter with a trial row but
    without the generator's final emit. The adapter identifies the factual
    rows newer than the persisted state and whether the last yielded proposal
    was consumed. Replaying those facts here preserves counters and the RNG
    sequence without re-evaluating the config.
    """
    extras = getattr(ctx, "extras", None) or {}
    raw = extras.get("turbo_recovery_outcomes", [])
    if not isinstance(raw, list):
        raise arm_api.ArmError(
            "turbo: extras.turbo_recovery_outcomes must be a list"
        )
    for index, row in enumerate(raw):
        if not isinstance(row, dict) or row.get("status") not in _EXECUTED:
            raise arm_api.ArmError(
                f"turbo: invalid recovery outcome at index {index}"
            )
        score = row.get("score")
        if row["status"] == "ok":
            try:
                valid_score = score is not None and math.isfinite(float(score))
            except (TypeError, ValueError, OverflowError):
                valid_score = False
            if not valid_score:
                raise arm_api.ArmError(
                    f"turbo: recovery outcome {index} has invalid score"
                )
        _apply_outcome(state, status=row["status"], score=score)
    consumed = extras.get("turbo_resume_proposal_consumed", False)
    if not isinstance(consumed, bool):
        raise arm_api.ArmError(
            "turbo: extras.turbo_resume_proposal_consumed must be boolean"
        )
    if consumed:
        state.proposal_index += 1


def _apply_outcome(state: TurboState, *, status: str, score: float | None) -> None:
    significant = (
        status == "ok"
        and score is not None
        and math.isfinite(float(score))
        and float(score)
        < state.best_score
        - SIGNIFICANT_RELATIVE_GAIN * abs(state.best_score)
    )
    if significant:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter >= state.success_tolerance:
        state.length = min(2.0 * state.length, LENGTH_MAX)
        state.success_counter = 0
    elif state.failure_counter >= state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    if status == "ok" and score is not None and math.isfinite(float(score)):
        state.best_score = min(state.best_score, float(score))
    state.outcomes_seen += 1
    state.restart_triggered = state.length < LENGTH_MIN


def _step_rng(state: TurboState) -> np.random.Generator:
    seed = np.random.SeedSequence(
        [state.rng_seed & 0xFFFFFFFF, state.proposal_index, 0x54555242]
    )
    return np.random.default_rng(seed)


def _categoricals_match(codec, params: dict, labels: dict) -> bool:
    return all(
        tune_tools._categorical_value_equal(params[dim.name], labels[dim.name])
        for dim in codec.categorical_dimensions
    )


def _training_data(ctx, moving: list[int], categorical_labels: dict):
    rows = [
        (params, score)
        for params, score in ctx.state.finite_unique_history()
        if _categoricals_match(ctx.codec, params, categorical_labels)
    ]
    X = []
    y = []
    for params, score in rows:
        z, _ = ctx.codec.encode(ctx.contract.cast(params))
        X.append([float(z[index]) for index in moving])
        y.append(float(score))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _fit_gp(X: np.ndarray, y: np.ndarray, *, seed: int, fit_steps: int):
    train_x = torch.as_tensor(X, dtype=torch.double)
    values = torch.as_tensor(y, dtype=torch.double)
    scale = values.std(unbiased=False)
    if not torch.isfinite(scale) or float(scale) < 1e-6:
        scale = torch.tensor(1.0, dtype=torch.double)
    train_y = (values - values.median()) / scale
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(*NOISE_BOUNDS)
    ).double()
    model = _ExactTurboGP(train_x, train_y, likelihood, X.shape[1]).double()
    model.covar_module.base_kernel.lengthscale = 0.5
    model.covar_module.outputscale = 1.0
    likelihood.noise = 0.005

    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            for _ in range(fit_steps):
                optimizer.zero_grad()
                loss = -mll(model(train_x), train_y)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite marginal likelihood")
                loss.backward()
                optimizer.step()
    except Exception as exc:
        raise arm_api.ArmError(f"turbo: GP fit failed: {exc}") from exc
    model.eval()
    likelihood.eval()
    lengthscales = (
        model.covar_module.base_kernel.lengthscale.detach()
        .reshape(-1)
        .cpu()
        .numpy()
    )
    return model, np.asarray(lengthscales, dtype=float)


def _trust_region(center: np.ndarray, lengthscales: np.ndarray, length: float):
    weights = lengthscales / float(lengthscales.mean())
    weights = weights / float(np.prod(weights) ** (1.0 / len(weights)))
    lower = np.clip(center - weights * length / 2.0, 0.0, 1.0)
    upper = np.clip(center + weights * length / 2.0, 0.0, 1.0)
    return lower, upper


def _executed_configs(ctx) -> list[dict]:
    return [
        trial.config for trial in ctx.state.trials if trial.status in _EXECUTED
    ]


def _candidate_pool(
    ctx,
    *,
    moving: list[int],
    categorical_labels: dict,
    center_full: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_candidates: int,
    rng: np.random.Generator,
):
    d = len(moving)
    sobol_seed = int(rng.integers(0, 2**31 - 1))
    sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=sobol_seed)
    raw = sobol.draw(n_candidates).double().cpu().numpy()
    raw = lower + (upper - lower) * raw
    mask = rng.random((n_candidates, d)) <= min(20.0 / d, 1.0)
    empty = np.flatnonzero(mask.sum(axis=1) == 0)
    if len(empty):
        mask[empty, rng.integers(0, d, size=len(empty))] = True
    raw = np.where(mask, raw, center_full[moving])

    executed = _executed_configs(ctx)
    identities: set[str] = set()
    params_pool: list[dict] = []
    canonical_pool: list[list[float]] = []
    outside = history_duplicates = pool_duplicates = 0
    for local_z in raw:
        full_z = center_full.copy()
        full_z[moving] = local_z
        params = ctx.codec.decode(full_z, categorical_labels)
        canonical_full, _ = ctx.codec.encode(params)
        canonical = canonical_full[moving]
        if np.any(canonical < lower - 1e-12) or np.any(canonical > upper + 1e-12):
            outside += 1
            continue
        identity = ctx.contract.params_identity(params)
        if ctx.contract.is_duplicate(params, executed):
            history_duplicates += 1
            continue
        if identity in identities:
            pool_duplicates += 1
            continue
        identities.add(identity)
        params_pool.append(params)
        canonical_pool.append([float(value) for value in canonical])
    return params_pool, np.asarray(canonical_pool), {
        "candidate_count_before": int(n_candidates),
        "candidate_count_after": len(params_pool),
        "candidate_outside_tr": outside,
        "candidate_history_duplicates": history_duplicates,
        "candidate_pool_duplicates": pool_duplicates,
    }


def _warmup_proposal(
    ctx,
    *,
    moving: list[int],
    categorical_labels: dict,
    center_full: np.ndarray,
    length: float,
    rng: np.random.Generator,
):
    lower = np.clip(center_full[moving] - length / 2.0, 0.0, 1.0)
    upper = np.clip(center_full[moving] + length / 2.0, 0.0, 1.0)
    executed = _executed_configs(ctx)
    duplicates = 0
    for _ in range(MAX_CONSECUTIVE_WARMUP_DRAWS):
        full_z = center_full.copy()
        full_z[moving] = rng.uniform(lower, upper)
        params = ctx.codec.decode(full_z, categorical_labels)
        if not ctx.contract.is_duplicate(params, executed):
            return params, duplicates
        duplicates += 1
    raise arm_api.ArmError(
        f"turbo: {MAX_CONSECUTIVE_WARMUP_DRAWS} consecutive warmup duplicates"
    )


class Turbo:
    name = "turbo"

    def active_dimensions(self, contract) -> int:
        return len(_moving_numeric_indices(contract))

    def calibration_report(self, ctx) -> dict:
        dimension = len(_moving_numeric_indices(ctx.contract))
        return {
            "turbo_policy_id": POLICY_ID,
            "turbo_length_init": LENGTH_INIT,
            "turbo_success_tolerance": SUCCESS_TOLERANCE,
            "turbo_failure_tolerance": _failure_tolerance(dimension)
            if dimension
            else None,
        }

    def run(self, ctx):
        moving = _moving_numeric_indices(ctx.contract)
        if not moving:
            raise arm_api.Unsupported(
                "turbo: checkpoint has no varying numeric dimension"
            )
        state = _load_state(ctx, len(moving))
        # Replay against the pre-proposal best before syncing the reconstructed
        # checkpoint incumbent; otherwise an interrupted improving outcome is
        # already the incumbent and is misclassified as a failure.
        _recover_interrupted_outcomes(ctx, state)
        state.best_score = min(
            state.best_score, float(ctx.state.incumbent_score)
        )
        if state.restart_triggered:
            raise arm_api.ArmError("turbo: persisted state already requests restart")

        extras = getattr(ctx, "extras", None) or {}
        fit_steps = int(extras.get("turbo_fit_steps", DEFAULT_FIT_STEPS))
        n_candidates = int(
            extras.get(
                "turbo_n_candidates",
                min(
                    DEFAULT_N_CANDIDATES_MAX,
                    max(DEFAULT_N_CANDIDATES_MIN, 200 * len(moving)),
                ),
            )
        )
        if fit_steps <= 0 or n_candidates <= 0:
            raise arm_api.ArmError(
                "turbo: turbo_fit_steps and turbo_n_candidates must be positive"
            )

        start_trial_index = len(ctx.state.trials)
        processed_outcomes = 0
        internal_duplicates = 0
        internal_resamples = 0
        last_lengthscales = None
        empty_pools = 0
        try:
            while True:
                if state.restart_triggered:
                    raise arm_api.ArmError(
                        "turbo: trust region reached restart threshold"
                    )
                center_full, categorical_labels = ctx.codec.encode(
                    ctx.state.incumbent_config
                )
                X, y = _training_data(ctx, moving, categorical_labels)
                rng = _step_rng(state)
                step_seed = int(rng.integers(0, 2**31 - 1))
                warmup = len(y) < 2
                if warmup:
                    params, duplicate_count = _warmup_proposal(
                        ctx,
                        moving=moving,
                        categorical_labels=categorical_labels,
                        center_full=center_full,
                        length=state.length,
                        rng=rng,
                    )
                    internal_duplicates += duplicate_count
                    internal_resamples += duplicate_count
                    lengthscales = None
                    diagnostics = {
                        "candidate_count_before": 1 + duplicate_count,
                        "candidate_count_after": 1,
                        "candidate_outside_tr": 0,
                        "candidate_history_duplicates": duplicate_count,
                        "candidate_pool_duplicates": 0,
                    }
                else:
                    model, lengthscales = _fit_gp(
                        X, y, seed=step_seed, fit_steps=fit_steps
                    )
                    last_lengthscales = [float(value) for value in lengthscales]
                    center = center_full[moving]
                    lower, upper = _trust_region(
                        center, lengthscales, state.length
                    )
                    pool, canonical, diagnostics = _candidate_pool(
                        ctx,
                        moving=moving,
                        categorical_labels=categorical_labels,
                        center_full=center_full,
                        lower=lower,
                        upper=upper,
                        n_candidates=n_candidates,
                        rng=rng,
                    )
                    internal_duplicates += diagnostics["candidate_history_duplicates"]
                    internal_duplicates += diagnostics["candidate_pool_duplicates"]
                    if not pool:
                        empty_pools += 1
                        internal_resamples += 1
                        if empty_pools >= MAX_CONSECUTIVE_EMPTY_POOLS:
                            raise arm_api.ArmError(
                                "turbo: canonical candidate pool empty on "
                                f"{MAX_CONSECUTIVE_EMPTY_POOLS} consecutive proposals"
                            )
                        state.proposal_index += 1
                        continue
                    empty_pools = 0
                    candidate_x = torch.as_tensor(canonical, dtype=torch.double)
                    try:
                        with torch.random.fork_rng(devices=[]):
                            torch.manual_seed(step_seed)
                            with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                                gpytorch.settings.fast_pred_samples():
                                draw = model(candidate_x).rsample().reshape(-1)
                        chosen = int(torch.argmin(draw).item())
                    except Exception as exc:
                        raise arm_api.ArmError(
                            f"turbo: posterior Thompson sample failed: {exc}"
                        ) from exc
                    params = pool[chosen]

                proposal_state = {
                    "turbo_state": state.to_json(),
                    "turbo_step": int(state.proposal_index),
                    "turbo_proposal_identity": ctx.contract.params_identity(params),
                    "tr_length": float(state.length),
                    "success_counter": int(state.success_counter),
                    "failure_counter": int(state.failure_counter),
                    "success_tolerance": int(state.success_tolerance),
                    "failure_tolerance": int(state.failure_tolerance),
                    "warmup_fallback": bool(warmup),
                    "gp_lengthscales": last_lengthscales,
                    **diagnostics,
                }
                state.proposal_index += 1
                feedback = yield arm_api.Proposal(
                    params=params,
                    source="turbo_warmup" if warmup else "turbo_ts",
                    arm_state=proposal_state,
                )
                if feedback.kind == "outcome":
                    _apply_outcome(
                        state, status=feedback.status, score=feedback.score
                    )
                    processed_outcomes += 1
        finally:
            # Budget exhaustion closes at the pending yield, after runner state
            # already contains the final trial. Reconcile any outcomes that did
            # not travel through Feedback before persisting the state.
            factual = [
                trial
                for trial in ctx.state.trials[start_trial_index:]
                if trial.status in _EXECUTED
            ]
            for trial in factual[processed_outcomes:]:
                _apply_outcome(state, status=trial.status, score=trial.score)
            ctx.emit(
                {
                    "turbo_final_state": state.to_json(),
                    "turbo_final_lengthscales": last_lengthscales,
                    "internal_duplicate_count": internal_duplicates,
                    "internal_resample_count": internal_resamples,
                }
            )


ARM = Turbo()
