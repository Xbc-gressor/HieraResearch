"""Pure-TPE control arm (PLAN-inner-smallspace follow-up, ``tpe_only``).

No LLM, no GP: ``optuna.samplers.TPESampler(multivariate=True)`` over the
production sampling surface (``bo_search.build_distributions``), which
natively handles log-float and categorical dimensions. Motivation: on <=2-dim
small spaces HEBO's per-step GP fit + EvolutionOpt costs as much wall-clock
as an LLM arm, while random/grid are cheap but non-adaptive (spaceship-000
narrow log-C band: both scored zero improvement in 24 evals where hebo_only
reached 0.1879). TPE is the standard cheap-adaptive middle point.

Everything except the sampler is ``random_search`` verbatim: same frozen
history injection, same clamp, same ask/tell feedback handling (ok -> tell
score; config-infeasible / task-preflight rejection -> tell
infeasible_value; other crash -> tell FAIL; duplicate -> cached value, else
FAIL). With TPE the told values actually shape the next suggestion, which is
the point of the arm.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "tuners"))

import arm_api  # noqa: E402
import bo_search  # noqa: E402
from _common import (  # noqa: E402
    cast_params_to_search_space,
    clamp_search_space_to_preflight,
)
from arms.current import _is_config_infeasible_detail  # noqa: E402


class TpeOnly:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "tpe_only"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        contract = ctx.contract
        checkpoint = ctx.checkpoint
        search_space = clamp_search_space_to_preflight(
            dict(contract.search_space),
            contract.base_params or None,
            contract.path,
            checkpoint.checkpoint_dir / "tune_report.json",
        )

        sampler = optuna.samplers.TPESampler(seed=ctx.seed, multivariate=True)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        distributions = bo_search.build_distributions(search_space)

        prior_rows = [
            (dict(row.params), float(row.score))
            for row in checkpoint.history
            if row.status == "ok" and row.score is not None and math.isfinite(row.score)
        ]
        incumbent_identity = contract.params_identity(checkpoint.incumbent.params)
        if not any(
            contract.params_identity(params) == incumbent_identity
            for params, _ in prior_rows
        ):
            prior_rows.insert(
                0, (dict(checkpoint.incumbent.params), checkpoint.incumbent.score)
            )
        prior_constraint_attrs = {
            "user_attrs": {bo_search.INFEASIBLE_ATTR: [0.0]},
            "system_attrs": {"constraints": (0.0,)},
        }
        bo_search._inject_prior_trials(
            study,
            [{"params": params, "score": score} for params, score in prior_rows],
            distributions,
            optuna.trial.create_trial,
            prior_constraint_attrs,
        )

        infeasible_value = max(
            (
                float(trial.value)
                for trial in study.trials
                if trial.value is not None and math.isfinite(float(trial.value))
            ),
            default=0.0,
        )

        known_scores: dict[str, float] = {}
        for params, score in prior_rows:
            if set(params) != set(search_space):
                continue
            try:
                identity = contract.params_identity(params)
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if score < known_scores.get(identity, math.inf):
                known_scores[identity] = float(score)
        known_infeasible: set[str] = set()

        while True:
            trial = study.ask()
            params = {
                key: bo_search.suggest(trial, key, entry)
                for key, entry in search_space.items()
            }
            params = cast_params_to_search_space(params, search_space)
            feedback = yield arm_api.Proposal(
                params=params,
                source="tpe",
                arm_state={"sampler": "tpe"},
            )
            identity = contract.params_identity(params)
            if feedback.kind == "outcome":
                if feedback.status == "ok":
                    study.tell(trial, float(feedback.score))
                    known_scores[identity] = float(feedback.score)
                    known_infeasible.discard(identity)
                elif _is_config_infeasible_detail(feedback.detail):
                    known_infeasible.add(identity)
                    study.tell(trial, infeasible_value)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
            elif feedback.stage == "task":
                known_infeasible.add(identity)
                study.tell(trial, infeasible_value)
            else:
                reason = feedback.reason or ""
                if not reason.startswith("duplicate"):
                    raise arm_api.ArmError(
                        "tpe_only: runner rejected an in-space TPE "
                        f"proposal: {reason!r}"
                    )
                if identity in known_scores:
                    study.tell(trial, known_scores[identity])
                elif identity in known_infeasible:
                    study.tell(trial, infeasible_value)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)


ARM = TpeOnly()
