"""Historical inner-v1 random-search arm — pure-random baseline over the
older production search distributions (ablation for Current's startup phase).

The archived experimental record shows that in first bouts the Current arm's entire
gain comes from its startup phase: with 1-2 frozen priors against
``n_startup_trials=8``, the first 6-7 of the 10 evaluations are plain
``RandomSampler`` draws that Optuna's TPESampler falls back to before the
surrogate engages. This arm reproduces exactly that behavior — nothing else:

- sampler: ``optuna.samplers.RandomSampler(seed=ctx.seed)`` — no model is
  ever fitted, no ``constraints_func`` (the startup phase it reproduces is
  likewise unaffected by constraints);
- distributions: ``bo_search.build_distributions(search_space)``; per-key
  suggest via ``bo_search.suggest`` — the identical production sampling
  surface Current draws from during startup (the production pre-search clamp
  ``clamp_search_space_to_preflight`` runs as in Current; it no-ops on frozen
  checkpoints);
- frozen history is injected exactly as Current does (same finite-prior row
  filter, same incumbent-insertion rule, same ``bo_search._inject_prior_trials``
  call), so a same-seed study starts from the identical state and — because
  TPESampler's startup fallback is literally ``RandomSampler(seed=seed)``
  consuming the same RNG stream — this arm's proposal sequence aligns with
  Current's startup-phase proposals draw for draw (asserted by test;
  holds while Current stays in startup, i.e. COMPLETE trials < 8);
- feedback handling mirrors Current's ask/tell loop verbatim (minus the
  feasibility user-attr marking, which is meaningless without a
  ``constraints_func``): ok -> ``tell(score)``; config-infeasible crash /
  task-preflight rejection -> ``tell(infeasible_value)``; any other crash ->
  ``tell(FAIL)``; deterministic duplicate rejection -> complete with the
  cached ``known_scores`` / ``known_infeasible`` value, else ``tell(FAIL)``.
  With no model none of this influences the sampler — scores exist only for
  the runner's incumbent tracking.

Deliberately absent (those are Current's production behaviors, NOT startup
behaviors, and this arm is the "is the startup advantage just random?"
ablation): rewarm (no session, zero LLM calls in ANY regime — first,
continuation, or deep), the deferred-config queue, and the
``select_method`` grid fallback (the ablation target is the bo startup
phase, so every space uses the bo distributions).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import arm_api  # noqa: E402
import bo_search  # noqa: E402
from _common import (  # noqa: E402
    cast_params_to_search_space,
    clamp_search_space_to_preflight,
)
from arms.current import _is_config_infeasible_detail  # noqa: E402


class RandomSearch:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "random_search"

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

        sampler = optuna.samplers.RandomSampler(seed=ctx.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        distributions = bo_search.build_distributions(search_space)

        # Frozen history as completed priors: identical filter / insertion /
        # injection as Current, so the study starts in the same state.
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

        # Same bookkeeping as Current's duplicate-feedback path.
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
                source="random",
                arm_state={"sampler": "random"},
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
                        "random_search: runner rejected an in-space random "
                        f"proposal: {reason!r}"
                    )
                if identity in known_scores:
                    study.tell(trial, known_scores[identity])
                elif identity in known_infeasible:
                    study.tell(trial, infeasible_value)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)


ARM = RandomSearch()
