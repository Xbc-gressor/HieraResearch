"""Current arm (PLAN §6.0) — reproduction of the production tuner pipeline.

This is the baseline arm: it runs the production ``tune_tools`` /
``bo_search`` / ``grid_search`` logic against the frozen checkpoint, adapted
to the runner's single-proposal generator protocol. It is NOT a
re-implementation from the plan text — every search-relevant decision is
made by the production functions:

- method selection: ``tune_tools.select_method`` (``n_dims <= 2`` -> grid,
  else bo = multivariate TPE; a fresh oversized grid rejects into its
  production fallback ``bo``);
- TPE: ``optuna.samplers.TPESampler`` with the exact production kwargs
  (``multivariate=True, group=True, n_startup_trials=8``,
  ``constraints_func=bo_search._infeasible_constraints``), seeded from
  ``ctx.seed``; distributions from ``bo_search.build_distributions``;
  per-key suggest via ``bo_search.suggest``;
- frozen history is injected into the study as completed trials via
  ``bo_search._inject_prior_trials`` (production warm-trials semantics:
  COMPLETE injected trials count toward ``n_startup_trials``, so with >= 8
  finite priors the very first TPE draw is surrogate-driven);
- rewarm (production Phase R): continuation/deep bouts only, one
  ``bench-rewarm-proposer`` call, up to ``rewarm_proposals=3`` configs,
  validated with production ``validate_proposals`` semantics (in-space via
  ``tune_tools._bounds_violations``, schema via ``_schema_accepts_value``,
  dedupe against every attempted config + earlier accepted proposals).
  Rejected/duplicate proposals are never evaluated and never consume B;
  accepted ones are enqueued ahead of everything (production order:
  proposals first, then deferred configs) and occupy the bout's first
  trial slots, naturally counting toward B;
- deferred queue: ``ctx.checkpoint.deferred_configs`` — visible ONLY to
  this arm (PLAN §七). Enqueued after the rewarm proposals via the
  production ``bo_search._enqueue_unique_deferred`` (grid path: production
  grid's cast + ``deduplicate_configs`` chain). Their evaluations count
  toward B — the PLAN §6.0 explicit deviation from production, where they
  are budget-neutral extras. A malformed deferred config fails the stage
  in production (``DeferredConfigError``); here it raises ``ArmError``;
- production capabilities kept: pre-search space clamp runs the real
  ``clamp_search_space_to_preflight`` with its full guard chain (it no-ops
  on frozen checkpoints — no runs/ task wiring, no VRAM telemetry — so
  feasibility information flows through the runner's task-preflight
  rejections, which are fed back to TPE as constrained-infeasible points
  exactly as production does), and preflight rejections / config-infeasible
  crashes are told to the study with the production penalty-value +
  ``INFEASIBLE_ATTR`` marking. This asymmetry over the other arms is
  inherent to Current (PLAN §6.0) and kept deliberately.

Batch-loop -> single-proposal adaptation: production drives
``study.optimize(objective, n_trials)``; this arm drives the equivalent
``study.ask()`` / ``study.tell()`` loop, yielding one ``Proposal`` per ask.
The sampler sees the identical trial sequence (enqueued WAITING trials
first, in enqueue order; suggest values fixed by the queue), because
production's objective body maps onto feedback handling one-to-one:

- ok outcome -> ``tell(score)`` (feasible), ``known_scores`` updated;
- crash that production classifies config-infeasible (evaluation timeout /
  OOM — mapped from the benchmark's crash ``detail`` string: a
  ``TimeoutError:`` prefix or "out of memory" text, the same two rules
  ``is_config_infeasible_error`` applies to exceptions) -> tell the
  pre-computed ``infeasible_value`` with the infeasible marking, identity
  added to ``known_infeasible``; any other crash -> ``tell(FAIL)`` (an
  unclassified crash is no region evidence);
- task-preflight rejection -> same as production: constrained-infeasible
  completion with the penalty value, identity into ``known_infeasible``;
- deterministic duplicate rejection -> production's ``known_scores`` /
  ``known_infeasible`` lookup (the study completes with the cached score /
  penalty instead of re-measuring); a duplicate with no known score (a
  crashed config) -> ``tell(FAIL)``.

Not reproduced (benchmark protocol overrides, PLAN §5.1): patience /
early stopping, stage chaining (a terminal failed bo stage falling back to
cmaes has no bout-local meaning here — the runner owns termination), and
any "space legally exhausted" terminal. The grid path exhausting its
deduped sweep before budget simply returns; the runner records that as
``arm_error`` ("arm exhausted early"). PLAN §5.1 measured only that the
raw config space cannot be exhausted by B=10 evaluations — a resolution-5
grid over a tiny 2-dim space CAN be; checkpoint selection pre-checks
expanded-grid size minus attempted overlap, keeping this corner
unreachable on the frozen corpus.

Explicit deviations from production Phase R (PLAN §6.0): the proposer gets
the §四 standard structured view (``llm.first_message_blocks``:
search_space / candidate / incumbent / history; ``tools=()``, no
``tune_report.json`` and no validate-proposals tool surface), while the
validation strength is unchanged (done in-arm with the production
``tune_tools`` primitives). First bouts never rewarm: no session is
created and zero LLM calls are made. ``session.totals()`` are emitted in
``finally``; rewarm proposals this arm itself recognizes as duplicates of
executed history (dropped before yielding) are counted under
``internal_duplicate_count``.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tuners"))

import arm_api  # noqa: E402
import llm  # noqa: E402
import tune_tools  # noqa: E402
import bo_search  # noqa: E402
import grid_search  # noqa: E402
from _common import (  # noqa: E402
    cast_params_to_search_space,
    clamp_search_space_to_preflight,
    deduplicate_configs,
    params_identity,
    split_configs_by_space,
)

REWARM_ROLE = "bench-rewarm-proposer"
REWARM_PROPOSALS = 3  # production tuner.rewarm_proposals default
N_STARTUP_TRIALS = 8  # production bo_search sampler kwarg
GRID_RESOLUTION = 5  # production grid_search --resolution default
GRID_MAX_TRIALS = 100  # production grid_search --max-trials fresh admission

REWARM_PROTOCOL = (
    "Current-arm rewarm (PLAN §6.0, production Phase R): this continuation "
    "bout rewarms the candidate's tuning. Propose 1-3 complete configs that "
    "your read of the evidence says are most promising (near the incumbent's "
    "best region unless the trials say it is exhausted). A deterministic "
    "validator keeps only in-space, schema-compatible configs that do not "
    "duplicate any executed history row or each other; accepted configs are "
    "evaluated FIRST inside this bout's budget (they displace search "
    "trials), rejected ones are never evaluated and consume nothing."
)

def _attempted_identities(params_rows, search_space) -> set[str]:
    """``_common.attempted_config_identities`` over checkpoint rows instead of
    a tune_report: every executed config (incumbent + history, any status),
    cast to the space with an exact key set."""
    identities: set[str] = set()
    for params in params_rows:
        try:
            normalized = cast_params_to_search_space(dict(params), search_space)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if set(normalized) == set(search_space):
            identities.add(params_identity(normalized))
    return identities


def _validate_rewarm_item(params, contract, search_space, seen: set) -> tuple[bool, bool]:
    """One item of production ``tune_tools.validate_proposals`` semantics
    (same order: object -> bounds -> schema -> identity dedupe). Returns
    ``(accepted, was_duplicate)``."""
    if not isinstance(params, dict):
        return False, False
    if tune_tools._bounds_violations(params, search_space):
        return False, False
    schema = contract.param_schema
    schema_bad = [
        key
        for key in search_space
        if key in schema
        and tune_tools._valid_schema_entry(schema[key])
        and not tune_tools._schema_accepts_value(schema[key], params[key])
    ]
    if schema_bad:
        return False, False
    identity = params_identity(cast_params_to_search_space(dict(params), search_space))
    if identity in seen:
        return False, True
    seen.add(identity)
    return True, False


def _is_config_infeasible_detail(detail) -> bool:
    """Production ``is_config_infeasible_error`` mapped onto the benchmark's
    crash detail string: evaluation timeouts (``objective`` renders the
    production ``TimeoutError`` label as a "TimeoutError: ..." prefix) and
    out-of-memory errors are config-infeasible; anything else is not."""
    if not detail:
        return False
    return detail.startswith("TimeoutError") or "out of memory" in detail.lower()


class Current:
    """arm_api protocol object; per-cell state lives in run()'s locals."""

    name = "current"

    def active_dimensions(self, contract) -> int:
        return len(contract.varying_dimensions)

    def run(self, ctx):
        contract = ctx.contract
        checkpoint = ctx.checkpoint
        search_space = dict(contract.search_space)
        # Production pre-search clamp (both search scripts call it). Frozen
        # checkpoints carry no runs/ task wiring or VRAM telemetry, so the
        # production guard chain returns the space unchanged.
        search_space = clamp_search_space_to_preflight(
            search_space,
            contract.base_params or None,
            contract.path,
            checkpoint.checkpoint_dir / "tune_report.json",
        )
        method = tune_tools.select_method(len(search_space))["method"]
        if method == "grid":
            total = 1
            for entry in search_space.values():
                total *= len(grid_search.expand_entry(entry, GRID_RESOLUTION))
            if total > GRID_MAX_TRIALS:
                # Production: a fresh oversized grid stage rejects, and the
                # orchestrator runs the select_method fallback ("bo").
                method = "bo"

        attempted_rows = [dict(checkpoint.incumbent.params)] + [
            dict(row.params) for row in checkpoint.history
        ]
        attempted = _attempted_identities(attempted_rows, search_space)

        # --- Phase R: rewarm proposals (continuation/deep bouts only) ------
        session = None
        rewarm_configs: list[dict] = []
        rewarm_rationale = None
        internal_duplicates = 0
        if checkpoint.regime != "first":
            factory = ctx.extras.get("session_factory")
            if factory is None:
                raise arm_api.ArmError(
                    "current: continuation/deep bouts require "
                    "ctx.extras['session_factory'] for the rewarm proposer"
                )
            session = factory(
                REWARM_ROLE,
                first_extras=llm.first_message_blocks(
                    checkpoint,
                    contract,
                    protocol=REWARM_PROTOCOL,
                    budget_remaining=ctx.budget,
                ),
            )
            receipt = None
            try:
                receipt = session.ask()
            except Exception:
                # Production's total-rejection path (validate-proposals exit 1):
                # proceed with the plain search, no rewarm proposals.
                receipt = None
            configs = receipt.get("configs") if isinstance(receipt, dict) else None
            if isinstance(configs, list):
                rewarm_rationale = receipt.get("rationale")
                # Validation-phase dedupe set. Production validate_proposals
                # builds a per-call local `seen` (attempted + accepted-so-far);
                # it must NOT alias ``attempted`` — the grid consumer chain
                # below reuses ``attempted`` for its own dedupe and would
                # silently drop every just-accepted proposal.
                rewarm_seen = set(attempted)
                for params in configs:
                    if len(rewarm_configs) >= REWARM_PROPOSALS:
                        break  # rewarm_proposals cap (production default 3)
                    accepted, was_duplicate = _validate_rewarm_item(
                        params, contract, search_space, rewarm_seen
                    )
                    if was_duplicate:
                        internal_duplicates += 1
                    if accepted:
                        rewarm_configs.append(params)

        rewarm_in_space, rewarm_outside = split_configs_by_space(
            rewarm_configs, search_space
        )
        deferred_in_space, deferred_outside = split_configs_by_space(
            [dict(params) for params in checkpoint.deferred_configs], search_space
        )
        ctx.emit(
            {
                "current_method": method,
                "rewarm_proposals_accepted": len(rewarm_in_space),
                "rewarm_skipped_outside_space": len(rewarm_outside),
                "deferred_skipped_outside_space": len(deferred_outside),
            }
        )

        try:
            if method == "grid":
                yield from self._run_grid(
                    ctx,
                    search_space,
                    attempted,
                    rewarm_in_space,
                    deferred_in_space,
                    rewarm_rationale,
                )
            else:
                yield from self._run_bo(
                    ctx,
                    search_space,
                    rewarm_in_space,
                    deferred_in_space,
                    rewarm_rationale,
                )
        finally:
            totals = (
                session.totals()
                if session is not None
                else {"llm_calls": 0, "llm_input_tokens": 0, "llm_output_tokens": 0}
            )
            totals["internal_duplicate_count"] = internal_duplicates
            ctx.emit(totals)

    def _run_bo(
        self,
        ctx,
        search_space,
        rewarm_in_space,
        deferred_in_space,
        rewarm_rationale,
    ):
        """Production bo_search as an ask/tell loop (see module docstring)."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        contract = ctx.contract
        checkpoint = ctx.checkpoint

        # Exact production sampler construction (bo_search.main).
        sampler = optuna.samplers.TPESampler(
            seed=ctx.seed,
            multivariate=True,
            group=True,
            n_startup_trials=N_STARTUP_TRIALS,
            constraints_func=bo_search._infeasible_constraints,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        distributions = bo_search.build_distributions(search_space)

        # Frozen history as completed priors (production warm-trials
        # semantics): every finite history row, plus the incumbent when the
        # freezer did not list it as a row (same insertion rule as runner).
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
        # Checkpoint crash rows are NOT injected as infeasible priors, and
        # ``known_infeasible`` starts empty. A production restart re-injects
        # preflight rejections and crashes flagged config-infeasible at record
        # time (``bo_search._inject_infeasible_trials``) and seeds
        # ``known_infeasible`` from them — but the frozen checkpoint schema
        # carries only ``status: ok|crash`` (PLAN §7: no tune_report copy), so
        # the flag is unrecoverable here. SANCTIONED DEVIATION: the bout-
        # internal feedback loop below re-marks any region the sampler
        # revisits; only the bout-start prior surface differs from a
        # production restart.

        # Production infeasible penalty: max finite study value, fixed here.
        infeasible_value = max(
            (
                float(trial.value)
                for trial in study.trials
                if trial.value is not None and math.isfinite(float(trial.value))
            ),
            default=0.0,
        )

        # Production enqueue order: validated rewarm proposals first, then
        # deferred configs (both via the production enqueue helper).
        try:
            n_rewarm = bo_search._enqueue_unique_deferred(
                study, rewarm_in_space, search_space, distributions
            )
            n_deferred = bo_search._enqueue_unique_deferred(
                study, deferred_in_space, search_space, distributions
            )
        except bo_search.DeferredConfigError as exc:
            raise arm_api.ArmError(
                f"current: enqueued configs failed production admission: {exc}"
            )

        # Production known_scores / known_infeasible bookkeeping, keyed by
        # ``contract.params_identity`` (cast + quantized) so the duplicate-
        # feedback lookups below agree with the runner's rejection decision;
        # the validation/grid chain above deliberately keeps production's raw
        # identity (it feeds production ``deduplicate_configs``).
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

        ask_index = 0
        while True:
            trial = study.ask()
            params = {
                key: bo_search.suggest(trial, key, entry)
                for key, entry in search_space.items()
            }
            params = cast_params_to_search_space(params, search_space)
            if ask_index < n_rewarm:
                source = "rewarm"
            elif ask_index < n_rewarm + n_deferred:
                source = "deferred"
            else:
                source = "tpe"
            ask_index += 1
            feedback = yield arm_api.Proposal(
                params=params,
                source=source,
                rationale=rewarm_rationale if source == "rewarm" else None,
                arm_state={"current_method": "bo"},
            )
            identity = contract.params_identity(params)
            if feedback.kind == "outcome":
                if feedback.status == "ok":
                    bo_search._set_feasibility(trial, feasible=True)
                    study.tell(trial, float(feedback.score))
                    known_scores[identity] = float(feedback.score)
                    known_infeasible.discard(identity)
                elif _is_config_infeasible_detail(feedback.detail):
                    bo_search._set_feasibility(trial, feasible=False)
                    known_infeasible.add(identity)
                    study.tell(trial, infeasible_value)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
            elif feedback.stage == "task":
                # Production: preflight rejection completes the trial as
                # constrained-infeasible with the penalty value.
                bo_search._set_feasibility(trial, feasible=False)
                known_infeasible.add(identity)
                study.tell(trial, infeasible_value)
            else:
                reason = feedback.reason or ""
                if not reason.startswith("duplicate"):
                    raise arm_api.ArmError(
                        "current: runner rejected an in-space TPE proposal: "
                        f"{reason!r}"
                    )
                # Production duplicate path: complete with the cached value
                # instead of re-measuring (no budget consumed here either).
                if identity in known_scores:
                    bo_search._set_feasibility(trial, feasible=True)
                    study.tell(trial, known_scores[identity])
                elif identity in known_infeasible:
                    bo_search._set_feasibility(trial, feasible=False)
                    study.tell(trial, infeasible_value)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)

    def _run_grid(
        self,
        ctx,
        search_space,
        attempted,
        rewarm_in_space,
        deferred_in_space,
        rewarm_rationale,
    ):
        """Production grid_search sweep: rewarm proposals, deferred configs,
        then the seeded-shuffle grid, deduped in that order (model-free:
        feedback only advances the sweep, as production's loop does)."""
        keys = list(search_space.keys())
        grids = [grid_search.expand_entry(search_space[key], GRID_RESOLUTION) for key in keys]
        combos = list(itertools.product(*grids))
        rng = random.Random(ctx.seed)  # production: random.Random(--seed)
        rng.shuffle(combos)

        proposals = [
            cast_params_to_search_space(dict(params), search_space)
            for params in rewarm_in_space
        ]
        proposals, _, seen = deduplicate_configs(proposals, seen=attempted)
        deferred = [
            cast_params_to_search_space(dict(params), search_space)
            for params in deferred_in_space
        ]
        deferred, _, seen = deduplicate_configs(deferred, seen=seen)
        grid_configs = [
            cast_params_to_search_space(dict(zip(keys, combo)), search_space)
            for combo in combos
        ]
        grid_configs, _, _ = deduplicate_configs(grid_configs, seen=seen)
        param_dicts = proposals + deferred + grid_configs
        n_proposals = len(proposals)
        n_deferred = len(deferred)

        index = 0
        while True:
            if index >= len(param_dicts):
                # Sweep exhausted before budget: the runner records this as
                # arm_error ("arm exhausted early"). Kept unreachable on the
                # frozen corpus by the selection-time grid-size pre-check
                # (see module docstring).
                return
            params = param_dicts[index]
            if index < n_proposals:
                source = "rewarm"
            elif index < n_proposals + n_deferred:
                source = "deferred"
            else:
                source = "grid"
            index += 1
            yield arm_api.Proposal(
                params=params,
                source=source,
                rationale=rewarm_rationale if source == "rewarm" else None,
                arm_state={"current_method": "grid"},
            )


ARM = Current()
