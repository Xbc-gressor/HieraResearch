"""Cell runner for the inner-tuner benchmark (PLAN §五) — the protocol machine
every arm runs under.

``run_cell`` drives one (checkpoint, arm, seed) cell:

1. Load the frozen checkpoint (checkpoint.json is authoritative; history
   scores are already re-measured), read the candidate contract, build the
   codec, and initialize the authoritative CellState: checkpoint incumbent +
   history rows as executed trials (history consumes NO budget), with
   ``identity_fn=contract.params_identity`` (cast + identity — the Task-1
   caveat: CellState's default identity_fn does NOT cast).
2. Write manifest.json (write-once).
3. Drive the arm's proposal generator (see arm_api for the protocol):
   deterministic preflight (schema_invalid / out_of_space / duplicate — no
   budget consumed), then task preflight (no budget), then the objective
   (1 budget consumed at start, per PLAN §5.1 "objective 已启动即消耗 1
   budget"; crash still consumes and never improves the incumbent).
   5 consecutive preflight rejections (deterministic + task counted together)
   terminate the cell as "arm_error"; any objective start resets the counter.
4. Termination: budget exhausted -> close the generator without sending the
   final feedback, status "ok". Unsupported -> "unsupported". ArmError, early
   return, or an unhandled arm exception -> "arm_error" (traceback recorded
   for the last case; never propagated). Pre-drive arm failures — the
   active_dimensions / calibration_report hooks or the run(ctx) call itself
   raising — take the same containment path: the manifest is still written
   with null/{} fallbacks for the fields that never came back, so once
   run_cell is entered the cell always yields manifest + events + result.json.
5. Write result.json (metrics + best config + counts + status) and return it.

Events (PLAN §十) use the artifacts.make_event envelope; ``kind`` marks
"cell_start" / "cell_end" / "evaluation" / "preflight_rejected". eval_index is
1-based over objective evaluations only (null elsewhere); transaction_id is
0-based and strictly increasing over ALL events. Deterministic preflight
reason codes start the Feedback.reason / event preflight_detail strings.
"""

from __future__ import annotations

import collections.abc
import hashlib
import platform
import random
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tuners"))

import arm_api  # noqa: E402
import artifacts  # noqa: E402
import checkpoint as checkpoint_mod  # noqa: E402
import codec as codec_mod  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import objective  # noqa: E402
import space as space_mod  # noqa: E402
import state as state_mod  # noqa: E402
import tune_tools  # noqa: E402

B = arm_api.B
POOL = arm_api.POOL
WARMUP = arm_api.WARMUP

MAX_CONSECUTIVE_PREFLIGHT_REJECTS = 5

SCHEMA_INVALID = "schema_invalid"
OUT_OF_SPACE = "out_of_space"
DUPLICATE = "duplicate"

_UNSET = object()  # feedback sentinel: marks "no proposal processed yet"


def run_cell(
    *,
    arm,
    checkpoint_dir,
    out_dir,
    seed: int,
    budget: int = B,
    model: dict | None = None,
    eval_fn=objective.evaluate,
    preflight_fn=objective.preflight,
    manifest_extra: dict | None = None,
    extras: dict | None = None,
) -> dict:
    """Run one (checkpoint, arm, seed) cell; write manifest/events/result
    under ``out_dir`` and return the result dict (same content as result.json).

    ``model`` is the LLM model/decoding record for the manifest (PLAN §5.3) —
    pass llm.LLMConfig(...).to_manifest(). The runner owns this because the
    model is a per-cell invariant across every arm: letting each arm supply it
    reintroduces exactly the drift §5.3 forbids.

    ``eval_fn`` / ``preflight_fn`` default to the real objective path and are
    the sanctioned test seam (must match objective.evaluate / .preflight
    signatures). ``manifest_extra`` is merged into the manifest ``extra`` slot
    (under the arm's calibration_report, if any); ``extras`` is handed to the
    arm as ctx.extras (arm-specific wiring, e.g. deferred_configs for Current
    only, LLM session factories for LLM arms).
    """
    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)

    # --- 1. frozen state -------------------------------------------------
    checkpoint = checkpoint_mod.load_checkpoint(checkpoint_dir)
    contract = space_mod.read_contract(checkpoint.candidate_path)
    codec = codec_mod.Codec(contract)
    _validate_checkpoint_for_run(checkpoint, contract)
    # state.trials carries the incumbent too when the freezing tool did not
    # list it as a history row: otherwise state.finite_unique_history() (the
    # live WARMUP surface, and what the numerical rankers fit on) would be
    # missing the best known point while the dedupe set still rejects it.
    history_trials = [
        state_mod.Trial(
            config=dict(row.params),
            score=row.score,
            status=row.status,
            source=row.origin,
        )
        for row in checkpoint.history
    ]
    incumbent_identity = contract.params_identity(checkpoint.incumbent.params)
    if not any(
        contract.params_identity(trial.config) == incumbent_identity
        for trial in history_trials
    ):
        history_trials.insert(
            0,
            state_mod.Trial(
                config=dict(checkpoint.incumbent.params),
                score=checkpoint.incumbent.score,
                status="ok",
                source="incumbent",
            ),
        )
    state = state_mod.CellState(
        incumbent_config=dict(checkpoint.incumbent.params),
        incumbent_score=checkpoint.incumbent.score,
        budget_remaining=budget,
        trials=history_trials,
        identity_fn=contract.params_identity,
    )
    # Duplicate reference set: every executed config — checkpoint history plus
    # the incumbent (in case the freezing tool did not list it as a row).
    executed_params = [dict(checkpoint.incumbent.params)] + [
        dict(row.params) for row in checkpoint.history
    ]
    # Evaluation subprocess interpreter: the task project recorded in the
    # checkpoint (real runs) or sys.executable (test fixtures). Resolved once —
    # it is a per-checkpoint invariant like the model record.
    python_cmd = objective.python_cmd_for_project(checkpoint.task.project)

    pending_arm_extras: dict = {}

    def emit(values: dict) -> None:
        pending_arm_extras.update(values)

    ctx = arm_api.CellContext(
        contract=contract,
        codec=codec,
        state=state,
        checkpoint=checkpoint,
        rng=random.Random(seed),
        np_rng=np.random.default_rng(seed),
        budget=budget,
        seed=seed,
        extras=dict(extras) if extras else {},
        emit=emit,
    )

    # --- 2. manifest (write-once) ----------------------------------------
    # The arm hooks are contained: any exception ends the cell as arm_error
    # (section 3 below) with null/{} fallbacks for the fields that never
    # came back, so one hook bug cannot abort a batch sweep without record.
    active_dimensions = None
    calibration: dict = {}
    pre_drive_traceback: str | None = None
    try:
        probe = getattr(arm, "active_dimensions", None)
        if callable(probe):
            active_dimensions = probe(contract)
        probe = getattr(arm, "calibration_report", None)
        if callable(probe):
            report = probe(ctx)
            if report:
                calibration = dict(report)
    except Exception:
        pre_drive_traceback = traceback.format_exc()
    checkpoint_bytes = (checkpoint_dir / checkpoint_mod.CHECKPOINT_FILENAME).read_bytes()
    artifacts.write_manifest(
        out_dir,
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "checkpoint_hash": hashlib.sha256(checkpoint_bytes).hexdigest(),
            # The frozen train.py is NOT covered by checkpoint_hash (and
            # BASE_PARAMS is exactly what the freezer rewrites into it).
            # objective.evaluate already pins this revision on every call, so
            # recording it here — rather than adding a second content hash —
            # makes tampering BETWEEN cells detectable too.
            "candidate_execution_revision": tune_tools._candidate_execution_revision(
                checkpoint.candidate_path
            ),
            "arm": arm.name,
            "seed": seed,
            "model": dict(model) if model else {},
            "dependencies": _dependency_versions(),
            "hardware": _hardware_info(),
            "active_dimensions": active_dimensions,
            "extra": {**(manifest_extra or {}), **calibration},
        },
    )

    # --- 3. drive the generator ------------------------------------------
    transaction_id = 0
    eval_index = 0
    consecutive_rejects = 0
    counts = {"invalid": 0, "duplicates": 0, "task_preflight_rejected": 0}
    arm_states: list[dict] = []
    cell_outcomes: list[state_mod.Trial] = []
    cell_status = "ok"
    reason: str | None = None
    arm_traceback: str | None = None
    cleanup_traceback: str | None = None

    def record_event(event: dict) -> None:
        nonlocal transaction_id
        event["transaction_id"] = transaction_id
        transaction_id += 1
        arm_states.append(event["arm_state"])
        artifacts.append_event(out_dir, event)

    def terminate(status: str, why: str | None, tb: str | None = None) -> None:
        nonlocal cell_status, reason, arm_traceback
        cell_status, reason, arm_traceback = status, why, tb

    def close_generator(gen, *, downgrade_status: bool = True) -> None:
        """Close the arm generator; record a raising cleanup.

        On the budget-exhaustion path the cell has ALREADY succeeded, and
        arm_api tells arms to clean up in try/finally — so a raising finally
        must not relabel a full 10-evaluation cell as arm_error (status is the
        natural §九 filter). There it is recorded as cleanup_traceback instead.
        """
        nonlocal cleanup_traceback
        try:
            gen.close()
        except Exception:
            cleanup_traceback = traceback.format_exc()
            if downgrade_status:
                terminate(
                    "arm_error",
                    "unhandled arm exception during generator close",
                    cleanup_traceback,
                )

    record_event(
        artifacts.make_event(
            kind="cell_start",
            checkpoint_id=checkpoint.checkpoint_id,
            arm=arm.name,
            seed=seed,
            budget=budget,
        )
    )

    gen = None
    if pre_drive_traceback is not None:
        terminate(
            "arm_error",
            "unhandled arm exception in pre-drive arm hook "
            "(active_dimensions / calibration_report)",
            pre_drive_traceback,
        )
    else:
        try:
            gen = arm.run(ctx)
        except Exception:
            terminate("arm_error", "unhandled arm exception", traceback.format_exc())
            gen = None
        if gen is not None and not isinstance(gen, collections.abc.Generator):
            terminate("arm_error", "arm.run(ctx) must return a generator")
            gen = None

    feedback = _UNSET
    while gen is not None:
        if state.budget_remaining <= 0:
            # Budget exhausted: close without sending the pending feedback
            # (the arm never receives the last outcome); status stays "ok".
            close_generator(gen, downgrade_status=False)
            break
        try:
            proposal = next(gen) if feedback is _UNSET else gen.send(feedback)
        except StopIteration:
            terminate(
                "arm_error",
                f"arm exhausted early: returned after {eval_index} evaluations "
                f"with budget_remaining={state.budget_remaining}",
            )
            break
        except arm_api.Unsupported as exc:
            terminate("unsupported", str(exc))
            break
        except arm_api.ArmError as exc:
            terminate("arm_error", str(exc))
            break
        except Exception:
            terminate("arm_error", "unhandled arm exception", traceback.format_exc())
            break
        if not isinstance(proposal, arm_api.Proposal):
            terminate(
                "arm_error",
                f"arm yielded a non-Proposal object: {type(proposal).__name__}",
            )
            close_generator(gen)
            break

        proposal_arm_state = {**proposal.arm_state, **pending_arm_extras}
        pending_arm_extras.clear()

        # Deterministic preflight: schema_invalid / out_of_space / duplicate.
        det_code, det_detail, cast = _deterministic_preflight(
            contract, proposal.params, executed_params
        )
        if det_code is not None:
            if det_code == DUPLICATE:
                counts["duplicates"] += 1
            else:
                counts["invalid"] += 1
            consecutive_rejects += 1
            reject_reason = f"{det_code}: {det_detail}"
            rejected_config = cast if cast is not None else proposal.params
            if isinstance(rejected_config, dict):
                state.record_outcome(
                    rejected_config,
                    status=state_mod.PREFLIGHT_REJECTED,
                    source=proposal.source,
                    rationale=proposal.rationale,
                )
            record_event(
                artifacts.make_event(
                    eval_index=None,
                    proposal=cast if cast is not None else proposal.params,
                    source=proposal.source,
                    rationale=proposal.rationale,
                    preflight_status="rejected",
                    status="preflight_rejected",
                    arm_state=proposal_arm_state,
                    kind="preflight_rejected",
                    preflight_stage="deterministic",
                    preflight_detail=reject_reason,
                )
            )
            if consecutive_rejects >= MAX_CONSECUTIVE_PREFLIGHT_REJECTS:
                terminate(
                    "arm_error",
                    f"{MAX_CONSECUTIVE_PREFLIGHT_REJECTS} consecutive preflight "
                    f"rejections (last: deterministic {reject_reason})",
                )
                close_generator(gen)
                break
            feedback = arm_api.Feedback.preflight_rejected(
                stage="deterministic", reason=reject_reason
            )
            continue

        # Task preflight (no-score feasibility probe; never consumes budget).
        preflight = preflight_fn(
            checkpoint.candidate_path,
            cast,
            preflight_fn=checkpoint.task.preflight_fn,
            per_runtime_limit=checkpoint.task.per_runtime_limit,
            python_cmd=python_cmd,
        )
        if preflight.status != "ok":
            counts["task_preflight_rejected"] += 1
            consecutive_rejects += 1
            state.record_outcome(
                cast,
                status=state_mod.PREFLIGHT_REJECTED,
                source=proposal.source,
                rationale=proposal.rationale,
            )
            # NOT added to executed_params: a task preflight can fail for
            # transient reasons (memory pressure), so the deterministic stage
            # must not permanently ban the config. The arm sees the row in
            # ctx.state.trials and is responsible for not re-proposing it;
            # the 5-consecutive-reject tripwire bounds the worst case.
            record_event(
                artifacts.make_event(
                    eval_index=None,
                    proposal=cast,
                    source=proposal.source,
                    rationale=proposal.rationale,
                    preflight_status="rejected",
                    status="preflight_rejected",
                    arm_state=proposal_arm_state,
                    kind="preflight_rejected",
                    preflight_stage="task",
                    preflight_detail=preflight.detail,
                )
            )
            if consecutive_rejects >= MAX_CONSECUTIVE_PREFLIGHT_REJECTS:
                terminate(
                    "arm_error",
                    f"{MAX_CONSECUTIVE_PREFLIGHT_REJECTS} consecutive preflight "
                    f"rejections (last: task {preflight.detail})",
                )
                close_generator(gen)
                break
            feedback = arm_api.Feedback.preflight_rejected(
                stage="task", reason=preflight.detail
            )
            continue

        # Objective start: the budget unit is consumed by record_outcome
        # below, AFTER eval_fn returns (crash rows included) — this relies on
        # the Task-2 contract that eval_fn returns crash outcomes instead of
        # raising. Everything from here to the flushed event is contained:
        # the budget is already spent, so a bookkeeping failure must still
        # leave a cell_end + result.json rather than discarding the work.
        consecutive_rejects = 0
        eval_index += 1
        incumbent_before = {
            "config": dict(state.incumbent_config),
            "score": state.incumbent_score,
        }
        try:
            outcome = eval_fn(
                checkpoint.candidate_path,
                cast,
                score_fn=checkpoint.task.score_fn,
                per_runtime_limit=checkpoint.task.per_runtime_limit,
                python_cmd=python_cmd,
            )
            trial = state.record_outcome(
                cast,
                status=outcome.status,
                score=outcome.score,
                source=proposal.source,
                rationale=proposal.rationale,
            )
            cell_outcomes.append(trial)
            executed_params.append(cast)
            record_event(
                artifacts.make_event(
                    eval_index=eval_index,
                    proposal=cast,
                    source=proposal.source,
                    rationale=proposal.rationale,
                    preflight_status="ok",
                    status=outcome.status,
                    score=outcome.score,
                    incumbent_before=incumbent_before,
                    incumbent_after={
                        "config": dict(state.incumbent_config),
                        "score": state.incumbent_score,
                    },
                    arm_state=proposal_arm_state,
                    kind="evaluation",
                    detail=outcome.detail,
                    elapsed_seconds=outcome.elapsed_seconds,
                )
            )
        except Exception:
            terminate(
                "arm_error",
                "runner bookkeeping failure after objective start",
                traceback.format_exc(),
            )
            close_generator(gen)
            break
        feedback = arm_api.Feedback.outcome(
            status=outcome.status,
            score=outcome.score,
            detail=outcome.detail,
            executed_params=dict(cast),
        )

    # --- 4/5. cell end + result ------------------------------------------
    # Flush whatever the arm emitted after its last proposal (a final-tally
    # emit in a try/finally is the natural place for LLM token counts): those
    # values would otherwise never reach an event, and so never be aggregated.
    record_event(
        artifacts.make_event(
            kind="cell_end",
            status=cell_status,
            reason=reason,
            evaluations=eval_index,
            arm_state=dict(pending_arm_extras),
        )
    )
    pending_arm_extras.clear()
    best_config, best_score = state.best_so_far()
    result = {
        "status": cell_status,
        "reason": reason,
        "traceback": arm_traceback,
        "cleanup_traceback": cleanup_traceback,
        "checkpoint_id": checkpoint.checkpoint_id,
        "arm": arm.name,
        "seed": seed,
        "best_config": best_config,
        "best_score": best_score,
        # Sampled here, not at manifest time: the arms import their optimizer
        # libraries lazily inside run(), so this is the only point where the
        # record reflects what the cell actually used.
        "dependencies": _dependency_versions(),
        **metrics_mod.compute_metrics(
            initial_incumbent_score=checkpoint.incumbent.score,
            outcomes=cell_outcomes,
            budget=budget,
            counts=counts,
            arm_states=arm_states,
        ),
    }
    artifacts.write_result(out_dir, result)
    return result


def _validate_checkpoint_for_run(checkpoint, contract) -> None:
    """Refuse to spend GPU hours on a checkpoint that must not be run.

    load_checkpoint deliberately accepts provisional checkpoints so
    ``freeze inspect`` can look at them; the selection-to-execution boundary
    is here. Two rules from PLAN §七:

    - re-measurement must have completed and been valid (otherwise history
      scores came from a different machine than the arm's own scores);
    - DEEP regimes need >= WARMUP finite unique observations, else
      the three numerical-ranker arms silently degenerate into
      llm_pool_self_rank and the experiment's central contrast collapses.
    """
    remeasure = (checkpoint.extra or {}).get("remeasure")
    if not remeasure or not remeasure.get("valid") or not remeasure.get("complete"):
        raise ValueError(
            f"checkpoint {checkpoint.checkpoint_id} has no valid completed "
            f"re-measurement (extra.remeasure={remeasure!r}); run "
            f"`freeze.py remeasure` before running cells on it"
        )
    if not checkpoint_mod.is_initial_regime(checkpoint.regime):
        finite = checkpoint.finite_unique_history(contract)
        if len(finite) < WARMUP:
            raise ValueError(
                f"checkpoint {checkpoint.checkpoint_id} (regime={checkpoint.regime}) "
                f"has {len(finite)} finite unique observations, below WARMUP={WARMUP}"
            )


def _deterministic_preflight(contract, params, executed_params):
    """-> (reason_code | None, detail | None, cast params | None).

    Order per PLAN §5.1: cast cleanly via the contract (schema_invalid), then
    in-bounds per SEARCH_SPACE (out_of_space — production _bounds_violations
    semantics on the cast params: exact key set, in-bounds values), then
    exact-duplicate against every executed history row (duplicate —
    contract.params_identity, i.e. cast + identity).
    """
    if not isinstance(params, dict):
        return SCHEMA_INVALID, f"params must be a dict, got {type(params).__name__}", None
    try:
        cast = contract.cast(params)
    except (TypeError, ValueError, ArithmeticError) as exc:
        # ArithmeticError covers OverflowError: an LLM emitting 1e400 on an int
        # dimension yields inf, and int(inf) raises past TypeError/ValueError.
        return SCHEMA_INVALID, f"params failed the contract cast: {exc}", None
    violations = tune_tools._bounds_violations(cast, contract.search_space)
    if violations:
        detail = "; ".join(f"{v['key']}: {v['reason']}" for v in violations)
        return OUT_OF_SPACE, detail, cast
    if contract.is_duplicate(cast, executed_params):
        return DUPLICATE, "exact duplicate of an already-executed config", cast
    return None, None, cast


def _dependency_versions() -> dict:
    """python/numpy always; the optimizer libraries only if already imported.

    NEVER imports them here — an arm that does not use HEBO must not pay for
    loading it, and importing a library the cell never used would misreport the
    environment. This is sampled at the END of a cell (into result.json), after
    the arm's lazy imports have happened; the manifest is written before the
    arm runs and would record none of them.
    """
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("optuna", "torch", "scipy", "sklearn", "gpytorch", "hebo", "pymoo", "GPy"):
        module = sys.modules.get(name)
        if module is not None:
            versions[name] = getattr(module, "__version__", "unknown")
    return versions


def _hardware_info() -> dict:
    """Best-effort, stdlib only (no GPU queries)."""
    return {"platform": platform.platform(), "machine": platform.machine()}
