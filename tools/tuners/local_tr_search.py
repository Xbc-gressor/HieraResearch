#!/usr/bin/env python3
"""Local trust-region FIRST bout (FIRST regime of inner policies
localtr8-hebo10-spsa10-v1 and localtr8-hebo10-hebo10-v1).

A FIRST bout is 8 objective evaluations. Deferred warm configs from
step 0+1 occupy slots inside those 8 (design §2 rule 4); remaining
slots are the inner-benchmark ``local_tr`` arm (PLAN §6.2) ported onto
the production Phase-C stage machinery:

- radius starts at 0.25 per bout (cold; nothing inherited across
  bouts) and is clamped to [0.025, 1.0];
- every proposal is a local draw around the current incumbent in the
  codec's normalized z-space;
- strict improvement grows the radius by 1.5x, anything else
  (including a crash) shrinks it by 0.5x; a preflight rejection is
  not an evaluation and leaves the radius untouched.

An interrupted bout resumes by counting this stage's already-charged
objective spends, skipping already-attempted deferred configs, and
reinjecting the last persisted radius. Phase-R rewarm proposals are
not part of this protocol.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inner_benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import (  # noqa: E402
    DEFAULT_SCORE_FN,
    EvaluationBudgetExhausted,
    append_preflight_attempt,
    append_trial,
    attempted_config_identities,
    cast_params_to_search_space,
    deep_tune_stage_elapsed,
    deep_tune_time_budget,
    is_config_infeasible_error,
    is_finite_score,
    params_identity,
    project_configs_into_space,
    read_deferred_configs,
    read_prior_trials,
    read_runtime_limit,
    read_tune_report,
    search_space_for_json,
    set_stage_meta,
    timed_eval,
    timed_preflight,
    write_json,
)
from failure_artifacts import record_failure  # noqa: E402
from tune_tools import (  # noqa: E402
    _bounds_violations,
    _read_literal_mapping,
)
from validate_tasks import ROOT, parse_task_toml  # noqa: E402

import arm_api  # noqa: E402
import checkpoint as checkpoint_mod  # noqa: E402
import codec as codec_mod  # noqa: E402
import space as space_mod  # noqa: E402
import state as state_mod  # noqa: E402
from arms.local_tr import ARM, R0, R_MAX, R_MIN, GROW, SHRINK  # noqa: E402


METHOD = "local_tr"
MAX_CONSECUTIVE_PREFLIGHT_REJECTS = 5


def _current_local_tr_stage(report: dict) -> dict | None:
    stages = report.get("phase_c", {}).get("stages", [])
    return next(
        (
            stage
            for stage in reversed(stages)
            if isinstance(stage, dict) and stage.get("method") == METHOD
        ),
        None,
    )


def _task_name(candidate_path: Path) -> str | None:
    parts = Path(candidate_path).resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "runs" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _task_section(candidate_path: Path, section: str) -> dict:
    name = _task_name(candidate_path)
    if name is None:
        return {}
    task_toml = ROOT / "tasks" / name / "task.toml"
    if not task_toml.is_file():
        return {}
    data = parse_task_toml(task_toml).get(section, {})
    return data if isinstance(data, dict) else {}


def _python_cmd(candidate_path: Path) -> list[str] | None:
    project = _task_section(candidate_path, "env").get("project")
    if not isinstance(project, str) or not project:
        return None
    return ["uv", "--project", str(ROOT / project), "run", "python"]


def _configured_score_fn(candidate_path: Path) -> str:
    name = _task_section(candidate_path, "evaluation").get("score_fn")
    return name if isinstance(name, str) and name else DEFAULT_SCORE_FN


def _configured_preflight_fn(candidate_path: Path) -> str | None:
    name = _task_section(candidate_path, "evaluation").get("preflight_fn")
    return name if isinstance(name, str) and name else None


def _best_so_far(report_path: Path, search_space: dict) -> tuple[dict, float] | None:
    best: tuple[dict, float] | None = None
    for trial in read_prior_trials(report_path):
        params, score = trial.get("params"), trial.get("score")
        if not isinstance(params, dict) or set(params) != set(search_space):
            continue
        if not is_finite_score(score):
            continue
        try:
            normalized = cast_params_to_search_space(dict(params), search_space)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        value = float(score)
        if best is None or value < best[1]:
            best = (normalized, value)
    return best


def _row_from_trial(trial: dict, origin: str, contract) -> checkpoint_mod.HistoryRow | None:
    params = trial.get("params")
    if not isinstance(params, dict):
        return None
    try:
        contract.params_identity(params)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    score = trial.get("score")
    if is_finite_score(score):
        status, value = "ok", float(score)
    elif trial.get("status") in ("failed", "crash"):
        status, value = "crash", None
    else:
        return None
    role = trial.get("role")
    return checkpoint_mod.HistoryRow(
        params=dict(params),
        score=value,
        status=status,
        origin=origin,
        role=role if isinstance(role, str) else None,
    )


def _history_rows(report: dict, incumbent_identity: str, contract) -> list:
    rows = []
    seen: set[str] = set()
    for trial in report.get("phase_a", {}).get("warm_start_configs", []):
        if not isinstance(trial, dict):
            continue
        row = _row_from_trial(trial, "phase_a", contract)
        if row is None:
            continue
        identity = contract.params_identity(row.params)
        if identity == incumbent_identity or identity in seen:
            continue
        rows.append(row)
        seen.add(identity)
    for stage in report.get("phase_c", {}).get("stages", []):
        if not isinstance(stage, dict):
            continue
        origin = f"bout_{stage.get('bout_index', 0)}"
        for trial in stage.get("trials", []):
            if not isinstance(trial, dict):
                continue
            row = _row_from_trial(trial, origin, contract)
            if row is None:
                continue
            identity = contract.params_identity(row.params)
            if identity == incumbent_identity or identity in seen:
                continue
            rows.append(row)
            seen.add(identity)
    return rows


def _stage_spent(stage: dict) -> int:
    spent = 0
    for row in stage.get("trials", []) if isinstance(stage, dict) else []:
        if not isinstance(row, dict):
            continue
        if is_finite_score(row.get("score")) or row.get("status") == "failed":
            spent += 1
    return spent


def _persisted_radius(stage: dict) -> float:
    state = stage.get("local_tr_state") if isinstance(stage, dict) else None
    if not isinstance(state, dict):
        return R0
    try:
        radius = float(state.get("tr_radius", R0))
    except (TypeError, ValueError):
        return R0
    return min(R_MAX, max(R_MIN, radius))


def _build_checkpoint(
    *,
    candidate_path: Path,
    report: dict,
    incumbent: tuple[dict, float],
    contract,
    remaining: int,
) -> checkpoint_mod.Checkpoint:
    incumbent_params, incumbent_score = incumbent
    incumbent_identity = contract.params_identity(incumbent_params)
    return checkpoint_mod.Checkpoint(
        checkpoint_id=candidate_path.parent.name,
        regime="first",
        stratum="first",
        source={"candidate_id": candidate_path.parent.name, "kind": "unknown"},
        checkpoint_dir=candidate_path.parent,
        candidate_relpath=".",
        candidate_path=candidate_path,
        task=checkpoint_mod.TaskSpec(
            score_fn=_configured_score_fn(candidate_path),
            preflight_fn=_configured_preflight_fn(candidate_path) or "preflight_config",
            per_runtime_limit=read_runtime_limit(candidate_path),
            project=_task_section(candidate_path, "env").get("project"),
        ),
        incumbent=checkpoint_mod.Incumbent(
            params=dict(incumbent_params), score=float(incumbent_score)
        ),
        incumbent_is_inherited_control=False,
        history=tuple(_history_rows(report, incumbent_identity, contract)),
        extra={"remaining_budget": remaining},
    )


def _build_cell_state(frozen, contract) -> state_mod.CellState:
    history_trials = [
        state_mod.Trial(
            config=dict(row.params),
            score=row.score,
            status=row.status,
            source=row.origin,
        )
        for row in frozen.history
    ]
    incumbent_identity = contract.params_identity(frozen.incumbent.params)
    if not any(
        contract.params_identity(trial.config) == incumbent_identity
        for trial in history_trials
    ):
        history_trials.insert(
            0,
            state_mod.Trial(
                config=dict(frozen.incumbent.params),
                score=frozen.incumbent.score,
                status="ok",
                source="incumbent",
            ),
        )
    return state_mod.CellState(
        incumbent_config=dict(frozen.incumbent.params),
        incumbent_score=frozen.incumbent.score,
        budget_remaining=frozen.extra["remaining_budget"],
        trials=history_trials,
        identity_fn=contract.params_identity,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--n-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    time_budget = deep_tune_time_budget(
        args.candidate_path,
        args.tune_report_json,
        METHOD,
    )
    bout_index = time_budget["bout_index"]
    search_space = _read_literal_mapping(args.candidate_path, "SEARCH_SPACE")
    base_params = _read_literal_mapping(args.candidate_path, "BASE_PARAMS")
    if not isinstance(base_params, dict):
        raise ValueError("local_tr requires BASE_PARAMS (the applied incumbent)")

    contract = space_mod.read_contract(args.candidate_path)
    incumbent = _best_so_far(args.tune_report_json, search_space)
    if incumbent is None:
        incumbent = (
            cast_params_to_search_space(dict(base_params), search_space),
            math.inf,
        )

    report = read_tune_report(args.tune_report_json)
    stage = _current_local_tr_stage(report) or {}
    prior_attempts = _stage_spent(stage)
    remaining = max(0, int(args.n_evals) - prior_attempts)
    if remaining <= 0:
        completed = sum(
            1
            for row in stage.get("trials", [])
            if isinstance(row, dict) and is_finite_score(row.get("score"))
        )
        set_stage_meta(
            args.tune_report_json,
            METHOD,
            bout_index=bout_index,
            status="ok",
            elapsed_seconds=deep_tune_stage_elapsed(time_budget),
            trials_attempted=prior_attempts,
            trials_completed=completed,
        )
        write_json(
            {
                "method": METHOD,
                "status": "ok",
                "reason": "bout already spent",
                "trials_attempted": prior_attempts,
                "trials_completed": completed,
            }
        )
        return 0

    frozen = _build_checkpoint(
        candidate_path=args.candidate_path,
        report=report,
        incumbent=incumbent,
        contract=contract,
        remaining=remaining,
    )
    codec = codec_mod.Codec(contract)
    cell_state = _build_cell_state(frozen, contract)
    python_cmd = _python_cmd(args.candidate_path)
    radius = _persisted_radius(stage)

    counters = {
        "objective_attempts": 0,
        "objective_completed": 0,
        "preflight_rejections": 0,
        "budget_exhausted": False,
        "budget_exhausted_scope": None,
        "consecutive_rejects": 0,
        "deferred_evaluated": 0,
        "deferred_skipped_outside_space": 0,
        "deferred_projected_into_space": 0,
        "deferred_skipped_already_seen": 0,
    }
    failure_refs: list[str] = []
    last_arm_state: dict = {"tr_radius": radius}
    pending_arm_extras: dict = {}
    close_status = "ok"
    close_reason = None
    preflight_enabled = _configured_preflight_fn(args.candidate_path) is not None

    def persist_arm_state() -> None:
        payload = {**last_arm_state, **pending_arm_extras, "tr_radius": radius}
        set_stage_meta(
            args.tune_report_json,
            METHOD,
            bout_index=bout_index,
            local_tr_state=payload,
        )

    def emit(values: dict) -> None:
        pending_arm_extras.update(values)

    def update_radius(feedback: arm_api.Feedback, incumbent_before: float) -> None:
        nonlocal radius
        if feedback.kind != "outcome":
            return
        if (
            feedback.status == "ok"
            and feedback.score is not None
            and feedback.score < incumbent_before
        ):
            radius = min(R_MAX, GROW * radius)
        else:
            radius = max(R_MIN, SHRINK * radius)

    def run_proposal(cast: dict, proposal: arm_api.Proposal) -> arm_api.Feedback:
        if preflight_enabled:
            try:
                preflight_result = timed_preflight(
                    cast,
                    args.candidate_path,
                    expected_execution_revision=time_budget[
                        "candidate_execution_revision"
                    ],
                    python_cmd=python_cmd,
                )
            except Exception as exc:
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="preflight",
                    method=METHOD,
                    params=cast,
                    error=exc,
                    traceback_text=traceback.format_exc(),
                )
                append_preflight_attempt(
                    args.tune_report_json,
                    source=METHOD,
                    params=cast,
                    status="failed",
                    failure=failure,
                )
                append_trial(
                    args.tune_report_json,
                    METHOD,
                    {
                        "params": cast,
                        "score": None,
                        "status": "preflight_rejected",
                        "source": proposal.source,
                        **failure,
                    },
                )
                counters["preflight_rejections"] += 1
                if failure["failure_ref"] not in failure_refs:
                    failure_refs.append(failure["failure_ref"])
                cell_state.record_outcome(
                    cast,
                    status=state_mod.PREFLIGHT_REJECTED,
                    source=proposal.source,
                    rationale=proposal.rationale,
                )
                return arm_api.Feedback.preflight_rejected(
                    stage="task", reason=str(exc)
                )
            append_preflight_attempt(
                args.tune_report_json,
                source=METHOD,
                params=cast,
                status="ok",
                result=preflight_result or {"status": "ok"},
            )
        try:
            score = timed_eval(
                None,
                None,
                cast,
                args.candidate_path,
                phase="phase_c",
                method=METHOD,
                expected_execution_revision=time_budget[
                    "candidate_execution_revision"
                ],
                python_cmd=python_cmd,
            )
        except EvaluationBudgetExhausted:
            raise
        except Exception as exc:
            counters["objective_attempts"] += 1
            failure = record_failure(
                report_path=args.tune_report_json,
                candidate_path=args.candidate_path,
                phase="phase_c",
                method=METHOD,
                params=cast,
                error=exc,
                traceback_text=traceback.format_exc(),
            )
            if failure["failure_ref"] not in failure_refs:
                failure_refs.append(failure["failure_ref"])
            config_infeasible = is_config_infeasible_error(exc)
            append_trial(
                args.tune_report_json,
                METHOD,
                {
                    "params": cast,
                    "score": None,
                    "status": "failed",
                    "config_infeasible": config_infeasible,
                    "source": proposal.source,
                    **failure,
                },
            )
            cell_state.record_outcome(
                cast, status=state_mod.CRASH, source=proposal.source
            )
            if not config_infeasible:
                raise
            return arm_api.Feedback.outcome(
                status="crash", score=None, detail=str(exc), executed_params=cast
            )
        counters["objective_attempts"] += 1
        counters["objective_completed"] += 1
        append_trial(
            args.tune_report_json,
            METHOD,
            {
                "params": cast,
                "score": score,
                "source": proposal.source,
                "rationale": proposal.rationale,
                "tr_radius": radius,
            },
        )
        cell_state.record_outcome(
            cast,
            status=state_mod.OK,
            score=float(score),
            source=proposal.source,
            rationale=proposal.rationale,
        )
        return arm_api.Feedback.outcome(
            status="ok", score=float(score), executed_params=cast
        )

    persist_arm_state()
    gen = None
    try:
        deferred_in_space, n_deferred_projected, deferred_dropped = (
            project_configs_into_space(
                read_deferred_configs(args.tune_report_json), search_space
            )
        )
        counters["deferred_skipped_outside_space"] = len(deferred_dropped)
        counters["deferred_projected_into_space"] = n_deferred_projected
        seen = attempted_config_identities(args.tune_report_json, search_space)
        for raw in deferred_in_space:
            if cell_state.budget_remaining <= 0:
                break
            try:
                cast = contract.cast(raw)
            except (TypeError, ValueError, ArithmeticError):
                counters["deferred_skipped_already_seen"] += 1
                continue
            identity = params_identity(cast)
            if identity in seen:
                counters["deferred_skipped_already_seen"] += 1
                continue
            proposal = arm_api.Proposal(params=cast, source="deferred")
            feedback = run_proposal(cast, proposal)
            seen.add(identity)
            if feedback.kind == "outcome":
                counters["deferred_evaluated"] += 1
            persist_arm_state()

        if cell_state.budget_remaining > 0 and not contract.varying_dimensions:
            if counters["objective_completed"] == 0:
                close_status = "rejected"
                close_reason = "local_tr: checkpoint has no varying dimension to move"
            else:
                close_reason = (
                    "deferred configs consumed the usable FIRST slots; "
                    "no varying dimension remains for local_tr"
                )
        elif cell_state.budget_remaining > 0:
            ctx = arm_api.CellContext(
                contract=contract,
                codec=codec,
                state=cell_state,
                checkpoint=frozen,
                rng=random.Random(args.seed),
                np_rng=np.random.default_rng(args.seed),
                budget=cell_state.budget_remaining,
                seed=int(args.seed),
                extras={"tr_radius": radius},
                emit=emit,
            )
            gen = ARM.run(ctx)
            feedback = None
            while cell_state.budget_remaining > 0:
                proposal = next(gen) if feedback is None else gen.send(feedback)
                if not isinstance(proposal, arm_api.Proposal):
                    raise arm_api.ArmError(
                        f"local_tr arm yielded a non-Proposal: {type(proposal).__name__}"
                    )
                last_arm_state = dict(proposal.arm_state or {})
                persist_arm_state()
                det_code = None
                try:
                    cast = contract.cast(proposal.params)
                except (TypeError, ValueError, ArithmeticError) as exc:
                    det_code = f"schema_invalid: {exc}"
                    cast = None
                else:
                    violations = _bounds_violations(cast, contract.search_space)
                    if violations:
                        det_code = "out_of_space: " + "; ".join(
                            f"{item['key']}: {item['reason']}" for item in violations
                        )
                    elif contract.is_duplicate(
                        cast,
                        [
                            trial.config
                            for trial in cell_state.trials
                            if trial.status in ("ok", "crash")
                        ],
                    ):
                        det_code = (
                            "duplicate: exact duplicate of an already-executed config"
                        )
                if det_code is not None:
                    counters["preflight_rejections"] += 1
                    counters["consecutive_rejects"] += 1
                    rejected = cast if cast is not None else proposal.params
                    if isinstance(rejected, dict):
                        append_trial(
                            args.tune_report_json,
                            METHOD,
                            {
                                "params": rejected,
                                "score": None,
                                "status": "preflight_rejected",
                                "source": proposal.source,
                                "reason": det_code,
                            },
                        )
                        cell_state.record_outcome(
                            rejected,
                            status=state_mod.PREFLIGHT_REJECTED,
                            source=proposal.source,
                            rationale=proposal.rationale,
                        )
                    if counters["consecutive_rejects"] >= MAX_CONSECUTIVE_PREFLIGHT_REJECTS:
                        raise arm_api.ArmError(
                            f"{MAX_CONSECUTIVE_PREFLIGHT_REJECTS} consecutive "
                            f"preflight rejections (last: {det_code})"
                        )
                    feedback = arm_api.Feedback.preflight_rejected(
                        stage="deterministic", reason=det_code
                    )
                    continue
                incumbent_before = cell_state.incumbent_score
                feedback = run_proposal(cast, proposal)
                if feedback.kind == "preflight_rejected":
                    counters["consecutive_rejects"] += 1
                    if counters["consecutive_rejects"] >= MAX_CONSECUTIVE_PREFLIGHT_REJECTS:
                        raise arm_api.ArmError(
                            f"{MAX_CONSECUTIVE_PREFLIGHT_REJECTS} consecutive "
                            f"preflight rejections (last: {feedback.reason})"
                        )
                else:
                    counters["consecutive_rejects"] = 0
                    update_radius(feedback, incumbent_before)
                persist_arm_state()
    except EvaluationBudgetExhausted as exc:
        counters["budget_exhausted"] = True
        counters["budget_exhausted_scope"] = exc.scope
    except arm_api.Unsupported as exc:
        close_status = "rejected"
        close_reason = str(exc)
    except arm_api.ArmError as exc:
        close_status = "failed"
        close_reason = str(exc)
    except StopIteration:
        if cell_state.budget_remaining > 0:
            close_status = "failed"
            close_reason = (
                f"local_tr arm exhausted early with "
                f"budget_remaining={cell_state.budget_remaining}"
            )
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:
                if close_status == "ok":
                    close_status = "failed"
                    close_reason = "local_tr arm raised during generator close"

    persist_arm_state()
    stage_elapsed = deep_tune_stage_elapsed(time_budget)
    if counters["budget_exhausted"] and counters["objective_completed"] == 0:
        set_stage_meta(
            args.tune_report_json,
            METHOD,
            bout_index=bout_index,
            status="budget_exhausted",
            elapsed_seconds=stage_elapsed,
            preflight_rejections=counters["preflight_rejections"],
            budget_exhausted_scope=counters["budget_exhausted_scope"],
        )
        write_json(
            {
                "method": METHOD,
                "status": "budget_exhausted",
                "reason": "evaluation allocation exhausted before score_fn",
                "budget_exhausted_scope": counters["budget_exhausted_scope"],
                "trials_completed": 0,
                "trials_attempted": counters["objective_attempts"],
                "preflight_rejections": counters["preflight_rejections"],
                "elapsed_seconds": round(stage_elapsed, 1),
            }
        )
        return 0

    if close_status == "ok" and counters["objective_completed"] == 0:
        close_status = "failed"
        close_reason = "local_tr produced no completed trial"

    totals = pending_arm_extras
    receipt = {
        "method": METHOD,
        "status": close_status,
        "trials_completed": counters["objective_completed"],
        "trials_attempted": counters["objective_attempts"],
        "preflight_rejections": counters["preflight_rejections"],
        "budget_exhausted": counters["budget_exhausted"],
        "deferred_evaluated": counters["deferred_evaluated"],
        "deferred_skipped_outside_space": counters["deferred_skipped_outside_space"],
        "deferred_projected_into_space": counters["deferred_projected_into_space"],
        "deferred_skipped_already_seen": counters["deferred_skipped_already_seen"],
        "tr_radius": radius,
        "internal_duplicate_count": totals.get("internal_duplicate_count", 0),
        "internal_resample_count": totals.get("internal_resample_count", 0),
        "failure_refs": failure_refs[-3:],
        "elapsed_seconds": round(stage_elapsed, 1),
        "search_space": search_space_for_json(search_space),
    }
    if close_reason is not None:
        receipt["reason"] = close_reason
    best = _best_so_far(args.tune_report_json, search_space)
    if best is not None:
        receipt["best_params"] = best[0]
        receipt["best_score"] = best[1]
    set_stage_meta(
        args.tune_report_json,
        METHOD,
        bout_index=bout_index,
        status=close_status,
        elapsed_seconds=stage_elapsed,
        preflight_rejections=counters["preflight_rejections"],
        budget_exhausted=counters["budget_exhausted"],
        budget_exhausted_scope=counters["budget_exhausted_scope"],
        time_limit_seconds=time_budget["limit_seconds"],
        deferred_evaluated=counters["deferred_evaluated"],
    )
    write_json(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
