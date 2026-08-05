"""Deterministic run-level state machine for HieraResearch experiments."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, CoordinatorStore, atomic_write_json
from .background import BackgroundBuilder
from .candidate import CandidateBuildError, CandidatePipeline
from .experience import ExperienceRefresh
from .llm import InferenceError, ModelGateway
from .models import (
    ActiveRound,
    CoordinatorPhase,
    CoordinatorState,
    RoundAction,
    RunIdentity,
    Transition,
)
from .process import ProcessError, ProcessInterrupted
from .semantic import IdeationContractError, SemanticAdmission
from .state_machine import next_transition
from .toolchain import ToolFailure, Toolchain, read_task_config
from .tuning import DeepTuner
from .upstream import (
    UpstreamBackoffPolicy,
    decide_upstream_recovery,
    is_retryable_upstream_failure,
    upstream_fields_reset,
)


@dataclass(frozen=True)
class RunControls:
    dimension_strategy: str | None = None
    llm_intelligence_score: float | None = None
    max_evaluations: int | None = None
    per_runtime_limit: float | None = None
    sync_environment: bool = True
    prepare_task: bool = True
    preflight_only: bool = False
    resume_blocked: bool = False
    # Run-level upstream recovery (502/503/...). None = library defaults.
    upstream_max_streak: int | None = None
    upstream_max_backoff_total_seconds: float | None = None
    upstream_base_backoff_seconds: float | None = None
    upstream_max_single_backoff_seconds: float | None = None


# Consecutive model-side empty rounds (every action dropped after ideation
# contract failures, no objective progress) the run tolerates before blocking.
# One retry separates a stochastic ideation blip from a provider that cannot
# currently produce a usable idea; a genuinely action-free SELECT is not
# counted here and still blocks immediately.
NO_PROGRESS_ROUND_LIMIT = 2


class ExperimentCoordinator:
    def __init__(
        self,
        identity: RunIdentity,
        toolchain: Toolchain,
        models: ModelGateway,
        controls: RunControls,
        *,
        sleep: Callable[[float], None] | None = None,
    ):
        self.identity = identity
        self.toolchain = toolchain
        self.models = models
        self.controls = controls
        self.store = CoordinatorStore(identity.run_dir)
        self.state: CoordinatorState | None = None
        self.task_config: dict[str, Any] | None = None
        self.semantic: SemanticAdmission | None = None
        self.candidates: CandidatePipeline | None = None
        self.experience: ExperienceRefresh | None = None
        self.deep_tuner: DeepTuner | None = None
        self._sleep = sleep or time.sleep
        self._upstream_policy = self._policy_from_controls(controls)

    @staticmethod
    def _policy_from_controls(controls: RunControls) -> UpstreamBackoffPolicy:
        kwargs: dict[str, Any] = {}
        if controls.upstream_max_streak is not None:
            kwargs["max_streak"] = controls.upstream_max_streak
        if controls.upstream_max_backoff_total_seconds is not None:
            kwargs["max_backoff_total_seconds"] = (
                controls.upstream_max_backoff_total_seconds
            )
        if controls.upstream_base_backoff_seconds is not None:
            kwargs["base_backoff_seconds"] = controls.upstream_base_backoff_seconds
        if controls.upstream_max_single_backoff_seconds is not None:
            kwargs["max_single_backoff_seconds"] = (
                controls.upstream_max_single_backoff_seconds
            )
        return UpstreamBackoffPolicy(**kwargs)

    def run(self, *, max_transitions: int | None = None) -> dict[str, Any]:
        if max_transitions is not None and max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        try:
            return self._run(max_transitions=max_transitions)
        except (ProcessInterrupted, KeyboardInterrupt) as exc:
            self._block(f"interrupted: {exc}")
            return self.status()
        except InferenceError as exc:
            # Retryable upstream should have been absorbed in the transition
            # loop; reaching here means exhaustion escaped or a pre-loop path
            # failed after recovery declined.
            if is_retryable_upstream_failure(exc):
                self._block(f"upstream_unrecoverable: {type(exc).__name__}: {exc}")
            else:
                self._block(f"{type(exc).__name__}: {exc}")
            return self.status()
        except (
            ArtifactError,
            CandidateBuildError,
            ProcessError,
            ToolFailure,
            ValueError,
        ) as exc:
            self._block(f"{type(exc).__name__}: {exc}")
            return self.status()
        except Exception as exc:
            self._block(f"unexpected {type(exc).__name__}: {exc}")
            raise

    def _run(self, *, max_transitions: int | None) -> dict[str, Any]:
        run_existed = self.identity.run_dir.exists()
        self.toolchain.initialize_run(
            self.identity.task_name,
            self.identity.tag,
            dimension_strategy=self.controls.dimension_strategy,
            llm_intelligence_score=self.controls.llm_intelligence_score,
            max_evaluations=self.controls.max_evaluations,
            per_runtime_limit=self.controls.per_runtime_limit,
        )
        self.task_config = read_task_config(
            self.identity.repo_root, self.identity.task_name
        )
        self.state = self.store.load(self.identity.task_name, self.identity.tag)
        self._resume_state_if_authorized()
        if self.state.phase is CoordinatorPhase.BLOCKED:
            return self.status()

        if self.controls.sync_environment:
            self._mark(Transition.INITIALIZE)
            self.toolchain.sync_environment(self.task_config)
        preflight_receipt = self.identity.run_dir / "environment_preflight.json"
        if (
            self.controls.prepare_task
            and not run_existed
            and not preflight_receipt.exists()
        ):
            self.toolchain.prepare_task(self.task_config)
        self._mark(Transition.ENVIRONMENT_PREFLIGHT)
        self.toolchain.environment_preflight(
            self.identity.task_name,
            self.identity.run_dir,
            self.task_config,
        )
        if self.controls.preflight_only:
            return self.status()

        self._build_services()
        self._mark(Transition.BUILD_BACKGROUND)
        assert self.background is not None
        while self.state.phase is CoordinatorPhase.RUNNING:
            try:
                completed_before = self._completed_invocation_count()
                self.background.ensure()
                # Reset only when a durable model invocation completed — a
                # no-op ensure() on already-valid artifacts must not wipe the
                # upstream budget after a crash mid-backoff.
                self._maybe_clear_upstream_after_model_success(completed_before)
                break
            except InferenceError as exc:
                if not self._recover_from_upstream(exc):
                    raise
                # Exhausted upstream budget → phase is BLOCKED; leave loop.
        if self.state.phase is not CoordinatorPhase.RUNNING:
            return self.status()
        self._reconcile_state()

        transitions = 0
        while self.state.phase is CoordinatorPhase.RUNNING:
            if max_transitions is not None and transitions >= max_transitions:
                break
            self._close_finished_round()
            brief = (
                self.toolchain.ledger_brief(self.identity.run_dir)
                if self.identity.ledger_path.exists()
                else None
            )
            transition = next_transition(
                self.state,
                ledger_exists=self.identity.ledger_path.exists(),
                ledger_brief=brief,
                has_provided_baseline=self._has_provided_baseline(),
                experience_refresh_pending=(
                    self.experience.has_pending(brief)
                    if self.experience is not None and brief is not None
                    else False
                ),
            )
            if transition is Transition.STOP:
                if brief and brief.get("phase") == "blocked":
                    self.state.phase = CoordinatorPhase.BLOCKED
                    self.state.stop_condition = str(
                        brief.get("active_stop_condition") or "ledger blocked"
                    )
                    self.store.save(self.state)
                break
            self._mark(transition)
            completed_before = self._completed_invocation_count()
            try:
                self._execute(transition, brief)
            except InferenceError as exc:
                if not self._recover_from_upstream(exc):
                    raise
                # Failed attempt does not consume max_transitions; after
                # backoff the same transition is re-derived from artifacts.
                # If recovery blocked the run, the while-guard exits.
                continue
            self._maybe_clear_upstream_after_model_success(completed_before)
            transitions += 1
        return self.status()

    @property
    def background(self) -> BackgroundBuilder | None:
        if self.task_config is None:
            return None
        return BackgroundBuilder(
            self.identity, self.toolchain, self.models, self.task_config
        )

    def _build_services(self) -> None:
        assert self.task_config is not None
        self.semantic = SemanticAdmission(self.identity, self.toolchain, self.models)
        self.candidates = CandidatePipeline(
            self.identity, self.toolchain, self.models, self.task_config
        )
        self.experience = ExperienceRefresh(
            self.identity, self.toolchain, self.models
        )
        self.deep_tuner = DeepTuner(
            self.identity, self.toolchain, self.task_config
        )

    def _execute(
        self,
        transition: Transition,
        brief: dict[str, Any] | None,
    ) -> None:
        if transition is Transition.ADMIT_BASELINE:
            self._admit_baseline()
        elif transition is Transition.ADMIT_ROUND:
            self._admit_round(brief)
        elif transition is Transition.MATERIALIZE_CANDIDATE:
            self._materialize_and_implement()
        elif transition is Transition.BUILD_TUNING_CONTRACT:
            self._build_candidate_contract()
        elif transition is Transition.PREFLIGHT_CANDIDATE:
            self._preflight_candidate()
        elif transition is Transition.EVALUATE_WARM_CONFIGS:
            self._evaluate_candidate()
        elif transition is Transition.DEEP_TUNE:
            self._deep_tune()
        elif transition is Transition.REFRESH_EXPERIENCE:
            if brief is None:
                raise ValueError("experience refresh requires a ledger brief")
            assert self.experience is not None
            outcome = self.experience.run(brief)
            if outcome.get("status") == "failed":
                # Model-quality failure at the refresh boundary: semantic
                # admission is ledger-gated on a processed experience delta,
                # so the truthful terminal state is an explicit block with the
                # recorded reason. A resume admits one fresh attempt.
                error = outcome.get("error") or {}
                self._block(
                    "experience_refresh_failed: "
                    f"dag-{outcome.get('dag_revision')}: "
                    f"{error.get('type', 'unknown')}: {error.get('message', '')}"
                )
        elif transition is Transition.COMPLETE:
            self.toolchain.set_phase(self.identity.run_dir, "completed")
            self.state.phase = CoordinatorPhase.COMPLETED
            self.state.stop_condition = "evaluation_budget_reached"
            self.store.save(self.state)
        else:
            raise ValueError(f"unsupported coordinator transition: {transition.value}")

    def _admit_baseline(self) -> None:
        if self.state.active_round is None:
            self.state.active_round = ActiveRound(
                round_id=self.state.next_round_id,
                actions=[RoundAction(op="fresh", run_id="000")],
                evaluations_before=0,
            )
            self.store.save(self.state)
        action = self.state.active_round.actions[0]
        if action.admitted:
            self.state.active_round.admission_complete = True
            self.store.save(self.state)
            return
        assert self.semantic is not None
        self.semantic.admit(
            run_id="000", op="fresh", parents=[], baseline_only=True
        )
        action.admitted = True
        self.state.active_round.admission_complete = True
        self.store.save(self.state)

    def _admit_round(self, brief: dict[str, Any] | None) -> None:
        if self.state.active_round is None:
            preflight = self.toolchain.background_preflight(self.identity.run_dir)
            if preflight.get("action") != "none":
                raise ValueError(f"background preflight rejected admission: {preflight}")
            decision = self.toolchain.got_decide(self.identity.run_dir)
            raw_actions = decision.get("actions")
            if not isinstance(raw_actions, list):
                raise ValueError("got_select actions must be a list")
            actions = [RoundAction.from_dict(value) for value in raw_actions]
            self.state.active_round = ActiveRound(
                round_id=self.state.next_round_id,
                actions=actions,
                evaluations_before=int((brief or {}).get("evaluations_attempted", 0)),
            )
            self.store.save(self.state)
        active = self.state.active_round
        assert active is not None
        assert self.semantic is not None
        dropped: list[RoundAction] = []
        reserved: set[str] = {
            action.run_id for action in active.actions if action.run_id is not None
        }
        for action in active.actions:
            if action.admitted:
                continue
            if action.run_id is None:
                current = (
                    self.toolchain.ledger_brief(self.identity.run_dir)
                    if self.identity.ledger_path.exists()
                    else {"next_run_id": "000"}
                )
                run_id = current.get("next_run_id")
                if not isinstance(run_id, str) or not run_id.isdigit():
                    raise ValueError(f"invalid next run id: {run_id!r}")
                # A dropped admission never reaches `admit_candidate`, so the
                # ledger's `next_run_id` does not advance past it. Without this
                # guard the next action in the same round would take the same
                # id, overwrite the first one's audit receipt, and — when the
                # inputs happen to match — replay its journaled inference.
                while run_id in reserved:
                    run_id = f"{int(run_id) + 1:0{len(run_id)}d}"
                action.run_id = run_id
                self.store.save(self.state)
            reserved.add(action.run_id)
            existing = (
                self.toolchain.ledger_record(self.identity.run_dir, action.run_id)
                if self.identity.ledger_path.exists()
                else None
            )
            if existing is None:
                try:
                    self.semantic.admit(
                        run_id=action.run_id,
                        op=action.op,
                        parents=action.parents,
                    )
                except IdeationContractError as exc:
                    if self._drop_failed_admission(action, exc):
                        dropped.append(action)
                        continue
                    raise
            else:
                self._verify_admitted_record(action, existing)
            action.admitted = True
            self.store.save(self.state)
        for action in dropped:
            active.actions.remove(action)
        active.admission_complete = True
        self.store.save(self.state)

    def _drop_failed_admission(
        self,
        action: RoundAction,
        exc: "IdeationContractError",
    ) -> bool:
        """Fail one admission without parking the run, when policy allows it.

        Ideation runs before the candidate has a ledger record, so the
        candidate-crash path that later stages degrade through does not exist
        yet: ``record-run`` requires ``add-record`` first. The durable receipt
        below is what keeps the drop auditable instead of silent. Returns False
        when the configured policy is ``block_run``, leaving the caller to
        propagate.
        """
        assert self.semantic is not None
        if self.semantic.ideation_failure_mode() != "drop_action":
            return False
        directory = (
            self.identity.run_dir / ".orchestrator" / "admission_failures"
        )
        round_id = self.state.active_round.round_id
        # The op/parents are part of the name because a dropped id can recur in
        # a later round: the ledger never advanced past it.
        stem = f"round-{round_id}-{action.run_id}"
        path = directory / f"{stem}.json"
        if path.exists():
            existing = sorted(directory.glob(f"{stem}-*.json"))
            path = directory / f"{stem}-{len(existing) + 2}.json"
        atomic_write_json(
            path,
            {
                "schema_version": 1,
                "kind": "ideation_failure",
                "run_id": action.run_id,
                "op": action.op,
                "parents": action.parents,
                "outcome": "action_dropped",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return True

    def _ideation_drops(self, round_id: int) -> int:
        """Count one round's validated ideation-failure drop receipts.

        The receipts written by ``_drop_failed_admission`` are the authority;
        deriving the count at the decision point keeps restarts and upgrades
        consistent without a second, driftable record in ``state.json``.
        Authority comes from the validated content, not the filename: a
        corrupt or contradictory receipt is artifact corruption and raises,
        which the run loop blocks on, rather than reclassifying the round.
        """
        directory = self.identity.run_dir / ".orchestrator" / "admission_failures"
        if not directory.is_dir():
            return 0
        drops = 0
        for path in sorted(directory.glob(f"round-{round_id}-*.json")):
            self._verify_drop_receipt(path)
            drops += 1
        return drops

    @staticmethod
    def _verify_drop_receipt(path: Path) -> None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactError(
                f"invalid admission-failure receipt {path}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactError(
                f"admission-failure receipt must be an object: {path}"
            )
        if (
            value.get("schema_version") != 1
            or value.get("kind") != "ideation_failure"
            or value.get("outcome") != "action_dropped"
        ):
            raise ArtifactError(
                f"admission-failure receipt contradicts its contract: {path}"
            )
        run_id = value.get("run_id")
        if not isinstance(run_id, str) or not run_id.isdigit():
            raise ArtifactError(
                f"admission-failure receipt has an invalid run_id: {path}"
            )
        if value.get("op") not in {"fresh", "improve", "crossover"}:
            raise ArtifactError(
                f"admission-failure receipt has an invalid op: {path}"
            )
        parents = value.get("parents")
        if not isinstance(parents, list) or not all(
            isinstance(parent, str) for parent in parents
        ):
            raise ArtifactError(
                f"admission-failure receipt has invalid parents: {path}"
            )
        if not isinstance(value.get("error"), str) or not value["error"]:
            raise ArtifactError(
                f"admission-failure receipt lacks its error: {path}"
            )

    def _materialize_and_implement(self) -> None:
        action = self._next_unresolved_action()
        assert self.candidates is not None
        try:
            if not action.materialized:
                provided = action.run_id == "000" and self._has_provided_baseline()
                self.candidates.materialize(action, provided_baseline=provided)
                action.materialized = True
                self.store.save(self.state)
            if not action.implemented:
                self.candidates.implement(action)
                action.implemented = True
                self.store.save(self.state)
        except CandidateBuildError as exc:
            self._close_candidate_build_failure(action, exc, stage="implementation")

    def _build_candidate_contract(self) -> None:
        action = self._next_unresolved_action()
        assert self.candidates is not None
        try:
            self.candidates.build_contract(action)
            action.contract_ready = True
            self.store.save(self.state)
        except CandidateBuildError as exc:
            self._close_candidate_build_failure(action, exc, stage="tuning_contract")

    def _evaluate_candidate(self) -> None:
        action = self._next_unresolved_action()
        assert self.candidates is not None
        try:
            outcome = self.candidates.evaluate(action)
        except CandidateBuildError as exc:
            self._close_candidate_build_failure(action, exc, stage="evaluation")
            return
        action.resolved = True
        self.store.save(self.state)
        if self.candidates.is_provided_baseline(action.run_id or "") and outcome.status == "crash":
            raise ValueError("provided baseline could not be evaluated")

    def _preflight_candidate(self) -> None:
        action = self._next_unresolved_action()
        assert self.candidates is not None
        try:
            self.candidates.preflight(action)
            action.preflight_ready = True
            self.store.save(self.state)
        except CandidateBuildError as exc:
            self._close_candidate_build_failure(action, exc, stage="preflight")

    def _close_candidate_build_failure(
        self,
        action: RoundAction,
        exc: CandidateBuildError,
        *,
        stage: str,
    ) -> None:
        """Close a candidate build failure and continue the round.

        LLM-authored candidates will fail; that failure belongs to the
        candidate, not the run.  Record the stage-tagged receipt, close the
        ledger record as a crash, and resolve the action so the next
        transition proceeds to the next candidate.  Only the provided
        baseline is run-fatal: the experiment anchors on it.
        """
        assert self.candidates is not None
        self.candidates.record_build_failure(action, exc, stage=stage)
        action.resolved = True
        self.store.save(self.state)
        if self.candidates.is_provided_baseline(action.run_id or ""):
            raise ValueError(
                f"provided baseline candidate failed at {stage}: {exc}"
            ) from exc

    def _deep_tune(self) -> None:
        active = self.state.active_round
        if active is None:
            raise ValueError("deep tuning requires an active round")
        assert self.deep_tuner is not None
        if active.deep_tune_selection is None:
            active.deep_tune_selection = self.deep_tuner.select()
            # Selection is the round's irreversible-effect reservation. Persist
            # it before a worker or finalizer can change objective/ledger state.
            self.store.save(self.state)
        outcome = self.deep_tuner.run(active.deep_tune_selection)
        if (
            outcome.tuned_run_id is not None
            and outcome.tuned_run_id != active.deep_tune_selection.run_id
        ):
            raise ArtifactError("deep-tune outcome changed the reserved candidate")
        active.deep_tune_outcome = outcome
        active.tuning_complete = True
        after = self.toolchain.ledger_brief(self.identity.run_dir)
        no_admissions = not active.actions
        no_objective_progress = (
            int(after.get("evaluations_attempted", 0)) == active.evaluations_before
        )
        stalled = no_admissions and no_objective_progress
        # An empty round has two distinct causes. A SELECT that returned no
        # actions (admission cap exhausted) means progress is genuinely
        # impossible; blocking immediately is correct. A round emptied by
        # dropped ideation failures is a model-side event — the objective
        # budget is untouched and the next round re-derives fresh actions —
        # so it earns a bounded tolerance before the run is parked. The cause
        # is recovered from the durable admission-failure receipts rather than
        # coordinator state, so a restarted (or upgraded) process classifies
        # the round the same way.
        if stalled and not self._ideation_drops(active.round_id):
            self.state.no_progress_cycles = 0
            self.store.save(self.state)
            self._block(
                "no_progress_possible: semantic admission cap returned no actions "
                "and deep tuning admitted no objective call"
            )
            return
        if stalled:
            self.state.no_progress_cycles += 1
        else:
            self.state.no_progress_cycles = 0
        self.store.save(self.state)
        if self.state.no_progress_cycles >= NO_PROGRESS_ROUND_LIMIT:
            self._block(
                "no_progress_possible: "
                f"{self.state.no_progress_cycles} consecutive rounds had every "
                "action dropped after ideation contract failures (see "
                ".orchestrator/admission_failures) and deep tuning admitted no "
                "objective call"
            )

    def _close_finished_round(self) -> None:
        active = self.state.active_round
        if active is None or not active.tuning_complete:
            return
        if any(not action.resolved for action in active.actions):
            raise ArtifactError("cannot close a round with unresolved candidates")
        self.store.complete_round(active)
        self.state.next_round_id = max(self.state.next_round_id, active.round_id + 1)
        self.state.active_round = None
        self.store.save(self.state)

    def _next_unresolved_action(self) -> RoundAction:
        active = self.state.active_round
        if active is None:
            raise ValueError("candidate transition requires an active round")
        for action in active.actions:
            if not action.resolved:
                return action
        raise ValueError("active round has no unresolved candidate")

    def _reconcile_state(self) -> None:
        if self.state is None:
            raise RuntimeError("coordinator state is not loaded")
        if not self.identity.ledger_path.exists():
            self.store.save(self.state)
            return
        brief = self.toolchain.ledger_brief(self.identity.run_dir)
        if (
            self.state.phase is CoordinatorPhase.COMPLETED
            and brief.get("phase") == "running"
        ):
            self.state.phase = CoordinatorPhase.RUNNING
            self.state.stop_condition = None
        active = self.state.active_round
        if active is None:
            pending = brief.get("pending_run_ids", [])
            if pending:
                actions: list[RoundAction] = []
                for run_id in pending:
                    record = self.toolchain.ledger_record(self.identity.run_dir, run_id)
                    if not isinstance(record, dict):
                        raise ArtifactError(f"pending ledger record disappeared: {run_id}")
                    action = RoundAction(
                        op=str(record.get("op")),
                        parents=[str(value) for value in record.get("source_run_ids", [])],
                        run_id=str(run_id),
                        admitted=True,
                    )
                    self._infer_candidate_stages(action, record)
                    actions.append(action)
                self.state.active_round = ActiveRound(
                    round_id=self.state.next_round_id,
                    actions=actions,
                    admission_complete=True,
                    evaluations_before=int(brief.get("evaluations_attempted", 0)),
                )
        else:
            for action in active.actions:
                if action.run_id is None:
                    continue
                record = self.toolchain.ledger_record(
                    self.identity.run_dir, action.run_id
                )
                if isinstance(record, dict):
                    self._verify_admitted_record(action, record)
                    action.admitted = True
                    self._infer_candidate_stages(action, record)
            if all(action.admitted for action in active.actions):
                active.admission_complete = True
        self.store.save(self.state)

    def _infer_candidate_stages(
        self, action: RoundAction, record: dict[str, Any]
    ) -> None:
        if action.run_id is None:
            return
        assert self.candidates is not None
        materialization_ready = self.candidates.materialization_is_ready(
            action, record
        )
        # Receipts gate only pre-evaluation transitions. A durable Phase A
        # report while the ledger record is still pending proves evaluation
        # already started: the warmstart worker rewrites train.py at startup
        # (apply_base_params), so every revision-bound receipt is stale by
        # design. Deriving stage readiness from those receipts would route
        # backwards into materialization/implementation/contract, which raise
        # on the mutated candidate and block the run permanently. Route to
        # EVALUATE_WARM_CONFIGS instead; evaluate() is terminal-first and
        # idempotent and owns forward-completion, crash closure, and debug
        # re-entry from there. evaluation_has_started still raises on a
        # malformed report, so genuine corruption is not masked.
        evaluation_started = self.candidates.evaluation_has_started(action)
        if evaluation_started and record.get("status") == "pending":
            action.materialized = True
            action.implemented = True
            action.contract_ready = True
            action.preflight_ready = True
            action.resolved = False
            return
        action.contract_ready = self.candidates.contract_is_ready(action)
        action.preflight_ready = self.candidates.preflight_is_ready(action)
        action.implemented = (
            action.contract_ready
            or self.candidates.tuning_values_are_ready(action)
            or self.candidates.tuning_schema_is_ready(action)
            or self.candidates.implementation_is_ready(action)
        )
        action.materialized = action.implemented or materialization_ready
        action.resolved = record.get("status") != "pending"

    @staticmethod
    def _verify_admitted_record(
        action: RoundAction, record: dict[str, Any]
    ) -> None:
        actual_parents = [str(value) for value in record.get("source_run_ids", [])]
        if record.get("op") != action.op or actual_parents != action.parents:
            raise ArtifactError(
                f"ledger record {action.run_id} does not match reserved round action"
            )

    def _resume_state_if_authorized(self) -> None:
        if self.state.phase is not CoordinatorPhase.BLOCKED:
            return
        if not self.controls.resume_blocked:
            return
        self.state.phase = CoordinatorPhase.RUNNING
        self.state.stop_condition = None
        # Manual reopen gets a fresh upstream budget; counters from the prior
        # blocked episode must not immediately re-trip exhaustion.
        self._clear_upstream_recovery(save=False)
        self.store.save(self.state)
        if self.identity.ledger_path.exists():
            self.toolchain.set_phase(self.identity.run_dir, "running")

    def _clear_upstream_recovery(self, *, save: bool) -> None:
        if self.state is None:
            return
        for key, value in upstream_fields_reset().items():
            setattr(self.state, key, value)
        if save:
            self.store.save(self.state)

    def _completed_invocation_count(self) -> int:
        """Count durable completed model receipts under .orchestrator/invocations."""
        root = self.identity.run_dir / ".orchestrator" / "invocations"
        if not root.is_dir():
            return 0
        count = 0
        for receipt_path in root.glob("*/receipt.json"):
            try:
                payload = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("status") == "completed":
                count += 1
        return count

    def _maybe_clear_upstream_after_model_success(self, completed_before: int) -> None:
        if self._completed_invocation_count() > completed_before:
            self._clear_upstream_recovery(save=True)

    def _recover_from_upstream(self, exc: InferenceError) -> bool:
        """Backoff and stay running on retryable upstream faults.

        Returns True when the caller should retry the interrupted work. Returns
        False when the error is not an upstream transport fault (caller should
        re-raise / outer handler blocks). When streak or wall-clock caps are
        hit, blocks the run and returns True so the main loop observes BLOCKED
        without re-raising.
        """
        if not is_retryable_upstream_failure(exc):
            return False
        if self.state is None:
            return False

        decision = decide_upstream_recovery(
            current_streak=self.state.upstream_failure_streak,
            current_backoff_total_seconds=self.state.upstream_backoff_total_seconds,
            error=exc,
            policy=self._upstream_policy,
        )
        self.state.upstream_failure_streak = decision.streak
        self.state.upstream_last_error = f"{type(exc).__name__}: {exc}"[:500]
        self.state.upstream_last_decision = decision.reason[:500]

        if decision.action == "block":
            self.state.upstream_backoff_total_seconds = decision.backoff_total_seconds
            self.store.save(self.state)
            self._block(decision.reason)
            return True

        self.state.upstream_backoff_total_seconds = decision.backoff_total_seconds
        self.store.save(self.state)
        if decision.sleep_seconds > 0:
            self._sleep(decision.sleep_seconds)
        return True

    def _has_provided_baseline(self) -> bool:
        if self.task_config is None:
            return False
        seed = self.task_config.get("seed")
        provided = seed.get("provided") if isinstance(seed, dict) else None
        return isinstance(provided, list) and bool(provided)

    def _mark(self, transition: Transition) -> None:
        if self.state is None:
            return
        self.state.last_transition = transition.value
        self.store.save(self.state)

    def _block(self, reason: str) -> None:
        reason = reason.strip()[:1000] or "unspecified blocker"
        if self.state is None:
            try:
                self.state = self.store.load(
                    self.identity.task_name, self.identity.tag
                )
            except Exception:
                self.state = CoordinatorState(
                    task_name=self.identity.task_name,
                    tag=self.identity.tag,
                )
        self.state.phase = CoordinatorPhase.BLOCKED
        self.state.stop_condition = reason
        if self.identity.ledger_path.exists():
            try:
                self.toolchain.set_phase(self.identity.run_dir, "blocked", reason)
            except Exception as persistence_error:
                self.state.stop_condition = (
                    reason
                    + "; ledger block persistence failed: "
                    + f"{type(persistence_error).__name__}: {persistence_error}"
                )[:1000]
        self.store.save(self.state)

    def status(self) -> dict[str, Any]:
        state = self.state or self.store.load(
            self.identity.task_name, self.identity.tag
        )
        brief: dict[str, Any] = {}
        if self.identity.ledger_path.exists():
            try:
                brief = self.toolchain.ledger_brief(self.identity.run_dir)
            except Exception as exc:
                brief = {"ledger_error": f"{type(exc).__name__}: {exc}"}
        active = state.active_round
        return {
            "task": self.identity.task_name,
            "tag": self.identity.tag,
            "run_dir": str(self.identity.run_dir),
            "phase": state.phase.value,
            "last_transition": state.last_transition,
            "active_round": None if active is None else active.round_id,
            "active_run_ids": []
            if active is None
            else [action.run_id for action in active.actions],
            "stop_condition": state.stop_condition,
            "upstream_failure_streak": state.upstream_failure_streak,
            "upstream_backoff_total_seconds": state.upstream_backoff_total_seconds,
            "upstream_last_error": state.upstream_last_error,
            "upstream_last_decision": state.upstream_last_decision,
            "ledger": brief,
        }
