"""Deterministic run-level state machine for HieraResearch experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactError, CoordinatorStore
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
from .process import ProcessInterrupted
from .semantic import SemanticAdmission
from .state_machine import next_transition
from .toolchain import ToolFailure, Toolchain, read_task_config
from .tuning import DeepTuner


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


class ExperimentCoordinator:
    def __init__(
        self,
        identity: RunIdentity,
        toolchain: Toolchain,
        models: ModelGateway,
        controls: RunControls,
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

    def run(self, *, max_transitions: int | None = None) -> dict[str, Any]:
        if max_transitions is not None and max_transitions <= 0:
            raise ValueError("max_transitions must be positive")
        try:
            return self._run(max_transitions=max_transitions)
        except (ProcessInterrupted, KeyboardInterrupt) as exc:
            self._block(f"interrupted: {exc}")
            return self.status()
        except (ArtifactError, ToolFailure, InferenceError, ValueError) as exc:
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
        self.background.ensure()
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
            self._execute(transition, brief)
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
            self.experience.run(brief)
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
                action.run_id = run_id
                self.store.save(self.state)
            existing = (
                self.toolchain.ledger_record(self.identity.run_dir, action.run_id)
                if self.identity.ledger_path.exists()
                else None
            )
            if existing is None:
                self.semantic.admit(
                    run_id=action.run_id,
                    op=action.op,
                    parents=action.parents,
                )
            else:
                self._verify_admitted_record(action, existing)
            action.admitted = True
            self.store.save(self.state)
        active.admission_complete = True
        self.store.save(self.state)

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
            self.candidates.record_build_failure(action, exc)
            action.resolved = True
            self.store.save(self.state)
            if self.candidates.is_provided_baseline(action.run_id or ""):
                raise ValueError("provided baseline implementation could not be prepared") from exc

    def _build_candidate_contract(self) -> None:
        action = self._next_unresolved_action()
        assert self.candidates is not None
        try:
            self.candidates.build_contract(action)
            action.contract_ready = True
            self.store.save(self.state)
        except CandidateBuildError as exc:
            self.candidates.record_build_failure(action, exc)
            action.resolved = True
            self.store.save(self.state)
            if self.candidates.is_provided_baseline(action.run_id or ""):
                raise ValueError("provided baseline tuning contract could not be prepared") from exc

    def _evaluate_candidate(self) -> None:
        action = self._next_unresolved_action()
        assert self.candidates is not None
        outcome = self.candidates.evaluate(action)
        action.resolved = True
        self.store.save(self.state)
        if self.candidates.is_provided_baseline(action.run_id or "") and outcome.status == "crash":
            raise ValueError("provided baseline could not be evaluated")

    def _preflight_candidate(self) -> None:
        action = self._next_unresolved_action()
        assert self.candidates is not None
        self.candidates.preflight(action)
        action.preflight_ready = True
        self.store.save(self.state)

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
        if no_admissions and no_objective_progress:
            self.state.no_progress_cycles += 1
        else:
            self.state.no_progress_cycles = 0
        self.store.save(self.state)
        if self.state.no_progress_cycles:
            self._block(
                "no_progress_possible: semantic admission cap returned no actions "
                "and deep tuning admitted no objective call"
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
        # Keep corruption detection independent from stage readiness. A report
        # is useful evidence that evaluation started, but only its own
        # revision-bound receipts can authorize the next contract/preflight
        # transition.
        self.candidates.evaluation_has_started(action)
        # A report proves that evaluation started, not that a current
        # candidate contract or no-score preflight receipt still matches the
        # candidate. Debug repairs can change those inputs after the report
        # was written, so recovery must trust only revision-bound receipts.
        action.contract_ready = self.candidates.contract_is_ready(action)
        action.preflight_ready = self.candidates.preflight_is_ready(action)
        action.implemented = (
            action.contract_ready
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
        self.store.save(self.state)
        if self.identity.ledger_path.exists():
            self.toolchain.set_phase(self.identity.run_dir, "running")

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
            "ledger": brief,
        }
