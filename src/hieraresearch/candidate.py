"""Candidate materialization, bounded authoring, Phase A, and debug escalation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    ArtifactError,
    atomic_write_json,
    file_revision,
    paths_revision,
)
from .debug import DebugPolicy, FailureEvidence, parse_debug_response
from .llm import AgentEditSpec, InferenceContractError, ModelGateway
from .models import DebugVerdict, RoundAction, RunIdentity
from .prompts import (
    CANDIDATE_WRITER_SYSTEM,
    CODE_REPAIR_SYSTEM,
    CONTRACT_BUILDER_SYSTEM,
    DEBUG_SYSTEM,
)
from .schemas import DEBUG_SCHEMA
from .toolchain import ToolFailure, Toolchain, parse_json_output


class CandidateBuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class CandidateOutcome:
    status: str
    best_score: float | None = None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def _bounded_text(path: Path, limit: int = 80_000) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if len(text) <= limit else text[:limit] + "\n[truncated]"


class CandidatePipeline:
    IMPLEMENTATION_RECEIPT = "_implementation_ready.json"
    CONTRACT_RECEIPT = "_contract_ready.json"
    PREFLIGHT_RECEIPT = "_preflight_ready.json"

    def __init__(
        self,
        identity: RunIdentity,
        toolchain: Toolchain,
        models: ModelGateway,
        task_config: dict[str, Any],
    ):
        self.identity = identity
        self.toolchain = toolchain
        self.models = models
        self.task_config = task_config
        self.debug_policy = DebugPolicy(identity.run_dir)

    def candidate_dir(self, run_id: str) -> Path:
        return self.identity.run_dir / "candidates" / run_id

    def is_provided_baseline(self, run_id: str) -> bool:
        brief = _read_json(self.candidate_dir(run_id) / "_candidate_brief.json")
        source = brief.get("implementation_source") if isinstance(brief, dict) else None
        return isinstance(source, dict) and source.get("kind") == "provided_entrypoint"

    def materialize(self, action: RoundAction, *, provided_baseline: bool = False) -> Path:
        if action.run_id is None:
            raise ValueError("cannot materialize an action without run_id")
        candidate_dir = self.candidate_dir(action.run_id)
        if candidate_dir.exists():
            record = self.toolchain.ledger_record(
                self.identity.run_dir,
                action.run_id,
            )
            if not isinstance(record, dict) or not self.materialization_is_ready(
                action, record
            ):
                raise ArtifactError(
                    f"candidate directory exists without a valid materialization: {candidate_dir}"
                )
            return candidate_dir
        return self.toolchain.materialize_candidate(
            self.identity.task_name,
            self.identity.tag,
            action.run_id,
            provided_baseline=provided_baseline,
        )

    def materialization_is_ready(
        self,
        action: RoundAction,
        record: dict[str, Any] | None = None,
    ) -> bool:
        if action.run_id is None:
            return False
        candidate_dir = self.candidate_dir(action.run_id)
        try:
            brief = _read_json(candidate_dir / "_candidate_brief.json")
        except ValueError:
            return False
        if not isinstance(brief, dict) or (
            brief.get("schema_version") != 4
            or brief.get("run_id") != action.run_id
            or brief.get("op") != action.op
            or brief.get("source_run_ids") != action.parents
        ):
            return False
        if record is not None:
            for key in (
                "idea",
                "change",
                "source_run_ids",
                "op",
                "semantic_point",
                "policy_receipt",
            ):
                if brief.get(key) != record.get(key):
                    return False
        candidate_config = self.task_config.get("candidate", {})
        if not isinstance(candidate_config, dict):
            return False
        entrypoint = candidate_config.get("entrypoint", "train.py")
        copy_files = candidate_config.get("copy_files", ["prepare.py", "train.py"])
        if not isinstance(entrypoint, str) or not isinstance(copy_files, list):
            return False
        source = brief.get("implementation_source")
        source_kind = source.get("kind") if isinstance(source, dict) else None
        required = [path for path in copy_files if path != entrypoint]
        if source_kind != "generated":
            required.append(entrypoint)
        if not all(
            isinstance(relative, str) and (candidate_dir / relative).is_file()
            for relative in required
        ):
            return False
        if source_kind == "provided_entrypoint":
            return self._provided_entrypoint_is_current(
                candidate_dir,
                entrypoint=entrypoint,
                source=source,
            )
        return True

    def implement(self, action: RoundAction) -> Path:
        if action.run_id is None:
            raise ValueError("cannot implement an action without run_id")
        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        if self.implementation_is_ready(action):
            return candidate_path
        if self.is_provided_baseline(action.run_id):
            if not self.materialization_is_ready(action):
                raise ArtifactError(
                    "provided baseline candidate does not match its task seed receipt"
                )
            self._validate_python(candidate_path)
            self._write_implementation_receipt(action.run_id, candidate_path, "provided")
            return candidate_path

        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        brief_path = candidate_dir / "_candidate_brief.json"
        read_roots = [task_dir, candidate_dir]
        input_paths = [
            brief_path,
            candidate_dir / "prepare.py",
            task_dir / "TASK.md",
            task_dir / "task.toml",
            self.identity.run_dir / "ledger.json",
            candidate_path,
        ]
        for parent in action.parents:
            parent_path = self.candidate_dir(parent) / "train.py"
            read_roots.append(parent_path)
            input_paths.append(parent_path)
        immutable_input_paths = tuple(
            path for path in input_paths if path != candidate_path
        )
        for purpose in (
            f"candidate_writer:{action.run_id}:repair",
            f"candidate_writer:{action.run_id}",
        ):
            if self.models.journal.completed_output_matches(
                purpose=purpose,
                schema_version=1,
                model=self.models.model,
                output_path=candidate_path,
                immutable_input_paths=immutable_input_paths,
            ):
                self._validate_python(candidate_path)
                self._write_implementation_receipt(
                    action.run_id, candidate_path, "model_receipt"
                )
                return candidate_path

        prompt = (
            f"Implement candidate {action.run_id} in {candidate_path}.\n"
            f"Operation: {action.op}; numeric parents: {action.parents or 'none'}.\n"
            "Read _candidate_brief.json first. The only authorized write is train.py."
        )
        last_error: BaseException | None = None
        for attempt in range(2):
            purpose = f"candidate_writer:{action.run_id}" + (":repair" if attempt else "")
            attempt_prompt = prompt
            if last_error is not None:
                attempt_prompt += (
                    "\nThe prior output failed the deterministic Python syntax/postcondition "
                    f"check. Repair it once. Error: {last_error}"
                )
            try:
                self.models.edit(
                    AgentEditSpec(
                        purpose=purpose,
                        schema_version=1,
                        cwd=self.identity.repo_root,
                        system_prompt=CANDIDATE_WRITER_SYSTEM,
                        prompt=attempt_prompt,
                        tools=("Read", "Glob", "Grep", "Write", "Edit"),
                        read_roots=tuple(read_roots),
                        write_paths=(candidate_path,),
                        input_paths=tuple(input_paths),
                        immutable_input_paths=immutable_input_paths,
                        max_turns=24,
                    ),
                    validate=lambda: self._validate_python(candidate_path),
                )
                self._write_implementation_receipt(
                    action.run_id, candidate_path, "agent_sdk"
                )
                return candidate_path
            except (ValueError, SyntaxError) as exc:
                last_error = exc
        raise CandidateBuildError(
            f"candidate writer failed for {action.run_id}: {last_error}"
        )

    def build_contract(self, action: RoundAction) -> None:
        if action.run_id is None:
            raise ValueError("cannot build a contract without run_id")
        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        space_path = candidate_dir / "_search_space.json"
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        provided = self.is_provided_baseline(action.run_id)
        k = 1 if provided else self._tuner_integer("K", 5, minimum=1)
        if self.contract_is_ready(action):
            return
        if candidate_path.is_file() and configs_path.is_file() and space_path.is_file():
            try:
                self._recover_authored_contract(
                    action,
                    candidate_path=candidate_path,
                    configs_path=configs_path,
                    space_path=space_path,
                    k=k,
                )
                return
            except (ToolFailure, ValueError, SyntaxError):
                pass
        lineage = self.toolchain.lineage_evidence(self.identity.run_dir, action.parents)
        base_prompt = (
            f"Candidate: {candidate_path}\nK: {k}\nParents: {action.parents or 'none'}\n"
            f"Provided baseline: {provided}\n\nBounded lineage evidence:\n"
            + json.dumps(lineage, ensure_ascii=False)[:40_000]
            + "\n\nWrite exactly _warm_configs.json, _search_space.json, and only the "
            "behavior-preserving PARAM_SCHEMA/make_model edits needed in train.py."
        )
        read_roots = [task_dir, candidate_dir]
        input_paths = [
            candidate_path,
            candidate_dir / "prepare.py",
            candidate_dir / "_candidate_brief.json",
            task_dir / "TASK.md",
            task_dir / "task.toml",
            self.identity.run_dir / "framework_cfg.json",
            self.identity.run_dir / "ledger.json",
            configs_path,
            space_path,
        ]
        for parent in action.parents:
            parent_path = self.candidate_dir(parent) / "train.py"
            read_roots.append(parent_path)
            input_paths.append(parent_path)

        first_error: BaseException | None = None
        for attempt in range(2):
            purpose = f"tuning_contract:{action.run_id}" + (":repair" if attempt else "")
            prompt = base_prompt
            if first_error is not None:
                prompt += (
                    "\n\nThe deterministic contract checks rejected the first attempt. "
                    "Correct only the reported issue once.\n"
                    + str(first_error)
                )
            try:
                self.models.edit(
                    AgentEditSpec(
                        purpose=purpose,
                        schema_version=1,
                        cwd=self.identity.repo_root,
                        system_prompt=CONTRACT_BUILDER_SYSTEM,
                        prompt=prompt,
                        tools=("Read", "Glob", "Grep", "Write", "Edit"),
                        read_roots=tuple(read_roots),
                        write_paths=(candidate_path, configs_path, space_path),
                        input_paths=tuple(input_paths),
                        immutable_input_paths=tuple(
                            path
                            for path in input_paths
                            if path not in {candidate_path, configs_path, space_path}
                        ),
                        max_turns=32,
                    ),
                    validate=lambda: self._validate_authored_contract(
                        candidate_path, configs_path, space_path, k
                    ),
                )
                self._finalize_contract(
                    action,
                    candidate_path=candidate_path,
                    configs_path=configs_path,
                    space_path=space_path,
                )
                return
            except (ToolFailure, ValueError, SyntaxError) as exc:
                first_error = exc
        raise CandidateBuildError(
            f"tuning contract failed for {action.run_id}: {first_error}"
        )

    def evaluate(self, action: RoundAction) -> CandidateOutcome:
        if action.run_id is None:
            raise ValueError("cannot evaluate an action without run_id")
        run_id = action.run_id
        candidate_dir = self.candidate_dir(run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        report_path = candidate_dir / "tune_report.json"
        provided = self.is_provided_baseline(run_id)
        k_eval = 1 if provided else self._tuner_integer("K_eval", 3, minimum=2)
        log_path = (
            self.identity.run_dir
            / ".orchestrator"
            / "workers"
            / f"{run_id}-warmstart.log"
        )

        for _ in range(9):
            result = self.toolchain.warmstart(
                candidate_path,
                configs_path,
                report_path,
                k_eval=k_eval,
                task_config=self.task_config,
                output_path=log_path,
            )
            if result.returncode == 0 and not result.timed_out:
                report = _read_json(report_path)
                phase_a = report.get("phase_a") if isinstance(report, dict) else None
                score = phase_a.get("best_warm_score") if isinstance(phase_a, dict) else None
                if (
                    not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not math.isfinite(float(score))
                ):
                    raise ValueError(f"successful Phase A lacks a finite best_warm_score: {report_path}")
                self.toolchain.project_screening_report(
                    self.identity.run_dir, run_id, report_path
                )
                record = self.toolchain.record_score(
                    self.identity.run_dir, run_id, float(score)
                )
                return CandidateOutcome(status=str(record.get("status")), best_score=float(score))
            if result.returncode == 3 and not result.timed_out:
                payload = parse_json_output(result.output)
                evidence = FailureEvidence.from_worker_payload(run_id, payload)
                if not self._debug_once(action, evidence, report_path):
                    return self._close_crash(run_id, report_path)
                continue
            if result.returncode == 4 and not result.timed_out:
                return self._close_budget_exhausted(run_id, report_path)
            raise ToolFailure("warm-config evaluation process", result)

        return self._close_crash(run_id, report_path)

    def preflight(self, action: RoundAction) -> dict[str, Any]:
        if action.run_id is None:
            raise ValueError("cannot preflight an action without run_id")
        if self.preflight_is_ready(action):
            value = _read_json(
                self.candidate_dir(action.run_id) / self.PREFLIGHT_RECEIPT
            )
            if not isinstance(value, dict):  # pragma: no cover - readiness checked it
                raise ArtifactError("candidate preflight receipt is not an object")
            return value

        candidate_dir = self.candidate_dir(action.run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        k_eval = (
            1
            if self.is_provided_baseline(action.run_id)
            else self._tuner_integer("K_eval", 3, minimum=2)
        )
        input_revision = paths_revision(self._preflight_input_paths(action.run_id))
        attempts_path = self.identity.run_dir / "evaluation_attempts.jsonl"
        attempts_before = paths_revision((attempts_path,))
        payload = self.toolchain.candidate_preflight(
            candidate_path,
            configs_path,
            k_eval=k_eval,
            task_config=self.task_config,
        )
        if paths_revision(self._preflight_input_paths(action.run_id)) != input_revision:
            raise ArtifactError("candidate inputs changed during no-score preflight")
        if paths_revision((attempts_path,)) != attempts_before:
            raise ArtifactError("candidate preflight changed objective attempt accounting")
        if (
            not isinstance(payload, dict)
            or payload.get("status") not in {"ok", "not_declared"}
            or payload.get("objective_calls") != 0
        ):
            raise ArtifactError(f"invalid candidate preflight result: {payload!r}")
        receipt = {
            "schema_version": 1,
            "run_id": action.run_id,
            "input_revision": input_revision,
            "k_eval": k_eval,
            "status": payload["status"],
            "objective_calls": 0,
            "result": payload,
        }
        atomic_write_json(candidate_dir / self.PREFLIGHT_RECEIPT, receipt)
        return receipt

    def record_build_failure(self, action: RoundAction, error: BaseException) -> CandidateOutcome:
        if action.run_id is None:
            raise ValueError("cannot close an action without run_id")
        receipt = self.identity.run_dir / ".orchestrator" / "candidate_failures" / f"{action.run_id}.json"
        atomic_write_json(
            receipt,
            {
                "schema_version": 1,
                "run_id": action.run_id,
                "stage": "authoring",
                "error": f"{type(error).__name__}: {error}",
                "objective_calls": 0,
            },
        )
        record = self.toolchain.record_crash(self.identity.run_dir, action.run_id)
        return CandidateOutcome(status=str(record.get("status", "crash")))

    def _debug_once(
        self,
        action: RoundAction,
        evidence: FailureEvidence,
        report_path: Path,
    ) -> bool:
        run_id = evidence.run_id
        candidate_dir = self.candidate_dir(run_id)
        candidate_path = candidate_dir / "train.py"
        configs_path = candidate_dir / "_warm_configs.json"
        space_path = candidate_dir / "_search_space.json"
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        try:
            self.debug_policy.reserve_analysis(evidence)
        except ArtifactError:
            return False

        artifact_value = evidence.failure_ref.get("artifact")
        failure_artifact = None
        if artifact_value is not None:
            if not isinstance(artifact_value, str) or not artifact_value:
                raise ArtifactError(
                    f"failure artifact reference is malformed for candidate {run_id}"
                )
            candidate_root = candidate_dir.resolve()
            failure_artifact = (candidate_root / artifact_value).resolve()
            try:
                failure_artifact.relative_to(candidate_root)
            except ValueError as exc:
                raise ArtifactError(
                    f"failure artifact escapes candidate directory: {artifact_value!r}"
                ) from exc
            if not failure_artifact.is_file():
                raise ArtifactError(
                    f"failure artifact does not exist for candidate {run_id}: "
                    f"{artifact_value}"
                )
        prompt = (
            f"Candidate: {run_id}\nFailure phase: {evidence.phase}\n"
            f"Crash config index: {evidence.crash_index}\n"
            "Crash config:\n"
            + json.dumps(evidence.crash_params, ensure_ascii=False)
            + "\nFailure receipt:\n"
            + json.dumps(evidence.failure_receipt, ensure_ascii=False)
            + "\n\nTask contract:\n"
            + _bounded_text(task_dir / "TASK.md", 30_000)
            + "\n\nCandidate source:\n"
            + _bounded_text(candidate_path)
        )
        input_paths = [
            candidate_path,
            configs_path,
            space_path,
            report_path,
            task_dir / "TASK.md",
            task_dir / "task.toml",
        ]
        if failure_artifact is not None:
            input_paths.append(failure_artifact)
        try:
            decision = self.models.infer(
                purpose=f"debug:{run_id}:{evidence.fingerprint}",
                schema_version=1,
                system_prompt=DEBUG_SYSTEM,
                prompt=prompt,
                schema=DEBUG_SCHEMA,
                input_paths=input_paths,
                parser=parse_debug_response,
                max_tokens=4096,
            )
        except InferenceContractError:
            return False

        if decision.verdict is DebugVerdict.ABANDON:
            return False
        if decision.verdict is DebugVerdict.CONFIG_INVALID:
            if evidence.crash_index == 0 and (
                action.parents or self.is_provided_baseline(run_id)
            ):
                return False
            configs = _read_json(configs_path)
            if not isinstance(configs, list) or evidence.crash_index >= len(configs):
                return False
            corrected = decision.corrected_config
            current = configs[evidence.crash_index]
            if (
                not isinstance(corrected, dict)
                or not isinstance(current, dict)
                or set(corrected) != set(current)
            ):
                return False
            configs[evidence.crash_index] = corrected
            atomic_write_json(configs_path, configs)
            if action.parents:
                self.toolchain.build_inheritance(candidate_path, configs_path)
            self.toolchain.check_search_space(candidate_path, space_path, configs_path)
            try:
                self.preflight(action)
            except ToolFailure:
                return False
            return True

        try:
            self.debug_policy.reserve_code_repair(run_id)
        except ArtifactError:
            return False
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        repair_prompt = (
            f"Repair {candidate_path} for this evidenced incompatibility.\n"
            f"Diagnosis: {decision.rationale}\nInstructions: {decision.repair_instructions}\n"
            f"Failure receipt: {json.dumps(evidence.failure_receipt, ensure_ascii=False)}"
        )
        self.models.edit(
            AgentEditSpec(
                purpose=f"debug_repair:{run_id}:{evidence.fingerprint}",
                schema_version=1,
                cwd=self.identity.repo_root,
                system_prompt=CODE_REPAIR_SYSTEM,
                prompt=repair_prompt,
                tools=("Read", "Grep", "Edit"),
                read_roots=(task_dir, candidate_dir),
                write_paths=(candidate_path,),
                input_paths=(
                    candidate_path,
                    candidate_dir / "prepare.py",
                    configs_path,
                    space_path,
                    report_path,
                    task_dir / "TASK.md",
                    task_dir / "task.toml",
                ),
                immutable_input_paths=(
                    candidate_dir / "prepare.py",
                    configs_path,
                    space_path,
                    report_path,
                    task_dir / "TASK.md",
                    task_dir / "task.toml",
                ),
                max_turns=12,
            ),
            validate=lambda: self._validate_python(candidate_path),
        )
        if action.parents:
            self.toolchain.build_inheritance(candidate_path, configs_path)
        self.toolchain.check_search_space(candidate_path, space_path, configs_path)
        try:
            self.preflight(action)
        except ToolFailure:
            return False
        return True

    def _close_budget_exhausted(self, run_id: str, report_path: Path) -> CandidateOutcome:
        status = self.toolchain.budget_status(self.identity.run_dir)
        attempts = 0
        for row in status.get("per_candidate", []):
            if isinstance(row, dict) and row.get("run_id") == run_id:
                attempts = int(row.get("evals", 0))
                break
        if attempts == 0:
            record = self.toolchain.resolve_unevaluated(
                self.identity.run_dir, self.identity.task_name, run_id
            )
            return CandidateOutcome(status=str(record.get("status", "unevaluated")))
        return self._close_crash(run_id, report_path)

    def _close_crash(self, run_id: str, report_path: Path) -> CandidateOutcome:
        if report_path.is_file():
            report = _read_json(report_path)
            phase_a = report.get("phase_a") if isinstance(report, dict) else None
            status = phase_a.get("status") if isinstance(phase_a, dict) else None
            if status == "ok":
                self.toolchain.project_screening_report(
                    self.identity.run_dir, run_id, report_path
                )
            elif status not in {"crashed", "preflight_failed", "budget_exhausted"}:
                raise ArtifactError(
                    f"cannot close candidate {run_id} from malformed Phase A "
                    f"status {status!r}: {report_path}"
                )
        record = self.toolchain.record_crash(self.identity.run_dir, run_id)
        return CandidateOutcome(status=str(record.get("status", "crash")))

    def _validate_authored_contract(
        self,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
        k: int,
    ) -> dict[str, Any]:
        self._validate_python(candidate_path)
        lint = self.toolchain.lint_schema(candidate_path)
        configs = _read_json(configs_path)
        space = _read_json(space_path)
        if not isinstance(configs, list) or len(configs) != k:
            raise ValueError(f"_warm_configs.json must contain exactly {k} configs")
        if not all(isinstance(item, dict) for item in configs):
            raise ValueError("every warm config must be an object")
        if not isinstance(space, dict) or not space:
            raise ValueError("_search_space.json must be a non-empty object")
        return lint

    def _finalize_contract(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
    ) -> None:
        if action.parents:
            self.toolchain.build_inheritance(candidate_path, configs_path)
        self.toolchain.check_search_space(candidate_path, space_path, configs_path)
        self.toolchain.apply_search_space(candidate_path, space_path)
        if action.parents:
            # apply_search_space changes train.py, so refresh the exact lineage
            # binding against the final authored candidate revision.
            self.toolchain.build_inheritance(candidate_path, configs_path)
        self.toolchain.lint_contract(
            candidate_path,
            require_base_params=False,
        )
        self._write_contract_receipt(
            action.run_id or "",
            candidate_path,
            configs_path,
            space_path,
        )

    def _recover_authored_contract(
        self,
        action: RoundAction,
        *,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
        k: int,
    ) -> None:
        configs = _read_json(configs_path)
        if not isinstance(configs, list) or len(configs) != k:
            raise ValueError(f"_warm_configs.json must contain exactly {k} configs")
        try:
            self.toolchain.lint_contract(
                candidate_path,
                require_base_params=False,
            )
        except ToolFailure:
            self._validate_authored_contract(candidate_path, configs_path, space_path, k)
        self._finalize_contract(
            action,
            candidate_path=candidate_path,
            configs_path=configs_path,
            space_path=space_path,
        )

    def implementation_is_ready(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        candidate_path = self.candidate_dir(action.run_id) / "train.py"
        receipt_path = self.candidate_dir(action.run_id) / self.IMPLEMENTATION_RECEIPT
        try:
            receipt = _read_json(receipt_path)
        except ValueError:
            return False
        return (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 1
            and receipt.get("run_id") == action.run_id
            and candidate_path.is_file()
            and receipt.get("candidate_revision") == file_revision(candidate_path)
        )

    def contract_is_ready(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        candidate_dir = self.candidate_dir(action.run_id)
        paths = {
            "candidate_revision": candidate_dir / "train.py",
            "configs_revision": candidate_dir / "_warm_configs.json",
            "space_revision": candidate_dir / "_search_space.json",
        }
        try:
            receipt = _read_json(candidate_dir / self.CONTRACT_RECEIPT)
        except ValueError:
            return False
        return (
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 1
            and receipt.get("run_id") == action.run_id
            and all(
                path.is_file() and receipt.get(key) == file_revision(path)
                for key, path in paths.items()
            )
        )

    def preflight_is_ready(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        receipt_path = self.candidate_dir(action.run_id) / self.PREFLIGHT_RECEIPT
        try:
            receipt = _read_json(receipt_path)
        except ValueError:
            return False
        if not isinstance(receipt, dict):
            return False
        try:
            input_revision = paths_revision(
                self._preflight_input_paths(action.run_id)
            )
        except (OSError, ValueError):
            return False
        return (
            receipt.get("schema_version") == 1
            and receipt.get("run_id") == action.run_id
            and receipt.get("input_revision") == input_revision
            and receipt.get("objective_calls") == 0
            and receipt.get("status") in {"ok", "not_declared"}
        )

    def evaluation_has_started(self, action: RoundAction) -> bool:
        if action.run_id is None:
            return False
        report_path = self.candidate_dir(action.run_id) / "tune_report.json"
        if not report_path.exists():
            return False
        report = _read_json(report_path)
        if not isinstance(report, dict) or not isinstance(report.get("phase_a"), dict):
            raise ArtifactError(f"malformed Phase A report: {report_path}")
        return True

    def _write_implementation_receipt(
        self,
        run_id: str,
        candidate_path: Path,
        source: str,
    ) -> None:
        atomic_write_json(
            candidate_path.parent / self.IMPLEMENTATION_RECEIPT,
            {
                "schema_version": 1,
                "run_id": run_id,
                "source": source,
                "candidate_revision": file_revision(candidate_path),
            },
        )

    def _write_contract_receipt(
        self,
        run_id: str,
        candidate_path: Path,
        configs_path: Path,
        space_path: Path,
    ) -> None:
        atomic_write_json(
            candidate_path.parent / self.CONTRACT_RECEIPT,
            {
                "schema_version": 1,
                "run_id": run_id,
                "candidate_revision": file_revision(candidate_path),
                "configs_revision": file_revision(configs_path),
                "space_revision": file_revision(space_path),
            },
        )

    def _tuner_integer(self, key: str, default: int, *, minimum: int) -> int:
        cfg = _read_json(self.identity.run_dir / "framework_cfg.json")
        section = cfg.get("tuner", {}) if isinstance(cfg, dict) else {}
        value = section.get(key, default) if isinstance(section, dict) else default
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"framework_cfg tuner.{key} must be an integer >= {minimum}")
        return value

    def _provided_entrypoint_is_current(
        self,
        candidate_dir: Path,
        *,
        entrypoint: str,
        source: dict[str, Any],
    ) -> bool:
        seed = self.task_config.get("seed")
        if not isinstance(seed, dict):
            return False
        seed_entrypoint = seed.get("entrypoint", "train.py")
        provided = seed.get("provided")
        if (
            seed_entrypoint != entrypoint
            or not isinstance(provided, list)
            or not provided
            or any(not isinstance(item, str) for item in provided)
        ):
            return False
        repo_root = self.identity.repo_root.resolve()
        task_dir = (repo_root / "tasks" / self.identity.task_name).resolve()
        task_entrypoint = (task_dir / entrypoint).resolve()
        try:
            task_entrypoint.relative_to(task_dir)
            expected_path = task_entrypoint.relative_to(repo_root).as_posix()
        except ValueError:
            return False
        if not task_entrypoint.is_file():
            return False
        provided_paths = {
            candidate.resolve()
            for item in provided
            for candidate in (
                (Path(item),)
                if Path(item).is_absolute()
                else (task_dir / item, repo_root / item)
            )
        }
        if task_entrypoint not in provided_paths:
            return False
        expected_revision = file_revision(task_entrypoint)
        candidate_path = candidate_dir / entrypoint
        return (
            source.get("path") == expected_path
            and source.get("sha256") == expected_revision
            and candidate_path.is_file()
            and file_revision(candidate_path) == expected_revision
        )

    def _preflight_input_paths(self, run_id: str) -> tuple[Path, ...]:
        candidate_dir = self.candidate_dir(run_id)
        task_dir = self.identity.repo_root / "tasks" / self.identity.task_name
        return (
            candidate_dir / "train.py",
            candidate_dir / "prepare.py",
            candidate_dir / "_warm_configs.json",
            candidate_dir / "_search_space.json",
            candidate_dir / "_parameter_transfer.json",
            candidate_dir / self.CONTRACT_RECEIPT,
            task_dir / "task.toml",
            self.identity.run_dir / "framework_cfg.json",
        )

    @staticmethod
    def _validate_python(path: Path) -> Path:
        if not path.is_file():
            raise ValueError(f"candidate writer did not create {path}")
        source = path.read_text(encoding="utf-8", errors="strict")
        if not source.strip():
            raise ValueError(f"candidate source is empty: {path}")
        compile(source, str(path), "exec")
        return path
