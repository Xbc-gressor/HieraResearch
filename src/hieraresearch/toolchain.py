"""Thin subprocess adapter over the repository's deterministic helpers."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

from .process import ProcessResult, ProcessRunner

try:  # pragma: no cover - exercised only on Python 3.10
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


class ToolFailure(RuntimeError):
    def __init__(self, label: str, result: ProcessResult):
        detail = result.output.strip()
        if len(detail) > 6000:
            detail = (
                detail[:3000]
                + "\n...[bounded diagnostic omitted]...\n"
                + detail[-3000:]
            )
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{label} failed with exit {result.returncode}{suffix}")
        self.label = label
        self.result = result


class ValidationRejected(ToolFailure):
    """A deterministic validator ran successfully and rejected authored data."""


def _validation_error(label: str, result: ProcessResult) -> ToolFailure:
    if result.returncode == 1 and not result.timed_out and not result.interrupted:
        return ValidationRejected(label, result)
    return ToolFailure(label, result)


def _validation_payload(
    label: str,
    result: ProcessResult,
    *,
    rejection_kind: str,
) -> dict[str, Any]:
    """Parse one typed validator receipt without guessing from exit code alone."""
    try:
        payload = parse_json_output(result.output)
    except ValueError as exc:
        raise ToolFailure(label, result) from exc
    errors = payload.get("errors") if isinstance(payload, dict) else None
    if not result.ok:
        recognized_rejection = bool(
            isinstance(payload, dict)
            and payload.get("ok") is False
            and payload.get("failure_kind") == rejection_kind
            and isinstance(errors, list)
            and errors
            and all(
                (isinstance(error, str) and bool(error))
                or (isinstance(error, dict) and bool(error))
                for error in errors
            )
        )
        if recognized_rejection:
            raise _validation_error(label, result)
        raise ToolFailure(label, result)
    if not (
        isinstance(payload, dict)
        and payload.get("ok") is True
        and (errors is None or errors == [])
        and payload.get("failure_kind") is None
    ):
        raise ToolFailure(label, result)
    return payload


def parse_json_output(output: str) -> Any:
    text = output.strip()
    if not text:
        raise ValueError("command produced no JSON output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(text) if char in "{["]
    for index in reversed(starts):
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not text[index + end :].strip():
            return value
    raise ValueError("command output does not end with a complete JSON value")


def read_task_config(repo_root: Path, task_name: str) -> dict[str, Any]:
    path = Path(repo_root) / "tasks" / task_name / "task.toml"
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"invalid task config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"task config must be a table: {path}")
    return value


class Toolchain:
    def __init__(
        self,
        repo_root: Path,
        runner: ProcessRunner,
        *,
        helper_timeout: float = 120.0,
        worker_timeout: float = 21600.0,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.runner = runner
        self.helper_timeout = helper_timeout
        self.worker_timeout = worker_timeout

    def _run(
        self,
        args: Iterable[str],
        *,
        timeout: float | None = None,
        output_path: Path | None = None,
        label: str,
        check: bool = True,
    ) -> ProcessResult:
        result = self.runner.run(
            list(args),
            cwd=self.repo_root,
            timeout=timeout or self.helper_timeout,
            output_path=output_path,
        )
        if check and not result.ok:
            raise ToolFailure(label, result)
        return result

    def _python(
        self,
        script: str,
        *args: str,
        timeout: float | None = None,
        output_path: Path | None = None,
        label: str | None = None,
        check: bool = True,
    ) -> ProcessResult:
        return self._run(
            [sys.executable, str(self.repo_root / script), *map(str, args)],
            timeout=timeout,
            output_path=output_path,
            label=label or script,
            check=check,
        )

    def _json_python(
        self,
        script: str,
        *args: str,
        timeout: float | None = None,
        label: str | None = None,
    ) -> Any:
        result = self._python(script, *args, timeout=timeout, label=label)
        try:
            return parse_json_output(result.output)
        except ValueError as exc:
            raise ToolFailure(label or script, result) from exc

    def _json_python_in_project(
        self,
        project: Path,
        script: str,
        *args: str,
        timeout: float | None = None,
        label: str | None = None,
    ) -> Any:
        result = self._run(
            [
                "uv",
                "--directory",
                str(project),
                "run",
                "python",
                str(self.repo_root / script),
                *map(str, args),
            ],
            timeout=timeout or self.helper_timeout,
            label=label or script,
        )
        try:
            return parse_json_output(result.output)
        except ValueError as exc:
            raise ToolFailure(label or script, result) from exc

    def _validation_python(
        self,
        script: str,
        *args: str,
        label: str,
        rejection_kind: str,
    ) -> ProcessResult:
        result = self._python(script, *args, label=label, check=False)
        _validation_payload(label, result, rejection_kind=rejection_kind)
        return result

    def _json_validation_python(
        self,
        script: str,
        *args: str,
        label: str,
        rejection_kind: str,
    ) -> Any:
        result = self._python(script, *args, label=label, check=False)
        return _validation_payload(
            label,
            result,
            rejection_kind=rejection_kind,
        )

    def initialize_run(
        self,
        task_name: str,
        tag: str,
        *,
        dimension_strategy: str | None,
        llm_intelligence_score: float | None,
        max_evaluations: int | None,
        per_runtime_limit: float | None,
    ) -> Path:
        args = [task_name, tag]
        if dimension_strategy is not None:
            args.extend(["--dimension-strategy", dimension_strategy])
        if llm_intelligence_score is not None:
            args.extend(["--llm-intelligence-score", str(llm_intelligence_score)])
        if max_evaluations is not None:
            args.extend(["--max-evaluations", str(max_evaluations)])
        if per_runtime_limit is not None:
            args.extend(["--timeout", str(per_runtime_limit)])
        self._python("tools/init_run.py", *args, label="initialize run")
        return self.repo_root / "runs" / task_name / tag

    def sync_environment(self, task_config: dict[str, Any]) -> None:
        project = self._task_project(task_config)
        self._run(
            ["uv", "--project", str(project), "sync"],
            timeout=self.worker_timeout,
            label="sync task environment",
        )

    def prepare_task(self, task_config: dict[str, Any]) -> None:
        command = task_config.get("run", {}).get("prepare_command")
        if command is None:
            return
        if not isinstance(command, str) or not command.strip():
            raise ValueError("run.prepare_command must be a non-empty string")
        task_project = self._task_project(task_config)
        parts = shlex.split(command)
        result = self.runner.run(parts, cwd=task_project, timeout=self.worker_timeout)
        if not result.ok:
            raise ToolFailure("task preparation", result)

    def environment_preflight(
        self,
        task_name: str,
        run_dir: Path,
        task_config: dict[str, Any],
    ) -> dict[str, Any]:
        project = self._task_project(task_config)
        result = self._run(
            [
                "uv",
                "--project",
                str(project),
                "run",
                "python",
                str(self.repo_root / "tools/preflight_env.py"),
                "--task",
                task_name,
                "--run-dir",
                str(run_dir),
            ],
            timeout=self.worker_timeout,
            label="environment preflight",
            check=False,
        )
        try:
            payload = parse_json_output(result.output)
        except ValueError as exc:
            raise ToolFailure("environment preflight", result) from exc
        if not result.ok or payload.get("status") != "ok":
            raise ToolFailure("environment preflight", result)
        return payload

    def validate_background(self, run_dir: Path, *, induced: bool, provided_baseline: bool = False) -> None:
        if induced:
            self._validation_python(
                "tools/background_contract.py",
                "catalog",
                "--path",
                str(run_dir / "dimension_catalog.json"),
                label="dimension catalog validation",
                rejection_kind="dimension_catalog_validation",
            )
        self.validate_background_retrieval(run_dir)

        args = [
            "validate",
            "--background",
            str(run_dir / "background.md"),
            "--retrieval-manifest",
            str(run_dir / "background_retrieval.json"),
        ]
        if provided_baseline:
            args.extend(
                ["--baseline-mechanisms", str(run_dir / "baseline_mechanisms.json")]
            )
        self._validation_python(
            "tools/background_contract.py",
            *args,
            label="background validation",
            rejection_kind="background_validation",
        )

    def validate_background_retrieval(self, run_dir: Path) -> None:
        self._validation_python(
            "tools/search_backends.py",
            "validate",
            "--manifest",
            str(run_dir / "background_retrieval.json"),
            label="background retrieval validation",
            rejection_kind="retrieval_validation",
        )

    def import_background_retrieval(self, run_dir: Path, draft_path: Path) -> dict[str, Any]:
        return self._json_validation_python(
            "tools/search_backends.py",
            "import-external",
            "--draft",
            str(draft_path),
            "--manifest",
            str(run_dir / "background_retrieval.json"),
            label="background retrieval import",
            rejection_kind="retrieval_draft_validation",
        )

    def background_catalog_receipt(self, catalog_path: Path | None = None) -> dict[str, str]:
        args = ["catalog", "--compact"]
        if catalog_path is not None:
            args.extend(["--path", str(catalog_path)])
        payload = self._json_validation_python(
            "tools/background_contract.py",
            *args,
            label="background catalog receipt",
            rejection_kind="dimension_catalog_validation",
        )
        receipt = payload.get("receipt") if isinstance(payload, dict) else None
        if (
            not isinstance(receipt, dict)
            or not isinstance(receipt.get("id"), str)
            or not isinstance(receipt.get("revision"), str)
        ):
            raise ValueError("background catalog command returned no valid receipt")
        return {"id": receipt["id"], "revision": receipt["revision"]}

    def background_preflight(self, run_dir: Path) -> dict[str, Any]:
        args = ["preflight", "--background", str(run_dir / "background.md")]
        if (run_dir / "ledger.json").exists():
            args.extend(["--ledger", str(run_dir / "ledger.json")])
        return self._json_python(
            "tools/background_contract.py", *args, label="background preflight"
        )

    def ledger_brief(self, run_dir: Path) -> dict[str, Any]:
        return self._json_python(
            "tools/ledger.py",
            "brief",
            "--ledger",
            str(run_dir / "ledger.json"),
            label="ledger brief",
        )

    def ledger_record(self, run_dir: Path, run_id: str) -> dict[str, Any] | None:
        return self._json_python(
            "tools/ledger.py",
            "show",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--run-id",
            run_id,
            label="ledger record",
        )

    def ledger_experience(self, run_dir: Path) -> Any:
        return self._json_python(
            "tools/ledger.py",
            "show",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--experience",
            label="ledger experience",
        )

    def set_phase(self, run_dir: Path, phase: str, reason: str | None = None) -> dict[str, Any]:
        args = ["set-phase", "--ledger", str(run_dir / "ledger.json"), "--phase", phase]
        if reason:
            args.extend(["--stop-condition", reason])
        return self._json_python("tools/ledger.py", *args, label=f"set phase {phase}")

    def got_decide(self, run_dir: Path) -> dict[str, Any]:
        return self._json_python(
            "tools/got_select.py",
            "decide",
            "--ledger",
            str(run_dir / "ledger.json"),
            label="graph selection",
        )

    def semantic_propose(
        self,
        run_dir: Path,
        run_id: str,
        op: str,
        parents: list[str],
        *,
        baseline_only: bool = False,
    ) -> Path:
        output = run_dir / ".semantic" / run_id / "proposals.json"
        args = [
            "propose",
            "--background",
            str(run_dir / "background.md"),
            "--ledger",
            str(run_dir / "ledger.json"),
            "--op",
            op,
            "--max-points",
            "24",
            "--output",
            str(output),
        ]
        if parents:
            args.extend(["--parents", ",".join(parents)])
        if baseline_only:
            args.append("--baseline-only")
        self._json_python("tools/semantic_search.py", *args, label="semantic proposal")
        return output

    def semantic_gain_context(self, run_dir: Path, proposals: Path) -> Path:
        output = proposals.parent / "gain-context.json"
        self._json_python(
            "tools/semantic_search.py",
            "gain-context",
            "--proposals",
            str(proposals),
            "--ledger",
            str(run_dir / "ledger.json"),
            "--output",
            str(output),
            label="semantic gain context",
        )
        return output

    def semantic_select(
        self,
        run_dir: Path,
        proposals: Path,
        *,
        policy: str | None = None,
        predictions: Path | None = None,
    ) -> tuple[Path, Path, dict[str, Any]]:
        point = proposals.parent / "point.json"
        receipt = proposals.parent / "policy.json"
        args = [
            "select",
            "--proposals",
            str(proposals),
            "--ledger",
            str(run_dir / "ledger.json"),
            "--point-output",
            str(point),
            "--receipt-output",
            str(receipt),
        ]
        if policy:
            args.extend(["--policy", policy])
        if predictions:
            args.extend(["--predictions", str(predictions)])
        payload = self._json_validation_python(
            "tools/semantic_search.py",
            *args,
            label="semantic selection",
            rejection_kind="prediction_validation",
        )
        return point, receipt, payload

    def admit_candidate(
        self,
        run_dir: Path,
        task_name: str,
        run_id: str,
        op: str,
        parents: list[str],
        point: Path,
        policy_receipt: Path,
        *,
        idea: str,
        change: str,
        candidate_name_hint: str,
        description: str,
    ) -> dict[str, Any]:
        args = [
            "add-record",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--task",
            task_name,
            "--run-id",
            run_id,
            "--kind",
            "optimization",
            "--op",
            op,
            "--idea",
            idea,
            "--change",
            change,
            "--background",
            str(run_dir / "background.md"),
            "--semantic-point",
            str(point),
            "--policy-receipt",
            str(policy_receipt),
            "--candidate-name-hint",
            candidate_name_hint,
            "--description",
            description,
        ]
        if parents:
            args.extend(["--source-run-ids", ",".join(parents)])
        return self._json_python(
            "tools/ledger.py",
            *args,
            label="candidate admission",
        )

    def materialize_candidate(
        self,
        task_name: str,
        tag: str,
        run_id: str,
        *,
        provided_baseline: bool = False,
    ) -> Path:
        args = [task_name, tag, run_id]
        args.append("--provided-baseline" if provided_baseline else "--skip-entrypoint")
        self._python("tools/new_candidate.py", *args, label="candidate materialization")
        return self.repo_root / "runs" / task_name / tag / "candidates" / run_id

    def lint_schema(self, candidate_path: Path) -> dict[str, Any]:
        return self._json_validation_python(
            "tools/tuners/tune_tools.py",
            "lint-schema",
            "--candidate-path",
            str(candidate_path),
            label="candidate schema lint",
            rejection_kind="candidate_schema_validation",
        )

    def provided_baseline_defaults(self, candidate_path: Path) -> dict[str, Any]:
        payload = self._json_validation_python(
            "tools/tuners/tune_tools.py",
            "read-default-params",
            "--candidate-path",
            str(candidate_path),
            label="provided baseline defaults",
            rejection_kind="default_params_validation",
        )
        defaults = payload.get("defaults") if isinstance(payload, dict) else None
        if not isinstance(defaults, dict):
            raise ValueError("provided baseline defaults returned no defaults object")
        return defaults

    def lint_contract(
        self,
        candidate_path: Path,
        *,
        require_base_params: bool = True,
    ) -> dict[str, Any]:
        args = [
            "lint-contract",
            "--candidate-path",
            str(candidate_path),
        ]
        if not require_base_params:
            args.append("--allow-missing-base-params")
        return self._json_validation_python(
            "tools/tuners/tune_tools.py",
            *args,
            label="candidate contract lint",
            rejection_kind="candidate_contract_validation",
        )

    def candidate_execution_revision(
        self,
        candidate_path: Path,
    ) -> dict[str, Any]:
        return self._json_python(
            "tools/tuners/tune_tools.py",
            "execution-revision",
            "--candidate-path",
            str(candidate_path),
            label="candidate execution revision",
        )

    def apply_base_params(
        self,
        candidate_path: Path,
        params_path: Path,
    ) -> dict[str, Any]:
        return self._json_python(
            "tools/apply_base_params.py",
            "--candidate-path",
            str(candidate_path),
            "--params-json",
            str(params_path),
            label="apply corrected base params",
        )

    def lineage_evidence(self, run_dir: Path, parents: list[str]) -> dict[str, Any]:
        if not parents:
            return {"per_parent": {}}
        return self._json_python(
            "tools/tuners/tune_tools.py",
            "lineage-evidence",
            "--run-dir",
            str(run_dir),
            "--source-run-ids",
            ",".join(parents),
            label="lineage evidence",
        )

    def build_inheritance(self, candidate_path: Path, configs_path: Path) -> dict[str, Any]:
        return self._json_python(
            "tools/tuners/tune_tools.py",
            "build-inheritance",
            "--candidate-path",
            str(candidate_path),
            "--configs-json",
            str(configs_path),
            label="parameter inheritance",
        )

    def check_search_space(
        self, candidate_path: Path, space_path: Path, configs_path: Path
    ) -> dict[str, Any]:
        return self._json_validation_python(
            "tools/tuners/tune_tools.py",
            "check-search-space",
            "--candidate-path",
            str(candidate_path),
            "--space-json",
            str(space_path),
            "--configs-json",
            str(configs_path),
            label="search-space validation",
            rejection_kind="search_space_validation",
        )

    def apply_search_space(self, candidate_path: Path, space_path: Path) -> None:
        self._python(
            "tools/apply_search_space.py",
            "--candidate-path",
            str(candidate_path),
            "--space-json",
            str(space_path),
            label="apply search space",
        )

    def candidate_preflight(
        self,
        candidate_path: Path,
        configs_path: Path,
        *,
        k_eval: int,
        task_config: dict[str, Any],
    ) -> dict[str, Any]:
        project = self._task_project(task_config)
        result = self._run(
            [
                "uv",
                "--project",
                str(project),
                "run",
                "python",
                str(self.repo_root / "tools/preflight_candidate.py"),
                "--candidate-path",
                str(candidate_path),
                "--configs-json",
                str(configs_path),
                "--k-eval",
                str(k_eval),
            ],
            timeout=self.worker_timeout,
            label="candidate preflight",
            check=False,
        )
        payload = _validation_payload(
            "candidate preflight",
            result,
            rejection_kind="candidate_preflight_validation",
        )
        if (
            payload.get("status") not in {"ok", "not_declared"}
            or payload.get("objective_calls") != 0
        ):
            raise ToolFailure("candidate preflight", result)
        return payload

    def warmstart(
        self,
        candidate_path: Path,
        configs_path: Path,
        report_path: Path,
        *,
        k_eval: int,
        task_config: dict[str, Any],
        output_path: Path,
    ) -> ProcessResult:
        project = self._task_project(task_config)
        return self._run(
            [
                "uv",
                "--project",
                str(project),
                "run",
                "python",
                str(self.repo_root / "tools/tuners/warmstart_eval.py"),
                "--candidate-path",
                str(candidate_path),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
                "--k-eval",
                str(k_eval),
            ],
            timeout=self.worker_timeout,
            output_path=output_path,
            label="warm-config evaluation",
            check=False,
        )

    def project_screening_report(self, run_dir: Path, run_id: str, report_path: Path) -> None:
        self._python(
            "tools/ledger.py",
            "set-tuning",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--run-id",
            run_id,
            "--from-report",
            str(report_path),
            label="project screening report",
        )

    def record_score(self, run_dir: Path, run_id: str, score: float) -> dict[str, Any]:
        return self._json_python(
            "tools/ledger.py",
            "record-run",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--run-id",
            run_id,
            "--final-best-score",
            str(score),
            label="record candidate score",
        )

    def record_crash(self, run_dir: Path, run_id: str) -> dict[str, Any]:
        return self._json_python(
            "tools/ledger.py",
            "record-run",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--run-id",
            run_id,
            "--status",
            "crash",
            label="record candidate crash",
        )

    def resolve_unevaluated(self, run_dir: Path, task_name: str, run_id: str) -> dict[str, Any]:
        return self._json_python(
            "tools/ledger.py",
            "resolve-unevaluated",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--task",
            task_name,
            "--run-id",
            run_id,
            label="resolve unevaluated candidate",
        )

    def budget_status(self, run_dir: Path) -> dict[str, Any]:
        return self._json_python(
            "tools/evaluation_budget.py",
            "status",
            "--run-dir",
            str(run_dir),
            label="evaluation budget status",
        )

    def objective_attempt_receipts(
        self,
        candidate_path: Path,
        *,
        phase: str,
        method: str,
    ) -> list[dict[str, Any]]:
        payload = self._json_python(
            "tools/evaluation_budget.py",
            "receipts",
            "--ref-path",
            str(candidate_path),
            "--phase",
            phase,
            "--method",
            method,
            label="objective attempt receipts",
        )
        receipts = payload.get("receipts") if isinstance(payload, dict) else None
        if not isinstance(receipts, list) or not all(
            isinstance(receipt, dict) for receipt in receipts
        ):
            raise ValueError("objective receipt reader returned malformed output")
        return receipts

    def select_tuning_candidate(self, run_dir: Path) -> dict[str, Any]:
        return self._json_python(
            "tools/tuners/tune_tools.py",
            "select-candidate",
            "--ledger",
            str(run_dir / "ledger.json"),
            label="deep-tune candidate selection",
        )

    def phase_c_action(self, candidate_path: Path, report_path: Path) -> dict[str, Any]:
        return self._json_python(
            "tools/tuners/tune_tools.py",
            "phase-c-action",
            "--candidate-path",
            str(candidate_path),
            "--tune-report-json",
            str(report_path),
            label="phase-c action",
        )

    def run_tuner(
        self,
        method: str,
        candidate_path: Path,
        report_path: Path,
        trial_cap: int,
        task_config: dict[str, Any],
        output_path: Path,
    ) -> ProcessResult:
        if method not in {"grid", "bo", "cmaes"}:
            raise ValueError(f"unknown tuning method: {method}")
        project = self._task_project(task_config)
        method_args = {
            "grid": ["--resolution", "5", "--max-trials", str(min(100, trial_cap)), "--patience", "6"],
            "bo": ["--n-trials", str(min(40, trial_cap))],
            "cmaes": ["--popsize", "8", "--max-evals", str(min(64, trial_cap)), "--patience", "20"],
        }[method]
        return self._run(
            [
                "uv",
                "--directory",
                str(project),
                "run",
                "python",
                str(self.repo_root / f"tools/tuners/{method}_search.py"),
                "--candidate-path",
                str(candidate_path),
                "--tune-report-json",
                str(report_path),
                *method_args,
            ],
            timeout=self.worker_timeout,
            output_path=output_path,
            label=f"{method} deep tuning",
            check=False,
        )

    def finalize_tuning(
        self,
        run_dir: Path,
        run_id: str,
        candidate_path: Path,
        report_path: Path,
        task_config: dict[str, Any],
    ) -> dict[str, Any]:
        project = self._task_project(task_config)
        return self._json_python_in_project(
            project,
            "tools/finalize_tuning.py",
            "--candidate-path",
            str(candidate_path),
            "--tune-report-json",
            str(report_path),
            "--ledger",
            str(run_dir / "ledger.json"),
            "--run-id",
            run_id,
            label="finalize deep tuning",
        )

    def verify_failure_artifact(
        self,
        report_path: Path,
        failure_id: str,
    ) -> dict[str, Any]:
        return self._json_python(
            "tools/tuners/tune_tools.py",
            "render-failure",
            "--tune-report-json",
            str(report_path),
            "--failure-id",
            failure_id,
            "--view",
            "verification",
            label="verify tuning failure artifact",
        )

    def experience_views(self, run_dir: Path) -> dict[str, Any]:
        ledger = run_dir / "ledger.json"
        background = run_dir / "background.md"
        commands = {
            "brief": ("tools/ledger.py", ["brief", "--ledger", str(ledger)]),
            "graph": (
                "tools/got_graph.py",
                ["render", "--ledger", str(ledger), "--incremental", "--top", "3", "--bottom", "3", "--format", "json"],
            ),
            "target_evidence": (
                "tools/background_contract.py",
                ["target-evidence", "--background", str(background), "--ledger", str(ledger), "--max-dimensions", "16", "--max-hypotheses", "32", "--max-edges-per-target", "5"],
            ),
            "space": (
                "tools/background_contract.py",
                ["render", "--background", str(background), "--ledger", str(ledger), "--max-hypotheses", "6"],
            ),
            "lineage": (
                "tools/background_contract.py",
                ["lineage", "--background", str(background), "--ledger", str(ledger), "--compact", "--limit", "8"],
            ),
        }
        result: dict[str, Any] = {}
        for key, (script, args) in commands.items():
            result[key] = self._json_python(
                script,
                *args,
                label=f"experience {key} view",
            )
        result["prior_experience"] = self.ledger_experience(run_dir)
        return result

    def validate_experience(self, run_dir: Path, experience_path: Path) -> None:
        self._json_validation_python(
            "tools/background_contract.py",
            "validate-experience",
            "--background",
            str(run_dir / "background.md"),
            "--ledger",
            str(run_dir / "ledger.json"),
            "--experience",
            str(experience_path),
            label="experience validation",
            rejection_kind="experience_validation",
        )

    def store_experience(self, run_dir: Path, experience_path: Path) -> None:
        self._python(
            "tools/ledger.py",
            "set-experience",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--background",
            str(run_dir / "background.md"),
            "--from-json",
            str(experience_path),
            label="store experience",
        )

    def apply_space_state(self, run_dir: Path) -> dict[str, Any]:
        return self._json_python(
            "tools/ledger.py",
            "apply-space-state",
            "--ledger",
            str(run_dir / "ledger.json"),
            "--background",
            str(run_dir / "background.md"),
            label="apply search-space state",
        )

    def background_render(self, run_dir: Path, *, max_hypotheses: int = 12) -> str:
        result = self._python(
            "tools/background_contract.py",
            "render",
            "--background",
            str(run_dir / "background.md"),
            "--ledger",
            str(run_dir / "ledger.json"),
            "--max-hypotheses",
            str(max_hypotheses),
            label="background render",
        )
        return result.output

    def _task_project(self, task_config: dict[str, Any]) -> Path:
        project = task_config.get("env", {}).get("project")
        if not isinstance(project, str) or not project:
            raise ValueError("task config env.project must be a non-empty string")
        path = (self.repo_root / project).resolve()
        try:
            path.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"task env.project escapes the repository: {project}") from exc
        return path
