from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hieraresearch.artifacts import (  # noqa: E402
    ArtifactError,
    atomic_write_json,
    file_revision,
)
from hieraresearch.candidate import CandidatePipeline  # noqa: E402
from hieraresearch import cli as coordinator_cli  # noqa: E402
from hieraresearch.coordinator import (  # noqa: E402
    ExperimentCoordinator,
    RunControls,
)
from hieraresearch.debug import (  # noqa: E402
    DebugPolicy,
    FailureEvidence,
    parse_debug_response,
)
from hieraresearch.models import (  # noqa: E402
    ActiveRound,
    CoordinatorState,
    RoundAction,
    RunIdentity,
    Transition,
)
from hieraresearch.process import ProcessRunner  # noqa: E402
from hieraresearch.state_machine import next_transition  # noqa: E402
from hieraresearch.toolchain import (  # noqa: E402
    ToolFailure,
    Toolchain,
    parse_json_output,
)
from tests.fixtures import background_text, fixture_registry  # noqa: E402


class PoisonModelBackend:
    def generate(self, **kwargs):  # pragma: no cover - the canary must not call this
        raise AssertionError(f"unexpected model judgment: {kwargs.get('purpose')}")

    def edit(self, **kwargs):  # pragma: no cover - the canary must not call this
        raise AssertionError(f"unexpected model edit: {kwargs.get('purpose')}")


class StartupToolchainStub:
    """Local-only helper boundary for the public CLI preflight canary."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.calls: list[tuple[object, ...]] = []

    def initialize_run(self, task_name: str, tag: str, **controls) -> Path:
        self.calls.append(("initialize_run", controls))
        run_dir = self.repo_root / "runs" / task_name / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def sync_environment(self, task_config: dict) -> None:  # pragma: no cover
        raise AssertionError(f"startup canary attempted environment sync: {task_config}")

    def prepare_task(self, task_config: dict) -> None:  # pragma: no cover
        raise AssertionError(f"startup canary attempted task preparation: {task_config}")

    def environment_preflight(
        self, task_name: str, run_dir: Path, task_config: dict
    ) -> dict:
        del task_config
        self.calls.append(("environment_preflight", task_name))
        receipt = {"status": "ok", "objective_calls": 0}
        atomic_write_json(run_dir / "environment_preflight.json", receipt)
        return receipt


class BaselineAdmissionToolchain(Toolchain):
    """Use real semantic/ledger helpers without requiring a task runtime."""

    def __init__(self, identity: RunIdentity):
        super().__init__(identity.repo_root, ProcessRunner(), helper_timeout=30.0)
        self.identity = identity
        self.background_validation_calls = 0

    def initialize_run(self, task_name: str, tag: str, **controls) -> Path:
        if (task_name, tag) != (self.identity.task_name, self.identity.tag):
            raise AssertionError("coordinator initialized the wrong run")
        if any(value is not None for value in controls.values()):
            raise AssertionError(f"unexpected run overrides: {controls}")
        return self.identity.run_dir

    def environment_preflight(
        self, task_name: str, run_dir: Path, task_config: dict
    ) -> dict:
        del task_name, task_config
        receipt = {"status": "ok", "objective_calls": 0}
        atomic_write_json(run_dir / "environment_preflight.json", receipt)
        return receipt

    def validate_background(
        self,
        run_dir: Path,
        *,
        induced: bool,
        provided_baseline: bool = False,
    ) -> None:
        self.background_validation_calls += 1
        super().validate_background(
            run_dir,
            induced=induced,
            provided_baseline=provided_baseline,
        )


class ObjectiveCanaryToolchain(BaselineAdmissionToolchain):
    """Run production workers without requiring a task-specific uv environment."""

    def candidate_preflight(
        self,
        candidate_path: Path,
        configs_path: Path,
        *,
        k_eval: int,
        task_config: dict,
    ) -> dict:
        del task_config
        result = self._run(
            [
                sys.executable,
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
        try:
            payload = parse_json_output(result.output)
        except ValueError as exc:
            raise ToolFailure("candidate preflight", result) from exc
        if (
            not result.ok
            or not isinstance(payload, dict)
            or payload.get("status") not in {"ok", "not_declared"}
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
        task_config: dict,
        output_path: Path,
    ):
        del task_config
        return self._run(
            [
                sys.executable,
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


def write_provided_baseline_background(
    identity: RunIdentity,
    *,
    space_id: str,
    constraint: str,
) -> None:
    run_dir = identity.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = fixture_registry()
    registry["space_id"] = space_id
    registry["dimensions"] = [registry["dimensions"][-1]]
    registry["relations"] = []
    registry["guidance"] = []
    registry["sources"] = []
    markdown = background_text(registry).replace(
        "- `g-01` — Filtering can remove rare legal examples in this exact regime.",
        f"- `task-constraint` — {constraint}",
    )
    (run_dir / "background.md").write_text(markdown, encoding="utf-8")
    atomic_write_json(
        run_dir / "background_retrieval.json",
        {
            "schema_version": 3,
            "lane_budgets": {"novelty": 2048, "grounding": 6000},
            "retrieval_condition": None,
            "queries": [],
            "coverage_exemptions": [],
            "results": [],
            "selected_keys": [],
            "backend_calls": [],
            "visits": [],
            "backend_failures": [],
        },
    )
    dimension = registry["dimensions"][0]
    baseline = dimension["hypotheses"][0]
    entrypoint = identity.repo_root / "tasks" / identity.task_name / "train.py"
    atomic_write_json(
        run_dir / "baseline_mechanisms.json",
        {
            "schema_version": 1,
            "kind": "baseline_mechanism_inventory",
            "entrypoint": {
                "path": entrypoint.relative_to(identity.repo_root).as_posix(),
                "sha256": file_revision(entrypoint),
            },
            "dimensions": {
                dimension["id"]: {
                    "interventions": baseline["scope"]["interventions"],
                    "citations": ["train.py:1"],
                }
            },
        },
    )


def write_objective_canary_task(repo_root: Path) -> Path:
    task_dir = repo_root / "tasks" / "objective-canary"
    task_dir.mkdir(parents=True)
    (task_dir / "TASK.md").write_text(
        """# Objective canary

## Evaluation Contract

The provided train.py is the baseline. evaluate_config is the single official,
lower-is-better score surface and must run exactly once.
""",
        encoding="utf-8",
    )
    (task_dir / "task.toml").write_text(
        """name = "objective-canary"
description = "Lightweight provided-baseline objective canary."

[env]
type = "uv"
project = "tasks/objective-canary"

[run]
working_dir = "tasks/objective-canary"
timeout_seconds = 30

[seed]
target_count = 1
provided = ["train.py"]
entrypoint = "train.py"

[evaluation]
score_fn = "evaluate_config"
preflight_fn = "preflight_config"

[result]
metric = "canary_loss"
parser = "tools/parse_result.py"
required_patterns = ["^score:"]
results_file = "ledger.json"

[constraints]
editable_files = ["train.py"]
readonly_files = ["prepare.py"]
allow_dependencies = false
""",
        encoding="utf-8",
    )
    (task_dir / "train.py").write_text(
        """PARAM_SCHEMA = {"x": "int"}
DEFAULT_PARAMS = {"x": 1}
SEARCH_SPACE = {"x": ("int", 1, 2)}

def make_model(params):
    return params["x"]
""",
        encoding="utf-8",
    )
    (task_dir / "prepare.py").write_text(
        """import json
from pathlib import Path


def preflight_config(make_model, params):
    return {"status": "ok", "value": make_model(params)}


def evaluate_config(make_model, params):
    run_dir = Path(__file__).resolve().parents[2]
    attempts = [
        json.loads(line)
        for line in (run_dir / "evaluation_attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    reservation = attempts[-1]
    if (
        reservation.get("kind") != "score_attempt"
        or reservation.get("attempt_id") != "eval-000001"
        or reservation.get("run_id") != "000"
    ):
        raise RuntimeError(f"score entered without its durable reservation: {reservation}")
    marker = Path(__file__).with_name("score_calls.jsonl")
    with marker.open("a", encoding="utf-8") as handle:
        payload = {"attempt_id": reservation["attempt_id"], "params": params}
        handle.write(json.dumps(payload) + "\\n")
        handle.flush()
    return float((make_model(params) - 2) ** 2)
""",
        encoding="utf-8",
    )
    return task_dir


class CandidatePreflightToolchainStub:
    def __init__(self, attempts_path: Path | None = None):
        self.calls: list[tuple[Path, Path, int]] = []
        self.attempts_path = attempts_path

    def candidate_preflight(
        self,
        candidate_path: Path,
        configs_path: Path,
        *,
        k_eval: int,
        task_config: dict,
    ) -> dict:
        del task_config
        self.calls.append((candidate_path, configs_path, k_eval))
        if self.attempts_path is not None:
            self.attempts_path.write_text('{"kind":"score_attempt"}\n')
        return {
            "status": "ok",
            "objective_calls": 0,
            "configs_checked": 1,
            "attempts": [{"index": 0, "result": {"status": "ok"}}],
        }


class CrashClosureToolchainStub:
    def __init__(self):
        self.projected = False
        self.crashed = False

    def budget_status(self, run_dir: Path) -> dict:
        del run_dir
        return {"per_candidate": [{"run_id": "001", "evals": 1}]}

    def project_screening_report(
        self, run_dir: Path, run_id: str, report_path: Path
    ) -> None:
        del run_dir, run_id, report_path
        self.projected = True

    def record_crash(self, run_dir: Path, run_id: str) -> dict:
        del run_dir
        self.crashed = True
        return {"run_id": run_id, "status": "crash"}


class OrchestratorBoundaryTests(unittest.TestCase):
    def test_contract_ready_candidate_preflights_before_warm_evaluation(self) -> None:
        action = RoundAction(
            op="fresh",
            run_id="000",
            admitted=True,
            materialized=True,
            implemented=True,
            contract_ready=True,
        )
        state = CoordinatorState(
            task_name="toy",
            tag="run",
            active_round=ActiveRound(
                round_id=0,
                actions=[action],
                admission_complete=True,
            ),
        )

        self.assertEqual(
            next_transition(
                state,
                ledger_exists=True,
                ledger_brief={"phase": "running"},
                has_provided_baseline=False,
            ),
            Transition.PREFLIGHT_CANDIDATE,
        )
        action.preflight_ready = True
        self.assertEqual(
            next_transition(
                state,
                ledger_exists=True,
                ledger_brief={"phase": "running"},
                has_provided_baseline=False,
            ),
            Transition.EVALUATE_WARM_CONFIGS,
        )

    def test_cli_preflight_only_stops_before_model_background_or_objective_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "tools").mkdir()
            (repo_root / "tools" / "ledger.py").write_text("", encoding="utf-8")
            task_dir = repo_root / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "TASK.md").write_text("# Toy task\n", encoding="utf-8")
            (task_dir / "task.toml").write_text("", encoding="utf-8")

            toolchain = StartupToolchainStub(repo_root)
            poison_backend = PoisonModelBackend()
            stdout = io.StringIO()
            with (
                patch.object(coordinator_cli, "Toolchain", return_value=toolchain),
                patch.object(
                    coordinator_cli,
                    "AnthropicMessagesBackend",
                    return_value=poison_backend,
                ),
                patch.object(
                    coordinator_cli,
                    "ClaudeAgentBackend",
                    return_value=poison_backend,
                ),
                redirect_stdout(stdout),
            ):
                exit_code = coordinator_cli.main(
                    [
                        "toy",
                        "startup-canary",
                        "--repo-root",
                        str(repo_root),
                        "--preflight-only",
                        "--no-sync",
                        "--no-prepare",
                    ]
                )

            status = json.loads(stdout.getvalue())
            run_dir = repo_root / "runs" / "toy" / "startup-canary"
            state = json.loads(
                (run_dir / ".orchestrator" / "state.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(status["phase"], "running")
            self.assertEqual(status["last_transition"], "environment_preflight")
            self.assertIsNone(status["active_round"])
            self.assertEqual(status["active_run_ids"], [])
            self.assertEqual(status["ledger"], {})
            self.assertEqual(state["last_transition"], "environment_preflight")
            self.assertIsNone(state["active_round"])
            self.assertFalse((run_dir / "background.md").exists())
            self.assertFalse((run_dir / "background_retrieval.json").exists())
            self.assertFalse((run_dir / "ledger.json").exists())
            self.assertFalse((run_dir / "evaluation_attempts.jsonl").exists())
            self.assertFalse((run_dir / "candidates").exists())
            self.assertEqual(
                json.loads(
                    (run_dir / "environment_preflight.json").read_text(
                        encoding="utf-8"
                    )
                )["objective_calls"],
                0,
            )
            self.assertEqual(
                [call[0] for call in toolchain.calls],
                [
                    "initialize_run",
                    "environment_preflight",
                ],
            )

    def test_provided_baseline_reaches_materialized_control_without_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ignore_caches = shutil.ignore_patterns("__pycache__", "*.pyc")
            shutil.copytree(ROOT / "tools", repo_root / "tools", ignore=ignore_caches)
            shutil.copytree(ROOT / "contracts", repo_root / "contracts")
            task_dir = repo_root / "tasks" / "autoresearch-baseline"
            task_dir.mkdir(parents=True)
            for name in ("TASK.md", "prepare.py", "task.toml", "train.py"):
                shutil.copy2(ROOT / "tasks" / "autoresearch-baseline" / name, task_dir / name)
            entrypoint = task_dir / "train.py"

            identity = RunIdentity(repo_root, "autoresearch-baseline", "admit-canary")
            run_dir = identity.run_dir
            run_dir.mkdir(parents=True)
            atomic_write_json(
                run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )

            write_provided_baseline_background(
                identity,
                space_id="provided-baseline-admission-canary",
                constraint="This fixture exercises admission only.",
            )

            toolchain = BaselineAdmissionToolchain(identity)
            coordinator = ExperimentCoordinator(
                identity,
                toolchain,
                PoisonModelBackend(),
                RunControls(sync_environment=False, prepare_task=False),
            )

            status = coordinator.run(max_transitions=2)
            state = json.loads(
                (run_dir / ".orchestrator" / "state.json").read_text(
                    encoding="utf-8"
                )
            )
            ledger = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
            action = state["active_round"]["actions"][0]
            record = ledger["records"][0]
            candidate_dir = run_dir / "candidates" / "000"
            candidate_brief = json.loads(
                (candidate_dir / "_candidate_brief.json").read_text(encoding="utf-8")
            )
            implementation_receipt = json.loads(
                (candidate_dir / CandidatePipeline.IMPLEMENTATION_RECEIPT).read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(toolchain.background_validation_calls, 1)
            self.assertEqual(status["phase"], "running")
            self.assertEqual(status["last_transition"], "materialize_candidate")
            self.assertEqual(status["ledger"]["evaluations_attempted"], 0)
            self.assertEqual(action["run_id"], "000")
            self.assertTrue(action["admitted"])
            self.assertTrue(action["materialized"])
            self.assertTrue(action["implemented"])
            self.assertFalse(action["contract_ready"])
            self.assertFalse(action["resolved"])
            self.assertEqual(record["run_id"], "000")
            self.assertEqual(record["op"], "fresh")
            self.assertEqual(record["source_run_ids"], [])
            self.assertEqual(record["status"], "pending")
            self.assertEqual(record["candidate_name"], "provided_baseline")
            self.assertEqual(
                file_revision(candidate_dir / "train.py"), file_revision(entrypoint)
            )
            self.assertEqual(
                candidate_brief["implementation_source"]["sha256"],
                file_revision(entrypoint),
            )
            self.assertEqual(implementation_receipt["source"], "provided")
            self.assertEqual(
                implementation_receipt["candidate_revision"],
                file_revision(entrypoint),
            )
            self.assertFalse((candidate_dir / "_warm_configs.json").exists())
            self.assertFalse((candidate_dir / "_search_space.json").exists())
            self.assertFalse((candidate_dir / "tune_report.json").exists())
            self.assertFalse((run_dir / "evaluation_attempts.jsonl").exists())

    def test_provided_baseline_executes_one_reserved_objective_and_closes_ledger(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            ignore_caches = shutil.ignore_patterns("__pycache__", "*.pyc")
            shutil.copytree(ROOT / "tools", repo_root / "tools", ignore=ignore_caches)
            shutil.copytree(ROOT / "contracts", repo_root / "contracts")
            write_objective_canary_task(repo_root)

            identity = RunIdentity(repo_root, "objective-canary", "one-score")
            run_dir = identity.run_dir
            write_provided_baseline_background(
                identity,
                space_id="provided-baseline-objective-canary",
                constraint="This fixture exercises one baseline score only.",
            )
            atomic_write_json(
                run_dir / "framework_cfg.json",
                {
                    "space_initialization": {
                        "dimension_strategy": "catalog_subset"
                    },
                    "max_evaluations": 1,
                    "per_runtime_limit": 10,
                },
            )

            toolchain = ObjectiveCanaryToolchain(identity)
            coordinator = ExperimentCoordinator(
                identity,
                toolchain,
                PoisonModelBackend(),
                RunControls(sync_environment=False, prepare_task=False),
            )

            materialized = coordinator.run(max_transitions=2)
            candidate_dir = run_dir / "candidates" / "000"
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 1}])
            atomic_write_json(
                candidate_dir / "_search_space.json",
                {"x": ["int", 1, 2]},
            )

            preflighted = coordinator.run(max_transitions=2)

            self.assertEqual(materialized["last_transition"], "materialize_candidate")
            self.assertEqual(preflighted["last_transition"], "preflight_candidate")
            self.assertFalse((run_dir / "evaluation_attempts.jsonl").exists())
            self.assertFalse((candidate_dir / "score_calls.jsonl").exists())
            self.assertEqual(
                json.loads(
                    (candidate_dir / CandidatePipeline.PREFLIGHT_RECEIPT).read_text(
                        encoding="utf-8"
                    )
                )["objective_calls"],
                0,
            )

            evaluated = coordinator.run(max_transitions=1)

            attempts = [
                json.loads(line)
                for line in (run_dir / "evaluation_attempts.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            reservations = [
                row for row in attempts if row.get("kind") == "score_attempt"
            ]
            score_calls = [
                json.loads(line)
                for line in (candidate_dir / "score_calls.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            state = json.loads(
                (run_dir / ".orchestrator" / "state.json").read_text(
                    encoding="utf-8"
                )
            )
            report = json.loads(
                (candidate_dir / "tune_report.json").read_text(encoding="utf-8")
            )
            ledger = json.loads((run_dir / "ledger.json").read_text(encoding="utf-8"))
            record = ledger["records"][0]

            self.assertEqual(evaluated["last_transition"], "evaluate_warm_configs")
            self.assertEqual(len(reservations), 1)
            self.assertEqual(
                {
                    key: reservations[0][key]
                    for key in ("attempt_id", "run_id", "phase", "method")
                },
                {
                    "attempt_id": "eval-000001",
                    "run_id": "000",
                    "phase": "phase_a",
                    "method": "warmstart",
                },
            )
            self.assertEqual(
                score_calls,
                [{"attempt_id": "eval-000001", "params": {"x": 1}}],
            )
            self.assertEqual(report["phase_a"]["status"], "ok")
            self.assertEqual(report["phase_a"]["trials_attempted"], 1)
            self.assertEqual(report["phase_a"]["best_warm_score"], 1.0)
            self.assertTrue(state["active_round"]["actions"][0]["resolved"])
            self.assertEqual(record["status"], "keep")
            self.assertEqual(record["final_best_score"], 1.0)
            self.assertEqual(record["trials_attempted"], 1)
            self.assertEqual(evaluated["ledger"]["evaluations_attempted"], 1)

    def test_debug_is_reserved_only_for_evidenced_candidate_crash(self) -> None:
        payload = {
            "status": "crashed",
            "phase": "a",
            "crash_index": 1,
            "crash_params": {"depth": 3},
            "failure_receipt": {"type": "ValueError", "message": "bad depth"},
            "failure_ref": {"failure_id": "failure-123", "artifact": "failure.json"},
        }
        evidence = FailureEvidence.from_worker_payload("001", payload)
        with tempfile.TemporaryDirectory() as tmp:
            policy = DebugPolicy(Path(tmp))
            policy.reserve_analysis(evidence)
            with self.assertRaises(ArtifactError):
                policy.reserve_analysis(evidence)

        with self.assertRaisesRegex(ValueError, "only evidenced"):
            FailureEvidence.from_worker_payload(
                "001",
                {**payload, "status": "budget_exhausted"},
            )
        with self.assertRaises(ValueError):
            parse_debug_response(
                {
                    "verdict": "insufficient_evidence",
                    "rationale": "uncertain",
                    "corrected_config": [],
                    "repair_instructions": "",
                }
            )

    def test_attempted_budget_exhaustion_closes_without_projecting_failed_phase_a(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            report_path = identity.run_dir / "candidates" / "001" / "tune_report.json"
            report_path.parent.mkdir(parents=True)
            atomic_write_json(
                report_path,
                {
                    "phase_a": {
                        "status": "budget_exhausted",
                        "trials_attempted": 1,
                        "warm_start_configs": [],
                    }
                },
            )
            toolchain = CrashClosureToolchainStub()
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )

            outcome = pipeline._close_budget_exhausted("001", report_path)

            self.assertEqual(outcome.status, "crash")
            self.assertFalse(toolchain.projected)
            self.assertTrue(toolchain.crashed)

    def test_parent_snapshot_is_not_a_completed_candidate_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            candidate_path = candidate_dir / "train.py"
            candidate_path.write_text("PARENT = True\n", encoding="utf-8")
            pipeline = CandidatePipeline(
                identity,
                toolchain=object(),
                models=object(),
                task_config={},
            )
            action = RoundAction(op="improve", parents=["000"], run_id="001")

            self.assertFalse(pipeline.implementation_is_ready(action))
            atomic_write_json(
                candidate_dir / pipeline.IMPLEMENTATION_RECEIPT,
                {
                    "schema_version": 1,
                    "run_id": "001",
                    "source": "agent_sdk",
                    "candidate_revision": file_revision(candidate_path),
                },
            )
            self.assertTrue(pipeline.implementation_is_ready(action))
            candidate_path.write_text("PARENT = False\n", encoding="utf-8")
            self.assertFalse(pipeline.implementation_is_ready(action))

    def test_provided_baseline_materialization_rejects_changed_seed_copy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            task_dir = repo_root / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            task_entrypoint = task_dir / "train.py"
            task_entrypoint.write_text("CONTROL = True\n", encoding="utf-8")
            identity = RunIdentity(repo_root, "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "000"
            candidate_dir.mkdir(parents=True)
            candidate_path = candidate_dir / "train.py"
            candidate_path.write_text("CONTROL = True\n", encoding="utf-8")
            (candidate_dir / "prepare.py").write_text("", encoding="utf-8")
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {
                    "schema_version": 4,
                    "run_id": "000",
                    "op": "fresh",
                    "source_run_ids": [],
                    "implementation_source": {
                        "kind": "provided_entrypoint",
                        "path": "tasks/toy/train.py",
                        "sha256": file_revision(task_entrypoint),
                    },
                },
            )
            pipeline = CandidatePipeline(
                identity,
                toolchain=object(),
                models=object(),
                task_config={
                    "seed": {"provided": ["train.py"], "entrypoint": "train.py"}
                },
            )
            action = RoundAction(op="fresh", run_id="000", admitted=True)

            self.assertTrue(pipeline.materialization_is_ready(action))
            candidate_path.write_text("TAMPERED = True\n", encoding="utf-8")
            self.assertFalse(pipeline.materialization_is_ready(action))
            with self.assertRaisesRegex(ArtifactError, "does not match"):
                pipeline.implement(action)
            self.assertFalse(
                (candidate_dir / pipeline.IMPLEMENTATION_RECEIPT).exists()
            )

    def test_candidate_preflight_receipt_is_no_score_and_revision_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "run")
            task_dir = repo_root / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "task.toml").write_text("", encoding="utf-8")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "train.py").write_text(
                "def make_model(env, params):\n    return params\n",
                encoding="utf-8",
            )
            (candidate_dir / "prepare.py").write_text("", encoding="utf-8")
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 1}])
            atomic_write_json(candidate_dir / "_search_space.json", {"x": ["int", 1, 2]})
            atomic_write_json(
                candidate_dir / CandidatePipeline.CONTRACT_RECEIPT,
                {"schema_version": 1},
            )
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {
                    "schema_version": 4,
                    "run_id": "001",
                    "op": "fresh",
                    "source_run_ids": [],
                    "implementation_source": {"kind": "generated"},
                },
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 3}},
            )
            toolchain = CandidatePreflightToolchainStub()
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            action = RoundAction(
                op="fresh",
                run_id="001",
                admitted=True,
                materialized=True,
                implemented=True,
                contract_ready=True,
            )

            receipt = pipeline.preflight(action)

            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["objective_calls"], 0)
            self.assertEqual(toolchain.calls[0][2], 3)
            self.assertTrue(pipeline.preflight_is_ready(action))
            self.assertFalse(
                (identity.run_dir / "evaluation_attempts.jsonl").exists()
            )
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 2}])
            self.assertFalse(pipeline.preflight_is_ready(action))

            attempts_path = identity.run_dir / "evaluation_attempts.jsonl"
            pipeline.toolchain = CandidatePreflightToolchainStub(attempts_path)
            with self.assertRaisesRegex(ArtifactError, "objective attempt accounting"):
                pipeline.preflight(action)


if __name__ == "__main__":
    unittest.main()
