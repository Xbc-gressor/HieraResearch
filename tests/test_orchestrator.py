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
    json_revision,
    paths_revision,
)
from hieraresearch.candidate import (  # noqa: E402
    CandidateBuildError,
    CandidatePipeline,
)
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
from hieraresearch.process import (  # noqa: E402
    ProcessError,
    ProcessResult,
    ProcessRunner,
)
from hieraresearch.state_machine import next_transition  # noqa: E402
from hieraresearch.toolchain import (  # noqa: E402
    ToolFailure,
    Toolchain,
    parse_json_output,
)


def immutable_failure_ref(failure_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "failure_id": failure_id,
        "artifact": f"_failures/{failure_id}.json",
        "sha256": json_revision({"failure_id": failure_id}),
    }


def execution_revision_fixture() -> dict[str, object]:
    revision = {
        "schema_version": 3,
        "structure_sha256": "sha256:" + "1" * 64,
    }
    revision["revision_sha256"] = json_revision(revision)
    return revision
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
    def __init__(self, *, attempts: int = 1, receipts: list[dict] | None = None):
        self.projected = False
        self.crashed = False
        self.attempts = attempts
        self.receipts = list(receipts or [])

    def budget_status(self, run_dir: Path) -> dict:
        del run_dir
        return {
            "per_candidate": [
                {"run_id": "001", "evals": self.attempts}
            ]
        }

    def project_screening_report(
        self, run_dir: Path, run_id: str, report_path: Path
    ) -> None:
        del run_dir, run_id, report_path
        self.projected = True

    def record_crash(self, run_dir: Path, run_id: str) -> dict:
        del run_dir
        self.crashed = True
        return {"run_id": run_id, "status": "crash"}

    def objective_attempt_receipts(self, *args, **kwargs) -> list[dict]:
        del args, kwargs
        return list(self.receipts)

    def candidate_execution_revision(self, candidate_path: Path) -> dict:
        del candidate_path
        return execution_revision_fixture()

    def verify_failure_artifact(
        self, report_path: Path, failure_id: str
    ) -> dict:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        terminal = report["phase_a"]["terminal_failure"]
        if terminal["failure_ref"]["failure_id"] != failure_id:
            raise AssertionError("failure verification identity mismatch")
        return {
            "failure_ref": terminal["failure_ref"],
            "failure_receipt": terminal["failure_receipt"],
        }


class DebugPreflightToolchainStub:
    def __init__(self):
        self.preflight_calls: list[tuple[Path, Path, int]] = []
        self.inheritance_calls = 0
        self.search_space_checks = 0
        self.base_params_calls: list[tuple[Path, Path]] = []
        self.crashes: list[str] = []

    def record_crash(self, run_dir: Path, run_id: str) -> dict:
        del run_dir
        self.crashes.append(run_id)
        return {"run_id": run_id, "status": "crash"}

    def build_inheritance(self, candidate_path: Path, configs_path: Path) -> dict:
        del candidate_path, configs_path
        self.inheritance_calls += 1
        return {"status": "ok"}

    def check_search_space(
        self, candidate_path: Path, space_path: Path, configs_path: Path
    ) -> dict:
        del candidate_path, space_path, configs_path
        self.search_space_checks += 1
        return {"ok": True}

    def lint_contract(self, candidate_path: Path) -> dict:
        return {
            "ok": True,
            "candidate_structure_sha256": file_revision(candidate_path),
        }

    def apply_base_params(self, candidate_path: Path, params_path: Path) -> dict:
        self.base_params_calls.append((candidate_path, params_path))
        return {"applied": True}

    def candidate_execution_revision(self, candidate_path: Path) -> dict:
        revision = execution_revision_fixture()
        revision["structure_sha256"] = file_revision(candidate_path)
        return revision

    def candidate_preflight(
        self,
        candidate_path: Path,
        configs_path: Path,
        *,
        k_eval: int,
        task_config: dict,
    ) -> dict:
        del task_config
        self.preflight_calls.append((candidate_path, configs_path, k_eval))
        return {"status": "ok", "objective_calls": 0}


class DebugModelStub:
    def __init__(self, decision):
        self.decision = decision
        self.infer_calls = 0
        self.edit_calls = 0

    def infer(self, **kwargs):
        del kwargs
        self.infer_calls += 1
        return self.decision

    def edit(self, spec, *, validate):
        self.edit_calls += 1
        target = spec.write_paths[0]
        target.write_text(
            target.read_text(encoding="utf-8")
            + f"REPAIR_{self.edit_calls} = True\n",
            encoding="utf-8",
        )
        return validate()



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

    def test_process_launch_failure_blocks_without_cli_exception(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "launch-failure")

            class LaunchFailureToolchain:
                def initialize_run(self, *args, **kwargs):
                    del args, kwargs
                    raise ProcessError("failed to start 'uv'")

            coordinator = ExperimentCoordinator(
                identity,
                toolchain=LaunchFailureToolchain(),
                models=object(),
                controls=RunControls(),
            )

            status = coordinator.run()

            self.assertEqual(status["phase"], "blocked")
            self.assertIn("ProcessError: failed to start 'uv'", status["stop_condition"])

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
            "objective_slot_consumed": True,
            "failure_category": "candidate_code_incompatibility",
            "candidate_execution_revision": execution_revision_fixture(),
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
        with self.assertRaisesRegex(ValueError, "candidate-code"):
            FailureEvidence.from_worker_payload(
                "001",
                {**payload, "failure_category": "timeout_or_resource"},
            )
        with self.assertRaisesRegex(ValueError, "Phase-A"):
            FailureEvidence.from_worker_payload(
                "001",
                {**payload, "objective_slot_consumed": False},
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

    def test_debug_repair_runs_bounded_preflight_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "run")
            task_dir = repo_root / "tasks" / "toy"
            candidate_dir = identity.run_dir / "candidates" / "001"
            task_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            (task_dir / "TASK.md").write_text("# task\n", encoding="utf-8")
            (task_dir / "task.toml").write_text("", encoding="utf-8")
            (candidate_dir / "prepare.py").write_text("", encoding="utf-8")
            (candidate_dir / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 1}])
            atomic_write_json(candidate_dir / "_search_space.json", {"x": ["int", 1, 2]})
            atomic_write_json(identity.run_dir / "framework_cfg.json", {"tuner": {"K_eval": 2}})

            evidence = FailureEvidence(
                run_id="001",
                phase="a",
                crash_index=0,
                crash_params={"x": 1},
                failure_receipt={"type": "ValueError", "message": "incompatible"},
                failure_ref={"failure_id": "failure-001"},
                objective_slot_consumed=True,
                failure_category="candidate_code_incompatibility",
            )
            models = DebugModelStub(
                parse_debug_response(
                    {
                        "verdict": "code_incompatible",
                        "rationale": "the implementation rejects a legal value",
                        "corrected_config": [],
                        "repair_instructions": "accept the legal value",
                    }
                )
            )
            toolchain = DebugPreflightToolchainStub()
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=models,
                task_config={},
            )

            repaired = pipeline._debug_once(
                RoundAction(op="fresh", run_id="001", admitted=True),
                evidence,
                candidate_dir / "tune_report.json",
            )

            self.assertTrue(repaired)
            self.assertEqual(models.edit_calls, 1)
            self.assertEqual(len(toolchain.preflight_calls), 1)
            self.assertEqual(toolchain.preflight_calls[0][2], 2)

            class UnavailablePreflight(DebugPreflightToolchainStub):
                def candidate_preflight(self, *args, **kwargs):
                    del args, kwargs
                    raise ToolFailure(
                        "candidate preflight",
                        ProcessResult(
                            args=("preflight",),
                            returncode=2,
                            output="preflight worker unavailable",
                            elapsed_seconds=0.0,
                        ),
                    )

            second_evidence = FailureEvidence(
                run_id="001",
                phase="a",
                crash_index=0,
                crash_params={"x": 1},
                failure_receipt={
                    "type": "ValueError",
                    "message": "still incompatible",
                },
                failure_ref={"failure_id": "failure-002"},
                objective_slot_consumed=True,
                failure_category="candidate_code_incompatibility",
            )
            (candidate_dir / pipeline.PREFLIGHT_RECEIPT).unlink()
            unavailable_models = DebugModelStub(models.decision)
            unavailable = CandidatePipeline(
                identity,
                toolchain=UnavailablePreflight(),
                models=unavailable_models,
                task_config={},
            )
            with self.assertRaisesRegex(ToolFailure, "worker unavailable"):
                unavailable._debug_once(
                    RoundAction(op="fresh", run_id="001", admitted=True),
                    second_evidence,
                    candidate_dir / "tune_report.json",
                )
            self.assertEqual(unavailable_models.edit_calls, 1)

    def test_debug_config_correction_must_change_the_crashing_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "run")
            task_dir = repo_root / "tasks" / "toy"
            candidate_dir = identity.run_dir / "candidates" / "001"
            task_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            (task_dir / "TASK.md").write_text("# task\n", encoding="utf-8")
            (task_dir / "task.toml").write_text("", encoding="utf-8")
            (candidate_dir / "prepare.py").write_text("", encoding="utf-8")
            (candidate_dir / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 1}])
            atomic_write_json(
                candidate_dir / "_search_space.json", {"x": ["int", 1, 2]}
            )
            evidence = FailureEvidence(
                run_id="001",
                phase="a",
                crash_index=0,
                crash_params={"x": 1},
                failure_receipt={"type": "ValueError", "message": "bad x"},
                failure_ref={"failure_id": "failure-same-config"},
                objective_slot_consumed=True,
                failure_category="candidate_code_incompatibility",
            )
            models = DebugModelStub(
                parse_debug_response(
                    {
                        "verdict": "config_invalid",
                        "rationale": "try a corrected value",
                        "corrected_config": [
                            {"key": "x", "value_json": "1"}
                        ],
                        "repair_instructions": "",
                    }
                )
            )
            toolchain = DebugPreflightToolchainStub()
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=models,
                task_config={},
            )

            self.assertFalse(
                pipeline._debug_once(
                    RoundAction(op="fresh", run_id="001", admitted=True),
                    evidence,
                    candidate_dir / "tune_report.json",
                )
            )
            self.assertEqual(toolchain.preflight_calls, [])
            self.assertEqual(
                json.loads((candidate_dir / "_warm_configs.json").read_text()),
                [{"x": 1}],
            )

    def test_debug_repair_resumes_after_preflight_without_reinference_or_reedit(
        self,
    ) -> None:
        class InterruptBeforeAuthorization(CandidatePipeline):
            def _authorize_phase_a_retry(self, run_id, terminal):
                del run_id, terminal
                raise RuntimeError("simulated interruption before retry authorization")

        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "run")
            task_dir = repo_root / "tasks" / "toy"
            candidate_dir = identity.run_dir / "candidates" / "001"
            task_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            (task_dir / "TASK.md").write_text("# task\n", encoding="utf-8")
            (task_dir / "task.toml").write_text("", encoding="utf-8")
            (candidate_dir / "prepare.py").write_text("", encoding="utf-8")
            (candidate_dir / "train.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 1}])
            atomic_write_json(
                candidate_dir / "_search_space.json", {"x": ["int", 1, 2]}
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            toolchain = DebugPreflightToolchainStub()
            execution_revision = toolchain.candidate_execution_revision(
                candidate_dir / "train.py"
            )
            evidence = FailureEvidence(
                run_id="001",
                phase="a",
                crash_index=0,
                crash_params={"x": 1},
                failure_receipt={"type": "ValueError", "message": "bad x"},
                failure_ref={"failure_id": "failure-resume"},
                objective_slot_consumed=True,
                failure_category="candidate_code_incompatibility",
                candidate_execution_revision=execution_revision,
            )
            terminal = {
                "status": "crashed",
                "phase": "a",
                "crash_index": 0,
                "crash_params": {"x": 1},
                "objective_slot_consumed": True,
                "failure_category": "candidate_code_incompatibility",
                "failure_receipt": evidence.failure_receipt,
                "failure_ref": evidence.failure_ref,
                "candidate_execution_revision": execution_revision,
            }
            models = DebugModelStub(
                parse_debug_response(
                    {
                        "verdict": "code_incompatible",
                        "rationale": "candidate rejects a legal value",
                        "corrected_config": [],
                        "repair_instructions": "accept the legal value",
                    }
                )
            )
            action = RoundAction(op="fresh", run_id="001", admitted=True)
            report_path = candidate_dir / "tune_report.json"
            interrupted = InterruptBeforeAuthorization(
                identity,
                toolchain=toolchain,
                models=models,
                task_config={},
            )

            with self.assertRaisesRegex(
                RuntimeError, "before retry authorization"
            ):
                interrupted._debug_once(
                    action,
                    evidence,
                    report_path,
                    terminal=terminal,
                )
            self.assertEqual(models.infer_calls, 1)
            self.assertEqual(models.edit_calls, 1)
            self.assertEqual(len(toolchain.preflight_calls), 1)

            resumed = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=models,
                task_config={},
            )
            self.assertTrue(
                resumed._debug_once(
                    action,
                    evidence,
                    report_path,
                    terminal=terminal,
                )
            )
            self.assertEqual(models.infer_calls, 1)
            self.assertEqual(models.edit_calls, 1)
            self.assertEqual(len(toolchain.preflight_calls), 1)
            repair = json.loads(
                (candidate_dir / resumed.PHASE_A_REPAIR_RECEIPT).read_text()
            )
            self.assertEqual(repair["status"], "completed")
            self.assertTrue(
                resumed._phase_a_retry_is_authorized("001", terminal)
            )

    def test_debug_rejects_failure_artifact_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 1}])
            atomic_write_json(candidate_dir / "_search_space.json", {"x": ["int", 1, 2]})
            evidence = FailureEvidence(
                run_id="001",
                phase="a",
                crash_index=0,
                crash_params={"x": 1},
                failure_receipt={"type": "ValueError"},
                failure_ref={"failure_id": "failure-escape", "artifact": "../outside.json"},
                objective_slot_consumed=True,
                failure_category="candidate_code_incompatibility",
            )
            pipeline = CandidatePipeline(
                identity,
                toolchain=DebugPreflightToolchainStub(),
                models=DebugModelStub(
                    parse_debug_response(
                        {
                            "verdict": "abandon",
                            "rationale": "the evidence is not repairable",
                            "corrected_config": [],
                            "repair_instructions": "",
                        }
                    )
                ),
                task_config={},
            )

            with self.assertRaisesRegex(ArtifactError, "escapes candidate directory"):
                pipeline._debug_once(
                    RoundAction(op="fresh", run_id="001", admitted=True),
                    evidence,
                    candidate_dir / "tune_report.json",
                )

    def test_debug_state_corruption_propagates_instead_of_closing_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            pipeline = CandidatePipeline(
                identity,
                toolchain=DebugPreflightToolchainStub(),
                models=object(),
                task_config={},
            )
            evidence = FailureEvidence(
                run_id="001",
                phase="a",
                crash_index=0,
                crash_params={"x": 1},
                failure_receipt={"type": "ValueError"},
                failure_ref={"failure_id": "failure-corrupt-state"},
                objective_slot_consumed=True,
                failure_category="candidate_code_incompatibility",
            )
            with patch.object(
                pipeline.debug_policy,
                "reserve_analysis",
                side_effect=ArtifactError("corrupt debug journal"),
            ):
                with self.assertRaisesRegex(ArtifactError, "corrupt debug journal"):
                    pipeline._debug_once(
                        RoundAction(op="fresh", run_id="001", admitted=True),
                        evidence,
                        identity.run_dir / "candidates" / "001" / "tune_report.json",
                    )

    def test_restart_does_not_promote_stale_report_to_contract_or_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "prepare.py").write_text("", encoding="utf-8")
            (candidate_dir / "train.py").write_text("VALUE = 1\n", encoding="utf-8")
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
                candidate_dir / CandidatePipeline.IMPLEMENTATION_RECEIPT,
                {
                    "schema_version": 1,
                    "run_id": "001",
                    "candidate_revision": file_revision(candidate_dir / "train.py"),
                },
            )
            atomic_write_json(candidate_dir / "tune_report.json", {"phase_a": {"status": "crashed"}})

            coordinator = ExperimentCoordinator(
                identity,
                toolchain=object(),
                models=object(),
                controls=RunControls(),
            )
            coordinator.task_config = {}
            coordinator.candidates = CandidatePipeline(
                identity,
                toolchain=object(),
                models=object(),
                task_config={},
            )
            action = RoundAction(op="fresh", run_id="001", admitted=True)

            coordinator._infer_candidate_stages(action, {"status": "pending"})

            self.assertTrue(action.materialized)
            self.assertTrue(action.implemented)
            self.assertFalse(action.contract_ready)
            self.assertFalse(action.preflight_ready)
            self.assertFalse(action.resolved)

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

    def test_admitted_non_debug_failure_closes_without_replaying(self) -> None:
        class FailedWorkerToolchain(CrashClosureToolchainStub):
            def __init__(self, payload: dict):
                super().__init__(
                    attempts=1,
                    receipts=[payload["objective_reservation"]],
                )
                self.payload = payload
                self.warmstart_calls = 0

            def warmstart(self, *args, **kwargs) -> ProcessResult:
                del kwargs
                self.warmstart_calls += 1
                report_path = Path(args[2])
                report = json.loads(report_path.read_text())
                report["phase_a"].update(
                    {
                        "candidate_code_revision": self.payload[
                            "candidate_execution_revision"
                        ],
                        "terminal_failure": self.payload,
                    }
                )
                atomic_write_json(report_path, report)
                return ProcessResult(
                    args=("warmstart",),
                    returncode=3,
                    output=json.dumps(self.payload),
                    elapsed_seconds=0.0,
                )

        for category in (
            "timeout_or_resource",
            "unknown_non_candidate_failure",
            "process_interruption",
        ):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as tmp:
                identity = RunIdentity(Path(tmp), "toy", "run")
                candidate_dir = identity.run_dir / "candidates" / "001"
                candidate_dir.mkdir(parents=True)
                atomic_write_json(
                    candidate_dir / "_candidate_brief.json",
                    {"implementation_source": {"kind": "generated"}},
                )
                atomic_write_json(
                    identity.run_dir / "framework_cfg.json",
                    {"tuner": {"K_eval": 2}},
                )
                atomic_write_json(
                    candidate_dir / "tune_report.json",
                    {
                        "phase_a": {
                            "status": "crashed",
                            "trials_attempted": 1,
                            "warm_start_configs": [
                                {"params": {"x": 1}, "status": "failed"}
                            ],
                        }
                    },
                )
                payload = {
                    "status": "crashed",
                    "phase": "a",
                    "crash_index": 0,
                    "crash_params": {"x": 1},
                    "objective_slot_consumed": True,
                    "failure_category": category,
                    "candidate_execution_revision": execution_revision_fixture(),
                    "objective_attempt_id": "eval-000001",
                    "objective_reservation": {
                        "schema_version": 1,
                        "kind": "score_attempt",
                        "attempt_id": "eval-000001",
                        "run_id": "001",
                        "phase": "phase_a",
                        "method": "warmstart",
                        "params_sha256": json_revision({"x": 1}),
                    },
                    "failure_receipt": {"frames": []},
                    "failure_ref": immutable_failure_ref("fail-1111111111111111"),
                }
                toolchain = FailedWorkerToolchain(payload)
                pipeline = CandidatePipeline(
                    identity,
                    toolchain=toolchain,
                    models=object(),
                    task_config={},
                )

                outcome = pipeline.evaluate(
                    RoundAction(op="fresh", run_id="001", admitted=True)
                )

                self.assertEqual(outcome.status, "crash")
                self.assertEqual(toolchain.warmstart_calls, 1)
                self.assertTrue(toolchain.crashed)

    def test_terminal_phase_a_failure_retries_close_without_warmstart(self) -> None:
        class FailFirstCloseToolchain(CrashClosureToolchainStub):
            def __init__(self):
                super().__init__(attempts=1)
                self.warmstart_calls = 0
                self.close_calls = 0

            def warmstart(self, *args, **kwargs) -> ProcessResult:
                del args, kwargs
                self.warmstart_calls += 1
                raise AssertionError("terminal recovery must not spawn warmstart")

            def record_crash(self, run_dir: Path, run_id: str) -> dict:
                self.close_calls += 1
                if self.close_calls == 1:
                    raise RuntimeError("simulated interruption before ledger close")
                return super().record_crash(run_dir, run_id)

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            candidate_execution_revision = execution_revision_fixture()
            reservation = {
                "schema_version": 1,
                "kind": "score_attempt",
                "attempt_id": "eval-000001",
                "run_id": "001",
                "phase": "phase_a",
                "method": "warmstart",
                "params_sha256": json_revision({"x": 1}),
            }
            terminal = {
                "status": "crashed",
                "phase": "a",
                "crash_index": 0,
                "crash_params": {"x": 1},
                "objective_slot_consumed": True,
                "failure_category": "process_interruption",
                "candidate_execution_revision": candidate_execution_revision,
                "objective_attempt_id": "eval-000001",
                "objective_reservation": reservation,
                "failure_receipt": {"frames": []},
                "failure_ref": immutable_failure_ref("fail-2222222222222222"),
            }
            report_path = candidate_dir / "tune_report.json"
            atomic_write_json(
                report_path,
                {
                    "phase_a": {
                        "status": "crashed",
                        "trials_attempted": 1,
                        "warm_start_configs": [],
                        "candidate_code_revision": candidate_execution_revision,
                        "terminal_failure": terminal,
                    }
                },
            )
            toolchain = FailFirstCloseToolchain()
            toolchain.receipts = [reservation]
            action = RoundAction(op="fresh", run_id="001", admitted=True)

            first = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "interruption before ledger close",
            ):
                first.evaluate(action)

            resumed = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            outcome = resumed.evaluate(action)

            self.assertEqual(outcome.status, "crash")
            self.assertEqual(toolchain.close_calls, 2)
            self.assertEqual(toolchain.warmstart_calls, 0)
            self.assertTrue(toolchain.crashed)

    def test_phase_a_retry_authorization_is_bound_to_repaired_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            task_dir = identity.repo_root / "tasks" / "toy"
            candidate_dir.mkdir(parents=True)
            task_dir.mkdir(parents=True)
            for path, content in (
                (candidate_dir / "train.py", "MODEL = 1\n"),
                (candidate_dir / "prepare.py", "# evaluator\n"),
                (candidate_dir / "_warm_configs.json", "[]\n"),
                (candidate_dir / "_search_space.json", "{}\n"),
                (candidate_dir / CandidatePipeline.CONTRACT_RECEIPT, "{}\n"),
                (candidate_dir / CandidatePipeline.PREFLIGHT_RECEIPT, "{}\n"),
                (task_dir / "task.toml", "\n"),
                (identity.run_dir / "framework_cfg.json", "{}\n"),
            ):
                path.write_text(content, encoding="utf-8")
            pipeline = CandidatePipeline(
                identity,
                toolchain=DebugPreflightToolchainStub(),
                models=object(),
                task_config={},
            )
            terminal = {
                "status": "crashed",
                "phase": "a",
                "crash_index": 0,
                "crash_params": {"x": 1},
                "objective_slot_consumed": True,
                "failure_category": "candidate_code_incompatibility",
                "failure_receipt": {"frames": []},
                "failure_ref": {"failure_id": "failure-repaired"},
            }

            pipeline._authorize_phase_a_retry("001", terminal)

            self.assertTrue(
                pipeline._phase_a_retry_is_authorized("001", terminal)
            )
            (candidate_dir / "train.py").write_text(
                "MODEL = 2\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ArtifactError, "retry receipt is stale"):
                pipeline._phase_a_retry_is_authorized("001", terminal)

    def test_authorized_retry_survives_base_materialization_then_timeout(self) -> None:
        class BaseMaterializingTimeoutToolchain(CrashClosureToolchainStub):
            def __init__(self, reservation):
                super().__init__(attempts=1, receipts=[reservation])
                self.calls = 0

            def warmstart(self, candidate_path, *args, **kwargs):
                del args, kwargs
                self.calls += 1
                if self.calls == 1:
                    candidate_path.write_text(
                        candidate_path.read_text(encoding="utf-8")
                        + "BASE_PARAMS = {'x': 2}\n",
                        encoding="utf-8",
                    )
                    return ProcessResult(
                        args=("warmstart",),
                        returncode=124,
                        output="timeout after base materialization",
                        elapsed_seconds=10.0,
                        timed_out=True,
                    )
                return ProcessResult(
                    args=("warmstart",),
                    returncode=2,
                    output="recovery worker unavailable",
                    elapsed_seconds=0.1,
                )

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "train.py").write_text(
                "SEARCH_SPACE = {'x': ['int', 1, 2]}\n",
                encoding="utf-8",
            )
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(candidate_dir / "_warm_configs.json", [{"x": 2}])
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            revision = execution_revision_fixture()
            reservation = {
                "schema_version": 1,
                "kind": "score_attempt",
                "attempt_id": "eval-000001",
                "run_id": "001",
                "phase": "phase_a",
                "method": "warmstart",
                "params_sha256": json_revision({"x": 1}),
            }
            failure_id = "fail-6666666666666666"
            terminal = {
                "status": "crashed",
                "phase": "a",
                "crash_index": 0,
                "crash_params": {"x": 1},
                "objective_slot_consumed": True,
                "failure_category": "candidate_code_incompatibility",
                "candidate_execution_revision": revision,
                "objective_attempt_id": "eval-000001",
                "objective_reservation": reservation,
                "failure_receipt": {"failure_id": failure_id, "frames": []},
                "failure_ref": immutable_failure_ref(failure_id),
            }
            report_path = candidate_dir / "tune_report.json"
            atomic_write_json(
                report_path,
                {
                    "phase_a": {
                        "status": "crashed",
                        "candidate_code_revision": revision,
                        "terminal_failure": terminal,
                    }
                },
            )
            toolchain = BaseMaterializingTimeoutToolchain(reservation)
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            pipeline._authorize_phase_a_retry("001", terminal)

            with patch("hieraresearch.candidate.time.sleep"):
                with self.assertRaisesRegex(
                    ToolFailure, "recovery worker unavailable"
                ):
                    pipeline.evaluate(
                        RoundAction(op="fresh", run_id="001", admitted=True)
                    )

            self.assertEqual(toolchain.calls, 2)
            self.assertTrue(
                pipeline._phase_a_retry_is_authorized("001", terminal)
            )

    def test_historical_retry_authorization_does_not_replay_new_terminal(self) -> None:
        class NoWarmstartToolchain(CrashClosureToolchainStub):
            def warmstart(self, *args, **kwargs):
                raise AssertionError("new terminal must be handled without replay")

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            pipeline = CandidatePipeline(
                identity,
                toolchain=NoWarmstartToolchain(attempts=2),
                models=object(),
                task_config={},
            )
            terminal_a = {
                "status": "crashed",
                "phase": "a",
                "crash_index": 0,
                "crash_params": {"x": 1},
                "objective_slot_consumed": True,
                "failure_category": "candidate_code_incompatibility",
                "failure_ref": {"failure_id": "failure-a"},
            }
            pipeline._authorize_phase_a_retry("001", terminal_a)

            revision = execution_revision_fixture()
            reservation = {
                "schema_version": 1,
                "kind": "score_attempt",
                "attempt_id": "eval-000002",
                "run_id": "001",
                "phase": "phase_a",
                "method": "warmstart",
                "params_sha256": json_revision({"x": 2}),
            }
            terminal_b = {
                "status": "crashed",
                "phase": "a",
                "crash_index": 1,
                "crash_params": {"x": 2},
                "objective_slot_consumed": True,
                "failure_category": "process_interruption",
                "candidate_execution_revision": revision,
                "objective_attempt_id": "eval-000002",
                "objective_reservation": reservation,
                "failure_receipt": {"frames": []},
                "failure_ref": immutable_failure_ref("fail-3333333333333333"),
            }
            pipeline.toolchain.receipts = [reservation]
            report_path = candidate_dir / "tune_report.json"
            atomic_write_json(
                report_path,
                {
                    "phase_a": {
                        "status": "crashed",
                        "trials_attempted": 2,
                        "warm_start_configs": [],
                        "candidate_code_revision": revision,
                        "terminal_failure": terminal_b,
                    }
                },
            )

            self.assertFalse(
                pipeline._phase_a_retry_is_authorized("001", terminal_b)
            )
            outcome = pipeline.evaluate(
                RoundAction(op="fresh", run_id="001", admitted=True)
            )
            self.assertEqual(outcome.status, "crash")
            self.assertTrue(pipeline.toolchain.crashed)

    def test_phase_a_repair_receipt_rejects_unbound_revision_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            path = candidate_dir / CandidatePipeline.PHASE_A_REPAIR_RECEIPT
            terminal = {
                "status": "crashed",
                "phase": "a",
                "failure_ref": {"failure_id": "failure-repair"},
            }
            terminal_revision = json_revision(terminal)
            before_revision = "sha256:" + "1" * 64
            base = {
                "schema_version": 1,
                "kind": "phase_a_debug_repair",
                "status": "planned",
                "run_id": "001",
                "failure_id": "failure-repair",
                "terminal_failure_revision": terminal_revision,
                "analysis_reservation_id": terminal_revision,
                "candidate_revision_before": before_revision,
                "decision": {
                    "verdict": "code_incompatible",
                    "rationale": "candidate rejects a legal value",
                    "corrected_config": None,
                    "repair_instructions": "accept the value",
                },
            }
            pipeline = CandidatePipeline(
                identity,
                toolchain=object(),
                models=object(),
                task_config={},
            )
            atomic_write_json(path, base)
            self.assertEqual(
                pipeline._matching_phase_a_repair(
                    path,
                    run_id="001",
                    terminal=terminal,
                ),
                base,
            )

            malformed = [
                {**base, "failure_id": ""},
                {**base, "terminal_failure_revision": "not-a-revision"},
                {**base, "unexpected": True},
                {
                    **base,
                    "analysis_reservation_id": "sha256:" + "2" * 64,
                },
                {
                    **base,
                    "status": "edit_completed",
                },
                {
                    **base,
                    "status": "planned",
                    "candidate_revision_after": "sha256:" + "3" * 64,
                },
                {
                    **base,
                    "status": "edit_completed",
                    "candidate_revision_after": "sha256:" + "3" * 64,
                    "decision": {
                        "verdict": "config_invalid",
                        "rationale": "correct the value",
                        "corrected_config": {"x": 2},
                        "repair_instructions": None,
                    },
                },
            ]
            for receipt in malformed:
                with self.subTest(receipt=receipt):
                    atomic_write_json(path, receipt)
                    with self.assertRaises(ArtifactError):
                        pipeline._matching_phase_a_repair(
                            path,
                            run_id="001",
                            terminal=terminal,
                        )

    def test_phase_a_retry_receipt_rejects_malformed_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            pipeline = CandidatePipeline(
                identity,
                toolchain=object(),
                models=object(),
                task_config={},
            )
            terminal = {
                "failure_ref": {"failure_id": "failure-retry"},
            }
            atomic_write_json(
                candidate_dir / CandidatePipeline.PHASE_A_RETRY_RECEIPT,
                {
                    "schema_version": 1,
                    "kind": "phase_a_debug_retry",
                    "status": "authorized",
                    "run_id": "001",
                    "failure_id": "failure-retry",
                    "terminal_failure_revision": "not-a-revision",
                    "retry_input_revision": "sha256:" + "1" * 64,
                },
            )

            with self.assertRaisesRegex(
                ArtifactError, "malformed Phase A retry receipt"
            ):
                pipeline._phase_a_retry_is_authorized("001", terminal)

    def test_terminal_failure_without_exact_reservation_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            revision = execution_revision_fixture()
            report_path = candidate_dir / "tune_report.json"
            atomic_write_json(
                report_path,
                {
                    "phase_a": {
                        "status": "crashed",
                        "trials_attempted": 1,
                        "warm_start_configs": [],
                        "candidate_code_revision": revision,
                        "terminal_failure": {
                            "status": "crashed",
                            "phase": "a",
                            "crash_index": 0,
                            "crash_params": {"x": 1},
                            "objective_slot_consumed": True,
                            "failure_category": "process_interruption",
                            "candidate_execution_revision": revision,
                            "failure_receipt": {"frames": []},
                            "failure_ref": immutable_failure_ref("fail-4444444444444444"),
                        },
                    }
                },
            )
            pipeline = CandidatePipeline(
                identity,
                toolchain=CrashClosureToolchainStub(attempts=1),
                models=object(),
                task_config={},
            )

            with self.assertRaisesRegex(ArtifactError, "exact objective reservation"):
                pipeline.evaluate(
                    RoundAction(op="fresh", run_id="001", admitted=True)
                )

    def test_terminal_failure_must_match_verified_immutable_artifact(self) -> None:
        class MismatchedEvidenceToolchain(CrashClosureToolchainStub):
            def verify_failure_artifact(self, report_path, failure_id):
                report = json.loads(report_path.read_text(encoding="utf-8"))
                terminal = report["phase_a"]["terminal_failure"]
                return {
                    "failure_ref": terminal["failure_ref"],
                    "failure_receipt": {"failure_id": failure_id, "frames": []},
                }

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            revision = execution_revision_fixture()
            failure_id = "fail-5555555555555555"
            report_path = candidate_dir / "tune_report.json"
            atomic_write_json(
                report_path,
                {
                    "phase_a": {
                        "candidate_code_revision": revision,
                        "terminal_failure": {
                            "status": "crashed",
                            "phase": "preflight",
                            "crash_index": 0,
                            "crash_params": {"x": 1},
                            "objective_slot_consumed": False,
                            "objective_attempt_id": None,
                            "objective_reservation": None,
                            "failure_category": "candidate_code_incompatibility",
                            "candidate_execution_revision": revision,
                            "failure_receipt": {
                                "failure_id": failure_id,
                                "frames": [{"line": 10}],
                            },
                            "failure_ref": immutable_failure_ref(failure_id),
                        },
                    }
                },
            )
            pipeline = CandidatePipeline(
                identity,
                toolchain=MismatchedEvidenceToolchain(attempts=0),
                models=object(),
                task_config={},
            )

            with self.assertRaisesRegex(
                ArtifactError, "does not match its immutable artifact"
            ):
                pipeline._phase_a_terminal_failure("001", report_path)

    def test_stdout_only_worker_failure_is_rejected_without_crash(self) -> None:
        class PreflightFailureToolchain(CrashClosureToolchainStub):
            def __init__(self):
                super().__init__(attempts=0)

            def warmstart(self, *args, **kwargs) -> ProcessResult:
                del args, kwargs
                return ProcessResult(
                    args=("warmstart",),
                    returncode=3,
                    output=json.dumps(
                        {
                            "status": "crashed",
                            "phase": "preflight",
                            "crash_index": 0,
                            "crash_params": {"x": 1},
                            "objective_slot_consumed": False,
                            "failure_category": "timeout_or_resource",
                            "failure_receipt": {"frames": []},
                            "failure_ref": {"failure_id": "failure-preflight"},
                        }
                    ),
                    elapsed_seconds=0.0,
                )

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            toolchain = PreflightFailureToolchain()
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )

            with self.assertRaisesRegex(
                ArtifactError, "returned no durable terminal"
            ):
                pipeline.evaluate(
                    RoundAction(op="fresh", run_id="001", admitted=True)
                )

            self.assertFalse(toolchain.crashed)

    def test_worker_timeout_recovery_cap_survives_pipeline_restart(self) -> None:
        class TimeoutToolchain(CrashClosureToolchainStub):
            def __init__(self, *, forbid_call: bool = False):
                super().__init__(attempts=0)
                self.calls = 0
                self.forbid_call = forbid_call

            def warmstart(self, *args, **kwargs) -> ProcessResult:
                del args, kwargs
                self.calls += 1
                if self.forbid_call:
                    raise AssertionError(
                        "exhausted recovery must block before worker spawn"
                    )
                return ProcessResult(
                    args=("warmstart",),
                    returncode=124,
                    output="worker timeout",
                    elapsed_seconds=10.0,
                    timed_out=True,
                )

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            action = RoundAction(op="fresh", run_id="001", admitted=True)
            first_toolchain = TimeoutToolchain()
            first = CandidatePipeline(
                identity,
                toolchain=first_toolchain,
                models=object(),
                task_config={},
            )

            with patch("hieraresearch.candidate.time.sleep"):
                with self.assertRaisesRegex(
                    ToolFailure, "warm-config evaluation process"
                ):
                    first.evaluate(action)
            self.assertEqual(first_toolchain.calls, 2)

            resumed_toolchain = TimeoutToolchain(forbid_call=True)
            resumed = CandidatePipeline(
                identity,
                toolchain=resumed_toolchain,
                models=object(),
                task_config={},
            )
            with self.assertRaisesRegex(
                ArtifactError, "worker recovery is already consumed"
            ):
                resumed.evaluate(action)
            self.assertEqual(resumed_toolchain.calls, 0)

    def test_one_recorded_worker_timeout_grants_only_remaining_attempt(self) -> None:
        class FailingWorkerToolchain(CrashClosureToolchainStub):
            def __init__(self):
                super().__init__(attempts=0)
                self.calls = 0

            def warmstart(self, *args, **kwargs) -> ProcessResult:
                del args, kwargs
                self.calls += 1
                return ProcessResult(
                    args=("warmstart",),
                    returncode=2,
                    output="worker failed before recovery",
                    elapsed_seconds=0.1,
                )

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            toolchain = FailingWorkerToolchain()
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            input_revision = pipeline._phase_a_worker_input_revision("001")
            pipeline._record_phase_a_worker_timeout(
                "001",
                input_revision,
                ProcessResult(
                    args=("warmstart",),
                    returncode=124,
                    output="timeout",
                    elapsed_seconds=10.0,
                    timed_out=True,
                ),
            )

            with patch("hieraresearch.candidate.time.sleep"):
                with self.assertRaisesRegex(
                    ToolFailure, "warm-config evaluation process"
                ):
                    pipeline.evaluate(
                        RoundAction(op="fresh", run_id="001", admitted=True)
                    )
            self.assertEqual(toolchain.calls, 1)

            restarted = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            with self.assertRaisesRegex(
                ArtifactError, "worker recovery is already consumed"
            ):
                restarted.evaluate(
                    RoundAction(op="fresh", run_id="001", admitted=True)
                )
            self.assertEqual(toolchain.calls, 1)

    def test_worker_recovery_intent_prevents_spawn_after_parent_interruption(
        self,
    ) -> None:
        class NoSpawnToolchain(CrashClosureToolchainStub):
            def __init__(self):
                super().__init__(attempts=0)
                self.calls = 0

            def warmstart(self, *args, **kwargs):
                del args, kwargs
                self.calls += 1
                raise AssertionError("consumed recovery must not spawn")

        class InterruptAfterRecoveryIntent(CandidatePipeline):
            def _consume_phase_a_worker_recovery(self, *args, **kwargs):
                super()._consume_phase_a_worker_recovery(*args, **kwargs)
                raise RuntimeError("simulated parent interruption after intent")

        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {"implementation_source": {"kind": "generated"}},
            )
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K_eval": 2}},
            )
            toolchain = NoSpawnToolchain()
            interrupted = InterruptAfterRecoveryIntent(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            input_revision = interrupted._phase_a_worker_input_revision("001")
            interrupted._record_phase_a_worker_timeout(
                "001",
                input_revision,
                ProcessResult(
                    args=("warmstart",),
                    returncode=124,
                    output="timeout",
                    elapsed_seconds=10.0,
                    timed_out=True,
                ),
            )

            with patch("hieraresearch.candidate.time.sleep"):
                with self.assertRaisesRegex(
                    RuntimeError, "parent interruption after intent"
                ):
                    interrupted.evaluate(
                        RoundAction(op="fresh", run_id="001", admitted=True)
                    )
            self.assertEqual(toolchain.calls, 0)

            resumed = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )
            with self.assertRaisesRegex(
                ArtifactError, "worker recovery is already consumed"
            ):
                resumed.evaluate(
                    RoundAction(op="fresh", run_id="001", admitted=True)
                )
            self.assertEqual(toolchain.calls, 0)

    def test_zero_objective_failure_cannot_be_recorded_as_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            report_path = (
                identity.run_dir
                / "candidates"
                / "001"
                / "tune_report.json"
            )
            report_path.parent.mkdir(parents=True)
            atomic_write_json(
                report_path,
                {"phase_a": {"status": "preflight_failed"}},
            )
            toolchain = CrashClosureToolchainStub(attempts=0)
            pipeline = CandidatePipeline(
                identity,
                toolchain=toolchain,
                models=object(),
                task_config={},
            )

            with self.assertRaisesRegex(
                CandidateBuildError,
                "before any objective reservation",
            ):
                pipeline._close_crash("001", report_path)

            self.assertFalse(toolchain.projected)
            self.assertFalse(toolchain.crashed)

    def test_parent_snapshot_is_not_a_completed_candidate_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "run")
            candidate_dir = identity.run_dir / "candidates" / "001"
            candidate_dir.mkdir(parents=True)
            candidate_path = candidate_dir / "train.py"
            candidate_path.write_text("PARENT = True\n", encoding="utf-8")
            atomic_write_json(
                candidate_dir / "_candidate_brief.json",
                {
                    "schema_version": 4,
                    "run_id": "001",
                    "op": "improve",
                    "source_run_ids": ["000"],
                    "implementation_source": {"kind": "generated"},
                },
            )
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
                    "schema_version": 2,
                    "run_id": "001",
                    "source": "agent_sdk",
                    "candidate_revision": file_revision(candidate_path),
                    "input_revision": paths_revision(
                        pipeline._candidate_authoring_context_paths(action)
                    ),
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
