from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures import background_text, fixture_registry  # noqa: E402
from hieraresearch.artifacts import atomic_write_json  # noqa: E402
from hieraresearch.experience import ExperienceRefresh  # noqa: E402
from hieraresearch.models import RunIdentity  # noqa: E402
from hieraresearch.process import (  # noqa: E402
    ProcessInterrupted,
    ProcessResult,
    ProcessRunner,
)
from hieraresearch.semantic import SemanticAdmission  # noqa: E402
from hieraresearch.toolchain import (  # noqa: E402
    ToolFailure,
    Toolchain,
    ValidationRejected,
)
from semantic_search import build_proposal_set  # noqa: E402


def process_result(
    *,
    returncode: int,
    output: str,
    timed_out: bool = False,
    interrupted: bool = False,
) -> ProcessResult:
    return ProcessResult(
        args=("validator",),
        returncode=returncode,
        output=output,
        elapsed_seconds=0.0,
        timed_out=timed_out,
        interrupted=interrupted,
    )


class StaticRunner:
    def __init__(self, result: ProcessResult):
        self.result = result

    def run(self, args, *, cwd, timeout, output_path=None):
        del args, cwd, timeout, output_path
        return self.result


class SequenceRunner:
    def __init__(self, results: list[ProcessResult]):
        self.results = list(results)

    def run(self, args, *, cwd, timeout, output_path=None):
        del args, cwd, timeout, output_path
        if not self.results:
            raise AssertionError("unexpected validator process call")
        return self.results.pop(0)


class RaisingRunner:
    def __init__(self, error: BaseException):
        self.error = error

    def run(self, args, *, cwd, timeout, output_path=None):
        del args, cwd, timeout, output_path
        raise self.error


class ScriptedModels:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError(f"unexpected model call: {kwargs['purpose']}")
        return kwargs["parser"](self.responses.pop(0))


class ExperienceToolchainStub:
    def __init__(self, first_failure: ToolFailure):
        self.first_failure = first_failure
        self.validation_calls = 0
        self.store_calls = 0
        self.experience = None

    def experience_views(self, run_dir):
        del run_dir
        return {}

    def validate_experience(self, run_dir, output):
        del run_dir, output
        self.validation_calls += 1
        if self.validation_calls == 1:
            raise self.first_failure

    def store_experience(self, run_dir, output):
        del run_dir
        self.store_calls += 1
        self.experience = json.loads(output.read_text())
        self.experience["dag_revision"] = int(output.stem.rsplit("-", 1)[1])

    def ledger_experience(self, run_dir):
        del run_dir
        return self.experience

    def apply_space_state(self, run_dir):
        del run_dir
        return {"ok": True}


class SemanticToolchainStub:
    def __init__(self, first_failure: ToolFailure):
        self.first_failure = first_failure
        self.selection_calls = 0

    def semantic_gain_context(self, run_dir, proposals):
        del run_dir
        output = proposals.parent / "gain-context.json"
        output.write_text("{}\n", encoding="utf-8")
        return output

    def semantic_select(self, run_dir, proposals, *, predictions=None):
        del run_dir, predictions
        self.selection_calls += 1
        if self.selection_calls == 1:
            raise self.first_failure
        point = proposals.parent / "point.json"
        receipt = proposals.parent / "policy.json"
        point.write_text("{}\n", encoding="utf-8")
        receipt.write_text("{}\n", encoding="utf-8")
        return point, receipt, {"ok": True}


class ModelValidationBoundaryTests(unittest.TestCase):
    def test_json_validator_adapter_separates_rejection_from_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "toy" / "smoke"
            experience_path = run_dir / "experience.json"
            structured_error = json.dumps(
                {
                    "ok": False,
                    "failure_kind": "experience_validation",
                    "errors": ["authored value rejected"],
                }
            )
            cases = [
                (
                    "deterministic rejection",
                    process_result(returncode=1, output=structured_error),
                    ValidationRejected,
                ),
                (
                    "validator exit 2",
                    process_result(returncode=2, output=structured_error),
                    ToolFailure,
                ),
                (
                    "different contract failure",
                    process_result(
                        returncode=1,
                        output=json.dumps(
                            {
                                "ok": False,
                                "failure_kind": "input_contract_invalid",
                                "errors": ["stale ledger"],
                            }
                        ),
                    ),
                    ToolFailure,
                ),
                (
                    "validator timeout",
                    process_result(
                        returncode=124,
                        output=structured_error,
                        timed_out=True,
                    ),
                    ToolFailure,
                ),
                (
                    "validator interruption",
                    process_result(
                        returncode=130,
                        output=structured_error,
                        interrupted=True,
                    ),
                    ToolFailure,
                ),
                (
                    "malformed validator output",
                    process_result(returncode=1, output="worker log only"),
                    ToolFailure,
                ),
                (
                    "wrong-shaped validator output",
                    process_result(returncode=1, output='{"errors": ["bad"]}'),
                    ToolFailure,
                ),
                (
                    "contradictory success",
                    process_result(returncode=0, output=structured_error),
                    ToolFailure,
                ),
            ]
            for label, result, expected in cases:
                with self.subTest(label=label):
                    toolchain = Toolchain(root, StaticRunner(result))
                    with self.assertRaises(expected) as raised:
                        toolchain.validate_experience(run_dir, experience_path)
                    if expected is ToolFailure:
                        self.assertNotIsInstance(raised.exception, ValidationRejected)

    def test_all_model_output_validators_emit_validation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "toy" / "smoke"
            candidate_dir = run_dir / "candidates" / "001"
            invocations = {
                "candidate schema": (
                    "candidate_schema_validation",
                    lambda toolchain: toolchain.lint_schema(
                        candidate_dir / "train.py"
                    ),
                ),
                "candidate contract": (
                    "candidate_contract_validation",
                    lambda toolchain: toolchain.lint_contract(
                        candidate_dir / "train.py"
                    ),
                ),
                "candidate search space": (
                    "search_space_validation",
                    lambda toolchain: toolchain.check_search_space(
                        candidate_dir / "train.py",
                        candidate_dir / "_search_space.json",
                        candidate_dir / "_warm_configs.json",
                    ),
                ),
                "provided defaults": (
                    "default_params_validation",
                    lambda toolchain: toolchain.provided_baseline_defaults(
                        candidate_dir / "train.py"
                    ),
                ),
                "candidate preflight": (
                    "candidate_preflight_validation",
                    lambda toolchain: toolchain.candidate_preflight(
                        candidate_dir / "train.py",
                        candidate_dir / "_warm_configs.json",
                        k_eval=2,
                        task_config={"env": {"project": "."}},
                    ),
                ),
                "dimension catalog": (
                    "dimension_catalog_validation",
                    lambda toolchain: toolchain.background_catalog_receipt(
                        run_dir / "dimension_catalog.json"
                    ),
                ),
                "background retrieval validation": (
                    "retrieval_validation",
                    lambda toolchain: toolchain.validate_background_retrieval(run_dir),
                ),
                "background retrieval import": (
                    "retrieval_draft_validation",
                    lambda toolchain: toolchain.import_background_retrieval(
                        run_dir,
                        run_dir / "background_retrieval.draft.json",
                    ),
                ),
                "experience": (
                    "experience_validation",
                    lambda toolchain: toolchain.validate_experience(
                        run_dir,
                        run_dir / "experience.json",
                    ),
                ),
                "semantic selection": (
                    "prediction_validation",
                    lambda toolchain: toolchain.semantic_select(
                        run_dir,
                        candidate_dir / "proposals.json",
                        predictions=candidate_dir / "predictions.json",
                    ),
                ),
            }
            for label, (failure_kind, invoke) in invocations.items():
                with self.subTest(label=label):
                    result = process_result(
                        returncode=1,
                        output=json.dumps(
                            {
                                "ok": False,
                                "failure_kind": failure_kind,
                                "errors": ["authored value rejected"],
                            }
                        ),
                    )
                    with self.assertRaises(ValidationRejected):
                        invoke(Toolchain(root, StaticRunner(result)))

            success = process_result(
                returncode=0,
                output='{"ok": true, "errors": []}',
            )
            background_rejection = process_result(
                returncode=1,
                output=json.dumps(
                    {
                        "ok": False,
                        "failure_kind": "background_validation",
                        "errors": ["authored value rejected"],
                    }
                ),
            )
            toolchain = Toolchain(
                root,
                SequenceRunner([success, background_rejection]),
            )
            with self.assertRaises(ValidationRejected):
                toolchain.validate_background(run_dir, induced=False)

    def test_background_validator_rejects_untyped_or_operational_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "toy" / "smoke"
            typed = json.dumps(
                {
                    "ok": False,
                    "failure_kind": "retrieval_validation",
                    "errors": ["invalid manifest"],
                }
            )
            cases = [
                process_result(returncode=2, output=typed),
                process_result(returncode=124, output=typed, timed_out=True),
                process_result(
                    returncode=1,
                    output='{"ok": false, "errors": ["invalid manifest"]}',
                ),
                process_result(returncode=1, output="validator log only"),
                process_result(returncode=0, output=typed),
            ]
            for result in cases:
                with self.subTest(result=result):
                    toolchain = Toolchain(root, StaticRunner(result))
                    with self.assertRaises(ToolFailure) as raised:
                        toolchain.validate_background_retrieval(run_dir)
                    self.assertNotIsInstance(
                        raised.exception,
                        ValidationRejected,
                    )

    def test_validator_process_interruption_propagates_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interrupted = ProcessInterrupted(
                process_result(
                    returncode=130,
                    output="interrupted",
                    interrupted=True,
                )
            )
            toolchain = Toolchain(root, RaisingRunner(interrupted))

            with self.assertRaises(ProcessInterrupted) as raised:
                toolchain.validate_experience(
                    root / "runs" / "toy" / "smoke",
                    root / "experience.json",
                )

            self.assertIs(raised.exception, interrupted)

    def test_unreadable_defaults_and_malformed_experience_view_are_operational(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.py"
            toolchain = Toolchain(ROOT, ProcessRunner(), helper_timeout=30.0)
            with self.assertRaises(ToolFailure) as raised:
                toolchain.provided_baseline_defaults(missing)
            self.assertNotIsInstance(raised.exception, ValidationRejected)

            malformed = Toolchain(
                Path(tmp),
                StaticRunner(
                    process_result(returncode=0, output="helper log only")
                ),
            )
            with self.assertRaises(ToolFailure):
                malformed.experience_views(Path(tmp) / "run")

    def test_real_background_producers_type_content_but_not_catalog_io(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed_background = root / "background.md"
            malformed_background.write_text(
                "# Background\n\nNo registry was authored.\n",
                encoding="utf-8",
            )
            toolchain = Toolchain(ROOT, ProcessRunner(), helper_timeout=30.0)

            with self.assertRaises(ValidationRejected):
                toolchain._json_validation_python(
                    "tools/background_contract.py",
                    "validate",
                    "--background",
                    str(malformed_background),
                    label="background validation",
                    rejection_kind="background_validation",
                )

            invalid_catalog = root / "dimension_catalog.json"
            invalid_catalog.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(ValidationRejected):
                toolchain.background_catalog_receipt(invalid_catalog)

            catalog_directory = root / "catalog-directory"
            catalog_directory.mkdir()
            with self.assertRaises(ToolFailure) as raised:
                toolchain.background_catalog_receipt(catalog_directory)
            self.assertNotIsInstance(raised.exception, ValidationRejected)

            valid_background = root / "valid-background.md"
            valid_background.write_text(
                background_text(fixture_registry()),
                encoding="utf-8",
            )
            malformed_baselines = root / "baseline_mechanisms.json"
            malformed_baselines.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(ValidationRejected):
                toolchain._json_validation_python(
                    "tools/background_contract.py",
                    "validate",
                    "--background",
                    str(valid_background),
                    "--baseline-mechanisms",
                    str(malformed_baselines),
                    label="background validation",
                    rejection_kind="background_validation",
                )

    def test_real_semantic_helper_types_only_prediction_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            semantic_dir = run_dir / ".semantic" / "001"
            semantic_dir.mkdir(parents=True)
            ledger = {"records": []}
            proposals = build_proposal_set(
                fixture_registry(),
                ledger,
                op="fresh",
                parents=[],
                max_points=2,
            )
            proposals_path = semantic_dir / "proposals.json"
            predictions_path = semantic_dir / "predictions.json"
            atomic_write_json(proposals_path, proposals)
            atomic_write_json(
                predictions_path,
                {
                    "schema_version": 3,
                    "proposal_set_revision": proposals["proposal_set_revision"],
                    "experience": {
                        "generation": None,
                        "updated_at_run": None,
                        "revision": None,
                    },
                    "predictions": [],
                },
            )
            atomic_write_json(run_dir / "ledger.json", ledger)
            toolchain = Toolchain(
                ROOT,
                ProcessRunner(),
                helper_timeout=30.0,
            )

            with self.assertRaises(ValidationRejected):
                toolchain.semantic_select(
                    run_dir,
                    proposals_path,
                    policy="gain",
                    predictions=predictions_path,
                )

            missing = semantic_dir / "missing-proposals.json"
            with self.assertRaises(ToolFailure) as raised:
                toolchain.semantic_select(
                    run_dir,
                    missing,
                    policy="gain",
                    predictions=predictions_path,
                )
            self.assertNotIsInstance(raised.exception, ValidationRejected)

            atomic_write_json(
                run_dir / "ledger.json",
                {
                    "records": [],
                    "experience": {"schema_version": 999},
                },
            )
            with self.assertRaises(ToolFailure) as raised:
                toolchain.semantic_select(
                    run_dir,
                    proposals_path,
                    policy="gain",
                    predictions=predictions_path,
                )
            self.assertNotIsInstance(raised.exception, ValidationRejected)

    def test_experience_correction_runs_only_for_validation_rejection(self) -> None:
        proposal = {"schema_version": 4}
        rejection = ValidationRejected(
            "experience validation",
            process_result(
                returncode=1,
                output='{"ok": false, "errors": ["bad snapshot"]}',
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            models = ScriptedModels([proposal, proposal])
            toolchain = ExperienceToolchainStub(rejection)

            result = ExperienceRefresh(identity, toolchain, models).run(
                {"dag_revision": 1}
            )

            self.assertEqual(result, {"ok": True})
            self.assertEqual(toolchain.validation_calls, 2)
            self.assertEqual(toolchain.store_calls, 1)
            self.assertEqual(
                [call["purpose"] for call in models.calls],
                ["experience_refresh:dag-1", "experience_refresh:dag-1:correction"],
            )

        infrastructure = ToolFailure(
            "experience validation",
            process_result(returncode=2, output="validator worker unavailable"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            models = ScriptedModels([proposal, proposal])
            toolchain = ExperienceToolchainStub(infrastructure)

            with self.assertRaisesRegex(ToolFailure, "worker unavailable"):
                ExperienceRefresh(identity, toolchain, models).run(
                    {"dag_revision": 1}
                )

            self.assertEqual(toolchain.validation_calls, 1)
            self.assertEqual(toolchain.store_calls, 0)
            self.assertEqual(
                [call["purpose"] for call in models.calls],
                ["experience_refresh:dag-1"],
            )

    def test_semantic_correction_runs_only_for_validation_rejection(self) -> None:
        prediction = {"schema_version": 3}
        rejection = ValidationRejected(
            "semantic selection",
            process_result(
                returncode=1,
                output='{"ok": false, "errors": ["bad predictions"]}',
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            identity, proposals = self._semantic_case(Path(tmp))
            models = ScriptedModels([prediction, prediction])
            toolchain = SemanticToolchainStub(rejection)
            admission = SemanticAdmission(identity, toolchain, models)

            admission._select_with_predictions(
                run_id="001",
                policy="gain",
                proposals=proposals,
            )

            self.assertEqual(toolchain.selection_calls, 2)
            self.assertEqual(
                [call["purpose"] for call in models.calls],
                ["semantic_predictions:001", "semantic_predictions:001:correction"],
            )

        infrastructure = ToolFailure(
            "semantic selection",
            process_result(returncode=2, output="selector worker unavailable"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            identity, proposals = self._semantic_case(Path(tmp))
            models = ScriptedModels([prediction, prediction])
            toolchain = SemanticToolchainStub(infrastructure)
            admission = SemanticAdmission(identity, toolchain, models)

            with self.assertRaisesRegex(ToolFailure, "worker unavailable"):
                admission._select_with_predictions(
                    run_id="001",
                    policy="gain",
                    proposals=proposals,
                )

            self.assertEqual(toolchain.selection_calls, 1)
            self.assertEqual(
                [call["purpose"] for call in models.calls],
                ["semantic_predictions:001"],
            )

    @staticmethod
    def _semantic_case(root: Path) -> tuple[RunIdentity, Path]:
        identity = RunIdentity(root, "toy", "smoke")
        task_dir = root / "tasks" / "toy"
        proposal_dir = identity.run_dir / ".semantic" / "001"
        task_dir.mkdir(parents=True)
        proposal_dir.mkdir(parents=True)
        (task_dir / "TASK.md").write_text("# toy\n", encoding="utf-8")
        (task_dir / "task.toml").write_text("", encoding="utf-8")
        (identity.run_dir / "background.md").write_text("# background\n")
        (identity.run_dir / "ledger.json").write_text("{}\n")
        proposals = proposal_dir / "proposals.json"
        proposals.write_text("{}\n", encoding="utf-8")
        return identity, proposals


if __name__ == "__main__":
    unittest.main()
