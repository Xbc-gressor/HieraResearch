from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hieraresearch.artifacts import (  # noqa: E402
    ArtifactError,
    InvocationJournal,
    atomic_write_json,
    atomic_write_text as real_atomic_write_text,
    file_revision,
    paths_revision,
)
from hieraresearch.candidate import (  # noqa: E402
    CandidateBuildError,
    CandidatePipeline,
    SourceValidationRejected,
    TuningValues,
)
from hieraresearch.llm import (  # noqa: E402
    InferenceContractError,
    InferenceError,
    InferenceRequestError,
    ModelGateway,
)
from hieraresearch.models import RoundAction, RunIdentity  # noqa: E402
from hieraresearch.process import ProcessResult, ProcessRunner  # noqa: E402
from hieraresearch.schemas import tuning_values_schema  # noqa: E402
from hieraresearch.toolchain import (  # noqa: E402
    ToolFailure,
    Toolchain,
    ValidationRejected,
    exact_command_string,
)


COMPUTED_SCHEMA = """\
KIND = "int"
PARAM_SCHEMA = {"depth": KIND}

def make_model(params):
    return params["depth"]
"""

VALID_SCHEMA = """\
PARAM_SCHEMA = {
    "depth": "int",
}

def make_model(params):
    return params["depth"]
"""

PROVIDED_SCHEMA = """\
DEFAULT_PARAMS = {
    "depth": 3,
    "learning_rate": 0.1,
    "solver": "lbfgs",
}

PARAM_SCHEMA = {
    "depth": "int",
    "learning_rate": "float",
    "solver": ["categorical", ["lbfgs", "adam"]],
}

def make_model(params):
    return params["depth"]
"""


def tuning_response(
    *values: int | float,
    kind: str = "int",
    low: int | float = 1,
    high: int | float = 5,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "warm_configs": [
            [{"key": "depth", "value": value}] for value in values
        ],
        "search_space": [
            {
                "key": "depth",
                "kind": kind,
                "low": low,
                "high": high,
                "log": False,
                "options": [],
            }
        ],
    }


def process_result(*, returncode: int, output: str) -> ProcessResult:
    return ProcessResult(
        args=("contract-test",),
        returncode=returncode,
        output=output,
        elapsed_seconds=0.0,
    )


class ScriptedModels:
    """Small ModelGateway stand-in that still runs the production parsers/gates."""

    def __init__(
        self,
        *,
        edit_sources: list[str | BaseException] | None = None,
        infer_responses: list[Any] | None = None,
        canonical_path: Path | None = None,
    ):
        self.edit_sources = list(edit_sources or [])
        self.infer_responses = list(infer_responses or [])
        self.canonical_path = canonical_path
        self.edit_specs = []
        self.infer_calls: list[dict[str, Any]] = []
        self.canonical_snapshots: list[str] = []

    def edit(self, spec, *, validate):
        self.edit_specs.append(spec)
        if self.canonical_path is not None:
            self.canonical_snapshots.append(
                self.canonical_path.read_text(encoding="utf-8")
            )
        if not self.edit_sources:
            raise AssertionError(f"unexpected edit call: {spec.purpose}")
        source = self.edit_sources.pop(0)
        if isinstance(source, BaseException):
            raise source
        if len(spec.write_paths) != 1:
            raise AssertionError("schema editor must have exactly one write target")
        spec.write_paths[0].write_text(source, encoding="utf-8")
        return validate()

    def infer(self, **kwargs):
        self.infer_calls.append(kwargs)
        if not self.infer_responses:
            raise AssertionError(f"unexpected inference call: {kwargs['purpose']}")
        response = self.infer_responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        try:
            return kwargs["parser"](response)
        except (TypeError, ValueError, KeyError) as exc:
            raise InferenceContractError(str(exc)) from exc


class RejectFirstSpaceToolchain(Toolchain):
    def __init__(self, first_failure: ToolFailure):
        super().__init__(ROOT, ProcessRunner(), helper_timeout=30.0)
        self.first_failure = first_failure
        self.check_calls = 0

    def check_search_space(self, candidate_path, space_path, configs_path):
        self.check_calls += 1
        if self.check_calls == 1:
            raise self.first_failure
        return super().check_search_space(candidate_path, space_path, configs_path)


class CrashBeforeAppliedReceiptPipeline(CandidatePipeline):
    """Expose the recovery window after candidate publish, before final receipt."""

    def _write_tuning_values_receipt(self, run_id: str, *, status: str, **kwargs):
        if status == "applied":
            raise RuntimeError("simulated interruption before applied receipt")
        return super()._write_tuning_values_receipt(
            run_id,
            status=status,
            **kwargs,
        )


class CrashBeforeSchemaReceiptPipeline(CandidatePipeline):
    """Expose the window after schema publish, before its ready receipt."""

    def _write_tuning_schema_receipt(self, *args, **kwargs):
        raise RuntimeError("simulated interruption before schema receipt")


class CrashBeforeContractFinalizationPipeline(CandidatePipeline):
    """Expose finalizing state before any contract publication side effect."""

    def _finalize_contract(self, *args, **kwargs):
        raise RuntimeError("simulated interruption before contract finalization")


class FailDraftSchemaValidatorOnceToolchain(Toolchain):
    def __init__(self):
        super().__init__(ROOT, ProcessRunner(), helper_timeout=30.0)
        self.lint_calls = 0

    def lint_schema(self, candidate_path):
        self.lint_calls += 1
        if self.lint_calls == 2:
            raise ToolFailure(
                "schema lint",
                process_result(returncode=2, output="validator worker unavailable"),
            )
        return super().lint_schema(candidate_path)


class MutateInheritanceThenFailOnceToolchain(Toolchain):
    def __init__(self):
        super().__init__(ROOT, ProcessRunner(), helper_timeout=30.0)
        self.inheritance_calls = 0
        self.lineage_calls = 0
        self.fail_lineage = False

    def lineage_evidence(self, run_dir, source_run_ids):
        self.lineage_calls += 1
        if self.fail_lineage:
            raise ToolFailure(
                "lineage evidence",
                process_result(returncode=2, output="lineage worker unavailable"),
            )
        return {"source_run_ids": list(source_run_ids), "parameters": {}}

    def build_inheritance(self, candidate_path, configs_path):
        self.inheritance_calls += 1
        configs = json.loads(Path(configs_path).read_text())
        configs[0] = {"depth": 4}
        atomic_write_json(Path(configs_path), configs)
        if self.inheritance_calls == 1:
            raise ToolFailure(
                "parameter inheritance",
                process_result(returncode=2, output="interrupted after config write"),
            )
        return {"ok": True}


class LedgerIdeaToolchain(Toolchain):
    def __init__(self):
        super().__init__(ROOT, ProcessRunner(), helper_timeout=30.0)
        self.lineage_calls = 0

    def lineage_evidence(self, run_dir, source_run_ids):
        self.lineage_calls += 1
        ledger = json.loads((Path(run_dir) / "ledger.json").read_text())
        records = {
            record["run_id"]: record
            for record in ledger.get("records", [])
        }
        return {
            "per_parent": {
                parent: {"idea": records.get(parent, {}).get("idea")}
                for parent in source_run_ids
            }
        }

    def build_inheritance(self, candidate_path, configs_path):
        return {"ok": True}


class ProvidedDefaultsUnavailableToolchain(Toolchain):
    def __init__(self):
        super().__init__(ROOT, ProcessRunner(), helper_timeout=30.0)

    def provided_baseline_defaults(self, candidate_path):
        raise ToolFailure(
            "provided baseline defaults",
            process_result(returncode=2, output="defaults worker unavailable"),
        )


class CandidateContractPipelineTests(unittest.TestCase):
    def _case(
        self,
        root: Path,
        *,
        train_source: str,
        k: int = 1,
        provided: bool = False,
    ) -> tuple[RunIdentity, RoundAction, Path]:
        identity = RunIdentity(root, "toy", "smoke")
        task_dir = root / "tasks" / "toy"
        candidate_dir = identity.run_dir / "candidates" / "001"
        task_dir.mkdir(parents=True)
        candidate_dir.mkdir(parents=True)
        (task_dir / "TASK.md").write_text("# toy contract\n", encoding="utf-8")
        (task_dir / "task.toml").write_text("", encoding="utf-8")
        (candidate_dir / "prepare.py").write_text("", encoding="utf-8")
        (candidate_dir / "train.py").write_text(train_source, encoding="utf-8")
        if provided:
            (task_dir / "train.py").write_text(train_source, encoding="utf-8")
        atomic_write_json(
            candidate_dir / "_candidate_brief.json",
            {
                "schema_version": 4,
                "run_id": "001",
                "op": "fresh",
                "source_run_ids": [],
                "implementation_source": {
                    "kind": (
                        "provided_entrypoint" if provided else "generated"
                    ),
                    **(
                        {
                            "path": "tasks/toy/train.py",
                            "sha256": file_revision(task_dir / "train.py"),
                        }
                        if provided
                        else {}
                    ),
                },
            },
        )
        atomic_write_json(identity.run_dir / "framework_cfg.json", {"tuner": {"K": k}})
        atomic_write_json(identity.run_dir / "ledger.json", {"records": []})
        return identity, RoundAction(op="fresh", run_id="001", admitted=True), candidate_dir

    @staticmethod
    def _toolchain() -> Toolchain:
        return Toolchain(ROOT, ProcessRunner(), helper_timeout=30.0)

    def test_schema_repairs_stay_in_draft_then_stage_b_publishes_canonical_files(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=COMPUTED_SCHEMA, k=2
            )
            candidate_path = candidate_dir / "train.py"
            models = ScriptedModels(
                edit_sources=[COMPUTED_SCHEMA, VALID_SCHEMA],
                infer_responses=[tuning_response(2, 4)],
                canonical_path=candidate_path,
            )
            pipeline = CandidatePipeline(identity, self._toolchain(), models, {})

            pipeline.build_contract(action)

            draft_path = candidate_dir / pipeline.TUNING_SCHEMA_DRAFT
            self.assertEqual(len(models.edit_specs), 2)
            self.assertTrue(
                all(spec.write_paths == (draft_path,) for spec in models.edit_specs)
            )
            expected_lint_command = exact_command_string(
                [
                    sys.executable,
                    str(ROOT / "tools" / "tuners" / "tune_tools.py"),
                    "lint-schema",
                    "--candidate-path",
                    str(draft_path),
                ]
            )
            self.assertTrue(
                all("Bash" in spec.tools for spec in models.edit_specs)
            )
            self.assertTrue(
                all(
                    spec.allowed_commands == frozenset({expected_lint_command})
                    for spec in models.edit_specs
                )
            )
            self.assertEqual(models.canonical_snapshots, [COMPUTED_SCHEMA] * 2)
            self.assertIn("SEARCH_SPACE", candidate_path.read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads((candidate_dir / "_warm_configs.json").read_text()),
                [{"depth": 2}, {"depth": 4}],
            )
            self.assertEqual(
                json.loads((candidate_dir / "_search_space.json").read_text()),
                {"depth": ["int", 1, 5]},
            )
            self.assertTrue(pipeline.contract_is_ready(action))
            self.assertTrue(pipeline.tuning_values_are_ready(action))

    def test_candidate_implementation_is_published_only_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source="SEED = True\n"
            )
            candidate_path = candidate_dir / "train.py"
            models = ScriptedModels(
                edit_sources=["def broken(:\n", "VALUE = 1\n"],
                canonical_path=candidate_path,
            )
            pipeline = CandidatePipeline(identity, self._toolchain(), models, {})

            published = pipeline.implement(action)

            draft_path = candidate_dir / pipeline.IMPLEMENTATION_DRAFT
            self.assertEqual(published, candidate_path)
            self.assertEqual(models.canonical_snapshots, ["SEED = True\n"] * 2)
            self.assertTrue(
                all(spec.write_paths == (draft_path,) for spec in models.edit_specs)
            )
            expected_validate_command = exact_command_string(
                [
                    sys.executable,
                    str(ROOT / "tools" / "validate_candidate_source.py"),
                    "--path",
                    str(draft_path),
                ]
            )
            self.assertTrue(
                all("Bash" in spec.tools for spec in models.edit_specs)
            )
            self.assertTrue(
                all(
                    spec.allowed_commands == frozenset({expected_validate_command})
                    for spec in models.edit_specs
                )
            )
            self.assertEqual(candidate_path.read_text(), "VALUE = 1\n")
            self.assertTrue(pipeline.implementation_is_ready(action))

    def test_agent_source_validator_command_matches_post_call_gate(self) -> None:
        cases = [
            # py_compile accepts a BOM; the strict gate must reject it.
            ("bom", b"\xef\xbb\xbfVALUE = 1\n", False),
            # py_compile honors the PEP 263 cookie and rejects UTF-8 bytes;
            # the strict gate decodes UTF-8 first and must accept.
            (
                "ascii_cookie_utf8",
                "# coding: ascii\nVALUE = 'café'\n".encode("utf-8"),
                True,
            ),
            ("non_ascii_utf8", "VALUE = 'café'  # naïve\n".encode("utf-8"), True),
            ("syntax_error", b"def broken(:\n", False),
            ("empty", b"\n", False),
            ("latin1_bytes", "VALUE = 'café'\n".encode("latin-1"), False),
            ("missing", None, False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = RunIdentity(root, "toy", "source-equivalence")
            toolchain = self._toolchain()
            pipeline = CandidatePipeline(identity, toolchain, ScriptedModels(), {})
            for name, payload, accepted in cases:
                with self.subTest(case=name):
                    path = root / f"{name}.py"
                    if payload is not None:
                        path.write_bytes(payload)
                    command = toolchain.candidate_source_validate_command(path)
                    completed = subprocess.run(
                        command, shell=True, capture_output=True, text=True
                    )
                    gate_error = None
                    try:
                        pipeline._validate_authored_python(path)
                    except SourceValidationRejected as exc:
                        gate_error = exc
                    self.assertEqual(completed.returncode == 0, accepted)
                    self.assertEqual(gate_error is None, accepted)
                    if not accepted:
                        verdict = json.loads(completed.stdout)
                        self.assertFalse(verdict["ok"])
                        self.assertEqual(
                            verdict["failure_kind"], "candidate_source_validation"
                        )

    def test_operational_ledger_phase_changes_do_not_reset_writer_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source="SEED = True\n"
            )
            non_upstream = InferenceError(
                "Claude Agent SDK edit ended with error_max_turns: "
                "max turns reached before a final result"
            )
            first_models = ScriptedModels(
                edit_sources=[non_upstream, non_upstream]
            )
            first = CandidatePipeline(
                identity,
                self._toolchain(),
                first_models,
                {},
            )

            with self.assertRaisesRegex(CandidateBuildError, "max turns reached"):
                first.implement(action)

            self.assertEqual(
                [spec.purpose for spec in first_models.edit_specs],
                ["candidate_writer:001", "candidate_writer:001:repair"],
            )
            state = json.loads(
                (candidate_dir / first.IMPLEMENTATION_STATE).read_text()
            )
            self.assertEqual(state["attempts_admitted"], 2)
            self.assertEqual(state["status"], "rejected")

            atomic_write_json(
                identity.run_dir / "ledger.json",
                {"phase": "blocked", "records": []},
            )
            no_third_edit = ScriptedModels(edit_sources=["VALUE = 1\n"])
            exhausted = CandidatePipeline(
                identity,
                self._toolchain(),
                no_third_edit,
                {},
            )
            with self.assertRaisesRegex(CandidateBuildError, "max turns reached"):
                exhausted.implement(action)

            state = json.loads(
                (candidate_dir / exhausted.IMPLEMENTATION_STATE).read_text()
            )
            self.assertEqual(state["attempts_admitted"], 2)
            self.assertEqual(no_third_edit.edit_specs, [])

    def test_upstream_transport_exhaustion_reopens_after_outer_recovery(
        self,
    ) -> None:
        """Coordinator backoff re-entry must issue another provider call on 502."""
        upstream = InferenceError(
            "Claude Messages request failed: Error code: 502 - "
            "{'error': {'message': 'Upstream service temporarily unavailable', "
            "'type': 'upstream_error'}, 'type': 'error'}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source="SEED = True\n"
            )
            first_models = ScriptedModels(edit_sources=[upstream])
            with self.assertRaisesRegex(InferenceError, "Error code: 502"):
                CandidatePipeline(
                    identity, self._toolchain(), first_models, {}
                ).implement(action)

            second_models = ScriptedModels(edit_sources=[upstream])
            with self.assertRaisesRegex(InferenceError, "Error code: 502"):
                CandidatePipeline(
                    identity, self._toolchain(), second_models, {}
                ).implement(action)

            # Local window exhausted — without outer reopen this would raise
            # retry-limit and never call the model.
            recovered_models = ScriptedModels(edit_sources=["VALUE = 1\n"])
            published = CandidatePipeline(
                identity, self._toolchain(), recovered_models, {}
            ).implement(action)

            self.assertEqual(published, candidate_dir / "train.py")
            self.assertEqual(len(recovered_models.edit_specs), 1)
            self.assertEqual(
                (candidate_dir / "train.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )
            state = json.loads(
                (candidate_dir / CandidatePipeline.IMPLEMENTATION_STATE).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(state["status"], "completed")

    def test_schema_transport_retry_resets_partial_draft_and_reuses_purpose(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=COMPUTED_SCHEMA
            )

            class PartialFailureModels(ScriptedModels):
                def edit(self, spec, *, validate):
                    del validate
                    self.edit_specs.append(spec)
                    spec.write_paths[0].write_text(
                        "PARTIAL_INFRASTRUCTURE_EDIT = True\n",
                        encoding="utf-8",
                    )
                    raise InferenceError(
                        "Claude Agent SDK edit failed: Error code: 502 - "
                        "{'error': {'message': 'Upstream service temporarily "
                        "unavailable', 'type': 'upstream_error'}, 'type': 'error'}"
                    )

            first_models = PartialFailureModels()
            first = CandidatePipeline(
                identity,
                self._toolchain(),
                first_models,
                {},
            )
            with self.assertRaisesRegex(InferenceError, "Error code: 502"):
                first.build_contract(action)

            class InspectingModels(ScriptedModels):
                def __init__(self):
                    super().__init__(
                        edit_sources=[VALID_SCHEMA],
                        infer_responses=[tuning_response(3)],
                    )
                    self.draft_before_edit: list[str] = []

                def edit(self, spec, *, validate):
                    self.draft_before_edit.append(
                        spec.write_paths[0].read_text(encoding="utf-8")
                    )
                    return super().edit(spec, validate=validate)

            resumed_models = InspectingModels()
            resumed = CandidatePipeline(
                identity,
                self._toolchain(),
                resumed_models,
                {},
            )
            resumed.build_contract(action)

            self.assertEqual(
                [first_models.edit_specs[0].purpose, resumed_models.edit_specs[0].purpose],
                ["tuning_schema:001", "tuning_schema:001"],
            )
            self.assertEqual(resumed_models.draft_before_edit, [COMPUTED_SCHEMA])
            self.assertTrue(resumed.contract_is_ready(action))

    def test_schema_receipt_survives_stage_b_infrastructure_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=COMPUTED_SCHEMA
            )
            candidate_path = candidate_dir / "train.py"
            interrupted_models = ScriptedModels(
                edit_sources=[VALID_SCHEMA],
                infer_responses=[InferenceError("messages API unavailable")],
                canonical_path=candidate_path,
            )
            first = CandidatePipeline(
                identity,
                self._toolchain(),
                interrupted_models,
                {},
            )

            with self.assertRaisesRegex(CandidateBuildError, "unavailable"):
                first.build_contract(action)

            self.assertTrue(first.tuning_schema_is_ready(action))
            self.assertFalse(first.contract_is_ready(action))
            state = json.loads(
                (candidate_dir / first.TUNING_SCHEMA_STATE).read_text()
            )
            self.assertEqual(state["status"], "completed")
            atomic_write_json(
                identity.run_dir / "ledger.json",
                {"phase": "blocked", "records": []},
            )

            resumed_models = ScriptedModels(
                infer_responses=[tuning_response(3)],
                canonical_path=candidate_path,
            )
            resumed = CandidatePipeline(
                identity,
                self._toolchain(),
                resumed_models,
                {},
            )
            resumed.build_contract(action)

            self.assertEqual(resumed_models.edit_specs, [])
            self.assertEqual(len(resumed_models.infer_calls), 1)
            self.assertEqual(
                resumed_models.infer_calls[0]["purpose"],
                "tuning_values:001",
            )
            values_state = json.loads(
                (candidate_dir / resumed.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(values_state["attempts_admitted"], 1)
            self.assertTrue(resumed.contract_is_ready(action))

    def test_stage_b_transport_retries_are_bounded_without_using_correction(self) -> None:
        upstream = InferenceError(
            "Claude Messages request failed: Error code: 502 - "
            "{'error': {'message': 'Upstream service temporarily unavailable', "
            "'type': 'upstream_error'}, 'type': 'error'}"
        )
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            purposes: list[str] = []
            for _ in range(2):
                models = ScriptedModels(infer_responses=[upstream])
                pipeline = CandidatePipeline(
                    identity,
                    self._toolchain(),
                    models,
                    {},
                )
                with self.assertRaisesRegex(InferenceError, "Error code: 502"):
                    pipeline.build_contract(action)
                purposes.append(models.infer_calls[0]["purpose"])

            state = json.loads(
                (candidate_dir / CandidatePipeline.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(purposes, ["tuning_values:001"] * 2)
            self.assertEqual(state["attempts_admitted"], 1)
            self.assertEqual(state["transport_failures"], 2)

            # The coordinator's outer backoff owns upstream recovery; re-entry
            # reopens the local window and replays the admitted attempt without
            # consuming the correction slot.
            recovered_models = ScriptedModels(
                infer_responses=[tuning_response(3)]
            )
            recovered = CandidatePipeline(
                identity,
                self._toolchain(),
                recovered_models,
                {},
            )
            recovered.build_contract(action)

            self.assertEqual(len(recovered_models.infer_calls), 1)
            self.assertEqual(
                recovered_models.infer_calls[0]["purpose"], "tuning_values:001"
            )
            state = json.loads(
                (candidate_dir / recovered.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(state["attempts_admitted"], 1)
            self.assertTrue(recovered.contract_is_ready(action))

    def test_stage_b_non_upstream_inference_error_crashes_candidate_not_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            models = ScriptedModels(
                infer_responses=[InferenceError("messages API unavailable")]
            )
            pipeline = CandidatePipeline(identity, self._toolchain(), models, {})

            with self.assertRaisesRegex(CandidateBuildError, "unavailable"):
                pipeline.build_contract(action)

            state = json.loads(
                (candidate_dir / pipeline.TUNING_VALUES_STATE).read_text()
            )
            # A non-transport inference failure is not billed to the upstream
            # transport window, and the admitted attempt stays replayable.
            self.assertEqual(state["transport_failures"], 0)
            self.assertEqual(state["status"], "infrastructure_failed")
            self.assertEqual(state["attempts_admitted"], 1)

            recovered_models = ScriptedModels(infer_responses=[tuning_response(3)])
            recovered = CandidatePipeline(
                identity, self._toolchain(), recovered_models, {}
            )
            recovered.build_contract(action)
            self.assertEqual(
                [call["purpose"] for call in recovered_models.infer_calls],
                ["tuning_values:001"],
            )
            self.assertTrue(recovered.contract_is_ready(action))

    def test_schema_non_upstream_inference_error_consumes_repairs_then_crashes_candidate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=COMPUTED_SCHEMA
            )
            max_turns = InferenceError(
                "Claude Agent SDK edit ended with error_max_turns: "
                "max turns reached before a final result"
            )
            models = ScriptedModels(edit_sources=[max_turns] * 3)
            pipeline = CandidatePipeline(identity, self._toolchain(), models, {})

            with self.assertRaisesRegex(CandidateBuildError, "max turns reached"):
                pipeline.build_contract(action)

            self.assertEqual(
                [spec.purpose for spec in models.edit_specs],
                [
                    "tuning_schema:001",
                    "tuning_schema:001:repair:1",
                    "tuning_schema:001:repair:2",
                ],
            )
            state = json.loads(
                (candidate_dir / pipeline.TUNING_SCHEMA_STATE).read_text()
            )
            self.assertEqual(state["attempts_admitted"], 3)
            self.assertEqual(state["status"], "rejected")

            no_more_edits = ScriptedModels(edit_sources=[VALID_SCHEMA])
            exhausted = CandidatePipeline(
                identity, self._toolchain(), no_more_edits, {}
            )
            with self.assertRaisesRegex(CandidateBuildError, "max turns reached"):
                exhausted.build_contract(action)
            self.assertEqual(no_more_edits.edit_specs, [])

    def test_values_correction_prompt_inlines_the_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, _ = self._case(Path(tmp), train_source=VALID_SCHEMA)
            rejection = ValidationRejected(
                "search-space validation",
                process_result(returncode=1, output="distinctive-contract-violation"),
            )
            toolchain = RejectFirstSpaceToolchain(rejection)
            models = ScriptedModels(
                infer_responses=[tuning_response(3), tuning_response(3)]
            )
            pipeline = CandidatePipeline(identity, toolchain, models, {})

            pipeline.build_contract(action)

            self.assertEqual(len(models.infer_calls), 2)
            correction = models.infer_calls[1]
            self.assertEqual(correction["purpose"], "tuning_values:001:correction:1")
            # The tool-less correction call cannot read files: the durable
            # diagnostic travels inside the prompt, not as a path reference.
            self.assertIn("distinctive-contract-violation", correction["prompt"])
            diagnostic_path = (
                identity.run_dir
                / ".orchestrator"
                / "candidate_contract_diagnostics"
                / "001-values.json"
            )
            self.assertTrue(diagnostic_path.is_file())
            self.assertNotIn(str(diagnostic_path), correction["prompt"])

    def test_stage_b_request_rejection_replays_only_after_schema_changes(self) -> None:
        class RequestBackend:
            def __init__(self):
                self.calls = 0

            def generate(self, **kwargs):
                del kwargs
                self.calls += 1
                if self.calls == 1:
                    raise InferenceRequestError("provider rejected schema")
                return tuning_response(3), {"request_id": "fixed-schema"}

        class NoEditor:
            def edit(self, **kwargs):
                raise AssertionError(f"unexpected edit: {kwargs}")

        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            backend = RequestBackend()
            gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(identity.run_dir),
                structured_backend=backend,
                edit_backend=NoEditor(),
            )
            first = CandidatePipeline(
                identity,
                self._toolchain(),
                gateway,
                {},
            )

            with self.assertRaisesRegex(
                InferenceRequestError, "provider rejected schema"
            ):
                first.build_contract(action)
            state = json.loads(
                (candidate_dir / first.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(state["status"], "request_failed")
            self.assertEqual(state["transport_failures"], 0)

            unchanged = CandidatePipeline(
                identity,
                self._toolchain(),
                gateway,
                {},
            )
            with self.assertRaisesRegex(
                InferenceRequestError, "recorded non-retryable"
            ):
                unchanged.build_contract(action)
            self.assertEqual(backend.calls, 1)

            original_schema = tuning_values_schema(1)
            fixed_schema = {**original_schema, "$comment": "provider-compatible-v2"}
            resumed = CandidatePipeline(
                identity,
                self._toolchain(),
                gateway,
                {},
            )
            with patch(
                "hieraresearch.candidate.tuning_values_schema",
                return_value=fixed_schema,
            ):
                resumed.build_contract(action)

            self.assertEqual(backend.calls, 2)
            self.assertTrue(resumed.contract_is_ready(action))

    def test_stage_b_replay_uses_frozen_lineage_after_ledger_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            action.op = "improve"
            action.parents = ["000"]
            brief = json.loads((candidate_dir / "_candidate_brief.json").read_text())
            brief["op"] = "improve"
            brief["source_run_ids"] = ["000"]
            atomic_write_json(candidate_dir / "_candidate_brief.json", brief)
            parent_dir = identity.run_dir / "candidates" / "000"
            parent_dir.mkdir(parents=True)
            (parent_dir / "train.py").write_text(VALID_SCHEMA, encoding="utf-8")
            atomic_write_json(parent_dir / "tune_report.json", {"phase_a": {}})
            atomic_write_json(
                identity.run_dir / "ledger.json",
                {"records": [{"run_id": "000", "idea": "alpha"}]},
            )
            toolchain = LedgerIdeaToolchain()
            failed_models = ScriptedModels(
                infer_responses=[InferenceError("messages API unavailable")]
            )
            first = CandidatePipeline(identity, toolchain, failed_models, {})

            with self.assertRaisesRegex(CandidateBuildError, "unavailable"):
                first.build_contract(action)

            first_prompt = failed_models.infer_calls[0]["prompt"]
            atomic_write_json(
                identity.run_dir / "ledger.json",
                {"records": [{"run_id": "000", "idea": "beta"}]},
            )
            resumed_models = ScriptedModels(
                infer_responses=[tuning_response(3)]
            )
            resumed = CandidatePipeline(identity, toolchain, resumed_models, {})

            resumed.build_contract(action)

            self.assertEqual(toolchain.lineage_calls, 1)
            self.assertEqual(resumed_models.infer_calls[0]["prompt"], first_prompt)
            self.assertIn('"idea": "alpha"', first_prompt)
            self.assertNotIn('"idea": "beta"', first_prompt)
            receipt = json.loads(
                (candidate_dir / resumed.TUNING_VALUES_RECEIPT).read_text()
            )
            self.assertEqual(
                receipt["lineage_evidence_revision"],
                file_revision(candidate_dir / resumed.TUNING_LINEAGE_EVIDENCE),
            )
            self.assertTrue(resumed.contract_is_ready(action))
            atomic_write_json(
                candidate_dir / resumed.TUNING_LINEAGE_EVIDENCE,
                {"tampered": True},
            )
            self.assertFalse(resumed.tuning_values_are_ready(action))
            self.assertFalse(resumed.contract_is_ready(action))

    def test_schema_publish_intent_survives_post_publish_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=COMPUTED_SCHEMA
            )
            candidate_path = candidate_dir / "train.py"
            models = ScriptedModels(
                edit_sources=[VALID_SCHEMA],
                canonical_path=candidate_path,
            )
            interrupted = CrashBeforeSchemaReceiptPipeline(
                identity,
                self._toolchain(),
                models,
                {},
            )
            interrupted._write_implementation_receipt(
                action.run_id or "",
                candidate_path,
                "agent_sdk",
                input_revision=paths_revision(
                    interrupted._candidate_authoring_context_paths(action)
                ),
            )
            implementation_revision = file_revision(candidate_path)

            with self.assertRaisesRegex(RuntimeError, "schema receipt"):
                interrupted.build_contract(action)

            state = json.loads(
                (candidate_dir / interrupted.TUNING_SCHEMA_STATE).read_text()
            )
            self.assertEqual(state["status"], "publishing")
            self.assertEqual(candidate_path.read_text(), VALID_SCHEMA)
            self.assertTrue(interrupted.implementation_is_ready(action))

            resumed_models = ScriptedModels(
                infer_responses=[tuning_response(3)],
                canonical_path=candidate_path,
            )
            resumed = CandidatePipeline(
                identity,
                self._toolchain(),
                resumed_models,
                {},
            )
            resumed.build_contract(action)

            self.assertEqual(resumed_models.edit_specs, [])
            completed_state = json.loads(
                (candidate_dir / resumed.TUNING_SCHEMA_STATE).read_text()
            )
            schema_receipt = json.loads(
                (candidate_dir / resumed.TUNING_SCHEMA_RECEIPT).read_text()
            )
            self.assertEqual(completed_state["status"], "completed")
            self.assertEqual(
                completed_state["source_candidate_revision"],
                implementation_revision,
            )
            self.assertEqual(
                schema_receipt["source_candidate_revision"],
                implementation_revision,
            )
            self.assertTrue(resumed.contract_is_ready(action))

    def test_schema_validator_outage_retries_validator_without_model_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=COMPUTED_SCHEMA
            )
            toolchain = FailDraftSchemaValidatorOnceToolchain()
            first_models = ScriptedModels(edit_sources=[VALID_SCHEMA])
            first = CandidatePipeline(identity, toolchain, first_models, {})

            with self.assertRaisesRegex(ToolFailure, "validator worker"):
                first.build_contract(action)

            state = json.loads(
                (candidate_dir / first.TUNING_SCHEMA_STATE).read_text()
            )
            self.assertEqual(state["status"], "validator_failed")
            self.assertEqual(state["attempts_admitted"], 1)

            resumed_models = ScriptedModels(
                infer_responses=[tuning_response(3)]
            )
            resumed = CandidatePipeline(identity, toolchain, resumed_models, {})
            resumed.build_contract(action)

            self.assertEqual(resumed_models.edit_specs, [])
            self.assertEqual(toolchain.lint_calls, 4)
            self.assertTrue(resumed.contract_is_ready(action))

    def test_rejected_schema_draft_tampering_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=COMPUTED_SCHEMA
            )
            pipeline = CandidatePipeline(
                identity,
                self._toolchain(),
                ScriptedModels(
                    edit_sources=[COMPUTED_SCHEMA, COMPUTED_SCHEMA, COMPUTED_SCHEMA]
                ),
                {},
            )

            with self.assertRaisesRegex(CandidateBuildError, "schema failed"):
                pipeline.build_contract(action)

            (candidate_dir / pipeline.TUNING_SCHEMA_DRAFT).write_text(
                VALID_SCHEMA,
                encoding="utf-8",
            )
            resumed = CandidatePipeline(
                identity,
                self._toolchain(),
                ScriptedModels(),
                {},
            )
            with self.assertRaisesRegex(ArtifactError, "outside authoring"):
                resumed.build_contract(action)

    def test_applying_values_receipt_forward_completes_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            models = ScriptedModels(infer_responses=[tuning_response(3)])
            interrupted = CrashBeforeAppliedReceiptPipeline(
                identity,
                self._toolchain(),
                models,
                {},
            )

            with self.assertRaisesRegex(RuntimeError, "applied receipt"):
                interrupted.build_contract(action)

            progress = json.loads(
                (candidate_dir / interrupted.TUNING_VALUES_RECEIPT).read_text()
            )
            self.assertEqual(progress["status"], "applying")
            self.assertTrue(interrupted.tuning_values_are_ready(action))
            self.assertFalse(interrupted.contract_is_ready(action))

            no_more_models = ScriptedModels()
            resumed = CandidatePipeline(
                identity,
                self._toolchain(),
                no_more_models,
                {},
            )
            resumed.build_contract(action)

            self.assertEqual(no_more_models.infer_calls, [])
            self.assertEqual(no_more_models.edit_specs, [])
            self.assertTrue(resumed.tuning_values_are_ready(action))
            self.assertTrue(resumed.contract_is_ready(action))

    def test_inheritance_side_effect_restores_from_finalization_intent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            action.op = "improve"
            action.parents = ["000"]
            brief = json.loads((candidate_dir / "_candidate_brief.json").read_text())
            brief["op"] = "improve"
            brief["source_run_ids"] = ["000"]
            atomic_write_json(candidate_dir / "_candidate_brief.json", brief)
            parent_dir = identity.run_dir / "candidates" / "000"
            parent_dir.mkdir(parents=True)
            (parent_dir / "train.py").write_text(VALID_SCHEMA, encoding="utf-8")
            atomic_write_json(parent_dir / "tune_report.json", {"phase_a": {}})
            toolchain = MutateInheritanceThenFailOnceToolchain()
            first = CandidatePipeline(
                identity,
                toolchain,
                ScriptedModels(infer_responses=[tuning_response(3)]),
                {},
            )

            with self.assertRaisesRegex(ToolFailure, "after config write"):
                first.build_contract(action)

            state = json.loads(
                (candidate_dir / first.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(state["status"], "finalizing")
            self.assertEqual(
                json.loads((candidate_dir / "_warm_configs.json").read_text()),
                [{"depth": 4}],
            )

            toolchain.fail_lineage = True
            resumed_models = ScriptedModels()
            resumed = CandidatePipeline(
                identity,
                toolchain,
                resumed_models,
                {},
            )
            resumed.build_contract(action)

            self.assertEqual(resumed_models.infer_calls, [])
            self.assertEqual(
                json.loads((candidate_dir / "_warm_configs.json").read_text()),
                [{"depth": 4}],
            )
            self.assertEqual(toolchain.inheritance_calls, 3)
            self.assertEqual(toolchain.lineage_calls, 1)
            self.assertTrue(resumed.contract_is_ready(action))

    def test_validated_value_drafts_forward_complete_a_split_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            pipeline = CandidatePipeline(
                identity,
                self._toolchain(),
                ScriptedModels(infer_responses=[tuning_response(3)]),
                {},
            )
            interrupted = False

            def interrupt_space_publish(path, value):
                nonlocal interrupted
                if Path(path).name == "_search_space.json" and not interrupted:
                    interrupted = True
                    real_atomic_write_text(
                        path,
                        json.dumps({"depth": ["int", 1, 2]}),
                    )
                    raise RuntimeError("simulated split value publication")
                return real_atomic_write_text(path, value)

            with patch(
                "hieraresearch.candidate.atomic_write_text",
                side_effect=interrupt_space_publish,
            ):
                with self.assertRaisesRegex(RuntimeError, "split value"):
                    pipeline.build_contract(action)

            state = json.loads(
                (candidate_dir / pipeline.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(state["status"], "draft_validated")
            self.assertTrue((candidate_dir / "_warm_configs.json").is_file())
            self.assertEqual(
                json.loads((candidate_dir / "_search_space.json").read_text()),
                {"depth": ["int", 1, 2]},
            )

            resumed_models = ScriptedModels()
            resumed = CandidatePipeline(
                identity,
                self._toolchain(),
                resumed_models,
                {},
            )
            resumed.build_contract(action)

            self.assertEqual(resumed_models.infer_calls, [])
            self.assertEqual(
                json.loads((candidate_dir / "_search_space.json").read_text()),
                {"depth": ["int", 1, 5]},
            )
            self.assertTrue(resumed.contract_is_ready(action))

    def test_provided_defaults_build_contract_without_model_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp),
                train_source=PROVIDED_SCHEMA,
                provided=True,
            )
            models = ScriptedModels()
            pipeline = CandidatePipeline(identity, self._toolchain(), models, {})

            pipeline.build_contract(action)

            configs = json.loads(
                (candidate_dir / "_warm_configs.json").read_text()
            )
            self.assertEqual(
                configs,
                [{"depth": 3, "learning_rate": 0.1, "solver": "lbfgs"}],
            )
            self.assertEqual(models.infer_calls, [])
            self.assertTrue(pipeline.contract_is_ready(action))

    def test_provided_finalization_recovery_does_not_reread_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp),
                train_source=PROVIDED_SCHEMA,
                provided=True,
            )
            interrupted = CrashBeforeContractFinalizationPipeline(
                identity,
                self._toolchain(),
                ScriptedModels(),
                {},
            )

            with self.assertRaisesRegex(RuntimeError, "contract finalization"):
                interrupted.build_contract(action)

            state = json.loads(
                (candidate_dir / interrupted.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(state["status"], "finalizing")
            resumed = CandidatePipeline(
                identity,
                ProvidedDefaultsUnavailableToolchain(),
                ScriptedModels(),
                {},
            )

            resumed.build_contract(action)

            self.assertTrue(resumed.contract_is_ready(action))

    def test_provided_implementation_receipt_does_not_hide_seed_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = RunIdentity(root, "toy", "smoke")
            task_dir = root / "tasks" / "toy"
            candidate_dir = identity.run_dir / "candidates" / "000"
            task_dir.mkdir(parents=True)
            candidate_dir.mkdir(parents=True)
            task_entrypoint = task_dir / "train.py"
            task_entrypoint.write_text(PROVIDED_SCHEMA, encoding="utf-8")
            (task_dir / "TASK.md").write_text("# toy\n", encoding="utf-8")
            (task_dir / "task.toml").write_text("", encoding="utf-8")
            (candidate_dir / "train.py").write_text(
                PROVIDED_SCHEMA,
                encoding="utf-8",
            )
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
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"tuner": {"K": 1}},
            )
            atomic_write_json(identity.run_dir / "ledger.json", {"records": []})
            action = RoundAction(op="fresh", run_id="000", admitted=True)
            task_config = {
                "candidate": {
                    "entrypoint": "train.py",
                    "copy_files": ["prepare.py", "train.py"],
                },
                "seed": {
                    "entrypoint": "train.py",
                    "provided": ["train.py"],
                },
            }
            pipeline = CandidatePipeline(
                identity,
                self._toolchain(),
                ScriptedModels(),
                task_config,
            )

            pipeline.implement(action)
            self.assertTrue(pipeline.implementation_is_ready(action))
            pipeline.build_contract(action)
            self.assertTrue(pipeline.tuning_values_are_ready(action))
            self.assertTrue(pipeline.contract_is_ready(action))

            task_entrypoint.write_text(
                PROVIDED_SCHEMA + "\nDRIFT = True\n",
                encoding="utf-8",
            )

            self.assertFalse(pipeline.implementation_is_ready(action))
            self.assertFalse(pipeline.tuning_values_are_ready(action))
            self.assertFalse(pipeline.contract_is_ready(action))
            with self.assertRaisesRegex(ArtifactError, "task seed"):
                pipeline.implement(action)

    def test_only_validation_rejection_gets_one_values_correction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, _ = self._case(Path(tmp), train_source=VALID_SCHEMA)
            rejection = ValidationRejected(
                "search-space validation",
                process_result(returncode=1, output="deterministic rejection"),
            )
            toolchain = RejectFirstSpaceToolchain(rejection)
            models = ScriptedModels(
                infer_responses=[tuning_response(3), tuning_response(3)]
            )
            pipeline = CandidatePipeline(identity, toolchain, models, {})

            pipeline.build_contract(action)

            self.assertEqual(toolchain.check_calls, 2)
            self.assertEqual(
                [call["purpose"] for call in models.infer_calls],
                ["tuning_values:001", "tuning_values:001:correction:1"],
            )

        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            infrastructure = ToolFailure(
                "search-space validation",
                process_result(returncode=2, output="worker could not start"),
            )
            toolchain = RejectFirstSpaceToolchain(infrastructure)
            models = ScriptedModels(
                infer_responses=[tuning_response(3), tuning_response(4)]
            )
            pipeline = CandidatePipeline(identity, toolchain, models, {})

            with self.assertRaisesRegex(ToolFailure, "worker could not start"):
                pipeline.build_contract(action)

            self.assertEqual(toolchain.check_calls, 1)
            self.assertEqual(
                [call["purpose"] for call in models.infer_calls],
                ["tuning_values:001"],
            )
            self.assertFalse((candidate_dir / pipeline.CONTRACT_RECEIPT).exists())
            state = json.loads(
                (candidate_dir / pipeline.TUNING_VALUES_STATE).read_text()
            )
            self.assertEqual(state["status"], "validator_failed")

            resumed_models = ScriptedModels()
            resumed = CandidatePipeline(
                identity,
                toolchain,
                resumed_models,
                {},
            )
            resumed.build_contract(action)

            self.assertEqual(resumed_models.infer_calls, [])
            self.assertEqual(toolchain.check_calls, 2)
            self.assertTrue(resumed.contract_is_ready(action))

    def test_parser_rejects_huge_integer_used_as_float_bound(self) -> None:
        huge = 1 << 100_000

        with self.assertRaisesRegex(ValueError, "finite bounds"):
            TuningValues.from_response(
                tuning_response(1.0, kind="float", low=huge, high=huge),
                k=1,
                expected_keys={"depth"},
            )

    def test_tuning_values_transport_schema_keeps_exact_k_in_python(self) -> None:
        schema = tuning_values_schema(5)
        warm_configs = schema["properties"]["warm_configs"]

        self.assertEqual(warm_configs["minItems"], 1)
        self.assertNotIn("maxItems", warm_configs)
        with self.assertRaisesRegex(ValueError, "exactly 5 configs"):
            TuningValues.from_response(
                tuning_response(1, 2),
                k=5,
                expected_keys={"depth"},
            )

    def test_validator_widening_is_published_from_staged_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity, action, candidate_dir = self._case(
                Path(tmp), train_source=VALID_SCHEMA
            )
            pipeline = CandidatePipeline(
                identity,
                self._toolchain(),
                ScriptedModels(
                    infer_responses=[tuning_response(10, low=1, high=5)]
                ),
                {},
            )

            pipeline.build_contract(action)

            space = json.loads(
                (candidate_dir / "_search_space.json").read_text()
            )
            self.assertLessEqual(space["depth"][1], 10)
            self.assertGreaterEqual(space["depth"][2], 10)
            proposal = json.loads(
                (candidate_dir / "_search_space_proposal_1.json").read_text()
            )
            self.assertEqual(proposal, {"depth": ["int", 1, 5]})
            self.assertTrue(pipeline.contract_is_ready(action))


if __name__ == "__main__":
    unittest.main()
