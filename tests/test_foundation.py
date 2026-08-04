from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hieraresearch.artifacts import (  # noqa: E402
    ArtifactError,
    InvocationJournal,
    atomic_write_json,
    file_revision,
    json_revision,
    paths_revision,
)
from hieraresearch.background import (  # noqa: E402
    BackgroundArtifactError,
    BackgroundBuilder,
    MAX_BACKGROUND_REPAIR_ATTEMPTS,
)
from hieraresearch.llm import (  # noqa: E402
    AgentEditSpec,
    AnthropicMessagesBackend,
    InferenceContractError,
    InferenceError,
    InferenceRequestError,
    ModelGateway,
    PathPolicy,
)
from hieraresearch.models import IdeaProposal, RunIdentity  # noqa: E402
from hieraresearch.process import ProcessResult, ProcessRunner  # noqa: E402
from hieraresearch.schemas import tuning_values_schema  # noqa: E402
from hieraresearch.toolchain import (  # noqa: E402
    ToolFailure,
    Toolchain,
    ValidationRejected,
    exact_command_string,
)


class StructuredStub:
    def __init__(self, response: object, mutate: Path | None = None):
        self.response = response
        self.mutate = mutate
        self.calls = 0

    def generate(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.mutate is not None:
            self.mutate.write_text("changed", encoding="utf-8")
        return self.response, {"backend": "stub"}


class UnusedEditor:
    def edit(self, **kwargs):  # pragma: no cover - a failing guard for these tests
        raise AssertionError(f"unexpected edit: {kwargs}")


class BackgroundToolchainStub(Toolchain):
    def __init__(self, repo_root: Path):
        super().__init__(repo_root, ProcessRunner(), helper_timeout=30.0)
        self.validation_calls: list[tuple[Path, bool, bool]] = []
        self.retrieval_imports: list[tuple[Path, Path]] = []

    def background_catalog_receipt(self, catalog_path=None):
        del catalog_path
        return {
            "id": "semantic-dimensions/v1",
            "revision": "sha256:" + "a" * 64,
        }

    def import_background_retrieval(self, run_dir: Path, draft_path: Path):
        self.retrieval_imports.append((run_dir, draft_path))
        atomic_write_json(
            run_dir / "background_retrieval.json",
            {"schema_version": 3, "kind": "canonical-test-manifest"},
        )
        return {"ok": True}

    def validate_background_retrieval(self, run_dir: Path):
        if not (run_dir / "background_retrieval.json").is_file():
            raise ValidationRejected(
                "background retrieval validation",
                ProcessResult(
                    args=("validator",),
                    returncode=1,
                    output='{"ok": false, "errors": ["manifest missing"]}',
                    elapsed_seconds=0.0,
                ),
            )

    def validate_background(
        self,
        run_dir: Path,
        *,
        induced: bool,
        provided_baseline: bool,
    ) -> None:
        self.validation_calls.append((run_dir, induced, provided_baseline))
        if not (run_dir / "background.md").is_file():
            raise AssertionError("background.md was not produced")
        if not (run_dir / "background_retrieval.json").is_file():
            raise AssertionError("background_retrieval.json was not produced")


class BackgroundModelStub:
    model = "recorded-test-model"

    def __init__(self):
        self.specs = []

    def edit(self, spec, *, validate):
        self.specs.append(spec)
        for path in spec.write_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.name == "background.md":
                path.write_text(
                    "# Frozen background\n\n"
                    "## Search space registry\n"
                    "```json\n"
                    '{"catalog":{"id":"semantic-dimensions/v1",'
                    '"revision":"coordinator-owned"}}\n'
                    "```\n",
                    encoding="utf-8",
                )
            elif path.name == "background_retrieval.draft.json":
                atomic_write_json(
                    path,
                    {
                        "schema_version": 1,
                        "kind": "external_retrieval_draft",
                    },
                )
            elif path.name == "dimension_catalog.json":
                atomic_write_json(
                    path,
                    {"schema_version": 1, "kind": "test-dimension-catalog"},
                )
            elif path.name == "baseline_mechanisms.json":
                atomic_write_json(path, {})
            else:  # pragma: no cover - guard for unexpected authored outputs
                raise AssertionError(f"unexpected background output: {path}")
        return validate()


def background_validator_commands(
    repo_root: Path,
    run_dir: Path,
    *,
    induced: bool,
    provided_baseline: bool,
    with_import: bool,
) -> set[str]:
    """Expected exact Bash commands for a background_research edit spec."""
    tools = repo_root.resolve() / "tools"
    commands = {
        exact_command_string(
            [
                sys.executable,
                str(tools / "search_backends.py"),
                "validate",
                "--manifest",
                str(run_dir / "background_retrieval.json"),
            ]
        ),
        exact_command_string(
            [
                sys.executable,
                str(tools / "background_contract.py"),
                "validate",
                "--background",
                str(run_dir / "background.md"),
                "--retrieval-manifest",
                str(run_dir / "background_retrieval.json"),
                *(
                    [
                        "--baseline-mechanisms",
                        str(run_dir / "baseline_mechanisms.json"),
                    ]
                    if provided_baseline
                    else []
                ),
            ]
        ),
    }
    if with_import:
        commands.add(
            exact_command_string(
                [
                    sys.executable,
                    str(tools / "search_backends.py"),
                    "import-external",
                    "--draft",
                    str(run_dir / "background_retrieval.draft.json"),
                    "--manifest",
                    str(run_dir / "background_retrieval.json"),
                ]
            )
        )
    if induced:
        commands.add(
            exact_command_string(
                [
                    sys.executable,
                    str(tools / "background_contract.py"),
                    "catalog",
                    "--path",
                    str(run_dir / "dimension_catalog.json"),
                ]
            )
        )
    return commands


class FoundationTests(unittest.TestCase):
    def test_anthropic_tuning_schema_uses_supported_array_cardinality(self) -> None:
        class SchemaCheckingMessages:
            def __init__(self):
                self.schema = None

            def create(self, **kwargs):
                self.schema = kwargs["output_config"]["format"]["schema"]

                def check(value):
                    if isinstance(value, dict):
                        if "minItems" in value:
                            if value["minItems"] not in {0, 1}:
                                raise ValueError("unsupported minItems")
                        for child in value.values():
                            check(child)
                    elif isinstance(value, list):
                        for child in value:
                            check(child)

                check(self.schema)
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text='{"ok": true}')],
                    usage=None,
                    _request_id="request-test",
                )

        messages = SchemaCheckingMessages()
        backend = AnthropicMessagesBackend(
            client=SimpleNamespace(messages=messages),
        )

        value, _ = backend.generate(
            purpose="tuning_values:test",
            model="test-model",
            system_prompt="system",
            prompt="prompt",
            schema=tuning_values_schema(5),
            max_tokens=100,
        )

        self.assertEqual(value, {"ok": True})
        warm_configs = messages.schema["properties"]["warm_configs"]
        self.assertEqual(warm_configs["minItems"], 1)
        self.assertNotIn("maxItems", warm_configs)

    def test_invalid_provider_request_is_nonretryable_for_exact_revision(self) -> None:
        import anthropic
        import httpx

        class RejectingMessages:
            def __init__(self):
                self.calls = 0

            def create(self, **kwargs):
                del kwargs
                self.calls += 1
                request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                response = httpx.Response(400, request=request)
                raise anthropic.BadRequestError(
                    "invalid JSON schema",
                    response=response,
                    body={
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "invalid JSON schema",
                        },
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.txt"
            source.write_text("stable", encoding="utf-8")
            messages = RejectingMessages()
            gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(root),
                structured_backend=AnthropicMessagesBackend(
                    client=SimpleNamespace(messages=messages)
                ),
                edit_backend=UnusedEditor(),
            )
            request = {
                "purpose": "tuning_values:001",
                "schema_version": 1,
                "system_prompt": "system",
                "prompt": "prompt",
                "schema": {"type": "object"},
                "input_paths": [source],
                "parser": lambda value: value,
            }

            with self.assertRaisesRegex(
                InferenceRequestError, "rejected the request contract"
            ):
                gateway.infer(**request)
            with self.assertRaisesRegex(
                InferenceRequestError, "recorded non-retryable"
            ):
                gateway.infer(**request)

            self.assertEqual(messages.calls, 1)

            changed_request = {
                **request,
                "schema": {"type": "object", "additionalProperties": False},
            }
            with self.assertRaises(InferenceRequestError):
                gateway.infer(**changed_request)
            self.assertEqual(messages.calls, 2)

    def test_model_receipt_replays_exact_input_and_rejects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.txt"
            source.write_text("stable", encoding="utf-8")
            response = {
                "idea": "try a bounded representation change",
                "change": "replace the candidate representation",
                "candidate_name_hint": "bounded-representation",
                "description": "A recorded structured proposal.",
            }
            backend = StructuredStub(response)
            gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(root),
                structured_backend=backend,
                edit_backend=UnusedEditor(),
            )
            request = {
                "purpose": "idea:001",
                "schema_version": 1,
                "system_prompt": "system",
                "prompt": "prompt",
                "schema": {"type": "object"},
                "input_paths": [source],
                "parser": IdeaProposal.from_dict,
            }

            first = gateway.infer(**request)
            second = gateway.infer(**request)

            self.assertEqual(first, second)
            self.assertEqual(backend.calls, 1)

            invocation = next(gateway.journal.root.iterdir())
            tampered = {**response, "idea": "tampered replay"}
            atomic_write_json(invocation / "response.json", tampered)
            with self.assertRaisesRegex(ArtifactError, "response revision mismatch"):
                gateway.infer(**request)

    def test_model_gateway_rejects_live_stale_and_malformed_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.txt"
            source.write_text("before", encoding="utf-8")
            response = {
                "idea": "try a bounded representation change",
                "change": "replace the candidate representation",
                "candidate_name_hint": "bounded-representation",
                "description": "A recorded structured proposal.",
            }
            request = {
                "purpose": "idea:stale",
                "schema_version": 1,
                "system_prompt": "system",
                "prompt": "prompt",
                "schema": {"type": "object"},
                "input_paths": [source],
                "parser": IdeaProposal.from_dict,
            }
            stale_gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(root / "stale-run"),
                structured_backend=StructuredStub(response, mutate=source),
                edit_backend=UnusedEditor(),
            )
            with self.assertRaisesRegex(ArtifactError, "inputs changed"):
                stale_gateway.infer(**request)

            source.write_text("stable", encoding="utf-8")
            malformed_gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(root / "malformed-run"),
                structured_backend=StructuredStub({"idea": "missing fields"}),
                edit_backend=UnusedEditor(),
            )
            with self.assertRaises(InferenceContractError):
                malformed_gateway.infer(
                    **{**request, "purpose": "idea:malformed"}
                )

    def test_completed_edit_requires_current_immutable_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            immutable = root / "brief.json"
            output = root / "train.py"
            immutable.write_text("old brief", encoding="utf-8")
            output.write_text("MODEL = 1\n", encoding="utf-8")
            journal = InvocationJournal(root)
            request = {
                "kind": "agent_edit",
                "write_paths": [str(output.resolve())],
                "immutable_input_revision": paths_revision([immutable]),
            }
            invocation = journal.begin(
                purpose="candidate_writer:001",
                schema_version=1,
                model="test-model",
                input_revision=json_revision(request),
                request=request,
            )
            journal.complete(
                invocation,
                response={
                    "result": "done",
                    "output_revisions": {
                        str(output.resolve()): file_revision(output),
                    },
                },
            )

            self.assertTrue(
                journal.completed_output_matches(
                    purpose="candidate_writer:001",
                    schema_version=1,
                    model="test-model",
                    output_path=output,
                    immutable_input_paths=(immutable,),
                )
            )
            immutable.write_text("new brief", encoding="utf-8")
            self.assertFalse(
                journal.completed_output_matches(
                    purpose="candidate_writer:001",
                    schema_version=1,
                    model="test-model",
                    output_path=output,
                    immutable_input_paths=(immutable,),
                )
            )

    def test_model_gateway_records_but_does_not_authorize_derived_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authored = root / "draft.json"
            derived = root / "manifest.json"

            class DerivedEditor:
                def edit(self, *, policy, **kwargs):
                    del kwargs
                    self.assertions = (
                        policy.decision(
                            "Write", {"file_path": str(authored)}
                        )[0],
                        policy.decision(
                            "Write", {"file_path": str(derived)}
                        )[0],
                    )
                    authored.write_text("{}\n", encoding="utf-8")
                    return "draft written", {"backend": "stub"}

            editor = DerivedEditor()
            gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(root),
                structured_backend=StructuredStub({}),
                edit_backend=editor,
            )

            gateway.edit(
                AgentEditSpec(
                    purpose="derived-output",
                    schema_version=1,
                    cwd=root,
                    system_prompt="system",
                    prompt="prompt",
                    tools=("Write",),
                    read_roots=(root,),
                    write_paths=(authored,),
                    input_paths=(authored, derived),
                    immutable_input_paths=(),
                    derived_output_paths=(derived,),
                ),
                validate=lambda: atomic_write_json(derived, {"canonical": True}),
            )

            self.assertEqual(editor.assertions, (True, False))
            response_path = next(
                (root / ".orchestrator" / "invocations").glob(
                    "derived-output-*"
                )
            ) / "response.json"
            response = json.loads(response_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(response["output_revisions"]),
                {str(authored.resolve()), str(derived.resolve())},
            )

    def test_model_gateway_normalizes_edit_backend_setup_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authored = root / "draft.py"

            class BrokenEditor:
                def edit(self, **kwargs):
                    del kwargs
                    raise ValueError("SDK options rejected")

            gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(root),
                structured_backend=StructuredStub({}),
                edit_backend=BrokenEditor(),
            )
            with self.assertRaisesRegex(
                InferenceError,
                "edit backend failed: SDK options rejected",
            ):
                gateway.edit(
                    AgentEditSpec(
                        purpose="candidate_writer:001",
                        schema_version=1,
                        cwd=root,
                        system_prompt="system",
                        prompt="prompt",
                        tools=("Write",),
                        read_roots=(root,),
                        write_paths=(authored,),
                        input_paths=(authored,),
                        immutable_input_paths=(),
                    ),
                    validate=lambda: authored,
                )

    def test_agent_path_policy_enforces_exact_edit_and_readonly_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "tasks" / "toy"
            candidate = root / "runs" / "toy" / "run" / "candidates" / "001"
            sibling = candidate.parent / "002"
            task.mkdir(parents=True)
            candidate.mkdir(parents=True)
            sibling.mkdir(parents=True)
            train = candidate / "train.py"
            prepare = candidate / "prepare.py"
            task_contract = task / "TASK.md"
            for path in (train, prepare, task_contract, sibling / "train.py"):
                path.write_text("x", encoding="utf-8")
            policy = PathPolicy(
                cwd=root,
                read_roots=(task, candidate),
                write_paths=(train,),
                allowed_tools=("Read", "Glob", "Grep", "Write", "Edit"),
            )

            self.assertTrue(policy.decision("Read", {"file_path": str(task_contract)})[0])
            self.assertTrue(policy.decision("Edit", {"file_path": str(train)})[0])
            self.assertFalse(policy.decision("Write", {"file_path": str(prepare)})[0])
            self.assertFalse(
                policy.decision("Edit", {"file_path": str(sibling / "train.py")})[0]
            )
            self.assertFalse(
                policy.decision(
                    "Glob",
                    {"path": str(candidate), "pattern": "../*/train.py"},
                )[0]
            )
            self.assertFalse(policy.decision("Bash", {"command": "true"})[0])

    def test_agent_path_policy_allows_only_exact_bash_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = (
                f"{sys.executable} {root / 'tools' / 'check.py'} validate "
                f"--manifest {root / 'run' / 'manifest.json'}"
            )
            policy = PathPolicy(
                cwd=root,
                read_roots=(root,),
                write_paths=(root / "run" / "background.md",),
                allowed_tools=("Read", "Write", "Bash"),
                allowed_commands={command},
            )

            self.assertTrue(policy.decision("Bash", {"command": command})[0])
            self.assertTrue(
                policy.decision("Bash", {"command": f"  {command}  "})[0]
            )
            self.assertFalse(
                policy.decision("Bash", {"command": f"{command} --strict"})[0]
            )
            self.assertFalse(
                policy.decision("Bash", {"command": f"{command} && rm -rf {root}"})[0]
            )
            self.assertFalse(
                policy.decision(
                    "Bash",
                    {
                        "command": (
                            f"{sys.executable} {root / 'tools' / 'check.py'} "
                            f"validate --manifest {root / 'other' / 'manifest.json'}"
                        )
                    },
                )[0]
            )
            self.assertFalse(policy.decision("Bash", {"command": command[:-3]})[0])
            self.assertFalse(policy.decision("Bash", {"command": ""})[0])
            self.assertFalse(policy.decision("Bash", {})[0])

            without_commands = PathPolicy(
                cwd=root,
                read_roots=(root,),
                write_paths=(),
                allowed_tools=("Bash",),
            )
            self.assertFalse(
                without_commands.decision("Bash", {"command": command})[0]
            )

    def test_agent_edit_spec_requires_bash_tool_for_allowed_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft = root / "draft.py"
            with self.assertRaisesRegex(ValueError, "Bash"):
                AgentEditSpec(
                    purpose="guarded",
                    schema_version=1,
                    cwd=root,
                    system_prompt="system",
                    prompt="prompt",
                    tools=("Read", "Write"),
                    read_roots=(root,),
                    write_paths=(draft,),
                    input_paths=(draft,),
                    immutable_input_paths=(),
                    allowed_commands=frozenset(
                        {f"{sys.executable} -m py_compile {draft}"}
                    ),
                )
            with self.assertRaisesRegex(ValueError, "non-empty"):
                AgentEditSpec(
                    purpose="guarded",
                    schema_version=1,
                    cwd=root,
                    system_prompt="system",
                    prompt="prompt",
                    tools=("Bash",),
                    read_roots=(root,),
                    write_paths=(draft,),
                    input_paths=(draft,),
                    immutable_input_paths=(),
                    allowed_commands=frozenset({"   "}),
                )

    def test_model_gateway_threads_allowed_commands_into_path_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            authored = root / "draft.py"
            command = f"{sys.executable} -m py_compile {authored}"

            class PolicyCapturingEditor:
                def edit(self, *, policy, **kwargs):
                    del kwargs
                    self.decisions = (
                        policy.decision("Bash", {"command": command})[0],
                        policy.decision("Bash", {"command": f"{command} && ls"})[0],
                    )
                    authored.write_text("x = 1\n", encoding="utf-8")
                    return "draft written", {"backend": "stub"}

            editor = PolicyCapturingEditor()
            gateway = ModelGateway(
                model="test-model",
                journal=InvocationJournal(root),
                structured_backend=StructuredStub({}),
                edit_backend=editor,
            )

            gateway.edit(
                AgentEditSpec(
                    purpose="candidate_writer:001",
                    schema_version=1,
                    cwd=root,
                    system_prompt="system",
                    prompt="prompt",
                    tools=("Write", "Bash"),
                    read_roots=(root,),
                    write_paths=(authored,),
                    allowed_commands=frozenset({command}),
                    input_paths=(authored,),
                    immutable_input_paths=(),
                ),
                validate=lambda: authored,
            )

            self.assertEqual(editor.decisions, (True, False))
            request_path = next(
                (root / ".orchestrator" / "invocations").glob(
                    "candidate_writer-001-*"
                )
            ) / "request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(request["allowed_commands"], [command])

    def test_timeout_kills_descendant_process_and_bounds_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pid_path = root / "child.pid"
            log_path = root / "worker.log"
            program = (
                "import pathlib,subprocess,sys,time;"
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid));"
                "print('x'*10000,flush=True);time.sleep(60)"
            )
            runner = ProcessRunner(capture_limit_bytes=128, kill_grace_seconds=0.2)

            result = runner.run(
                [sys.executable, "-c", program],
                cwd=root,
                timeout=0.4,
                output_path=log_path,
            )

            self.assertTrue(result.timed_out)
            self.assertEqual(result.returncode, 124)
            self.assertTrue(result.output_truncated)
            self.assertLessEqual(len(result.output.encode()), 128)
            child_pid = int(pid_path.read_text())
            deadline = time.monotonic() + 2
            state = self._process_state(child_pid)
            while state not in {None, "Z"} and time.monotonic() < deadline:
                time.sleep(0.02)
                state = self._process_state(child_pid)
            self.assertIn(state, {None, "Z"})

    def test_background_builder_creates_and_reuses_validated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "slice")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            toolchain = BackgroundToolchainStub(identity.repo_root)
            models = BackgroundModelStub()
            builder = BackgroundBuilder(identity, toolchain, models, task_config={})

            builder.ensure()
            builder.ensure()

            self.assertEqual(len(models.specs), 1)
            spec = models.specs[0]
            self.assertEqual(spec.purpose, "background_research")
            self.assertEqual(
                {path.name for path in spec.write_paths},
                {"background.md", "background_retrieval.draft.json"},
            )
            self.assertEqual(
                {path.name for path in spec.derived_output_paths},
                {"background_retrieval.json"},
            )
            self.assertIn("Bash", spec.tools)
            self.assertEqual(
                spec.allowed_commands,
                background_validator_commands(
                    repo_root,
                    identity.run_dir,
                    induced=False,
                    provided_baseline=False,
                    with_import=True,
                ),
            )
            self.assertEqual(len(toolchain.validation_calls), 2)
            self.assertEqual(len(toolchain.retrieval_imports), 1)
            self.assertIn(
                "sha256:" + "a" * 64,
                (identity.run_dir / "background.md").read_text(encoding="utf-8"),
            )
            self.assertTrue(all(not induced for _, induced, _ in toolchain.validation_calls))
            self.assertTrue(
                all(not provided for _, _, provided in toolchain.validation_calls)
            )

    def test_background_research_spec_allowlists_exact_validator_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "induced-provided")
            task_dir = repo_root / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            (task_dir / "train.py").write_text("SEED = True\n", encoding="utf-8")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "llm_induced"}},
            )
            toolchain = BackgroundToolchainStub(identity.repo_root)
            models = BackgroundModelStub()
            builder = BackgroundBuilder(
                identity,
                toolchain,
                models,
                task_config={
                    "seed": {"provided": ["train.py"], "entrypoint": "train.py"}
                },
            )

            builder.ensure()

            self.assertEqual(len(models.specs), 1)
            spec = models.specs[0]
            self.assertEqual(spec.purpose, "background_research")
            self.assertIn("Bash", spec.tools)
            self.assertEqual(
                spec.allowed_commands,
                background_validator_commands(
                    repo_root,
                    identity.run_dir,
                    induced=True,
                    provided_baseline=True,
                    with_import=True,
                ),
            )
            for command in sorted(spec.allowed_commands):
                self.assertIn(command, spec.prompt)
            self.assertIn("import-external", spec.prompt)
            self.assertEqual(
                toolchain.validation_calls, [(identity.run_dir, True, True)]
            )

    def test_background_builder_does_not_research_on_retrieval_infrastructure_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "retrieval-infrastructure")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            atomic_write_json(
                identity.run_dir / "background_retrieval.json",
                {"schema_version": 3},
            )

            class UnavailableRetrievalValidator(BackgroundToolchainStub):
                def validate_background_retrieval(self, run_dir):
                    raise ToolFailure(
                        "background retrieval validation",
                        ProcessResult(
                            args=("validator",),
                            returncode=2,
                            output="validator worker unavailable",
                            elapsed_seconds=0.0,
                        ),
                    )

            models = BackgroundModelStub()
            builder = BackgroundBuilder(
                identity,
                UnavailableRetrievalValidator(identity.repo_root),
                models,
                task_config={},
            )

            with self.assertRaisesRegex(ToolFailure, "worker unavailable"):
                builder.ensure()

            self.assertEqual(models.specs, [])

    def test_background_builder_does_not_repair_validator_infrastructure_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "validator-infrastructure")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )

            class UnavailableBackgroundValidator(BackgroundToolchainStub):
                def validate_background(self, run_dir, *, induced, provided_baseline):
                    super().validate_background(
                        run_dir,
                        induced=induced,
                        provided_baseline=provided_baseline,
                    )
                    raise ToolFailure(
                        "background validation",
                        ProcessResult(
                            args=("validator",),
                            returncode=2,
                            output="validator worker unavailable",
                            elapsed_seconds=0.0,
                        ),
                    )

            models = BackgroundModelStub()
            builder = BackgroundBuilder(
                identity,
                UnavailableBackgroundValidator(identity.repo_root),
                models,
                task_config={},
            )

            with self.assertRaisesRegex(ToolFailure, "worker unavailable"):
                builder.ensure()

            self.assertEqual(len(models.specs), 1)
            self.assertEqual(models.specs[0].purpose, "background_research")

    def test_existing_background_validator_infrastructure_is_not_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "existing-infrastructure")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            (identity.run_dir / "background.md").write_text(
                "# existing background\n",
                encoding="utf-8",
            )
            atomic_write_json(
                identity.run_dir / "background_retrieval.json",
                {"schema_version": 3},
            )

            class UnavailableBackgroundValidator(BackgroundToolchainStub):
                def validate_background(self, run_dir, *, induced, provided_baseline):
                    raise ToolFailure(
                        "background validation",
                        ProcessResult(
                            args=("validator",),
                            returncode=2,
                            output="validator worker unavailable",
                            elapsed_seconds=0.0,
                        ),
                    )

            models = BackgroundModelStub()
            builder = BackgroundBuilder(
                identity,
                UnavailableBackgroundValidator(identity.repo_root),
                models,
                task_config={},
            )

            with self.assertRaisesRegex(ToolFailure, "worker unavailable"):
                builder.ensure()

            self.assertEqual(models.specs, [])
            self.assertFalse(
                (
                    identity.run_dir
                    / ".orchestrator"
                    / "background_repair_diagnostic.json"
                ).exists()
            )

    def test_background_builder_bounds_layered_repairs_across_derived_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "layered-repair")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )

            class LayeredToolchain(BackgroundToolchainStub):
                def import_background_retrieval(self, run_dir, draft_path):
                    if not self.retrieval_imports:
                        self.retrieval_imports.append((run_dir, draft_path))
                        raise BackgroundArtifactError("draft contract rejected")
                    return super().import_background_retrieval(run_dir, draft_path)

                def validate_background(self, run_dir, *, induced, provided_baseline):
                    super().validate_background(
                        run_dir,
                        induced=induced,
                        provided_baseline=provided_baseline,
                    )
                    if len(self.validation_calls) == 1:
                        raise BackgroundArtifactError("registry contract rejected")

            toolchain = LayeredToolchain(identity.repo_root)
            models = BackgroundModelStub()
            BackgroundBuilder(identity, toolchain, models, task_config={}).ensure()

            self.assertEqual(
                [spec.purpose for spec in models.specs],
                [
                    "background_research",
                    "background_research:repair:1",
                    "background_research:repair:2",
                ],
            )
            self.assertEqual(len(toolchain.retrieval_imports), 2)
            self.assertEqual(len(toolchain.validation_calls), 2)
            final_spec = models.specs[-1]
            self.assertEqual({path.name for path in final_spec.write_paths}, {"background.md"})
            self.assertEqual(final_spec.derived_output_paths, ())
            self.assertNotIn("WebSearch", final_spec.tools)
            self.assertNotIn("WebFetch", final_spec.tools)
            self.assertIn("is frozen for this repair", final_spec.prompt)

    def test_background_builder_reuses_valid_retrieval_during_registry_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "registry-repair")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            (identity.run_dir / "background.md").write_text(
                "# Invalid registry draft\n", encoding="utf-8"
            )
            atomic_write_json(
                identity.run_dir / "background_retrieval.json",
                {"schema_version": 3, "kind": "canonical-test-manifest"},
            )

            class RegistryRepairToolchain(BackgroundToolchainStub):
                def validate_background(self, run_dir, *, induced, provided_baseline):
                    super().validate_background(
                        run_dir,
                        induced=induced,
                        provided_baseline=provided_baseline,
                    )
                    if len(self.validation_calls) == 1:
                        raise BackgroundArtifactError("registry rejected")

            toolchain = RegistryRepairToolchain(identity.repo_root)
            models = BackgroundModelStub()
            BackgroundBuilder(identity, toolchain, models, task_config={}).ensure()

            self.assertEqual(len(models.specs), 1)
            spec = models.specs[0]
            self.assertEqual(spec.purpose, "background_research:repair:1")
            self.assertEqual(
                {path.name for path in spec.write_paths}, {"background.md"}
            )
            self.assertEqual(spec.derived_output_paths, ())
            self.assertNotIn("WebSearch", spec.tools)
            self.assertNotIn("WebFetch", spec.tools)
            self.assertIn("Bash", spec.tools)
            self.assertEqual(
                spec.allowed_commands,
                background_validator_commands(
                    repo_root,
                    identity.run_dir,
                    induced=False,
                    provided_baseline=False,
                    with_import=False,
                ),
            )
            self.assertEqual(toolchain.retrieval_imports, [])
            diagnostic_path = (
                identity.run_dir / ".orchestrator" / "background_repair_diagnostic.json"
            )
            self.assertIn(diagnostic_path, spec.input_paths)
            self.assertIn(diagnostic_path, spec.immutable_input_paths)
            self.assertIn(str(diagnostic_path), spec.prompt)
            self.assertEqual(
                json.loads(diagnostic_path.read_text(encoding="utf-8"))["message"],
                "registry rejected",
            )

    def test_background_builder_persists_complete_validation_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "complete-diagnostic")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            expected_errors = [f"contract error {index}" for index in range(200)]

            class ReportingToolchain(BackgroundToolchainStub):
                def validate_background(self, run_dir, *, induced, provided_baseline):
                    super().validate_background(
                        run_dir,
                        induced=induced,
                        provided_baseline=provided_baseline,
                    )
                    if len(self.validation_calls) == 1:
                        raise ValidationRejected(
                            "background validation",
                            ProcessResult(
                                args=("validator",),
                                returncode=1,
                                output=json.dumps({"ok": False, "errors": expected_errors}),
                                elapsed_seconds=0.0,
                            ),
                        )

            models = BackgroundModelStub()
            BackgroundBuilder(
                identity, ReportingToolchain(identity.repo_root), models, task_config={}
            ).ensure()

            diagnostic_path = (
                identity.run_dir / ".orchestrator" / "background_repair_diagnostic.json"
            )
            diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            self.assertEqual(diagnostic["validator_errors"], expected_errors)
            self.assertFalse(diagnostic["validator_errors_truncated"])
            repair_spec = models.specs[-1]
            self.assertIn(diagnostic_path, repair_spec.input_paths)
            self.assertIn(str(diagnostic_path), repair_spec.prompt)

    def test_background_builder_resumed_started_attempt_keeps_admission_and_diagnostic(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "started-resume")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            expected_errors = ["contract error 1", "contract error 2"]

            class RejectOnceToolchain(BackgroundToolchainStub):
                def validate_background(self, run_dir, *, induced, provided_baseline):
                    super().validate_background(
                        run_dir,
                        induced=induced,
                        provided_baseline=provided_baseline,
                    )
                    if len(self.validation_calls) == 1:
                        raise ValidationRejected(
                            "background validation",
                            ProcessResult(
                                args=("validator",),
                                returncode=1,
                                output=json.dumps(
                                    {"ok": False, "errors": expected_errors}
                                ),
                                elapsed_seconds=0.0,
                            ),
                        )

            class InterruptedModels(BackgroundModelStub):
                def edit(self, spec, *, validate):
                    if self.specs:
                        self.specs.append(spec)
                        raise RuntimeError("worker killed mid-repair")
                    return super().edit(spec, validate=validate)

            models = InterruptedModels()
            builder = BackgroundBuilder(
                identity, RejectOnceToolchain(identity.repo_root), models, task_config={}
            )
            with self.assertRaisesRegex(RuntimeError, "worker killed"):
                builder.ensure()

            state_path = (
                identity.run_dir / ".orchestrator" / "background_authoring.json"
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "started")
            self.assertEqual(state["attempts_admitted"], 2)
            self.assertEqual(state["purpose"], "background_research:repair:1")
            self.assertEqual(state["last_error_kind"], "validation_rejected")

            # A killed writer may leave required artifacts missing; the resume
            # must then rebuild the rejection from the authoring state alone.
            (identity.run_dir / "background.md").unlink()
            resumed_models = BackgroundModelStub()
            BackgroundBuilder(
                identity,
                BackgroundToolchainStub(identity.repo_root),
                resumed_models,
                task_config={},
            ).ensure()

            self.assertEqual(
                [spec.purpose for spec in resumed_models.specs],
                ["background_research:repair:1"],
            )
            final_state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(final_state["status"], "completed")
            self.assertEqual(final_state["attempts_admitted"], 2)
            diagnostic = json.loads(
                (
                    identity.run_dir
                    / ".orchestrator"
                    / "background_repair_diagnostic.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostic["error_type"], "ValidationRejected")
            self.assertEqual(diagnostic["validator_errors"], expected_errors)

    def test_background_builder_bounds_repairs_after_five_rejections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "bounded-repairs")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )

            class RejectingToolchain(BackgroundToolchainStub):
                def validate_background(self, run_dir, *, induced, provided_baseline):
                    super().validate_background(
                        run_dir,
                        induced=induced,
                        provided_baseline=provided_baseline,
                    )
                    raise BackgroundArtifactError("registry remains invalid")

            toolchain = RejectingToolchain(identity.repo_root)
            models = BackgroundModelStub()
            builder = BackgroundBuilder(identity, toolchain, models, task_config={})

            with self.assertRaisesRegex(ValueError, "registry remains invalid"):
                builder.ensure()

            self.assertEqual(len(models.specs), MAX_BACKGROUND_REPAIR_ATTEMPTS + 1)
            self.assertEqual(models.specs[0].purpose, "background_research")
            self.assertEqual(
                models.specs[-1].purpose,
                f"background_research:repair:{MAX_BACKGROUND_REPAIR_ATTEMPTS}",
            )
            self.assertEqual(len(toolchain.retrieval_imports), 1)

            resumed_models = BackgroundModelStub()
            resumed = BackgroundBuilder(
                identity,
                toolchain,
                resumed_models,
                task_config={},
            )
            with self.assertRaisesRegex(ValueError, "registry remains invalid"):
                resumed.ensure()

            state = json.loads(
                (
                    identity.run_dir
                    / ".orchestrator"
                    / "background_authoring.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(resumed_models.specs, [])
            self.assertEqual(
                state["attempts_admitted"],
                MAX_BACKGROUND_REPAIR_ATTEMPTS + 1,
            )

    def test_background_builder_absorbs_non_upstream_inference_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "inference-repair")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            max_turns = InferenceError(
                "Claude Agent SDK edit ended with error_max_turns: "
                "max turns reached before a final result"
            )

            class FailOnceModels(BackgroundModelStub):
                def edit(self, spec, *, validate):
                    if not self.specs:
                        self.specs.append(spec)
                        raise max_turns
                    return super().edit(spec, validate=validate)

            models = FailOnceModels()
            builder = BackgroundBuilder(
                identity, BackgroundToolchainStub(identity.repo_root), models, task_config={}
            )
            builder.ensure()

            self.assertEqual(
                [spec.purpose for spec in models.specs],
                ["background_research", "background_research:repair:1"],
            )
            # An inference failure is not a validator rejection: the repair
            # prompt carries the error text but no stale diagnostic reference.
            self.assertIn("max turns reached", models.specs[1].prompt)
            self.assertNotIn("background_repair_diagnostic", models.specs[1].prompt)
            state = json.loads(
                (
                    identity.run_dir / ".orchestrator" / "background_authoring.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "completed")
            self.assertEqual(state["attempts_admitted"], 2)

    def test_background_builder_reraises_upstream_inference_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "upstream-background")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )

            class UpstreamModels(BackgroundModelStub):
                def edit(self, spec, *, validate):
                    del validate
                    self.specs.append(spec)
                    raise InferenceError(
                        "Claude Agent SDK edit failed: Error code: 502 - "
                        "{'error': {'message': 'Upstream service temporarily "
                        "unavailable', 'type': 'upstream_error'}, 'type': 'error'}"
                    )

            models = UpstreamModels()
            builder = BackgroundBuilder(
                identity, BackgroundToolchainStub(identity.repo_root), models, task_config={}
            )
            with self.assertRaisesRegex(InferenceError, "Error code: 502"):
                builder.ensure()

            # The coordinator's upstream backoff owns this retry: exactly one
            # admitted attempt was consumed and no repair slot was burned.
            self.assertEqual(len(models.specs), 1)
            state = json.loads(
                (
                    identity.run_dir / ".orchestrator" / "background_authoring.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "started")
            self.assertEqual(state["attempts_admitted"], 1)

    def test_background_builder_bounds_non_upstream_inference_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "bounded-inference")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )

            class AlwaysFailingModels(BackgroundModelStub):
                def edit(self, spec, *, validate):
                    del validate
                    self.specs.append(spec)
                    raise InferenceError("Claude Agent SDK returned no final result")

            models = AlwaysFailingModels()
            builder = BackgroundBuilder(
                identity, BackgroundToolchainStub(identity.repo_root), models, task_config={}
            )
            with self.assertRaisesRegex(ValueError, "no final result"):
                builder.ensure()

            self.assertEqual(len(models.specs), MAX_BACKGROUND_REPAIR_ATTEMPTS + 1)
            state = json.loads(
                (
                    identity.run_dir / ".orchestrator" / "background_authoring.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(state["attempts_admitted"], MAX_BACKGROUND_REPAIR_ATTEMPTS + 1)
            self.assertEqual(state["last_error_kind"], "inference")
            self.assertEqual(state["status"], "rejected")

    def test_frozen_background_rejects_stale_provided_baseline_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            identity = RunIdentity(repo_root, "toy", "frozen")
            task_dir = repo_root / "tasks" / "toy"
            task_dir.mkdir(parents=True)
            entrypoint = task_dir / "train.py"
            entrypoint.write_text("CURRENT = True\n", encoding="utf-8")
            identity.run_dir.mkdir(parents=True)
            atomic_write_json(
                identity.run_dir / "framework_cfg.json",
                {"space_initialization": {"dimension_strategy": "catalog_subset"}},
            )
            (identity.run_dir / "background.md").write_text(
                "# Frozen background\n", encoding="utf-8"
            )
            atomic_write_json(
                identity.run_dir / "background_retrieval.json",
                {"schema_version": 3},
            )
            atomic_write_json(
                identity.run_dir / "baseline_mechanisms.json",
                {
                    "entrypoint": {
                        "path": "tasks/toy/train.py",
                        "sha256": "sha256:" + "0" * 64,
                    }
                },
            )
            atomic_write_json(identity.ledger_path, {"records": []})
            toolchain = BackgroundToolchainStub(identity.repo_root)
            models = BackgroundModelStub()
            builder = BackgroundBuilder(
                identity,
                toolchain,
                models,
                task_config={
                    "seed": {"provided": ["train.py"], "entrypoint": "train.py"}
                },
            )

            with self.assertRaisesRegex(
                BackgroundArtifactError, "entrypoint.sha256 does not match"
            ):
                builder.ensure()

            self.assertEqual(models.specs, [])
            self.assertEqual(len(toolchain.validation_calls), 1)

    @staticmethod
    def _process_state(pid: int) -> str | None:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().split()
        except OSError:
            return None
        return fields[2] if len(fields) > 2 else None


if __name__ == "__main__":
    unittest.main()
