from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path


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
)
from hieraresearch.llm import (  # noqa: E402
    InferenceContractError,
    ModelGateway,
    PathPolicy,
)
from hieraresearch.models import IdeaProposal, RunIdentity  # noqa: E402
from hieraresearch.process import ProcessRunner  # noqa: E402


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


class BackgroundToolchainStub:
    def __init__(self):
        self.validation_calls: list[tuple[Path, bool, bool]] = []

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
                path.write_text("# Frozen background\n", encoding="utf-8")
            elif path.name == "background_retrieval.json":
                atomic_write_json(path, {"schema_version": 1})
            else:  # pragma: no cover - catalog/provided variants are not in this slice test
                raise AssertionError(f"unexpected background output: {path}")
        return validate()


class FoundationTests(unittest.TestCase):
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
            toolchain = BackgroundToolchainStub()
            models = BackgroundModelStub()
            builder = BackgroundBuilder(identity, toolchain, models, task_config={})

            builder.ensure()
            builder.ensure()

            self.assertEqual(len(models.specs), 1)
            spec = models.specs[0]
            self.assertEqual(spec.purpose, "background_research")
            self.assertEqual(
                {path.name for path in spec.write_paths},
                {"background.md", "background_retrieval.json"},
            )
            self.assertEqual(len(toolchain.validation_calls), 2)
            self.assertTrue(all(not induced for _, induced, _ in toolchain.validation_calls))
            self.assertTrue(
                all(not provided for _, _, provided in toolchain.validation_calls)
            )

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
            toolchain = BackgroundToolchainStub()
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
