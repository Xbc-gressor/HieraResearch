from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import hieraresearch.experience as experience_module  # noqa: E402
from hieraresearch.artifacts import ArtifactError, atomic_write_json  # noqa: E402
from hieraresearch.experience import ExperienceRefresh  # noqa: E402
from hieraresearch.llm import InferenceContractError, InferenceError  # noqa: E402
from hieraresearch.models import (  # noqa: E402
    CoordinatorState,
    RunIdentity,
    Transition,
)
from hieraresearch.process import ProcessResult, ProcessRunner  # noqa: E402
from hieraresearch.state_machine import next_transition  # noqa: E402
from hieraresearch.toolchain import Toolchain, ValidationRejected  # noqa: E402
from fixtures import background_text, fixture_registry, record  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402
from semantic_space import complete_point, space_receipt  # noqa: E402


PROPOSAL = {
    "schema_version": 4,
    "updated_at_run": "000",
    "generation": 1,
    "summary": "bounded evidence",
    "promising_regions": [],
    "lessons": [],
    "bottlenecks": [],
    "dimension_evidence": [],
    "hypothesis_evidence": [],
}


class SimulatedKill(BaseException):
    pass


class RecordingModels:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, **kwargs):
        self.calls += 1
        return kwargs["parser"](dict(PROPOSAL))


class RecoveryToolchain:
    def __init__(self, *, apply_failure: str | None = None) -> None:
        self.apply_failure = apply_failure
        self.views_calls = 0
        self.validation_calls = 0
        self.store_calls = 0
        self.apply_calls = 0
        self.stored_experience: dict | None = {"dag_revision": 2}
        self.space_revision = 0
        self.space_applied = False

    def experience_views(self, run_dir):
        del run_dir
        self.views_calls += 1
        return {"brief": {"dag_revision": 3}}

    def validate_experience(self, run_dir, output):
        del run_dir
        self.validation_calls += 1
        if json.loads(output.read_text()) != PROPOSAL:
            raise AssertionError("unexpected experience proposal")

    def store_experience(self, run_dir, output):
        del run_dir
        self.store_calls += 1
        self.stored_experience = json.loads(output.read_text())
        self.stored_experience["dag_revision"] = 3

    def ledger_experience(self, run_dir):
        del run_dir
        return self.stored_experience

    def apply_space_state(self, run_dir):
        del run_dir
        self.apply_calls += 1
        if self.apply_failure == "before" and self.apply_calls == 1:
            raise RuntimeError("space-state worker failed")
        if not self.space_applied:
            self.space_applied = True
            self.space_revision += 1
        if self.apply_failure == "after" and self.apply_calls == 1:
            raise SimulatedKill("killed after ledger apply")
        return {"ok": True, "revision": self.space_revision}


class InterruptFirstStoreToolchain(Toolchain):
    def __init__(self) -> None:
        super().__init__(ROOT, ProcessRunner(), helper_timeout=30.0)
        self.store_calls = 0

    def store_experience(self, run_dir, output):
        self.store_calls += 1
        if self.store_calls == 1:
            raise SimulatedKill("killed before experience store")
        return super().store_experience(run_dir, output)


class RejectingToolchain(RecoveryToolchain):
    def validate_experience(self, run_dir, output):
        del run_dir, output
        self.validation_calls += 1
        raise ValidationRejected(
            "validate-experience",
            ProcessResult(
                args=("validate-experience",),
                returncode=1,
                output="rejected: evidence cites an unknown run",
                elapsed_seconds=0.0,
            ),
        )


class FailingModels:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.calls = 0

    def infer(self, **kwargs):
        self.calls += 1
        raise self.error


def stale_brief() -> dict:
    return {
        "dag_revision": 3,
        "experience_dag_revision": 2,
        "experience_dag_delta": 1,
        "semantic_admission_blocked": True,
        "experience_refresh_required": True,
    }


def stored_brief() -> dict:
    return {
        "dag_revision": 3,
        "experience_dag_revision": 3,
        "experience_dag_delta": 0,
        "semantic_admission_blocked": False,
        "experience_refresh_required": False,
    }


class ExperienceRecoveryTests(unittest.TestCase):
    def test_prepared_recovery_uses_the_authoritative_ledger_brief_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = RunIdentity(root, "hard-interactions", "recovery")
            run_dir = identity.run_dir
            run_dir.mkdir(parents=True)
            registry = fixture_registry()
            baseline = complete_point(registry)
            entry = record(
                "000",
                "fresh",
                [],
                baseline,
                score=0.5,
                status="keep",
            )
            entry["dag_revision"] = 1
            prior_experience = {**PROPOSAL, "generation": 0, "summary": "prior"}
            prior_experience["dag_revision"] = 0
            (run_dir / "background.md").write_text(
                background_text(registry), encoding="utf-8"
            )
            atomic_write_json(
                run_dir / "ledger.json",
                {
                    "task": "hard-interactions",
                    "tag": "recovery",
                    "metric": "validation_loss",
                    "search_space": space_receipt(registry),
                    "search_space_state": empty_search_space_state(),
                    "dag_revision": 1,
                    "records": [entry],
                    "experience": prior_experience,
                },
            )
            models = RecordingModels()
            toolchain = InterruptFirstStoreToolchain()
            refresh = ExperienceRefresh(identity, toolchain, models)
            brief = toolchain.ledger_brief(run_dir)
            self.assertNotIn("experience_cursor", brief)
            self.assertEqual(brief["experience_dag_revision"], 0)
            self.assertTrue(brief["experience_refresh_required"])

            with self.assertRaises(SimulatedKill):
                refresh.run(brief)

            result = refresh.run(toolchain.ledger_brief(run_dir))

            self.assertTrue(result["ok"])
            self.assertEqual(models.calls, 1)
            self.assertEqual(toolchain.store_calls, 2)
            stored = toolchain.ledger_experience(run_dir)
            self.assertEqual(stored["dag_revision"], 1)
            self.assertEqual(stored["summary"], PROPOSAL["summary"])
            self.assertFalse(
                toolchain.ledger_brief(run_dir)["experience_refresh_required"]
            )

    def test_store_then_apply_failure_resumes_without_inference_or_second_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            models = RecordingModels()
            toolchain = RecoveryToolchain(apply_failure="before")
            refresh = ExperienceRefresh(identity, toolchain, models)

            with self.assertRaisesRegex(RuntimeError, "space-state worker failed"):
                refresh.run(stale_brief())

            receipt_path = (
                identity.run_dir
                / ".orchestrator"
                / "experience-refresh-dag-3.json"
            )
            self.assertEqual(json.loads(receipt_path.read_text())["status"], "stored")
            self.assertTrue(refresh.has_pending(stored_brief()))
            self.assertEqual(
                next_transition(
                    CoordinatorState(task_name="toy", tag="smoke"),
                    ledger_exists=True,
                    ledger_brief=stored_brief(),
                    has_provided_baseline=False,
                    experience_refresh_pending=True,
                ),
                Transition.REFRESH_EXPERIENCE,
            )

            result = refresh.run(stored_brief())

            self.assertEqual(result, {"ok": True, "revision": 1})
            self.assertEqual(models.calls, 1)
            self.assertEqual(toolchain.views_calls, 1)
            self.assertEqual(toolchain.validation_calls, 1)
            self.assertEqual(toolchain.store_calls, 1)
            self.assertEqual(toolchain.apply_calls, 2)
            self.assertEqual(json.loads(receipt_path.read_text())["status"], "completed")
            self.assertFalse(refresh.has_pending(stored_brief()))

    def test_kill_after_store_reconciles_prepared_receipt_without_second_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            models = RecordingModels()
            toolchain = RecoveryToolchain()
            refresh = ExperienceRefresh(identity, toolchain, models)
            interrupted = False

            def interrupt_stored_receipt(path, value):
                nonlocal interrupted
                if (
                    isinstance(value, dict)
                    and value.get("status") == "stored"
                    and not interrupted
                ):
                    interrupted = True
                    raise SimulatedKill("killed after experience store")
                atomic_write_json(path, value)

            with mock.patch.object(
                experience_module,
                "atomic_write_json",
                side_effect=interrupt_stored_receipt,
            ):
                with self.assertRaises(SimulatedKill):
                    refresh.run(stale_brief())

            receipt_path = (
                identity.run_dir
                / ".orchestrator"
                / "experience-refresh-dag-3.json"
            )
            self.assertEqual(json.loads(receipt_path.read_text())["status"], "prepared")
            self.assertEqual(toolchain.store_calls, 1)

            result = refresh.run(stored_brief())

            self.assertEqual(result, {"ok": True, "revision": 1})
            self.assertEqual(models.calls, 1)
            self.assertEqual(toolchain.store_calls, 1)
            self.assertEqual(toolchain.apply_calls, 1)
            self.assertEqual(json.loads(receipt_path.read_text())["status"], "completed")

    def test_kill_after_apply_retries_idempotent_apply_and_completes_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            models = RecordingModels()
            toolchain = RecoveryToolchain(apply_failure="after")
            refresh = ExperienceRefresh(identity, toolchain, models)

            with self.assertRaises(SimulatedKill):
                refresh.run(stale_brief())

            self.assertEqual(toolchain.space_revision, 1)
            result = refresh.run(stored_brief())

            self.assertEqual(result, {"ok": True, "revision": 1})
            self.assertEqual(toolchain.space_revision, 1)
            self.assertEqual(toolchain.apply_calls, 2)
            self.assertEqual(models.calls, 1)
            self.assertEqual(toolchain.store_calls, 1)
            receipt = json.loads(
                (
                    identity.run_dir
                    / ".orchestrator"
                    / "experience-refresh-dag-3.json"
                ).read_text()
            )
            self.assertEqual(receipt["completion_mode"], "reconciled")

    def test_tampered_prepared_proposal_blocks_before_recovery_side_effects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            models = RecordingModels()
            toolchain = RecoveryToolchain()
            refresh = ExperienceRefresh(identity, toolchain, models)
            interrupted = False

            def interrupt_stored_receipt(path, value):
                nonlocal interrupted
                if (
                    isinstance(value, dict)
                    and value.get("status") == "stored"
                    and not interrupted
                ):
                    interrupted = True
                    raise SimulatedKill("killed after experience store")
                atomic_write_json(path, value)

            with mock.patch.object(
                experience_module,
                "atomic_write_json",
                side_effect=interrupt_stored_receipt,
            ):
                with self.assertRaises(SimulatedKill):
                    refresh.run(stale_brief())

            proposal_path = (
                identity.run_dir
                / ".orchestrator"
                / "experience-proposal-dag-3.json"
            )
            proposal_path.write_text('{"schema_version": 4}\n')

            with self.assertRaisesRegex(ArtifactError, "changed after validation"):
                refresh.run(stored_brief())

            self.assertEqual(models.calls, 1)
            self.assertEqual(toolchain.store_calls, 1)
            self.assertEqual(toolchain.apply_calls, 0)

    def test_second_strike_rejection_degrades_to_a_terminal_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            models = RecordingModels()
            toolchain = RejectingToolchain()
            refresh = ExperienceRefresh(identity, toolchain, models)

            result = refresh.run(stale_brief())

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["dag_revision"], 3)
            self.assertEqual(result["error"]["type"], "ValidationRejected")
            self.assertEqual(models.calls, 2)
            self.assertEqual(toolchain.validation_calls, 2)
            self.assertEqual(toolchain.store_calls, 0)
            self.assertEqual(toolchain.apply_calls, 0)
            self.assertEqual(toolchain.stored_experience, {"dag_revision": 2})
            receipt_path = (
                identity.run_dir / ".orchestrator" / "experience-refresh-dag-3.json"
            )
            receipt = json.loads(receipt_path.read_text())
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(receipt["completion_mode"], "failed")
            self.assertEqual(receipt["error"]["type"], "ValidationRejected")
            self.assertIn("unknown run", receipt["error"]["message"])
            self.assertFalse(refresh.has_pending(stale_brief()))

            # Re-entry (an explicit resume after the coordinator blocked on the
            # failure) admits one fresh attempt instead of idling forever.
            again = refresh.run(stale_brief())

            self.assertEqual(again["status"], "failed")
            self.assertEqual(models.calls, 4)
            self.assertEqual(toolchain.validation_calls, 4)
            self.assertEqual(
                json.loads(receipt_path.read_text())["completion_mode"], "failed"
            )

            later_brief = {
                **stale_brief(),
                "dag_revision": 4,
                "experience_dag_delta": 2,
            }
            later = refresh.run(later_brief)

            self.assertEqual(later["status"], "failed")
            self.assertEqual(models.calls, 6)
            later_receipt = json.loads(
                (
                    identity.run_dir
                    / ".orchestrator"
                    / "experience-refresh-dag-4.json"
                ).read_text()
            )
            self.assertEqual(later_receipt["status"], "completed")
            self.assertEqual(later_receipt["completion_mode"], "failed")
            self.assertFalse(refresh.has_pending(later_brief))
            self.assertFalse(refresh.has_pending(stale_brief()))

    def test_model_quality_inference_failures_degrade_to_failed_receipts(self) -> None:
        cases = (
            InferenceContractError("structured snapshot failed its schema"),
            InferenceError("Claude Agent SDK stopped with 'max_turns'"),
        )
        for error in cases:
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as tmp,
            ):
                identity = RunIdentity(Path(tmp), "toy", "smoke")
                models = FailingModels(error)
                toolchain = RecoveryToolchain()
                refresh = ExperienceRefresh(identity, toolchain, models)

                result = refresh.run(stale_brief())

                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error"]["type"], type(error).__name__)
                self.assertEqual(models.calls, 1)
                self.assertEqual(toolchain.store_calls, 0)
                self.assertEqual(toolchain.apply_calls, 0)
                receipt = json.loads(
                    (
                        identity.run_dir
                        / ".orchestrator"
                        / "experience-refresh-dag-3.json"
                    ).read_text()
                )
                self.assertEqual(receipt["status"], "completed")
                self.assertEqual(receipt["completion_mode"], "failed")
                self.assertEqual(receipt["error"]["type"], type(error).__name__)
                self.assertFalse(refresh.has_pending(stale_brief()))

    def test_upstream_inference_error_reraises_without_a_failed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "smoke")
            upstream = InferenceError(
                "Claude Messages request failed: Error code: 502 - "
                "{'error': {'message': 'Upstream service temporarily unavailable', "
                "'type': 'upstream_error'}, 'type': 'error'}"
            )
            models = FailingModels(upstream)
            toolchain = RecoveryToolchain()
            refresh = ExperienceRefresh(identity, toolchain, models)

            with self.assertRaises(InferenceError):
                refresh.run(stale_brief())

            self.assertEqual(models.calls, 1)
            self.assertEqual(toolchain.store_calls, 0)
            self.assertFalse(
                (
                    identity.run_dir / ".orchestrator" / "experience-refresh-dag-3.json"
                ).exists()
            )
            self.assertFalse(refresh.has_pending(stale_brief()))


if __name__ == "__main__":
    unittest.main()
