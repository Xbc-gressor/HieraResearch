from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hieraresearch.artifacts import ArtifactError  # noqa: E402
from hieraresearch.coordinator import ExperimentCoordinator, RunControls  # noqa: E402
from hieraresearch.models import (  # noqa: E402
    ActiveRound,
    CoordinatorPhase,
    CoordinatorState,
    DeepTuneOutcome,
    DeepTuneSelection,
    RunIdentity,
)
from hieraresearch.tuning import DeepTuner  # noqa: E402


class LedgerBriefToolchain:
    def ledger_brief(self, run_dir: Path) -> dict[str, int]:
        del run_dir
        return {"evaluations_attempted": 4}


class CommittedThenCrashedTuner:
    def __init__(self, selection: DeepTuneSelection):
        self.selection = selection
        self.select_calls = 0
        self.committed_run_ids: list[str | None] = []

    def select(self) -> DeepTuneSelection:
        self.select_calls += 1
        return self.selection

    def run(self, selection: DeepTuneSelection) -> DeepTuneOutcome:
        self.committed_run_ids.append(selection.run_id)
        raise RuntimeError("simulated crash after selected tuning side effect")


class ResumePinnedTuner:
    def __init__(self, outcome: DeepTuneOutcome):
        self.outcome = outcome
        self.run_selections: list[DeepTuneSelection] = []

    def select(self) -> DeepTuneSelection:  # pragma: no cover - contract canary
        raise AssertionError("restart recomputed the round's tuning selection")

    def run(self, selection: DeepTuneSelection) -> DeepTuneOutcome:
        self.run_selections.append(selection)
        return self.outcome


class IdleTuner:
    """A deep tuner with nothing eligible: no-op selection, no objective call."""

    def __init__(self):
        self.selection = DeepTuneSelection(
            run_id=None,
            reason="no tunable candidate",
            trial_cap=None,
            input_revision="sha256:" + "0" * 64,
        )

    def select(self) -> DeepTuneSelection:
        return self.selection

    def run(self, selection: DeepTuneSelection) -> DeepTuneOutcome:
        assert selection is self.selection
        return DeepTuneOutcome(
            tuned_run_id=None,
            ledger_updated=False,
            reason=selection.reason,
        )


class PinnedToolchain:
    def __init__(self):
        self.phase_c_paths: list[tuple[Path, Path]] = []
        self.finalized_run_ids: list[str] = []

    def select_tuning_candidate(self, run_dir: Path) -> dict:
        del run_dir
        raise AssertionError("DeepTuner.run(selection) performed a fresh selection")

    def phase_c_action(
        self,
        candidate_path: Path,
        report_path: Path,
    ) -> dict[str, str]:
        self.phase_c_paths.append((candidate_path, report_path))
        return {"action": "finalize", "reason": "terminal_ok"}

    def finalize_tuning(
        self,
        run_dir: Path,
        run_id: str,
        candidate_path: Path,
        report_path: Path,
        task_config: dict[str, object],
    ) -> dict[str, object]:
        del run_dir, candidate_path, report_path, task_config
        self.finalized_run_ids.append(run_id)
        return {
            "status": "ok",
            "run_id": run_id,
            "ledger_updated": True,
            "final_best_score": 0.5,
        }

    def ledger_record(self, run_dir: Path, run_id: str) -> dict[str, object]:
        del run_dir, run_id
        return {"tune": True, "final_best_score": 0.5}


class CoordinatorDeepTuneRestartTests(unittest.TestCase):
    def test_selection_is_durable_before_side_effect_and_reused_after_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "restart")
            selection = DeepTuneSelection(
                run_id="000",
                reason="best untuned candidate",
                trial_cap=2,
                input_revision="sha256:" + "a" * 64,
            )
            state = CoordinatorState(
                task_name=identity.task_name,
                tag=identity.tag,
                active_round=ActiveRound(
                    round_id=3,
                    actions=[],
                    admission_complete=True,
                    evaluations_before=3,
                ),
            )
            first = ExperimentCoordinator(
                identity,
                LedgerBriefToolchain(),
                models=object(),
                controls=RunControls(),
            )
            first.state = state
            first.store.save(state)
            crashed = CommittedThenCrashedTuner(selection)
            first.deep_tuner = crashed

            with self.assertRaisesRegex(
                RuntimeError,
                "after selected tuning side effect",
            ):
                first._deep_tune()

            durable = first.store.load(identity.task_name, identity.tag)
            self.assertEqual(crashed.select_calls, 1)
            self.assertEqual(crashed.committed_run_ids, ["000"])
            self.assertEqual(durable.active_round.deep_tune_selection, selection)
            self.assertIsNone(durable.active_round.deep_tune_outcome)

            outcome = DeepTuneOutcome(
                tuned_run_id="000",
                ledger_updated=True,
                reason=selection.reason,
            )
            resumed = ExperimentCoordinator(
                identity,
                LedgerBriefToolchain(),
                models=object(),
                controls=RunControls(),
            )
            resumed.state = durable
            retry = ResumePinnedTuner(outcome)
            resumed.deep_tuner = retry

            resumed._deep_tune()

            completed = resumed.store.load(identity.task_name, identity.tag)
            self.assertEqual(retry.run_selections, [selection])
            self.assertEqual(completed.active_round.deep_tune_selection, selection)
            self.assertEqual(completed.active_round.deep_tune_outcome, outcome)

            resumed._close_finished_round()

            receipt = resumed.store.root / "rounds" / "000003.json"
            self.assertTrue(receipt.is_file())
            recorded = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual(
                recorded["round"]["deep_tune_selection"]["run_id"],
                "000",
            )
            self.assertTrue(
                recorded["round"]["deep_tune_outcome"]["ledger_updated"]
            )
            self.assertIsNone(resumed.state.active_round)
            self.assertEqual(
                resumed.store.complete_round(completed.active_round),
                receipt,
            )

            recorded["round"]["deep_tune_selection"]["input_revision"] = (
                "sha256:not-a-real-revision"
            )
            resumed.store.path.write_text(
                json.dumps(
                    {
                        **completed.to_dict(),
                        "active_round": recorded["round"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "lowercase sha256"):
                resumed.store.load(identity.task_name, identity.tag)

    def test_deep_tuner_executes_only_the_persisted_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            identity = RunIdentity(Path(tmp), "toy", "pinned")
            toolchain = PinnedToolchain()
            selection = DeepTuneSelection(
                run_id="007",
                reason="persisted round selection",
                trial_cap=3,
                input_revision="sha256:" + "b" * 64,
            )

            outcome = DeepTuner(identity, toolchain, task_config={}).run(selection)

            candidate_dir = identity.run_dir / "candidates" / "007"
            self.assertEqual(
                toolchain.phase_c_paths,
                [(candidate_dir / "train.py", candidate_dir / "tune_report.json")],
            )
            self.assertEqual(toolchain.finalized_run_ids, ["007"])
            self.assertEqual(outcome.tuned_run_id, "007")
            self.assertTrue(outcome.ledger_updated)
            self.assertEqual(outcome.reason, "terminal_ok")


class NoProgressGuardTests(unittest.TestCase):
    """A round emptied by dropped ideation failures is not semantic exhaustion.

    Live regression: two malformed idea responses dropped both actions of a
    round and the guard blocked the run while most of the objective budget
    remained. A model-side empty round earns one bounded retry; only
    consecutive repeats — or a genuinely action-free SELECT — prove progress
    is impossible. The cause is recovered from the durable admission-failure
    receipts, not from coordinator memory.
    """

    @staticmethod
    def _write_drop_receipts(identity: RunIdentity, round_id: int, count: int) -> None:
        directory = identity.run_dir / ".orchestrator" / "admission_failures"
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            (directory / f"round-{round_id}-{index + 1:03d}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "ideation_failure",
                        "run_id": f"{index + 1:03d}",
                        "op": "fresh",
                        "parents": [],
                        "outcome": "action_dropped",
                        "error": "InferenceContractError: bad fields",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    @classmethod
    def _coordinator(
        cls, root: Path, *, evaluations_attempted: int, ideation_drops: int
    ) -> ExperimentCoordinator:
        identity = RunIdentity(root, "toy", "guard")
        toolchain = SimpleNamespace(
            ledger_brief=lambda run_dir: {
                "evaluations_attempted": evaluations_attempted
            },
        )
        coordinator = ExperimentCoordinator(
            identity,
            toolchain,
            models=object(),
            controls=RunControls(),
        )
        coordinator.state = CoordinatorState(
            task_name=identity.task_name,
            tag=identity.tag,
            active_round=ActiveRound(
                round_id=0,
                actions=[],
                admission_complete=True,
                evaluations_before=4,
            ),
        )
        coordinator.deep_tuner = IdleTuner()
        cls._write_drop_receipts(identity, round_id=0, count=ideation_drops)
        return coordinator

    def test_ideation_emptied_round_earns_one_bounded_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(
                Path(tmp), evaluations_attempted=4, ideation_drops=2
            )

            coordinator._deep_tune()

            self.assertEqual(coordinator.state.phase, CoordinatorPhase.RUNNING)
            self.assertEqual(coordinator.state.no_progress_cycles, 1)

            # A second consecutive ideation-emptied round proves the model
            # cannot currently produce a usable idea; block with the reason
            # that points at the durable failure receipts.
            coordinator.state.active_round = ActiveRound(
                round_id=1,
                actions=[],
                admission_complete=True,
                evaluations_before=4,
            )
            self._write_drop_receipts(coordinator.identity, round_id=1, count=2)
            coordinator._deep_tune()

            self.assertEqual(coordinator.state.phase, CoordinatorPhase.BLOCKED)
            self.assertIn("no_progress_possible", coordinator.state.stop_condition)
            self.assertIn("ideation", coordinator.state.stop_condition)

    def test_semantically_empty_round_still_blocks_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(
                Path(tmp), evaluations_attempted=4, ideation_drops=0
            )

            coordinator._deep_tune()

            self.assertEqual(coordinator.state.phase, CoordinatorPhase.BLOCKED)
            self.assertIn(
                "semantic admission", coordinator.state.stop_condition
            )

    def test_objective_progress_resets_the_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(
                Path(tmp), evaluations_attempted=5, ideation_drops=2
            )
            coordinator.state.no_progress_cycles = 1

            coordinator._deep_tune()

            self.assertEqual(coordinator.state.phase, CoordinatorPhase.RUNNING)
            self.assertEqual(coordinator.state.no_progress_cycles, 0)

    def test_resumed_round_recovers_the_drop_cause_from_receipts(self) -> None:
        """The guard's cause survives a process restart via durable receipts.

        Upgrade window: a process parked between an ideation-emptied
        admission and deep tuning carries no marker in `state.json`. The
        admission-failure receipts on disk are the authoritative record, so a
        coordinator that reloads the state from disk must still tolerate the
        round instead of blocking as semantic exhaustion.
        """
        with tempfile.TemporaryDirectory() as tmp:
            parked = self._coordinator(
                Path(tmp), evaluations_attempted=4, ideation_drops=2
            )
            parked.store.save(parked.state)

            resumed = ExperimentCoordinator(
                parked.identity,
                SimpleNamespace(
                    ledger_brief=lambda run_dir: {"evaluations_attempted": 4}
                ),
                models=object(),
                controls=RunControls(),
            )
            resumed.state = resumed.store.load(
                parked.identity.task_name, parked.identity.tag
            )
            resumed.deep_tuner = IdleTuner()

            resumed._deep_tune()

            self.assertEqual(resumed.state.phase, CoordinatorPhase.RUNNING)
            self.assertEqual(resumed.state.no_progress_cycles, 1)

    def test_malformed_drop_receipt_blocks_as_corruption(self) -> None:
        """The authority is the receipt's content, never its filename.

        An unparseable file in the admission-failure directory is artifact
        corruption; classifying the round from it would change control flow
        on evidence nobody validated, so the guard refuses with ArtifactError
        instead.
        """
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(
                Path(tmp), evaluations_attempted=4, ideation_drops=0
            )
            directory = (
                coordinator.identity.run_dir
                / ".orchestrator"
                / "admission_failures"
            )
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "round-0-001.json").write_text(
                "not json{", encoding="utf-8"
            )

            with self.assertRaises(ArtifactError):
                coordinator._deep_tune()

    def test_foreign_receipt_is_not_counted_as_an_ideation_drop(self) -> None:
        """A contradictory receipt must not reclassify a semantic-empty round.

        `kind: unrelated` / `outcome: success` content under a matching
        filename is not evidence of a dropped ideation failure; treating it
        as such would convert an immediate block into a retry on fabricated
        grounds.
        """
        with tempfile.TemporaryDirectory() as tmp:
            coordinator = self._coordinator(
                Path(tmp), evaluations_attempted=4, ideation_drops=0
            )
            directory = (
                coordinator.identity.run_dir
                / ".orchestrator"
                / "admission_failures"
            )
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "round-0-001.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "unrelated",
                        "run_id": "001",
                        "op": "fresh",
                        "parents": [],
                        "outcome": "success",
                        "error": "",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactError):
                coordinator._deep_tune()


if __name__ == "__main__":
    unittest.main()
