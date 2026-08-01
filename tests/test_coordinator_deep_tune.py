from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hieraresearch.coordinator import ExperimentCoordinator, RunControls  # noqa: E402
from hieraresearch.models import (  # noqa: E402
    ActiveRound,
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
    ) -> dict[str, object]:
        del run_dir, candidate_path, report_path
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


if __name__ == "__main__":
    unittest.main()
