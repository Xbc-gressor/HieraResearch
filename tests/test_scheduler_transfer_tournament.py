"""Deterministic tests for the anchor + donor-transfer challenger scheduler."""

from __future__ import annotations

import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

from tools.scheduler.contract import (  # noqa: E402
    CandidateView,
    ResourceContract,
)
from tools.scheduler.session import contract_for, decide_for_run  # noqa: E402
from tools.scheduler.state import SchedulerState, state_from_snapshot  # noqa: E402
from tools.scheduler.store import SchedulerStore  # noqa: E402
from tools.scheduler.transfer_tournament import (  # noqa: E402
    AmbiguousAnchorError,
    decide,
    full_tuning_reserve,
    generation_admission_cap,
    generation_candidate_reservation,
    generation_reserve,
)
from tools.scheduler import calibrate as scheduler_calibrate  # noqa: E402
from tools.scheduler import cli as scheduler_cli  # noqa: E402
import got_select  # noqa: E402
import inner_policy  # noqa: E402
from tuners.tune_tools import (  # noqa: E402
    _scheduler_selection,
    select_candidate,
)


CONTRACT = ResourceContract(
    bout_trials=10,
    max_bouts=3,
    k_eval=2,
    first_bout_trials=24,
    bout_cost_schedule=(24, 10, 10),
    transferred_first_bout_trials=10,
)


def candidate(
    run_id: str,
    score: float,
    bouts: int = 0,
    gain: float | None = None,
    **kwargs,
) -> CandidateView:
    return CandidateView(
        run_id=run_id,
        best_score=score,
        bouts_used=bouts,
        previous_gain=gain,
        **kwargs,
    )


def donor_candidate(
    run_id: str,
    score: float,
    bouts: int = 0,
    gain: float | None = None,
    *,
    donor_finite: bool = True,
    donor_snapshot_id: str | None = "donor-test",
) -> CandidateView:
    return candidate(
        run_id,
        score,
        bouts,
        gain,
        initialization_mode="global_donor",
        donor_snapshot_id=donor_snapshot_id,
        donor_evaluated=True,
        donor_finite=donor_finite,
    )


def state(*candidates: CandidateView, budget: int = 125) -> SchedulerState:
    return SchedulerState(
        global_best=min(item.best_score for item in candidates),
        remaining_budget=budget,
        candidates=tuple(candidates),
        contract=CONTRACT,
        n_roots=5,
        n_seed=5,
    )


class TransferTournamentPhaseTest(unittest.TestCase):
    """§6.1: the 44 -> 20 -> 10 -> 0 reserve ladder, all from facts."""

    def test_reserve_and_cap_track_the_four_phases(self) -> None:
        self.assertEqual(full_tuning_reserve(CONTRACT), 44)

        before = state(candidate("000", 1.2), candidate("001", 1.1))
        self.assertEqual(generation_reserve(before), 44)
        self.assertEqual(generation_admission_cap(before), 40)

        anchored = state(
            candidate("000", 1.0, bouts=1, gain=0.2),
            candidate("001", 1.1),
            budget=90,
        )
        self.assertEqual(generation_reserve(anchored), 20)
        self.assertEqual(generation_admission_cap(anchored), 35)

        first_segment = state(
            candidate("000", 1.0, bouts=1, gain=0.2),
            donor_candidate("001", 1.05, bouts=1, gain=0.05),
            budget=30,
        )
        self.assertEqual(generation_reserve(first_segment), 10)
        self.assertEqual(generation_admission_cap(first_segment), 0)

        # An anchor DEEP continuation spends a segment just the same.
        anchor_segment = state(
            candidate("000", 1.0, bouts=2, gain=0.0),
            budget=30,
        )
        self.assertEqual(generation_reserve(anchor_segment), 10)
        self.assertEqual(generation_admission_cap(anchor_segment), 0)

        complete = state(
            candidate("000", 1.0, bouts=2, gain=0.0),
            donor_candidate("001", 1.05, bouts=1, gain=0.05),
            budget=30,
        )
        self.assertEqual(generation_reserve(complete), 0)
        self.assertEqual(generation_admission_cap(complete), 15)

    def test_seed_gate_defers_while_roots_arrive(self) -> None:
        seeded = state(candidate("000", 1.2), candidate("001", 1.1))
        gated = SchedulerState(
            global_best=1.1,
            remaining_budget=125,
            candidates=seeded.candidates,
            contract=CONTRACT,
            n_roots=2,
            n_seed=5,
        )
        decision = decide(gated)
        self.assertEqual(decision.action, "DEFER")
        self.assertIn("seed set incomplete", decision.reason)

    def test_anchor_is_the_best_ordinary_untuned_seed(self) -> None:
        # A better-scoring global_donor candidate is never the anchor: the
        # anchor bout is an ordinary 24-eval INITIAL.
        decision = decide(
            state(
                candidate("000", 1.2),
                candidate("001", 1.1),
                donor_candidate("002", 1.0),
                budget=100,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "001"))

    def test_budget_below_the_full_reserve_stops(self) -> None:
        decision = decide(
            state(candidate("000", 1.2), candidate("001", 1.1), budget=43)
        )
        self.assertEqual(decision.action, "STOP")
        self.assertIn("44", decision.reason)

    def test_competing_ordinary_anchors_fail_closed(self) -> None:
        contested = state(
            candidate("000", 1.0, bouts=1),
            candidate("001", 1.1, bouts=1),
            budget=90,
        )
        with self.assertRaises(AmbiguousAnchorError):
            decide(contested)
        with self.assertRaises(AmbiguousAnchorError):
            generation_reserve(contested)


class TransferTournamentPostAnchorTest(unittest.TestCase):
    """§6.1/§6.2: challenger selection, the responder rule, and fallbacks."""

    def _anchored(self, *extra: CandidateView, budget: int) -> SchedulerState:
        return state(
            candidate("000", 1.0, bouts=1, gain=0.2), *extra, budget=budget
        )

    def test_generation_continues_while_the_reserve_leaves_room(self) -> None:
        decision = decide(
            self._anchored(donor_candidate("001", 1.05), budget=44)
        )
        self.assertEqual(decision.action, "DEFER")
        self.assertEqual(
            decision.evidence_mode["phase"], "post_anchor_generation"
        )

    def test_first_segment_goes_to_the_best_finite_donor_challenger(self) -> None:
        decision = decide(
            self._anchored(
                donor_candidate("001", 1.05),
                donor_candidate("002", 1.03),
                # A crashed donor row never earns the TRANSFERRED segment.
                donor_candidate("003", 1.01, donor_finite=False),
                budget=20,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "002"))
        self.assertEqual(
            decision.evidence_mode["donor_eligible_challenger_ids"],
            ["002", "001"],
        )
        self.assertEqual(
            decision.evidence_mode["donor_snapshot_id"], "donor-test"
        )

    def test_no_challenger_continues_the_anchor(self) -> None:
        decision = decide(self._anchored(candidate("001", 1.1), budget=20))
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "000"))
        self.assertIn("no eligible donor challenger", decision.reason)

    def test_no_challenger_and_no_anchor_continuation_stops(self) -> None:
        decision = decide(
            state(
                candidate(
                    "000",
                    1.0,
                    bouts=1,
                    gain=0.2,
                    has_unresolved_descendant=True,
                ),
                candidate("001", 1.1),
                budget=20,
            )
        )
        self.assertEqual(decision.action, "STOP")

    def test_positive_gain_stays_with_the_responder(self) -> None:
        decision = decide(
            self._anchored(
                donor_candidate("001", 1.05, bouts=1, gain=0.05),
                donor_candidate("002", 1.03),
                budget=30,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "001"))
        self.assertEqual(decision.evidence_mode["phase"], "post_anchor_segments")
        self.assertEqual(
            decision.evidence_mode["post_anchor_segments_spent"], 1
        )
        self.assertEqual(
            decision.evidence_mode["first_post_anchor_target"], "001"
        )
        self.assertEqual(decision.evidence_mode["anchor_run_id"], "000")

    def test_zero_gain_switches_to_an_untouched_challenger(self) -> None:
        decision = decide(
            self._anchored(
                donor_candidate("001", 1.05, bouts=1, gain=0.0),
                donor_candidate("002", 1.03),
                budget=30,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "002"))
        self.assertIn("zero gain", decision.reason)

    def test_zero_gain_without_challenger_continues_the_anchor(self) -> None:
        decision = decide(
            self._anchored(
                donor_candidate("001", 1.05, bouts=1, gain=0.0),
                budget=30,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "000"))
        self.assertIn("continue the anchor", decision.reason)

    def test_zero_gain_on_an_anchor_first_segment_stops(self) -> None:
        # The first segment was already the anchor's DEEP continuation; a
        # zero-gain anchor earns no further segment.
        decision = decide(
            state(candidate("000", 1.0, bouts=2, gain=0.0), budget=30)
        )
        self.assertEqual(decision.action, "STOP")

    def test_ineligible_responder_switches_instead_of_stopping(self) -> None:
        # A positive-gain responder whose next bout is blocked (here: an
        # unresolved primary descendant) yields the segment to the best
        # untouched challenger rather than stopping.
        decision = decide(
            state(
                candidate("000", 1.0, bouts=1, gain=0.2),
                candidate(
                    "001",
                    1.05,
                    bouts=1,
                    gain=0.05,
                    initialization_mode="global_donor",
                    donor_snapshot_id="donor-test",
                    donor_evaluated=True,
                    donor_finite=True,
                    has_unresolved_descendant=True,
                ),
                donor_candidate("002", 1.03),
                budget=30,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "002"))
        self.assertIn("responder ineligible", decision.reason)


class TransferTournamentTerminalBudgetTest(unittest.TestCase):
    """§9.8: terminal screening fidelity once both segments are spent."""

    K3_CONTRACT = ResourceContract(
        bout_trials=10,
        max_bouts=3,
        k_eval=3,
        first_bout_trials=24,
        bout_cost_schedule=(24, 10, 10),
        transferred_first_bout_trials=10,
    )

    def _complete(self, budget: int) -> SchedulerState:
        return SchedulerState(
            global_best=1.0,
            remaining_budget=budget,
            candidates=(
                candidate("000", 1.0, bouts=2, gain=0.0),
                donor_candidate("001", 1.05, bouts=1, gain=0.0),
            ),
            contract=self.K3_CONTRACT,
            n_roots=5,
            n_seed=5,
        )

    def test_full_screening_runs_normally(self) -> None:
        complete = self._complete(6)
        self.assertEqual(generation_admission_cap(complete), 2)
        self.assertEqual(generation_candidate_reservation(complete), 3)
        self.assertEqual(decide(complete).action, "DEFER")

    def test_two_left_admit_a_single_two_row_screen(self) -> None:
        complete = self._complete(2)
        self.assertEqual(generation_admission_cap(complete), 1)
        self.assertEqual(generation_candidate_reservation(complete), 2)
        self.assertEqual(decide(complete).action, "DEFER")

    def test_one_left_completes_with_an_explicit_unused_budget_reason(self) -> None:
        complete = self._complete(1)
        self.assertEqual(generation_admission_cap(complete), 0)
        decision = decide(complete)
        self.assertEqual(decision.action, "STOP")
        self.assertIn("unused", decision.reason)

    def test_reserve_is_never_invaded_before_completion(self) -> None:
        anchored = SchedulerState(
            global_best=1.0,
            remaining_budget=21,
            candidates=(candidate("000", 1.0, bouts=1, gain=0.2),),
            contract=self.K3_CONTRACT,
            n_roots=5,
            n_seed=5,
        )
        self.assertEqual(generation_reserve(anchored), 20)
        self.assertEqual(generation_admission_cap(anchored), 0)


class TransferTournamentReplayTest(unittest.TestCase):
    """§9.6: snapshot round-trip and receipt-version dispatch."""

    def test_snapshot_round_trip_preserves_transfer_facts(self) -> None:
        original = state(
            candidate("000", 1.0, bouts=1, gain=0.2),
            donor_candidate("001", 1.05),
            budget=90,
        )
        restored = state_from_snapshot(original.snapshot())
        contract = restored.contract
        self.assertEqual(contract.bout_cost_schedule, (24, 10, 10))
        self.assertEqual(contract.transferred_first_bout_trials, 10)
        restored_candidate = next(
            item for item in restored.candidates if item.run_id == "001"
        )
        self.assertEqual(
            restored_candidate.initialization_mode, "global_donor"
        )
        self.assertEqual(restored_candidate.donor_snapshot_id, "donor-test")
        self.assertTrue(restored_candidate.donor_evaluated)
        self.assertTrue(restored_candidate.donor_finite)

    def test_decide_is_a_pure_function_of_the_snapshot(self) -> None:
        # Replay re-derives the decision from the stored snapshot alone, so
        # both sides must agree in every phase.
        states = [
            state(candidate("000", 1.2), candidate("001", 1.1)),
            state(candidate("000", 1.0, bouts=1, gain=0.2), budget=20),
            state(
                candidate("000", 1.0, bouts=1, gain=0.2),
                donor_candidate("001", 1.05, bouts=1, gain=0.0),
                donor_candidate("002", 1.03),
                budget=30,
            ),
            state(
                candidate("000", 1.0, bouts=2, gain=0.0),
                donor_candidate("001", 1.05, bouts=1, gain=0.05),
                budget=30,
            ),
        ]
        for original in states:
            with self.subTest(phase=original.candidates):
                replayed = state_from_snapshot(original.snapshot())
                self.assertEqual(
                    (decide(original).action, decide(original).run_id),
                    (decide(replayed).action, decide(replayed).run_id),
                )


class TransferTournamentSessionTest(unittest.TestCase):
    """Live-run wiring: contract, receipt version, idempotence, replay."""

    def _run_dir(self, tmp: str, *, max_evaluations: int = 100) -> Path:
        run_dir = Path(tmp) / "runs" / "toy" / "tag"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(
            json.dumps(
                {
                    "max_evaluations": max_evaluations,
                    "tuner": {
                        "scheduler_policy": "anchor_transfer_challenger_v1",
                        "inner_policy": "hebo24-transfer10-hebo10",
                        "deep_tune_budget_fraction": None,
                        "deep_tune_per_candidate_cap": 44,
                    },
                    "got": {"n_seed": 2},
                }
            )
        )
        return run_dir

    def _write_attempts(self, run_dir: Path, count: int) -> None:
        attempt = json.dumps({"schema_version": 1, "kind": "score_attempt"})
        (run_dir / "evaluation_attempts.jsonl").write_text(
            "\n".join([attempt] * count) + ("\n" if count else "")
        )

    def test_contract_prices_the_transferred_first_bout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            ledger = run_dir / "ledger.json"
            ledger.write_text(json.dumps({"records": []}))
            contract = contract_for(ledger)
            self.assertEqual(contract.bout_cost_schedule, (24, 10, 10))
            self.assertEqual(contract.transferred_first_bout_trials, 10)
            self.assertIsNone(contract.numeric_required_from_bout_index)

    def test_live_decide_receipt_idempotence_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            self._write_attempts(run_dir, 0)
            ledger = run_dir / "ledger.json"
            ledger.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "run_id": "000",
                                "status": "keep",
                                "best_warm_score": 1.2,
                            },
                            {
                                "run_id": "001",
                                "status": "keep",
                                "best_warm_score": 1.1,
                            },
                        ]
                    }
                )
            )

            view = decide_for_run(ledger)

            self.assertEqual(
                view["policy_version"], "scheduler-anchor-transfer-challenger-v1"
            )
            self.assertEqual((view["action"], view["run_id"]), ("TUNE", "001"))
            self.assertIsNone(view["prior_id"])
            mode = view["evidence_mode"]
            self.assertEqual(mode["phase"], "early_anchor")
            self.assertEqual(mode["generation_reserve"], 44)
            self.assertIsNone(mode["anchor_run_id"])

            # Same round, same snapshot: the open decision is reused, never
            # re-minted.
            again = decide_for_run(ledger)
            self.assertTrue(again["reused_open_decision"])
            self.assertEqual(again["decision_id"], view["decision_id"])

            # Replay dispatches on the receipt's policy_version.
            output = io.StringIO()
            with redirect_stdout(output):
                code = scheduler_cli.cmd_replay(
                    SimpleNamespace(
                        ledger=str(ledger),
                        decision_id=view["decision_id"],
                    )
                )
            replayed = json.loads(output.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(replayed["reproduced"])

            replay_all = scheduler_calibrate.replay_all(SchedulerStore(run_dir))
            self.assertTrue(replay_all["reproducible"])

    def test_session_selects_the_finite_donor_challenger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            self._write_attempts(run_dir, 80)  # remaining 20 == the reserve
            report_dir = run_dir / "candidates" / "001"
            report_dir.mkdir(parents=True)
            (report_dir / "tune_report.json").write_text(
                json.dumps(
                    {
                        "phase_a": {
                            "status": "ok",
                            "initialization_mode": "global_donor",
                            "global_donor_transfer": {
                                "donor": {"snapshot_id": "donor-abc"}
                            },
                            "global_donor_observation": {"status": "finite"},
                        },
                        "phase_c": {"stages": []},
                    }
                )
            )
            ledger = run_dir / "ledger.json"
            ledger.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "run_id": "000",
                                "op": "fresh",
                                "status": "keep",
                                "best_warm_score": 1.2,
                                "final_best_score": 1.0,
                                "tuning_bouts": 1,
                                "tune": True,
                            },
                            {
                                "run_id": "001",
                                "op": "fresh",
                                "status": "keep",
                                "best_warm_score": 0.9,
                            },
                        ]
                    }
                )
            )

            selection = _scheduler_selection(ledger, None)

            self.assertEqual(selection["run_id"], "001")
            self.assertEqual(selection["bout_regime"], inner_policy.TRANSFERRED)
            self.assertEqual(selection["budget_allocation"]["trial_cap"], 10)
            scheduler = selection["scheduler"]
            self.assertEqual(
                scheduler["policy_version"],
                "scheduler-anchor-transfer-challenger-v1",
            )
            mode = scheduler["evidence_mode"]
            self.assertEqual(mode["anchor_run_id"], "000")
            self.assertEqual(mode["donor_eligible_challenger_ids"], ["001"])
            self.assertEqual(mode["donor_snapshot_id"], "donor-abc")


class TransferTournamentGotSelectTest(unittest.TestCase):
    """got_select reserve/cap wiring in both output modes."""

    def _run_dir(self, tmp: str) -> Path:
        run_dir = Path(tmp) / "runs" / "toy" / "tag"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(
            json.dumps(
                {
                    "max_evaluations": 100,
                    "tuner": {
                        "scheduler_policy": "anchor_transfer_challenger_v1",
                        "inner_policy": "hebo24-transfer10-hebo10",
                        "deep_tune_budget_fraction": None,
                        "deep_tune_per_candidate_cap": 44,
                    },
                }
            )
        )
        return run_dir

    def _decide(self, ledger: Path, **overrides) -> dict:
        output = io.StringIO()
        with redirect_stdout(output):
            got_select.cmd_decide(
                SimpleNamespace(ledger=str(ledger), cfg=None, **overrides)
            )
        return json.loads(output.getvalue())

    def test_round_one_bootstrap_carries_the_44_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            payload = self._decide(run_dir / "ledger.json")
            self.assertEqual(
                payload["diag"]["tournament_generation_reserve"], 44
            )
            self.assertEqual(
                payload["diag"]["candidate_admission_cap"], 18
            )

    def test_post_anchor_reserve_blocks_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            attempt = json.dumps({"schema_version": 1, "kind": "score_attempt"})
            # Anchor done (24 charged): reserve drops to 20; with 22 left no
            # three-row screen may be admitted.
            (run_dir / "evaluation_attempts.jsonl").write_text(
                "\n".join([attempt] * 78) + "\n"
            )
            report_dir = run_dir / "candidates" / "001"
            report_dir.mkdir(parents=True)
            (report_dir / "tune_report.json").write_text(
                json.dumps(
                    {
                        "phase_a": {
                            "status": "ok",
                            "initialization_mode": "global_donor",
                            "global_donor_transfer": {
                                "donor": {"snapshot_id": "donor-abc"}
                            },
                            "global_donor_observation": {"status": "finite"},
                        },
                        "phase_c": {"stages": []},
                    }
                )
            )
            ledger = run_dir / "ledger.json"
            ledger.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "run_id": "000",
                                "op": "fresh",
                                "status": "keep",
                                "best_warm_score": 1.2,
                                "final_best_score": 1.1,
                                "tuning_bouts": 1,
                                "tune": True,
                            },
                            {
                                "run_id": "001",
                                "op": "fresh",
                                "status": "keep",
                                "best_warm_score": 1.15,
                            },
                        ]
                    }
                )
            )
            payload = self._decide(ledger)
            self.assertEqual(payload["actions"], [])
            self.assertEqual(payload["diag"]["objective_remaining"], 22)
            self.assertEqual(
                payload["diag"]["tournament_generation_reserve"], 20
            )
            self.assertEqual(payload["diag"]["candidate_admission_cap"], 0)

    def test_lanes_mode_records_the_same_reserve_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            document = self._decide(run_dir / "ledger.json", mode="lanes")
            self.assertEqual(
                document["budget"]["tournament_generation_reserve"], 44
            )
            self.assertEqual(document["budget"]["tournament_admission_cap"], 18)
            self.assertEqual(document["budget"]["admission_cap"], 18)
            self.assertEqual(
                document["budget"]["candidate_objective_reservation"], 3
            )


class LegacySelectionFailClosedTest(unittest.TestCase):
    """The mode-unaware legacy path refuses the transfer inner policy."""

    def test_selected_candidate_result_rejects_the_transfer_policy(self) -> None:
        ledger = {
            "records": [
                {"run_id": "000", "status": "keep", "best_warm_score": 1.0}
            ]
        }
        with self.assertRaisesRegex(ValueError, "anchor_transfer_challenger_v1"):
            select_candidate(
                ledger,
                n_min=1,
                top_percentile=80.0,
                bout_trials=10,
                policy_id=inner_policy.HEBO_TRANSFER_HEBO_POLICY_ID,
            )


if __name__ == "__main__":
    unittest.main()
