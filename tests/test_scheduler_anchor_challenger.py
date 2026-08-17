"""Deterministic tests for the early-anchor / late-challenger scheduler."""

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
    ineligibility_reason,
)
from tools.scheduler.session import contract_for, decide_for_run  # noqa: E402
from tools.scheduler.state import SchedulerState, state_from_snapshot  # noqa: E402
from tools.scheduler.tournament import (  # noqa: E402
    decide,
    full_tuning_reserve,
    generation_admission_cap,
    generation_reserve,
)
import got_select  # noqa: E402
from tools.scheduler import cli as scheduler_cli  # noqa: E402
from tuners.tune_tools import _scheduler_selection  # noqa: E402


CONTRACT = ResourceContract(
    bout_trials=10,
    max_bouts=3,
    k_eval=2,
    first_bout_trials=24,
    bout_cost_schedule=(24, 10, 10),
    numeric_required_from_bout_index=1,
)


def candidate(
    run_id: str,
    score: float,
    bouts: int = 0,
    gain: float | None = None,
) -> CandidateView:
    return CandidateView(
        run_id=run_id,
        best_score=score,
        bouts_used=bouts,
        previous_gain=gain,
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


class TournamentPolicyTest(unittest.TestCase):
    def test_reserve_tracks_anchor_and_challenger_phases(self) -> None:
        before = state(candidate("000", 1.2), candidate("001", 1.1))
        self.assertEqual(full_tuning_reserve(CONTRACT), 68)
        self.assertEqual(generation_reserve(before), 68)
        self.assertEqual(generation_admission_cap(before), 28)
        self.assertEqual(
            state_from_snapshot(before.snapshot()).contract.bout_cost_schedule,
            (24, 10, 10),
        )
        self.assertEqual(
            state_from_snapshot(
                before.snapshot()
            ).contract.numeric_required_from_bout_index,
            1,
        )

        after_anchor = state(
            candidate("000", 1.0, bouts=1, gain=0.2),
            candidate("001", 1.1),
            budget=90,
        )
        self.assertEqual(generation_reserve(after_anchor), 44)
        self.assertEqual(generation_admission_cap(after_anchor), 23)

    def test_best_seed_becomes_early_anchor(self) -> None:
        decision = decide(
            state(candidate("000", 1.2), candidate("001", 1.1), budget=100)
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "001"))

    def test_reserve_boundary_initializes_best_untuned_challenger(self) -> None:
        decision = decide(
            state(
                candidate("000", 1.05, bouts=1, gain=0.05),
                candidate("001", 1.04),
                candidate("002", 1.08),
                budget=44,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "001"))

    def test_later_bout_stays_with_responder(self) -> None:
        decision = decide(
            state(
                candidate("000", 1.01, bouts=2, gain=0.01),
                candidate("001", 1.02, bouts=1, gain=0.03),
                budget=10,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "000"))

    def test_ineligible_responder_switches_instead_of_stopping(self) -> None:
        # A positive-gain responder whose next bout is an SPSA DEEP bout
        # without a movable continuous dimension: the reserved second later
        # bout must switch to the other initialized candidate, not STOP.
        decision = decide(
            state(
                CandidateView(
                    run_id="000",
                    best_score=1.01,
                    bouts_used=2,
                    previous_gain=0.05,
                    has_movable_continuous=False,
                ),
                candidate("001", 1.02, bouts=1, gain=0.03),
                budget=10,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "001"))

    def test_later_bout_switches_after_zero_gain(self) -> None:
        decision = decide(
            state(
                candidate("000", 1.01, bouts=2, gain=0.0),
                candidate("001", 1.02, bouts=1, gain=0.03),
                budget=10,
            )
        )
        self.assertEqual((decision.action, decision.run_id), ("TUNE", "001"))

    def test_two_later_bouts_complete_the_tournament(self) -> None:
        decision = decide(
            state(
                candidate("000", 1.01, bouts=2, gain=0.0),
                candidate("001", 1.00, bouts=2, gain=0.02),
                budget=20,
            )
        )
        self.assertEqual(decision.action, "STOP")

    def test_turbo_requires_numeric_space_only_after_initial(self) -> None:
        categorical = CandidateView(
            run_id="000",
            best_score=1.0,
            bouts_used=0,
            has_movable_numeric=False,
        )
        self.assertIsNone(ineligibility_reason(categorical, 68, CONTRACT))
        self.assertEqual(
            ineligibility_reason(
                CandidateView(
                    run_id="000",
                    best_score=1.0,
                    bouts_used=1,
                    has_movable_numeric=False,
                ),
                44,
                CONTRACT,
            ),
            "no varying numeric dimension for this bout",
        )


class TournamentSessionTest(unittest.TestCase):
    def test_mixup_turbo_contract_comes_from_inner_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 100,
                        "tuner": {
                            "scheduler_policy": "anchor_challenger_v1",
                            "inner_policy": "mixup24-turbo20-v1",
                            "deep_tune_budget_fraction": None,
                            "deep_tune_per_candidate_cap": 44,
                        },
                        "got": {"n_seed": 2},
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
            (run_dir / "evaluation_attempts.jsonl").write_text("")

            contract = contract_for(ledger)

            self.assertEqual(contract.bout_cost_schedule, (24, 10, 10))
            self.assertEqual(contract.lifetime_cost(), 44)
            self.assertEqual(contract.numeric_required_from_bout_index, 1)
            selection = _scheduler_selection(ledger, None)
            self.assertEqual(
                selection["budget_allocation"]["trial_cap"], 24
            )
            self.assertEqual(
                selection["budget_allocation"]["bout_trials"], 24
            )

            config_path = run_dir / "framework_cfg.json"
            config = json.loads(config_path.read_text())
            config["tuner"]["inner_policy"] = "hebo24-turbo20-v1"
            config_path.write_text(json.dumps(config))
            hebo_contract = contract_for(ledger)
            self.assertEqual(hebo_contract.bout_cost_schedule, (24, 10, 10))
            self.assertEqual(hebo_contract.lifetime_cost(), 44)
            self.assertEqual(hebo_contract.numeric_required_from_bout_index, 1)

    def test_live_session_routes_to_tournament_without_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 100,
                        "tuner": {
                            "scheduler_policy": "anchor_challenger_v1",
                            "inner_policy": "localtr8-hebo10-hebo10-v1",
                        },
                        "got": {"n_seed": 2},
                    }
                )
            )
            (run_dir / "evaluation_attempts.jsonl").write_text("")
            ledger = run_dir / "ledger.json"
            ledger.write_text(
                json.dumps(
                    {
                        "records": [
                            {"run_id": "000", "status": "keep", "best_warm_score": 1.2},
                            {"run_id": "001", "status": "keep", "best_warm_score": 1.1},
                        ]
                    }
                )
            )

            view = decide_for_run(ledger)

            self.assertEqual(view["policy_version"], "scheduler-anchor-challenger-v1")
            self.assertEqual((view["action"], view["run_id"]), ("TUNE", "001"))
            self.assertEqual(view["state"].contract.bout_cost_schedule, (8, 10, 10))
            self.assertEqual(view["prior_id"], None)

            # Replay must dispatch on the receipt's policy_version: running
            # the recorded tournament decision back through the v3.2 rollout
            # policy would report a false divergence.
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

    def test_round_one_bootstrap_caps_without_a_ledger(self) -> None:
        # SELECT runs before the first add-record creates ledger.json; the
        # reserve cap must ride the same missing-ledger bootstrap instead of
        # reading the file (regression: FileNotFoundError on round 1).
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "toy" / "tag"
            run_dir.mkdir(parents=True)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 100,
                        "tuner": {
                            "scheduler_policy": "anchor_challenger_v1",
                            "inner_policy": "localtr8-hebo10-hebo10-v1",
                            "K_eval": 2,
                        },
                    }
                )
            )
            output = io.StringIO()
            with redirect_stdout(output):
                got_select.cmd_decide(
                    SimpleNamespace(ledger=str(run_dir / "ledger.json"), cfg=None)
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(payload["diag"]["tournament_generation_reserve"], 36)
            self.assertEqual(payload["diag"]["candidate_admission_cap"], 32)

    def test_got_select_cannot_consume_the_hard_reserve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "toy" / "tag"
            run_dir.mkdir(parents=True)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 100,
                        "tuner": {
                            "scheduler_policy": "anchor_challenger_v1",
                            "inner_policy": "localtr8-hebo10-hebo10-v1",
                            "K_eval": 2,
                        },
                    }
                )
            )
            attempt = json.dumps({"schema_version": 1, "kind": "score_attempt"})
            # One initialized anchor under the current 8/10/10 inner policy
            # leaves a hard reserve of 8+10+10=28.  With only 29 remaining,
            # no two-evaluation candidate may be admitted.
            (run_dir / "evaluation_attempts.jsonl").write_text(
                "\n".join([attempt] * 71) + "\n"
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
            output = io.StringIO()
            with redirect_stdout(output):
                got_select.cmd_decide(
                    SimpleNamespace(ledger=str(ledger), cfg=None)
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(payload["actions"], [])
            self.assertEqual(payload["diag"]["objective_remaining"], 29)
            self.assertEqual(payload["diag"]["tournament_generation_reserve"], 28)
            self.assertEqual(payload["diag"]["candidate_admission_cap"], 0)


if __name__ == "__main__":
    unittest.main()
