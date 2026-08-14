"""Deterministic checks over the scheduler v3.2 policy arm.

The whole-suite pass rate says nothing about this arm — it is behind an
isolated `tuner.scheduler_policy` switch that no other test turns on. These
are the checks the design calls for that can run with no objective, no GPU,
and no LLM: admission and budget accounting, exact-target closure, CRN
pairing identity, evidence derivation from artifacts, decision idempotence,
and snapshot replay.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.scheduler.contract import (  # noqa: E402
    CandidateView,
    ResourceContract,
    ineligibility_reason,
)
from tools.scheduler.evidence import (  # noqa: E402
    FIRST,
    INFRA_FAILURE,
    LATER,
    PRIOR_ID,
    SCIENTIFIC_INVALID,
    VALID,
    ArrivalModel,
    ArrivalRecord,
    TuningModel,
    TuningRecord,
    admitted_prefix,
    derive_tuning_records,
)
from tools.scheduler.policy import (  # noqa: E402
    CoverageConfig,
    PolicyConfig,
    decide,
)
from tools.scheduler.reconcile import reconcile  # noqa: E402
from tools.scheduler.rollout import (  # noqa: E402
    ReferencePolicy,
    RolloutConfig,
    crn_draw,
    evaluate_actions,
)
from tools.scheduler.session import decide_for_run  # noqa: E402
from tools.scheduler.state import (  # noqa: E402
    SchedulerState,
    state_from_snapshot,
)
from tools.scheduler.store import SchedulerStore  # noqa: E402


def candidate(run_id: str, score: float, bouts: int = 0, **kwargs) -> CandidateView:
    return CandidateView(
        run_id=run_id, best_score=score, bouts_used=bouts, **kwargs
    )


def state(
    *candidates: CandidateView,
    budget: int = 100,
    contract: ResourceContract | None = None,
    n_roots: int | None = None,
    n_seed: int = 0,
) -> SchedulerState:
    contract = contract or ResourceContract()
    scores = [c.best_score for c in candidates if not c.crashed]
    return SchedulerState(
        global_best=min(scores) if scores else float("inf"),
        remaining_budget=budget,
        candidates=tuple(candidates),
        contract=contract,
        n_roots=n_roots,
        n_seed=n_seed,
    )


def tuning_model(*records: TuningRecord, use_prior: bool = False) -> TuningModel:
    return TuningModel.from_records(records, use_prior=use_prior)


def bout(klass: str, gain: float, *, status: str = VALID, cost: int = 10):
    return TuningRecord(bout_class=klass, status=status, gain=gain, cost=cost)


class AdmissionTest(unittest.TestCase):
    """A bout is admitted at full B or not at all (§2)."""

    def test_partial_budget_admits_no_bout(self):
        contract = ResourceContract()
        # Regime costs (design §2): a FIRST bout admits at full B_FIRST=8,
        # a CONTINUE/DEEP bout at full B=10 — never truncated.
        first = candidate("001", 1.0)
        self.assertIsNone(ineligibility_reason(first, 8, contract))
        self.assertIn("cannot admit a full", ineligibility_reason(first, 7, contract))
        later = candidate("002", 1.0, bouts=1)
        self.assertIsNone(ineligibility_reason(later, 10, contract))
        self.assertIn("cannot admit a full", ineligibility_reason(later, 9, contract))

    def test_deep_bout_requires_a_movable_continuous_dimension(self):
        contract = ResourceContract()
        deep = candidate("001", 1.0, bouts=2, has_movable_continuous=False)
        self.assertEqual(
            ineligibility_reason(deep, 1000, contract),
            "no movable continuous dimension for a DEEP bout",
        )
        # The same candidate is admitted while its next bout is CONTINUE.
        pre_deep = candidate("001", 1.0, bouts=1, has_movable_continuous=False)
        self.assertIsNone(ineligibility_reason(pre_deep, 1000, contract))

    def test_lifetime_cost_is_policy_aware(self):
        self.assertEqual(ResourceContract().lifetime_cost(), 38)
        self.assertEqual(
            ResourceContract(first_bout_trials=10).lifetime_cost(), 40
        )
        self.assertEqual(
            ResourceContract(bout_trials=8, first_bout_trials=8).lifetime_cost(),
            32,
        )

    def test_bout_cap_is_permanent(self):
        contract = ResourceContract()
        capped = candidate("001", 1.0, bouts=contract.max_bouts)
        self.assertEqual(
            ineligibility_reason(capped, 1000, contract),
            f"bout cap reached ({contract.max_bouts})",
        )

    def test_reasons_are_ordered_permanent_first(self):
        contract = ResourceContract()
        both = candidate("001", 1.0, bouts=contract.max_bouts, crashed=True)
        self.assertEqual(ineligibility_reason(both, 0, contract), "crashed")

    def test_defer_stops_being_an_action_below_k_eval(self):
        contract = ResourceContract()
        self.assertTrue(state(budget=contract.k_eval).defer_available())
        self.assertFalse(state(budget=contract.k_eval - 1).defer_available())

    def test_terminal_when_neither_action_is_admissible(self):
        empty = state(candidate("001", 1.0), budget=1)
        self.assertTrue(empty.terminal())
        self.assertEqual(empty.actions(), [])


class TransitionTest(unittest.TestCase):
    """Equations (2)/(4): raw score transition and budget accounting."""

    def test_positive_gain_moves_candidate_and_global_best(self):
        after = state(candidate("001", 5.0), candidate("002", 7.0)).apply_bout(
            "002", 3.0
        )
        self.assertEqual(after.candidates[1].best_score, 4.0)
        self.assertEqual(after.global_best, 4.0)
        # "002" was untuned: the bout charges the FIRST-regime cost B_FIRST=8.
        self.assertEqual(after.remaining_budget, 92)

    def test_negative_gain_is_floored_but_still_charges(self):
        after = state(candidate("001", 5.0)).apply_bout("001", -2.0)
        self.assertEqual(after.candidates[0].best_score, 5.0)
        self.assertEqual(after.candidates[0].previous_gain, -2.0)
        self.assertEqual(after.remaining_budget, 92)

    def test_continuation_bout_charges_the_full_b(self):
        after = state(candidate("001", 5.0, bouts=1)).apply_bout("001", 1.0)
        self.assertEqual(after.remaining_budget, 90)

    def test_gain_below_headroom_does_not_move_the_global_best(self):
        after = state(candidate("001", 5.0), candidate("002", 9.0)).apply_bout(
            "002", 2.0
        )
        self.assertEqual(after.candidates[1].best_score, 7.0)
        self.assertEqual(after.global_best, 5.0)

    def test_failed_arrival_charges_cost_and_adds_no_candidate(self):
        after = state(candidate("001", 5.0)).admit_arrivals(
            [("new-1", None, 2), ("new-2", 3.0, 2)]
        )
        self.assertEqual([c.run_id for c in after.candidates], ["001", "new-2"])
        self.assertEqual(after.remaining_budget, 96)
        self.assertEqual(after.global_best, 3.0)


class EvidenceDerivationTest(unittest.TestCase):
    """`(Z, D, cost)` comes from the candidate's own tuning report."""

    def _report(self, stages: list[dict], warm: float = 10.0) -> dict:
        return {
            "phase_a": {"best_warm_score": warm},
            "phase_c": {"stages": stages},
        }

    def _records(self, report: dict) -> list[TuningRecord]:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = run_dir / "candidates" / "001" / "tune_report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(report))
            # What `finalize_tuning` writes once every bout in the report has
            # closed: only closed bouts are derived as outcomes.
            closed = len(
                {
                    stage.get("bout_index", 0)
                    for stage in report["phase_c"]["stages"]
                }
            )
            return derive_tuning_records(
                run_dir,
                {"records": [{"run_id": "001", "tuning_bouts": closed}]},
            )

    def test_first_and_later_classes_and_signed_gain(self):
        records = self._records(
            self._report(
                [
                    {
                        "bout_index": 0,
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"score": 8.0}, {"score": 12.0}],
                    },
                    {
                        "bout_index": 1,
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"score": 9.0}],
                    },
                ]
            )
        )
        self.assertEqual([r.bout_class for r in records], [FIRST, LATER])
        self.assertEqual(records[0].gain, 2.0)
        # The second bout's best (9.0) is worse than the running incumbent
        # (8.0), so D is negative — a real observation, not a missing one.
        self.assertEqual(records[1].gain, -1.0)
        self.assertEqual([r.cost for r in records], [2, 1])

    def test_an_open_bout_is_not_an_outcome(self):
        """An interrupted bout must not freeze a fragment in the log.

        The evidence log is append-only and deduped by `(run_id, bout_index)`,
        so a record derived from a `running` stage's partial trials would be
        the *only* record that bout ever contributes — the recovered, complete
        bout would be skipped as already seen.
        """
        report = self._report(
            [
                {
                    "bout_index": 0,
                    "method": "bo",
                    "status": "running",
                    "trials": [{"score": 9.8}, {"score": 9.9}],
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = run_dir / "candidates" / "001" / "tune_report.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(report))
            ledger = {"records": [{"run_id": "001", "tuning_bouts": 0}]}
            self.assertEqual(derive_tuning_records(run_dir, ledger), [])

            # Recovered: the stage resumed under the same bout index, found a
            # far better score, and finalized.
            report["phase_c"]["stages"][0].update(
                status="ok", trials=[{"score": 9.8}, {"score": 9.9}, {"score": 4.0}]
            )
            path.write_text(json.dumps(report))
            ledger["records"][0]["tuning_bouts"] = 1
            records = derive_tuning_records(run_dir, ledger)
            self.assertEqual([(r.gain, r.cost) for r in records], [(6.0, 3)])

    def test_all_rejected_stages_are_scientific_invalid(self):
        records = self._records(
            self._report(
                [
                    {
                        "bout_index": 0,
                        "method": "grid",
                        "status": "rejected",
                        "trials": [],
                    }
                ]
            )
        )
        self.assertEqual(records[0].status, SCIENTIFIC_INVALID)
        self.assertEqual(records[0].cost, 0)

    def test_preflight_rejected_trials_are_scientific_invalid(self):
        records = self._records(
            self._report(
                [
                    {
                        "bout_index": 0,
                        "method": "bo",
                        "status": "ok",
                        "trials": [{"status": "preflight_rejected"}],
                    }
                ]
            )
        )
        self.assertEqual(records[0].status, SCIENTIFIC_INVALID)

    def test_scoreless_evaluated_bout_is_infra_failure(self):
        records = self._records(
            self._report(
                [
                    {
                        "bout_index": 0,
                        "method": "bo",
                        "status": "failed",
                        "trials": [{"score": None}],
                    }
                ]
            )
        )
        self.assertEqual(records[0].status, INFRA_FAILURE)

    def test_cost_counts_objective_attempts_not_trial_rows(self):
        """A preflight-rejected leg leaves a trial row but reserved no
        objective slot (`timed_preflight` never calls `reserve_evaluation`),
        while every scored or failed row passed `reserve_evaluation` on the
        way in. Cost must be the latter, matching evaluation_attempts.jsonl —
        the degenerate DEEP shape (plus legs scored, minus legs rejected, no
        gradient update) is a valid outcome whose true cost is 10, not 20.
        """
        records = self._records(
            self._report(
                [
                    {
                        "bout_index": 0,
                        "method": "spsa",
                        "status": "ok",
                        "pairs_attempted": 10,
                        "updates_applied": 0,
                        "preflight_rejections": 9,
                        "trials": [
                            *[{"score": 9.0 + i * 0.1} for i in range(10)],
                            {"score": None, "status": "failed"},
                            *[{"status": "preflight_rejected"} for _ in range(9)],
                        ],
                    }
                ]
            )
        )
        self.assertEqual(records[0].status, VALID)
        self.assertEqual(records[0].cost, 11)
        # Termination shape rides along as diagnostics — recorded, never
        # conditioned on (§4.3) — so a degenerate bout stays distinguishable
        # from a complete one without changing the model's inputs.
        self.assertEqual(records[0].diagnostics["preflight_rejected_trials"], 9)
        self.assertEqual(records[0].diagnostics["pairs_attempted"], 10)
        self.assertEqual(records[0].diagnostics["updates_applied"], 0)


class ArrivalEvidenceTest(unittest.TestCase):
    """Gaps are anchored to the pre-episode raw global best (eq. 7)."""

    def _run(self, *rounds: list[dict]) -> ArrivalRecord | None:
        """Reconcile one generation round at a time and return the last episode.

        Episodes are grouped by what reconciliation had not yet consumed, so
        a run's rounds have to arrive in sequence: replaying the whole ledger
        at once is one episode, not several, and no candidate in it has a
        prior global best to be anchored against.
        """
        from tools.scheduler.evidence import EvidenceLog

        with tempfile.TemporaryDirectory() as tmp:
            records: list[dict] = []
            for batch in rounds:
                records.extend(batch)
                reconcile(Path(tmp), {"records": list(records)})
            log = EvidenceLog(Path(tmp) / ".scheduler" / "evidence.jsonl")
            episodes = log.arrival_records()
            return episodes[-1] if episodes else None

    def test_anchor_includes_tuning_improvement(self):
        episode = self._run(
            [
                {
                    "run_id": "000",
                    "status": "keep",
                    "best_warm_score": 10.0,
                    "final_best_score": 5.0,
                    "tune": True,
                    "tuning_bouts": 1,
                }
            ],
            [{"run_id": "001", "status": "keep", "best_warm_score": 8.0}],
        )
        # Anchored on the tuned global best 5.0, a warm score of 8.0 is a
        # +3.0 gap (worse), not the -2.0 a warm-only anchor would report.
        self.assertEqual(episode.warm_gaps, (3.0,))

    def test_failed_arrival_is_none_not_zero(self):
        episode = self._run(
            [{"run_id": "000", "status": "keep", "best_warm_score": 4.0}],
            [{"run_id": "001", "status": "keep", "best_warm_score": None}],
        )
        self.assertEqual(episode.warm_gaps, (None,))
        self.assertEqual(episode.usable_gaps, ())

    def test_admitted_prefix_preserves_failures_and_truncates(self):
        episode = ArrivalRecord(
            warm_gaps=(1.0, None, -2.0),
            planned_count=3,
            per_candidate_cost=2,
        )
        self.assertEqual(
            admitted_prefix(episode, 100), [(1.0, 2), (None, 2), (-2.0, 2)]
        )
        self.assertEqual(admitted_prefix(episode, 5), [(1.0, 2), (None, 2)])

    def test_unanchored_first_episode_is_not_evidence_but_advances(self):
        episode = self._run(
            [{"run_id": "000", "status": "keep", "best_warm_score": 4.0}]
        )
        self.assertEqual(episode.run_ids, ("000",))
        self.assertEqual(episode.warm_gaps, ())
        self.assertFalse(ArrivalModel.from_records([episode]).episodes)


class FrozenPriorTest(unittest.TestCase):
    """The design prior is versioned, class-separated, and FIRST-costed."""

    def test_prior_id_and_class_costs(self):
        from tools.scheduler.evidence import frozen_tuning_records

        records = frozen_tuning_records()
        first = [r for r in records if r.bout_class == FIRST]
        later = [r for r in records if r.bout_class == LATER]
        self.assertEqual(len(first), 4)
        self.assertEqual(len(later), 4)
        self.assertTrue(all(r.cost == 8 for r in first))
        self.assertTrue(all(r.cost == 10 for r in later))
        self.assertTrue(all(r.diagnostics.get("prior_id") == PRIOR_ID for r in records))
        self.assertGreater(max(r.gain for r in first), max(r.gain for r in later))


class ModelSupportTest(unittest.TestCase):
    """Exact class wins; otherwise the frozen prior, never the other class."""

    def test_prior_is_not_exact_support(self):
        model = TuningModel.from_records(
            [bout(FIRST, 1.0) for _ in range(3)], use_prior=True
        )
        self.assertTrue(model.supported(LATER))
        self.assertFalse(model.exact_supported(LATER))
        self.assertEqual(model.usage_mode(LATER), "prior")
        self.assertEqual(model.usage_mode(FIRST), "exact")

    def test_empty_model_without_prior_is_unsupported(self):
        model = tuning_model()
        self.assertFalse(model.supported(FIRST))
        self.assertEqual(model.usage_mode(FIRST), "unsupported")

    def test_empty_model_with_prior_is_supported(self):
        model = TuningModel.from_records([], use_prior=True)
        self.assertTrue(model.supported(FIRST))
        self.assertTrue(model.supported(LATER))
        self.assertEqual(model.usage_mode(FIRST), "prior")
        self.assertEqual(model.usage_mode(LATER), "prior")
        self.assertGreaterEqual(model.exact_count(FIRST), 0)
        self.assertFalse(model.exact_supported(FIRST))

    def test_first_is_never_pooled_into_later(self):
        model = TuningModel.from_records(
            [bout(FIRST, 1.0) for _ in range(3)], use_prior=False
        )
        self.assertTrue(model.supported(FIRST))
        self.assertFalse(model.supported(LATER))
        self.assertEqual(model.usage_mode(LATER), "unsupported")


class SeedSetGateTest(unittest.TestCase):
    """Defer while reserved fresh roots are still arriving."""

    def test_incomplete_seed_set_defers_without_rollout(self):
        decision = decide(
            state(
                candidate("000", 1.0),
                candidate("001", 1.03),
                n_roots=2,
                n_seed=5,
            ),
            TuningModel.from_records([], use_prior=True),
            ArrivalModel.from_records([], use_prior=True),
        )
        self.assertEqual(decision.action, "DEFER")
        self.assertIsNone(decision.run_id)
        self.assertTrue(decision.reason.startswith("seed set incomplete"))
        self.assertFalse(decision.values)
        self.assertTrue(decision.evidence_mode["seed_set_incomplete"])
        self.assertEqual(decision.evidence_mode["n_roots"], 2)
        self.assertEqual(decision.evidence_mode["n_seed"], 5)

    def test_complete_seed_set_uses_rollout(self):
        decision = decide(
            state(
                candidate("000", 1.0),
                candidate("001", 1.03),
                n_roots=5,
                n_seed=5,
            ),
            TuningModel.from_records([], use_prior=True),
            ArrivalModel.from_records([], use_prior=True),
            config=PolicyConfig(rollout=RolloutConfig(scenarios=4)),
        )
        self.assertFalse(decision.reason.startswith("seed set incomplete"))
        self.assertTrue(decision.values)
        self.assertFalse(decision.evidence_mode["seed_set_incomplete"])

    def test_gate_does_not_fire_when_defer_cannot_buy_a_round(self):
        # FIRST still fits; a generation slot does not.
        contract = ResourceContract(k_eval=20, first_bout_trials=8)
        decision = decide(
            state(
                candidate("000", 1.0),
                budget=15,
                contract=contract,
                n_roots=2,
                n_seed=5,
            ),
            TuningModel.from_records([], use_prior=True),
            ArrivalModel.from_records([], use_prior=True),
            config=PolicyConfig(rollout=RolloutConfig(scenarios=4)),
        )
        self.assertFalse(decision.reason.startswith("seed set incomplete"))
        self.assertEqual(decision.action, "TUNE")
        self.assertEqual(decision.run_id, "000")


class CoverageTest(unittest.TestCase):
    """Sample-seeking coverage is off; cold start uses the frozen prior."""

    def _decide(self, tuning, arrival, spent=0, **kwargs):
        return decide(
            state(candidate("001", 5.0), candidate("002", 6.0, bouts=1)),
            tuning,
            arrival,
            config=PolicyConfig(
                rollout=RolloutConfig(scenarios=8),
                coverage=CoverageConfig(**kwargs),
            ),
            coverage_spent=spent,
        )

    def test_thin_later_does_not_force_a_later_bout(self):
        decision = self._decide(
            TuningModel.from_records(
                [bout(FIRST, 1.0) for _ in range(3)], use_prior=True
            ),
            ArrivalModel.from_records([], use_prior=True),
        )
        self.assertFalse(decision.reason.startswith("coverage:"))
        self.assertEqual(decision.evidence_mode[LATER], "prior")
        self.assertEqual(decision.evidence_mode[FIRST], "exact")

    def test_empty_current_run_does_not_force_first(self):
        decision = self._decide(
            TuningModel.from_records([], use_prior=True),
            ArrivalModel.from_records([], use_prior=True),
        )
        self.assertFalse(decision.reason.startswith("coverage:"))
        self.assertFalse(decision.reason.startswith("fallback"))
        self.assertTrue(decision.values)
        self.assertIn(decision.action, {"TUNE", "DEFER"})
        self.assertEqual(decision.evidence_mode[FIRST], "prior")
        self.assertEqual(decision.evidence_mode[LATER], "prior")
        self.assertEqual(decision.evidence_mode["arrival"], "prior")
        self.assertEqual(decision.evidence_mode["prior_id"], PRIOR_ID)

    def test_coverage_gate_is_gone_even_with_old_thresholds(self):
        decision = self._decide(
            TuningModel.from_records([], use_prior=True),
            ArrivalModel.from_records([], use_prior=True),
            budget_cap=6,
            min_first=3,
            min_later=3,
        )
        self.assertFalse(decision.reason.startswith("coverage:"))

    def test_evidence_mode_is_recorded(self):
        decision = self._decide(
            TuningModel.from_records(
                [bout(FIRST, 1.0) for _ in range(3)], use_prior=True
            ),
            ArrivalModel.from_records([], use_prior=True),
        )
        self.assertEqual(decision.evidence_mode[LATER], "prior")
        self.assertEqual(decision.evidence_mode[FIRST], "exact")


class CRNTest(unittest.TestCase):
    """Pairing keys on future-event identity, not draw order."""

    def test_same_event_same_draw_across_branches(self):
        self.assertEqual(crn_draw(3, "arrival", 0), crn_draw(3, "arrival", 0))
        self.assertNotEqual(crn_draw(3, "arrival", 0), crn_draw(4, "arrival", 0))
        self.assertNotEqual(
            crn_draw(3, "tune", "001", 0), crn_draw(3, "tune", "002", 0)
        )

    def test_draws_are_uniform_in_range(self):
        draws = [crn_draw(i, "arrival", 0) for i in range(64)]
        self.assertTrue(all(0.0 <= d < 1.0 for d in draws))
        self.assertEqual(len(set(draws)), 64)


class RolloutTest(unittest.TestCase):
    """Full-remaining-budget simulation and its cost/Z semantics."""

    def _models(self, *records):
        arrival = ArrivalModel.from_records(
            [
                ArrivalRecord(
                    warm_gaps=(0.5,), planned_count=1, per_candidate_cost=2
                ),
                ArrivalRecord(
                    warm_gaps=(2.0,), planned_count=1, per_candidate_cost=2
                ),
            ]
        )
        return tuning_model(*records), arrival

    def test_invalid_bout_applies_no_gain(self):
        tuning, arrival = self._models(
            *(bout(FIRST, 5.0, status=INFRA_FAILURE) for _ in range(3))
        )
        values = evaluate_actions(
            state(candidate("001", 1.0), budget=20),
            tuning,
            arrival,
            RolloutConfig(scenarios=4),
        )
        tune = next(v for v in values if v.action == "TUNE")
        # Every sampled bout is an infra failure, so no simulated trajectory
        # can improve the global best through tuning.
        self.assertEqual(tune.q_hat, 0.0)

    def test_valid_bout_produces_improvement(self):
        tuning, arrival = self._models(*(bout(FIRST, 5.0) for _ in range(3)))
        values = evaluate_actions(
            state(candidate("001", 1.0), budget=20),
            tuning,
            arrival,
            RolloutConfig(scenarios=4),
        )
        tune = next(v for v in values if v.action == "TUNE")
        self.assertGreater(tune.q_hat, 0.0)

    def test_zero_cost_bout_charges_nothing_but_burns_a_bout(self):
        # A bout whose whole method chain was rejected reserves no objective
        # slot. Charging it a full B would drain the simulated budget faster
        # than the real one; charging it nothing is only safe because the
        # bout cap still terminates the candidate.
        current = state(candidate("001", 1.0), budget=20)
        for _ in range(current.contract.max_bouts):
            current = current.apply_bout("001", 0.0, cost=0)
        self.assertEqual(current.remaining_budget, 20)
        self.assertEqual(current.eligible(), [])

    def test_invalid_bout_cost_is_the_observed_cost(self):
        from tools.scheduler.rollout import _bout_cost

        self.assertEqual(_bout_cost(bout(FIRST, 0.0, cost=0), 10), 0)
        self.assertEqual(_bout_cost(bout(FIRST, 0.0, cost=4), 10), 4)
        # A record can never charge more than the contract's B.
        self.assertEqual(_bout_cost(bout(FIRST, 0.0, cost=99), 10), 10)

    def test_reference_policy_prefers_smallest_headroom(self):
        chosen = ReferencePolicy().choose(
            state(candidate("001", 5.0), candidate("002", 9.0))
        )
        self.assertEqual(chosen, ("TUNE", "001"))

    def test_tuning_round_still_receives_its_generation(self):
        """The driver generates every round; a bout cannot postpone that.

        `experiment.py` runs step 2 (generate + warm evaluate) before step 3
        (at most one bout) and does not gate step 2 on the scheduler. A
        rollout that admitted arrivals only on DEFER would let TUNE chain
        bouts against a candidate pool the real run would have grown.
        """
        from tools.scheduler.rollout import _simulate

        tuning, _ = self._models(*(bout(FIRST, 0.0) for _ in range(3)))
        # Arrivals are the only source of improvement here: every bout is a
        # flat zero. Scores are lower-is-better, so a negative gap is an
        # arrival that beats the anchor. Whatever the TUNE branch reaches,
        # it reached through a generation episode that followed the bout.
        arrival = ArrivalModel.from_records(
            [
                ArrivalRecord(
                    warm_gaps=(-3.0,), planned_count=1, per_candidate_cost=2
                ),
                ArrivalRecord(
                    warm_gaps=(-3.0,), planned_count=1, per_candidate_cost=2
                ),
            ]
        )
        improvement = _simulate(
            state(candidate("001", 5.0), budget=40),
            ("TUNE", "001"),
            0,
            tuning,
            arrival,
            RolloutConfig(scenarios=1),
        )
        self.assertGreater(improvement, 0.0)


class TieRuleTest(unittest.TestCase):
    """Exact ties resolve the way pi_ref would, not lexicographically."""

    def test_tie_matches_reference_ordering(self):
        # Long horizon: every candidate can reach its bout cap, so the
        # terminal best is order-invariant and all TUNE values tie exactly.
        decision = decide(
            state(candidate("009", 5.0), candidate("001", 9.0), budget=200),
            tuning_model(*(bout(FIRST, 1.0) for _ in range(3))),
            ArrivalModel.from_records(
                [
                    ArrivalRecord(
                        warm_gaps=(1.0,), planned_count=1, per_candidate_cost=2
                    )
                ]
                * 2
            ),
            config=PolicyConfig(
                rollout=RolloutConfig(scenarios=8),
                coverage=CoverageConfig(budget_cap=0),
            ),
        )
        self.assertEqual(decision.action, "TUNE")
        self.assertEqual(decision.run_id, "009")


def run_dir_fixture(
    tmp: str,
    *,
    budget: int = 100,
    tuner: dict | None = None,
    got: dict | None = None,
) -> Path:
    """A synthetic v3.2 run directory: two warm candidates, no objective."""
    run_dir = Path(tmp) / "runs" / "toy" / "tag"
    run_dir.mkdir(parents=True)
    config = {
        "max_evaluations": budget,
        "tuner": {"scheduler_policy": "v3_2", **(tuner or {})},
    }
    if got:
        config["got"] = got
    (run_dir / "framework_cfg.json").write_text(json.dumps(config))
    (run_dir / "evaluation_attempts.jsonl").write_text("")
    (run_dir / "ledger.json").write_text(
        json.dumps(
            {
                "records": [
                    {"run_id": "000", "status": "keep", "best_warm_score": 5.0},
                    {"run_id": "001", "status": "keep", "best_warm_score": 7.0},
                ]
            }
        )
    )
    return run_dir


def execute_bout(ledger: Path, run_id: str, *, best: float = 3.0, trials: int = 10):
    """Write the artifacts a completed bout on `run_id` would have left."""
    run_dir = ledger.parent
    report = run_dir / "candidates" / run_id / "tune_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "phase_a": {"best_warm_score": 5.0},
                "phase_c": {
                    "stages": [
                        {
                            "bout_index": 0,
                            "method": "bo",
                            "status": "ok",
                            "trials": [{"score": best}] * trials,
                        }
                    ]
                },
            }
        )
    )
    data = json.loads(ledger.read_text())
    for record in data["records"]:
        if record["run_id"] == run_id:
            record.update(
                {"tune": True, "tuning_bouts": 1, "final_best_score": best}
            )
    ledger.write_text(json.dumps(data))


class RunIntegrationTest(unittest.TestCase):
    """End-to-end over a synthetic run directory: no objective, no LLM."""

    def _run_dir(self, tmp: str, **kwargs) -> Path:
        return run_dir_fixture(tmp, **kwargs)

    def test_decision_is_idempotent_for_the_same_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            ledger = run_dir / "ledger.json"
            first = decide_for_run(ledger, scenarios=4)
            second = decide_for_run(ledger, scenarios=4)
            self.assertFalse(first["reused_open_decision"])
            self.assertTrue(second["reused_open_decision"])
            self.assertEqual(first["decision_id"], second["decision_id"])
            store = SchedulerStore(run_dir)
            self.assertEqual(
                len([r for r in store.decisions() if r["kind"] == "scheduler_decision"]),
                1,
            )

    def test_repeated_queries_do_not_inflate_the_decision_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            ledger = run_dir / "ledger.json"
            for _ in range(4):
                decide_for_run(ledger, scenarios=4)
            store = SchedulerStore(run_dir)
            self.assertEqual(store.coverage_spent(), 0)
            self.assertEqual(
                len([r for r in store.decisions() if r["kind"] == "scheduler_decision"]),
                1,
            )

    def test_cold_start_is_not_sample_seeking_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            view = decide_for_run(run_dir / "ledger.json", scenarios=4)
            self.assertFalse(str(view["reason"]).startswith("coverage:"))
            self.assertEqual(view["evidence_mode"][FIRST], "prior")
            self.assertEqual(view["evidence_mode"][LATER], "prior")
            self.assertEqual(view["evidence_mode"]["arrival"], "prior")
            self.assertEqual(view["prior_id"], PRIOR_ID)
            # Two warm roots, default n_seed=5: the seed-set gate defers.
            self.assertEqual(view["action"], "DEFER")
            self.assertTrue(str(view["reason"]).startswith("seed set incomplete"))
            self.assertEqual(view["evidence_mode"]["n_roots"], 2)
            self.assertEqual(view["evidence_mode"]["n_seed"], 5)

    def test_complete_seed_set_on_a_live_run_uses_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp, got={"n_seed": 2})
            view = decide_for_run(run_dir / "ledger.json", scenarios=4)
            self.assertFalse(str(view["reason"]).startswith("seed set incomplete"))
            self.assertIn(view["action"], {"TUNE", "DEFER"})
            store = SchedulerStore(run_dir)
            receipt = next(
                row
                for row in store.decisions()
                if row["kind"] == "scheduler_decision"
            )
            self.assertIn("q_hat", receipt)
            self.assertFalse(receipt["evidence_mode"]["seed_set_incomplete"])

    def test_executed_action_binds_the_decision_and_reopens_deciding(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            ledger = run_dir / "ledger.json"
            first = decide_for_run(ledger, scenarios=4)
            if first["action"] == "TUNE":
                execute_bout(ledger, first["run_id"])
            else:
                data = json.loads(ledger.read_text())
                data["records"].append(
                    {"run_id": "002", "status": "keep", "best_warm_score": 6.0}
                )
                ledger.write_text(json.dumps(data))

            second = decide_for_run(ledger, scenarios=4)
            self.assertNotEqual(first["decision_id"], second["decision_id"])
            outcomes = [
                row
                for row in SchedulerStore(run_dir).decisions()
                if row["kind"] == "scheduler_outcome"
            ]
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(outcomes[0]["decision_id"], first["decision_id"])
            self.assertEqual(outcomes[0]["executed_action"], first["action"])
            if first["action"] == "TUNE":
                self.assertEqual(outcomes[0]["executed_run_id"], first["run_id"])
                self.assertEqual(outcomes[0]["status"], VALID)
                self.assertEqual(outcomes[0]["consumed_evaluations"], 10)

    def test_snapshot_round_trips_through_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            view = decide_for_run(run_dir / "ledger.json", scenarios=4)
            store = SchedulerStore(run_dir)
            snapshot = store.get_snapshot(view["state_snapshot_id"])
            rebuilt = state_from_snapshot(snapshot)
            self.assertEqual(rebuilt.snapshot(), snapshot)

    def test_unbounded_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"tuner": {"scheduler_policy": "legacy"}})
            )
            with self.assertRaises(ValueError):
                decide_for_run(run_dir / "ledger.json", scenarios=4)

    def test_simulated_budget_matches_the_run_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp, budget=57)
            view = decide_for_run(run_dir / "ledger.json", scenarios=4)
            self.assertEqual(view["remaining_budget"], 57)
            self.assertEqual(view["state"].remaining_budget, 57)


class ConfigCompatibilityTest(unittest.TestCase):
    """v3.2 rejects configs whose admission layer contradicts its contract."""

    def _cfg(self, tmp: str, config: dict) -> Path:
        path = Path(tmp) / "framework_cfg.json"
        path.write_text(json.dumps(config))
        return path

    def _read(self, config: dict):
        from run_cfg import RunConfigError, read_framework_cfg

        with tempfile.TemporaryDirectory() as tmp:
            return read_framework_cfg(self._cfg(tmp, config)), RunConfigError

    def _expect_error(self, config: dict, needle: str):
        from run_cfg import RunConfigError, read_framework_cfg

        with tempfile.TemporaryDirectory() as tmp:
            path = self._cfg(tmp, config)
            with self.assertRaises(RunConfigError) as ctx:
                read_framework_cfg(path)
            self.assertIn(needle, str(ctx.exception))

    def test_default_v3_2_config_is_accepted(self):
        config, _ = self._read(
            {"max_evaluations": 120, "tuner": {"scheduler_policy": "v3_2"}}
        )
        self.assertEqual(config["max_evaluations"], 120)

    def test_unbounded_budget_rejected(self):
        self._expect_error(
            {"tuner": {"scheduler_policy": "v3_2"}}, "requires max_evaluations"
        )

    def test_deep_tune_fraction_rejected(self):
        self._expect_error(
            {
                "max_evaluations": 120,
                "tuner": {
                    "scheduler_policy": "v3_2",
                    "deep_tune_budget_fraction": 0.4,
                },
            },
            "deep_tune_budget_fraction",
        )

    def test_per_candidate_cap_of_38_is_the_full_contract(self):
        config, _ = self._read(
            {
                "max_evaluations": 120,
                "tuner": {
                    "scheduler_policy": "v3_2",
                    "deep_tune_per_candidate_cap": 38,
                },
            }
        )
        self.assertEqual(config["tuner"]["deep_tune_per_candidate_cap"], 38)

    def test_per_candidate_cap_of_37_is_rejected(self):
        self._expect_error(
            {
                "max_evaluations": 120,
                "tuner": {
                    "scheduler_policy": "v3_2",
                    "deep_tune_per_candidate_cap": 37,
                },
            },
            "8 + 10 x 3 = 38",
        )

    def test_per_candidate_cap_below_bout_contract_rejected(self):
        self._expect_error(
            {
                "max_evaluations": 120,
                "tuner": {
                    "scheduler_policy": "v3_2",
                    "deep_tune_per_candidate_cap": 30,
                },
            },
            "below the v3.2 bout contract",
        )

    def test_legacy_runs_keep_their_overrides(self):
        config, _ = self._read(
            {
                "tuner": {
                    "scheduler_policy": "legacy",
                    "deep_tune_budget_fraction": 0.4,
                    "deep_tune_per_candidate_cap": 30,
                }
            }
        )
        self.assertEqual(config["tuner"]["deep_tune_per_candidate_cap"], 30)


class CalibrationTest(unittest.TestCase):
    """The read-only calibration view over a synthetic run's artifacts."""

    def _executed_run(self, tmp: str):
        """A run with one decision made, executed, and reconciled."""
        # Two roots already present: set n_seed so the seed-set gate is
        # not the thing being calibrated.
        run_dir = run_dir_fixture(tmp, got={"n_seed": 2})
        ledger = run_dir / "ledger.json"
        first = decide_for_run(ledger, scenarios=4)
        self.assertEqual(first["action"], "TUNE")
        execute_bout(ledger, first["run_id"])
        decide_for_run(ledger, scenarios=4)
        return run_dir, ledger

    def test_calibration_reports_closure_and_replay(self):
        from tools.scheduler.calibrate import calibrate

        with tempfile.TemporaryDirectory() as tmp:
            _, ledger = self._executed_run(tmp)
            report = calibrate(ledger, scenarios=4)
            closure = report["execution_closure"]
            self.assertEqual(closure["target_mismatches"], [])
            self.assertEqual(closure["bound"], 1)
            # The second decision is still open — it has not executed yet.
            self.assertEqual(len(closure["unbound_decision_ids"]), 1)
            self.assertTrue(report["replay"]["reproducible"])

    def test_calibration_is_read_only(self):
        from tools.scheduler.calibrate import calibrate

        with tempfile.TemporaryDirectory() as tmp:
            run_dir, ledger = self._executed_run(tmp)
            before = {
                path: path.read_bytes()
                for path in sorted(run_dir.rglob("*"))
                if path.is_file()
            }
            calibrate(ledger, scenarios=4)
            after = {
                path: path.read_bytes()
                for path in sorted(run_dir.rglob("*"))
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_budget_accounting_matches_charged_cost(self):
        from tools.scheduler.calibrate import calibrate

        with tempfile.TemporaryDirectory() as tmp:
            _, ledger = self._executed_run(tmp)
            accounting = calibrate(ledger, scenarios=4)["budget_accounting"]
            self.assertEqual(accounting["scheduler_charged"], 10)
            # The synthetic attempt log is empty, so the bout's cost looks
            # overcharged. That is exactly the disagreement the check exists
            # to surface.
            self.assertTrue(accounting["overcharged"])


if __name__ == "__main__":
    unittest.main()
