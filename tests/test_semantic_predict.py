from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import ContractError, validate_ledger  # noqa: E402
from semantic_predict import (  # noqa: E402
    PredictError,
    build_pairs,
    render_precedents,
    retrieve_precedents,
    tally,
)
from semantic_search import (  # noqa: E402
    build_proposal_set,
    select_proposal,
    shortlist_proposals,
)
from tests.fixtures import belief_ledger, fixture_registry  # noqa: E402


PRECEDENT_RECORDS = [
    {
        "run_id": "000",
        "status": "keep",
        "change": "raise the learning rate warmup span for the embedding matrix",
    },
    {
        "run_id": "001",
        "status": "discard",
        "change": "raise the learning rate warmup span for the output head",
        "failure_reason": "diverged after step 400",
    },
    {
        "run_id": "002",
        "status": "crash",
        "change": "swap the tokenizer for a byte-level vocabulary",
    },
    {"run_id": "003", "status": "pending", "change": "raise the learning rate warmup span"},
]


class PrecedentRetrievalTests(unittest.TestCase):
    def test_retrieval_labels_outcomes_and_excludes_unresolved_records(self) -> None:
        hits = retrieve_precedents(
            "raise the learning rate warmup span for the embedding matrix",
            PRECEDENT_RECORDS,
            threshold=0.1,
        )
        by_run = {hit["run_id"]: hit["outcome"] for hit in hits}
        self.assertEqual(by_run.get("000"), "worked")
        self.assertEqual(by_run.get("001"), "did not work")
        # An unresolved record is not evidence and must not be labeled.
        self.assertNotIn("003", by_run)
        # A crash is evidence that the change did not work, not a missing
        # observation.
        crash = retrieve_precedents(
            "swap the tokenizer for a byte-level vocabulary",
            PRECEDENT_RECORDS,
            threshold=0.1,
        )
        self.assertEqual(crash[0]["run_id"], "002")
        self.assertEqual(crash[0]["outcome"], "did not work")

    def test_threshold_drops_weak_neighbors_rather_than_padding(self) -> None:
        hits = retrieve_precedents(
            "swap the tokenizer for a byte-level vocabulary",
            PRECEDENT_RECORDS,
            threshold=0.99,
        )
        self.assertEqual([hit["run_id"] for hit in hits], ["002"])

    def test_rendered_precedent_is_change_plus_label_only(self) -> None:
        hits = retrieve_precedents(
            "raise the learning rate warmup span for the output head",
            PRECEDENT_RECORDS,
            threshold=0.1,
        )
        lines = render_precedents(hits)
        self.assertTrue(lines)
        for line in lines:
            self.assertEqual(line.count("\n"), 0)
            self.assertNotIn("diverged", line)
            self.assertTrue(line.endswith("worked") or line.endswith("did not work"))


class StrictConsensusTallyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = ["p-a", "p-b", "p-c"]

    def _sweep(self, winner_of: dict[str, str], confidence: float = 0.8) -> list[dict]:
        verdicts = []
        for pair in build_pairs(self.ids):
            verdicts.append(
                {
                    "a": pair["a"],
                    "b": pair["b"],
                    "winner": winner_of[pair["pair_id"]],
                    "confidence": confidence,
                }
            )
        return verdicts

    def test_both_orders_are_scheduled_for_every_pair(self) -> None:
        pairs = build_pairs(self.ids)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len({p["pair_id"] for p in pairs}), 3)
        for pair in pairs:
            self.assertIn({"a": pair["b"], "b": pair["a"], "pair_id": pair["pair_id"]}, pairs)

    def test_agreed_pairs_score_and_disagreement_abstains(self) -> None:
        verdicts = self._sweep({"p-a|p-b": "p-a", "p-a|p-c": "p-a", "p-b|p-c": "p-b"})
        # Flip one order of p-b|p-c so that pair disagrees with itself.
        for verdict in verdicts:
            if {verdict["a"], verdict["b"]} == {"p-b", "p-c"} and verdict["a"] == "p-c":
                verdict["winner"] = "p-c"
        result = tally(self.ids, verdicts)
        self.assertEqual(result["winner"], "p-a")
        self.assertEqual(result["votes"], {"p-a": 2, "p-b": 0, "p-c": 0})
        self.assertEqual(result["abstentions"], 1)
        self.assertEqual(result["decided_by"], "votes")

    def test_total_abstention_falls_back_to_acquisition_rank(self) -> None:
        verdicts = []
        for pair in build_pairs(self.ids):
            # Every judge picks whichever candidate was presented first, so no
            # pair ever agrees across orders.
            verdicts.append({"a": pair["a"], "b": pair["b"], "winner": pair["a"]})
        result = tally(self.ids, verdicts, base_rank={"p-c": 1, "p-a": 2, "p-b": 3})
        self.assertEqual(result["abstentions"], 3)
        self.assertEqual(result["coverage"], 0.0)
        self.assertEqual(result["winner"], "p-c")
        self.assertEqual(result["decided_by"], "acquisition_rank_fallback")

    def test_incomplete_sweep_is_rejected(self) -> None:
        verdicts = self._sweep({"p-a|p-b": "p-a", "p-a|p-c": "p-a", "p-b|p-c": "p-b"})
        partial = [v for v in verdicts if {v["a"], v["b"]} != {"p-b", "p-c"}]
        with self.assertRaises(PredictError):
            tally(self.ids, partial)
        single_order = [v for v in verdicts if not (v["a"] == "p-c" and v["b"] == "p-b")]
        with self.assertRaises(PredictError):
            tally(self.ids, single_order)


class PredictReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.ledger = belief_ledger(self.registry)
        self.proposals = build_proposal_set(
            self.registry, self.ledger, op="fresh", parents=[]
        )

    def _shortlist(self, size: int = 3) -> list[dict]:
        return shortlist_proposals(
            self.proposals, size=size, policy="coverage", ledger=self.ledger
        )

    def test_shortlist_is_the_acquisition_head(self) -> None:
        shortlist = self._shortlist()
        self.assertEqual([item["rank"] for item in shortlist], [1, 2, 3])
        _, receipt = select_proposal(
            self.proposals, policy="coverage", ledger=self.ledger
        )
        self.assertEqual(
            [item["point_id"] for item in shortlist],
            receipt["ranked_point_ids"][:3],
        )

    def test_forced_winner_records_its_true_acquisition_rank(self) -> None:
        shortlist = self._shortlist()
        runner_up = shortlist[1]["point_id"]
        predict = {
            "method": "pairwise_strict_consensus",
            "winner": runner_up,
            "candidates": [item["point_id"] for item in shortlist],
            "votes": {item["point_id"]: 0 for item in shortlist},
            "coverage": 1.0,
            "abstentions": 0,
            "decided_by": "votes",
            "ranking": [item["point_id"] for item in shortlist],
        }
        point, receipt = select_proposal(
            self.proposals,
            policy="coverage",
            ledger=self.ledger,
            forced_point_id=runner_up,
            predict=predict,
        )
        self.assertEqual(receipt["selected_point_id"], runner_up)
        # The ranking keeps acquisition order, so base_rank shows how far the
        # tournament moved the choice.
        self.assertEqual(receipt["budget"]["base_rank"], 2)
        self.assertEqual(receipt["ranked_point_ids"][1], runner_up)
        self.assertEqual(receipt["predict"]["winner"], runner_up)
        self.assertEqual(point["point_id"], runner_up)

    def test_plain_selection_carries_no_predict_block(self) -> None:
        _, receipt = select_proposal(
            self.proposals, policy="coverage", ledger=self.ledger
        )
        self.assertEqual(receipt["schema_version"], 8)
        self.assertNotIn("predict", receipt)
        self.assertEqual(receipt["budget"]["base_rank"], 1)

    def test_forced_point_outside_the_proposal_set_is_rejected(self) -> None:
        with self.assertRaises(ContractError):
            select_proposal(
                self.proposals,
                policy="coverage",
                ledger=self.ledger,
                forced_point_id="point-not-in-this-space",
            )

    def test_receipt_selecting_below_the_top_needs_a_predict_block(self) -> None:
        shortlist = self._shortlist()
        runner_up = shortlist[1]["point_id"]
        predict = {
            "method": "pairwise_strict_consensus",
            "winner": runner_up,
            "candidates": [item["point_id"] for item in shortlist],
            "votes": {item["point_id"]: 0 for item in shortlist},
            "coverage": 1.0,
            "abstentions": 0,
            "decided_by": "votes",
            "ranking": [item["point_id"] for item in shortlist],
        }
        point, receipt = select_proposal(
            self.proposals,
            policy="coverage",
            ledger=self.ledger,
            forced_point_id=runner_up,
            predict=predict,
        )
        record = {
            "run_id": "900",
            "source_run_ids": [],
            "semantic_point": point,
            "semantic_edges": [],
            "status": "pending",
            "policy_receipt": receipt,
            "dag_revision": 1,
        }
        tampered = dict(self.ledger)
        tampered["records"] = list(self.ledger["records"]) + [record]
        stripped = dict(receipt)
        stripped.pop("predict")
        record["policy_receipt"] = stripped
        errors = validate_ledger(self.registry, tampered)
        self.assertTrue(
            any("predict receipt" in error for error in errors),
            errors,
        )

    def test_predict_winner_must_come_from_the_ranked_field(self) -> None:
        shortlist = self._shortlist()
        selected = shortlist[0]["point_id"]
        _, receipt = select_proposal(
            self.proposals, policy="coverage", ledger=self.ledger
        )
        receipt["predict"] = {
            "method": "pairwise_strict_consensus",
            "winner": selected,
            "candidates": [selected, "point-never-ranked"],
            "votes": {selected: 1, "point-never-ranked": 0},
            "coverage": 1.0,
            "abstentions": 0,
            "decided_by": "votes",
            "ranking": [selected, "point-never-ranked"],
        }
        record = {
            "run_id": "901",
            "source_run_ids": [],
            "semantic_point": self.proposals["proposals"][0]["point"],
            "semantic_edges": [],
            "status": "pending",
            "policy_receipt": receipt,
            "dag_revision": 1,
        }
        tampered = dict(self.ledger)
        tampered["records"] = list(self.ledger["records"]) + [record]
        errors = validate_ledger(self.registry, tampered)
        self.assertTrue(
            any("were never ranked" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
