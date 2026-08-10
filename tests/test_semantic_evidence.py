from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from got_graph import render_incremental  # noqa: E402
import ledger as ledger_tools  # noqa: E402
from semantic_evidence import (  # noqa: E402
    DIRECT_COMPARATOR_CAPABILITY,
    SemanticEvidenceError,
    _json_sha256,
    build_semantic_edges,
    comparator_coverage,
    edge_index,
    edge_observation,
    experience_cited_ids,
    hypothesis_carriers,
    mechanical_gain_direction,
    normalize_coverage,
    render_target_evidence,
    target_evaluation_state,
    validate_parameter_transfer_binding,
    validate_parameter_transfer_evidence,
    validate_lineage_snapshots,
    validate_semantic_edges,
)
from semantic_space import complete_point, space_revision  # noqa: E402
from tests.fixtures import (  # noqa: E402
    attach_matched_transfer,
    belief_ledger,
    fixture_registry,
)


def _append(
    records: list[dict],
    run_id: str,
    parents: list[str],
    point: dict,
    *,
    status: str,
    score: float | None,
    dag_revision: int,
    depth: str = "tuned",
) -> dict:
    record = {
        "run_id": run_id,
        "source_run_ids": parents,
        "semantic_point": point,
        "status": status,
        "final_best_score": score,
        "evaluation_depth": depth,
        "dag_revision": dag_revision,
    }
    record["semantic_edges"] = build_semantic_edges(records, record)
    records.append(record)
    if parents and status in {"keep", "discard"} and score is not None:
        parent = next(item for item in records[:-1] if item["run_id"] == parents[0])
        attach_matched_transfer(parent, record)
    return record


def _category_ledger(registry: dict) -> dict:
    """Four direct, two confounded, and one crash edge for hyp-data-filtered."""
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    filtered_cv = complete_point(
        registry,
        {
            "dim-data-curation": "hyp-data-filtered",
            "dim-validation-selection": "hyp-valid-cv",
        },
    )
    records: list[dict] = []
    rows = [
        ("000", [], baseline, "keep", 0.40),
        ("001", ["000"], filtered, "discard", 0.50),
        ("002", [], baseline, "keep", 0.41),
        ("003", ["002"], filtered, "discard", 0.52),
        ("004", [], baseline, "keep", 0.42),
        ("005", ["004"], filtered, "keep", 0.38),
        ("006", [], baseline, "keep", 0.43),
        ("007", ["006"], filtered, "discard", 0.53),
        ("008", [], baseline, "keep", 0.44),
        ("009", ["008"], filtered_cv, "discard", 0.54),
        ("010", [], baseline, "keep", 0.45),
        ("011", ["010"], filtered_cv, "keep", 0.39),
        ("012", [], baseline, "keep", 0.46),
        ("013", ["012"], filtered, "crash", None),
    ]
    for revision, (run_id, parents, point, status, score) in enumerate(rows, 1):
        _append(
            records,
            run_id,
            parents,
            point,
            status=status,
            score=score,
            dag_revision=revision,
        )
    return {"records": records, "dag_revision": len(rows)}


def _mixed_ledger(registry: dict) -> dict:
    """One single-dimension filtered edge and one multi-dimension stacked edge."""
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    stacked = complete_point(
        registry,
        {
            "dim-model-architecture": "hyp-model-multibranch",
            "dim-ensemble": "hyp-ensemble-stacking",
        },
    )
    records: list[dict] = []
    _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
    _append(records, "001", ["000"], filtered, status="discard", score=0.50, dag_revision=2)
    _append(records, "002", ["000"], stacked, status="keep", score=0.35, dag_revision=3)
    return {"records": records, "dag_revision": 3}


class SemanticEdgeReceiptTests(unittest.TestCase):
    def test_experience_citations_share_one_schema_walk(self) -> None:
        experience = {
            "promising_regions": [{"evidence": ["000", "001"]}],
            "lessons": [{"evidence": ["001", "002"]}],
            "bottlenecks": [],
            "dimension_evidence": [{
                "evidence_run_ids": ["003"],
                "evidence_edge_ids": ["sedge-000-003"],
            }],
            "hypothesis_evidence": [{
                "evidence_run_ids": ["004"],
                "evidence_edge_ids": ["sedge-003-004"],
            }],
        }
        run_ids, edge_ids = experience_cited_ids(experience)
        self.assertEqual(run_ids, {"000", "001", "002", "003", "004"})
        self.assertEqual(edge_ids, {"sedge-000-003", "sedge-003-004"})
        self.assertEqual(experience_cited_ids(None), (set(), set()))

    def test_builds_one_persistent_receipt_per_parent(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        changed = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        parent = {"run_id": "000", "semantic_point": baseline}
        child = {
            "run_id": "001",
            "source_run_ids": ["000"],
            "semantic_point": changed,
        }

        self.assertEqual(
            build_semantic_edges([parent], child),
            [{
                "schema_version": 1,
                "edge_id": "sedge-000-001",
                "space_revision": changed["space_revision"],
                "parent_run_id": "000",
                "child_run_id": "001",
                "change_class": "single_dimension",
                "changes": [{
                    "dimension_id": "dim-data-curation",
                    "operation": "hypothesis_changed",
                    "from_hypothesis_id": "hyp-data-raw",
                    "to_hypothesis_id": "hyp-data-filtered",
                }],
            }],
        )

    def test_fresh_record_builds_no_receipts(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        child = {"run_id": "000", "source_run_ids": [], "semantic_point": baseline}
        self.assertEqual(build_semantic_edges([], child), [])

    def test_build_rejects_missing_parent_with_domain_error(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        child = {
            "run_id": "001",
            "source_run_ids": ["999"],
            "semantic_point": baseline,
        }
        with self.assertRaisesRegex(
            SemanticEvidenceError,
            r"record 001 parent 999 is not an earlier record",
        ):
            build_semantic_edges([], child)

    def test_unchanged_points_produce_same_point_receipt(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        parent = {"run_id": "000", "semantic_point": baseline}
        child = {"run_id": "001", "source_run_ids": ["000"], "semantic_point": baseline}
        (receipt,) = build_semantic_edges([parent], child)
        self.assertEqual(receipt["change_class"], "same_point")
        self.assertEqual(receipt["changes"], [])

    def test_activation_and_deactivation_operations(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        multibranch = complete_point(
            registry, {"dim-model-architecture": "hyp-model-multibranch"}
        )
        parent = {"run_id": "000", "semantic_point": baseline}
        child = {"run_id": "001", "source_run_ids": ["000"], "semantic_point": multibranch}
        (receipt,) = build_semantic_edges([parent], child)
        changes = {change["dimension_id"]: change for change in receipt["changes"]}
        self.assertEqual(
            changes["dim-model-architecture"],
            {
                "dimension_id": "dim-model-architecture",
                "operation": "hypothesis_changed",
                "from_hypothesis_id": "hyp-model-linear",
                "to_hypothesis_id": "hyp-model-multibranch",
            },
        )
        self.assertEqual(
            changes["dim-ensemble"],
            {
                "dimension_id": "dim-ensemble",
                "operation": "dimension_activated",
                "from_hypothesis_id": None,
                "to_hypothesis_id": "hyp-ensemble-identity",
            },
        )

        back = {"run_id": "002", "source_run_ids": ["001"], "semantic_point": baseline}
        (receipt,) = build_semantic_edges(
            [{"run_id": "001", "semantic_point": multibranch}], back
        )
        changes = {change["dimension_id"]: change for change in receipt["changes"]}
        self.assertEqual(
            changes["dim-ensemble"],
            {
                "dimension_id": "dim-ensemble",
                "operation": "dimension_deactivated",
                "from_hypothesis_id": "hyp-ensemble-identity",
                "to_hypothesis_id": None,
            },
        )

    def test_multi_dimension_receipt_keeps_registry_order(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        stacked = complete_point(
            registry,
            {
                "dim-model-architecture": "hyp-model-multibranch",
                "dim-ensemble": "hyp-ensemble-stacking",
            },
        )
        parent = {"run_id": "000", "semantic_point": baseline}
        child = {"run_id": "001", "source_run_ids": ["000"], "semantic_point": stacked}
        self.assertEqual(
            build_semantic_edges([parent], child),
            [{
                "schema_version": 1,
                "edge_id": "sedge-000-001",
                "space_revision": stacked["space_revision"],
                "parent_run_id": "000",
                "child_run_id": "001",
                "change_class": "multi_dimension",
                "changes": [
                    {
                        "dimension_id": "dim-model-architecture",
                        "operation": "hypothesis_changed",
                        "from_hypothesis_id": "hyp-model-linear",
                        "to_hypothesis_id": "hyp-model-multibranch",
                    },
                    {
                        "dimension_id": "dim-validation-selection",
                        "operation": "hypothesis_changed",
                        "from_hypothesis_id": "hyp-valid-holdout",
                        "to_hypothesis_id": "hyp-valid-cv",
                    },
                    {
                        "dimension_id": "dim-ensemble",
                        "operation": "dimension_activated",
                        "from_hypothesis_id": None,
                        "to_hypothesis_id": "hyp-ensemble-stacking",
                    },
                ],
            }],
        )

    def test_crossover_receipt_order_follows_source_run_ids(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        stacked = complete_point(
            registry,
            {
                "dim-model-architecture": "hyp-model-multibranch",
                "dim-ensemble": "hyp-ensemble-stacking",
            },
        )
        parents = [
            {"run_id": "000", "semantic_point": baseline},
            {"run_id": "002", "semantic_point": stacked},
        ]
        child = {
            "run_id": "003",
            "source_run_ids": ["002", "000"],
            "semantic_point": filtered,
        }
        receipts = build_semantic_edges(parents, child)
        self.assertEqual(
            [receipt["edge_id"] for receipt in receipts],
            ["sedge-002-003", "sedge-000-003"],
        )
        self.assertEqual(receipts[0]["change_class"], "multi_dimension")
        self.assertEqual(receipts[1]["change_class"], "single_dimension")

    def test_validate_accepts_mechanical_receipts(self) -> None:
        registry = fixture_registry()
        records = belief_ledger(registry)["records"]
        self.assertEqual(validate_semantic_edges(records[:1], records[1], registry), [])
        self.assertEqual(validate_semantic_edges(records[:3], records[3], registry), [])
        self.assertEqual(validate_semantic_edges([], records[0], registry), [])

    def test_validate_rejects_forged_receipts(self) -> None:
        registry = fixture_registry()
        records = belief_ledger(registry)["records"]
        prior, child = records[:1], records[1]

        forged = copy.deepcopy(child)
        forged["semantic_edges"][0]["changes"][0]["operation"] = "dimension_activated"
        self.assertNotEqual(validate_semantic_edges(prior, forged, registry), [])

        forged = copy.deepcopy(child)
        forged["semantic_edges"][0]["edge_id"] = "sedge-000-002"
        self.assertNotEqual(validate_semantic_edges(prior, forged, registry), [])

        forged = copy.deepcopy(child)
        forged["semantic_edges"][0]["space_revision"] = "sha256:" + "0" * 64
        self.assertNotEqual(validate_semantic_edges(prior, forged, registry), [])

        forged = copy.deepcopy(child)
        forged["semantic_edges"][0]["changes"] = []
        self.assertNotEqual(validate_semantic_edges(prior, forged, registry), [])

        forged = copy.deepcopy(child)
        forged["semantic_edges"] = []
        self.assertNotEqual(validate_semantic_edges(prior, forged, registry), [])

        missing_parent = copy.deepcopy(child)
        self.assertNotEqual(validate_semantic_edges([], missing_parent, registry), [])


class SemanticEdgeObservationTests(unittest.TestCase):
    def test_edge_index_collects_persisted_receipts(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        index = edge_index(ledger)
        self.assertEqual(sorted(index), ["sedge-000-001", "sedge-002-003"])
        self.assertEqual(index["sedge-000-001"]["parent_run_id"], "000")

    def test_edge_observation_reports_terminal_delta(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        first = edge_observation(ledger, "sedge-000-001")
        self.assertEqual(first["score_basis"], "paired_semantic_control")
        self.assertEqual(first["parent_score"], 0.4)
        self.assertEqual(first["child_score"], 0.5)
        self.assertEqual(first["semantic_delta"], 0.1)
        self.assertEqual(first["tuning_delta"], 0.0)
        self.assertEqual(first["total_delta"], 0.1)
        self.assertEqual(
            first["parameter_transfer_receipt_sha256"],
            ledger["records"][1]["parameter_transfer"]["receipt"][
                "receipt_sha256"
            ],
        )
        second = edge_observation(ledger, "sedge-002-003")
        self.assertEqual(second["score_basis"], "paired_semantic_control")
        self.assertEqual(second["parent_score"], 0.41)
        self.assertEqual(second["child_score"], 0.52)
        self.assertEqual(second["delta"], 0.11)
        self.assertEqual(edge_observation(ledger, "sedge-000-999"), {})

    def test_edge_observation_nulls_crash_delta(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        records: list[dict] = []
        _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append(records, "001", ["000"], filtered, status="crash", score=None, dag_revision=2)
        ledger = {"records": records, "dag_revision": 2}
        self.assertEqual(
            edge_observation(ledger, "sedge-000-001"),
            {
                "edge_id": "sedge-000-001",
                "change_class": "single_dimension",
                "parent_run_id": "000",
                "child_run_id": "001",
                "parent_status": "keep",
                "child_status": "crash",
                "parent_score": 0.4,
                "child_score": None,
                "delta": None,
                "score_basis": "independently_tuned_final",
            },
        )

    def test_legacy_single_dimension_edge_is_confounded(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        _append(
            records,
            "000",
            [],
            baseline,
            status="keep",
            score=0.40,
            dag_revision=1,
        )
        child = {
            "run_id": "001",
            "source_run_ids": ["000"],
            "semantic_point": filtered,
            "status": "discard",
            "final_best_score": 0.50,
            "dag_revision": 2,
        }
        child["semantic_edges"] = build_semantic_edges(records, child)
        records.append(child)
        ledger = {"records": records}
        self.assertEqual(
            comparator_coverage(
                ledger,
                ["sedge-000-001"],
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            ),
            {
                "direct_tuned_edges": 0,
                "direct_lightly_tuned_edges": 0,
                "direct_noncrash_edges": 0,
                "confounded_noncrash_edges": 1,
                "crash_edges": 0,
            },
        )
        self.assertEqual(
            edge_observation(ledger, "sedge-000-001")["score_basis"],
            "independently_tuned_final",
        )

    def test_comparator_coverage_counts_only_cited_touching_receipts(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        both = ["sedge-000-001", "sedge-002-003"]
        expected = {
            "direct_tuned_edges": 2,
            "direct_lightly_tuned_edges": 0,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        }
        self.assertEqual(
            comparator_coverage(
                ledger, both, target_kind="hypothesis", target_id="hyp-data-filtered"
            ),
            expected,
        )
        self.assertEqual(
            comparator_coverage(
                ledger, both, target_kind="dimension", target_id="dim-data-curation"
            ),
            expected,
        )
        self.assertEqual(
            comparator_coverage(
                ledger, both, target_kind="hypothesis", target_id="hyp-model-linear"
            ),
            {"direct_tuned_edges": 0, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )
        self.assertEqual(
            comparator_coverage(
                ledger,
                ["sedge-000-001"],
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            ),
            {"direct_tuned_edges": 1, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )
        self.assertEqual(
            comparator_coverage(
                ledger,
                ["sedge-000-999"],
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            ),
            {"direct_tuned_edges": 0, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )

    def test_lightly_tuned_edges_have_their_own_bucket(self) -> None:
        # A direct comparator whose child is tuned_lightly increments
        # direct_lightly_tuned_edges, not direct_tuned_edges and not
        # direct_noncrash_edges.
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append(
            records,
            "001",
            ["000"],
            filtered,
            status="discard",
            score=0.50,
            dag_revision=2,
            depth="tuned_lightly",
        )
        ledger = {"records": records, "dag_revision": 2}
        self.assertEqual(
            comparator_coverage(
                ledger,
                ["sedge-000-001"],
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            ),
            {
                "direct_tuned_edges": 0,
                "direct_lightly_tuned_edges": 1,
                "direct_noncrash_edges": 0,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
        )

    def test_comparator_covered_accepts_two_lightly_tuned_edges(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append(
            records,
            "001",
            ["000"],
            filtered,
            status="discard",
            score=0.50,
            dag_revision=2,
            depth="tuned_lightly",
        )
        _append(records, "002", [], baseline, status="keep", score=0.41, dag_revision=3)
        _append(
            records,
            "003",
            ["002"],
            filtered,
            status="discard",
            score=0.52,
            dag_revision=4,
            depth="tuned_lightly",
        )
        ledger = {"records": records, "dag_revision": 4}
        edges = ["sedge-000-001", "sedge-002-003"]

        def state(run_ids: list[str], edge_ids: list[str]) -> str:
            return target_evaluation_state(
                ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_run_ids=run_ids,
                evidence_edge_ids=edge_ids,
            )

        self.assertEqual(state(["000", "001", "002", "003"], edges), "comparator_covered")
        # One tuned plus one lightly-tuned direct edge also covers.
        records[3]["evaluation_depth"] = "tuned"
        self.assertEqual(state(["000", "001", "002", "003"], edges), "comparator_covered")
        # A single lightly-tuned edge is a non-crash observation, not a failure.
        records[3]["evaluation_depth"] = "tuned_lightly"
        self.assertEqual(state([], ["sedge-000-001"]), "observed")

    def test_mechanical_gain_direction_depth_bar(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append(records, "001", ["000"], filtered, status="keep", score=0.30, dag_revision=2)
        _append(records, "002", [], baseline, status="keep", score=0.41, dag_revision=3)
        _append(records, "003", ["002"], filtered, status="keep", score=0.31, dag_revision=4)
        _append(records, "004", [], baseline, status="keep", score=0.42, dag_revision=5)
        _append(records, "005", ["004"], filtered, status="keep", score=0.32, dag_revision=6)
        ledger = {"records": records, "dag_revision": 6}
        edges = ["sedge-000-001", "sedge-002-003", "sedge-004-005"]

        def direction(edge_ids: list[str]) -> str:
            return mechanical_gain_direction(
                ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_edge_ids=edge_ids,
            )

        # Legacy: two agreeing tuned controls orient a direction.
        self.assertEqual(direction(edges[:2]), "positive")
        # Two agreeing lightly-tuned controls stay below the bar and abstain.
        for run_id in ("001", "003", "005"):
            records[int(run_id)]["evaluation_depth"] = "tuned_lightly"
        self.assertEqual(direction(edges[:2]), "none")
        # Three agreeing lightly-tuned controls clear the depth bar.
        self.assertEqual(direction(edges), "positive")
        # Mixed tuned/lightly-tuned controls also orient in numbers.
        records[1]["evaluation_depth"] = "tuned"
        self.assertEqual(direction(edges), "positive")

    def test_legacy_coverage_receipts_normalize(self) -> None:
        # The four-key (schema-2) and three-key (schema-1) coverage shapes read
        # forward with direct_lightly_tuned_edges at 0.
        four_key = {
            "direct_tuned_edges": 2,
            "direct_noncrash_edges": 1,
            "confounded_noncrash_edges": 1,
            "crash_edges": 0,
        }
        self.assertEqual(
            normalize_coverage(four_key),
            {
                "direct_tuned_edges": 2,
                "direct_lightly_tuned_edges": 0,
                "direct_noncrash_edges": 1,
                "confounded_noncrash_edges": 1,
                "crash_edges": 0,
            },
        )
        three_key = {
            "direct_noncrash_edges": 1,
            "confounded_noncrash_edges": 1,
            "crash_edges": 0,
        }
        self.assertEqual(
            normalize_coverage(three_key),
            {
                "direct_tuned_edges": 0,
                "direct_lightly_tuned_edges": 0,
                "direct_noncrash_edges": 1,
                "confounded_noncrash_edges": 1,
                "crash_edges": 0,
            },
        )
        five_key = {
            "direct_tuned_edges": 1,
            "direct_lightly_tuned_edges": 2,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        }
        self.assertEqual(normalize_coverage(five_key), five_key)
        self.assertIsNone(normalize_coverage({"direct_tuned_edges": 1}))

    def test_gain_direction_uses_repeated_control_scores_not_final_tuning(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        _append(
            records,
            "000",
            [],
            baseline,
            status="keep",
            score=0.40,
            dag_revision=1,
        )
        _append(
            records,
            "001",
            ["000"],
            filtered,
            status="keep",
            score=0.30,
            dag_revision=2,
        )
        _append(
            records,
            "002",
            [],
            baseline,
            status="keep",
            score=0.41,
            dag_revision=3,
        )
        _append(
            records,
            "003",
            ["002"],
            filtered,
            status="keep",
            score=0.31,
            dag_revision=4,
        )
        ledger = {"records": records}
        edges = ["sedge-000-001", "sedge-002-003"]
        self.assertEqual(
            mechanical_gain_direction(
                ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_edge_ids=edges,
            ),
            "positive",
        )

        # A later tuning win belongs to the inner loop; it cannot rewrite the
        # fixed semantic-control deltas or their direction.
        records[1]["final_best_score"] = 0.10
        records[3]["final_best_score"] = 0.60
        self.assertEqual(
            mechanical_gain_direction(
                ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_edge_ids=edges,
            ),
            "positive",
        )
        self.assertEqual(
            edge_observation(ledger, edges[1])["semantic_delta"],
            -0.1,
        )
        self.assertEqual(
            edge_observation(ledger, edges[1])["tuning_delta"],
            0.29,
        )

    def test_reset_or_tampered_transfer_is_never_a_direct_comparator(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        parent = _append(
            records,
            "000",
            [],
            baseline,
            status="keep",
            score=0.40,
            dag_revision=1,
        )
        child = {
            "run_id": "001",
            "source_run_ids": ["000"],
            "semantic_point": filtered,
            "status": "discard",
            "final_best_score": 0.50,
            "dag_revision": 2,
        }
        child["semantic_edges"] = build_semantic_edges(records, child)
        records.append(child)
        attach_matched_transfer(parent, child, reset=True)
        ledger = {"records": records}
        edge_ids = ["sedge-000-001"]
        self.assertEqual(validate_parameter_transfer_evidence(child), [])
        self.assertEqual(
            comparator_coverage(
                ledger,
                edge_ids,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            )["confounded_noncrash_edges"],
            1,
        )

        attach_matched_transfer(parent, child)
        child["parameter_transfer"]["warm_start_observations"][0]["params"][
            "shared"
        ] = 9.0
        self.assertTrue(validate_parameter_transfer_evidence(child))
        self.assertEqual(
            comparator_coverage(
                ledger,
                edge_ids,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            )["direct_noncrash_edges"],
            0,
        )

    def test_rehashed_parent_claim_must_match_durable_ledger_snapshot(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        parent = _append(
            records,
            "000",
            [],
            baseline,
            status="keep",
            score=0.40,
            dag_revision=1,
        )
        child = _append(
            records,
            "001",
            ["000"],
            filtered,
            status="discard",
            score=0.50,
            dag_revision=2,
        )
        ledger = {"records": records}
        self.assertEqual(validate_parameter_transfer_binding(ledger, child), [])

        forged = copy.deepcopy(child)
        transfer = forged["parameter_transfer"]
        receipt = transfer["receipt"]
        receipt["primary_parent"]["incumbent_score"] = 0.10
        receipt["primary_parent"]["ledger_record_sha256"] = "sha256:" + "f" * 64
        unhashed = dict(receipt)
        unhashed.pop("receipt_sha256")
        receipt["receipt_sha256"] = _json_sha256(unhashed)
        transfer["inherited_control"]["parent_incumbent_score"] = 0.10
        transfer["inherited_control"]["receipt_sha256"] = receipt["receipt_sha256"]
        for observation in transfer["warm_start_observations"]:
            observation[
                "parameter_transfer_receipt_sha256"
            ] = receipt["receipt_sha256"]

        # The forged receipt is internally complete and self-hashed, but it
        # cannot become semantic evidence because its parent snapshot is false.
        self.assertEqual(validate_parameter_transfer_evidence(forged), [])
        binding_errors = validate_parameter_transfer_binding(
            {"records": [parent, forged]}, forged
        )
        self.assertTrue(
            any("durable lineage snapshot" in error for error in binding_errors),
            binding_errors,
        )
        self.assertEqual(
            comparator_coverage(
                {"records": [parent, forged]},
                ["sedge-000-001"],
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            )["direct_noncrash_edges"],
            0,
        )

    def test_snapshotted_parent_revision_survives_later_tuning(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        parent = ledger["records"][0]
        child = ledger["records"][1]
        self.assertEqual(validate_parameter_transfer_binding(ledger, child), [])

        ledger_tools._capture_transfer_parent_snapshot(ledger, child)
        self.assertNotIn("receipt_sha256", ledger["lineage_snapshots"][0])
        wrong_shape = copy.deepcopy(ledger)
        wrong_shape["lineage_snapshots"][0]["receipt_sha256"] = "sha256:obsolete"
        self.assertIn(
            "ledger.lineage_snapshots[0] has an invalid shape",
            validate_lineage_snapshots(wrong_shape),
        )
        old_record_hash = child["parameter_transfer"]["receipt"][
            "primary_parent"
        ]["ledger_record_sha256"]
        parent["final_best_score"] = 0.35
        parent["applied_incumbent"]["score"] = 0.35

        self.assertNotEqual(_json_sha256(parent), old_record_hash)
        self.assertEqual(validate_lineage_snapshots(ledger), [])
        self.assertEqual(validate_parameter_transfer_binding(ledger, child), [])

    def test_unpaired_parent_parameter_control_is_uncertainty_only(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        child = ledger["records"][1]
        transfer = child["parameter_transfer"]
        receipt = transfer["receipt"]
        receipt["semantic_control"] = {
            "status": "unverified",
            "reason": "no_same_child_code_control_treatment_pair",
        }
        transfer["warm_start_observations"] = [
            transfer["warm_start_observations"][0]
        ]
        unhashed = dict(receipt)
        unhashed.pop("receipt_sha256")
        receipt["receipt_sha256"] = _json_sha256(unhashed)
        transfer["inherited_control"]["receipt_sha256"] = receipt["receipt_sha256"]
        transfer["warm_start_observations"][0][
            "parameter_transfer_receipt_sha256"
        ] = receipt["receipt_sha256"]

        self.assertEqual(validate_parameter_transfer_binding(ledger, child), [])
        coverage = comparator_coverage(
            ledger,
            ["sedge-000-001"],
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
        )
        self.assertEqual(coverage["direct_noncrash_edges"], 0)
        self.assertEqual(coverage["confounded_noncrash_edges"], 1)

    def test_production_capability_gate_keeps_synthetic_pairs_confounded(
        self,
    ) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        ledger["direct_comparator_capability"] = dict(
            DIRECT_COMPARATOR_CAPABILITY
        )
        edge_ids = ["sedge-000-001", "sedge-002-003"]

        coverage = comparator_coverage(
            ledger,
            edge_ids,
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
        )

        self.assertEqual(coverage["direct_noncrash_edges"], 0)
        self.assertEqual(coverage["confounded_noncrash_edges"], 2)
        self.assertEqual(
            target_evaluation_state(
                ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_run_ids=["000", "001", "002", "003"],
                evidence_edge_ids=edge_ids,
            ),
            "observed",
        )
        self.assertEqual(
            mechanical_gain_direction(
                ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_edge_ids=edge_ids,
            ),
            "none",
        )
        self.assertEqual(
            render_target_evidence(
                registry,
                ledger,
                target_ids=["hyp-data-filtered"],
            )["direct_comparator_capability"],
            DIRECT_COMPARATOR_CAPABILITY,
        )

    def test_rehashed_parent_params_cannot_replace_applied_snapshot(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        child = copy.deepcopy(ledger["records"][1])
        transfer = child["parameter_transfer"]
        receipt = transfer["receipt"]
        primary = receipt["primary_parent"]
        projection = receipt["projection"]
        semantic_control = receipt["semantic_control"]
        primary["incumbent_params"] = {"shared": 999.0}
        projection["params"] = {"shared": 999.0}
        projection["params_sha256"] = _json_sha256(projection["params"])
        projection["copied"] = [{"key": "shared", "value": 999.0}]
        treatment = {"shared": 1000.0}
        semantic_control["control_params_sha256"] = projection["params_sha256"]
        semantic_control["treatment_params_sha256"] = _json_sha256(treatment)
        observations = transfer["warm_start_observations"]
        observations[0]["params"] = projection["params"]
        observations[0]["params_sha256"] = projection["params_sha256"]
        observations[1]["params"] = treatment
        observations[1]["params_sha256"] = semantic_control[
            "treatment_params_sha256"
        ]
        transfer["inherited_control"]["params_sha256"] = projection[
            "params_sha256"
        ]
        unhashed = dict(receipt)
        unhashed.pop("receipt_sha256")
        receipt["receipt_sha256"] = _json_sha256(unhashed)
        transfer["inherited_control"]["receipt_sha256"] = receipt["receipt_sha256"]
        for observation in observations:
            observation[
                "parameter_transfer_receipt_sha256"
            ] = receipt["receipt_sha256"]

        forged_ledger = {"records": [ledger["records"][0], child]}
        self.assertEqual(validate_parameter_transfer_evidence(child), [])
        errors = validate_parameter_transfer_binding(forged_ledger, child)
        self.assertTrue(
            any("incumbent_params" in error for error in errors),
            errors,
        )

    def test_legacy_policy_cannot_gain_direct_comparator_status(self) -> None:
        registry = fixture_registry()
        for version in (4, 5):
            with self.subTest(version=version):
                ledger = belief_ledger(registry)
                child = ledger["records"][1]
                child["policy_receipt"]["schema_version"] = version
                coverage = comparator_coverage(
                    ledger,
                    ["sedge-000-001"],
                    target_kind="hypothesis",
                    target_id="hyp-data-filtered",
                )
                self.assertEqual(coverage["direct_noncrash_edges"], 0)
                self.assertEqual(coverage["confounded_noncrash_edges"], 1)

    def test_pending_edge_is_not_noncrash_comparator_coverage(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        records: list[dict] = []
        _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append(
            records,
            "001",
            ["000"],
            filtered,
            status="pending",
            score=None,
            dag_revision=2,
        )
        ledger = {"records": records, "dag_revision": 2}
        self.assertEqual(
            comparator_coverage(
                ledger,
                ["sedge-000-001"],
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
            ),
            {
                "direct_tuned_edges": 0,
                "direct_lightly_tuned_edges": 0,
                "direct_noncrash_edges": 0,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
        )
        self.assertEqual(
            target_evaluation_state(
                ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_run_ids=[],
                evidence_edge_ids=["sedge-000-001"],
            ),
            "unevaluated",
        )
        view = render_target_evidence(
            registry,
            ledger,
            target_ids=["hyp-data-filtered"],
        )
        (block,) = view["hypothesis_targets"]
        self.assertEqual(block["evaluation_state"], "unevaluated")
        self.assertEqual(block["evidence_run_ids"], [])
        self.assertEqual(block["evidence_edge_ids"], [])
        self.assertEqual(
            block["comparator_coverage"],
            {
                "direct_tuned_edges": 0,
                "direct_lightly_tuned_edges": 0,
                "direct_noncrash_edges": 0,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
        )

    def test_comparator_coverage_separates_confounded_and_crash(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        stacked = complete_point(
            registry,
            {
                "dim-model-architecture": "hyp-model-multibranch",
                "dim-ensemble": "hyp-ensemble-stacking",
            },
        )
        records: list[dict] = []
        _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append(records, "001", ["000"], stacked, status="keep", score=0.35, dag_revision=2)
        _append(records, "002", ["000"], filtered, status="crash", score=None, dag_revision=3)
        ledger = {"records": records, "dag_revision": 3}
        edge_ids = ["sedge-000-001", "sedge-000-002"]
        self.assertEqual(
            comparator_coverage(
                ledger, edge_ids, target_kind="hypothesis", target_id="hyp-model-multibranch"
            ),
            {"direct_tuned_edges": 0, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 1, "crash_edges": 0},
        )
        self.assertEqual(
            comparator_coverage(
                ledger, edge_ids, target_kind="hypothesis", target_id="hyp-data-filtered"
            ),
            {"direct_tuned_edges": 0, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 1},
        )

    def test_target_evaluation_state_machine(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        state = target_evaluation_state(
            ledger,
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
            evidence_run_ids=[],
            evidence_edge_ids=[],
        )
        self.assertEqual(state, "unevaluated")
        state = target_evaluation_state(
            ledger,
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
            evidence_run_ids=["000", "001"],
            evidence_edge_ids=["sedge-000-001"],
        )
        self.assertEqual(state, "observed")
        state = target_evaluation_state(
            ledger,
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
            evidence_run_ids=["000", "001", "002", "003"],
            evidence_edge_ids=["sedge-000-001", "sedge-002-003"],
        )
        self.assertEqual(state, "comparator_covered")

        mixed = _mixed_ledger(registry)
        state = target_evaluation_state(
            mixed,
            target_kind="hypothesis",
            target_id="hyp-model-multibranch",
            evidence_run_ids=["000", "002"],
            evidence_edge_ids=["sedge-000-002"],
        )
        self.assertEqual(state, "observed")

        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        records: list[dict] = []
        _append(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append(records, "001", ["000"], filtered, status="crash", score=None, dag_revision=2)
        crashed = {"records": records, "dag_revision": 2}
        state = target_evaluation_state(
            crashed,
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
            evidence_run_ids=["000", "001"],
            evidence_edge_ids=["sedge-000-001"],
        )
        self.assertEqual(state, "failed")


class TargetEvidenceViewTests(unittest.TestCase):
    def test_bounded_view_recovers_old_direct_edges(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        baseline = complete_point(registry)
        records = ledger["records"]
        parent = "002"
        for offset in range(6):
            run_id = f"{4 + offset:03d}"
            _append(
                records,
                run_id,
                [parent],
                baseline,
                status="keep",
                score=0.39 - 0.01 * offset,
                dag_revision=5 + offset,
            )
            parent = run_id
        ledger["dag_revision"] = 10
        ledger["experience"] = {"dag_revision": 9}

        graph_view = render_incremental(ledger, top=1, bottom=1)
        self.assertEqual([node["id"] for node in graph_view["delta_nodes"]], ["009"])
        graph_edges = {
            (edge["parent"], edge["child"]) for edge in graph_view["delta_edges"]
        }
        self.assertNotIn(("000", "001"), graph_edges)
        self.assertNotIn(("002", "003"), graph_edges)

        view = render_target_evidence(registry, ledger, target_ids=["hyp-data-filtered"])
        self.assertEqual(view["schema_version"], 2)
        self.assertEqual(view["space_revision"], space_revision(registry))
        self.assertEqual(view["dag_revision"], 10)
        self.assertEqual(view["experience_dag_revision"], 9)
        self.assertEqual(
            view["bounds"],
            {
                "max_dimensions": 16,
                "max_hypotheses": 32,
                "max_edges_per_target": 5,
                "max_runs_per_target": 5,
            },
        )
        self.assertEqual(view["dimension_targets"], [])
        self.assertEqual(view["omitted_target_ids"], {"dimensions": [], "hypotheses": []})
        self.assertEqual(len(view["hypothesis_targets"]), 1)
        block = view["hypothesis_targets"][0]
        self.assertEqual(block["target_id"], "hyp-data-filtered")
        self.assertEqual(block["evaluation_state"], "comparator_covered")
        self.assertEqual(block["evidence_run_ids"], ["000", "001", "002", "003"])
        self.assertEqual(block["evidence_edge_ids"], ["sedge-000-001", "sedge-002-003"])
        self.assertEqual(
            block["comparator_coverage"],
            {"direct_tuned_edges": 2, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )
        self.assertEqual(
            block["available_comparator_coverage"],
            {"direct_tuned_edges": 2, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )
        self.assertEqual(
            block["omitted_edge_counts"],
            {"direct_tuned_edges": 0, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )
        self.assertEqual(
            block["edges"],
            [
                edge_observation(ledger, "sedge-000-001"),
                edge_observation(ledger, "sedge-002-003"),
            ],
        )

    def test_category_balancing_and_edge_cap(self) -> None:
        registry = fixture_registry()
        ledger = _category_ledger(registry)
        view = render_target_evidence(
            registry, ledger, max_edges_per_target=5, target_ids=["hyp-data-filtered"]
        )
        (block,) = view["hypothesis_targets"]
        self.assertEqual(
            block["evidence_edge_ids"],
            [
                "sedge-002-003",
                "sedge-004-005",
                "sedge-006-007",
                "sedge-010-011",
                "sedge-012-013",
            ],
        )
        self.assertEqual(
            block["comparator_coverage"],
            {"direct_tuned_edges": 3, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 1, "crash_edges": 1},
        )
        self.assertEqual(
            block["available_comparator_coverage"],
            {"direct_tuned_edges": 4, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 2, "crash_edges": 1},
        )
        self.assertEqual(
            block["omitted_edge_counts"],
            {"direct_tuned_edges": 1, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 1, "crash_edges": 0},
        )
        self.assertEqual(block["evidence_run_ids"], ["002", "003", "004", "005", "006"])
        self.assertEqual(block["evaluation_state"], "comparator_covered")

    def test_small_edge_cap_is_enforced_across_categories(self) -> None:
        registry = fixture_registry()
        ledger = _category_ledger(registry)
        view = render_target_evidence(
            registry, ledger, max_edges_per_target=2, target_ids=["hyp-data-filtered"]
        )
        (block,) = view["hypothesis_targets"]
        self.assertEqual(
            block["evidence_edge_ids"],
            ["sedge-004-005", "sedge-006-007"],
        )
        self.assertEqual(
            block["comparator_coverage"],
            {"direct_tuned_edges": 2, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )
        self.assertEqual(
            block["available_comparator_coverage"],
            {"direct_tuned_edges": 4, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 2, "crash_edges": 1},
        )
        self.assertEqual(
            block["omitted_edge_counts"],
            {"direct_tuned_edges": 2, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 2, "crash_edges": 1},
        )
        self.assertEqual(view["bounds"]["max_edges_per_target"], 2)
        self.assertLessEqual(len(block["edges"]), view["bounds"]["max_edges_per_target"])

    def test_target_count_caps_and_omitted_ids(self) -> None:
        registry = fixture_registry()
        ledger = _mixed_ledger(registry)
        view = render_target_evidence(registry, ledger, max_dimensions=2, max_hypotheses=2)
        self.assertEqual(
            [block["target_id"] for block in view["dimension_targets"]],
            ["dim-data-curation", "dim-model-architecture"],
        )
        self.assertEqual(
            view["omitted_target_ids"]["dimensions"],
            ["dim-validation-selection", "dim-ensemble"],
        )
        self.assertEqual(
            [block["target_id"] for block in view["hypothesis_targets"]],
            ["hyp-data-raw", "hyp-data-filtered"],
        )
        self.assertEqual(
            view["omitted_target_ids"]["hypotheses"],
            [
                "hyp-model-linear",
                "hyp-model-multibranch",
                "hyp-valid-holdout",
                "hyp-valid-cv",
                "hyp-ensemble-stacking",
            ],
        )

    def test_priority_orders_new_preserved_then_other(self) -> None:
        registry = fixture_registry()
        ledger = _mixed_ledger(registry)
        ledger["experience"] = {
            "dag_revision": 2,
            "hypothesis_evidence": [{"target_id": "hyp-data-filtered"}],
        }
        view = render_target_evidence(registry, ledger)
        self.assertEqual(
            [block["target_id"] for block in view["hypothesis_targets"]],
            [
                "hyp-model-linear",
                "hyp-model-multibranch",
                "hyp-valid-holdout",
                "hyp-valid-cv",
                "hyp-ensemble-stacking",
                "hyp-data-filtered",
                "hyp-data-raw",
            ],
        )
        self.assertEqual(
            [block["target_id"] for block in view["dimension_targets"]],
            [
                "dim-model-architecture",
                "dim-validation-selection",
                "dim-ensemble",
                "dim-data-curation",
            ],
        )

    def test_dimension_target_block(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        view = render_target_evidence(registry, ledger, target_ids=["dim-data-curation"])
        self.assertEqual(view["hypothesis_targets"], [])
        (block,) = view["dimension_targets"]
        self.assertEqual(block["target_id"], "dim-data-curation")
        self.assertEqual(block["evidence_edge_ids"], ["sedge-000-001", "sedge-002-003"])
        self.assertEqual(
            block["comparator_coverage"],
            {"direct_tuned_edges": 2, "direct_lightly_tuned_edges": 0, "direct_noncrash_edges": 0, "confounded_noncrash_edges": 0, "crash_edges": 0},
        )
        self.assertEqual(block["evaluation_state"], "comparator_covered")

    def test_explicit_target_ids_bypass_target_caps(self) -> None:
        registry = fixture_registry()
        ledger = _mixed_ledger(registry)
        view = render_target_evidence(
            registry,
            ledger,
            max_dimensions=1,
            max_hypotheses=1,
            target_ids=["dim-ensemble", "hyp-valid-cv", "hyp-data-filtered"],
        )
        self.assertEqual(
            [block["target_id"] for block in view["dimension_targets"]], ["dim-ensemble"]
        )
        self.assertEqual(
            [block["target_id"] for block in view["hypothesis_targets"]],
            ["hyp-valid-cv", "hyp-data-filtered"],
        )
        self.assertEqual(view["omitted_target_ids"], {"dimensions": [], "hypotheses": []})

    def test_empty_ledger_has_no_targets(self) -> None:
        registry = fixture_registry()
        view = render_target_evidence(registry, {"records": []})
        self.assertEqual(view["dimension_targets"], [])
        self.assertEqual(view["hypothesis_targets"], [])
        self.assertEqual(view["omitted_target_ids"], {"dimensions": [], "hypotheses": []})
        self.assertEqual(view["dag_revision"], 0)
        self.assertEqual(view["experience_dag_revision"], 0)

    def test_view_bound_and_target_id_validation(self) -> None:
        registry = fixture_registry()
        ledger = belief_ledger(registry)
        for kwargs in (
            {"max_dimensions": 0},
            {"max_dimensions": 17},
            {"max_hypotheses": 0},
            {"max_hypotheses": 33},
            {"max_edges_per_target": 1},
            {"max_edges_per_target": 6},
        ):
            with self.subTest(**kwargs), self.assertRaises(SemanticEvidenceError):
                render_target_evidence(registry, ledger, **kwargs)
        render_target_evidence(
            registry,
            ledger,
            max_dimensions=16,
            max_hypotheses=32,
            max_edges_per_target=5,
        )
        render_target_evidence(
            registry,
            ledger,
            max_dimensions=1,
            max_hypotheses=1,
            max_edges_per_target=2,
        )
        with self.assertRaises(SemanticEvidenceError):
            render_target_evidence(registry, ledger, target_ids=["hyp-unknown"])


if __name__ == "__main__":
    unittest.main()


def _append_plain(
    records: list[dict],
    run_id: str,
    parents: list[str],
    point: dict,
    *,
    status: str,
    score: float | None,
    dag_revision: int,
    warm: float | None = None,
) -> dict:
    record = {
        "run_id": run_id,
        "source_run_ids": parents,
        "semantic_point": point,
        "status": status,
        "final_best_score": score,
        "evaluation_depth": "screening",
        "dag_revision": dag_revision,
    }
    if warm is not None:
        record["best_warm_score"] = warm
    record["semantic_edges"] = build_semantic_edges(records, record)
    records.append(record)
    return record


class TestHypothesisCarriers(unittest.TestCase):
    def _registry_points(self):
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        return registry, baseline, filtered

    def test_two_independent_negative_contexts(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="discard", score=0.50, dag_revision=2)
        _append_plain(records, "002", [], baseline, status="keep", score=0.41, dag_revision=3)
        _append_plain(records, "003", ["002"], filtered, status="discard", score=0.52, dag_revision=4)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 2)
        self.assertEqual(result["positive"], 0)
        self.assertEqual(result["negative_contexts"], ["000", "002"])
        self.assertEqual(result["positive_contexts"], [])

    def test_crash_child_never_counts(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="crash", score=None, dag_revision=2)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)

    def test_positive_context_reported(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="keep", score=0.38, dag_revision=2)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 1)

    def test_mixed_context_counts_neither(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="discard", score=0.50, dag_revision=2)
        _append_plain(records, "002", ["000"], filtered, status="keep", score=0.38, dag_revision=3)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)

    def test_equal_scores_do_not_count(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="keep", score=0.40, dag_revision=2)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)

    def test_warm_scores_pair_with_warm_not_final(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1, warm=0.45)
        _append_plain(records, "001", ["000"], filtered, status="keep", score=0.50, dag_revision=2, warm=0.44)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 1)

    def test_missing_pair_does_not_count(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        child = _append_plain(records, "001", ["000"], filtered, status="keep", score=None, dag_revision=2, warm=0.60)
        del child["final_best_score"]
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)



class TestSchema7TransferRequirement(unittest.TestCase):
    def test_schema7_nonfresh_terminal_requires_parameter_transfer(self):
        record = {
            "run_id": "001",
            "source_run_ids": ["000"],
            "op": "improve",
            "status": "keep",
            "final_best_score": 0.5,
            "policy_receipt": {"schema_version": 7},
        }
        errors = validate_parameter_transfer_evidence(record)
        self.assertTrue(
            any("parameter_transfer is required" in error for error in errors),
            errors,
        )

    def test_schema7_fresh_still_forbids_parameter_transfer(self):
        record = {
            "run_id": "000",
            "source_run_ids": [],
            "op": "fresh",
            "status": "keep",
            "final_best_score": 0.5,
            "policy_receipt": {"schema_version": 7},
        }
        self.assertEqual(validate_parameter_transfer_evidence(record), [])
