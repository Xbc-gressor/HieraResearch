from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import got_select  # noqa: E402
import semantic_search  # noqa: E402
import slate  # noqa: E402
from background_contract import ContractError  # noqa: E402
from ledger_core import records_prefix_digest  # noqa: E402
from semantic_space import complete_point  # noqa: E402
from tests.fixtures import background_text, fixture_registry, record  # noqa: E402


def _lane(lane_id: str, op: str, parents: list, value) -> dict:
    return {"lane_id": lane_id, "op": op, "parents": parents, "lane_value": value}


def _proposal(point: dict, coverage: float, parent_diffs=None, deprioritized=()) -> dict:
    return {
        "point_id": point["point_id"],
        "point": point,
        "coverage": coverage,
        "parent_diffs": parent_diffs or [],
        "deprioritized_hypotheses": sorted(deprioritized),
        "budget_lane": "deprioritized" if deprioritized else "active",
    }


def _pset(revision: str, proposals: list) -> dict:
    return {"proposal_set_revision": revision, "proposals": proposals}


def _rec(
    run_id: str,
    *,
    op: str = "fresh",
    parents: tuple = (),
    warm: float | None = 0.5,
    status: str = "keep",
    idea: str | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "op": op,
        "source_run_ids": list(parents),
        "status": status,
        "best_warm_score": warm,
        "idea": idea if idea is not None else f"idea {run_id}",
    }


def _pool_doc(registry, entries, **overrides) -> dict:
    """Minimal hand-built pool artifact carrying what the consumers read."""
    pool = []
    for rank, (point, coverage, carrier, alternatives) in enumerate(entries, start=1):
        pool.append(
            {
                "label": f"C{rank}",
                "coverage_rank": rank,
                "point_id": point["point_id"],
                "point": point,
                "coverage": coverage,
                "deprioritized_hypotheses": [],
                "carrier": carrier,
                "carrier_alternatives": alternatives,
                "summary": slate.bare_point_summary(point, carrier, [], registry),
            }
        )
    doc = {
        "schema_version": 1,
        "gen_no": 1,
        "ledger_snapshot": {"record_count": 0},
        "budget": {"objective_remaining": None, "admission_cap": None},
        "pool_size": 6,
        "lanes": [],
        "lanes_digest": "sha256:" + "0" * 64,
        "proposal_set_revisions": {},
        "lanes_without_proposals": [],
        "pool": pool,
    }
    doc.update(overrides)
    doc["pool_digest"] = slate.recompute_pool_digest(doc)
    doc["generation_seed"] = slate.recompute_generation_seed(doc)
    return doc


class PoolCarrierTests(unittest.TestCase):
    def setUp(self):
        self.registry = fixture_registry()
        self.points = {
            name: complete_point(self.registry, overrides)
            for name, overrides in {
                "base": None,
                "filtered": {"dim-data-curation": "hyp-data-filtered"},
                "multi": {"dim-model-architecture": "hyp-model-multibranch"},
                "cv": {"dim-validation-selection": "hyp-valid-cv"},
                "stacked": {
                    "dim-model-architecture": "hyp-model-multibranch",
                    "dim-ensemble": "hyp-ensemble-stacking",
                    "dim-validation-selection": "hyp-valid-cv",
                },
            }.items()
        }

    def _points(self, *names):
        return [self.points[name] for name in names]

    def test_lane_representative_guarantee(self):
        p_a, p_b1, p_b2, p_b3 = self._points("base", "filtered", "cv", "multi")
        lanes = [
            _lane("lane-00", "improve", ["003"], 0.4),
            _lane("lane-01", "improve", ["004"], 0.6),
        ]
        sets = {
            "lane-00": _pset("sha256:" + "a" * 64, [_proposal(p_a, 0.85)]),
            "lane-01": _pset("sha256:" + "b" * 64, [
                _proposal(p_b1, 0.95),
                _proposal(p_b2, 0.93),
                _proposal(p_b3, 0.91),
            ]),
        }
        result = slate.build_pool(lanes, sets, self.registry, 6)
        self.assertEqual(len(result["pool"]), 4)

        result = slate.build_pool(lanes, sets, self.registry, 3)
        ids = [entry["point_id"] for entry in result["pool"]]
        # Lane-00's nominee is seated even though p_b3 outranks it; the final
        # numbering still follows coverage order.
        self.assertEqual(ids, [p_b1["point_id"], p_b2["point_id"], p_a["point_id"]])
        self.assertEqual([entry["label"] for entry in result["pool"]], ["C1", "C2", "C3"])

    def test_duplicate_point_occupies_one_seat_for_both_lanes(self):
        p1, p2 = self._points("base", "filtered")
        lanes = [
            _lane("lane-00", "improve", ["003"], 0.4),
            _lane("lane-01", "crossover", ["003", "004"], 0.7),
        ]
        sets = {
            "lane-00": _pset("sha256:" + "a" * 64, [_proposal(p1, 0.9)]),
            "lane-01": _pset("sha256:" + "b" * 64, [
                _proposal(p1, 0.9),
                _proposal(p2, 0.2),
            ]),
        }
        result = slate.build_pool(lanes, sets, self.registry, 6)
        self.assertEqual(len(result["pool"]), 2)
        first = result["pool"][0]
        self.assertEqual(first["point_id"], p1["point_id"])
        # The higher-valued crossover lane carries the shared point.
        self.assertEqual(first["carrier"]["op"], "crossover")
        self.assertEqual(len(first["carrier_alternatives"]), 1)
        self.assertEqual(
            first["carrier_alternatives"][0]["lane_id"], "lane-00"
        )
        self.assertEqual(
            result["proposal_set_revisions"],
            {"lane-00": "sha256:" + "a" * 64, "lane-01": "sha256:" + "b" * 64},
        )

    def test_coverage_fill_and_deprioritized_ordering(self):
        p_dep, p_act, p_low = self._points("filtered", "cv", "base")
        lanes = [_lane("lane-00", "fresh", [], None)]
        sets = {
            "lane-00": _pset("sha256:" + "a" * 64, [
                _proposal(p_low, 0.4),
                _proposal(p_dep, 0.99, deprioritized=("hyp-data-filtered",)),
                _proposal(p_act, 0.99),
            ]),
        }
        result = slate.build_pool(lanes, sets, self.registry, 3)
        ids = [entry["point_id"] for entry in result["pool"]]
        # Equal coverage: the deprioritized point ranks behind the active one.
        self.assertEqual(ids, [p_act["point_id"], p_dep["point_id"], p_low["point_id"]])
        self.assertEqual(
            result["pool"][1]["deprioritized_hypotheses"], ["hyp-data-filtered"]
        )

    def test_carrier_tie_breaks(self):
        (point,) = self._points("filtered")
        diff = [{"parent_run_id": "003", "changes": []}]
        lanes = [
            _lane("lane-00", "fresh", [], None),
            _lane("lane-01", "crossover", ["003", "004"], 0.5),
            _lane("lane-02", "improve", ["005"], 0.5),
            _lane("lane-03", "improve", ["003"], 0.5),
        ]
        sets = {
            lane["lane_id"]: _pset(
                "sha256:" + "9" * 64,
                [_proposal(point, 0.9, parent_diffs=diff if lane["op"] != "fresh" else [])],
            )
            for lane in lanes
        }
        entry = slate.build_pool(lanes, sets, self.registry, 6)["pool"][0]
        # value tie → improve > crossover > fresh; parent tuple 003 < 005.
        self.assertEqual(entry["carrier"]["lane_id"], "lane-03")
        self.assertEqual(
            [c["lane_id"] for c in entry["carrier_alternatives"]],
            ["lane-02", "lane-01", "lane-00"],
        )

        # A fresh carrier (lane_value null = -inf) loses to any valued lane.
        lanes = [_lane("lane-00", "fresh", [], None), _lane("lane-01", "improve", ["003"], 0.0)]
        sets = {
            "lane-00": _pset("sha256:" + "8" * 64, [_proposal(point, 0.9)]),
            "lane-01": _pset("sha256:" + "9" * 64, [_proposal(point, 0.9, parent_diffs=diff)]),
        }
        entry = slate.build_pool(lanes, sets, self.registry, 6)["pool"][0]
        self.assertEqual(entry["carrier"]["op"], "improve")

    def test_empty_pool_when_no_lane_proposes(self):
        lanes = [_lane("lane-00", "fresh", [], None)]
        result = slate.build_pool(lanes, {}, self.registry, 6)
        self.assertEqual(result["pool"], [])
        self.assertEqual(result["lanes_without_proposals"], ["lane-00"])


class A1ContextTests(unittest.TestCase):
    def _pool(self, *carrier_parents):
        return {
            "pool": [
                {
                    "label": f"C{index + 1}",
                    "coverage_rank": index + 1,
                    "carrier": {
                        "op": "improve" if len(parents) == 1 else "crossover",
                        "parents": list(parents),
                    },
                }
                for index, parents in enumerate(carrier_parents)
            ]
        }

    def test_parent_ancestor_sibling_anchor_rows(self):
        records = [
            _rec("000", warm=0.50),
            _rec("001", op="improve", parents=("000",), warm=0.45),
            _rec("002", op="improve", parents=("001",), warm=0.40),
            _rec("003", op="improve", parents=("002",), warm=0.60),
            _rec("004", op="improve", parents=("002",), warm=0.55),
            _rec("005", warm=0.90),
        ]
        context = slate.build_a1_context({"records": records}, self._pool(("002",)))
        self.assertEqual(
            [(row["run_id"], row["role"]) for row in context["rows"]],
            [
                ("002", "parent"),
                ("001", "ancestor"),
                ("000", "ancestor"),
                ("004", "sibling"),
                ("003", "sibling"),
                ("005", "anchor_high"),
            ],
        )
        parent_row = context["rows"][0]
        self.assertEqual(parent_row["delta"], 0.40 - 0.45)
        self.assertIn("d=-0.050000", parent_row["line"])
        self.assertIn("run 002 [improve<-001]", parent_row["line"])
        # 004 (sibling, warm 0.55) and 000 are deduplicated out of the anchors.
        self.assertEqual(context["prefix_record_count"], 6)
        self.assertEqual(
            context["prefix_digest"], records_prefix_digest(records)
        )

    def test_multi_parent_round_robin(self):
        records = [
            _rec("000", warm=0.70),
            _rec("001", op="improve", parents=("000",), warm=0.60),
            _rec("002", op="improve", parents=("001",), warm=0.50),
            _rec("007", warm=0.71),
            _rec("008", op="improve", parents=("007",), warm=0.61),
            _rec("009", op="improve", parents=("008",), warm=0.51),
            _rec("010", op="improve", parents=("009",), warm=0.41),
            _rec("011", op="improve", parents=("002",), warm=0.81),
            _rec("012", op="improve", parents=("002",), warm=0.82),
            _rec("013", op="improve", parents=("010",), warm=0.83),
        ]
        pool = self._pool(("002",), ("010", "000"))
        context = slate.build_a1_context({"records": records}, pool)
        roles = [(row["run_id"], row["role"]) for row in context["rows"]]
        # Direct parents follow pool coverage rank, then carrier parent order.
        self.assertEqual(
            roles[:3],
            [("002", "parent"), ("010", "parent"), ("000", "parent")],
        )
        # Ancestors: chain(002)=[001,000], chain(010)=[009,008,007],
        # chain(000)=[]; 000 is already a direct parent, so its ancestor
        # occurrence is skipped and the round-robin continues.
        self.assertEqual(
            roles[3:7],
            [
                ("001", "ancestor"),
                ("009", "ancestor"),
                ("008", "ancestor"),
                ("007", "ancestor"),
            ],
        )
        # Siblings of 002: newest first 012, 011; siblings of 010: 013.
        self.assertEqual(
            roles[7:10],
            [("012", "sibling"), ("013", "sibling"), ("011", "sibling")],
        )

    def test_rows_never_exceed_the_hard_cap(self):
        records = []
        # 12 distinct carrier parents, each with a 3-deep private chain and
        # 4 private children; plus 10 anchor-only records with extreme scores.
        for index in range(12):
            parent = 100 + index
            chain = [200 + 3 * index + level for level in range(3)]
            records.append(
                _rec(str(chain[2]), warm=0.5)
            )
            records.append(
                _rec(str(chain[1]), op="improve", parents=(str(chain[2]),), warm=0.5)
            )
            records.append(
                _rec(str(chain[0]), op="improve", parents=(str(chain[1]),), warm=0.5)
            )
            records.append(
                _rec(str(parent), op="improve", parents=(str(chain[0]),), warm=0.5)
            )
            for child in range(4):
                records.append(
                    _rec(
                        str(400 + 4 * index + child),
                        op="improve",
                        parents=(str(parent),),
                        warm=0.5,
                    )
                )
        for index in range(5):
            records.append(_rec(str(500 + index), warm=0.01 + index * 0.001))
            records.append(_rec(str(600 + index), warm=9.0 + index * 0.1))
        parents = [str(100 + index) for index in range(12)]
        pool = self._pool(*[(parents[i], parents[i + 1]) for i in range(0, 12, 2)])
        context = slate.build_a1_context({"records": records}, pool)
        self.assertEqual(len(context["rows"]), 46)
        roles = {}
        for row in context["rows"]:
            roles[row["role"]] = roles.get(row["role"], 0) + 1
        self.assertEqual(
            roles,
            {"parent": 12, "ancestor": 12, "sibling": 12, "anchor_low": 5, "anchor_high": 5},
        )

    def test_rows_ignore_final_scores_and_later_tuning_state(self):
        records = [
            dict(
                _rec("000", warm=0.50),
                final_best_score=0.40,
                evaluation_depth="tuned",
            ),
            dict(
                _rec("001", op="improve", parents=("000",), warm=0.45),
                final_best_score=0.44,
                evaluation_depth="screening",
            ),
        ]
        pool = self._pool(("001",))
        baseline = slate.build_a1_context({"records": records}, pool)
        mutated = [dict(record) for record in records]
        mutated[0]["final_best_score"] = 0.01
        mutated[0]["tuning_bouts"] = 9
        mutated[1]["final_best_score"] = 0.99
        changed = slate.build_a1_context({"records": mutated}, pool)
        self.assertEqual(baseline["rows"], changed["rows"])
        self.assertEqual(baseline["rendered_text"], changed["rendered_text"])
        self.assertEqual(baseline["prefix_digest"], changed["prefix_digest"])
        # ... while a selection-safe field does move the digest.
        warmed = [dict(record) for record in records]
        warmed[0]["best_warm_score"] = 0.51
        self.assertNotEqual(
            baseline["prefix_digest"],
            slate.build_a1_context({"records": warmed}, pool)["prefix_digest"],
        )

    def test_idea_excerpt_is_single_line_and_bounded(self):
        idea = "  first\n\nsecond   third " + "x" * 200
        records = [_rec("000", warm=0.5, idea=idea)]
        context = slate.build_a1_context({"records": records}, self._pool(("000",)))
        excerpt = context["rows"][0]["idea"]
        self.assertEqual(excerpt, " ".join(idea.split())[:140])
        self.assertEqual(len(excerpt), 140)
        self.assertNotIn("\n", context["rendered_text"].split("idea: ")[1])

    def test_crash_and_missing_warm_rendering(self):
        records = [
            _rec("000", warm=None, status="crash"),
            _rec("001", warm=None),
        ]
        context = slate.build_a1_context(
            {"records": records}, self._pool(("000",), ("001",))
        )
        lines = context["rendered_text"].splitlines()
        self.assertIn("run 000 [fresh] warm=crash", lines[1])
        self.assertIn("run 001 [fresh] warm=n/a", lines[2])


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.pool = {
            "pool": [
                {"label": f"C{rank}", "coverage_rank": rank, "point_id": f"p{rank}"}
                for rank in range(1, 5)
            ]
        }

    def _judgment(self, stage, ranking, status="valid"):
        return {"stage": stage, "status": status, "ranking": ranking}

    def test_consensus_orders_by_mean_presented_rank(self):
        # C2 outranks C1 on mean presented rank despite coverage order.
        decision = slate.aggregate_regular(
            self.pool,
            [
                self._judgment("regular-0", ["C2", "C1", "C3", "C4"]),
                self._judgment("regular-1", ["C2", "C1", "C4", "C3"]),
            ],
        )
        self.assertEqual(decision["status"], "selected")
        self.assertEqual(decision["path"], "consensus")
        self.assertEqual(decision["slate"], ["C2", "C1"])

    def test_mean_rank_tie_falls_back_to_coverage(self):
        decision = slate.aggregate_regular(
            self.pool,
            [
                self._judgment("regular-0", ["C2", "C1", "C3", "C4"]),
                self._judgment("regular-1", ["C1", "C2", "C4", "C3"]),
            ],
        )
        self.assertEqual(decision["slate"], ["C1", "C2"])

    def test_disagreement_requires_boundary_and_boundary_decides(self):
        decision = slate.aggregate_regular(
            self.pool,
            [
                self._judgment("regular-0", ["C1", "C2", "C3", "C4"]),
                self._judgment("regular-1", ["C3", "C1", "C4", "C2"]),
            ],
        )
        self.assertEqual(decision["status"], "boundary_required")
        # Union in coverage order, so the boundary shuffle has a canonical input.
        self.assertEqual(decision["boundary_labels"], ["C1", "C2", "C3"])

        boundary = slate.aggregate_boundary(
            self.pool, self._judgment("boundary", ["C3", "C2", "C1"])
        )
        self.assertEqual(boundary["status"], "selected")
        self.assertEqual(boundary["path"], "boundary")
        self.assertEqual(boundary["slate"], ["C3", "C2"])

        failed = slate.aggregate_boundary(
            self.pool, self._judgment("boundary", None, status="failed")
        )
        self.assertEqual(failed["status"], "fallback")
        self.assertEqual(failed["slate"], ["C1", "C2"])

    def test_invalid_permutations_are_rejected(self):
        labels = ["C1", "C2", "C3"]
        self.assertEqual(
            slate.validate_judge_ranking(labels, {"ranking": ["C2", "C3", "C1"]}), []
        )
        for ranking in (
            ["C1", "C1", "C2"],
            ["C1", "C2"],
            ["C1", "C2", "C9"],
            "C1,C2,C3",
        ):
            with self.subTest(ranking=ranking):
                self.assertTrue(slate.validate_judge_ranking(labels, {"ranking": ranking}))
        self.assertTrue(slate.validate_judge_ranking(labels, {"no_ranking": []}))
        self.assertTrue(slate.validate_judge_ranking(labels, None))

    def test_regular_failure_falls_back_to_coverage(self):
        decision = slate.aggregate_regular(
            self.pool,
            [
                self._judgment("regular-0", ["C4", "C3", "C2", "C1"]),
                self._judgment("regular-1", None, status="failed"),
            ],
        )
        self.assertEqual(decision["status"], "fallback")
        self.assertEqual(decision["path"], "coverage_fallback")
        self.assertEqual(decision["slate"], ["C1", "C2"])
        self.assertIn("regular-1", decision["reason"])

    def test_presented_order_is_stable_per_stage(self):
        pool_doc = _pool_doc(fixture_registry(), [])
        seed = "sha256:" + "1" * 64
        first = slate.presented_order(self.pool, seed, "regular-0")
        self.assertEqual(first, slate.presented_order(self.pool, seed, "regular-0"))
        self.assertEqual(sorted(first), ["C1", "C2", "C3", "C4"])
        other = slate.presented_order(self.pool, seed, "regular-1")
        self.assertEqual(sorted(other), ["C1", "C2", "C3", "C4"])
        # Boundary stages shuffle exactly the union subset.
        subset = slate.presented_order(self.pool, seed, "boundary", labels=["C2", "C4"])
        self.assertEqual(sorted(subset), ["C2", "C4"])
        del pool_doc


class ManifestTests(unittest.TestCase):
    def _docs(self):
        registry = fixture_registry()
        point = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        carrier = {
            "lane_id": "lane-00",
            "op": "improve",
            "parents": ["003"],
            "parent_diffs": [],
            "lane_value": 0.7,
            "proposal_set_revision": "sha256:" + "a" * 64,
        }
        point2 = complete_point(registry)
        carrier2 = {
            "lane_id": "lane-01",
            "op": "crossover",
            "parents": ["003", "004"],
            "parent_diffs": [],
            "lane_value": 0.6,
            "proposal_set_revision": "sha256:" + "b" * 64,
        }
        pool = _pool_doc(registry, [(point, 0.9, carrier, []), (point2, 0.8, carrier2, [])])
        context = {"schema_version": 1, "gen_no": 1, "rows": [], "rendered_text": "x"}
        judge = {
            "schema_version": 1,
            "gen_no": 1,
            "generation_seed": pool["generation_seed"],
            "pool_digest": pool["pool_digest"],
            "context_digest": slate.digest(context),
            "aggregation": {"path": "consensus", "reason": None, "slate": ["C1", "C2"]},
            "stages": {},
            "judge_cost": {"session_ids": {}, "models": {}},
        }
        manifest = slate.build_manifest(pool, judge, pool["budget"], ["013", "014"])
        return pool, context, judge, manifest

    def test_manifest_recomputes_clean(self):
        pool, context, judge, manifest = self._docs()
        self.assertEqual(slate.verify_manifest(manifest, pool, context, judge), [])
        self.assertEqual(
            manifest["generation_id"], slate.recompute_generation_id(manifest)
        )
        self.assertEqual(
            manifest["slate"][0]["candidate_id"],
            slate.candidate_id(manifest["slate"][0]["point_id"], "improve", ["003"]),
        )
        self.assertEqual(
            manifest["policy"]["config"],
            {"pool_size": 6, "slate_size": 2, "regular_rollouts": 2},
        )
        self.assertEqual(manifest["cardinality"]["pool_actual"], 2)
        self.assertEqual([s["run_id"] for s in manifest["slate"]], ["013", "014"])

    def test_tampering_breaks_the_digest_chain(self):
        pool, context, judge, manifest = self._docs()

        tampered_pool = json.loads(json.dumps(pool))
        tampered_pool["pool"][0]["point"]["point_id"] = "point-forged"
        self.assertNotEqual(
            slate.recompute_pool_digest(tampered_pool), pool["pool_digest"]
        )

        tampered_context = dict(context, rendered_text="forged history")
        self.assertNotEqual(slate.digest(tampered_context), judge["context_digest"])

        tampered_judge = json.loads(json.dumps(judge))
        tampered_judge["aggregation"]["slate"] = ["C2", "C1"]
        self.assertNotEqual(slate.digest(tampered_judge), manifest["judge_digest"])

        tampered_manifest = json.loads(json.dumps(manifest))
        tampered_manifest["slate"][0]["run_id"] = "099"
        self.assertNotEqual(
            slate.recompute_generation_id(tampered_manifest), manifest["generation_id"]
        )
        errors = slate.verify_manifest(tampered_manifest, pool, context, judge)
        self.assertTrue(errors)

    def test_slot_count_must_match_reserved_ids(self):
        pool, context, judge, manifest = self._docs()
        del context, manifest
        with self.assertRaisesRegex(ContractError, "reserved run ids"):
            slate.build_manifest(pool, judge, pool["budget"], ["013"])


class LanesModeTests(unittest.TestCase):
    def _decide(self, run_dir: Path, *, mode, cfg=None, output=None):
        args = SimpleNamespace(
            ledger=str(run_dir / "ledger.json"), cfg=cfg, mode=mode, output=output
        )
        out = io.StringIO()
        with redirect_stdout(out):
            got_select.cmd_decide(args)
        return json.loads(out.getvalue())

    def _write_records(self, run_dir: Path, records: list) -> None:
        (run_dir / "ledger.json").write_text(json.dumps({"records": records}))

    def test_stall_merges_homogeneous_fresh_actions_into_one_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._write_records(
                run_dir,
                [
                    _rec("000", warm=0.4) | {"final_best_score": 0.4},
                    _rec("001", op="improve", parents=("000",), warm=0.5)
                    | {"final_best_score": 0.5},
                    _rec("002", op="improve", parents=("000",), warm=0.6)
                    | {"final_best_score": 0.6},
                ],
            )
            doc = self._decide(run_dir, mode="lanes", cfg='{"n_seed":1,"S":2,"B":2}')
            self.assertEqual(doc["schema_version"], 1)
            self.assertEqual(
                doc["lanes"],
                [{"lane_id": "lane-00", "op": "fresh", "parents": [], "lane_value": None}],
            )
            self.assertEqual(doc["gen_no"], 1)
            snapshot = doc["ledger_snapshot"]
            self.assertEqual(snapshot["record_count"], 3)
            self.assertEqual(snapshot["dag_revision"], 0)
            self.assertEqual(
                snapshot["records_digest"],
                records_prefix_digest(
                    json.loads((run_dir / "ledger.json").read_text())["records"]
                ),
            )
            self.assertEqual(
                snapshot["experience"],
                {"generation": None, "updated_at_run": None, "revision": None},
            )

    def test_pucb_lanes_carry_q_and_ignore_the_admission_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            records = [
                _rec(f"00{index}", warm=0.4 + index * 0.05)
                | {"final_best_score": 0.4 + index * 0.05}
                for index in range(5)
            ]
            self._write_records(run_dir, records)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 3})
            )
            (run_dir / "evaluation_attempts.jsonl").write_text(
                "\n".join(json.dumps({"schema_version": 1, "kind": "score_attempt"}) for _ in range(2))
                + "\n"
            )
            default = self._decide(run_dir, mode="actions")
            # Legacy behavior: the cap truncates actions, here to nothing.
            self.assertEqual(default["actions"], [])
            self.assertEqual(default["diag"]["candidate_admission_cap"], 0)
            self.assertEqual(set(default), {"kind", "actions", "diag"})

            out_path = run_dir / ".semantic" / "gen-0001" / "lanes.json"
            doc = self._decide(run_dir, mode="lanes", output=str(out_path))
            self.assertEqual(doc["budget"]["admission_cap"], 0)
            self.assertEqual(doc["budget"]["objective_remaining"], 1)
            self.assertEqual(doc["budget"]["candidate_objective_reservation"], 3)
            self.assertTrue(doc["lanes"])
            for lane in doc["lanes"]:
                self.assertIsInstance(lane["lane_value"], float)
                self.assertNotEqual(lane["op"], "fresh")
            keys = {(lane["op"], tuple(lane["parents"])) for lane in doc["lanes"]}
            self.assertEqual(len(keys), len(doc["lanes"]))
            self.assertEqual(json.loads(out_path.read_text()), doc)

    def test_lanes_budget_keeps_tournament_reserve_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
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
            doc = self._decide(run_dir, mode="lanes")
            self.assertEqual(doc["budget"]["tournament_generation_reserve"], 36)
            self.assertEqual(doc["budget"]["admission_cap"], 32)
            self.assertEqual(len(doc["lanes"]), 1)

    def test_records_digest_is_selection_safe(self):
        records = [
            dict(_rec("000", warm=0.4), final_best_score=0.3),
            dict(_rec("001", op="improve", parents=("000",), warm=0.5), final_best_score=0.45),
        ]
        digest = records_prefix_digest(records)
        mutated = [dict(record) for record in records]
        mutated[0]["final_best_score"] = 0.01
        mutated[1]["tuning_bouts"] = 4
        self.assertEqual(records_prefix_digest(mutated), digest)
        warmed = [dict(record) for record in records]
        warmed[1]["best_warm_score"] = 0.55
        self.assertNotEqual(records_prefix_digest(warmed), digest)


class CliPipelineTests(unittest.TestCase):
    """The Patch-A completion gate: full replay from saved artifacts only."""

    def _setup_run(self, run_dir: Path) -> dict:
        registry = fixture_registry()
        (run_dir / "background.md").write_text(background_text(registry))
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        base = complete_point(registry)
        records = []
        scores = [0.40, 0.50, 0.41, 0.52, 0.39]
        for index, score in enumerate(scores):
            point = filtered if index % 2 else base
            entry = record(
                f"00{index}", "fresh", [], point, score=score, status="keep"
            )
            entry["best_warm_score"] = score + 0.05
            records.append(entry)
        (run_dir / "ledger.json").write_text(json.dumps({"records": records}))
        return registry

    def _run_lanes_and_construct(self, run_dir: Path, gen: Path) -> dict:
        lanes_path = gen / "lanes.json"
        with redirect_stdout(io.StringIO()):
            got_select.cmd_decide(
                SimpleNamespace(
                    ledger=str(run_dir / "ledger.json"),
                    cfg=None,
                    mode="lanes",
                    output=str(lanes_path),
                )
            )
        lanes_doc = json.loads(lanes_path.read_text())
        proposals_dir = gen / "proposals"
        proposals_dir.mkdir(parents=True, exist_ok=True)
        for lane in lanes_doc["lanes"]:
            with redirect_stdout(io.StringIO()):
                semantic_search.cmd_propose(
                    SimpleNamespace(
                        background=run_dir / "background.md",
                        ledger=run_dir / "ledger.json",
                        op=lane["op"],
                        parents=",".join(lane["parents"]),
                        max_points=128,
                        baseline_only=False,
                        output=proposals_dir / f"{lane['lane_id']}.json",
                    )
                )
        out = io.StringIO()
        with redirect_stdout(out):
            slate.cmd_construct(
                SimpleNamespace(
                    lanes=lanes_path,
                    proposals_dir=proposals_dir,
                    ledger=run_dir / "ledger.json",
                    background=run_dir / "background.md",
                    pool_size=6,
                    pool_output=gen / "pool.json",
                    context_output=gen / "context.json",
                )
            )
        status = json.loads(out.getvalue())
        self.assertTrue(status["ok"])
        self.assertGreaterEqual(status["pool_size"], 3)
        return lanes_doc

    def _judge(self, gen: Path, rankings: dict) -> dict:
        judgments = gen / "judgments"
        judgments.mkdir(exist_ok=True)
        for stage, top2 in rankings.items():
            with redirect_stdout(io.StringIO()):
                slate.cmd_prepare_judge(
                    SimpleNamespace(
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        stage=stage,
                        labels=None,
                        task_brief=None,
                        output=judgments / f"{stage}.input.json",
                    )
                )
            input_doc = json.loads((judgments / f"{stage}.input.json").read_text())
            order = input_doc["presented_order"]
            ranking = [label for label in order if label in top2] + [
                label for label in order if label not in top2
            ]
            receipt = judgments / f"{stage}.receipt.json"
            receipt.write_text(json.dumps({"ranking": ranking, "rationale": "t"}))
            with redirect_stdout(io.StringIO()) as out:
                code = slate.cmd_validate_judge(
                    SimpleNamespace(
                        input=judgments / f"{stage}.input.json",
                        receipt=receipt,
                        session_id=f"sess-{stage}",
                        model="grok/grok-4.6",
                        output=judgments / f"{stage}.json",
                    )
                )
            self.assertEqual(code, 0, out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            slate.cmd_aggregate(
                SimpleNamespace(
                    pool=gen / "pool.json",
                    context=gen / "context.json",
                    judgments_dir=judgments,
                    output=gen / "judge.json",
                )
            )
        return json.loads(out.getvalue())

    def test_consensus_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._setup_run(run_dir)
            gen = run_dir / ".semantic" / "gen-0001"
            lanes_doc = self._run_lanes_and_construct(run_dir, gen)
            self.assertTrue(lanes_doc["lanes"])

            pool = json.loads((gen / "pool.json").read_text())
            context = json.loads((gen / "context.json").read_text())
            self.assertTrue(any(row["role"] == "parent" for row in context["rows"]))
            self.assertEqual(
                context["prefix_digest"],
                lanes_doc["ledger_snapshot"]["records_digest"],
            )

            status = self._judge(gen, {"regular-0": ["C1", "C3"], "regular-1": ["C3", "C1"]})
            self.assertEqual(status["status"], "selected")
            self.assertEqual(status["path"], "consensus")
            # Mean presented ranks disagree with coverage here only if the
            # orders differ; the slate set is the agreed top-2 either way.
            self.assertEqual(sorted(status["slate"]), ["C1", "C3"])

            with redirect_stdout(io.StringIO()):
                slate.cmd_build_manifest(
                    SimpleNamespace(
                        lanes=gen / "lanes.json",
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        judge=gen / "judge.json",
                        reserved_run_ids="005,006",
                        output=gen / "generation.json",
                    )
                )
            manifest = json.loads((gen / "generation.json").read_text())
            judge = json.loads((gen / "judge.json").read_text())
            self.assertEqual(slate.verify_manifest(manifest, pool, context, judge), [])
            self.assertEqual(
                [slot["run_id"] for slot in manifest["slate"]], ["005", "006"]
            )
            self.assertEqual(
                manifest["ledger_snapshot"]["records_digest"],
                lanes_doc["ledger_snapshot"]["records_digest"],
            )
            # A committed manifest consumes the generation number.
            self.assertEqual(got_select._generation_number(run_dir), 2)

    def test_boundary_pipeline_two_phase_aggregate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._setup_run(run_dir)
            gen = run_dir / ".semantic" / "gen-0001"
            self._run_lanes_and_construct(run_dir, gen)
            status = self._judge(
                gen, {"regular-0": ["C1", "C2"], "regular-1": ["C1", "C3"]}
            )
            self.assertEqual(status["status"], "boundary_required")
            union = status["boundary_labels"]
            self.assertEqual(sorted(union), ["C1", "C2", "C3"])

            judgments = gen / "judgments"
            with redirect_stdout(io.StringIO()):
                slate.cmd_prepare_judge(
                    SimpleNamespace(
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        stage="boundary",
                        labels=",".join(union),
                        task_brief=None,
                        output=judgments / "boundary.input.json",
                    )
                )
            input_doc = json.loads((judgments / "boundary.input.json").read_text())
            self.assertEqual(sorted(input_doc["presented_order"]), sorted(union))
            ranking = list(input_doc["presented_order"])
            (judgments / "boundary.receipt.json").write_text(
                json.dumps({"ranking": ranking, "rationale": "t"})
            )
            with redirect_stdout(io.StringIO()):
                slate.cmd_validate_judge(
                    SimpleNamespace(
                        input=judgments / "boundary.input.json",
                        receipt=judgments / "boundary.receipt.json",
                        session_id="sess-boundary",
                        model=None,
                        output=judgments / "boundary.json",
                    )
                )
            out = io.StringIO()
            with redirect_stdout(out):
                slate.cmd_aggregate(
                    SimpleNamespace(
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        judgments_dir=judgments,
                        output=gen / "judge.json",
                    )
                )
            final = json.loads(out.getvalue())
            self.assertEqual(final["status"], "selected")
            self.assertEqual(final["path"], "boundary")
            self.assertEqual(final["slate"], ranking[:2])
            judge = json.loads((gen / "judge.json").read_text())
            self.assertEqual(
                set(judge["aggregation"]["regular_top2"]["regular-0"]), {"C1", "C2"}
            )
            self.assertIn("boundary", judge["stages"])

    def test_invalid_receipt_marks_the_stage_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._setup_run(run_dir)
            gen = run_dir / ".semantic" / "gen-0001"
            self._run_lanes_and_construct(run_dir, gen)
            judgments = gen / "judgments"
            judgments.mkdir(exist_ok=True)
            with redirect_stdout(io.StringIO()):
                slate.cmd_prepare_judge(
                    SimpleNamespace(
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        stage="regular-0",
                        labels=None,
                        task_brief=None,
                        output=judgments / "regular-0.input.json",
                    )
                )
            (judgments / "bad.receipt.json").write_text(
                json.dumps({"ranking": ["C1", "C1"], "rationale": "t"})
            )
            out = io.StringIO()
            with redirect_stdout(out):
                code = slate.cmd_validate_judge(
                    SimpleNamespace(
                        input=judgments / "regular-0.input.json",
                        receipt=judgments / "bad.receipt.json",
                        session_id=None,
                        model=None,
                        output=judgments / "regular-0.json",
                    )
                )
            self.assertEqual(code, 1)
            artifact = json.loads((judgments / "regular-0.json").read_text())
            self.assertEqual(artifact["status"], "failed")
            self.assertIsNone(artifact["ranking"])
            self.assertTrue(artifact["errors"])

    def test_construct_refuses_a_mutated_or_pending_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._setup_run(run_dir)
            gen = run_dir / ".semantic" / "gen-0001"
            self._run_lanes_and_construct(run_dir, gen)

            ledger = json.loads((run_dir / "ledger.json").read_text())
            ledger["records"].append(
                record("005", "fresh", [], complete_point(fixture_registry()), score=0.7, status="pending")
            )
            (run_dir / "ledger.json").write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ContractError, "grew or shrank"):
                slate.cmd_construct(
                    SimpleNamespace(
                        lanes=gen / "lanes.json",
                        proposals_dir=gen / "proposals",
                        ledger=run_dir / "ledger.json",
                        background=run_dir / "background.md",
                        pool_size=6,
                        pool_output=gen / "pool2.json",
                        context_output=gen / "context2.json",
                    )
                )

    def test_construct_blocks_a_pending_prefix_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            registry = self._setup_run(run_dir)
            ledger = json.loads((run_dir / "ledger.json").read_text())
            ledger["records"].append(
                record("005", "fresh", [], complete_point(registry), score=0.7, status="pending")
            )
            (run_dir / "ledger.json").write_text(json.dumps(ledger))
            # The snapshot honestly describes the pending prefix; the block is
            # the non-terminal status, not a digest mismatch.
            snapshot = {
                "record_count": len(ledger["records"]),
                "records_digest": records_prefix_digest(ledger["records"]),
                "dag_revision": 0,
                "search_space_state_revision": 0,
                "experience": {"generation": None, "updated_at_run": None, "revision": None},
            }
            gen = run_dir / ".semantic" / "gen-0001"
            gen.mkdir(parents=True)
            (gen / "lanes.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "gen_no": 1,
                        "ledger_snapshot": snapshot,
                        "budget": {"objective_remaining": None, "admission_cap": None},
                        "lanes": [
                            {"lane_id": "lane-00", "op": "fresh", "parents": [], "lane_value": None}
                        ],
                    }
                )
            )
            (gen / "proposals").mkdir()
            with self.assertRaisesRegex(ContractError, "non-terminal"):
                slate.cmd_construct(
                    SimpleNamespace(
                        lanes=gen / "lanes.json",
                        proposals_dir=gen / "proposals",
                        ledger=run_dir / "ledger.json",
                        background=run_dir / "background.md",
                        pool_size=6,
                        pool_output=gen / "pool.json",
                        context_output=gen / "context.json",
                    )
                )

    def test_degraded_empty_pool_needs_no_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            self._setup_run(run_dir)
            gen = run_dir / ".semantic" / "gen-0001"
            gen.mkdir(parents=True)
            lanes_path = gen / "lanes.json"
            with redirect_stdout(io.StringIO()):
                got_select.cmd_decide(
                    SimpleNamespace(
                        ledger=str(run_dir / "ledger.json"),
                        cfg=None,
                        mode="lanes",
                        output=str(lanes_path),
                    )
                )
            (gen / "proposals").mkdir()  # no lane produced a proposal
            with redirect_stdout(io.StringIO()):
                slate.cmd_construct(
                    SimpleNamespace(
                        lanes=lanes_path,
                        proposals_dir=gen / "proposals",
                        ledger=run_dir / "ledger.json",
                        background=run_dir / "background.md",
                        pool_size=6,
                        pool_output=gen / "pool.json",
                        context_output=gen / "context.json",
                    )
                )
            pool = json.loads((gen / "pool.json").read_text())
            self.assertEqual(pool["pool"], [])
            self.assertTrue(pool["lanes_without_proposals"])

            out = io.StringIO()
            with redirect_stdout(out):
                slate.cmd_aggregate(
                    SimpleNamespace(
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        judgments_dir=gen / "judgments",
                        output=gen / "judge.json",
                    )
                )
            status = json.loads(out.getvalue())
            self.assertEqual(status, {"status": "selected", "slate": [], "path": "degraded_empty_pool"})
            judge = json.loads((gen / "judge.json").read_text())
            self.assertFalse(judge["cardinality"]["judge_called"])

            with redirect_stdout(io.StringIO()):
                slate.cmd_build_manifest(
                    SimpleNamespace(
                        lanes=lanes_path,
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        judge=gen / "judge.json",
                        reserved_run_ids="",
                        output=gen / "generation.json",
                    )
                )
            manifest = json.loads((gen / "generation.json").read_text())
            self.assertEqual(manifest["slate"], [])
            self.assertEqual(manifest["cardinality"]["pool_actual"], 0)
            self.assertEqual(
                slate.verify_manifest(
                    manifest,
                    pool,
                    json.loads((gen / "context.json").read_text()),
                    judge,
                ),
                [],
            )


if __name__ == "__main__":
    unittest.main()
