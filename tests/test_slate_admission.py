"""Judged-slate atomic admission: schema-8 receipts and the two-seat transaction.

The manifest/pool/judge fixtures are built with the real ``tools/slate.py``
helpers, so every digest binding admission checks is the genuine one; the
per-seat plans are the fake stand-ins for the Patch-C slate-plan-writer.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import got_select  # noqa: E402
import ledger as ledger_cli  # noqa: E402
import ledger_admission  # noqa: E402
import semantic_search  # noqa: E402
import slate  # noqa: E402
from background_contract import validate_ledger  # noqa: E402
from ledger_admission import (  # noqa: E402
    AdmissionError,
    AdmissionRequest,
    SlateAdmissionRequest,
    admit_record,
    admit_slate_atomic,
)
from ledger_core import experience_receipt, records_prefix_digest  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402
from semantic_space import complete_point, digest, space_receipt  # noqa: E402
from tests.fixtures import (  # noqa: E402
    background_text,
    fixture_registry,
    policy_receipt,
    record as fixture_record,
)


def _experience(dag_revision: int) -> dict:
    return {
        "schema_version": 3,
        "updated_at_run": "004",
        "generation": 1,
        "summary": "",
        "promising_regions": [],
        "lessons": [],
        "bottlenecks": [],
        "dimension_evidence": [],
        "hypothesis_evidence": [],
        "dag_revision": dag_revision,
    }


def _ledger_data(registry: dict) -> dict:
    """Five terminal fresh records; the judged generation starts above them."""
    base = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = []
    scores = [0.40, 0.50, 0.41, 0.52, 0.39]
    for index, score in enumerate(scores):
        point = filtered if index % 2 else base
        entry = fixture_record(
            f"00{index}", "fresh", [], point, score=score, status="keep"
        )
        entry["best_warm_score"] = score + 0.05
        records.append(entry)
    return {
        "task": "hard-interactions",
        "tag": "slate-admission-test",
        "metric": "validation-loss",
        "records": records,
        "items": {},
        "lineage_snapshots": [],
        "dag_revision": 5,
        "search_space": space_receipt(registry),
        "search_space_state": empty_search_space_state(),
        "experience": _experience(5),
    }


def _carrier(lane_id: str, op: str, parents: list, value, revision: str) -> dict:
    return {
        "lane_id": lane_id,
        "op": op,
        "parents": parents,
        "parent_diffs": [],
        "lane_value": value,
        "proposal_set_revision": revision,
    }


def _stage_artifact(stage: str, pool_doc: dict, context_digest: str, ranking: list) -> dict:
    return {
        "schema_version": 1,
        "stage": stage,
        "gen_no": pool_doc["gen_no"],
        "generation_seed": pool_doc["generation_seed"],
        "pool_digest": pool_doc["pool_digest"],
        "context_digest": context_digest,
        "presented_order": slate.presented_order(
            pool_doc, pool_doc["generation_seed"], stage
        ),
        "status": "valid",
        "ranking": list(ranking),
        "rationale": "fixture judge",
        "errors": [],
        "receipt_path": None,
        "session_id": f"sess-{stage}",
        "model": "fixture",
    }


class SlateAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)
        self.registry = fixture_registry()
        (self.run_dir / "background.md").write_text(background_text(self.registry))
        self.data = _ledger_data(self.registry)
        self.gen = self.run_dir / ".semantic" / "gen-0001"
        self.manifest = self._write_generation()
        self._write_plans()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # ---------- generation fixture ----------

    def _write_generation(self) -> dict:
        """A judged consensus generation whose digests all recompute."""
        snapshot = {
            "record_count": len(self.data["records"]),
            "records_digest": records_prefix_digest(self.data["records"]),
            "dag_revision": self.data["dag_revision"],
            "search_space_state_revision": 0,
            "experience": experience_receipt(self.data),
        }
        revisions = {
            f"lane-0{index}": f"sha256:{chr(ord('a') + index) * 64}"
            for index in range(3)
        }
        points = [
            complete_point(
                self.registry, {"dim-model-architecture": "hyp-model-multibranch"}
            ),
            complete_point(
                self.registry, {"dim-validation-selection": "hyp-valid-cv"}
            ),
            complete_point(
                self.registry, {"dim-data-curation": "hyp-data-filtered"}
            ),
        ]
        carriers = [
            _carrier("lane-00", "improve", ["004"], 0.7, revisions["lane-00"]),
            _carrier("lane-01", "improve", ["003"], 0.6, revisions["lane-01"]),
            _carrier("lane-02", "fresh", [], None, revisions["lane-02"]),
        ]
        pool = []
        for rank, (point, carrier) in enumerate(zip(points, carriers), start=1):
            pool.append(
                {
                    "label": f"C{rank}",
                    "coverage_rank": rank,
                    "point_id": point["point_id"],
                    "point": point,
                    "coverage": 1.0 - rank * 0.1,
                    "deprioritized_hypotheses": [],
                    "carrier": carrier,
                    "carrier_alternatives": [],
                }
            )
        pool_doc = {
            "schema_version": 1,
            "gen_no": 1,
            "ledger_snapshot": snapshot,
            "budget": {"objective_remaining": 90, "admission_cap": 2},
            "pool_size": 6,
            "space": space_receipt(self.registry),
            "lanes": [],
            "lanes_digest": "sha256:" + "0" * 64,
            "proposal_set_revisions": revisions,
            "lanes_without_proposals": [],
            "pool": pool,
        }
        pool_doc["pool_digest"] = slate.recompute_pool_digest(pool_doc)
        pool_doc["generation_seed"] = slate.recompute_generation_seed(pool_doc)

        context_doc = {
            "schema_version": 1,
            "gen_no": 1,
            "ledger_snapshot": snapshot,
            "rendered_text": "fixture measured history",
        }
        context_digest = digest(context_doc)
        stages = {
            "regular-0": _stage_artifact(
                "regular-0", pool_doc, context_digest, ["C2", "C1", "C3"]
            ),
            "regular-1": _stage_artifact(
                "regular-1", pool_doc, context_digest, ["C1", "C2", "C3"]
            ),
        }
        judge_doc = {
            "schema_version": 1,
            "gen_no": 1,
            "generation_seed": pool_doc["generation_seed"],
            "pool_digest": pool_doc["pool_digest"],
            "context_digest": context_digest,
            "budget": {"objective_remaining": 90, "admission_cap": 2},
            "cardinality": {"pool_target": 6, "pool_actual": 3, "judge_called": True},
            "stages": stages,
            "aggregation": {
                "path": "consensus",
                "reason": None,
                "slate": ["C1", "C2"],
                "regular_top2": {
                    "regular-0": ["C2", "C1"],
                    "regular-1": ["C1", "C2"],
                },
            },
            "judge_cost": {"session_ids": {}, "models": {}},
        }
        manifest = slate.build_manifest(
            pool_doc, judge_doc, pool_doc["budget"], ["005", "006"]
        )
        self.gen.mkdir(parents=True, exist_ok=True)
        (self.gen / "pool.json").write_text(json.dumps(pool_doc, indent=2) + "\n")
        (self.gen / "context.json").write_text(json.dumps(context_doc, indent=2) + "\n")
        (self.gen / "judge.json").write_text(json.dumps(judge_doc, indent=2) + "\n")
        (self.gen / "generation.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        return manifest

    def _write_plans(self) -> None:
        plans = self.gen / "plans"
        plans.mkdir(exist_ok=True)
        for slot in self.manifest["slate"]:
            (plans / f"slot-{slot['slot']}.json").write_text(
                json.dumps(
                    {
                        "slot": slot["slot"],
                        "idea": f"Fixture slate idea for {slot['run_id']}.",
                        "change": f"Fixture change for slot {slot['slot']}.",
                        "candidate_name": f"slate_{slot['run_id']}",
                    }
                )
            )

    def _request(self) -> SlateAdmissionRequest:
        return SlateAdmissionRequest(
            background_path=self.run_dir / "background.md",
            catalog_path=None,
            manifest_path=self.gen / "generation.json",
            plans_dir=self.gen / "plans",
            run_dir=self.run_dir,
        )

    def _rewrite_manifest(self, manifest: dict) -> None:
        (self.gen / "generation.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )

    # ---------- atomic admission (design section 10, item 5) ----------

    def test_two_seats_admitted_atomically(self) -> None:
        pre_experience = experience_receipt(self.data)
        admitted = admit_slate_atomic(self.data, self._request())
        self.assertEqual([record["run_id"] for record in admitted], ["005", "006"])
        self.assertEqual([record["status"] for record in admitted], ["pending", "pending"])
        self.assertEqual(len(self.data["records"]), 7)

        manifest_digest = "sha256:" + hashlib.sha256(
            (self.gen / "generation.json").read_bytes()
        ).hexdigest()
        for index, record in enumerate(admitted):
            slot = self.manifest["slate"][index]
            self.assertEqual(record["semantic_point"], slot["point"])
            self.assertEqual(record["op"], slot["carrier"]["op"])
            self.assertEqual(record["source_run_ids"], slot["carrier"]["parents"])
            self.assertEqual(
                record["candidate_name"], f"slate_{slot['run_id']}"
            )
            receipt = record["policy_receipt"]
            self.assertEqual(receipt["schema_version"], 8)
            self.assertEqual(receipt["policy"]["name"], "judged_slate")
            self.assertEqual(receipt["generation_id"], self.manifest["generation_id"])
            self.assertEqual(
                receipt["judge"],
                {
                    "manifest_path": ".semantic/gen-0001/generation.json",
                    "manifest_digest": manifest_digest,
                    "slate_index": index,
                    "candidate_id": slot["candidate_id"],
                    "aggregation": "consensus",
                },
            )
            self.assertEqual(
                receipt["carrier_proposal_set_revision"],
                slot["carrier"]["proposal_set_revision"],
            )
            self.assertEqual(
                receipt["budget"],
                {"selection_index": 6 + index, "admission_cap": 2},
            )
        # The batch revalidates cleanly, schema-6 prefix and schema-8 seats alike.
        self.assertEqual(validate_ledger(self.registry, self.data), [])

    def test_both_seats_share_the_pre_admission_snapshot(self) -> None:
        pre = copy.deepcopy(self.data)
        seen = []
        real_build = ledger_admission.build_semantic_edges

        def spy(records, record):
            seen.append(len(records))
            return real_build(records, record)

        with mock.patch.object(
            ledger_admission, "build_semantic_edges", side_effect=spy
        ):
            admitted = admit_slate_atomic(self.data, self._request())
        # Semantic edges were computed twice against the same 5-record prefix.
        self.assertEqual(seen, [5, 5])
        shared = experience_receipt(pre)
        for record in admitted:
            self.assertEqual(record["policy_receipt"]["experience"], shared)
            self.assertEqual(
                record["policy_receipt"]["search_space_state_revision"], 0
            )
            edge_parents = {
                edge["parent_run_id"] for edge in record["semantic_edges"]
            }
            self.assertTrue(edge_parents <= {"000", "001", "002", "003", "004"})
        # Slot 1's edges never reference its same-generation sibling.
        self.assertNotIn(
            "005",
            {edge["parent_run_id"] for edge in admitted[1]["semantic_edges"]},
        )
        self.assertIsNone(admitted[0]["route_provenance"])
        self.assertIsNone(admitted[1]["route_provenance"])

    def test_invalid_second_seat_leaves_the_ledger_untouched(self) -> None:
        plans = self.gen / "plans"
        (plans / "slot-1.json").write_text(
            json.dumps(
                {"slot": 1, "idea": "", "change": "x", "candidate_name": "y"}
            )
        )
        with self.assertRaisesRegex(AdmissionError, "slot 1"):
            admit_slate_atomic(self.data, self._request())
        self.assertEqual(len(self.data["records"]), 5)

    def test_cli_failure_leaves_the_ledger_bytes_identical(self) -> None:
        ledger_path = self.run_dir / "ledger.json"
        ledger_path.write_text(json.dumps(self.data, indent=2))
        before = ledger_path.read_bytes()
        plans = self.gen / "plans"
        (plans / "slot-1.json").write_text(json.dumps({"slot": 1, "idea": "x"}))
        args = SimpleNamespace(
            ledger=str(ledger_path),
            task="hard-interactions",
            background=str(self.run_dir / "background.md"),
            catalog=None,
            manifest=str(self.gen / "generation.json"),
            plans_dir=str(plans),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                ledger_cli.cmd_admit_slate(args)
        self.assertEqual(ledger_path.read_bytes(), before)
        self.assertFalse((self.run_dir / "loop_state.md").exists())

    # ---------- judge binding negative cases ----------

    def test_tampered_manifest_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["slate"][0]["candidate_id"] = "sha256:" + "f" * 64
        self._rewrite_manifest(manifest)
        with self.assertRaisesRegex(AdmissionError, "generation artifacts"):
            admit_slate_atomic(self.data, self._request())
        self.assertEqual(len(self.data["records"]), 5)

    def test_run_id_collision_is_rejected(self) -> None:
        pool_doc = json.loads((self.gen / "pool.json").read_text())
        judge_doc = json.loads((self.gen / "judge.json").read_text())
        manifest = slate.build_manifest(
            pool_doc, judge_doc, pool_doc["budget"], ["004", "005"]
        )
        self._rewrite_manifest(manifest)
        with self.assertRaisesRegex(AdmissionError, "already exists"):
            admit_slate_atomic(self.data, self._request())
        self.assertEqual(len(self.data["records"]), 5)

    def test_prefix_drift_is_rejected(self) -> None:
        self.data["records"][0]["best_warm_score"] = 0.01
        with self.assertRaisesRegex(AdmissionError, "records_digest"):
            admit_slate_atomic(self.data, self._request())
        self.assertEqual(len(self.data["records"]), 5)

    def test_plan_slot_misalignment_is_rejected(self) -> None:
        (self.gen / "plans" / "slot-0.json").write_text(
            json.dumps(
                {
                    "slot": 1,
                    "idea": "misfiled plan",
                    "change": "misfiled change",
                    "candidate_name": "slate_005",
                }
            )
        )
        with self.assertRaisesRegex(AdmissionError, "slot-0"):
            admit_slate_atomic(self.data, self._request())
        self.assertEqual(len(self.data["records"]), 5)

    def test_tampered_presented_order_is_rejected_on_replay(self) -> None:
        judge_doc = json.loads((self.gen / "judge.json").read_text())
        stage = judge_doc["stages"]["regular-0"]
        stage["presented_order"] = list(reversed(stage["presented_order"]))
        (self.gen / "judge.json").write_text(json.dumps(judge_doc, indent=2) + "\n")
        pool_doc = json.loads((self.gen / "pool.json").read_text())
        # The manifest chain is rebuilt over the tampered judge, so only the
        # deterministic replay can catch this.
        manifest = slate.build_manifest(
            pool_doc, judge_doc, pool_doc["budget"], ["005", "006"]
        )
        self._rewrite_manifest(manifest)
        with self.assertRaisesRegex(AdmissionError, "does not replay"):
            admit_slate_atomic(self.data, self._request())
        self.assertEqual(len(self.data["records"]), 5)

    def test_carrier_revision_outside_the_proposal_sets_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["slate"][0]["carrier"]["proposal_set_revision"] = "sha256:" + "9" * 64
        manifest["generation_id"] = slate.recompute_generation_id(manifest)
        self._rewrite_manifest(manifest)
        with self.assertRaisesRegex(AdmissionError, "proposal revision"):
            admit_slate_atomic(self.data, self._request())
        self.assertEqual(len(self.data["records"]), 5)

    # ---------- receipt schema 8 shape ----------

    def test_schema8_receipt_shape_is_enforced(self) -> None:
        admit_slate_atomic(self.data, self._request())
        record = next(r for r in self.data["records"] if r["run_id"] == "005")

        broken = copy.deepcopy(self.data)
        broken_record = next(r for r in broken["records"] if r["run_id"] == "005")
        del broken_record["policy_receipt"]["judge"]["manifest_digest"]
        errors = validate_ledger(self.registry, broken)
        self.assertTrue(any("judge" in error for error in errors))

        broken = copy.deepcopy(self.data)
        broken_record = next(r for r in broken["records"] if r["run_id"] == "005")
        broken_record["policy_receipt"]["budget"]["selection_index"] = 7
        errors = validate_ledger(self.registry, broken)
        self.assertTrue(any("selection_index" in error for error in errors))

        self.assertEqual(validate_ledger(self.registry, self.data), [])
        self.assertEqual(record["policy_receipt"]["schema_version"], 8)

    # ---------- schema 6/7 regression ----------

    def test_schema6_admission_still_works_after_a_slate(self) -> None:
        admit_slate_atomic(self.data, self._request())
        point = complete_point(
            self.registry, {"dim-validation-selection": "hyp-valid-cv"}
        )
        point_path = self.run_dir / "point-007.json"
        point_path.write_text(json.dumps(point))
        receipt = policy_receipt("improve", ["006"], point, selection_index=8)
        receipt["experience"].update(
            {
                "generation": 1,
                "updated_at_run": "004",
                "revision": digest(self.data["experience"]),
            }
        )
        receipt_path = self.run_dir / "receipt-007.json"
        receipt_path.write_text(json.dumps(receipt))
        record = admit_record(
            self.data,
            AdmissionRequest(
                run_id="007",
                kind="optimization",
                idea="Fixture regression candidate after a slate.",
                change="fixture regression change",
                source_run_ids="006",
                op="improve",
                background_path=self.run_dir / "background.md",
                catalog_path=None,
                semantic_point_path=point_path,
                policy_receipt_path=receipt_path,
                candidate_name_hint="fixture_007",
            ),
        )
        self.assertEqual(record["policy_receipt"]["schema_version"], 6)
        self.assertEqual(len(self.data["records"]), 8)


class SlateAdmissionCliTests(unittest.TestCase):
    """The full pipeline: lanes -> construct -> judge -> manifest -> admit-slate."""

    def test_admit_slate_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            registry = fixture_registry()
            (run_dir / "background.md").write_text(background_text(registry))
            data = _ledger_data(registry)
            ledger_path = run_dir / "ledger.json"
            ledger_path.write_text(json.dumps(data, indent=2))

            gen = run_dir / ".semantic" / "gen-0001"
            gen.mkdir(parents=True)
            lanes_path = gen / "lanes.json"
            with contextlib.redirect_stdout(io.StringIO()):
                got_select.cmd_decide(
                    SimpleNamespace(
                        ledger=str(ledger_path),
                        cfg=None,
                        mode="lanes",
                        output=str(lanes_path),
                    )
                )
            lanes_doc = json.loads(lanes_path.read_text())
            proposals_dir = gen / "proposals"
            proposals_dir.mkdir()
            for lane in lanes_doc["lanes"]:
                with contextlib.redirect_stdout(io.StringIO()):
                    semantic_search.cmd_propose(
                        SimpleNamespace(
                            background=run_dir / "background.md",
                            ledger=ledger_path,
                            op=lane["op"],
                            parents=",".join(lane["parents"]),
                            max_points=128,
                            baseline_only=False,
                            output=proposals_dir / f"{lane['lane_id']}.json",
                        )
                    )
            with contextlib.redirect_stdout(io.StringIO()):
                slate.cmd_construct(
                    SimpleNamespace(
                        lanes=lanes_path,
                        proposals_dir=proposals_dir,
                        ledger=ledger_path,
                        background=run_dir / "background.md",
                        pool_size=6,
                        pool_output=gen / "pool.json",
                        context_output=gen / "context.json",
                    )
                )
            pool_doc = json.loads((gen / "pool.json").read_text())
            self.assertGreaterEqual(len(pool_doc["pool"]), 3)

            judgments = gen / "judgments"
            judgments.mkdir()
            for stage in ("regular-0", "regular-1"):
                with contextlib.redirect_stdout(io.StringIO()):
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
                top2 = {"C1", "C3"}
                ranking = [label for label in order if label in top2] + [
                    label for label in order if label not in top2
                ]
                receipt = judgments / f"{stage}.receipt.json"
                receipt.write_text(json.dumps({"ranking": ranking, "rationale": "t"}))
                with contextlib.redirect_stdout(io.StringIO()):
                    code = slate.cmd_validate_judge(
                        SimpleNamespace(
                            input=judgments / f"{stage}.input.json",
                            receipt=receipt,
                            session_id=f"sess-{stage}",
                            model="fixture",
                            output=judgments / f"{stage}.json",
                        )
                    )
                self.assertEqual(code, 0)
            with contextlib.redirect_stdout(io.StringIO()):
                slate.cmd_aggregate(
                    SimpleNamespace(
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        judgments_dir=judgments,
                        output=gen / "judge.json",
                    )
                )
            with contextlib.redirect_stdout(io.StringIO()):
                slate.cmd_build_manifest(
                    SimpleNamespace(
                        lanes=lanes_path,
                        pool=gen / "pool.json",
                        context=gen / "context.json",
                        judge=gen / "judge.json",
                        reserved_run_ids="005,006",
                        donor_snapshot=None,
                        no_donor=False,
                        output=gen / "generation.json",
                    )
                )
            manifest = json.loads((gen / "generation.json").read_text())
            plans = gen / "plans"
            plans.mkdir()
            for slot in manifest["slate"]:
                (plans / f"slot-{slot['slot']}.json").write_text(
                    json.dumps(
                        {
                            "slot": slot["slot"],
                            "idea": f"Pipeline slate idea for {slot['run_id']}.",
                            "change": f"Pipeline change for slot {slot['slot']}.",
                            "candidate_name": f"slate_{slot['run_id']}",
                        }
                    )
                )

            with contextlib.redirect_stdout(io.StringIO()):
                code = ledger_cli.cmd_admit_slate(
                    SimpleNamespace(
                        ledger=str(ledger_path),
                        task="hard-interactions",
                        background=str(run_dir / "background.md"),
                        catalog=None,
                        manifest=str(gen / "generation.json"),
                        plans_dir=str(plans),
                    )
                )
            self.assertEqual(code, 0)

            admitted = json.loads(ledger_path.read_text())
            self.assertEqual(len(admitted["records"]), 7)
            seats = {
                record["run_id"]: record for record in admitted["records"][-2:]
            }
            self.assertEqual(sorted(seats), ["005", "006"])
            manifest_digest = "sha256:" + hashlib.sha256(
                (gen / "generation.json").read_bytes()
            ).hexdigest()
            for slot in manifest["slate"]:
                record = seats[slot["run_id"]]
                self.assertEqual(record["status"], "pending")
                receipt = record["policy_receipt"]
                self.assertEqual(receipt["schema_version"], 8)
                self.assertEqual(receipt["judge"]["manifest_digest"], manifest_digest)
                self.assertEqual(receipt["judge"]["slate_index"], slot["slot"])
                # No evaluation budget is configured in this run: the lanes
                # budget records admission_cap=null, which the receipt keeps.
                self.assertIsNone(receipt["budget"]["admission_cap"])
            self.assertEqual(validate_ledger(registry, admitted), [])
            self.assertTrue((run_dir / "loop_state.md").exists())


if __name__ == "__main__":
    unittest.main()
