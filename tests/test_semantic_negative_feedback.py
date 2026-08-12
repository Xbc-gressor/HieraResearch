"""Objective-free smoke over the two semantic negative-feedback channels.

The attempt downside (selection) and the planned-route memory (generation) are
exercised through the real CLIs against a synthetic ledger.  No objective is
ever called: every score here is written directly by ``record-run``.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import validate_ledger  # noqa: E402
from ledger import _load_ledger, _set_experience, cmd_add_record, record_run  # noqa: E402
from semantic_attempts import (  # noqa: E402
    DEFAULT_ATTEMPT_CONFIG,
    attempt_adjustment,
    classify_attempts,
)
from semantic_routes import (  # noqa: E402
    DEFAULT_ROUTE_CONFIG,
    build_route_memory,
    route_arm_active,
    route_config,
)
from semantic_space import complete_point, digest  # noqa: E402
from tests.fixtures import (  # noqa: E402
    attach_matched_transfer,
    background_text,
    fixture_registry,
    policy_receipt,
)


def _experience(run_id: str, generation: int = 0) -> dict:
    return {
        "schema_version": 3,
        "updated_at_run": run_id,
        "generation": generation,
        "summary": "",
        "promising_regions": [],
        "lessons": [],
        "bottlenecks": [],
        "dimension_evidence": [],
        "hypothesis_evidence": [],
    }


class NegativeFeedbackSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.baseline = complete_point(self.registry)
        self.filtered = complete_point(
            self.registry, {"dim-data-curation": "hyp-data-filtered"}
        )

    def _add(self, run_dir: Path, run_id, op, parents, point, *, route=None) -> None:
        point_path = run_dir / "point.json"
        policy_path = run_dir / "policy.json"
        ledger_path = run_dir / "ledger.json"
        point_path.write_text(json.dumps(point))
        receipt = policy_receipt(
            op, parents, point, selection_index=int(run_id) + 1, schema_version=6
        )
        if ledger_path.exists():
            experience = json.loads(ledger_path.read_text()).get("experience")
            if isinstance(experience, dict):
                receipt["experience"].update(
                    {
                        "generation": experience["generation"],
                        "updated_at_run": experience["updated_at_run"],
                        "revision": digest(experience),
                    }
                )
        policy_path.write_text(json.dumps(receipt))
        # The run keeps its proposal sets; the offline replay reads them to
        # recover every ranked point's coverage.
        semantic = run_dir / ".semantic" / run_id
        semantic.mkdir(parents=True, exist_ok=True)
        (semantic / "proposals.json").write_text(
            json.dumps(
                {
                    "action": {"op": op, "parents": parents},
                    "proposals": [
                        {
                            "point_id": point["point_id"],
                            "coverage": receipt["components"]["coverage"],
                        }
                    ],
                }
            )
        )
        route_path = None
        if route is not None:
            route_path = run_dir / "route.json"
            route_path.write_text(json.dumps(route))
        args = types.SimpleNamespace(
            ledger=str(ledger_path),
            task="hard-interactions",
            run_id=run_id,
            kind="optimization",
            op=op,
            source_run_ids=",".join(parents),
            background=str(run_dir / "background.md"),
            catalog=None,
            semantic_point=str(point_path),
            policy_receipt=str(policy_path),
            idea=f"Complete fixture solution {run_id} at the selected point.",
            change=f"fixture change for {op} run {run_id}",
            candidate_name_hint=f"fixture_{run_id}",
            description=None,
            route_provenance=None if route_path is None else str(route_path),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cmd_add_record(args), 0)

    def _record(self, run_dir: Path, run_id: str, score) -> None:
        ledger_path = run_dir / "ledger.json"
        data = json.loads(ledger_path.read_text())
        record = next(r for r in data["records"] if r["run_id"] == run_id)
        if record["source_run_ids"] and score is not None:
            # A scored non-fresh candidate must carry a transfer receipt; the
            # attempt statistic itself has no opinion about parameter transfer.
            parent = next(
                r for r in data["records"] if r["run_id"] == record["source_run_ids"][0]
            )
            attach_matched_transfer(parent, record, control_score=score)
            ledger_path.write_text(json.dumps(data))
        record_run(
            run_dir / "ledger.json",
            "hard-interactions",
            run_id,
            final_best_score=score,
            status="crash" if score is None else "auto",
        )
        data = _load_ledger(run_dir / "ledger.json")
        _set_experience(
            run_dir / "ledger.json",
            data,
            _experience(run_id, generation=int(run_id)),
            validated_dag_revision=data["dag_revision"],
        )

    def test_attempt_observations_survive_a_tuning_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "background.md").write_text(background_text(self.registry))
            self._add(run_dir, "000", "fresh", [], self.baseline)
            self._record(run_dir, "000", 0.5)
            self._add(run_dir, "001", "improve", ["000"], self.filtered)
            self._record(run_dir, "001", 0.7)

            ledger = json.loads((run_dir / "ledger.json").read_text())
            rows = {row["run_id"]: row for row in classify_attempts(ledger)}
            # 001 got worse than the parent it inherited from: a screen_fail.
            self.assertEqual(rows["001"]["outcome"], "screen_fail")
            # A fresh candidate has no comparable parent, so it is unpaired
            # rather than neutral.
            self.assertEqual(rows["000"]["outcome"], "unpaired")

            # A later tuning close rewrites the mutable score fields; the
            # frozen observation must not follow them.
            record = next(r for r in ledger["records"] if r["run_id"] == "001")
            record["final_best_score"] = 0.1
            record["best_warm_score"] = 0.1
            self.assertEqual(
                classify_attempts(ledger)[1]["outcome"], "screen_fail"
            )
            self.assertEqual(validate_ledger(self.registry, ledger), [])

    def test_repaired_crash_replaces_the_attempt_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "background.md").write_text(background_text(self.registry))
            self._add(run_dir, "000", "fresh", [], self.baseline)
            ledger_path = run_dir / "ledger.json"

            record_run(
                ledger_path,
                "hard-interactions",
                "000",
                status="crash",
            )
            crashed = json.loads(ledger_path.read_text())
            self.assertEqual(crashed["attempt_observations"][0]["status"], "crash")

            record_run(
                ledger_path,
                "hard-interactions",
                "000",
                final_best_score=0.9,
                status="discard",
            )
            repaired = json.loads(ledger_path.read_text())
            observation = repaired["attempt_observations"][0]
            self.assertEqual(len(repaired["attempt_observations"]), 1)
            self.assertEqual(
                (observation["status"], observation["screening_score"]),
                ("discard", 0.9),
            )
            self.assertEqual(classify_attempts(repaired)[0]["outcome"], "unpaired")
            self.assertEqual(validate_ledger(self.registry, repaired), [])

            inconsistent = json.loads(json.dumps(repaired))
            inconsistent["attempt_observations"][0].update(
                status="crash", screening_score=None
            )
            self.assertTrue(
                any(
                    "crash/non-crash state must match" in error
                    for error in validate_ledger(self.registry, inconsistent)
                )
            )

    def test_attempt_adjustment_directions(self) -> None:
        cfg = dict(DEFAULT_ATTEMPT_CONFIG)

        def rows(outcomes, point="P"):
            return [
                {"run_id": str(i), "point_id": point, "op": "improve",
                 "outcome": outcome, "delta": None, "parent_run_id": None}
                for i, outcome in enumerate(outcomes)
            ]

        def value(outcomes, reference=()):
            return attempt_adjustment(
                rows(outcomes) + list(reference), point_id="P", op="improve", cfg=cfg
            )[0]

        neutral_reference = rows(["screen_success"] * 6, point="Q")
        self.assertEqual(value([]), 0.0)
        # A success never buys a bonus.
        self.assertEqual(value(["screen_success"] * 3, neutral_reference), 0.0)
        self.assertLess(value(["screen_fail"] * 3, neutral_reference), 0.0)
        self.assertLess(value(["screen_neutral"] * 3, neutral_reference), 0.0)
        self.assertLess(value(["crash"]), 0.0)
        # Unpaired attempts dilute a crash but cannot cancel it to zero.
        self.assertLess(value(["crash"] + ["unpaired"] * 20), 0.0)
        # New non-failures soft-reopen the point.
        self.assertGreater(
            value(["screen_fail"] * 3 + ["screen_success"] * 6, neutral_reference),
            value(["screen_fail"] * 3, neutral_reference),
        )
        # The channel is bounded, so it cannot generally overwhelm coverage.
        self.assertGreaterEqual(
            value(["crash"] * 20 + ["screen_fail"] * 20),
            -float(cfg["attempt_cap"]),
        )

    def test_route_arm_requires_and_validates_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "background.md").write_text(background_text(self.registry))
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"semantic_search": {"n_route_sketches": 2,
                                                "route_memory": True}})
            )
            cfg = route_config({"n_route_sketches": 2, "route_memory": True})
            self.assertTrue(route_arm_active(cfg))
            self.assertFalse(route_arm_active(route_config(None)))
            # One sketch is the explicit planning baseline, not "no planning":
            # it must still produce and persist provenance, or a 1-vs-3
            # comparison would also be a planning-vs-no-planning comparison.
            self.assertTrue(route_arm_active(route_config({"n_route_sketches": 1})))

            # Arm on, provenance absent: admission fails loudly instead of
            # silently running the no-memory arm.
            with self.assertRaisesRegex(SystemExit, "route arm requires"):
                self._add(run_dir, "000", "fresh", [], self.baseline)

            memory = build_route_memory({}, self.baseline, "fresh", cfg)
            self.assertEqual(memory["rows"], [])
            route = {
                "schema_version": 1,
                "point_id": memory["point_id"],
                "op": "fresh",
                "n_route_sketches": 2,
                "route_memory": True,
                "memory_rows": [],
                "sketches": [
                    {"sketch_id": "r1", "route": "wide shallow trunk"},
                    {"sketch_id": "r2", "route": "narrow deep trunk"},
                ],
                "preference_order": ["r2", "r1"],
                "chosen_sketch_id": "r2",
                "chosen_route": "narrow deep trunk",
            }
            forged = dict(route, chosen_route="something else entirely")
            with self.assertRaisesRegex(SystemExit, "invalid route provenance"):
                self._add(run_dir, "000", "fresh", [], self.baseline, route=forged)

            self._add(run_dir, "000", "fresh", [], self.baseline, route=route)
            self._record(run_dir, "000", 0.5)

            ledger = json.loads((run_dir / "ledger.json").read_text())
            stored = ledger["records"][0]["route_provenance"]
            self.assertEqual(stored["chosen_route"], "narrow deep trunk")
            self.assertEqual(validate_ledger(self.registry, ledger), [])

            # The next candidate at the same point sees that planned route
            # together with the attempt's realized outcome.
            memory = build_route_memory(ledger, self.baseline, "fresh", cfg)
            row = memory["rows"][0]
            self.assertEqual(
                (row["run_id"], row["relation"], row["route"], row["outcome"]),
                ("000", "same_point", "narrow deep trunk", "unpaired"),
            )
            self.assertTrue(row["route_available"])

            # A neighbor point is offered as transfer context, marked as such.
            neighbor = build_route_memory(ledger, self.filtered, "fresh", cfg)
            self.assertEqual(neighbor["rows"][0]["relation"], "neighbor")
            self.assertEqual(neighbor["rows"][0]["distance"], 1)

            # With the memory arm off no rows are shown, and the envelope
            # still reports which arm produced that.
            off = build_route_memory(ledger, self.baseline, "fresh",
                                     route_config({"n_route_sketches": 2}))
            self.assertEqual(off["rows"], [])
            self.assertFalse(off["route_memory"])

    def test_default_arms_are_off_and_record_no_provenance(self) -> None:
        self.assertEqual(DEFAULT_ROUTE_CONFIG["n_route_sketches"], 0)
        self.assertFalse(DEFAULT_ROUTE_CONFIG["route_memory"])
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "background.md").write_text(background_text(self.registry))
            self._add(run_dir, "000", "fresh", [], self.baseline)
            ledger = json.loads((run_dir / "ledger.json").read_text())
            self.assertIsNone(ledger["records"][0]["route_provenance"])

    def test_offline_replay_reports_the_required_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "background.md").write_text(background_text(self.registry))
            self._add(run_dir, "000", "fresh", [], self.baseline)
            self._record(run_dir, "000", 0.5)
            self._add(run_dir, "001", "improve", ["000"], self.filtered)
            self._record(run_dir, "001", 0.7)
            self._add(run_dir, "002", "improve", ["000"], self.filtered)
            self._record(run_dir, "002", None)

            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "replay_attempts.py"),
                 "--ledger", str(run_dir / "ledger.json")],
                cwd=ROOT, capture_output=True, text=True,
            )
            report = json.loads(proc.stdout)
            self.assertTrue(report["ok"], proc.stdout + proc.stderr)
            self.assertTrue(report["partition"]["exhaustive"])
            self.assertEqual(report["partition"]["counts"]["crash"], 1)
            self.assertEqual(report["partition"]["counts"]["screen_fail"], 1)
            self.assertEqual(report["partition"]["counts"]["unpaired"], 1)
            self.assertTrue(report["directions"]["crash_survives_unpaired"])
            self.assertTrue(report["directions"]["all_success_is_zero"])
            # Every recorded coverage-family selection replayed exactly: its
            # score was rebuilt from the persisted slate and the pre-admission
            # prefix, and it equals the receipt's own acquisition score.
            replay = report["selection_replay"]
            self.assertEqual(replay["selections"], 3)
            self.assertEqual(replay["replayed_exactly"], 3, replay["unreplayable"])
            self.assertIn("overlap", report["channel_overlap"])

    def test_replay_refuses_to_pass_on_an_empty_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            ledger.write_text(json.dumps({"records": []}))
            proc = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "replay_attempts.py"),
                 "--ledger", str(ledger)],
                cwd=ROOT, capture_output=True, text=True,
            )
            report = json.loads(proc.stdout)
            # The synthetic direction probes pass in a vacuum; a run with no
            # observations and no selections must still not report success.
            self.assertTrue(report["directions"]["fail_is_negative"])
            self.assertFalse(report["ok"])
            self.assertEqual(proc.returncode, 1)


if __name__ == "__main__":
    unittest.main()
