from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import finalize_tuning  # noqa: E402
import ledger  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402
from semantic_evidence import _json_sha256  # noqa: E402
from tune_tools import _candidate_execution_revision  # noqa: E402


def _report(*, stage_status: str) -> dict:
    return {
        "phase_a": {
            "status": "ok",
            "warm_start_configs": [{"params": {"x": 1.0}, "score": 1.0}],
            "best_warm_params": {"x": 1.0},
            "best_warm_score": 1.0,
            "trials_attempted": 1,
            "elapsed_seconds": 1.0,
            "search_space": {"x": ["float", 0.0, 2.0]},
        },
        "preflight": {"attempts": []},
        "phase_c": {
            "stages": [
                {
                    "method": "grid",
                    "status": stage_status,
                    "trials": [{"params": {"x": 2.0}, "score": 0.8}],
                    "elapsed_seconds": 2.5,
                }
            ]
        },
    }


def _record() -> dict:
    return {
        "run_id": "001",
        "semantic_point": {},
        "policy_receipt": {},
        "status": "keep",
        "final_best_score": 1.0,
        "best_warm_score": 1.0,
        "trials_attempted": 1,
        "tune": False,
        "dag_revision": 1,
    }


class TuningFinalizationTests(unittest.TestCase):
    def _fixture(self, root: Path, *, stage_status: str) -> tuple[Path, Path, Path]:
        run_dir = root / "run"
        candidate_dir = run_dir / "candidates" / "001"
        candidate_dir.mkdir(parents=True)
        candidate_path = candidate_dir / "train.py"
        candidate_path.write_text(
            "PARAM_SCHEMA = {'x': 'float'}\n"
            "SEARCH_SPACE = {'x': ('float', 0.0, 2.0)}\n"
            "BASE_PARAMS = {'x': 1.0}\n"
            "def make_model(params):\n"
            "    return params\n"
        )
        (candidate_dir / "prepare.py").write_text(
            "def evaluate_config(make_model, params):\n"
            "    return float(params['x'])\n"
        )
        report_path = candidate_dir / "tune_report.json"
        report = _report(stage_status=stage_status)
        report["phase_a"]["candidate_code_revision"] = (
            _candidate_execution_revision(candidate_path)
        )
        report_path.write_text(json.dumps(report, indent=2))
        ledger_path = run_dir / "ledger.json"
        ledger_path.write_text(
            json.dumps(
                {
                    "task": "autoresearch-baseline",
                    "tag": "test",
                    "metric": "val_bpb",
                    "dag_revision": 1,
                    "search_space_state": empty_search_space_state(),
                    "records": [_record()],
                },
                indent=2,
            )
        )
        return candidate_path, report_path, ledger_path

    def test_interrupted_stage_cannot_mutate_candidate_report_or_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="running",
            )
            before = {
                path: path.read_bytes()
                for path in (candidate_path, report_path, ledger_path)
            }

            with self.assertRaisesRegex(ValueError, "status is not terminal"):
                finalize_tuning.finalize(
                    candidate_path=candidate_path,
                    report_path=report_path,
                    ledger_path=ledger_path,
                    run_id="001",
                    task_name="autoresearch-baseline",
                )

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

            legacy_close = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "set-tuning",
                    "--ledger",
                    str(ledger_path),
                    "--task",
                    "autoresearch-baseline",
                    "--run-id",
                    "001",
                    "--from-report",
                    str(report_path),
                    "--mark-tuned",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(legacy_close.returncode, 0)
            self.assertIn("--mark-tuned is disabled", legacy_close.stderr)
            self.assertEqual(ledger_path.read_bytes(), before[ledger_path])

    def test_invalid_report_is_rejected_before_any_mutation(self) -> None:
        """Finalization is all-or-nothing: a rejected report changes nothing."""

        def no_finite_trial(report: dict) -> None:
            report["phase_c"]["stages"][0]["trials"] = []

        def wrong_method_chain(report: dict) -> None:
            report["phase_c"]["stages"][0]["method"] = "bo"

        def forged_warm_best(report: dict) -> None:
            report["phase_a"]["best_warm_score"] = 0.1

        def drifted_search_space(report: dict) -> None:
            report["phase_a"]["search_space"]["x"][2] = 3.0

        def out_of_space_trial(report: dict) -> None:
            report["phase_c"]["stages"][0]["trials"].append(
                {"params": {"x": 99.0}, "score": 9.0}
            )

        cases = [
            ("must contain a finite trial", no_finite_trial),
            ("method chain", wrong_method_chain),
            ("best_warm_score", forged_warm_best),
            ("does not match", drifted_search_space),
            ("violates SEARCH_SPACE", out_of_space_trial),
        ]
        for needle, mutate in cases:
            with self.subTest(needle=needle):
                with tempfile.TemporaryDirectory() as tmp:
                    candidate_path, report_path, ledger_path = self._fixture(
                        Path(tmp), stage_status="ok"
                    )
                    report = json.loads(report_path.read_text())
                    mutate(report)
                    report_path.write_text(json.dumps(report, indent=2))
                    before = {
                        path: path.read_bytes()
                        for path in (candidate_path, report_path, ledger_path)
                    }

                    with self.assertRaisesRegex(ValueError, needle):
                        finalize_tuning.finalize(
                            candidate_path=candidate_path,
                            report_path=report_path,
                            ledger_path=ledger_path,
                            run_id="001",
                            task_name="autoresearch-baseline",
                        )

                    for path, content in before.items():
                        self.assertEqual(path.read_bytes(), content)

    def test_failed_stage_partial_trial_is_finalized(self) -> None:
        """A crashed last invocation must not discard the stage's proven rows.

        Trial rows are durable and bound to the candidate on disk by
        admission-time revision validation, so `failed` describes how the search
        ended, not whether its observations count. Discarding them stranded real,
        already-charged evaluations (run 0802-sonnet-ex125-1/007 lost a 1.050231
        incumbent this way).
        """
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="failed"
            )
            ledger_data = json.loads(ledger_path.read_text())
            ledger_data["records"][0].update(
                {
                    "phase_c_method": "bo",
                    "phase_b_decision": "continue",
                    "warm_percentile": 99,
                }
            )
            ledger_path.write_text(json.dumps(ledger_data, indent=2))

            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )

            # The stage's 0.8 trial beats the 1.0 warm incumbent and is applied.
            self.assertEqual(result["final_best_score"], 0.8)
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["final_best_score"], 0.8)
            self.assertEqual(record["phase_c_method"], "grid")
            self.assertEqual(record["evaluation_depth"], "tuned_lightly")

    def test_phase_c_losing_to_warm_incumbent_still_records_tuned_depth(self) -> None:
        """Depth measures evaluation effort, not which row won the argmin.

        A Phase-C stage that scored trials but failed to beat the warm
        incumbent is the most common Phase-C outcome.  Depth is now graded —
        one scored Phase-C trial promotes screening to "tuned_lightly";
        "tuned" requires tuner.tuned_threshold attempts.
        """
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            report = json.loads(report_path.read_text())
            # The scored Phase-C trial loses to the 1.0 warm incumbent.
            report["phase_c"]["stages"][0]["trials"] = [
                {"params": {"x": 2.0}, "score": 1.5}
            ]
            report_path.write_text(json.dumps(report, indent=2))

            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )

            self.assertEqual(result["final_best_score"], 1.0)
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["final_best_score"], 1.0)
            self.assertIsNone(record["phase_c_method"])
            self.assertEqual(record["evaluation_depth"], "tuned_lightly")

    def test_first_bout_close_records_progressive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            self.assertEqual(result["final_best_score"], 0.8)
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["tuning_bouts"], 1)
            self.assertIs(record["last_bout_improved"], True)
            self.assertEqual(record["evaluation_depth"], "tuned_lightly")
            self.assertIs(record["tune"], True)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["last_finalized_stage_index"], 0)

    def test_second_bout_close_recovers_best_across_bouts(self) -> None:
        """A continuation bout closes against every stage, not just its own."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"].append(
                {
                    "method": "grid",
                    "bout_index": 1,
                    "status": "ok",
                    "trials": [{"params": {"x": 0.7}, "score": 0.7}],
                    "elapsed_seconds": 1.0,
                }
            )
            report_path.write_text(json.dumps(report, indent=2))
            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            self.assertEqual(result["final_best_score"], 0.7)
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["tuning_bouts"], 2)
            self.assertIs(record["last_bout_improved"], True)
            self.assertEqual(record["evaluation_depth"], "tuned_lightly")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["last_finalized_stage_index"], 1)
            self.assertIn("'x': 0.7", candidate_path.read_text())

    def test_non_improving_bout_marks_non_responder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"].append(
                {
                    "method": "grid",
                    "bout_index": 1,
                    "status": "ok",
                    "trials": [{"params": {"x": 1.9}, "score": 1.9}],
                    "elapsed_seconds": 1.0,
                }
            )
            report_path.write_text(json.dumps(report, indent=2))
            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            self.assertEqual(result["final_best_score"], 0.8)  # bout 0's best
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["tuning_bouts"], 2)
            self.assertIs(record["last_bout_improved"], False)

    def test_depth_becomes_tuned_at_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"][0]["trials"] = [
                {"params": {"x": 1.5 + i / 100.0}, "score": 1.5 + i / 100.0}
                for i in range(20)
            ]
            report_path.write_text(json.dumps(report, indent=2))
            finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["evaluation_depth"], "tuned")

    def test_phase_a_ledger_ingestion_rejects_untrustworthy_reports(self) -> None:
        """A phase_a-only report must still agree with the candidate on disk."""

        def search_space_drift(report: dict, candidate_path: Path) -> None:
            report["phase_a"]["search_space"]["x"][2] = 3.0

        def out_of_space_warm_row(report: dict, candidate_path: Path) -> None:
            report["phase_a"]["warm_start_configs"].append(
                {"params": {"x": 9.0}, "score": 9.0}
            )

        def invalid_candidate_contract(report: dict, candidate_path: Path) -> None:
            candidate_path.write_text(
                candidate_path.read_text() + "SEARCH_SPACE.update({})\n"
            )
            report["phase_a"]["candidate_code_revision"] = (
                _candidate_execution_revision(candidate_path)
            )

        cases = [
            ("phase_a.search_space does not match", search_space_drift),
            ("violates SEARCH_SPACE", out_of_space_warm_row),
            ("candidate tuning contract is invalid", invalid_candidate_contract),
        ]
        for needle, mutate in cases:
            with self.subTest(needle=needle):
                with tempfile.TemporaryDirectory() as tmp:
                    candidate_path, report_path, ledger_path = self._fixture(
                        Path(tmp), stage_status="ok"
                    )
                    report = json.loads(report_path.read_text())
                    report["phase_c"]["stages"] = []
                    mutate(report, candidate_path)
                    report_path.write_text(json.dumps(report, indent=2))
                    ledger_before = ledger_path.read_bytes()

                    with self.assertRaisesRegex(ValueError, needle):
                        finalize_tuning.ledger._tuning_record_from_report(report_path)

                    self.assertEqual(ledger_path.read_bytes(), ledger_before)

    def test_direct_ledger_finalization_rejects_invalid_candidate_contract(
        self,
    ) -> None:
        for defect in ("out_of_bounds", "module_scope_mutation"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                candidate_path, report_path, ledger_path = self._fixture(
                    Path(tmp), stage_status="ok"
                )
                source = candidate_path.read_text().replace(
                    "BASE_PARAMS = {'x': 1.0}",
                    "BASE_PARAMS = {'x': 2.0}",
                )
                report = json.loads(report_path.read_text())
                if defect == "out_of_bounds":
                    source = source.replace(
                        "SEARCH_SPACE = {'x': ('float', 0.0, 2.0)}",
                        "SEARCH_SPACE = {'x': ('float', 0.0, 1.0)}",
                    )
                    report["phase_a"]["search_space"]["x"][2] = 1.0
                else:
                    source += "SEARCH_SPACE.update({})\n"
                candidate_path.write_text(source)
                report["phase_a"]["candidate_code_revision"] = (
                    _candidate_execution_revision(candidate_path)
                )
                report.update(
                    {
                        "final_best_params": {"x": 2.0},
                        "final_best_score": 0.8,
                        "applied_to_base_params": True,
                    }
                )
                report_path.write_text(json.dumps(report, indent=2))
                before = {
                    path: path.read_bytes()
                    for path in (candidate_path, report_path, ledger_path)
                }

                with self.assertRaisesRegex(
                    ValueError,
                    "candidate tuning contract is invalid|violates SEARCH_SPACE",
                ):
                    finalize_tuning.ledger.finalize_tuning(
                        ledger_path,
                        "autoresearch-baseline",
                        "001",
                        report_path,
                    )

                for path, content in before.items():
                    self.assertEqual(path.read_bytes(), content)

    def test_finalization_rejects_stale_candidate_or_evaluator_revision(self) -> None:
        for changed_file in ("candidate", "prepare", "search_space"):
            with self.subTest(changed_file=changed_file), tempfile.TemporaryDirectory() as tmp:
                candidate_path, report_path, ledger_path = self._fixture(
                    Path(tmp), stage_status="ok"
                )
                if changed_file == "candidate":
                    candidate_path.write_text(
                        candidate_path.read_text().replace(
                            "    return params\n",
                            "    return dict(params)\n",
                        )
                    )
                elif changed_file == "prepare":
                    prepare_path = candidate_path.parent / "prepare.py"
                    prepare_path.write_text(prepare_path.read_text() + "\n# revision\n")
                else:
                    candidate_path.write_text(
                        candidate_path.read_text().replace(
                            "SEARCH_SPACE = {'x': ('float', 0.0, 2.0)}",
                            "SEARCH_SPACE = {'x': ('float', 0.0, 3.0)}",
                        )
                    )
                before = {
                    path: path.read_bytes()
                    for path in (candidate_path, report_path, ledger_path)
                }

                with self.assertRaisesRegex(ValueError, "execution revision"):
                    finalize_tuning.finalize(
                        candidate_path=candidate_path,
                        report_path=report_path,
                        ledger_path=ledger_path,
                        run_id="001",
                        task_name="autoresearch-baseline",
                    )

                for path, content in before.items():
                    self.assertEqual(path.read_bytes(), content)

    def test_rejected_primary_then_successful_fallback_is_finalizable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"] = [
                {"method": "grid", "status": "rejected", "trials": []},
                {
                    "method": "bo",
                    "status": "ok",
                    "trials": [{"params": {"x": 2.0}, "score": 0.8}],
                },
            ]
            report_path.write_text(json.dumps(report, indent=2))

            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )

            self.assertEqual(result["final_best_score"], 0.8)
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["phase_c_method"], "bo")

    def test_exhausted_rejected_chain_closes_to_warm_incumbent(self) -> None:
        report = _report(stage_status="rejected")
        report["phase_c"]["stages"] = [
            {"method": "grid", "status": "rejected", "trials": []},
            {"method": "bo", "status": "rejected", "trials": []},
        ]

        result = finalize_tuning.finalizable_tuning_result(report)

        self.assertEqual(result["best_score"], 1.0)
        self.assertIsNone(result["phase_c_method"])

    def test_fixed_space_no_search_receipt_is_finalizable(self) -> None:
        params = {"a": 1.0, "b": 2.0, "c": "only"}
        report = {
            "phase_a": {
                "status": "ok",
                "warm_start_configs": [{"params": params, "score": 0.5}],
                "best_warm_params": params,
                "best_warm_score": 0.5,
                "search_space": {
                    "a": ["float", 0.0, 2.0],
                    "b": ["float", 1.0, 3.0],
                    "c": ["categorical", ["only", "other"]],
                },
            },
            "phase_c": {
                "stages": [
                    {"method": "bo", "status": "rejected", "trials": []},
                    {
                        "method": "cmaes",
                        "status": "no_search_needed",
                        "trials": [],
                        "trials_attempted": 0,
                        "trials_completed": 0,
                        "fixed_search_space": True,
                        "early_stop_reason": "fixed_search_space",
                        "fixed_incumbent_params": params,
                        "fixed_incumbent_score": 0.5,
                        "effective_search_space": {
                            "a": ["float", 1.0, 1.0],
                            "b": ["float", 2.0, 2.0],
                            "c": ["categorical", ["only"]],
                        },
                    },
                ]
            },
        }

        result = finalize_tuning.finalizable_tuning_result(report)

        self.assertEqual(result["best_params"], params)
        self.assertEqual(result["best_score"], 0.5)
        self.assertIsNone(result["phase_c_method"])

    def test_time_exhausted_is_a_proven_terminal_boundary(self) -> None:
        report = _report(stage_status="time_exhausted")
        stage = report["phase_c"]["stages"][0]
        stage.update(
            {
                "early_stop_reason": "time_budget",
                "elapsed_seconds": 10.0,
                "time_limit_seconds": 10.0,
            }
        )

        result = finalize_tuning.finalizable_tuning_result(report)

        self.assertEqual(result["best_score"], 0.8)
        self.assertEqual(result["phase_c_method"], "grid")

        stage["elapsed_seconds"] = 1.0
        with self.assertRaisesRegex(ValueError, "cumulative elapsed"):
            finalize_tuning.finalizable_tuning_result(report)

    def test_time_exhaustion_counts_elapsed_across_fallback_stages(self) -> None:
        report = _report(stage_status="rejected")
        report["phase_c"]["stages"] = [
            {
                "method": "grid",
                "status": "rejected",
                "trials": [],
                "elapsed_seconds": 9.5,
            },
            {
                "method": "bo",
                "status": "time_exhausted",
                "trials": [],
                "elapsed_seconds": 0.5,
                "early_stop_reason": "time_budget",
                "time_limit_seconds": 10.0,
            },
        ]

        result = finalize_tuning.finalizable_tuning_result(report)

        self.assertEqual(result["best_score"], 1.0)
        self.assertIsNone(result["phase_c_method"])

    def test_terminal_stage_closes_once_and_reconciles_strict_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            attempt_rows = [
                {
                    "schema_version": 1,
                    "kind": "score_attempt",
                    "attempt_id": f"eval-{index:06d}",
                    "run_id": "001",
                }
                for index in range(1, 4)
            ]
            (ledger_path.parent / "evaluation_attempts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in attempt_rows)
            )

            first = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            after_first = {
                path: path.read_bytes()
                for path in (candidate_path, report_path, ledger_path)
            }
            second = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )

            self.assertEqual(first, second)
            for path, content in after_first.items():
                self.assertEqual(path.read_bytes(), content)

            module = ast.parse(candidate_path.read_text())
            self.assertFalse(candidate_path.with_suffix(".py.tmp").exists())
            assignment = next(
                node
                for node in module.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "BASE_PARAMS"
                    for target in node.targets
                )
            )
            self.assertEqual(ast.literal_eval(assignment.value), {"x": 2.0})

            closed_report = json.loads(report_path.read_text())
            self.assertEqual(closed_report["final_best_params"], {"x": 2.0})
            self.assertEqual(closed_report["final_best_score"], 0.8)
            self.assertTrue(closed_report["applied_to_base_params"])

            stored = json.loads(ledger_path.read_text())
            record = stored["records"][0]
            self.assertTrue(record["tune"])
            self.assertEqual(record["status"], "keep")
            self.assertEqual(record["final_best_score"], 0.8)
            self.assertEqual(record["phase_c_method"], "grid")
            self.assertEqual(record["trials_completed"], 2)
            self.assertEqual(record["trials_attempted"], 3)
            self.assertEqual(record["elapsed_seconds"], 3.5)
            self.assertTrue(record["applied"])
            self.assertEqual(stored["dag_revision"], 2)

    def test_invalid_run_state_is_rejected_before_mutation(self) -> None:
        """A corrupt ledger target or attempt log stops before any write."""

        def missing_record(ledger_path: Path) -> None:
            ledger_data = json.loads(ledger_path.read_text())
            ledger_data["records"] = []
            ledger_path.write_text(json.dumps(ledger_data, indent=2))

        def corrupt_attempt_log(ledger_path: Path) -> None:
            (ledger_path.parent / "evaluation_attempts.jsonl").write_text("{broken\n")

        cases = [
            ("no record for run_id", missing_record),
            ("invalid evaluation_attempts.jsonl", corrupt_attempt_log),
        ]
        for needle, corrupt in cases:
            with self.subTest(needle=needle):
                with tempfile.TemporaryDirectory() as tmp:
                    candidate_path, report_path, ledger_path = self._fixture(
                        Path(tmp),
                        stage_status="ok",
                    )
                    corrupt(ledger_path)
                    before_candidate = candidate_path.read_bytes()
                    before_report = report_path.read_bytes()

                    with self.assertRaisesRegex(ValueError, needle):
                        finalize_tuning.finalize(
                            candidate_path=candidate_path,
                            report_path=report_path,
                            ledger_path=ledger_path,
                            run_id="001",
                            task_name="autoresearch-baseline",
                        )

                    self.assertEqual(candidate_path.read_bytes(), before_candidate)
                    self.assertEqual(report_path.read_bytes(), before_report)

    def test_ledger_failure_rolls_back_candidate_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            before = {
                path: path.read_bytes()
                for path in (candidate_path, report_path, ledger_path)
            }

            with mock.patch.object(
                finalize_tuning.ledger,
                "finalize_tuning",
                side_effect=OSError("simulated ledger write failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated ledger write failure"):
                    finalize_tuning.finalize(
                        candidate_path=candidate_path,
                        report_path=report_path,
                        ledger_path=ledger_path,
                        run_id="001",
                        task_name="autoresearch-baseline",
                    )

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
            self.assertFalse(
                (candidate_path.parent / finalize_tuning.RECOVERY_JOURNAL_FILENAME).exists()
            )

    def test_interrupted_recovery_journal_restores_preclose_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            candidate_before = candidate_path.read_bytes()
            report_before = report_path.read_bytes()
            journal_path = (
                candidate_path.parent / finalize_tuning.RECOVERY_JOURNAL_FILENAME
            )
            finalize_tuning._write_recovery_journal(
                journal_path,
                run_id="001",
                candidate_path=candidate_path.resolve(),
                report_path=report_path.resolve(),
                candidate_before=candidate_before,
                report_before=report_before,
            )
            candidate_path.write_text(candidate_path.read_text().replace("1.0", "2.0"))
            closed = json.loads(report_path.read_text())
            closed["final_best_params"] = {"x": 2.0}
            closed["final_best_score"] = 0.8
            closed["applied_to_base_params"] = True
            report_path.write_text(json.dumps(closed))

            finalize_tuning._recover_interrupted_finalization(
                journal_path,
                ledger_path=ledger_path.resolve(),
                candidate_path=candidate_path.resolve(),
                report_path=report_path.resolve(),
                run_id="001",
            )

            self.assertEqual(candidate_path.read_bytes(), candidate_before)
            self.assertEqual(report_path.read_bytes(), report_before)
            self.assertFalse(journal_path.exists())

    def test_post_ledger_bookkeeping_failure_keeps_committed_forward_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            candidate_path.write_text(
                "PARAM_SCHEMA = {'x': ('float', 'log')}\n"
                "SEARCH_SPACE = {'x': ('float', 0.1, 2.0, 'log')}\n"
                "BASE_PARAMS = {'x': 1.0}\n"
                "def make_model(params):\n"
                "    return params\n"
            )
            report = json.loads(report_path.read_text())
            report["phase_a"]["search_space"] = {
                "x": ["float", 0.1, 2.0, "log"]
            }
            report["phase_a"]["candidate_code_revision"] = (
                _candidate_execution_revision(candidate_path)
            )
            report_path.write_text(json.dumps(report, indent=2))

            with mock.patch.object(
                finalize_tuning.ledger,
                "_write_loop_state",
                side_effect=OSError("simulated loop-state failure"),
            ):
                with self.assertRaisesRegex(OSError, "simulated loop-state failure"):
                    finalize_tuning.finalize(
                        candidate_path=candidate_path,
                        report_path=report_path,
                        ledger_path=ledger_path,
                        run_id="001",
                        task_name="autoresearch-baseline",
                    )

            report = json.loads(report_path.read_text())
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertTrue(report["applied_to_base_params"])
            self.assertTrue(record["tune"])
            self.assertEqual(record["final_best_score"], 0.8)
            self.assertEqual(
                record["applied_incumbent"]["param_schema"],
                {"x": ["float", "log"]},
            )
            self.assertNotIn("params_sha256", record["applied_incumbent"])
            self.assertNotIn("param_schema_sha256", record["applied_incumbent"])
            self.assertTrue(
                finalize_tuning._ledger_committed_final_state(
                    ledger_path.resolve(),
                    report_path.resolve(),
                    "001",
                )
            )
            self.assertFalse(
                (candidate_path.parent / finalize_tuning.RECOVERY_JOURNAL_FILENAME).exists()
            )

    def test_cross_run_ledger_is_rejected_before_any_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path, report_path, own_ledger = self._fixture(
                root / "run-a",
                stage_status="ok",
            )
            _, _, foreign_ledger = self._fixture(
                root / "run-b",
                stage_status="ok",
            )
            before = {
                path: path.read_bytes()
                for path in (
                    candidate_path,
                    report_path,
                    own_ledger,
                    foreign_ledger,
                )
            }

            with self.assertRaisesRegex(
                ValueError, "ledger.json owned by the candidate"
            ):
                finalize_tuning.finalize(
                    candidate_path=candidate_path,
                    report_path=report_path,
                    ledger_path=foreign_ledger,
                    run_id="001",
                    task_name="autoresearch-baseline",
                )

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_cross_run_phase_a_report_is_rejected_by_ledger_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_path, report_path, own_ledger = self._fixture(
                root / "run-a",
                stage_status="ok",
            )
            _, _, foreign_ledger = self._fixture(
                root / "run-b",
                stage_status="ok",
            )
            before = {
                path: path.read_bytes()
                for path in (
                    candidate_path,
                    report_path,
                    own_ledger,
                    foreign_ledger,
                )
            }

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "set-tuning",
                    "--ledger",
                    str(foreign_ledger),
                    "--task",
                    "autoresearch-baseline",
                    "--run-id",
                    "001",
                    "--from-report",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("candidate-owned", result.stderr)
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_set_tuning_from_report_replaces_every_tuning_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"] = []
            report_path.write_text(json.dumps(report, indent=2))

            data = json.loads(ledger_path.read_text())
            data["records"][0].update(
                {
                    "warm_percentile": 99,
                    "phase_b_decision": "continue",
                    "phase_c_method": "bo",
                    "applied": True,
                    "parameter_transfer": {"stale": True},
                    "applied_incumbent": {"stale": True},
                }
            )
            ledger_path.write_text(json.dumps(data, indent=2))
            expected = finalize_tuning.ledger._tuning_record_from_report(
                report_path
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "set-tuning",
                    "--ledger",
                    str(ledger_path),
                    "--task",
                    "autoresearch-baseline",
                    "--run-id",
                    "001",
                    "--from-report",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            stored = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(
                {
                    key: stored.get(key)
                    for key in finalize_tuning.ledger.TUNING_FIELDS
                },
                expected,
            )
            self.assertIsNone(stored["warm_percentile"])
            self.assertIsNone(stored["phase_b_decision"])
            self.assertIsNone(stored["phase_c_method"])
            self.assertIsNone(stored["applied"])
            self.assertIsNone(stored["parameter_transfer"])

    def test_set_tuning_individual_flags_keep_patch_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, _, ledger_path = self._fixture(Path(tmp), stage_status="ok")
            data = json.loads(ledger_path.read_text())
            data["records"][0].update(
                {
                    "phase_b_decision": "continue",
                    "phase_c_method": "bo",
                    "applied": True,
                }
            )
            ledger_path.write_text(json.dumps(data, indent=2))

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "set-tuning",
                    "--ledger",
                    str(ledger_path),
                    "--task",
                    "autoresearch-baseline",
                    "--run-id",
                    "001",
                    "--best-warm-score",
                    "2.5",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            stored = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(stored["best_warm_score"], 2.5)
            self.assertEqual(stored["phase_b_decision"], "continue")
            self.assertEqual(stored["phase_c_method"], "bo")
            self.assertTrue(stored["applied"])

    def test_missing_transfer_is_rejected_before_durable_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            data = json.loads(ledger_path.read_text())
            data["records"][0].update(
                {
                    "op": "improve",
                    "source_run_ids": ["000"],
                    "policy_receipt": {"schema_version": 6},
                    "parameter_transfer": None,
                }
            )
            ledger_path.write_text(json.dumps(data, indent=2))
            before = {
                path: path.read_bytes()
                for path in (candidate_path, report_path, ledger_path)
            }

            with self.assertRaisesRegex(ValueError, "parameter_transfer is required"):
                finalize_tuning.finalize(
                    candidate_path=candidate_path,
                    report_path=report_path,
                    ledger_path=ledger_path,
                    run_id="001",
                    task_name="autoresearch-baseline",
                )

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_report_authored_paired_control_cannot_enter_through_public_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            report = json.loads(report_path.read_text())
            report["phase_a"]["parameter_transfer"] = {
                "schema_version": 2,
                "semantic_control": {"status": "paired"},
            }
            report_path.write_text(json.dumps(report, indent=2))
            before = {
                path: path.read_bytes()
                for path in (candidate_path, report_path, ledger_path)
            }

            with self.assertRaisesRegex(
                ValueError,
                "report-authored paired semantic controls are not admissible",
            ):
                finalize_tuning.finalize(
                    candidate_path=candidate_path,
                    report_path=report_path,
                    ledger_path=ledger_path,
                    run_id="001",
                    task_name="autoresearch-baseline",
                )
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

            phase_a = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "set-tuning",
                    "--ledger",
                    str(ledger_path),
                    "--task",
                    "autoresearch-baseline",
                    "--run-id",
                    "001",
                    "--from-report",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(phase_a.returncode, 0)
            self.assertIn(
                "Phase-A tuning updates cannot consume a report with Phase-C stages",
                phase_a.stderr,
            )
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_parent_cannot_be_retuned_while_child_binding_is_in_flight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            data = json.loads(ledger_path.read_text())
            child = _record()
            child.update(
                {
                    "run_id": "002",
                    "source_run_ids": ["001"],
                    "status": "pending",
                    "final_best_score": None,
                }
            )
            data["records"].append(child)
            ledger_path.write_text(json.dumps(data, indent=2))
            before = {
                path: path.read_bytes()
                for path in (candidate_path, report_path, ledger_path)
            }

            with self.assertRaisesRegex(ValueError, "in-flight or invalid"):
                finalize_tuning.finalize(
                    candidate_path=candidate_path,
                    report_path=report_path,
                    ledger_path=ledger_path,
                    run_id="001",
                    task_name="autoresearch-baseline",
                )

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_malformed_scored_child_binding_blocks_parent_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp),
                stage_status="ok",
            )
            data = json.loads(ledger_path.read_text())
            parent = data["records"][0]
            child = _record()
            child.update(
                {
                    "run_id": "002",
                    "source_run_ids": ["001"],
                    "op": "improve",
                    "status": "discard",
                    "final_best_score": 0.9,
                    "parameter_transfer": {
                        "receipt": {
                            "primary_parent": {
                                "run_id": "001",
                                "ledger_record_sha256": _json_sha256(parent),
                            }
                        }
                    },
                }
            )
            data["records"].append(child)
            ledger_path.write_text(json.dumps(data, indent=2))
            before = {
                path: path.read_bytes()
                for path in (candidate_path, report_path, ledger_path)
            }

            with self.assertRaisesRegex(ValueError, "in-flight or invalid"):
                finalize_tuning.finalize(
                    candidate_path=candidate_path,
                    report_path=report_path,
                    ledger_path=ledger_path,
                    run_id="001",
                    task_name="autoresearch-baseline",
                )

            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)

    def test_no_binding_terminal_child_does_not_freeze_parent(self) -> None:
        for child_status in ("crash", "unevaluated"):
            with self.subTest(child_status=child_status):
                with tempfile.TemporaryDirectory() as tmp:
                    candidate_path, report_path, ledger_path = self._fixture(
                        Path(tmp),
                        stage_status="ok",
                    )
                    data = json.loads(ledger_path.read_text())
                    child = _record()
                    child.update(
                        {
                            "run_id": "002",
                            "source_run_ids": ["001"],
                            "status": child_status,
                            "final_best_score": (
                                float("inf") if child_status == "crash" else None
                            ),
                            "parameter_transfer": None,
                        }
                    )
                    data["records"].append(child)
                    ledger_path.write_text(json.dumps(data, indent=2))

                    result = finalize_tuning.finalize(
                        candidate_path=candidate_path,
                        report_path=report_path,
                        ledger_path=ledger_path,
                        run_id="001",
                        task_name="autoresearch-baseline",
                    )

                    self.assertEqual(result["final_best_score"], 0.8)
                    stored = json.loads(ledger_path.read_text())
                    self.assertTrue(stored["records"][0]["tune"])


class LedgerProgressiveFieldsTest(unittest.TestCase):
    def test_new_record_carries_progressive_defaults(self):
        record = ledger._new_record("001")
        self.assertEqual(record["tuning_bouts"], 0)
        self.assertIsNone(record["last_bout_improved"])

    def test_legacy_records_normalize_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            legacy = {
                "run_id": "001",
                "semantic_point": {},
                "policy_receipt": {},
                "status": "keep",
                "tune": True,
                "final_best_score": 0.9,
            }
            fresh = {
                "run_id": "002",
                "semantic_point": {},
                "policy_receipt": {},
                "status": "keep",
                "tune": False,
                "final_best_score": 1.1,
            }
            ledger_path.write_text(json.dumps({
                "task": "autoresearch-baseline",
                "tag": "test",
                "metric": "val_bpb",
                "search_space_state": empty_search_space_state(),
                "records": [legacy, fresh],
            }))
            data = ledger._load_ledger(ledger_path)
            by_id = {r["run_id"]: r for r in data["records"]}
            self.assertEqual(by_id["001"]["tuning_bouts"], 1)
            self.assertIsNone(by_id["001"]["last_bout_improved"])
            self.assertEqual(by_id["002"]["tuning_bouts"], 0)
            self.assertIsNone(by_id["002"]["last_bout_improved"])


if __name__ == "__main__":
    unittest.main()
