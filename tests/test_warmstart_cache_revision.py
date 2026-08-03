from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import warmstart_eval  # noqa: E402
import evaluation_budget  # noqa: E402


class WarmstartCacheRevisionTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        provided: bool,
    ) -> tuple[Path, Path, Path]:
        candidate = root / "candidates" / "001" / "train.py"
        candidate.parent.mkdir(parents=True)
        defaults = "DEFAULT_PARAMS = {'x': 1}\n" if provided else ""
        candidate.write_text(
            (
                defaults
                + """
BEHAVIOR = 0
PARAM_SCHEMA = {"x": "int"}
SEARCH_SPACE = {"x": ("int", 1, 2)}

def make_model(params):
    return params["x"] + BEHAVIOR
""".lstrip()
            )
        )
        (candidate.parent / "prepare.py").write_text(
            """
def evaluate_config(make_model, params):
    return float(make_model(params))
""".lstrip()
        )
        if provided:
            brief = {
                "schema_version": 3,
                "implementation_source": {
                    "kind": "provided_entrypoint",
                    "path": "tasks/unit/train.py",
                    "sha256": "sha256:" + "0" * 64,
                },
            }
            configs = [{"x": 1}]
        else:
            brief = {
                "schema_version": 4,
                "run_id": "001",
                "source_run_ids": [],
                "primary_parent": None,
                "implementation_source": {"kind": "generated"},
            }
            configs = [{"x": 1}, {"x": 2}]
        (candidate.parent / "_candidate_brief.json").write_text(json.dumps(brief))
        configs_path = candidate.parent / "_warm_configs.json"
        configs_path.write_text(json.dumps(configs))
        return candidate, configs_path, candidate.parent / "tune_report.json"

    def _run(
        self,
        candidate: Path,
        configs_path: Path,
        report_path: Path,
    ) -> mock.Mock:
        timed_eval = mock.Mock(
            side_effect=lambda _evaluate, _make_model, params, *_args, **_kwargs: (
                float(params["x"])
            )
        )
        argv = [
            "warmstart_eval.py",
            "--candidate-path",
            str(candidate),
            "--configs-json",
            str(configs_path),
            "--tune-report-json",
            str(report_path),
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(warmstart_eval, "timed_eval", timed_eval),
            mock.patch("sys.stdout", new=io.StringIO()),
        ):
            self.assertEqual(warmstart_eval.main(), 0)
        return timed_eval

    def test_phase_a_crash_receipt_marks_reserved_candidate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]
            output = io.StringIO()
            evaluation_error = RuntimeError("candidate bug")
            evaluation_error.objective_attempt_admitted = True
            evaluation_error.objective_slot_consumed = True
            failure = {
                "error": "RuntimeError: candidate bug",
                "failure_ref": {
                    "schema_version": 1,
                    "failure_id": "fail-unit",
                    "artifact": "_failures/fail-unit.json",
                    "sha256": "sha256:" + "0" * 64,
                },
                "failure_receipt": {
                    "frames": [{"path": str(candidate), "line": 8}],
                },
            }
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    warmstart_eval,
                    "timed_eval",
                    side_effect=evaluation_error,
                ),
                mock.patch.object(
                    warmstart_eval,
                    "record_failure",
                    return_value=failure,
                ),
                mock.patch("sys.stdout", new=output),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                self.assertEqual(warmstart_eval.main(), warmstart_eval.CRASHED)

            receipt = json.loads(output.getvalue())
            self.assertTrue(receipt["objective_slot_consumed"])
            self.assertEqual(
                receipt["failure_category"],
                "candidate_code_incompatibility",
            )
            report = json.loads(report_path.read_text())
            failed = report["phase_a"]["warm_start_configs"][0]
            self.assertEqual(
                failed["failure_category"],
                "candidate_code_incompatibility",
            )

    def test_reservation_failure_does_not_create_a_failed_warm_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]
            record_failure = mock.Mock()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    warmstart_eval,
                    "timed_eval",
                    side_effect=OSError("attempt log unavailable"),
                ),
                mock.patch.object(
                    warmstart_eval,
                    "record_failure",
                    record_failure,
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                with self.assertRaisesRegex(OSError, "attempt log unavailable"):
                    warmstart_eval.main()

            record_failure.assert_not_called()
            phase_a = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(phase_a["trials_attempted"], 0)
            self.assertEqual(phase_a["warm_start_configs"], [])

    def test_standalone_candidate_crash_records_crashed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            # No framework_cfg.json anywhere above the candidate: the real
            # timed_eval takes the unbudgeted standalone path, and a candidate
            # exception must still become the recorded CRASHED receipt.
            (candidate.parent / "prepare.py").write_text(
                """
def evaluate_config(make_model, params):
    raise RuntimeError("candidate bug")
""".lstrip()
            )
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]
            output = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("sys.stdout", new=output),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                self.assertEqual(warmstart_eval.main(), warmstart_eval.CRASHED)

            receipt = json.loads(output.getvalue())
            self.assertEqual(receipt["status"], "crashed")
            self.assertEqual(receipt["phase"], "a")
            self.assertFalse(receipt["objective_slot_consumed"])
            self.assertNotIn("objective_reservation", receipt)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["phase_a"]["status"], "crashed")
            self.assertFalse(
                report["phase_a"]["terminal_failure"]["objective_slot_consumed"]
            )
            self.assertFalse((Path(tmp) / "evaluation_attempts.jsonl").exists())

    def test_preflight_crash_receipt_never_claims_an_objective_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]
            output = io.StringIO()
            failure = {
                "error": "RuntimeError: candidate preflight bug",
                "failure_ref": {
                    "schema_version": 1,
                    "failure_id": "fail-preflight-unit",
                    "artifact": "_failures/fail-preflight-unit.json",
                    "sha256": "sha256:" + "0" * 64,
                },
                "failure_receipt": {
                    "frames": [{"path": str(candidate), "line": 8}],
                },
            }
            timed_eval = mock.Mock()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    warmstart_eval,
                    "resolve_preflight_fn",
                    return_value=object(),
                ),
                mock.patch.object(
                    warmstart_eval,
                    "timed_preflight",
                    side_effect=RuntimeError("candidate preflight bug"),
                ),
                mock.patch.object(warmstart_eval, "timed_eval", timed_eval),
                mock.patch.object(
                    warmstart_eval,
                    "record_failure",
                    return_value=failure,
                ),
                mock.patch("sys.stdout", new=output),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                self.assertEqual(warmstart_eval.main(), warmstart_eval.CRASHED)

            timed_eval.assert_not_called()
            receipt = json.loads(output.getvalue())
            self.assertEqual(receipt["phase"], "preflight")
            self.assertFalse(receipt["objective_slot_consumed"])
            self.assertEqual(
                receipt["failure_category"],
                "candidate_code_incompatibility",
            )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["phase_a"]["status"], "preflight_failed")
            self.assertEqual(report["phase_a"]["trials_attempted"], 0)

    def test_restart_forward_closes_reservation_appended_before_result(self) -> None:
        class SimulatedProcessKill(BaseException):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "interrupted"
            candidate, configs_path, report_path = self._fixture(
                run_dir,
                provided=True,
            )
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 3})
            )
            (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]

            def kill_after_append(
                _evaluate, _make_model, params, candidate_path, **_kwargs
            ):
                evaluation_budget.reserve_evaluation(
                    candidate_path,
                    params=params,
                    phase="phase_a",
                    method="warmstart",
                )
                raise SimulatedProcessKill()

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    warmstart_eval,
                    "timed_eval",
                    side_effect=kill_after_append,
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                with self.assertRaises(SimulatedProcessKill):
                    warmstart_eval.main()

            interrupted = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(interrupted["active_objective"]["status"], "intent_persisted")
            self.assertEqual(interrupted["trials_attempted"], 0)

            score_again = mock.Mock()
            output = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(warmstart_eval, "timed_eval", score_again),
                mock.patch("sys.stdout", new=output),
            ):
                self.assertEqual(warmstart_eval.main(), warmstart_eval.CRASHED)

            score_again.assert_not_called()
            terminal = json.loads(output.getvalue())
            self.assertEqual(terminal["failure_category"], "process_interruption")
            self.assertTrue(terminal["objective_slot_consumed"])
            self.assertEqual(terminal["objective_attempt_id"], "eval-000001")
            report = json.loads(report_path.read_text())
            phase_a = report["phase_a"]
            self.assertNotIn("active_objective", phase_a)
            self.assertEqual(phase_a["trials_attempted"], 1)
            self.assertEqual(phase_a["terminal_failure"], terminal)
            failed = phase_a["warm_start_configs"][0]
            self.assertEqual(failed["objective_attempt_id"], "eval-000001")
            self.assertEqual(
                failed["candidate_execution_revision_sha256"],
                terminal["candidate_execution_revision"]["revision_sha256"],
            )

    def test_second_kill_during_reservation_recovery_cannot_erase_intent(self) -> None:
        class SimulatedProcessKill(BaseException):
            pass

        class SimulatedRecoveryKill(BaseException):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "double-interrupted"
            candidate, configs_path, report_path = self._fixture(
                run_dir,
                provided=True,
            )
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 3})
            )
            (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]

            def kill_after_append(
                _evaluate, _make_model, params, candidate_path, **_kwargs
            ):
                evaluation_budget.reserve_evaluation(
                    candidate_path,
                    params=params,
                    phase="phase_a",
                    method="warmstart",
                )
                raise SimulatedProcessKill()

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    warmstart_eval,
                    "timed_eval",
                    side_effect=kill_after_append,
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                with self.assertRaises(SimulatedProcessKill):
                    warmstart_eval.main()

            real_write = warmstart_eval.write_tune_report

            def kill_after_recovery_projection(path, report):
                real_write(path, report)
                active = report.get("phase_a", {}).get("active_objective")
                if (
                    isinstance(active, dict)
                    and active.get("status") == "reservation_persisted"
                ):
                    raise SimulatedRecoveryKill()

            score_again = mock.Mock()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(warmstart_eval, "timed_eval", score_again),
                mock.patch.object(
                    warmstart_eval,
                    "write_tune_report",
                    side_effect=kill_after_recovery_projection,
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                with self.assertRaises(SimulatedRecoveryKill):
                    warmstart_eval.main()
            score_again.assert_not_called()

            recovered = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(
                recovered["active_objective"]["status"],
                "reservation_persisted",
            )

            output = io.StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(warmstart_eval, "timed_eval", score_again),
                mock.patch("sys.stdout", new=output),
            ):
                self.assertEqual(warmstart_eval.main(), warmstart_eval.CRASHED)

            score_again.assert_not_called()
            terminal = json.loads(output.getvalue())
            self.assertEqual(terminal["failure_category"], "process_interruption")
            self.assertEqual(terminal["objective_attempt_id"], "eval-000001")
            attempts = [
                json.loads(line)
                for line in (run_dir / "evaluation_attempts.jsonl")
                .read_text()
                .splitlines()
            ]
            reservations = [
                row for row in attempts if row.get("kind") == "score_attempt"
            ]
            self.assertEqual(len(reservations), 1)

    def _assert_revision_bound_resume(self, *, provided: bool) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=provided,
            )
            expected_evaluations = 1 if provided else 2

            first_eval = self._run(candidate, configs_path, report_path)
            self.assertEqual(first_eval.call_count, expected_evaluations)
            first_report = json.loads(report_path.read_text())
            first_revision = first_report["phase_a"]["candidate_code_revision"]
            for call in first_eval.call_args_list:
                self.assertEqual(
                    call.kwargs["expected_execution_revision"],
                    first_revision,
                )
            self.assertEqual(first_revision["schema_version"], 3)
            self.assertTrue(first_revision["structure_sha256"].startswith("sha256:"))
            self.assertTrue(
                first_revision["search_space_sha256"].startswith("sha256:")
            )
            self.assertTrue(first_revision["prepare_sha256"].startswith("sha256:"))
            self.assertTrue(first_revision["revision_sha256"].startswith("sha256:"))
            self.assertEqual(
                {
                    row["candidate_structure_sha256"]
                    for row in first_report["phase_a"]["warm_start_configs"]
                },
                {first_revision["structure_sha256"]},
            )
            self.assertEqual(
                {
                    row["candidate_execution_revision_sha256"]
                    for row in first_report["phase_a"]["warm_start_configs"]
                },
                {first_revision["revision_sha256"]},
            )

            resumed_eval = self._run(candidate, configs_path, report_path)
            resumed_eval.assert_not_called()
            resumed_report = json.loads(report_path.read_text())
            self.assertEqual(
                resumed_report["phase_a"]["candidate_code_revision"],
                first_revision,
            )
            self.assertEqual(
                resumed_report["phase_a"]["trials_attempted"],
                expected_evaluations,
            )

            # These observations/closing claims belong to the first execution
            # revision and must not survive a later code/evaluator rewrite.
            resumed_report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "ok",
                        "trials": [{"params": {"x": 1}, "score": -999.0}],
                    }
                ]
            }
            resumed_report["preflight"] = {
                "attempts": [{"status": "ok", "revision": "old"}],
                "invocations": 1,
                "status": "ok",
            }
            resumed_report["final_best_params"] = {"x": 1}
            resumed_report["final_best_score"] = -999.0
            resumed_report["applied_to_base_params"] = True
            report_path.write_text(json.dumps(resumed_report))

            candidate.write_text(
                candidate.read_text().replace("BEHAVIOR = 0", "BEHAVIOR = 100")
            )
            changed_eval = self._run(candidate, configs_path, report_path)
            self.assertEqual(changed_eval.call_count, expected_evaluations)
            changed_report = json.loads(report_path.read_text())
            changed_revision = changed_report["phase_a"]["candidate_code_revision"]
            self.assertNotEqual(changed_revision, first_revision)
            self.assertEqual(
                changed_report["phase_a"]["trials_attempted"],
                2 * expected_evaluations,
            )
            self.assertEqual(
                {
                    row["candidate_structure_sha256"]
                    for row in changed_report["phase_a"]["warm_start_configs"]
                },
                {changed_revision["structure_sha256"]},
            )
            self.assertNotIn("phase_c", changed_report)
            self.assertNotIn("final_best_params", changed_report)
            self.assertNotIn("final_best_score", changed_report)
            self.assertNotIn("applied_to_base_params", changed_report)
            self.assertEqual(
                changed_report["preflight"],
                {"attempts": [], "invocations": 0},
            )

    def test_prepare_change_invalidates_every_cached_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            self.assertEqual(
                self._run(candidate, configs_path, report_path).call_count,
                2,
            )
            first_revision = json.loads(report_path.read_text())["phase_a"][
                "candidate_code_revision"
            ]

            prepare_path = candidate.parent / "prepare.py"
            prepare_path.write_text(
                prepare_path.read_text().replace(
                    "float(make_model(params))",
                    "float(make_model(params)) + 100.0",
                )
            )
            resumed_eval = self._run(candidate, configs_path, report_path)
            second_revision = json.loads(report_path.read_text())["phase_a"][
                "candidate_code_revision"
            ]

            self.assertEqual(resumed_eval.call_count, 2)
            self.assertNotEqual(
                first_revision["prepare_sha256"],
                second_revision["prepare_sha256"],
            )
            self.assertNotEqual(
                first_revision["revision_sha256"],
                second_revision["revision_sha256"],
            )

    def test_fresh_candidate_cache_is_invalidated_by_code_revision(self) -> None:
        self._assert_revision_bound_resume(provided=False)

    def test_provided_candidate_cache_is_invalidated_by_code_revision(self) -> None:
        self._assert_revision_bound_resume(provided=True)

    def test_search_space_change_invalidates_every_cached_score(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            self.assertEqual(
                self._run(candidate, configs_path, report_path).call_count,
                2,
            )
            first_report = json.loads(report_path.read_text())
            first_revision = first_report["phase_a"]["candidate_code_revision"]

            candidate.write_text(
                candidate.read_text().replace(
                    'SEARCH_SPACE = {"x": ("int", 1, 2)}',
                    'SEARCH_SPACE = {"x": ("int", 1, 3)}',
                )
            )
            configs_path.write_text(json.dumps([{"x": 1}, {"x": 3}]))
            resumed_eval = self._run(candidate, configs_path, report_path)

            self.assertEqual(resumed_eval.call_count, 2)
            self.assertEqual(
                [call.args[2] for call in resumed_eval.call_args_list],
                [{"x": 1}, {"x": 3}],
            )
            report = json.loads(report_path.read_text())
            second_revision = report["phase_a"]["candidate_code_revision"]
            self.assertEqual(
                second_revision["structure_sha256"],
                first_revision["structure_sha256"],
            )
            self.assertNotEqual(
                second_revision["search_space_sha256"],
                first_revision["search_space_sha256"],
            )
            self.assertNotEqual(
                second_revision["revision_sha256"],
                first_revision["revision_sha256"],
            )
            self.assertEqual(report["phase_a"]["trials_attempted"], 4)

    def test_invalid_nonzero_config_fails_before_any_persistent_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            configs_path.write_text(json.dumps([{"x": 1}, {"x": "2"}]))
            candidate_before = candidate.read_bytes()
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]
            timed_eval = mock.Mock()

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(warmstart_eval, "timed_eval", timed_eval),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                with self.assertRaises(SystemExit):
                    warmstart_eval.main()

            timed_eval.assert_not_called()
            self.assertEqual(candidate.read_bytes(), candidate_before)
            self.assertFalse(report_path.exists())

    def test_interrupted_cache_replay_preserves_unreplayed_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            self.assertEqual(
                self._run(candidate, configs_path, report_path).call_count,
                2,
            )
            original_write = warmstart_eval.write_tune_report

            def interrupt_after_first_replay(path: Path, report: dict) -> None:
                original_write(path, report)
                phase_a = report.get("phase_a", {})
                if (
                    phase_a.get("status") == "running"
                    and len(phase_a.get("warm_start_configs", [])) == 1
                ):
                    raise RuntimeError("simulated replay interruption")

            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    warmstart_eval,
                    "write_tune_report",
                    side_effect=interrupt_after_first_replay,
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "simulated replay interruption",
                ):
                    warmstart_eval.main()

            interrupted = json.loads(report_path.read_text())["phase_a"]
            self.assertEqual(len(interrupted["warm_start_configs"]), 1)
            self.assertEqual(
                len(interrupted["warm_score_cache"]["rows"]),
                2,
            )

            resumed_eval = self._run(candidate, configs_path, report_path)
            resumed_eval.assert_not_called()

    def test_budget_exhaustion_recovers_cached_suffix_before_closing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate, configs_path, report_path = self._fixture(
                Path(tmp),
                provided=False,
            )
            candidate.write_text(
                candidate.read_text().replace(
                    'SEARCH_SPACE = {"x": ("int", 1, 2)}',
                    'SEARCH_SPACE = {"x": ("int", 1, 3)}',
                )
            )
            self.assertEqual(
                self._run(candidate, configs_path, report_path).call_count,
                2,
            )
            configs_path.write_text(json.dumps([{"x": 3}, {"x": 2}]))
            exhausted = warmstart_eval.EvaluationBudgetExhausted(
                used=2,
                budget=2,
                run_dir=candidate.parent.parent,
            )
            timed_eval = mock.Mock(side_effect=exhausted)
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs_path),
                "--tune-report-json",
                str(report_path),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(warmstart_eval, "timed_eval", timed_eval),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertEqual(warmstart_eval.main(), 0)

            timed_eval.assert_called_once()
            report = json.loads(report_path.read_text())
            phase_a = report["phase_a"]
            self.assertEqual(phase_a["status"], "ok")
            self.assertTrue(phase_a["budget_exhausted"])
            self.assertEqual(phase_a["best_warm_params"], {"x": 2})
            self.assertEqual(
                [row["params"] for row in phase_a["warm_start_configs"]],
                [{"x": 2}],
            )
            recovered = phase_a["warm_start_configs"][0]
            self.assertEqual(recovered["proposed_index"], 1)
            self.assertEqual(
                recovered["candidate_execution_revision_sha256"],
                phase_a["candidate_code_revision"]["revision_sha256"],
            )
            self.assertEqual(
                phase_a["deferred_configs"],
                [{"params": {"x": 3}}],
            )


if __name__ == "__main__":
    unittest.main()
