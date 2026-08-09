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
            self.assertNotIn(
                "cache_sha256",
                first_report["phase_a"]["warm_score_cache"],
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

            # Reports produced before the simplification remain resumable; the
            # now-redundant field is tolerated and omitted on the next write.
            first_report["phase_a"]["warm_score_cache"]["cache_sha256"] = (
                "sha256:legacy"
            )
            report_path.write_text(json.dumps(first_report))
            resumed_eval = self._run(candidate, configs_path, report_path)
            resumed_eval.assert_not_called()
            resumed_report = json.loads(report_path.read_text())
            self.assertNotIn(
                "cache_sha256",
                resumed_report["phase_a"]["warm_score_cache"],
            )
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
