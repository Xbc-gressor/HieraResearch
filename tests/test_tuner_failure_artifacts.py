from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

from failure_artifacts import record_failure, render_failure  # noqa: E402
from _common import is_finite_score, read_prior_trials, timed_eval  # noqa: E402
from tune_tools import select_best, select_candidate, summarize  # noqa: E402


TRACEBACK = """Traceback (most recent call last):
  File \"/tmp/run/candidates/007/train.py\", line 41, in make_model
    return build(depth)
  File \"/usr/lib/example.py\", line 9, in build
    raise ValueError(\"depth must be positive\")
ValueError: depth must be positive
"""


class FailureArtifactTests(unittest.TestCase):
    def test_same_failure_is_stable_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "tune_report.json"
            report_path.write_text("{}")
            kwargs = {
                "report_path": report_path,
                "candidate_path": Path(tmp) / "train.py",
                "phase": "phase_a",
                "method": "warmstart",
                "params": {"depth": 0},
                "error": ValueError("depth must be positive"),
                "traceback_text": TRACEBACK,
            }

            first = record_failure(**kwargs)
            artifact = report_path.parent / first["failure_ref"]["artifact"]
            original_bytes = artifact.read_bytes()
            second = record_failure(**kwargs)

            self.assertEqual(first, second)
            self.assertEqual(artifact.read_bytes(), original_bytes)
            self.assertEqual(len(list((report_path.parent / "_failures").glob("*.json"))), 1)
            self.assertRegex(first["failure_ref"]["failure_id"], r"^fail-[0-9a-f]{16}$")
            self.assertRegex(first["failure_ref"]["sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(
                first["failure_receipt"]["content_sha256"], r"^sha256:[0-9a-f]{64}$"
            )

    def test_trial_fields_are_compact_and_full_traceback_is_retrievable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "tune_report.json"
            report_path.write_text("{}")
            failure = record_failure(
                report_path=report_path,
                candidate_path=Path("/tmp/run/candidates/007/train.py"),
                phase="phase_a",
                method="warmstart",
                params={"depth": 0},
                error=ValueError("depth must be positive"),
                traceback_text=TRACEBACK,
            )

            self.assertNotIn("error_traceback", failure)
            self.assertEqual(failure["failure_receipt"]["exception"], "ValueError: depth must be positive")
            self.assertEqual(
                [frame["traceback_line"] for frame in failure["failure_receipt"]["frames"]],
                [2, 4],
            )
            self.assertEqual(failure["failure_receipt"]["retained_traceback_lines"], 5)
            self.assertEqual(failure["failure_receipt"]["omitted_traceback_lines"], 1)
            self.assertEqual(
                render_failure(report_path, failure["failure_ref"]["failure_id"], view="full"),
                TRACEBACK,
            )

    def test_legacy_inline_traceback_remains_compatible(self) -> None:
        report = {
            "phase_a": {
                "warm_start_configs": [
                    {
                        "params": {"depth": 0},
                        "score": None,
                        "status": "failed",
                        "error": "ValueError: depth must be positive",
                        "error_traceback": TRACEBACK,
                    },
                    {"params": {"depth": 2}, "score": 0.4},
                ],
                "elapsed_seconds": 1.5,
            }
        }

        self.assertEqual(select_best(report)["best_score"], 0.4)
        self.assertEqual(summarize(report)["trials_completed"], 1)
        self.assertEqual(summarize(report)["trials_attempted"], 2)

    def test_non_finite_scores_are_not_successful_trials(self) -> None:
        report = {
            "phase_a": {
                "warm_start_configs": [
                    {"params": {"depth": 1}, "score": float("inf")},
                    {"params": {"depth": 2}, "score": float("nan")},
                    {"params": {"depth": 3}, "score": 0.4},
                ],
                "best_warm_score": float("inf"),
            },
            "phase_c": {
                "stages": [
                    {
                        "method": "bo",
                        "trials": [
                            {"params": {"depth": 4}, "score": float("-inf")},
                        ],
                    }
                ]
            },
        }

        self.assertFalse(is_finite_score(float("inf")))
        self.assertFalse(is_finite_score(float("nan")))
        self.assertEqual(select_best(report)["best_score"], 0.4)
        self.assertIsNone(summarize(report)["best_warm_score"])
        self.assertEqual(summarize(report)["trials_completed"], 1)
        self.assertEqual(summarize(report)["trials_attempted"], 4)
        self.assertIsNone(
            select_candidate(
                {
                    "records": [
                        {
                            "run_id": "000",
                            "status": "keep",
                            "best_warm_score": float("inf"),
                            "tune": False,
                        }
                    ]
                },
                n_min=1,
                top_percentile=0,
            )["run_id"]
        )

        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "tune_report.json"
            report_path.write_text(json.dumps(report))
            self.assertEqual(
                read_prior_trials(report_path),
                [{"params": {"depth": 3}, "score": 0.4}],
            )

    def test_attempt_summary_preserves_warm_retries_and_phase_c_failures(self) -> None:
        report = {
            "phase_a": {
                "warm_start_configs": [{"params": {"depth": 2}, "score": 0.4}],
                "trials_attempted": 3,
            },
            "phase_c": {
                "stages": [{
                    "method": "bo",
                    "trials": [
                        {"params": {"depth": 3}, "score": None, "status": "failed"},
                        {"params": {"depth": 4}, "score": 0.3},
                    ],
                }],
            },
        }

        summary = summarize(report)
        self.assertEqual(summary["trials_completed"], 2)
        self.assertEqual(summary["trials_attempted"], 5)

    def test_timed_eval_rejects_non_finite_in_process_result(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite score"):
            timed_eval(
                lambda make_model, params: float("inf"),
                object(),
                {},
                Path("/tmp/no-framework-config/candidate.py"),
            )

    @mock.patch("_common.read_runtime_limit", return_value=5)
    @mock.patch("_common.subprocess.Popen")
    def test_timed_eval_surfaces_child_process_error(self, popen, _read_limit) -> None:
        process = popen.return_value
        process.communicate.return_value = ("training output", "ValueError: child failed")
        process.returncode = 1

        with self.assertRaisesRegex(RuntimeError, "child failed"):
            timed_eval(object(), object(), {}, Path("/tmp/candidate.py"))

    @mock.patch("_common.read_runtime_limit", return_value=5)
    @mock.patch("_common.subprocess.Popen")
    @mock.patch("_common.os.killpg")
    @mock.patch("_common.os.getpgid", return_value=1234)
    def test_timed_eval_surfaces_timeout(
        self, _getpgid, _killpg, popen, _read_limit
    ) -> None:
        process = popen.return_value
        process.pid = 1234
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="eval", timeout=5),
            ("", ""),
        ]

        with self.assertRaisesRegex(TimeoutError, "per_runtime_limit=5s"):
            timed_eval(object(), object(), {}, Path("/tmp/candidate.py"))


if __name__ == "__main__":
    unittest.main()
