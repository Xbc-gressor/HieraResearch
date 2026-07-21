from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

from failure_artifacts import record_failure, render_failure  # noqa: E402
from tune_tools import select_best, summarize  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
