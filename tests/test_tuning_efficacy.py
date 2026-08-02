from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluation_budget import attempt_log_summary  # noqa: E402
import tuning_efficacy  # noqa: E402


class TuningEfficacyTests(unittest.TestCase):
    def _run_dir(self, root: Path) -> Path:
        run_dir = root / "run"
        run_dir.mkdir()
        (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
        return run_dir

    def test_attempt_summary_uses_canonical_baseline_and_sync_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(Path(tmp))
            rows = [
                {
                    "schema_version": 1,
                    "kind": "baseline",
                    "per_candidate": {"000": 2},
                },
                {
                    "schema_version": 1,
                    "kind": "sync",
                    "per_candidate": {"001": 1},
                },
                {
                    "schema_version": 1,
                    "kind": "score_attempt",
                    "run_id": "001",
                    "phase": "phase_a",
                },
                {
                    "schema_version": 1,
                    "kind": "score_attempt",
                    "run_id": "001",
                },
            ]
            (run_dir / "evaluation_attempts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

            summary = attempt_log_summary(run_dir)
            self.assertIsNotNone(summary)
            self.assertEqual(summary["evaluations_done"], 5)
            self.assertEqual(summary["phase_counts"], {"phase_a": 1})
            self.assertEqual(summary["carried_evaluations"], 3)
            self.assertEqual(summary["unclassified_score_attempts"], 1)

            output = io.StringIO()
            with redirect_stdout(output):
                result = tuning_efficacy.main(["tuning_efficacy.py", str(run_dir)])
            self.assertEqual(result, 0)
            self.assertIn("admitted evaluations: 5", output.getvalue())
            self.assertIn("baseline/sync=3", output.getvalue())
            self.assertIn("unclassified_score_attempts=1", output.getvalue())

    def test_malformed_attempt_log_fails_visibly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(Path(tmp))
            (run_dir / "evaluation_attempts.jsonl").write_text("{broken\n")

            error = io.StringIO()
            with redirect_stderr(error):
                result = tuning_efficacy.main(["tuning_efficacy.py", str(run_dir)])
            self.assertEqual(result, 2)
            self.assertIn("invalid evaluation_attempts.jsonl line 1", error.getvalue())

    def test_malformed_tune_report_fails_visibly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(Path(tmp))
            report = run_dir / "candidates" / "000" / "tune_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{broken\n")

            error = io.StringIO()
            with redirect_stderr(error), redirect_stdout(io.StringIO()):
                result = tuning_efficacy.main(["tuning_efficacy.py", str(run_dir)])
            self.assertEqual(result, 2)
            self.assertIn(str(report), error.getvalue())


if __name__ == "__main__":
    unittest.main()
