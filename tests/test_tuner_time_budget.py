from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TUNERS = ROOT / "tools" / "tuners"
sys.path.insert(0, str(TUNERS))

from _common import (  # noqa: E402
    EvaluationFailure,
    EvaluationTimeout,
    read_global_eval_budget,
    read_search_wall_limit,
    timed_eval,
)
from tune_tools import select_best, summarize  # noqa: E402


TRAIN = """\
BASE_PARAMS = {"delay": 0.0}
SEARCH_SPACE = {"delay": ("float", 0.0, 2.0)}
def make_model(dataset, params):
    return params
"""

PREPARE = """\
import time
def evaluate_config(make_model, params):
    time.sleep(params.get("delay", 0.0))
    if params.get("fail"):
        raise RuntimeError("fixture failure")
    return params.get("score", -0.5)
"""


class TunerTimeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "runs" / "tabular-blind" / "test-run"
        self.candidate = self.run_dir / "candidates" / "000"
        self.candidate.mkdir(parents=True)
        (self.candidate / "train.py").write_text(TRAIN)
        (self.candidate / "prepare.py").write_text(PREPARE)
        (self.run_dir / "framework_cfg.json").write_text(json.dumps({
            "per_runtime_limit": 0.15,
            "tuner": {"search_wall_seconds": 7},
        }))

    def tearDown(self):
        self.tmp.cleanup()

    def test_timeout_is_typed_failure_not_infinity(self):
        started = time.monotonic()
        with self.assertRaises(EvaluationTimeout) as caught:
            timed_eval(None, None, {"delay": 1.0}, self.candidate / "train.py")
        self.assertEqual(caught.exception.status, "timeout")
        self.assertLess(time.monotonic() - started, 2.0)

    def test_search_wall_override_is_deterministic(self):
        self.assertEqual(read_search_wall_limit(self.candidate / "train.py"), 7)

    def test_global_budget_reports_ledger_remaining(self):
        cfg = json.loads((self.run_dir / "framework_cfg.json").read_text())
        cfg["max_evaluations"] = 10
        (self.run_dir / "framework_cfg.json").write_text(json.dumps(cfg))
        (self.run_dir / "ledger.json").write_text(json.dumps({"records": [
            {"run_id": "000", "trials_completed": 3},
            {"run_id": "001", "trials_completed": 4},
        ]}))
        self.assertEqual(
            read_global_eval_budget(self.candidate / "train.py"),
            {"maximum": 10, "used": 7, "remaining": 3},
        )

    def test_child_error_is_typed_failure_not_infinity(self):
        with self.assertRaises(EvaluationFailure) as caught:
            timed_eval(None, None, {"delay": 0.0, "fail": True}, self.candidate / "train.py")
        self.assertEqual(caught.exception.status, "error")
        self.assertIn("fixture failure", str(caught.exception))

    def test_warmstart_timeout_is_crash_and_not_cacheable(self):
        configs = self.candidate / "_warm_configs.json"
        report = self.candidate / "tune_report.json"
        configs.write_text(json.dumps([{"delay": 1.0}]))
        proc = subprocess.run(
            [
                sys.executable,
                str(TUNERS / "warmstart_eval.py"),
                "--candidate-path", str(self.candidate / "train.py"),
                "--configs-json", str(configs),
                "--tune-report-json", str(report),
                "--k-eval", "1",
            ],
            text=True,
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["failure_status"], "timeout")
        phase_a = json.loads(report.read_text())["phase_a"]
        self.assertEqual(phase_a["status"], "crashed")
        self.assertIsNone(phase_a["warm_start_configs"][0]["score"])
        self.assertEqual(phase_a["warm_start_configs"][0]["status"], "timeout")

    def test_reducer_ignores_nonfinite_scores_but_counts_attempts(self):
        report = {
            "phase_a": {
                "warm_start_configs": [
                    {"params": {"x": 1}, "score": -0.6},
                    {"params": {"x": 2}, "score": float("inf")},
                ],
                "best_warm_score": -0.6,
            },
            "phase_c": {"stages": [{"method": "bo", "trials": [
                {"params": {"x": 3}, "score": None, "status": "timeout"},
                {"params": {"x": 4}, "score": -0.7},
            ]}]},
        }
        self.assertEqual(select_best(report)["best_score"], -0.7)
        self.assertEqual(summarize(report)["trials_completed"], 4)

    def test_bo_partial_batches_resume_to_total_target(self):
        (self.run_dir / "framework_cfg.json").write_text(json.dumps({
            "per_runtime_limit": 1,
            "tuner": {"search_wall_seconds": 0.08, "max_consecutive_failures": 5},
        }))
        report_path = self.candidate / "tune_report.json"
        report_path.write_text(json.dumps({
            "phase_a": {
                "warm_start_configs": [{"params": {"delay": 0.0}, "score": -0.5}],
                "deferred_configs": [],
                "best_warm_score": -0.5,
                "search_space": {"delay": ["float", 0.0, 2.0]},
                "status": "ok",
            }
        }))
        statuses = []
        for _ in range(8):
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TUNERS / "bo_search.py"),
                    "--candidate-path", str(self.candidate / "train.py"),
                    "--tune-report-json", str(report_path),
                    "--n-trials", "4",
                    "--seed", "7",
                ],
                text=True,
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            statuses.append(payload["status"])
            if payload["status"] == "ok":
                break
        self.assertIn("partial", statuses)
        self.assertEqual(statuses[-1], "ok")
        stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
        self.assertEqual(len(stage["trials"]), 4)
        self.assertEqual(stage["target_trials"], 4)

    def test_grid_partial_batches_skip_recorded_configs(self):
        (self.run_dir / "framework_cfg.json").write_text(json.dumps({
            "per_runtime_limit": 1,
            "tuner": {"search_wall_seconds": 0.08, "max_consecutive_failures": 5},
        }))
        report_path = self.candidate / "tune_report.json"
        report_path.write_text(json.dumps({
            "phase_a": {
                "warm_start_configs": [{"params": {"delay": 0.0}, "score": -0.5}],
                "deferred_configs": [],
                "best_warm_score": -0.5,
                "search_space": {"delay": ["float", 0.0, 2.0]},
                "status": "ok",
            }
        }))
        statuses = []
        for _ in range(8):
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TUNERS / "grid_search.py"),
                    "--candidate-path", str(self.candidate / "train.py"),
                    "--tune-report-json", str(report_path),
                    "--resolution", "5",
                    "--max-trials", "5",
                    "--patience", "6",
                    "--seed", "7",
                ],
                text=True,
                capture_output=True,
                timeout=5,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            statuses.append(payload["status"])
            if payload["status"] == "ok":
                break
        self.assertIn("partial", statuses)
        self.assertEqual(statuses[-1], "ok")
        trials = json.loads(report_path.read_text())["phase_c"]["stages"][0]["trials"]
        self.assertEqual(len(trials), 5)
        self.assertEqual(len({json.dumps(t["params"], sort_keys=True) for t in trials}), 5)

    def test_bo_caps_total_target_to_global_remaining_budget(self):
        (self.run_dir / "framework_cfg.json").write_text(json.dumps({
            "max_evaluations": 10,
            "per_runtime_limit": 1,
            "tuner": {"search_wall_seconds": 2, "max_consecutive_failures": 5},
        }))
        (self.run_dir / "ledger.json").write_text(json.dumps({
            "records": [{"run_id": "000", "trials_completed": 9}]
        }))
        report_path = self.candidate / "tune_report.json"
        report_path.write_text(json.dumps({
            "phase_a": {
                "warm_start_configs": [{"params": {"delay": 0.0}, "score": -0.5}],
                "deferred_configs": [],
                "best_warm_score": -0.5,
                "search_space": {"delay": ["float", 0.0, 2.0]},
                "status": "ok",
            }
        }))
        proc = subprocess.run(
            [
                sys.executable,
                str(TUNERS / "bo_search.py"),
                "--candidate-path", str(self.candidate / "train.py"),
                "--tune-report-json", str(report_path),
                "--n-trials", "4",
            ],
            text=True,
            capture_output=True,
            timeout=5,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["target_trials"], 1)
        self.assertEqual(payload["requested_target_trials"], 4)
        self.assertEqual(payload["early_stop_reason"], "global_eval_budget")


if __name__ == "__main__":
    unittest.main()
