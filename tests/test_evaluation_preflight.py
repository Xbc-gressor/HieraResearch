from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import evaluation_budget  # noqa: E402
import preflight_env  # noqa: E402
from _common import timed_eval, timed_preflight  # noqa: E402


def _run_dir(root: Path, *, budget: int) -> tuple[Path, Path]:
    run_dir = root / "runs" / "unit" / "strict"
    candidate = run_dir / "candidates" / "001" / "train.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# candidate\n")
    (run_dir / "framework_cfg.json").write_text(
        json.dumps({"max_evaluations": budget})
    )
    (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
    return run_dir, candidate


class EvaluationBudgetTests(unittest.TestCase):
    def test_reservation_refuses_before_score_fn_at_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=2)
            calls: list[dict] = []

            def score(_make_model, params):
                calls.append(dict(params))
                return 0.5

            self.assertEqual(
                timed_eval(
                    score,
                    object(),
                    {"x": 1},
                    candidate,
                    phase="phase_a",
                    method="warmstart",
                ),
                0.5,
            )
            self.assertEqual(
                timed_eval(
                    score,
                    object(),
                    {"x": 2},
                    candidate,
                    phase="phase_c",
                    method="grid",
                ),
                0.5,
            )
            with self.assertRaises(evaluation_budget.EvaluationBudgetExhausted):
                timed_eval(
                    score,
                    object(),
                    {"x": 3},
                    candidate,
                    phase="phase_c",
                    method="grid",
                )

            self.assertEqual(calls, [{"x": 1}, {"x": 2}])
            status = evaluation_budget.budget_status(run_dir)
            self.assertEqual(status["evaluations_done"], 2)
            self.assertEqual(status["remaining"], 0)
            ledger_view = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "ledger.py"),
                    "evaluations",
                    "--ledger",
                    str(run_dir / "ledger.json"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                json.loads(ledger_view.stdout)["evaluations_done"],
                2,
            )
            rows = [
                json.loads(line)
                for line in (run_dir / evaluation_budget.ATTEMPT_LOG)
                .read_text()
                .splitlines()
            ]
            self.assertEqual(
                [row["kind"] for row in rows],
                ["baseline", "score_attempt", "score_attempt"],
            )

    def test_legacy_sync_reconciles_per_candidate_without_hiding_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=10)
            evaluation_budget.reserve_evaluation(
                candidate, params={"x": 1}, phase="phase_a", method="warmstart"
            )
            evaluation_budget.reserve_evaluation(
                candidate, params={"x": 2}, phase="phase_a", method="warmstart"
            )

            # Candidate 001's aggregate is stale, while candidate 002 was
            # recorded by an older caller that never wrote the reservation log.
            (run_dir / "ledger.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {"run_id": "001", "trials_attempted": 1},
                            {"run_id": "002", "trials_attempted": 2},
                        ]
                    }
                )
            )
            status = evaluation_budget.budget_status(run_dir, create=True)

            self.assertEqual(status["evaluations_done"], 4)
            self.assertEqual(
                {row["run_id"]: row["evals"] for row in status["per_candidate"]},
                {"001": 2, "002": 2},
            )
            sync = [
                json.loads(line)
                for line in (run_dir / evaluation_budget.ATTEMPT_LOG)
                .read_text()
                .splitlines()
                if '"kind": "sync"' in line
            ]
            self.assertEqual(sync[-1]["per_candidate"], {"002": 2})

    def test_concurrent_reservations_cannot_overshoot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=3)
            script = (
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
                "from evaluation_budget import "
                "EvaluationBudgetExhausted, reserve_evaluation\n"
                "try:\n"
                "    reserve_evaluation(sys.argv[1], params={'x': sys.argv[2]}, "
                "phase='phase_c', method='bo')\n"
                "except EvaluationBudgetExhausted:\n"
                "    raise SystemExit(2)\n"
            )
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", script, str(candidate), str(index)]
                )
                for index in range(8)
            ]
            returncodes = [process.wait(timeout=10) for process in processes]

            self.assertEqual(returncodes.count(0), 3)
            self.assertEqual(returncodes.count(2), 5)
            self.assertEqual(
                evaluation_budget.budget_status(run_dir)["evaluations_done"],
                3,
            )


class EnvironmentPreflightTests(unittest.TestCase):
    def test_environment_hook_runs_without_score_surface_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            task_dir = fake_root / "tasks" / "unit"
            task_dir.mkdir(parents=True)
            (task_dir / "task.toml").write_text(
                """
[evaluation]
score_fn = "score"
environment_preflight_fn = "check_environment"
""".lstrip()
            )
            (task_dir / "prepare.py").write_text(
                """
score_calls = 0

def score(make_model, params):
    global score_calls
    score_calls += 1
    raise AssertionError("score must not run during environment preflight")

def check_environment():
    return {"asset": "ready"}
""".lstrip()
            )

            with mock.patch.object(preflight_env, "ROOT", fake_root):
                receipt = preflight_env.run_preflight("unit", fake_root / "run")

            self.assertEqual(receipt["status"], "ok")
            self.assertEqual(receipt["objective_calls"], 0)
            self.assertEqual(receipt["hook_result"], {"asset": "ready"})

    def test_candidate_preflight_is_isolated_and_does_not_reserve_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = (
                Path(tmp)
                / "runs"
                / "autoresearch-baseline"
                / "unit"
            )
            candidate = run_dir / "candidates" / "001" / "train.py"
            candidate.parent.mkdir(parents=True)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 1})
            )
            (run_dir / "ledger.json").write_text(json.dumps({"records": []}))
            (candidate.parent / "prepare.py").write_text(
                """
def evaluate_config(make_model, params):
    raise AssertionError("objective surface must not run")

def preflight_config(make_model, params):
    return {"status": "ok", "seen": make_model(None, params)}
""".lstrip()
            )
            candidate.write_text(
                """
BASE_PARAMS = {"x": 1}
SEARCH_SPACE = {"x": ("int", 1, 2)}

def make_model(env, params):
    return params["x"]
""".lstrip()
            )

            self.assertEqual(
                timed_preflight({"x": 2}, candidate),
                {"status": "ok", "seen": 2},
            )
            self.assertFalse((run_dir / evaluation_budget.ATTEMPT_LOG).exists())
            self.assertEqual(
                timed_eval(
                    lambda _make_model, params: float(params["x"]),
                    object(),
                    {"x": 2},
                    candidate,
                    phase="phase_a",
                    method="warmstart",
                ),
                2.0,
            )
            self.assertEqual(
                evaluation_budget.budget_status(run_dir)["evaluations_done"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
