from __future__ import annotations

import io
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
import run_cfg  # noqa: E402
import _common  # noqa: E402
import warmstart_eval  # noqa: E402
from _common import timed_eval, timed_preflight  # noqa: E402
from warmstart_eval import (  # noqa: E402
    select_warm_config_indices,
    validate_provided_baseline_configs,
)


def _plain_make_model(*args, **kwargs):
    return None


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
    def test_warm_config_selection_is_uniform_without_replacement_and_replayed(
        self,
    ) -> None:
        selection = select_warm_config_indices(5, 3, {}, seed=17)

        self.assertEqual(selection["method"], "uniform_without_replacement")
        self.assertEqual(selection["seed"], 17)
        self.assertEqual(sorted(selection["permutation"]), list(range(5)))
        self.assertEqual(
            selection["selected_indices"],
            selection["permutation"][:3],
        )
        self.assertEqual(
            selection["deferred_indices"],
            selection["permutation"][3:],
        )

        replayed = select_warm_config_indices(
            5,
            3,
            {"warm_config_selection": selection},
            seed=999,
        )
        self.assertEqual(replayed, selection)

    def test_warm_config_selection_rejects_contract_changes_on_resume(self) -> None:
        selection = select_warm_config_indices(5, 3, {}, seed=17)

        with self.assertRaisesRegex(ValueError, "k_eval changed"):
            select_warm_config_indices(
                5,
                2,
                {"warm_config_selection": selection},
            )
        with self.assertRaisesRegex(ValueError, "count changed"):
            select_warm_config_indices(
                6,
                3,
                {"warm_config_selection": selection},
            )

    def test_schema4_selection_always_includes_and_replays_control_zero(self) -> None:
        for seed in range(20):
            selection = select_warm_config_indices(
                7,
                3,
                {},
                seed=seed,
                mandatory_indices=(0,),
            )
            self.assertEqual(selection["schema_version"], 2)
            self.assertEqual(
                selection["method"],
                "mandatory_then_uniform_without_replacement",
            )
            self.assertEqual(selection["mandatory_indices"], [0])
            self.assertEqual(selection["selected_indices"][0], 0)
            self.assertNotIn(0, selection["deferred_indices"])
            self.assertEqual(
                select_warm_config_indices(
                    7,
                    3,
                    {"warm_config_selection": selection},
                    seed=seed + 100,
                    mandatory_indices=(0,),
                ),
                selection,
            )

    def test_provided_baseline_allows_only_its_exact_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate" / "train.py"
            candidate.parent.mkdir()
            candidate.write_text("DEFAULT_PARAMS = {'depth': 8, 'lr': 0.04}\n")
            (candidate.parent / "_candidate_brief.json").write_text(json.dumps({
                "schema_version": 3,
                "implementation_source": {
                    "kind": "provided_entrypoint",
                    "path": "tasks/unit/train.py",
                    "sha256": "sha256:" + "0" * 64,
                },
            }))
            defaults = [{"depth": 8, "lr": 0.04}]

            validate_provided_baseline_configs(candidate, defaults, 1)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                validate_provided_baseline_configs(
                    candidate,
                    defaults + [{"depth": 9, "lr": 0.04}],
                    1,
                )
            with self.assertRaisesRegex(ValueError, "literal DEFAULT_PARAMS"):
                validate_provided_baseline_configs(
                    candidate,
                    [{"depth": 9, "lr": 0.04}],
                    1,
                )
            candidate.write_text("DEFAULT_PARAMS = build_defaults()\n")
            with self.assertRaisesRegex(ValueError, "module-level literal"):
                validate_provided_baseline_configs(candidate, defaults, 1)

    def test_warmstart_requires_a_valid_candidate_origin_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate" / "train.py"
            candidate.parent.mkdir()
            candidate.write_text("DEFAULT_PARAMS = {'depth': 8}\n")
            brief_path = candidate.parent / "_candidate_brief.json"

            with self.assertRaisesRegex(ValueError, "requires candidate brief"):
                validate_provided_baseline_configs(candidate, [{"depth": 8}], 1)

            brief_path.write_text("{")
            with self.assertRaisesRegex(ValueError, "invalid candidate brief"):
                validate_provided_baseline_configs(candidate, [{"depth": 8}], 1)

            brief_path.write_text(json.dumps({
                "schema_version": 3,
                "implementation_source": {"kind": "unknown"},
            }))
            with self.assertRaisesRegex(ValueError, "implementation_source.kind"):
                validate_provided_baseline_configs(candidate, [{"depth": 8}], 1)

            brief_path.write_text(json.dumps({
                "schema_version": 3,
                "implementation_source": {"kind": "generated"},
            }))
            validate_provided_baseline_configs(
                candidate,
                [{"depth": 8}, {"depth": 9}],
                2,
            )

    def test_schema4_nonfresh_missing_transfer_fails_before_candidate_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidates" / "002" / "train.py"
            candidate.parent.mkdir(parents=True)
            candidate.write_text("# unchanged\n")
            (candidate.parent / "_candidate_brief.json").write_text(
                json.dumps(
                    {
                        "schema_version": 4,
                        "run_id": "002",
                        "source_run_ids": ["001"],
                        "primary_parent": {"schema_version": 1},
                        "implementation_source": {
                            "kind": "primary_parent_snapshot",
                        },
                    }
                )
            )
            configs = candidate.parent / "_warm_configs.json"
            configs.write_text(json.dumps([{"x": 1}]))
            report = candidate.parent / "tune_report.json"
            argv = [
                "warmstart_eval.py",
                "--candidate-path",
                str(candidate),
                "--configs-json",
                str(configs),
                "--tune-report-json",
                str(report),
                "--k-eval",
                "1",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    warmstart_eval.apply_base_params,
                    "apply",
                ) as apply_mock,
                mock.patch.object(sys, "stderr", io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                warmstart_eval.main()

            self.assertEqual(raised.exception.code, 2)
            apply_mock.assert_not_called()
            self.assertEqual(candidate.read_text(), "# unchanged\n")

    def test_reservation_refuses_before_score_fn_at_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=2)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 2,
                        "tuner": {
                            "deep_tune_budget_fraction": 1.0,
                            "deep_tune_per_candidate_cap": 2,
                        },
                    }
                )
            )
            calls: list[dict] = []

            def score(_make_model, params):
                calls.append(dict(params))
                return 0.5

            self.assertEqual(
                timed_eval(
                    score,
                    _plain_make_model,
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
                    _plain_make_model,
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
                    _plain_make_model,
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
                ["score_attempt", "score_attempt"],
            )

    def test_deep_tune_total_and_per_candidate_caps_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=10)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 10,
                        "tuner": {
                            "deep_tune_budget_fraction": 0.2,
                            "deep_tune_per_candidate_cap": 1,
                        },
                    }
                )
            )
            other = run_dir / "candidates" / "002" / "train.py"
            other.parent.mkdir(parents=True)
            other.write_text("# candidate\n")

            evaluation_budget.reserve_evaluation(
                candidate,
                params={"x": 1},
                phase="phase_c",
                method="grid",
            )
            with self.assertRaises(
                evaluation_budget.EvaluationBudgetExhausted
            ) as candidate_cap:
                evaluation_budget.reserve_evaluation(
                    candidate,
                    params={"x": 2},
                    phase="phase_c",
                    method="grid",
                )
            self.assertTrue(
                candidate_cap.exception.scope.startswith("deep_tune_candidate")
            )

            evaluation_budget.reserve_evaluation(
                other,
                params={"x": 3},
                phase="phase_c",
                method="grid",
            )
            with self.assertRaises(
                evaluation_budget.EvaluationBudgetExhausted
            ) as total_cap:
                evaluation_budget.reserve_evaluation(
                    run_dir / "candidates" / "003" / "train.py",
                    params={"x": 4},
                    phase="phase_c",
                    method="grid",
                )
            self.assertEqual(total_cap.exception.scope, "deep_tune_total")

            status = evaluation_budget.budget_status(run_dir)
            self.assertEqual(status["deep_tune"]["attempts"], 2)
            self.assertEqual(status["deep_tune"]["remaining"], 0)

    def test_deep_tune_fraction_uses_strict_floor_for_small_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=1)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 1,
                        "tuner": {
                            "deep_tune_budget_fraction": 0.4,
                            "deep_tune_per_candidate_cap": 1,
                        },
                    }
                )
            )

            status = evaluation_budget.budget_status(run_dir)
            self.assertEqual(status["deep_tune"]["total_cap"], 0)
            with self.assertRaises(
                evaluation_budget.EvaluationBudgetExhausted
            ) as exhausted:
                evaluation_budget.reserve_evaluation(
                    candidate,
                    params={"x": 1},
                    phase="phase_c",
                    method="grid",
                )
            self.assertEqual(exhausted.exception.scope, "deep_tune_total")
            self.assertEqual(
                evaluation_budget.budget_status(run_dir)["evaluations_done"],
                0,
            )

    def test_zero_fraction_disables_phase_c_without_global_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=10)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "tuner": {
                            "deep_tune_budget_fraction": 0,
                            "deep_tune_per_candidate_cap": 1,
                        }
                    }
                )
            )

            status = evaluation_budget.budget_status(run_dir)
            self.assertIsNone(status["budget"])
            self.assertEqual(status["deep_tune"]["total_cap"], 0)
            with self.assertRaises(
                evaluation_budget.EvaluationBudgetExhausted
            ) as exhausted:
                evaluation_budget.reserve_evaluation(
                    candidate,
                    params={"x": 1},
                    phase="phase_c",
                    method="grid",
                )
            self.assertEqual(exhausted.exception.scope, "deep_tune_total")

    def test_corrupt_framework_cfg_fails_fast_instead_of_lifting_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=1)
            # The file is designed to be hand-edited; a malformed edit must not
            # silently degrade to "no budget configured".
            (run_dir / "framework_cfg.json").write_text('{"max_evaluations": 1,')
            with self.assertRaises(run_cfg.RunConfigError):
                evaluation_budget.reserve_evaluation(
                    candidate, params={"x": 1}, phase="phase_c", method="grid"
                )
            with self.assertRaises(run_cfg.RunConfigError):
                evaluation_budget.budget_status(run_dir)

    def test_legal_json_with_invalid_hard_limits_fails_fast(self) -> None:
        invalid_configs = [
            {"max_evaluations": "1"},
            {"max_evaluations": True},
            {"max_evaluations": 0},
            {"per_runtime_limit": "bad"},
            {"per_runtime_limit": 0},
            {"per_runtime_limit": float("nan")},
            {"per_runtime_limit": float("inf")},
            {"tuner": {"K_eval": "3"}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=1)
            for config in invalid_configs:
                with self.subTest(config=config):
                    (run_dir / "framework_cfg.json").write_text(json.dumps(config))
                    with self.assertRaises(run_cfg.RunConfigError):
                        evaluation_budget.reserve_evaluation(
                            candidate,
                            params={"x": 1},
                            phase="phase_c",
                            method="grid",
                        )
                    self.assertFalse(
                        (run_dir / evaluation_budget.ATTEMPT_LOG).exists()
                    )

    def test_corrupt_task_contract_does_not_disable_runtime_hooks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_root = Path(tmp)
            task_dir = fake_root / "tasks" / "unit"
            task_dir.mkdir(parents=True)
            (task_dir / "task.toml").write_text("[evaluation]\nbroken line\n")
            candidate = (
                fake_root
                / "runs"
                / "unit"
                / "tag"
                / "candidates"
                / "001"
                / "train.py"
            )
            prepare = type(
                "Prepare",
                (),
                {"evaluate_config": staticmethod(lambda _model, _params: 0.0)},
            )

            with mock.patch.object(_common, "ROOT", fake_root):
                with self.assertRaisesRegex(ValueError, "expected key = value"):
                    _common.resolve_score_fn(prepare, candidate)
                with self.assertRaisesRegex(ValueError, "expected key = value"):
                    timed_preflight({}, candidate)

    def test_concurrent_reservations_cannot_overshoot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir, candidate = _run_dir(Path(tmp), budget=3)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 3,
                        "tuner": {
                            "deep_tune_budget_fraction": 1.0,
                            "deep_tune_per_candidate_cap": 3,
                        },
                    }
                )
            )
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

    def test_hillclimb_reserve_cli_enforces_the_same_hard_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "hillclimb"
            run_dir.mkdir(parents=True)
            candidate = run_dir / "train.py"
            candidate.write_text("VALUE = 1\n")
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 1})
            )
            (run_dir / "results.tsv").write_text(
                "step\tscore\tstatus\tdescription\n"
            )
            command = [
                sys.executable,
                str(ROOT / "tools" / "evaluation_budget.py"),
                "reserve",
                "--ref-path",
                str(candidate),
                "--phase",
                "hillclimb",
                "--method",
                "direct",
            ]

            first = subprocess.run(command, capture_output=True, text=True)
            second = subprocess.run(command, capture_output=True, text=True)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(json.loads(first.stdout)["status"], "reserved")
            (run_dir / "results.tsv").write_text(
                "step\tscore\tstatus\tdescription\n"
                "0\t1.0\tkeep\tbaseline\n"
            )
            self.assertEqual(
                evaluation_budget.budget_status(run_dir)["evaluations_done"],
                1,
            )
            self.assertEqual(second.returncode, 4, second.stderr)
            self.assertEqual(json.loads(second.stdout)["status"], "exhausted")
            self.assertEqual(
                evaluation_budget.budget_status(run_dir)["evaluations_done"],
                1,
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
                    _plain_make_model,
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

    def test_standalone_preflight_uses_default_params_without_tuner_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = (
                Path(tmp)
                / "runs"
                / "autoresearch-baseline"
                / "hillclimb"
            )
            run_dir.mkdir(parents=True)
            candidate = run_dir / "train.py"
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 1, "preflight_runtime_limit": 30})
            )
            (run_dir / "prepare.py").write_text(
                """
def evaluate_config(make_model, params):
    raise AssertionError("objective surface must not run during preflight")

def preflight_config(make_model, params):
    return {"status": "ok", "seen": make_model(None, params)}
""".lstrip()
            )
            candidate.write_text(
                """
DEFAULT_PARAMS = {"x": 7}

def make_model(env, params):
    return params["x"]
""".lstrip()
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "preflight_candidate.py"),
                    "--candidate-path",
                    str(candidate),
                ],
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["status"], "ok")
            self.assertEqual(payload["params_source"], "DEFAULT_PARAMS")
            self.assertEqual(payload["result"]["seen"], 7)
            self.assertFalse((run_dir / evaluation_budget.ATTEMPT_LOG).exists())


if __name__ == "__main__":
    unittest.main()
