from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import _common  # noqa: E402
import bo_search  # noqa: E402
import cmaes_search  # noqa: E402
import grid_search  # noqa: E402
from _common import (  # noqa: E402
    DEEP_TUNE_INVOCATION_STARTED_AT,
    DeepTuneStageAdmissionError,
    DeepTuneTimeExhausted,
    PhaseCObjectiveRecoveryError,
    append_trial,
    bind_phase_c_objective_reservation,
    commit_phase_c_objective_trial,
    deduplicate_configs,
    deep_tune_stage_elapsed,
    deep_tune_time_budget,
    ensure_deep_tune_time_remaining,
    prepare_phase_c_objective_attempt,
    set_stage_meta,
    timed_eval,
    timed_preflight,
)
from grid_search import main as grid_main  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    phase_c_action,
)


def _plain_make_model(params):
    return params


class DeepTuneGovernanceTest(unittest.TestCase):
    def _fixture(self, root: Path, *, n_dims: int = 1, limit: float = 10.0):
        candidate = root / "candidates" / "001" / "train.py"
        candidate.parent.mkdir(parents=True)
        entries = ", ".join(
            f"'x{index}': ('float', 0.0, 1.0)"
            for index in range(n_dims)
        )
        schema = ", ".join(
            f"'x{index}': 'float'"
            for index in range(n_dims)
        )
        base = ", ".join(
            f"'x{index}': 0.0"
            for index in range(n_dims)
        )
        candidate.write_text(
            f"PARAM_SCHEMA = {{{schema}}}\n"
            f"SEARCH_SPACE = {{{entries}}}\n"
            f"BASE_PARAMS = {{{base}}}\n"
            "def make_model(params):\n"
            "    return params\n"
        )
        (candidate.parent / "prepare.py").write_text(
            "def evaluate_config(make_model, params):\n"
            "    return 0.0\n"
        )
        (root / "framework_cfg.json").write_text(
            json.dumps(
                {"tuner": {"deep_tune_time_limit_seconds": limit}}
            )
        )
        report = {
            "phase_a": {
                "status": "ok",
                "candidate_code_revision": _candidate_execution_revision(candidate),
                "search_space": {
                    f"x{index}": ["float", 0.0, 1.0]
                    for index in range(n_dims)
                },
                "warm_start_configs": [
                    {
                        "params": {
                            f"x{index}": 0.0
                            for index in range(n_dims)
                        },
                        "score": 1.0,
                    }
                ],
                "best_warm_params": {
                    f"x{index}": 0.0
                    for index in range(n_dims)
                },
                "best_warm_score": 1.0,
            }
        }
        report_path = candidate.parent / "tune_report.json"
        report_path.write_text(json.dumps(report))
        return candidate, report_path

    def test_running_start_is_persisted_and_recovered_after_interruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            with mock.patch.object(_common.time, "monotonic", return_value=10.0), \
                    mock.patch.object(_common.time, "time", return_value=100.0):
                first = deep_tune_time_budget(candidate, report_path, "grid")

            self.assertEqual(first["stage_used_seconds"], 0.0)
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(stage["status"], "running")
            self.assertEqual(stage[DEEP_TUNE_INVOCATION_STARTED_AT], 100.0)
            first["_phase_c_lock_handle"].close()

            with mock.patch.object(_common.time, "monotonic", return_value=20.0), \
                    mock.patch.object(_common.time, "time", return_value=104.0):
                resumed = deep_tune_time_budget(candidate, report_path, "grid")

            self.assertEqual(resumed["stage_used_seconds"], 4.0)
            # No wall-clock limit exists (the fixture's legacy
            # deep_tune_time_limit_seconds key must parse but be ignored):
            # remaining is unbounded and the receipt value is null.
            self.assertEqual(resumed["remaining_seconds"], math.inf)
            self.assertIsNone(resumed["limit_seconds"])
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(stage["recovered_interrupted_invocations"], 1)
            self.assertEqual(stage["elapsed_seconds"], 4.0)
            self.assertEqual(stage[DEEP_TUNE_INVOCATION_STARTED_AT], 104.0)
            resumed["_phase_c_lock_handle"].close()

    def test_phase_c_recovery_retries_intent_without_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "intent-only"
            candidate, report_path = self._fixture(run_dir)
            first = deep_tune_time_budget(candidate, report_path, "grid")
            prepare_phase_c_objective_attempt(
                report_path,
                candidate,
                "grid",
                {"x0": 0.25},
                first["candidate_execution_revision"],
            )
            first["_phase_c_lock_handle"].close()

            resumed = deep_tune_time_budget(candidate, report_path, "grid")
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertNotIn(_common.PHASE_C_ACTIVE_OBJECTIVE_ATTEMPT, stage)
            self.assertEqual(stage["trials"], [])
            self.assertEqual(
                _common.objective_attempt_receipts(
                    candidate,
                    phase="phase_c",
                    method="grid",
                ),
                [],
            )
            resumed["_phase_c_lock_handle"].close()

    def test_phase_c_recovery_consumes_orphaned_reservation_without_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "reserved-only"
            candidate, report_path = self._fixture(run_dir)
            first = deep_tune_time_budget(candidate, report_path, "grid")
            intent = prepare_phase_c_objective_attempt(
                report_path,
                candidate,
                "grid",
                {"x0": 0.25},
                first["candidate_execution_revision"],
            )
            receipt = _common.reserve_evaluation(
                candidate,
                params={"x0": 0.25},
                phase="phase_c",
                method="grid",
            )
            self.assertIsNotNone(receipt)
            # Simulate SIGKILL before the callback can bind the receipt.
            first["_phase_c_lock_handle"].close()

            resumed = deep_tune_time_budget(candidate, report_path, "grid")
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertNotIn(_common.PHASE_C_ACTIVE_OBJECTIVE_ATTEMPT, stage)
            self.assertEqual(len(stage["trials"]), 1)
            recovered = stage["trials"][0]
            self.assertEqual(recovered["params"], intent["params"])
            self.assertIsNone(recovered["score"])
            self.assertEqual(recovered["status"], "failed")
            self.assertEqual(
                recovered["failure_category"],
                _common.PHASE_C_INTERRUPTION_CATEGORY,
            )
            self.assertEqual(
                recovered["objective_reservation"]["attempt_id"],
                receipt["attempt_id"],
            )
            self.assertEqual(
                recovered["candidate_execution_revision_sha256"],
                first["candidate_execution_revision"]["revision_sha256"],
            )
            resumed["_phase_c_lock_handle"].close()

            # Recovery is idempotent: a later restart neither appends a second
            # failure row nor reserves another objective slot.
            again = deep_tune_time_budget(candidate, report_path, "grid")
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(len(stage["trials"]), 1)
            self.assertEqual(
                len(
                    _common.objective_attempt_receipts(
                        candidate,
                        phase="phase_c",
                        method="grid",
                    )
                ),
                1,
            )
            again["_phase_c_lock_handle"].close()

    def test_phase_c_recovery_blocks_mismatched_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "mismatch"
            candidate, report_path = self._fixture(run_dir)
            first = deep_tune_time_budget(candidate, report_path, "grid")
            prepare_phase_c_objective_attempt(
                report_path,
                candidate,
                "grid",
                {"x0": 0.25},
                first["candidate_execution_revision"],
            )
            _common.reserve_evaluation(
                candidate,
                params={"x0": 0.75},
                phase="phase_c",
                method="grid",
            )
            first["_phase_c_lock_handle"].close()

            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError,
                "reservation does not match",
            ):
                deep_tune_time_budget(candidate, report_path, "grid")
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertIn(_common.PHASE_C_ACTIVE_OBJECTIVE_ATTEMPT, stage)
            self.assertEqual(stage["trials"], [])

    def test_phase_c_commit_clears_intent_with_exact_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "unit" / "commit"
            candidate, report_path = self._fixture(run_dir)
            budget = deep_tune_time_budget(candidate, report_path, "grid")
            params = {"x0": 0.25}
            intent = prepare_phase_c_objective_attempt(
                report_path,
                candidate,
                "grid",
                params,
                budget["candidate_execution_revision"],
            )
            receipt = _common.reserve_evaluation(
                candidate,
                params=params,
                phase="phase_c",
                method="grid",
            )
            bind_phase_c_objective_reservation(
                report_path,
                candidate,
                "grid",
                intent,
                receipt,
            )

            with self.assertRaises(PhaseCObjectiveRecoveryError):
                append_trial(
                    report_path,
                    "grid",
                    {"params": params, "score": 0.5},
                )
            with self.assertRaises(PhaseCObjectiveRecoveryError):
                set_stage_meta(report_path, "grid", status="ok")

            committed = commit_phase_c_objective_trial(
                report_path,
                candidate,
                "grid",
                intent,
                {"params": params, "score": 0.5},
            )
            self.assertEqual(committed["objective_reservation"], receipt)
            self.assertEqual(
                committed["candidate_execution_revision_sha256"],
                budget["candidate_execution_revision"]["revision_sha256"],
            )
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertNotIn(_common.PHASE_C_ACTIVE_OBJECTIVE_ATTEMPT, stage)
            self.assertEqual(stage["trials"], [committed])
            budget["_phase_c_lock_handle"].close()

    def test_active_tuners_do_not_replay_recovered_orphan(self):
        class BOTrial:
            def __init__(
                self,
                params,
                *,
                value=None,
                user_attrs=None,
                system_attrs=None,
            ):
                self.params = dict(params)
                self.value = value
                self.user_attrs = dict(user_attrs or {})
                self.system_attrs = dict(system_attrs or {})

            def set_user_attr(self, key, value):
                self.user_attrs[key] = value

            def suggest_float(self, key, low, high, *, log=False):
                return self.params[key]

            def suggest_int(self, key, low, high):
                return self.params[key]

            def suggest_categorical(self, key, choices):
                return self.params[key]

        class BOStudy:
            def __init__(self, proposal):
                self.proposal = dict(proposal)
                self.trials = []
                self.stopped = False

            def add_trial(self, trial):
                self.trials.append(trial)

            def enqueue_trial(self, params, *, skip_if_exists=False):
                raise AssertionError("no deferred config should be enqueued")

            def stop(self):
                self.stopped = True

            def optimize(
                self,
                objective,
                *,
                n_trials,
                timeout,
                show_progress_bar,
                callbacks,
                catch,
            ):
                trial = BOTrial(self.proposal)
                try:
                    trial.value = objective(trial)
                except catch:
                    trial.value = None
                self.trials.append(trial)
                for callback in callbacks:
                    callback(self, trial)

        for method, n_dims, module in [
            ("grid", 1, grid_search),
            ("bo", 3, bo_search),
            ("cmaes", 3, cmaes_search),
        ]:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                run_dir = Path(tmp) / "runs" / "unit" / method
                candidate, report_path = self._fixture(run_dir, n_dims=n_dims)
                if method == "cmaes":
                    report = json.loads(report_path.read_text())
                    report["phase_c"] = {
                        "stages": [
                            {
                                "method": "bo",
                                "status": "rejected",
                                "trials": [],
                            }
                        ]
                    }
                    report_path.write_text(json.dumps(report))

                budget = deep_tune_time_budget(
                    candidate,
                    report_path,
                    method,
                )
                orphan_params = {
                    f"x{index}": 1.0 for index in range(n_dims)
                }
                prepare_phase_c_objective_attempt(
                    report_path,
                    candidate,
                    method,
                    orphan_params,
                    budget["candidate_execution_revision"],
                )
                first_receipt = _common.reserve_evaluation(
                    candidate,
                    params=orphan_params,
                    phase="phase_c",
                    method=method,
                )
                budget["_phase_c_lock_handle"].close()

                argv = [
                    f"{method}_search.py",
                    "--candidate-path",
                    str(candidate),
                    "--tune-report-json",
                    str(report_path),
                ]
                if method == "grid":
                    argv.extend(["--resolution", "2", "--max-trials", "2"])
                elif method == "bo":
                    argv.extend(["--n-trials", "1"])
                else:
                    argv.extend(["--max-evals", "1", "--popsize", "1"])

                timed_eval_mock = mock.Mock(
                    side_effect=AssertionError(
                        "recovered orphan reached objective reservation again"
                    )
                )
                contexts = [
                    mock.patch.object(module, "timed_eval", timed_eval_mock),
                    mock.patch.object(module, "write_json"),
                    mock.patch.object(sys, "argv", argv),
                ]
                if method == "bo":
                    study = BOStudy(orphan_params)
                    distribution = lambda *args, **kwargs: object()
                    fake_optuna = types.SimpleNamespace(
                        samplers=types.SimpleNamespace(
                            TPESampler=distribution
                        ),
                        distributions=types.SimpleNamespace(
                            FloatDistribution=distribution,
                            IntDistribution=distribution,
                            CategoricalDistribution=distribution,
                        ),
                        trial=types.SimpleNamespace(
                            create_trial=lambda **kwargs: BOTrial(
                                kwargs["params"],
                                value=kwargs.get("value"),
                                user_attrs=kwargs.get("user_attrs"),
                                system_attrs=kwargs.get("system_attrs"),
                            )
                        ),
                        logging=types.SimpleNamespace(
                            WARNING=30,
                            set_verbosity=lambda level: None,
                        ),
                        create_study=lambda **kwargs: study,
                    )
                    contexts.append(
                        mock.patch.dict(sys.modules, {"optuna": fake_optuna})
                    )
                elif method == "cmaes":
                    strategy = mock.Mock()
                    strategy.stop.return_value = False
                    strategy.ask.return_value = [
                        [orphan_params[f"x{index}"] for index in range(n_dims)]
                    ]
                    fake_cma = types.SimpleNamespace(
                        CMAEvolutionStrategy=lambda *args, **kwargs: strategy
                    )
                    contexts.append(
                        mock.patch.dict(sys.modules, {"cma": fake_cma})
                    )

                with contexts[0], contexts[1], contexts[2]:
                    if len(contexts) == 4:
                        with contexts[3]:
                            self.assertEqual(module.main(), 0)
                    else:
                        self.assertEqual(module.main(), 0)

                timed_eval_mock.assert_not_called()
                receipts = _common.objective_attempt_receipts(
                    candidate,
                    phase="phase_c",
                    method=method,
                )
                self.assertEqual(receipts, [first_receipt])
                report = json.loads(report_path.read_text())
                stage = next(
                    item
                    for item in report["phase_c"]["stages"]
                    if item["method"] == method
                )
                self.assertEqual(len(stage["trials"]), 1)
                self.assertEqual(
                    stage["trials"][0]["failure_category"],
                    _common.PHASE_C_INTERRUPTION_CATEGORY,
                )

    def test_elapsed_is_full_precision_and_never_blocks_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The tiny legacy cap must not stop anything: elapsed accounting is
            # informational only since the wall clock was removed.
            candidate, report_path = self._fixture(Path(tmp), limit=0.05)
            with mock.patch.object(
                _common.time, "monotonic", side_effect=[10.0, 10.037, 10.051]
            ), mock.patch.object(_common.time, "time", return_value=100.0):
                budget = deep_tune_time_budget(candidate, report_path, "grid")
                self.assertAlmostEqual(deep_tune_stage_elapsed(budget), 0.037)
                ensure_deep_tune_time_remaining(budget)

            budget["_phase_c_lock_handle"].close()
            set_stage_meta(
                report_path,
                "grid",
                status="time_exhausted",
                elapsed_seconds=0.037,
            )
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(stage["elapsed_seconds"], 0.037)
            self.assertNotIn(DEEP_TUNE_INVOCATION_STARTED_AT, stage)

    def test_wrong_fallback_and_terminal_rerun_are_rejected_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "requires rejected prefix"
            ):
                deep_tune_time_budget(candidate, report_path, "bo")

            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {"method": "grid", "status": "rejected", "trials": []}
                ]
            }
            report_path.write_text(json.dumps(report))
            budget = deep_tune_time_budget(candidate, report_path, "bo")
            set_stage_meta(
                report_path,
                "bo",
                status="failed",
                elapsed_seconds=0.01,
            )
            budget["_phase_c_lock_handle"].close()
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "cannot be rerun"
            ):
                deep_tune_time_budget(candidate, report_path, "bo")

    def test_untrustworthy_phase_a_is_rejected_before_stage_mutation(self):
        """Stage admission is a gate: a rejected report writes no phase_c."""

        def forged_warm_best(report: dict) -> None:
            report["phase_a"]["best_warm_score"] = 2.0

        def out_of_bounds_warm_row(report: dict) -> None:
            report["phase_a"]["warm_start_configs"] = [
                {"params": {"x0": 9.0}, "score": 1.0}
            ]
            report["phase_a"]["best_warm_params"] = {"x0": 9.0}

        def unapplied_incumbent(report: dict) -> None:
            report["phase_a"]["warm_start_configs"] = [
                {"params": {"x0": 1.0}, "score": 1.0}
            ]
            report["phase_a"]["best_warm_params"] = {"x0": 1.0}

        def drifted_search_space(report: dict) -> None:
            report["phase_a"]["search_space"]["x0"][2] = 0.5

        cases = [
            ("best_warm_score", forged_warm_best),
            ("violates SEARCH_SPACE", out_of_bounds_warm_row),
            # The winning warm config must already be applied to BASE_PARAMS,
            # or Phase C would deep-tune from a different implementation.
            ("BASE_PARAMS", unapplied_incumbent),
            ("phase_a.search_space", drifted_search_space),
        ]
        for needle, mutate in cases:
            with self.subTest(needle=needle), tempfile.TemporaryDirectory() as tmp:
                candidate, report_path = self._fixture(Path(tmp))
                report = json.loads(report_path.read_text())
                mutate(report)
                report_path.write_text(json.dumps(report))

                with self.assertRaisesRegex(DeepTuneStageAdmissionError, needle):
                    deep_tune_time_budget(candidate, report_path, "grid")
                self.assertNotIn("phase_c", json.loads(report_path.read_text()))

    def test_stale_warm_execution_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            candidate.write_text(
                candidate.read_text() + "EXECUTION_CHANGED = True\n"
            )
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "execution revision"
            ):
                deep_tune_time_budget(candidate, report_path, "grid")
            report = json.loads(report_path.read_text())
            self.assertNotIn("phase_c", report)

    def test_phase_c_action_resumes_primary_and_fallback_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())

            self.assertEqual(
                phase_c_action(report, candidate)["method"],
                "grid",
            )
            report["phase_c"] = {
                "stages": [
                    {"method": "grid", "status": "running", "trials": []}
                ]
            }
            resumed = phase_c_action(report, candidate)
            self.assertEqual(
                (resumed["action"], resumed["method"], resumed["reason"]),
                ("run", "grid", "resume_interrupted_stage"),
            )
            report["phase_c"]["stages"][0]["status"] = "rejected"
            fallback = phase_c_action(report, candidate)
            self.assertEqual(
                (fallback["action"], fallback["method"], fallback["reason"]),
                ("run", "bo", "run_deterministic_fallback"),
            )

    def test_phase_c_action_cli_exposes_the_validated_resume_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "tuners" / "tune_tools.py"),
                    "phase-c-action",
                    "--candidate-path",
                    str(candidate),
                    "--tune-report-json",
                    str(report_path),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            decision = json.loads(completed.stdout)
            self.assertEqual(
                (decision["action"], decision["method"], decision["n_dims"]),
                ("run", "grid", 1),
            )

    def test_phase_c_action_finalizes_terminal_or_exhausted_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "ok",
                        "trials": [
                            {"params": {"x0": 0.5}, "score": 0.5}
                        ],
                    }
                ]
            }
            terminal = phase_c_action(report, candidate)
            self.assertEqual(terminal["action"], "finalize")
            self.assertEqual(terminal["best_score"], 0.5)

            report["phase_c"]["stages"] = [
                {"method": "grid", "status": "rejected", "trials": []},
                {"method": "bo", "status": "rejected", "trials": []},
            ]
            exhausted = phase_c_action(report, candidate)
            self.assertEqual(exhausted["action"], "finalize")
            self.assertEqual(exhausted["best_score"], 1.0)

    def test_phase_c_action_stops_on_budget_exhaustion_and_rejects_bad_history(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "budget_exhausted",
                        "trials": [],
                    }
                ]
            }
            stopped = phase_c_action(report, candidate)
            self.assertEqual(
                (stopped["action"], stopped["reason"]),
                ("stop", "evaluation_budget_reached"),
            )

            report["phase_c"]["stages"][0] = {
                "method": "grid",
                "status": "rejected",
                "trials": [{"params": {"x0": 0.5}, "score": 0.5}],
            }
            with self.assertRaisesRegex(ValueError, "cannot contain trials"):
                phase_c_action(report, candidate)

    def test_phase_c_action_rejects_malformed_state_and_forged_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            original = json.loads(report_path.read_text())

            for malformed in ([], "running", {"stages": None}):
                with self.subTest(malformed=malformed):
                    report = json.loads(json.dumps(original))
                    report["phase_c"] = malformed
                    with self.assertRaisesRegex(
                        ValueError, "phase_c|phase_c.stages"
                    ):
                        phase_c_action(report, candidate)

            forged = json.loads(json.dumps(original))
            forged["applied_to_base_params"] = True
            with self.assertRaisesRegex(ValueError, "not finalizable"):
                phase_c_action(forged, candidate)

            report_path.write_text(
                json.dumps({**original, "phase_c": []})
            )
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "phase_c must be an object"
            ):
                deep_tune_time_budget(candidate, report_path, "grid")

    def test_phase_c_action_rejects_malformed_or_out_of_bounds_trial_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            original = json.loads(report_path.read_text())
            bad_trials = (
                "bad",
                [7],
                [{"params": {"x0": 9.0}, "score": 9.0}],
                [
                    {
                        "params": {"x0": 0.5},
                        "score": -999.0,
                        "status": "preflight_rejected",
                    }
                ],
                [{"params": {"x0": 0.5}, "score": None}],
            )
            for trials in bad_trials:
                with self.subTest(trials=trials):
                    report = json.loads(json.dumps(original))
                    report["phase_c"] = {
                        "stages": [
                            {
                                "method": "grid",
                                "status": "running",
                                "trials": trials,
                            }
                        ]
                    }
                    with self.assertRaisesRegex(
                        ValueError,
                        "must be a list|must contain params|violates|"
                        "finite score|null score",
                    ):
                        phase_c_action(report, candidate)

    def test_running_journal_survives_interruption_and_stays_admissible(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp), limit=1.0)
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "running",
                        "trials": [],
                        "elapsed_seconds": 1.0,
                    }
                ]
            }
            report_path.write_text(json.dumps(report))

            # Past any legacy cap, admission and journaling must be unaffected.
            first = deep_tune_time_budget(candidate, report_path, "grid")
            self.assertEqual(first["remaining_seconds"], math.inf)
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(stage["status"], "running")
            self.assertIn(DEEP_TUNE_INVOCATION_STARTED_AT, stage)

            # Simulate process death: the OS releases the lock while the running
            # journal remains. The next invocation must be admissible and recover it.
            first["_phase_c_lock_handle"].close()
            resumed = deep_tune_time_budget(candidate, report_path, "grid")
            self.assertEqual(resumed["remaining_seconds"], math.inf)
            resumed["_phase_c_lock_handle"].close()

    def test_candidate_phase_c_lock_rejects_concurrent_tuner(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            first = deep_tune_time_budget(candidate, report_path, "grid")
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "another Phase-C tuner"
            ):
                deep_tune_time_budget(candidate, report_path, "grid")

            first["_phase_c_lock_handle"].close()
            resumed = deep_tune_time_budget(candidate, report_path, "grid")
            resumed["_phase_c_lock_handle"].close()

    def test_eval_uses_live_minimum_phase_timeout_and_charges_reservation(self):
        remaining = mock.Mock(side_effect=[5.0, 3.0])
        with mock.patch.object(
            _common, "reserve_evaluation"
        ) as reserve, mock.patch.object(
            _common, "read_runtime_limit", return_value=10.0
        ), mock.patch.object(
            _common,
            "_communicate_with_limit",
            return_value=("RESULT:0.25\n", "", 0),
        ) as communicate:
            score = timed_eval(
                object(),
                _plain_make_model,
                {"x": 1},
                Path("/tmp/candidate.py"),
                phase="phase_c",
                method="grid",
                phase_time_limit_seconds=remaining,
            )

        self.assertEqual(score, 0.25)
        reserve.assert_called_once()
        self.assertEqual(communicate.call_args.kwargs["limit"], 3.0)

    def test_phase_bound_eval_timeout_reports_reserved_attempt(self):
        with mock.patch.object(
            _common, "reserve_evaluation"
        ) as reserve, mock.patch.object(
            _common, "read_runtime_limit", return_value=None
        ), mock.patch.object(
            _common,
            "_communicate_with_limit",
            side_effect=TimeoutError("phase timeout"),
        ):
            with self.assertRaises(DeepTuneTimeExhausted) as caught:
                timed_eval(
                    object(),
                    _plain_make_model,
                    {},
                    Path("/tmp/candidate.py"),
                    phase="phase_c",
                    method="bo",
                    phase_time_limit_seconds=2.0,
                )

        reserve.assert_called_once()
        self.assertTrue(caught.exception.attempt_reserved)

    def test_preflight_uses_minimum_of_task_and_phase_time(self):
        with mock.patch.object(
            _common, "_configured_preflight_name", return_value="preflight"
        ), mock.patch.object(
            _common, "read_preflight_limit", return_value=10.0
        ), mock.patch.object(
            _common,
            "_communicate_with_limit",
            return_value=('PREFLIGHT:{"status":"ok"}\n', "", 0),
        ) as communicate:
            result = timed_preflight(
                {},
                Path("/tmp/candidate.py"),
                phase_time_limit_seconds=3.0,
            )

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(communicate.call_args.kwargs["limit"], 3.0)

    def test_search_space_change_is_rejected_before_stage_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            candidate.write_text(
                candidate.read_text().replace(
                    "'x0': ('float', 0.0, 1.0)",
                    "'x0': ('float', 0.0, 2.0)",
                )
            )

            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "execution revision"
            ):
                deep_tune_time_budget(candidate, report_path, "grid")

            report = json.loads(report_path.read_text())
            self.assertNotIn("phase_c", report)

    def test_deduplicate_configs_preserves_first_unseen_order(self):
        unique, skipped, seen = deduplicate_configs(
            [{"x": 1}, {"x": 2}, {"x": 1}, {"x": 3}],
            seen={_common.params_identity({"x": 2})},
        )
        self.assertEqual(unique, [{"x": 1}, {"x": 3}])
        self.assertEqual(skipped, 2)
        self.assertEqual(len(seen), 3)

    def test_resumed_grid_does_not_re_evaluate_deferred_or_grid_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, report_path = self._fixture(root)
            candidate.write_text(
                "PARAM_SCHEMA = {'mode': ('categorical', ['a', 'b'])}\n"
                "SEARCH_SPACE = {'mode': ('categorical', ['a', 'b'])}\n"
                "BASE_PARAMS = {'mode': 'a'}\n"
                "def make_model(params):\n"
                "    return params\n"
            )
            revision = _candidate_execution_revision(candidate)
            report_path.write_text(
                json.dumps(
                    {
                        "phase_a": {
                            "status": "ok",
                            "candidate_code_revision": revision,
                            "search_space": {
                                "mode": ["categorical", ["a", "b"]]
                            },
                            "warm_start_configs": [
                                {"params": {"mode": "a"}, "score": 1.0}
                            ],
                            "best_warm_params": {"mode": "a"},
                            "best_warm_score": 1.0,
                            "deferred_configs": [
                                {"params": {"mode": "a"}},
                                {"params": {"mode": "b"}},
                            ],
                        },
                        "phase_c": {
                            "stages": [
                                {
                                    "method": "grid",
                                    "status": "running",
                                    "trials": [
                                        {
                                            "params": {"mode": "b"},
                                            "score": 0.5,
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                )
            )
            train_module = mock.Mock(
                SEARCH_SPACE={"mode": ("categorical", ["a", "b"])},
                BASE_PARAMS={"mode": "a"},
                make_model=object(),
            )

            with mock.patch(
                "grid_search.load_candidate_modules",
                return_value=(train_module, object()),
            ), mock.patch(
                "grid_search.resolve_score_fn", return_value=object()
            ), mock.patch(
                "grid_search.resolve_preflight_fn", return_value=None
            ), mock.patch(
                "grid_search.timed_eval"
            ) as timed_eval_mock, mock.patch(
                "grid_search.write_json"
            ) as write_result, mock.patch.object(
                sys,
                "argv",
                [
                    "grid_search.py",
                    "--candidate-path",
                    str(candidate),
                    "--tune-report-json",
                    str(report_path),
                    "--resolution",
                    "2",
                    "--max-trials",
                    "1",
                ],
            ):
                self.assertEqual(grid_main(), 0)

            timed_eval_mock.assert_not_called()
            result = write_result.call_args.args[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["best_params"], {"mode": "b"})
            self.assertEqual(result["deferred_skipped_already_seen"], 2)
            self.assertEqual(result["grid_skipped_already_seen"], 2)

    def test_grid_rechecks_time_after_preflight_before_objective_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, report_path = self._fixture(root)
            train_module = mock.Mock(
                SEARCH_SPACE={"x0": ("float", 0.0, 1.0)},
                BASE_PARAMS={"x0": 0.0},
                make_model=object(),
            )
            time_checks = [None] * 6 + [
                DeepTuneTimeExhausted("expired after preflight")
            ]

            with mock.patch(
                "grid_search.load_candidate_modules",
                return_value=(train_module, object()),
            ), mock.patch(
                "grid_search.resolve_score_fn", return_value=object()
            ), mock.patch(
                "grid_search.resolve_preflight_fn", return_value=object()
            ), mock.patch(
                "grid_search.clamp_search_space_to_preflight",
                return_value=train_module.SEARCH_SPACE,
            ) as clamp_mock, mock.patch(
                "grid_search.ensure_deep_tune_time_remaining",
                side_effect=time_checks,
            ), mock.patch(
                "grid_search.timed_preflight", return_value={"status": "ok"}
            ) as preflight_mock, mock.patch(
                "grid_search.timed_eval"
            ) as timed_eval_mock, mock.patch(
                "grid_search.write_json"
            ) as write_result, mock.patch.object(
                sys,
                "argv",
                [
                    "grid_search.py",
                    "--candidate-path",
                    str(candidate),
                    "--tune-report-json",
                    str(report_path),
                ],
            ):
                self.assertEqual(grid_main(), 0)

            preflight_mock.assert_called_once()
            self.assertTrue(
                callable(
                    clamp_mock.call_args.kwargs["phase_time_limit_seconds"]
                )
            )
            self.assertTrue(
                callable(
                    preflight_mock.call_args.kwargs[
                        "phase_time_limit_seconds"
                    ]
                )
            )
            timed_eval_mock.assert_not_called()
            self.assertEqual(
                write_result.call_args.args[0]["status"],
                "time_exhausted",
            )


if __name__ == "__main__":
    unittest.main()
