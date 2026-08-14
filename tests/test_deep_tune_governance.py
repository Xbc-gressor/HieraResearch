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

import _common  # noqa: E402
from _common import (  # noqa: E402
    DEEP_TUNE_INVOCATION_STARTED_AT,
    DeepTuneStageAdmissionError,
    deduplicate_configs,
    deep_tune_stage_elapsed,
    deep_tune_time_budget,
    read_pending_proposals,
    set_stage_meta,
)
from grid_search import main as grid_main  # noqa: E402
from tune_tools import (  # noqa: E402
    _candidate_execution_revision,
    _last_bout_was_first,
    close_exhausted_stage,
    finalizable_tuning_result,
    phase_c_action,
    select_candidate,
    validate_proposals,
)


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
                {"tuner": {"deep_tune_time_limit_seconds": limit,
                           # Stage-mechanics tests predate the regime-
                           # conditioned inner policy: keep the legacy
                           # uniform chains (grid primary at <=2 dims).
                           "inner_policy": "legacy"}}
            )
        )
        report = {
            "inner_policy": "legacy",
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
            # No wall-clock limit exists; the receipt value remains null.
            self.assertIsNone(resumed["limit_seconds"])
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(stage["recovered_interrupted_invocations"], 1)
            self.assertEqual(stage["elapsed_seconds"], 4.0)
            self.assertEqual(stage[DEEP_TUNE_INVOCATION_STARTED_AT], 104.0)
            resumed["_phase_c_lock_handle"].close()

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

            budget["_phase_c_lock_handle"].close()
            set_stage_meta(
                report_path,
                "grid",
                status="failed",
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

    def test_phase_c_action_finalizes_budget_exhaustion_and_rejects_bad_history(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            # A budget death with no trial of its own still finalizes — on the
            # warm incumbent here, or on a better Phase-C row when one exists.
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "budget_exhausted",
                        "trials": [],
                    }
                ]
            }
            closed = phase_c_action(report, candidate)
            self.assertEqual(
                (closed["action"], closed["reason"]),
                ("finalize", "terminal_budget_exhausted"),
            )
            self.assertEqual(closed["best_score"], 1.0)

            report["phase_c"]["stages"][0]["trials"] = [
                {"params": {"x0": 0.5}, "score": 0.5}
            ]
            with_trial = phase_c_action(report, candidate)
            self.assertEqual(with_trial["action"], "finalize")
            self.assertEqual(with_trial["best_score"], 0.5)

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
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(stage["status"], "running")
            self.assertIn(DEEP_TUNE_INVOCATION_STARTED_AT, stage)

            # Simulate process death: the OS releases the lock while the running
            # journal remains. The next invocation must be admissible and recover it.
            first["_phase_c_lock_handle"].close()
            resumed = deep_tune_time_budget(candidate, report_path, "grid")
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

class BoutAdmissionTest(unittest.TestCase):
    def _fixture(self, root: Path):
        run_dir = root / "run"
        candidate_dir = run_dir / "candidates" / "001"
        candidate_dir.mkdir(parents=True)
        candidate = candidate_dir / "train.py"
        candidate.write_text(
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
        # Admission-mechanics tests predate the regime-conditioned inner
        # policy: keep the legacy uniform chains (grid primary at <=2 dims).
        (run_dir / "framework_cfg.json").write_text(
            json.dumps({"tuner": {"inner_policy": "legacy"}})
        )
        report_path = candidate_dir / "tune_report.json"
        report = {
            "inner_policy": "legacy",
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
        }
        report["phase_a"]["candidate_code_revision"] = (
            _candidate_execution_revision(candidate)
        )
        report_path.write_text(json.dumps(report))
        return candidate, report_path

    def _close_bout_zero(self, candidate, report_path, *, improved: bool) -> None:
        """Simulate a finalized bout 0 (warm incumbent stays best)."""
        report = json.loads(report_path.read_text())
        report["phase_c"] = {
            "stages": [
                {
                    "method": "grid",
                    "status": "ok",
                    "trials": [
                        {"params": {"x": 0.5 if improved else 1.5},
                         "score": 0.5 if improved else 1.5}
                    ],
                    "elapsed_seconds": 1.0,
                }
            ]
        }
        best = 0.5 if improved else 1.0
        report["final_best_params"] = {"x": best}
        report["final_best_score"] = best
        report["applied_to_base_params"] = True
        report["last_finalized_stage_index"] = 0
        report_path.write_text(json.dumps(report))
        # finalize would have rewritten BASE_PARAMS to the applied winner
        source = candidate.read_text().replace(
            "BASE_PARAMS = {'x': 1.0}", f"BASE_PARAMS = {{'x': {best}}}"
        )
        candidate.write_text(source)

    def _finalize_active_bout(
        self, candidate, report_path, *, x, score, best
    ) -> None:
        """Close the running stage admission created, then finalize its bout."""
        report = json.loads(report_path.read_text())
        stage = report["phase_c"]["stages"][-1]
        stage.pop(DEEP_TUNE_INVOCATION_STARTED_AT, None)
        stage["status"] = "ok"
        stage["trials"] = [{"params": {"x": x}, "score": score}]
        stage["elapsed_seconds"] = 1.0
        report["final_best_params"] = {"x": best}
        report["final_best_score"] = best
        report["applied_to_base_params"] = True
        report["last_finalized_stage_index"] = (
            len(report["phase_c"]["stages"]) - 1
        )
        report_path.write_text(json.dumps(report))
        source = candidate.read_text()
        start = source.index("BASE_PARAMS = ")
        end = source.index("\n", start)
        candidate.write_text(
            source[:start] + f"BASE_PARAMS = {{'x': {best}}}" + source[end:]
        )

    def _run_validate_proposals_cli(
        self, tmp, candidate, report_path, proposals
    ) -> dict:
        proposals_path = Path(tmp) / "proposals.json"
        proposals_path.write_text(json.dumps(proposals))
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "tuners" / "tune_tools.py"),
                "validate-proposals",
                "--candidate-path", str(candidate),
                "--tune-report-json", str(report_path),
                "--proposals-json", str(proposals_path),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    def test_new_bout_admitted_after_finalized_bout(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            self._close_bout_zero(candidate, report_path, improved=False)
            budget = deep_tune_time_budget(candidate, report_path, "grid")
            try:
                self.assertEqual(budget["bout_index"], 1)
                report = json.loads(report_path.read_text())
                stages = report["phase_c"]["stages"]
                self.assertEqual(len(stages), 2)
                self.assertEqual(stages[1].get("bout_index"), 1)
                self.assertEqual(stages[1]["status"], "running")
            finally:
                budget["_phase_c_lock_handle"].close()

    def test_new_bout_refused_when_previous_bout_not_finalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "ok",
                        "trials": [{"params": {"x": 1.5}, "score": 1.5}],
                        "elapsed_seconds": 1.0,
                    }
                ]
            }
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "must be finalized"
            ):
                deep_tune_time_budget(candidate, report_path, "grid")

    def test_terminal_rerun_within_same_bout_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "ok",
                        "trials": [{"params": {"x": 1.5}, "score": 1.5}],
                        "elapsed_seconds": 1.0,
                    },
                    {
                        "method": "grid",
                        "bout_index": 1,
                        "status": "rejected",
                        "trials": [],
                    },
                    {
                        "method": "bo",
                        "bout_index": 1,
                        "status": "ok",
                        "trials": [{"params": {"x": 1.4}, "score": 1.4}],
                        "elapsed_seconds": 1.0,
                    },
                ]
            }
            report_path.write_text(json.dumps(report))
            # bo is the active final stage of bout 1 and terminal; re-admitting
            # it is a same-bout rerun (only the PRIMARY method of a terminal
            # bout can start a new bout, and that requires a finalized close).
            with self.assertRaisesRegex(DeepTuneStageAdmissionError, "cannot be rerun"):
                deep_tune_time_budget(candidate, report_path, "bo")

    def test_validate_proposals_accepts_valid_and_rejects_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            # Continuation-only: a finalized bout must exist first.
            self._close_bout_zero(candidate, report_path, improved=False)
            proposals = [
                {"x": 0.5},                    # valid, novel
                {"x": 1.0},                    # already attempted (warm row)
                {"x": 9.0},                    # outside SEARCH_SPACE
                {"x": 0.5},                    # duplicate of the accepted one
                "not-a-dict",                  # malformed
            ]
            result = validate_proposals(candidate, report_path, proposals)
            self.assertTrue(result["ok"])
            self.assertEqual(result["proposed_count"], 5)
            self.assertEqual(result["accepted"], [{"x": 0.5}])
            reasons = [r["reason"] for r in result["rejected"]]
            self.assertEqual(
                reasons,
                ["already_attempted", "out_of_space", "already_attempted",
                 "params_must_be_object"],
            )

    def test_validate_proposals_all_rejected_is_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            self._close_bout_zero(candidate, report_path, improved=False)
            result = validate_proposals(candidate, report_path, [{"x": 1.0}])
            self.assertFalse(result["ok"])
            self.assertEqual(result["accepted"], [])

    def test_validate_proposals_rejected_before_first_bout(self):
        """Re-warm proposals scope to continuations: no finalized bout, no admission."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            proposals = [{"x": 0.5}, {"x": 0.75}]
            result = validate_proposals(candidate, report_path, proposals)
            self.assertFalse(result["ok"])
            self.assertEqual(result["proposed_count"], 2)
            self.assertEqual(result["accepted"], [])
            self.assertEqual(
                [r["reason"] for r in result["rejected"]],
                ["first_bout_has_no_rewarm"] * 2,
            )
            # The CLI exits 1 and writes nothing to the report.
            proposals_path = Path(tmp) / "proposals.json"
            proposals_path.write_text(json.dumps(proposals))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "tuners" / "tune_tools.py"),
                    "validate-proposals",
                    "--candidate-path", str(candidate),
                    "--tune-report-json", str(report_path),
                    "--proposals-json", str(proposals_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 1)
            self.assertNotIn("phase_c", json.loads(report_path.read_text()))

    def test_validate_proposals_fails_closed_on_inconsistent_close(self):
        """An applied_to_base_params claim without finalizable provenance is an
        error, not a rejection: has_applied_close raises and it propagates."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            report["applied_to_base_params"] = True  # forged close
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "not finalizable"):
                validate_proposals(candidate, report_path, [{"x": 0.25}])

    def test_validate_proposals_cli_writes_pending_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            self._close_bout_zero(candidate, report_path, improved=False)
            result = self._run_validate_proposals_cli(
                tmp, candidate, report_path, [{"x": 0.5}, {"x": 9.0}]
            )
            self.assertTrue(result["ok"])
            report = json.loads(report_path.read_text())
            self.assertEqual(
                report["phase_c"]["pending_proposals"], [{"x": 0.5}]
            )
            # The list is tagged with the bout it targets (the next new bout).
            self.assertEqual(
                report["phase_c"]["pending_proposals_bout_index"], 1
            )

    def test_stale_pending_proposals_cleared_at_new_bout_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            self._close_bout_zero(candidate, report_path, improved=False)
            self._run_validate_proposals_cli(
                tmp, candidate, report_path, [{"x": 0.6}]
            )
            # Bout 1 admission keeps THIS bout's freshly validated proposals:
            # the orchestrator validates before launching the search, so the
            # clear must not wipe the list written for the bout now starting.
            budget = deep_tune_time_budget(candidate, report_path, "grid")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["phase_c"]["pending_proposals"], [{"x": 0.6}])
            self.assertEqual(report["phase_c"]["pending_proposals_bout_index"], 1)
            budget["_phase_c_lock_handle"].close()
            # Bout 1 attempts the proposal and finalizes; the consumed list
            # stays in the report (nothing deletes it at consumption).
            self._finalize_active_bout(
                candidate, report_path, x=0.6, score=0.6, best=0.6
            )
            # Bout 2 starts with no new validate-proposals call: the leftover
            # from bout 1 must be cleared at admission, never re-consumed.
            budget = deep_tune_time_budget(candidate, report_path, "grid")
            try:
                self.assertEqual(budget["bout_index"], 2)
                report = json.loads(report_path.read_text())
                self.assertNotIn("pending_proposals", report["phase_c"])
                self.assertNotIn(
                    "pending_proposals_bout_index", report["phase_c"]
                )
                self.assertEqual(read_pending_proposals(report_path), [])
            finally:
                budget["_phase_c_lock_handle"].close()

    def test_fresh_proposals_for_new_bout_survive_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            self._close_bout_zero(candidate, report_path, improved=False)
            self._run_validate_proposals_cli(
                tmp, candidate, report_path, [{"x": 0.6}]
            )
            budget = deep_tune_time_budget(candidate, report_path, "grid")
            budget["_phase_c_lock_handle"].close()
            # The proposal is never attempted (the bout closes on another
            # trial), so the tagged list is a genuine leftover.
            self._finalize_active_bout(
                candidate, report_path, x=1.8, score=1.8, best=1.0
            )
            # A fresh validate-proposals for bout 2 rewrites the list with the
            # new tag; admission must keep it, not clear it as stale.
            self._run_validate_proposals_cli(
                tmp, candidate, report_path, [{"x": 0.7}]
            )
            budget = deep_tune_time_budget(candidate, report_path, "grid")
            try:
                self.assertEqual(budget["bout_index"], 2)
                report = json.loads(report_path.read_text())
                self.assertEqual(
                    report["phase_c"]["pending_proposals"], [{"x": 0.7}]
                )
                self.assertEqual(
                    report["phase_c"]["pending_proposals_bout_index"], 2
                )
            finally:
                budget["_phase_c_lock_handle"].close()

    def test_continuation_bout_proposals_displace_search_trials(self):
        """A continuation bout spends at most trial_cap objective attempts:
        proposals first, then deferred, then the grid sweep."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, report_path = self._fixture(root)
            report = json.loads(report_path.read_text())
            report["phase_a"]["deferred_configs"] = [
                {"params": {"x": 0.25}},
                {"params": {"x": 1.75}},
            ]
            report_path.write_text(json.dumps(report))
            self._close_bout_zero(candidate, report_path, improved=False)
            self._run_validate_proposals_cli(
                tmp, candidate, report_path, [{"x": 0.5}, {"x": 0.75}]
            )
            train_module = mock.Mock(
                SEARCH_SPACE={"x": ("float", 0.0, 2.0)},
                BASE_PARAMS={"x": 1.0},
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
                "grid_search.timed_eval",
                # Improving scores keep patience from firing before the cap.
                side_effect=[0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6],
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
                    "3",
                    "--max-trials",
                    "5",
                ],
            ):
                self.assertEqual(grid_main(), 0)

            # trial_cap (5) bounds the whole bout: 2 proposals + 2 deferred +
            # 1 grid point. The unseen grid sweep had 2 more points
            # ({0.0, 2.0}); without the cap the bout would spend 6.
            attempted = [
                call.args[2] for call in timed_eval_mock.call_args_list
            ]
            self.assertEqual(len(attempted), 5)
            self.assertEqual(
                attempted[:4],
                [{"x": 0.5}, {"x": 0.75}, {"x": 0.25}, {"x": 1.75}],
            )
            self.assertIn(attempted[4], ({"x": 0.0}, {"x": 2.0}))
            result = write_result.call_args.args[0]
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["trials_attempted"], 5)
            self.assertEqual(result["early_stop_reason"], "max_trials")


class CloseExhaustedStageTest(unittest.TestCase):
    """The deterministic close for a stage no invocation will ever resume."""

    def _run_fixture(self, root: Path, *, max_evaluations: int, phase_c_attempts: int):
        run_dir = root / "runs" / "autoresearch-baseline" / "tag"
        candidate = run_dir / "candidates" / "001" / "train.py"
        candidate.parent.mkdir(parents=True)
        candidate.write_text(
            "PARAM_SCHEMA = {'x0': 'float'}\n"
            "SEARCH_SPACE = {'x0': ('float', 0.0, 1.0)}\n"
            "BASE_PARAMS = {'x0': 0.0}\n"
            "def make_model(params):\n"
            "    return params\n"
        )
        (candidate.parent / "prepare.py").write_text(
            "def evaluate_config(make_model, params):\n"
            "    return 0.0\n"
        )
        (run_dir / "framework_cfg.json").write_text(
            json.dumps(
                {
                    "max_evaluations": max_evaluations,
                    "tuner": {
                        "deep_tune_budget_fraction": 0.4,
                        "deep_tune_per_candidate_cap": 2,
                        "inner_policy": "legacy",
                    },
                }
            )
        )
        rows = [{
            "schema_version": 1,
            "kind": "score_attempt",
            "attempt_id": "eval-000001",
            "run_id": "001",
            "phase": "phase_a",
            "method": "warmstart",
            "params_sha256": "sha256:" + "0" * 64,
        }]
        for index in range(phase_c_attempts):
            rows.append({
                "schema_version": 1,
                "kind": "score_attempt",
                "attempt_id": f"eval-{index + 2:06d}",
                "run_id": "001",
                "phase": "phase_c",
                "method": "grid",
                "params_sha256": f"sha256:{index:064d}",
            })
        (run_dir / "evaluation_attempts.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
        report = {
            "inner_policy": "legacy",
            "phase_a": {
                "status": "ok",
                "trials_attempted": 1,
                "candidate_code_revision": _candidate_execution_revision(candidate),
                "search_space": {"x0": ["float", 0.0, 1.0]},
                "warm_start_configs": [{"params": {"x0": 0.0}, "score": 1.0}],
                "best_warm_params": {"x0": 0.0},
                "best_warm_score": 1.0,
            },
            "phase_c": {
                "stages": [
                    {
                        # First method of the 1-dim deterministic chain, so the
                        # closed report is finalizable without a rejected prefix.
                        "method": "grid",
                        "status": "running",
                        "trials": [{"params": {"x0": 0.5}, "score": 0.5}],
                    }
                ]
            },
        }
        report_path = candidate.parent / "tune_report.json"
        report_path.write_text(json.dumps(report))
        return candidate, report_path

    def test_refuses_while_the_budget_still_admits_a_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._run_fixture(
                Path(tmp), max_evaluations=100, phase_c_attempts=1
            )
            with self.assertRaisesRegex(ValueError, "still.*admits a reservation"):
                close_exhausted_stage(candidate, report_path)
            # The refusal must not mutate the stage.
            stage = json.loads(report_path.read_text())["phase_c"]["stages"][0]
            self.assertEqual(stage["status"], "running")

    def test_closes_at_the_per_candidate_cap_and_frees_its_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._run_fixture(
                Path(tmp), max_evaluations=100, phase_c_attempts=2
            )
            result = close_exhausted_stage(candidate, report_path)
            self.assertEqual(result["action"], "closed")
            self.assertEqual(result["budget_scope"], "deep_tune_candidate:001")
            self.assertEqual(result["trials_at_close"], 1)

            report = json.loads(report_path.read_text())
            stage = report["phase_c"]["stages"][0]
            self.assertEqual(stage["status"], "budget_exhausted")
            self.assertTrue(stage["closed_without_invocation"])
            self.assertEqual(stage["trials_at_close"], 1)

            # The point of the close: the stranded 0.5 trial is now finalizable.
            self.assertEqual(
                finalizable_tuning_result(report)["best_score"], 0.5
            )

            # Idempotent.
            again = close_exhausted_stage(candidate, report_path)
            self.assertEqual(again["action"], "noop")

    def test_closes_when_the_global_budget_is_spent(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._run_fixture(
                Path(tmp), max_evaluations=1, phase_c_attempts=1
            )
            result = close_exhausted_stage(candidate, report_path)
            self.assertEqual(
                (result["action"], result["budget_scope"]), ("closed", "global")
            )

    def test_phase_c_action_advises_the_close_instead_of_a_dead_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._run_fixture(
                Path(tmp), max_evaluations=100, phase_c_attempts=2
            )
            report = json.loads(report_path.read_text())
            # The candidate is at its per-candidate cap, so "run" advice could
            # never succeed: the helper must route to the deterministic close
            # itself rather than leaving it to prompt prose.
            advice = phase_c_action(report, candidate)
            self.assertEqual(advice["action"], "close_exhausted_stage")
            self.assertEqual(advice["reason"], "evaluation_budget_reached")
            self.assertEqual(advice["budget_scope"], "deep_tune_candidate:001")

    def test_phase_c_action_still_resumes_a_stage_the_budget_can_fund(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._run_fixture(
                Path(tmp), max_evaluations=100, phase_c_attempts=1
            )
            report = json.loads(report_path.read_text())
            advice = phase_c_action(report, candidate)
            self.assertEqual(
                (advice["action"], advice["method"], advice["reason"]),
                ("run", "grid", "resume_interrupted_stage"),
            )


def _candidate_record(
    run_id: str,
    warm: float,
    *,
    tune: bool = False,
    bouts: int = 0,
    improved: bool | None = None,
    final: float | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "status": "keep",
        "best_warm_score": warm,
        "tune": tune,
        "tuning_bouts": bouts,
        "last_bout_improved": improved,
        "final_best_score": final if final is not None else warm,
    }


class ProgressiveSelectCandidateTest(unittest.TestCase):
    def _ledger(self, records) -> dict:
        return {"records": records}

    def test_first_bout_still_requires_percentile_gate(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)
        self.assertEqual(result["bout_index"], 0)

    def test_continuation_selected_when_fresh_fails_gate(self):
        # Fresh candidates carry the WORST warm scores, so the best fresh
        # warm percentile is 75 (< 80) and the gate refuses a first bout.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00),
            # tuned responder: continuation pools ignore warm rank
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=True, final=0.80),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "005")
        self.assertIs(result["is_continuation"], True)
        self.assertEqual(result["bout_index"], 1)

    def test_fresh_first_bout_beats_continuation(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.40, tune=True, bouts=1,
                              improved=True, final=0.70),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)

    def test_continuation_ranks_fewest_bouts_then_tuned_score(self):
        # Fresh warm scores are worst (best fresh percentile 50 < 80), so the
        # choice falls to continuations: fewer bouts beats better tuned score.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00, tune=True, bouts=2,
                              improved=True, final=0.60),
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=True, final=0.90),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "005")  # fewer bouts wins over better score

    def test_non_responders_never_selected(self):
        # Fresh gate fails (percentile 75) and the only tuned candidate did
        # not improve in its last bout — and it is not the run-best
        # (004's 1.00 beats its 1.05), so no incumbent retry either.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00),
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=False, final=1.05),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertIsNone(result["run_id"])
        self.assertIn("non-responder", result["reason"])

    def test_trial_cap_includes_bout_trials(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
        ])
        allocation = {
            "remaining": 100,
            "deep_tune": {
                "remaining": None,
                "total_cap": None,
                "per_candidate_cap": 20,
                "per_candidate": [],
                "time_limit_seconds": None,
            },
        }
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0,
            bout_trials=8, budget_allocation=allocation,
        )
        self.assertEqual(result["run_id"], "001")
        self.assertEqual(result["budget_allocation"]["trial_cap"], 8)

    def test_alternation_responder_follows_first_bout(self):
        # After a first bout, the waiting responder wins even though the best
        # fresh candidate passes the percentile gate.
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=1,
                              improved=True, final=0.94),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=True
        )
        self.assertEqual(result["run_id"], "006")
        self.assertIs(result["is_continuation"], True)
        self.assertIn("alternation", result["reason"])

    def test_alternation_fresh_follows_continuation(self):
        # Same population, but the last bout was a continuation: the
        # gate-passing fresh candidate wins.
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=2,
                              improved=True, final=0.94),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=False
        )
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)

    def test_alternation_none_preserves_legacy_order(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=1,
                              improved=True, final=0.94),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)

    def test_alternation_without_responder_falls_back_to_fresh(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=1,
                              improved=False, final=0.95),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=True
        )
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)

    def test_incumbent_retry_after_failed_first_bout(self):
        # Fresh candidates carry the worst warm scores (gate fails); the only
        # tuned candidate failed its first bout but holds the run's best final
        # score — it earns one confirmation bout.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00),
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=False, final=0.70),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "005")
        self.assertIs(result["is_continuation"], True)
        self.assertIn("incumbent retry", result["reason"])

    def test_non_incumbent_non_responder_not_retried(self):
        # Same shape, but a better final score exists elsewhere: no retry.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 0.80),  # better than the non-responder
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=False, final=0.90),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        # 004 passes the fresh gate (best fresh, percentile 100).
        self.assertEqual(result["run_id"], "004")
        self.assertIs(result["is_continuation"], False)

    def test_retry_used_up_after_two_bouts(self):
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00),
            _candidate_record("005", 0.95, tune=True, bouts=2,
                              improved=False, final=0.70),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertIsNone(result["run_id"])

    def test_responder_outranks_incumbent_retry(self):
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00),
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=False, final=0.70),
            _candidate_record("006", 1.05, tune=True, bouts=1,
                              improved=True, final=0.75),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=True
        )
        self.assertEqual(result["run_id"], "006")
        self.assertIs(result["is_continuation"], True)

    def test_incumbent_retry_beats_fresh_under_alternation(self):
        # After a first bout, the incumbent's confirmation bout precedes the
        # fresh gate even when the best fresh candidate passes it.
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.85, tune=True, bouts=1,
                              improved=False, final=0.80),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=True
        )
        self.assertEqual(result["run_id"], "006")
        self.assertIs(result["is_continuation"], True)


class LastBoutWasFirstTest(unittest.TestCase):
    def _attempts(self, root: Path, rows: list[dict]) -> Path:
        path = root / "evaluation_attempts.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_last_finalized_first_bout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._attempts(root, [
                {"kind": "score_attempt", "phase": "phase_a", "run_id": "001"},
                {"kind": "score_attempt", "phase": "phase_c", "run_id": "001"},
            ])
            ledger = {"records": [
                {"run_id": "001", "status": "keep", "tune": True, "tuning_bouts": 1},
            ]}
            self.assertIs(_last_bout_was_first(root, ledger), True)

    def test_last_finalized_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._attempts(root, [
                {"kind": "score_attempt", "phase": "phase_c", "run_id": "001"},
            ])
            ledger = {"records": [
                {"run_id": "001", "status": "keep", "tune": True, "tuning_bouts": 2},
            ]}
            self.assertIs(_last_bout_was_first(root, ledger), False)

    def test_in_flight_bout_yields_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._attempts(root, [
                {"kind": "score_attempt", "phase": "phase_c", "run_id": "001"},
            ])
            ledger = {"records": [
                {"run_id": "001", "status": "keep"},  # no tune flag yet
            ]}
            self.assertIsNone(_last_bout_was_first(root, ledger))

    def test_missing_attempts_file_yields_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_last_bout_was_first(Path(tmp), {"records": []}))


if __name__ == "__main__":
    unittest.main()
