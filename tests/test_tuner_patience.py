from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

from _common import PatienceMonitor, prior_patience_state, write_tune_report  # noqa: E402


class PatienceMonitorTest(unittest.TestCase):
    def test_failed_trials_count_toward_patience(self):
        monitor = PatienceMonitor(patience=3, start_best=1.0)
        self.assertFalse(monitor.update_failed())
        self.assertFalse(monitor.update_failed())
        self.assertTrue(monitor.update_failed())
        # A failure streak must never move the incumbent best.
        self.assertEqual(monitor.best, 1.0)

    def test_improvement_resets_failure_streak(self):
        monitor = PatienceMonitor(patience=2, start_best=1.0)
        self.assertFalse(monitor.update_failed())
        self.assertFalse(monitor.update(0.9))  # improvement resets
        self.assertFalse(monitor.update_failed())
        self.assertTrue(monitor.update_failed())

    def test_start_since_seeds_streak_for_resume(self):
        # A resumed search must continue the persisted streak, not restart it:
        # with 2 prior non-improving trials and patience 3, one more
        # non-improvement stops the search.
        monitor = PatienceMonitor(patience=3, start_best=1.0, start_since=2)
        self.assertTrue(monitor.update_failed())
        monitor = PatienceMonitor(patience=3, start_best=1.0, start_since=2)
        self.assertFalse(monitor.update(0.95))  # improvement still resets
        self.assertFalse(monitor.update_failed())


def _write_report(
    path: Path, warm_scores, stage_trials, bout_index: int | None = None
) -> None:
    stage = {
        "method": "bo",
        "trials": [
            {"params": {"x": float(i)}, "score": s, "status": st}
            for i, (s, st) in enumerate(stage_trials)
        ],
    }
    if bout_index is not None:
        stage["bout_index"] = bout_index
    write_tune_report(
        path,
        {
            "phase_a": {
                "warm_start_configs": [
                    {"params": {"x": float(i)}, "score": s}
                    for i, s in enumerate(warm_scores)
                ]
            },
            "phase_c": {
                "stages": (
                    [stage] if stage_trials or bout_index is not None else []
                )
            },
        },
    )


def _write_two_bout_report(path: Path) -> None:
    write_tune_report(
        path,
        {
            "phase_a": {
                "warm_start_configs": [{"params": {"x": 1.0}, "score": 1.07}]
            },
            "phase_c": {
                "stages": [
                    {
                        "method": "bo",
                        "trials": [
                            {"params": {"x": 2.0}, "score": 1.05},
                            {"params": {"x": 3.0}, "score": None, "status": "failed"},
                        ],
                    },
                    {
                        "method": "bo",
                        "bout_index": 1,
                        "trials": [
                            {"params": {"x": 4.0}, "score": 1.06},
                        ],
                    },
                ]
            },
        }
    )


class PriorPatienceStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.report_path = Path(self.tmp.name) / "tune_report.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_empty_report(self):
        _write_report(self.report_path, [], [])
        self.assertEqual(prior_patience_state(self.report_path), (None, 0))

    def test_streak_counts_since_last_improvement(self):
        # warm: 1.10 then 1.07 (best). stage: 1.08 worse, 1.05 better,
        # then failed, worse, preflight_rejected → streak 3, best 1.05.
        _write_report(
            self.report_path,
            [1.10, 1.07],
            [
                (1.08, "ok"),
                (1.05, "ok"),
                (None, "failed"),
                (1.06, "ok"),
                (None, "preflight_rejected"),
            ],
        )
        best, streak = prior_patience_state(self.report_path)
        self.assertEqual(best, 1.05)
        self.assertEqual(streak, 3)

    def test_failures_before_any_success_count(self):
        _write_report(
            self.report_path,
            [],
            [(None, "failed"), (None, "preflight_rejected"), (None, "failed")],
        )
        best, streak = prior_patience_state(self.report_path)
        self.assertIsNone(best)
        self.assertEqual(streak, 3)

    def test_fidelity_control_cannot_set_or_reset_patience_best(self):
        write_tune_report(
            self.report_path,
            {
                "phase_a": {
                    "warm_start_configs": [
                        {
                            "params": {"x": 1.0},
                            "score": 0.1,
                            "role": "inherited_control",
                        },
                        {"params": {"x": 2.0}, "score": 0.4},
                    ]
                },
                "phase_c": {
                    "stages": [
                        {
                            "trials": [
                                {"params": {"x": 3.0}, "score": 0.3},
                                {"params": {"x": 1.0}, "score": 0.05},
                            ]
                        }
                    ]
                },
            },
        )

        self.assertEqual(prior_patience_state(self.report_path), (0.3, 1))


class BoutPatienceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.report_path = Path(self.tmp.name) / "tune_report.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_streak_scoped_to_current_bout_best_is_global(self):
        _write_two_bout_report(self.report_path)
        best, streak = prior_patience_state(self.report_path)
        self.assertEqual(best, 1.05)
        self.assertEqual(streak, 1)  # only bout 1's non-improving trial counts

    def test_explicit_earlier_bout_replays_that_bout(self):
        _write_two_bout_report(self.report_path)
        best, streak = prior_patience_state(self.report_path, bout_index=0)
        self.assertEqual(best, 1.05)
        self.assertEqual(streak, 1)  # bout 0: improvement then failed trial


if __name__ == "__main__":
    unittest.main()
