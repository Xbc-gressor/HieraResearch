from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_cfg  # noqa: E402
from run_cfg import RunConfigError, read_framework_cfg  # noqa: E402


class RunConfigTunerValidationTests(unittest.TestCase):
    def _read(self, tuner: dict) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "framework_cfg.json"
            path.write_text(json.dumps({"tuner": tuner}))
            return run_cfg.read_framework_cfg(path)

    def test_valid_tuner_overrides_are_preserved(self) -> None:
        tuner = {
            "K_eval": 3,
            "n_min": 1,
            "top_percentile": 0,
            "bo_n_trials": 1,
            "bo_patience": 1,
            "bo_patience_cap": 1,
            "bo_patience_floor": 2,
            "deep_tune_budget_fraction": 0.4,
            "deep_tune_per_candidate_cap": 20,
            "deep_tune_time_limit_seconds": 3600,
        }

        self.assertEqual(self._read(tuner)["tuner"], tuner)

        tuner["top_percentile"] = 99.999
        self.assertEqual(self._read(tuner)["tuner"], tuner)

    def test_nullable_overrides_match_consumer_default_semantics(self) -> None:
        tuner = {
            "K_eval": None,
            "n_min": None,
            "bo_patience": None,
        }

        self.assertEqual(self._read(tuner)["tuner"], tuner)

    def test_n_min_must_be_a_positive_integer(self) -> None:
        for value in (0, -1, 1.5, True, "5"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    run_cfg.RunConfigError,
                    "tuner.n_min must be a positive integer or null",
                ):
                    self._read({"n_min": value})

    def test_k_eval_reserves_a_selectable_row_beyond_the_control(self) -> None:
        with self.assertRaisesRegex(
            run_cfg.RunConfigError,
            "tuner.K_eval must be at least 2",
        ):
            self._read({"K_eval": 1})

    def test_k_proposes_a_row_beyond_the_control(self) -> None:
        with self.assertRaisesRegex(
            run_cfg.RunConfigError,
            "tuner.K must be at least 2",
        ):
            self._read({"K": 1})

    def test_top_percentile_must_be_finite_and_half_open_bounded(self) -> None:
        for value in (
            -0.1,
            100,
            10**1000,
            float("nan"),
            float("inf"),
            True,
            "80",
            None,
        ):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    run_cfg.RunConfigError,
                    r"tuner.top_percentile must be a finite number in \[0, 100\)",
                ):
                    self._read({"top_percentile": value})

    def test_bo_integer_overrides_reject_invalid_values(self) -> None:
        for key in (
            "bo_n_trials",
            "bo_patience",
            "bo_patience_cap",
            "bo_patience_floor",
        ):
            for value in (0, -1, 1.5, True, "5"):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(
                        run_cfg.RunConfigError,
                        f"tuner.{key} must be a positive integer",
                    ):
                        self._read({key: value})

    def test_nonnullable_bo_overrides_reject_explicit_null(self) -> None:
        for key in ("bo_n_trials", "bo_patience_cap", "bo_patience_floor"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(
                    run_cfg.RunConfigError,
                    f"tuner.{key} must be a positive integer",
                ):
                    self._read({key: None})

    def test_adaptive_patience_floor_cannot_exceed_cap(self) -> None:
        for tuner in (
            {"bo_patience_floor": 21},
            {"bo_patience_cap": 11},
            {"bo_patience_floor": 10, "bo_patience_cap": 9},
        ):
            with self.subTest(tuner=tuner):
                with self.assertRaisesRegex(
                    run_cfg.RunConfigError,
                    "bo_patience_floor must be less than or equal",
                ):
                    self._read(tuner)

    def test_fixed_patience_does_not_require_an_ordered_adaptive_pair(self) -> None:
        tuner = {
            "bo_patience": 8,
            "bo_patience_floor": 20,
            "bo_patience_cap": 10,
        }

        self.assertEqual(self._read(tuner)["tuner"], tuner)

    def test_deep_tune_budget_controls_are_bounded(self) -> None:
        for value in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(fraction=value):
                with self.assertRaisesRegex(
                    run_cfg.RunConfigError,
                    "deep_tune_budget_fraction",
                ):
                    self._read({"deep_tune_budget_fraction": value})
        # null is the default: no run-level Phase-C share at all.
        self.assertIsNone(
            self._read({"deep_tune_budget_fraction": None})["tuner"][
                "deep_tune_budget_fraction"
            ]
        )
        for key in ("deep_tune_per_candidate_cap",):
            for value in (0, -1, 1.5, True, None):
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(
                        run_cfg.RunConfigError,
                        f"tuner.{key} must be a positive integer",
                    ):
                        self._read({key: value})
        for value in (0, -1, float("nan"), True, "3600", None):
            with self.subTest(time_limit=value):
                with self.assertRaisesRegex(
                    run_cfg.RunConfigError,
                    "deep_tune_time_limit_seconds",
                ):
                    self._read({"deep_tune_time_limit_seconds": value})

    def test_mixup_turbo_requires_complete_anchor_contract(self) -> None:
        base = {
            "max_evaluations": 68,
            "tuner": {
                "scheduler_policy": "anchor_challenger_v1",
                "inner_policy": "mixup24-turbo20-v1",
                "deep_tune_budget_fraction": None,
                "deep_tune_per_candidate_cap": 44,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "framework_cfg.json"
            path.write_text(json.dumps(base))
            self.assertEqual(
                read_framework_cfg(path)["tuner"]["inner_policy"],
                "mixup24-turbo20-v1",
            )

            hebo = json.loads(json.dumps(base))
            hebo["tuner"]["inner_policy"] = "hebo24-turbo20-v1"
            path.write_text(json.dumps(hebo))
            self.assertEqual(
                read_framework_cfg(path)["tuner"]["inner_policy"],
                "hebo24-turbo20-v1",
            )

            hebo_only = json.loads(json.dumps(base))
            hebo_only["tuner"]["inner_policy"] = "hebo24-hebo20"
            path.write_text(json.dumps(hebo_only))
            self.assertEqual(
                read_framework_cfg(path)["tuner"]["inner_policy"],
                "hebo24-hebo20",
            )

            bad_scheduler = json.loads(json.dumps(base))
            bad_scheduler["tuner"]["scheduler_policy"] = "v3_2"
            path.write_text(json.dumps(bad_scheduler))
            with self.assertRaisesRegex(
                RunConfigError, "requires tuner.scheduler_policy"
            ):
                read_framework_cfg(path)

            short_cap = json.loads(json.dumps(base))
            short_cap["tuner"]["deep_tune_per_candidate_cap"] = 43
            path.write_text(json.dumps(short_cap))
            with self.assertRaisesRegex(RunConfigError, "three-bout schedule"):
                read_framework_cfg(path)

            short_run = json.loads(json.dumps(base))
            short_run["max_evaluations"] = 67
            path.write_text(json.dumps(short_run))
            with self.assertRaisesRegex(RunConfigError, "tournament reserve"):
                read_framework_cfg(path)


class ProgressiveTunerKnobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg_path = Path(self.tmp.name) / "framework_cfg.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, tuner: dict) -> None:
        self.cfg_path.write_text(json.dumps({"tuner": tuner}))

    def test_valid_progressive_knobs_parse(self):
        self._write({"bout_trials": 8, "tuned_threshold": 16, "rewarm_proposals": 3})
        cfg = read_framework_cfg(self.cfg_path)
        self.assertEqual(cfg["tuner"]["bout_trials"], 8)
        self.assertEqual(cfg["tuner"]["tuned_threshold"], 16)
        self.assertEqual(cfg["tuner"]["rewarm_proposals"], 3)

    def test_invalid_progressive_knobs_rejected(self):
        for key in ("bout_trials", "tuned_threshold", "rewarm_proposals"):
            for bad in (0, -1, 2.5, "8", True):
                with self.subTest(key=key, bad=bad):
                    self._write({key: bad})
                    with self.assertRaises(RunConfigError):
                        read_framework_cfg(self.cfg_path)


if __name__ == "__main__":
    unittest.main()
