from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_cfg  # noqa: E402


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
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, None):
            with self.subTest(fraction=value):
                with self.assertRaisesRegex(
                    run_cfg.RunConfigError,
                    "deep_tune_budget_fraction",
                ):
                    self._read({"deep_tune_budget_fraction": value})
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


if __name__ == "__main__":
    unittest.main()
