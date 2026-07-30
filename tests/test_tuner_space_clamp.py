from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import _common  # noqa: E402
from _common import (  # noqa: E402
    clamp_search_space_to_preflight,
    params_within_search_space,
    split_configs_by_space,
)
from grid_search import expand_entry  # noqa: E402

TOTAL_VRAM = 80000.0
HEADROOM = 0.85  # matches _common.SPACE_CLAMP_HEADROOM

SPACE = {
    "device_batch_size": ("int", 64, 249),
    "depth": ("int", 4, 12),
    "matrix_lr": ("float", 0.001, 0.1, "log"),
}
BASE = {"device_batch_size": 128, "depth": 8, "matrix_lr": 0.04}


def fake_preflight(params, candidate_path):
    """OOM above device_batch_size 150; peak telemetry scales with batch size."""
    if params["device_batch_size"] > 150:
        raise RuntimeError("CUDA out of memory")
    return {"status": "ok", "peak_vram_mb": params["device_batch_size"] * 300.0}


class ClampTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        run_dir = Path(self.tmp.name)
        (run_dir / "framework_cfg.json").write_text("{}")
        (run_dir / "environment_preflight.json").write_text(json.dumps(
            {"hook_result": {"total_vram_mb": TOTAL_VRAM}}
        ))
        cand_dir = run_dir / "candidates" / "000"
        cand_dir.mkdir(parents=True)
        (cand_dir / "train.py").write_text("# candidate")
        self.candidate_path = cand_dir / "train.py"
        self.report_path = cand_dir / "tune_report.json"
        self.probes = []

    def tearDown(self):
        self.tmp.cleanup()

    def _run_clamp(self, preflight=fake_preflight, space=SPACE, base=BASE):
        self.probes.append("marker")  # count via appended preflight attempts
        with mock.patch.object(
            _common, "_configured_preflight_name", return_value="preflight_config"
        ), mock.patch.object(_common, "timed_preflight", side_effect=preflight):
            return clamp_search_space_to_preflight(
                space, base, self.candidate_path, self.report_path
            )

    def _probe_count(self):
        report = json.loads(self.report_path.read_text())
        return len(report.get("preflight", {}).get("attempts", []))

    def test_feasible_corner_costs_one_probe_and_changes_nothing(self):
        def all_ok(params, candidate_path):
            return {"status": "ok", "peak_vram_mb": 40000.0}

        clamped = self._run_clamp(preflight=all_ok)
        self.assertEqual(clamped, SPACE)
        self.assertEqual(self._probe_count(), 1)
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertTrue(receipt["corner_feasible"])
        self.assertEqual(receipt["outcome"], "already_feasible")

    def test_oom_corner_clamps_only_the_failing_dim(self):
        clamped = self._run_clamp()
        # device_batch_size bisects to the largest feasible bound (150);
        # depth and matrix_lr were feasible and stay untouched.
        self.assertEqual(clamped["device_batch_size"], ("int", 64, 150))
        self.assertEqual(clamped["depth"], ("int", 4, 12))
        self.assertEqual(clamped["matrix_lr"], ("float", 0.001, 0.1, "log"))
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertTrue(receipt["corner_feasible"])
        self.assertEqual(receipt["outcome"], "clamped_feasible")
        self.assertGreater(self._probe_count(), 1)

    def test_result_is_cached_across_calls(self):
        first = self._run_clamp()
        probes_after_first = self._probe_count()
        second = self._run_clamp()
        self.assertEqual(first, second)
        self.assertEqual(self._probe_count(), probes_after_first)

    def test_legacy_or_unverified_cache_is_reprobed(self):
        def all_ok(params, candidate_path):
            return {"status": "ok", "peak_vram_mb": 40000.0}

        self._run_clamp(preflight=all_ok)
        probes_after_first = self._probe_count()
        report = json.loads(self.report_path.read_text())
        legacy = report["search_space_clamp"]
        legacy["schema_version"] = 1
        legacy.pop("algorithm_version")
        legacy["outcome"] = "unclamped"
        legacy["corner_feasible"] = False
        self.report_path.write_text(json.dumps(report))

        clamped = self._run_clamp(preflight=all_ok)

        self.assertEqual(clamped, SPACE)
        self.assertGreater(self._probe_count(), probes_after_first)
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertEqual(receipt["schema_version"], 2)
        self.assertEqual(receipt["algorithm_version"], 2)
        self.assertEqual(receipt["outcome"], "already_feasible")
        self.assertTrue(receipt["corner_feasible"])

    def test_missing_vram_telemetry_is_a_noop(self):
        (Path(self.tmp.name) / "environment_preflight.json").unlink()
        clamped = self._run_clamp()
        self.assertEqual(clamped, SPACE)
        self.assertFalse(self.report_path.exists())

    def test_infeasible_base_is_a_noop(self):
        base = dict(BASE, device_batch_size=200)  # base itself OOMs
        clamped = self._run_clamp(base=base)
        self.assertEqual(clamped, SPACE)
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertFalse(receipt["corner_feasible"])
        self.assertEqual(receipt["outcome"], "base_infeasible_noop")

    def test_peak_beyond_headroom_counts_as_infeasible(self):
        # Passes preflight (no exception) but peaks above the headroom: the
        # full-run OOM pattern. dbs 200 peaks at 0.9 * total -> infeasible.
        def tight(params, candidate_path):
            return {"status": "ok", "peak_vram_mb": params["device_batch_size"] * 360.0}

        clamped = self._run_clamp(preflight=tight)
        # 68000 MB headroom / 360 MB-per-unit -> largest feasible dbs is 188.
        self.assertLessEqual(clamped["device_batch_size"][2], 188)
        self.assertGreaterEqual(clamped["device_batch_size"][2], 128)

    @staticmethod
    def _bilinear_preflight(params, candidate_path):
        """OOM above 80 GB; peak scales with batch x depth (interaction)."""
        peak = 300.0 * params["device_batch_size"] * params["depth"] / 8
        if peak > 80000.0:
            raise RuntimeError("CUDA out of memory")
        return {"status": "ok", "peak_vram_mb": peak}

    def test_unresolvable_interaction_collapses_to_base_box(self):
        # With pull rounds disabled, the bisected corner still OOMs jointly:
        # the clamp must not return a known-infeasible box — it collapses the
        # memory contributors to the (proven-feasible) base point instead.
        with mock.patch.object(_common, "SPACE_CLAMP_MAX_PULL_ROUNDS", 0):
            clamped = self._run_clamp(preflight=self._bilinear_preflight)
        self.assertEqual(clamped["device_batch_size"], ("int", 64, 128))
        self.assertEqual(clamped["depth"], ("int", 4, 8))
        # Memory-neutral dims are not contributors and stay untouched.
        self.assertEqual(clamped["matrix_lr"], ("float", 0.001, 0.1, "log"))
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertTrue(receipt["corner_feasible"])
        self.assertEqual(receipt["outcome"], "collapsed_to_base")

    def test_probe_budget_exhaustion_falls_back_to_base_box(self):
        with mock.patch.object(_common, "SPACE_CLAMP_MAX_PROBES", 4):
            clamped = self._run_clamp(preflight=self._bilinear_preflight)
        # Budget hit during bisection: collapse every clampable dim to base.
        self.assertEqual(clamped["device_batch_size"], ("int", 64, 128))
        self.assertEqual(clamped["depth"], ("int", 4, 8))
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertTrue(receipt["corner_feasible"])
        self.assertEqual(receipt["outcome"], "collapsed_to_base")
        self.assertLessEqual(len(receipt["probes"]), 4)

    def test_probe_elapsed_seconds_are_recorded(self):
        self._run_clamp()
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertTrue(receipt["probes"])
        for entry in receipt["probes"]:
            self.assertIn("elapsed_seconds", entry)

    def test_admission_guard_runs_before_and_after_each_probe(self):
        checks = []

        def all_ok(params, candidate_path):
            checks.append("preflight")
            return {"status": "ok", "peak_vram_mb": 40000.0}

        with mock.patch.object(
            _common, "_configured_preflight_name", return_value="preflight_config"
        ), mock.patch.object(_common, "timed_preflight", side_effect=all_ok):
            clamp_search_space_to_preflight(
                SPACE,
                BASE,
                self.candidate_path,
                self.report_path,
                admission_check=lambda: checks.append("guard"),
            )

        self.assertEqual(checks, ["guard", "preflight", "guard"])


class WithinSpaceTest(unittest.TestCase):
    def test_bounds_and_categorical(self):
        space = {"x": ("int", 1, 5), "mode": ("categorical", ["a", "b"])}
        self.assertTrue(params_within_search_space({"x": 3, "mode": "a"}, space))
        self.assertFalse(params_within_search_space({"x": 6, "mode": "a"}, space))
        self.assertFalse(params_within_search_space({"x": 3, "mode": "c"}, space))
        self.assertFalse(params_within_search_space({"x": 3}, space))

    def test_split_configs_by_space(self):
        space = {"x": ("int", 1, 5)}
        inside, outside = split_configs_by_space(
            [{"x": 3}, {"x": 9}, {"x": 1}], space
        )
        self.assertEqual(inside, [{"x": 3}, {"x": 1}])
        self.assertEqual(outside, [{"x": 9}])


class FixedGridEntryTest(unittest.TestCase):
    def test_fixed_numeric_ranges_expand_once(self):
        self.assertEqual(expand_entry(("int", 4, 4), resolution=5), [4])
        self.assertEqual(expand_entry(("float", 1.25, 1.25), resolution=5), [1.25])
        self.assertEqual(
            expand_entry(("float", 0.01, 0.01, "log"), resolution=5),
            [0.01],
        )


if __name__ == "__main__":
    unittest.main()
