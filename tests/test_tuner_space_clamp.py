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


def fake_preflight(params, candidate_path, **kwargs):
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
        def all_ok(params, candidate_path, **kwargs):
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
        def tight(params, candidate_path, **kwargs):
            return {"status": "ok", "peak_vram_mb": params["device_batch_size"] * 360.0}

        clamped = self._run_clamp(preflight=tight)
        # 68000 MB headroom / 360 MB-per-unit -> largest feasible dbs is 188.
        self.assertLessEqual(clamped["device_batch_size"][2], 188)
        self.assertGreaterEqual(clamped["device_batch_size"][2], 128)

    @staticmethod
    def _bilinear_preflight(params, candidate_path, **kwargs):
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

        def all_ok(params, candidate_path, **kwargs):
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

    def test_sub_maximal_envelope_cannot_certify_feasibility(self):
        """A probe that admits it measured the wrong shape is not evidence.

        Run 0802-sonnet-ex125-1/007: a sequence-length curriculum's first step
        passed at 44.7 GB, the clamp declared the corner feasible, and five full
        evaluations then OOMed at 74.9 GB. A cheap peak plus
        `envelope_covers_worst_case: false` must not widen the space.
        """
        def half_context(params, candidate_path, **kwargs):
            # Comfortably inside the headroom, but only half the real shape.
            return {
                "status": "ok",
                "peak_vram_mb": 40000.0,
                "probe_seq_len": 1024,
                "max_seq_len": 2048,
                "envelope_covers_worst_case": False,
            }

        clamped = self._run_clamp(preflight=half_context)

        # Nothing may be certified on that receipt: the corner is not feasible
        # and no dim can be attributed, so the space collapses to base rather
        # than being trusted at its upper bound.
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertFalse(receipt["corner_feasible"])
        self.assertLessEqual(
            clamped["device_batch_size"][2], SPACE["device_batch_size"][2]
        )
        self.assertIs(receipt["probes"][0]["envelope_covers_worst_case"], False)

    def test_full_context_envelope_is_trusted(self):
        """The same peak with a worst-case receipt does certify the corner."""
        def full_context(params, candidate_path, **kwargs):
            return {
                "status": "ok",
                "peak_vram_mb": 40000.0,
                "probe_seq_len": 2048,
                "max_seq_len": 2048,
                "envelope_covers_worst_case": True,
            }

        clamped = self._run_clamp(preflight=full_context)
        self.assertEqual(clamped, SPACE)
        receipt = json.loads(self.report_path.read_text())["search_space_clamp"]
        self.assertTrue(receipt["corner_feasible"])
        self.assertEqual(receipt["outcome"], "already_feasible")

    def test_resource_probe_is_preferred_when_the_task_declares_one(self):
        """The clamp asks for the worst-case oracle, not the correctness one."""
        modes = []

        def record_mode(params, candidate_path, **kwargs):
            modes.append(kwargs.get("probe_mode"))
            return {
                "status": "ok",
                "peak_vram_mb": 40000.0,
                "envelope_covers_worst_case": True,
            }

        with mock.patch.object(
            _common, "_configured_preflight_name", return_value="preflight_config"
        ), mock.patch.object(
            _common,
            "_configured_resource_probe_name",
            return_value="resource_probe_config",
        ), mock.patch.object(_common, "timed_preflight", side_effect=record_mode):
            clamp_search_space_to_preflight(
                SPACE, BASE, self.candidate_path, self.report_path
            )

        self.assertEqual(modes, ["resource"])


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
