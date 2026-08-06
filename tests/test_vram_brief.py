from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import _common  # noqa: E402
from tune_tools import (  # noqa: E402
    SPACE_CLAMP_HEADROOM,
    VRAM_BRIEF_MAX_OBSERVATIONS,
    build_vram_brief,
)

TOTAL_VRAM = 81920.0


def _attempt(params: dict, peak: float, *, covers=True, source="warmstart") -> dict:
    return {
        "params": params,
        "source": source,
        "status": "ok",
        "result": {
            "status": "ok",
            "peak_vram_mb": peak,
            "envelope_covers_worst_case": covers,
        },
    }


class VramBriefTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_env(self, hook: dict | None = None):
        payload = {"hook_result": hook} if hook is not None else {}
        (self.run_dir / "environment_preflight.json").write_text(json.dumps(payload))

    def _write_report(self, run_id: str, attempts: list[dict]):
        cand = self.run_dir / "candidates" / run_id
        cand.mkdir(parents=True, exist_ok=True)
        (cand / "tune_report.json").write_text(
            json.dumps({"preflight": {"attempts": attempts}})
        )

    def test_brief_construction(self):
        self._write_env({"device": "NVIDIA A800-SXM4-80GB", "total_vram_mb": TOTAL_VRAM})
        self._write_report("000", [
            _attempt({"depth": 8, "device_batch_size": 128}, 47764.3),
            # Duplicate params collapse to one observation.
            _attempt({"depth": 8, "device_batch_size": 128}, 47764.3),
        ])
        self._write_report("003", [
            _attempt({"depth": 12, "device_batch_size": 128}, 61000.0),
            _attempt({"depth": 4, "device_batch_size": 64}, 21000.0),
        ])

        brief = build_vram_brief(self.run_dir)

        self.assertTrue(brief["available"])
        self.assertEqual(brief["device"], "NVIDIA A800-SXM4-80GB")
        self.assertEqual(brief["total_vram_mb"], TOTAL_VRAM)
        self.assertEqual(
            brief["feasible_ceiling_mb"], round(SPACE_CLAMP_HEADROOM * TOTAL_VRAM, 1)
        )
        # Four usable attempts found; one pair is a duplicate, so three distinct.
        self.assertEqual(brief["observation_count"], 4)
        self.assertEqual(brief["distinct_count"], 3)
        self.assertFalse(brief["truncated"])
        self.assertEqual(brief["skipped_reports"], 0)

        observed = [(o["run_id"], o["peak_vram_mb"]) for o in brief["observations"]]
        self.assertEqual(observed, [("000", 47764.3), ("003", 61000.0), ("003", 21000.0)])
        # Full params survive: the consumer decides which keys drive memory.
        self.assertEqual(
            brief["observations"][0]["params"], {"depth": 8, "device_batch_size": 128}
        )

    def test_cap_keeps_extremes_and_marks_truncation(self):
        self._write_env({"device": "gpu", "total_vram_mb": TOTAL_VRAM})
        peaks = [1000.0 * i for i in range(1, VRAM_BRIEF_MAX_OBSERVATIONS + 4)]
        self._write_report("000", [
            _attempt({"depth": i}, peak) for i, peak in enumerate(peaks)
        ])

        brief = build_vram_brief(self.run_dir)

        self.assertTrue(brief["truncated"])
        self.assertEqual(len(brief["observations"]), VRAM_BRIEF_MAX_OBSERVATIONS)
        kept = {o["peak_vram_mb"] for o in brief["observations"]}
        # The binding constraint and the low anchor must both survive the cap.
        self.assertIn(max(peaks), kept)
        self.assertIn(min(peaks), kept)

    def test_degrades_without_receipt(self):
        brief = build_vram_brief(self.run_dir)
        self.assertFalse(brief["available"])
        self.assertIn("total_vram_mb", brief["reason"])

        self._write_env({"device": "gpu"})  # receipt present, VRAM missing
        self.assertFalse(build_vram_brief(self.run_dir)["available"])

    def test_headroom_matches_clamp(self):
        # tune_tools re-declares the constant to stay stdlib-only (_common
        # imports numpy). They must not drift.
        self.assertEqual(SPACE_CLAMP_HEADROOM, _common.SPACE_CLAMP_HEADROOM)


if __name__ == "__main__":
    unittest.main()
