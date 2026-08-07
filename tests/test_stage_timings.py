from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import stage_timings  # noqa: E402


class StageTimingsTest(unittest.TestCase):
    def test_records_and_derives_spans_from_a_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "task" / "tag"
            run_dir.mkdir(parents=True)
            (run_dir / "framework_cfg.json").write_text("{}")
            manifest = run_dir / "background_retrieval.json"
            manifest.write_text("{}")

            stage_timings.record(manifest, stage="retrieve", action="search", ok=True)
            stage_timings.record(manifest, stage="retrieve", action="visit", ok=True)
            payload = json.loads((run_dir / stage_timings.TIMINGS_FILENAME).read_text())
            self.assertEqual(len(payload["events"]), 2)

            derived = stage_timings.derive(run_dir)
            self.assertEqual(derived["retrieval_calls"], 2)
            self.assertIsNotNone(derived["stages"]["plan"]["seconds"])
            # background.md was never written, so distill has no end boundary
            # and reports no duration rather than an imputed one.
            self.assertIsNone(derived["stages"]["distill"]["end"])
            self.assertIsNone(derived["stages"]["distill"]["seconds"])

    def test_recording_outside_a_run_directory_is_a_silent_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            orphan = Path(tmp) / "background_retrieval.json"
            orphan.write_text("{}")
            stage_timings.record(orphan, stage="retrieve", action="search", ok=True)
            self.assertEqual(list(Path(tmp).iterdir()), [orphan])


if __name__ == "__main__":
    unittest.main()
