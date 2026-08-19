from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from driver.loops import baseline_tune


class BaselineTuneTest(unittest.TestCase):
    def test_phase_c_action_failure_uses_persisted_block_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "toy" / "baseline"
            candidate = run_dir / "candidates" / "000" / "train.py"
            report = candidate.parent / "tune_report.json"
            failed = subprocess.CalledProcessError(
                1,
                ["phase-c-action"],
                stderr="broken action",
            )

            with mock.patch.object(
                baseline_tune,
                "_or_block",
                side_effect=baseline_tune.RunBlocked("blocked"),
            ) as block:
                with self.assertRaises(baseline_tune.RunBlocked):
                    baseline_tune._phase_c_action(
                        run_dir,
                        candidate,
                        report,
                        root,
                        mock.Mock(side_effect=failed),
                        mock.sentinel.events,
                    )

            block.assert_called_once_with(
                run_dir,
                root,
                mock.ANY,
                mock.sentinel.events,
                "phase-c-action failed: broken action",
            )


if __name__ == "__main__":
    unittest.main()
