import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def run_cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "driver", *args],
            cwd=ROOT, capture_output=True, text=True,
        )

    def test_requires_task_and_tag(self) -> None:
        result = self.run_cli("run", "only-task")
        self.assertNotEqual(result.returncode, 0)

    def test_rejects_unknown_loop(self) -> None:
        result = self.run_cli("run", "t", "tag", "--loop", "nope")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid choice", result.stderr)

    def test_model_required_for_new_run(self) -> None:
        result = self.run_cli("run", "t", "tag", "--loop", "hillclimb")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--model", result.stderr)


if __name__ == "__main__":
    unittest.main()
