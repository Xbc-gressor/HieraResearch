from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "tuners"))

import _common  # noqa: E402


def _make_candidate(run_dir: Path, *, eval_body: str) -> Path:
    candidate = run_dir / "candidates" / "000" / "train.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text(
        'PARAM_SCHEMA = {"x": "int"}\n'
        'BASE_PARAMS = {"x": 1}\n'
        'SEARCH_SPACE = {"x": ("int", 1, 2)}\n\n'
        "def make_model(params):\n"
        "    return params['x']\n"
    )
    (candidate.parent / "prepare.py").write_text(eval_body)
    return candidate


class EvalTraceTests(unittest.TestCase):
    def _run_dir(self, tmp: str, cfg: dict) -> Path:
        run_dir = Path(tmp) / "runs" / "unit" / "tag"
        run_dir.mkdir(parents=True)
        (run_dir / "framework_cfg.json").write_text(json.dumps(cfg))
        return run_dir

    def test_success_writes_bounded_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp, {})
            candidate = _make_candidate(
                run_dir,
                eval_body=(
                    "def evaluate_config(make_model, params):\n"
                    "    print('trace-marker')\n"
                    "    print('x' * (100 * 1024))\n"
                    "    return float(make_model(params))\n"
                ),
            )
            score = _common.timed_eval(
                None,
                None,
                {"x": 1},
                candidate,
                phase="phase_a",
                method="warmstart",
                python_cmd=[sys.executable],
            )
            self.assertEqual(score, 1.0)
            traces = list((candidate.parent / "_traces").glob("*.log"))
            self.assertEqual(len(traces), 1)
            content = traces[0].read_text()
            self.assertIn("attempt_id: eval-000001", content)
            self.assertIn("phase: phase_a", content)
            self.assertIn("method: warmstart", content)
            self.assertIn("returncode: 0", content)
            self.assertIn("timed_out: false", content)
            self.assertIn("elapsed_seconds:", content)
            self.assertIn("max_rss_kb:", content)
            self.assertIn("trace-marker", content)
            self.assertIn("[trace truncated,", content)
            self.assertIn("[stderr]", content)

    def test_timeout_writes_partial_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._run_dir(tmp, {"per_runtime_limit": 1})
            candidate = _make_candidate(
                run_dir,
                eval_body=(
                    "import time\n"
                    "def evaluate_config(make_model, params):\n"
                    "    print('partial-marker', flush=True)\n"
                    "    time.sleep(30)\n"
                    "    return 0.0\n"
                ),
            )
            with self.assertRaises(TimeoutError):
                _common.timed_eval(
                    None,
                    None,
                    {"x": 1},
                    candidate,
                    phase="phase_a",
                    method="warmstart",
                    python_cmd=[sys.executable],
                )
            traces = list((candidate.parent / "_traces").glob("*.log"))
            self.assertEqual(len(traces), 1)
            content = traces[0].read_text()
            self.assertIn("timed_out: true", content)
            self.assertIn("returncode: None", content)
            self.assertIn("partial-marker", content)


if __name__ == "__main__":
    unittest.main()
