from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from init_run import initialize_run  # noqa: E402


class InitRunDimensionStrategyTests(unittest.TestCase):
    def _repo(self, root: Path) -> None:
        tasks = root / "tasks"
        tasks.mkdir()
        template = json.loads((ROOT / "tasks" / "framework_cfg.example.json").read_text())
        (tasks / "framework_cfg.example.json").write_text(json.dumps(template))

    def _strategy(self, run_dir: Path) -> str:
        config = json.loads((run_dir / "framework_cfg.json").read_text())
        return config["space_initialization"]["dimension_strategy"]

    def test_default_strategy_comes_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(repo_root, "toy", "default")

            self.assertEqual(self._strategy(run_dir), "catalog_subset")
            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["per_runtime_limit"], 60)

    def test_explicit_induced_strategy_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(
                repo_root,
                "toy",
                "induced",
                dimension_strategy="llm_induced",
            )

            self.assertEqual(self._strategy(run_dir), "llm_induced")

    def test_same_strategy_is_idempotent_after_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)
            run_dir = initialize_run(
                repo_root,
                "toy",
                "resume",
                dimension_strategy="llm_induced",
            )
            (run_dir / "dimension_catalog.json").write_text("{}")

            initialize_run(
                repo_root,
                "toy",
                "resume",
                dimension_strategy="llm_induced",
            )

            self.assertEqual(self._strategy(run_dir), "llm_induced")

    def test_strategy_can_change_before_semantic_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)
            run_dir = initialize_run(repo_root, "toy", "unfrozen")

            initialize_run(
                repo_root,
                "toy",
                "unfrozen",
                dimension_strategy="llm_induced",
            )

            self.assertEqual(self._strategy(run_dir), "llm_induced")

    def test_conflicting_strategy_is_rejected_after_artifacts_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)
            run_dir = initialize_run(
                repo_root,
                "toy",
                "frozen",
                dimension_strategy="llm_induced",
            )
            (run_dir / "background.md").write_text("# frozen")

            with self.assertRaisesRegex(ValueError, "cannot change dimension strategy"):
                initialize_run(
                    repo_root,
                    "toy",
                    "frozen",
                    dimension_strategy="catalog_subset",
                )

            self.assertEqual(self._strategy(run_dir), "llm_induced")

    def test_run_limits_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(
                repo_root,
                "toy",
                "bounded",
                max_evaluations=37,
                per_runtime_limit=12.5,
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["max_evaluations"], 37)
            self.assertEqual(config["per_runtime_limit"], 12.5)

    def test_run_limits_can_change_when_resuming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)
            run_dir = initialize_run(repo_root, "toy", "resume-limits")
            (run_dir / "background.md").write_text("# frozen")

            initialize_run(
                repo_root,
                "toy",
                "resume-limits",
                max_evaluations=400,
                per_runtime_limit=90,
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["max_evaluations"], 400)
            self.assertEqual(config["per_runtime_limit"], 90)

    def test_run_limits_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            with self.assertRaisesRegex(ValueError, "positive integer"):
                initialize_run(
                    repo_root,
                    "toy",
                    "bad-budget",
                    max_evaluations=0,
                )
            with self.assertRaisesRegex(ValueError, "positive number"):
                initialize_run(
                    repo_root,
                    "toy",
                    "bad-timeout",
                    per_runtime_limit=-1,
                )


if __name__ == "__main__":
    unittest.main()
