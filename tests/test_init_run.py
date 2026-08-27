from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from init_run import _task_runtime_limit, initialize_run  # noqa: E402


class InitRunDimensionStrategyTests(unittest.TestCase):
    def _repo(self, root: Path, *, task_timeout: int | None = None) -> None:
        tasks = root / "tasks"
        tasks.mkdir()
        template = json.loads((ROOT / "tasks" / "framework_cfg.example.json").read_text())
        (tasks / "framework_cfg.example.json").write_text(json.dumps(template))
        if task_timeout is not None:
            task = tasks / "toy"
            task.mkdir()
            (task / "task.toml").write_text(
                f"[run]\ntimeout_seconds = {task_timeout}\n"
            )

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
            self.assertEqual(
                config["semantic_search"]["policy"],
                "coverage_attempt",
            )
            self.assertEqual(
                config["tuner"]["scheduler_policy"],
                "anchor_challenger_v1",
            )
            self.assertEqual(
                config["tuner"]["inner_policy"],
                "hebo24-hebo20",
            )
            self.assertEqual(config["tuner"]["K_eval"], 3)
            self.assertEqual(config["tuner"]["deep_tune_per_candidate_cap"], 44)

    def test_policy_comparison_arms_are_persisted_from_explicit_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(
                repo_root,
                "toy",
                "comparison",
                semantic_policy="coverage",
                scheduler_policy="legacy",
                inner_policy="localtr8-hebo10-spsa10-v1",
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["semantic_search"]["policy"], "coverage")
            self.assertEqual(config["tuner"]["scheduler_policy"], "legacy")
            self.assertEqual(
                config["tuner"]["inner_policy"],
                "localtr8-hebo10-spsa10-v1",
            )

            self.assertEqual(
                initialize_run(
                    repo_root,
                    "toy",
                    "selfrank",
                    inner_policy="selfrank8-hebo10-hebo10",
                ).name,
                "selfrank",
            )
            self.assertEqual(
                json.loads(
                    (repo_root / "runs/toy/selfrank/framework_cfg.json").read_text()
                )["tuner"]["inner_policy"],
                "selfrank8-hebo10-hebo10",
            )

    def test_24_plus_20_policies_set_scheduler_and_lifetime_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(
                repo_root,
                "toy",
                "mixup-turbo",
                scheduler_policy="anchor_challenger_v1",
                inner_policy="mixup24-turbo20-v1",
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(
                config["tuner"]["scheduler_policy"],
                "anchor_challenger_v1",
            )
            self.assertEqual(
                config["tuner"]["inner_policy"],
                "mixup24-turbo20-v1",
            )
            self.assertEqual(
                config["tuner"]["deep_tune_per_candidate_cap"],
                44,
            )

            hebo_run_dir = initialize_run(
                repo_root,
                "toy",
                "hebo-turbo",
                scheduler_policy="anchor_challenger_v1",
                inner_policy="hebo24-turbo20-v1",
            )
            hebo_config = json.loads(
                (hebo_run_dir / "framework_cfg.json").read_text()
            )
            self.assertEqual(
                hebo_config["tuner"]["inner_policy"],
                "hebo24-turbo20-v1",
            )
            self.assertEqual(
                hebo_config["tuner"]["deep_tune_per_candidate_cap"],
                44,
            )

            hebo_only_run_dir = initialize_run(
                repo_root,
                "toy",
                "hebo-only",
                scheduler_policy="anchor_challenger_v1",
                inner_policy="hebo24-hebo20",
            )
            hebo_only_config = json.loads(
                (hebo_only_run_dir / "framework_cfg.json").read_text()
            )
            self.assertEqual(
                hebo_only_config["tuner"]["inner_policy"],
                "hebo24-hebo20",
            )
            self.assertEqual(
                hebo_only_config["tuner"]["deep_tune_per_candidate_cap"],
                44,
            )

            with self.assertRaisesRegex(
                ValueError, "requires scheduler_policy anchor_challenger_v1"
            ):
                initialize_run(
                    repo_root,
                    "toy",
                    "mixup-wrong-scheduler",
                    scheduler_policy="v3_2",
                    inner_policy="mixup24-turbo20-v1",
                )

    def test_existing_run_is_not_rewritten_to_new_policy_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)
            run_dir = repo_root / "runs" / "toy" / "historical"
            run_dir.mkdir(parents=True)
            original = {"max_evaluations": 17}
            config_path = run_dir / "framework_cfg.json"
            config_path.write_text(json.dumps(original))

            initialize_run(repo_root, "toy", "historical")

            self.assertEqual(json.loads(config_path.read_text()), original)

    def test_new_production_defaults_have_budget_without_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            (repo_root / "tasks").mkdir()

            run_dir = initialize_run(repo_root, "toy", "no-template")

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["max_evaluations"], 200)
            self.assertEqual(
                config["tuner"]["scheduler_policy"],
                "anchor_challenger_v1",
            )
            self.assertEqual(
                config["tuner"]["inner_policy"],
                "hebo24-hebo20",
            )
            self.assertEqual(config["tuner"]["deep_tune_per_candidate_cap"], 44)

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

    def test_new_run_inherits_task_runtime_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root, task_timeout=900)

            run_dir = initialize_run(repo_root, "toy", "task-timeout")

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["per_runtime_limit"], 900)

    def test_corrupt_task_toml_does_not_drop_the_default_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root, task_timeout=900)
            task_toml = repo_root / "tasks" / "toy" / "task.toml"
            task_toml.write_text("[run]\nbroken line\n")

            with self.assertRaisesRegex(ValueError, "expected key = value"):
                _task_runtime_limit(repo_root, "toy")

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

    def test_judged_slate_policy_is_registered_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(
                repo_root, "toy", "judged", semantic_policy="judged_slate"
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["semantic_search"]["policy"], "judged_slate")
            self.assertEqual(config["judged_slate"]["pool_size"], 6)

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

    def test_llm_intelligence_score_is_persisted_without_clobbering_section(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(
                repo_root,
                "toy",
                "intelligence",
                llm_intelligence_score=72.5,
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            semantic_search = config["semantic_search"]
            self.assertEqual(semantic_search["llm_intelligence_score"], 72.5)
            self.assertEqual(
                semantic_search["policy"],
                "coverage_attempt",
            )
            self.assertEqual(semantic_search["uncertainty_weight"], 0.5)

    def test_llm_intelligence_score_can_change_before_semantic_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)
            run_dir = initialize_run(
                repo_root,
                "toy",
                "intelligence-unfrozen",
                llm_intelligence_score=25,
            )

            initialize_run(
                repo_root,
                "toy",
                "intelligence-unfrozen",
                llm_intelligence_score=80,
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(
                config["semantic_search"]["llm_intelligence_score"],
                80,
            )

    def test_llm_intelligence_score_is_frozen_with_semantic_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)
            run_dir = initialize_run(
                repo_root,
                "toy",
                "intelligence-frozen",
                llm_intelligence_score=60,
            )
            (run_dir / "ledger.json").write_text("{}")

            initialize_run(
                repo_root,
                "toy",
                "intelligence-frozen",
                llm_intelligence_score=60,
            )
            with self.assertRaisesRegex(
                ValueError,
                "cannot change llm intelligence score",
            ):
                initialize_run(
                    repo_root,
                    "toy",
                    "intelligence-frozen",
                    llm_intelligence_score=61,
                )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(
                config["semantic_search"]["llm_intelligence_score"],
                60,
            )

    def test_llm_intelligence_score_must_be_finite_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            for index, value in enumerate((-0.1, 100.1, float("nan"), float("inf"))):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"finite number in \[0, 100\]",
                    ):
                        initialize_run(
                            repo_root,
                            "toy",
                            f"bad-intelligence-{index}",
                            llm_intelligence_score=value,
                        )

            low = initialize_run(
                repo_root,
                "toy",
                "intelligence-low",
                llm_intelligence_score=0,
            )
            high = initialize_run(
                repo_root,
                "toy",
                "intelligence-high",
                llm_intelligence_score=100,
            )
            self.assertEqual(
                json.loads((low / "framework_cfg.json").read_text())[
                    "semantic_search"
                ]["llm_intelligence_score"],
                0,
            )
            self.assertEqual(
                json.loads((high / "framework_cfg.json").read_text())[
                    "semantic_search"
                ]["llm_intelligence_score"],
                100,
            )

    def test_warm_config_count_is_persisted_and_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(repo_root, "toy", "k-warm", k_warm=8)

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["tuner"]["K"], 8)

            (run_dir / "ledger.json").write_text("{}")
            # Idempotent re-application stays legal; a change does not.
            initialize_run(repo_root, "toy", "k-warm", k_warm=8)
            with self.assertRaisesRegex(ValueError, "cannot change K after"):
                initialize_run(repo_root, "toy", "k-warm", k_warm=5)

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["tuner"]["K"], 8)

    def test_warm_config_count_must_leave_a_row_beyond_the_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            for index, value in enumerate((1, 0, -3)):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(
                        ValueError, "k_warm must be an integer of at least 2"
                    ):
                        initialize_run(
                            repo_root, "toy", f"bad-k-warm-{index}", k_warm=value
                        )

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


class InitRunTransferPolicyTests(unittest.TestCase):
    """The transfer policy pair freezes explicitly and stays off by default."""

    def _repo(self, root: Path, *, template_edit=None) -> None:
        tasks = root / "tasks"
        tasks.mkdir()
        template = json.loads((ROOT / "tasks" / "framework_cfg.example.json").read_text())
        if template_edit is not None:
            template_edit(template)
        (tasks / "framework_cfg.example.json").write_text(json.dumps(template))

    def test_transfer_pair_freezes_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(
                repo_root,
                "toy",
                "transfer",
                scheduler_policy="anchor_transfer_challenger_v1",
                inner_policy="hebo24-transfer10-hebo10",
            )

            config_path = run_dir / "framework_cfg.json"
            config = json.loads(config_path.read_text())
            self.assertEqual(
                config["tuner"]["scheduler_policy"],
                "anchor_transfer_challenger_v1",
            )
            self.assertEqual(
                config["tuner"]["inner_policy"],
                "hebo24-transfer10-hebo10",
            )
            self.assertEqual(config["tuner"]["deep_tune_per_candidate_cap"], 44)
            self.assertEqual(config["max_evaluations"], 200)
            # The frozen pair passes the run_cfg validation layer.
            from run_cfg import read_framework_cfg

            read_framework_cfg(config_path)

    def test_transfer_inner_raises_a_low_lifetime_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            # A legacy template keeps the copied config valid while its cap
            # (40) sits below the transfer pair's three-bout lifetime (44).
            self._repo(
                repo_root,
                template_edit=lambda template: template["tuner"].update(
                    scheduler_policy="legacy",
                    inner_policy="legacy",
                    deep_tune_per_candidate_cap=40,
                ),
            )

            run_dir = initialize_run(
                repo_root,
                "toy",
                "transfer-cap",
                scheduler_policy="anchor_transfer_challenger_v1",
                inner_policy="hebo24-transfer10-hebo10",
            )

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(config["tuner"]["deep_tune_per_candidate_cap"], 44)

    def test_transfer_policies_reject_mispairing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            # The template defaults anchor_challenger_v1 x hebo24-hebo20, so
            # passing only one side of the pair mismatches the other.
            with self.assertRaisesRegex(
                ValueError, "requires scheduler_policy anchor_transfer_challenger_v1"
            ):
                initialize_run(
                    repo_root,
                    "toy",
                    "inner-only",
                    inner_policy="hebo24-transfer10-hebo10",
                )
            with self.assertRaisesRegex(
                ValueError, "requires inner tuner policy hebo24-transfer10-hebo10"
            ):
                initialize_run(
                    repo_root,
                    "toy",
                    "scheduler-only",
                    scheduler_policy="anchor_transfer_challenger_v1",
                )
            with self.assertRaisesRegex(
                ValueError, "requires scheduler_policy anchor_transfer_challenger_v1"
            ):
                initialize_run(
                    repo_root,
                    "toy",
                    "wrong-scheduler",
                    scheduler_policy="anchor_challenger_v1",
                    inner_policy="hebo24-transfer10-hebo10",
                )
            with self.assertRaisesRegex(
                ValueError, "requires inner tuner policy hebo24-transfer10-hebo10"
            ):
                initialize_run(
                    repo_root,
                    "toy",
                    "wrong-inner",
                    scheduler_policy="anchor_transfer_challenger_v1",
                    inner_policy="hebo24-hebo20",
                )

    def test_default_pair_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._repo(repo_root)

            run_dir = initialize_run(repo_root, "toy", "default")

            config = json.loads((run_dir / "framework_cfg.json").read_text())
            self.assertEqual(
                config["tuner"]["scheduler_policy"], "anchor_challenger_v1"
            )
            self.assertEqual(config["tuner"]["inner_policy"], "hebo24-hebo20")


if __name__ == "__main__":
    unittest.main()
