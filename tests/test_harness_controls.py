from __future__ import annotations

import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import new_candidate  # noqa: E402
import got_select  # noqa: E402
import parse_result  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402


class LegacyResultParserTests(unittest.TestCase):
    def test_parser_calls_current_record_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "run.log"
            ledger_path = tmp_path / "ledger.json"
            log_path.write_text("val_bpb: 0.95\npeak_vram_mb: 1024\n")
            calls: list[dict] = []

            def strict_record_run(
                ledger_path_arg,
                task_name,
                run_id,
                *,
                final_best_score=None,
                status="auto",
                candidate_name=None,
                description=None,
            ):
                calls.append(
                    {
                        "ledger": ledger_path_arg,
                        "task": task_name,
                        "run_id": run_id,
                        "score": final_best_score,
                        "status": status,
                        "candidate_name": candidate_name,
                        "description": description,
                    }
                )
                return {
                    "run_id": run_id,
                    "status": "keep",
                    "final_best_score": final_best_score,
                }

            argv = [
                "parse_result.py",
                str(log_path),
                "--ledger",
                str(ledger_path),
                "--task",
                "autoresearch-baseline",
            ]
            with (
                mock.patch.object(
                    parse_result,
                    "load_task_config",
                    return_value={
                        "result": {
                            "metric": "val_bpb",
                            "required_patterns": [
                                "^val_bpb:",
                                "^peak_vram_mb:",
                            ],
                        }
                    },
                ),
                mock.patch.object(parse_result, "record_run", strict_record_run),
                mock.patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(parse_result.main(), 0)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["score"], 0.95)


class UsageAndLifecycleTests(unittest.TestCase):
    def test_got_select_reserves_k_eval_capacity_before_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "framework_cfg.json").write_text(
                json.dumps(
                    {
                        "max_evaluations": 2,
                        "tuner": {"K_eval": 3},
                    }
                )
            )
            ledger_path = run_dir / "ledger.json"
            output = io.StringIO()
            with redirect_stdout(output):
                got_select.cmd_decide(
                    SimpleNamespace(ledger=str(ledger_path), cfg=None)
                )
            payload = json.loads(output.getvalue())

            self.assertEqual(payload["actions"], [])
            self.assertEqual(payload["diag"]["objective_remaining"], 2)
            self.assertEqual(payload["diag"]["candidate_admission_cap"], 0)

    def test_candidate_brief_contains_only_implementation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            primary = Path(tmp) / "candidates" / "003" / "train.py"
            primary.parent.mkdir(parents=True)
            primary.write_text("PRIMARY = True\n")
            ledger_path.write_text(json.dumps({"records": [{
                "run_id": "007", "op": "crossover", "idea": "combined result",
                "change": "vs 003: keep core; vs 005: use features",
                "source_run_ids": ["003", "005"], "candidate_name": "combo",
                "semantic_point": {"point_id": "point-abc"},
                "policy_receipt": {"policy": {"name": "coverage"}},
                "final_best_score": 0.1, "tuning": {"large": "payload"},
            }]}))
            brief = new_candidate.candidate_brief(ledger_path, "007")
            self.assertEqual(brief["source_run_ids"], ["003", "005"])
            self.assertEqual(brief["idea"], "combined result")
            self.assertEqual(brief["semantic_point"]["point_id"], "point-abc")
            self.assertEqual(brief["policy_receipt"]["policy"]["name"], "coverage")
            self.assertEqual(brief["schema_version"], 4)
            self.assertEqual(brief["primary_parent"]["run_id"], "003")
            self.assertRegex(
                brief["primary_parent"]["sha256"],
                r"^sha256:[0-9a-f]{64}$",
            )
            self.assertEqual(
                brief["implementation_source"]["kind"],
                "primary_parent_snapshot",
            )
            self.assertNotIn("final_best_score", brief)
            self.assertNotIn("tuning", brief)

            ledger_path.write_text(json.dumps({"records": [{
                "run_id": "008", "op": "fresh", "idea": "legacy idea",
                "change": "from scratch", "source_run_ids": [],
            }]}))
            self.assertIsNone(new_candidate.candidate_brief(ledger_path, "008"))

    def test_provided_baseline_copies_declared_entrypoint_with_source_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            task_dir = repo_root / "tasks" / "unit"
            task_dir.mkdir(parents=True)
            source = task_dir / "train.py"
            source.write_text("CANDIDATE_NAME = 'provided'\n")
            (task_dir / "prepare.py").write_text("VALUE = 1\n")
            (task_dir / "task.toml").write_text(
                """
[seed]
provided = ["train.py"]
entrypoint = "train.py"

[candidate]
copy_files = ["prepare.py", "train.py"]
entrypoint = "train.py"
""".strip()
                + "\n"
            )
            run_dir = repo_root / "runs" / "unit" / "tag"
            run_dir.mkdir(parents=True)
            (run_dir / "ledger.json").write_text(json.dumps({"records": [{
                "run_id": "000",
                "op": "fresh",
                "idea": "Use the task-provided baseline.",
                "change": "provided baseline at point-base",
                "source_run_ids": [],
                "candidate_name": "provided_baseline",
                "semantic_point": {"point_id": "point-base"},
                "policy_receipt": {"policy": {"name": "coverage"}},
            }]}))

            argv = [
                "new_candidate.py",
                "unit",
                "tag",
                "000",
                "--provided-baseline",
            ]
            with (
                mock.patch.object(new_candidate, "ROOT", repo_root),
                mock.patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(new_candidate.main(), 0)

            candidate_dir = run_dir / "candidates" / "000"
            self.assertEqual((candidate_dir / "train.py").read_text(), source.read_text())
            brief = json.loads((candidate_dir / "_candidate_brief.json").read_text())
            implementation = brief["implementation_source"]
            self.assertEqual(brief["schema_version"], 4)
            self.assertEqual(implementation["kind"], "provided_entrypoint")
            self.assertEqual(implementation["path"], "tasks/unit/train.py")
            self.assertRegex(implementation["sha256"], r"^sha256:[0-9a-f]{64}$")

    def test_nonfresh_candidate_starts_as_exact_primary_parent_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            task_dir = repo_root / "tasks" / "unit"
            task_dir.mkdir(parents=True)
            (task_dir / "prepare.py").write_text("VALUE = 1\n")
            (task_dir / "train.py").write_text("TASK_TEMPLATE = True\n")
            (task_dir / "task.toml").write_text(
                """
[candidate]
copy_files = ["prepare.py", "train.py"]
entrypoint = "train.py"
""".strip()
                + "\n"
            )
            run_dir = repo_root / "runs" / "unit" / "tag"
            parent = run_dir / "candidates" / "001" / "train.py"
            parent.parent.mkdir(parents=True)
            parent.write_text("PARENT_STRATEGY = {'depth': 9}\n")
            (run_dir / "ledger.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "run_id": "002",
                                "op": "improve",
                                "idea": "Preserve and extend the parent.",
                                "change": "Add one isolated mechanism.",
                                "source_run_ids": ["001"],
                                "candidate_name": "child",
                                "semantic_point": {"point_id": "point-child"},
                                "policy_receipt": {
                                    "policy": {"name": "gain_uncertainty_nocost"}
                                },
                            }
                        ]
                    }
                )
            )

            argv = [
                "new_candidate.py",
                "unit",
                "tag",
                "002",
                "--skip-entrypoint",
            ]
            with (
                mock.patch.object(new_candidate, "ROOT", repo_root),
                mock.patch.object(sys, "argv", argv),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(new_candidate.main(), 0)

            child_dir = run_dir / "candidates" / "002"
            self.assertEqual(
                (child_dir / "train.py").read_bytes(),
                parent.read_bytes(),
            )
            brief = json.loads((child_dir / "_candidate_brief.json").read_text())
            self.assertEqual(brief["primary_parent"]["run_id"], "001")
            self.assertEqual(
                brief["implementation_source"]["sha256"],
                brief["primary_parent"]["sha256"],
            )

    def test_ledger_budget_prefers_attempts_and_reads_legacy_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps({
                "records": [
                    {
                        "run_id": "001",
                        "status": "discard",
                        "trials_completed": 2,
                        "trials_attempted": 5,
                    },
                    {
                        "run_id": "002",
                        "status": "keep",
                        "trials_completed": 3,
                    },
                ],
                "search_space_state": empty_search_space_state(),
            }))
            result = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "ledger.py"), "evaluations",
                 "--ledger", str(ledger_path)],
                check=True, capture_output=True, text=True,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["evaluations_done"], 8)
            self.assertEqual([row["evals"] for row in payload["per_candidate"]], [5, 3])

    def test_autoresearch_batch_schema_uses_independent_coordinates(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "tuners" / "tune_tools.py"),
             "lint-schema",
             "--candidate-path", str(ROOT / "tasks" / "autoresearch-baseline" / "train.py")],
            check=True, capture_output=True, text=True,
        )

        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertIn("grad_accum_steps", payload["keys"])
        self.assertNotIn("total_batch_size", payload["keys"])

    def test_brief_and_explicit_completion_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps({
                "records": [{
                    "run_id": "001", "status": "keep", "op": "fresh",
                    "final_best_score": 0.2, "trials_attempted": 2,
                    "trials_completed": 2,
                }],
                "search_space_state": empty_search_space_state(),
            }))
            brief = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "ledger.py"), "brief",
                 "--ledger", str(ledger_path), "--budget", "2"],
                check=True, capture_output=True, text=True,
            )
            self.assertEqual(json.loads(brief.stdout)["phase"], "completed")

            # Any positive remainder still refuses completion: got_select's
            # admission cap bounds only NEW candidates, while an admitted
            # candidate's Phase C reserves per trial and can spend the tail.
            refused = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "ledger.py"), "set-phase",
                 "--ledger", str(ledger_path), "--phase", "completed", "--budget", "4"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(refused.returncode, 0)

            remainder = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "ledger.py"), "set-phase",
                 "--ledger", str(ledger_path), "--phase", "completed", "--budget", "3"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(remainder.returncode, 0)

            subprocess.run(
                [sys.executable, str(ROOT / "tools" / "ledger.py"), "set-phase",
                 "--ledger", str(ledger_path), "--phase", "completed", "--budget", "2"],
                check=True, capture_output=True, text=True,
            )
            stored = json.loads(ledger_path.read_text())
            self.assertEqual(stored["run_state"]["phase"], "completed")
            self.assertEqual(stored["run_state"]["evaluation_budget"], 2)
            self.assertIn("phase: completed", (ledger_path.parent / "loop_state.md").read_text())


if __name__ == "__main__":
    unittest.main()
