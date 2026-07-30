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

import harness_guard  # noqa: E402
import harness_watch  # noqa: E402
import new_candidate  # noqa: E402
import got_select  # noqa: E402
import parse_result  # noqa: E402
from search_space_state import empty_search_space_state  # noqa: E402


class DelegationGuardTests(unittest.TestCase):
    def test_blocks_observed_role_collapse(self) -> None:
        prompt = "After writing train.py, also perform step 0+1 since budget is tight."
        self.assertIsNotNone(harness_guard.delegation_violation("candidate-writer", prompt))

    def test_allows_narrow_writer_assignment(self) -> None:
        prompt = "Write train.py only. Do not perform step 0+1 or warm-start evaluation."
        self.assertIsNone(harness_guard.delegation_violation("candidate-writer", prompt))

    def test_compacts_valid_writer_receipt(self) -> None:
        raw = """<task id=\"ses_test\" state=\"completed\"><task_result>
status: written
candidate_path: /tmp/run/candidates/001/train.py
candidate_name: compact_tree
wrote: true
risk_flags: none
confidence: high
diff: this must never survive
</task_result></task>"""
        result = harness_guard.compact_task_result("candidate-writer", raw)
        self.assertIn("receipt_contract: ok", result)
        self.assertIn("child_session_id: ses_test", result)
        self.assertNotIn("this must never survive", result)

    def test_flags_writer_evaluation_overreach(self) -> None:
        raw = """status: discard
candidate_path: /tmp/run/candidates/001/train.py
wrote: true
best_warm: 0.2
trials_completed: 3
"""
        result = harness_guard.compact_task_result("candidate-writer", raw)
        self.assertIn("receipt_contract: invalid", result)
        self.assertIn("scope_violation:", result)

    def test_rejects_non_receipt_idea_output(self) -> None:
        raw = """status: orchestration_only
generation_run_ids: none
actions: none
risk_flags: invalid coordinator control output
"""
        result = harness_guard.compact_task_result("idea-generator", raw)
        self.assertIn("receipt_contract: invalid", result)
        self.assertIn("missing_fields:", result)

    def test_accepts_budget_admission_no_action_receipt(self) -> None:
        raw = """generation_run_ids: none
selection_reason: objective_budget_admission_cap
ledger: /tmp/run/ledger.json
"""
        result = harness_guard.compact_task_result("idea-generator", raw)
        self.assertIn("receipt_contract: ok", result)
        self.assertIn("generation_run_ids: none", result)

    def test_compacts_colon_delimited_semantic_receipts(self) -> None:
        raw = """run_id: 004
op: improve
parents: 001
point_id: point-first
policy: coverage
candidate: first
ledger: /tmp/run/ledger.json

run_id: 005
op: improve
parents: 001,003
point_id: point-second
policy: gain_uncertainty
candidate: second
ledger: /tmp/run/ledger.json
"""
        result = harness_guard.compact_task_result("idea-generator", raw)
        self.assertIn("receipt_contract: ok", result)
        self.assertIn("run_id: 004; 005", result)
        self.assertIn("op: improve; improve", result)
        self.assertIn("point_id: point-first; point-second", result)

        experience = """updated_at_run: 005
generation: 2
evidence_runs: 5
search_space_state_revision: 3
decision_ids: sdec-000003
ledger: /tmp/run/ledger.json
"""
        result = harness_guard.compact_task_result("experience-extractor", experience)
        self.assertIn("receipt_contract: ok", result)

    def test_compacts_experience_state_decision_receipt(self) -> None:
        decided = """updated_at_run: 007
generation: 2
evidence_runs: 5
search_space_state_revision: 3
decision_ids: sdec-000003
ledger: /tmp/run/ledger.json
"""
        result = harness_guard.compact_task_result("experience-extractor", decided)
        self.assertIn("receipt_contract: ok", result)
        self.assertIn("search_space_state_revision: 3", result)
        self.assertIn("decision_ids: sdec-000003", result)

        noop = """updated_at_run: 007
generation: 2
evidence_runs: 5
search_space_state_revision: 3
decision_ids: none
ledger: /tmp/run/ledger.json
"""
        result = harness_guard.compact_task_result("experience-extractor", noop)
        self.assertIn("receipt_contract: ok", result)
        self.assertIn("decision_ids: none", result)


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

    def test_claude_usage_deduplicates_stream_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transcript = Path(tmp) / "session.jsonl"
            rows = [
                {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 10}}},
                {"type": "assistant", "message": {"id": "m1", "usage": {"input_tokens": 12}}},
                {"type": "assistant", "message": {"id": "m2", "usage": {
                    "input_tokens": 5, "output_tokens": 3,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 7,
                }}},
            ]
            transcript.write_text("\n".join(json.dumps(row) for row in rows))
            usage, messages = harness_watch._claude_usage(transcript)
            self.assertEqual(messages, 2)
            self.assertEqual(usage.input, 17)
            self.assertEqual(usage.output, 3)
            self.assertEqual(usage.cache_read, 20)
            self.assertEqual(usage.cache_write, 7)

    def test_snapshot_derives_completion_from_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "framework_cfg.json").write_text(json.dumps({"max_evaluations": 3}))
            ledger = {"task": "unit", "tag": "test", "records": [
                {"run_id": "001", "status": "keep", "trials_attempted": 2},
            ]}
            (run_dir / "ledger.json").write_text(json.dumps(ledger))
            self.assertEqual(harness_watch._run_snapshot(run_dir)["phase"], "running")
            ledger["records"][0]["trials_attempted"] = 3
            (run_dir / "ledger.json").write_text(json.dumps(ledger))
            snapshot = harness_watch._run_snapshot(run_dir)
            self.assertEqual(snapshot["phase"], "completed")
            self.assertEqual(snapshot["remaining"], 0)
            ledger["run_state"] = {"phase": "blocked", "active_stop_condition": "contract mismatch"}
            (run_dir / "ledger.json").write_text(json.dumps(ledger))
            self.assertEqual(harness_watch._run_snapshot(run_dir)["phase"], "blocked")

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

            refused = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "ledger.py"), "set-phase",
                 "--ledger", str(ledger_path), "--phase", "completed", "--budget", "3"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(refused.returncode, 0)

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
