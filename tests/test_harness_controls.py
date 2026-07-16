from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import harness_guard  # noqa: E402
import harness_watch  # noqa: E402
import new_candidate  # noqa: E402


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

    def test_rejects_orchestration_action_as_idea_status(self) -> None:
        raw = """status: background_refresh_required
generation_run_ids: none
actions: none
risk_flags: legacy directions exhausted
"""
        result = harness_guard.compact_task_result("idea-generator", raw)
        self.assertIn("receipt_contract: invalid", result)
        self.assertIn("invalid_fields: status", result)


class UsageAndLifecycleTests(unittest.TestCase):
    def test_candidate_brief_contains_only_implementation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps({"records": [{
                "run_id": "007", "op": "crossover", "idea": "combined result",
                "change": "vs 003: keep core; vs 005: use features",
                "source_run_ids": ["003", "005"], "candidate_name": "combo",
                "final_best_score": 0.1, "tuning": {"large": "payload"},
            }]}))
            brief = new_candidate.candidate_brief(ledger_path, "007")
            self.assertEqual(brief["source_run_ids"], ["003", "005"])
            self.assertEqual(brief["idea"], "combined result")
            self.assertNotIn("final_best_score", brief)
            self.assertNotIn("tuning", brief)

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

    def test_brief_and_explicit_completion_are_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps({"records": [{
                "run_id": "001", "status": "keep", "op": "fresh",
                "final_best_score": 0.2, "trials_attempted": 2,
                "trials_completed": 2,
            }]}))
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
