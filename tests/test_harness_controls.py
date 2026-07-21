from __future__ import annotations

import json
from pathlib import Path
import re
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
    def test_runtime_guidance_does_not_police_roadmap_phases(self) -> None:
        runtime_paths = [
            *sorted((ROOT / ".claude" / "agents").glob("*.md")),
            *sorted((ROOT / ".opencode" / "agents").glob("*.md")),
            ROOT / ".claude" / "rules" / "ledger.md",
            ROOT / ".opencode" / "rules" / "ledger.md",
        ]
        forbidden = (
            re.compile(r"\bP[1-4]\b"),
            re.compile(r"\bprun(?:e|ed|ing)\b", re.IGNORECASE),
            re.compile(r"convergence/regret", re.IGNORECASE),
            re.compile(r"bottleneck retrieval", re.IGNORECASE),
            re.compile(r"dynamic dimensions", re.IGNORECASE),
        )
        for path in runtime_paths:
            text = path.read_text()
            for pattern in forbidden:
                self.assertIsNone(pattern.search(text), (path.relative_to(ROOT), pattern.pattern))

    def test_autoresearch_prompts_use_progressive_disclosure(self) -> None:
        prompt_specs = (
            (".claude/agents/autoresearch-experiment.md", ".claude/rules/ledger.md"),
            (".opencode/agents/autoresearch-experiment.md", ".opencode/rules/ledger.md"),
        )
        for relative_path, ledger_rule in prompt_specs:
            prompt = (ROOT / relative_path).read_text()
            normalized_prompt = " ".join(prompt.split())
            self.assertLess(len(prompt.split()), 2600, relative_path)
            self.assertIn(
                "Continue rounds until the evaluation budget is exhausted or a hard stop occurs.",
                normalized_prompt,
            )
            self.assertIn(
                "Recheck the budget before deep tuning; if it is exhausted, return to step 0 without spawning the tuner.",
                normalized_prompt,
            )
            self.assertIn(f"Read `{ledger_rule}` only when", prompt)
            for required in (
                "source_run_ids",
                "semantic_point",
                "policy_receipt",
                "background_contract.py preflight",
                "new_candidate.py",
                "ledger.py brief",
                "ledger.py set-phase",
            ):
                self.assertIn(required, prompt, (relative_path, required))
            for obsolete in (
                "caller explicitly requested only one candidate",
                "run-000.log",
                "parsed run",
                "result.parser",
                "## Run Directory Layout",
                "## Ledger Schema",
                "## Delegation Rules",
                "### Scoring is recorded",
            ):
                self.assertNotIn(obsolete, prompt, (relative_path, obsolete))

    def test_experience_extractors_use_supported_graph_render_flags(self) -> None:
        for relative_path in (
            ".claude/agents/experience-extractor.md",
            ".opencode/agents/experience-extractor.md",
        ):
            prompt = (ROOT / relative_path).read_text()
            self.assertIn("--incremental --top 3 --bottom 3 --format json", prompt)
            self.assertNotIn("--top-k", prompt)
            self.assertNotIn("--bottom-k", prompt)

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
ledger: /tmp/run/ledger.json
"""
        result = harness_guard.compact_task_result("experience-extractor", experience)
        self.assertIn("receipt_contract: ok", result)


class UsageAndLifecycleTests(unittest.TestCase):
    def test_candidate_brief_contains_only_implementation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
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
            self.assertNotIn("final_best_score", brief)
            self.assertNotIn("tuning", brief)

            ledger_path.write_text(json.dumps({"records": [{
                "run_id": "008", "op": "fresh", "idea": "legacy idea",
                "change": "from scratch", "source_run_ids": [],
            }]}))
            self.assertIsNone(new_candidate.candidate_brief(ledger_path, "008"))

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
