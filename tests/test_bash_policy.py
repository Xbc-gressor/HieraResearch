import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.events import EventsLog  # noqa: E402
from driver.roles import ROLES  # noqa: E402
from driver.session import SDKSessionRunner, adapter_command_verdict  # noqa: E402


class AdapterCommandVerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "runs" / "mle-x" / "tag"
        self.run_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def verdict(self, command: str) -> str | None:
        return adapter_command_verdict(command, self.run_dir)

    def test_allows_each_search_backends_subcommand(self) -> None:
        for subcommand in ("search", "status", "results", "visit", "read",
                           "validate"):
            command = (f"python tools/search_backends.py {subcommand} "
                       f"--manifest {self.run_dir}/background_retrieval.json")
            self.assertIsNone(self.verdict(command), subcommand)

    def test_allows_python3_and_background_contract_subcommands(self) -> None:
        for subcommand in ("catalog", "validate"):
            command = (f"python3 tools/background_contract.py {subcommand} "
                       f"--background {self.run_dir}/background.md")
            self.assertIsNone(self.verdict(command), subcommand)

    def test_json_metachars_inside_quotes_do_not_trip_operator_check(self) -> None:
        command = (
            "python tools/search_backends.py search "
            f"--manifest {self.run_dir}/background_retrieval.json "
            '--query-spec \'{"text": "a; b | c $ d \\"quoted\\""}\''
        )
        self.assertIsNone(self.verdict(command))

    def test_denies_non_adapter_commands(self) -> None:
        for command in (
            "curl https://kaggle.com/c/some-competition",
            "wget https://example.com/paper.pdf",
            "git clone https://github.com/team/solution",
            "gh repo view team/solution",
            "pip install some-package",
            'bash -c "echo hi"',
        ):
            self.assertIsNotNone(self.verdict(command), command)

    def test_denies_operators_on_an_allowlisted_prefix(self) -> None:
        base = ("python tools/search_backends.py status "
                f"--manifest {self.run_dir}/background_retrieval.json")
        for command in (
            f"{base} && curl https://kaggle.com",
            f"{base} ; curl https://kaggle.com",
            f"{base} | head",
            f"{base} > status.txt",
            f"{base} 2>/dev/null",
        ):
            reason = self.verdict(command)
            self.assertIsNotNone(reason, command)
            self.assertIn("operator", reason, command)

    def test_denies_command_substitution(self) -> None:
        for command in (
            "python tools/search_backends.py status --manifest $(pwd)/m.json",
            "python tools/search_backends.py status --manifest `pwd`/m.json",
        ):
            self.assertIsNotNone(self.verdict(command), command)

    def test_manifest_must_stay_inside_run_dir(self) -> None:
        sibling = self.run_dir.parent / "other-tag"
        reason = self.verdict(
            f"python tools/search_backends.py search --manifest "
            f"{sibling}/background_retrieval.json")
        self.assertIn("inside this run's directory", reason)
        self.assertIsNotNone(self.verdict(
            "python tools/search_backends.py status --manifest "
            "../other-tag/background_retrieval.json"))
        self.assertIsNone(self.verdict(
            "python tools/search_backends.py status "
            "--manifest background_retrieval.json"))

    def test_denies_unparseable_command(self) -> None:
        self.assertIsNotNone(self.verdict(
            'python tools/search_backends.py search --manifest "unterminated'))


class AdapterHookTests(unittest.TestCase):
    def test_background_role_deny_is_retryable_and_emits_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            hook = runner._capability_hook(ROLES["background-researcher"],
                                           run_dir=run_dir)
            verdict = asyncio.run(hook(
                {"tool_name": "Bash",
                 "tool_input": {"command": "curl https://kaggle.com/c/x"}},
                None, {}))
            decision = verdict["hookSpecificOutput"]
            self.assertEqual(decision["permissionDecision"], "deny")
            self.assertIn("search_backends.py",
                          decision["permissionDecisionReason"])
            rows = [json.loads(line)
                    for line in (run_dir / "driver_events.jsonl")
                    .read_text().splitlines()]
            denied = [r for r in rows if r.get("kind") == "guard_denied"
                      and r.get("guard") == "bash_adapter"]
            self.assertEqual(len(denied), 1)
            self.assertEqual(denied[0]["command_head"],
                             "curl https://kaggle.com/c/x")
            self.assertTrue(denied[0]["reason"])

    def test_background_role_allows_adapter_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            hook = runner._capability_hook(ROLES["background-researcher"],
                                           run_dir=run_dir)
            verdict = asyncio.run(hook(
                {"tool_name": "Bash",
                 "tool_input": {"command":
                                "python tools/search_backends.py validate "
                                f"--manifest {run_dir}/background_retrieval.json"}},
                None, {}))
            self.assertEqual(verdict, {})

    def test_benign_trailing_forms_are_rewritten_not_denied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            hook = runner._capability_hook(ROLES["background-researcher"],
                                           run_dir=run_dir)
            base = ("python tools/background_contract.py validate "
                    "--background background.md")
            output = asyncio.run(hook(
                {"tool_name": "Bash",
                 "tool_input": {"command": f"{base} 2>&1 | head -50"}},
                None, {}))["hookSpecificOutput"]
            self.assertEqual(output["permissionDecision"], "allow")
            self.assertEqual(output["updatedInput"]["command"], base)
            denied = asyncio.run(hook(
                {"tool_name": "Bash",
                 "tool_input": {"command": f"{base} > out.txt"}},
                None, {}))["hookSpecificOutput"]
            self.assertEqual(denied["permissionDecision"], "deny")
            self.assertIn("Edit", denied["permissionDecisionReason"])

    def test_other_roles_bash_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            hook = runner._capability_hook(ROLES["hillclimb-editor"],
                                           run_dir=run_dir)
            verdict = asyncio.run(hook(
                {"tool_name": "Bash",
                 "tool_input": {"command": "curl https://example.com"}},
                None, {}))
            self.assertEqual(verdict, {})
            self.assertFalse((run_dir / "driver_events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
