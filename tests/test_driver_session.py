import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.events import EventsLog  # noqa: E402
from driver.receipts import ReceiptStore  # noqa: E402
from driver.roles import ROLES, InvocationContext, RoleDefinition  # noqa: E402
from driver.session import (  # noqa: E402
    FakeSessionRunner,
    InvocationFailed,
    SDKSessionRunner,
)


def make_ctx(run_dir: Path, **kw) -> InvocationContext:
    return InvocationContext(task="t", tag="tag", run_dir=run_dir,
                             invocation_id=1, **kw)


SIMPLE_ROLE = RoleDefinition(
    name="hillclimb-editor",
    prompt_file="hillclimb-editor.md",
    tools=("Read", "Write"),
    disallowed=("Agent", "Task", "Skill"),
    receipt_schema={"edited": "bool", "summary": "str"},
)


class CapabilityHookTests(unittest.TestCase):
    def test_denies_tool_outside_positive_set(self) -> None:
        runner = SDKSessionRunner(model="m", events=EventsLog(Path(tempfile.mkdtemp())))
        hook = runner._capability_hook(SIMPLE_ROLE)
        verdict = asyncio.run(hook({"tool_name": "Bash", "tool_input": {}}, None, {}))
        decision = verdict["hookSpecificOutput"]
        self.assertEqual(decision["permissionDecision"], "deny")
        self.assertIn("hillclimb-editor", decision["permissionDecisionReason"])

    def test_allows_positive_set_and_receipt_tool(self) -> None:
        runner = SDKSessionRunner(model="m", events=EventsLog(Path(tempfile.mkdtemp())))
        hook = runner._capability_hook(SIMPLE_ROLE)
        for name in ("Read", "Write", "mcp__receipts__submit_receipt"):
            verdict = asyncio.run(hook({"tool_name": name, "tool_input": {}}, None, {}))
            self.assertEqual(verdict, {}, name)


class BashPatternTests(unittest.TestCase):
    def _hook(self, role: RoleDefinition):
        runner = SDKSessionRunner(model="m", events=EventsLog(Path(tempfile.mkdtemp())))
        return runner._capability_hook(role)

    def test_denies_bash_command_off_prefix(self) -> None:
        hook = self._hook(ROLES["crash-diagnosis"])
        verdict = asyncio.run(hook(
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf x"}},
            None, {}))
        decision = verdict["hookSpecificOutput"]
        self.assertEqual(decision["permissionDecision"], "deny")
        self.assertIn("crash-diagnosis", decision["permissionDecisionReason"])
        self.assertIn("render-failure", decision["permissionDecisionReason"])

    def test_allows_render_failure_command(self) -> None:
        hook = self._hook(ROLES["crash-diagnosis"])
        verdict = asyncio.run(hook(
            {"tool_name": "Bash",
             "tool_input": {"command": "python tools/tuners/tune_tools.py "
                                       "render-failure --run-dir runs/t/x"}},
            None, {}))
        self.assertEqual(verdict, {})

    def test_role_without_bash_patterns_keeps_name_only(self) -> None:
        role = RoleDefinition(
            name="shell-role",
            prompt_file="shell-role.md",
            tools=("Bash",),
            disallowed=(),
            receipt_schema={},
        )
        hook = self._hook(role)
        verdict = asyncio.run(hook(
            {"tool_name": "Bash", "tool_input": {"command": "rm -rf x"}},
            None, {}))
        self.assertEqual(verdict, {})


class FakeSystemMessage:
    def __init__(self, session_id: str):
        self.subtype = "init"
        self.data = {"session_id": session_id}


class FakeResultMessage:
    def __init__(self):
        self.session_id = "sess-fake"
        self.is_error = False
        self.num_turns = 1
        self.total_cost_usd = 0.01
        self.usage = {"input_tokens": 10}


class FakeClient:
    """Scripted stand-in for ClaudeSDKClient.

    behavior: list of per-query outcomes. Each outcome is a dict:
      {"accept": bool}  — simulate submit_receipt accepted this turn
    The fake "fixes" postconditions (creates train.py) when accept=True.
    """

    def __init__(self, run_dir: Path, store: ReceiptStore, invocation_id: int,
                 behavior: list[dict]):
        self.run_dir = run_dir
        self.store = store
        self.invocation_id = invocation_id
        self.behavior = list(behavior)
        self.queries: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def receive_response(self):
        yield FakeSystemMessage("sess-fake")
        outcome = self.behavior.pop(0) if self.behavior else {"accept": False}
        if outcome.get("accept"):
            (self.run_dir / "train.py").write_text("# edited\n")
            self.store.persist_receipt(
                "hillclimb-editor", self.invocation_id,
                {"edited": True, "summary": "ok"})
        yield FakeResultMessage()


def make_runner(run_dir: Path, behavior: list[dict]) -> SDKSessionRunner:
    store = ReceiptStore(run_dir)

    def factory(options):
        return FakeClient(run_dir, store, 1, behavior)

    return SDKSessionRunner(model="m", events=EventsLog(run_dir),
                            client_factory=factory)


class VerifyRepairTests(unittest.TestCase):
    def test_success_first_try(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = make_runner(run_dir, [{"accept": True}])
            receipt = runner.run(SIMPLE_ROLE, make_ctx(run_dir))
            self.assertEqual(receipt["edited"], True)
            self.assertEqual(
                ReceiptStore(run_dir).load_session_id("hillclimb-editor", 1),
                "sess-fake",
            )

    def test_corrective_followup_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = make_runner(run_dir, [{"accept": False}, {"accept": True}])
            receipt = runner.run(SIMPLE_ROLE, make_ctx(run_dir))
            self.assertTrue(receipt["edited"])

    def test_invocation_failed_after_bounded_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = make_runner(run_dir, [])  # never accepts
            with self.assertRaises(InvocationFailed) as cm:
                runner.run(SIMPLE_ROLE, make_ctx(run_dir))
            self.assertEqual(cm.exception.role, "hillclimb-editor")
            self.assertTrue(cm.exception.problems)


class FakeSessionRunnerTests(unittest.TestCase):
    def test_scripted_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = FakeSessionRunner([
                {"receipt": {"edited": True, "summary": "ok"}},
                {"fail": ["postcondition: train.py missing"]},
            ])
            receipt = runner.run(ROLES["hillclimb-editor"], make_ctx(run_dir))
            self.assertTrue(receipt["edited"])
            with self.assertRaises(InvocationFailed):
                runner.run(ROLES["hillclimb-editor"], make_ctx(run_dir))


if __name__ == "__main__":
    unittest.main()
