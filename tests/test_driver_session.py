import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.events import EventsLog  # noqa: E402
from driver.receipts import ReceiptStore, build_receipt_server  # noqa: E402
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

    def test_tuner_cannot_bypass_driver_owned_job_with_nohup(self) -> None:
        hook = self._hook(ROLES["tuner-orchestrator"])
        verdict = asyncio.run(hook(
            {"tool_name": "Bash", "tool_input": {
                "command": "nohup uv run python tools/tuners/bo_search.py &"
            }},
            None, {}))
        decision = verdict["hookSpecificOutput"]
        self.assertEqual(decision["permissionDecision"], "deny")
        self.assertIn("driver_job", decision["permissionDecisionReason"])


class DriverJobHandoffTests(unittest.TestCase):
    def test_intermediate_job_skips_terminal_postconditions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            ctx = InvocationContext(
                task="t", tag="tag", run_dir=run_dir,
                invocation_id=1, run_id="007",
            )
            receipt = {
                "run_id": "007",
                "status": "driver_job",
                "ledger_updated": False,
                "driver_job": {"kind": "warmstart", "run_id": "007", "k_eval": 2},
            }
            self.assertEqual(
                runner._problems(ROLES["tunable-contract-extractor"], ctx, receipt),
                [],
            )

    def test_terminal_fields_cannot_smuggle_an_extra_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            ctx = InvocationContext(
                task="t", tag="tag", run_dir=run_dir,
                invocation_id=1, run_id="007",
            )
            receipt = {
                "run_id": "007",
                "status": "keep",
                "ledger_updated": True,
                "driver_job": {"kind": "warmstart", "run_id": "007", "k_eval": 2},
            }
            problems = runner._problems(
                ROLES["tunable-contract-extractor"], ctx, receipt
            )
            self.assertEqual(len(problems), 1)
            self.assertIn("status='driver_job'", problems[0])


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


class EarlyInterruptTests(unittest.TestCase):
    """Once the receipts MCP tool accepts a payload mid-stream, _drain must
    interrupt the client (post-acceptance turns are pure waste) yet keep
    draining so the ResultMessage's usage accounting is still emitted."""

    class InterruptibleFakeClient:
        def __init__(self, accepted: list):
            self.accepted = accepted
            self.interrupt_calls = 0

        async def receive_response(self):
            yield FakeSystemMessage("sess-fake")
            # The in-process MCP tool runs between streamed messages.
            self.accepted.append({"edited": True, "summary": "ok"})
            yield object()  # post-acceptance assistant message: pure waste
            yield FakeResultMessage()

        async def interrupt(self):
            self.interrupt_calls += 1

    def test_interrupts_once_and_still_emits_session_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            events = EventsLog(run_dir)
            runner = SDKSessionRunner(model="m", events=events)
            accepted: list[dict] = []
            client = self.InterruptibleFakeClient(accepted)
            asyncio.run(runner._drain(client, SIMPLE_ROLE, make_ctx(run_dir),
                                      ReceiptStore(run_dir), accepted))
            self.assertEqual(client.interrupt_calls, 1)
            rows = [json.loads(line)
                    for line in (run_dir / "driver_events.jsonl")
                    .read_text().splitlines()]
            ends = [r for r in rows if r.get("kind") == "session_end"]
            self.assertEqual(len(ends), 1)
            self.assertEqual(ends[0]["usage"], {"input_tokens": 10})

    def test_no_acceptance_no_interrupt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            client = self.InterruptibleFakeClient([])
            asyncio.run(runner._drain(client, SIMPLE_ROLE, make_ctx(run_dir),
                                      ReceiptStore(run_dir), []))
            self.assertEqual(client.interrupt_calls, 0)

    class CorrectiveFakeClient:
        """Two drains share one accepted list — the corrective-turn shape:
        drain 1 accepts a receipt, then a postcondition failure sends the
        session into a corrective drain (drain 2)."""

        def __init__(self, accepted: list, *, resubmit: bool):
            self.accepted = accepted
            self.resubmit = resubmit
            self.interrupt_calls = 0
            self.drains = 0

        def receive_response(self):
            self.drains += 1
            drain_no = self.drains

            async def stream():
                yield FakeSystemMessage("sess-fake")
                if drain_no == 1 or self.resubmit:
                    self.accepted.append({"edited": True, "summary": "ok"})
                yield object()  # assistant work (e.g. fixing files)
                yield FakeResultMessage()

            return stream()

        async def interrupt(self):
            self.interrupt_calls += 1

    def _two_drains(self, run_dir: Path, *, resubmit: bool):
        runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
        accepted: list[dict] = []
        client = self.CorrectiveFakeClient(accepted, resubmit=resubmit)
        for _ in range(2):
            asyncio.run(runner._drain(client, SIMPLE_ROLE, make_ctx(run_dir),
                                      ReceiptStore(run_dir), accepted))
        return client

    def test_stale_receipt_does_not_interrupt_corrective_turn(self) -> None:
        # Receipt accepted, then postcondition failed: the corrective turn
        # must get to work — only a NEW acceptance may interrupt a drain.
        with tempfile.TemporaryDirectory() as tmp:
            client = self._two_drains(Path(tmp), resubmit=False)
            self.assertEqual(client.interrupt_calls, 1)

    def test_corrective_resubmission_interrupts_again(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = self._two_drains(Path(tmp), resubmit=True)
            self.assertEqual(client.interrupt_calls, 2)


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


class OptionsTests(unittest.TestCase):
    def test_options_set_is_sandbox_and_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            store = ReceiptStore(run_dir)
            server, _ = build_receipt_server(
                "hillclimb-editor", SIMPLE_ROLE.receipt_schema, store, 1)
            runner = SDKSessionRunner(model="m", events=EventsLog(run_dir))
            options = runner._build_options(SIMPLE_ROLE, make_ctx(run_dir), server)
            # Claude Code refuses bypassPermissions under root without this.
            self.assertEqual(options.env.get("IS_SANDBOX"), "1")
            self.assertEqual(options.permission_mode, "bypassPermissions")
            self.assertEqual(options.setting_sources, [])


if __name__ == "__main__":
    unittest.main()
