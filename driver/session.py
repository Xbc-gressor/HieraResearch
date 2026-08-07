"""Session layer: one role invocation = one SDK session + verify-repair loop.

The loop: run the session → check the accepted receipt + postconditions →
corrective follow-up in the SAME session with concrete diagnostics → repeat
up to role.corrective_attempts → InvocationFailed. Escalation beyond that
is the loops' job (role-specific, see spec Error handling).
"""

from __future__ import annotations

import json

import anyio
from pathlib import Path
from typing import Callable, Protocol

from .events import EventsLog
from .receipts import ReceiptStore, build_receipt_server
from .roles import PROMPT_DIR, REPO_ROOT, InvocationContext, RoleDefinition

RECEIPT_TOOL = "mcp__receipts__submit_receipt"


class InvocationFailed(Exception):
    def __init__(self, role: str, problems: list[str]):
        self.role = role
        self.problems = problems
        super().__init__(f"{role}: " + "; ".join(problems))


class SessionRunner(Protocol):
    def run(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        """Return the accepted receipt payload; raise InvocationFailed."""
        ...


class SDKSessionRunner:
    def __init__(
        self,
        model: str,
        events: EventsLog,
        cli_path: str | None = None,
        client_factory: Callable | None = None,
    ):
        self.model = model
        self.events = events
        self.cli_path = cli_path
        self._client_factory = client_factory

    # -- public -------------------------------------------------------------

    def run(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        return anyio.run(self._run_async, role, ctx)

    # -- internals ------------------------------------------------------------

    def _capability_hook(self, role: RoleDefinition):
        """Fail-closed: deny every tool outside the role's positive set.

        A PreToolUse hook (not canUseTool — that callback is shadowed under
        bypassPermissions and never reached; hooks still run). When the role
        declares bash_patterns, Bash commands must also start with one of
        those prefixes.
        """
        allowed = set(role.tools) | {RECEIPT_TOOL}

        def deny(reason: str) -> dict:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }

        async def hook(input_data, tool_use_id, context):
            name = input_data.get("tool_name", "")
            if name not in allowed:
                return deny(f"role {role.name} may not use tool {name}")
            if name == "Bash" and role.bash_patterns:
                # Coarse prefix matching — compound commands (&&, ;, |) can
                # evade it. This is minimal containment, not a sandbox.
                command = (input_data.get("tool_input") or {}).get("command", "")
                if not any(command.startswith(p) for p in role.bash_patterns):
                    return deny(
                        f"role {role.name} may only run Bash commands starting "
                        f"with: {', '.join(role.bash_patterns)}")
            return {}

        return hook

    def _system_prompt(self, role: RoleDefinition, ctx: InvocationContext) -> str:
        prompt_path = PROMPT_DIR / role.prompt_file
        # driver/prompts/ lands in a later task; tolerate its absence so the
        # runner is testable before the prompt files exist.
        body = (prompt_path.read_text(encoding="utf-8")
                if prompt_path.exists() else "")
        return (
            body
            + "\n\n---\n\n## Invocation context\n\n"
            + ctx.user_message()
            + "\n\n"
            + ctx.postcondition_checklist(role)
        )

    def _default_client_factory(self, options):
        from claude_agent_sdk import ClaudeSDKClient

        return ClaudeSDKClient(options=options)

    def _build_options(self, role: RoleDefinition, ctx: InvocationContext, server):
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        kwargs = {}
        if self.cli_path:
            kwargs["cli_path"] = self.cli_path
        if ctx.resume_session_id:
            kwargs["resume"] = ctx.resume_session_id
        return ClaudeAgentOptions(
            system_prompt=self._system_prompt(role, ctx),
            cwd=REPO_ROOT,
            model=self.model,
            permission_mode="bypassPermissions",
            # Claude Code refuses bypassPermissions under root unless it is
            # told the session runs in a sandbox; our runs do (dedicated
            # server/container). Harmless for non-root users.
            env={"IS_SANDBOX": "1"},
            setting_sources=[],
            disallowed_tools=list(role.disallowed),
            mcp_servers={"receipts": server},
            hooks={"PreToolUse": [HookMatcher(matcher=None,
                                              hooks=[self._capability_hook(role)])]},
            **kwargs,
        )

    async def _drain(self, client, role: RoleDefinition, ctx: InvocationContext,
                     store: ReceiptStore) -> None:
        from claude_agent_sdk import ResultMessage, SystemMessage

        async for msg in client.receive_response():
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                session_id = msg.data.get("session_id")
                if session_id:
                    # Persist IMMEDIATELY on init — a later kill can then
                    # resume this exact conversation via resume=<session_id>.
                    store.persist_session_id(role.name, ctx.invocation_id, session_id)
            elif isinstance(msg, ResultMessage):
                self.events.emit(
                    "session_end",
                    role=role.name,
                    invocation_id=ctx.invocation_id,
                    session_id=msg.session_id,
                    is_error=msg.is_error,
                    num_turns=msg.num_turns,
                    total_cost_usd=msg.total_cost_usd,
                    usage=msg.usage,
                )
            # duck-typed test fakes
            elif getattr(msg, "subtype", None) == "init":
                session_id = msg.data.get("session_id")
                if session_id:
                    store.persist_session_id(role.name, ctx.invocation_id, session_id)
            elif hasattr(msg, "num_turns"):
                self.events.emit(
                    "session_end",
                    role=role.name,
                    invocation_id=ctx.invocation_id,
                    session_id=msg.session_id,
                    is_error=msg.is_error,
                    num_turns=msg.num_turns,
                    total_cost_usd=msg.total_cost_usd,
                    usage=msg.usage,
                )

    @staticmethod
    def _latest_receipt(store: ReceiptStore, role: RoleDefinition,
                        ctx: InvocationContext, accepted: list[dict]) -> dict | None:
        """Newest accepted receipt payload, or None.

        The real SDK path appends to ``accepted`` via the receipts MCP tool;
        injected test clients persist directly through the store, so fall
        back to the on-disk receipt — the durable artifact is the source of
        truth either way.
        """
        if accepted:
            return accepted[-1]
        path = store.receipt_path(role.name, ctx.invocation_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def _problems(self, role: RoleDefinition, ctx: InvocationContext,
                  receipt: dict | None) -> list[str]:
        problems = [] if receipt is not None else ["no accepted receipt"]
        for check in role.postconditions:
            error = check(ctx)
            if error:
                problems.append(error)
        return problems

    @staticmethod
    def _corrective_message(problems: list[str]) -> str:
        items = "\n".join(f"- {p}" for p in problems)
        return (
            "Your previous turn did NOT satisfy this invocation's requirements. "
            "Fix each problem below using your tools, then call "
            f"{RECEIPT_TOOL} again:\n{items}"
        )

    async def _run_async(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        store = ReceiptStore(ctx.run_dir)
        server, accepted = build_receipt_server(
            role.name, role.receipt_schema, store, ctx.invocation_id
        )
        options = self._build_options(role, ctx, server)
        factory = self._client_factory or self._default_client_factory
        self.events.emit("session_start", role=role.name,
                         invocation_id=ctx.invocation_id,
                         resume=bool(ctx.resume_session_id))
        async with factory(options) as client:
            await client.query(ctx.user_message())
            await self._drain(client, role, ctx, store)
            receipt = self._latest_receipt(store, role, ctx, accepted)
            problems = self._problems(role, ctx, receipt)
            attempts = 0
            while problems and attempts < role.corrective_attempts:
                attempts += 1
                self.events.emit("corrective_followup", role=role.name,
                                 invocation_id=ctx.invocation_id,
                                 attempt=attempts, problems=problems)
                await client.query(self._corrective_message(problems))
                await self._drain(client, role, ctx, store)
                receipt = self._latest_receipt(store, role, ctx, accepted)
                problems = self._problems(role, ctx, receipt)
        if problems:
            raise InvocationFailed(role.name, problems)
        return receipt


class FakeSessionRunner:
    """Scripted runner for loop tests. Each script entry:

    {"receipt": {...},
     "side_effects": Callable[[InvocationContext], None] | None,
     "fail": list[str] | None}
    """

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[tuple[str, InvocationContext]] = []

    def run(self, role: RoleDefinition, ctx: InvocationContext) -> dict:
        self.calls.append((role.name, ctx))
        # Mirror the real runner's init-time persistence so loop tests can
        # exercise same-session resume (resume=<session_id>).
        ReceiptStore(ctx.run_dir).persist_session_id(
            role.name, ctx.invocation_id, f"fake-sess-{ctx.invocation_id:04d}"
        )
        if not self.script:
            raise InvocationFailed(role.name, ["fake runner: script exhausted"])
        entry = self.script.pop(0)
        if entry.get("side_effects"):
            entry["side_effects"](ctx)
        if entry.get("fail"):
            raise InvocationFailed(role.name, entry["fail"])
        ReceiptStore(ctx.run_dir).persist_receipt(
            role.name, ctx.invocation_id, entry["receipt"]
        )
        return entry["receipt"]
