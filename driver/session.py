"""Session layer: one role invocation = one SDK session + verify-repair loop.

The loop: run the session → check the accepted receipt + postconditions →
corrective follow-up in the SAME session with concrete diagnostics → repeat
up to role.corrective_attempts → InvocationFailed. Escalation beyond that
is the loops' job (role-specific, see spec Error handling).

A drain is interrupted as soon as a schema-valid receipt is accepted: the
receipt is the role's final artifact, so post-acceptance turns can only be
waste (observed: models rambling or resubmitting for 10-30 extra turns).
Corrective follow-ups may still resubmit (latest wins, see receipts.py).
"""

from __future__ import annotations

import json

import anyio
from pathlib import Path
from typing import Callable, Protocol

from .events import EventsLog
from .receipts import ReceiptStore, build_receipt_server
from .roles import (
    PROMPT_DIR,
    REPO_ROOT,
    InvocationContext,
    RoleDefinition,
    driver_job_handoff_problem,
)

RECEIPT_TOOL = "mcp__receipts__submit_receipt"

# Consecutive identical (tool, input) calls before the repetition breaker
# trips. Observed incident: a model retried one hallucinated Edit verbatim
# 880+ times until the context overflowed. Healthy retries re-read the file
# or change the input, so the fingerprint necessarily changes.
REPETITION_LIMIT = 5


def new_breaker() -> dict:
    return {"fp": None, "count": 0, "tripped": None}


class InvocationFailed(Exception):
    def __init__(self, role: str, problems: list[str], *, invocation_id: int | None = None):
        self.role = role
        self.problems = problems
        self.invocation_id = invocation_id
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

    def _capability_hook(self, role: RoleDefinition, breaker: dict | None = None):
        """Fail-closed: deny every tool outside the role's positive set.

        A PreToolUse hook (not canUseTool — that callback is shadowed under
        bypassPermissions and never reached; hooks still run). When the role
        declares bash_patterns, Bash commands must also start with one of
        those prefixes. The same hook hosts the repetition breaker: the same
        (tool, input) call REPETITION_LIMIT times in a row trips it, later
        calls are denied, and the drain interrupts the session.
        """
        if breaker is None:
            breaker = new_breaker()
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
            fp = (input_data.get("tool_name", ""),
                  json.dumps(input_data.get("tool_input"), sort_keys=True))
            if fp == breaker["fp"]:
                breaker["count"] += 1
            else:
                breaker["fp"], breaker["count"] = fp, 1
            if breaker["tripped"] is None and breaker["count"] >= REPETITION_LIMIT:
                breaker["tripped"] = (f"{fp[0]} invoked {breaker['count']}x "
                                      "with identical input")
            if breaker["tripped"] is not None:
                return deny(f"repetition breaker tripped: {breaker['tripped']}")
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
            if name == "Bash" and role.forbidden_bash_substrings:
                command = (input_data.get("tool_input") or {}).get("command", "")
                forbidden = next(
                    (part for part in role.forbidden_bash_substrings if part in command),
                    None,
                )
                if forbidden is not None:
                    return deny(
                        f"role {role.name} may not launch long objective work via "
                        f"Bash ({forbidden!r}); submit a driver_job receipt"
                    )
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

    def _build_options(self, role: RoleDefinition, ctx: InvocationContext,
                       server, breaker: dict | None = None):
        if breaker is None:
            breaker = new_breaker()
        from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

        kwargs = {}
        if self.cli_path:
            kwargs["cli_path"] = self.cli_path
        if ctx.resume_session_id:
            kwargs["resume"] = ctx.resume_session_id
        if role.max_turns is not None:
            kwargs["max_turns"] = role.max_turns
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
                                              hooks=[self._capability_hook(
                                                  role, breaker)])]},
            **kwargs,
        )

    async def _drain(self, client, role: RoleDefinition, ctx: InvocationContext,
                     store: ReceiptStore, accepted: list[dict],
                     breaker: dict | None = None) -> dict | None:
        if breaker is None:
            breaker = new_breaker()
        from claude_agent_sdk import ResultMessage, SystemMessage

        interrupted = False
        # {"is_error", "subtype"} of the session's ResultMessage, so the caller
        # can tell an error-terminated session (e.g. error_max_turns) apart
        # from an ordinary no-receipt turn.
        result_info = None
        # Receipts accepted by EARLIER drains (e.g. the one whose postcondition
        # failure triggered this corrective turn) must not interrupt this
        # drain — only a NEW acceptance ends this turn's useful work.
        accepted_baseline = len(accepted)
        async for msg in client.receive_response():
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                session_id = msg.data.get("session_id")
                if session_id:
                    # Persist IMMEDIATELY on init — a later kill can then
                    # resume this exact conversation via resume=<session_id>.
                    store.persist_session_id(role.name, ctx.invocation_id, session_id)
            elif isinstance(msg, ResultMessage):
                result_info = {"is_error": msg.is_error, "subtype": msg.subtype}
                store.mark_session_ended(role.name, ctx.invocation_id,
                                         not msg.is_error, msg.subtype)
                self.events.emit(
                    "session_end",
                    role=role.name,
                    invocation_id=ctx.invocation_id,
                    session_id=msg.session_id,
                    is_error=msg.is_error,
                    num_turns=msg.num_turns,
                    total_cost_usd=msg.total_cost_usd,
                    usage=msg.usage,
                    # error detail matters most when is_error is true; these
                    # are None/empty on success and cheap to carry always
                    stop_reason=msg.stop_reason,
                    api_error_status=msg.api_error_status,
                    errors=msg.errors,
                    result_excerpt=(msg.result or "")[:500] or None,
                )
            # duck-typed test fakes
            elif getattr(msg, "subtype", None) == "init":
                session_id = msg.data.get("session_id")
                if session_id:
                    store.persist_session_id(role.name, ctx.invocation_id, session_id)
            elif hasattr(msg, "num_turns"):
                result_info = {
                    "is_error": msg.is_error,
                    "subtype": getattr(msg, "subtype", None),
                }
                store.mark_session_ended(role.name, ctx.invocation_id,
                                         not msg.is_error,
                                         getattr(msg, "subtype", None))
                self.events.emit(
                    "session_end",
                    role=role.name,
                    invocation_id=ctx.invocation_id,
                    session_id=msg.session_id,
                    is_error=msg.is_error,
                    num_turns=msg.num_turns,
                    total_cost_usd=msg.total_cost_usd,
                    usage=msg.usage,
                    stop_reason=getattr(msg, "stop_reason", None),
                    api_error_status=getattr(msg, "api_error_status", None),
                    errors=getattr(msg, "errors", None),
                    result_excerpt=(getattr(msg, "result", None) or "")[:500]
                                   or None,
                )
            # A schema-valid receipt ends the invocation's useful work — the
            # receipt protocol defines it as the role's FINAL artifact
            # ("Call after ALL on-disk work is complete"). Without an
            # interrupt the model keeps going after "receipt accepted":
            # observed up to 31 extra turns and ~$5 per invocation in
            # degenerate repetition loops. Keep draining after the interrupt
            # so the ResultMessage (usage accounting) still lands.
            if (len(accepted) > accepted_baseline
                    or breaker["tripped"] is not None) and not interrupted:
                if breaker["tripped"] is not None:
                    self.events.emit("repetition_breaker", role=role.name,
                                     invocation_id=ctx.invocation_id,
                                     description=breaker["tripped"])
                interrupted = True
                interrupt = getattr(client, "interrupt", None)
                if interrupt is not None:
                    try:
                        await interrupt()
                    except Exception:
                        # The receipt is already persisted; a failed
                        # interrupt must not fail the invocation.
                        pass
        return result_info

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
        # A typed long-job request is an intermediate handoff, not a completed
        # role invocation. The driver executes it synchronously and resumes the
        # same session; terminal postconditions are checked on that later turn.
        if receipt is not None and isinstance(receipt.get("driver_job"), dict):
            handoff_problem = driver_job_handoff_problem(role.name, receipt)
            if handoff_problem:
                problems.append(handoff_problem)
            return problems
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
        breaker = new_breaker()
        options = self._build_options(role, ctx, server, breaker)
        factory = self._client_factory or self._default_client_factory
        self.events.emit("session_start", role=role.name,
                         invocation_id=ctx.invocation_id,
                         resume=bool(ctx.resume_session_id))
        async with factory(options) as client:
            await client.query(ctx.user_message())
            result = await self._drain(client, role, ctx, store, accepted,
                                       breaker)
            receipt = self._latest_receipt(store, role, ctx, accepted)
            problems = self._problems(role, ctx, receipt)
            if breaker["tripped"] is not None:
                problems.append(
                    f"repetition breaker tripped: {breaker['tripped']}")
            attempts = 0
            while (problems and attempts < role.corrective_attempts
                   and breaker["tripped"] is None):
                if result and result["is_error"]:
                    # The session ended on an error result (e.g.
                    # error_max_turns from a role's max_turns cap): the CLI
                    # process may be dead, and a corrective turn cannot
                    # produce the receipt anyway. Fail directly instead of
                    # querying a spent session, so callers' InvocationFailed
                    # handlers see every bounded-role cutoff.
                    problems.append(
                        f"session ended with error result: {result['subtype']}"
                    )
                    break
                attempts += 1
                self.events.emit("corrective_followup", role=role.name,
                                 invocation_id=ctx.invocation_id,
                                 attempt=attempts, problems=problems)
                await client.query(self._corrective_message(problems))
                result = await self._drain(client, role, ctx, store, accepted,
                                           breaker)
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
