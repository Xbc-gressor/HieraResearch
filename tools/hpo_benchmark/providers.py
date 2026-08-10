"""Fresh-call proposal-provider boundary used by LLM-backed benchmark arms."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


@dataclass(frozen=True)
class ProviderResponse:
    output: Mapping[str, Any]
    raw_output: str
    model: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FreshProposalProvider(Protocol):
    """One complete call with no resumable-session argument or return value."""

    def complete(
        self, prompt: str, *, output_schema: Mapping[str, Any]
    ) -> ProviderResponse: ...


class ReplayProposalProvider:
    """No-network provider for CPU tests and recorded-proposal replays."""

    def __init__(self, outputs: list[Mapping[str, Any]], *, model: str = "replay"):
        self._outputs = list(outputs)
        self.model = model
        self.calls: list[dict[str, Any]] = []

    def complete(
        self, prompt: str, *, output_schema: Mapping[str, Any]
    ) -> ProviderResponse:
        if not self._outputs:
            raise RuntimeError("replay provider has no proposal left")
        output = dict(self._outputs.pop(0))
        self.calls.append({"prompt": prompt, "output_schema": dict(output_schema)})
        return ProviderResponse(
            output=output,
            raw_output=json.dumps(output, ensure_ascii=False),
            model=self.model,
            metadata={"replay_index": len(self.calls) - 1},
        )


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


class ClaudeCLIProposalProvider:
    """Fresh, tool-free Claude CLI invocation with structured JSON output."""

    def __init__(
        self,
        *,
        model: str,
        cwd: Path,
        cli_path: str = "claude",
        max_budget_usd: float | None = None,
        retries: int = 2,
        command_runner: CommandRunner = subprocess.run,
    ):
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.model = model
        self.cwd = Path(cwd)
        self.cli_path = cli_path
        self.max_budget_usd = max_budget_usd
        self.retries = retries
        self._command_runner = command_runner
        self.calls = 0
        self.attempts = 0

    def complete(
        self, prompt: str, *, output_schema: Mapping[str, Any]
    ) -> ProviderResponse:
        command: list[str] = [
            self.cli_path,
            "--print",
            "--safe-mode",
            "--tools",
            "",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--json-schema",
            json.dumps(output_schema, ensure_ascii=False, allow_nan=False),
            "--model",
            self.model,
        ]
        if self.max_budget_usd is not None:
            command.extend(["--max-budget-usd", str(self.max_budget_usd)])
        command.append(prompt)
        attempts_this_call = 0
        for _ in range(self.retries + 1):
            completed = self._command_runner(
                command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            attempts_this_call += 1
            self.attempts += 1
            if completed.returncode == 0:
                break
        else:
            detail = (completed.stderr or completed.stdout).strip()
            if len(detail) > 2000:
                detail = "...[output truncated]...\n" + detail[-2000:]
            raise RuntimeError(
                "Claude proposal call failed after "
                f"{attempts_this_call} attempts; last exit code "
                f"{completed.returncode}: {detail}"
            )
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Claude proposal call did not return a JSON envelope") from exc
        output = envelope.get("structured_output")
        if not isinstance(output, Mapping):
            raise RuntimeError("Claude proposal call returned no structured_output object")
        self.calls += 1
        metadata = {
            key: envelope[key]
            for key in (
                "session_id",
                "total_cost_usd",
                "duration_ms",
                "duration_api_ms",
                "num_turns",
                "usage",
                "modelUsage",
            )
            if key in envelope
        }
        metadata["provider_call_index"] = self.calls - 1
        metadata["provider_attempt_count"] = attempts_this_call
        metadata["provider_retry_count"] = attempts_this_call - 1
        metadata["provider_total_attempts"] = self.attempts
        return ProviderResponse(
            output=dict(output),
            raw_output=json.dumps(output, ensure_ascii=False, allow_nan=False),
            model=self.model,
            metadata=metadata,
        )
