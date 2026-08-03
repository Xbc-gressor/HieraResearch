"""Retryable upstream API failure classification and coordinator backoff policy.

Python owns recovery: transient provider outages (502/503 and related transport
faults) must not hard-block a run until a configured streak or cumulative
backoff wall-clock budget is exhausted. Contract, request, and artifact
failures remain non-retryable at this layer.

Call sites may still enforce a short local transport attempt window. When that
window is exhausted the raised error is only coordinator-retryable if the
*underlying* failure is upstream; the call site must reopen a local window on
re-entry so backoff is not a pure sleep loop.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


# Default run-level policy (not per-call transport retries inside call sites).
DEFAULT_UPSTREAM_MAX_STREAK = 8
DEFAULT_UPSTREAM_MAX_BACKOFF_TOTAL_SECONDS = 2 * 60 * 60
DEFAULT_UPSTREAM_BASE_BACKOFF_SECONDS = 30.0
DEFAULT_UPSTREAM_MAX_SINGLE_BACKOFF_SECONDS = 15 * 60
DEFAULT_UPSTREAM_BACKOFF_MULTIPLIER = 2.0


_UPSTREAM_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})

_LOCAL_TRANSPORT_BUDGET_EXHAUSTED = re.compile(
    r"transport retry limit reached", re.I
)

# Signatures of the underlying provider/transport fault (not local budget text).
_UPSTREAM_MESSAGE_PATTERNS = (
    re.compile(r"\bError code:\s*(408|425|429|500|502|503|504|529)\b", re.I),
    re.compile(r"\bstatus[_ ]?code[=:\s]+(408|425|429|500|502|503|504|529)\b", re.I),
    re.compile(r"\bupstream_error\b", re.I),
    re.compile(r"\boverloaded_error\b", re.I),
    re.compile(r"\brate_limit(?:_error)?\b", re.I),
    re.compile(r"temporarily unavailable", re.I),
    re.compile(r"no available accounts", re.I),
    re.compile(r"service unavailable", re.I),
    re.compile(r"bad gateway", re.I),
    re.compile(r"gateway timeout", re.I),
    re.compile(r"connection reset", re.I),
    re.compile(r"connection aborted", re.I),
    re.compile(r"remote (?:end|server) closed", re.I),
    re.compile(r"timed? out", re.I),
    re.compile(r"APIConnectionError", re.I),
    re.compile(r"InternalServerError", re.I),
    re.compile(r"ServiceUnavailable", re.I),
)


@dataclass(frozen=True)
class UpstreamBackoffPolicy:
    """Run-level caps for coordinator auto-backoff on upstream failures."""

    max_streak: int = DEFAULT_UPSTREAM_MAX_STREAK
    max_backoff_total_seconds: float = DEFAULT_UPSTREAM_MAX_BACKOFF_TOTAL_SECONDS
    base_backoff_seconds: float = DEFAULT_UPSTREAM_BASE_BACKOFF_SECONDS
    max_single_backoff_seconds: float = DEFAULT_UPSTREAM_MAX_SINGLE_BACKOFF_SECONDS
    backoff_multiplier: float = DEFAULT_UPSTREAM_BACKOFF_MULTIPLIER

    def __post_init__(self) -> None:
        if self.max_streak < 1:
            raise ValueError("max_streak must be >= 1")
        if self.max_backoff_total_seconds <= 0:
            raise ValueError("max_backoff_total_seconds must be positive")
        if self.base_backoff_seconds <= 0:
            raise ValueError("base_backoff_seconds must be positive")
        if self.max_single_backoff_seconds < self.base_backoff_seconds:
            raise ValueError(
                "max_single_backoff_seconds must be >= base_backoff_seconds"
            )
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1")


@dataclass(frozen=True)
class UpstreamRecoveryDecision:
    """Pure outcome of one upstream failure against durable counters."""

    action: str  # "backoff" | "block"
    sleep_seconds: float
    streak: int
    backoff_total_seconds: float
    reason: str


def exception_status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from SDK / wrapped exceptions."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        code = getattr(current, "status_code", None)
        if isinstance(code, int) and not isinstance(code, bool):
            return code
        response = getattr(current, "response", None)
        if response is not None:
            resp_code = getattr(response, "status_code", None)
            if isinstance(resp_code, int) and not isinstance(resp_code, bool):
                return resp_code
        current = current.__cause__ or current.__context__
    return None


def text_matches_upstream_provider_failure(text: str) -> bool:
    """True when *text* describes a provider/transport fault (not local budget)."""
    if not text or not str(text).strip():
        return False
    return any(pattern.search(text) for pattern in _UPSTREAM_MESSAGE_PATTERNS)


def is_retryable_upstream_failure(exc: BaseException) -> bool:
    """Return True when *exc* is a transient provider/transport failure.

    Non-retryable inference contract/request errors must not match even if their
    message string is noisy. A call-site message of the form
    ``transport retry limit reached: <cause>`` is retryable only when *cause*
    itself looks like an upstream provider failure — never solely because the
    local attempt window is exhausted.
    """
    # Local import avoids a hard cycle at module import time in some test paths.
    from .llm import InferenceContractError, InferenceRequestError

    if isinstance(exc, (InferenceRequestError, InferenceContractError)):
        return False

    code = exception_status_code(exc)
    if code in _UPSTREAM_STATUS_CODES:
        return True

    text = _exception_text(exc)
    if not text:
        return False

    budget = _LOCAL_TRANSPORT_BUDGET_EXHAUSTED.search(text)
    if budget is not None:
        # Only the nested cause decides; exhausted local budget alone is not
        # an upstream signal and must not drive coordinator sleep loops.
        return text_matches_upstream_provider_failure(text[budget.end() :])

    return text_matches_upstream_provider_failure(text)


def last_error_is_upstream_transport(last_error: object) -> bool:
    """Whether a persisted call-site ``last_error`` string is upstream-class."""
    if not isinstance(last_error, str) or not last_error.strip():
        return False
    from .llm import InferenceError

    return is_retryable_upstream_failure(InferenceError(last_error))


def backoff_sleep_seconds(streak_after_failure: int, policy: UpstreamBackoffPolicy) -> float:
    """Exponential backoff for the given 1-based streak, capped per attempt."""
    if streak_after_failure < 1:
        raise ValueError("streak_after_failure must be >= 1")
    exponent = streak_after_failure - 1
    raw = policy.base_backoff_seconds * (policy.backoff_multiplier**exponent)
    return float(min(policy.max_single_backoff_seconds, raw))


def decide_upstream_recovery(
    *,
    current_streak: int,
    current_backoff_total_seconds: float,
    error: BaseException | str,
    policy: UpstreamBackoffPolicy,
) -> UpstreamRecoveryDecision:
    """Advance counters for one upstream failure and choose backoff vs block.

    ``current_*`` are the durable values *before* this failure. On backoff, the
    returned streak/total already include this failure and the planned sleep.
    """
    if isinstance(error, str):
        from .llm import InferenceError

        if not is_retryable_upstream_failure(InferenceError(error)):
            raise ValueError(
                "decide_upstream_recovery requires a retryable upstream error"
            )
    elif not is_retryable_upstream_failure(error):
        raise ValueError("decide_upstream_recovery requires a retryable upstream error")

    streak = int(current_streak) + 1
    err_text = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    err_text = err_text.strip()[:500] or "upstream failure"

    if streak > policy.max_streak:
        return UpstreamRecoveryDecision(
            action="block",
            sleep_seconds=0.0,
            streak=streak,
            backoff_total_seconds=float(current_backoff_total_seconds),
            reason=(
                f"upstream_failure_streak_exhausted:{streak}/{policy.max_streak}: "
                f"{err_text}"
            ),
        )

    sleep_for = backoff_sleep_seconds(streak, policy)
    projected_total = float(current_backoff_total_seconds) + sleep_for
    if projected_total > policy.max_backoff_total_seconds + 1e-9:
        return UpstreamRecoveryDecision(
            action="block",
            sleep_seconds=0.0,
            streak=streak,
            backoff_total_seconds=float(current_backoff_total_seconds),
            reason=(
                "upstream_backoff_wall_clock_exhausted:"
                f"{current_backoff_total_seconds:.1f}+{sleep_for:.1f}s>"
                f"{policy.max_backoff_total_seconds:.1f}s: {err_text}"
            ),
        )

    return UpstreamRecoveryDecision(
        action="backoff",
        sleep_seconds=sleep_for,
        streak=streak,
        backoff_total_seconds=projected_total,
        reason=f"upstream_backoff:{sleep_for:.1f}s:streak={streak}: {err_text}",
    )


def upstream_fields_reset() -> dict[str, Any]:
    """Durable coordinator fields after a successful model invocation."""
    return {
        "upstream_failure_streak": 0,
        "upstream_backoff_total_seconds": 0.0,
        "upstream_last_error": None,
        "upstream_last_decision": None,
    }


def parse_upstream_failure_streak(value: object, *, present: bool) -> int:
    """Strict non-negative int; absent field defaults to 0."""
    if not present or value is None:
        return 0
    if type(value) is not int:  # reject bool and numeric strings
        raise ValueError(
            f"upstream_failure_streak must be a non-negative int, got {value!r}"
        )
    if value < 0:
        raise ValueError(f"upstream_failure_streak must be >= 0, got {value}")
    return value


def parse_upstream_backoff_total_seconds(value: object, *, present: bool) -> float:
    """Strict non-negative finite float; absent field defaults to 0.0."""
    if not present or value is None:
        return 0.0
    if type(value) is bool or type(value) not in (int, float):
        raise ValueError(
            "upstream_backoff_total_seconds must be a non-negative finite number, "
            f"got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(
            "upstream_backoff_total_seconds must be a non-negative finite number, "
            f"got {value!r}"
        )
    return number


def parse_optional_upstream_text(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{field} must be a string or null, got {value!r}")
    return value


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = [f"{type(exc).__name__}: {exc}"]
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        body = getattr(current, "body", None)
        if body is not None:
            parts.append(repr(body))
        current = current.__cause__ or current.__context__
    return "\n".join(parts)
