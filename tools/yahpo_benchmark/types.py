from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    params: dict[str, Any]
    raw_target: float
    score: float


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    total_cost_usd: float = 0.0
    attempts: list[dict[str, Any]] = field(default_factory=list)

    def add(self, entry: dict[str, Any]) -> None:
        self.calls += 1
        self.input_tokens += int(entry.get("input_tokens", 0) or 0)
        self.output_tokens += int(entry.get("output_tokens", 0) or 0)
        self.cache_read_input_tokens += int(
            entry.get("cache_read_input_tokens", 0) or 0
        )
        self.cache_creation_input_tokens += int(
            entry.get("cache_creation_input_tokens", 0) or 0
        )
        self.total_cost_usd += float(entry.get("total_cost_usd", 0.0) or 0.0)
        self.attempts.append(dict(entry))

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "total_cost_usd": self.total_cost_usd,
            "attempts": list(self.attempts),
        }
