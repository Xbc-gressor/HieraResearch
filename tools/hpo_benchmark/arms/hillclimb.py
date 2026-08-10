"""Fresh-session LLM hillclimb example arm."""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from typing import Any

from ..core import (
    BenchmarkContext,
    EvaluationOutcome,
    Observation,
    PolicyContractError,
    Proposal,
    ProposalBatch,
    select_incumbent,
)
from ..providers import FreshProposalProvider, ProviderResponse
from ..summary import FocusedSummaryBuilder
from ._repair import complete_with_repair, random_config


OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["changes", "reason"],
    "properties": {
        "changes": {"type": "object"},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}


class LLMHillclimbArm:
    name = "llm_hillclimb"

    def __init__(
        self,
        provider: FreshProposalProvider,
        *,
        summary_builder: FocusedSummaryBuilder | None = None,
        corrective_attempts: int = 3,
    ):
        if corrective_attempts < 0:
            raise ValueError("corrective_attempts must be non-negative")
        self.provider = provider
        self.summary_builder = summary_builder or FocusedSummaryBuilder()
        self.corrective_attempts = corrective_attempts
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self.calls = 0
        self.provider_attempts = 0
        self.corrective_calls = 0
        self.degraded_calls = 0

    def initialize(self, context: BenchmarkContext) -> None:
        self.context = context
        self.observations = list(context.observations)
        self.calls = 0
        self.provider_attempts = 0
        self.corrective_calls = 0
        self.degraded_calls = 0

    def _problems(self, output: Mapping[str, Any]) -> list[str]:
        assert self.context is not None
        problems: list[str] = []
        changes = output.get("changes")
        if not isinstance(changes, Mapping):
            problems.append(
                "changes must be an object mapping dimension names to new values"
            )
        elif not changes:
            problems.append("changes must be non-empty")
        else:
            unknown = sorted(set(changes) - set(self.context.space.names))
            if unknown:
                problems.append(
                    "changes contains keys that are not search-space dimensions: "
                    + ", ".join(unknown)
                    + ". Change only dimensions listed in search_space."
                )
            else:
                incumbent = select_incumbent(
                    [
                        observation
                        for observation in self.observations
                        if set(observation.params)
                        == set(self.context.space.names)
                    ]
                )
                merged = dict(incumbent.params)
                merged.update(changes)
                try:
                    projected = self.context.space.project(merged)
                except (TypeError, ValueError) as exc:
                    problems.append(f"changes cannot be projected: {exc}")
                else:
                    incumbent_key = self.context.space.canonical(incumbent.params)
                    candidate_key = self.context.space.canonical(projected)
                    if candidate_key == incumbent_key:
                        problems.append(
                            "changes are a no-op after projection; change the incumbent"
                        )
                    elif candidate_key in {
                        self.context.space.canonical(observation.params)
                        for observation in self.observations
                        if set(observation.params)
                        == set(self.context.space.names)
                    }:
                        problems.append(
                            "changes revisit a configuration already present in history"
                        )
        reason = output.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            problems.append("reason must be a non-empty string")
        return problems

    def _fallback_changes(self, incumbent: Mapping[str, Any]) -> dict[str, Any]:
        """Degraded-mode proposal: a random config distinct from history."""
        assert self.context is not None
        rng = random.Random(self.context.seed * 10007 + self.calls * 31)
        blocked = {
            self.context.space.canonical(observation.params)
            for observation in self.observations
            if set(observation.params) == set(self.context.space.names)
        }
        blocked.add(self.context.space.canonical(incumbent))
        candidate = random_config(self.context.space, rng)
        for _ in range(64):
            if self.context.space.canonical(candidate) not in blocked:
                return candidate
            candidate = random_config(self.context.space, rng)
        raise PolicyContractError(
            "cannot sample a hillclimb fallback distinct from history"
        )

    def _usable_changes(
        self,
        response: ProviderResponse,
        problems: list[str],
        summary: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        """Return (changes, reason); degrade to a safe fallback if needed."""
        output = response.output
        if not problems:
            return output["changes"], output["reason"].strip()
        assert self.context is not None
        self.degraded_calls += 1
        repaired = "provider response repaired: " + "; ".join(problems)
        # Repair: keep only valid dimension changes that actually move the
        # incumbent; otherwise fall back to a fresh random configuration.
        changes = output.get("changes")
        if isinstance(changes, Mapping):
            incumbent = summary["incumbent"]["params"]
            valid = {
                name: value
                for name, value in changes.items()
                if name in self.context.space.names
            }
            merged = dict(incumbent)
            merged.update(valid)
            try:
                projected = self.context.space.project(merged)
            except (TypeError, ValueError):
                projected = None
            blocked = {
                self.context.space.canonical(observation.params)
                for observation in self.observations
                if set(observation.params) == set(self.context.space.names)
            }
            if projected is not None:
                candidate_key = self.context.space.canonical(projected)
                if (
                    valid
                    and candidate_key
                    != self.context.space.canonical(incumbent)
                    and candidate_key not in blocked
                ):
                    return (
                        {name: projected[name] for name in valid},
                        repaired,
                    )
        fallback = self._fallback_changes(summary["incumbent"]["params"])
        return fallback, "degraded fallback: " + "; ".join(problems)

    def ask(self, remaining_budget: int) -> ProposalBatch:
        if self.context is None:
            raise PolicyContractError("hillclimb arm was not initialized")
        summary = self.summary_builder.build(
            self.context, self.observations, remaining_budget=remaining_budget
        )
        prompt = (
            "Propose one sparse hyperparameter hillclimb step. Scores are "
            "lower-is-better. Change only dimensions that are likely to improve "
            "the incumbent. Return JSON matching the supplied schema.\n\n"
            + json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        )
        response, problems, final_prompt = complete_with_repair(
            self.provider,
            prompt,
            OUTPUT_SCHEMA,
            validate=self._problems,
            corrective_attempts=self.corrective_attempts,
        )
        self.calls += 1
        self.provider_attempts += int(
            response.metadata["repair_provider_attempts"]
        )
        self.corrective_calls += int(
            response.metadata["repair_corrective_calls"]
        )
        changes, reason = self._usable_changes(response, problems, summary)
        params = dict(summary["incumbent"]["params"])
        params.update(changes)
        return ProposalBatch(
            proposals=(
                Proposal(
                    params=params,
                    origin=self.name,
                    metadata={
                        "changes": dict(changes),
                        "reason": reason,
                        "repair_problems": list(problems),
                        "degraded": bool(problems),
                        "prompt": final_prompt,
                        "raw_output": response.raw_output,
                        "model": response.model,
                        "provider_metadata": dict(response.metadata),
                    },
                ),
            ),
            metadata={"fresh_provider_call": self.calls},
        )

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        self.observations.extend(outcome.observation for outcome in outcomes)

    def snapshot(self) -> Mapping[str, Any]:
        incumbent = select_incumbent(self.observations)
        return {
            "provider_calls": self.calls,
            "provider_attempts": self.provider_attempts,
            "corrective_calls": self.corrective_calls,
            "degraded_calls": self.degraded_calls,
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }


def create_arm(*, provider: FreshProposalProvider, **_: Any) -> LLMHillclimbArm:
    return LLMHillclimbArm(provider)
