"""Fresh-session LLM hillclimb example arm."""

from __future__ import annotations

import json
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
from ..providers import FreshProposalProvider
from ..summary import FocusedSummaryBuilder


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
    ):
        self.provider = provider
        self.summary_builder = summary_builder or FocusedSummaryBuilder()
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self.calls = 0

    def initialize(self, context: BenchmarkContext) -> None:
        self.context = context
        self.observations = list(context.observations)
        self.calls = 0

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
        response = self.provider.complete(prompt, output_schema=OUTPUT_SCHEMA)
        output = response.output
        changes = output.get("changes")
        reason = output.get("reason")
        if not isinstance(changes, Mapping) or not changes:
            raise PolicyContractError("hillclimb provider must return non-empty changes")
        unknown = set(changes) - set(self.context.space.names)
        if unknown:
            raise PolicyContractError(f"hillclimb provider changed unknown dimensions: {sorted(unknown)}")
        if not isinstance(reason, str) or not reason.strip():
            raise PolicyContractError("hillclimb provider must return a non-empty reason")
        params = dict(summary["incumbent"]["params"])
        params.update(changes)
        self.calls += 1
        return ProposalBatch(
            proposals=(
                Proposal(
                    params=params,
                    origin=self.name,
                    metadata={
                        "changes": dict(changes),
                        "reason": reason,
                        "prompt": prompt,
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
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }


def create_arm(*, provider: FreshProposalProvider, **_: Any) -> LLMHillclimbArm:
    return LLMHillclimbArm(provider)
