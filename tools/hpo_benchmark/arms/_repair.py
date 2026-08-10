"""Bounded corrective re-ask for fresh LLM proposal calls.

Mirrors the driver's verify-repair loop (driver/session.py): when a fresh
provider response violates the arm's output contract, re-ask in the same call
series with the concrete problems appended to the prompt, up to a bound.  Arms
own their validation function and their graceful-degradation salvage; this
module owns only the re-ask loop and the space-level random fallback configs
shared by several arms.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from typing import Any

from ..core import SearchSpace
from ..providers import FreshProposalProvider, ProviderResponse


ValidateFn = Callable[[Mapping[str, Any]], list[str]]


def complete_with_repair(
    provider: FreshProposalProvider,
    prompt: str,
    output_schema: Mapping[str, Any],
    *,
    validate: ValidateFn,
    corrective_attempts: int = 3,
) -> tuple[ProviderResponse, list[str], str]:
    """Call ``provider.complete`` and repair contract violations.

    ``validate(output)`` returns a list of concrete problems (empty means the
    response is acceptable).  Each rejected attempt appends its problems to the
    prompt and re-asks with a fresh call, mirroring the driver's corrective
    follow-up.  Returns ``(response, problems, final_prompt)`` where ``problems``
    are those of the last validation (empty on success) and ``final_prompt`` is
    the prompt of the last call.  The caller decides how to salvage a response
    that still fails.
    """
    if corrective_attempts < 0:
        raise ValueError("corrective_attempts must be non-negative")
    response = provider.complete(prompt, output_schema=output_schema)
    problems = validate(response.output)
    for _ in range(corrective_attempts):
        if not problems:
            break
        items = "\n".join(f"- {problem}" for problem in problems)
        prompt = (
            prompt
            + "\n\nYour previous response was rejected because:\n"
            + items
            + "\nReturn one corrected response that fixes every problem."
        )
        response = provider.complete(prompt, output_schema=output_schema)
        problems = validate(response.output)
    return response, problems, prompt


def random_config(space: SearchSpace, rng: random.Random) -> dict[str, Any]:
    """Draw one uniformly random full-space configuration, projected."""
    params: dict[str, Any] = {}
    for dimension in space.dimensions:
        if dimension.kind == "categorical":
            params[dimension.name] = rng.choice(dimension.choices)
        else:
            assert dimension.low is not None and dimension.high is not None
            low, high = float(dimension.low), float(dimension.high)
            if dimension.log:
                low, high = math.log(low), math.log(high)
                params[dimension.name] = math.exp(low + rng.random() * (high - low))
            else:
                params[dimension.name] = low + rng.random() * (high - low)
    return space.project(params)
