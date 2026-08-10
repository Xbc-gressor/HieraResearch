"""Faithful benchmark arm for the current progressive-tuning policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

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
from ._repair import complete_with_repair


REWARM_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["params", "reason"],
                "properties": {
                    "params": {"type": "object"},
                    "reason": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}


class ProposalBackend(Protocol):
    """Small ask/tell seam used by tests and the real Optuna backend."""

    def ask(self) -> Mapping[str, Any]: ...

    def tell(self, outcome: EvaluationOutcome) -> None: ...

    def snapshot(self) -> Mapping[str, Any]: ...


BackendFactory = Callable[[BenchmarkContext], ProposalBackend]


def _full_space_observations(context: BenchmarkContext) -> list[Observation]:
    names = set(context.space.names)
    return [
        observation
        for observation in context.observations
        if set(observation.params) == names
    ]


class _OptunaTPEBackend:
    """Production-equivalent multivariate/group TPE in ask/tell form."""

    def __init__(self, context: BenchmarkContext):
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - exercised in task env
            raise RuntimeError(
                "CurrentArm requires optuna in the task environment once its "
                "deferred/rewarm queue is exhausted"
            ) from exc

        self._optuna = optuna
        self._space = context.space
        self._pending = None
        self._asked = 0
        self._told = 0
        self._infeasible_attr = "hiera_infeasible"

        def constraints(trial):
            return tuple(
                float(value)
                for value in trial.user_attrs.get(self._infeasible_attr, (0.0,))
            )

        sampler = optuna.samplers.TPESampler(
            seed=context.seed,
            multivariate=True,
            group=True,
            n_startup_trials=8,
            constraints_func=constraints,
        )
        self._study = optuna.create_study(direction="minimize", sampler=sampler)
        self._distributions = self._build_distributions()
        self._penalty = self._initial_penalty(context.observations)
        self._inject_history(_full_space_observations(context))

    def _build_distributions(self) -> dict[str, Any]:
        optuna = self._optuna
        distributions: dict[str, Any] = {}
        for dimension in self._space.dimensions:
            if dimension.kind == "float":
                distributions[dimension.name] = optuna.distributions.FloatDistribution(
                    float(dimension.low), float(dimension.high), log=dimension.log
                )
            elif dimension.kind == "int":
                distributions[dimension.name] = optuna.distributions.IntDistribution(
                    int(dimension.low), int(dimension.high), step=dimension.step
                )
            else:
                distributions[dimension.name] = optuna.distributions.CategoricalDistribution(
                    list(dimension.choices)
                )
        return distributions

    @staticmethod
    def _initial_penalty(observations: Sequence[Observation]) -> float:
        scores = [
            float(observation.score)
            for observation in observations
            if observation.score is not None and math.isfinite(float(observation.score))
        ]
        worst = max(scores, default=0.0)
        return worst + max(1.0, abs(worst))

    def _inject_history(self, observations: Sequence[Observation]) -> None:
        for observation in observations:
            feasible = (
                observation.status == "ok"
                and observation.score is not None
                and math.isfinite(float(observation.score))
            )
            value = float(observation.score) if feasible else self._penalty
            self._study.add_trial(
                self._optuna.trial.create_trial(
                    params=dict(self._space.project(observation.params)),
                    distributions=self._distributions,
                    value=value,
                    user_attrs={self._infeasible_attr: [0.0 if feasible else 1.0]},
                    system_attrs={"constraints": (0.0 if feasible else 1.0,)},
                )
            )

    def ask(self) -> Mapping[str, Any]:
        if self._pending is not None:
            raise PolicyContractError("TPE ask called with an unresolved trial")
        trial = self._study.ask()
        params: dict[str, Any] = {}
        for dimension in self._space.dimensions:
            if dimension.kind == "float":
                params[dimension.name] = trial.suggest_float(
                    dimension.name,
                    float(dimension.low),
                    float(dimension.high),
                    log=dimension.log,
                )
            elif dimension.kind == "int":
                params[dimension.name] = trial.suggest_int(
                    dimension.name,
                    int(dimension.low),
                    int(dimension.high),
                    step=dimension.step,
                )
            else:
                params[dimension.name] = trial.suggest_categorical(
                    dimension.name, list(dimension.choices)
                )
        self._pending = trial
        self._asked += 1
        return params

    def tell(self, outcome: EvaluationOutcome) -> None:
        if self._pending is None:
            raise PolicyContractError("TPE tell called without a pending trial")
        observation = outcome.observation
        feasible = (
            observation.status == "ok"
            and observation.score is not None
            and math.isfinite(float(observation.score))
        )
        self._pending.set_user_attr(
            self._infeasible_attr, [0.0 if feasible else 1.0]
        )
        value = float(observation.score) if feasible else self._penalty
        self._study.tell(self._pending, value)
        self._pending = None
        self._told += 1

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "backend": "optuna_multivariate_group_tpe",
            "n_startup_trials": 8,
            "asked": self._asked,
            "told": self._told,
            "history_size": len(self._study.trials) - self._told,
        }


class CurrentArm:
    """Current baseline: queued configs first, then production-shaped TPE.

    First bouts consume ``deferred_configs`` before TPE. Continuation bouts make
    one fresh provider call for at most three rewarm proposals, evaluate those
    first, and then enter TPE. Both queues displace TPE draws inside the fixed
    benchmark budget.
    """

    name = "current"

    def __init__(
        self,
        *,
        deferred_configs: Sequence[Mapping[str, Any]] = (),
        provider: FreshProposalProvider | None = None,
        summary_builder: FocusedSummaryBuilder | None = None,
        max_rewarm_proposals: int = 3,
        tpe_backend_factory: BackendFactory | None = None,
        corrective_attempts: int = 3,
    ):
        if not 0 <= max_rewarm_proposals <= 3:
            raise ValueError("max_rewarm_proposals must be between 0 and 3")
        if corrective_attempts < 0:
            raise ValueError("corrective_attempts must be non-negative")
        self.deferred_configs = [dict(config) for config in deferred_configs]
        self.provider = provider
        self.summary_builder = summary_builder or FocusedSummaryBuilder()
        self.max_rewarm_proposals = max_rewarm_proposals
        self.tpe_backend_factory = tpe_backend_factory or _OptunaTPEBackend
        self.corrective_attempts = corrective_attempts
        self.context: BenchmarkContext | None = None
        self.observations: list[Observation] = []
        self._queue: list[Proposal] = []
        self._backend: ProposalBackend | None = None
        self._pending_backend = False
        self._rewarm_generated = False
        self._rewarm_degraded = False
        self._provider_calls = 0
        self._queue_evaluated = 0
        self._skipped_queue_configs = 0

    def initialize(self, context: BenchmarkContext) -> None:
        if context.regime == "continuation" and self.deferred_configs:
            raise PolicyContractError(
                "Current continuation bouts cannot consume first-bout deferred configs"
            )
        if context.regime == "continuation" and self.max_rewarm_proposals and self.provider is None:
            raise PolicyContractError(
                "Current continuation bouts require a fresh rewarm proposal provider"
            )
        self.context = context
        self.observations = list(context.observations)
        self._backend = None
        self._pending_backend = False
        self._rewarm_generated = False
        self._rewarm_degraded = False
        self._provider_calls = 0
        self._queue_evaluated = 0
        self._skipped_queue_configs = 0
        self._queue = []
        if context.regime == "first":
            self._queue = self._prepare_queue(
                self.deferred_configs, origin="current_deferred"
            )

    def _prepare_queue(
        self,
        configs: Sequence[Mapping[str, Any]],
        *,
        origin: str,
        reasons: Sequence[str] | None = None,
        shared_metadata: Mapping[str, Any] | None = None,
    ) -> list[Proposal]:
        assert self.context is not None
        names = set(self.context.space.names)
        seen = {
            self.context.space.canonical(observation.params)
            for observation in self.observations
            if set(observation.params) == names
        }
        queued: list[Proposal] = []
        for index, config in enumerate(configs):
            if set(config) != names:
                self._skipped_queue_configs += 1
                continue
            try:
                projected = self.context.space.project(config)
                key = self.context.space.canonical(projected)
            except ValueError:
                self._skipped_queue_configs += 1
                continue
            if key in seen:
                self._skipped_queue_configs += 1
                continue
            seen.add(key)
            metadata = {"queue_index": index, **dict(shared_metadata or {})}
            if reasons is not None:
                metadata["reason"] = reasons[index]
            queued.append(Proposal(projected, origin=origin, metadata=metadata))
        return queued

    def _rewarm_problems(self, output: Mapping[str, Any]) -> list[str]:
        rows = output.get("proposals")
        if not isinstance(rows, list):
            return ["proposals must be a list of configuration objects"]
        return []

    def _generate_rewarm(self, remaining_budget: int) -> None:
        assert self.context is not None and self.provider is not None
        summary = self.summary_builder.build(
            self.context, self.observations, remaining_budget=remaining_budget
        )
        prompt = (
            f"Propose up to {self.max_rewarm_proposals} promising continuation "
            "configurations. Scores are lower-is-better. Prefer the incumbent's "
            "best region unless the evidence says it is exhausted. Return JSON "
            "matching the supplied schema.\n\n"
            + json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
        )
        response, problems, final_prompt = complete_with_repair(
            self.provider,
            prompt,
            REWARM_OUTPUT_SCHEMA,
            validate=self._rewarm_problems,
            corrective_attempts=self.corrective_attempts,
        )
        self._provider_calls += 1
        if problems:
            # Degraded: skip the rewarm queue and fall through to TPE.
            self._rewarm_degraded = True
            self._queue = []
            return
        rows = response.output.get("proposals")
        rows = rows[: self.max_rewarm_proposals]
        configs: list[Mapping[str, Any]] = []
        reasons: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                self._skipped_queue_configs += 1
                continue
            params, reason = row.get("params"), row.get("reason")
            if not isinstance(params, Mapping) or not isinstance(reason, str) or not reason.strip():
                self._skipped_queue_configs += 1
                continue
            configs.append(params)
            reasons.append(reason)
        self._queue = self._prepare_queue(
            configs,
            origin="current_rewarm",
            reasons=reasons,
            shared_metadata={
                "prompt": final_prompt,
                "prompt_hash": hashlib.sha256(final_prompt.encode("utf-8")).hexdigest(),
                "raw_output": response.raw_output,
                "model": response.model,
                "provider_metadata": dict(response.metadata),
                "repair_problems": list(problems),
                "degraded": bool(problems),
            },
        )

    def _ensure_backend(self) -> ProposalBackend:
        assert self.context is not None
        if self._backend is None:
            backend_context = BenchmarkContext(
                checkpoint_id=self.context.checkpoint_id,
                regime=self.context.regime,
                space=self.context.space,
                observations=tuple(self.observations),
                budget=self.context.budget,
                seed=self.context.seed,
                metadata=self.context.metadata,
            )
            self._backend = self.tpe_backend_factory(backend_context)
        return self._backend

    def ask(self, remaining_budget: int) -> ProposalBatch:
        if self.context is None:
            raise PolicyContractError("Current arm was not initialized")
        if self._pending_backend:
            raise PolicyContractError("Current arm received ask before TPE tell")
        if self.context.regime == "continuation" and not self._rewarm_generated:
            self._rewarm_generated = True
            if self.max_rewarm_proposals:
                self._generate_rewarm(remaining_budget)
        if self._queue:
            proposal = self._queue.pop(0)
            return ProposalBatch(
                proposals=(proposal,),
                metadata={"phase": proposal.origin, "queue_remaining": len(self._queue)},
            )
        backend = self._ensure_backend()
        self._pending_backend = True
        return ProposalBatch(
            proposals=(
                Proposal(
                    params=dict(backend.ask()),
                    origin="current_tpe",
                    metadata={
                        "sampler": "optuna_tpe",
                        "multivariate": True,
                        "group": True,
                        "n_startup_trials": 8,
                    },
                ),
            ),
            metadata={"phase": "current_tpe"},
        )

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        if len(outcomes) != 1:
            raise PolicyContractError("Current arm expects one outcome per ask")
        self.observations.append(outcomes[0].observation)
        if self._pending_backend:
            assert self._backend is not None
            self._backend.tell(outcomes[0])
            self._pending_backend = False
        elif outcomes[0].observation.consumes_budget:
            self._queue_evaluated += 1

    def snapshot(self) -> Mapping[str, Any]:
        incumbent = select_incumbent(self.observations)
        return {
            "regime": None if self.context is None else self.context.regime,
            "provider_calls": self._provider_calls,
            "queue_evaluated": self._queue_evaluated,
            "queue_remaining": len(self._queue),
            "skipped_queue_configs": self._skipped_queue_configs,
            "rewarm_degraded": self._rewarm_degraded,
            "tpe": None if self._backend is None else dict(self._backend.snapshot()),
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }


def create_arm(**dependencies: Any) -> CurrentArm:
    allowed = {
        "deferred_configs",
        "provider",
        "summary_builder",
        "max_rewarm_proposals",
        "tpe_backend_factory",
        "corrective_attempts",
    }
    return CurrentArm(**{key: value for key, value in dependencies.items() if key in allowed})
