"""Common contracts and deterministic runner for inner-tuner benchmarks.

The core owns comparison semantics: lower-is-better incumbent tracking,
post-checkpoint budget accounting, parameter projection, deduplication, and
artifacts.  Arms own only proposal policy through initialize/ask/tell/snapshot.
Objectives are injected, so CPU tests and later task subprocess adapters share
the same runner without importing task or GPU dependencies here.
"""

from __future__ import annotations

import importlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, runtime_checkable


DimensionKind = Literal["float", "int", "categorical"]
ObservationStatus = Literal["ok", "crash", "rejected"]


class PolicyContractError(RuntimeError):
    """An arm returned a batch that the shared runner cannot execute fairly."""


class ConfigInfeasibleError(RuntimeError):
    """One admitted configuration failed for a known config-specific reason."""


@dataclass(frozen=True)
class SearchDimension:
    name: str
    kind: DimensionKind
    low: float | int | None = None
    high: float | int | None = None
    choices: tuple[Any, ...] = ()
    log: bool = False
    step: int = 1

    def __post_init__(self) -> None:
        if self.kind in {"float", "int"}:
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError(f"invalid bounds for dimension {self.name!r}")
            if self.log and (self.low <= 0 or self.high <= 0):
                raise ValueError(f"log dimension {self.name!r} requires positive bounds")
            if self.kind == "int" and self.step < 1:
                raise ValueError(f"integer dimension {self.name!r} requires step >= 1")
        elif self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"categorical dimension {self.name!r} has no choices")
            if self.low is not None or self.high is not None:
                raise ValueError(f"categorical dimension {self.name!r} cannot have bounds")
        else:
            raise ValueError(f"unknown dimension kind {self.kind!r}")

    @classmethod
    def from_legacy(cls, name: str, spec: Sequence[Any]) -> "SearchDimension":
        if not spec:
            raise ValueError(f"empty search-space entry for {name!r}")
        kind = spec[0]
        if kind == "float" and len(spec) in {3, 4}:
            return cls(
                name=name,
                kind="float",
                low=float(spec[1]),
                high=float(spec[2]),
                log=len(spec) == 4 and spec[3] == "log",
            )
        if kind == "int" and len(spec) == 3:
            return cls(name=name, kind="int", low=int(spec[1]), high=int(spec[2]))
        if kind == "categorical" and len(spec) == 2:
            return cls(name=name, kind="categorical", choices=tuple(spec[1]))
        raise ValueError(f"invalid search-space entry for {name!r}: {spec!r}")

    def project(self, value: Any) -> Any:
        if self.kind == "categorical":
            if value not in self.choices:
                raise ValueError(
                    f"dimension {self.name!r} expects one of {list(self.choices)!r}"
                )
            return value
        if isinstance(value, bool):
            raise ValueError(f"dimension {self.name!r} does not accept booleans")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"dimension {self.name!r} expects a number") from None
        if not math.isfinite(numeric):
            raise ValueError(f"dimension {self.name!r} expects a finite number")
        assert self.low is not None and self.high is not None
        numeric = min(float(self.high), max(float(self.low), numeric))
        if self.kind == "float":
            return numeric
        low = int(self.low)
        projected = low + round((numeric - low) / self.step) * self.step
        return min(int(self.high), max(low, int(projected)))

    def normalized(self, value: Any) -> float:
        projected = self.project(value)
        if self.kind == "categorical":
            raise TypeError(
                f"categorical dimension {self.name!r} is nominal, not ordinal"
            )
        assert self.low is not None and self.high is not None
        if self.low == self.high:
            return 0.0
        if self.log:
            lo, hi, val = math.log(float(self.low)), math.log(float(self.high)), math.log(
                float(projected)
            )
            return (val - lo) / (hi - lo)
        return (float(projected) - float(self.low)) / (
            float(self.high) - float(self.low)
        )

    def describe(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name, "kind": self.kind}
        if self.kind == "categorical":
            result["choices"] = list(self.choices)
        else:
            result.update({"low": self.low, "high": self.high})
            if self.kind == "int":
                result["step"] = self.step
            if self.log:
                result["log"] = True
        return result


@dataclass(frozen=True)
class SearchSpace:
    dimensions: tuple[SearchDimension, ...]

    def __post_init__(self) -> None:
        names = [dimension.name for dimension in self.dimensions]
        if not names or len(names) != len(set(names)):
            raise ValueError("search space requires unique, non-empty dimensions")

    @classmethod
    def from_legacy(cls, space: Mapping[str, Sequence[Any]]) -> "SearchSpace":
        return cls(tuple(SearchDimension.from_legacy(name, spec) for name, spec in space.items()))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    @property
    def active_dimensions(self) -> tuple[SearchDimension, ...]:
        return tuple(
            dimension
            for dimension in self.dimensions
            if (
                len(dimension.choices) > 1
                if dimension.kind == "categorical"
                else dimension.low != dimension.high
            )
        )

    def project(self, params: Mapping[str, Any]) -> dict[str, Any]:
        missing = set(self.names) - set(params)
        extra = set(params) - set(self.names)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing {sorted(missing)}")
            if extra:
                parts.append(f"unknown {sorted(extra)}")
            raise ValueError("parameter keys do not match search space: " + ", ".join(parts))
        return {dimension.name: dimension.project(params[dimension.name]) for dimension in self.dimensions}

    def canonical(self, params: Mapping[str, Any]) -> str:
        return json.dumps(
            self.project(params),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def distance(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
        left, right = self.project(left), self.project(right)
        terms = []
        for dimension in self.dimensions:
            if dimension.kind == "categorical":
                terms.append(0.0 if left[dimension.name] == right[dimension.name] else 1.0)
            else:
                terms.append(
                    (dimension.normalized(left[dimension.name]) - dimension.normalized(right[dimension.name])) ** 2
                )
        return math.sqrt(sum(terms) / len(terms))

    def describe(self) -> list[dict[str, Any]]:
        return [dimension.describe() for dimension in self.dimensions]


@dataclass(frozen=True)
class Observation:
    observation_id: str
    params: Mapping[str, Any]
    score: float | None
    status: ObservationStatus
    origin: str
    eligible_incumbent: bool = True
    consumes_budget: bool = True
    failure: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkContext:
    checkpoint_id: str
    regime: Literal["first", "continuation"]
    space: SearchSpace
    observations: tuple[Observation, ...]
    budget: int
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget < 1:
            raise ValueError("benchmark budget must be positive")


@dataclass(frozen=True)
class Proposal:
    params: Mapping[str, Any]
    origin: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProposalBatch:
    proposals: tuple[Proposal, ...]
    atomic: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationOutcome:
    observation: Observation
    incumbent_before_id: str
    incumbent_after_id: str


@runtime_checkable
class Policy(Protocol):
    name: str

    def initialize(self, context: BenchmarkContext) -> None: ...

    def ask(self, remaining_budget: int) -> ProposalBatch: ...

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None: ...

    def snapshot(self) -> Mapping[str, Any]: ...


class Objective(Protocol):
    """Return preflight rejections; raise unknown failures unchanged.

    Evaluation may raise ``ConfigInfeasibleError`` for an admitted config whose
    failure is known to be config-specific and safe to record as ``+inf``.
    """

    def preflight(self, params: Mapping[str, Any]) -> str | None: ...

    def evaluate(self, params: Mapping[str, Any]) -> float: ...


@dataclass
class FunctionObjective:
    evaluate_fn: Callable[[Mapping[str, Any]], float]
    preflight_fn: Callable[[Mapping[str, Any]], str | None] | None = None

    def preflight(self, params: Mapping[str, Any]) -> str | None:
        return None if self.preflight_fn is None else self.preflight_fn(params)

    def evaluate(self, params: Mapping[str, Any]) -> float:
        return float(self.evaluate_fn(params))


def select_incumbent(observations: Sequence[Observation]) -> Observation:
    """Return the best finite, incumbent-eligible observation."""
    eligible = [
        observation
        for observation in observations
        if observation.status == "ok"
        and observation.eligible_incumbent
        and observation.score is not None
        and math.isfinite(observation.score)
    ]
    if not eligible:
        raise ValueError("checkpoint requires at least one finite eligible incumbent")
    return min(eligible, key=lambda observation: float(observation.score))


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _observation_record(observation: Observation) -> dict[str, Any]:
    return {
        "observation_id": observation.observation_id,
        "params": dict(observation.params),
        "score": _jsonable(observation.score),
        "status": observation.status,
        "origin": observation.origin,
        "eligible_incumbent": observation.eligible_incumbent,
        "consumes_budget": observation.consumes_budget,
        "failure": observation.failure,
        "metadata": dict(observation.metadata),
    }


class ArtifactStore:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.json"
        self.events_path = self.output_dir / "events.jsonl"
        self.result_path = self.output_dir / "result.json"
        self.failure_path = self.output_dir / "failure.json"
        self._event_count = 0
        self._initialized = False

    def initialize(self, manifest: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        occupied = [
            path.name
            for path in (
                self.manifest_path,
                self.events_path,
                self.result_path,
                self.failure_path,
            )
            if path.exists()
        ]
        if occupied:
            raise FileExistsError(
                f"benchmark output already contains artifacts: {', '.join(occupied)}"
            )
        self.manifest_path.write_text(
            json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._initialized = True

    def event(self, kind: str, **payload: Any) -> None:
        record = {"event_index": self._event_count, "kind": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False) + "\n"
            )
        self._event_count += 1

    def result(self, result: Mapping[str, Any]) -> None:
        self.result_path.write_text(
            json.dumps(_jsonable(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def failure(self, failure: Mapping[str, Any]) -> None:
        self.failure_path.write_text(
            json.dumps(_jsonable(failure), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


class BenchmarkRunner:
    """Run one checkpoint × arm × seed cell with strict admitted budget."""

    def __init__(
        self,
        context: BenchmarkContext,
        objective: Objective,
        output_dir: Path,
        *,
        max_proposal_batches: int | None = None,
    ):
        self.context = context
        self.objective = objective
        self.store = ArtifactStore(output_dir)
        self.max_proposal_batches = (
            max_proposal_batches
            if max_proposal_batches is not None
            else max(10, context.budget * 10)
        )
        if self.max_proposal_batches < context.budget:
            raise ValueError("max_proposal_batches cannot be smaller than evaluation budget")

    def run(self, policy: Policy) -> dict[str, Any]:
        try:
            return self._run(policy)
        except Exception as exc:
            if self.store._initialized:
                self.store.failure(
                    {
                        "schema_version": 1,
                        "checkpoint_id": self.context.checkpoint_id,
                        "arm": getattr(policy, "name", None),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            raise

    def _run(self, policy: Policy) -> dict[str, Any]:
        if not isinstance(policy, Policy):
            raise TypeError("policy does not implement the benchmark arm contract")
        observations = list(self.context.observations)
        incumbent = select_incumbent(observations)
        initial_incumbent = incumbent
        seen = {
            self.context.space.canonical(observation.params)
            for observation in observations
            if set(observation.params) == set(self.context.space.names)
        }
        self.store._event_count = 0
        self.store.initialize(
            {
                "schema_version": 1,
                "checkpoint_id": self.context.checkpoint_id,
                "regime": self.context.regime,
                "budget": self.context.budget,
                "seed": self.context.seed,
                "arm": policy.name,
                "space": self.context.space.describe(),
                "initial_observations": [
                    _observation_record(observation)
                    for observation in self.context.observations
                ],
                "initial_incumbent": _observation_record(initial_incumbent),
                "metadata": dict(self.context.metadata),
            }
        )
        policy.initialize(self.context)
        consumed = 0
        proposal_index = 0
        best_by_evaluation: list[float] = []
        proposal_batches = 0

        while consumed < self.context.budget:
            if proposal_batches >= self.max_proposal_batches:
                raise PolicyContractError(
                    f"arm {policy.name!r} exhausted {self.max_proposal_batches} "
                    "proposal batches before filling its evaluation budget"
                )
            proposal_batches += 1
            remaining = self.context.budget - consumed
            batch = policy.ask(remaining)
            if not isinstance(batch, ProposalBatch) or not batch.proposals:
                raise PolicyContractError(f"arm {policy.name!r} returned an empty proposal batch")
            if len(batch.proposals) > remaining:
                raise PolicyContractError(
                    f"arm {policy.name!r} returned {len(batch.proposals)} proposals "
                    f"with only {remaining} evaluations remaining"
                )
            self.store.event(
                "proposal_batch",
                remaining_budget=remaining,
                atomic=batch.atomic,
                metadata=dict(batch.metadata),
                proposals=[
                    {"origin": proposal.origin, "params": dict(proposal.params), "metadata": dict(proposal.metadata)}
                    for proposal in batch.proposals
                ],
            )

            prepared: list[
                tuple[Proposal, dict[str, Any] | None, str | None, str | None]
            ] = []
            batch_seen: set[str] = set()
            for proposal in batch.proposals:
                try:
                    params = self.context.space.project(proposal.params)
                    key = self.context.space.canonical(params)
                except (TypeError, ValueError) as exc:
                    params = None
                    failure = f"{type(exc).__name__}: {exc}"
                    rejection_kind = "contract"
                else:
                    if key in seen or key in batch_seen:
                        failure = "duplicate configuration"
                        rejection_kind = "duplicate"
                    else:
                        failure = self.objective.preflight(params)
                        rejection_kind = "preflight" if failure is not None else None
                    batch_seen.add(key)
                prepared.append((proposal, params, failure, rejection_kind))

            if batch.atomic and any(
                failure is not None for _, _, failure, _ in prepared
            ):
                direct_rejections = {
                    rejection_kind
                    for _, _, failure, rejection_kind in prepared
                    if failure is not None
                }
                peer_kind = (
                    "atomic_peer"
                    if direct_rejections <= {"preflight", "duplicate"}
                    else "contract"
                )
                prepared = [
                    (
                        proposal,
                        params,
                        failure or "atomic peer was rejected",
                        rejection_kind or peer_kind,
                    )
                    for proposal, params, failure, rejection_kind in prepared
                ]

            outcomes: list[EvaluationOutcome] = []
            consumed_this_batch = 0
            rejection_kinds: list[str] = []
            for proposal, params, failure, rejection_kind in prepared:
                proposal_index += 1
                observation_id = f"trial-{proposal_index:04d}"
                incumbent_before = incumbent
                if failure is not None or params is None:
                    assert rejection_kind is not None
                    rejection_kinds.append(rejection_kind)
                    if rejection_kind == "preflight" and params is not None:
                        seen.add(self.context.space.canonical(params))
                    observation = Observation(
                        observation_id=observation_id,
                        params=dict(proposal.params) if params is None else params,
                        score=None,
                        status="rejected",
                        origin=proposal.origin,
                        eligible_incumbent=False,
                        consumes_budget=False,
                        failure=failure,
                        metadata=dict(proposal.metadata),
                    )
                else:
                    seen.add(self.context.space.canonical(params))
                    try:
                        score = float(self.objective.evaluate(params))
                        if not math.isfinite(score):
                            raise ValueError("objective returned a non-finite score")
                        status: ObservationStatus = "ok"
                        failure = None
                    except ConfigInfeasibleError as exc:
                        score = math.inf
                        status = "crash"
                        failure = f"{type(exc).__name__}: {exc}"
                    observation = Observation(
                        observation_id=observation_id,
                        params=params,
                        score=score,
                        status=status,
                        origin=proposal.origin,
                        eligible_incumbent=True,
                        consumes_budget=True,
                        failure=failure,
                        metadata=dict(proposal.metadata),
                    )
                    consumed += 1
                    consumed_this_batch += 1
                    if status == "ok" and score < float(incumbent.score):
                        incumbent = observation
                    best_by_evaluation.append(float(incumbent.score))
                observations.append(observation)
                outcome = EvaluationOutcome(
                    observation=observation,
                    incumbent_before_id=incumbent_before.observation_id,
                    incumbent_after_id=incumbent.observation_id,
                )
                outcomes.append(outcome)
                self.store.event(
                    "evaluation",
                    admitted_index=consumed if observation.consumes_budget else None,
                    observation=_observation_record(observation),
                    incumbent_before_id=outcome.incumbent_before_id,
                    incumbent_after_id=outcome.incumbent_after_id,
                )

            policy.tell(tuple(outcomes))
            if consumed_this_batch == 0:
                if rejection_kinds and all(
                    kind in {"preflight", "duplicate", "atomic_peer"}
                    for kind in rejection_kinds
                ):
                    continue
                raise PolicyContractError(
                    f"arm {policy.name!r} made no budget-consuming progress in its batch"
                )

        initial_score = float(initial_incumbent.score)
        improvement_curve = [initial_score - score for score in best_by_evaluation]
        improvements = {
            str(prefix): initial_score - best_by_evaluation[prefix - 1]
            for prefix in (2, 4, 6, 10)
            if prefix <= len(best_by_evaluation)
        }
        first_improvement = next(
            (index for index, score in enumerate(best_by_evaluation, start=1) if score < initial_score),
            None,
        )
        result = {
            "schema_version": 1,
            "checkpoint_id": self.context.checkpoint_id,
            "regime": self.context.regime,
            "arm": policy.name,
            "seed": self.context.seed,
            "evaluations_consumed": consumed,
            "proposal_batches": proposal_batches,
            "initial_incumbent_score": initial_score,
            "final_incumbent_score": float(incumbent.score),
            "final_incumbent_id": incumbent.observation_id,
            "improvement": initial_score - float(incumbent.score),
            "beat_initial_incumbent": float(incumbent.score) < initial_score,
            "improvement_at": improvements,
            "improvement_auc": sum(improvement_curve),
            "first_improvement_evaluation": first_improvement,
            "best_by_evaluation": best_by_evaluation,
            "crash_count": sum(
                observation.status == "crash"
                for observation in observations[len(self.context.observations) :]
            ),
            "rejection_count": sum(
                observation.status == "rejected"
                for observation in observations[len(self.context.observations) :]
            ),
            "policy_snapshot": dict(policy.snapshot()),
        }
        self.store.result(result)
        return result


def load_arm(spec: str) -> Callable[..., Policy]:
    """Load ``package.module:factory`` without a shared arm registry."""
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("arm spec must be 'package.module:factory'")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"arm factory {spec!r} is not callable")
    return factory
