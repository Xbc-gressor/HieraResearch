"""Pure SMAC random-forest/EI benchmark arm."""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from ..core import (
    BenchmarkContext,
    EvaluationOutcome,
    PolicyContractError,
    Proposal,
    ProposalBatch,
    select_incumbent,
)


class SMACBackend(Protocol):
    def ask(self) -> Mapping[str, Any]: ...

    def tell(self, outcome: EvaluationOutcome) -> None: ...

    def snapshot(self) -> Mapping[str, Any]: ...


BackendFactory = Callable[[BenchmarkContext], SMACBackend]


class _RealSMACBackend:
    """Lazy SMAC adapter so the shared CPU core has no hard dependency."""

    def __init__(self, context: BenchmarkContext):
        try:
            from ConfigSpace import (
                Categorical,
                Configuration,
                ConfigurationSpace,
                Constant,
                Float,
                Integer,
            )
            from smac import HyperparameterOptimizationFacade, Scenario
            from smac.runhistory.dataclasses import TrialInfo, TrialValue
            from smac.runhistory.enumerations import StatusType
        except ImportError as exc:  # pragma: no cover - exercised in task env
            raise RuntimeError(
                "PureSMACArm requires smac and ConfigSpace in the task environment"
            ) from exc

        self._Configuration = Configuration
        self._TrialInfo = TrialInfo
        self._TrialValue = TrialValue
        self._StatusType = StatusType
        self._seed = context.seed
        self._pending = None
        self._asked = 0
        self._told = 0
        self._history_injected = 0
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="hieraresearch-pure-smac-"
        )

        configspace = ConfigurationSpace(seed=context.seed)
        for dimension in context.space.dimensions:
            if dimension.kind == "categorical" and len(dimension.choices) == 1:
                hyperparameter = Constant(
                    dimension.name, dimension.choices[0]
                )
            elif (
                dimension.kind in {"float", "int"}
                and dimension.low == dimension.high
            ):
                hyperparameter = Constant(dimension.name, dimension.low)
            elif dimension.kind == "float":
                hyperparameter = Float(
                    dimension.name,
                    (float(dimension.low), float(dimension.high)),
                    log=dimension.log,
                )
            elif dimension.kind == "int":
                hyperparameter = Integer(
                    dimension.name,
                    (int(dimension.low), int(dimension.high)),
                    log=dimension.log,
                )
            else:
                hyperparameter = Categorical(
                    dimension.name, list(dimension.choices)
                )
            configspace.add(hyperparameter)
        self._configspace = configspace

        full_history = [
            observation
            for observation in context.observations
            if set(observation.params) == set(context.space.names)
        ]
        finite_scores = [
            float(observation.score)
            for observation in full_history
            if observation.score is not None and math.isfinite(float(observation.score))
        ]
        worst = max(finite_scores, default=0.0)
        self._penalty = worst + max(1.0, abs(worst))
        scenario = Scenario(
            configspace,
            n_trials=max(1, len(full_history) + context.budget * 10),
            deterministic=True,
            output_directory=Path(self._temporary_directory.name),
            seed=context.seed,
        )
        # The checkpoint observations are SMAC's startup data. Leaving the
        # facade's default Sobol design enabled would spend this benchmark's
        # entire ten-evaluation budget before RF/EI is ever consulted.
        initial_design = HyperparameterOptimizationFacade.get_initial_design(
            scenario, n_configs=0
        )
        self._smac = HyperparameterOptimizationFacade(
            scenario,
            lambda config, seed=0: 0.0,
            initial_design=initial_design,
            overwrite=True,
            logging_level=False,
        )
        self._inject_history(context, full_history)

    def _inject_history(self, context, observations) -> None:
        seen: set[str] = set()
        for observation in observations:
            params = context.space.project(observation.params)
            key = context.space.canonical(params)
            if key in seen:
                continue
            seen.add(key)
            feasible = (
                observation.status == "ok"
                and observation.score is not None
                and math.isfinite(float(observation.score))
            )
            status = self._StatusType.SUCCESS if feasible else self._StatusType.CRASHED
            value = float(observation.score) if feasible else self._penalty
            self._smac.tell(
                self._TrialInfo(
                    self._Configuration(self._configspace, values=params),
                    seed=self._seed,
                ),
                self._TrialValue(cost=value, status=status),
            )
            self._history_injected += 1

    def ask(self) -> Mapping[str, Any]:
        if self._pending is not None:
            raise PolicyContractError("SMAC ask called with an unresolved trial")
        self._pending = self._smac.ask()
        self._asked += 1
        return dict(self._pending.config)

    def tell(self, outcome: EvaluationOutcome) -> None:
        if self._pending is None:
            raise PolicyContractError("SMAC tell called without a pending trial")
        observation = outcome.observation
        feasible = (
            observation.status == "ok"
            and observation.score is not None
            and math.isfinite(float(observation.score))
        )
        status = self._StatusType.SUCCESS if feasible else self._StatusType.CRASHED
        value = float(observation.score) if feasible else self._penalty
        self._smac.tell(
            self._pending,
            self._TrialValue(cost=value, status=status),
        )
        self._pending = None
        self._told += 1

    def snapshot(self) -> Mapping[str, Any]:
        return {
            "backend": "smac",
            "surrogate": "random_forest",
            "acquisition": "expected_improvement",
            "initial_design_configs": 0,
            "history_injected": self._history_injected,
            "asked": self._asked,
            "told": self._told,
        }


class PureSMACArm:
    """Let SMAC select every post-checkpoint configuration without an LLM."""

    name = "pure_smac"

    def __init__(self, *, backend_factory: BackendFactory | None = None):
        self.backend_factory = backend_factory or _RealSMACBackend
        self.context: BenchmarkContext | None = None
        self.backend: SMACBackend | None = None
        self.observations = []
        self._pending = False

    def initialize(self, context: BenchmarkContext) -> None:
        if not context.space.active_dimensions:
            raise PolicyContractError("Pure SMAC requires a non-fixed dimension")
        self.context = context
        self.observations = list(context.observations)
        self.backend = self.backend_factory(context)
        self._pending = False

    def ask(self, remaining_budget: int) -> ProposalBatch:
        if self.backend is None:
            raise PolicyContractError("Pure SMAC arm was not initialized")
        if self._pending:
            raise PolicyContractError("Pure SMAC received ask before tell")
        if remaining_budget < 1:
            raise PolicyContractError("Pure SMAC requires remaining budget")
        self._pending = True
        return ProposalBatch(
            proposals=(
                Proposal(
                    params=dict(self.backend.ask()),
                    origin=self.name,
                    metadata={
                        "backend": "smac",
                        "surrogate": "random_forest",
                        "acquisition": "expected_improvement",
                    },
                ),
            ),
            metadata={
                "backend": "smac",
                "surrogate": "random_forest",
                "acquisition": "expected_improvement",
            },
        )

    def tell(self, outcomes: tuple[EvaluationOutcome, ...]) -> None:
        if self.backend is None or not self._pending:
            raise PolicyContractError("Pure SMAC received tell without a pending trial")
        if len(outcomes) != 1:
            raise PolicyContractError("Pure SMAC expects one outcome per ask")
        self.backend.tell(outcomes[0])
        self.observations.append(outcomes[0].observation)
        self._pending = False

    def snapshot(self) -> Mapping[str, Any]:
        if self.backend is None:
            raise PolicyContractError("Pure SMAC arm was not initialized")
        incumbent = select_incumbent(self.observations)
        return {
            **dict(self.backend.snapshot()),
            "incumbent_id": incumbent.observation_id,
            "incumbent_score": incumbent.score,
        }


def create_arm(**dependencies: Any) -> PureSMACArm:
    return PureSMACArm(
        **{
            key: dependencies[key]
            for key in ("backend_factory",)
            if key in dependencies
        }
    )
