from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TaskSpec:
    scenario: str
    instance: str
    target: str
    learner_family: str
    n_samples: int
    n_numeric_features: int
    n_categorical_features: int
    n_classes: int
    task_type: str = "multiclass classification"

    @property
    def key(self) -> str:
        return f"{self.scenario}__{self.instance}"

    def as_dict(self) -> dict:
        return asdict(self)

    def public_metadata(self) -> dict:
        """Dataset facts safe to expose to the LLM (no name or identifier)."""
        return {
            "task_type": self.task_type,
            "n_samples": self.n_samples,
            "n_numeric_features": self.n_numeric_features,
            "n_categorical_features": self.n_categorical_features,
            "n_classes": self.n_classes,
        }


# OpenML qualities were frozen on 2026-08-23. NumberOfFeatures includes the
# symbolic target, so categorical predictor counts subtract that one target.
PILOT_TASKS = (
    TaskSpec("lcbench", "167168", "val_accuracy", "feed-forward neural network", 846, 18, 0, 4),
    TaskSpec("lcbench", "189906", "val_accuracy", "feed-forward neural network", 2310, 19, 0, 7),
    TaskSpec("rbv2_glmnet", "375", "acc", "elastic-net generalized linear model", 9961, 14, 0, 9),
    TaskSpec("rbv2_rpart", "14", "acc", "decision tree", 2000, 76, 0, 10),
    TaskSpec("rbv2_ranger", "16", "acc", "random forest", 2000, 64, 0, 10),
    TaskSpec("rbv2_ranger", "42", "acc", "random forest", 683, 0, 35, 19),
    TaskSpec("rbv2_xgboost", "12", "acc", "gradient-boosted trees", 2000, 216, 0, 10),
    TaskSpec("rbv2_xgboost", "40499", "acc", "gradient-boosted trees", 5500, 40, 0, 11),
)

_BY_KEY = {task.key: task for task in PILOT_TASKS}
SMOKE_TASKS = (_BY_KEY["rbv2_glmnet__375"], _BY_KEY["rbv2_xgboost__40499"])


def task_from_dict(data: dict) -> TaskSpec:
    return TaskSpec(**data)
