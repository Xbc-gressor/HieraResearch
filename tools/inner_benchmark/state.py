"""Authoritative per-cell state for the inner-tuner benchmark (PLAN §3.2).

The LLM session is never the source of truth — the runner owns this state.
Scores are always lower-is-better. Crash rows and unexecuted
(preflight-rejected) proposals carry score=None and are never silently
coerced; +inf is the scalar representation of a crash for consumers that
need one, but this layer keeps None. Executed rows (ok/crash) consume one
unit of budget each; preflight-rejected proposals were never executed and
consume none.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tuners"))

from _common import params_identity as _default_identity  # noqa: E402

OK = "ok"
CRASH = "crash"
PREFLIGHT_REJECTED = "preflight_rejected"
STATUSES = (OK, CRASH, PREFLIGHT_REJECTED)


@dataclass
class Trial:
    """One recorded row: an executed trial or an unexecuted proposal."""

    config: dict
    score: float | None  # finite or +inf when executed ok; None on crash/rejected
    status: str  # "ok" | "crash" | "preflight_rejected"
    source: str | None = None
    rationale: str | None = None


@dataclass
class CellState:
    """Authoritative state of one (checkpoint, arm, seed) cell, owned by the runner.

    ``identity_fn`` drives dedupe in finite_unique_history; the runner should
    pass CandidateContract.params_identity (production cast + identity). The
    default is production params_identity without cast, which distinguishes
    e.g. 3 from 3.0.
    """

    incumbent_config: dict
    incumbent_score: float
    budget_remaining: int
    trials: list[Trial] = field(default_factory=list)
    arm_state: dict = field(default_factory=dict)
    identity_fn: Callable[[dict], str] = _default_identity

    def is_strict_improvement(self, score: float) -> bool:
        """Only a finite evaluated score strictly below the incumbent counts."""
        return math.isfinite(score) and score < self.incumbent_score

    def record_outcome(self, config: dict, *, status: str, score=None,
                       source: str | None = None,
                       rationale: str | None = None) -> Trial:
        """Record one executed trial or unexecuted proposal.

        Executed rows (ok/crash) consume one unit of budget; preflight-rejected
        proposals consume none. The incumbent updates only on strict
        improvement by a finite evaluated score. No silent coercion: ok rows
        require a real (non-NaN) score, crash/rejected rows require score=None.
        """
        if status not in STATUSES:
            raise ValueError(f"unknown trial status {status!r}; expected one of {STATUSES}")
        if status == OK:
            if not isinstance(score, (int, float)) or math.isnan(float(score)):
                raise ValueError(
                    f"executed trials need a real score, got {score!r}; "
                    "crash rows use status='crash' with score=None"
                )
            score = float(score)
        elif score is not None:
            raise ValueError(
                f"{status} rows carry score=None; refusing to silently coerce {score!r}"
            )
        trial = Trial(config=dict(config), score=score, status=status,
                      source=source, rationale=rationale)
        self.trials.append(trial)
        if status != PREFLIGHT_REJECTED:
            self.budget_remaining -= 1
        if status == OK and self.is_strict_improvement(score):
            self.incumbent_config = dict(config)
            self.incumbent_score = score
        return trial

    def best_so_far(self) -> tuple[dict, float]:
        """(incumbent config, incumbent score)."""
        return dict(self.incumbent_config), self.incumbent_score

    def finite_unique_history(self) -> list[tuple[dict, float]]:
        """(config, finite score) pairs in execution order — the fittable
        history for numerical rankers.

        Crash rows, preflight-rejected proposals, and non-finite scores are
        excluded; duplicate configs (by identity_fn) keep their first
        occurrence.
        """
        seen: set[str] = set()
        history: list[tuple[dict, float]] = []
        for trial in self.trials:
            if trial.status != OK or not math.isfinite(trial.score):
                continue
            identity = self.identity_fn(trial.config)
            if identity in seen:
                continue
            seen.add(identity)
            history.append((dict(trial.config), trial.score))
        return history
