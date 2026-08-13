"""Exact mechanical state of one scheduler decision (design §3, §4.1, §8).

The state is built from authoritative run artifacts — the ledger, the
strict attempt log, and each candidate's tuning report — never from a role
receipt. It carries two kinds of field, and the distinction is the whole
point of the module:

* **mechanical** fields decide which actions exist and how budget and the
  candidate pool change once one runs. They are cheap to observe exactly,
  so they are modeled exactly;
* **diagnostic** fields (budget progress, headroom, generation kind, stall
  regime, DAG summary, ...) are recorded for replay and calibration but do
  not condition any predictive model in v1. Recording a variable and
  conditioning on it are separate decisions; §4.3 keeps them separate until
  evidence justifies a promotion.

The score transition is the design's equation (2)/(4):

    m_i' = m_i - max(D_i, 0)          candidate best-so-far
    Δ_g  = (D_i - h_i)_+              immediate raw global improvement

with `D` the raw bout improvement and `h_i = m_i - g` the headroom. No
noise adjustment enters here: the scheduler optimizes the raw terminal best
the run actually records.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .contract import (
    CandidateView,
    ResourceContract,
    defer_available,
    eligible_candidates,
    ineligibility_reason,
)


def is_finite_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class SchedulerState:
    """One decision point's exact mechanical state."""

    global_best: float
    remaining_budget: int
    candidates: tuple[CandidateView, ...]
    contract: ResourceContract = field(default_factory=ResourceContract)
    #: Recorded, never conditioned on (design §4.3).
    diagnostics: dict = field(default_factory=dict)

    def headroom(self, candidate: CandidateView) -> float:
        """h_i = m_i - g, how far this candidate is from the global best."""
        return candidate.best_score - self.global_best

    def eligible(self) -> list[CandidateView]:
        return eligible_candidates(
            self.candidates, self.remaining_budget, self.contract
        )

    def defer_available(self) -> bool:
        return defer_available(self.remaining_budget, self.contract)

    def terminal(self) -> bool:
        """No admissible full-bout TUNE and no executable DEFER (§7)."""
        return not self.eligible() and not self.defer_available()

    def actions(self) -> list[tuple[str, str | None]]:
        """The action set A(S): exact TUNE targets plus DEFER when available."""
        actions: list[tuple[str, str | None]] = [
            ("TUNE", candidate.run_id) for candidate in self.eligible()
        ]
        if self.defer_available():
            actions.append(("DEFER", None))
        return actions

    # -- transitions ------------------------------------------------------
    #
    # Both the rollout simulator and the realized-outcome recorder go
    # through these, so a simulated trajectory and a real one apply the
    # same arithmetic to the same fields.

    def apply_bout(
        self,
        run_id: str,
        gain: float,
        *,
        cost: int | None = None,
    ) -> "SchedulerState":
        """Charge one bout on `run_id` and apply equation (2).

        `cost` defaults to the full bout. A terminated bout that consumed
        fewer evaluations passes its actual consumption so simulated and
        realized budget accounting agree (§4.2).
        """
        charged = self.contract.bout_trials if cost is None else int(cost)
        if charged < 0:
            raise ValueError("bout cost must be non-negative")
        updated = []
        for candidate in self.candidates:
            if candidate.run_id != run_id:
                updated.append(candidate)
                continue
            improvement = max(float(gain), 0.0)
            updated.append(
                replace(
                    candidate,
                    best_score=candidate.best_score - improvement,
                    bouts_used=candidate.bouts_used + 1,
                    previous_gain=float(gain),
                    # A first bout's deferred-warm supply is consumed
                    # inside that bout; later bouts have none.
                    deferred_warm_backlog=0,
                )
            )
        return replace(
            self,
            candidates=tuple(updated),
            remaining_budget=max(0, self.remaining_budget - charged),
            global_best=min(
                (candidate.best_score for candidate in updated),
                default=self.global_best,
            ),
        )

    def admit_arrivals(
        self,
        arrivals: Iterable[tuple[str, float | None, int]],
    ) -> "SchedulerState":
        """Add admitted candidates from one generation round.

        Each arrival is `(run_id, warm_score, cost)`. The caller has already
        truncated the episode to the prefix the remaining budget admits
        (§5); this method only applies the resulting state change.

        A `None` warm score is an arrival that produced no usable
        candidate. It charges its cost and adds nothing — a failed
        generation is a real way to spend budget, and modeling it as a
        candidate at the global best would make DEFER look strictly
        productive.
        """
        candidates = list(self.candidates)
        remaining = self.remaining_budget
        for run_id, warm_score, cost in arrivals:
            remaining = max(0, remaining - int(cost))
            if warm_score is None:
                continue
            candidates.append(
                CandidateView(
                    run_id=run_id,
                    best_score=float(warm_score),
                    bouts_used=0,
                )
            )
        return replace(
            self,
            candidates=tuple(candidates),
            remaining_budget=remaining,
            global_best=min(
                (candidate.best_score for candidate in candidates),
                default=self.global_best,
            ),
        )

    # -- artifact ---------------------------------------------------------

    def snapshot(self) -> dict:
        """The immutable compact state artifact of design §8.

        A mutable ledger revision is not a historical snapshot; this dict is
        what a replay reads to reconstruct the decision exactly.
        """
        return {
            "schema_version": 1,
            "kind": "scheduler_state_snapshot",
            "global_best": self.global_best,
            "remaining_budget": self.remaining_budget,
            "contract": {
                "bout_trials": self.contract.bout_trials,
                "max_bouts": self.contract.max_bouts,
                "k_eval": self.contract.k_eval,
            },
            "candidates": [
                {
                    "run_id": candidate.run_id,
                    "best_score": candidate.best_score,
                    "bouts_used": candidate.bouts_used,
                    "previous_gain": candidate.previous_gain,
                    "headroom": self.headroom(candidate),
                    "deferred_warm_backlog": candidate.deferred_warm_backlog,
                    "eligible": ineligibility_reason(
                        candidate, self.remaining_budget, self.contract
                    )
                    is None,
                    "ineligible_reason": ineligibility_reason(
                        candidate, self.remaining_budget, self.contract
                    ),
                }
                for candidate in self.candidates
            ],
            "defer_available": self.defer_available(),
            "diagnostics": self.diagnostics,
        }


def state_from_snapshot(snapshot: dict) -> SchedulerState:
    """Rebuild a state from its immutable artifact (replay path)."""
    contract_fields = snapshot.get("contract", {})
    contract = ResourceContract(
        bout_trials=int(contract_fields.get("bout_trials", 10)),
        max_bouts=int(contract_fields.get("max_bouts", 4)),
        k_eval=int(contract_fields.get("k_eval", 2)),
    )
    candidates = tuple(
        CandidateView(
            run_id=str(row["run_id"]),
            best_score=float(row["best_score"]),
            bouts_used=int(row.get("bouts_used", 0)),
            previous_gain=row.get("previous_gain"),
            deferred_warm_backlog=int(row.get("deferred_warm_backlog", 0)),
            # `eligible` in a snapshot is the derived verdict; the causes
            # that are not budget-dependent are restored here so the
            # predicate recomputes the same answer.
            has_unresolved_descendant=(
                row.get("ineligible_reason") == "unresolved primary descendant"
            ),
            crashed=row.get("ineligible_reason") == "crashed",
        )
        for row in snapshot.get("candidates", [])
    )
    return SchedulerState(
        global_best=float(snapshot["global_best"]),
        remaining_budget=int(snapshot["remaining_budget"]),
        candidates=candidates,
        contract=contract,
        diagnostics=snapshot.get("diagnostics", {}),
    )


# =============================================================================
# Building the state from authoritative run artifacts
# =============================================================================


def _unresolved_descendant_ids(ledger: dict) -> set[str]:
    """Run ids whose primary descendants are not yet bound.

    Tuning such a parent would race a child that inherited its parameters,
    so it is mechanically ineligible. The shared lineage predicate is the
    authority; a local re-derivation could disagree with the ledger's own
    binding rules.
    """
    try:
        from semantic_evidence import unbound_primary_descendants
    except ImportError:  # pragma: no cover - tools/ not on the path
        return set()
    blocked = set()
    for record in ledger.get("records", []):
        run_id = str(record.get("run_id"))
        if unbound_primary_descendants(ledger, run_id):
            blocked.add(run_id)
    return blocked


def candidate_score(record: dict) -> float | None:
    """The candidate's current best: its tuned score, else its warm score.

    `final_best_score` is only comparable to `best_warm_score` after a bout
    has run; before that the warm score *is* the candidate's best-so-far.
    Ranking one against the other across candidates is exactly the
    like-for-like violation AGENTS.md forbids — but `m_i` here is a
    per-candidate state variable feeding a transition, not a cross-candidate
    ranking key, and the global best it feeds is the run's own raw best.
    """
    if record.get("tune") and is_finite_score(record.get("final_best_score")):
        return float(record["final_best_score"])
    if is_finite_score(record.get("best_warm_score")):
        return float(record["best_warm_score"])
    return None


def build_state(
    ledger: dict,
    *,
    remaining_budget: int,
    contract: ResourceContract | None = None,
    deferred_backlog: dict[str, int] | None = None,
    previous_gains: dict[str, float] | None = None,
    diagnostics: dict | None = None,
) -> SchedulerState:
    """Assemble the exact mechanical state from the ledger and budget."""
    contract = contract or ResourceContract()
    blocked = _unresolved_descendant_ids(ledger)
    backlog = deferred_backlog or {}
    gains = previous_gains or {}
    candidates = []
    for record in ledger.get("records", []):
        run_id = str(record.get("run_id"))
        score = candidate_score(record)
        if score is None:
            # No finite observation yet: not a scheduler-visible candidate.
            continue
        candidates.append(
            CandidateView(
                run_id=run_id,
                best_score=score,
                bouts_used=int(record.get("tuning_bouts") or 0),
                previous_gain=gains.get(run_id),
                has_unresolved_descendant=run_id in blocked,
                crashed=record.get("status") == "crash",
                deferred_warm_backlog=int(backlog.get(run_id, 0)),
            )
        )
    scores = [
        candidate.best_score for candidate in candidates if not candidate.crashed
    ]
    return SchedulerState(
        global_best=min(scores) if scores else math.inf,
        remaining_budget=int(remaining_budget),
        candidates=tuple(candidates),
        contract=contract,
        diagnostics=diagnostics or {},
    )


def previous_gains(run_dir: Path, ledger: dict) -> dict[str, float]:
    """Each candidate's `D` from its most recent completed bout.

    Derived from the same tuning reports the evidence models read, because
    no ledger field carries it: the ledger records where a candidate ended
    up, not what its last bout moved. This is a diagnostic (§4.3) — it is
    written into the snapshot for calibration and never conditions a
    predictive model — so deriving it here rather than adding a field to a
    shared ledger schema keeps the change inside the scheduler.
    """
    from .evidence import derive_tuning_records

    latest: dict[str, tuple[int, float]] = {}
    for record in derive_tuning_records(run_dir, ledger):
        if record.run_id is None:
            continue
        index = record.bout_index or 0
        seen = latest.get(record.run_id)
        if seen is None or index >= seen[0]:
            latest[record.run_id] = (index, record.gain)
    return {run_id: gain for run_id, (_, gain) in latest.items()}


def load_state(
    ledger_path: Path,
    *,
    contract: ResourceContract | None = None,
    diagnostics: dict | None = None,
) -> SchedulerState:
    """Build the state for a live run directory."""
    ledger_path = Path(ledger_path)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    run_dir = ledger_path.parent
    remaining = _remaining_budget(run_dir)
    return build_state(
        ledger,
        remaining_budget=remaining,
        contract=contract,
        deferred_backlog=deferred_warm_backlog(run_dir, ledger),
        previous_gains=previous_gains(run_dir, ledger),
        diagnostics=diagnostics,
    )


def _remaining_budget(run_dir: Path) -> int:
    from evaluation_budget import budget_status

    status = budget_status(Path(run_dir))
    remaining = status.get("remaining")
    if not isinstance(remaining, int):
        # An unbounded run has no scheduling problem to solve: every action
        # is affordable forever, so there is no budget to allocate. Failing
        # here is honest; returning 0 would make every state look terminal.
        raise ValueError(
            f"{run_dir}: scheduler v3.2 requires a bounded evaluation budget "
            "(set max_evaluations in framework_cfg.json)"
        )
    return int(remaining)


def deferred_warm_backlog(run_dir: Path, ledger: dict) -> dict[str, int]:
    """Per-candidate count of proposed-but-unevaluated warm configs.

    These are step-0+1 configs a FIRST bout tries before its own search.
    They occupy slots inside `B`, so the count is mechanical state a FIRST
    bout's cost accounting needs, not a predictive covariate.
    """
    backlog: dict[str, int] = {}
    for record in ledger.get("records", []):
        run_id = str(record.get("run_id"))
        report_path = (
            Path(run_dir) / "candidates" / run_id / "tune_report.json"
        )
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        deferred = report.get("phase_a", {}).get("deferred_configs")
        if isinstance(deferred, list) and deferred:
            backlog[run_id] = len(deferred)
    return backlog
