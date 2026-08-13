"""Current-run empirical evidence and the plug-in models over it (§4, §5, §6).

Two deliberately coarse models:

**Tuning transitions.** `P(Z, D | q)` for `q in {FIRST, LATER}` — nothing
finer. With two historical runs of ~12 bouts each and cross-run pooling off,
conditioning further (progress × headroom × previous-gain bucket × generator
kind × stall × ...) leaves most cells with 0–2 records. That is not a more
faithful model; it is recording single accidents as conditional
distributions. The minimal-sufficient-predictive-state rule (§4.3) admits a
new conditioning variable only after it demonstrates stable out-of-sample
predictive value.

**Arrivals.** An exchangeable marginal over whole ordered generation
episodes (§5). Real arrivals certainly depend on generator kind, stall
regime, and DAG state; the current run does not carry enough episodes to
estimate that dependence, so v1 resamples episodes whole and keeps the
diagnostics for calibration instead.

Both are plug-in empirical distributions over `EVIDENCE_SCOPE =
current_run` (§6). Other runs inform offline design, never the online model:
sharing live evidence across arms would make replicates start from different
states and make any E2E difference unattributable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

EVIDENCE_SCOPE = "current_run"
EVIDENCE_FILENAME = "evidence.jsonl"

FIRST = "FIRST"
LATER = "LATER"

#: Outcome status of one bout (`Z` in equation (5)).
VALID = "valid"
SCIENTIFIC_INVALID = "scientific_invalid"
INFRA_FAILURE = "infra_failure"


@dataclass(frozen=True)
class TuningRecord:
    """One realized tuning bout: the `(Z, D)` pair plus its actual cost."""

    bout_class: str
    status: str
    gain: float
    cost: int
    run_id: str | None = None
    bout_index: int | None = None
    #: Recorded, not conditioned on.
    diagnostics: dict | None = None

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "tuning_transition",
            "bout_class": self.bout_class,
            "status": self.status,
            "gain": self.gain,
            "cost": self.cost,
            "run_id": self.run_id,
            "bout_index": self.bout_index,
            "diagnostics": self.diagnostics or {},
        }


@dataclass(frozen=True)
class ArrivalRecord:
    """One ordered, partially admissible generation episode (§5).

    `warm_gaps` are anchored to the episode's own pre-generation global
    best, which is what makes an episode resamplable into a different
    future state: a gap is relative, an absolute warm score is not. The
    anchor is the run's *raw global best* at that moment — including any
    improvement earlier tuning already produced — because that is the `g`
    equation (7) re-anchors against.

    A gap of `None` marks an arrival that produced no usable candidate. It
    still consumed its evaluations, so it stays in the episode: dropping it
    would make generation look cheaper and more productive than it is, and
    turning it into `0.0` would invent a candidate exactly at the global
    best.
    """

    warm_gaps: tuple[float | None, ...]
    planned_count: int
    per_candidate_cost: int
    statuses: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    diagnostics: dict | None = None

    @property
    def usable_gaps(self) -> tuple[float, ...]:
        return tuple(gap for gap in self.warm_gaps if gap is not None)

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "kind": "arrival_episode",
            "warm_gaps": list(self.warm_gaps),
            "planned_count": self.planned_count,
            "per_candidate_cost": self.per_candidate_cost,
            "statuses": list(self.statuses),
            "run_ids": list(self.run_ids),
            "diagnostics": self.diagnostics or {},
        }


# =============================================================================
# Evidence store (append-only, current run)
# =============================================================================


class EvidenceLog:
    """Append-only current-run evidence, with a stable read cursor.

    The cursor is the record count at read time. A decision receipt cites
    it so a replay can rebuild the exact evidence state the decision saw,
    rather than the evidence that accumulated afterwards.
    """

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, record: TuningRecord | ArrivalRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_json(), sort_keys=True) + "\n")

    def rows(self, *, cursor: int | None = None) -> list[dict]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows if cursor is None else rows[:cursor]

    def cursor(self) -> int:
        return len(self.rows())

    def tuning_records(self, *, cursor: int | None = None) -> list[TuningRecord]:
        return [
            TuningRecord(
                bout_class=row.get("bout_class", LATER),
                status=row.get("status", VALID),
                gain=float(row.get("gain", 0.0)),
                cost=int(row.get("cost", 0)),
                run_id=row.get("run_id"),
                bout_index=row.get("bout_index"),
                diagnostics=row.get("diagnostics"),
            )
            for row in self.rows(cursor=cursor)
            if row.get("kind") == "tuning_transition"
        ]

    def arrival_records(self, *, cursor: int | None = None) -> list[ArrivalRecord]:
        return [
            ArrivalRecord(
                warm_gaps=tuple(
                    None if gap is None else float(gap)
                    for gap in row.get("warm_gaps", [])
                ),
                planned_count=int(row.get("planned_count", 0)),
                per_candidate_cost=int(row.get("per_candidate_cost", 2)),
                statuses=tuple(row.get("statuses", [])),
                run_ids=tuple(str(rid) for rid in row.get("run_ids", [])),
                diagnostics=row.get("diagnostics"),
            )
            for row in self.rows(cursor=cursor)
            if row.get("kind") == "arrival_episode"
        ]


# =============================================================================
# Plug-in empirical models
# =============================================================================


@dataclass(frozen=True)
class TuningModel:
    """Empirical `P(Z, D | q)` with a frozen fallback for thin classes.

    `min_support` is the smallest number of same-class records the model
    will estimate from. Below it, the class falls back to the pooled
    FIRST+LATER records — a coarser but non-degenerate estimate — and below
    the pooled threshold the model reports itself unsupported so the policy
    can take its coverage or fallback path (§6) instead of sampling from a
    distribution that does not exist.
    """

    first: tuple[TuningRecord, ...]
    later: tuple[TuningRecord, ...]
    min_support: int = 3
    pooled_min_support: int = 3

    @classmethod
    def from_records(
        cls,
        records: Sequence[TuningRecord],
        *,
        min_support: int = 3,
        pooled_min_support: int = 3,
    ) -> "TuningModel":
        return cls(
            first=tuple(r for r in records if r.bout_class == FIRST),
            later=tuple(r for r in records if r.bout_class == LATER),
            min_support=min_support,
            pooled_min_support=pooled_min_support,
        )

    def support(self, bout_class: str) -> tuple[TuningRecord, ...]:
        """The record set this class actually draws from."""
        exact = self.first if bout_class == FIRST else self.later
        if len(exact) >= self.min_support:
            return exact
        pooled = self.first + self.later
        if len(pooled) >= self.pooled_min_support:
            return pooled
        return ()

    def exact_count(self, bout_class: str) -> int:
        return len(self.first if bout_class == FIRST else self.later)

    def exact_supported(self, bout_class: str) -> bool:
        """Whether this class has enough evidence of its OWN kind.

        Separate from `supported` because the two answer different
        questions. `supported` asks whether the rollout can sample at all;
        this asks whether it would be sampling the right distribution.
        Pooling FIRST into LATER is a usable stopgap, but it erases exactly
        the FIRST/LATER difference the model exists to represent, so
        coverage keeps trying to collect the real class while its shared cap
        lasts.
        """
        exact = self.first if bout_class == FIRST else self.later
        return len(exact) >= self.min_support

    def usage_mode(self, bout_class: str) -> str:
        """How this class is being estimated: exact, pooled, or unsupported."""
        if self.exact_supported(bout_class):
            return "exact"
        return "pooled" if self.supported(bout_class) else "unsupported"

    def supported(self, bout_class: str) -> bool:
        return bool(self.support(bout_class))

    def sample(self, bout_class: str, uniform: float) -> TuningRecord:
        """Draw one `(Z, D, cost)` outcome.

        `uniform` is a draw in [0, 1) supplied by the caller's scenario, not
        generated here: paired rollouts key their randomness by future-event
        identity, which only the caller knows.
        """
        support = self.support(bout_class)
        if not support:
            raise ValueError(f"tuning model has no support for class {bout_class}")
        index = min(int(uniform * len(support)), len(support) - 1)
        return support[index]

    def summary(self) -> dict:
        def stats(records: Sequence[TuningRecord]) -> dict:
            gains = [r.gain for r in records]
            return {
                "count": len(records),
                "mean_gain": (sum(gains) / len(gains)) if gains else None,
                "max_gain": max(gains) if gains else None,
                "positive_fraction": (
                    sum(1 for g in gains if g > 0) / len(gains) if gains else None
                ),
            }

        return {
            "scope": EVIDENCE_SCOPE,
            "min_support": self.min_support,
            "FIRST": stats(self.first),
            "LATER": stats(self.later),
        }


@dataclass(frozen=True)
class ArrivalModel:
    """Exchangeable marginal over ordered generation episodes (§5)."""

    episodes: tuple[ArrivalRecord, ...]
    min_support: int = 2

    @classmethod
    def from_records(
        cls,
        records: Sequence[ArrivalRecord],
        *,
        min_support: int = 2,
    ) -> "ArrivalModel":
        # An episode with no arrivals at all carries no information — it
        # only marks which candidates reconciliation has consumed. An
        # episode whose arrivals all failed DOES carry information (that
        # generation can spend a round and return nothing), so it is kept.
        return cls(
            episodes=tuple(r for r in records if r.warm_gaps),
            min_support=min_support,
        )

    def supported(self) -> bool:
        return len(self.episodes) >= self.min_support

    def sample(self, uniform: float) -> ArrivalRecord:
        if not self.episodes:
            raise ValueError("arrival model has no observed episodes")
        index = min(int(uniform * len(self.episodes)), len(self.episodes) - 1)
        return self.episodes[index]

    def summary(self) -> dict:
        counts = [len(episode.usable_gaps) for episode in self.episodes]
        gaps = [gap for episode in self.episodes for gap in episode.usable_gaps]
        failed = sum(
            1
            for episode in self.episodes
            for gap in episode.warm_gaps
            if gap is None
        )
        return {
            "scope": EVIDENCE_SCOPE,
            "episodes": len(self.episodes),
            "mean_admitted": (sum(counts) / len(counts)) if counts else None,
            "mean_warm_gap": (sum(gaps) / len(gaps)) if gaps else None,
            "improving_gap_fraction": (
                sum(1 for gap in gaps if gap < 0) / len(gaps) if gaps else None
            ),
            "failed_arrivals": failed,
        }


def admitted_prefix(
    episode: ArrivalRecord,
    remaining_budget: int,
) -> list[tuple[float | None, int]]:
    """The ordered prefix of an episode the remaining budget admits (§5).

    Arrival is ordered and partially admissible: the driver walks the
    generated list and stops when the budget runs out, so a resampled
    episode must be truncated the same way rather than admitted whole or
    dropped whole.

    A `None` gap is preserved, not filtered: a failed arrival spent its
    slots, and the simulator has to charge that cost without gaining a
    candidate. Dropping it here would silently convert a wasted generation
    into a free one.
    """
    prefix: list[tuple[float | None, int]] = []
    budget = remaining_budget
    cost = max(1, episode.per_candidate_cost)
    for gap in episode.warm_gaps:
        if budget < cost:
            break
        budget -= cost
        prefix.append((gap, cost))
    return prefix


# =============================================================================
# Derivation from durable run artifacts
# =============================================================================


def bout_class(bouts_used_before: int) -> str:
    return FIRST if bouts_used_before == 0 else LATER


def derive_tuning_records(run_dir: Path, ledger: dict) -> list[TuningRecord]:
    """Reconstruct realized bout outcomes from candidates' tuning reports.

    Used to seed the evidence log for a run whose bouts predate the
    scheduler, and by calibration to check the log against the artifacts it
    claims to summarize. The live path appends outcomes as they happen; this
    is the reconciliation path.

    Only *closed* bouts are outcomes. `tune_report.json` is durable while a
    bout is still in flight — an interrupted stage stays `running` with its
    already-charged trials on disk, and a rejected stage sits mid-method-chain
    until the fallback runs — so reading the report alone would report a
    3-trial fragment as if it were a 10-trial bout. That is not a transient
    error: the evidence log is append-only and deduped by
    `(run_id, bout_index)`, so the fragment would be frozen and the completed
    bout never recorded. The ledger's `tuning_bouts` is the run's own count of
    bouts that reached `finalize`, and it is the same number `SchedulerState`
    reads for `bouts_used`, so gating on it keeps the evidence and the
    mechanical state describing the same set of bouts.
    """
    records: list[TuningRecord] = []
    for record in ledger.get("records", []):
        run_id = str(record.get("run_id"))
        report_path = Path(run_dir) / "candidates" / run_id / "tune_report.json"
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        closed = int(record.get("tuning_bouts") or 0)
        records.extend(_records_from_report(report, run_id, closed_bouts=closed))
    return records


def _bout_status(bout_stages: list[dict], scored: int, attempted: int) -> str:
    """`Z` for one bout, from the stage and trial statuses it recorded.

    The three classes are not interchangeable and the rollout treats them
    differently, so the derivation has to distinguish them from artifacts
    rather than collapsing everything scoreless into "infra":

    * `valid` — at least one trial produced a finite score. The bout
      observed the objective; `D` means something.
    * `scientific_invalid` — trials were proposed and every one of them was
      rejected before reaching the objective (preflight), or the tuner's
      whole method chain was rejected. The machinery worked; the *configs*
      were unusable. Nothing was learned about the objective, but the bout
      is not evidence of a broken run either.
    * `infra_failure` — the bout tried to evaluate and could not.
    """
    if scored:
        return VALID
    statuses = {
        stage.get("status") for stage in bout_stages if isinstance(stage, dict)
    }
    if statuses and statuses <= {"rejected", "no_search_needed"}:
        return SCIENTIFIC_INVALID
    if attempted and attempted == sum(
        1
        for stage in bout_stages
        for trial in stage.get("trials", [])
        if isinstance(trial, dict) and trial.get("status") == "preflight_rejected"
    ):
        return SCIENTIFIC_INVALID
    return INFRA_FAILURE


def _records_from_report(
    report: dict, run_id: str, *, closed_bouts: int | None = None
) -> list[TuningRecord]:
    phase_a = report.get("phase_a", {})
    warm_best = phase_a.get("best_warm_score")
    if not _finite(warm_best):
        return []
    stages = report.get("phase_c", {}).get("stages", [])
    if not isinstance(stages, list):
        return []

    by_bout: dict[int, list[dict]] = {}
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        by_bout.setdefault(int(stage.get("bout_index", 0) or 0), []).append(stage)

    incumbent = float(warm_best)
    out: list[TuningRecord] = []
    for index in sorted(by_bout):
        # Bout indices are sequential and a new bout only opens once the
        # previous one finalized, so the closed bouts are exactly [0, n).
        if closed_bouts is not None and index >= closed_bouts:
            break
        bout_stages = by_bout[index]
        scores = [
            float(trial["score"])
            for stage in bout_stages
            for trial in stage.get("trials", [])
            if isinstance(trial, dict) and _finite(trial.get("score"))
        ]
        attempted = sum(
            len(stage.get("trials", []))
            for stage in bout_stages
            if isinstance(stage.get("trials"), list)
        )
        best = min(scores) if scores else None
        # D_i is the raw improvement to the candidate's OWN best (eq. 2),
        # so it stays signed: a bout that found nothing better is evidence
        # of a negative-or-zero draw, not a missing observation. The floor
        # at zero belongs to the state transition, not to the record.
        gain = incumbent - best if best is not None else 0.0
        stage_statuses = [
            stage.get("status")
            for stage in bout_stages
            if isinstance(stage, dict)
        ]
        out.append(
            TuningRecord(
                bout_class=bout_class(index),
                status=_bout_status(bout_stages, len(scores), attempted),
                gain=gain,
                cost=attempted,
                run_id=run_id,
                bout_index=index,
                # Termination shape: recorded for calibration and for the
                # cost model's "did this bout stop early, and why" question.
                # Recorded, never conditioned on (§4.3).
                diagnostics={
                    "stage_statuses": stage_statuses,
                    "scored_trials": len(scores),
                    "terminated_on_budget": "budget_exhausted" in stage_statuses,
                    "methods": [
                        stage.get("method")
                        for stage in bout_stages
                        if isinstance(stage, dict)
                    ],
                },
            )
        )
        if best is not None:
            incumbent = min(incumbent, best)
    return out


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
