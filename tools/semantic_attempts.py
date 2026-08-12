#!/usr/bin/env python3
"""Policy-conditioned attempt downside for semantic point selection.

This module owns one cheap signal the carrier posterior cannot express: how
did candidates previously built *at this exact semantic point, under this
exact action type*, actually turn out — under the current writer, route and
evaluation policy.  It is therefore an empirical risk of re-selecting the
point, **not** a causal statement about the mechanism the point encodes.

Two deliberate separations:

* the statistic reads a helper-owned, one-row-per-run observation store
  captured at attempt time (``ledger.attempt_observations``), never the mutable
  ``best_warm_score`` / ``final_best_score`` fields a later tuning bout
  rewrites;
* the resulting adjustment stays out of the carrier posterior so the two
  channels can be compared independently (``coverage_attempt`` vs
  ``coverage_experience``).

Outcome partition (mutually exclusive, exhaustive over completed attempts):
``crash``, ``screen_success``, ``screen_fail``, ``screen_neutral``,
``unpaired``.  ``unevaluated`` candidates never enter the store at all.
"""

from __future__ import annotations

import math
from typing import Any


OBSERVATION_SCHEMA_VERSION = 1
OBSERVATION_KIND = "attempt_screening_observation"
OBSERVATION_STORE_KEY = "attempt_observations"
OBSERVATION_FIELDS = {
    "schema_version",
    "kind",
    "run_id",
    "point_id",
    "op",
    "primary_parent_run_id",
    "status",
    "screening_score",
}

OUTCOMES = (
    "crash",
    "screen_success",
    "screen_fail",
    "screen_neutral",
    "unpaired",
)

# Prior/weight shape is deliberately calibratable: PROPOSAL §3.3 leaves the
# magnitudes to offline replay rather than freezing them here.
DEFAULT_ATTEMPT_CONFIG = {
    # Pre-frozen noise threshold tau on the child-minus-parent warm delta.
    "attempt_noise_threshold": 0.0,
    "attempt_screen_weight": 0.20,
    "attempt_crash_weight": 0.30,
    # Shrinkage mass: how much paired exposure a point needs before its own
    # failure rate is trusted at full strength.
    "attempt_screen_prior_strength": 2.0,
    # Smoothing mass for the absolute crash rate.  Strictly positive so a
    # single observed crash can never be diluted to exactly zero by
    # co-occurring unpaired attempts.
    "attempt_crash_prior_strength": 1.0,
    "attempt_cap": 0.60,
}


class AttemptError(ValueError):
    """The attempt observation store violates its contract."""


def _finite(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------


def capture_attempt_observation(data: dict[str, Any], record: dict[str, Any]) -> None:
    """Freeze one candidate's screening observation at its terminal transition.

    Called by ``tools/ledger.py record-run``, the single point where a
    candidate's step 0+1 screening result becomes terminal.  At that moment
    ``final_best_score`` still equals the warm screening score; a later Phase-C
    bout rewrites both it and ``best_warm_score``, which is exactly why the
    classification cannot read them back.  The first write for a run id wins,
    except that a crash may be replaced once by a later finite screening
    result when the repair-capable extractor completes the same candidate.
    """
    run_id = str(record.get("run_id"))
    store = data.setdefault(OBSERVATION_STORE_KEY, [])
    status = record.get("status")
    if status not in {"keep", "discard", "crash"}:
        return
    point = record.get("semantic_point")
    point_id = point.get("point_id") if isinstance(point, dict) else None
    if not isinstance(point_id, str) or not point_id:
        return
    parents = record.get("source_run_ids")
    parents = parents if isinstance(parents, list) else []
    score = None if status == "crash" else _finite(record.get("final_best_score"))
    observation = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "kind": OBSERVATION_KIND,
        "run_id": run_id,
        "point_id": point_id,
        "op": record.get("op"),
        "primary_parent_run_id": str(parents[0]) if parents else None,
        "status": status,
        "screening_score": score,
    }
    existing_index = next(
        (
            index
            for index, item in enumerate(store)
            if str(item.get("run_id")) == run_id
        ),
        None,
    )
    if existing_index is None:
        store.append(observation)
        return
    existing = store[existing_index]
    if (
        isinstance(existing, dict)
        and existing.get("status") == "crash"
        and status in {"keep", "discard"}
        and score is not None
    ):
        # A repair session can legitimately turn the candidate's provisional
        # crash into its one completed screening result.  Phase-C never calls
        # this helper, so this exception cannot overwrite the warm snapshot
        # with a later tuning-lowered score.
        store[existing_index] = observation


def validate_attempt_observations(ledger: dict[str, Any]) -> list[str]:
    """Validate the helper-owned attempt observation store."""
    store = ledger.get(OBSERVATION_STORE_KEY, [])
    if store is None:
        return []
    if not isinstance(store, list):
        return [f"ledger.{OBSERVATION_STORE_KEY} must be a list"]
    records = ledger.get("records")
    records_by_run = {
        str(record.get("run_id")): record
        for record in records
        if isinstance(record, dict)
    } if isinstance(records, list) else {}
    errors: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(store):
        where = f"ledger.{OBSERVATION_STORE_KEY}[{index}]"
        if not isinstance(item, dict) or set(item) != OBSERVATION_FIELDS:
            errors.append(f"{where} has an invalid shape")
            continue
        run_id = str(item.get("run_id"))
        if run_id in seen:
            errors.append(f"{where} duplicates run {run_id}")
        seen.add(run_id)
        parent = item.get("primary_parent_run_id")
        if (
            item.get("schema_version") != OBSERVATION_SCHEMA_VERSION
            or item.get("kind") != OBSERVATION_KIND
            or not run_id.isdigit()
            or not isinstance(item.get("point_id"), str)
            or item.get("op") not in {"fresh", "improve", "crossover"}
            or not (parent is None or (isinstance(parent, str) and parent.isdigit()))
            or item.get("status") not in {"keep", "discard", "crash"}
        ):
            errors.append(f"{where} is not a valid attempt observation")
            continue
        score = item.get("screening_score")
        if item["status"] == "crash":
            if score is not None:
                errors.append(f"{where} crash observation must carry no score")
        elif score is not None and _finite(score) is None:
            errors.append(f"{where} screening_score must be null or finite")
        record = records_by_run.get(run_id)
        if isinstance(record, dict) and (
            (item.get("status") == "crash")
            != (record.get("status") == "crash")
        ):
            errors.append(
                f"{where} crash/non-crash state must match its ledger record"
            )
    return errors


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def classify_attempts(
    ledger: dict[str, Any], *, noise_threshold: float = 0.0
) -> list[dict[str, Any]]:
    """Classify every completed attempt into exactly one outcome.

    A non-crash attempt is paired only when both the child and the parent that
    actually supplied the inherited implementation carry a finite screening
    observation; everything else is ``unpaired`` (a fresh candidate normally
    is).  Warm screening pairs with warm screening only — never with a
    tuning-lowered final score.
    """
    store = ledger.get(OBSERVATION_STORE_KEY, []) or []
    by_run = {str(item.get("run_id")): item for item in store if isinstance(item, dict)}
    tau = abs(float(noise_threshold))
    rows: list[dict[str, Any]] = []
    for item in store:
        if not isinstance(item, dict):
            continue
        row = {
            "run_id": str(item.get("run_id")),
            "point_id": item.get("point_id"),
            "op": item.get("op"),
            "delta": None,
            "parent_run_id": item.get("primary_parent_run_id"),
        }
        if item.get("status") == "crash":
            rows.append({**row, "outcome": "crash"})
            continue
        child_score = _finite(item.get("screening_score"))
        parent = by_run.get(str(item.get("primary_parent_run_id")))
        parent_score = (
            _finite(parent.get("screening_score")) if isinstance(parent, dict) else None
        )
        if child_score is None or parent_score is None:
            rows.append({**row, "outcome": "unpaired"})
            continue
        delta = child_score - parent_score
        if delta < -tau:
            outcome = "screen_success"
        elif delta > tau:
            outcome = "screen_fail"
        else:
            outcome = "screen_neutral"
        rows.append({**row, "outcome": outcome, "delta": round(delta, 12)})
    return rows


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in OUTCOMES}
    for row in rows:
        counts[row["outcome"]] += 1
    return counts


# ---------------------------------------------------------------------------
# adjustment
# ---------------------------------------------------------------------------


def attempt_adjustment(
    rows: list[dict[str, Any]],
    *,
    point_id: str,
    op: str,
    cfg: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Return the non-positive attempt adjustment for one ``(point, op)``.

    ``A(P, a) = A_screen(P, a) + A_crash(P, a) <= 0``.

    ``A_screen`` penalizes only the part of the point's non-improvement rate
    that exceeds the same-op base rate computed over *other* points
    (leave-one-point-out, so a point cannot inflate its own reference), scaled
    by how much paired exposure the point actually has.  New non-failure
    observations therefore dilute an old penalty and the point can soft-reopen.

    ``A_crash`` is absolute: observing a crash produces a negative value
    without having to exceed any global crash base rate.  Its smoothing mass
    keeps the value strictly negative once a crash exists, so co-occurring
    unpaired attempts cannot cancel it to zero.

    No relevant exposure of either kind yields exactly ``0.0``.
    """
    same_op = [row for row in rows if row["op"] == op]
    mine = [row for row in same_op if row["point_id"] == point_id]
    others = [row for row in same_op if row["point_id"] != point_id]

    counts = _counts(mine)
    paired = counts["screen_success"] + counts["screen_fail"] + counts["screen_neutral"]
    other_counts = _counts(others)
    other_paired = (
        other_counts["screen_success"]
        + other_counts["screen_fail"]
        + other_counts["screen_neutral"]
    )
    # Laplace-smoothed leave-one-point-out base rate of "did not improve".
    base_rate = (other_counts["screen_fail"] + other_counts["screen_neutral"] + 1.0) / (
        other_paired + 2.0
    )

    screen_prior = float(cfg["attempt_screen_prior_strength"])
    if paired:
        point_rate = (counts["screen_fail"] + counts["screen_neutral"]) / paired
        confidence = paired / (paired + screen_prior)
        screen = (
            -float(cfg["attempt_screen_weight"])
            * confidence
            * max(0.0, point_rate - base_rate)
        )
    else:
        point_rate = None
        screen = 0.0

    exposure = sum(counts.values())
    if counts["crash"]:
        crash_rate = counts["crash"] / (
            exposure + float(cfg["attempt_crash_prior_strength"])
        )
        crash = -float(cfg["attempt_crash_weight"]) * crash_rate
    else:
        crash_rate = 0.0
        crash = 0.0

    value = max(-abs(float(cfg["attempt_cap"])), screen + crash)
    detail = {
        "op": op,
        "counts": counts,
        "paired": paired,
        "base_rate": round(base_rate, 10),
        "point_rate": None if point_rate is None else round(point_rate, 10),
        "crash_rate": round(crash_rate, 10),
        "screen": round(screen, 10),
        "crash": round(crash, 10),
        "run_ids": sorted(row["run_id"] for row in mine),
    }
    return round(value, 10), detail


def attempt_priors(
    proposal_set: dict[str, Any],
    ledger: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, tuple[float, dict[str, Any]]]:
    """Per-proposal attempt adjustment for the proposal set's own action type."""
    op = proposal_set["action"]["op"]
    rows = classify_attempts(
        ledger, noise_threshold=float(cfg["attempt_noise_threshold"])
    )
    priors: dict[str, tuple[float, dict[str, Any]]] = {}
    for proposal in proposal_set["proposals"]:
        priors[proposal["point_id"]] = attempt_adjustment(
            rows, point_id=proposal["point_id"], op=op, cfg=cfg
        )
    return priors


def validate_attempt_detail(detail: Any) -> list[str]:
    """Validate one receipt's recorded attempt statistic."""
    fields = {
        "op",
        "counts",
        "paired",
        "base_rate",
        "point_rate",
        "crash_rate",
        "screen",
        "crash",
        "run_ids",
    }
    if not isinstance(detail, dict) or set(detail) != fields:
        return ["attempts must record the exact attempt statistic fields"]
    counts = detail.get("counts")
    if not isinstance(counts, dict) or set(counts) != set(OUTCOMES) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in counts.values()
    ):
        return ["attempts.counts must cover the full outcome partition"]
    run_ids = detail.get("run_ids")
    if not isinstance(run_ids, list) or any(
        not isinstance(item, str) or not item.isdigit() for item in run_ids
    ):
        return ["attempts.run_ids must list the contributing numeric run ids"]
    if len(run_ids) != sum(counts.values()):
        return ["attempts.run_ids must match the counted exposure"]
    if detail.get("op") not in {"fresh", "improve", "crossover"}:
        return ["attempts.op must be the selection's action type"]
    for key in ("base_rate", "crash_rate", "paired"):
        if _finite(detail.get(key)) is None:
            return [f"attempts.{key} must be finite"]
    for key in ("screen", "crash"):
        value = _finite(detail.get(key))
        if value is None or value > 0.0:
            return [f"attempts.{key} must be a finite non-positive number"]
    point_rate = detail.get("point_rate")
    if point_rate is not None and _finite(point_rate) is None:
        return ["attempts.point_rate must be null or finite"]
    return []
