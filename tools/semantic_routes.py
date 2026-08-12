#!/usr/bin/env python3
"""Bounded planned-route memory for semantic idea generation.

The attempt downside (``semantic_attempts.py``) tells *selection* that a point
has been expensive under the current policy.  It cannot tell *generation* what
was already tried there.  This module supplies that second channel: a bounded,
relevance-ordered list of previously **planned** routes at or near the point
about to be built, plus how each of those attempts turned out.

Two boundaries this module keeps deliberately sharp:

* a route record is **planned provenance only**.  It says what the generator
  intended before the writer ran; it proves nothing about the code that was
  actually produced.  No realized-route classification happens here.
* neighbor rows are *transfer context for generation*.  They never enter the
  target point's attempt posterior — that statistic is stratified by exact
  ``(point_id, op)`` and lives in ``semantic_attempts.py``.

Older records predate route provenance.  Their rows carry ``route: null`` and
``route_available: false``; nothing is ever fabricated for them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from run_cfg import load_run_cfg
from semantic_attempts import classify_attempts
from semantic_space import point_id as _point_id, selected_assignments


ROUTE_MEMORY_SCHEMA_VERSION = 1
PROVENANCE_SCHEMA_VERSION = 1

# Generation-side switches (PROPOSAL §4.3).  These are NOT acquisition weights
# and deliberately stay out of the policy receipt's `policy.config`: the route
# arm is audited through the persisted provenance itself.
DEFAULT_ROUTE_CONFIG = {
    # 0 = no explicit planning step at all (the pre-proposal behavior, and the
    # default).  1 = the explicit single-route planning baseline: the generator
    # states one route and persists it, so a breadth arm at 3 differs from it
    # only in breadth and self-ranking, not in whether planning happened.
    # >=2 sketches that many distinct routes.  Unrelated to tuner.K.
    "n_route_sketches": 0,
    "route_memory": False,
    "route_memory_rows": 6,
    # How many differing selected assignments still counts as a neighbor.
    "route_memory_max_distance": 2,
}


class RouteError(ValueError):
    """Route memory or route provenance violates its contract."""


def route_config(section: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve the route arm from a ``framework_cfg.json.semantic_search`` block."""
    cfg = dict(DEFAULT_ROUTE_CONFIG)
    for key, value in (section or {}).items():
        if key in cfg:
            cfg[key] = value
    if not isinstance(cfg["route_memory"], bool):
        raise RouteError("semantic_search.route_memory must be a boolean")
    sketches = cfg["n_route_sketches"]
    if not isinstance(sketches, int) or isinstance(sketches, bool) or sketches < 0:
        raise RouteError("semantic_search.n_route_sketches must be a non-negative integer")
    if sketches == 0 and cfg["route_memory"]:
        raise RouteError(
            "semantic_search.route_memory requires an explicit planning step "
            "(n_route_sketches >= 1); memory is shown to a planner, not to nobody"
        )
    for key in ("route_memory_rows", "route_memory_max_distance"):
        value = cfg[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise RouteError(f"semantic_search.{key} must be a positive integer")
    return cfg


def route_arm_active(cfg: dict[str, Any]) -> bool:
    """True when generation must produce and persist route provenance.

    One sketch is already an arm: the explicit-planning baseline.  Making it
    provenance-bearing is what keeps a 1-vs-3 comparison attributable to route
    breadth rather than to the presence of planning.
    """
    return int(cfg["n_route_sketches"]) >= 1


# ---------------------------------------------------------------------------
# memory
# ---------------------------------------------------------------------------


def _distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    """Number of dimensions whose selected hypothesis differs."""
    a = selected_assignments(left)
    b = selected_assignments(right)
    return sum(
        1 for key in set(a) | set(b) if a.get(key) != b.get(key)
    )


def build_route_memory(
    ledger: dict[str, Any],
    point: dict[str, Any],
    op: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Bounded relevance-ordered planned-route memory for one target point.

    Ordering (PROPOSAL §4.2): exact ``(point_id, op)`` matches first, then
    same-op semantic-assignment neighbors; within each group, nearer first and
    newer first.  The list is truncated to ``route_memory_rows``.
    """
    target_id = point.get("point_id") or _point_id(point)
    max_distance = int(cfg["route_memory_max_distance"])
    outcomes = {
        row["run_id"]: row["outcome"]
        for row in classify_attempts(ledger)
    }
    rows: list[dict[str, Any]] = []
    for record in ledger.get("records", []):
        if not isinstance(record, dict) or record.get("op") != op:
            continue
        record_point = record.get("semantic_point")
        if not isinstance(record_point, dict):
            continue
        record_id = record_point.get("point_id")
        exact = record_id == target_id
        distance = 0 if exact else _distance(record_point, point)
        if not exact and distance > max_distance:
            continue
        provenance = record.get("route_provenance")
        route = (
            provenance.get("chosen_route")
            if isinstance(provenance, dict)
            else None
        )
        run_id = str(record.get("run_id"))
        rows.append(
            {
                "run_id": run_id,
                "point_id": record_id,
                "relation": "same_point" if exact else "neighbor",
                "distance": distance,
                # Terminal classification when the attempt completed; a record
                # still in flight (or terminally unevaluated) has none.
                "outcome": outcomes.get(run_id),
                "route": route,
                "route_available": isinstance(route, str) and bool(route),
            }
        )
    rows.sort(key=lambda row: (row["relation"] != "same_point", row["distance"],
                              -int(row["run_id"])))
    return {
        "schema_version": ROUTE_MEMORY_SCHEMA_VERSION,
        "point_id": target_id,
        "op": op,
        "n_route_sketches": int(cfg["n_route_sketches"]),
        "route_memory": bool(cfg["route_memory"]),
        # An inactive memory arm still emits the envelope, so the generator's
        # read is unconditional and the arm cannot be confused with a helper
        # that simply failed to run.
        "rows": rows[: int(cfg["route_memory_rows"])] if cfg["route_memory"] else [],
    }


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


PROVENANCE_FIELDS = {
    "schema_version",
    "point_id",
    "op",
    "n_route_sketches",
    "route_memory",
    "memory_rows",
    "sketches",
    "preference_order",
    "chosen_sketch_id",
    "chosen_route",
}
# A candidate that no generator planned.  The provided baseline is installed
# verbatim from the task, so there is no route to record; saying so explicitly
# keeps the opt-out auditable instead of fabricating LLM provenance.
NOT_APPLICABLE_FIELDS = {"schema_version", "status", "reason"}


def is_not_applicable(provenance: Any) -> bool:
    return (
        isinstance(provenance, dict)
        and set(provenance) == NOT_APPLICABLE_FIELDS
        and provenance.get("status") == "not_applicable"
    )


def validate_not_applicable(provenance: Any) -> list[str]:
    errors: list[str] = []
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        errors.append(
            f"route_provenance.schema_version must be {PROVENANCE_SCHEMA_VERSION}"
        )
    reason = provenance.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        errors.append("route_provenance.reason must say why no route was planned")
    return errors


def validate_route_provenance(
    provenance: Any,
    *,
    memory: dict[str, Any],
) -> list[str]:
    """Validate one persisted planned-route record against its own memory view.

    ``memory`` is the deterministically recomputed view for this candidate, so
    an inconsistent or absent route record fails loudly instead of silently
    degrading the run into the no-memory arm.
    """
    if is_not_applicable(provenance):
        return validate_not_applicable(provenance)
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_FIELDS:
        return ["route_provenance must record the exact planned-route fields"]
    errors: list[str] = []
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        errors.append(
            f"route_provenance.schema_version must be {PROVENANCE_SCHEMA_VERSION}"
        )
    for key in ("point_id", "op", "n_route_sketches", "route_memory"):
        if provenance.get(key) != memory[key]:
            errors.append(
                f"route_provenance.{key} must equal the recomputed memory view"
            )
    # The whole rows, not just their run ids: a row's `outcome` moves as the
    # sibling it describes finishes, so only the stored snapshot can say what
    # the generator was actually shown.
    if provenance.get("memory_rows") != memory["rows"]:
        errors.append(
            "route_provenance.memory_rows must be exactly the rows the helper "
            "produced for this candidate"
        )
    sketches = provenance.get("sketches")
    if not isinstance(sketches, list) or len(sketches) != memory["n_route_sketches"]:
        return errors + [
            "route_provenance.sketches must hold one entry per configured "
            "route sketch"
        ]
    ids: list[str] = []
    for index, sketch in enumerate(sketches):
        where = f"route_provenance.sketches[{index}]"
        if not isinstance(sketch, dict) or set(sketch) != {"sketch_id", "route"}:
            errors.append(f"{where} must carry a sketch_id and a route")
            continue
        sketch_id = sketch.get("sketch_id")
        route = sketch.get("route")
        if not isinstance(sketch_id, str) or not sketch_id.strip():
            errors.append(f"{where}.sketch_id must be a non-empty string")
            continue
        if not isinstance(route, str) or not route.strip():
            errors.append(f"{where}.route must be a concrete non-empty route")
        ids.append(sketch_id)
    if len(set(ids)) != len(ids):
        errors.append("route_provenance.sketches must use distinct sketch ids")
    order = provenance.get("preference_order")
    if not isinstance(order, list) or sorted(order, key=str) != sorted(ids):
        errors.append(
            "route_provenance.preference_order must rank exactly the sketches"
        )
    chosen = provenance.get("chosen_sketch_id")
    if chosen not in ids:
        errors.append("route_provenance.chosen_sketch_id must name one sketch")
    else:
        route = next(s["route"] for s in sketches if s["sketch_id"] == chosen)
        if provenance.get("chosen_route") != route:
            errors.append(
                "route_provenance.chosen_route must repeat the chosen sketch's "
                "route verbatim"
            )
    if not isinstance(provenance.get("chosen_route"), str) or not str(
        provenance.get("chosen_route")
    ).strip():
        errors.append("route_provenance.chosen_route must be a non-empty string")
    return errors


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_memory(args: argparse.Namespace) -> int:
    ledger = json.loads(Path(args.ledger).read_text()) if Path(args.ledger).exists() else {}
    point = json.loads(Path(args.point).read_text())
    cfg = route_config(load_run_cfg(args.ledger, "semantic_search"))
    memory = build_route_memory(ledger, point, args.op, cfg)
    payload = json.dumps(memory, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    memory = sub.add_parser("memory", help="bounded planned-route memory")
    memory.add_argument("--ledger", required=True, type=Path)
    memory.add_argument("--point", required=True, type=Path)
    memory.add_argument("--op", required=True,
                        choices=["fresh", "improve", "crossover"])
    memory.add_argument("--output", type=Path)
    memory.set_defaults(func=cmd_memory)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (RouteError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
