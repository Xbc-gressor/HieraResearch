#!/usr/bin/env python3
"""slate.py — judged-slate deterministic kernel (no LLM, no driver imports).

One generation of the `judged_slate` arm, minus every model call:

    lanes.json (got_select decide --mode lanes)
      + per-lane proposal sets (semantic_search.py propose)
      -> construct      seats the pool, freezes carriers, writes the shared A1
                        context (pool.json + context.json)
      -> prepare-judge  stable per-stage presented order + bounded prompt
                        payload (<stage>.input.json)
      -> validate-judge receipt permutation check -> <stage>.json
      -> aggregate      consensus / boundary / coverage fallback -> judge.json
      -> build-manifest immutable generation.json (under the transfer
                        scheduler policy: plus the generation's donor binding)
      -> replay         re-derive every generation's deterministic decisions
                        from the artifacts + ledger prefix; exit 1 on mismatch

The decision core is a set of pure functions over already-parsed dicts; the
CLI facade owns all file I/O. Every artifact is written via a temporary file
plus os.replace(). Stable seeds never use Python's process-local hash().

Scores are always lower-is-better; coverage order is the only fallback order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

from background_contract import ContractError, load_registry
from ledger_core import (
    experience_receipt,
    records_prefix_digest,
    search_space_state_revision,
)
from run_cfg import read_framework_cfg
from semantic_evidence import LIFECYCLE_TERMINAL_STATUSES
from semantic_search import validate_proposal_set
from semantic_space import (
    canonical_json,
    digest,
    dimension_map,
    hypothesis_map,
    selected_assignments,
    space_receipt,
)


SCHEMA_VERSION = 1
SLATE_SIZE = 2
REGULAR_STAGES = ("regular-0", "regular-1")
BOUNDARY_STAGE = "boundary"
STAGES = (*REGULAR_STAGES, BOUNDARY_STAGE)
DEFAULT_POOL_SIZE = 6
POOL_SIZE_MIN = 3
POOL_SIZE_MAX = 12

# Carrier tie-break after lane value: improve > crossover > fresh.
OP_PRIORITY = {"improve": 0, "crossover": 1, "fresh": 2}

# A1 measured-history bounds (production values for the D1/D2 row semantics).
A1_LIMITS = {
    "ancestor_levels": 3,
    "ancestor_rows": 12,
    "sibling_rows_per_parent": 4,
    "sibling_rows": 12,
    "anchor_low": 5,
    "anchor_high": 5,
    "idea_chars": 140,
}


# ============================ decision core (pure) ============================


def _coverage_key(entry: dict) -> tuple:
    """The proposal/pool coverage order: coverage desc, deprioritized last."""
    return (
        -float(entry["coverage"]),
        bool(entry["deprioritized_hypotheses"]),
        entry["point_id"],
    )


def _carrier_key(carrier: dict) -> tuple:
    """Highest lane value wins; a fresh lane (value None) counts as -inf."""
    value = carrier.get("lane_value")
    score = -float(value) if value is not None else float("inf")
    parents = tuple(int(parent) for parent in carrier.get("parents", []))
    return (score, OP_PRIORITY[carrier["op"]], parents, carrier["lane_id"])


def bare_point_summary(
    point: dict,
    carrier: dict,
    deprioritized_hypotheses: list,
    registry: dict,
) -> dict:
    """The bounded, leak-safe candidate view a judge may consume."""
    dimensions = dimension_map(registry)
    hypotheses = hypothesis_map(registry)
    selected = []
    for dimension_id, hypothesis_id in selected_assignments(point).items():
        dimension = dimensions.get(dimension_id) or {}
        if hypothesis_id == dimension.get("baseline_hypothesis_id"):
            continue
        _, hypothesis = hypotheses.get(hypothesis_id, (None, {}))
        selected.append(
            {
                "dimension_id": dimension_id,
                "hypothesis_id": hypothesis_id,
                "title": hypothesis.get("title"),
                "claim": hypothesis.get("claim"),
            }
        )
    return {
        "carrier": {"op": carrier["op"], "parents": list(carrier["parents"])},
        "parent_diffs": carrier["parent_diffs"],
        "hypotheses": selected,
        "deprioritized_hypotheses": list(deprioritized_hypotheses),
    }


def build_pool(
    lanes: list,
    proposal_sets: dict,
    registry: dict,
    pool_size: int,
) -> dict:
    """Seat one generation's candidate pool from per-lane proposal sets.

    Seating: every lane with at least one proposal nominates its coverage
    leader (a repeated point occupies one seat and satisfies every nominating
    lane); remaining seats fill in union coverage order; the final pool is
    numbered in coverage order, which is also the fallback order. A pool
    smaller than the target is kept as-is (never padded with duplicates); a
    generation where no lane produced any proposal yields an empty pool.
    """
    if (
        not isinstance(pool_size, int)
        or isinstance(pool_size, bool)
        or not POOL_SIZE_MIN <= pool_size <= POOL_SIZE_MAX
    ):
        raise ContractError(
            f"pool_size must be an integer in [{POOL_SIZE_MIN}, {POOL_SIZE_MAX}]"
        )
    entries: dict[str, dict] = {}
    nominees: list[str] = []
    lanes_without_proposals: list[str] = []
    proposal_set_revisions: dict[str, Any] = {}
    for lane in lanes:
        lane_id = lane["lane_id"]
        proposal_set = proposal_sets.get(lane_id)
        proposals = (
            proposal_set.get("proposals") if isinstance(proposal_set, dict) else None
        )
        if not proposals:
            lanes_without_proposals.append(lane_id)
            continue
        proposal_set_revisions[lane_id] = proposal_set.get("proposal_set_revision")
        nominees.append(min(proposals, key=_coverage_key)["point_id"])
        for proposal in proposals:
            carrier = {
                "lane_id": lane_id,
                "op": lane["op"],
                "parents": list(lane["parents"]),
                "parent_diffs": proposal["parent_diffs"],
                "lane_value": lane.get("lane_value"),
                "proposal_set_revision": proposal_set.get("proposal_set_revision"),
            }
            entry = entries.get(proposal["point_id"])
            if entry is None:
                entries[proposal["point_id"]] = {
                    "point_id": proposal["point_id"],
                    "point": proposal["point"],
                    "coverage": proposal["coverage"],
                    "deprioritized_hypotheses": list(
                        proposal["deprioritized_hypotheses"]
                    ),
                    "carriers": [carrier],
                }
            else:
                entry["carriers"].append(carrier)
    ordered = sorted(entries.values(), key=_coverage_key)
    nominated = [e for e in ordered if e["point_id"] in set(nominees)]
    seated_ids = {e["point_id"] for e in nominated[:pool_size]}
    for entry in ordered:
        if len(seated_ids) >= pool_size:
            break
        seated_ids.add(entry["point_id"])
    pool = []
    for rank, entry in enumerate(
        (e for e in ordered if e["point_id"] in seated_ids), start=1
    ):
        carriers = sorted(entry["carriers"], key=_carrier_key)
        chosen, alternatives = carriers[0], carriers[1:]
        pool.append(
            {
                "label": f"C{rank}",
                "coverage_rank": rank,
                "point_id": entry["point_id"],
                "point": entry["point"],
                "coverage": entry["coverage"],
                "deprioritized_hypotheses": entry["deprioritized_hypotheses"],
                "carrier": chosen,
                "carrier_alternatives": alternatives,
                "summary": bare_point_summary(
                    entry["point"],
                    chosen,
                    entry["deprioritized_hypotheses"],
                    registry,
                ),
            }
        )
    return {
        "pool": pool,
        "lanes_without_proposals": lanes_without_proposals,
        "proposal_set_revisions": proposal_set_revisions,
    }


def _rid_key(run_id: Any) -> tuple:
    text = str(run_id)
    return (0, int(text)) if text.isdigit() else (1, text)


def _finite_warm(record: dict) -> float | None:
    value = record.get("best_warm_score")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    return None


def _a1_row(record: dict, role: str, by_id: dict, idea_chars: int) -> dict:
    run_id = str(record.get("run_id"))
    parents = [str(item) for item in (record.get("source_run_ids") or [])]
    op = record.get("op")
    if not isinstance(op, str) or not op:
        op = "fresh" if not parents else ("improve" if len(parents) == 1 else "crossover")
    status = record.get("status")
    warm = _finite_warm(record)
    if status == "crash":
        warm_text = "crash"
    elif warm is not None:
        warm_text = f"{warm:.6f}"
    else:
        warm_text = "n/a"
    delta = None
    if parents:
        parent_warm = _finite_warm(by_id.get(parents[0]) or {})
        if warm is not None and parent_warm is not None:
            delta = warm - parent_warm
    idea = " ".join(str(record.get("idea") or "").split())[:idea_chars]
    bracket = f"{op}<-{','.join(parents)}" if parents else op
    line = f"run {run_id} [{bracket}] warm={warm_text}"
    if delta is not None:
        line += f" d={delta:+.6f}"
    line += f" idea: {idea}"
    return {
        "run_id": run_id,
        "role": role,
        "op": op,
        "parents": parents,
        "status": status,
        "warm_score": warm,
        "delta": delta,
        "idea": idea,
        "line": line,
    }


def build_a1_context(ledger_prefix: dict, pool: dict, limits: dict = A1_LIMITS) -> dict:
    """The single leak-safe measured-history view shared by all judge rollouts.

    Row budget: direct carrier parents (in pool coverage order), then each
    parent's primary-parent chain (round-robin across parents), then each
    parent's already-executed primary children (newest first, round-robin),
    then the 5 lowest / 5 highest finite warm scores as global anchors. A
    run already selected by an earlier category is skipped, so every emitted
    row is distinct and the total never exceeds
    12 parents + 12 ancestors + 12 siblings + 10 anchors = 46 rows.
    """
    records = list(ledger_prefix.get("records", []))
    by_id = {
        str(record.get("run_id")): record
        for record in records
        if isinstance(record, dict)
    }

    direct: list[str] = []
    for entry in pool.get("pool", []):
        for parent in entry["carrier"]["parents"]:
            if parent not in by_id:
                raise ContractError(
                    f"carrier parent {parent} is missing from the ledger prefix"
                )
            if parent not in direct:
                direct.append(parent)

    selected: list[tuple[str, str]] = []
    used: set[str] = set()

    def take(run_id: str, role: str) -> bool:
        if run_id in used or run_id not in by_id:
            return False
        used.add(run_id)
        selected.append((run_id, role))
        return True

    for parent in direct:
        take(parent, "parent")

    chains: list[list[str]] = []
    for parent in direct:
        chain: list[str] = []
        current = parent
        for _ in range(int(limits["ancestor_levels"])):
            parents = [
                str(item) for item in (by_id[current].get("source_run_ids") or [])
            ]
            if not parents or parents[0] not in by_id:
                break
            current = parents[0]
            chain.append(current)
        chains.append(chain)
    ancestor_count = 0
    for level in range(int(limits["ancestor_levels"])):
        if ancestor_count >= int(limits["ancestor_rows"]):
            break
        for chain in chains:
            if ancestor_count >= int(limits["ancestor_rows"]):
                break
            if level < len(chain) and take(chain[level], "ancestor"):
                ancestor_count += 1

    children_of: dict[str, list[str]] = {parent: [] for parent in direct}
    for record in records:
        parents = record.get("source_run_ids") or []
        if parents and str(parents[0]) in children_of:
            children_of[str(parents[0])].append(str(record.get("run_id")))
    per_parent = int(limits["sibling_rows_per_parent"])
    for parent in direct:
        # records iterate in selection order; newest first, bounded per parent.
        children_of[parent] = list(reversed(children_of[parent]))[:per_parent]
    sibling_count = 0
    for offset in range(per_parent):
        if sibling_count >= int(limits["sibling_rows"]):
            break
        for parent in direct:
            if sibling_count >= int(limits["sibling_rows"]):
                break
            children = children_of[parent]
            if offset < len(children) and take(children[offset], "sibling"):
                sibling_count += 1

    finite = [
        (warm, str(record.get("run_id")))
        for record in records
        for warm in [_finite_warm(record)]
        if warm is not None
    ]
    lows = sorted(finite, key=lambda item: (item[0], _rid_key(item[1])))[
        : int(limits["anchor_low"])
    ]
    highs = sorted(
        finite, key=lambda item: (item[0], _rid_key(item[1])), reverse=True
    )[: int(limits["anchor_high"])]
    for _, run_id in lows:
        take(run_id, "anchor_low")
    for _, run_id in highs:
        take(run_id, "anchor_high")

    rows = [
        _a1_row(by_id[run_id], role, by_id, int(limits["idea_chars"]))
        for run_id, role in selected
    ]
    header = (
        "Measured history (lower-is-better warm screening scores; "
        "d = child minus primary parent):"
    )
    lines = [row["line"] for row in rows]
    rendered_text = (
        "\n".join([header, *lines]) if lines else header + "\n(no measured runs yet)"
    )
    return {
        "limits": dict(limits),
        "prefix_record_count": len(records),
        "prefix_digest": records_prefix_digest(records),
        "rows": rows,
        "rendered_text": rendered_text,
    }


def _stable_shuffle(labels: list, generation_seed: str, stage: str) -> list:
    """Deterministic per-stage order; md5's first 8 bytes as the shuffle seed."""
    material = canonical_json({"generation_seed": generation_seed, "stage": stage})
    seed = int.from_bytes(hashlib.md5(material.encode("utf-8")).digest()[:8], "big")
    order = list(labels)
    random.Random(seed).shuffle(order)
    return order


def presented_order(
    pool: dict,
    generation_seed: str,
    stage: str,
    labels: list | None = None,
) -> list:
    """Stable presented order for one stage; `labels` restricts the boundary pool."""
    base = labels if labels is not None else [e["label"] for e in pool["pool"]]
    return _stable_shuffle(base, generation_seed, stage)


def validate_judge_ranking(labels: list, receipt: Any) -> list:
    """A judge receipt is valid iff `ranking` is an exact permutation of labels."""
    if not isinstance(receipt, dict):
        return ["judge receipt must be a JSON object"]
    ranking = receipt.get("ranking")
    if not isinstance(ranking, list) or any(
        not isinstance(item, str) for item in ranking
    ):
        return ["judge receipt ranking must be a list of presented labels"]
    errors = []
    unknown = sorted(set(ranking) - set(labels))
    missing = sorted(set(labels) - set(ranking))
    duplicated = sorted({item for item in ranking if ranking.count(item) > 1})
    if unknown:
        errors.append(f"ranking contains unknown labels {unknown}")
    if duplicated:
        errors.append(f"ranking contains duplicate labels {duplicated}")
    if missing:
        errors.append(f"ranking is missing labels {missing}")
    return errors


def coverage_fallback(pool: dict, slate_size: int, reason: str) -> dict:
    slate = [entry["label"] for entry in pool["pool"][:slate_size]]
    return {
        "status": "fallback",
        "path": "coverage_fallback",
        "slate": slate,
        "reason": reason,
    }


def aggregate_regular(pool: dict, judgments: list) -> dict:
    """Aggregate the regular rollouts.

    Any failed required rollout falls back to coverage order immediately.
    Identical top-2 sets aggregate by mean presented rank (ties: coverage
    order); disjoint sets escalate to the boundary rollout over their union.
    """
    failed = sorted(
        judgment.get("stage") for judgment in judgments
        if judgment.get("status") != "valid"
    )
    if failed:
        return coverage_fallback(
            pool, SLATE_SIZE, f"regular judge failed: {', '.join(failed)}"
        )
    coverage_rank = {entry["label"]: entry["coverage_rank"] for entry in pool["pool"]}
    regular_top2 = {
        judgment["stage"]: list(judgment["ranking"][:SLATE_SIZE])
        for judgment in judgments
    }
    top_sets = {frozenset(labels) for labels in regular_top2.values()}
    if len(top_sets) == 1:
        mean_rank = {}
        for label in next(iter(top_sets)):
            ranks = [judgment["ranking"].index(label) for judgment in judgments]
            mean_rank[label] = sum(ranks) / len(ranks)
        slate = sorted(
            mean_rank, key=lambda label: (mean_rank[label], coverage_rank[label])
        )
        return {
            "status": "selected",
            "path": "consensus",
            "slate": slate,
            "reason": None,
            "regular_top2": regular_top2,
        }
    union = [
        entry["label"]
        for entry in pool["pool"]
        if entry["label"] in set().union(*top_sets)
    ]
    return {
        "status": "boundary_required",
        "path": "boundary_required",
        "boundary_labels": union,
        "reason": None,
        "regular_top2": regular_top2,
    }


def aggregate_boundary(pool: dict, boundary: dict) -> dict:
    """The boundary rollout's top-2, in its ranked order; failure falls back."""
    if boundary.get("status") != "valid":
        return coverage_fallback(pool, SLATE_SIZE, "boundary judge failed")
    return {
        "status": "selected",
        "path": "boundary",
        "slate": list(boundary["ranking"][:SLATE_SIZE]),
        "reason": None,
    }


def _check_stage_artifact(
    artifact: dict, pool_doc: dict, context_digest: str, labels: list | None
) -> None:
    stage = artifact.get("stage")
    if artifact.get("gen_no") != pool_doc.get("gen_no"):
        raise ContractError(f"{stage}: artifact belongs to a different generation")
    if artifact.get("pool_digest") != pool_doc.get("pool_digest"):
        raise ContractError(f"{stage}: artifact was validated against a different pool")
    if artifact.get("context_digest") != context_digest:
        raise ContractError(f"{stage}: artifact was validated against a different context")
    expected = presented_order(
        pool_doc, pool_doc["generation_seed"], stage, labels=labels
    )
    if artifact.get("presented_order") != expected:
        raise ContractError(
            f"{stage}: presented order does not recompute from the generation seed"
        )


def decide_aggregation(pool: dict, stages: dict, context_digest: str) -> tuple[dict, bool]:
    """The aggregation decision tree over already-parsed stage artifacts.

    Degraded paths (empty pool, zero cap, or a slate the pool cannot fill
    past one seat) never consume judge output.  The judged path re-binds each
    stage artifact to this pool/context and its stable presented order before
    aggregating.  Returns ``(decision, judge_called)``.
    """
    budget = pool.get("budget") or {}
    admission_cap = budget.get("admission_cap")
    entries = pool["pool"]
    pool_n = len(entries)
    if pool_n == 0 or admission_cap == 0:
        reason = (
            "no lane produced a proposal"
            if pool_n == 0
            else "admission cap is zero"
        )
        return {
            "status": "selected",
            "path": "degraded_empty_pool" if pool_n == 0 else "admission_cap_zero",
            "slate": [],
            "reason": reason,
        }, False
    if (admission_cap is not None and admission_cap <= 1) or pool_n <= 1:
        return {
            "status": "selected",
            "path": "judge_skipped_cardinality",
            "slate": [entries[0]["label"]],
            "reason": f"admission_cap={admission_cap}, pool_size={pool_n}",
        }, False
    if pool_n <= SLATE_SIZE:
        return {
            "status": "selected",
            "path": "judge_skipped_pool_le_B",
            "slate": [entry["label"] for entry in entries],
            "reason": f"pool_size {pool_n} <= slate size {SLATE_SIZE}",
        }, False
    regular = []
    for stage in REGULAR_STAGES:
        artifact = stages.get(stage)
        if artifact is None:
            raise ContractError(f"missing {stage} judge artifact")
        _check_stage_artifact(artifact, pool, context_digest, labels=None)
        regular.append(artifact)
    decision = aggregate_regular(pool, regular)
    if decision["status"] == "boundary_required":
        boundary = stages.get(BOUNDARY_STAGE)
        if boundary is not None:
            _check_stage_artifact(
                boundary,
                pool,
                context_digest,
                labels=decision["boundary_labels"],
            )
            boundary_decision = aggregate_boundary(pool, boundary)
            boundary_decision["regular_top2"] = decision["regular_top2"]
            boundary_decision["boundary_labels"] = decision["boundary_labels"]
            decision = boundary_decision
    return decision, True


def replay_aggregation(pool: dict, judge: dict) -> list:
    """Recompute the aggregation a judge.json records from pool + stage artifacts.

    An empty result means the recorded aggregation, the judge_called fact, and
    every stage's pool/context/presented-order binding match what the
    deterministic rules produce from the same inputs.  Used at ledger
    admission and by the replay tooling.
    """
    stages = judge.get("stages")
    stages = stages if isinstance(stages, dict) else {}
    try:
        decision, judge_called = decide_aggregation(
            pool, stages, judge.get("context_digest")
        )
    except ContractError as exc:
        return [f"aggregation does not replay: {exc}"]
    expected = {
        key: decision.get(key)
        for key in ("path", "reason", "slate", "regular_top2", "boundary_labels")
        if decision.get(key) is not None or key in ("reason", "slate")
    }
    errors = []
    if judge.get("aggregation") != expected:
        errors.append("judge.json aggregation does not match the replayed decision")
    cardinality = judge.get("cardinality")
    if (
        not isinstance(cardinality, dict)
        or cardinality.get("judge_called") is not judge_called
    ):
        errors.append(
            "judge.json cardinality.judge_called does not match the replayed path"
        )
    return errors


def candidate_id(point_id_value: str, op: str, parents: list) -> str:
    return digest(
        {"point_id": point_id_value, "op": op, "parents": [str(p) for p in parents]}
    )


def build_manifest(
    pool: dict,
    judge: dict,
    budget: dict,
    reserved_run_ids: list,
    donor_snapshot: dict | None = None,
) -> dict:
    """The immutable generation manifest; `generation_id` is its content id.

    ``donor_snapshot`` is the generation's donor binding (design §3.1) and is
    present only under the transfer scheduler policy; older policies pass
    None and their manifests stay byte-identical.
    """
    if donor_snapshot is not None:
        errors = _donor_binding_shape_errors(donor_snapshot)
        if errors:
            raise ContractError(
                "invalid donor_snapshot binding: " + "; ".join(errors)
            )
    by_label = {entry["label"]: entry for entry in pool["pool"]}
    slate_labels = list(judge["aggregation"].get("slate") or [])
    if len(reserved_run_ids) != len(slate_labels):
        raise ContractError(
            f"reserved run ids ({len(reserved_run_ids)}) must equal the slate "
            f"size ({len(slate_labels)})"
        )
    slate = []
    for slot, label in enumerate(slate_labels):
        entry = by_label.get(label)
        if entry is None:
            raise ContractError(f"slate label {label} is not in the pool")
        carrier = entry["carrier"]
        slate.append(
            {
                "slot": slot,
                "run_id": reserved_run_ids[slot],
                "label": label,
                "candidate_id": candidate_id(
                    entry["point_id"], carrier["op"], carrier["parents"]
                ),
                "point_id": entry["point_id"],
                "point": entry["point"],
                "carrier": carrier,
                "carrier_alternatives": entry["carrier_alternatives"],
            }
        )
    core = {
        "schema_version": SCHEMA_VERSION,
        "gen_no": pool["gen_no"],
        "ledger_snapshot": pool["ledger_snapshot"],
        "policy": {
            "name": "judged_slate",
            "config": {
                "pool_size": pool["pool_size"],
                "slate_size": SLATE_SIZE,
                "regular_rollouts": len(REGULAR_STAGES),
            },
        },
        "budget": budget,
        "cardinality": {
            "pool_target": pool["pool_size"],
            "pool_actual": len(pool["pool"]),
            "slate_size": len(slate),
        },
        "lanes_digest": pool["lanes_digest"],
        "pool_digest": pool["pool_digest"],
        "context_digest": judge["context_digest"],
        "judge_digest": digest(judge),
        "proposal_set_revisions": pool["proposal_set_revisions"],
        "aggregation": judge["aggregation"],
        "reserved_run_ids": list(reserved_run_ids),
        "slate": slate,
        "judge_cost": judge.get("judge_cost"),
    }
    if donor_snapshot is not None:
        core["donor_snapshot"] = donor_snapshot
    return {**core, "generation_id": digest(core)}


# ---------- generation donor binding (transfer scheduler policy, design §3.1) ----------

TRANSFER_SCHEDULER_POLICY = "anchor_transfer_challenger_v1"
# Mirrors tune_tools.GLOBAL_DONOR_TRANSFER_FILENAME (tuners is a script-level
# package; importing it here would drag its sys.path setup into replay).
DONOR_RECEIPT_FILENAME = "_global_donor_transfer.json"


def _donor_binding_shape_errors(binding) -> list:
    """Structural contract of the manifest's ``donor_snapshot`` binding."""
    if not isinstance(binding, dict):
        return ["donor_snapshot binding must be an object"]
    status = binding.get("status")
    if status not in ("bound", "no_donor"):
        return ["donor_snapshot.status must be 'bound' or 'no_donor'"]
    if status == "no_donor":
        if any(
            binding.get(key) is not None
            for key in ("snapshot_id", "path", "digest")
        ):
            return ["a no_donor binding must carry null snapshot_id/path/digest"]
        return []
    errors = []
    snapshot_id = binding.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.startswith("donor-"):
        errors.append("a bound donor_snapshot needs a donor- snapshot_id")
    path = binding.get("path")
    if (
        not isinstance(path, str)
        or not path
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        errors.append("a bound donor_snapshot needs a run-relative path")
    binding_digest = binding.get("digest")
    if not isinstance(binding_digest, str) or not binding_digest.startswith(
        "sha256:"
    ):
        errors.append("a bound donor_snapshot needs a sha256: byte digest")
    return errors


def no_donor_binding() -> dict:
    """The explicit no-donor binding (the normal pre-anchor state)."""
    return {"status": "no_donor", "snapshot_id": None, "path": None, "digest": None}


def donor_binding_from_snapshot(snapshot_path: Path, run_dir: Path) -> dict:
    """Verify a donor snapshot artifact and build the manifest binding for it.

    ``snapshot_path`` may be absolute or run-dir-relative; the recorded
    binding path is always run-relative so the manifest survives a moved
    run directory.  A corrupt or inconsistent snapshot is a hard error
    (design §8): the caller blocks the run rather than degrading to the
    ordinary warm pool.
    """
    from scheduler import donor as donor_snapshots  # noqa: PLC0415

    run_dir = Path(run_dir).resolve()
    path = Path(snapshot_path)
    if not path.is_absolute():
        path = run_dir / path
    path = path.resolve()
    try:
        relative = path.relative_to(run_dir)
    except ValueError:
        raise ContractError(
            f"donor snapshot {path} is not inside the run directory {run_dir}"
        ) from None
    snapshot = donor_snapshots.load_donor_snapshot(path)
    return {
        "status": "bound",
        "snapshot_id": snapshot["snapshot_id"],
        "path": relative.as_posix(),
        "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# ---------- digest recompute helpers (replay/tamper checks; Patch D reuses) ----------


def pool_payload(pool_doc: dict) -> dict:
    return {
        key: value
        for key, value in pool_doc.items()
        if key not in {"pool_digest", "generation_seed"}
    }


def recompute_pool_digest(pool_doc: dict) -> str:
    return digest(pool_payload(pool_doc))


def recompute_generation_seed(pool_doc: dict) -> str:
    return digest(
        {
            "gen_no": pool_doc["gen_no"],
            "ledger_snapshot": pool_doc["ledger_snapshot"],
            "pool_digest": recompute_pool_digest(pool_doc),
        }
    )


def recompute_generation_id(manifest: dict) -> str:
    return digest(
        {key: value for key, value in manifest.items() if key != "generation_id"}
    )


def recompute_candidate_id(slot: dict) -> str:
    carrier = slot["carrier"]
    return candidate_id(slot["point_id"], carrier["op"], carrier["parents"])


def verify_manifest(manifest: dict, pool: dict, context: dict, judge: dict) -> list:
    """Recompute every digest binding between the four generation artifacts."""
    errors = []
    if manifest.get("generation_id") != recompute_generation_id(manifest):
        errors.append("generation_id does not match the manifest core")
    pool_digest = recompute_pool_digest(pool)
    if pool.get("pool_digest") != pool_digest:
        errors.append("pool_digest does not match the pool payload")
    if pool.get("generation_seed") != recompute_generation_seed(pool):
        errors.append("generation_seed does not match gen_no/snapshot/pool digest")
    context_digest = digest(context)
    judge_digest = digest(judge)
    if manifest.get("pool_digest") != pool_digest:
        errors.append("manifest pool_digest does not match pool.json")
    if manifest.get("lanes_digest") != pool.get("lanes_digest"):
        errors.append("manifest lanes_digest does not match pool.json")
    if manifest.get("context_digest") != context_digest:
        errors.append("manifest context_digest does not match context.json")
    if manifest.get("judge_digest") != judge_digest:
        errors.append("manifest judge_digest does not match judge.json")
    if judge.get("pool_digest") != pool_digest:
        errors.append("judge.json pool_digest does not match pool.json")
    if judge.get("context_digest") != context_digest:
        errors.append("judge.json context_digest does not match context.json")
    for slot in manifest.get("slate", []):
        if slot.get("candidate_id") != recompute_candidate_id(slot):
            errors.append(f"slot {slot.get('slot')} candidate_id does not recompute")
    return errors


# ============================ CLI facade (I/O) ============================


def _load_object(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def write_json_atomic(path: Path, value: dict) -> None:
    """Temp file + os.replace; provisional artifacts may be overwritten."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2) + "\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_judged_slate_config(ledger_path: Path) -> dict:
    """The run's `judged_slate` config section (defaults when unconfigured)."""
    path = Path(ledger_path).parent / "framework_cfg.json"
    if not path.is_file():
        return {"pool_size": DEFAULT_POOL_SIZE}
    section = read_framework_cfg(path).get("judged_slate") or {}
    return {"pool_size": section.get("pool_size", DEFAULT_POOL_SIZE)}


def _run_scheduler_policy(run_dir: Path) -> str | None:
    """The run's frozen tuner.scheduler_policy (None when unconfigured)."""
    path = Path(run_dir) / "framework_cfg.json"
    if not path.is_file():
        return None
    tuner = read_framework_cfg(path).get("tuner")
    if not isinstance(tuner, dict):
        return None
    policy = tuner.get("scheduler_policy")
    return policy if isinstance(policy, str) else None


def _check_lanes_document(lanes_doc: dict) -> None:
    if lanes_doc.get("schema_version") != 1:
        raise ContractError("lanes document schema_version must be 1")
    gen_no = lanes_doc.get("gen_no")
    if not isinstance(gen_no, int) or isinstance(gen_no, bool) or gen_no < 1:
        raise ContractError("lanes document gen_no must be a positive integer")
    snapshot = lanes_doc.get("ledger_snapshot")
    if not isinstance(snapshot, dict):
        raise ContractError("lanes document ledger_snapshot must be an object")
    record_count = snapshot.get("record_count")
    if (
        not isinstance(record_count, int)
        or isinstance(record_count, bool)
        or record_count < 0
    ):
        raise ContractError("ledger_snapshot.record_count must be a non-negative integer")
    if not isinstance(snapshot.get("records_digest"), str):
        raise ContractError("ledger_snapshot.records_digest must be a string")
    if not isinstance(snapshot.get("experience"), dict):
        raise ContractError("ledger_snapshot.experience must be an object")
    if not isinstance(lanes_doc.get("budget"), dict):
        raise ContractError("lanes document budget must be an object")
    lanes = lanes_doc.get("lanes")
    if not isinstance(lanes, list):
        raise ContractError("lanes document lanes must be a list")
    seen = set()
    for lane in lanes:
        if not isinstance(lane, dict):
            raise ContractError("each lane must be an object")
        lane_id = lane.get("lane_id")
        if not isinstance(lane_id, str) or not lane_id:
            raise ContractError("each lane needs a non-empty lane_id")
        if lane_id in seen:
            raise ContractError(f"duplicate lane_id {lane_id}")
        seen.add(lane_id)
        if lane.get("op") not in OP_PRIORITY:
            raise ContractError(f"lane {lane_id} op must be fresh, improve, or crossover")
        parents = lane.get("parents")
        if not isinstance(parents, list) or any(
            not isinstance(parent, str) for parent in parents
        ):
            raise ContractError(f"lane {lane_id} parents must be a string list")
        value = lane.get("lane_value")
        if value is not None and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ContractError(f"lane {lane_id} lane_value must be a finite number or null")


def _verify_prefix(lanes_doc: dict, ledger: dict) -> list:
    """Bind the lanes snapshot to the ledger re-read at construct time.

    Returns the prefix records. The generation-start prefix must still match
    exactly (count, selection-safe digest, experience and search-space
    revisions), and it may not contain non-terminal records: a pending record
    is a candidate whose measurement has not completed, not history.
    """
    snapshot = lanes_doc["ledger_snapshot"]
    records = ledger.get("records", [])
    if not isinstance(records, list):
        raise ContractError("ledger.records must be a list")
    if len(records) != snapshot["record_count"]:
        raise ContractError(
            f"ledger prefix grew or shrank: {len(records)} records, "
            f"lanes snapshot expected {snapshot['record_count']}"
        )
    if records_prefix_digest(records) != snapshot["records_digest"]:
        raise ContractError(
            "ledger prefix digest mismatch: records changed since lanes were decided"
        )
    if experience_receipt(ledger) != snapshot["experience"]:
        raise ContractError("ledger experience snapshot changed since lanes were decided")
    if search_space_state_revision(ledger) != snapshot.get("search_space_state_revision"):
        raise ContractError(
            "ledger search_space_state revision changed since lanes were decided"
        )
    non_terminal = [
        str(record.get("run_id"))
        for record in records
        if record.get("status") not in LIFECYCLE_TERMINAL_STATUSES
    ]
    if non_terminal:
        raise ContractError(
            f"generation prefix contains non-terminal records {non_terminal}; "
            "resolve pending candidates before constructing a slate"
        )
    return records


def _load_proposal_sets(lanes_doc: dict, proposals_dir: Path, space: dict) -> dict:
    """Load and bind every lane's proposal set; shared by construct and replay."""
    lanes_by_id = {lane["lane_id"]: lane for lane in lanes_doc["lanes"]}
    proposal_sets: dict[str, dict] = {}
    proposals_dir = Path(proposals_dir)
    if proposals_dir.is_dir():
        for file in sorted(proposals_dir.glob("lane-*.json")):
            lane_id = file.stem
            if lane_id not in lanes_by_id:
                raise ContractError(f"{file}: no matching lane in the lanes document")
            proposal_set = _load_object(file)
            errors = validate_proposal_set(proposal_set)
            if errors:
                raise ContractError(
                    f"{file}: invalid proposal set: " + "; ".join(errors)
                )
            proposal_sets[lane_id] = proposal_set
    for lane_id, proposal_set in proposal_sets.items():
        lane = lanes_by_id[lane_id]
        action = proposal_set.get("action")
        if action != {"op": lane["op"], "parents": lane["parents"]}:
            raise ContractError(
                f"proposal set for {lane_id} was built for action {action}, "
                f"not the lane's ({lane['op']}, {lane['parents']})"
            )
        if proposal_set.get("space") != space:
            raise ContractError(
                f"proposal set for {lane_id} was built against a different search space"
            )
        if proposal_set.get("search_space_state_revision") != lanes_doc[
            "ledger_snapshot"
        ].get("search_space_state_revision"):
            raise ContractError(
                f"proposal set for {lane_id} was built against a different "
                "search_space_state revision"
            )
    return proposal_sets


def cmd_construct(args: argparse.Namespace) -> int:
    lanes_doc = _load_object(args.lanes)
    _check_lanes_document(lanes_doc)
    ledger_path = Path(args.ledger)
    ledger = _load_object(ledger_path) if ledger_path.exists() else {"records": []}
    records = _verify_prefix(lanes_doc, ledger)
    registry = load_registry(args.background)
    space = space_receipt(registry)

    proposal_sets = _load_proposal_sets(lanes_doc, args.proposals_dir, space)

    pool_size = args.pool_size
    if pool_size is None:
        pool_size = load_judged_slate_config(ledger_path)["pool_size"]
    snapshot = lanes_doc["ledger_snapshot"]
    core = build_pool(lanes_doc["lanes"], proposal_sets, registry, pool_size)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "gen_no": lanes_doc["gen_no"],
        "ledger_snapshot": snapshot,
        "budget": lanes_doc["budget"],
        "pool_size": pool_size,
        "space": space,
        "search_space_state_revision": snapshot.get("search_space_state_revision"),
        "lanes": lanes_doc["lanes"],
        "lanes_digest": digest(lanes_doc),
        "proposal_set_revisions": core["proposal_set_revisions"],
        "lanes_without_proposals": core["lanes_without_proposals"],
        "pool": core["pool"],
    }
    pool_digest = digest(payload)
    generation_seed = digest(
        {
            "gen_no": lanes_doc["gen_no"],
            "ledger_snapshot": snapshot,
            "pool_digest": pool_digest,
        }
    )
    pool_doc = {
        **payload,
        "pool_digest": pool_digest,
        "generation_seed": generation_seed,
    }
    write_json_atomic(args.pool_output, pool_doc)

    context_core = build_a1_context({"records": records}, pool_doc)
    context_doc = {
        "schema_version": SCHEMA_VERSION,
        "gen_no": lanes_doc["gen_no"],
        "ledger_snapshot": snapshot,
        **context_core,
    }
    write_json_atomic(args.context_output, context_doc)
    print(
        json.dumps(
            {
                "ok": True,
                "gen_no": lanes_doc["gen_no"],
                "pool_size": len(core["pool"]),
                "pool_digest": pool_digest,
                "generation_seed": generation_seed,
                "lanes_without_proposals": core["lanes_without_proposals"],
            },
            separators=(",", ":"),
        )
    )
    return 0


def _render_candidate_block(candidate: dict) -> str:
    summary = candidate["summary"]
    carrier = summary["carrier"]
    lines = [f"Candidate {candidate['label']} ({candidate['point_id']})"]
    if carrier["op"] == "fresh":
        lines.append("Carrier if selected: fresh (implemented from scratch)")
    else:
        lines.append(
            f"Carrier if selected: op={carrier['op']}, "
            f"parents={','.join(carrier['parents'])}"
        )
    for parent_diff in summary["parent_diffs"]:
        parent = parent_diff["parent_run_id"]
        changes = parent_diff["changes"]
        if not changes:
            lines.append(f"Changes vs parent {parent}: none (same semantic point)")
            continue
        rendered = "; ".join(
            f"{change['dimension_id']}: "
            f"{change.get('from_hypothesis_id') or '-'} -> "
            f"{change.get('to_hypothesis_id') or '-'}"
            for change in changes
        )
        lines.append(f"Changes vs parent {parent}: {rendered}")
    hypotheses = summary["hypotheses"]
    if hypotheses:
        lines.append("Non-baseline hypotheses:")
        for hypothesis in hypotheses:
            lines.append(
                f"- {hypothesis['dimension_id']} / {hypothesis['hypothesis_id']}: "
                f"{hypothesis['title']} — {hypothesis['claim']}"
            )
    deprioritized = summary["deprioritized_hypotheses"]
    if deprioritized:
        lines.append(f"Deprioritized by runtime evidence: {', '.join(deprioritized)}")
    return "\n".join(lines)


def cmd_prepare_judge(args: argparse.Namespace) -> int:
    pool_doc = _load_object(args.pool)
    context_doc = _load_object(args.context)
    if context_doc.get("gen_no") != pool_doc.get("gen_no"):
        raise ContractError("pool and context belong to different generations")
    stage = args.stage
    labels = None
    if stage == BOUNDARY_STAGE:
        labels = [item.strip() for item in (args.labels or "").split(",") if item.strip()]
        if not labels:
            raise ContractError("the boundary stage requires --labels")
        known = {entry["label"] for entry in pool_doc["pool"]}
        unknown = sorted(set(labels) - known)
        if unknown:
            raise ContractError(f"boundary labels not in the pool: {unknown}")
    order = presented_order(
        pool_doc, pool_doc["generation_seed"], stage, labels=labels
    )
    by_label = {entry["label"]: entry for entry in pool_doc["pool"]}
    candidates = [
        {
            "label": entry["label"],
            "point_id": entry["point_id"],
            "summary": entry["summary"],
        }
        for entry in (by_label[label] for label in order)
    ]
    task_brief = None
    if args.task_brief:
        task_brief = Path(args.task_brief).read_text().strip()
    parts = []
    if task_brief:
        parts.append(task_brief)
    parts.append(context_doc["rendered_text"])
    parts.append("\n\n".join(_render_candidate_block(c) for c in candidates))
    doc = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "gen_no": pool_doc["gen_no"],
        "generation_seed": pool_doc["generation_seed"],
        "pool_digest": pool_doc["pool_digest"],
        "context_digest": digest(context_doc),
        "presented_order": order,
        "candidates": candidates,
        "history_text": context_doc["rendered_text"],
        "prompt_text": "\n\n".join(parts) + "\n",
    }
    write_json_atomic(args.output, doc)
    print(
        json.dumps(
            {"ok": True, "stage": stage, "presented_order": order},
            separators=(",", ":"),
        )
    )
    return 0


def cmd_validate_judge(args: argparse.Namespace) -> int:
    input_doc = _load_object(args.input)
    receipt = _load_object(args.receipt) if args.receipt else None
    if receipt is None:
        errors = ["no judge receipt was produced"]
    else:
        errors = validate_judge_ranking(input_doc["presented_order"], receipt)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "stage": input_doc["stage"],
        "gen_no": input_doc["gen_no"],
        "generation_seed": input_doc["generation_seed"],
        "pool_digest": input_doc["pool_digest"],
        "context_digest": input_doc["context_digest"],
        "presented_order": input_doc["presented_order"],
        "status": "failed" if errors else "valid",
        "ranking": None if errors else list(receipt["ranking"]),
        "rationale": None if errors else receipt.get("rationale"),
        "errors": errors,
        "receipt_path": str(args.receipt) if args.receipt else None,
        "session_id": args.session_id,
        "model": args.model,
    }
    write_json_atomic(args.output, artifact)
    print(
        json.dumps(
            {"ok": not errors, "stage": input_doc["stage"], "status": artifact["status"], "errors": errors},
            separators=(",", ":"),
        )
    )
    return 0 if not errors else 1


def _judge_cost(stages: dict) -> dict:
    return {
        "session_ids": {
            stage: artifact["session_id"]
            for stage, artifact in stages.items()
            if artifact.get("session_id")
        },
        "models": {
            stage: artifact["model"]
            for stage, artifact in stages.items()
            if artifact.get("model")
        },
    }


def cmd_aggregate(args: argparse.Namespace) -> int:
    pool_doc = _load_object(args.pool)
    context_doc = _load_object(args.context)
    if context_doc.get("gen_no") != pool_doc.get("gen_no"):
        raise ContractError("pool and context belong to different generations")
    context_digest = digest(context_doc)
    budget = pool_doc.get("budget") or {}
    admission_cap = budget.get("admission_cap")
    entries = pool_doc["pool"]
    pool_n = len(entries)

    judgments_dir = Path(args.judgments_dir)
    stages = {}
    for stage in STAGES:
        path = judgments_dir / f"{stage}.json"
        if path.is_file():
            stages[stage] = _load_object(path)

    try:
        decision, judge_called = decide_aggregation(pool_doc, stages, context_digest)
    except ContractError as exc:
        raise ContractError(f"{exc} in {judgments_dir}") from exc

    aggregation = {
        key: decision.get(key)
        for key in ("path", "reason", "slate", "regular_top2", "boundary_labels")
        if decision.get(key) is not None or key in ("reason", "slate")
    }
    judge_doc = {
        "schema_version": SCHEMA_VERSION,
        "gen_no": pool_doc["gen_no"],
        "generation_seed": pool_doc["generation_seed"],
        "pool_digest": pool_doc["pool_digest"],
        "context_digest": context_digest,
        "budget": {
            "objective_remaining": budget.get("objective_remaining"),
            "admission_cap": admission_cap,
        },
        "cardinality": {
            "pool_target": pool_doc["pool_size"],
            "pool_actual": pool_n,
            "judge_called": judge_called,
        },
        # Degraded paths never call the judge; leftover artifacts from an
        # aborted pre-manifest attempt of the same gen_no must not leak in.
        "stages": stages if judge_called else {},
        "aggregation": aggregation,
        "judge_cost": _judge_cost(stages) if judge_called else {"session_ids": {}, "models": {}},
    }
    write_json_atomic(args.output, judge_doc)

    status = decision["status"]
    if status == "boundary_required":
        payload = {
            "status": "boundary_required",
            "boundary_labels": decision["boundary_labels"],
        }
    elif status == "fallback":
        payload = {
            "status": "fallback",
            "slate": decision["slate"],
            "reason": decision.get("reason"),
        }
    else:
        payload = {
            "status": "selected",
            "slate": decision["slate"],
            "path": aggregation["path"],
        }
    print(json.dumps(payload, separators=(",", ":")))
    return 0


def cmd_build_manifest(args: argparse.Namespace) -> int:
    lanes_doc = _load_object(args.lanes)
    pool_doc = _load_object(args.pool)
    context_doc = _load_object(args.context)
    judge_doc = _load_object(args.judge)
    if not (
        lanes_doc.get("gen_no")
        == pool_doc.get("gen_no")
        == context_doc.get("gen_no")
        == judge_doc.get("gen_no")
    ):
        raise ContractError("generation artifacts disagree on gen_no")
    if digest(lanes_doc) != pool_doc.get("lanes_digest"):
        raise ContractError("lanes.json no longer matches the pool's lanes_digest")
    if judge_doc.get("pool_digest") != pool_doc.get("pool_digest"):
        raise ContractError("judge.json was aggregated against a different pool")
    if judge_doc.get("context_digest") != digest(context_doc):
        raise ContractError("judge.json was aggregated against a different context")
    if not judge_doc.get("aggregation", {}).get("slate") and judge_doc.get(
        "aggregation", {}
    ).get("path") == "boundary_required":
        raise ContractError("aggregation is still waiting for the boundary judge")
    reserved = [
        item.strip() for item in args.reserved_run_ids.split(",") if item.strip()
    ]
    # The generation layout pins the run directory two levels above the
    # manifest (<run_dir>/.semantic/gen-NNNN/generation.json); the frozen
    # scheduler policy there decides whether a donor binding is required,
    # and an explicit binding anywhere else is meaningless.
    run_dir = Path(args.output).resolve().parents[2]
    scheduler_policy = _run_scheduler_policy(run_dir)
    if scheduler_policy == TRANSFER_SCHEDULER_POLICY:
        if args.donor_snapshot is None and not args.no_donor:
            raise ContractError(
                f"scheduler_policy {TRANSFER_SCHEDULER_POLICY!r} requires an "
                "explicit donor binding: --donor-snapshot <artifact> or "
                "--no-donor"
            )
        donor_snapshot = (
            no_donor_binding()
            if args.no_donor
            else donor_binding_from_snapshot(args.donor_snapshot, run_dir)
        )
    else:
        if args.donor_snapshot is not None or args.no_donor:
            raise ContractError(
                "--donor-snapshot/--no-donor require the run's "
                f"scheduler_policy to be {TRANSFER_SCHEDULER_POLICY!r} "
                f"(this run has {scheduler_policy!r})"
            )
        donor_snapshot = None
    manifest = build_manifest(
        pool_doc, judge_doc, pool_doc.get("budget") or {}, reserved,
        donor_snapshot=donor_snapshot,
    )
    write_json_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "generation_id": manifest["generation_id"],
                "slate": [slot["run_id"] for slot in manifest["slate"]],
            },
            separators=(",", ":"),
        )
    )
    return 0


# ============================ replay ============================
#
# `slate.py replay` re-derives every deterministic decision of each recorded
# generation from the artifacts on disk and the ledger prefix, and reports any
# disagreement.  It never calls the judge and never scores whether the judge
# chose well; it only verifies that the system executed the recorded rules.
# A green run therefore means "the artifacts and the ledger are one consistent
# execution", nothing more.


def _pool_diff(rebuilt: list, stored: list) -> str:
    """Localize the first pool mismatch for the replay report."""
    if len(rebuilt) != len(stored):
        return f"stored pool has {len(stored)} entries, recompute gives {len(rebuilt)}"
    for want, got in zip(rebuilt, stored):
        if want != got:
            fields = sorted(
                key for key in set(want) | set(got) if want.get(key) != got.get(key)
            )
            return f"pool entry {got.get('label')} differs in {fields}"
    return "pool differs"


def _coverage_order_subset(pool_doc: dict, labels: list) -> list:
    """The boundary-stage label list in canonical (coverage) order."""
    wanted = set(labels)
    return [entry["label"] for entry in pool_doc["pool"] if entry["label"] in wanted]


def _replay_pool_errors(
    pool_doc: dict, lanes_doc: dict, proposal_sets: dict, registry: dict
) -> list:
    """Seating, coverage order, carriers, and the lane/proposal bindings."""
    errors = []
    if digest(lanes_doc) != pool_doc.get("lanes_digest"):
        errors.append("pool lanes_digest does not match lanes.json")
    try:
        core = build_pool(
            lanes_doc["lanes"], proposal_sets, registry, pool_doc["pool_size"]
        )
    except (ContractError, KeyError, TypeError) as exc:
        return errors + [f"pool does not rebuild from lanes + proposals: {exc}"]
    if core["pool"] != pool_doc.get("pool"):
        errors.append("pool does not recompute: " + _pool_diff(core["pool"], pool_doc.get("pool") or []))
    if core["proposal_set_revisions"] != pool_doc.get("proposal_set_revisions"):
        errors.append("pool proposal_set_revisions do not match the proposal sets")
    if core["lanes_without_proposals"] != pool_doc.get("lanes_without_proposals"):
        errors.append("pool lanes_without_proposals does not recompute")
    if recompute_pool_digest(pool_doc) != pool_doc.get("pool_digest"):
        errors.append("pool_digest does not match the pool payload")
    if recompute_generation_seed(pool_doc) != pool_doc.get("generation_seed"):
        errors.append("generation_seed does not match gen_no/snapshot/pool digest")
    return errors


def _replay_context_errors(context_doc: dict, pool_doc: dict, prefix: list) -> list:
    """The A1 view: prefix digest, row selection/dedup/caps, rendered text."""
    errors = []
    snapshot = pool_doc["ledger_snapshot"]
    if context_doc.get("gen_no") != pool_doc.get("gen_no"):
        errors.append("context.json belongs to a different generation than pool.json")
    if context_doc.get("ledger_snapshot") != snapshot:
        errors.append("context.json ledger_snapshot differs from pool.json")
    if context_doc.get("prefix_record_count") != snapshot.get("record_count"):
        errors.append("context prefix_record_count does not match the snapshot")
    prefix_digest = records_prefix_digest(prefix)
    if prefix_digest != snapshot.get("records_digest"):
        errors.append("the ledger prefix no longer digests to the snapshot")
    if context_doc.get("prefix_digest") != prefix_digest:
        errors.append("context prefix_digest does not match the ledger prefix")
    try:
        rebuilt = build_a1_context({"records": prefix}, pool_doc)
    except ContractError as exc:
        return errors + [f"A1 context does not rebuild: {exc}"]
    if rebuilt["rows"] != context_doc.get("rows"):
        errors.append("A1 rows do not recompute from the ledger prefix")
    if rebuilt["rendered_text"] != context_doc.get("rendered_text"):
        errors.append("A1 rendered_text does not recompute from the rows")
    if rebuilt["limits"] != context_doc.get("limits"):
        errors.append("A1 limits differ from the recorded ones")
    return errors


def _replay_stage_input_errors(
    input_doc: dict, pool_doc: dict, context_doc: dict
) -> list:
    """One prepared judge input: order, candidate payloads, embedded history."""
    stage = input_doc.get("stage")
    if stage not in STAGES:
        return [f"stage input has unknown stage {stage!r}"]
    errors = []
    if input_doc.get("gen_no") != pool_doc.get("gen_no"):
        errors.append(f"{stage}: input belongs to a different generation")
    if input_doc.get("generation_seed") != pool_doc.get("generation_seed"):
        errors.append(f"{stage}: input generation_seed differs from pool.json")
    if input_doc.get("pool_digest") != pool_doc.get("pool_digest"):
        errors.append(f"{stage}: input was prepared against a different pool")
    if input_doc.get("context_digest") != digest(context_doc):
        errors.append(f"{stage}: input was prepared against a different context")
    order = input_doc.get("presented_order")
    if not isinstance(order, list):
        errors.append(f"{stage}: input presented_order must be a list")
        order = []
    labels = (
        None
        if stage in REGULAR_STAGES
        else _coverage_order_subset(pool_doc, order)
    )
    expected = presented_order(
        pool_doc, pool_doc["generation_seed"], stage, labels=labels
    )
    if order != expected:
        errors.append(f"{stage}: input presented order does not recompute")
    candidates = input_doc.get("candidates")
    by_label = {entry["label"]: entry for entry in pool_doc["pool"]}
    if not isinstance(candidates, list) or [
        candidate.get("label") if isinstance(candidate, dict) else None
        for candidate in candidates
    ] != order:
        errors.append(f"{stage}: input candidates do not follow the presented order")
        candidates = []
    for candidate in candidates:
        entry = by_label.get(candidate.get("label"))
        if entry is None:
            errors.append(f"{stage}: candidate {candidate.get('label')} is not in the pool")
            continue
        if candidate.get("point_id") != entry["point_id"]:
            errors.append(f"{stage}: candidate {entry['label']} point_id differs from the pool")
        if candidate.get("summary") != entry["summary"]:
            errors.append(f"{stage}: candidate {entry['label']} summary differs from the pool")
    if input_doc.get("history_text") != context_doc.get("rendered_text"):
        errors.append(f"{stage}: input history_text differs from context.json")
    prompt = input_doc.get("prompt_text")
    if not isinstance(prompt, str) or context_doc.get("rendered_text", "") not in prompt:
        errors.append(f"{stage}: prompt_text does not embed the A1 history")
    else:
        for candidate in candidates:
            if _render_candidate_block(candidate) not in prompt:
                errors.append(
                    f"{stage}: prompt_text lacks candidate {candidate.get('label')}'s block"
                )
    return errors


def _replay_stage_artifact_errors(
    artifact: dict, pool_doc: dict, context_digest: str
) -> list:
    """One validated stage artifact: pool/context binding and ranking shape."""
    stage = artifact.get("stage")
    if stage not in STAGES:
        return [f"stage artifact has unknown stage {stage!r}"]
    order = artifact.get("presented_order")
    labels = (
        None
        if stage in REGULAR_STAGES
        else _coverage_order_subset(pool_doc, order if isinstance(order, list) else [])
    )
    errors = []
    try:
        _check_stage_artifact(artifact, pool_doc, context_digest, labels)
    except ContractError as exc:
        errors.append(str(exc))
    status = artifact.get("status")
    if status == "valid":
        errors += [
            f"{stage}: {error}"
            for error in validate_judge_ranking(
                order if isinstance(order, list) else [],
                {"ranking": artifact.get("ranking")},
            )
        ]
    elif status == "failed":
        if artifact.get("ranking") is not None:
            errors.append(f"{stage}: a failed artifact must not carry a ranking")
    else:
        errors.append(f"{stage}: unknown status {status!r}")
    return errors


def _replay_judge_errors(
    judge_doc: dict,
    pool_doc: dict,
    context_doc: dict,
    disk_stages: dict,
    manifest_present: bool,
) -> list:
    """judge.json: recorded facts, embedded-vs-disk stages, aggregation replay."""
    errors = []
    if judge_doc.get("gen_no") != pool_doc.get("gen_no"):
        errors.append("judge.json belongs to a different generation than pool.json")
    if judge_doc.get("generation_seed") != pool_doc.get("generation_seed"):
        errors.append("judge.json generation_seed differs from pool.json")
    budget = pool_doc.get("budget") or {}
    if judge_doc.get("budget") != {
        "objective_remaining": budget.get("objective_remaining"),
        "admission_cap": budget.get("admission_cap"),
    }:
        errors.append("judge.json budget facts do not match pool.json")
    cardinality = judge_doc.get("cardinality") or {}
    if (
        cardinality.get("pool_target") != pool_doc.get("pool_size")
        or cardinality.get("pool_actual") != len(pool_doc["pool"])
    ):
        errors.append("judge.json cardinality facts do not match pool.json")
    errors += replay_aggregation(pool_doc, judge_doc)
    embedded = judge_doc.get("stages") or {}
    if not cardinality.get("judge_called") and embedded:
        errors.append("judge.json embeds stages although the judge was never called")
    for stage, artifact in embedded.items():
        disk = disk_stages.get(stage)
        if disk is None:
            errors.append(f"judge.json embeds {stage} but judgments/{stage}.json is missing")
        elif disk != artifact:
            errors.append(f"judgments/{stage}.json differs from the embedded stage artifact")
    if manifest_present and (judge_doc.get("aggregation") or {}).get(
        "path"
    ) == "boundary_required":
        errors.append("the manifest exists but the aggregation still awaits a boundary judge")
    return errors


def _replay_manifest_errors(
    manifest: dict, pool_doc: dict, context_doc: dict, judge_doc: dict, gen_no: int
) -> list:
    """The digest chain plus every manifest fact that must equal its sources."""
    errors = verify_manifest(manifest, pool_doc, context_doc, judge_doc)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema_version must be 1")
    if manifest.get("gen_no") != gen_no:
        errors.append(f"manifest gen_no {manifest.get('gen_no')} does not match gen-{gen_no:04d}")
    policy = manifest.get("policy") or {}
    if policy.get("name") != "judged_slate":
        errors.append("manifest policy.name must be judged_slate")
    if policy.get("config") != {
        "pool_size": pool_doc["pool_size"],
        "slate_size": SLATE_SIZE,
        "regular_rollouts": len(REGULAR_STAGES),
    }:
        errors.append("manifest policy.config does not match the frozen protocol")
    if manifest.get("budget") != pool_doc.get("budget"):
        errors.append("manifest budget does not match pool.json")
    slate_slots = manifest.get("slate") if isinstance(manifest.get("slate"), list) else []
    if manifest.get("cardinality") != {
        "pool_target": pool_doc["pool_size"],
        "pool_actual": len(pool_doc["pool"]),
        "slate_size": len(slate_slots),
    }:
        errors.append("manifest cardinality does not match pool.json and the slate")
    if manifest.get("ledger_snapshot") != pool_doc.get("ledger_snapshot"):
        errors.append("manifest ledger_snapshot differs from pool.json")
    if manifest.get("proposal_set_revisions") != pool_doc.get("proposal_set_revisions"):
        errors.append("manifest proposal_set_revisions differ from pool.json")
    if manifest.get("reserved_run_ids") != [slot.get("run_id") for slot in slate_slots]:
        errors.append("manifest reserved_run_ids do not equal the slate's run ids")
    by_label = {entry["label"]: entry for entry in pool_doc["pool"]}
    for slot in slate_slots:
        entry = by_label.get(slot.get("label")) if isinstance(slot, dict) else None
        if entry is None:
            errors.append(f"manifest slot {slot.get('slot') if isinstance(slot, dict) else '?'} label is not in the pool")
            continue
        if slot.get("point") != entry["point"] or slot.get("point_id") != entry["point_id"]:
            errors.append(f"manifest slot {slot['slot']} point differs from the pool entry")
        if (
            slot.get("carrier") != entry["carrier"]
            or slot.get("carrier_alternatives") != entry["carrier_alternatives"]
        ):
            errors.append(f"manifest slot {slot['slot']} carrier differs from the pool entry")
    return errors


def _replay_seat_errors(manifest: dict, pool_doc: dict, ledger: dict) -> tuple[list, list]:
    """The schema-8 binding of the manifest's seats in the ledger."""
    errors, notes = [], []
    records = [r for r in ledger.get("records", []) if isinstance(r, dict)]
    snapshot = manifest.get("ledger_snapshot") or {}
    record_count = snapshot.get("record_count")
    generation_id = manifest.get("generation_id")
    gen_no = manifest.get("gen_no")
    slate_slots = [slot for slot in (manifest.get("slate") or []) if isinstance(slot, dict)]
    seat_ids = {str(slot.get("run_id")) for slot in slate_slots}
    by_id = {str(record.get("run_id")): record for record in records}
    seats = [by_id.get(str(slot.get("run_id"))) for slot in slate_slots]
    present = [seat is not None for seat in seats]
    if slate_slots and any(present) and not all(present):
        errors.append(
            "the ledger holds only part of the manifest's slate; "
            "atomic admission admits every seat or none"
        )
    elif slate_slots and not any(present):
        notes.append("the slate is not admitted in the ledger yet")
    strangers = [
        str(record.get("run_id"))
        for record in records
        if str(record.get("run_id")) not in seat_ids
        and (record.get("policy_receipt") or {}).get("generation_id") == generation_id
    ]
    if strangers:
        errors.append(f"records {strangers} reference this generation without being seats")
    if not isinstance(record_count, int):
        errors.append("manifest ledger_snapshot.record_count is missing")
        return errors, notes
    for slot, record in zip(slate_slots, seats):
        if record is None:
            continue
        seat = f"seat {slot.get('run_id')}"
        carrier = slot.get("carrier") or {}
        receipt = record.get("policy_receipt")
        if not isinstance(receipt, dict):
            errors.append(f"{seat}: the record carries no policy receipt")
            continue
        if receipt.get("schema_version") != 8:
            errors.append(f"{seat}: policy receipt schema_version must be 8")
        receipt_policy = receipt.get("policy") or {}
        if receipt_policy.get("name") != "judged_slate":
            errors.append(f"{seat}: policy.name must be judged_slate")
        if receipt_policy.get("config") != (manifest.get("policy") or {}).get("config"):
            errors.append(f"{seat}: policy.config differs from the manifest")
        if receipt.get("generation_id") != generation_id:
            errors.append(f"{seat}: generation_id differs from the manifest")
        judge = receipt.get("judge") or {}
        expected_path = f".semantic/gen-{gen_no:04d}/generation.json"
        if judge.get("manifest_path") != expected_path:
            errors.append(f"{seat}: judge.manifest_path must be {expected_path}")
        if judge.get("slate_index") != slot.get("slot"):
            errors.append(f"{seat}: judge.slate_index does not match the manifest slot")
        if judge.get("candidate_id") != slot.get("candidate_id"):
            errors.append(f"{seat}: judge.candidate_id differs from the manifest slot")
        if judge.get("aggregation") != (manifest.get("aggregation") or {}).get("path"):
            errors.append(f"{seat}: judge.aggregation differs from the manifest")
        if receipt.get("carrier_proposal_set_revision") != carrier.get("proposal_set_revision"):
            errors.append(f"{seat}: carrier proposal revision differs from the manifest")
        budget = receipt.get("budget") or {}
        if budget.get("selection_index") != record_count + slot.get("slot", -1) + 1:
            errors.append(f"{seat}: budget.selection_index does not follow the pre-admission count")
        if budget.get("admission_cap") != (manifest.get("budget") or {}).get("admission_cap"):
            errors.append(f"{seat}: budget.admission_cap differs from the manifest")
        if receipt.get("experience") != snapshot.get("experience"):
            errors.append(f"{seat}: experience snapshot differs from the manifest")
        if receipt.get("search_space_state_revision") != snapshot.get("search_space_state_revision"):
            errors.append(f"{seat}: search_space_state_revision differs from the manifest")
        if receipt.get("space") != pool_doc.get("space"):
            errors.append(f"{seat}: space receipt differs from the generation's space")
        position = record_count + slot.get("slot", 0)
        if position >= len(records) or records[position] is not record:
            errors.append(f"{seat}: the seat is not at the expected post-prefix ledger position")
        if record.get("semantic_point") != slot.get("point"):
            errors.append(f"{seat}: the record's point differs from the manifest slot")
        if record.get("op") != carrier.get("op") or [
            str(parent) for parent in (record.get("source_run_ids") or [])
        ] != [str(parent) for parent in (carrier.get("parents") or [])]:
            errors.append(f"{seat}: the record's op/parents differ from the carrier")
    return errors, notes


def _replay_donor_binding_errors(
    manifest: dict, run_dir: Path, ledger: dict
) -> tuple[list, list]:
    """The generation's donor binding: snapshot integrity + seat receipts (§3.1).

    Recomputes the bound snapshot's content id and byte digest against the
    manifest binding, re-derives the selection over the snapshot's frozen
    eligible frontier, and checks that every eligible donor predates the
    generation (a donor admitted later could not have been bound).  A seat
    without a candidate-local receipt is simply not implemented yet; a seat
    whose receipt references a different snapshot — or any receipt under a
    no_donor binding — breaks the binding.
    """
    from scheduler import donor as donor_snapshots  # noqa: PLC0415

    errors: list[str] = []
    notes: list[str] = []
    binding = manifest.get("donor_snapshot")
    shape = _donor_binding_shape_errors(binding)
    if shape:
        return ["manifest donor_snapshot: " + error for error in shape], notes
    if binding["status"] == "bound":
        snapshot_path = run_dir / binding["path"]
        if not snapshot_path.is_file():
            errors.append(f"bound donor snapshot {binding['path']} is missing")
        else:
            try:
                snapshot = donor_snapshots.load_donor_snapshot(snapshot_path)
            except ValueError as exc:
                errors.append(f"bound donor snapshot {binding['path']}: {exc}")
            else:
                if snapshot["snapshot_id"] != binding["snapshot_id"]:
                    errors.append(
                        "the bound snapshot's content id differs from the "
                        "manifest binding"
                    )
                byte_digest = (
                    "sha256:"
                    + hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                )
                if byte_digest != binding["digest"]:
                    errors.append(
                        "the bound snapshot's bytes differ from the manifest "
                        "binding digest"
                    )
                eligible = snapshot["eligible"]
                if eligible:
                    try:
                        chosen = min(
                            eligible,
                            key=lambda entry: (
                                entry["final_best_score"],
                                entry["run_id"],
                            ),
                        )
                    except (KeyError, TypeError) as exc:
                        errors.append(
                            f"the bound snapshot's eligible entries lack "
                            f"selection fields: {exc}"
                        )
                    else:
                        if snapshot["selected"]["run_id"] != chosen["run_id"]:
                            errors.append(
                                "the bound snapshot's selected donor is not "
                                "the frozen frontier's minimum"
                            )
                record_count = (manifest.get("ledger_snapshot") or {}).get(
                    "record_count"
                )
                if isinstance(record_count, int) and not isinstance(
                    record_count, bool
                ):
                    prefix_ids = {
                        str(record.get("run_id"))
                        for record in ledger.get("records", [])[:record_count]
                        if isinstance(record, dict)
                    }
                    late = [
                        entry["run_id"]
                        for entry in eligible
                        if entry["run_id"] not in prefix_ids
                    ]
                    if late:
                        errors.append(
                            f"donor snapshot eligible {late} are not in the "
                            "generation-start ledger prefix"
                        )
    for slot in manifest.get("slate") or []:
        if not isinstance(slot, dict):
            continue
        run_id = str(slot.get("run_id"))
        receipt_path = run_dir / "candidates" / run_id / DONOR_RECEIPT_FILENAME
        if not receipt_path.is_file():
            notes.append(f"seat {run_id} has no donor receipt yet")
            continue
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"seat {run_id} donor receipt is unreadable: {exc}")
            continue
        donor = receipt.get("donor") if isinstance(receipt, dict) else None
        receipt_snapshot = (
            donor.get("snapshot_id") if isinstance(donor, dict) else None
        )
        if binding["status"] == "no_donor":
            errors.append(
                f"seat {run_id} carries a donor receipt under a no_donor "
                "binding"
            )
        elif receipt_snapshot != binding["snapshot_id"]:
            errors.append(
                f"seat {run_id} donor receipt references {receipt_snapshot}, "
                f"not the bound {binding['snapshot_id']}"
            )
    return errors, notes


def _gen_dir_number(gen_dir: Path) -> int | None:
    name = gen_dir.name
    if name.startswith("gen-") and name[4:].isdigit():
        return int(name[4:])
    return None


def replay_generation(gen_dir: Path, ledger: dict, registry: dict | None) -> dict:
    """Replay one `.semantic/gen-*/` directory against the deterministic rules.

    Returns a per-generation report: ``checks`` maps each check group to its
    pass/fail, ``errors`` lists every mismatch, ``notes`` records legitimate
    in-flight states (a provisional generation pre-manifest, a slate not yet
    admitted).  Only the rules the artifacts themselves record are replayed;
    the judge is never re-called and its choice is never re-scored.
    """
    gen_dir = Path(gen_dir)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    notes: list[str] = []

    def run_check(name, fn):
        found = fn()
        checks[name] = not found
        errors.extend(found)

    gen_no = _gen_dir_number(gen_dir)
    if gen_no is None:
        return {
            "generation": gen_dir.name,
            "gen_no": None,
            "provisional": True,
            "checks": {},
            "notes": [],
            "errors": [f"{gen_dir.name}: not a gen-NNNN directory"],
        }
    lanes_path = gen_dir / "lanes.json"
    if not lanes_path.is_file():
        return {
            "generation": gen_dir.name,
            "gen_no": gen_no,
            "provisional": True,
            "checks": {},
            "notes": [],
            "errors": [f"{gen_dir.name}: lanes.json is missing"],
        }
    lanes_doc = _load_object(lanes_path)

    def check_lanes():
        found = []
        try:
            _check_lanes_document(lanes_doc)
        except ContractError as exc:
            found.append(f"lanes.json: {exc}")
        if lanes_doc.get("gen_no") != gen_no:
            found.append(f"lanes.json gen_no {lanes_doc.get('gen_no')} does not match gen-{gen_no:04d}")
        return found

    run_check("lanes", check_lanes)

    pool_path = gen_dir / "pool.json"
    context_path = gen_dir / "context.json"
    judge_path = gen_dir / "judge.json"
    manifest_path = gen_dir / "generation.json"
    if pool_path.is_file() != context_path.is_file():
        errors.append("pool.json and context.json must be written together by construct")
        checks["pool"] = False
        return {
            "generation": gen_dir.name,
            "gen_no": gen_no,
            "provisional": True,
            "checks": checks,
            "notes": notes,
            "errors": errors,
        }
    provisional = not pool_path.is_file()
    if provisional:
        notes.append("pre-construct provisional generation; nothing to replay yet")
        if judge_path.is_file() or manifest_path.is_file():
            errors.append("judge.json/generation.json cannot exist before pool.json")
        return {
            "generation": gen_dir.name,
            "gen_no": gen_no,
            "provisional": True,
            "checks": checks,
            "notes": notes,
            "errors": errors,
        }

    pool_doc = _load_object(pool_path)
    context_doc = _load_object(context_path)

    def check_pool():
        if registry is None:
            return ["background registry unavailable; the pool cannot be rebuilt"]
        try:
            proposal_sets = _load_proposal_sets(
                lanes_doc, gen_dir / "proposals", pool_doc.get("space")
            )
        except ContractError as exc:
            return [f"proposals: {exc}"]
        return _replay_pool_errors(pool_doc, lanes_doc, proposal_sets, registry)

    run_check("pool", check_pool)

    records = [r for r in ledger.get("records", []) if isinstance(r, dict)]
    snapshot = pool_doc.get("ledger_snapshot") or {}
    record_count = snapshot.get("record_count")
    if not isinstance(record_count, int) or record_count > len(records):
        run_check(
            "context",
            lambda: [
                f"the ledger has {len(records)} records, shorter than the "
                f"generation prefix {record_count}"
            ],
        )
    else:
        prefix = records[:record_count]
        run_check(
            "context", lambda: _replay_context_errors(context_doc, pool_doc, prefix)
        )

    judgments_dir = gen_dir / "judgments"
    context_digest = digest(context_doc)
    disk_inputs: dict[str, dict] = {}
    disk_stages: dict[str, dict] = {}
    if judgments_dir.is_dir():
        for stage in STAGES:
            input_path = judgments_dir / f"{stage}.input.json"
            if input_path.is_file():
                disk_inputs[stage] = _load_object(input_path)
            artifact_path = judgments_dir / f"{stage}.json"
            if artifact_path.is_file():
                disk_stages[stage] = _load_object(artifact_path)

    def check_presented():
        found = []
        for stage, input_doc in disk_inputs.items():
            found += _replay_stage_input_errors(input_doc, pool_doc, context_doc)
        for stage, artifact in disk_stages.items():
            found += _replay_stage_artifact_errors(artifact, pool_doc, context_digest)
        return found

    run_check("presented_orders", check_presented)

    judge_doc = None
    if judge_path.is_file():
        judge_doc = _load_object(judge_path)
        run_check(
            "aggregation",
            lambda: _replay_judge_errors(
                judge_doc, pool_doc, context_doc, disk_stages, manifest_path.is_file()
            ),
        )
    elif manifest_path.is_file():
        checks["aggregation"] = False
        errors.append("generation.json exists but judge.json is missing")
    else:
        notes.append("pre-aggregate provisional generation")

    if manifest_path.is_file() and judge_doc is not None:
        manifest = _load_object(manifest_path)
        run_check(
            "manifest",
            lambda: _replay_manifest_errors(
                manifest, pool_doc, context_doc, judge_doc, gen_no
            ),
        )
        seat_errors, seat_notes = _replay_seat_errors(
            manifest, pool_doc, ledger
        )
        checks["ledger_binding"] = not seat_errors
        errors.extend(seat_errors)
        notes.extend(seat_notes)
        if "donor_snapshot" in manifest:
            binding_errors, binding_notes = _replay_donor_binding_errors(
                manifest, gen_dir.parent.parent, ledger
            )
            checks["donor_binding"] = not binding_errors
            errors.extend(binding_errors)
            notes.extend(binding_notes)
    return {
        "generation": gen_dir.name,
        "gen_no": gen_no,
        "provisional": not manifest_path.is_file(),
        "checks": checks,
        "notes": notes,
        "errors": errors,
    }


def cmd_replay(args: argparse.Namespace) -> int:
    ledger_path = Path(args.ledger)
    ledger = _load_object(ledger_path)
    semantic_dir = (
        Path(args.semantic_dir)
        if args.semantic_dir
        else ledger_path.parent / ".semantic"
    )
    background = (
        Path(args.background)
        if args.background
        else ledger_path.parent / "background.md"
    )
    registry = load_registry(background) if background.is_file() else None
    generations = []
    if semantic_dir.is_dir():
        for gen_dir in sorted(semantic_dir.glob("gen-*")):
            if gen_dir.is_dir():
                generations.append(replay_generation(gen_dir, ledger, registry))
    if not generations and not semantic_dir.is_dir():
        raise ContractError(f"no .semantic directory at {semantic_dir}")
    ok = all(not generation["errors"] for generation in generations)
    report = {
        "ok": ok,
        "ledger": str(ledger_path),
        "semantic_dir": str(semantic_dir),
        "generations_checked": len(generations),
        "generations": [
            generation if args.verbose else {**generation, "notes": []}
            for generation in generations
        ],
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    construct = sub.add_parser(
        "construct", help="verify the prefix, seat the pool, write pool+context"
    )
    construct.add_argument("--lanes", type=Path, required=True)
    construct.add_argument("--proposals-dir", type=Path, required=True)
    construct.add_argument("--ledger", type=Path, required=True)
    construct.add_argument("--background", type=Path, required=True)
    construct.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="defaults to the run's judged_slate.pool_size (or 6)",
    )
    construct.add_argument("--pool-output", type=Path, required=True)
    construct.add_argument("--context-output", type=Path, required=True)
    construct.set_defaults(func=cmd_construct)

    prepare = sub.add_parser(
        "prepare-judge", help="write one stage's presented order + prompt payload"
    )
    prepare.add_argument("--pool", type=Path, required=True)
    prepare.add_argument("--context", type=Path, required=True)
    prepare.add_argument("--stage", choices=STAGES, required=True)
    prepare.add_argument(
        "--labels", help="boundary stage only: comma-separated union labels"
    )
    prepare.add_argument(
        "--task-brief",
        type=Path,
        help="optional frozen task/run contract text prepended to the payload",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=cmd_prepare_judge)

    validate = sub.add_parser(
        "validate-judge", help="check a receipt is an exact presented-label permutation"
    )
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--receipt", type=Path)
    validate.add_argument("--session-id")
    validate.add_argument("--model")
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(func=cmd_validate_judge)

    aggregate = sub.add_parser(
        "aggregate", help="regular/boundary aggregation -> judge.json + status"
    )
    aggregate.add_argument("--pool", type=Path, required=True)
    aggregate.add_argument("--context", type=Path, required=True)
    aggregate.add_argument("--judgments-dir", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.set_defaults(func=cmd_aggregate)

    manifest = sub.add_parser(
        "build-manifest", help="write the immutable generation.json manifest"
    )
    manifest.add_argument("--lanes", type=Path, required=True)
    manifest.add_argument("--pool", type=Path, required=True)
    manifest.add_argument("--context", type=Path, required=True)
    manifest.add_argument("--judge", type=Path, required=True)
    manifest.add_argument(
        "--reserved-run-ids", required=True, help="comma-separated, one per slot"
    )
    donor = manifest.add_mutually_exclusive_group()
    donor.add_argument(
        "--donor-snapshot",
        type=Path,
        default=None,
        help=(
            "donor snapshot artifact (absolute or run-dir-relative) to bind "
            "into the manifest; transfer scheduler policy only"
        ),
    )
    donor.add_argument(
        "--no-donor",
        action="store_true",
        help=(
            "bind an explicit no_donor state (transfer scheduler policy "
            "before any donor exists); one of --donor-snapshot/--no-donor is "
            "required under that policy and rejected elsewhere"
        ),
    )
    manifest.add_argument("--output", type=Path, required=True)
    manifest.set_defaults(func=cmd_build_manifest)

    replay = sub.add_parser(
        "replay",
        help="re-derive every recorded generation's deterministic decisions "
        "from the artifacts and the ledger prefix; exit 1 on any mismatch",
    )
    replay.add_argument("--ledger", type=Path, required=True)
    replay.add_argument(
        "--semantic-dir",
        type=Path,
        help="where the run keeps its generations (default <ledger dir>/.semantic)",
    )
    replay.add_argument(
        "--background",
        type=Path,
        help="registry source for rebuilding pools (default <ledger dir>/background.md)",
    )
    replay.add_argument("--output", type=Path, help="write the replay report here")
    replay.add_argument(
        "--verbose",
        action="store_true",
        help="keep the per-generation notes in the report",
    )
    replay.set_defaults(func=cmd_replay)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (ContractError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"ok": False, "errors": [str(exc)]}, indent=2),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
