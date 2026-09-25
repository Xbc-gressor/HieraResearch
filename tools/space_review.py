"""Bounded runtime review material and proposals; no scheduling or SDK calls.

CLI: material --background ... --ledger ... --output ...
     validate --background ... --ledger ... --review ...
The driver supplies the goal, remaining budget and role invocation. Optional
external retrieval uses the existing search_backends adapter and faithfulness
audit, after separate admission by the driver.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
from typing import Any

from background_contract import load_registry, render_space, derive_hypothesis_selection
from search_space_state import compose_effective_selection, replay_search_space_state, validate_point_eligibility
from search_backends import merged_results
from semantic_space import (coverage_from_records, complete_point, hypothesis_map, resolve_dimension_catalog,
                            resolve_dimension_strategy, selected_assignments, space_revision,
                            validate_point, validate_record_point)
from space_revisions import apply_expansion, load_registry_history


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_retrieval_request(request: Any) -> list[str]:
    if not isinstance(request, dict) or set(request) != {"knowledge_gap", "decision_impact", "queries"}:
        return ["retrieval_request needs knowledge_gap, decision_impact and queries"]
    errors = [f"retrieval_request.{key} must be non-empty" for key in ("knowledge_gap", "decision_impact")
              if not _text(request.get(key))]
    queries = request.get("queries")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 3 or any(not _text(q) for q in queries):
        errors.append("retrieval_request.queries must contain 1–3 non-empty methodology questions")
    return errors


def validate_review(review: Any, registry: dict, *, ledger: dict, catalog: dict | None = None,
                    dimension_strategy: str = "catalog_subset", registry_history: dict | None = None) -> list[str]:
    if not isinstance(review, dict):
        return ["review must be an object"]
    allowed = {"decision", "reason", "basis", "base_revision", "delta", "probe"}
    errors = []
    if set(review) - allowed:
        errors.append("review has unknown fields")
    if review.get("base_revision") != space_revision(registry):
        errors.append("review base_revision is stale")
    if not _text(review.get("reason")):
        errors.append("review.reason must explain the decision against the goal and remaining budget")
    basis = review.get("basis")
    if not isinstance(basis, list) or not basis or any(not _text(ref) for ref in basis):
        errors.append("review.basis must cite research or runtime material")
    if review.get("decision") == "continue":
        if review.get("delta") or review.get("probe"):
            errors.append("continue cannot publish a delta or probe")
        return errors
    if review.get("decision") != "expand":
        return errors + ["review.decision must be continue or expand"]
    try:
        expanded, _ = apply_expansion(registry, review.get("delta"), catalog=catalog,
                                     dimension_strategy=dimension_strategy, records=ledger.get("records", []))
    except ValueError as exc:
        return errors + [str(exc)]
    probe = review.get("probe")
    if not isinstance(probe, dict):
        return errors + ["expand requires a probe"]
    errors.extend(validate_point(probe.get("point"), expanded))
    runtime = replay_search_space_state(expanded, ledger.get("search_space_state") or {})
    effective = compose_effective_selection(expanded, derive_hypothesis_selection(expanded), runtime)
    errors.extend(validate_point_eligibility(probe.get("point"), expanded, effective))
    if isinstance(probe.get("point"), dict):
        new_ids = set(hypothesis_map(expanded)) - set(hypothesis_map(registry))
        if not new_ids.intersection(selected_assignments(probe["point"]).values()):
            errors.append("probe must exercise the new route")
    parents = probe.get("parents")
    op = probe.get("op")
    n_parents = {"fresh": 0, "improve": 1, "crossover": 2}.get(op) if isinstance(op, str) else None
    if not isinstance(parents, list) or any(not isinstance(p, str) for p in parents):
        errors.append("probe.parents must be a string list")
    elif n_parents is None or len(parents) != n_parents or len(set(parents)) != n_parents:
        errors.append("probe carrier has invalid operator or parent count")
    else:
        by_id = {r.get("run_id"): r for r in ledger.get("records", [])}
        for parent_id in parents:
            record = by_id.get(parent_id)
            if record is None or record.get("status") not in {"keep", "discard"}:
                errors.append(f"probe parent {parent_id} is not an available scored candidate")
            else:
                errors.extend(validate_record_point(record.get("semantic_point"), registry, registry_history))
    for key in ("implementation_seconds", "screening_seconds"):
        cost = probe.get(key)
        if isinstance(cost, bool) or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            errors.append(f"probe.{key} must be a positive finite estimate")
    if not _text(probe.get("expected_observation")):
        errors.append("probe.expected_observation must be non-empty")
    return errors


def _bounded(value: Any, limit: int, chars: int = 1200) -> Any:
    """Bound display only; originals remain available at named run-local inputs."""
    if isinstance(value, str):
        return value[:chars]
    if isinstance(value, list):
        return [_bounded(item, limit, chars) for item in value[:limit]]
    if isinstance(value, dict):
        return {key: _bounded(item, limit, chars) for key, item in list(value.items())[:32]}
    return value


def _execution_material(run_dir: Path, records: list[dict], limit: int) -> dict:
    """Read retained cost/failure receipts; do not infer training diagnostics."""
    path = run_dir / "evaluation_attempts.jsonl"
    elapsed = 0.0
    completions = 0
    attempts = 0
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "score_attempt":
                    attempts += 1
                seconds = row.get("duration_seconds")
                if row.get("kind") == "score_completion" and isinstance(seconds, (int, float)) and math.isfinite(seconds):
                    elapsed += seconds
                    completions += 1
    diagnostics = []
    for record in records[-limit:]:
        run_id = str(record.get("run_id"))
        if not run_id.isdigit():
            continue
        report_path = run_dir / "candidates" / run_id / "tune_report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text())
        warm = report.get("phase_a", {})
        diagnostics.append({
            "run_id": run_id, "source": str(report_path.relative_to(run_dir)),
            "status": warm.get("status"), "circuit_breaker": warm.get("circuit_breaker"),
            "warm_results": [{k: row.get(k) for k in ("score", "status", "config_infeasible",
                "failure_kind", "error_type", "error", "message", "elapsed_seconds") if k in row}
                for row in warm.get("warm_start_configs", [])[-limit:]],
        })
    return {"source": str(path), "objective_attempts": attempts if path.exists() else None,
            "recorded_completions": completions,
            "completed_eval_seconds": elapsed if completions else None,
            "candidate_diagnostics": _bounded(diagnostics, limit),
            "notice": "Recorded completions only; missing completion time and diagnostics are unknown."}


def _coverage(registry: dict, ledger: dict, history: dict) -> dict:
    """Attempted and scored counts for every searchable hypothesis; never display-bounded."""
    records = ledger.get("records", [])
    attempted = coverage_from_records(registry, records, registry_history=history)
    scored = coverage_from_records(registry, [r for r in records if r.get("status") in {"keep", "discard"}],
                                   registry_history=history)
    counts = {h["hypothesis_id"]: h["count"] for d in scored["dimensions"] for h in d["hypotheses"]}
    runtime = replay_search_space_state(registry, ledger.get("search_space_state") or {})
    status = compose_effective_selection(registry, derive_hypothesis_selection(registry), runtime)
    tried, untried = [], []
    for dimension in attempted["dimensions"]:
        if dimension["mode"] != "searchable":
            continue
        for hypothesis in dimension["hypotheses"]:
            hid = hypothesis["hypothesis_id"]
            row = {"dimension_id": dimension["dimension_id"], "hypothesis_id": hid,
                   "status": status[hid]["effective_status"]}
            if hypothesis["count"]:
                tried.append({**row, "attempts": hypothesis["count"], "scored": counts.get(hid, 0)})
            else:
                untried.append(row)
    return {"space_revision": attempted["space_revision"], "n_valid_records": attempted["n_valid_records"],
            "n_unique_points": attempted["n_unique_points"], "tried": tried, "untried": untried,
            "n_invalid_records": len(attempted["invalid_records"]),
            "invalid_records": _bounded(attempted["invalid_records"][-3:], 3, 600)}


def build_review_material(background: Path, *, ledger: dict, limit: int = 12,
                          goal: str | None = None, remaining_seconds: float | None = None) -> dict:
    if not 1 <= limit <= 32:
        raise ValueError("material limit must be in [1, 32]")
    background = Path(background)
    registry = load_registry(background)
    history = load_registry_history(background)
    manifest_path = background.parent / "background_retrieval.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    # Include unregistered/deferred/contradictory research, not only registry sources.
    raw = background.read_text(errors="replace").split("## Search space registry", 1)[0]
    sections = re.split(r"(?m)(?=^## )", raw)
    research = [{"path": background.name, "section": s.splitlines()[0] if s.splitlines() else "",
                 "excerpt": s[:1600]} for s in sections if s.strip()]
    hits = merged_results(manifest) if manifest else []
    visits = [{key: v.get(key) for key in ("url", "status", "view", "content_file")}
              for v in (manifest or {}).get("visits", [])]
    records = ledger.get("records", [])
    def fact(record):
        return {key: record.get(key) for key in ("run_id", "op", "source_run_ids", "status",
            "semantic_point", "best_warm_score", "final_best_score", "evaluation_depth",
            "unevaluated_receipt", "error", "failure", "cost", "tune_summary") if key in record}
    coverage = _coverage(registry, ledger, history)
    # Counts are computed over all records. Recent examples are not evidence of
    # the whole run's score distribution or of training/convergence behavior.
    packet = {
        "schema_version": 1, "base_revision": space_revision(registry),
        "goal": goal, "remaining_seconds": remaining_seconds,
        "research": _bounded(research, 32, 1600),
        "research_index": {"background": str(background), "retrieval_manifest": str(manifest_path),
                           "total_sections": len(research), "total_hits": len(hits),
                           "total_visits": len(visits), "hits": _bounded(hits, limit),
                           "visits": _bounded(visits, limit)},
        "space": _bounded(render_space(registry, ledger, max_hypotheses=min(limit, 4), registry_history=history), 16, 400),
        "reserves": _bounded(registry.get("reserves", []), 12),
        "guidance": _bounded(registry.get("guidance", []), limit),
        "search": {"total_records": len(records), "status_counts": dict(Counter(r.get("status") for r in records)),
                   "recent": _bounded([fact(r) for r in records[-limit:]], limit),
                   "failure_examples": _bounded([fact(r) for r in records if r.get("status") in {"crash", "aborted", "unevaluated"}][-3:], 3),
                   "scored_examples": _bounded([fact(r) for r in records if r.get("status") in {"keep", "discard"}][-3:], 3),
                   "attempt_observations": _bounded(ledger.get("attempt_observations", [])[-limit:], limit)},
        "execution": _execution_material(background.parent, records, limit),
        "coverage": coverage,
        "experience": _bounded(ledger.get("experience") or {}, limit),
        "read_more": [str(background), str(manifest_path), str(background.parent / "ledger.json"),
                      str(background.parent / ".semantic/space-revisions.json")],
        "limitations": ["Missing diagnostics remain unknown; scalar scores do not establish causal bottlenecks.",
                         "Display is bounded; reread retained research via the manifest, including unregistered hits.",
                         "Only compare scores in the same evaluation domain; differing investment is a confound."],
    }
    # The source index survives reduction, so clipping display does not turn
    # the first selected hypotheses or papers into a knowledge whitelist.
    display_limit = limit
    while len(json.dumps(packet, ensure_ascii=False)) > 60000 and display_limit > 1:
        display_limit = max(1, display_limit // 2)
        for key in ("research", "space", "guidance", "experience"):
            packet[key] = _bounded(packet[key], display_limit, 600)
        for key in ("hits", "visits"):
            packet["research_index"][key] = _bounded(packet["research_index"][key], display_limit, 600)
        # These lists are tails of the run; keep their most recent items.
        for key in ("recent", "failure_examples", "scored_examples", "attempt_observations"):
            packet["search"][key] = _bounded(packet["search"][key][-display_limit:], display_limit, 600)
        packet["execution"]["candidate_diagnostics"] = _bounded(
            packet["execution"]["candidate_diagnostics"][-display_limit:], display_limit, 600)
    progress_path = background.parent / ".semantic/space-review-state.json"
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        packet["expansion_progress"] = {"reviews": _bounded(state["reviews"][-4:], 4),
                                        "probes": _bounded(state["probes"], 4)}
        # Each entry is the boundary where the system best (including
        # rewrite/tune) was first seen lower, and the operation behind it.
        packet["best_history"] = state["best_history"]
    packet["display_limit"] = display_limit
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["material", "validate", "validate-retrieval", "complete-point"])
    parser.add_argument("--background", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--goal")
    parser.add_argument("--remaining-seconds", type=float)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.command == "validate-retrieval":
        if args.review is None:
            parser.error("validate-retrieval requires --review")
        errors = validate_retrieval_request(json.loads(args.review.read_text()))
        value = {"ok": not errors, "errors": errors}
    else:
        if args.background is None or args.ledger is None:
            parser.error("material/validate require --background and --ledger")
        ledger = json.loads(args.ledger.read_text())
        if args.command == "material":
            value = build_review_material(args.background, ledger=ledger, limit=args.limit,
                                          goal=args.goal, remaining_seconds=args.remaining_seconds)
        elif args.command == "complete-point":
            if args.review is None:
                parser.error("complete-point requires --review")
            value = json.loads(args.review.read_text())
            expanded, _ = apply_expansion(load_registry(args.background), value['delta'],
                catalog=resolve_dimension_catalog(args.background),
                dimension_strategy=resolve_dimension_strategy(args.background), records=ledger.get('records', []))
            point = complete_point(expanded, value['probe'].pop('assignments', {}))
            if point is None:
                raise ValueError("probe assignments cannot form a compatible point")
            value['probe']['point'] = point
        else:
            if args.review is None:
                parser.error("validate requires --review")
            errors = validate_review(json.loads(args.review.read_text()), load_registry(args.background),
                ledger=ledger, catalog=resolve_dimension_catalog(args.background),
                dimension_strategy=resolve_dimension_strategy(args.background),
                registry_history=load_registry_history(args.background))
            value = {"ok": not errors, "errors": errors}
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0 if value.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
