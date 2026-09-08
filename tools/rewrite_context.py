#!/usr/bin/env python3
"""Render the rewrite-editor's bounded intelligence brief (context.md).

The rewrite loop re-renders <candidate>/_rewrite/context.md before every
bout; this file is the editor's information advantage over blind hillclimb.
Six sections, fixed order:

1. Bout history     — _rewrite/bouts.jsonl, last 20 entries;
2. Eval traces      — header metadata of the last 5 attempts, this run's
                      newest (_traces/) first, source history (_traces_src/)
                      filling the remainder, with the stderr tail attached
                      for crashes;
3. Experience       — experience.seed.json lessons/bottlenecks in full, plus
                      the dimension/hypothesis evidence hitting this point;
4. Semantic point   — the selected assignments anchored to the run-level
                      background.md registry (matching relations and guidance
                      first, then per-dimension blocks: dimension definition/
                      boundary/selection_reason, selected hypothesis detail,
                      sibling boundary list);
5. Source material  — snippets behind the selected hypotheses' evidence,
                      resolved evidence -> sources[].url -> canonical_key ->
                      background_retrieval.json merged results (retained visit
                      content files as fallback);
6. Candidate status — current best, baseline, tuning delta/summary, idea.

Budgets: section 4 <= 10KB, every other section <= 4KB, total <= 20KB; long
free-text fields <= 250 chars; truncation is marked with ...[truncated];
missing artifacts render as (none). The run directory is found by walking
up from the candidate path (candidates always live at <run>/candidates/<id>).

Usage:
    python tools/rewrite_context.py --candidate <dir> [--sections a,b,c]

--sections takes a comma-separated subset of
bout_history,traces,experience,background,sources,status (the loop's
--context ablation switch passes straight through). stdout:
{"context_md": "<path>"}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from background_contract import ContractError, load_registry
from rewrite_bout import current_best, load_bouts
from search_backends import canonical_key, merged_results, resolve_visit_content

FIELD_LIMIT = 250
SNIPPET_LIMIT = 600
STDERR_TAIL_LIMIT = 2048
SECTION_LIMIT = 4096
POINT_SECTION_LIMIT = 10240
TOTAL_LIMIT = 20480
BOUT_WINDOW = 20
TRACE_WINDOW = 5
GUIDANCE_CAP = 5
SOURCE_CAP = 5
TRUNCATED = "...[truncated]"
NONE = "(none)"

SECTION_ORDER = ("bout_history", "traces", "experience", "background", "sources", "status")
SECTION_TITLES = {
    "bout_history": "## Bout history",
    "traces": "## Eval traces",
    "experience": "## Experience",
    "background": "## Semantic point",
    "sources": "## Source material",
    "status": "## Candidate status",
}
# The scope facets guidance matching looks at (per the render contract).
GUIDANCE_FACETS = ("model_families", "interventions", "metrics")


def _text(value, limit: int = FIELD_LIMIT) -> str:
    """One-line rendering of a free-text field, clipped with a marker."""
    if value is None:
        return NONE
    flat = " ".join(str(value).split())
    if len(flat) > limit:
        return flat[: limit - len(TRUNCATED)] + TRUNCATED
    return flat


def _cap(text: str, limit: int) -> str:
    """Hard size cap on a rendered block, cut at a line boundary."""
    if len(text) <= limit:
        return text
    marker = "\n" + TRUNCATED + "\n"
    body = text[: limit - len(marker)]
    newline = body.rfind("\n")
    if newline > 0:
        body = body[:newline]
    return body + marker


def _load_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dir(candidate: Path) -> Path:
    """Ascend from the candidate path: <run>/candidates/<id> -> <run>."""
    parent = candidate.parent
    return parent.parent if parent.name == "candidates" else parent


def _selected_assignments(manifest: dict | None) -> list[dict]:
    point = (manifest or {}).get("semantic_point")
    if not isinstance(point, dict):
        return []
    return [
        assignment
        for assignment in point.get("assignments") or []
        if isinstance(assignment, dict) and assignment.get("state") == "selected"
    ]


def _load_registry(run_dir: Path) -> dict | None:
    path = run_dir / "background.md"
    if not path.is_file():
        return None
    try:
        return load_registry(path)
    except ContractError as exc:
        raise SystemExit(str(exc)) from None


def _registry_dimensions(registry: dict) -> dict:
    return {
        dimension.get("id"): dimension
        for dimension in registry.get("dimensions") or []
        if isinstance(dimension, dict)
    }


def _dimension_hypotheses(dimension: dict) -> dict:
    return {
        hypothesis.get("id"): hypothesis
        for hypothesis in dimension.get("hypotheses") or []
        if isinstance(hypothesis, dict)
    }


def _bout_history_lines(candidate: Path) -> list[str]:
    bouts = load_bouts(candidate)[-BOUT_WINDOW:]
    if not bouts:
        return [NONE]
    lines = []
    for entry in bouts:
        score = entry.get("score")
        score_text = f"{score:.6g}" if isinstance(score, (int, float)) else NONE
        lines.append(
            f"- bout {entry.get('bout')}: {entry.get('outcome')} | score {score_text} "
            f"| {_text(entry.get('summary'))} | basis: {_text(entry.get('basis'))}"
        )
    return lines


def _parse_trace(path: Path) -> tuple[dict, str]:
    """Split one trace log into its header dict and the raw stderr section."""
    text = path.read_text(encoding="utf-8", errors="replace")
    header = {}
    for line in text.splitlines():
        if not line.strip():
            break
        key, sep, value = line.partition(":")
        if sep:
            header[key.strip()] = value.strip()
    marker = "\n[stderr]\n"
    index = text.find(marker)
    stderr = text[index + len(marker) :] if index != -1 else ""
    return header, stderr


def _traces_lines(candidate: Path) -> list[str]:
    entries = []
    for label in ("_traces_src", "_traces"):
        directory = candidate / label
        if not directory.is_dir():
            continue
        for path in directory.glob("*.log"):
            # this run's logs always outrank the source history: attempt ids
            # count from each run's own global counter, so the source's (often
            # far larger) ids would otherwise bury this run's fresh traces
            rank = 1 if label == "_traces" else 0
            entries.append((rank, path.stem, label, path))
    entries.sort(key=lambda item: (item[0], item[1]))
    if not entries:
        return [NONE]
    lines = []
    for _, stem, label, path in entries[-TRACE_WINDOW:]:
        header, stderr = _parse_trace(path)
        lines.append(f"### {stem} ({label}/)")
        lines.append(
            " | ".join(
                f"{key}: {header.get(key, NONE)}"
                for key in (
                    "phase",
                    "method",
                    "returncode",
                    "timed_out",
                    "elapsed_seconds",
                    "max_rss_kb",
                )
            )
        )
        crashed = header.get("timed_out") == "true" or header.get("returncode") != "0"
        if crashed and stderr.strip():
            lines.append("stderr (tail):")
            lines.append("```")
            lines.append(stderr.rstrip()[-STDERR_TAIL_LIMIT:])
            lines.append("```")
    return lines


def _experience_lines(run_dir: Path, dimension_ids: set, hypothesis_ids: set) -> list[str]:
    experience = _load_json(run_dir / "experience.seed.json")
    if experience is None:
        return [NONE]
    lines = ["### Lessons"]
    lessons = experience.get("lessons") or []
    if not lessons:
        lines.append(NONE)
    for item in lessons:
        evidence = ", ".join(str(run_id) for run_id in item.get("evidence") or [])
        lines.append(
            f"- [{item.get('kind')}/{item.get('confidence')}] "
            f"{_text(item.get('claim'))} (evidence: {evidence})"
        )
    lines.append("### Bottlenecks")
    bottlenecks = experience.get("bottlenecks") or []
    if not bottlenecks:
        lines.append(NONE)
    for item in bottlenecks:
        evidence = ", ".join(str(run_id) for run_id in item.get("evidence") or [])
        lines.append(
            f"- [{item.get('confidence')}] {_text(item.get('claim'))} "
            f"(evidence: {evidence})"
        )
    lines.append("### Evidence on this semantic point")
    hits = [
        item
        for field, wanted in (
            ("dimension_evidence", dimension_ids),
            ("hypothesis_evidence", hypothesis_ids),
        )
        for item in experience.get(field) or []
        if isinstance(item, dict) and item.get("target_id") in wanted
    ]
    if not hits:
        lines.append(NONE)
    for item in hits:
        coverage = json.dumps(
            item.get("comparator_coverage"), ensure_ascii=False, sort_keys=True
        )
        lines.append(
            f"- {item.get('target_id')}: state={item.get('evaluation_state')} "
            f"assessment={item.get('assessment')} confidence={item.get('confidence')} "
            f"coverage={coverage} | {_text(item.get('claim'))}"
        )
    return lines


def _guidance_hit(scope, union: dict) -> bool:
    """A guidance entry hits when any watched facet intersects the point's union."""
    if not isinstance(scope, dict):
        return False
    for facet in GUIDANCE_FACETS:
        values = set(scope.get(facet) or [])
        point_values = union[facet]
        if not values or not point_values:
            continue
        if "*" in values or "*" in point_values or values & point_values:
            return True
    return False


def _semantic_point_lines(registry: dict | None, selected: list[dict]) -> list[str]:
    if registry is None or not selected:
        return [NONE]
    dimensions = _registry_dimensions(registry)
    dimension_ids = {assignment.get("dimension_id") for assignment in selected}
    hypothesis_ids = {assignment.get("hypothesis_id") for assignment in selected}
    scope_union = {facet: set() for facet in GUIDANCE_FACETS}
    for assignment in selected:
        dimension = dimensions.get(assignment.get("dimension_id"))
        if dimension is None:
            continue
        hypothesis = _dimension_hypotheses(dimension).get(assignment.get("hypothesis_id"))
        scope = (hypothesis or {}).get("scope")
        if isinstance(scope, dict):
            for facet in GUIDANCE_FACETS:
                scope_union[facet] |= set(scope.get(facet) or [])
    lines = []
    # Relations and guidance lead: the editor's most actionable intelligence.
    relations = [
        relation
        for relation in registry.get("relations") or []
        if isinstance(relation, dict)
        and isinstance(relation.get("when"), dict)
        and relation["when"].get("dimension_id") in dimension_ids
        and hypothesis_ids & set(relation["when"].get("hypothesis_ids") or [])
    ]
    lines.append("### Relations at this point")
    if not relations:
        lines.append(NONE)
    for relation in relations:
        then = relation.get("then")
        then_text = (
            json.dumps(then, ensure_ascii=False, sort_keys=True)
            if isinstance(then, (dict, list))
            else _text(then)
        )
        lines.append(
            f"- {relation.get('id')} [{relation.get('type')}] -> "
            f"{relation.get('target_dimension_id')} | then: {then_text}"
        )
    guidance = [
        item
        for item in registry.get("guidance") or []
        if isinstance(item, dict) and _guidance_hit(item.get("scope"), scope_union)
    ]
    guidance.sort(key=lambda item: (item.get("section") != "pitfall", str(item.get("id"))))
    lines.append("### Guidance")
    if not guidance:
        lines.append(NONE)
    for item in guidance[:GUIDANCE_CAP]:
        lines.append(
            f"- [{item.get('id')} {item.get('section')}/{item.get('effect')}] "
            f"{_text(item.get('claim'))}"
        )
    # Per-dimension blocks follow (audit metadata like provenance/reopen_when
    # is deliberately not rendered).
    for assignment in selected:
        dimension_id = assignment.get("dimension_id")
        hypothesis_id = assignment.get("hypothesis_id")
        lines.append(f"### {dimension_id} -> {hypothesis_id}")
        dimension = dimensions.get(dimension_id)
        if dimension is None:
            lines.append("(dimension not in registry)")
            continue
        lines.append(f"- definition: {_text(dimension.get('definition'))}")
        lines.append(f"- boundary: {_text(dimension.get('boundary'))}")
        lines.append(f"- selection_reason: {_text(dimension.get('selection_reason'))}")
        hypotheses = _dimension_hypotheses(dimension)
        hypothesis = hypotheses.get(hypothesis_id)
        if hypothesis is None:
            lines.append(f"- selected hypothesis {hypothesis_id} not in registry")
        else:
            lines.append(f"- selected: {_text(hypothesis.get('title'))}")
            for field in (
                "claim",
                "testable_expectation",
                "claim_scope",
                "status",
                "literature_credibility",
                "credibility_rationale",
            ):
                lines.append(f"  - {field}: {_text(hypothesis.get(field))}")
        siblings = [
            (sibling.get("id"), sibling.get("title"))
            for sibling in dimension.get("hypotheses") or []
            if isinstance(sibling, dict) and sibling.get("id") != hypothesis_id
        ]
        if siblings:
            lines.append("- not selected (what this point does not do):")
            for sibling_id, sibling_title in siblings:
                lines.append(f"  - {sibling_id}: {_text(sibling_title)}")
    return lines


def _retrieval_match(entries: list, key: str, url: str) -> dict | None:
    """First entry matching by canonical_key, else by exact url."""
    for entry in entries:
        if entry.get("canonical_key") == key:
            return entry
    for entry in entries:
        if entry.get("url") == url:
            return entry
    return None


def _retrieval_lookup(entries: list, key: str, url: str, field: str) -> str | None:
    match = _retrieval_match(entries, key, url)
    if match is None:
        return None
    value = match.get(field)
    return value if isinstance(value, str) and value.strip() else None


def _source_material_lines(
    registry: dict | None,
    retrieval: dict | None,
    manifest_dir: Path,
    selected: list[dict],
) -> list[str]:
    if registry is None or not selected:
        return [NONE]
    dimensions = _registry_dimensions(registry)
    cited: dict[str, list[str]] = {}
    for assignment in selected:
        dimension = dimensions.get(assignment.get("dimension_id")) or {}
        hypothesis = _dimension_hypotheses(dimension).get(assignment.get("hypothesis_id"))
        if hypothesis is None:
            continue
        for link in hypothesis.get("evidence") or []:
            if isinstance(link, dict) and isinstance(link.get("source_id"), str):
                cited.setdefault(link["source_id"], []).append(
                    f"{link.get('role')} in {hypothesis.get('id')}"
                )
    if not cited:
        return [NONE]
    sources = {
        source.get("id"): source
        for source in registry.get("sources") or []
        if isinstance(source, dict)
    }
    results = merged_results(retrieval or {})
    visits = [
        entry for entry in (retrieval or {}).get("visits") or [] if isinstance(entry, dict)
    ]
    lines = []
    for source_id in sorted(cited)[:SOURCE_CAP]:
        source = sources.get(source_id)
        if source is None:
            lines.append(f"### {source_id}\n(not in registry sources)")
            continue
        url = source.get("url") or ""
        key = canonical_key(url)
        text = _retrieval_lookup(results, key, url, "snippet")
        if text is None:
            visit = _retrieval_match(visits, key, url)
            text = resolve_visit_content(visit, manifest_dir)
        lines.append(f"### {source_id} — {_text(source.get('title'))}")
        lines.append(f"url: {url} | canonical_key: {key} | cited: {'; '.join(cited[source_id])}")
        lines.append(_text(text, SNIPPET_LIMIT) if text else "(no retrieval content)")
    return lines


def _status_lines(candidate: Path, manifest: dict | None) -> list[str]:
    manifest = manifest or {}
    try:
        best_text = f"{current_best(candidate):.6g}"
    except ValueError:
        best_text = NONE
    tune_summary = manifest.get("tune_summary")
    return [
        f"- current_best: {best_text}",
        f"- baseline_score: {_text(manifest.get('baseline_score'))}",
        f"- warm_to_tuned_delta: {_text(manifest.get('warm_to_tuned_delta'))}",
        "- tune_summary: "
        + (
            json.dumps(tune_summary, ensure_ascii=False, sort_keys=True)
            if tune_summary is not None
            else NONE
        ),
        f"- idea: {_text(manifest.get('idea'))}",
        f"- change: {_text(manifest.get('change'))}",
    ]


def render_context(candidate, sections=None) -> str:
    """Render the full context.md text for one candidate directory."""
    candidate = Path(candidate)
    run_dir = _run_dir(candidate)
    manifest = _load_json(candidate / "_import.json")
    selected = _selected_assignments(manifest)
    dimension_ids = {assignment.get("dimension_id") for assignment in selected}
    hypothesis_ids = {assignment.get("hypothesis_id") for assignment in selected}
    registry = _load_registry(run_dir)
    retrieval = _load_json(run_dir / "background_retrieval.json")
    bodies = {
        "bout_history": _bout_history_lines(candidate),
        "traces": _traces_lines(candidate),
        "experience": _experience_lines(run_dir, dimension_ids, hypothesis_ids),
        "background": _semantic_point_lines(registry, selected),
        "sources": _source_material_lines(registry, retrieval, run_dir, selected),
        "status": _status_lines(candidate, manifest),
    }
    wanted = SECTION_ORDER if sections is None else [s for s in SECTION_ORDER if s in sections]
    parts = []
    for key in wanted:
        section = SECTION_TITLES[key] + "\n" + "\n".join(bodies[key]) + "\n"
        limit = POINT_SECTION_LIMIT if key == "background" else SECTION_LIMIT
        parts.append(_cap(section, limit))
    document = "# Rewrite context\n\n" + "\n".join(parts)
    return _cap(document, TOTAL_LIMIT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--sections",
        default=None,
        help="comma-separated subset of " + ",".join(SECTION_ORDER),
    )
    args = parser.parse_args()
    candidate = args.candidate
    if not candidate.is_dir():
        raise SystemExit(f"candidate directory not found: {candidate}")
    if args.sections is None:
        sections = None
    else:
        sections = [item.strip() for item in args.sections.split(",") if item.strip()]
        unknown = sorted(set(sections) - set(SECTION_ORDER))
        if unknown:
            raise SystemExit(
                f"unknown sections {unknown}; choose from {list(SECTION_ORDER)}"
            )
    document = render_context(candidate, sections)
    output = candidate / "_rewrite" / "context.md"
    output.parent.mkdir(exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(json.dumps({"context_md": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
