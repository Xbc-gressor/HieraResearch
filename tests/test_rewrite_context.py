"""Tests for tools/rewrite_context.py (rewrite-editor context rendering)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import rewrite_context  # noqa: E402


def _registry() -> dict:
    return {
        "space_id": "synthetic",
        "dimensions": [
            {
                "id": "dim-step",
                "definition": "How the step size adapts between generations.",
                "boundary": "Exact numeric settings belong to inner HPO.",
                "selection_reason": "The step-size rule is the mechanism TASK.md credits.",
                "hypotheses": [
                    {
                        "id": "hyp-step-csa",
                        "title": "CSA with identity covariance",
                        "claim": "CSA on an isotropic Gaussian beats fixed-sigma ES.",
                        "testable_expectation": "Lower suite mean than fixed-sigma ES.",
                        "claim_scope": "(mu/mu_I, lambda)-CSA-ES on sphere problems.",
                        "status": "active",
                        "literature_credibility": "corroborated",
                        "credibility_rationale": "Hansen's tutorial motivates CSA.",
                        "reopen_when": "Default CSA damping is retuned for large lambda.",
                        "provenance": [{"kind": "literature", "ref": "src-01"}],
                        "scope": {
                            "model_families": ["gaussian_es"],
                            "interventions": ["csa_identity"],
                            "metrics": ["mean_log10"],
                        },
                        "evidence": [
                            {"source_id": "src-01", "role": "supports"},
                            {"source_id": "src-02", "role": "context"},
                        ],
                    },
                    {"id": "hyp-step-full-cma", "title": "Full covariance CMA"},
                ],
            },
            {
                "id": "dim-bounds",
                "definition": "How infeasible coordinates are handled.",
                "boundary": "Nothing outside the box is ever evaluated.",
                "selection_reason": "Bound handling changes where evaluations land.",
                "hypotheses": [
                    {
                        "id": "hyp-bounds-mirror",
                        "title": "Inward reflection at the bounds",
                        "scope": {"model_families": ["gaussian_es"]},
                        "evidence": [],
                    },
                ],
            },
        ],
        "relations": [
            {
                "id": "rel-1",
                "type": "activates",
                "when": {"dimension_id": "dim-step", "hypothesis_ids": ["hyp-step-csa"]},
                "then": {"note": "step-size adaptation active"},
                "target_dimension_id": "dim-bounds",
            },
            {
                "id": "rel-2",
                "type": "requires",
                "when": {"dimension_id": "dim-other", "hypothesis_ids": ["hyp-other"]},
                "then": None,
                "target_dimension_id": "dim-step",
            },
        ],
        "guidance": [
            {
                "id": "g-01",
                "section": "deprioritize",
                "effect": "deprioritize",
                "claim": "Deprioritized but scope-matching entry.",
                "scope": {"model_families": ["gaussian_es"]},
            },
            {
                "id": "g-02",
                "section": "pitfall",
                "effect": "caution",
                "claim": "Origin-centered init plateaus far from the optimum.",
                "scope": {"model_families": ["gaussian_es"], "metrics": ["mean_log10"]},
            },
            {
                "id": "g-03",
                "section": "pitfall",
                "effect": "caution",
                "claim": "Unrelated pitfall about particle swarms.",
                "scope": {"model_families": ["particle_swarm"]},
            },
        ],
        "sources": [
            {
                "id": "src-01",
                "url": "https://arxiv.org/abs/1604.00772",
                "title": "The CMA Evolution Strategy: A Tutorial",
            },
            {
                "id": "src-02",
                "url": "https://example.com/es-notes",
                "title": "ES field notes",
            },
        ],
    }


def _write_run(
    run: Path,
    *,
    retrieval: dict | None = None,
    retrieval_files: dict[str, str] | None = None,
) -> Path:
    run.mkdir(parents=True)
    (run / "background.md").write_text(
        "# Background\n\n## Search space registry\n\n```json\n"
        + json.dumps(_registry(), indent=1)
        + "\n```\n",
        encoding="utf-8",
    )
    if retrieval is None:
        retrieval = {
            "schema_version": 4,
            "rounds": [
                {
                    "round_id": "r-01",
                    "results": [
                        {
                            "canonical_key": "arxiv:1604.00772",
                            "url": "https://arxiv.org/abs/1604.00772",
                            "title": "The CMA Evolution Strategy: A Tutorial",
                            "snippet": "CSA updates the global step size from the evolution path.",
                        }
                    ],
                }
            ],
            "visits": [
                {
                    "url": "https://example.com/es-notes",
                    "canonical_key": "example.com/es-notes",
                    "status": "success",
                    "view": "full_text",
                    "content_file": "retrieval/000-example-com-es-notes.txt",
                    "content_chars": len("Field notes on mirror bounds handling."),
                }
            ],
        }
        retrieval_files = {
            "retrieval/000-example-com-es-notes.txt": "Field notes on mirror bounds handling."
        }
    (run / "background_retrieval.json").write_text(json.dumps(retrieval), encoding="utf-8")
    for relative, content in (retrieval_files or {}).items():
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (run / "experience.seed.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "summary": "Two runs observed.",
                "lessons": [
                    {
                        "kind": "feasibility",
                        "claim": "Mirror bounds never crashed.",
                        "evidence": ["000"],
                        "confidence": "high",
                    }
                ],
                "bottlenecks": [
                    {
                        "claim": "Rosenbrock plateaus without covariance.",
                        "evidence": ["001"],
                        "confidence": "low",
                    }
                ],
                "dimension_evidence": [
                    {
                        "target_id": "dim-step",
                        "evaluation_state": "comparator_covered",
                        "assessment": "promising",
                        "claim": "CSA variants improved twice.",
                        "comparator_coverage": {"direct_tuned_edges": 2},
                        "confidence": "med",
                    },
                    {
                        "target_id": "dim-elsewhere",
                        "evaluation_state": "observed",
                        "assessment": "mixed",
                        "claim": "irrelevant dimension evidence",
                        "comparator_coverage": {},
                        "confidence": "low",
                    },
                ],
                "hypothesis_evidence": [
                    {
                        "target_id": "hyp-step-csa",
                        "evaluation_state": "observed",
                        "assessment": "promising",
                        "claim": "CSA beat fixed sigma in its bout.",
                        "comparator_coverage": {"direct_noncrash_edges": 1},
                        "confidence": "med",
                    },
                    {
                        "target_id": "hyp-elsewhere",
                        "evaluation_state": "observed",
                        "assessment": "mixed",
                        "claim": "irrelevant hypothesis evidence",
                        "comparator_coverage": {},
                        "confidence": "low",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def _write_candidate(run: Path, *, baseline: float = 1.5) -> Path:
    candidate = run / "candidates" / "001"
    candidate.mkdir(parents=True)
    (candidate / "_import.json").write_text(
        json.dumps(
            {
                "source": "runs/x/y",
                "baseline_score": baseline,
                "warm_to_tuned_delta": 0.25,
                "idea": "Single isotropic CSA-ES with mirror bounds.",
                "change": "from scratch at point-abc",
                "semantic_point": {
                    "assignments": [
                        {
                            "dimension_id": "dim-step",
                            "state": "selected",
                            "hypothesis_id": "hyp-step-csa",
                        },
                        {
                            "dimension_id": "dim-bounds",
                            "state": "selected",
                            "hypothesis_id": "hyp-bounds-mirror",
                        },
                        {"dimension_id": "dim-skip", "state": "desupported"},
                    ]
                },
                "tune_summary": {"inner_policy": "hebo24", "final_best_score": 1.5},
            }
        ),
        encoding="utf-8",
    )
    return candidate


def _trace(attempt_id: str, *, returncode: str = "0", stderr: str = "all good") -> str:
    return (
        f"attempt_id: {attempt_id}\n"
        "phase: rewrite\n"
        "method: rewrite\n"
        f"returncode: {returncode}\n"
        "timed_out: false\n"
        "elapsed_seconds: 12.3\n"
        "max_rss_kb: 1024\n"
        "\n[stdout]\nok\n"
        f"\n[stderr]\n{stderr}\n"
    )


def _add_bout(candidate: Path, bout: int, outcome: str, score, summary: str) -> None:
    rewrite_dir = candidate / "_rewrite"
    rewrite_dir.mkdir(exist_ok=True)
    with (rewrite_dir / "bouts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "bout": bout,
                    "attempt_id": f"eval-{bout:06d}",
                    "score": score,
                    "outcome": outcome,
                    "summary": summary,
                    "basis": f"basis {bout}",
                    "snapshot": f"bout-{bout:03d}.pre.py",
                }
            )
            + "\n"
        )


def _section(doc: str, title: str) -> str:
    start = doc.index(title)
    nxt = doc.find("\n## ", start + 1)
    return doc[start:] if nxt == -1 else doc[start:nxt]


def _headings(doc: str) -> list[str]:
    return [line for line in doc.splitlines() if line.startswith("## ")]


def test_cli_renders_all_six_sections_in_order(tmp_path, monkeypatch, capsys) -> None:
    run = _write_run(tmp_path / "run")
    candidate = _write_candidate(run)
    _add_bout(candidate, 1, "kept", 1.4, "tightened the sigma guard")
    _add_bout(candidate, 2, "reverted_crash", None, "removed bounds reflection")
    traces = candidate / "_traces"
    traces.mkdir()
    (traces / "eval-000002.log").write_text(
        _trace("eval-000002", returncode="1", stderr="IndexError: boom"), encoding="utf-8"
    )

    monkeypatch.setattr(
        sys, "argv", ["rewrite_context.py", "--candidate", str(candidate)]
    )
    assert rewrite_context.main() == 0
    payload = json.loads(capsys.readouterr().out)
    context_path = Path(payload["context_md"])
    assert context_path == candidate / "_rewrite" / "context.md"
    doc = context_path.read_text(encoding="utf-8")

    assert _headings(doc) == [
        "## Bout history",
        "## Eval traces",
        "## Experience",
        "## Semantic point",
        "## Source material",
        "## Candidate status",
    ]
    # bout history: both bouts, crash score rendered without a number
    assert "bout 1: kept | score 1.4 | tightened the sigma guard | basis: basis 1" in doc
    assert "bout 2: reverted_crash" in doc
    # traces: header metadata plus the crash's stderr tail
    assert "eval-000002 (_traces/)" in doc
    assert "returncode: 1" in doc
    assert "IndexError: boom" in doc
    # experience: lessons/bottlenecks plus only the evidence hitting this point
    assert "Mirror bounds never crashed." in doc
    assert "Rosenbrock plateaus without covariance." in doc
    assert "dim-step" in doc and "CSA beat fixed sigma in its bout." in doc
    assert "irrelevant dimension evidence" not in doc
    assert "irrelevant hypothesis evidence" not in doc
    # semantic point: relations/guidance lead, per-dimension blocks follow
    point = _section(doc, "## Semantic point")
    first_block = point.index("### dim-step -> hyp-step-csa")
    assert point.index("### Relations at this point") < first_block
    assert point.index("### Guidance") < first_block
    assert "How the step size adapts between generations." in point
    assert "CSA with identity covariance" in point
    assert "Hansen's tutorial motivates CSA." in point
    assert "hyp-step-full-cma: Full covariance CMA" in point
    assert "dim-skip" not in doc  # non-selected assignments are ignored
    # audit metadata is not editor-actionable: provenance/reopen_when dropped
    assert "provenance" not in doc and "literature:src-01" not in doc
    assert "reopen_when" not in doc and "Default CSA damping" not in doc
    # relations: only the one hitting both when.dimension_id and hypothesis_ids
    assert "rel-1" in point and "rel-2" not in point
    # guidance: scope-matching entries, pitfall before deprioritize, no misses
    assert "Origin-centered init plateaus" in point
    assert "Deprioritized but scope-matching entry." in point
    assert point.index("g-02") < point.index("g-01")
    assert "Unrelated pitfall" not in point
    # source material: snippet via arxiv canonical key, visit head via url equality
    sources = _section(doc, "## Source material")
    assert "CSA updates the global step size from the evolution path." in sources
    assert "Field notes on mirror bounds handling." in sources
    assert "(no retrieval content)" not in sources
    # candidate status: best from the kept bout, baseline, delta, idea
    status = _section(doc, "## Candidate status")
    assert "current_best: 1.4" in status
    assert "baseline_score: 1.5" in status
    assert "warm_to_tuned_delta: 0.25" in status
    assert "Single isotropic CSA-ES with mirror bounds." in status
    assert "hebo24" in status


def test_sections_filter_renders_only_the_requested_sections(tmp_path) -> None:
    run = _write_run(tmp_path / "run")
    candidate = _write_candidate(run)

    doc = rewrite_context.render_context(candidate, ["status", "traces"])

    assert _headings(doc) == ["## Eval traces", "## Candidate status"]


def test_missing_artifacts_render_none(tmp_path) -> None:
    candidate = tmp_path / "run" / "candidates" / "001"
    candidate.mkdir(parents=True)

    doc = rewrite_context.render_context(candidate)

    assert _headings(doc) == [
        "## Bout history",
        "## Eval traces",
        "## Experience",
        "## Semantic point",
        "## Source material",
        "## Candidate status",
    ]
    assert _section(doc, "## Bout history").count("(none)") == 1
    assert _section(doc, "## Eval traces").count("(none)") == 1
    assert _section(doc, "## Experience").count("(none)") == 1
    assert _section(doc, "## Semantic point").count("(none)") == 1
    assert _section(doc, "## Source material").count("(none)") == 1
    assert "current_best: (none)" in _section(doc, "## Candidate status")


def test_truncation_marks_overlong_fields_and_sections(tmp_path) -> None:
    run = _write_run(tmp_path / "run")
    candidate = _write_candidate(run)
    registry = _registry()
    registry["dimensions"][0]["hypotheses"][0]["claim"] = "C" * 1000
    (run / "background.md").write_text(
        "# Background\n\n## Search space registry\n\n```json\n"
        + json.dumps(registry, indent=1)
        + "\n```\n",
        encoding="utf-8",
    )
    for bout in range(1, 21):
        _add_bout(candidate, bout, "kept", 1.0, "S" * 200 + f" {bout}")

    doc = rewrite_context.render_context(candidate)

    point = _section(doc, "## Semantic point")
    assert "C" * 251 not in point
    assert "C" * 236 + "...[truncated]" in point
    history = _section(doc, "## Bout history")
    assert len(history) <= 4096
    assert history.rstrip().endswith("...[truncated]")


def test_semantic_point_section_cap_is_10kb(tmp_path) -> None:
    def render_with_extra_dimensions(extra: int) -> str:
        run = _write_run(tmp_path / f"run-{extra}")
        candidate = _write_candidate(run)
        registry = _registry()
        manifest = json.loads((candidate / "_import.json").read_text(encoding="utf-8"))
        assignments = manifest["semantic_point"]["assignments"]
        filler = "D" * 240
        for index in range(extra):
            dimension_id = f"dim-extra-{index:02d}"
            hypothesis_id = f"hyp-extra-{index:02d}"
            registry["dimensions"].append(
                {
                    "id": dimension_id,
                    "definition": filler,
                    "boundary": filler,
                    "selection_reason": filler,
                    "hypotheses": [{"id": hypothesis_id, "title": f"Extra {index}"}],
                }
            )
            assignments.append(
                {
                    "dimension_id": dimension_id,
                    "state": "selected",
                    "hypothesis_id": hypothesis_id,
                }
            )
        (run / "background.md").write_text(
            "# Background\n\n## Search space registry\n\n```json\n"
            + json.dumps(registry, indent=1)
            + "\n```\n",
            encoding="utf-8",
        )
        (candidate / "_import.json").write_text(json.dumps(manifest), encoding="utf-8")
        return _section(rewrite_context.render_context(candidate), "## Semantic point")

    surviving = render_with_extra_dimensions(8)
    assert len(surviving) > 8192  # beyond the old 8KB cap...
    assert "...[truncated]" not in surviving  # ...yet rendered in full
    capped = render_with_extra_dimensions(14)
    assert len(capped) <= 10240
    assert capped.rstrip().endswith("...[truncated]")


def test_source_chain_falls_back_to_visit_content(tmp_path) -> None:
    run = _write_run(
        tmp_path / "run",
        retrieval={
            "schema_version": 4,
            "rounds": [],
            "visits": [
                {
                    "url": "https://arxiv.org/abs/1604.00772",
                    "canonical_key": "arxiv:1604.00772",
                    "status": "success",
                    "view": "full_text",
                    "content_file": "retrieval/000-arxiv-1604-00772.txt",
                    "content_chars": len("Visited tutorial full text head."),
                },
                {
                    "url": "https://example.com/es-notes",
                    "canonical_key": "example.com/es-notes?cached=1",
                    "status": "success",
                    "view": "full_text",
                    "content_file": "retrieval/001-example-com-es-notes.txt",
                    "content_chars": len("Field notes via url equality."),
                },
            ],
        },
        retrieval_files={
            "retrieval/000-arxiv-1604-00772.txt": "Visited tutorial full text head.",
            "retrieval/001-example-com-es-notes.txt": "Field notes via url equality.",
        },
    )
    candidate = _write_candidate(run)

    sources = _section(rewrite_context.render_context(candidate), "## Source material")

    assert "Visited tutorial full text head." in sources
    assert "Field notes via url equality." in sources
    assert "(no retrieval content)" not in sources


def test_bout_history_and_trace_windows(tmp_path) -> None:
    run = _write_run(tmp_path / "run")
    candidate = _write_candidate(run)
    for bout in range(1, 26):
        _add_bout(candidate, bout, "reverted_worse", 1.6, f"bout {bout} summary")
    src = candidate / "_traces_src"
    new = candidate / "_traces"
    src.mkdir()
    new.mkdir()
    for index in range(1, 4):
        (src / f"eval-{index:06d}.log").write_text(
            _trace(f"eval-{index:06d}"), encoding="utf-8"
        )
    for index in range(4, 7):
        (new / f"eval-{index:06d}.log").write_text(
            _trace(f"eval-{index:06d}"), encoding="utf-8"
        )
    (new / "eval-000007.log").write_text(
        _trace("eval-000007", returncode="1", stderr="E" * 3000), encoding="utf-8"
    )

    doc = rewrite_context.render_context(candidate)

    history = _section(doc, "## Bout history")
    assert "bout 5:" not in history  # only the last 20 bouts
    assert "bout 6: reverted_worse" in history
    assert "bout 25: reverted_worse" in history
    traces = _section(doc, "## Eval traces")
    assert "eval-000002" not in traces  # only the last 5 attempts
    assert "eval-000003 (_traces_src/)" in traces
    assert "eval-000007 (_traces/)" in traces
    assert "E" * 2048 in traces  # crash stderr tail capped at 2KB
    assert "E" * 2049 not in traces


def test_trace_window_prefers_this_run_over_source_history(tmp_path) -> None:
    """Source attempt ids continue the source run's global counter, so they
    usually dwarf this run's; the window must still show this run's traces."""
    run = _write_run(tmp_path / "run")
    candidate = _write_candidate(run)
    src = candidate / "_traces_src"
    new = candidate / "_traces"
    src.mkdir()
    new.mkdir()
    for index in range(40, 46):
        (src / f"eval-{index:06d}.log").write_text(
            _trace(f"eval-{index:06d}"), encoding="utf-8"
        )
    (new / "eval-000001.log").write_text(_trace("eval-000001"), encoding="utf-8")

    traces = _section(rewrite_context.render_context(candidate), "## Eval traces")

    assert "eval-000001 (_traces/)" in traces
    assert "eval-000045 (_traces_src/)" in traces
    assert "eval-000041" not in traces  # source history fills the remainder
