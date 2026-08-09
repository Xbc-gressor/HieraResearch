#!/usr/bin/env python3
"""Network-free regression checks for the background retrieval layer."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from search_backends import (
    LANE_BUDGETS,
    FrozenCorpusBackend,
    SearchBackend,
    _validate_query_plan,
    add_visit,
    canonical_key,
    dispatch_search,
    merge_candidates,
    new_manifest,
    select_balanced,
    validate_manifest,
)


class FakeBackend(SearchBackend):
    def __init__(self, name: str, rows: dict[str, list[dict]] | None = None, fail: bool = False):
        self.name = name
        self.rows = rows or {}
        self.fail = fail

    async def search(self, query: str, max_results: int) -> dict:
        if self.fail:
            raise RuntimeError("backend unavailable")
        items = self.rows.get(query, [])[:max_results]
        return {"items": items, "raw_response": {"items": items}, "metadata": {}}


def main() -> int:
    assert canonical_key("https://arxiv.org/abs/2203.11171v4") == "arxiv:2203.11171"
    assert canonical_key("https://www.alphaxiv.org/pdf/2203.11171") == "arxiv:2203.11171"
    assert canonical_key("https://Example.com/a/?utm_source=x") == "example.com/a"

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = Path(tmp) / "frozen.json"
        corpus_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "corpus_id": "fixture-v1",
                    "cutoff": "2025-01-01",
                    "created_at": "2025-01-02T00:00:00Z",
                    "provenance": "Synthetic network-free regression fixture.",
                    "prepared_before_task_ids": True,
                    "items": [
                        {
                            "url": "https://paper.test/trees",
                            "title": "Regularized trees",
                            "abstract": "Regularized tree models for tabular classification.",
                            "text": "Full retained method and results for regularized trees.",
                        }
                    ],
                }
            )
        )
        frozen = FrozenCorpusBackend(corpus_path)
        response = asyncio.run(frozen.search("regularized trees", 5))
        assert response["items"][0]["url"] == "https://paper.test/trees"
        assert "Full retained" in frozen.read("https://paper.test/trees")
        assert response["metadata"]["corpus_id"] == "fixture-v1"

        manifest_path = Path(tmp) / "retrieval.json"
        search_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(manifest_path),
                "--frozen-corpus",
                str(corpus_path),
                "--query-spec",
                json.dumps(
                    {
                        "text": "regularized trees",
                        "target_dimension_ids": ["dim-method-choice"],
                        "evidence_roles": ["hypothesis"],
                    }
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert search_run.returncode == 0, search_run.stderr or search_run.stdout
        visit_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "visit",
                "--manifest",
                str(manifest_path),
                "--frozen-corpus",
                str(corpus_path),
                "--url",
                "https://paper.test/trees",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert visit_run.returncode == 0, visit_run.stderr or visit_run.stdout
        frozen_manifest = json.loads(manifest_path.read_text())
        assert frozen_manifest["retrieval_condition"] == "frozen"
        assert frozen_manifest["backend_calls"][0]["raw_response"]
        assert frozen_manifest["visits"][0]["content"]
        assert validate_manifest(frozen_manifest) == []

        implicit_external = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(Path(tmp) / "must-not-run.json"),
                "--query-spec",
                json.dumps(
                    {
                        "text": "regularized trees",
                        "target_dimension_ids": ["dim-method-choice"],
                        "evidence_roles": ["hypothesis"],
                    }
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert implicit_external.returncode == 1
        assert "select --backend explicitly" in implicit_external.stderr

        fake_deepxiv = Path(tmp) / "deepxiv"
        fake_deepxiv.write_text(
            """#!/usr/bin/env python3
import json
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("deepxiv fake-1")
elif args and args[0] == "paper" and "--head" in args:
    paper_id = args[1]
    print(json.dumps({
        "title": "Progressive fixture",
        "abstract": "A metadata-only abstract.",
        "sections": [] if paper_id in {"2409.05592", "2409.05593"} else [
            {"name": "Introduction", "idx": 1, "tldr": "Motivation and context.", "token_count": 700},
            {"name": "Method", "idx": 2, "tldr": "The proposed mechanism.", "token_count": 1800},
            {"name": "Results", "idx": 3, "tldr": "Comparators and ablations.", "token_count": 1900},
            {"name": "Limitations", "idx": 4, "tldr": "Known failure regimes.", "token_count": 600},
            {"name": "References", "idx": 5, "tldr": "", "token_count": 1000}
        ]
    }))
elif args and args[0] == "paper" and "--section" in args:
    name = args[args.index("--section") + 1]
    print(json.dumps({
        "section": name,
        "content": "Primary source body for " + name + ". " + ("evidence " * 100)
    }))
elif args and args[0] == "paper" and "--preview" in args:
    if args[1] == "2409.05593":
        print("preview unavailable", file=sys.stderr)
        raise SystemExit(3)
    print(json.dumps({"content": "Fallback primary-source preview."}))
else:
    print("unsupported fake DeepXiv invocation", file=sys.stderr)
    raise SystemExit(2)
"""
        )
        fake_deepxiv.chmod(0o755)
        progressive_manifest_path = Path(tmp) / "progressive.json"
        progressive_manifest_path.write_text(json.dumps(new_manifest()))
        progressive_env = dict(os.environ)
        progressive_env["PATH"] = str(Path(tmp)) + os.pathsep + progressive_env.get("PATH", "")
        progressive_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "visit",
                "--manifest",
                str(progressive_manifest_path),
                "--url",
                "https://arxiv.org/abs/2409.05591",
                "--view",
                "auto",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert progressive_run.returncode == 0, progressive_run.stderr or progressive_run.stdout
        progressive_manifest = json.loads(progressive_manifest_path.read_text())
        successful_visits = [
            visit for visit in progressive_manifest["visits"] if visit["status"] == "success"
        ]
        assert successful_visits[0]["view"] == "head"
        section_visits = [
            visit for visit in successful_visits if visit["view"] == "section"
        ]
        assert {visit["section"] for visit in section_visits} == {
            "Method",
            "Results",
            "Limitations",
        }
        assert "Primary source body for Method" in progressive_run.stdout
        assert sum(visit["content_chars"] for visit in successful_visits) <= (
            LANE_BUDGETS["grounding"] * 4
        )
        assert validate_manifest(progressive_manifest) == []

        preview_manifest_path = Path(tmp) / "preview-fallback.json"
        preview_manifest_path.write_text(json.dumps(new_manifest()))
        preview_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "visit",
                "--manifest",
                str(preview_manifest_path),
                "--url",
                "https://arxiv.org/abs/2409.05592",
                "--view",
                "auto",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert preview_run.returncode == 0, preview_run.stderr or preview_run.stdout
        preview_manifest = json.loads(preview_manifest_path.read_text())
        assert [visit["view"] for visit in preview_manifest["visits"]] == [
            "head",
            "preview",
        ]
        assert validate_manifest(preview_manifest) == []

        head_only_manifest_path = Path(tmp) / "head-only-failure.json"
        head_only_manifest_path.write_text(json.dumps(new_manifest()))
        head_only_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "visit",
                "--manifest",
                str(head_only_manifest_path),
                "--url",
                "https://arxiv.org/abs/2409.05593",
                "--view",
                "auto",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert head_only_run.returncode == 1
        assert "head metadata but no substantive" in head_only_run.stderr
        head_only_manifest = json.loads(head_only_manifest_path.read_text())
        assert [visit["view"] for visit in head_only_manifest["visits"]] == [
            "head",
            "preview",
        ]
        assert head_only_manifest["visits"][-1]["status"] == "failed"
        assert validate_manifest(head_only_manifest) == []

    queries = [
        {
            "id": "q-01",
            "text": "regularized trees",
            "lane": "grounding",
            "target_dimension_ids": ["dim-method-choice"],
            "evidence_roles": ["hypothesis"],
        },
        {
            "id": "q-02",
            "text": "tree baseline failures",
            "lane": "grounding",
            "target_dimension_ids": ["dim-method-choice"],
            "evidence_roles": ["baseline", "failure_mode"],
        },
    ]
    deepxiv = FakeBackend(
        "deepxiv",
        {
            "regularized trees": [
                {
                    "url": "https://arxiv.org/abs/2203.11171v2",
                    "title": "Shared paper",
                    "snippet": "short",
                },
                {"url": "https://only-a.test", "title": "A", "snippet": "a"},
            ],
            "tree baseline failures": [
                {
                    "url": "https://arxiv.org/abs/2203.11171",
                    "title": "Shared paper",
                    "snippet": "longer shared snippet",
                },
                {"url": "https://only-b.test", "title": "B", "snippet": "b"},
            ],
        },
    )
    jina = FakeBackend(
        "jina",
        {
            "regularized trees": [
                {
                    "url": "https://www.alphaxiv.org/pdf/2203.11171",
                    "title": "Shared paper mirror",
                    "snippet": "mirror",
                }
            ]
        },
    )
    broken = FakeBackend("broken", fail=True)
    raw, failures, calls = asyncio.run(dispatch_search(queries, [deepxiv, jina, broken], 10))
    assert len(failures) == 2

    merged = merge_candidates(raw)
    shared = next(item for item in merged if item["canonical_key"] == "arxiv:2203.11171")
    assert shared["query_support"] == 2
    assert shared["backend_support"] == 2
    assert shared["snippet"] == "longer shared snippet"

    selected = select_balanced(merged, ["q-01", "q-02"])
    assert selected[0] == "arxiv:2203.11171"
    assert "only-a.test/" in selected
    assert "only-b.test/" in selected

    manifest = new_manifest()
    manifest["retrieval_condition"] = "open_world"
    manifest["queries"] = queries
    manifest["results"] = merged
    manifest["selected_keys"] = selected
    manifest["backend_calls"] = calls
    manifest["backend_failures"] = failures
    add_visit(
        manifest,
        url="https://arxiv.org/abs/2203.11171",
        lane="grounding",
        backend="deepxiv",
        view="section",
        section="Method",
        status="success",
        content="inspected method and results" * 40,
    )
    manifest_errors = validate_manifest(manifest)
    assert manifest_errors == [], manifest_errors

    broken_budget = new_manifest()
    broken_budget["lane_budgets"] = {"novelty": 3000, "grounding": 2000}
    errors = validate_manifest(broken_budget)
    assert any("grounding token budget must exceed" in error for error in errors), errors

    zero_content = new_manifest()
    add_visit(
        zero_content,
        url="https://example.test/source",
        lane="novelty",
        backend="claude-webfetch",
        view="abstract",
        status="success",
        content="",
    )
    errors = validate_manifest(zero_content)
    assert any("content_chars must be positive" in error for error in errors), errors

    assert LANE_BUDGETS["grounding"] > LANE_BUDGETS["novelty"]

    def plan_query(roles, targets=()):
        return {
            "id": "q-01",
            "text": "planned",
            "lane": "grounding",
            "target_dimension_ids": list(targets),
            "evidence_roles": list(roles),
        }

    for roles in (["hypothesis"], ["relation"], ["baseline", "hypothesis"]):
        errors, _ = _validate_query_plan([plan_query(roles)], [])
        assert any("must target at least one dimension" in e for e in errors), (roles, errors)
    errors, _ = _validate_query_plan(
        [plan_query(["baseline", "failure_mode", "counterevidence"])], []
    )
    assert errors == [], errors
    errors, _ = _validate_query_plan(
        [plan_query(["relation"], targets=("dim-method-choice",))], []
    )
    assert errors == [], errors
    print("Search backend, ranking, and visit-integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
