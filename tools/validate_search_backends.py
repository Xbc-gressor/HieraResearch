#!/usr/bin/env python3
"""Network-free regression checks for the background retrieval layer."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from search_backends import (
    LANE_BUDGETS,
    FrozenCorpusBackend,
    SearchBackend,
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
                "--query",
                "regularized trees",
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
        assert frozen_manifest["visits"][0]["content_sha256"]
        assert validate_manifest(frozen_manifest) == []

        implicit_external = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(Path(tmp) / "must-not-run.json"),
                "--query",
                "regularized trees",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert implicit_external.returncode == 1
        assert "select --backend explicitly" in implicit_external.stderr

    queries = [
        {"id": "q-01", "text": "regularized trees", "lane": "grounding"},
        {"id": "q-02", "text": "tree baseline failures", "lane": "grounding"},
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
    print("Search backend, ranking, and visit-integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
