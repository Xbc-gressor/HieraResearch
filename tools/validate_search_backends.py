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
    _check_deepxiv_progression,
    _deepxiv_section_content,
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
        offline_env = dict(os.environ)
        offline_env["HIERA_RETRIEVAL_OFFLINE"] = "1"
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
            env=offline_env,
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
            env=offline_env,
        )
        assert visit_run.returncode == 0, visit_run.stderr or visit_run.stdout
        frozen_manifest = json.loads(manifest_path.read_text())
        assert frozen_manifest["retrieval_condition"] == "frozen"
        assert frozen_manifest["backend_calls"][0]["raw_response"]
        assert frozen_manifest["visits"][0]["content_sha256"]
        assert validate_manifest(frozen_manifest) == []

        append_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(manifest_path),
                "--frozen-corpus",
                str(corpus_path),
                "--query",
                "tabular classification",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=offline_env,
        )
        assert append_run.returncode == 0, append_run.stderr or append_run.stdout
        appended_manifest = json.loads(manifest_path.read_text())
        assert [query["id"] for query in appended_manifest["queries"]] == ["q-01", "q-02"]
        assert len(appended_manifest["backend_calls"]) == 2
        assert len(appended_manifest["candidates"]) == 2
        assert len(appended_manifest["visits"]) == 1
        assert appended_manifest["results"][0]["query_support"] == 2
        assert validate_manifest(appended_manifest) == []

        frozen_probe = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "probe",
                "--backend",
                "frozen",
                "--frozen-corpus",
                str(corpus_path),
            ],
            text=True,
            capture_output=True,
            check=False,
            env=offline_env,
        )
        assert frozen_probe.returncode == 0, frozen_probe.stderr or frozen_probe.stdout
        assert json.loads(frozen_probe.stdout)["source"] == "frozen_corpus"

        native_results = Path(tmp) / "native-results.json"
        native_results.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "url": "https://arxiv.org/abs/2203.11171v3",
                            "title": "A native search result",
                            "snippet": "A retained native search snippet.",
                        }
                    ]
                }
            )
        )
        native_manifest_path = Path(tmp) / "native-retrieval.json"
        native_record = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "record-search",
                "--manifest",
                str(native_manifest_path),
                "--query",
                "native tree search",
                "--results-file",
                str(native_results),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert native_record.returncode == 0, native_record.stderr or native_record.stdout
        native_manifest = json.loads(native_manifest_path.read_text())
        assert native_manifest["retrieval_condition"] == "open_world"
        assert native_manifest["results"][0]["canonical_key"] == "arxiv:2203.11171"
        assert native_manifest["backend_calls"][0]["backend"] == "claude-websearch"
        assert validate_manifest(native_manifest) == []

        native_record_second = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "record-search",
                "--manifest",
                str(native_manifest_path),
                "--query",
                "native failure-mode search",
                "--results-file",
                str(native_results),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert native_record_second.returncode == 0, (
            native_record_second.stderr or native_record_second.stdout
        )
        native_manifest = json.loads(native_manifest_path.read_text())
        assert len(native_manifest["results"]) == 1
        assert native_manifest["results"][0]["query_support"] == 2
        assert native_manifest["selected_keys"] == ["arxiv:2203.11171"]
        assert validate_manifest(native_manifest) == []

        disabled_env = dict(os.environ)
        disabled_env["HIERA_RETRIEVAL_DISABLE_BACKENDS"] = "deepxiv"
        disabled_probe = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "probe",
                "--backend",
                "deepxiv",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=disabled_env,
        )
        assert disabled_probe.returncode == 1
        assert "disabled" in disabled_probe.stdout

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

        offline_live = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(Path(tmp) / "offline-live.json"),
                "--backend",
                "deepxiv",
                "--query",
                "regularized trees",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=offline_env,
        )
        assert offline_live.returncode == 1
        assert "requires the frozen backend" in offline_live.stderr

        offline_direct_visit = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "visit",
                "--manifest",
                str(manifest_path),
                "--url",
                "https://paper.test/trees",
                "--visit-backend",
                "direct",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=offline_env,
        )
        assert offline_direct_visit.returncode == 1
        assert "requires --frozen-corpus" in offline_direct_visit.stderr

        fake_bin = Path(tmp) / "bin"
        fake_bin.mkdir()
        fake_deepxiv = fake_bin / "deepxiv"
        fake_deepxiv.write_text(
            """#!/usr/bin/env python3
import json
import sys

if "--version" in sys.argv:
    print("deepxiv-fixture-1")
elif "--head" in sys.argv:
    print(json.dumps({"title": "Fixture", "sections": [{"name": "Methods"}]}))
elif "--section" in sys.argv:
    print(json.dumps({"arxiv_id": "2203.11171", "section": "Methods", "content": "S" * 30000}))
else:
    raise SystemExit(2)
"""
        )
        fake_deepxiv.chmod(0o755)
        fake_env = dict(os.environ)
        fake_env["PATH"] = str(fake_bin) + os.pathsep + fake_env.get("PATH", "")
        deepxiv_manifest_path = Path(tmp) / "deepxiv-retrieval.json"
        for view_args in (["--view", "head"], ["--view", "section", "--section", "Methods"]):
            deepxiv_visit = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("search_backends.py")),
                    "visit",
                    "--manifest",
                    str(deepxiv_manifest_path),
                    "--url",
                    "https://arxiv.org/abs/2203.11171",
                    *view_args,
                ],
                text=True,
                capture_output=True,
                check=False,
                env=fake_env,
            )
            assert deepxiv_visit.returncode == 0, (
                deepxiv_visit.stderr or deepxiv_visit.stdout
            )
        deepxiv_manifest = json.loads(deepxiv_manifest_path.read_text())
        retained_section = deepxiv_manifest["visits"][1]
        assert retained_section["view"] == "section"
        assert retained_section["content"] == "S" * (LANE_BUDGETS["grounding"] * 4)
        assert retained_section["original_content_chars"] == 30000
        assert retained_section["content_truncated"] is True
        assert validate_manifest(deepxiv_manifest) == []

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
        section="Methods",
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

    section_body = "method details " * 4000
    extracted = _deepxiv_section_content(
        json.dumps(
            {"arxiv_id": "2203.11171", "section": "Methods", "content": section_body}
        )
    )
    assert extracted == section_body

    progressive = new_manifest()
    add_visit(
        progressive,
        url="https://arxiv.org/abs/2203.11171",
        lane="grounding",
        backend="deepxiv",
        view="head",
        status="success",
        content=json.dumps(
            {"title": "Fixture", "sections": [{"name": "Methods"}, {"name": "Results"}]}
        ),
    )
    _check_deepxiv_progression(
        progressive, "https://arxiv.org/abs/2203.11171", "grounding", "section", "Methods"
    )
    try:
        _check_deepxiv_progression(
            progressive,
            "https://arxiv.org/abs/2203.11171",
            "grounding",
            "section",
            "Method",
        )
        raise AssertionError("partial section names must be rejected")
    except RuntimeError as exc:
        assert "exactly match" in str(exc)
    try:
        _check_deepxiv_progression(
            progressive,
            "https://arxiv.org/abs/2203.11171",
            "grounding",
            "full_text",
            None,
        )
        raise AssertionError("full text must not bypass available named sections")
    except RuntimeError as exc:
        assert "fallback" in str(exc)
    add_visit(
        progressive,
        url="https://arxiv.org/abs/2203.11171",
        lane="grounding",
        backend="deepxiv",
        view="section",
        section="Method",
        status="failed",
        error="fixture invalid section name",
    )
    try:
        _check_deepxiv_progression(
            progressive,
            "https://arxiv.org/abs/2203.11171",
            "grounding",
            "full_text",
            None,
        )
        raise AssertionError("an invalid section request must not unlock full text")
    except RuntimeError as exc:
        assert "fallback" in str(exc)
    add_visit(
        progressive,
        url="https://arxiv.org/abs/2203.11171",
        lane="grounding",
        backend="deepxiv",
        view="section",
        section="Methods",
        status="failed",
        error="fixture section failure",
    )
    _check_deepxiv_progression(
        progressive,
        "https://arxiv.org/abs/2203.11171",
        "grounding",
        "full_text",
        None,
    )

    truncated = new_manifest()
    add_visit(
        truncated,
        url="https://example.test/source",
        lane="grounding",
        backend="fixture",
        view="full_text",
        status="success",
        content="retained",
        original_content_chars=100,
    )
    assert truncated["visits"][0]["content_truncated"] is True
    assert validate_manifest(truncated) == []

    assert LANE_BUDGETS["grounding"] > LANE_BUDGETS["novelty"]
    print("Search backend, ranking, and visit-integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
