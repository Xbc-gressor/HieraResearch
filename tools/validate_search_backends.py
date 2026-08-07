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
        search_cmd = [
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
        ]
        search_run = subprocess.run(
            search_cmd,
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

        research_run = subprocess.run(
            search_cmd,
            text=True,
            capture_output=True,
            check=False,
        )
        assert research_run.returncode == 0, research_run.stderr or research_run.stdout
        researched_manifest = json.loads(manifest_path.read_text())
        assert [visit["url"] for visit in researched_manifest["visits"]] == [
            "https://paper.test/trees"
        ], "re-running search must retain prior visit receipts"
        assert validate_manifest(researched_manifest) == []

        corpus_path_2 = Path(tmp) / "frozen-v2.json"
        corpus_path_2.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "corpus_id": "fixture-v2",
                    "cutoff": "2025-06-01",
                    "created_at": "2025-06-02T00:00:00Z",
                    "provenance": "Synthetic network-free regression fixture, revised.",
                    "prepared_before_task_ids": True,
                    "items": [
                        {
                            "url": "https://paper.test/trees-v2",
                            "title": "Regularized trees revisited",
                            "abstract": "Regularized tree models, revised edition.",
                            "text": "Revised retained method and results for regularized trees.",
                        }
                    ],
                }
            )
        )
        other_corpus_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(manifest_path),
                "--frozen-corpus",
                str(corpus_path_2),
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
        assert other_corpus_run.returncode == 0, (
            other_corpus_run.stderr or other_corpus_run.stdout
        )
        other_corpus_manifest = json.loads(manifest_path.read_text())
        assert other_corpus_manifest["visits"] == [], (
            "re-search under a different frozen corpus must not retain prior visits"
        )
        assert validate_manifest(other_corpus_manifest) == []

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
import os
import sys
from pathlib import Path

args = sys.argv[1:]
if args == ["--version"]:
    print("deepxiv fake-1")
elif args and args[0] == "search":
    print(json.dumps({"result": [
        {
            "arxiv_id": "2409.05591",
            "title": "Progressive fixture",
            "abstract": "A metadata-only abstract.",
            "score": 0.87,
            "categories": ["cs.LG", "stat.ML"],
            "venue": "ICLR",
            "citation_count": 12
        },
        {
            "arxiv_id": "2409.05592",
            "title": "Second fixture",
            "tldr": "Second fixture summary.",
            "score": 0.41,
            "categories": ["cs.CV"],
            "venue": None,
            "citation_count": 0
        }
    ]}))
elif args and args[0] == "paper" and "--head" in args:
    paper_id = args[1]
    if paper_id in {"2409.05594", "2409.05595"}:
        state = Path(os.environ["FAKE_DEEPXIV_STATE"]) / (paper_id + ".count")
        seen = int(state.read_text()) if state.exists() else 0
        state.write_text(str(seen + 1))
        if paper_id == "2409.05595" or seen < 2:
            print("Daily limit reached. Visit https://data.rag.ac.cn/register", file=sys.stderr)
            raise SystemExit(1)
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
    if name == "Nonexistent":
        print("ValueError: Section 'Nonexistent' not found", file=sys.stderr)
        raise SystemExit(1)
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
        state_dir = Path(tmp) / "fake-state"
        state_dir.mkdir()
        progressive_env["FAKE_DEEPXIV_STATE"] = str(state_dir)
        progressive_env["HIERA_RETRY_BASE_SECONDS"] = "0"
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

        def visit_args(manifest_path, *urls, extra=()):
            command = [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "visit",
                "--manifest",
                str(manifest_path),
            ]
            for url in urls:
                command += ["--url", url]
            return command + list(extra)

        # A batch reads every URL in one process and writes the manifest once.
        batch_manifest_path = Path(tmp) / "batch.json"
        batch_manifest_path.write_text(json.dumps(new_manifest()))
        batch_run = subprocess.run(
            visit_args(
                batch_manifest_path,
                "https://arxiv.org/abs/2409.05591",
                "https://arxiv.org/abs/2409.05592",
            ),
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert batch_run.returncode == 0, batch_run.stderr or batch_run.stdout
        batch_manifest = json.loads(batch_manifest_path.read_text())
        assert {visit["canonical_key"] for visit in batch_manifest["visits"]} == {
            "arxiv:2409.05591",
            "arxiv:2409.05592",
        }
        assert "arxiv.org/abs/2409.05591 =====" in batch_run.stdout
        assert validate_manifest(batch_manifest) == []

        # Partial failure is its own exit code, and names the failed URL.
        mixed_manifest_path = Path(tmp) / "mixed-batch.json"
        mixed_manifest_path.write_text(json.dumps(new_manifest()))
        mixed_run = subprocess.run(
            visit_args(
                mixed_manifest_path,
                "https://arxiv.org/abs/2409.05591",
                "https://arxiv.org/abs/2409.05593",
            ),
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert mixed_run.returncode == 2, mixed_run.stderr or mixed_run.stdout
        assert "2409.05593" in mixed_run.stderr
        assert "2409.05591" not in mixed_run.stderr
        assert validate_manifest(json.loads(mixed_manifest_path.read_text())) == []

        all_failed_manifest_path = Path(tmp) / "all-failed-batch.json"
        all_failed_manifest_path.write_text(json.dumps(new_manifest()))
        all_failed_run = subprocess.run(
            visit_args(
                all_failed_manifest_path,
                "https://arxiv.org/abs/2409.05593",
                "https://arxiv.org/abs/2409.05595",
            ),
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert all_failed_run.returncode == 1, all_failed_run.stdout

        # An explicit head read is a success even though head is not substantive.
        head_view_manifest_path = Path(tmp) / "head-view.json"
        head_view_manifest_path.write_text(json.dumps(new_manifest()))
        head_view_run = subprocess.run(
            visit_args(
                head_view_manifest_path,
                "https://arxiv.org/abs/2409.05593",
                extra=("--view", "head"),
            ),
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert head_view_run.returncode == 0, head_view_run.stderr or head_view_run.stdout
        head_view_manifest = json.loads(head_view_manifest_path.read_text())
        assert [visit["view"] for visit in head_view_manifest["visits"]] == ["head"]
        assert validate_manifest(head_view_manifest) == []

        # A 429-class failure is retried; a non-429 failure is not.
        retry_manifest_path = Path(tmp) / "retry.json"
        retry_manifest_path.write_text(json.dumps(new_manifest()))
        retry_run = subprocess.run(
            visit_args(retry_manifest_path, "https://arxiv.org/abs/2409.05594"),
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert retry_run.returncode == 0, retry_run.stderr or retry_run.stdout
        retry_manifest = json.loads(retry_manifest_path.read_text())
        assert retry_manifest["visits"][0]["retry_count"] == 2, retry_manifest["visits"][0]
        assert validate_manifest(retry_manifest) == []

        rate_limited_manifest_path = Path(tmp) / "rate-limited.json"
        rate_limited_manifest_path.write_text(json.dumps(new_manifest()))
        rate_limited_run = subprocess.run(
            visit_args(rate_limited_manifest_path, "https://arxiv.org/abs/2409.05595"),
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert rate_limited_run.returncode == 1
        rate_limited_manifest = json.loads(rate_limited_manifest_path.read_text())
        assert rate_limited_manifest["visits"][0]["retry_count"] == 2

        section_failure_manifest_path = Path(tmp) / "section-no-retry.json"
        section_failure_manifest_path.write_text(json.dumps(new_manifest()))
        section_failure_run = subprocess.run(
            visit_args(
                section_failure_manifest_path,
                "https://arxiv.org/abs/2409.05592",
                extra=("--view", "section", "--section", "Nonexistent"),
            ),
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        # The named section is absent, so the read falls back to auto ranking
        # and both receipts are kept. 2409.05592 has no sections, so the
        # fallback lands on preview.
        assert section_failure_run.returncode == 0, section_failure_run.stderr
        section_failure_manifest = json.loads(section_failure_manifest_path.read_text())
        assert [visit["view"] for visit in section_failure_manifest["visits"]] == [
            "section",
            "head",
            "preview",
        ]
        assert section_failure_manifest["visits"][0]["status"] == "failed"
        assert section_failure_manifest["visits"][0]["retry_count"] == 0, (
            "a missing section is not a rate limit and must not be retried"
        )
        assert validate_manifest(section_failure_manifest) == []

        deepxiv_search_path = Path(tmp) / "deepxiv-search.json"
        deepxiv_search_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(deepxiv_search_path),
                "--lane",
                "grounding",
                "--query-spec",
                json.dumps(
                    {
                        "text": "fixture mechanism",
                        "target_dimension_ids": ["dim-method-choice"],
                        "evidence_roles": ["hypothesis"],
                    }
                ),
                "--backend",
                "deepxiv",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert deepxiv_search_run.returncode == 0, (
            deepxiv_search_run.stderr or deepxiv_search_run.stdout
        )
        deepxiv_manifest = json.loads(deepxiv_search_path.read_text())
        assert [item["external_id"] for item in deepxiv_manifest["results"]] == [
            "2409.05591",
            "2409.05592",
        ]
        assert deepxiv_manifest["results"][0]["retrieval_signals"] == [
            {
                "query_id": "q-01",
                "backend": "deepxiv",
                "rank": 1,
                "score": 0.87,
                "categories": ["cs.LG", "stat.ML"],
                "venue": "ICLR",
                "citation_count": 12,
            }
        ]
        assert deepxiv_manifest["results"][1]["retrieval_signals"][0]["score"] == 0.41
        assert deepxiv_manifest["results"][1]["retrieval_signals"][0]["venue"] is None
        assert validate_manifest(deepxiv_manifest) == []

        fetched = Path(tmp) / "fetched-content.txt"
        fetched.write_text("external web evidence " * 50)
        record_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "record-visit",
                "--manifest",
                str(deepxiv_search_path),
                "--lane",
                "grounding",
                "--backend",
                "claude-webfetch",
                "--view",
                "page",
                "--status",
                "success",
                "--content-file",
                str(fetched),
                "--url",
                "https://example.test/evidence",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert record_run.returncode == 0, record_run.stderr or record_run.stdout

        deepxiv_spec = json.dumps(
            {
                "text": "fixture mechanism",
                "target_dimension_ids": ["dim-method-choice"],
                "evidence_roles": ["hypothesis"],
            }
        )
        same_condition_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(deepxiv_search_path),
                "--lane",
                "grounding",
                "--query-spec",
                deepxiv_spec,
                "--backend",
                "deepxiv",
            ],
            text=True,
            capture_output=True,
            check=False,
            env=progressive_env,
        )
        assert same_condition_run.returncode == 0, (
            same_condition_run.stderr or same_condition_run.stdout
        )
        assert len(json.loads(deepxiv_search_path.read_text())["visits"]) == 1, (
            "re-search under the same retrieval condition must retain visits"
        )

        changed_condition_run = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("search_backends.py")),
                "search",
                "--manifest",
                str(deepxiv_search_path),
                "--lane",
                "grounding",
                "--query-spec",
                json.dumps(
                    {
                        "text": "regularized trees",
                        "target_dimension_ids": ["dim-method-choice"],
                        "evidence_roles": ["hypothesis"],
                    }
                ),
                "--frozen-corpus",
                str(corpus_path),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        assert changed_condition_run.returncode == 0, (
            changed_condition_run.stderr or changed_condition_run.stdout
        )
        assert json.loads(deepxiv_search_path.read_text())["visits"] == [], (
            "re-search under a changed retrieval condition must not retain visits"
        )

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
                    "score": 0.05,
                },
                {"url": "https://only-a.test", "title": "A", "snippet": "a", "score": 0.01},
            ],
            "tree baseline failures": [
                {
                    "url": "https://arxiv.org/abs/2203.11171",
                    "title": "Shared paper",
                    "snippet": "longer shared snippet",
                    "score": 0.05,
                },
                {"url": "https://only-b.test", "title": "B", "snippet": "b", "score": 0.99},
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
    assert [item["canonical_key"] for item in merged] == [
        "arxiv:2203.11171",
        "only-a.test/",
        "only-b.test/",
    ], "retrieval scores must not influence merge ordering"
    shared = next(item for item in merged if item["canonical_key"] == "arxiv:2203.11171")
    assert shared["query_support"] == 2
    assert shared["backend_support"] == 2
    assert shared["snippet"] == "longer shared snippet"
    assert shared["retrieval_signals"] == [
        {
            "query_id": "q-01",
            "backend": "deepxiv",
            "rank": 1,
            "score": 0.05,
            "categories": None,
            "venue": None,
            "citation_count": None,
        },
        {
            "query_id": "q-01",
            "backend": "jina",
            "rank": 1,
            "score": None,
            "categories": None,
            "venue": None,
            "citation_count": None,
        },
        {
            "query_id": "q-02",
            "backend": "deepxiv",
            "rank": 1,
            "score": 0.05,
            "categories": None,
            "venue": None,
            "citation_count": None,
        },
    ]
    only_b = next(item for item in merged if item["canonical_key"] == "only-b.test/")
    assert only_b["retrieval_signals"][0]["score"] == 0.99

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

    # Sustained rate limiting halves concurrency rather than aborting, and only
    # gives up once there is nothing left to slow down.
    import search_backends

    waves: list[int] = []
    original_visit_source = search_backends._visit_source

    def stub(url: str, **kwargs) -> dict:
        return {
            "url": url,
            "attempts": [],
            "ok": False,
            "error": "RuntimeError: Daily limit reached.",
            "rate_limited": True,
            "rendered": "",
        }

    original_gather = asyncio.gather

    async def counting_gather(*coroutines, **kwargs):
        waves.append(len(coroutines))
        return await original_gather(*coroutines, **kwargs)

    search_backends._visit_source = stub
    asyncio.gather = counting_gather
    try:
        outcomes, gave_up = asyncio.run(
            search_backends._dispatch_visits(
                [f"https://arxiv.org/abs/24{index:02d}.00001" for index in range(12)],
                max_concurrency=4,
                read_kwargs={"view": "auto", "section": None},
            )
        )
    finally:
        search_backends._visit_source = original_visit_source
        asyncio.gather = original_gather
    # Each trip halves and restarts the streak, so three more consecutive
    # rate-limited URLs are needed before halving again.
    assert waves == [4, 2, 2, 1, 1, 1], waves
    assert gave_up, "a trip at concurrency 1 must stop issuing requests"
    assert len(outcomes) == 12, "every URL still gets a receipt-bearing outcome"
    assert sum(1 for outcome in outcomes if "skipped" in (outcome["error"] or "")) == 1

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
