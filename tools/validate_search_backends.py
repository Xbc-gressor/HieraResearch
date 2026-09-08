#!/usr/bin/env python3
"""Network-free regression checks for the background retrieval layer.

Everything here runs offline: the frozen-corpus backend, in-process fakes, a
loopback-only HTTP server, and a fake ``deepxiv`` executable.  Covered:

* append-only rounds — a second ``search`` leaves the first round's queries
  and the global visits untouched, and round/query ids stay monotonic;
* result-card field surfacing — authors/date/citation_count/tldr and friends
  flow from backend rows through merge, the reducer, and the rendered cards;
* the empty three-state — a zero-hit query records ``empty``, never enters
  ``backend_failures``, and still validates;
* error-page rejection — an error page or an empty page fetched over HTTP is
  recorded as a failed visit, never a successful one.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from search_backends import (
    VISIT_CONTENT_STORE_CHARS,
    FrozenCorpusBackend,
    SearchBackend,
    _render_result_card,
    _validate_query_plan,
    canonical_key,
    dispatch_search,
    merge_candidates,
    merged_results,
    new_manifest,
    validate_manifest,
)

TOOL = Path(__file__).with_name("search_backends.py")


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


def run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )


def write_frozen_corpus(path: Path) -> None:
    path.write_text(
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
                        "authors": ["Ada Lovelace", "Alan Turing"],
                        "date": "2024-05-01",
                        "citation_count": 17,
                        "tldr": "Penalized tree ensembles for small tabular data.",
                        "venue": "ToyConf",
                        "categories": ["cs.LG"],
                        "github_url": "https://github.com/toy/trees",
                        "score": 0.91,
                    },
                    {
                        "url": "https://paper.test/schedules",
                        "title": "Learning rate schedules",
                        "abstract": "Cyclic learning rate schedules for convex problems.",
                        "text": "Full retained method and results for learning rate schedules.",
                    },
                ],
            }
        )
    )


def check_dispatch_merge_and_cards() -> None:
    """In-process: three-state calls, dedup merge, card field surfacing."""
    queries = [
        {"id": "q-01", "text": "regularized trees"},
        {"id": "q-02", "text": "tree baseline failures"},
    ]
    deepxiv = FakeBackend(
        "deepxiv",
        {
            "regularized trees": [
                {
                    "url": "https://arxiv.org/abs/2203.11171v2",
                    "title": "Shared paper",
                    "snippet": "short",
                    "authors": ["Grace Hopper"],
                    "date": "2022-03-21",
                    "citation_count": 42,
                    "tldr": "A shared paper with a metadata trail.",
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
        "jina-search",
        {
            "regularized trees": [
                {
                    "url": "https://www.alphaxiv.org/pdf/2203.11171",
                    "title": "Shared paper mirror",
                    "snippet": "mirror",
                }
            ]
            # "tree baseline failures" absent: a zero-hit call.
        },
    )
    broken = FakeBackend("broken", fail=True)
    raw, failures, calls = asyncio.run(dispatch_search(queries, [deepxiv, jina, broken], 10))

    # Three states: failed backends are failures; zero-hit calls are empty.
    assert len(failures) == 2 and all(f["backend"] == "broken" for f in failures)
    by_pair = {(c["query_id"], c["backend"]): c for c in calls}
    assert by_pair[("q-01", "broken")]["status"] == "failed"
    assert by_pair[("q-02", "jina-search")]["status"] == "empty"
    assert by_pair[("q-01", "deepxiv")]["status"] == "success"
    assert all(c["backend"] != "broken" or "raw_response" not in c for c in calls)

    merged = merge_candidates(raw)
    shared = next(item for item in merged if item["canonical_key"] == "arxiv:2203.11171")
    assert shared["query_support"] == 2
    assert shared["backend_support"] == 2
    assert shared["snippet"] == "longer shared snippet"
    # Result-card metadata survives the merge.
    assert shared["authors"] == ["Grace Hopper"]
    assert shared["date"] == "2022-03-21"
    assert shared["citation_count"] == 42
    assert shared["tldr"] == "A shared paper with a metadata trail."

    card = _render_result_card(shared)
    assert "authors: Grace Hopper" in card
    assert "date: 2022-03-21" in card
    assert "citations: 42" in card
    assert "tldr: A shared paper with a metadata trail." in card


def check_append_only_rounds(tmp: Path) -> None:
    """CLI over the frozen corpus: rounds append, nothing is rewritten."""
    tmp.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp / "frozen.json"
    write_frozen_corpus(corpus_path)
    manifest_path = tmp / "retrieval.json"

    first = run_cli(
        "search",
        "--manifest", str(manifest_path),
        "--frozen-corpus", str(corpus_path),
        "--query-spec", json.dumps(
            {
                "text": "regularized trees",
                "target_dimension_ids": ["dim-method-choice"],
                "evidence_roles": ["hypothesis"],
            }
        ),
    )
    assert first.returncode == 0, first.stderr or first.stdout
    # The CLI surfaces the result-card fields, not just the bare title.
    assert "authors: Ada Lovelace, Alan Turing" in first.stdout
    assert "date: 2024-05-01" in first.stdout
    assert "citations: 17" in first.stdout
    assert "tldr: Penalized tree ensembles" in first.stdout

    visit = run_cli(
        "visit",
        "--manifest", str(manifest_path),
        "--frozen-corpus", str(corpus_path),
        "--url", "https://paper.test/trees",
    )
    assert visit.returncode == 0, visit.stderr or visit.stdout

    before = json.loads(manifest_path.read_text())
    assert [r["round_id"] for r in before["rounds"]] == ["r-01"]
    assert [q["id"] for q in before["rounds"][0]["queries"]] == ["q-01"]
    assert len(before["visits"]) == 1

    second = run_cli(
        "search",
        "--manifest", str(manifest_path),
        "--frozen-corpus", str(corpus_path),
        "--query-spec", json.dumps(
            {
                "text": "learning rate schedules",
                "target_dimension_ids": ["dim-schedule"],
                "evidence_roles": ["hypothesis"],
            }
        ),
    )
    assert second.returncode == 0, second.stderr or second.stdout

    after = json.loads(manifest_path.read_text())
    # Append-only: round one and the global visit survive the second search.
    assert [r["round_id"] for r in after["rounds"]] == ["r-01", "r-02"]
    assert after["rounds"][0] == before["rounds"][0]
    assert after["visits"] == before["visits"]
    # Query ids are monotone across rounds.
    assert [q["id"] for q in after["rounds"][1]["queries"]] == ["q-02"]

    visit_entry = after["visits"][0]
    assert visit_entry["status"] == "success" and visit_entry["view"] == "full_text"
    assert "content" not in visit_entry  # content is externalized
    retained = tmp / visit_entry["content_file"]
    assert retained.is_file()
    assert len(retained.read_text()) == visit_entry["content_chars"]

    # The reducer carries per-round provenance and the card metadata through.
    view = {item["canonical_key"]: item for item in merged_results(after)}
    trees = view["paper.test/trees"]
    assert trees["rounds"] == ["r-01"]
    assert trees["authors"] == ["Ada Lovelace", "Alan Turing"]
    assert trees["date"] == "2024-05-01"
    assert trees["tldr"] == "Penalized tree ensembles for small tabular data."
    assert view["paper.test/schedules"]["rounds"] == ["r-02"]

    # Historical rounds stay browsable; the manifest validates on disk.
    history = run_cli("results", "--manifest", str(manifest_path), "--round", "r-01")
    assert history.returncode == 0, history.stderr
    assert "Regularized trees" in history.stdout
    assert "Learning rate schedules" not in history.stdout
    validated = run_cli("validate", "--manifest", str(manifest_path))
    assert validated.returncode == 0, validated.stdout


def check_empty_three_state(tmp: Path) -> None:
    """A zero-hit query records empty, is no failure, and still validates."""
    tmp.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp / "frozen.json"
    write_frozen_corpus(corpus_path)
    manifest_path = tmp / "empty.json"

    run = run_cli(
        "search",
        "--manifest", str(manifest_path),
        "--frozen-corpus", str(corpus_path),
        "--query-spec", json.dumps(
            {"text": "zzz absent from the corpus", "evidence_roles": ["baseline"]}
        ),
    )
    assert run.returncode == 1  # zero hits: the round is a diagnostic failure
    assert "empty: q-01" in run.stdout

    manifest = json.loads(manifest_path.read_text())
    round_ = manifest["rounds"][0]
    assert [call["status"] for call in round_["backend_calls"]] == ["empty"]
    assert round_["backend_failures"] == []
    assert round_["results"] == []
    assert validate_manifest(manifest, manifest_dir=tmp) == []

    status = run_cli("status", "--manifest", str(manifest_path))
    assert status.returncode == 0, status.stderr
    assert 'q-01 [empty]' in status.stdout


def check_frozen_isolation(tmp: Path) -> None:
    """The frozen condition is exclusive; backend choice is always explicit."""
    tmp.mkdir(parents=True, exist_ok=True)
    corpus_path = tmp / "frozen.json"
    write_frozen_corpus(corpus_path)
    spec = json.dumps({"text": "regularized trees", "evidence_roles": ["hypothesis"]})

    implicit = run_cli(
        "search", "--manifest", str(tmp / "must-not-run.json"), "--query-spec", spec
    )
    assert implicit.returncode == 1
    assert "select --backend explicitly" in implicit.stderr
    assert not (tmp / "must-not-run.json").exists()

    mixed = run_cli(
        "search",
        "--manifest", str(tmp / "mixed.json"),
        "--frozen-corpus", str(corpus_path),
        "--backend", "deepxiv",
        "--query-spec", spec,
    )
    assert mixed.returncode == 1
    assert "permits only the frozen backend" in mixed.stderr


def check_error_pages_rejected(tmp: Path) -> None:
    """Loopback HTTP: error pages and empty pages never record success."""
    tmp.mkdir(parents=True, exist_ok=True)
    error_page = (
        "<html><body><h1>Just a moment...</h1><p>Please enable JavaScript "
        "and cookies to continue while the anti-bot challenge completes. "
        "This interstitial retains no readable page content for a reader.</p>"
        "</body></html>"
    )
    ok_page = (
        "A retained primary source describing the method, the ablations, and "
        "the failure regimes of the fixture mechanism in enough prose to "
        "clear the minimum content length for a successful visit receipt. "
        "Additional sentences keep the readable body well above that bar."
    )
    pages = {"/error": error_page, "/empty": "", "/ok": ok_page}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = pages.get(self.path)
            if body is None:
                self.send_error(404)
                return
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        manifest_path = tmp / "web.json"
        manifest_path.write_text(json.dumps(new_manifest()))

        for path, needle in (
            ("/error", "error-page marker"),
            ("/empty", "empty content"),
        ):
            run = run_cli(
                "visit",
                "--manifest", str(manifest_path),
                "--url", base + path,
                "--visit-backend", "direct",
            )
            assert run.returncode == 1, (path, run.stdout)
            assert needle in run.stderr, run.stderr

        control = run_cli(
            "visit",
            "--manifest", str(manifest_path),
            "--url", base + "/ok",
            "--visit-backend", "direct",
        )
        assert control.returncode == 0, control.stderr
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    manifest = json.loads(manifest_path.read_text())
    statuses = [visit["status"] for visit in manifest["visits"]]
    assert statuses == ["failed", "failed", "success"], statuses
    assert validate_manifest(manifest, manifest_dir=tmp) == []


FAKE_DEEPXIV = """#!/usr/bin/env python3
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


def check_deepxiv_progressive(tmp: Path) -> None:
    """Fake deepxiv CLI: progressive read, preview fallback, head-only failure."""
    tmp.mkdir(parents=True, exist_ok=True)
    fake_deepxiv = tmp / "deepxiv"
    fake_deepxiv.write_text(FAKE_DEEPXIV)
    fake_deepxiv.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(tmp) + os.pathsep + env.get("PATH", "")

    def visit(manifest_path: Path, paper: str) -> subprocess.CompletedProcess:
        return run_cli(
            "visit",
            "--manifest", str(manifest_path),
            "--url", f"https://arxiv.org/abs/{paper}",
            "--view", "auto",
            env=env,
        )

    progressive_manifest_path = tmp / "progressive.json"
    progressive_run = visit(progressive_manifest_path, "2409.05591")
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
    assert all("content" not in visit for visit in progressive_manifest["visits"])
    assert sum(visit["content_chars"] for visit in successful_visits) <= (
        VISIT_CONTENT_STORE_CHARS
    )
    assert validate_manifest(progressive_manifest, manifest_dir=tmp) == []

    preview_manifest_path = tmp / "preview-fallback.json"
    preview_run = visit(preview_manifest_path, "2409.05592")
    assert preview_run.returncode == 0, preview_run.stderr or preview_run.stdout
    preview_manifest = json.loads(preview_manifest_path.read_text())
    assert [visit["view"] for visit in preview_manifest["visits"]] == [
        "head",
        "preview",
    ]
    assert validate_manifest(preview_manifest, manifest_dir=tmp) == []

    head_only_manifest_path = tmp / "head-only-failure.json"
    head_only_run = visit(head_only_manifest_path, "2409.05593")
    assert head_only_run.returncode == 1
    assert "head metadata but no substantive" in head_only_run.stderr
    head_only_manifest = json.loads(head_only_manifest_path.read_text())
    assert [visit["view"] for visit in head_only_manifest["visits"]] == [
        "head",
        "preview",
    ]
    assert head_only_manifest["visits"][-1]["status"] == "failed"
    assert validate_manifest(head_only_manifest, manifest_dir=tmp) == []


def check_query_plan() -> None:
    """Query shape: targets are optional intent records; roles stay closed."""
    errors, _ = _validate_query_plan(
        [{"id": "q-01", "text": "exploratory", "evidence_roles": ["hypothesis"]}]
    )
    assert errors == [], errors

    errors, _ = _validate_query_plan(
        [
            {
                "id": "q-01",
                "text": "planned",
                "target_dimension_ids": ["dim-method-choice"],
                "evidence_roles": ["baseline", "failure_mode", "counterevidence"],
            }
        ]
    )
    assert errors == [], errors

    for bad, needle in (
        (
            {"id": "q-01", "text": "t", "lane": "grounding", "evidence_roles": ["baseline"]},
            "unknown fields",
        ),
        (
            {"id": "q-01", "text": "t", "evidence_roles": ["inner_hpo_prior"]},
            "evidence_roles must be",
        ),
        (
            {"id": "q-01", "text": "t", "evidence_roles": []},
            "evidence_roles must be",
        ),
        (
            {"id": "q-01", "text": "t", "target_dimension_ids": ["not-a-dim"]},
            "target_dimension_ids must be",
        ),
    ):
        errors, _ = _validate_query_plan([bad])
        assert any(needle in error for error in errors), (bad, errors)


def main() -> int:
    assert canonical_key("https://arxiv.org/abs/2203.11171v4") == "arxiv:2203.11171"
    assert canonical_key("https://www.alphaxiv.org/pdf/2203.11171") == "arxiv:2203.11171"
    assert canonical_key("https://Example.com/a/?utm_source=x") == "example.com/a"

    check_dispatch_merge_and_cards()
    check_query_plan()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        corpus_path = tmp_path / "corpus.json"
        write_frozen_corpus(corpus_path)
        frozen = FrozenCorpusBackend(corpus_path)
        response = asyncio.run(frozen.search("regularized trees", 5))
        assert response["items"][0]["url"] == "https://paper.test/trees"
        assert response["items"][0]["tldr"].startswith("Penalized tree")
        assert "Full retained" in frozen.read("https://paper.test/trees")
        assert response["metadata"]["corpus_id"] == "fixture-v1"

        check_append_only_rounds(tmp_path / "rounds")
        check_empty_three_state(tmp_path / "empty")
        check_frozen_isolation(tmp_path / "isolation")
        check_error_pages_rejected(tmp_path / "web")
        check_deepxiv_progressive(tmp_path / "deepxiv")

    print("Retrieval rounds, result-card, three-state, and visit-integrity checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
