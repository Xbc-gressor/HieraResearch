"""Competition-policy compliance gates: query/result/body classifiers, audit
store, manifest shape, and exit-code accounting.  Everything is network-free:
backends and fetches are faked or monkeypatched in-process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import competition_policy  # noqa: E402
import search_backends  # noqa: E402


MLSP = ("mle-mlsp-birds", "mlsp-2013-birds")
STATOIL = ("mle-statoil-iceberg", "statoil-iceberg-classifier-challenge")
SPOOKY = ("mle-spooky", "spooky-author-identification")
DENOISING = ("mle-denoising", "denoising-dirty-documents")

BLOCKED_QUERIES = [
    (*MLSP, "MLSP 2013 bird classification challenge winning solutions"),
    (*STATOIL, "Statoil iceberg classifier Kaggle competition best solutions"),
    (*SPOOKY, "Spooky Author Identification winning solutions"),
]
ALLOWED_QUERIES = [
    (*SPOOKY, "author identification best methods"),
    (*DENOISING, "denoising documents code"),
]


def _profile(task: str, competition_id: str) -> dict:
    profile = competition_policy.derive_profile(task, competition_id)
    assert profile is not None
    return profile


def _run_dir(
    tmp_path: Path, task: str, competition_id: str, *, write_profile: bool = True
) -> tuple[Path, Path]:
    """A runs/<task>/<tag> tree with (optionally) the identity profile."""
    run_dir = tmp_path / "runs" / task / "t1"
    run_dir.mkdir(parents=True)
    if write_profile:
        (run_dir / "task_identity_profile.json").write_text(
            json.dumps(_profile(task, competition_id))
        )
    return run_dir, run_dir / "background_retrieval.json"


def _search_args(manifest: Path, texts: list[str]) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        query_spec=[
            json.dumps({"text": text, "evidence_roles": ["hypothesis"]})
            for text in texts
        ],
        backend=["fake"],
        frozen_corpus=None,
        max_results=5,
    )


def _visit_args(manifest: Path, url: str, *, visit_backend: str = "jina-read",
                view: str = "auto") -> argparse.Namespace:
    return argparse.Namespace(
        manifest=manifest,
        url=url,
        view=view,
        section=None,
        frozen_corpus=None,
        visit_backend=visit_backend,
    )


class _FakeBackend(search_backends.SearchBackend):
    """In-memory backend with a jina-shaped raw payload."""

    name = "fake"
    version = "fake-1"

    def __init__(self, rows: dict[str, list[dict]] | None = None, fail: bool = False):
        self.rows = rows or {}
        self.fail = fail
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int) -> dict:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("backend unavailable")
        items = [dict(row) for row in self.rows.get(query, [])[:max_results]]
        return {"items": items, "raw_response": {"data": items}, "metadata": {}}


def _install_backend(monkeypatch: pytest.MonkeyPatch, backend: _FakeBackend) -> None:
    monkeypatch.setattr(
        search_backends, "build_backends", lambda names, frozen_corpus: ([backend], [])
    )


def _forbid_backend_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(names, frozen_corpus):
        raise AssertionError("backend construction must be skipped")

    monkeypatch.setattr(search_backends, "build_backends", boom)


def _forbid_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):
        raise AssertionError("blocked visits must never fetch")

    for name in ("_web_read", "_direct_visit", "_deepxiv_read",
                 "_deepxiv_progressive_read"):
        monkeypatch.setattr(search_backends, name, boom)


def _results_cmd_args(manifest: Path) -> argparse.Namespace:
    return argparse.Namespace(manifest=manifest, round=None)


# --------------------------------------------------------------------------
# policy module: profile derivation and the identity predicate
# --------------------------------------------------------------------------


def test_derive_profile_is_mechanical() -> None:
    profile = _profile(*MLSP)
    assert profile["profile_version"] == 1
    assert profile["competition_policy_version"] == (
        competition_policy.COMPETITION_POLICY_VERSION
    )
    assert profile["competition_id"] == "mlsp-2013-birds"
    assert profile["task_name"] == "mle-mlsp-birds"
    assert profile["slug_tokens"] == ["mlsp", "2013", "birds"]
    assert profile["identity_tokens"] == ["mlsp", "2013", "birds"]
    assert profile["discriminative_tokens"] == ["mlsp", "birds"]  # no pure years
    assert profile["proper_tokens"] == ["mlsp"]
    assert profile["kaggle_competition_path"] == "kaggle.com/competitions/mlsp-2013-birds"
    assert profile["generated_at"]
    assert competition_policy.derive_profile("mle-x", None) is None

    spooky = _profile(*SPOOKY)
    assert spooky["identity_tokens"] == ["spooky"]  # author/identification are generic
    assert spooky["proper_tokens"] == []  # spooky is a plain English word
    statoil = _profile(*STATOIL)
    assert statoil["proper_tokens"] == ["statoil"]


def test_identity_hit_rules() -> None:
    mlsp, spooky = _profile(*MLSP), _profile(*SPOOKY)
    # slug token sequence with a stopword inserted inside the window
    hit, rule = competition_policy.identity_hit(
        "notes on the Spooky Author Identification task", spooky
    )
    assert hit and rule == "slug_sequence"
    hit, rule = competition_policy.identity_hit("mlsp feature baselines", mlsp)
    assert hit and rule == "proper_token"
    # proper tokens match whole tokens only
    assert competition_policy.identity_hit("mlspx is unrelated", mlsp) == (False, "")
    # >=2 discriminative tokens co-occur; a pure year never counts
    quest = _profile("mle-google-quest", "google-quest-challenge")
    assert quest["proper_tokens"] == []
    hit, rule = competition_policy.identity_hit("google quest data", quest)
    assert hit and rule == "discriminative_tokens"
    assert competition_policy.identity_hit("2013 birds audio", mlsp) == (False, "")
    assert competition_policy.identity_hit("anything", None) == (False, "")


def test_kaggle_competition_url() -> None:
    mlsp = _profile(*MLSP)
    assert competition_policy.is_kaggle_competition_url(
        "https://www.kaggle.com/competitions/mlsp-2013-birds", mlsp)
    assert competition_policy.is_kaggle_competition_url(
        "https://kaggle.com/c/mlsp-2013-birds/discussion/123", mlsp)
    assert not competition_policy.is_kaggle_competition_url(
        "https://www.kaggle.com/competitions/other-competition", mlsp)
    assert not competition_policy.is_kaggle_competition_url(
        "https://example.com/kaggle.com/competitions/mlsp-2013-birds", mlsp)
    assert not competition_policy.is_kaggle_competition_url(
        "https://www.kaggle.com/competitions/mlsp-2013-birds", None)


# --------------------------------------------------------------------------
# query gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task,cid,query", BLOCKED_QUERIES)
def test_blocked_query_never_dispatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, task, cid, query
) -> None:
    _run_dir(tmp_path, task, cid)
    manifest_path = tmp_path / "runs" / task / "t1" / "background_retrieval.json"
    _forbid_backend_construction(monkeypatch)
    rc = search_backends.cmd_search(_search_args(manifest_path, [query]))
    assert rc == competition_policy.POLICY_EXIT_CODE
    out, err = capsys.readouterr()
    assert "query blocked by competition policy" in out
    assert "query blocked by competition policy" in err

    manifest = json.loads(manifest_path.read_text())
    assert manifest["competition_id"] == cid
    assert manifest["competition_policy_version"] == (
        competition_policy.COMPETITION_POLICY_VERSION
    )
    round_ = manifest["rounds"][0]
    query_entry = round_["queries"][0]
    assert query_entry["policy_status"] == "blocked"
    assert query_entry["policy_categories"]
    (call,) = round_["backend_calls"]
    assert call["backend"] == "policy-gate"
    assert call["backend_version"] == str(competition_policy.COMPETITION_POLICY_VERSION)
    assert call["status"] == "blocked" and "raw_response" not in call
    assert round_["results"] == [] and round_["backend_failures"] == []
    assert search_backends.validate_manifest(manifest, manifest_dir=manifest_path.parent) == []


@pytest.mark.parametrize("task,cid,query", ALLOWED_QUERIES)
def test_generic_methodology_query_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task, cid, query
) -> None:
    _, manifest_path = _run_dir(tmp_path, task, cid)
    backend = _FakeBackend({query: [{"url": "https://paper.test/m",
                                     "title": "A generic method", "snippet": "s"}]})
    _install_backend(monkeypatch, backend)
    rc = search_backends.cmd_search(_search_args(manifest_path, [query]))
    assert rc == 0
    assert backend.queries == [query]  # dispatched verbatim
    manifest = json.loads(manifest_path.read_text())
    round_ = manifest["rounds"][0]
    assert round_["queries"][0]["policy_status"] == "allowed"
    assert round_["results"][0]["policy_status"] == "allowed"
    assert round_["blocked_results"] == []


# --------------------------------------------------------------------------
# result gate
# --------------------------------------------------------------------------


def test_kaggle_competition_url_blocked_as_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP)
    query = "bird acoustic feature methods"
    kaggle_row = {
        "url": "https://www.kaggle.com/competitions/mlsp-2013-birds",
        "title": "ZZZ-KAGGLE-PAGE-TITLE",
        "snippet": "ZZZ-KAGGLE-PAGE-SNIPPET",
    }
    generic_row = {"url": "https://paper.test/generic", "title": "Generic", "snippet": "g"}
    backend = _FakeBackend({query: [kaggle_row, generic_row]})
    _install_backend(monkeypatch, backend)
    rc = search_backends.cmd_search(_search_args(manifest_path, [query]))
    assert rc == 0  # one row survives
    out, _ = capsys.readouterr()
    assert "ZZZ-KAGGLE-PAGE-TITLE" not in out and "ZZZ-KAGGLE-PAGE-SNIPPET" not in out

    manifest = json.loads(manifest_path.read_text())
    round_ = manifest["rounds"][0]
    assert [r["url"] for r in round_["results"]] == ["https://paper.test/generic"]
    (blocked,) = round_["blocked_results"]
    assert blocked["url"] == kaggle_row["url"]
    assert blocked["domain"] == "kaggle.com"
    assert blocked["basis"] == "direct"
    assert "kaggle_competition_url" in blocked["rule_categories"]
    assert blocked["query_ids"] == ["q-01"]
    assert blocked["pointer"].startswith("pa-")
    # the blocked row is scrubbed from the retained raw payload
    (call,) = round_["backend_calls"]
    assert [r["url"] for r in call["raw_response"]["data"]] == [generic_row["url"]]
    blob = json.dumps(manifest)
    assert "ZZZ-KAGGLE-PAGE-TITLE" not in blob and "ZZZ-KAGGLE-PAGE-SNIPPET" not in blob
    # the content survives only in the audit store, behind the opaque pointer
    audit = manifest_path.parent / ".policy_audit" / "blocked-items.jsonl"
    lines = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [line["id"] for line in lines] == [blocked["pointer"]]
    assert lines[0]["title"] == "ZZZ-KAGGLE-PAGE-TITLE"
    assert search_backends.validate_manifest(manifest, manifest_dir=manifest_path.parent) == []


def test_solution_shaped_rows_blocked_and_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP)
    query = "audio classification feature methods"
    rows = [
        {  # competition-summary PDF
            "url": "https://example.com/mlsp-summary.pdf",
            "title": "ZZZ-PDF: MLSP 2013 birds competition summary",
            "snippet": "ZZZ-PDF-SNIPPET how top teams approached it",
        },
        {  # participant paper
            "url": "https://example.com/participant-paper",
            "title": "ZZZ-PAPER: our approach to the MLSP challenge",
            "snippet": "ZZZ-PAPER-SNIPPET",
        },
        {  # solution repo
            "url": "https://github.com/someone/mlsp-2013-birds",
            "title": "ZZZ-REPO: mlsp-2013-birds winning solution",
            "snippet": "ZZZ-REPO-SNIPPET source code",
        },
        {  # methodology control: identity hit, no solution signal
            "url": "https://arxiv.org/abs/1234.5678",
            "title": "A benchmark study of audio classification",
            "snippet": "We evaluate on the MLSP 2013 birds dataset and report AUC.",
        },
    ]
    backend = _FakeBackend({query: rows})
    _install_backend(monkeypatch, backend)
    rc = search_backends.cmd_search(_search_args(manifest_path, [query]))
    assert rc == 0
    out, _ = capsys.readouterr()
    for marker in ("ZZZ-PDF", "ZZZ-PAPER", "ZZZ-REPO"):
        assert marker not in out
    manifest = json.loads(manifest_path.read_text())
    blob = json.dumps(manifest)
    for marker in ("ZZZ-PDF", "ZZZ-PAPER", "ZZZ-REPO"):
        assert marker not in blob
    round_ = manifest["rounds"][0]
    assert [r["url"] for r in round_["results"]] == ["https://arxiv.org/abs/1234.5678"]
    assert len(round_["blocked_results"]) == 3
    audit_lines = (
        manifest_path.parent / ".policy_audit" / "blocked-items.jsonl"
    ).read_text().splitlines()
    assert len(audit_lines) == 3
    # replay passes the same filter and shows the blocked count
    capsys.readouterr()
    rc = search_backends.cmd_results(_results_cmd_args(manifest_path))
    assert rc == 0
    out, _ = capsys.readouterr()
    assert "blocked: 3" in out
    for marker in ("ZZZ-PDF", "ZZZ-PAPER", "ZZZ-REPO"):
        assert marker not in out
    assert search_backends.validate_manifest(manifest, manifest_dir=manifest_path.parent) == []


def test_generic_leaderboard_words_without_identity_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP)
    query = "how kaggle leaderboards work"
    row = {
        "url": "https://blog.example.com/kaggle-tips",
        "title": "How Kaggle leaderboards and kernels work",
        "snippet": "leaderboard kernel notebook score",
    }
    _install_backend(monkeypatch, _FakeBackend({query: [row]}))
    rc = search_backends.cmd_search(_search_args(manifest_path, [query]))
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rounds"][0]["blocked_results"] == []
    assert not (manifest_path.parent / ".policy_audit").exists()


def test_all_results_blocked_round_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP)
    query = "bird acoustic feature methods"
    row = {
        "url": "https://github.com/someone/mlsp-2013-birds",
        "title": "mlsp-2013-birds winning solution",
        "snippet": "source code",
    }
    _install_backend(monkeypatch, _FakeBackend({query: [row]}))
    rc = search_backends.cmd_search(_search_args(manifest_path, [query]))
    assert rc == competition_policy.POLICY_EXIT_CODE


# --------------------------------------------------------------------------
# visit gate
# --------------------------------------------------------------------------


def test_kaggle_competition_url_blocked_as_direct_visit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    run_dir, manifest_path = _run_dir(tmp_path, *STATOIL)
    _forbid_fetch(monkeypatch)
    args = _visit_args(
        manifest_path,
        "https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge",
    )
    rc = search_backends.cmd_visit(args)
    assert rc == competition_policy.POLICY_EXIT_CODE
    out, err = capsys.readouterr()
    assert out == "" and "visit blocked by competition policy" in err
    assert not (run_dir / "retrieval").exists()
    manifest = json.loads(manifest_path.read_text())
    (visit,) = manifest["visits"]
    assert visit["status"] == "blocked" and visit["policy_status"] == "blocked"
    assert "kaggle_competition_url" in visit["policy_categories"]
    assert "content_file" not in visit
    assert visit["pointer"].startswith("pa-")
    assert search_backends.validate_manifest(manifest, manifest_dir=run_dir) == []


def test_blocked_visit_body_scan_writes_no_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    run_dir, manifest_path = _run_dir(tmp_path, *MLSP)
    body = (
        "An informal post about audio pipelines. " * 10
        + " The winning solution to the MLSP 2013 birds challenge reached rank 1 "
        "with 0.954 private."
    )
    monkeypatch.setattr(
        search_backends, "_web_read", lambda url: (body, "jina-read", None)
    )
    rc = search_backends.cmd_visit(
        _visit_args(manifest_path, "https://blog.example.com/solution-post")
    )
    assert rc == competition_policy.POLICY_EXIT_CODE
    out, err = capsys.readouterr()
    assert "winning solution" not in out and "rank 1" not in out
    assert "visit blocked by competition policy" in err
    assert not (run_dir / "retrieval").exists()
    manifest = json.loads(manifest_path.read_text())
    (visit,) = manifest["visits"]
    assert visit["status"] == "blocked" and "content_file" not in visit
    # the audit line keeps only a bounded context snippet, never the full body
    audit_lines = (
        run_dir / ".policy_audit" / "blocked-items.jsonl"
    ).read_text().splitlines()
    (line,) = [json.loads(item) for item in audit_lines]
    assert len(line["context"]) <= 200
    assert body not in json.dumps(line)


def test_methodology_visit_allowed_and_receipt_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest_path = _run_dir(tmp_path, *MLSP)
    body = (
        "A benchmark study of audio classification methods. We evaluate on the "
        "MLSP 2013 birds dataset alongside other corpora and report AUC with "
        "ablations over feature extractors and pooling strategies. " * 3
    )
    monkeypatch.setattr(
        search_backends, "_web_read", lambda url: (body, "jina-read", None)
    )
    rc = search_backends.cmd_visit(
        _visit_args(manifest_path, "https://arxiv.org/abs/1234.5678")
    )
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    (visit,) = manifest["visits"]
    assert visit["status"] == "success" and visit["policy_status"] == "allowed"
    assert visit["content_file"] and visit["content_chars"] > 0
    assert (run_dir / visit["content_file"]).is_file()
    assert not (run_dir / ".policy_audit").exists()
    assert search_backends.validate_manifest(manifest, manifest_dir=run_dir) == []


def test_visit_inherited_identity_blocks_without_local_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP)
    # The originating query identity-hits but has no intent, so both the query
    # and its innocuous row are allowed; the body then trips the phrase scan.
    query = "MLSP 2013 birds acoustic feature extraction"
    row = {
        "url": "https://example.com/methods-post",
        "title": "Some notes on audio features",
        "snippet": "Feature engineering notes.",
    }
    _install_backend(monkeypatch, _FakeBackend({query: [row]}))
    assert search_backends.cmd_search(_search_args(manifest_path, [query])) == 0
    body = "A long technical post. " * 30 + " The winning approach was late-fusion stacking."
    monkeypatch.setattr(
        search_backends, "_web_read", lambda url: (body, "jina-read", None)
    )
    rc = search_backends.cmd_visit(_visit_args(manifest_path, row["url"]))
    assert rc == competition_policy.POLICY_EXIT_CODE
    manifest = json.loads(manifest_path.read_text())
    (visit,) = manifest["visits"]
    assert visit["status"] == "blocked"
    assert "inherited_identity" in visit["policy_categories"]


def test_visit_phrase_without_identity_or_binding_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest_path = _run_dir(tmp_path, *MLSP)
    body = (
        "An essay about Kaggle in general: the winning approach to any "
        "competition is careful cross-validation, and first-place finishes "
        "usually involve ensembling. " * 5
    )
    monkeypatch.setattr(
        search_backends, "_web_read", lambda url: (body, "jina-read", None)
    )
    rc = search_backends.cmd_visit(
        _visit_args(manifest_path, "https://blog.example.com/kaggle-craft")
    )
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["visits"][0]["status"] == "success"
    assert not (run_dir / ".policy_audit").exists()


def test_blocked_deepxiv_progressive_visit_persists_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, manifest_path = _run_dir(tmp_path, *MLSP)
    attempts = [
        {"backend": "deepxiv", "backend_version": "fake-1", "view": "head",
         "section": None, "status": "success", "content": '{"sections": []}',
         "error": None},
        {"backend": "deepxiv", "backend_version": "fake-1", "view": "preview",
         "section": None, "status": "success",
         "content": "The MLSP 2013 birds winning solution used rank 1 stacking.",
         "error": None},
    ]
    monkeypatch.setattr(
        search_backends, "_deepxiv_progressive_read", lambda url, manifest: attempts
    )
    rc = search_backends.cmd_visit(
        _visit_args(manifest_path, "https://arxiv.org/abs/2409.00001",
                    visit_backend="auto")
    )
    assert rc == competition_policy.POLICY_EXIT_CODE
    assert not (run_dir / "retrieval").exists()
    manifest = json.loads(manifest_path.read_text())
    (visit,) = manifest["visits"]
    assert visit["status"] == "blocked" and "content_file" not in visit


# --------------------------------------------------------------------------
# status accounting and replay
# --------------------------------------------------------------------------


def test_status_treats_blocked_as_independent_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP)
    blocked = "MLSP 2013 bird classification challenge winning solutions"
    empty = "zzz absent everywhere"
    ok = "generic tree method"
    backend = _FakeBackend({ok: [{"url": "https://paper.test/t", "title": "T",
                                  "snippet": "t"}]})
    _install_backend(monkeypatch, backend)
    rc = search_backends.cmd_search(_search_args(manifest_path, [blocked, empty, ok]))
    assert rc == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["rounds"][0]["backend_failures"] == []
    capsys.readouterr()
    rc = search_backends.cmd_status(argparse.Namespace(manifest=manifest_path))
    assert rc == 0
    out, _ = capsys.readouterr()
    assert "q-01 [blocked]" in out
    assert "q-02 [empty]" in out
    assert "q-03" not in out.split("queries with no results:")[1]
    assert "policy blocks: queries: 1 | results: 0 | visits: 0" in out


def test_exit_codes_distinguish_blocked_empty_and_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP)
    # all-empty round (no blocks) keeps today's exit 1
    _install_backend(monkeypatch, _FakeBackend({}))
    rc = search_backends.cmd_search(_search_args(manifest_path, ["zzz nothing"]))
    assert rc == 1
    outcomes = search_backends._round_query_outcomes(
        json.loads(manifest_path.read_text())["rounds"][0]
    )
    assert outcomes[0]["outcome"] == "empty"

    _, fail_manifest = _run_dir(tmp_path / "f", *MLSP)
    _install_backend(monkeypatch, _FakeBackend(fail=True))
    rc = search_backends.cmd_search(_search_args(fail_manifest, ["generic method"]))
    assert rc == 1  # backend failure, distinct from a policy block

    _, blocked_manifest = _run_dir(tmp_path / "b", *MLSP)
    _forbid_backend_construction(monkeypatch)
    rc = search_backends.cmd_search(
        _search_args(blocked_manifest, [BLOCKED_QUERIES[0][2]])
    )
    assert rc == competition_policy.POLICY_EXIT_CODE


# --------------------------------------------------------------------------
# artifact scan (product side)
# --------------------------------------------------------------------------


def test_scan_artifact_text() -> None:
    mlsp = _profile(*MLSP)
    text = (
        "Approach notes for the MLSP 2013 birds task: the winning solution "
        "reached first-place with 0.954 private leaderboard."
    )
    hits = competition_policy.scan_artifact_text(text, mlsp)
    assert hits
    assert all(len(hit["context"]) <= 160 for hit in hits)
    assert all(hit["categories"] for hit in hits)

    generic_leaderboard = (
        "We evaluate on the MLSP 2013 birds dataset. The leaderboard mechanism "
        "on Kaggle refreshes periodically and rewards ensembling."
    )
    assert competition_policy.scan_artifact_text(generic_leaderboard, mlsp) == []
    ranked_leaderboard = (
        "For the MLSP 2013 birds task the final leaderboard rank 3 used X."
    )
    assert competition_policy.scan_artifact_text(ranked_leaderboard, mlsp)
    assert competition_policy.scan_artifact_text("anything", None) == []


# --------------------------------------------------------------------------
# profile resolution and fail-closed
# --------------------------------------------------------------------------


def test_fail_closed_when_profile_missing_for_known_competition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _, manifest_path = _run_dir(tmp_path, *MLSP, write_profile=False)
    _forbid_backend_construction(monkeypatch)
    _forbid_fetch(monkeypatch)
    rc = search_backends.cmd_search(_search_args(manifest_path, ["generic query"]))
    assert rc == competition_policy.POLICY_EXIT_CODE
    rc = search_backends.cmd_visit(_visit_args(manifest_path, "https://x.example"))
    assert rc == competition_policy.POLICY_EXIT_CODE
    _, err = capsys.readouterr()
    assert "competition policy profile required but missing/unparseable" in err
    assert not manifest_path.exists()  # nothing dispatched, nothing saved

    # run_metadata carrying a bare competition_id fails closed the same way
    (manifest_path.parent / "run_metadata.json").write_text(
        json.dumps({"competition_id": MLSP[1]})
    )
    rc = search_backends.cmd_search(_search_args(manifest_path, ["generic query"]))
    assert rc == competition_policy.POLICY_EXIT_CODE

    # an embedded profile resolves instead
    (manifest_path.parent / "run_metadata.json").write_text(
        json.dumps({
            "competition_id": MLSP[1],
            "task_identity_profile": _profile(*MLSP),
        })
    )
    resolution = competition_policy.resolve_profile(manifest_path)
    assert resolution.profile is not None and resolution.fail_closed_reason is None


def test_non_mle_and_adhoc_paths_pass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "retrieval.json"  # not a runs/<task>/<tag> tree
    query = "MLSP 2013 bird classification challenge winning solutions"
    backend = _FakeBackend({query: [{"url": "https://paper.test/x", "title": "X",
                                     "snippet": "x"}]})
    _install_backend(monkeypatch, backend)
    rc = search_backends.cmd_search(_search_args(manifest_path, [query]))
    assert rc == 0
    assert backend.queries == [query]
    manifest = json.loads(manifest_path.read_text())
    assert "competition_id" not in manifest
    assert "policy_status" not in manifest["rounds"][0]["queries"][0]
    assert "blocked_results" not in manifest["rounds"][0]
