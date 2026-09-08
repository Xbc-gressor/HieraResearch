#!/usr/bin/env python3
"""Local-first retrieval, optional backends, and auditable visit receipts.

Frozen-corpus search and every artifact-integrity check use only the standard
library. DeepXiv, Jina (jina-search / jina-reader), and direct fetches are
explicit, optional adapters; every receipt in the manifest is produced by this
tool itself.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 4
EVIDENCE_ROLES = {
    "hypothesis",
    "baseline",
    "failure_mode",
    "counterevidence",
    "relation",
}
# Safety cap on stored visit content (~100k tokens); stdout uses bounded views.
VISIT_CONTENT_STORE_CHARS = 400_000
VISIT_HEAD_CHARS = 4000
READ_WINDOW_CHARS = 8000
HIGH_RANK_THRESHOLD = 5
# Minimum chars for a web visit to count as success; shorter means an error page.
VISIT_MIN_CONTENT_CHARS = 200
_WEB_ERROR_MARKERS = (
    "no html available",
    "just a moment",  # anti-bot challenge interstitial
    "enable javascript",
    "verify you are human",
)
RESULT_METADATA_FIELDS = (
    "authors",
    "date",
    "citation_count",
    "tldr",
    "venue",
    "categories",
    "github_url",
    "score",
)
HTTP_TIMEOUT = 45
DEEPXIV_MAX_SECTIONS = 3
SUBSTANTIVE_VIEWS = {"section", "preview", "full_text", "page"}

_ARXIV_RE = re.compile(
    r"(?:arxiv\.org|alphaxiv\.org)/(?:abs|pdf)/([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_RE = re.compile(r"<[^>]+>")
_DIMENSION_RE = re.compile(r"^dim-[a-z0-9][a-z0-9-]*$")
_WORD_RE = re.compile(r"[a-z0-9]+")
_SECTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def arxiv_id(url: str) -> str | None:
    match = _ARXIV_RE.search(str(url or ""))
    return _VERSION_RE.sub("", match.group(1)) if match else None


def canonical_key(url: str) -> str:
    """Return a stable work/page identity for deduplication and visit joins."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    paper_id = arxiv_id(raw)
    if paper_id:
        return f"arxiv:{paper_id.lower()}"
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return raw.lower().rstrip("/")
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{parts.port}" if parts.port else ""
    path = re.sub(r"/+", "/", parts.path or "/").rstrip("/") or "/"
    return f"{host}{port}{path}"


def canonical_url(url: str) -> str:
    paper_id = arxiv_id(url)
    if paper_id:
        return f"https://arxiv.org/abs/{paper_id}"
    raw = str(url or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return raw
    path = re.sub(r"/+", "/", parts.path or "/").rstrip("/") or "/"
    return urllib.parse.urlunsplit(
        ((parts.scheme or "https").lower(), parts.netloc.lower(), path, "", "")
    )


def new_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rounds": [],
        "visits": [],
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return new_manifest()
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: retrieval manifest must be a JSON object")
    return data


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _validate_query_plan(
    queries: Any, *, where: str = "retrieval queries"
) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    query_ids: set[str] = set()
    query_texts: set[str] = set()

    if not isinstance(queries, list):
        return [f"{where} must be a list"], query_ids
    for index, query in enumerate(queries):
        label = f"{where}[{index}]"
        if not isinstance(query, dict):
            errors.append(f"{label} must be an object")
            continue
        unknown = sorted(
            set(query) - {"id", "text", "target_dimension_ids", "evidence_roles"}
        )
        if unknown:
            errors.append(f"{label} has unknown fields {unknown}")
        query_id = query.get("id")
        if not isinstance(query_id, str) or not re.fullmatch(r"q-\d{2,}", query_id):
            errors.append(f"{label}.id must match q-NN")
        elif query_id in query_ids:
            errors.append(f"duplicate retrieval query id {query_id}")
        else:
            query_ids.add(query_id)
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            errors.append(f"{label}.text must be non-empty")
        elif query["text"].strip().casefold() in query_texts:
            errors.append(f"{label}.text duplicates another query")
        else:
            query_texts.add(query["text"].strip().casefold())

        targets = query.get("target_dimension_ids")
        if targets is None:
            targets = []
        elif not (
            isinstance(targets, list)
            and all(
                isinstance(item, str) and _DIMENSION_RE.fullmatch(item) for item in targets
            )
        ):
            errors.append(f"{label}.target_dimension_ids must be a list of dim-* ids")
            targets = []
        elif len(targets) != len(set(targets)):
            errors.append(f"{label}.target_dimension_ids must not contain duplicates")

        roles = query.get("evidence_roles")
        valid_roles = (
            isinstance(roles, list)
            and bool(roles)
            and all(isinstance(role, str) and role in EVIDENCE_ROLES for role in roles)
        )
        if not valid_roles:
            errors.append(
                f"{label}.evidence_roles must be a non-empty list drawn from "
                f"{sorted(EVIDENCE_ROLES)}"
            )
            continue
        if len(roles) != len(set(roles)):
            errors.append(f"{label}.evidence_roles must not contain duplicates")
    return errors, query_ids


def validate_manifest(
    manifest: dict[str, Any], manifest_dir: Path | None = None
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"retrieval manifest schema_version must be {SCHEMA_VERSION}")

    rounds = manifest.get("rounds")
    if not isinstance(rounds, list):
        errors.append("retrieval manifest rounds must be a list")
        rounds = []
    seen_round_ids: set[str] = set()
    all_query_ids: set[str] = set()
    for round_index, round_ in enumerate(rounds):
        rwhere = f"retrieval rounds[{round_index}]"
        if not isinstance(round_, dict):
            errors.append(f"{rwhere} must be an object")
            continue
        round_id = round_.get("round_id")
        if not isinstance(round_id, str) or not re.fullmatch(r"r-\d{2,}", round_id):
            errors.append(f"{rwhere}.round_id must match r-NN")
        elif round_id in seen_round_ids:
            errors.append(f"duplicate retrieval round id {round_id}")
        else:
            seen_round_ids.add(round_id)
        if not isinstance(round_.get("created_at"), str) or not round_["created_at"]:
            errors.append(f"{rwhere}.created_at must be non-empty")

        plan_errors, query_ids = _validate_query_plan(
            round_.get("queries"), where=f"{rwhere}.queries"
        )
        errors.extend(plan_errors)
        reused = sorted(query_ids & all_query_ids)
        if reused:
            errors.append(f"{rwhere} reuses query ids from earlier rounds: {reused}")
        all_query_ids |= query_ids

        result_keys: set[str] = set()
        results = round_.get("results", [])
        if not isinstance(results, list):
            errors.append(f"{rwhere}.results must be a list")
            results = []
        for index, result in enumerate(results):
            where = f"{rwhere}.results[{index}]"
            if not isinstance(result, dict):
                errors.append(f"{where} must be an object")
                continue
            key = result.get("canonical_key")
            if not isinstance(key, str) or not key:
                errors.append(f"{where}.canonical_key must be non-empty")
            elif key in result_keys:
                errors.append(f"duplicate merged retrieval result {key} in {round_id}")
            else:
                result_keys.add(key)
            result_query_ids = result.get("query_ids")
            if not isinstance(result_query_ids, list) or not set(result_query_ids) <= query_ids:
                errors.append(f"{where}.query_ids contains unknown queries")
            if result.get("query_support") != len(set(result_query_ids or [])):
                errors.append(f"{where}.query_support does not match distinct query_ids")

        called_queries: set[str] = set()
        calls = round_.get("backend_calls", [])
        if not isinstance(calls, list):
            errors.append(f"{rwhere}.backend_calls must be a list")
            calls = []
        for index, call in enumerate(calls):
            where = f"{rwhere}.backend_calls[{index}]"
            if not isinstance(call, dict):
                errors.append(f"{where} must be an object")
                continue
            if call.get("query_id") not in query_ids:
                errors.append(f"{where}.query_id is unknown in {round_id}")
            else:
                called_queries.add(call["query_id"])
            if call.get("status") not in {"success", "empty", "failed"}:
                errors.append(f"{where}.status must be success, empty, or failed")
            if not isinstance(call.get("retrieved_at"), str) or not call["retrieved_at"]:
                errors.append(f"{where}.retrieved_at must be non-empty")
            if not isinstance(call.get("backend_version"), str) or not call["backend_version"]:
                errors.append(f"{where}.backend_version must be non-empty")
            if call.get("status") in {"success", "empty"} and "raw_response" not in call:
                errors.append(f"{where}.raw_response must be retained")
        missing_calls = sorted(query_ids - called_queries)
        if missing_calls:
            errors.append(f"{rwhere} queries have no backend call records: {missing_calls}")

    visits = manifest.get("visits", [])
    if not isinstance(visits, list):
        errors.append("retrieval manifest visits must be a list")
        visits = []
    for index, visit in enumerate(visits):
        where = f"retrieval visits[{index}]"
        if not isinstance(visit, dict):
            errors.append(f"{where} must be an object")
            continue
        if visit.get("status") not in {"success", "failed"}:
            errors.append(f"{where}.status must be success or failed")
        if not isinstance(visit.get("backend_version"), str) or not visit["backend_version"]:
            errors.append(f"{where}.backend_version must be non-empty")
        url = visit.get("url")
        if not isinstance(url, str) or canonical_key(url) != visit.get("canonical_key"):
            errors.append(f"{where}.canonical_key does not match url")
        if visit.get("content") is not None:
            errors.append(f"{where} must not retain inline content; use content_file")
        content_file = visit.get("content_file")
        content_chars = visit.get("content_chars")
        if visit.get("status") == "success":
            if not isinstance(content_file, str) or not content_file:
                errors.append(f"{where}.content_file must reference a retained content file")
            if not isinstance(content_chars, int) or content_chars <= 0:
                errors.append(f"{where}.content_chars must be positive for a successful visit")
            if (
                isinstance(content_file, str)
                and content_file
                and isinstance(content_chars, int)
                and manifest_dir is not None
            ):
                content_path = manifest_dir / content_file
                if not content_path.is_file():
                    errors.append(f"{where}.content_file is missing: {content_file}")
                else:
                    retained = len(content_path.read_text(encoding="utf-8", errors="replace"))
                    if retained != content_chars:
                        errors.append(
                            f"{where}.content_chars does not match retained content file"
                        )
        elif content_file is not None:
            errors.append(f"{where}.content_file is only valid for a successful visit")
        view = visit.get("view")
        section = visit.get("section")
        if view == "section" and (not isinstance(section, str) or not section.strip()):
            errors.append(f"{where}.section must be non-empty for a section view")
        elif view != "section" and section is not None:
            errors.append(f"{where}.section is only valid for a section view")
        if not isinstance(visit.get("retrieved_at"), str) or not visit["retrieved_at"]:
            errors.append(f"{where}.retrieved_at must be non-empty")
    return errors


class SearchBackend(ABC):
    name: str
    version: str = "unknown"

    @abstractmethod
    async def search(self, query: str, max_results: int) -> dict[str, Any]:
        raise NotImplementedError


class FrozenCorpusBackend(SearchBackend):
    """Auditable local lexical search over a pinned JSON corpus."""

    name = "frozen"

    def __init__(self, path: Path) -> None:
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            raise RuntimeError(
                "frozen corpus must declare schema_version, corpus_id, cutoff, provenance, and items"
            )
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise RuntimeError("frozen corpus must be an object with an items list")
        if payload.get("schema_version") != 1:
            raise RuntimeError("frozen corpus schema_version must be 1")
        if not isinstance(payload.get("corpus_id"), str) or not payload["corpus_id"].strip():
            raise RuntimeError("frozen corpus must declare a non-empty corpus_id")
        if not isinstance(payload.get("cutoff"), str) or not payload["cutoff"].strip():
            raise RuntimeError("frozen corpus must declare a non-empty cutoff")
        if not isinstance(payload.get("created_at"), str) or not payload["created_at"].strip():
            raise RuntimeError("frozen corpus must declare created_at")
        if not isinstance(payload.get("provenance"), str) or not payload["provenance"].strip():
            raise RuntimeError("frozen corpus must declare provenance")
        if not isinstance(payload.get("prepared_before_task_ids"), bool):
            raise RuntimeError("frozen corpus must declare prepared_before_task_ids as boolean")
        seen: set[str] = set()
        for index, item in enumerate(payload["items"]):
            if not isinstance(item, dict) or not item.get("url") or not item.get("title"):
                raise RuntimeError(f"frozen corpus item {index} needs url and title")
            if not any(isinstance(item.get(field), str) and item[field].strip()
                       for field in ("text", "full_text", "abstract")):
                raise RuntimeError(f"frozen corpus item {index} has no retained content")
            key = canonical_key(item["url"])
            if not key or key in seen:
                raise RuntimeError(f"frozen corpus item {index} has duplicate/invalid URL")
            seen.add(key)
        self.payload = payload
        self.path = path.resolve()
        self.version = str(payload.get("corpus_id") or path.stem)
        self.corpus_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    async def search(self, query: str, max_results: int) -> dict[str, Any]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.payload["items"]:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            text = " ".join(
                str(item.get(field) or "") for field in ("title", "abstract", "text", "keywords")
            ).lower()
            tokens = set(re.findall(r"[a-z0-9]+", text))
            overlap = len(terms & tokens)
            phrase = 2 if query.lower() in text else 0
            score = overlap + phrase
            if score:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("title", ""))))
        items = []
        for _, item in scored[:max_results]:
            row: dict[str, Any] = {
                "url": item["url"],
                "title": item.get("title") or "No title",
                "snippet": item.get("abstract") or item.get("text") or "",
                "external_id": item.get("external_id") or item.get("arxiv_id"),
            }
            for field in RESULT_METADATA_FIELDS:
                if item.get(field) is not None:
                    row[field] = item[field]
            items.append(row)
        return {
            "items": items,
            "raw_response": {"matches": [item for _, item in scored[:max_results]]},
            "metadata": {
                "corpus_path": str(self.path),
                "corpus_id": self.version,
                "corpus_sha256": self.corpus_sha256,
                "cutoff": self.payload.get("cutoff"),
                "created_at": self.payload.get("created_at"),
                "provenance": self.payload.get("provenance"),
                "prepared_before_task_ids": self.payload.get("prepared_before_task_ids"),
                "client_license": "not_applicable",
                "invocation_surface": "local frozen lexical search",
            },
        }

    def read(self, url: str) -> str:
        key = canonical_key(url)
        for item in self.payload["items"]:
            if isinstance(item, dict) and canonical_key(item.get("url", "")) == key:
                content = item.get("text") or item.get("full_text") or item.get("abstract")
                if not isinstance(content, str) or not content.strip():
                    raise RuntimeError(f"frozen corpus has no retained content for {url}")
                return content
        raise RuntimeError(f"source is absent from frozen corpus: {url}")


class DeepXivBackend(SearchBackend):
    name = "deepxiv"

    def __init__(self) -> None:
        executable = shutil.which("deepxiv")
        if not executable:
            raise RuntimeError("deepxiv executable not found on PATH")
        self.command = [executable]
        self.env = None
        version = subprocess.run(
            self.command + ["--version"], text=True, capture_output=True,
            timeout=10, check=False,
        )
        self.version = (version.stdout or version.stderr).strip() or "installed-cli-unknown"

    async def search(self, query: str, max_results: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._search_sync, query, max_results)

    def _search_sync(self, query: str, max_results: int) -> dict[str, Any]:
        process = subprocess.run(
            self.command + ["search", query, "--limit", str(max_results), "--format", "json"],
            cwd=ROOT,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        if process.returncode != 0:
            message = (process.stderr or process.stdout).strip().splitlines()
            raise RuntimeError(message[-1] if message else f"exit {process.returncode}")
        payload = json.loads(process.stdout)
        rows = []
        if isinstance(payload, dict):
            rows = payload.get("results") or payload.get("result") or []
        if not isinstance(rows, list):
            rows = []
        output: list[dict[str, Any]] = []
        for row in rows[:max_results]:
            if not isinstance(row, dict):
                continue
            paper_id = row.get("arxiv_id")
            url = row.get("url") or (f"https://arxiv.org/abs/{paper_id}" if paper_id else "")
            if not url:
                continue
            tldr = row.get("tldr")
            if isinstance(tldr, dict):
                tldr = tldr.get("text")
            authors = row.get("authors")
            if isinstance(authors, list):
                authors = [
                    a.get("name", str(a)) if isinstance(a, dict) else str(a)
                    for a in authors
                ]
            citation_count = row.get("citation_count")
            if citation_count is None:
                citation_count = row.get("citation")
            item: dict[str, Any] = {
                "url": url,
                "title": row.get("title") or "No title",
                "snippet": row.get("abstract") or tldr or "",
                "external_id": paper_id,
            }
            for field, value in (
                ("authors", authors),
                ("date", row.get("date") or row.get("published")),
                ("citation_count", citation_count),
                ("tldr", tldr),
                ("venue", row.get("venue")),
                ("categories", row.get("categories")),
                ("github_url", row.get("github_url")),
                ("score", row.get("score")),
            ):
                if value is not None:
                    item[field] = value
            output.append(item)
        return {
            "items": output,
            "raw_response": payload,
            "metadata": {
                "client_license": "MIT",
                "invocation_surface": "deepxiv CLI",
            },
        }


class JinaSearchBackend(SearchBackend):
    name = "jina-search"
    version = "hosted-api-unknown"

    def __init__(self) -> None:
        if not os.environ.get("JINA_API_KEY"):
            raise RuntimeError(
                "JINA_API_KEY is not set; keyless s.jina.ai always returns 401"
            )

    async def search(self, query: str, max_results: int) -> dict[str, Any]:
        return await asyncio.to_thread(self._search_sync, query, max_results)

    def _search_sync(self, query: str, max_results: int) -> dict[str, Any]:
        url = "https://s.jina.ai/?q=" + urllib.parse.quote(query)
        headers = {"Accept": "application/json", "User-Agent": "HieraResearch/1"}
        if os.environ.get("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            payload = json.loads(response.read().decode("utf-8", errors="replace"))
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        output: list[dict[str, Any]] = []
        for row in (rows or [])[:max_results]:
            if not isinstance(row, dict) or not row.get("url"):
                continue
            output.append(
                {
                    "url": row["url"],
                    "title": row.get("title") or "No title",
                    "snippet": row.get("description") or (row.get("content") or "")[:500],
                }
            )
        return {
            "items": output,
            "raw_response": payload,
            "metadata": {
                "client_license": "stdlib adapter",
                "hosted_service": "Jina Search",
            },
        }


def build_backends(
    names: list[str], frozen_corpus: Path | None = None
) -> tuple[list[SearchBackend], list[dict[str, str]]]:
    backends: list[SearchBackend] = []
    failures: list[dict[str, str]] = []
    for name in dict.fromkeys(n.lower() for n in names):
        try:
            if name == "frozen":
                if frozen_corpus is None:
                    raise RuntimeError("frozen backend requires --frozen-corpus")
                backends.append(FrozenCorpusBackend(frozen_corpus))
            elif name == "deepxiv":
                backends.append(DeepXivBackend())
            elif name == "jina-search":
                backends.append(JinaSearchBackend())
            else:
                failures.append({"backend": name, "error": "unknown backend"})
        except Exception as exc:
            failures.append({"backend": name, "error": f"{type(exc).__name__}: {exc}"})
    return backends, failures


def merge_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = canonical_key(candidate.get("url", ""))
        if not key:
            continue
        current = merged.get(key)
        if current is None:
            current = {
                "canonical_key": key,
                "url": canonical_url(candidate["url"]),
                "title": candidate.get("title") or "No title",
                "snippet": candidate.get("snippet") or "",
                "query_ids": [],
                "queries": [],
                "backends": [],
                "best_rank": candidate.get("rank", 9999),
                "rank_sum": 0,
                "hit_count": 0,
                "external_id": candidate.get("external_id"),
            }
            merged[key] = current
        current["rank_sum"] += candidate.get("rank", 9999)
        current["hit_count"] += 1
        current["best_rank"] = min(current["best_rank"], candidate.get("rank", 9999))
        for field, value in (("query_ids", candidate.get("query_id")),
                             ("queries", candidate.get("query")),
                             ("backends", candidate.get("backend"))):
            if value and value not in current[field]:
                current[field].append(value)
        if len(candidate.get("snippet") or "") > len(current["snippet"]):
            current["snippet"] = candidate["snippet"]
        if candidate.get("rank", 9999) <= current["best_rank"] and candidate.get("title"):
            current["title"] = candidate["title"]
        if not current.get("external_id") and candidate.get("external_id"):
            current["external_id"] = candidate["external_id"]
        for field in RESULT_METADATA_FIELDS:
            value = candidate.get(field)
            if value not in (None, "", []) and current.get(field) in (None, "", []):
                current[field] = value

    output = list(merged.values())
    for item in output:
        item["query_support"] = len(item["query_ids"])
        item["backend_support"] = len(item["backends"])
        item["average_rank"] = item.pop("rank_sum") / max(item.pop("hit_count"), 1)
    output.sort(
        key=lambda item: (
            -item["query_support"],
            -item["backend_support"],
            item["best_rank"],
            item["average_rank"],
            item["canonical_key"],
        )
    )
    return output


def merged_results(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Derived global dedup view over all rounds; not authoritative state."""
    merged: dict[str, dict[str, Any]] = {}
    for round_ in manifest.get("rounds", []):
        if not isinstance(round_, dict):
            continue
        round_id = round_.get("round_id")
        results = round_.get("results", [])
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            key = result.get("canonical_key")
            if not isinstance(key, str) or not key:
                continue
            entry = merged.get(key)
            if entry is None:
                entry = {
                    "canonical_key": key,
                    "url": result.get("url"),
                    "title": result.get("title") or "No title",
                    "snippet": result.get("snippet") or "",
                    "external_id": result.get("external_id"),
                    "rounds": [],
                    "query_ids": [],
                    "queries": [],
                    "backends": [],
                    "best_rank": 9999,
                    "appearances": [],
                }
                merged[key] = entry
            if isinstance(round_id, str) and round_id not in entry["rounds"]:
                entry["rounds"].append(round_id)
            for field in ("query_ids", "queries", "backends"):
                for value in result.get(field) or []:
                    if value and value not in entry[field]:
                        entry[field].append(value)
            best = result.get("best_rank")
            if isinstance(best, (int, float)):
                entry["best_rank"] = min(entry["best_rank"], best)
            if len(result.get("snippet") or "") > len(entry["snippet"]):
                entry["snippet"] = result["snippet"]
            if not entry.get("external_id") and result.get("external_id"):
                entry["external_id"] = result["external_id"]
            for field in RESULT_METADATA_FIELDS:
                value = result.get(field)
                if value not in (None, "", []) and entry.get(field) in (None, "", []):
                    entry[field] = value
            entry["appearances"].append(
                {
                    "round_id": round_id,
                    "query_ids": list(result.get("query_ids") or []),
                    "backends": list(result.get("backends") or []),
                    "best_rank": result.get("best_rank"),
                    "average_rank": result.get("average_rank"),
                }
            )
    output = list(merged.values())
    for item in output:
        item["query_support"] = len(item["query_ids"])
        item["backend_support"] = len(item["backends"])
    output.sort(
        key=lambda item: (
            -item["query_support"],
            -item["backend_support"],
            item["best_rank"],
            item["canonical_key"],
        )
    )
    return output


def resolve_visit_content(visit: dict[str, Any], manifest_dir: Path) -> str | None:
    """Read a visit's retained content file; None when there is none to read."""
    content_file = visit.get("content_file") if isinstance(visit, dict) else None
    if not isinstance(content_file, str) or not content_file:
        return None
    path = manifest_dir / content_file
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


_VERIFICATION_RANK = {"snippet_only": 0, "preview": 1, "section": 2, "full_text": 3}
_VIEW_VERIFICATION = {
    "preview": "preview",
    "section": "section",
    "full_text": "full_text",
    "page": "full_text",
}


def verification_statuses(manifest: dict[str, Any]) -> dict[str, str]:
    """Per-key verification tier, derivable from the manifest alone.

    Every canonical key with a tool-recorded receipt starts at
    ``snippet_only`` (no substantive read: a search hit, or only head/brief
    triage visits).  A successful visit raises the tier to its view's level:
    ``preview``, ``section``, or ``full_text`` (``page`` counts as
    ``full_text``); ``head``/``brief`` never raise it.
    """
    tiers: dict[str, str] = {}
    for result in merged_results(manifest):
        tiers.setdefault(result["canonical_key"], "snippet_only")
    for visit in manifest.get("visits", []):
        if not isinstance(visit, dict) or visit.get("status") != "success":
            continue
        key = visit.get("canonical_key")
        if not isinstance(key, str) or not key:
            continue
        current = tiers.setdefault(key, "snippet_only")
        raised = _VIEW_VERIFICATION.get(str(visit.get("view")))
        if raised is not None and _VERIFICATION_RANK[raised] > _VERIFICATION_RANK[current]:
            tiers[key] = raised
    return tiers


def unexplored_leads(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """High-rank merged hits with no visit receipt — the status dashboard's leads."""
    visited_keys = {
        visit.get("canonical_key")
        for visit in manifest.get("visits", [])
        if isinstance(visit, dict) and visit.get("canonical_key")
    }
    return [
        entry
        for entry in merged_results(manifest)
        if isinstance(entry.get("best_rank"), (int, float))
        and entry["best_rank"] <= HIGH_RANK_THRESHOLD
        and entry["canonical_key"] not in visited_keys
    ]


def _clip(text: Any, limit: int = 200) -> str:
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _render_result_card(result: dict[str, Any]) -> str:
    queries = ",".join(str(q) for q in result.get("query_ids") or [])
    lines = [f"[{queries} rank {result.get('best_rank')}] {_clip(result.get('title'), 120)}"]
    meta = [str(result.get("url") or result.get("canonical_key") or "")]
    authors = result.get("authors")
    if isinstance(authors, list):
        authors = ", ".join(str(a) for a in authors)
    if authors:
        meta.append(f"authors: {_clip(authors, 120)}")
    if result.get("date"):
        meta.append(f"date: {result['date']}")
    if result.get("citation_count") is not None:
        meta.append(f"citations: {result['citation_count']}")
    lines.append("  " + " | ".join(meta))
    if result.get("tldr"):
        lines.append(f"  tldr: {_clip(result['tldr'])}")
    if result.get("snippet"):
        lines.append(f"  snippet: {_clip(result['snippet'])}")
    return "\n".join(lines)


def _round_query_outcomes(round_: dict[str, Any]) -> list[dict[str, str]]:
    """Per-query outcome within one round: ok / empty (success, zero hits) / failed."""
    hit_queries: set[str] = set()
    for result in round_.get("results", []):
        if isinstance(result, dict):
            hit_queries.update(str(q) for q in result.get("query_ids") or [])
    call_statuses: dict[str, list[str]] = {}
    for call in round_.get("backend_calls", []):
        if isinstance(call, dict) and call.get("query_id"):
            call_statuses.setdefault(str(call["query_id"]), []).append(
                str(call.get("status"))
            )
    outcomes: list[dict[str, str]] = []
    for query in round_.get("queries", []):
        if not isinstance(query, dict):
            continue
        qid = str(query.get("id"))
        statuses = call_statuses.get(qid, [])
        if qid in hit_queries:
            outcome = "ok"
        elif "empty" in statuses or "success" in statuses:
            # explicit empty first; bare success with zero hits is the T2-era fallback
            outcome = "empty"
        else:
            outcome = "failed"
        outcomes.append(
            {"id": qid, "text": str(query.get("text") or ""), "outcome": outcome}
        )
    return outcomes


_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+(.+?)\s*$")


def _markdown_sections(content: str) -> list[dict[str, Any]]:
    headings = [
        (match.start(), match.group(1).strip())
        for match in _HEADING_RE.finditer(content)
    ]
    return [
        {
            "name": name,
            "start": start,
            "end": headings[index + 1][0] if index + 1 < len(headings) else len(content),
        }
        for index, (start, name) in enumerate(headings)
    ]


def _render_head_view(url: str, content: str, manifest: Path) -> str:
    if len(content) <= VISIT_HEAD_CHARS:
        return content
    lines = [
        content[:VISIT_HEAD_CHARS],
        f"\n… [{len(content) - VISIT_HEAD_CHARS} more chars of {len(content)}]",
    ]
    sections = _markdown_sections(content)
    if sections:
        lines.append("sections:")
        for section in sections:
            lines.append(
                f"  {section['name']}  (chars {section['start']}-{section['end']})"
            )
    else:
        lines.append(f"no markdown headings; page by offset ({READ_WINDOW_CHARS} chars):")
        for start in range(0, len(content), READ_WINDOW_CHARS):
            lines.append(f"  chars {start}-{min(start + READ_WINDOW_CHARS, len(content))}")
    lines.append(
        f"continue: python tools/search_backends.py read "
        f"--manifest {shlex.quote(str(manifest))} "
        f"--url {shlex.quote(url)} [--section NAME | --offset N]"
    )
    return "\n".join(lines)


async def dispatch_search(
    queries: list[dict[str, Any]], backends: list[SearchBackend], max_results: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    tasks: list[tuple[dict[str, Any], SearchBackend, asyncio.Task]] = []
    for query in queries:
        for backend in backends:
            tasks.append(
                (query, backend, asyncio.create_task(backend.search(query["text"], max_results)))
            )
    raw: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    calls: list[dict[str, Any]] = []
    for query, backend, task in tasks:
        retrieved_at = datetime.now(timezone.utc).isoformat()
        try:
            response = await task
        except Exception as exc:
            failure = {"query_id": query["id"], "backend": backend.name,
                       "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            calls.append(
                {**failure, "status": "failed", "retrieved_at": retrieved_at,
                 "backend_version": backend.version}
            )
            continue
        rows = response.get("items", []) if isinstance(response, dict) else []
        raw_response = response.get("raw_response") if isinstance(response, dict) else None
        calls.append(
            {
                "query_id": query["id"],
                "backend": backend.name,
                "backend_version": backend.version,
                "status": "success" if rows else "empty",
                "retrieved_at": retrieved_at,
                "raw_response": raw_response,
                "metadata": response.get("metadata", {}) if isinstance(response, dict) else {},
            }
        )
        for rank, row in enumerate(rows, start=1):
            raw.append(
                {**row, "query_id": query["id"], "query": query["text"],
                 "backend": backend.name, "rank": rank}
            )
    return raw, failures, calls


def _strip_html(raw: str) -> str:
    text = _TAG_RE.sub(" ", raw)
    text = _HTML_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _web_read(url: str) -> tuple[str, str, str | None]:
    """jina-reader first, direct fallback; the note records any reader failure."""
    headers = {"Accept": "text/plain", "X-Return-Format": "markdown",
               "User-Agent": "HieraResearch/1"}
    if os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
    reader_error: str | None = None
    try:
        request = urllib.request.Request("https://r.jina.ai/" + url, headers=headers)
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            text = response.read().decode("utf-8", errors="replace")
        if text.strip():
            return text, "jina-read", None
        reader_error = "jina-reader returned empty content"
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        reader_error = f"jina-reader: {type(exc).__name__}: {exc}"
    try:
        return _direct_visit(url), "direct", reader_error
    except Exception as exc:
        raise RuntimeError(f"{exc} (after {reader_error})") from exc


def _web_content_error(content: str) -> str | None:
    """Rejection reason for a fetched web page, or None when it looks like content."""
    stripped = content.strip()
    if not stripped:
        return "empty content"
    lowered = stripped[:2000].casefold()
    for marker in _WEB_ERROR_MARKERS:
        if marker in lowered:
            return f"error-page marker {marker!r}"
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and not payload.get("data") and (
            "code" in payload or "message" in payload
        ):
            return "JSON error response instead of page content"
    if len(stripped) < VISIT_MIN_CONTENT_CHARS:
        return f"content too short ({len(stripped)} chars)"
    return None


def _direct_visit(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "HieraResearch/1"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or url.lower().split("?", 1)[0].endswith(".pdf"):
            raise RuntimeError(
                "direct PDF extraction is unavailable; use retained frozen text or an explicit auditable reader"
            )
        raw = response.read().decode("utf-8", errors="replace")
    return _strip_html(raw)


def _deepxiv_read(
    url: str,
    view: str,
    section: str | None,
    *,
    backend: DeepXivBackend | None = None,
) -> tuple[str, str, str]:
    paper_id = arxiv_id(url)
    if not paper_id:
        raise RuntimeError("URL is not an arXiv paper")
    backend = backend or DeepXivBackend()
    command = backend.command + ["paper", paper_id, "--format", "json"]
    if view == "brief":
        command.append("--brief")
    elif view == "head":
        command.append("--head")
    elif view == "preview":
        command.append("--preview")
    elif view == "full_text":
        command.append("--raw")
    elif view == "section":
        if not section:
            raise RuntimeError("section view requires --section")
        command.extend(["--section", section])
    else:
        raise RuntimeError(f"unsupported DeepXiv view {view}")
    process = subprocess.run(
        command, cwd=ROOT, env=backend.env, text=True, capture_output=True,
        timeout=120, check=False,
    )
    if process.returncode != 0:
        message = (process.stderr or process.stdout).strip().splitlines()
        raise RuntimeError(message[-1] if message else f"exit {process.returncode}")
    return process.stdout, "deepxiv", backend.version


def _head_sections(content: str) -> list[dict[str, Any]]:
    """Normalize DeepXiv's dict/list section-map variants."""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return []
    containers: list[Any] = [payload]
    if isinstance(payload, dict):
        for key in ("data", "result", "paper"):
            value = payload.get(key)
            if isinstance(value, dict):
                containers.append(value)
    sections: Any = None
    for container in containers:
        if isinstance(container, dict) and isinstance(container.get("sections"), (dict, list)):
            sections = container["sections"]
            break
    normalized: list[dict[str, Any]] = []
    if isinstance(sections, dict):
        rows = [
            (
                name,
                info if isinstance(info, dict) else {"tldr": str(info or "")},
                position,
            )
            for position, (name, info) in enumerate(sections.items())
        ]
    elif isinstance(sections, list):
        rows = []
        for position, info in enumerate(sections):
            if isinstance(info, dict):
                name = info.get("name") or info.get("title") or info.get("section")
                rows.append((name, info, position))
            elif isinstance(info, str):
                rows.append((info, {}, position))
    else:
        rows = []
    seen: set[str] = set()
    for name, info, position in rows:
        if not isinstance(name, str) or not name.strip():
            continue
        clean_name = name.strip()
        key = clean_name.casefold()
        if key in seen:
            continue
        seen.add(key)
        idx = info.get("idx", position)
        normalized.append(
            {
                "name": clean_name,
                "idx": idx if isinstance(idx, (int, float)) else position,
                "tldr": str(info.get("tldr") or info.get("summary") or ""),
                "token_count": info.get("token_count"),
            }
        )
    return normalized


def _source_query_context(manifest: dict[str, Any], url: str) -> tuple[str, set[str]]:
    key = canonical_key(url)
    query_ids: set[str] = set()
    for result in merged_results(manifest):
        if result.get("canonical_key") == key:
            query_ids.update(
                item for item in result.get("query_ids", []) if isinstance(item, str)
            )
    texts: list[str] = []
    roles: set[str] = set()
    for round_ in manifest.get("rounds", []):
        if not isinstance(round_, dict):
            continue
        for query in round_.get("queries", []):
            if not isinstance(query, dict) or query.get("id") not in query_ids:
                continue
            if isinstance(query.get("text"), str):
                texts.append(query["text"])
            roles.update(
                role for role in query.get("evidence_roles", []) if isinstance(role, str)
            )
    return " ".join(texts), roles


def _select_deepxiv_sections(
    head_content: str,
    *,
    query_text: str = "",
    evidence_roles: set[str] | None = None,
    limit: int = DEEPXIV_MAX_SECTIONS,
) -> list[str]:
    """Choose source-body sections for the evidence question, not document order."""
    sections = _head_sections(head_content)
    if not sections or limit <= 0:
        return []
    context_terms = {
        term
        for term in _WORD_RE.findall(query_text.casefold())
        if len(term) > 2 and term not in _SECTION_STOPWORDS
    }
    roles = evidence_roles or set()
    ranked: list[tuple[float, float, str]] = []
    for section in sections:
        name = section["name"]
        lowered = name.casefold()
        if re.search(
            r"\b(abstract|references?|bibliography|acknowledg(?:e)?ments?)\b",
            lowered,
        ):
            continue
        score = 25.0
        priorities = (
            (r"\b(methods?|methodology|approach|algorithm|model|architecture|training)\b", 100),
            (r"\b(experiments?|results?|evaluation|benchmark|ablation|analysis)\b", 95),
            (r"\b(limitations?|discussion|failure|error analysis)\b", 90),
            (r"\b(conclusions?|future work)\b", 65),
            (r"\b(introduction|background|related work|preliminar(?:y|ies))\b", 35),
        )
        for pattern, priority in priorities:
            if re.search(pattern, lowered):
                score = max(score, float(priority))
        if "counterevidence" in roles or "failure_mode" in roles:
            if re.search(r"\b(limitations?|discussion|failure|error|analysis)\b", lowered):
                score += 20
        if "hypothesis" in roles or "relation" in roles:
            if re.search(r"\b(methods?|approach|algorithm|model|architecture)\b", lowered):
                score += 15
        section_terms = set(
            _WORD_RE.findall(f"{name} {section.get('tldr', '')}".casefold())
        )
        score += min(len(context_terms & section_terms), 8) * 4
        ranked.append((-score, float(section["idx"]), name))
    ranked.sort()
    return [name for _, _, name in ranked[:limit]]


def _deepxiv_progressive_read(
    url: str, manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    """Triage an arXiv paper, then fetch evidence-bearing body content."""
    backend = DeepXivBackend()
    head, backend_name, backend_version = _deepxiv_read(
        url, "head", None, backend=backend
    )
    attempts: list[dict[str, Any]] = [
        {
            "backend": backend_name,
            "backend_version": backend_version,
            "view": "head",
            "section": None,
            "status": "success",
            "content": head,
            "error": None,
        }
    ]
    query_text, evidence_roles = _source_query_context(manifest, url)
    section_names = _select_deepxiv_sections(
        head, query_text=query_text, evidence_roles=evidence_roles
    )
    for section_name in section_names:
        try:
            content, _, _ = _deepxiv_read(
                url, "section", section_name, backend=backend
            )
            attempts.append(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "view": "section",
                    "section": section_name,
                    "status": "success",
                    "content": content,
                    "error": None,
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "view": "section",
                    "section": section_name,
                    "status": "failed",
                    "content": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not any(
        attempt["status"] == "success" and attempt["view"] == "section"
        for attempt in attempts
    ):
        try:
            content, _, _ = _deepxiv_read(url, "preview", None, backend=backend)
            attempts.append(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "view": "preview",
                    "section": None,
                    "status": "success",
                    "content": content,
                    "error": None,
                }
            )
        except Exception as exc:
            attempts.append(
                {
                    "backend": backend_name,
                    "backend_version": backend_version,
                    "view": "preview",
                    "section": None,
                    "status": "failed",
                    "content": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return attempts


def _retain_progressive_content(
    attempts: list[dict[str, Any]], budget_chars: int
) -> None:
    """Share one source-reading budget across triage and body receipts."""
    successful = [
        attempt for attempt in attempts if attempt["status"] == "success"
    ]
    head = next(
        (attempt for attempt in successful if attempt["view"] == "head"), None
    )
    body = [attempt for attempt in successful if attempt["view"] in SUBSTANTIVE_VIEWS]
    head_budget = min(4000, max(budget_chars // 6, 1)) if head else 0
    if head:
        head["content"] = head["content"][:head_budget]
    remaining = max(budget_chars - head_budget, 0)
    body_budget = max(remaining // len(body), 1) if body else 0
    for attempt in body:
        attempt["content"] = attempt["content"][:body_budget]


def add_visit(
    manifest: dict[str, Any], manifest_dir: Path, *, url: str, backend: str, view: str,
    status: str, content: str | None = None, error: str | None = None,
    backend_version: str = "unknown", section: str | None = None,
    note: str | None = None,
) -> None:
    visits = manifest.setdefault("visits", [])
    visit: dict[str, Any] = {
        "url": canonical_url(url),
        "canonical_key": canonical_key(url),
        "backend": backend,
        "backend_version": backend_version,
        "view": view,
        "section": section,
        "status": status,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    if note:
        visit["note"] = note
    if status == "success" and content:
        slug = re.sub(r"[^a-z0-9]+", "-", visit["canonical_key"].lower()).strip("-")[:60]
        filename = f"{len(visits):03d}-{slug or 'source'}.txt"
        retrieval_dir = manifest_dir / "retrieval"
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        (retrieval_dir / filename).write_text(content, encoding="utf-8")
        visit["content_file"] = f"retrieval/{filename}"
        visit["content_chars"] = len(content)
    visits.append(visit)


def _parse_cli_objects(
    values: list[str], *, label: str, fields: set[str]
) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for index, raw in enumerate(values, start=1):
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} {index} is not valid JSON: {exc.msg}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{label} {index} must be a JSON object")
        unknown = sorted(set(item) - fields)
        if unknown:
            raise ValueError(f"{label} {index} has unknown fields {unknown}")
        parsed.append(item)
    return parsed


def _reject_legacy_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"retrieval manifest schema_version must be {SCHEMA_VERSION}; "
            "regenerate this disposable run artifact"
        )


def _next_query_number(manifest: dict[str, Any]) -> int:
    highest = 0
    for round_ in manifest.get("rounds", []):
        if not isinstance(round_, dict):
            continue
        for query in round_.get("queries", []):
            if isinstance(query, dict):
                match = re.fullmatch(r"q-(\d+)", str(query.get("id") or ""))
                if match:
                    highest = max(highest, int(match.group(1)))
    return highest + 1


def cmd_search(args: argparse.Namespace) -> int:
    specs = _parse_cli_objects(
        args.query_spec,
        label="--query-spec",
        fields={"text", "target_dimension_ids", "evidence_roles"},
    )
    manifest = load_manifest(args.manifest)
    _reject_legacy_manifest(manifest)
    names = args.backend or (["frozen"] if args.frozen_corpus else [])
    if not names:
        print(
            json.dumps(
                {
                    "ok": False,
                    "errors": [
                        "select --backend explicitly, or provide --frozen-corpus for the reproducible default"
                    ],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    if args.frozen_corpus is not None and any(name != "frozen" for name in names):
        print(
            json.dumps(
                {"ok": False, "errors": ["--frozen-corpus permits only the frozen backend"]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    if args.frozen_corpus is None and "frozen" in names:
        print(
            json.dumps(
                {"ok": False, "errors": ["the frozen backend requires --frozen-corpus"]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    start = _next_query_number(manifest)
    queries = [
        {
            "id": f"q-{number:02d}",
            "text": spec.get("text"),
            "target_dimension_ids": spec.get("target_dimension_ids"),
            "evidence_roles": spec.get("evidence_roles"),
        }
        for number, spec in enumerate(specs, start=start)
    ]
    plan_errors, _ = _validate_query_plan(queries)
    if plan_errors:
        print(json.dumps({"ok": False, "errors": plan_errors}, indent=2), file=sys.stderr)
        return 1

    backends, unavailable = build_backends(names, args.frozen_corpus)
    raw, failures, calls = asyncio.run(dispatch_search(queries, backends, args.max_results))
    for failure in unavailable:
        for query in queries:
            calls.append(
                {
                    "query_id": query["id"],
                    "backend": failure["backend"],
                    "backend_version": "unavailable",
                    "status": "failed",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "error": failure["error"],
                }
            )
    results = merge_candidates(raw)
    rounds = manifest.setdefault("rounds", [])
    round_id = f"r-{len(rounds) + 1:02d}"
    rounds.append(
        {
            "round_id": round_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "queries": queries,
            "backend_calls": calls,
            "backend_failures": unavailable + failures,
            "results": results,
        }
    )
    save_manifest(args.manifest, manifest)
    errors = validate_manifest(manifest, manifest_dir=args.manifest.parent)
    outcomes = _round_query_outcomes(rounds[-1])
    empty = [o["id"] for o in outcomes if o["outcome"] == "empty"]
    failed = [o["id"] for o in outcomes if o["outcome"] == "failed"]
    summary = (
        f"round {round_id} | queries: {len(queries)} | hits: {len(results)} "
        f"| max_results: {args.max_results} | empty: {','.join(empty) or '-'} "
        f"| failed: {','.join(failed) or '-'} | validate: {'ok' if not errors else 'FAILED'}"
    )
    round_failures = unavailable + failures
    if round_failures:
        summary += "\nfailures: " + "; ".join(
            f"{failure['backend']}:{failure.get('query_id', '-')} {failure['error']}"
            for failure in round_failures
        )
    if empty:
        summary += (
            "\nempty calls returned zero results; treat them as diagnostic failures "
            "(check backend credentials/quota, or widen the query)"
        )
    print(summary)
    for result in results:
        print()
        print(_render_result_card(result))
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2), file=sys.stderr)
    return 0 if results and not errors else 1


def cmd_visit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _reject_legacy_manifest(manifest)
    view = args.view
    if args.section and view != "section":
        print("visit failed: --section is only valid with --view section", file=sys.stderr)
        return 1
    try:
        note = None
        if args.frozen_corpus:
            frozen = FrozenCorpusBackend(args.frozen_corpus)
            content = frozen.read(args.url)
            backend = "frozen"
            backend_version = frozen.version
            if view == "auto":
                view = "full_text"
        elif args.visit_backend == "jina-read":
            content, backend, note = _web_read(args.url)
            backend_version = "hosted-api-unknown" if backend == "jina-read" else "stdlib"
            if view == "auto":
                view = "full_text"
        elif args.visit_backend == "direct":
            content, backend = _direct_visit(args.url), "direct"
            backend_version = "stdlib"
            if view == "auto":
                view = "full_text"
        elif arxiv_id(args.url) and view == "auto":
            attempts = _deepxiv_progressive_read(args.url, manifest)
            _retain_progressive_content(attempts, VISIT_CONTENT_STORE_CHARS)
            first_visit = len(manifest.get("visits", []))
            for attempt in attempts:
                add_visit(
                    manifest,
                    args.manifest.parent,
                    url=args.url,
                    backend=attempt["backend"],
                    view=attempt["view"],
                    status=attempt["status"],
                    content=attempt["content"],
                    error=attempt["error"],
                    backend_version=attempt["backend_version"],
                    section=attempt["section"],
                )
            save_manifest(args.manifest, manifest)
            body_attempts = [
                attempt
                for attempt in attempts
                if attempt["status"] == "success"
                and attempt["view"] in SUBSTANTIVE_VIEWS
            ]
            rendered = []
            for index, attempt in enumerate(attempts):
                if attempt["status"] != "success":
                    continue
                label = attempt["view"]
                if attempt["section"]:
                    label += f": {attempt['section']}"
                body = attempt["content"]
                chunk = body[:VISIT_HEAD_CHARS]
                if len(body) > VISIT_HEAD_CHARS:
                    # the receipt index pins the exact visit: a --url/--view
                    # scan would page through the last section instead
                    chunk += (
                        f"\n… [{len(body) - VISIT_HEAD_CHARS} more chars; continue: read "
                        f"--manifest {shlex.quote(str(args.manifest))} "
                        f"--visit {first_visit + index}]"
                    )
                rendered.append(f"## DeepXiv {label}\n\n{chunk}")
            if rendered:
                print("\n\n".join(rendered))
            if not body_attempts:
                print(
                    "visit failed: DeepXiv returned head metadata but no substantive "
                    "section or preview content",
                    file=sys.stderr,
                )
                return 1
            return 0
        elif arxiv_id(args.url) and view in {"brief", "head", "preview", "section", "full_text"}:
            content, backend, backend_version = _deepxiv_read(args.url, view, args.section)
        else:
            content, backend, note = _web_read(args.url)
            backend_version = "hosted-api-unknown" if backend == "jina-read" else "stdlib"
            if view == "auto":
                view = "full_text"
        content = content[:VISIT_CONTENT_STORE_CHARS]
        problem = (
            _web_content_error(content) if backend in {"jina-read", "direct"} else None
        )
        if problem:
            if note:
                problem = f"{problem} (after {note})"
            add_visit(manifest, args.manifest.parent, url=args.url, backend=backend,
                      view=view, status="failed", error=problem,
                      backend_version=backend_version, section=args.section)
            save_manifest(args.manifest, manifest)
            print(f"visit failed: {problem}", file=sys.stderr)
            return 1
        add_visit(manifest, args.manifest.parent, url=args.url, backend=backend, view=view,
                  status="success", content=content, backend_version=backend_version,
                  section=args.section, note=note)
        save_manifest(args.manifest, manifest)
        print(_render_head_view(canonical_url(args.url), content, args.manifest))
        return 0
    except Exception as exc:
        add_visit(manifest, args.manifest.parent, url=args.url, backend="auto", view=view,
                  status="failed", error=f"{type(exc).__name__}: {exc}",
                  backend_version="unknown", section=args.section)
        save_manifest(args.manifest, manifest)
        print(f"visit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def cmd_read(args: argparse.Namespace) -> int:
    """Read back stored visit content; never appends a visit receipt."""
    manifest = load_manifest(args.manifest)
    _reject_legacy_manifest(manifest)
    visit = None
    if args.visit is not None:
        visits = manifest.get("visits", [])
        if 0 <= args.visit < len(visits):
            visit = visits[args.visit]
    else:
        if not args.url:
            print("read failed: --url is required without --visit", file=sys.stderr)
            return 1
        key = canonical_key(args.url)
        for candidate in reversed(manifest.get("visits", [])):
            if not isinstance(candidate, dict):
                continue
            if candidate.get("status") != "success" or not candidate.get("content_file"):
                continue
            if key and candidate.get("canonical_key") != key:
                continue
            if args.view and candidate.get("view") != args.view:
                continue
            visit = candidate
    if not isinstance(visit, dict) or not visit.get("content_file"):
        print("read failed: no stored successful visit matches", file=sys.stderr)
        return 1
    content_path = args.manifest.parent / visit["content_file"]
    if not content_path.is_file():
        print(f"read failed: missing content file {visit['content_file']}", file=sys.stderr)
        return 1
    content = content_path.read_text(encoding="utf-8", errors="replace")
    if args.section:
        sections = _markdown_sections(content)
        match = next(
            (s for s in sections if s["name"].casefold() == args.section.casefold()), None
        )
        if match is None:
            available = ", ".join(s["name"] for s in sections) or "(no markdown headings)"
            print(f"read failed: no section {args.section!r}; available: {available}",
                  file=sys.stderr)
            return 1
        body = content[match["start"]:match["end"]]
        print(body[: args.length])
        if len(body) > args.length:
            print(f"\n… [section continues: --offset {match['start'] + args.length}]")
        return 0
    start = args.offset
    if start < 0 or start >= len(content):
        print(f"read failed: offset {start} outside content of {len(content)} chars",
              file=sys.stderr)
        return 1
    print(content[start: start + args.length])
    if start + args.length < len(content):
        print(f"\n… [continue: --offset {start + args.length}]")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _reject_legacy_manifest(manifest)
    rounds = [r for r in manifest.get("rounds", []) if isinstance(r, dict)]
    if not rounds:
        print("results failed: manifest has no rounds yet", file=sys.stderr)
        return 1
    if args.round is None:
        round_ = rounds[-1]
    else:
        wanted = args.round
        if not wanted.startswith("r-"):
            try:
                wanted = f"r-{int(wanted):02d}"
            except ValueError:
                pass
        round_ = next((r for r in rounds if r.get("round_id") == wanted), None)
        if round_ is None:
            available = ", ".join(str(r.get("round_id")) for r in rounds)
            print(
                f"results failed: no round {args.round!r}; available: {available}",
                file=sys.stderr,
            )
            return 1
    results = round_.get("results", [])
    print(
        f"round {round_.get('round_id')} | queries: {len(round_.get('queries', []))} "
        f"| hits: {len(results)}"
    )
    for result in results:
        print()
        print(_render_result_card(result))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _reject_legacy_manifest(manifest)
    rounds = [r for r in manifest.get("rounds", []) if isinstance(r, dict)]
    visits = [v for v in manifest.get("visits", []) if isinstance(v, dict)]

    query_dims: dict[str, list[str]] = {}
    for round_ in rounds:
        for query in round_.get("queries", []):
            if not isinstance(query, dict) or not isinstance(query.get("id"), str):
                continue
            targets = query.get("target_dimension_ids")
            query_dims[query["id"]] = (
                [t for t in targets if isinstance(t, str)]
                if isinstance(targets, list)
                else []
            )

    result_queries: dict[str, set[str]] = {}
    dim_results: dict[str, set[str]] = {}
    for round_ in rounds:
        for result in round_.get("results", []):
            if not isinstance(result, dict) or not result.get("canonical_key"):
                continue
            key = result["canonical_key"]
            for qid in result.get("query_ids") or []:
                result_queries.setdefault(key, set()).add(qid)
                for dim in query_dims.get(qid, []):
                    dim_results.setdefault(dim, set()).add(key)

    visited_keys = {v.get("canonical_key") for v in visits if v.get("canonical_key")}
    dim_visits: dict[str, int] = {}
    for visit in visits:
        dims: set[str] = set()
        for qid in result_queries.get(visit.get("canonical_key"), ()):
            dims.update(query_dims.get(qid, []))
        for dim in dims:
            dim_visits[dim] = dim_visits.get(dim, 0) + 1

    global_view = merged_results(manifest)
    total_queries = sum(len(r.get("queries", [])) for r in rounds)
    tier_counts: dict[str, int] = {}
    for tier in verification_statuses(manifest).values():
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    verification = " ".join(
        f"{tier}={tier_counts[tier]}"
        for tier in ("full_text", "section", "preview", "snippet_only")
        if tier_counts.get(tier)
    )
    lines = [
        f"rounds: {len(rounds)} | queries: {total_queries} | "
        f"unique results: {len(global_view)} | visits: {len(visits)}",
        f"verification: {verification or '(no receipts)'}",
        "dimensions (results / visits):",
    ]
    dims = sorted(set(dim_results) | set(dim_visits))
    if not dims:
        lines.append("  (no query targets recorded)")
    for dim in dims:
        lines.append(f"  {dim}: {len(dim_results.get(dim, set()))} / {dim_visits.get(dim, 0)}")

    leads = unexplored_leads(manifest)
    lines.append(
        f"unvisited high-rank hits (best_rank <= {HIGH_RANK_THRESHOLD}): {len(leads)}"
    )
    for entry in leads[:20]:
        lines.append(
            f"  [rank {entry['best_rank']}] {_clip(entry.get('title'), 100)} — "
            f"{entry['canonical_key']} (rounds: {','.join(entry['rounds'])})"
        )
    if len(leads) > 20:
        lines.append(f"  … and {len(leads) - 20} more")

    lines.append("queries with no results:")
    reported = False
    for round_ in rounds:
        for outcome in _round_query_outcomes(round_):
            if outcome["outcome"] == "ok":
                continue
            reported = True
            lines.append(
                f"  {round_.get('round_id')} {outcome['id']} [{outcome['outcome']}] "
                f"\"{_clip(outcome['text'], 80)}\""
            )
    if not reported:
        lines.append("  (none)")
    print("\n".join(lines))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest, manifest_dir=args.manifest.parent)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="fan queries across usable backends")
    search.add_argument("--manifest", type=Path, required=True)
    search.add_argument(
        "--query-spec",
        action="append",
        required=True,
        help="JSON object with text, target_dimension_ids, and evidence_roles",
    )
    search.add_argument("--backend", action="append",
                        choices=["frozen", "deepxiv", "jina-search"])
    search.add_argument(
        "--frozen-corpus",
        type=Path,
        help="pinned local JSON corpus; implies the frozen backend when --backend is omitted",
    )
    search.add_argument("--max-results", type=int, default=50)
    search.set_defaults(func=cmd_search)

    visit = sub.add_parser("visit", help="read a source and append a visit receipt")
    visit.add_argument("--manifest", type=Path, required=True)
    visit.add_argument("--url", required=True)
    visit.add_argument(
        "--view", choices=["auto", "brief", "head", "preview", "section", "full_text"],
        default="auto",
    )
    visit.add_argument("--section")
    visit.add_argument("--frozen-corpus", type=Path)
    visit.add_argument(
        "--visit-backend",
        choices=["auto", "direct", "jina-read"],
        default="auto",
        help="direct skips the reader; auto uses DeepXiv for arXiv and jina-reader (direct fallback) otherwise",
    )
    visit.set_defaults(func=cmd_visit)

    read = sub.add_parser(
        "read", help="read back stored visit content (section/offset); appends no receipt"
    )
    read.add_argument("--manifest", type=Path, required=True)
    read.add_argument("--url")
    read.add_argument(
        "--visit", type=int, help="visit receipt index; overrides --url selection"
    )
    read.add_argument("--view", help="restrict --url selection to a receipt view")
    read.add_argument("--section", help="markdown heading name within the stored content")
    read.add_argument("--offset", type=int, default=0)
    read.add_argument("--length", type=int, default=READ_WINDOW_CHARS)
    read.set_defaults(func=cmd_read)

    status = sub.add_parser("status", help="triage dashboard over rounds and visits")
    status.add_argument("--manifest", type=Path, required=True)
    status.set_defaults(func=cmd_status)

    results = sub.add_parser("results", help="browse a past round's result cards")
    results.add_argument("--manifest", type=Path, required=True)
    results.add_argument(
        "--round", help="round id (r-NN or bare number); default: latest round"
    )
    results.set_defaults(func=cmd_results)

    validate = sub.add_parser("validate", help="validate a retrieval manifest")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.set_defaults(func=cmd_validate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
