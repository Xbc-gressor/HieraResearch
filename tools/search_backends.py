#!/usr/bin/env python3
"""Local-first retrieval, optional backends, and auditable visit receipts.

Frozen-corpus search and every artifact-integrity check use only the standard
library. DeepXiv, Jina, and Claude-native WebSearch/WebFetch are explicit,
optional coverage adapters; successful external visits can be recorded with
``record-visit`` so the same manifest integrity checks still apply.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import os
import re
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
SCHEMA_VERSION = 3
EXTERNAL_DRAFT_SCHEMA_VERSION = 1
LANE_BUDGETS = {"novelty": 2048, "grounding": 6000}
EVIDENCE_ROLES = {
    "hypothesis",
    "baseline",
    "failure_mode",
    "counterevidence",
    "relation",
    "inner_hpo_prior",
}
INNER_HPO_ROLE = "inner_hpo_prior"
DIMENSION_BOUND_ROLES = {"hypothesis", "relation"}
MAX_SHARED = 6
MAX_SELECTED = 18
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


def is_substantive_grounding_visit(visit: Any) -> bool:
    """Whether a receipt contains source body text suitable for grounding."""
    return (
        isinstance(visit, dict)
        and visit.get("status") == "success"
        and visit.get("lane") == "grounding"
        and visit.get("view") in SUBSTANTIVE_VIEWS
    )


def new_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "lane_budgets": dict(LANE_BUDGETS),
        "retrieval_condition": None,
        "queries": [],
        "coverage_exemptions": [],
        "results": [],
        "selected_keys": [],
        "backend_calls": [],
        "visits": [],
        "backend_failures": [],
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
    queries: Any, coverage_exemptions: Any
) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    query_ids: set[str] = set()
    query_texts: set[str] = set()
    targeted_dimensions: set[str] = set()

    if not isinstance(queries, list):
        return ["retrieval manifest queries must be a list"], query_ids
    for index, query in enumerate(queries):
        where = f"retrieval queries[{index}]"
        if not isinstance(query, dict):
            errors.append(f"{where} must be an object")
            continue
        unknown = sorted(
            set(query) - {"id", "text", "lane", "target_dimension_ids", "evidence_roles"}
        )
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
        query_id = query.get("id")
        if not isinstance(query_id, str) or not re.fullmatch(r"q-\d{2,}", query_id):
            errors.append(f"{where}.id must match q-NN")
        elif query_id in query_ids:
            errors.append(f"duplicate retrieval query id {query_id}")
        else:
            query_ids.add(query_id)
        if query.get("lane") not in LANE_BUDGETS:
            errors.append(f"{where}.lane must be one of {sorted(LANE_BUDGETS)}")
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            errors.append(f"{where}.text must be non-empty")
        elif query["text"].strip().casefold() in query_texts:
            errors.append(f"{where}.text duplicates another query")
        else:
            query_texts.add(query["text"].strip().casefold())

        targets = query.get("target_dimension_ids")
        valid_targets = isinstance(targets, list) and all(
            isinstance(item, str) and _DIMENSION_RE.fullmatch(item) for item in targets
        )
        if not valid_targets:
            errors.append(f"{where}.target_dimension_ids must be a list of dim-* ids")
            targets = []
        elif len(targets) != len(set(targets)):
            errors.append(f"{where}.target_dimension_ids must not contain duplicates")
        else:
            targeted_dimensions.update(targets)

        roles = query.get("evidence_roles")
        valid_roles = (
            isinstance(roles, list)
            and bool(roles)
            and all(isinstance(role, str) and role in EVIDENCE_ROLES for role in roles)
        )
        if not valid_roles:
            errors.append(
                f"{where}.evidence_roles must be a non-empty list drawn from "
                f"{sorted(EVIDENCE_ROLES)}"
            )
            continue
        if len(roles) != len(set(roles)):
            errors.append(f"{where}.evidence_roles must not contain duplicates")
        if INNER_HPO_ROLE in roles:
            if roles != [INNER_HPO_ROLE]:
                errors.append(
                    f"{where} inner_hpo_prior must be the query's only evidence role"
                )
            if targets:
                errors.append(f"{where} inner_hpo_prior must not target semantic dimensions")
        elif not targets and set(roles) & DIMENSION_BOUND_ROLES:
            errors.append(
                f"{where} hypothesis/relation queries must target at least one dimension"
            )

    if not isinstance(coverage_exemptions, list):
        errors.append("retrieval manifest coverage_exemptions must be a list")
        return errors, query_ids
    exempted_dimensions: set[str] = set()
    for index, exemption in enumerate(coverage_exemptions):
        where = f"retrieval coverage_exemptions[{index}]"
        if not isinstance(exemption, dict):
            errors.append(f"{where} must be an object")
            continue
        unknown = sorted(set(exemption) - {"dimension_id", "rationale"})
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
        dimension_id = exemption.get("dimension_id")
        if not isinstance(dimension_id, str) or not _DIMENSION_RE.fullmatch(dimension_id):
            errors.append(f"{where}.dimension_id must be a dim-* id")
        elif dimension_id in exempted_dimensions:
            errors.append(f"duplicate retrieval coverage exemption {dimension_id}")
        else:
            exempted_dimensions.add(dimension_id)
        if not isinstance(exemption.get("rationale"), str) or not exemption["rationale"].strip():
            errors.append(f"{where}.rationale must be non-empty")
    overlap = sorted(targeted_dimensions & exempted_dimensions)
    if overlap:
        errors.append(f"retrieval coverage exemptions duplicate query targets: {overlap}")
    return errors, query_ids


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"retrieval manifest schema_version must be {SCHEMA_VERSION}")
    draft_revision = manifest.get("external_draft_revision")
    if draft_revision is not None and re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(draft_revision)
    ) is None:
        errors.append("retrieval manifest external_draft_revision must be a sha256 digest")
    budgets = manifest.get("lane_budgets")
    if not isinstance(budgets, dict):
        errors.append("retrieval manifest lane_budgets must be an object")
    else:
        for lane, default in LANE_BUDGETS.items():
            value = budgets.get(lane)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"retrieval manifest lane_budgets.{lane} must be positive")
            elif (
                lane == "grounding"
                and isinstance(budgets.get("novelty"), int)
                and value <= budgets["novelty"]
            ):
                errors.append("grounding token budget must exceed novelty token budget")

    plan_errors, query_ids = _validate_query_plan(
        manifest.get("queries"), manifest.get("coverage_exemptions")
    )
    errors.extend(plan_errors)
    condition = manifest.get("retrieval_condition")
    if query_ids and (
        not isinstance(condition, str)
        or condition not in {"frozen", "open_world", "mixed"}
    ):
        errors.append("retrieval_condition must describe a populated search")

    result_keys: set[str] = set()
    for index, result in enumerate(manifest.get("results", [])):
        where = f"retrieval results[{index}]"
        if not isinstance(result, dict):
            errors.append(f"{where} must be an object")
            continue
        key = result.get("canonical_key")
        if not isinstance(key, str) or not key:
            errors.append(f"{where}.canonical_key must be non-empty")
        elif key in result_keys:
            errors.append(f"duplicate merged retrieval result {key}")
        else:
            result_keys.add(key)
        result_query_ids = result.get("query_ids")
        if not isinstance(result_query_ids, list) or not set(result_query_ids) <= query_ids:
            errors.append(f"{where}.query_ids contains unknown queries")
        if result.get("query_support") != len(set(result_query_ids or [])):
            errors.append(f"{where}.query_support does not match distinct query_ids")

    selected = manifest.get("selected_keys")
    if not isinstance(selected, list) or not set(selected) <= result_keys:
        errors.append("retrieval selected_keys must reference merged results")

    called_queries: set[str] = set()
    for index, call in enumerate(manifest.get("backend_calls", [])):
        where = f"retrieval backend_calls[{index}]"
        if not isinstance(call, dict):
            errors.append(f"{where} must be an object")
            continue
        if call.get("query_id") not in query_ids:
            errors.append(f"{where}.query_id is unknown")
        else:
            called_queries.add(call["query_id"])
        if call.get("status") not in {"success", "failed"}:
            errors.append(f"{where}.status must be success or failed")
        if not isinstance(call.get("retrieved_at"), str) or not call["retrieved_at"]:
            errors.append(f"{where}.retrieved_at must be non-empty")
        if not isinstance(call.get("backend_version"), str) or not call["backend_version"]:
            errors.append(f"{where}.backend_version must be non-empty")
        if call.get("status") == "success":
            raw = call.get("raw_response")
            digest = hashlib.sha256(
                json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            if call.get("response_sha256") != digest:
                errors.append(f"{where}.response_sha256 does not match raw_response")
    missing_calls = sorted(query_ids - called_queries)
    if missing_calls:
        errors.append(f"retrieval queries have no backend call records: {missing_calls}")

    for index, visit in enumerate(manifest.get("visits", [])):
        where = f"retrieval visits[{index}]"
        if not isinstance(visit, dict):
            errors.append(f"{where} must be an object")
            continue
        if visit.get("lane") not in LANE_BUDGETS:
            errors.append(f"{where}.lane must be one of {sorted(LANE_BUDGETS)}")
        if visit.get("status") not in {"success", "failed"}:
            errors.append(f"{where}.status must be success or failed")
        if not isinstance(visit.get("backend_version"), str) or not visit["backend_version"]:
            errors.append(f"{where}.backend_version must be non-empty")
        url = visit.get("url")
        if not isinstance(url, str) or canonical_key(url) != visit.get("canonical_key"):
            errors.append(f"{where}.canonical_key does not match url")
        budget = visit.get("budget_tokens")
        expected_budget = (budgets or {}).get(visit.get("lane"))
        if budget != expected_budget:
            errors.append(f"{where}.budget_tokens does not match its lane")
        if visit.get("status") == "success" and (
            not isinstance(visit.get("content_chars"), int) or visit["content_chars"] <= 0
        ):
            errors.append(f"{where}.content_chars must be positive for a successful visit")
        if visit.get("status") == "success":
            content = visit.get("content")
            if not isinstance(content, str) or not content:
                errors.append(f"{where}.content must be retained for a successful visit")
            else:
                if visit.get("content_chars") != len(content):
                    errors.append(f"{where}.content_chars does not match retained content")
                digest = hashlib.sha256(content.encode()).hexdigest()
                if visit.get("content_sha256") != digest:
                    errors.append(f"{where}.content_sha256 does not match retained content")
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
        items = [
            {
                "url": item["url"],
                "title": item.get("title") or "No title",
                "snippet": item.get("abstract") or item.get("text") or "",
                "external_id": item.get("external_id") or item.get("arxiv_id"),
            }
            for _, item in scored[:max_results]
        ]
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
        rows = payload.get("result", []) if isinstance(payload, dict) else []
        output: list[dict[str, Any]] = []
        for row in rows[:max_results]:
            if not isinstance(row, dict):
                continue
            paper_id = row.get("arxiv_id")
            url = row.get("url") or (f"https://arxiv.org/abs/{paper_id}" if paper_id else "")
            if not url:
                continue
            output.append(
                {
                    "url": url,
                    "title": row.get("title") or "No title",
                    "snippet": row.get("abstract") or row.get("tldr") or "",
                    "external_id": paper_id,
                }
            )
        return {
            "items": output,
            "raw_response": payload,
            "metadata": {
                "client_license": "MIT",
                "invocation_surface": "deepxiv CLI",
            },
        }


class JinaSearchBackend(SearchBackend):
    name = "jina"
    version = "hosted-api-unknown"

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
            elif name == "jina":
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


def select_balanced(results: list[dict[str, Any]], query_ids: list[str]) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()

    def add(item: dict[str, Any]) -> bool:
        key = item["canonical_key"]
        if key in seen:
            return False
        seen.add(key)
        selected.append(key)
        return True

    for item in results:
        if item["query_support"] > 1:
            add(item)
        if len(selected) >= min(MAX_SHARED, MAX_SELECTED):
            return selected

    while len(selected) < MAX_SELECTED:
        progressed = False
        for query_id in query_ids:
            for item in results:
                if query_id in item["query_ids"] and add(item):
                    progressed = True
                    break
            if len(selected) >= MAX_SELECTED:
                return selected
        if not progressed:
            break
    return selected


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
        serialized = json.dumps(raw_response, sort_keys=True, ensure_ascii=False).encode()
        calls.append(
            {
                "query_id": query["id"],
                "backend": backend.name,
                "backend_version": backend.version,
                "status": "success",
                "retrieved_at": retrieved_at,
                "response_sha256": hashlib.sha256(serialized).hexdigest(),
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


def _jina_visit(url: str) -> tuple[str, str]:
    headers = {"Accept": "text/plain", "X-Return-Format": "markdown",
               "User-Agent": "HieraResearch/1"}
    if os.environ.get("JINA_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
    try:
        request = urllib.request.Request("https://r.jina.ai/" + url, headers=headers)
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            text = response.read().decode("utf-8", errors="replace")
        if text.strip():
            return text, "jina-reader"
    except (urllib.error.URLError, TimeoutError, ValueError):
        pass
    return _direct_visit(url), "direct"


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
    for result in manifest.get("results", []):
        if isinstance(result, dict) and result.get("canonical_key") == key:
            query_ids.update(
                item for item in result.get("query_ids", []) if isinstance(item, str)
            )
    texts: list[str] = []
    roles: set[str] = set()
    for query in manifest.get("queries", []):
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
    manifest: dict[str, Any], *, url: str, lane: str, backend: str, view: str,
    status: str, content: str | None = None, error: str | None = None,
    backend_version: str = "unknown", section: str | None = None,
) -> None:
    budgets = manifest.setdefault("lane_budgets", dict(LANE_BUDGETS))
    manifest.setdefault("visits", []).append(
        {
            "url": canonical_url(url),
            "canonical_key": canonical_key(url),
            "lane": lane,
            "backend": backend,
            "backend_version": backend_version,
            "view": view,
            "section": section,
            "status": status,
            "budget_tokens": budgets[lane],
            "content_chars": len(content or ""),
            "content_sha256": hashlib.sha256((content or "").encode()).hexdigest()
            if content
            else None,
            "content": content,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
    )


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(("https://", "http://")):
        return False
    try:
        return bool(urllib.parse.urlsplit(value).hostname)
    except ValueError:
        return False


def import_external_draft(draft: Any) -> dict[str, Any]:
    """Canonicalize a bounded WebSearch/WebFetch research draft.

    Runtime-native web tools cannot invoke this module while they are running.
    They therefore retain the semantic query fields, raw result rows, and exact
    fetched text in a deliberately small draft.  Python owns every mechanical
    receipt: canonical identities, hashes, timestamps, ranks, lane budgets, and
    balanced selection.
    """

    errors: list[str] = []
    if not isinstance(draft, dict):
        raise ValueError("external retrieval draft must be a JSON object")
    allowed_top = {
        "schema_version",
        "kind",
        "retrieval_condition",
        "queries",
        "coverage_exemptions",
        "visits",
        "backend_failures",
    }
    unknown_top = sorted(set(draft) - allowed_top)
    if unknown_top:
        errors.append(f"external retrieval draft has unknown fields {unknown_top}")
    if draft.get("schema_version") != EXTERNAL_DRAFT_SCHEMA_VERSION:
        errors.append(
            "external retrieval draft schema_version must be "
            f"{EXTERNAL_DRAFT_SCHEMA_VERSION}"
        )
    if draft.get("kind") != "external_retrieval_draft":
        errors.append("external retrieval draft kind must be external_retrieval_draft")
    if draft.get("retrieval_condition") != "open_world":
        errors.append("external retrieval draft must declare open_world retrieval")

    raw_queries = draft.get("queries")
    if not isinstance(raw_queries, list) or not raw_queries:
        errors.append("external retrieval draft queries must be a non-empty list")
        raw_queries = []
    queries: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    backend_calls: list[dict[str, Any]] = []
    recorded_at = datetime.now(timezone.utc).isoformat()
    query_fields = {
        "id",
        "text",
        "target_dimension_ids",
        "evidence_roles",
        "backend",
        "backend_version",
        "status",
        "results",
        "error",
    }
    result_fields = {"url", "title", "snippet", "external_id"}
    for index, raw_query in enumerate(raw_queries):
        where = f"external retrieval draft queries[{index}]"
        if not isinstance(raw_query, dict):
            errors.append(f"{where} must be an object")
            continue
        unknown = sorted(set(raw_query) - query_fields)
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
        query = {
            "id": raw_query.get("id"),
            "text": raw_query.get("text"),
            "lane": "grounding",
            "target_dimension_ids": raw_query.get("target_dimension_ids"),
            "evidence_roles": raw_query.get("evidence_roles"),
        }
        queries.append(query)
        backend = raw_query.get("backend")
        backend_version = raw_query.get("backend_version")
        status = raw_query.get("status")
        if not isinstance(backend, str) or not backend.strip():
            errors.append(f"{where}.backend must be non-empty")
        if not isinstance(backend_version, str) or not backend_version.strip():
            errors.append(f"{where}.backend_version must be non-empty")
        if status not in {"success", "failed"}:
            errors.append(f"{where}.status must be success or failed")
        rows = raw_query.get("results")
        if not isinstance(rows, list):
            errors.append(f"{where}.results must be a list")
            rows = []
        clean_rows: list[dict[str, Any]] = []
        for result_index, row in enumerate(rows):
            row_where = f"{where}.results[{result_index}]"
            if not isinstance(row, dict):
                errors.append(f"{row_where} must be an object")
                continue
            result_unknown = sorted(set(row) - result_fields)
            if result_unknown:
                errors.append(f"{row_where} has unknown fields {result_unknown}")
            url = row.get("url")
            title = row.get("title")
            snippet = row.get("snippet")
            if not _is_http_url(url):
                errors.append(f"{row_where}.url must be a valid non-empty URL")
                continue
            if not isinstance(title, str) or not title.strip():
                errors.append(f"{row_where}.title must be non-empty")
                continue
            if not isinstance(snippet, str):
                errors.append(f"{row_where}.snippet must be a string")
                continue
            clean = {
                "url": canonical_url(url),
                "title": title.strip(),
                "snippet": snippet,
            }
            external_id = row.get("external_id")
            if external_id is not None:
                if not isinstance(external_id, str) or not external_id.strip():
                    errors.append(f"{row_where}.external_id must be non-empty when present")
                else:
                    clean["external_id"] = external_id.strip()
            clean_rows.append(clean)
            candidates.append(
                {
                    **clean,
                    "query_id": query["id"],
                    "query": query["text"],
                    "backend": backend,
                    "rank": result_index + 1,
                }
            )
        if status == "failed" and clean_rows:
            errors.append(f"{where} failed query must not retain successful results")
        error = raw_query.get("error")
        if status == "failed" and (not isinstance(error, str) or not error.strip()):
            errors.append(f"{where}.error must explain a failed query")
        raw_response = {"results": clean_rows}
        call = {
            "query_id": query["id"],
            "backend": backend,
            "backend_version": backend_version,
            "status": status,
            "retrieved_at": recorded_at,
            "raw_response": raw_response if status == "success" else None,
            "response_sha256": (
                hashlib.sha256(
                    json.dumps(
                        raw_response, sort_keys=True, ensure_ascii=False
                    ).encode()
                ).hexdigest()
                if status == "success"
                else None
            ),
            "error": error if status == "failed" else None,
        }
        backend_calls.append(call)

    exemptions = draft.get("coverage_exemptions")
    if not isinstance(exemptions, list):
        errors.append("external retrieval draft coverage_exemptions must be a list")
        exemptions = []
    plan_errors, _ = _validate_query_plan(queries, exemptions)
    errors.extend(plan_errors)

    raw_visits = draft.get("visits")
    if not isinstance(raw_visits, list):
        errors.append("external retrieval draft visits must be a list")
        raw_visits = []
    visit_fields = {
        "url",
        "backend",
        "backend_version",
        "view",
        "section",
        "status",
        "content",
        "error",
    }
    prepared_visits: list[dict[str, Any]] = []
    retained_chars = 0
    for index, raw_visit in enumerate(raw_visits):
        where = f"external retrieval draft visits[{index}]"
        if not isinstance(raw_visit, dict):
            errors.append(f"{where} must be an object")
            continue
        unknown = sorted(set(raw_visit) - visit_fields)
        if unknown:
            errors.append(f"{where} has unknown fields {unknown}")
        url = raw_visit.get("url")
        backend = raw_visit.get("backend")
        backend_version = raw_visit.get("backend_version")
        view = raw_visit.get("view")
        section = raw_visit.get("section")
        status = raw_visit.get("status")
        content = raw_visit.get("content")
        error = raw_visit.get("error")
        if not _is_http_url(url):
            errors.append(f"{where}.url must be a valid non-empty URL")
        if not isinstance(backend, str) or not backend.strip():
            errors.append(f"{where}.backend must be non-empty")
        if not isinstance(backend_version, str) or not backend_version.strip():
            errors.append(f"{where}.backend_version must be non-empty")
        if view not in SUBSTANTIVE_VIEWS | {"brief", "head"}:
            errors.append(f"{where}.view must be a supported receipt view")
        if view == "section" and (not isinstance(section, str) or not section.strip()):
            errors.append(f"{where}.section must be non-empty for a section view")
        if view != "section" and section is not None:
            errors.append(f"{where}.section is only valid for a section view")
        if status not in {"success", "failed"}:
            errors.append(f"{where}.status must be success or failed")
        if status == "success" and (not isinstance(content, str) or not content.strip()):
            errors.append(f"{where}.content must retain fetched text for a successful visit")
        if status == "failed" and (not isinstance(error, str) or not error.strip()):
            errors.append(f"{where}.error must explain a failed visit")
        if isinstance(content, str) and status == "success":
            retained_chars += len(content)
        prepared_visits.append(
            {
                "url": url,
                "backend": backend,
                "backend_version": backend_version,
                "view": view,
                "section": section,
                "status": status,
                "content": content if status == "success" else None,
                "error": error if status == "failed" else None,
            }
        )
    if retained_chars > LANE_BUDGETS["grounding"] * 4:
        errors.append(
            "external retrieval draft retained content exceeds the grounding lane budget"
        )

    failures = draft.get("backend_failures")
    if not isinstance(failures, list) or any(not isinstance(item, dict) for item in failures):
        errors.append("external retrieval draft backend_failures must be a list of objects")
        failures = []
    if errors:
        raise ValueError("invalid external retrieval draft: " + "; ".join(errors))

    merged = merge_candidates(candidates)
    manifest = new_manifest()
    manifest.update(
        {
            "external_draft_revision": "sha256:"
            + hashlib.sha256(
                json.dumps(
                    draft,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "retrieval_condition": "open_world",
            "queries": queries,
            "coverage_exemptions": exemptions,
            "results": merged,
            "selected_keys": select_balanced(merged, [query["id"] for query in queries]),
            "backend_calls": backend_calls,
            "backend_failures": failures,
        }
    )
    for visit in prepared_visits:
        add_visit(manifest, lane="grounding", **visit)
    manifest_errors = validate_manifest(manifest)
    if manifest_errors:
        raise ValueError(
            "canonical external retrieval manifest is invalid: "
            + "; ".join(manifest_errors)
        )
    return manifest


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


def cmd_search(args: argparse.Namespace) -> int:
    specs = _parse_cli_objects(
        args.query_spec,
        label="--query-spec",
        fields={"text", "target_dimension_ids", "evidence_roles"},
    )
    exemptions = _parse_cli_objects(
        args.coverage_exemption or [],
        label="--coverage-exemption",
        fields={"dimension_id", "rationale"},
    )
    queries = [
        {
            "id": f"q-{index:02d}",
            "text": spec.get("text"),
            "lane": args.lane,
            "target_dimension_ids": spec.get("target_dimension_ids"),
            "evidence_roles": spec.get("evidence_roles"),
        }
        for index, spec in enumerate(specs, start=1)
    ]
    plan_errors, _ = _validate_query_plan(queries, exemptions)
    if plan_errors:
        print(json.dumps({"ok": False, "errors": plan_errors}, indent=2), file=sys.stderr)
        return 1

    if args.manifest.exists():
        existing = load_manifest(args.manifest)
        _reject_legacy_manifest(existing)
    manifest = new_manifest()
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
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "lane_budgets": dict(LANE_BUDGETS),
            "retrieval_condition": (
                "frozen"
                if set(names) == {"frozen"}
                else "open_world"
                if "frozen" not in names
                else "mixed"
            ),
            "queries": queries,
            "coverage_exemptions": exemptions,
            "results": results,
            "selected_keys": select_balanced(results, [q["id"] for q in queries]),
            "backend_calls": calls,
            "visits": [],
            "backend_failures": unavailable + failures,
        }
    )
    save_manifest(args.manifest, manifest)
    errors = validate_manifest(manifest)
    print(json.dumps({"ok": not errors, "selected": manifest["selected_keys"],
                      "failures": manifest["backend_failures"], "errors": errors}, indent=2))
    return 0 if results and not errors else 1


def cmd_visit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _reject_legacy_manifest(manifest)
    budget = manifest.get("lane_budgets", LANE_BUDGETS).get(args.lane, LANE_BUDGETS[args.lane])
    view = args.view
    if args.section and view != "section":
        print("visit failed: --section is only valid with --view section", file=sys.stderr)
        return 1
    try:
        if args.frozen_corpus:
            frozen = FrozenCorpusBackend(args.frozen_corpus)
            content = frozen.read(args.url)
            backend = "frozen"
            backend_version = frozen.version
            if view == "auto":
                view = "full_text"
        elif args.visit_backend == "jina":
            content, backend = _jina_visit(args.url)
            backend_version = "hosted-api-unknown" if backend == "jina-reader" else "stdlib"
            if view == "auto":
                view = "full_text"
        elif args.visit_backend == "direct":
            content, backend = _direct_visit(args.url), "direct"
            backend_version = "stdlib"
            if view == "auto":
                view = "full_text"
        elif arxiv_id(args.url) and view == "auto":
            attempts = _deepxiv_progressive_read(args.url, manifest)
            _retain_progressive_content(attempts, budget * 4)
            for attempt in attempts:
                add_visit(
                    manifest,
                    url=args.url,
                    lane=args.lane,
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
            for attempt in attempts:
                if attempt["status"] != "success":
                    continue
                label = attempt["view"]
                if attempt["section"]:
                    label += f": {attempt['section']}"
                rendered.append(f"## DeepXiv {label}\n\n{attempt['content']}")
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
            content, backend = _direct_visit(args.url), "direct"
            backend_version = "stdlib"
            if view == "auto":
                view = "full_text"
        content = content[: budget * 4]
        add_visit(manifest, url=args.url, lane=args.lane, backend=backend, view=view,
                  status="success", content=content, backend_version=backend_version,
                  section=args.section)
        save_manifest(args.manifest, manifest)
        print(content)
        return 0
    except Exception as exc:
        add_visit(manifest, url=args.url, lane=args.lane, backend="auto", view=view,
                  status="failed", error=f"{type(exc).__name__}: {exc}",
                  backend_version="unknown", section=args.section)
        save_manifest(args.manifest, manifest)
        print(f"visit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def cmd_record_visit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _reject_legacy_manifest(manifest)
    content = args.content_file.read_text(errors="replace") if args.content_file else None
    if args.status == "success" and not content:
        print(
            json.dumps(
                {"ok": False, "errors": ["successful external visits require --content-file"]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    add_visit(manifest, url=args.url, lane=args.lane, backend=args.backend,
              view=args.view, status=args.status, content=content, error=args.error,
              backend_version=args.backend_version, section=args.section)
    save_manifest(args.manifest, manifest)
    errors = validate_manifest(manifest)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def cmd_import_external(args: argparse.Namespace) -> int:
    draft = json.loads(args.draft.read_text(encoding="utf-8"))
    manifest = import_external_draft(draft)
    reused = False
    if args.manifest.is_file():
        existing = load_manifest(args.manifest)
        reused = (
            existing.get("external_draft_revision")
            == manifest["external_draft_revision"]
            and validate_manifest(existing) == []
        )
        if reused:
            manifest = existing
    if not reused:
        save_manifest(args.manifest, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "queries": len(manifest["queries"]),
                "results": len(manifest["results"]),
                "visits": len(manifest["visits"]),
                "reused": reused,
            },
            indent=2,
        )
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest)
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
    search.add_argument(
        "--coverage-exemption",
        action="append",
        help="JSON object with dimension_id and rationale",
    )
    search.add_argument("--backend", action="append", choices=["frozen", "deepxiv", "jina"])
    search.add_argument(
        "--frozen-corpus",
        type=Path,
        help="pinned local JSON corpus; implies the frozen backend when --backend is omitted",
    )
    search.add_argument("--lane", choices=sorted(LANE_BUDGETS), default="grounding")
    search.add_argument("--max-results", type=int, default=10)
    search.set_defaults(func=cmd_search)

    visit = sub.add_parser("visit", help="read a source and append a visit receipt")
    visit.add_argument("--manifest", type=Path, required=True)
    visit.add_argument("--url", required=True)
    visit.add_argument("--lane", choices=sorted(LANE_BUDGETS), default="grounding")
    visit.add_argument(
        "--view", choices=["auto", "brief", "head", "preview", "section", "full_text"],
        default="auto",
    )
    visit.add_argument("--section")
    visit.add_argument("--frozen-corpus", type=Path)
    visit.add_argument(
        "--visit-backend",
        choices=["auto", "direct", "jina"],
        default="auto",
        help="Jina is explicit and optional; auto uses DeepXiv for arXiv and direct fetch otherwise",
    )
    visit.set_defaults(func=cmd_visit)

    record = sub.add_parser(
        "record-visit", help="record a successful/failed visit performed by an external tool"
    )
    record.add_argument("--manifest", type=Path, required=True)
    record.add_argument("--url", required=True)
    record.add_argument("--lane", choices=sorted(LANE_BUDGETS), default="grounding")
    record.add_argument("--backend", required=True)
    record.add_argument("--backend-version", default="unknown")
    record.add_argument("--view", required=True)
    record.add_argument("--section")
    record.add_argument("--status", choices=["success", "failed"], required=True)
    record.add_argument("--content-file", type=Path)
    record.add_argument("--error")
    record.set_defaults(func=cmd_record_visit)

    external = sub.add_parser(
        "import-external",
        help="canonicalize a bounded runtime WebSearch/WebFetch draft",
    )
    external.add_argument("--draft", type=Path, required=True)
    external.add_argument("--manifest", type=Path, required=True)
    external.set_defaults(func=cmd_import_external)

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
