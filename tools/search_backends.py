#!/usr/bin/env python3
"""Local-first retrieval, optional backends, and auditable visit receipts.

Frozen-corpus search and every artifact-integrity check use only the standard
library. DeepXiv, Jina, and Claude-native WebSearch/WebFetch are explicit,
optional coverage adapters; successful native searches and external visits can
be retained with ``record-search`` and ``record-visit`` so the same manifest
integrity checks still apply.
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
SCHEMA_VERSION = 1
LANE_BUDGETS = {"novelty": 2048, "grounding": 6000}
MAX_SHARED = 6
MAX_SELECTED = 18
HTTP_TIMEOUT = 45
OFFLINE_ENV = "HIERA_RETRIEVAL_OFFLINE"
DISABLED_BACKENDS_ENV = "HIERA_RETRIEVAL_DISABLE_BACKENDS"

_ARXIV_RE = re.compile(
    r"(?:arxiv\.org|alphaxiv\.org)/(?:abs|pdf)/([a-z-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?",
    re.IGNORECASE,
)
_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_RE = re.compile(r"<[^>]+>")


def offline_mode() -> bool:
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def disabled_backends() -> set[str]:
    """Return explicitly disabled live adapters for controlled ablations."""
    return {
        name.strip().lower()
        for name in os.environ.get(DISABLED_BACKENDS_ENV, "").split(",")
        if name.strip()
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
        "lane_budgets": dict(LANE_BUDGETS),
        "retrieval_condition": None,
        "queries": [],
        "candidates": [],
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


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"retrieval manifest schema_version must be {SCHEMA_VERSION}")
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

    query_ids: set[str] = set()
    query_texts: set[str] = set()
    for index, query in enumerate(manifest.get("queries", [])):
        where = f"retrieval queries[{index}]"
        if not isinstance(query, dict):
            errors.append(f"{where} must be an object")
            continue
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
    condition = manifest.get("retrieval_condition")
    if query_ids and condition not in {"frozen", "open_world", "mixed"}:
        errors.append("retrieval_condition must describe a populated search")

    candidates = manifest.get("candidates", [])
    if not isinstance(candidates, list):
        errors.append("retrieval candidates must be a list")
        candidates = []
    query_text_by_id = {
        query.get("id"): query.get("text")
        for query in manifest.get("queries", [])
        if isinstance(query, dict)
    }
    for index, candidate in enumerate(candidates):
        where = f"retrieval candidates[{index}]"
        if not isinstance(candidate, dict):
            errors.append(f"{where} must be an object")
            continue
        if not isinstance(candidate.get("url"), str) or not canonical_key(candidate["url"]):
            errors.append(f"{where}.url must be a non-empty URL")
        query_id = candidate.get("query_id")
        if query_id not in query_ids:
            errors.append(f"{where}.query_id is unknown")
        elif candidate.get("query") != query_text_by_id.get(query_id):
            errors.append(f"{where}.query does not match its query_id")
        if not isinstance(candidate.get("backend"), str) or not candidate["backend"]:
            errors.append(f"{where}.backend must be non-empty")
        if not isinstance(candidate.get("rank"), int) or candidate["rank"] <= 0:
            errors.append(f"{where}.rank must be a positive integer")

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
        has_original_size = "original_content_chars" in visit
        has_truncated_flag = "content_truncated" in visit
        if has_original_size != has_truncated_flag:
            errors.append(
                f"{where} must record original_content_chars and content_truncated together"
            )
        elif has_original_size:
            original_size = visit.get("original_content_chars")
            retained_size = visit.get("content_chars")
            if not isinstance(original_size, int) or original_size < 0:
                errors.append(f"{where}.original_content_chars must be non-negative")
            elif isinstance(retained_size, int) and original_size < retained_size:
                errors.append(
                    f"{where}.original_content_chars cannot be smaller than content_chars"
                )
            if not isinstance(visit.get("content_truncated"), bool):
                errors.append(f"{where}.content_truncated must be boolean")
            elif isinstance(original_size, int) and isinstance(retained_size, int):
                if visit["content_truncated"] != (original_size > retained_size):
                    errors.append(
                        f"{where}.content_truncated does not match retained content size"
                    )
        if not isinstance(visit.get("retrieved_at"), str) or not visit["retrieved_at"]:
            errors.append(f"{where}.retrieved_at must be non-empty")
        if visit.get("view") == "section" and not (
            isinstance(visit.get("section"), str) and visit["section"].strip()
        ):
            errors.append(f"{where}.section must name the retained section")
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
        if self.name in disabled_backends():
            raise RuntimeError(f"DeepXiv is disabled by {DISABLED_BACKENDS_ENV}")
        executable = shutil.which("deepxiv")
        if executable:
            self.command = [executable]
            self.env = None
            self.source = "installed_cli"
            version = subprocess.run(
                self.command + ["--version"], text=True, capture_output=True,
                timeout=10, check=False,
            )
            self.version = (version.stdout or version.stderr).strip() or "installed-cli-unknown"
            return
        sibling = ROOT.parent / "deepxiv_sdk" / "deepxiv_sdk" / "cli.py"
        if not sibling.exists():
            raise RuntimeError("DeepXiv CLI and sibling checkout are unavailable")
        self.command = [sys.executable, "-m", "deepxiv_sdk.deepxiv_sdk.cli"]
        self.env = dict(os.environ)
        self.source = "sibling_checkout"
        old_path = self.env.get("PYTHONPATH")
        self.env["PYTHONPATH"] = str(ROOT.parent) + (os.pathsep + old_path if old_path else "")
        version_file = ROOT.parent / "deepxiv_sdk" / "deepxiv_sdk" / "__init__.py"
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)', version_file.read_text())
        self.version = match.group(1) if match else "sibling-unknown"

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

    def __init__(self) -> None:
        if self.name in disabled_backends():
            raise RuntimeError(f"Jina is disabled by {DISABLED_BACKENDS_ENV}")

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


def describe_backend(backend: SearchBackend) -> dict[str, Any]:
    """Return a secret-free, network-free backend availability description."""
    description: dict[str, Any] = {
        "backend": backend.name,
        "available": True,
        "backend_version": backend.version,
    }
    source = getattr(backend, "source", None)
    if source:
        description["source"] = source
    if isinstance(backend, FrozenCorpusBackend):
        description.update(
            {
                "source": "frozen_corpus",
                "corpus_path": str(backend.path),
                "corpus_sha256": backend.corpus_sha256,
            }
        )
    elif isinstance(backend, JinaSearchBackend):
        description["source"] = "hosted_service"
    return description


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


def expand_merged_results(
    results: list[dict[str, Any]], queries: dict[str, str]
) -> list[dict[str, Any]]:
    """Expand merged rows enough to preserve support when appending a search receipt."""
    candidates: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict) or not result.get("url"):
            continue
        query_ids = [qid for qid in result.get("query_ids", []) if qid in queries]
        backends = [str(name) for name in result.get("backends", []) if name]
        if not query_ids:
            continue
        if not backends:
            backends = ["unknown"]
        common = {
            "url": result["url"],
            "title": result.get("title") or "No title",
            "snippet": result.get("snippet") or "",
            "external_id": result.get("external_id"),
            "rank": result.get("best_rank", 9999),
        }
        for index, query_id in enumerate(query_ids):
            candidates.append(
                {
                    **common,
                    "query_id": query_id,
                    "query": queries[query_id],
                    "backend": backends[min(index, len(backends) - 1)],
                }
            )
        for backend in backends[len(query_ids) :]:
            candidates.append(
                {
                    **common,
                    "query_id": query_ids[0],
                    "query": queries[query_ids[0]],
                    "backend": backend,
                }
            )
    return candidates


def retained_candidates(
    manifest: dict[str, Any], queries: dict[str, str]
) -> list[dict[str, Any]]:
    """Return exact retained hits, migrating older result-only manifests."""
    candidates = manifest.get("candidates")
    if isinstance(candidates, list) and candidates:
        return [dict(item) for item in candidates if isinstance(item, dict)]
    return expand_merged_results(manifest.get("results", []), queries)


def resolve_queries(
    manifest: dict[str, Any], texts: list[str], lane: str
) -> list[dict[str, str]]:
    """Append new query identities and reuse exact queries on later dispatches."""
    queries = manifest.setdefault("queries", [])
    if not isinstance(queries, list):
        raise ValueError("retrieval manifest queries must be a list")
    by_text = {
        str(item.get("text", "")).strip().casefold(): item
        for item in queries
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    }
    existing_numbers = [
        int(match.group(1))
        for item in queries
        if isinstance(item, dict)
        and (match := re.fullmatch(r"q-(\d{2,})", str(item.get("id", ""))))
    ]
    next_number = max(existing_numbers, default=0) + 1
    resolved: list[dict[str, str]] = []
    dispatched_ids: set[str] = set()
    for raw_text in texts:
        text = raw_text.strip()
        if not text:
            raise ValueError("retrieval queries must be non-empty")
        folded = text.casefold()
        query = by_text.get(folded)
        if query is None:
            query = {"id": f"q-{next_number:02d}", "text": text, "lane": lane}
            next_number += 1
            queries.append(query)
            by_text[folded] = query
        elif query.get("lane") != lane:
            raise ValueError(
                f"query {query.get('id')} already belongs to lane {query.get('lane')}"
            )
        query_id = str(query["id"])
        if query_id not in dispatched_ids:
            resolved.append(
                {"id": query_id, "text": str(query["text"]), "lane": str(query["lane"])}
            )
            dispatched_ids.add(query_id)
    return resolved


def combined_condition(existing: Any, current: str) -> str:
    if existing not in {"frozen", "open_world", "mixed"}:
        return current
    if existing == current:
        return current
    return "mixed"


async def dispatch_search(
    queries: list[dict[str, str]], backends: list[SearchBackend], max_results: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    tasks: list[tuple[dict[str, str], SearchBackend, asyncio.Task]] = []
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
    if "jina" in disabled_backends():
        raise RuntimeError(f"Jina is disabled by {DISABLED_BACKENDS_ENV}")
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
    if "direct" in disabled_backends():
        raise RuntimeError(f"direct visiting is disabled by {DISABLED_BACKENDS_ENV}")
    request = urllib.request.Request(url, headers={"User-Agent": "HieraResearch/1"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        content_type = response.headers.get("Content-Type", "").lower()
        if "pdf" in content_type or url.lower().split("?", 1)[0].endswith(".pdf"):
            raise RuntimeError(
                "direct PDF extraction is unavailable; use retained frozen text or an explicit auditable reader"
            )
        raw = response.read().decode("utf-8", errors="replace")
    return _strip_html(raw)


def _deepxiv_section_content(raw: str) -> str:
    """Extract the section body before applying the visit retention budget."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("DeepXiv returned invalid JSON for a section") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("content"), str):
        raise RuntimeError("DeepXiv section response has no textual content")
    content = payload["content"]
    if not content.strip():
        raise RuntimeError("DeepXiv returned an empty section")
    return content


def _deepxiv_read(url: str, view: str, section: str | None) -> tuple[str, str, str]:
    paper_id = arxiv_id(url)
    if not paper_id:
        raise RuntimeError("URL is not an arXiv paper")
    backend = DeepXivBackend()
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
    content = process.stdout
    if view == "section":
        content = _deepxiv_section_content(content)
    return content, "deepxiv", backend.version


def _deepxiv_head_state(
    manifest: dict[str, Any], url: str, lane: str
) -> tuple[int, list[str]] | None:
    """Return the latest retained DeepXiv head and its exact section names."""
    key = canonical_key(url)
    visits = manifest.get("visits", [])
    for index in range(len(visits) - 1, -1, -1):
        visit = visits[index]
        if not (
            isinstance(visit, dict)
            and visit.get("canonical_key") == key
            and visit.get("lane") == lane
            and visit.get("backend") == "deepxiv"
            and visit.get("view") == "head"
            and visit.get("status") == "success"
        ):
            continue
        try:
            payload = json.loads(visit.get("content", ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("retained DeepXiv head is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("retained DeepXiv head must be a JSON object")
        raw_sections = payload.get("sections", [])
        if not isinstance(raw_sections, list):
            raise RuntimeError("retained DeepXiv head has an invalid section map")
        names: list[str] = []
        for section in raw_sections:
            name = section.get("name") if isinstance(section, dict) else section
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return index, names
    return None


def _check_deepxiv_progression(
    manifest: dict[str, Any], url: str, lane: str, view: str, section: str | None
) -> None:
    """Require auditable triage before expensive DeepXiv paper reads."""
    if view not in {"section", "full_text"}:
        return
    state = _deepxiv_head_state(manifest, url, lane)
    if state is None:
        raise RuntimeError(
            f"DeepXiv {view} requires a prior successful {lane}-lane head visit"
        )
    head_index, section_names = state
    if view == "section":
        requested = (section or "").strip()
        if requested not in section_names:
            available = ", ".join(section_names) or "none"
            raise RuntimeError(
                "--section must exactly match the retained DeepXiv head; "
                f"available sections: {available}"
            )
        return
    if not section_names:
        return
    key = canonical_key(url)
    failed_section = any(
        isinstance(visit, dict)
        and visit.get("canonical_key") == key
        and visit.get("lane") == lane
        and visit.get("view") == "section"
        and visit.get("status") == "failed"
        and visit.get("section") in section_names
        for visit in manifest.get("visits", [])[head_index + 1 :]
    )
    if not failed_section:
        raise RuntimeError(
            "DeepXiv full_text is a fallback: read a named section first, or use it "
            "after a recorded section failure"
        )


def add_visit(
    manifest: dict[str, Any], *, url: str, lane: str, backend: str, view: str,
    status: str, content: str | None = None, error: str | None = None,
    backend_version: str = "unknown", section: str | None = None,
    original_content_chars: int | None = None,
) -> None:
    budgets = manifest.setdefault("lane_budgets", dict(LANE_BUDGETS))
    retained_chars = len(content or "")
    original_chars = retained_chars if original_content_chars is None else original_content_chars
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
            "content_chars": retained_chars,
            "original_content_chars": original_chars,
            "content_truncated": original_chars > retained_chars,
            "content_sha256": hashlib.sha256((content or "").encode()).hexdigest()
            if content
            else None,
            "content": content,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "error": error,
        }
    )


def cmd_probe(args: argparse.Namespace) -> int:
    """Check adapter discovery without contacting a remote service."""
    if offline_mode() and args.backend != "frozen":
        payload = {
            "ok": False,
            "backend": args.backend,
            "available": False,
            "error": f"{OFFLINE_ENV}=1 permits only the frozen backend",
        }
        print(json.dumps(payload, indent=2))
        return 1
    backends, failures = build_backends([args.backend], args.frozen_corpus)
    if not backends:
        error = failures[0]["error"] if failures else "backend unavailable"
        print(
            json.dumps(
                {
                    "ok": False,
                    "backend": args.backend,
                    "available": False,
                    "error": error,
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps({"ok": True, **describe_backend(backends[0])}, indent=2))
    return 0


def _native_result_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("results", payload.get("items", payload.get("data", [])))
    else:
        rows = []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows if isinstance(rows, list) else [], start=1):
        if not isinstance(row, dict) or not isinstance(row.get("url"), str):
            continue
        url = row["url"].strip()
        if not url:
            continue
        rank = row.get("rank", index)
        normalized.append(
            {
                "url": url,
                "title": row.get("title") or row.get("name") or "No title",
                "snippet": row.get("snippet")
                or row.get("description")
                or str(row.get("content") or "")[:500],
                "external_id": row.get("external_id") or row.get("arxiv_id"),
                "rank": rank if isinstance(rank, int) and rank > 0 else index,
            }
        )
    return normalized


def cmd_record_search(args: argparse.Namespace) -> int:
    """Append a Claude-native search receipt and merge its result URLs."""
    if offline_mode():
        print(
            json.dumps(
                {"ok": False, "errors": [f"{OFFLINE_ENV}=1 rejects external search receipts"]},
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    manifest = load_manifest(args.manifest)
    existing_errors = validate_manifest(manifest)
    if existing_errors:
        print(json.dumps({"ok": False, "errors": existing_errors}, indent=2), file=sys.stderr)
        return 1
    query = resolve_queries(manifest, [args.query], args.lane)[0]
    queries = manifest["queries"]

    retrieved_at = datetime.now(timezone.utc).isoformat()
    call: dict[str, Any] = {
        "query_id": query["id"],
        "backend": args.backend,
        "backend_version": args.backend_version,
        "status": args.status,
        "retrieved_at": retrieved_at,
    }
    new_candidates: list[dict[str, Any]] = []
    if args.status == "success":
        if args.results_file is None:
            print(
                json.dumps(
                    {"ok": False, "errors": ["successful search requires --results-file"]},
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
        payload = json.loads(args.results_file.read_text())
        call["raw_response"] = payload
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        call["response_sha256"] = hashlib.sha256(serialized).hexdigest()
        call["metadata"] = {
            "invocation_surface": "Claude native WebSearch",
            "receipt_file": args.results_file.name,
        }
        for row in _native_result_rows(payload):
            new_candidates.append(
                {
                    **row,
                    "query_id": query["id"],
                    "query": query["text"],
                    "backend": args.backend,
                }
            )
    else:
        call["error"] = args.error or "native search failed without a supplied reason"
        manifest.setdefault("backend_failures", []).append(
            {
                "query_id": query["id"],
                "backend": args.backend,
                "error": call["error"],
            }
        )

    manifest.setdefault("backend_calls", []).append(call)
    query_text = {
        item["id"]: item["text"]
        for item in queries
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    old_candidates = retained_candidates(manifest, query_text)
    candidates = old_candidates + new_candidates
    results = merge_candidates(candidates)
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "lane_budgets": dict(LANE_BUDGETS),
            "retrieval_condition": combined_condition(
                manifest.get("retrieval_condition"), "open_world"
            ),
            "candidates": candidates,
            "results": results,
            "selected_keys": select_balanced(results, [item["id"] for item in queries]),
        }
    )
    errors = validate_manifest(manifest)
    if not errors:
        save_manifest(args.manifest, manifest)
    print(
        json.dumps(
            {
                "ok": not errors,
                "query_id": query["id"],
                "recorded_results": len(new_candidates),
                "selected": manifest["selected_keys"],
                "errors": errors,
            },
            indent=2,
        )
    )
    return 0 if args.status == "success" and new_candidates and not errors else 1


def cmd_search(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    existing_errors = validate_manifest(manifest)
    if existing_errors:
        print(json.dumps({"ok": False, "errors": existing_errors}, indent=2), file=sys.stderr)
        return 1
    names = args.backend or (["frozen"] if args.frozen_corpus else [])
    if offline_mode() and (set(names) != {"frozen"} or args.frozen_corpus is None):
        print(
            json.dumps(
                {
                    "ok": False,
                    "errors": [
                        f"{OFFLINE_ENV}=1 requires the frozen backend and --frozen-corpus"
                    ],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
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
    queries = resolve_queries(manifest, args.query, args.lane)
    backends, unavailable = build_backends(names, args.frozen_corpus)
    raw, failures, calls = asyncio.run(dispatch_search(queries, backends, args.max_results))
    unavailable_events: list[dict[str, str]] = []
    for failure in unavailable:
        for query in queries:
            event = {
                "query_id": query["id"],
                "backend": failure["backend"],
                "error": failure["error"],
            }
            unavailable_events.append(event)
            calls.append(
                {
                    **event,
                    "backend_version": "unavailable",
                    "status": "failed",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    query_text = {
        item["id"]: item["text"]
        for item in manifest["queries"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    old_candidates = retained_candidates(manifest, query_text)
    candidates = old_candidates + raw
    results = merge_candidates(candidates)
    current_condition = (
        "frozen"
        if set(names) == {"frozen"}
        else "open_world"
        if "frozen" not in names
        else "mixed"
    )
    manifest.update(
        {
            "schema_version": SCHEMA_VERSION,
            "lane_budgets": dict(LANE_BUDGETS),
            "retrieval_condition": combined_condition(
                manifest.get("retrieval_condition"), current_condition
            ),
            "candidates": candidates,
            "results": results,
            "selected_keys": select_balanced(
                results, [item["id"] for item in manifest["queries"]]
            ),
            "backend_calls": manifest.get("backend_calls", []) + calls,
            "backend_failures": manifest.get("backend_failures", [])
            + unavailable_events
            + failures,
        }
    )
    errors = validate_manifest(manifest)
    if not errors:
        save_manifest(args.manifest, manifest)
    print(json.dumps({"ok": not errors, "selected": manifest["selected_keys"],
                      "dispatch_results": len(raw),
                      "failures": unavailable_events + failures,
                      "errors": errors}, indent=2))
    return 0 if raw and not errors else 1


def cmd_visit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    existing_errors = validate_manifest(manifest)
    if existing_errors:
        print(json.dumps({"ok": False, "errors": existing_errors}, indent=2), file=sys.stderr)
        return 1
    if offline_mode() and args.frozen_corpus is None:
        print(
            f"visit failed: {OFFLINE_ENV}=1 requires --frozen-corpus",
            file=sys.stderr,
        )
        return 1
    if args.view == "section" and not (args.section and args.section.strip()):
        print("visit failed: --view section requires --section", file=sys.stderr)
        return 1
    if args.view != "section" and args.section:
        print("visit failed: --section is valid only with --view section", file=sys.stderr)
        return 1
    budget = manifest.get("lane_budgets", LANE_BUDGETS).get(args.lane, LANE_BUDGETS[args.lane])
    view = args.view
    attempted_backend = "auto"
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
            view = "head"
            attempted_backend = "deepxiv"
            content, backend, backend_version = _deepxiv_read(args.url, view, args.section)
        elif arxiv_id(args.url) and view in {"brief", "head", "preview", "section", "full_text"}:
            attempted_backend = "deepxiv"
            _check_deepxiv_progression(
                manifest, args.url, args.lane, view, args.section
            )
            content, backend, backend_version = _deepxiv_read(args.url, view, args.section)
        else:
            content, backend = _direct_visit(args.url), "direct"
            backend_version = "stdlib"
            if view == "auto":
                view = "full_text"
        if not content.strip():
            raise RuntimeError("reader returned empty content")
        original_content_chars = len(content)
        content = content[: budget * 4]
        add_visit(manifest, url=args.url, lane=args.lane, backend=backend, view=view,
                  status="success", content=content, backend_version=backend_version,
                  section=args.section, original_content_chars=original_content_chars)
        errors = validate_manifest(manifest)
        if errors:
            manifest["visits"].pop()
            raise RuntimeError("invalid visit receipt: " + "; ".join(errors))
        save_manifest(args.manifest, manifest)
        print(content)
        return 0
    except Exception as exc:
        add_visit(manifest, url=args.url, lane=args.lane, backend=attempted_backend, view=view,
                  status="failed", error=f"{type(exc).__name__}: {exc}",
                  backend_version="unknown", section=args.section)
        save_manifest(args.manifest, manifest)
        print(f"visit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def cmd_record_visit(args: argparse.Namespace) -> int:
    if offline_mode():
        print(
            json.dumps(
                {
                    "ok": False,
                    "errors": [f"{OFFLINE_ENV}=1 rejects externally recorded visits"],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    manifest = load_manifest(args.manifest)
    existing_errors = validate_manifest(manifest)
    if existing_errors:
        print(json.dumps({"ok": False, "errors": existing_errors}, indent=2), file=sys.stderr)
        return 1
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
    errors = validate_manifest(manifest)
    if not errors:
        save_manifest(args.manifest, manifest)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def cmd_validate(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    errors = validate_manifest(manifest)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser(
        "probe", help="check local adapter discovery without contacting a remote service"
    )
    probe.add_argument("--backend", required=True, choices=["frozen", "deepxiv", "jina"])
    probe.add_argument("--frozen-corpus", type=Path)
    probe.set_defaults(func=cmd_probe)

    search = sub.add_parser("search", help="fan queries across usable backends")
    search.add_argument("--manifest", type=Path, required=True)
    search.add_argument("--query", action="append", required=True)
    search.add_argument("--backend", action="append", choices=["frozen", "deepxiv", "jina"])
    search.add_argument(
        "--frozen-corpus",
        type=Path,
        help="pinned local JSON corpus; implies the frozen backend when --backend is omitted",
    )
    search.add_argument("--lane", choices=sorted(LANE_BUDGETS), default="grounding")
    search.add_argument("--max-results", type=int, default=10)
    search.set_defaults(func=cmd_search)

    record_search = sub.add_parser(
        "record-search", help="record and merge results returned by Claude WebSearch"
    )
    record_search.add_argument("--manifest", type=Path, required=True)
    record_search.add_argument("--query", required=True)
    record_search.add_argument("--lane", choices=sorted(LANE_BUDGETS), default="grounding")
    record_search.add_argument("--backend", default="claude-websearch")
    record_search.add_argument("--backend-version", default="hosted-provider-unknown")
    record_search.add_argument("--status", choices=["success", "failed"], default="success")
    record_search.add_argument("--results-file", type=Path)
    record_search.add_argument("--error")
    record_search.set_defaults(func=cmd_record_search)

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
