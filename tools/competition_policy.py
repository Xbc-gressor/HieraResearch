#!/usr/bin/env python3
"""Competition-policy compliance gates for retrieval (deterministic, stdlib only).

MLE-bench bans looking at other participants' solutions to the same
competition.  These gates reject competition-specific solution information
before it enters the model's context while letting generic methodology
research through: every gate requires a competition-identity hit conjoined
with a solution/participant signal, so identity alone or intent alone never
blocks.  A task identity profile is derived mechanically from the competition
slug and task name — no per-competition hand-maintained tables.
"""

from __future__ import annotations

import json
import re
import tomllib
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

COMPETITION_POLICY_VERSION = 1
POLICY_EXIT_CODE = 2
PROFILE_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent

# Window (~chars) between an identity hit and a high-confidence phrase.
PROXIMITY_CHARS = 400
# Window for qualifying "leaderboard"/"grandmaster"/"public leaderboard".
SIGNAL_QUALIFY_CHARS = 80

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\d{4}")

# Fixed global generic-word list: problem-type / modality / framing words that
# never identify a specific competition.
GENERIC_WORDS = frozenset({
    # stopword glue
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "the", "to", "via", "with",
    # benchmark / competition framing
    "mle", "kaggle", "competition", "competitions", "contest", "challenge",
    "benchmark", "task", "prize", "track",
    # problem-type words
    "author", "identification", "classifier", "classification", "prediction",
    "predict", "predicting", "detection", "detecting", "segmentation",
    "recognition", "estimation", "regression", "ranking", "recommendation",
    "retrieval", "generation", "translation", "summarization", "clustering",
    "forecasting", "modeling", "modelling", "analysis", "extraction",
    "denoising", "understanding", "matching", "scoring", "completion",
    # modality words
    "image", "images", "text", "texts", "document", "documents", "audio",
    "video", "videos", "signal", "signals", "speech", "language", "data",
    "dataset", "datasets", "sentence", "sentences", "series", "time",
    # ml-generic
    "machine", "learning", "deep", "neural", "model", "models", "ai", "ml",
})

# Modest common-English set for the acronym/coinage proxy.  Over-blocking is
# acceptable: every gate also requires an intent/signal conjunction.
COMMON_ENGLISH_WORDS = frozenset({
    "about", "after", "again", "all", "also", "any", "because", "been",
    "before", "being", "between", "both", "but", "can", "could", "day",
    "did", "does", "down", "each", "even", "few", "first", "get", "give",
    "good", "great", "has", "have", "her", "here", "him", "his", "how",
    "its", "just", "know", "last", "like", "long", "made", "make", "many",
    "may", "more", "most", "much", "must", "new", "not", "now", "off",
    "old", "once", "one", "only", "other", "our", "out", "over", "own",
    "part", "people", "place", "put", "said", "same", "see", "she",
    "should", "small", "some", "state", "still", "such", "take", "than",
    "that", "their", "them", "then", "there", "these", "they", "thing",
    "think", "this", "those", "through", "two", "under", "until", "upon",
    "use", "used", "very", "want", "was", "way", "well", "were", "what",
    "when", "where", "which", "while", "who", "will", "work", "world",
    "would", "year", "you", "your",
    # ordinary words appearing in known competition slugs
    "aerial", "bird", "birds", "cactus", "commentary", "conductor",
    "conductors", "dirty", "google", "iceberg", "insults", "quest",
    "social", "spaceship", "spooky", "titanic", "transparent",
})

SOLUTION_INTENT = (
    "solution", "solutions", "winning", "winner", "winners", "won", "top",
    "leaderboard", "kernel", "kernels", "notebook", "notebooks", "score",
    "scores", "champion", "first place", "first-place", "1st", "gold medal",
    "grandmaster", "prize", "ranking", "standings",
)
PARTICIPANT_INTENT = (
    "team", "teams", "entrant", "entrants", "participant", "participants",
    "repo", "repos", "repository", "repositories", "source code",
    "postmortem", "competition summary", "writeup", "write-up",
    "my approach", "our approach", "submission", "submissions",
)

# Simple substring phrases; "public leaderboard" and "grandmaster" are
# qualified below (rank/score or competition context nearby), and the
# rank/score regexes carry patterns like "rank 1" / "1st of" / "top 1%" /
# "0.954 private".  Bare leaderboard/kernel/notebook are generic words and
# never trigger a block by themselves.
HIGH_CONFIDENCE_SOLUTION_PHRASES = (
    "winning solution", "winning approach", "winning entry",
    "first-place", "first place", "1st place",
    "top solution", "top team", "top teams", "top finisher",
    "private leaderboard", "private score",
    "public leaderboard",
    "competition postmortem", "post-competition",
    "winner's", "winners'", "teams used", "team used",
    "gold medal",
    "grandmaster",
)

RANK_SCORE_PATTERNS = (
    re.compile(r"\brank\s*#?\d+", re.IGNORECASE),
    re.compile(r"\b\d+(?:st|nd|rd|th)\s+of\b", re.IGNORECASE),
    re.compile(r"\btop\s*\d+(?:\.\d+)?\s*%", re.IGNORECASE),
    re.compile(r"\b\d+\.\d+\s+(?:private|public)\b", re.IGNORECASE),
)
_RANK_SCORE_NEAR_RE = re.compile(
    r"\d|%|\brank\b|\bplace\b|\bscore\b|\bposition\b", re.IGNORECASE
)
# Bare years (the identity itself often carries one) are not rank/score semantics.
_YEAR_TOKEN_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_COMPETITION_CONTEXT_RE = re.compile(r"kaggle|competition", re.IGNORECASE)
_QUALIFIED_PHRASES = {
    "public leaderboard": _RANK_SCORE_NEAR_RE,
    "grandmaster": _COMPETITION_CONTEXT_RE,
}

RULE_SLUG_SEQUENCE = "slug_sequence"
RULE_PROPER_TOKEN = "proper_token"
RULE_DISCRIMINATIVE = "discriminative_tokens"
RULE_KAGGLE_URL = "kaggle_competition_url"
RULE_INHERITED = "inherited_identity"
_IDENTITY_RULE_ORDER = (RULE_SLUG_SEQUENCE, RULE_PROPER_TOKEN, RULE_DISCRIMINATIVE)


def _tokens(text: str) -> list[str]:
    return [m.group().lower() for m in _TOKEN_RE.finditer(str(text or ""))]


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [
        (m.group().lower(), m.start(), m.end())
        for m in _TOKEN_RE.finditer(str(text or ""))
    ]


def _phrase_spans(text: str, phrase: str) -> list[tuple[int, int]]:
    pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def derive_profile(task_name: str, competition_id: str | None) -> dict | None:
    """Mechanical identity profile from the competition slug + task name."""
    if not competition_id or not str(competition_id).strip():
        return None
    competition_id = str(competition_id).strip()
    slug_tokens = _tokens(competition_id)
    name_tokens = _tokens(task_name)
    combined = list(dict.fromkeys(slug_tokens + name_tokens))
    identity_tokens = [t for t in combined if t not in GENERIC_WORDS]
    discriminative = [t for t in identity_tokens if not _YEAR_RE.fullmatch(t)]
    proper = [
        t for t in discriminative
        if len(t) >= 4 and not t.isdigit() and t not in COMMON_ENGLISH_WORDS
    ]
    return {
        "profile_version": PROFILE_VERSION,
        "competition_policy_version": COMPETITION_POLICY_VERSION,
        "competition_id": competition_id,
        "task_name": task_name,
        "display_name": " ".join(slug_tokens).title(),
        "slug_tokens": slug_tokens,
        "identity_tokens": identity_tokens,
        "discriminative_tokens": discriminative,
        "proper_tokens": proper,
        "kaggle_competition_path": f"kaggle.com/competitions/{competition_id}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@dataclass(frozen=True)
class ProfileResolution:
    profile: dict | None
    fail_closed_reason: str | None


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _coerce_profile(data: object) -> dict | None:
    if not isinstance(data, dict):
        return None
    competition_id = data.get("competition_id")
    if not isinstance(competition_id, str) or not competition_id.strip():
        return None
    for field in ("slug_tokens", "identity_tokens", "discriminative_tokens",
                  "proper_tokens"):
        if not isinstance(data.get(field), list):
            return None
    return data


def _task_competition_id(task_name: str) -> str | None:
    try:
        with (REPO_ROOT / "tasks" / task_name / "task.toml").open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    competition_id = (data.get("mlebench") or {}).get("competition_id")
    if isinstance(competition_id, str) and competition_id.strip():
        return competition_id.strip()
    return None


def resolve_profile(manifest_path: Path) -> ProfileResolution:
    """Resolve the task identity profile for a retrieval manifest.

    Order: task_identity_profile.json beside the manifest; run_metadata.json
    (competition_id + embedded profile); path inference from a
    runs/<task>/<tag> layout against tasks/<task>/task.toml.  Fail closed only
    when a competition_id is known but no parseable profile was found;
    non-MLE and uninferable ad-hoc paths resolve to profile=None.
    """
    manifest_dir = Path(manifest_path).parent
    profile = _coerce_profile(_read_json(manifest_dir / "task_identity_profile.json"))
    if profile is not None:
        return ProfileResolution(profile, None)
    competition_id = None
    metadata = _read_json(manifest_dir / "run_metadata.json")
    if isinstance(metadata, dict):
        embedded = _coerce_profile(metadata.get("task_identity_profile"))
        if embedded is not None:
            return ProfileResolution(embedded, None)
        candidate = metadata.get("competition_id")
        if isinstance(candidate, str) and candidate.strip():
            competition_id = candidate.strip()
    if competition_id is None:
        run_dir = manifest_dir
        task_dir = run_dir.parent
        if task_dir.parent.name == "runs":
            competition_id = _task_competition_id(task_dir.name)
    if competition_id:
        return ProfileResolution(
            None,
            f"competition {competition_id!r} has no parseable task identity "
            f"profile beside {Path(manifest_path).name}",
        )
    return ProfileResolution(None, None)


def _identity_hits(text: str, profile: dict | None) -> list[dict]:
    """Every identity occurrence in text: rule label + char span."""
    if not profile:
        return []
    tokens = _token_spans(text)
    if not tokens:
        return []
    words = [word for word, _, _ in tokens]
    hits: list[dict] = []
    slug = [str(t) for t in profile.get("slug_tokens") or []]
    if slug and len(words) >= len(slug):
        window = len(slug) + 2  # subsequence match tolerates inserted stopwords
        for start in range(0, len(words) - len(slug) + 1):
            pos = 0
            last = start
            for index in range(start, min(start + window, len(words))):
                if pos < len(slug) and words[index] == slug[pos]:
                    pos += 1
                    last = index
            if pos == len(slug):
                hits.append({
                    "rule": RULE_SLUG_SEQUENCE,
                    "start": tokens[start][1],
                    "end": tokens[last][2],
                })
    proper = {str(t) for t in profile.get("proper_tokens") or []}
    for word, start, end in tokens:
        if word in proper:
            hits.append({"rule": RULE_PROPER_TOKEN, "token": word,
                         "start": start, "end": end})
    discriminative = {
        str(t) for t in profile.get("discriminative_tokens") or []
        if not _YEAR_RE.fullmatch(str(t))
    }
    if len(discriminative) >= 2:
        matched = [
            (word, start, end) for word, start, end in tokens
            if word in discriminative
        ]
        if len({word for word, _, _ in matched}) >= 2:
            for word, start, end in matched:
                hits.append({"rule": RULE_DISCRIMINATIVE, "token": word,
                             "start": start, "end": end})
    return hits


def identity_hit(text: str, profile: dict | None) -> tuple[bool, str]:
    """Whether text identifies the competition; returns the matched rule."""
    hits = _identity_hits(text, profile)
    if not hits:
        return False, ""
    for rule in _IDENTITY_RULE_ORDER:
        if any(hit["rule"] == rule for hit in hits):
            return True, rule
    return False, ""


def is_kaggle_competition_url(url: str, profile: dict | None) -> bool:
    """kaggle.com URL whose path addresses this competition; identity by itself."""
    if not profile:
        return False
    competition_id = str(profile.get("competition_id") or "").strip().lower()
    raw = str(url or "").strip()
    if not competition_id or not raw:
        return False
    if "://" not in raw:
        raw = "https://" + raw
    try:
        parts = urllib.parse.urlsplit(raw)
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host != "kaggle.com":
        return False
    path = (parts.path or "").lower()
    if f"/competitions/{competition_id}" in path:
        return True
    return competition_id in [segment for segment in path.split("/") if segment]


def _intent_hits(text: str) -> list[dict]:
    text = str(text or "")
    spans = _token_spans(text)
    words = {word for word, _, _ in spans}
    hits: list[dict] = []
    for lexicon, label in (
        (SOLUTION_INTENT, "solution_intent"),
        (PARTICIPANT_INTENT, "participant_intent"),
    ):
        for term in lexicon:
            if " " in term or "-" in term:
                term_spans = _phrase_spans(text, term)
            elif term in words:
                term_spans = [(s, e) for w, s, e in spans if w == term]
            else:
                term_spans = []
            for start, end in term_spans:
                hits.append({"label": f"{label}:{term}", "start": start, "end": end})
    return hits


def _qualified_phrase_hits(
    text: str, phrase: str, qualifier: re.Pattern, label: str
) -> list[dict]:
    hits: list[dict] = []
    for start, end in _phrase_spans(text, phrase):
        before = _YEAR_TOKEN_RE.sub("", text[max(0, start - SIGNAL_QUALIFY_CHARS):start])
        after = _YEAR_TOKEN_RE.sub("", text[end:end + SIGNAL_QUALIFY_CHARS])
        if qualifier.search(before) or qualifier.search(after):
            hits.append({"label": label, "start": start, "end": end})
    return hits


def _high_confidence_hits(text: str, *, include_leaderboard: bool = False) -> list[dict]:
    text = str(text or "")
    hits: list[dict] = []
    for phrase in HIGH_CONFIDENCE_SOLUTION_PHRASES:
        qualifier = _QUALIFIED_PHRASES.get(phrase)
        if qualifier is not None:
            hits.extend(_qualified_phrase_hits(
                text, phrase, qualifier, f"phrase:{phrase} (qualified)"))
            continue
        for start, end in _phrase_spans(text, phrase):
            hits.append({"label": f"phrase:{phrase}", "start": start, "end": end})
    for pattern in RANK_SCORE_PATTERNS:
        for match in pattern.finditer(text):
            hits.append({
                "label": f"rank_score:{match.group().strip().lower()}",
                "start": match.start(),
                "end": match.end(),
            })
    if include_leaderboard:
        hits.extend(_qualified_phrase_hits(
            text, "leaderboard", _RANK_SCORE_NEAR_RE,
            "phrase:leaderboard (rank/score-adjacent)"))
    return hits


def _span_gap(a: dict, b: dict) -> int:
    return max(0, max(a["start"] - b["end"], b["start"] - a["end"]))


def _context_around(text: str, start: int, end: int, limit: int) -> str:
    text = str(text or "")
    pad = max(limit - (end - start), 0)
    lo = max(0, start - pad // 2)
    hi = min(len(text), end + pad - pad // 2)
    snippet = re.sub(r"\s+", " ", text[lo:hi]).strip()
    if len(snippet) > limit:
        snippet = snippet[: limit - 1].rstrip() + "…"
    return snippet


@dataclass(frozen=True)
class Verdict:
    action: str  # "allow" | "block"
    categories: tuple[str, ...] = ()
    basis: str | None = None  # "direct" | "inherited"
    context: str | None = None


_ALLOW = Verdict("allow")


def _identity_rule(hits: list[dict]) -> str:
    for rule in _IDENTITY_RULE_ORDER:
        if any(hit["rule"] == rule for hit in hits):
            return rule
    return ""


def _labels(hits: list[dict], limit: int = 4) -> tuple[str, ...]:
    return tuple(dict.fromkeys(hit["label"] for hit in hits))[:limit]


def classify_query(text: str, profile: dict | None) -> Verdict:
    """Block iff the query both identifies the competition and shows
    solution/participant intent.  Identity alone or intent alone allows."""
    if not profile:
        return _ALLOW
    hit, rule = identity_hit(text, profile)
    if not hit:
        return _ALLOW
    intents = _intent_hits(text)
    if not intents:
        return _ALLOW
    categories = (rule, *_labels(intents))
    return Verdict(
        "block", categories, "direct",
        _context_around(text, intents[0]["start"], intents[0]["end"], 200),
    )


def _fields_text(fields: dict) -> str:
    authors = fields.get("authors")
    if isinstance(authors, (list, tuple)):
        authors = " ".join(str(a) for a in authors)
    return " ".join(
        str(part)
        for part in (
            fields.get("url"), fields.get("title"), authors,
            fields.get("snippet"), fields.get("tldr"),
        )
        if part
    )


def classify_result(
    fields: dict, inherited_identity: bool, profile: dict | None
) -> Verdict:
    """Gate one backend row (url/title/authors/snippet/tldr)."""
    if not profile:
        return _ALLOW
    fields = fields or {}
    url = str(fields.get("url") or "")
    if is_kaggle_competition_url(url, profile):
        return Verdict("block", (RULE_KAGGLE_URL,), "direct",
                       _context_around(url, 0, len(url), 200))
    text = _fields_text(fields)
    hits = _identity_hits(text, profile)
    if hits:
        signals = _intent_hits(text) + _high_confidence_hits(text)
        if signals:
            categories = (_identity_rule(hits), *_labels(signals))
            return Verdict(
                "block", categories, "direct",
                _context_around(text, signals[0]["start"], signals[0]["end"], 200),
            )
    if inherited_identity:
        confident = _high_confidence_hits(text)
        if confident:
            categories = (RULE_INHERITED, *_labels(confident))
            return Verdict(
                "block", categories, "inherited",
                _context_around(text, confident[0]["start"], confident[0]["end"], 200),
            )
    return _ALLOW


def classify_body(
    url: str, text: str, inherited_identity: bool, profile: dict | None
) -> Verdict:
    """Gate a fetched page body.  Never blocks on generic words alone and
    never denies unclassifiable pages by default."""
    if not profile:
        return _ALLOW
    if is_kaggle_competition_url(url, profile):
        url = str(url or "")
        return Verdict("block", (RULE_KAGGLE_URL,), "direct",
                       _context_around(url, 0, len(url), 200))
    text = str(text or "")
    hits = _identity_hits(text, profile)
    confident = _high_confidence_hits(text)
    if hits:
        near = [
            phrase for phrase in confident
            if any(_span_gap(phrase, hit) <= PROXIMITY_CHARS for hit in hits)
        ]
        if near:
            categories = (_identity_rule(hits), *_labels(near))
            return Verdict(
                "block", categories, "direct",
                _context_around(text, near[0]["start"], near[0]["end"], 200),
            )
    elif inherited_identity and confident:
        categories = (RULE_INHERITED, *_labels(confident))
        return Verdict(
            "block", categories, "inherited",
            _context_around(text, confident[0]["start"], confident[0]["end"], 200),
        )
    return _ALLOW


def scan_artifact_text(text: str, profile: dict | None) -> list[dict]:
    """Product-side scan (e.g. background.md): identity hit conjoined with a
    high-confidence solution phrase within ~400 chars.  Bare "leaderboard"
    counts only when rank/score semantics sit within ~80 chars."""
    if not profile or not text:
        return []
    text = str(text)
    hits = _identity_hits(text, profile)
    if not hits:
        return []
    rule = _identity_rule(hits)
    found: list[dict] = []
    for signal in _high_confidence_hits(text, include_leaderboard=True):
        near = [hit for hit in hits if _span_gap(signal, hit) <= PROXIMITY_CHARS]
        if not near:
            continue
        found.append({
            "categories": [rule, signal["label"]],
            "context": _context_around(
                text,
                min(near[0]["start"], signal["start"]),
                max(near[0]["end"], signal["end"]),
                160,
            ),
        })
        if len(found) >= 50:
            break
    return found
