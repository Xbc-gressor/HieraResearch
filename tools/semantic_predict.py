"""semantic_predict.py — PREDICT 层 / pre-execution candidate judgment.

Between SELECT (which points are eligible) and EXECUTE (which costs objective
budget), rank a shortlist of concrete candidate sketches and run only the
winner. Every call here is inference-only: it spends no objective slot.

Three deterministic pieces; the LLM sits only in the middle, as a pairwise
judge:

- ``precedents``: focused, outcome-labeled retrieval over prior ledger records.
  One line per precedent — the change text and a binary worked/did-not-work
  label, nothing else.
- ``pairs``: the ordered comparison schedule for a shortlist. Every unordered
  pair appears in both presentation orders.
- ``tally``: strict-consensus vote counting over the returned verdicts. Both
  orders must agree for a pair to produce a vote; a disagreement abstains.

Design evidence (Rehearse, arXiv:2607.27687, on the same nanochat loop this
repo benchmarks against):

- Focused outcome-labeled retrieval beat a generated history summary (83.5% vs
  80.9% late-run selective accuracy) and a full-history dump (70.8%). Adding
  the recorded failure *reason* to a retrieved line LOWERED accuracy to 80.6%;
  a curated reason lowered it further to 78.7%. So a precedent line carries the
  change and the label and nothing else — this is a deliberate omission, not an
  unfinished field.
- Keeping only worked precedents scored 78.9%; only did-not-work, 76.1%. Both
  labels are required.
- Pairwise strict consensus, rather than argmax over absolute rubric scores:
  absolute judge scores can look well-calibrated while best-of-N selection
  fails, and two-order agreement suppresses position bias.
- A no-LLM rule that steers *away* from prior did-not-work attempts scored
  47.8% — below chance — because the right move is frequently a near neighbor
  of a failed one (raising a parameter whose lowering just failed). Retrieval
  here therefore surfaces neighbors for the judge to read; it never scores a
  candidate by distance from failure. Do not reintroduce that shortcut.

Retrieval uses IDF-weighted token cosine over the ledger's own change/idea text
rather than a sentence encoder: this repo declares zero dependencies, and the
source reports the mechanism is flat across similarity thresholds 0.40-0.55 and
was demonstrated with a deliberately small encoder. ``--threshold`` and the
pluggable ``similarity`` seam below are where a stronger retriever would land.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# Fixed before any benchmark measurement, from the source's reported sweep over
# {0.30 ... 0.60}; below 0.40 pulls in loosely related attempts, above it
# coverage thins. Selective accuracy was flat across 0.40-0.55.
DEFAULT_THRESHOLD = 0.40
DEFAULT_SHORTLIST = 3
DEFAULT_MAX_PRECEDENTS = 8

# Outcome labels are binary by contract. `keep` is the only worked outcome; a
# discard and a crash are both "did not work" — the judge is choosing what to
# run next, and for that decision a crash and a regression carry the same
# lesson. Pending/unevaluated records have no outcome and are never retrieved.
WORKED = "worked"
DID_NOT_WORK = "did not work"
_OUTCOME_BY_STATUS = {
    "keep": WORKED,
    "discard": DID_NOT_WORK,
    "crash": DID_NOT_WORK,
}

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._+-]*")
# Deliberately small: this filters grammatical filler only. Domain words stay,
# including the numbers and parameter names that make two changes comparable.
_STOPWORDS = frozenset(
    """a an and are as at be by for from in into is it its of on or that the
    to with without use using via at least most more less than then this these
    those which while when where over under between""".split()
)


class PredictError(RuntimeError):
    """Contract violation in the predict layer."""


# --------------------------------------------------------------- run config
_CFG_DEFAULTS = {
    "enabled": True,
    "shortlist_size": DEFAULT_SHORTLIST,
    "similarity_threshold": DEFAULT_THRESHOLD,
    "max_precedents": DEFAULT_MAX_PRECEDENTS,
}


def predict_cfg(ref_path: Any) -> dict[str, Any]:
    """The `predict` section of the nearest framework_cfg.json, with defaults.

    Resolution order is CLI flag > run config > these defaults, matching every
    other deterministic helper. A run that never writes the section gets the
    defaults and the layer stays on.
    """
    cfg = dict(_CFG_DEFAULTS)
    if ref_path is None:
        return cfg
    try:
        from run_cfg import load_run_cfg
    except ImportError:  # pragma: no cover - tools/ always on sys.path in-run
        return cfg
    section = load_run_cfg(ref_path, "predict")
    for key in _CFG_DEFAULTS:
        if key in section:
            cfg[key] = section[key]
    if not isinstance(cfg["enabled"], bool):
        raise PredictError("predict.enabled must be true or false")
    size = cfg["shortlist_size"]
    if not isinstance(size, int) or isinstance(size, bool) or not 2 <= size <= 8:
        raise PredictError("predict.shortlist_size must be an integer in [2, 8]")
    threshold = cfg["similarity_threshold"]
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        raise PredictError("predict.similarity_threshold must be a number in [0, 1]")
    limit = cfg["max_precedents"]
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise PredictError("predict.max_precedents must be a positive integer")
    return cfg


# ---------------------------------------------------------------- similarity
def tokenize(text: str) -> Counter:
    """Bag of content tokens. Numbers and parameter names are content."""
    if not isinstance(text, str):
        return Counter()
    return Counter(
        tok
        for tok in _TOKEN.findall(text.lower())
        if tok not in _STOPWORDS and len(tok) > 1
    )


def _idf(corpus: list[Counter]) -> dict[str, float]:
    """Smoothed IDF over the retrieval corpus.

    Without it, boilerplate shared by every candidate description ("the model",
    "training") dominates the cosine and every pair looks similar.
    """
    n = len(corpus)
    if n == 0:
        return {}
    df: Counter = Counter()
    for bag in corpus:
        df.update(bag.keys())
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def similarity(a: Counter, b: Counter, idf: dict[str, float]) -> float:
    """IDF-weighted cosine in [0, 1].

    The seam a sentence encoder would replace: swap this for an embedding dot
    product and the rest of the layer is unchanged.
    """
    if not a or not b:
        return 0.0
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0
    num = sum(a[t] * b[t] * idf.get(t, 1.0) ** 2 for t in shared)
    na = math.sqrt(sum((a[t] * idf.get(t, 1.0)) ** 2 for t in a))
    nb = math.sqrt(sum((b[t] * idf.get(t, 1.0)) ** 2 for t in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


# ----------------------------------------------------------------- retrieval
def _record_text(record: dict[str, Any]) -> str:
    """The embedded view of an executed attempt: what was changed.

    `change` is the parent-relative delta and is the closest analogue to the
    source's change description; `description` and `idea` back it up when a
    record predates that field or leaves it terse.
    """
    for key in ("change", "description", "idea"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def outcome_of(record: dict[str, Any]) -> str | None:
    status = record.get("status")
    if not isinstance(status, str):
        return None
    return _OUTCOME_BY_STATUS.get(status)


def retrieve_precedents(
    query: str,
    records: Iterable[dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    limit: int = DEFAULT_MAX_PRECEDENTS,
) -> list[dict[str, Any]]:
    """Outcome-labeled prior attempts similar to ``query``.

    Returns at most ``limit`` entries, most similar first. Entries below
    ``threshold`` are dropped and nothing is returned below it — a thin result
    is the correct output for a genuinely novel candidate, and padding it with
    weak neighbors is what degraded the full-dump condition.
    """
    if not 0.0 <= threshold <= 1.0:
        raise PredictError("--threshold must be in [0, 1]")
    if limit < 1:
        raise PredictError("--max-precedents must be >= 1")
    scored = [
        (record, _record_text(record), outcome_of(record))
        for record in records
        if isinstance(record, dict)
    ]
    usable = [
        (record, text, outcome)
        for record, text, outcome in scored
        if outcome is not None and text
    ]
    if not usable:
        return []
    bags = [tokenize(text) for _, text, _ in usable]
    query_bag = tokenize(query)
    idf = _idf(bags + [query_bag])
    hits = []
    for (record, text, outcome), bag in zip(usable, bags):
        score = similarity(query_bag, bag, idf)
        if score >= threshold:
            hits.append(
                {
                    "run_id": str(record.get("run_id")),
                    "change": text,
                    "outcome": outcome,
                    "similarity": round(score, 4),
                }
            )
    # Descending similarity, then run_id, so a tie never depends on ledger order.
    hits.sort(key=lambda item: (-item["similarity"], item["run_id"]))
    return hits[:limit]


def render_precedents(hits: list[dict[str, Any]]) -> list[str]:
    """One line per precedent: the change and the label. Nothing else.

    Adding a failure reason here measurably lowered selective accuracy in the
    source ablation. Keep this renderer lossy on purpose.
    """
    return [f"{hit['change']} — {hit['outcome']}" for hit in hits]


# ------------------------------------------------------------------ pairing
def build_pairs(candidate_ids: list[str]) -> list[dict[str, str]]:
    """Every unordered pair in both presentation orders.

    Both orders are what make a verdict strict-consensus rather than a single
    biased read; judges are measurably position-sensitive.
    """
    unique = list(dict.fromkeys(candidate_ids))
    if len(unique) != len(candidate_ids):
        raise PredictError("shortlist candidate ids must be unique")
    if len(unique) < 2:
        raise PredictError("a tournament needs at least 2 candidates")
    pairs = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            pairs.append({"pair_id": f"{unique[i]}|{unique[j]}", "a": unique[i], "b": unique[j]})
            pairs.append({"pair_id": f"{unique[i]}|{unique[j]}", "a": unique[j], "b": unique[i]})
    return pairs


# ------------------------------------------------------------------- tallying
def tally(
    candidate_ids: list[str],
    verdicts: list[dict[str, Any]],
    *,
    base_rank: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Strict-consensus tournament result.

    A pair scores a vote only when both presentation orders name the same
    winner. A disagreement is an abstention and gives no vote to either side —
    that abstention is the mechanism, not a gap to fill in: it concentrates
    commitment on the comparisons the judge actually gets right.

    Ties break by mean confidence over won comparisons, then by the acquisition
    ranking that produced the shortlist, then by id. The result is fully
    determined by its inputs.
    """
    unique = list(dict.fromkeys(candidate_ids))
    if len(unique) < 2:
        raise PredictError("a tournament needs at least 2 candidates")
    known = set(unique)
    ranks = base_rank or {}

    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for verdict in verdicts:
        if not isinstance(verdict, dict):
            raise PredictError("each verdict must be an object")
        a, b, winner = verdict.get("a"), verdict.get("b"), verdict.get("winner")
        if a not in known or b not in known:
            raise PredictError(f"verdict names unknown candidate: {a!r} vs {b!r}")
        if a == b:
            raise PredictError("a verdict cannot compare a candidate with itself")
        if winner not in {a, b}:
            raise PredictError(
                f"verdict winner {winner!r} must be one of the compared candidates"
            )
        confidence = verdict.get("confidence", 1.0)
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise PredictError("verdict confidence must be a number in [0, 1]")
        pair_id = "|".join(sorted((a, b)))
        order = "forward" if (a, b) == tuple(sorted((a, b))) else "reverse"
        slot = by_pair.setdefault(pair_id, {})
        if order in slot:
            raise PredictError(f"duplicate verdict for {pair_id} in {order} order")
        slot[order] = {"winner": winner, "confidence": float(confidence)}

    expected = {
        "|".join(sorted((unique[i], unique[j])))
        for i in range(len(unique))
        for j in range(i + 1, len(unique))
    }
    missing = sorted(expected - set(by_pair))
    if missing:
        raise PredictError(f"missing verdicts for pairs: {missing}")

    votes = Counter({cid: 0 for cid in unique})
    confidence_sum: Counter = Counter()
    comparisons = []
    for pair_id in sorted(by_pair):
        slot = by_pair[pair_id]
        if len(slot) != 2:
            raise PredictError(f"pair {pair_id} needs both presentation orders")
        forward, reverse = slot["forward"], slot["reverse"]
        agreed = forward["winner"] == reverse["winner"]
        mean_confidence = round((forward["confidence"] + reverse["confidence"]) / 2.0, 6)
        if agreed:
            votes[forward["winner"]] += 1
            confidence_sum[forward["winner"]] += mean_confidence
        comparisons.append(
            {
                "pair_id": pair_id,
                "agreed": agreed,
                "winner": forward["winner"] if agreed else None,
                "mean_confidence": mean_confidence if agreed else None,
            }
        )

    def mean_confidence_of(cid: str) -> float:
        won = votes[cid]
        return confidence_sum[cid] / won if won else 0.0

    order = sorted(
        unique,
        key=lambda cid: (-votes[cid], -mean_confidence_of(cid), ranks.get(cid, 10**6), cid),
    )
    winner = order[0]
    abstentions = sum(1 for item in comparisons if not item["agreed"])
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "pairwise_strict_consensus",
        "winner": winner,
        "candidates": unique,
        "votes": {cid: votes[cid] for cid in unique},
        "mean_confidence": {cid: round(mean_confidence_of(cid), 6) for cid in unique},
        "ranking": order,
        "comparisons": comparisons,
        "coverage": round((len(comparisons) - abstentions) / len(comparisons), 6),
        "abstentions": abstentions,
        # A tournament in which every pair abstained fell through to the
        # acquisition ranking; the receipt must say so rather than imply the
        # judge chose.
        "decided_by": "votes" if votes[winner] > 0 else "acquisition_rank_fallback",
    }


# ----------------------------------------------------------------------- CLI
def _load(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        raise PredictError(f"missing input file: {path}") from None
    except json.JSONDecodeError as exc:
        raise PredictError(f"{path}: invalid JSON: {exc}") from None


def _write(path: Path | None, payload: Any) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _records(ledger_path: Path | None) -> list[dict[str, Any]]:
    if ledger_path is None or not Path(ledger_path).exists():
        return []
    data = _load(ledger_path)
    records = data.get("records", []) if isinstance(data, dict) else []
    if not isinstance(records, list):
        raise PredictError("ledger.records must be a list")
    return records


def cmd_precedents(args: argparse.Namespace) -> int:
    records = _records(args.ledger)
    cfg = predict_cfg(args.cfg or args.ledger)
    if args.threshold is None:
        args.threshold = float(cfg["similarity_threshold"])
    if args.max_precedents is None:
        args.max_precedents = int(cfg["max_precedents"])
    if args.sketches:
        sketches = _load(args.sketches)
        entries = sketches.get("sketches", sketches) if isinstance(sketches, dict) else sketches
        if not isinstance(entries, list):
            raise PredictError("sketches must be a list, or an object with a 'sketches' list")
        blocks = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise PredictError("each sketch must be an object")
            query = entry.get("change") or entry.get("idea") or ""
            hits = retrieve_precedents(
                query,
                records,
                threshold=args.threshold,
                limit=args.max_precedents,
            )
            blocks.append(
                {
                    "candidate_id": entry.get("candidate_id") or entry.get("point_id"),
                    "point_id": entry.get("point_id"),
                    "precedents": hits,
                    "lines": render_precedents(hits),
                }
            )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "threshold": args.threshold,
            "corpus_size": sum(1 for r in records if outcome_of(r) is not None),
            "by_candidate": blocks,
        }
    else:
        if not args.query:
            raise PredictError("provide either --sketches or --query")
        hits = retrieve_precedents(
            args.query, records, threshold=args.threshold, limit=args.max_precedents
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "threshold": args.threshold,
            "corpus_size": sum(1 for r in records if outcome_of(r) is not None),
            "precedents": hits,
            "lines": render_precedents(hits),
        }
    _write(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_pairs(args: argparse.Namespace) -> int:
    sketches = _load(args.sketches)
    entries = sketches.get("sketches", sketches) if isinstance(sketches, dict) else sketches
    if not isinstance(entries, list):
        raise PredictError("sketches must be a list, or an object with a 'sketches' list")
    ids = [str(entry.get("candidate_id") or entry.get("point_id")) for entry in entries]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidates": list(dict.fromkeys(ids)),
        "pairs": build_pairs(ids),
    }
    _write(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_tally(args: argparse.Namespace) -> int:
    verdict_doc = _load(args.verdicts)
    verdicts = (
        verdict_doc.get("verdicts", verdict_doc)
        if isinstance(verdict_doc, dict)
        else verdict_doc
    )
    if not isinstance(verdicts, list):
        raise PredictError("verdicts must be a list, or an object with a 'verdicts' list")
    if args.shortlist:
        shortlist = _load(args.shortlist)
        entries = shortlist.get("shortlist", []) if isinstance(shortlist, dict) else shortlist
        ids = [str(entry.get("point_id")) for entry in entries]
        ranks = {
            str(entry.get("point_id")): int(entry.get("rank", index + 1))
            for index, entry in enumerate(entries)
        }
    else:
        ids = list(dict.fromkeys(str(v.get("a")) for v in verdicts if isinstance(v, dict)))
        ids += [
            str(v.get("b"))
            for v in verdicts
            if isinstance(v, dict) and str(v.get("b")) not in ids
        ]
        ranks = {}
    result = tally(ids, verdicts, base_rank=ranks)
    _write(args.output, result)
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    prec = sub.add_parser(
        "precedents", help="focused outcome-labeled retrieval over prior records"
    )
    prec.add_argument("--ledger", type=Path)
    prec.add_argument("--sketches", type=Path, help="shortlist sketches to retrieve for")
    prec.add_argument("--query", help="single free-text change description")
    prec.add_argument("--threshold", type=float, default=None)
    prec.add_argument("--max-precedents", type=int, default=None)
    prec.add_argument("--cfg", type=Path, help="framework_cfg.json (else nearest to --ledger)")
    prec.add_argument("--output", type=Path)
    prec.set_defaults(func=cmd_precedents)

    pairs = sub.add_parser("pairs", help="ordered comparison schedule (both orders)")
    pairs.add_argument("--sketches", type=Path, required=True)
    pairs.add_argument("--output", type=Path)
    pairs.set_defaults(func=cmd_pairs)

    tal = sub.add_parser("tally", help="strict-consensus vote count over verdicts")
    tal.add_argument("--verdicts", type=Path, required=True)
    tal.add_argument("--shortlist", type=Path, help="shortlist file supplying tie-break ranks")
    tal.add_argument("--output", type=Path)
    tal.set_defaults(func=cmd_tally)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except PredictError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    raise SystemExit(main())
