"""Deterministic qualification tiering + pool freezing (no LLM, no GPU).

Input: one qualification bundle per pool (assembled by the experimenter from
production/B-2 artifacts), holding the anchor carrier results and every bank
candidate with its frozen qualification result and coverage rank. This tool
only transforms that data:

    natural cohort      the pool IS the historical production-shape pool
                        (exactly 6 candidates); tiers are annotations only
    competitive cohort  keep competitive candidates; fewer than 6 -> a
                        rejection receipt (never request new candidates or
                        re-pick after seeing scores); else the first 6 by
                        original production coverage rank — never a
                        score-ordered top-6

freeze_checks are computed mechanically at manifest time: whether the pinned
train.py reads env.train_budget_seconds (a non-reader's low-fidelity runs are
truncation, not a compressed schedule — annotated, not excluded) and whether
the candidate's historical qualification stdout summary parses a
training_seconds field (the runtime budget check depends on it).

A written pool manifest is immutable: identical re-runs are no-ops, any
divergence is an error. Qualification results may be referenced afterwards but
never overwritten by this experiment's low/full scores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import manifest  # noqa: E402


class QualificationError(RuntimeError):
    pass


def _anchor(bundle: dict) -> dict:
    """Strongest finite carrier anchor (lower is better)."""
    anchors = [
        anchor
        for anchor in bundle.get("anchors", [])
        if manifest.is_finite_number(anchor.get("qualification_score"))
    ]
    if not anchors:
        raise QualificationError(
            "no finite carrier anchor; the pool cannot be tiered"
        )
    return min(anchors, key=lambda a: float(a["qualification_score"]))


def _freeze_checks(entry: dict) -> dict:
    train_path = Path(entry["candidate_path"])
    if not train_path.is_file():
        raise QualificationError(f"missing pinned candidate {train_path}")
    stdout_path = entry.get("qualification_stdout_path")
    has_training_seconds = False
    if stdout_path:
        has_training_seconds = manifest.summary_has_training_seconds(
            Path(stdout_path).read_text(errors="replace")
        )
    return {
        "reads_env_train_budget_seconds": manifest.reads_env_train_budget_seconds(
            train_path
        ),
        "summary_has_training_seconds": has_training_seconds,
    }


def _frozen_candidate(entry: dict, tier: dict) -> dict:
    params = entry["params"]
    params_digest = manifest.params_digest(params)
    revision = entry["candidate_execution_revision"]
    candidate_id = entry.get("candidate_id") or manifest.candidate_fallback_id(
        entry.get("source_prefix"),
        entry["point_id"],
        entry["op"],
        entry.get("parents", []),
        revision,
        params_digest,
    )
    return {
        "candidate_id": candidate_id,
        "coverage_rank": int(entry["coverage_rank"]),
        "point_id": entry["point_id"],
        "op": entry["op"],
        "parents": list(entry.get("parents", [])),
        "judge_label": entry.get("judge_label"),
        "candidate_path": str(entry["candidate_path"]),
        "candidate_execution_revision": revision,
        "prepare_digest": manifest.sha256_file(
            Path(entry["candidate_path"]).parent / "prepare.py"
        ),
        "params": params,
        "params_digest": params_digest,
        "quality": tier,
        "freeze_checks": _freeze_checks(entry),
    }


def freeze_pool(bundle: dict) -> dict:
    """Turn one qualification bundle into a frozen pool manifest, or a
    rejection receipt (status != 'frozen') for an insufficient bank."""
    cohort = bundle.get("cohort")
    if cohort not in ("natural", "competitive"):
        raise QualificationError(f"unknown cohort {cohort!r}")
    anchor = _anchor(bundle)
    anchor_score = float(anchor["qualification_score"])

    tiered = []
    for entry in bundle["bank"]:
        qualification = entry.get("qualification") or {}
        score = qualification.get("score")
        tier_name = manifest.quality_tier(score, anchor_score)
        tiered.append((entry, {
            "tier": tier_name,
            "qualification_score": score if manifest.is_finite_number(score) else None,
            "qualification_result": qualification.get("result_digest"),
            "rule": (
                f"score <= {manifest.COMPETITIVE_FACTOR:g} * anchor"
                if tier_name == "competitive"
                else f"score <= {manifest.BORDERLINE_FACTOR:g} * anchor"
                if tier_name == "borderline"
                else "crash/non-finite or worse than "
                f"{manifest.BORDERLINE_FACTOR:g} * anchor"
            ),
        }))

    if cohort == "natural":
        if len(tiered) != manifest.POOL_SIZE:
            raise QualificationError(
                f"natural pool must hold exactly {manifest.POOL_SIZE} "
                f"candidates, got {len(tiered)}"
            )
        selected = tiered
    else:
        competitive = [
            item for item in tiered if item[1]["tier"] == "competitive"
        ]
        if len(competitive) < manifest.POOL_SIZE:
            return {
                "schema_version": manifest.SCHEMA_VERSION,
                "status": "insufficient_competitive_bank",
                "pool_id": bundle["pool_id"],
                "source_prefix": bundle.get("source_prefix"),
                "anchor": {
                    "candidate_id": anchor.get("candidate_id"),
                    "qualification_score": anchor_score,
                    "result_digest": anchor.get("result_digest"),
                },
                "bank_size": len(tiered),
                "competitive_count": len(competitive),
                "tiers": {
                    tier_name: sum(
                        1 for item in tiered if item[1]["tier"] == tier_name
                    )
                    for tier_name in manifest.QUALITY_TIERS
                },
            }
        selected = sorted(
            competitive, key=lambda item: int(item[0]["coverage_rank"])
        )[: manifest.POOL_SIZE]

    candidates = [_frozen_candidate(entry, tier) for entry, tier in selected]
    doc = {
        "schema_version": manifest.SCHEMA_VERSION,
        "status": "frozen",
        "pool_id": bundle["pool_id"],
        "cohort": cohort,
        "source_prefix": bundle.get("source_prefix"),
        "production_pool_digest": bundle.get("production_pool_digest"),
        "judge_bundle_digest": bundle.get("judge_bundle_digest"),
        "anchor": {
            "candidate_id": anchor.get("candidate_id"),
            "qualification_score": anchor_score,
            "result_digest": anchor.get("result_digest"),
        },
        "candidates": candidates,
    }
    manifest.validate_pool_manifest(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="qualification bundle JSON")
    parser.add_argument("--output", required=True,
                        help="pool manifest path (immutable once written)")
    parser.add_argument("--rejection-output", default=None,
                        help="where a rejection receipt goes "
                             "(default: <output>.rejected.json)")
    args = parser.parse_args(argv)

    bundle = manifest.load_json(Path(args.input))
    doc = freeze_pool(bundle)
    if doc["status"] != "frozen":
        rejection_path = Path(
            args.rejection_output or (args.output + ".rejected.json")
        )
        manifest.atomic_write_json(rejection_path, doc)
        print(
            f"pool {doc['pool_id']}: {doc['status']} "
            f"(competitive {doc['competitive_count']}/{doc['bank_size']}); "
            f"receipt at {rejection_path}"
        )
        return 1
    written = manifest.freeze_immutable(Path(args.output), doc)
    print(
        f"pool {doc['pool_id']}: frozen {len(doc['candidates'])} candidates"
        + ("" if written else " (already frozen, unchanged)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
