#!/usr/bin/env python3
"""Offline replay of the attempt downside over a recorded run.

This is the calibration and falsification surface required before any E2E
budget is spent (PROPOSAL §5.1).  It calls no objective, mutates nothing, and
reads only what a past selection could itself have read: the run's kept
proposal sets, the ledger prefix that existed before each admission, and the
policy config.

A selection counts as replayed only once the rebuilt score for the point that
actually won equals that receipt's own ``acquisition_score``.  That equality is
what proves the prefix and the config were reconstructed correctly, so a run
with nothing to replay cannot report success.

Reported checks:

* the outcome partition is mutually exclusive and complete, and ``unpaired``
  is counted apart from ``screen_neutral``;
* crash / fail / neutral push the adjustment down and a success never buys a
  positive bonus;
* an observed crash stays strictly negative even when unpaired attempts
  co-occur;
* every past coverage-family selection replays exactly, and how often the
  channel would have moved a contested slate's winner;
* how much record overlap the carrier and attempt channels have.

Weights are calibrated from signal incidence, score margin, and selection
perturbation only.  Do not pick them by looking at which point historically
won or at any downstream final score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from background_contract import ContractError
from semantic_attempts import (
    DEFAULT_ATTEMPT_CONFIG,
    OBSERVATION_STORE_KEY,
    OUTCOMES,
    attempt_adjustment,
    classify_attempts,
)
from semantic_search import (
    ATTEMPT_POLICIES,
    CARRIER_POLICIES,
    COVERAGE_POLICIES,
    DEFAULT_POLICY_CONFIG,
    _carrier_priors,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected a JSON object")
    return value


def _partition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: sum(1 for row in rows if row["outcome"] == name) for name in OUTCOMES}
    return {
        "counts": counts,
        "total": len(rows),
        "exhaustive": sum(counts.values()) == len(rows),
        "unpaired_counted_separately": True,
    }


def _direction_checks(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Synthetic single-point probes: does each outcome move the sign it must?"""

    def probe(outcomes: list[str], op: str = "improve") -> float:
        synthetic = [
            {"run_id": str(index), "point_id": "P", "op": op, "outcome": outcome,
             "delta": None, "parent_run_id": None}
            for index, outcome in enumerate(outcomes)
        ]
        # Keep the observed population as the leave-one-point-out reference so
        # the probe is calibrated against real base rates, not a vacuum.
        reference = [row for row in rows if row["point_id"] != "P"]
        value, _ = attempt_adjustment(
            synthetic + reference, point_id="P", op=op, cfg=cfg
        )
        return value

    crash_only = probe(["crash"])
    crash_with_unpaired = probe(["crash", "unpaired", "unpaired", "unpaired"])
    return {
        "all_success_non_positive": probe(["screen_success"] * 3) <= 0.0,
        "all_success_is_zero": probe(["screen_success"] * 3) == 0.0,
        "fail_is_negative": probe(["screen_fail"] * 3) < 0.0,
        "neutral_is_negative": probe(["screen_neutral"] * 3) < 0.0,
        "crash_is_negative": crash_only < 0.0,
        "crash_survives_unpaired": crash_with_unpaired < 0.0,
        "success_dilutes_fail": probe(["screen_fail"] * 3 + ["screen_success"] * 3)
        > probe(["screen_fail"] * 3),
        "no_exposure_is_zero": probe([]) == 0.0,
        "values": {
            "success": probe(["screen_success"] * 3),
            "fail": probe(["screen_fail"] * 3),
            "neutral": probe(["screen_neutral"] * 3),
            "crash": crash_only,
            "crash_with_unpaired": crash_with_unpaired,
        },
    }


def _prefix_state(
    ledger: dict[str, Any], records: list[dict[str, Any]], index: int
) -> dict[str, Any]:
    """The ledger as it stood immediately before ``records[index]`` was admitted.

    Records are appended in admission order, so the prefix is exact.  An
    observation can only exist for an already-admitted record, which bounds the
    observation store to the same prefix; whether that bound was tight for a
    given selection is not assumed — it is verified below by reproducing the
    receipt's own acquisition score.
    """
    prefix = records[:index]
    visible_runs = {str(item.get("run_id")) for item in prefix}
    return {
        **ledger,
        "records": prefix,
        "attempt_observations": [
            item
            for item in ledger.get(OBSERVATION_STORE_KEY, []) or []
            if str(item.get("run_id")) in visible_runs
        ],
    }


def _argmax(scores: dict[str, float]) -> str:
    """The winner under the selector's own ordering: score desc, point id asc."""
    return min(scores, key=lambda point: (-scores[point], point))


def _replay_selections(
    ledger: dict[str, Any], cfg: dict[str, Any], semantic_dir: Path
) -> dict[str, Any]:
    """Replay each past coverage-family selection from stored state alone.

    A selection is only counted once it has been *reproduced*: the persisted
    proposal set supplies every ranked point's coverage, the pre-admission
    prefix supplies the carrier and attempt channels, and the resulting score
    for the point that actually won must equal the receipt's own
    ``acquisition_score``.  Anything else is reported as unreplayable rather
    than folded into the perturbation rate.
    """
    records = [r for r in ledger.get("records", []) if isinstance(r, dict)]
    total = 0
    reproduced = 0
    perturbed = 0
    margins: list[float] = []
    flips: list[dict[str, Any]] = []
    unreplayable: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        receipt = record.get("policy_receipt")
        if not isinstance(receipt, dict):
            continue
        policy = receipt.get("policy") or {}
        policy_name = policy.get("name")
        if policy_name not in COVERAGE_POLICIES:
            continue
        total += 1
        run_id = str(record.get("run_id"))
        ranked = receipt.get("ranked_point_ids")
        selected = record.get("semantic_point", {}).get("point_id")
        if not isinstance(ranked, list) or not ranked or selected not in ranked:
            unreplayable.append({"run_id": run_id, "reason": "no ranked slate"})
            continue
        proposals_path = semantic_dir / run_id / "proposals.json"
        if not proposals_path.is_file():
            unreplayable.append({"run_id": run_id, "reason": "proposal set not kept"})
            continue
        proposal_set = _load(proposals_path)
        proposals = proposal_set.get("proposals")
        if not isinstance(proposals, list):
            unreplayable.append({"run_id": run_id, "reason": "malformed proposal set"})
            continue
        coverage = {
            str(item.get("point_id")): float(item["coverage"])
            for item in proposals
            if isinstance(item, dict) and isinstance(item.get("coverage"), (int, float))
        }
        if set(coverage) != set(ranked):
            unreplayable.append(
                {"run_id": run_id, "reason": "proposal set does not match the slate"}
            )
            continue

        prefix = _prefix_state(ledger, records, index)
        receipt_cfg = policy.get("config") if isinstance(policy.get("config"), dict) else {}
        live_cfg = dict(DEFAULT_POLICY_CONFIG)
        live_cfg.update({k: v for k, v in receipt_cfg.items() if k in live_cfg})
        op = (receipt.get("action") or {}).get("op")

        carriers = (
            _carrier_priors(proposal_set, prefix, live_cfg)
            if policy_name in CARRIER_POLICIES
            else {}
        )
        base = {
            point: coverage[point] + (carriers[point][0] if carriers else 0.0)
            for point in ranked
        }
        # Reproduce the score the selector actually recorded, using the config
        # the receipt itself declares.
        live_rows = classify_attempts(
            prefix, noise_threshold=float(live_cfg["attempt_noise_threshold"])
        )
        live_attempt = {
            point: (
                attempt_adjustment(live_rows, point_id=point, op=op, cfg=live_cfg)[0]
                if policy_name in ATTEMPT_POLICIES
                else 0.0
            )
            for point in ranked
        }
        recomputed = base[selected] + live_attempt[selected]
        recorded = receipt.get("acquisition_score")
        if not isinstance(recorded, (int, float)) or abs(
            recomputed - float(recorded)
        ) > 1e-9:
            unreplayable.append(
                {
                    "run_id": run_id,
                    "reason": "recomputed score does not reproduce the receipt",
                    "recomputed": round(recomputed, 10),
                    "recorded": recorded,
                }
            )
            continue
        reproduced += 1

        # Now the counterfactual: the same slate scored with the attempt
        # channel under the weights being calibrated.
        replay_rows = classify_attempts(
            prefix, noise_threshold=float(cfg["attempt_noise_threshold"])
        )
        after = {
            point: base[point]
            + attempt_adjustment(replay_rows, point_id=point, op=op, cfg=cfg)[0]
            for point in ranked
        }
        others = [point for point in ranked if point != selected]
        if not others:
            # A one-point slate has no margin to perturb; counting it would
            # dilute the perturbation rate with selections that had no choice.
            continue
        margin = round(after[selected] - max(after[point] for point in others), 10)
        margins.append(margin)
        if _argmax(after) != selected:
            perturbed += 1
            flips.append(
                {
                    "run_id": run_id,
                    "from_point_id": selected,
                    "to_point_id": _argmax(after),
                    "margin": margin,
                }
            )
    return {
        "selections": total,
        "replayed_exactly": reproduced,
        "unreplayable": unreplayable[:10],
        "perturbed": perturbed,
        "contested_slates": len(margins),
        "perturbation_rate": (
            None if not margins else round(perturbed / len(margins), 6)
        ),
        "worst_margin": min(margins, default=None),
        "flips": flips[:10],
    }


def _channel_overlap(ledger: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How many records feed both the carrier posterior and the attempt statistic."""
    carrier_runs: set[str] = set()
    for record in ledger.get("records", []):
        if not isinstance(record, dict) or record.get("status") not in {"keep", "discard"}:
            continue
        edges = record.get("semantic_edges")
        if isinstance(edges, list) and edges:
            carrier_runs.add(str(record.get("run_id")))
    attempt_runs = {row["run_id"] for row in rows}
    return {
        "carrier_records": len(carrier_runs),
        "attempt_records": len(attempt_runs),
        "overlap": len(carrier_runs & attempt_runs),
        "attempt_only": len(attempt_runs - carrier_runs),
        "carrier_only": len(carrier_runs - attempt_runs),
    }


def cmd_replay(args: argparse.Namespace) -> int:
    ledger = _load(args.ledger)
    cfg = {key: DEFAULT_POLICY_CONFIG[key] for key in DEFAULT_ATTEMPT_CONFIG}
    # Same source the live selection reads, so a replay is reproducible from
    # the run's own state.
    run_cfg = args.ledger.parent / "framework_cfg.json"
    if run_cfg.is_file():
        section = _load(run_cfg).get("semantic_search") or {}
        cfg.update({key: value for key, value in section.items() if key in cfg})
    if args.cfg:
        override = json.loads(args.cfg)
        unknown = sorted(set(override) - set(cfg))
        if unknown:
            raise ContractError(f"unknown attempt config keys {unknown}")
        cfg.update(override)

    rows = classify_attempts(
        ledger, noise_threshold=float(cfg["attempt_noise_threshold"])
    )
    semantic_dir = args.semantic_dir or (args.ledger.parent / ".semantic")
    replay = _replay_selections(ledger, cfg, semantic_dir)
    report = {
        "config": cfg,
        "partition": _partition(rows),
        "directions": _direction_checks(rows, cfg),
        "selection_replay": replay,
        "channel_overlap": _channel_overlap(ledger, rows),
        "attempts": rows if args.verbose else None,
    }
    checks = report["directions"]
    # A green result must mean the run supplied evidence, not that it supplied
    # nothing to disagree with: the synthetic direction probes pass on an empty
    # ledger, so they alone can never carry the exit code.
    evidence = {
        "has_attempts": len(rows) > 0,
        "has_selections": replay["selections"] > 0,
        "all_selections_replayed": (
            replay["selections"] > 0
            and replay["replayed_exactly"] == replay["selections"]
        ),
    }
    report["evidence"] = evidence
    report["ok"] = bool(
        report["partition"]["exhaustive"]
        and checks["all_success_is_zero"]
        and checks["fail_is_negative"]
        and checks["neutral_is_negative"]
        and checks["crash_is_negative"]
        and checks["crash_survives_unpaired"]
        and checks["no_exposure_is_zero"]
        and all(evidence.values())
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--cfg", help="JSON attempt-weight overrides")
    parser.add_argument("--semantic-dir", type=Path,
                        help="where the run kept its proposal sets "
                             "(default <ledger dir>/.semantic)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verbose", action="store_true",
                        help="include the per-attempt classification rows")
    parser.set_defaults(func=cmd_replay)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
