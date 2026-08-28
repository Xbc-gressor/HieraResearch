"""Deterministic pool-local Top-2 Recall analyzer (no LLM, no GPU, no ledger).

Inputs are frozen artifacts only: the experiment manifest, pool manifests,
production judge artifacts (judge.json + regular stage artifacts per pool),
and terminal job results. Scores are lower-is-better; any non-ok status maps
to +inf. Ties are broken by frozen candidate id.

Selectors (per experiment §6):

    J2         production judged-slate aggregation. Its top-2 IS the recorded
               aggregation slate (never recomputed). The full ranking needed
               for restricted views is derived mechanically: slate first, the
               remaining candidates by mean rank over the two regular
               rollouts' output rankings, ties by frozen coverage order; a
               judge-failure fallback is the coverage order itself.
    F2(t)      rank all pool candidates by the fidelity-t score.
    J4->F2(t)  judge top-4 shortlist (experiment §4.1), then rank the
               shortlist by the fidelity-t score.

The J4 shortlist derivation (union of regular top-2s, filled by mean output
rank, ties by coverage; coverage top-4 when the required judge failed) is
saved per pool as hybrid_shortlist.json with its input judgment digests. It
never alters the pure J2 aggregation.

Oracle GT is the matrix H300 score; when adjudication repeats exist for a
candidate the analyzer applies the frozen protocol (median of the matrix
score and exactly two adjudication repeats) and never decides on its own
whether to rerun or how else to aggregate.

Verdicts (experiment §7) are mechanical: the N-layer stop/proceed decision
with the frozen P* selection chain, and the C-layer confirmation gate for a
previously frozen P*.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import manifest  # noqa: E402

INF = float("inf")
SHORTLIST_SIZE = 4
TOP_K = 2


class AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


def load_job_rows(jobs_dir: Path, experiment_id: str) -> list[dict]:
    rows = []
    for request_path in sorted(Path(jobs_dir).glob("*/request.json")):
        request = manifest.load_json(request_path)
        if request.get("experiment_id") != experiment_id:
            continue
        result_path = request_path.parent / manifest.RESULT_FILENAME
        if not result_path.exists():
            continue
        result = manifest.load_json(result_path)
        errors = manifest.validate_result_against_request(result, request)
        if errors:
            raise AnalysisError(
                f"{result_path} does not bind to its request: "
                + "; ".join(errors)
            )
        rows.append({"request": request, "result": result})
    return rows


def pool_observations(pool: dict, rows: list[dict], fidelities: list[int]) -> dict:
    """Index terminal results for one pool; enforce matrix uniqueness."""
    candidate_ids = {c["candidate_id"] for c in pool["candidates"]}
    matrix: dict[str, dict[int, dict]] = {}
    adjudication: dict[str, list[dict]] = {}
    calibration: list[dict] = []
    for row in rows:
        request, result = row["request"], row["result"]
        if request["pool_id"] != pool["pool_id"]:
            continue
        if request["candidate_id"] not in candidate_ids:
            raise AnalysisError(
                f"result for unknown candidate {request['candidate_id']} in "
                f"pool {pool['pool_id']}"
            )
        purpose = request["purpose"]
        if purpose == "matrix":
            if request["evaluation_path"] != "adapter":
                continue  # official-path jobs belong to the equivalence gate
            per = matrix.setdefault(request["candidate_id"], {})
            fidelity = request["requested_train_seconds"]
            if fidelity in per:
                raise AnalysisError(
                    f"duplicate matrix result for {request['candidate_id']} "
                    f"@{fidelity}s in pool {pool['pool_id']}"
                )
            per[fidelity] = result
        elif purpose == "adjudication":
            adjudication.setdefault(request["candidate_id"], []).append(
                {"repeat_index": request["repeat_index"], "result": result}
            )
        else:
            calibration.append(result)
    for candidate in pool["candidates"]:
        per = matrix.get(candidate["candidate_id"], {})
        missing = [f for f in fidelities if f not in per]
        if missing:
            raise AnalysisError(
                f"pool {pool['pool_id']} candidate "
                f"{candidate['candidate_id']} missing matrix results at "
                f"{missing}"
            )
    return {
        "matrix": matrix,
        "adjudication": adjudication,
        "calibration": calibration,
    }


def observed_score(result: dict) -> float:
    if result["status"] == "ok":
        return float(result["score"])
    return INF


def oracle_scores(pool: dict, observations: dict) -> dict[str, float]:
    """Matrix H300, replaced by the frozen median-of-3 protocol wherever
    adjudication repeats exist."""
    out = {}
    for candidate in pool["candidates"]:
        cid = candidate["candidate_id"]
        base = observed_score(
            observations["matrix"][cid][manifest.FULL_TRAIN_SECONDS]
        )
        repeats = observations["adjudication"].get(cid)
        if repeats:
            if sorted(r["repeat_index"] for r in repeats) != [1, 2]:
                raise AnalysisError(
                    f"adjudication for {cid} must be exactly repeats 1 and 2, "
                    f"got {sorted(r['repeat_index'] for r in repeats)}"
                )
            scores = [base] + [
                observed_score(r["result"]) for r in repeats
            ]
            out[cid] = statistics.median(scores)
        else:
            out[cid] = base
    return out


# ---------------------------------------------------------------------------
# rankings
# ---------------------------------------------------------------------------


def coverage_order(pool: dict) -> list[str]:
    return [
        c["candidate_id"]
        for c in sorted(
            pool["candidates"],
            key=lambda c: (c["coverage_rank"], c["candidate_id"]),
        )
    ]


def score_ranking(scores: dict[str, float], members: list[str]) -> list[str]:
    return sorted(members, key=lambda cid: (scores[cid], cid))


def _label_map(pool: dict) -> dict[str, str]:
    mapping = {}
    for candidate in pool["candidates"]:
        label = candidate.get("judge_label")
        if not label:
            raise AnalysisError(
                f"pool {pool['pool_id']} candidate "
                f"{candidate['candidate_id']} has no judge_label"
            )
        mapping[label] = candidate["candidate_id"]
    return mapping


def _to_cids(labels: list[str], label_map: dict[str, str], where: str) -> list[str]:
    unknown = [label for label in labels if label not in label_map]
    if unknown:
        raise AnalysisError(f"{where}: labels {unknown} not in pool manifest")
    return [label_map[label] for label in labels]


def _regular_rankings(stage_docs: dict, label_map: dict, pool_id: str) -> list[list[str]]:
    rankings = []
    for stage in ("regular-0", "regular-1"):
        doc = stage_docs.get(stage)
        if doc is None or doc.get("status") != "valid":
            raise AnalysisError(
                f"pool {pool_id}: regular stage {stage} is not valid"
            )
        rankings.append(_to_cids(doc["ranking"], label_map, f"{pool_id}/{stage}"))
    return rankings


def _mean_rank_order(
    members: list[str],
    rankings: list[list[str]],
    coverage: list[str],
) -> list[str]:
    coverage_pos = {cid: i for i, cid in enumerate(coverage)}
    mean_rank = {
        cid: sum(r.index(cid) for r in rankings) / len(rankings)
        for cid in members
    }
    return sorted(members, key=lambda cid: (mean_rank[cid], coverage_pos[cid]))


def judge_failed(judge_doc: dict) -> bool:
    return judge_doc["aggregation"]["path"] == "coverage_fallback"


def j2_full_ranking(pool: dict, judge_doc: dict, stage_docs: dict) -> list[str]:
    coverage = coverage_order(pool)
    label_map = _label_map(pool)
    if judge_failed(judge_doc):
        return coverage
    slate = _to_cids(
        judge_doc["aggregation"]["slate"], label_map,
        f"{pool['pool_id']}/judge.json slate",
    )
    rankings = _regular_rankings(stage_docs, label_map, pool["pool_id"])
    rest = [cid for cid in coverage if cid not in slate]
    return slate + _mean_rank_order(rest, rankings, coverage)


def j4_shortlist(pool: dict, judge_doc: dict, stage_docs: dict) -> tuple[list[str], dict]:
    coverage = coverage_order(pool)
    if judge_failed(judge_doc):
        shortlist = coverage[:SHORTLIST_SIZE]
        return shortlist, {
            "path": "coverage_fallback",
            "reason": judge_doc["aggregation"].get("reason"),
            "shortlist": shortlist,
        }
    label_map = _label_map(pool)
    rankings = _regular_rankings(stage_docs, label_map, pool["pool_id"])
    union = []
    for ranking in rankings:
        for cid in ranking[:TOP_K]:
            if cid not in union:
                union.append(cid)
    union = _mean_rank_order(union, rankings, coverage)
    fill = _mean_rank_order(
        [cid for cid in coverage if cid not in union], rankings, coverage
    )
    shortlist = (union + fill)[:SHORTLIST_SIZE]
    derivation = {
        "path": "regular_union_mean_rank",
        "regular_top2": [ranking[:TOP_K] for ranking in rankings],
        "union": union,
        "fill": fill[: max(0, SHORTLIST_SIZE - len(union))],
        "shortlist": shortlist,
        "input_judgment_digests": {
            stage: manifest.digest(stage_docs[stage])
            for stage in ("regular-0", "regular-1")
        },
    }
    return shortlist, derivation


# ---------------------------------------------------------------------------
# per-pool analysis
# ---------------------------------------------------------------------------


def restrict(ranking: list[str], members: set[str]) -> list[str]:
    return [cid for cid in ranking if cid in members]


def recall_at_2(oracle_top2: list[str], selected: list[str]) -> float:
    return len(set(oracle_top2) & set(selected[:TOP_K])) / TOP_K


def analyze_pool(
    pool: dict,
    rows: list[dict],
    fidelities: list[int],
    judge_doc: dict,
    stage_docs: dict,
) -> dict:
    low_fidelities = [f for f in fidelities if f != manifest.FULL_TRAIN_SECONDS]
    observations = pool_observations(pool, rows, fidelities)
    gt = oracle_scores(pool, observations)
    all_members = [c["candidate_id"] for c in pool["candidates"]]
    competitive = {
        c["candidate_id"]
        for c in pool["candidates"]
        if c["quality"]["tier"] == "competitive"
    }

    low_scores = {
        fidelity: {
            cid: observed_score(observations["matrix"][cid][fidelity])
            for cid in all_members
        }
        for fidelity in low_fidelities
    }
    elapsed = {
        fidelity: {
            cid: float(
                observations["matrix"][cid][fidelity].get(
                    "elapsed_accelerator_seconds"
                )
                or 0.0
            )
            for cid in all_members
        }
        for fidelity in fidelities
    }

    rankings: dict[str, list[str]] = {
        "J2": j2_full_ranking(pool, judge_doc, stage_docs)
    }
    shortlist, shortlist_derivation = j4_shortlist(pool, judge_doc, stage_docs)
    for fidelity in low_fidelities:
        ordered = score_ranking(low_scores[fidelity], all_members)
        rankings[f"F2({fidelity})"] = ordered
        hybrid = score_ranking(low_scores[fidelity], shortlist)
        rankings[f"J4->F2({fidelity})"] = hybrid + [
            cid for cid in ordered if cid not in shortlist
        ]

    selectors = list(rankings)
    report: dict = {
        "pool_id": pool["pool_id"],
        "cohort": pool["cohort"],
        "competitive": sorted(competitive),
        "truncation_semantics_candidates": sorted(
            c["candidate_id"]
            for c in pool["candidates"]
            if not c["freeze_checks"].get("reads_env_train_budget_seconds")
        ),
        "oracle_scores": {cid: gt[cid] for cid in sorted(gt)},
        "low_fidelity_scores": {
            str(fidelity): dict(sorted(low_scores[fidelity].items()))
            for fidelity in low_fidelities
        },
        "j4_shortlist": shortlist,
        "j4_shortlist_derivation": shortlist_derivation,
        "judge_cost": judge_doc.get("judge_cost"),
        "selections": {},
        "recall": {},
        "j4_retention": {},
        "costs": {},
    }

    for restriction, members in (
        ("all", set(all_members)),
        ("competitive", competitive),
    ):
        oracle_rank = restrict(score_ranking(gt, all_members), members)
        oracle_top2 = oracle_rank[:TOP_K]
        report["selections"][restriction] = {"oracle_top2": oracle_top2}
        report["recall"][restriction] = {}
        for selector in selectors:
            selected = restrict(rankings[selector], members)[:TOP_K]
            report["selections"][restriction][selector] = selected
            report["recall"][restriction][selector] = recall_at_2(
                oracle_top2, selected
            )
        report["j4_retention"][restriction] = (
            len(set(oracle_top2) & set(restrict(shortlist, members))) / TOP_K
        )

    pool_n = len(all_members)
    for selector in selectors:
        if selector == "J2":
            nominal = 0
            actual = 0.0
        elif selector.startswith("J4->F2("):
            fidelity = int(selector[len("J4->F2("):-1])
            nominal = SHORTLIST_SIZE * fidelity
            actual = sum(elapsed[fidelity][cid] for cid in shortlist)
        else:
            fidelity = int(selector[len("F2("):-1])
            nominal = pool_n * fidelity
            actual = sum(elapsed[fidelity].values())
        report["costs"][selector] = {
            "nominal_train_seconds": nominal,
            "actual_accelerator_seconds": actual,
        }
    report["h300_actual_accelerator_seconds"] = sum(
        elapsed[manifest.FULL_TRAIN_SECONDS].values()
    )
    return report


# ---------------------------------------------------------------------------
# macro + verdicts
# ---------------------------------------------------------------------------


def macro_recall(pool_reports: list[dict], selector: str, restriction: str) -> float:
    return sum(
        report["recall"][restriction][selector] for report in pool_reports
    ) / len(pool_reports)


def _selector_fidelity(selector: str) -> int:
    return int(selector[selector.index("(") + 1:-1])


def n_layer_verdict(pool_reports: list[dict], low_fidelities: list[int]) -> dict:
    """Experiment §7.1: stop unless some low-fidelity selector strictly beats
    J2 on N/competitive macro Recall@2; otherwise freeze P* by the chain
    (recall desc, actual GPU-s asc, smaller t, hybrid first)."""
    j2 = macro_recall(pool_reports, "J2", "competitive")
    candidates = []
    for fidelity in low_fidelities:
        for selector in (f"F2({fidelity})", f"J4->F2({fidelity})"):
            candidates.append(
                {
                    "selector": selector,
                    "recall_competitive": macro_recall(
                        pool_reports, selector, "competitive"
                    ),
                    "recall_all": macro_recall(pool_reports, selector, "all"),
                    "actual_accelerator_seconds": sum(
                        report["costs"][selector]["actual_accelerator_seconds"]
                        for report in pool_reports
                    ),
                    "t": fidelity,
                    "hybrid": selector.startswith("J4->"),
                }
            )
    better = [c for c in candidates if c["recall_competitive"] > j2]
    verdict = {
        "j2_recall_competitive": j2,
        "j2_recall_all": macro_recall(pool_reports, "J2", "all"),
        "selectors": candidates,
    }
    if not better:
        verdict["decision"] = "stop"
        verdict["frozen_selector"] = None
        return verdict
    frozen = min(
        better,
        key=lambda c: (
            -c["recall_competitive"],
            c["actual_accelerator_seconds"],
            c["t"],
            0 if c["hybrid"] else 1,
        ),
    )
    verdict["decision"] = "proceed_to_C"
    verdict["frozen_selector"] = frozen
    return verdict


def c_layer_verdict(
    pool_reports: list[dict],
    frozen_selector: str,
    n_verdict: dict,
    gate_status: str | None,
) -> dict:
    """Experiment §7.2 confirmation gate for the frozen P*."""
    j2_c = macro_recall(pool_reports, "J2", "competitive")
    p_c = macro_recall(pool_reports, frozen_selector, "competitive")
    checks = {
        "c_competitive_margin": {
            "value": p_c - j2_c,
            "required": 0.125,
            "pass": p_c - j2_c >= 0.125,
        }
    }
    n_j2 = n_verdict["j2_recall_competitive"]
    n_p = next(
        (
            c["recall_competitive"]
            for c in n_verdict["selectors"]
            if c["selector"] == frozen_selector
        ),
        None,
    )
    checks["n_competitive_not_below_j2"] = {
        "p_star": n_p,
        "j2": n_j2,
        "pass": n_p is not None and n_p >= n_j2,
    }
    if frozen_selector.startswith("J4->"):
        unexplained = []
        for report in pool_reports:
            oracle_top2 = report["selections"]["competitive"]["oracle_top2"]
            selected = report["selections"]["competitive"][frozen_selector]
            missed = [cid for cid in oracle_top2 if cid not in selected]
            shortlist = set(report["j4_shortlist"])
            # A miss that IS in the shortlist was lost by the low-fidelity
            # evaluator, not by the judge top-4 — unexplained by retention.
            unexplained.extend(
                f"{report['pool_id']}:{cid}"
                for cid in missed
                if cid in shortlist
            )
        checks["hybrid_failures_explained_by_shortlist"] = {
            "unexplained": unexplained,
            "pass": not unexplained,
        }
    if gate_status is not None:
        checks["equivalence_gate"] = {
            "status": gate_status,
            "pass": gate_status == "pass",
        }
    return {
        "frozen_selector": frozen_selector,
        "j2_recall_competitive": j2_c,
        "p_star_recall_competitive": p_c,
        "checks": checks,
        "decision": (
            "enter_production_e2e"
            if all(check["pass"] for check in checks.values())
            else "not_supported"
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def analyze(
    experiment: dict,
    pools: list[dict],
    jobs_dir: Path,
    judges_dir: Path,
    *,
    layer: str,
    frozen_selector: str | None = None,
    n_report: dict | None = None,
    gate_status: str | None = None,
) -> dict:
    manifest.validate_experiment(experiment)
    fidelities = [int(f) for f in experiment["fidelities"]]
    low_fidelities = [f for f in fidelities if f != manifest.FULL_TRAIN_SECONDS]
    rows = load_job_rows(jobs_dir, experiment["experiment_id"])
    pool_reports = []
    for pool in pools:
        manifest.validate_pool_manifest(pool)
        judge_dir = Path(judges_dir) / pool["pool_id"]
        judge_doc = manifest.load_json(judge_dir / "judge.json")
        stage_docs = {}
        for stage in ("regular-0", "regular-1"):
            stage_path = judge_dir / f"{stage}.json"
            if stage_path.exists():
                stage_docs[stage] = manifest.load_json(stage_path)
        pool_reports.append(
            analyze_pool(pool, rows, fidelities, judge_doc, stage_docs)
        )

    selectors = ["J2"] + [
        name
        for fidelity in low_fidelities
        for name in (f"F2({fidelity})", f"J4->F2({fidelity})")
    ]
    report = {
        "schema_version": manifest.SCHEMA_VERSION,
        "experiment_id": experiment["experiment_id"],
        "layer": layer,
        "pools": pool_reports,
        "macro": {
            restriction: {
                selector: macro_recall(pool_reports, selector, restriction)
                for selector in selectors
            }
            for restriction in ("all", "competitive")
        },
        "macro_j4_retention": {
            restriction: sum(
                r["j4_retention"][restriction] for r in pool_reports
            ) / len(pool_reports)
            for restriction in ("all", "competitive")
        },
    }
    if layer == "N":
        report["verdict"] = n_layer_verdict(pool_reports, low_fidelities)
    else:
        if not frozen_selector or n_report is None:
            raise AnalysisError(
                "C-layer analysis needs --frozen-selector and --n-report"
            )
        report["verdict"] = c_layer_verdict(
            pool_reports, frozen_selector, n_report["verdict"], gate_status
        )
    return report


def _print_summary(report: dict) -> None:
    macro = report["macro"]
    selectors = list(macro["all"])
    print(f"layer {report['layer']}  pools {len(report['pools'])}")
    print(f"{'selector':<14}{'all':>8}{'competitive':>14}")
    for selector in selectors:
        print(
            f"{selector:<14}{macro['all'][selector]:>8.3f}"
            f"{macro['competitive'][selector]:>14.3f}"
        )
    verdict = report["verdict"]
    print(f"verdict: {verdict.get('decision')}")
    if verdict.get("frozen_selector"):
        frozen = verdict["frozen_selector"]
        if isinstance(frozen, dict):
            print(f"frozen P*: {frozen['selector']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--pool", action="append", required=True)
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--judges-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", choices=["N", "C"], default="N")
    parser.add_argument("--frozen-selector", default=None)
    parser.add_argument("--n-report", default=None)
    parser.add_argument("--gate-verdict", default=None)
    args = parser.parse_args(argv)

    gate_status = None
    if args.gate_verdict:
        gate_status = manifest.load_json(Path(args.gate_verdict)).get("status")
    report = analyze(
        manifest.load_json(Path(args.experiment)),
        [manifest.load_json(Path(p)) for p in args.pool],
        Path(args.jobs_dir),
        Path(args.judges_dir),
        layer=args.layer,
        frozen_selector=args.frozen_selector,
        n_report=(
            manifest.load_json(Path(args.n_report)) if args.n_report else None
        ),
        gate_status=gate_status,
    )
    output = Path(args.output)
    manifest.atomic_write_json(output, report)
    for pool_report in report["pools"]:
        manifest.atomic_write_json(
            output.parent / "hybrid_shortlist" / f"{pool_report['pool_id']}.json",
            {
                "pool_id": pool_report["pool_id"],
                "shortlist": pool_report["j4_shortlist"],
                "derivation": pool_report["j4_shortlist_derivation"],
            },
        )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
