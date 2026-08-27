#!/usr/bin/env python3
"""Step 1 — warm-config evaluator (sequential, resumable, stop-on-crash).

Run by `tunable-contract-extractor` (segment ③) after it has produced a
candidate `train.py` (PARAM_SCHEMA + SEARCH_SPACE + make_model, NO BASE_PARAMS
yet) and written `_warm_configs.json` (the K warm configs). For schema-4
candidates, config 0 is a mandatory control; non-fresh candidates must bind it
to a validated primary-parent parameter-transfer receipt. It samples the
remaining K_eval slots uniformly without replacement and evaluates those
configs on the score fn
(`prepare`'s `score_fn`), and seeds the candidate's tuning. Sequential +
resumable + stop-on-crash, so the extractor can diagnose + fix ONE crash at a
time without re-evaluating what already passed:

1. CREATE `BASE_PARAMS` (so the candidate is a complete contract that imports).
2. Persist the mandatory indices, random permutation, seed, and
   selected/deferred indices in
   `phase_a.warm_config_selection`, then evaluate the selected configs in that
   order. **Resume** reuses both that selection and any config already scored in
   a prior report (passed configs are cached by params). On the FIRST
   not-yet-scored config that raises, record it (original proposed index + FULL
   traceback) and STOP — exit `3` (CRASHED). The caller diagnoses it
   (config-invalid → edit that slot in `_warm_configs.json`; code-incompatible →
   edit `train.py`), then re-runs this to resume.
3. When every selected config has a score (no crash), pick the best finite warm
   row, including an inherited config-0 control when it wins, write it into
   `BASE_PARAMS`, finalize `phase_a` (warm_start_configs +
   best_warm_score + best_warm_params + search_space), exit `0`.

An optional task-owned preflight runs before each score attempt in an isolated
subprocess. It is a real-shape feasibility check, not a smoke score, and never
reserves an objective slot. There is no `base_score`. Run from the task uv env:
`uv --project tasks/<task> run python tools/tuners/warmstart_eval.py ...`
(`--project` selects the task env without chdir, so repo-relative paths resolve;
`--directory` would chdir into the task dir and break them).

Legacy schema-3 candidate briefs remain readable for in-progress resumes. New
schema-4 briefs make config 0 mandatory and require a validated
`_parameter_transfer.json` for every non-fresh candidate. A brief stamped
`implementation_source.kind=provided_entrypoint` enables the observed-control
guard: exactly one warm config may run, `k_eval` must be one, and a literal
`DEFAULT_PARAMS` must match that config exactly.

Under the global-donor policy pair (`anchor_transfer_challenger_v1` x
`hebo24-transfer10-hebo10`), `--donor-snapshot` binds the generation's donor
snapshot and the candidate must carry a matching helper-written
`_global_donor_transfer.json`.  A compatible donor row is a mandatory but
failable warm treatment: its preflight rejection or objective crash is
persisted as the donor observation and the remaining selected rows continue,
while a donor row that also carries the lineage control stays fail-closed.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # tools/ for apply_base_params
import apply_base_params  # noqa: E402
from _common import (  # noqa: E402
    EvaluationBudgetExhausted,
    resolve_score_fn,
    resolve_preflight_fn,
    timed_eval,
    timed_preflight,
    cast_params_to_search_space,
    is_finite_score,
    load_candidate_modules,
    read_tune_report,
    search_space_for_json,
    write_json,
    write_tune_report,
)
from failure_artifacts import record_failure  # noqa: E402
from tune_tools import (  # noqa: E402
    GLOBAL_DONOR_TRANSFER_FILENAME,
    PARAMETER_TRANSFER_FILENAME,
    _bounds_violations,
    _candidate_execution_revision,
    _global_donor_policy_active,
    _json_native,
    _read_param_schema,
    _read_search_space,
    _validate_global_donor_receipt,
    _validate_schema_values,
    finite_warm_incumbent_rows,
    lint_contract,
    validate_parameter_transfer,
)

CRASHED = 3  # a not-yet-scored config raised; the caller diagnoses + fixes + resumes
BUDGET_EXHAUSTED = 4  # no score_fn call was started; coordinator ends the run


# =============================================================================
# Candidate and warm-config contracts
# =============================================================================


def _params_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True, default=str)


def _validated_warm_configs(candidate_path: Path, configs: list) -> tuple[list, dict]:
    """Validate every proposed row before BASE_PARAMS/report mutation."""
    schema = _read_param_schema(candidate_path)
    search_space = _read_search_space(candidate_path)
    validated = []
    for index, config in enumerate(configs):
        try:
            _validate_schema_values(
                config,
                schema,
                label=f"warm config {index}",
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        violations = _bounds_violations(config, search_space)
        if violations:
            raise ValueError(
                f"warm config {index} violates finalized SEARCH_SPACE: "
                + json.dumps(violations, ensure_ascii=False)
            )
        validated.append(dict(config))
    return validated, search_space


def select_warm_config_indices(
    population_size: int,
    k_eval: int,
    previous_phase_a: dict,
    *,
    seed: int | None = None,
    mandatory_indices: tuple[int, ...] = (),
) -> dict:
    """Create or replay an auditable selection, optionally pinning controls."""
    if population_size < 1:
        raise ValueError("warm-config population must be non-empty")
    if not 1 <= k_eval <= population_size:
        raise ValueError("k_eval must be within the warm-config population")
    if not isinstance(previous_phase_a, dict):
        raise ValueError("tune_report.phase_a must be an object")
    if (
        any(not isinstance(index, int) or isinstance(index, bool)
            for index in mandatory_indices)
        or len(set(mandatory_indices)) != len(mandatory_indices)
        or any(index < 0 or index >= population_size for index in mandatory_indices)
    ):
        raise ValueError("mandatory warm-config indices must be unique in-range integers")
    if len(mandatory_indices) > k_eval:
        raise ValueError("k_eval cannot be smaller than the mandatory control set")

    previous = previous_phase_a.get("warm_config_selection")
    if previous is not None:
        expected_schema = 2 if mandatory_indices else 1
        if (
            not isinstance(previous, dict)
            or previous.get("schema_version") != expected_schema
        ):
            raise ValueError(
                "phase_a.warm_config_selection schema does not match "
                "the candidate control contract"
            )
        if previous.get("population_size") != population_size:
            raise ValueError(
                "warm-config count changed after selection; edit failed configs in place"
            )
        if previous.get("k_eval") != k_eval:
            raise ValueError("k_eval changed after warm configs were selected")

        method = previous.get("method")
        valid_methods = (
            {"mandatory_then_uniform_without_replacement"}
            if mandatory_indices
            else {"uniform_without_replacement"}
        )
        if method not in valid_methods:
            raise ValueError(f"unknown warm-config selection method: {method!r}")
        permutation = previous.get("permutation")
        if (
            not isinstance(permutation, list)
            or any(
                not isinstance(index, int) or isinstance(index, bool)
                for index in permutation
            )
            or sorted(permutation) != list(range(population_size))
        ):
            raise ValueError(
                "warm-config selection permutation must contain every index exactly once"
            )
        if previous.get("selected_indices") != permutation[:k_eval]:
            raise ValueError("warm-config selected_indices do not match its permutation")
        if previous.get("deferred_indices") != permutation[k_eval:]:
            raise ValueError("warm-config deferred_indices do not match its permutation")
        if mandatory_indices:
            if previous.get("mandatory_indices") != list(mandatory_indices):
                raise ValueError("warm-config mandatory_indices changed after selection")
            if permutation[:len(mandatory_indices)] != list(mandatory_indices):
                raise ValueError("mandatory warm configs must lead the permutation")

        previous_seed = previous.get("seed")
        if (
            not isinstance(previous_seed, int)
            or isinstance(previous_seed, bool)
            or previous_seed < 0
        ):
            raise ValueError("uniform warm-config selection requires a nonnegative seed")
        remaining = [
            index
            for index in range(population_size)
            if index not in mandatory_indices
        ]
        expected = list(mandatory_indices) + random.Random(
            previous_seed
        ).sample(remaining, len(remaining))
        if permutation != expected:
            raise ValueError("warm-config permutation does not match its persisted seed")
        return dict(previous)

    if previous_phase_a:
        raise ValueError(
            "existing phase_a is missing warm_config_selection; start a fresh run"
        )

    if seed is None:
        selection_seed = random.SystemRandom().randrange(1 << 63)
    elif not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("warm-config selection seed must be a nonnegative integer")
    else:
        selection_seed = seed
    remaining = [
        index
        for index in range(population_size)
        if index not in mandatory_indices
    ]
    permutation = list(mandatory_indices) + random.Random(
        selection_seed
    ).sample(remaining, len(remaining))
    method = (
        "mandatory_then_uniform_without_replacement"
        if mandatory_indices
        else "uniform_without_replacement"
    )

    result = {
        "schema_version": 2 if mandatory_indices else 1,
        "method": method,
        "seed": selection_seed,
        "population_size": population_size,
        "k_eval": k_eval,
        "permutation": permutation,
        "selected_indices": permutation[:k_eval],
        "deferred_indices": permutation[k_eval:],
    }
    if mandatory_indices:
        result["mandatory_indices"] = list(mandatory_indices)
    return result


def _literal_default_params(candidate_path: Path) -> dict | None:
    tree = ast.parse(candidate_path.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_PARAMS"
            for target in targets
        ):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, dict) else None
    return None


def validate_provided_baseline_configs(
    candidate_path: Path,
    configs: list,
    k_eval: int | None,
) -> dict:
    """Validate origin and return the warm-control contract for this candidate."""
    brief_path = candidate_path.parent / "_candidate_brief.json"
    try:
        brief = json.loads(brief_path.read_text())
    except OSError as exc:
        raise ValueError(f"warmstart requires candidate brief {brief_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid candidate brief {brief_path}: {exc}") from exc
    schema_version = brief.get("schema_version") if isinstance(brief, dict) else None
    if schema_version not in {3, 4}:
        raise ValueError("warmstart requires a schema-3 or schema-4 candidate brief")
    if (
        schema_version == 4
        and brief.get("run_id") != Path(candidate_path).resolve().parent.name
    ):
        raise ValueError("schema-4 candidate brief run_id does not match its directory")
    source = brief.get("implementation_source")
    if not isinstance(source, dict):
        raise ValueError("candidate brief requires an implementation_source object")
    source_kind = source.get("kind")
    recognized = {"generated", "legacy_generated", "provided_entrypoint"}
    if schema_version == 4:
        recognized.add("primary_parent_snapshot")
    if source_kind not in recognized:
        raise ValueError(
            "candidate brief implementation_source.kind must be generated, "
            "legacy_generated, provided_entrypoint, or primary_parent_snapshot"
        )
    source_run_ids = brief.get("source_run_ids")
    if schema_version == 4:
        if not isinstance(source_run_ids, list):
            raise ValueError("schema-4 candidate brief requires source_run_ids")
        if source_run_ids:
            if source_kind != "primary_parent_snapshot":
                raise ValueError(
                    "schema-4 non-fresh candidate requires a primary-parent snapshot"
                )
            if not isinstance(brief.get("primary_parent"), dict):
                raise ValueError(
                    "schema-4 non-fresh candidate requires primary_parent metadata"
                )
        elif brief.get("primary_parent") is not None:
            raise ValueError("schema-4 fresh candidate cannot declare primary_parent")
    if source_kind != "provided_entrypoint":
        return {
            "brief": brief,
            "schema_version": schema_version,
            "source_kind": source_kind,
            "mandatory_indices": (0,) if schema_version == 4 else (),
            "requires_parameter_transfer": (
                schema_version == 4 and bool(source_run_ids)
            ),
        }
    source_path = source.get("path")
    source_sha256 = source.get("sha256")
    if (
        not isinstance(source_path, str)
        or not source_path
        or not isinstance(source_sha256, str)
        or len(source_sha256) != 71
        or not source_sha256.startswith("sha256:")
        or any(char not in "0123456789abcdef" for char in source_sha256[7:])
    ):
        raise ValueError(
            "provided_entrypoint implementation_source requires path and sha256 receipt"
        )
    if len(configs) != 1:
        raise ValueError("provided baseline requires exactly one warm config")
    if k_eval not in (None, 1):
        raise ValueError("provided baseline requires k_eval=1")
    defaults = _literal_default_params(candidate_path)
    if defaults is None:
        raise ValueError(
            "provided baseline requires a module-level literal DEFAULT_PARAMS"
        )
    if configs[0] != defaults:
        raise ValueError(
            "provided baseline warm config must equal its literal DEFAULT_PARAMS"
        )
    return {
        "brief": brief,
        "schema_version": schema_version,
        "source_kind": source_kind,
        "mandatory_indices": (0,) if schema_version == 4 else (),
        "requires_parameter_transfer": False,
    }


# =============================================================================
# Invocation setup and resumable state
# =============================================================================


@dataclass
class WarmstartRun:
    """Mutable state shared by the named stages of one Phase-A invocation."""

    candidate_path: Path
    report_path: Path
    configs: list[dict]
    deferred: list[dict]
    selected_indices: list[int]
    selection: dict
    parameter_transfer: dict | None
    candidate_code_revision: dict
    search_space: dict
    make_model: Callable[..., Any]
    evaluate: Callable[..., float]
    preflight_enabled: bool
    report: dict
    preflight_report: dict
    cache_rows: dict[str, dict]
    cache: dict[str, float]
    trials_attempted: int
    started: float
    donor_policy_active: bool
    donor_warm_config_index: int | None
    donor_failure_row: dict | None
    donor_preflight_rejection: dict | None
    warm_rows: list[dict] = field(default_factory=list)

    @property
    def phase_a(self) -> dict:
        return self.report["phase_a"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--configs-json", required=True, type=Path,
                        help="JSON list of K warm config dicts (_warm_configs.json)")
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--k-eval", type=int, default=None,
                        help="uniformly sample k-eval configs without replacement for "
                             "evaluation now (best finite row = screening score, "
                             "including an inherited control when it wins); the rest are "
                             "DEFERRED — stored params-only and evaluated later by the deep-tuner "
                             "(BO enqueue / grid prepend) only if this candidate is selected. "
                             "The sampled permutation is persisted for resume. Default = all "
                             "(no deferral).")
    parser.add_argument(
        "--donor-snapshot",
        type=Path,
        default=None,
        help=(
            "generation-bound global donor snapshot (transfer policy pair "
            "only); requires the candidate-local _global_donor_transfer.json "
            "written by inject-global-donor"
        ),
    )
    parser.add_argument(
        "--target-k-eval",
        type=int,
        default=None,
        help=(
            "configured full-fidelity screening target. When the driver assigns "
            "a smaller terminal --k-eval, Phase A records tail_degraded fidelity"
        ),
    )
    return parser


def _load_parameter_transfer(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    all_configs: list[dict],
    control_contract: dict,
) -> dict | None:
    """Load and validate the mandatory primary-parent transfer, if any."""
    if not control_contract["requires_parameter_transfer"]:
        return None

    receipt_path = args.candidate_path.parent / PARAMETER_TRANSFER_FILENAME
    try:
        parameter_transfer = json.loads(receipt_path.read_text())
    except OSError as exc:
        parser.error(
            f"schema-4 non-fresh candidate requires {receipt_path}: {exc}"
        )
    except json.JSONDecodeError as exc:
        parser.error(f"invalid parameter-transfer receipt {receipt_path}: {exc}")
    try:
        validate_parameter_transfer(
            args.candidate_path,
            all_configs,
            parameter_transfer,
        )
    except ValueError as exc:
        parser.error(str(exc))
    return parameter_transfer


def _load_global_donor_transfer(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    all_configs: list[dict],
    candidate_code_revision: dict,
    previous_phase_a: dict,
) -> dict | None:
    """Validate the global-donor receipt against its generation-bound snapshot.

    Returns the candidate-local receipt when the candidate carries one, else
    None (the no_donor binding).  Every inconsistency — missing/stale receipt,
    snapshot mismatch, population drift, or a post-selection binding change —
    fails before BASE_PARAMS is written or any objective slot is consumed
    (design §4.2, §7.1, §8).
    """
    receipt_path = args.candidate_path.parent / GLOBAL_DONOR_TRANSFER_FILENAME
    selection_recorded = isinstance(
        previous_phase_a.get("warm_config_selection"), dict
    )
    if args.donor_snapshot is None:
        if receipt_path.exists():
            parser.error(
                f"global-donor receipt {receipt_path} exists but no "
                "--donor-snapshot was passed (the no_donor binding); refusing "
                "to silently drop the donor"
            )
        if (
            selection_recorded
            and previous_phase_a.get("initialization_mode") == "global_donor"
        ):
            parser.error(
                "phase_a is bound to a global-donor initialization but no "
                "--donor-snapshot was passed"
            )
        return None

    from scheduler import donor as donor_snapshots  # noqa: PLC0415

    try:
        snapshot = donor_snapshots.load_donor_snapshot(args.donor_snapshot)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        receipt = json.loads(receipt_path.read_text())
    except OSError as exc:
        parser.error(
            f"--donor-snapshot requires the helper-written receipt "
            f"{receipt_path}: {exc}"
        )
    except json.JSONDecodeError as exc:
        parser.error(f"invalid global-donor receipt {receipt_path}: {exc}")

    try:
        _validate_global_donor_receipt(
            receipt,
            args.candidate_path.parent.name,
            args.candidate_path,
        )
        if receipt["donor"]["snapshot_id"] != snapshot["snapshot_id"]:
            raise ValueError(
                f"global-donor receipt is bound to snapshot "
                f"{receipt['donor']['snapshot_id']}; the run passed "
                f"{snapshot['snapshot_id']}"
            )
        ordinary_count = receipt.get("ordinary_config_count")
        if (
            not isinstance(ordinary_count, int)
            or isinstance(ordinary_count, bool)
            or ordinary_count < 1
        ):
            raise ValueError(
                "global-donor receipt has an invalid ordinary_config_count"
            )
        if receipt["status"] == "ok":
            donor_index = receipt["warm_config_index"]
            expected = ordinary_count + (0 if receipt["deduplicated"] else 1)
            if len(all_configs) != expected or donor_index >= len(all_configs):
                raise ValueError(
                    "warm-config population does not match the global-donor "
                    "receipt"
                )
            if not receipt["deduplicated"] and donor_index != ordinary_count:
                raise ValueError(
                    "global-donor receipt warm_config_index is inconsistent "
                    "with its ordinary_config_count"
                )
            if all_configs[donor_index] != receipt["projection"]["params"]:
                raise ValueError(
                    "the donor warm row does not match the global-donor "
                    "receipt projection"
                )
        elif len(all_configs) != ordinary_count:
            raise ValueError(
                "warm-config population does not match the global-donor "
                "receipt"
            )
        if selection_recorded:
            # The donor binding freezes with warm selection; the receipt
            # embedded in phase_a is the reference (§8).  A helper-frozen
            # receipt may legitimately carry a stale execution revision after
            # a post-selection code fix, so freshness checks stop applying.
            mode = "global_donor" if receipt["status"] == "ok" else "ordinary"
            if previous_phase_a.get("initialization_mode") != mode:
                raise ValueError(
                    "the donor binding changed after warm selection"
                )
            previous_receipt = previous_phase_a.get("global_donor_transfer")
            if not isinstance(previous_receipt, dict):
                raise ValueError(
                    "phase_a predates the global-donor receipt; refusing to "
                    "bind a donor after warm selection"
                )
            frozen = (
                "status",
                "warm_config_index",
                "deduplicated",
                "dedup_ordinary_index",
                "ordinary_config_count",
            )
            if (
                any(
                    previous_receipt.get(field) != receipt.get(field)
                    for field in frozen
                )
                or previous_receipt.get("donor", {}).get("snapshot_id")
                != receipt["donor"]["snapshot_id"]
                or previous_receipt.get("projection", {}).get("params")
                != receipt["projection"]["params"]
            ):
                raise ValueError(
                    "the global-donor receipt changed after warm selection"
                )
        elif (
            receipt["candidate"]["execution_revision"] != candidate_code_revision
            or receipt["candidate"]["param_schema"]
            != _json_native(_read_param_schema(args.candidate_path))
        ):
            raise ValueError(
                "global-donor receipt is stale relative to the candidate; "
                "re-run inject-global-donor before warm evaluation"
            )
    except ValueError as exc:
        parser.error(str(exc))
    return receipt


def _mandatory_warm_indices(
    control_contract: dict,
    parameter_transfer: dict | None,
    donor_transfer: dict | None,
) -> tuple[int, ...]:
    """Unified mandatory set once lineage and donor receipts are validated (§4.2)."""
    if donor_transfer is None or donor_transfer["status"] != "ok":
        return control_contract["mandatory_indices"]
    indices = {donor_transfer["warm_config_index"]}
    if parameter_transfer is not None:
        # The lineage control keeps its mandatory slot next to the donor row;
        # a deduplicated donor at index 0 collapses the two roles into one.
        indices.add(0)
    return tuple(sorted(indices))


def _restore_warm_score_cache(
    *,
    all_configs: list[dict],
    search_space: dict,
    previous_phase_a: dict,
    candidate_code_revision: dict,
    parameter_transfer: dict | None,
) -> tuple[dict[str, dict], dict[str, float], bool]:
    """Recover only scores bound to the current code and transfer revision."""
    phase_revision_matches = (
        previous_phase_a.get("candidate_code_revision")
        == candidate_code_revision
    )
    current_param_keys = {
        _params_key(cast_params_to_search_space(dict(config), search_space))
        for config in all_configs
    }
    cache_rows: dict[str, dict] = {}

    def admit(trial: dict) -> None:
        if (
            not isinstance(trial, dict)
            or not isinstance(trial.get("params"), dict)
            or not is_finite_score(trial.get("score"))
        ):
            return
        key = _params_key(trial["params"])
        if key not in current_param_keys:
            return
        cache_rows[key] = {
            "params": trial["params"],
            "score": trial["score"],
        }

    if phase_revision_matches:
        previous_rows = previous_phase_a.get("warm_start_configs", [])
        if not isinstance(previous_rows, list):
            previous_rows = []
        for trial in previous_rows:
            if isinstance(trial, dict):
                admit(trial)

    previous_cache = previous_phase_a.get("warm_score_cache")
    if (
        isinstance(previous_cache, dict)
        and set(previous_cache)
        == {"schema_version", "candidate_execution_revision", "rows"}
        and previous_cache.get("schema_version") == 1
        and previous_cache.get("candidate_execution_revision")
        == candidate_code_revision
        and isinstance(previous_cache.get("rows"), list)
    ):
        for trial in previous_cache["rows"]:
            admit(trial)

    previous_transfer = previous_phase_a.get("parameter_transfer")
    if parameter_transfer is not None:
        if (
            not isinstance(previous_transfer, dict)
            or previous_transfer != parameter_transfer
        ):
            cache_rows = {}
    elif previous_transfer is not None:
        cache_rows = {}

    cache = {key: row["score"] for key, row in cache_rows.items()}
    return cache_rows, cache, phase_revision_matches


def _cache_receipt(run: WarmstartRun) -> dict:
    return {
        "schema_version": 1,
        "candidate_execution_revision": run.candidate_code_revision,
        "rows": [run.cache_rows[key] for key in sorted(run.cache_rows)],
    }


def _trial_receipt(run: WarmstartRun, proposed_index: int) -> dict:
    receipt = {
        "proposed_index": proposed_index,
    }
    roles = []
    if run.parameter_transfer is not None and proposed_index == 0:
        roles.append("inherited_control")
    if (
        run.donor_warm_config_index is not None
        and proposed_index == run.donor_warm_config_index
    ):
        roles.append("global_donor")
    # A deduplicated donor row can carry both roles; keep the historical
    # plain string when there is exactly one.
    if len(roles) == 1:
        receipt["role"] = roles[0]
    elif roles:
        receipt["role"] = roles
    return receipt


def _global_donor_observation(run: WarmstartRun) -> dict | None:
    """The persisted donor-row observation (facts only, no eligibility call)."""
    index = run.donor_warm_config_index
    if index is None:
        return None
    for row in run.warm_rows:
        if row.get("proposed_index") != index:
            continue
        if is_finite_score(row.get("score")):
            return {
                "warm_config_index": index,
                "status": "finite",
                "score": row["score"],
                "failure_ref": None,
            }
        return {
            "warm_config_index": index,
            "status": "crash",
            "score": None,
            "failure_ref": row.get("failure_ref"),
        }
    if run.donor_preflight_rejection is not None:
        return {
            "warm_config_index": index,
            "status": "preflight_rejected",
            "score": None,
            "failure_ref": run.donor_preflight_rejection.get("failure_ref"),
        }
    if run.donor_failure_row is not None:
        # Restored crash row whose replay position has not been reached yet.
        return {
            "warm_config_index": index,
            "status": "crash",
            "score": None,
            "failure_ref": run.donor_failure_row.get("failure_ref"),
        }
    return {
        "warm_config_index": index,
        "status": "not_evaluated",
        "score": None,
        "failure_ref": None,
    }


def _stamp_donor_facts(run: WarmstartRun) -> None:
    """Refresh the donor observation and per-outcome row counts (§4.3)."""
    if not run.donor_policy_active:
        return
    finite = crashed = 0
    for row in run.warm_rows:
        if is_finite_score(row.get("score")):
            finite += 1
        elif row.get("status") == "failed":
            crashed += 1
    run.phase_a["k_finite"] = finite
    run.phase_a["k_crashed"] = crashed
    run.phase_a["k_preflight_rejected"] = (
        1 if run.donor_preflight_rejection is not None else 0
    )
    run.phase_a["global_donor_observation"] = _global_donor_observation(run)


def _prepare_run(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> WarmstartRun:
    """Validate inputs, materialize BASE_PARAMS, and persist running Phase A."""

    # Validate every immutable input before the first candidate/report write.
    contract = lint_contract(args.candidate_path, require_base_params=False)
    if not contract["ok"]:
        parser.error(
            "candidate tuning contract is invalid before BASE_PARAMS "
            f"materialization: {json.dumps(contract['errors'], ensure_ascii=False)}"
        )

    with open(args.configs_json) as f:
        all_configs = json.load(f)
    if not isinstance(all_configs, list) or not all_configs:
        parser.error(f"--configs-json must be a non-empty list, got {type(all_configs).__name__}")
    try:
        control_contract = validate_provided_baseline_configs(
            args.candidate_path,
            all_configs,
            args.k_eval,
        )
    except ValueError as exc:
        parser.error(str(exc))

    parameter_transfer = _load_parameter_transfer(
        args,
        parser,
        all_configs,
        control_contract,
    )

    # This is the mandatory data boundary. `check-search-space` remains useful
    # for proposing/validating the space, but skipping it must never let an
    # invalid row mutate BASE_PARAMS, the report, or consume an objective slot.
    try:
        all_configs, _literal_search_space = _validated_warm_configs(
            args.candidate_path,
            all_configs,
        )
        candidate_code_revision = _candidate_execution_revision(
            args.candidate_path
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    # Config 0 is mandatory for schema-4 candidates; remaining eval-now slots
    # are sampled without replacement. Persist and replay the full permutation
    # so a crash/resume never redraws the subset. k_eval=None / >=len means
    # every config is selected (no deferral).
    k_eval = (
        len(all_configs)
        if args.k_eval is None
        else max(1, min(args.k_eval, len(all_configs)))
    )
    target_k_eval = (
        k_eval
        if args.target_k_eval is None
        else max(1, min(args.target_k_eval, len(all_configs)))
    )
    if target_k_eval < k_eval:
        parser.error("--target-k-eval cannot be smaller than --k-eval")
    if parameter_transfer is not None and k_eval < 2:
        parser.error(
            "schema-4 non-fresh candidates require k_eval>=2 to evaluate the "
            "inherited config 0 plus at least one alternative warm config"
        )
    previous_report = read_tune_report(args.tune_report_json)
    previous_phase_a = previous_report.get("phase_a", {})
    previous_target = previous_phase_a.get("screening_target_k_eval")
    if previous_target is not None and previous_target != target_k_eval:
        parser.error("screening target changed after warm configs were selected")

    # The global-donor policy pair binds the generation's donor snapshot here;
    # old policies never read the donor receipt at all.
    try:
        donor_policy_active = _global_donor_policy_active(args.candidate_path)
    except ValueError as exc:
        parser.error(str(exc))
    donor_transfer = None
    if donor_policy_active:
        donor_transfer = _load_global_donor_transfer(
            args,
            parser,
            all_configs=all_configs,
            candidate_code_revision=candidate_code_revision,
            previous_phase_a=previous_phase_a,
        )
    mandatory_indices = _mandatory_warm_indices(
        control_contract,
        parameter_transfer,
        donor_transfer,
    )
    donor_index = (
        donor_transfer["warm_config_index"]
        if donor_transfer is not None and donor_transfer["status"] == "ok"
        else None
    )
    initialization_mode = "global_donor" if donor_index is not None else "ordinary"
    try:
        selection = select_warm_config_indices(
            len(all_configs),
            k_eval,
            previous_phase_a,
            mandatory_indices=mandatory_indices,
        )
    except ValueError as exc:
        parser.error(str(exc))
    selected_indices = selection["selected_indices"]
    deferred_indices = selection["deferred_indices"]
    configs = [all_configs[index] for index in selected_indices]
    deferred = [all_configs[index] for index in deferred_indices]

    # BASE_PARAMS must exist before importing the complete candidate contract;
    # the AST-only validation above intentionally precedes this first write.
    apply_base_params.apply(args.candidate_path, dict(configs[0]))

    train_module, prepare_module = load_candidate_modules(
        args.candidate_path,
        expected_execution_revision=candidate_code_revision,
    )
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)
    preflight_enabled = resolve_preflight_fn(prepare_module, args.candidate_path) is not None

    # Resume cache: configs already scored in a prior run, keyed by params. A
    # config the caller edited (config-invalid fix) gets new params → cache miss
    # → re-evaluated; a config that crashed has no score → re-evaluated; passed
    # configs are reused only under the same parameter-transfer/code revision.
    cache_rows, cache, phase_revision_matches = _restore_warm_score_cache(
        all_configs=all_configs,
        search_space=search_space,
        previous_phase_a=previous_phase_a,
        candidate_code_revision=candidate_code_revision,
        parameter_transfer=parameter_transfer,
    )

    # Restore a persisted donor-only failure from a previous invocation: the
    # donor crash/rejection IS the transfer observation, so resume neither
    # redraws nor re-evaluates it (unlike a fidelity-control crash, which the
    # caller fixes and re-runs).
    donor_failure_row = None
    donor_preflight_rejection = None
    if (
        donor_index is not None
        and not (parameter_transfer is not None and donor_index == 0)
        and phase_revision_matches
        and previous_phase_a.get("parameter_transfer") == parameter_transfer
    ):
        previous_observation = previous_phase_a.get("global_donor_observation")
        if (
            isinstance(previous_observation, dict)
            and previous_observation.get("warm_config_index") == donor_index
        ):
            if previous_observation.get("status") == "crash":
                for row in previous_phase_a.get("warm_start_configs", []):
                    if (
                        isinstance(row, dict)
                        and row.get("proposed_index") == donor_index
                        and not is_finite_score(row.get("score"))
                    ):
                        donor_failure_row = dict(row)
                        break
                if donor_failure_row is None:
                    donor_failure_row = {
                        "params": cast_params_to_search_space(
                            dict(all_configs[donor_index]),
                            search_space,
                        ),
                        "score": None,
                        "status": "failed",
                        "proposed_index": donor_index,
                        "role": "global_donor",
                        "failure_ref": previous_observation.get("failure_ref"),
                    }
            elif previous_observation.get("status") == "preflight_rejected":
                donor_preflight_rejection = previous_observation

    # Rebuild the running Phase-A view while preserving its cumulative budget.
    trials_attempted = previous_phase_a.get("trials_attempted", 0)
    if (
        not isinstance(trials_attempted, int)
        or isinstance(trials_attempted, bool)
        or trials_attempted < 0
    ):
        parser.error("phase_a.trials_attempted must be a nonnegative integer")

    report = previous_report
    if not phase_revision_matches:
        # Phase-C rows and closing fields are observations of the old
        # candidate/evaluator revision.  Keeping them while stamping a new
        # Phase-A revision would falsely re-attribute stale scores to the new
        # code.  Preserve the strict cumulative attempt counter above, but
        # discard every revision-bound derived/terminal artifact.
        report.pop("phase_c", None)
        report.pop("preflight", None)
        for field in (
            "final_best_params",
            "final_best_score",
            "applied_to_base_params",
        ):
            report.pop(field, None)
    preflight_report = report.setdefault("preflight", {"attempts": [], "invocations": 0})
    if preflight_enabled:
        preflight_report["invocations"] = len(preflight_report.get("attempts", []))
        preflight_report["status"] = "running"
    phase_a = {
        "warm_start_configs": [],
        "warm_config_selection": selection,
        "screening_target_k_eval": target_k_eval,
        "screening_actual_k_eval": k_eval,
        "screening_fidelity": (
            "tail_degraded" if k_eval < target_k_eval else "full"
        ),
        # deferred = proposed-but-not-evaluated-now; the deep-tuner evaluates these
        # first (BO enqueue / grid prepend) only if this candidate is promoted.
        "deferred_configs": [{"params": cast_params_to_search_space(dict(d), search_space)}
                             for d in deferred],
        "search_space": search_space_for_json(search_space),
        "trials_attempted": trials_attempted,
        "status": "running",
        "candidate_code_revision": candidate_code_revision,
        # Keep every still-relevant finite row while `warm_start_configs` is
        # replayed incrementally. A kill between cached rows cannot erase the
        # untouched suffix and force duplicate objective calls on the next run.
        "warm_score_cache": {
            "schema_version": 1,
            "candidate_execution_revision": candidate_code_revision,
            "rows": [cache_rows[key] for key in sorted(cache_rows)],
        },
    }
    report["phase_a"] = phase_a
    if parameter_transfer is not None:
        phase_a["parameter_transfer"] = parameter_transfer
        phase_a["inherited_control"] = {
            "warm_config_index": 0,
            "selected": 0 in selected_indices,
            "primary_parent_run_id": parameter_transfer["primary_parent"]["run_id"],
            "parent_incumbent_score": parameter_transfer["primary_parent"][
                "incumbent_score"
            ],
        }
    if donor_policy_active:
        # Facts only: initialization_mode is the single interpretation source
        # for the first tuning bout; eligibility judgments live downstream.
        phase_a["initialization_mode"] = initialization_mode
        if donor_transfer is None:
            phase_a["global_donor_transfer"] = None
        else:
            embedded = dict(donor_transfer)
            if donor_transfer["status"] == "ok":
                # The receipt's reserved fields are computed here, where
                # mandatory roles are owned; the helper-owned file itself
                # stays untouched.
                embedded["mandatory_role_indices"] = list(mandatory_indices)
                embedded["k_eval"] = k_eval
            phase_a["global_donor_transfer"] = embedded
        if donor_index is None:
            observation = None
        elif donor_failure_row is not None:
            observation = {
                "warm_config_index": donor_index,
                "status": "crash",
                "score": None,
                "failure_ref": donor_failure_row.get("failure_ref"),
            }
        elif donor_preflight_rejection is not None:
            observation = {
                "warm_config_index": donor_index,
                "status": "preflight_rejected",
                "score": None,
                "failure_ref": donor_preflight_rejection.get("failure_ref"),
            }
        else:
            observation = {
                "warm_config_index": donor_index,
                "status": "not_evaluated",
                "score": None,
                "failure_ref": None,
            }
        phase_a["global_donor_observation"] = observation
    write_tune_report(args.tune_report_json, report)

    return WarmstartRun(
        candidate_path=args.candidate_path,
        report_path=args.tune_report_json,
        configs=configs,
        deferred=deferred,
        selected_indices=selected_indices,
        selection=selection,
        parameter_transfer=parameter_transfer,
        candidate_code_revision=candidate_code_revision,
        search_space=search_space,
        make_model=make_model,
        evaluate=evaluate,
        preflight_enabled=preflight_enabled,
        report=report,
        preflight_report=preflight_report,
        cache_rows=cache_rows,
        cache=cache,
        trials_attempted=trials_attempted,
        started=time.time(),
        donor_policy_active=donor_policy_active,
        donor_warm_config_index=donor_index,
        donor_failure_row=donor_failure_row,
        donor_preflight_rejection=donor_preflight_rejection,
    )


# =============================================================================
# Sequential evaluation and terminal paths
# =============================================================================


def _preflight_config(
    run: WarmstartRun,
    *,
    params: dict,
    proposed_index: int,
    evaluation_position: int,
    fatal: bool = True,
) -> str:
    """Run and persist one no-score feasibility check.

    Returns "evaluate" to proceed to the objective, "stop" after a fatal
    rejection (the historical fail-closed path), or "skip" when a donor-only
    row was rejected: the rejection is persisted as the donor observation and
    the evaluation loop continues with the remaining selected rows.
    """
    if not run.preflight_enabled:
        return "evaluate"
    try:
        result = timed_preflight(
            params,
            run.candidate_path,
            expected_execution_revision=run.candidate_code_revision,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        sys.stderr.write(tb)
        failure = record_failure(
            report_path=run.report_path,
            candidate_path=run.candidate_path,
            phase="preflight",
            method="warmstart",
            params=params,
            error=exc,
            traceback_text=tb,
        )
        run.preflight_report.setdefault("attempts", []).append(
            {
                "params": params,
                "source": "warmstart",
                "status": "failed",
                **failure,
            }
        )
        run.preflight_report["invocations"] = len(
            run.preflight_report["attempts"]
        )
        if run.donor_warm_config_index == proposed_index:
            run.donor_preflight_rejection = failure
        run.phase_a["warm_start_configs"] = run.warm_rows
        _stamp_donor_facts(run)
        if not fatal:
            # A donor-only rejection consumes no objective slot and does not
            # stop the screening; it is the persisted transfer observation.
            write_tune_report(run.report_path, run.report)
            return "skip"
        run.preflight_report["status"] = "failed"
        run.phase_a["status"] = "preflight_failed"
        write_tune_report(run.report_path, run.report)
        write_json(
            {
                "phase": "preflight",
                "status": "crashed",
                "crash_index": proposed_index,
                "evaluation_position": evaluation_position,
                "crash_params": params,
                "objective_slot_consumed": False,
                **failure,
            }
        )
        return "stop"

    run.preflight_report.setdefault("attempts", []).append(
        {
            "params": params,
            "source": "warmstart",
            "status": "ok",
            "result": result or {"status": "ok"},
        }
    )
    run.preflight_report["invocations"] = len(
        run.preflight_report["attempts"]
    )
    write_tune_report(run.report_path, run.report)
    return "evaluate"


def _finish_budget_exhausted(
    run: WarmstartRun,
    *,
    current_position: int,
    error: EvaluationBudgetExhausted,
) -> int:
    """Close Phase A from cached observations when no new slot is available."""
    if run.preflight_enabled:
        run.preflight_report["status"] = "ok"

    # `recovered` IS the live row list so donor-fact stamping below sees the
    # cache-recovered suffix too (e.g. a donor row scored in a prior run).
    recovered = run.warm_rows
    truly_unscored = []
    for position in range(current_position, len(run.configs)):
        remaining_params = cast_params_to_search_space(
            dict(run.configs[position]),
            run.search_space,
        )
        remaining_key = _params_key(remaining_params)
        if remaining_key in run.cache:
            recovered.append(
                {
                    "params": remaining_params,
                    "score": run.cache[remaining_key],
                    **_trial_receipt(
                        run,
                        run.selected_indices[position],
                    ),
                }
            )
        else:
            truly_unscored.append(run.configs[position])

    unscored = truly_unscored + run.deferred
    run.phase_a["warm_start_configs"] = recovered
    run.phase_a["deferred_configs"] = [
        {
            "params": cast_params_to_search_space(
                dict(config),
                run.search_space,
            )
        }
        for config in unscored
    ]
    selectable = finite_warm_incumbent_rows(recovered)
    if selectable:
        best_params, best_warm_score = min(
            (
                (trial["params"], trial["score"])
                for trial in selectable
            ),
            key=lambda item: item[1],
        )
        apply_base_params.apply(run.candidate_path, best_params)
        elapsed = time.time() - run.started
        run.phase_a.update(
            {
                "best_warm_score": best_warm_score,
                "best_warm_params": best_params,
                "k_evaluated": len(recovered),
                "k_survived": len(recovered),
                "k_deferred": len(unscored),
                "elapsed_seconds": round(elapsed, 1),
                "status": "ok",
                "budget_exhausted": True,
            }
        )
        _stamp_donor_facts(run)
        write_tune_report(run.report_path, run.report)
        write_json(
            {
                "phase": "a",
                "status": "ok",
                "budget_exhausted": True,
                "k_evaluated": len(recovered),
                "trials_attempted": run.trials_attempted,
                "best_warm_score": best_warm_score,
                "best_warm_params": best_params,
                "elapsed_seconds": round(elapsed, 1),
            }
        )
        return 0

    run.phase_a["status"] = "budget_exhausted"
    _stamp_donor_facts(run)
    write_tune_report(run.report_path, run.report)
    write_json(
        {
            "phase": "a",
            "status": "budget_exhausted",
            "reason": f"{error}; no finite warm observation" if recovered else str(error),
            "objective_slot_consumed": False,
        }
    )
    return BUDGET_EXHAUSTED


def _record_objective_failure(
    run: WarmstartRun,
    *,
    params: dict,
    proposed_index: int,
    evaluation_position: int,
    trial_receipt: dict,
    error: Exception,
    fatal: bool = True,
) -> int | None:
    run.trials_attempted += 1
    run.phase_a["trials_attempted"] = run.trials_attempted
    tb = traceback.format_exc()
    sys.stderr.write(tb)
    failure = record_failure(
        report_path=run.report_path,
        candidate_path=run.candidate_path,
        phase="phase_a",
        method="warmstart",
        params=params,
        error=error,
        traceback_text=tb,
    )
    run.warm_rows.append(
        {
            "params": params,
            "score": None,
            "status": "failed",
            **trial_receipt,
            **failure,
        }
    )
    run.phase_a["warm_start_configs"] = run.warm_rows
    _stamp_donor_facts(run)
    if not fatal:
        # The donor row's crash is the transfer observation, not a candidate
        # crash; the remaining selected rows still get evaluated (§4.3).
        write_tune_report(run.report_path, run.report)
        return None
    run.phase_a["status"] = "crashed"
    write_tune_report(run.report_path, run.report)
    write_json(
        {
            "phase": "a",
            "status": "crashed",
            "crash_index": proposed_index,
            "evaluation_position": evaluation_position,
            "crash_params": params,
            **failure,
        }
    )
    return CRASHED


def _finish_phase_a(run: WarmstartRun) -> int:
    """Apply the best finite row and persist the successful Phase A."""
    selectable = finite_warm_incumbent_rows(run.warm_rows)
    if not selectable:
        # Reachable only when every selected row failed non-fatally (the
        # donor-only treatment path); the candidate follows the crash path.
        run.phase_a["warm_start_configs"] = run.warm_rows
        run.phase_a["status"] = "crashed"
        _stamp_donor_facts(run)
        write_tune_report(run.report_path, run.report)
        write_json(
            {
                "phase": "a",
                "status": "crashed",
                "reason": "no finite warm row",
                "k_evaluated": len(run.configs),
                "trials_attempted": run.trials_attempted,
            }
        )
        return CRASHED
    best_params, best_warm_score = min(
        ((trial["params"], trial["score"]) for trial in selectable),
        key=lambda item: item[1],
    )
    apply_base_params.apply(run.candidate_path, best_params)
    elapsed = time.time() - run.started

    run.phase_a.update(
        {
            "best_warm_score": best_warm_score,
            "best_warm_params": best_params,
            "k_evaluated": len(run.configs),
            "k_survived": len(run.warm_rows),
            "k_deferred": len(run.deferred),
            "elapsed_seconds": round(elapsed, 1),
            "status": "ok",
        }
    )
    _stamp_donor_facts(run)
    if run.preflight_enabled:
        run.preflight_report["status"] = "ok"
    write_tune_report(run.report_path, run.report)

    completion = {
        "phase": "a",
        "status": "ok",
        "k_evaluated": len(run.configs),
        "k_survived": len(run.warm_rows),
        "trials_attempted": run.trials_attempted,
        "warm_config_selection": run.selection,
        **(
            {"inherited_control": run.phase_a["inherited_control"]}
            if run.parameter_transfer is not None
            else {}
        ),
        "best_warm_score": best_warm_score,
        "best_warm_params": best_params,
        "elapsed_seconds": round(elapsed, 1),
    }
    if run.donor_policy_active:
        completion["initialization_mode"] = run.phase_a["initialization_mode"]
        completion["global_donor_observation"] = run.phase_a[
            "global_donor_observation"
        ]
    write_json(completion)
    return 0


def _evaluate_selected_configs(run: WarmstartRun) -> int:
    """Evaluate selected configs sequentially, stopping at the first failure."""

    for position, raw in enumerate(run.configs):
        proposed_index = run.selected_indices[position]
        params = cast_params_to_search_space(dict(raw), run.search_space)
        trial_receipt = _trial_receipt(run, proposed_index)
        # A donor-only row is a failable treatment, not a fidelity control:
        # its recorded failure does not stop the remaining selected rows.  A
        # donor row that also carries the lineage control stays fail-closed.
        donor_only = (
            run.donor_warm_config_index is not None
            and proposed_index == run.donor_warm_config_index
            and not (run.parameter_transfer is not None and proposed_index == 0)
        )
        if donor_only and run.donor_failure_row is not None:
            run.warm_rows.append(dict(run.donor_failure_row))
            run.phase_a["warm_start_configs"] = run.warm_rows
            _stamp_donor_facts(run)
            write_tune_report(run.report_path, run.report)
            continue
        if donor_only and run.donor_preflight_rejection is not None:
            continue

        preflight = _preflight_config(
            run,
            params=params,
            proposed_index=proposed_index,
            evaluation_position=position,
            fatal=not donor_only,
        )
        if preflight == "stop":
            return CRASHED
        if preflight == "skip":
            continue

        key = _params_key(params)
        if key in run.cache:
            run.warm_rows.append(
                {
                    "params": params,
                    "score": run.cache[key],
                    **trial_receipt,
                }
            )
        else:
            try:
                score = timed_eval(
                    run.evaluate,
                    run.make_model,
                    params,
                    run.candidate_path,
                    phase="phase_a",
                    method="warmstart",
                )
            except EvaluationBudgetExhausted as exc:
                return _finish_budget_exhausted(
                    run,
                    current_position=position,
                    error=exc,
                )
            except Exception as exc:
                outcome = _record_objective_failure(
                    run,
                    params=params,
                    proposed_index=proposed_index,
                    evaluation_position=position,
                    trial_receipt=trial_receipt,
                    error=exc,
                    fatal=not donor_only,
                )
                if outcome is not None:
                    return outcome
                continue
            run.trials_attempted += 1
            run.phase_a["trials_attempted"] = run.trials_attempted
            run.warm_rows.append(
                {"params": params, "score": score, **trial_receipt}
            )
            run.cache[key] = score
            run.cache_rows[key] = {"params": params, "score": score}
            run.phase_a["warm_score_cache"] = _cache_receipt(run)

        run.phase_a["warm_start_configs"] = run.warm_rows
        _stamp_donor_facts(run)
        write_tune_report(run.report_path, run.report)

    return _finish_phase_a(run)


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    run = _prepare_run(args, parser)
    return _evaluate_selected_configs(run)


if __name__ == "__main__":
    raise SystemExit(main())
