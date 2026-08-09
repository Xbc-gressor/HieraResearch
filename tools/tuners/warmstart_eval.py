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
3. When every selected config has a score (no crash), pick the best selectable
   warm row, excluding an inherited config-0 fidelity control, write it into
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
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
import time
import traceback
from pathlib import Path

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
    PARAMETER_TRANSFER_FILENAME,
    _bounds_violations,
    _candidate_execution_revision,
    _read_param_schema,
    _read_search_space,
    _validate_schema_values,
    finite_warm_incumbent_rows,
    lint_contract,
    validate_parameter_transfer,
)

CRASHED = 3  # a not-yet-scored config raised; the caller diagnoses + fixes + resumes
BUDGET_EXHAUSTED = 4  # no score_fn call was started; coordinator ends the run


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
            {"mandatory_then_uniform_without_replacement", "legacy_prefix_resume"}
            if mandatory_indices
            else {"uniform_without_replacement", "legacy_prefix_resume"}
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
        if method in {
            "uniform_without_replacement",
            "mandatory_then_uniform_without_replacement",
        }:
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
        elif previous_seed is not None:
            raise ValueError("legacy warm-config selection must have a null seed")
        return dict(previous)

    if previous_phase_a:
        # Reports created before selection receipts used the first K_eval
        # configs. Preserve that already-started experiment instead of silently
        # changing its sampled set during an upgrade.
        permutation = list(range(population_size))
        method = "legacy_prefix_resume"
        selection_seed = None
    else:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--configs-json", required=True, type=Path,
                        help="JSON list of K warm config dicts (_warm_configs.json)")
    parser.add_argument("--tune-report-json", required=True, type=Path)
    parser.add_argument("--k-eval", type=int, default=None,
                        help="uniformly sample k-eval configs without replacement for "
                             "evaluation now (best selectable row = screening score; "
                             "an inherited control is observation-only); the rest are "
                             "DEFERRED — stored params-only and evaluated later by the deep-tuner "
                             "(BO enqueue / grid prepend) only if this candidate is selected. "
                             "The sampled permutation is persisted for resume. Default = all "
                             "(no deferral).")
    args = parser.parse_args()

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

    parameter_transfer = None
    if control_contract["requires_parameter_transfer"]:
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

    # This is the mandatory data boundary. `check-search-space` remains useful
    # for proposing/expanding the space, but skipping it must never let an
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
    k_eval = len(all_configs) if args.k_eval is None else max(1, min(args.k_eval, len(all_configs)))
    if parameter_transfer is not None and k_eval < 2:
        parser.error(
            "schema-4 non-fresh candidates require k_eval>=2: inherited "
            "config 0 is a fidelity observation and cannot be the incumbent"
        )
    previous_report = read_tune_report(args.tune_report_json)
    previous_phase_a = previous_report.get("phase_a", {})
    try:
        selection = select_warm_config_indices(
            len(all_configs),
            k_eval,
            previous_phase_a,
            mandatory_indices=control_contract["mandatory_indices"],
        )
    except ValueError as exc:
        parser.error(str(exc))
    selected_indices = selection["selected_indices"]
    deferred_indices = selection["deferred_indices"]
    configs = [all_configs[index] for index in selected_indices]
    deferred = [all_configs[index] for index in deferred_indices]

    # Ensure BASE_PARAMS exists (create on the first run, rewrite later) so
    # load_candidate_modules' REQUIRED_SYMBOLS check passes; AST reads SEARCH_SPACE
    # from the file, so this precedes the import.
    apply_base_params.apply(args.candidate_path, dict(configs[0]))

    train_module, prepare_module = load_candidate_modules(
        args.candidate_path,
        expected_execution_revision=candidate_code_revision,
    )
    search_space = train_module.SEARCH_SPACE
    make_model = train_module.make_model
    evaluate = resolve_score_fn(prepare_module, args.candidate_path)
    preflight_enabled = resolve_preflight_fn(prepare_module, args.candidate_path) is not None
    current_structure_sha256 = candidate_code_revision["structure_sha256"]
    current_execution_revision_sha256 = candidate_code_revision["revision_sha256"]

    # Resume cache: configs already scored in a prior run, keyed by params. A
    # config the caller edited (config-invalid fix) gets new params → cache miss
    # → re-evaluated; a config that crashed has no score → re-evaluated; passed
    # configs are reused only under the same parameter-transfer/code revision.
    prev = previous_phase_a.get("warm_start_configs", [])
    if not isinstance(prev, list):
        prev = []
    previous_code_revision = previous_phase_a.get("candidate_code_revision")
    phase_revision_matches = previous_code_revision == candidate_code_revision
    current_param_keys = {
        _params_key(cast_params_to_search_space(dict(config), search_space))
        for config in all_configs
    }
    cache_rows: dict[str, dict] = {}

    def _admit_cache_row(trial: dict) -> None:
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
        for trial in prev:
            if (
                isinstance(trial, dict)
                and trial.get("candidate_execution_revision_sha256")
                == current_execution_revision_sha256
            ):
                _admit_cache_row(trial)

    previous_cache = previous_phase_a.get("warm_score_cache")
    if (
        isinstance(previous_cache, dict)
        and previous_cache.get("schema_version") == 1
        and previous_cache.get("candidate_execution_revision")
        == candidate_code_revision
        and isinstance(previous_cache.get("rows"), list)
    ):
        for trial in previous_cache["rows"]:
            _admit_cache_row(trial)

    if parameter_transfer is not None:
        previous_transfer = previous_phase_a.get("parameter_transfer")
        if (
            not isinstance(previous_transfer, dict)
            or previous_transfer != parameter_transfer
        ):
            cache_rows = {}
    elif previous_phase_a.get("parameter_transfer") is not None:
        cache_rows = {}
    cache = {
        key: row["score"]
        for key, row in cache_rows.items()
    }

    def _cache_receipt() -> dict:
        return {
            "schema_version": 1,
            "candidate_execution_revision": candidate_code_revision,
            "rows": [
                cache_rows[key]
                for key in sorted(cache_rows)
            ],
        }

    trials_attempted = previous_phase_a.get("trials_attempted")
    if not isinstance(trials_attempted, int) or isinstance(trials_attempted, bool) \
            or trials_attempted < 0:
        # Backward-compatible recovery for reports written before the explicit
        # attempt counter: every persisted warm row came from one score_fn call.
        trials_attempted = len(prev)

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
    report["phase_a"] = {
        "warm_start_configs": [],
        "warm_config_selection": selection,
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
        "warm_score_cache": _cache_receipt(),
    }
    if parameter_transfer is not None:
        report["phase_a"]["parameter_transfer"] = parameter_transfer
        report["phase_a"]["inherited_control"] = {
            "warm_config_index": 0,
            "selected": 0 in selected_indices,
            "primary_parent_run_id": parameter_transfer["primary_parent"]["run_id"],
            "parent_incumbent_score": parameter_transfer["primary_parent"][
                "incumbent_score"
            ],
            "params_sha256": parameter_transfer["projection"]["params_sha256"],
            "receipt_sha256": parameter_transfer["receipt_sha256"],
        }
    write_tune_report(args.tune_report_json, report)

    started = time.time()
    wsc: list[dict] = []

    def _trial_receipt(proposed_index: int) -> dict:
        receipt = {
            "proposed_index": proposed_index,
            "candidate_structure_sha256": current_structure_sha256,
            "candidate_execution_revision_sha256": (
                current_execution_revision_sha256
            ),
        }
        if parameter_transfer is not None and proposed_index == 0:
            receipt.update(
                {
                    "role": "inherited_control",
                    "parameter_transfer_receipt_sha256": parameter_transfer[
                        "receipt_sha256"
                    ],
                    "params_sha256": parameter_transfer["projection"][
                        "params_sha256"
                    ],
                }
            )
        return receipt

    for i, raw in enumerate(configs):
        proposed_index = selected_indices[i]
        params = cast_params_to_search_space(dict(raw), search_space)
        trial_receipt = _trial_receipt(proposed_index)
        if preflight_enabled:
            try:
                result = timed_preflight(
                    params,
                    args.candidate_path,
                    expected_execution_revision=candidate_code_revision,
                )
            except Exception as exc:
                tb = traceback.format_exc()
                sys.stderr.write(tb)
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="preflight",
                    method="warmstart",
                    params=params,
                    error=exc,
                    traceback_text=tb,
                )
                preflight_report.setdefault("attempts", []).append(
                    {
                        "params": params,
                        "source": "warmstart",
                        "status": "failed",
                        **failure,
                    }
                )
                preflight_report["invocations"] = len(preflight_report["attempts"])
                preflight_report["status"] = "failed"
                report["phase_a"]["warm_start_configs"] = wsc
                report["phase_a"]["status"] = "preflight_failed"
                write_tune_report(args.tune_report_json, report)
                write_json(
                    {
                        "phase": "preflight",
                        "status": "crashed",
                        "crash_index": proposed_index,
                        "evaluation_position": i,
                        "crash_params": params,
                        "objective_slot_consumed": False,
                        **failure,
                    }
                )
                return CRASHED
            preflight_report.setdefault("attempts", []).append(
                {
                    "params": params,
                    "source": "warmstart",
                    "status": "ok",
                    "result": result or {"status": "ok"},
                }
            )
            preflight_report["invocations"] = len(preflight_report["attempts"])
            write_tune_report(args.tune_report_json, report)

        key = _params_key(params)
        if key in cache:
            wsc.append({"params": params, "score": cache[key], **trial_receipt})
        else:
            try:
                score = timed_eval(
                    evaluate,
                    make_model,
                    params,
                    args.candidate_path,
                    phase="phase_a",
                    method="warmstart",
                )
            except EvaluationBudgetExhausted as exc:
                if preflight_enabled:
                    preflight_report["status"] = "ok"
                recovered = list(wsc)
                truly_unscored = []
                for remaining_position in range(i, len(configs)):
                    remaining_params = cast_params_to_search_space(
                        dict(configs[remaining_position]),
                        search_space,
                    )
                    remaining_key = _params_key(remaining_params)
                    if remaining_key in cache:
                        recovered.append({
                            "params": remaining_params,
                            "score": cache[remaining_key],
                            **_trial_receipt(
                                selected_indices[remaining_position]
                            ),
                        })
                    else:
                        truly_unscored.append(configs[remaining_position])
                unscored = truly_unscored + deferred
                report["phase_a"]["warm_start_configs"] = recovered
                report["phase_a"]["deferred_configs"] = [
                    {
                        "params": cast_params_to_search_space(
                            dict(config),
                            search_space,
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
                    apply_base_params.apply(args.candidate_path, best_params)
                    elapsed = time.time() - started
                    report["phase_a"].update(
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
                    write_tune_report(args.tune_report_json, report)
                    write_json(
                        {
                            "phase": "a",
                            "status": "ok",
                            "budget_exhausted": True,
                            "k_evaluated": len(recovered),
                            "trials_attempted": trials_attempted,
                            "best_warm_score": best_warm_score,
                            "best_warm_params": best_params,
                            "elapsed_seconds": round(elapsed, 1),
                        }
                    )
                    return 0
                report["phase_a"]["status"] = "budget_exhausted"
                write_tune_report(args.tune_report_json, report)
                write_json(
                    {
                        "phase": "a",
                        "status": "budget_exhausted",
                        "reason": (
                            f"{exc}; no finite selectable warm observation "
                            "beyond the inherited fidelity control"
                            if recovered
                            else str(exc)
                        ),
                        "objective_slot_consumed": False,
                    }
                )
                return BUDGET_EXHAUSTED
            except Exception as exc:
                trials_attempted += 1
                report["phase_a"]["trials_attempted"] = trials_attempted
                tb = traceback.format_exc()
                sys.stderr.write(tb)
                failure = record_failure(
                    report_path=args.tune_report_json,
                    candidate_path=args.candidate_path,
                    phase="phase_a",
                    method="warmstart",
                    params=params,
                    error=exc,
                    traceback_text=tb,
                )
                wsc.append({
                    "params": params,
                    "score": None,
                    "status": "failed",
                    **trial_receipt,
                    **failure,
                })
                report["phase_a"]["warm_start_configs"] = wsc
                report["phase_a"]["status"] = "crashed"
                write_tune_report(args.tune_report_json, report)
                write_json({"phase": "a", "status": "crashed",
                            "crash_index": proposed_index,
                            "evaluation_position": i,
                            "crash_params": params,
                            **failure})
                return CRASHED
            trials_attempted += 1
            report["phase_a"]["trials_attempted"] = trials_attempted
            wsc.append({"params": params, "score": score, **trial_receipt})
            cache[key] = score
            cache_rows[key] = {"params": params, "score": score}
            report["phase_a"]["warm_score_cache"] = _cache_receipt()
        report["phase_a"]["warm_start_configs"] = wsc
        write_tune_report(args.tune_report_json, report)

    # ---- every sampled config scored → best selectable row → BASE_PARAMS ----
    selectable = finite_warm_incumbent_rows(wsc)
    if not selectable:
        raise RuntimeError(
            "warm evaluation produced no finite selectable row beyond the "
            "inherited fidelity control"
        )
    best_params, best_warm_score = min(
        ((t["params"], t["score"]) for t in selectable),
        key=lambda t: t[1],
    )
    apply_base_params.apply(args.candidate_path, best_params)
    elapsed = time.time() - started

    report["phase_a"].update({
        "best_warm_score": best_warm_score,
        "best_warm_params": best_params,
        "k_evaluated": len(configs),
        "k_survived": len(wsc),
        "k_deferred": len(deferred),
        "elapsed_seconds": round(elapsed, 1),
        "status": "ok",
    })
    if preflight_enabled:
        preflight_report["status"] = "ok"
    write_tune_report(args.tune_report_json, report)

    write_json({
        "phase": "a",
        "status": "ok",
        "k_evaluated": len(configs),
        "k_survived": len(wsc),
        "trials_attempted": trials_attempted,
        "warm_config_selection": selection,
        **(
            {"inherited_control": report["phase_a"]["inherited_control"]}
            if parameter_transfer is not None
            else {}
        ),
        "best_warm_score": best_warm_score,
        "best_warm_params": best_params,
        "elapsed_seconds": round(elapsed, 1),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
