"""Pure tuning-report projections and lineage binding for the run ledger.

The functions here validate and transform in-memory ledger data.  They never
write ``ledger.json``; committing the resulting record remains the job of
``tools/ledger.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from ledger_core import (
    TUNING_FIELDS,
    best_kept_value,
    get_record,
    is_improvement,
    json_digest,
    json_native,
)
from semantic_evidence import (
    unbound_primary_descendants,
    validate_parameter_transfer_binding,
)


TUNER_DIR = Path(__file__).resolve().parent / "tuners"


def _load_tune_tools():
    if str(TUNER_DIR) not in sys.path:
        sys.path.insert(0, str(TUNER_DIR))
    import tune_tools

    return tune_tools


def tuning_record_from_report(report_path: Path) -> dict:
    """Project a Phase-A report onto the ledger tuning fields."""
    tune_tools = _load_tune_tools()
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    tune_tools.validate_phase_a_candidate_state(
        report,
        report_path.parent / "train.py",
    )
    stages = report.get("phase_c", {}).get("stages", [])
    if stages:
        raise ValueError(
            "Phase-A tuning updates cannot consume a report with Phase-C "
            "stages; use tools/finalize_tuning.py after a finalizable close"
        )
    fields = tune_tools.tuning_record(report)
    phase_a = report.get("phase_a")
    fields["applied_incumbent"] = (
        applied_incumbent_from_report(report_path)
        if isinstance(phase_a, dict) and phase_a.get("status") == "ok"
        else None
    )
    return {key: fields.get(key) for key in TUNING_FIELDS}


def finalized_tuning_record_from_report(
    report_path: Path,
    *,
    validate_revision: bool = True,
) -> dict:
    """Project a terminal Phase-C report onto the ledger tuning fields."""
    tune_tools = _load_tune_tools()
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    tune_tools.validate_report_trial_rows(report, report_path.parent / "train.py")
    ledger_guess = report_path.parent.parent.parent / "ledger.json"
    fields = tune_tools.finalized_tuning_record(
        report,
        tuned_threshold=tune_tools.load_tuned_threshold(ledger_guess),
    )
    fields["applied_incumbent"] = applied_incumbent_from_report(
        report_path,
        require_final=True,
        validate_revision=validate_revision,
    )
    return fields


def applied_incumbent_from_report(
    report_path: Path,
    *,
    require_final: bool = False,
    validate_revision: bool = True,
) -> dict:
    """Snapshot the exact applied candidate state represented by a report."""
    tune_tools = _load_tune_tools()
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if require_final:
        result = tune_tools.finalizable_tuning_result(report, require_applied=True)
        params = result["best_params"]
        score = float(result["best_score"])
        source = "finalized_phase_c"
    else:
        closing_present = any(
            report.get(key) is not None
            for key in ("final_best_params", "final_best_score")
        )
        if closing_present:
            result = tune_tools.finalizable_tuning_result(
                report, require_applied=True
            )
            params = result["best_params"]
            score = float(result["best_score"])
            source = "finalized_phase_c"
        else:
            phase_a_best = tune_tools.validated_phase_a_incumbent(report)
            params = phase_a_best["params"]
            score = phase_a_best["score"]
            source = "applied_phase_a"

    candidate_path = report_path.parent / "train.py"
    if not candidate_path.is_file():
        raise ValueError(
            f"cannot snapshot applied incumbent without candidate {candidate_path}"
        )
    contract = tune_tools.lint_contract(candidate_path)
    if not contract.get("ok"):
        details = "; ".join(
            str(error.get("detail", error))
            for error in contract.get("errors", [])
            if isinstance(error, dict)
        )
        raise ValueError(
            "candidate tuning contract is invalid"
            + (f": {details}" if details else "")
        )
    if validate_revision:
        tune_tools.validate_candidate_execution_revision(report, candidate_path)

    param_schema = tune_tools._read_literal_mapping(candidate_path, "PARAM_SCHEMA")
    search_space = tune_tools._read_literal_mapping(candidate_path, "SEARCH_SPACE")
    phase_a = report.get("phase_a")
    reported_search_space = (
        phase_a.get("search_space") if isinstance(phase_a, dict) else None
    )
    if not isinstance(reported_search_space, dict):
        raise ValueError("tune report phase_a.search_space must be an object")
    try:
        reported_search_space = json_native(reported_search_space)
        search_space = json_native(search_space)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"tune report phase_a.search_space is not canonical JSON: {exc}"
        ) from exc
    if reported_search_space != search_space:
        raise ValueError(
            "tune report phase_a.search_space does not match the candidate "
            "SEARCH_SPACE literal"
        )

    tune_tools._validate_schema_values(
        params,
        param_schema,
        label="applied incumbent params",
    )
    violations = tune_tools._bounds_violations(params, search_space)
    if violations:
        raise ValueError(
            "applied incumbent params violate the candidate SEARCH_SPACE: "
            + json.dumps(violations, ensure_ascii=False)
        )
    if tune_tools._read_literal_mapping(candidate_path, "BASE_PARAMS") != params:
        raise ValueError(
            "candidate BASE_PARAMS do not equal the report's applied incumbent"
        )
    snapshot = {
        "schema_version": 1,
        "source": source,
        "score": score,
        "params": json_native(params),
        "param_schema": json_native(param_schema),
        "entrypoint_sha256": (
            "sha256:" + hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        ),
        "tune_report_sha256": (
            "sha256:" + hashlib.sha256(report_path.read_bytes()).hexdigest()
        ),
    }
    return snapshot


def validate_tuning_report_ownership(
    ledger_path: Path,
    run_id: str,
    report_path: Path,
) -> Path:
    """Bind a tuning report to exactly one candidate in exactly one run."""
    ledger_path = Path(ledger_path).resolve()
    report_path = Path(report_path).resolve()
    expected_candidate_dir = ledger_path.parent / "candidates" / str(run_id)
    expected_report = expected_candidate_dir / "tune_report.json"
    if report_path != expected_report:
        raise ValueError(
            "tune report must be the candidate-owned "
            f"{expected_report}; got {report_path}"
        )
    candidate_path = expected_candidate_dir / "train.py"
    if not candidate_path.is_file():
        raise ValueError(f"candidate entrypoint does not exist: {candidate_path}")
    return candidate_path


def capture_transfer_parent_snapshot(data: dict, child: dict) -> None:
    """Persist the exact parent revision cited by one transfer receipt."""
    transfer = child.get("parameter_transfer")
    receipt = transfer.get("receipt") if isinstance(transfer, dict) else None
    primary = receipt.get("primary_parent") if isinstance(receipt, dict) else None
    if not isinstance(primary, dict):
        return
    parent_run_id = str(primary.get("run_id"))
    record_hash = primary.get("ledger_record_sha256")
    snapshots = data.setdefault("lineage_snapshots", [])
    if not isinstance(snapshots, list):
        raise ValueError("ledger.lineage_snapshots must be a list")
    existing = [
        item
        for item in snapshots
        if isinstance(item, dict)
        and item.get("parent_run_id") == parent_run_id
        and item.get("ledger_record_sha256") == record_hash
    ]
    if existing:
        if len(existing) != 1:
            raise ValueError(
                f"parent revision {parent_run_id}/{record_hash} is duplicated"
            )
        return
    parent = get_record(data, parent_run_id)
    if parent is None or json_digest(parent) != record_hash:
        raise ValueError(
            f"record {child.get('run_id')} cites a parent revision that is "
            "neither current nor durably snapshotted"
        )
    score = parent.get("final_best_score")
    applied = parent.get("applied_incumbent")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not isinstance(applied, dict)
    ):
        raise ValueError(
            f"record {child.get('run_id')} cannot snapshot parent {parent_run_id} "
            "without a finite score and exact applied incumbent"
        )
    snapshots.append(
        {
            "schema_version": 1,
            "kind": "parameter_transfer_parent_snapshot",
            "parent_run_id": parent_run_id,
            "ledger_record_sha256": record_hash,
            "final_best_score": float(score),
            "applied_incumbent": copy.deepcopy(applied),
            "captured_by_run_id": str(child.get("run_id")),
        }
    )


def preserve_descendant_bindings(data: dict, run_id: str) -> None:
    """Snapshot every settled primary-child binding before parent mutation."""
    target = str(run_id)
    unbound = set(unbound_primary_descendants(data, target))
    for child in data.get("records", []):
        if not isinstance(child, dict):
            continue
        child_run_id = str(child.get("run_id"))
        if child_run_id in unbound:
            continue
        parents = child.get("source_run_ids")
        if not isinstance(parents, list) or not parents or str(parents[0]) != target:
            continue
        transfer = child.get("parameter_transfer")
        receipt = transfer.get("receipt") if isinstance(transfer, dict) else None
        primary = receipt.get("primary_parent") if isinstance(receipt, dict) else None
        if not isinstance(primary, dict) or str(primary.get("run_id")) != target:
            continue
        try:
            capture_transfer_parent_snapshot(data, child)
        except ValueError:
            unbound.add(child_run_id)
    if unbound:
        raise ValueError(
            f"record {run_id} has in-flight or invalid primary descendants "
            f"{sorted(unbound)}; finish their parameter-transfer binding before "
            "mutating the parent"
        )


def prospective_finalized_record(
    data: dict,
    record: dict,
    report_path: Path,
    *,
    strict_attempts: int,
    validate_revision: bool = True,
) -> dict:
    """Build and validate the exact record a tuning close would persist."""
    finalized = finalized_tuning_record_from_report(
        report_path,
        validate_revision=validate_revision,
    )
    updates = {key: finalized.get(key) for key in TUNING_FIELDS}
    updates["trials_attempted"] = max(
        int(updates.get("trials_attempted") or 0),
        int(strict_attempts),
    )
    score = float(finalized["final_best_score"])
    already_closed = (
        record.get("tune") is True
        and record.get("metric") == data.get("metric")
        and isinstance(record.get("final_best_score"), (int, float))
        and not isinstance(record.get("final_best_score"), bool)
        and math.isfinite(float(record["final_best_score"]))
        and float(record["final_best_score"]) == score
        and all(record.get(key) == value for key, value in updates.items())
    )
    prospective = copy.deepcopy(record)
    if not already_closed:
        prospective["metric"] = data.get("metric")
        prospective["final_best_score"] = score
        prospective["status"] = (
            "keep"
            if is_improvement(
                score,
                best_kept_value(
                    data, exclude_run_id=str(record.get("run_id"))
                ),
            )
            else "discard"
        )
        prospective.update(updates)
        prospective["tune"] = True

    transfer_errors = validate_parameter_transfer_binding(data, prospective)
    if transfer_errors:
        raise ValueError("; ".join(transfer_errors))
    return prospective
