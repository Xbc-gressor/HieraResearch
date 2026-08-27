"""Run-global donor snapshots for the transfer scheduler (design §3.1).

Under the ``anchor_transfer_challenger_v1`` x ``hebo24-transfer10-hebo10``
policy pair, later candidates warm-start from one run-global donor: the best
candidate that completed and finalized at least one real Phase-C segment.
This module freezes the eligible donor frontier as a content-addressed
immutable artifact under ``<run_dir>/.scheduler/donors/donor-<digest>.json``
so a whole generation of candidates binds to the same donor even if the
ledger keeps moving between seats.

Selection rules (design §3.1):

1. the ledger record is not a crash, has ``tune=true`` and
   ``tuning_bouts >= 1`` — a warm-only winner never becomes a donor;
2. ``final_best_score`` is finite and the applied incumbent comes from a
   finalized tune report;
3. the authoritative candidate files/report reproduce the ledger's
   ``applied_incumbent`` snapshot (the same byte-level cross-check the
   primary-parent inheritance helper performs);
4. the selected donor minimizes ``(final_best_score, run_id)``.

A record that fails a rule is excluded from the frontier; the reasons are
returned (not persisted) for post-hoc analysis.  The snapshot id digests the
canonical payload minus ``snapshot_id`` itself, so rebuilding over the same
ledger state is idempotent: same content, same id, no rewrite.  With no
eligible donor the build returns ``status=no_donor`` and writes nothing —
fabricating an empty snapshot would make a missing donor look like a bound
one.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_KIND = "global_donor_snapshot"
SNAPSHOT_ID_PREFIX = "donor-"
SCHEDULER_DIRNAME = ".scheduler"
DONORS_DIRNAME = "donors"
SELECTION_RULE = "min(final_best_score, run_id)"


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _content_digest(value) -> str:
    return hashlib.blake2b(_canonical_json(value), digest_size=12).hexdigest()


def donors_dir(run_dir: Path) -> Path:
    return Path(run_dir) / SCHEDULER_DIRNAME / DONORS_DIRNAME


def snapshot_id_for(payload: dict) -> str:
    """Content id of a snapshot payload (everything except ``snapshot_id``)."""
    return SNAPSHOT_ID_PREFIX + _content_digest(payload)


def verify_donor_snapshot(snapshot: dict) -> dict:
    """Return the snapshot, raising ValueError unless it is internally consistent."""
    if not isinstance(snapshot, dict):
        raise ValueError("donor snapshot must be an object")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("donor snapshot schema_version must be 1")
    if snapshot.get("kind") != SNAPSHOT_KIND:
        raise ValueError(f"donor snapshot kind must be {SNAPSHOT_KIND!r}")
    snapshot_id = snapshot.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.startswith(
        SNAPSHOT_ID_PREFIX
    ):
        raise ValueError("donor snapshot has an invalid snapshot_id")
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    if snapshot_id_for(payload) != snapshot_id:
        raise ValueError(
            "donor snapshot id does not match its canonical payload; the "
            "artifact is corrupt or was edited"
        )
    eligible = snapshot.get("eligible")
    if not isinstance(eligible, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("run_id"), str)
        for entry in eligible
    ):
        raise ValueError("donor snapshot eligible must be a list of donor facts")
    selected = snapshot.get("selected")
    if (
        not isinstance(selected, dict)
        or not isinstance(selected.get("run_id"), str)
        or not isinstance(selected.get("params"), dict)
        or not isinstance(selected.get("param_schema"), dict)
        or not isinstance(selected.get("score"), (int, float))
        or isinstance(selected.get("score"), bool)
        or not math.isfinite(float(selected["score"]))
    ):
        raise ValueError("donor snapshot selected lacks a complete donor state")
    return snapshot


def load_donor_snapshot(path: Path) -> dict:
    """Read and verify an immutable donor snapshot; corruption is a hard error."""
    path = Path(path)
    try:
        snapshot = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read donor snapshot {path}: {exc}") from exc
    return verify_donor_snapshot(snapshot)


def _load_tune_tools():
    # Lazy, mirroring ledger_tuning._load_tune_tools: tools/scheduler must not
    # import the tuner module at module import time (the tuners package is a
    # namespace package under tools/).
    tools_dir = Path(__file__).resolve().parent.parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from tuners import tune_tools  # noqa: PLC0415

    return tune_tools


def _eligibility_failure(record: dict) -> str | None:
    """Rules 1-2: lifecycle shape and a finite finalized score."""
    if record.get("status") == "crash":
        return "crash record"
    if record.get("tune") is not True:
        return "tune is not true"
    bouts = record.get("tuning_bouts")
    if isinstance(bouts, bool) or not isinstance(bouts, int) or bouts < 1:
        return "tuning_bouts < 1 (warm-only)"
    score = record.get("final_best_score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        return "final_best_score is not finite"
    return None


def build_donor_snapshot(run_dir: Path) -> dict:
    """Freeze the current eligible donor frontier as an immutable artifact.

    Returns ``{"status": "ok", "snapshot_id", "path", "snapshot", "excluded"}``
    with ``path`` relative to ``run_dir``, or ``{"status": "no_donor", ...}``
    without writing anything when no candidate qualifies.
    """
    run_dir = Path(run_dir).resolve()
    ledger_path = run_dir / "ledger.json"
    try:
        ledger = json.loads(ledger_path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read run ledger {ledger_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid run ledger {ledger_path}: {exc}") from exc
    records = ledger.get("records", []) if isinstance(ledger, dict) else []

    tune_tools = _load_tune_tools()
    eligible: list[dict] = []
    excluded: list[dict] = []
    incumbents: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            continue
        reason = _eligibility_failure(record)
        incumbent = None
        if reason is None:
            train_path = run_dir / "candidates" / run_id / "train.py"
            if not train_path.is_file():
                reason = "candidate entrypoint is missing"
            else:
                try:
                    # Rule 3: the same byte-level applied-incumbent cross-check
                    # the inheritance helper performs (report, BASE_PARAMS,
                    # ledger record, and the ledger's applied_incumbent
                    # snapshot must all agree with the files on disk).
                    incumbent = tune_tools.authoritative_parent_incumbent(
                        train_path,
                        run_id,
                    )
                except ValueError as exc:
                    reason = f"applied incumbent not reproducible: {exc}"
                else:
                    # Rule 2: the applied incumbent must come from a finalized
                    # Phase-C close, not a pre-close Phase-A snapshot.
                    if incumbent["source"] != "finalized_phase_c":
                        reason = "applied incumbent is not a finalized Phase-C close"
        if reason is not None:
            excluded.append({"run_id": run_id, "reason": reason})
            continue
        score = float(record["final_best_score"])
        eligible.append(
            {
                "run_id": run_id,
                "final_best_score": score,
                "tuning_bouts": int(record["tuning_bouts"]),
                "evaluation_depth": record.get("evaluation_depth"),
                # Pins the exact record content this selection saw; ledger
                # records legitimately change with later bouts.
                "ledger_record_digest": _content_digest(record),
            }
        )
        incumbents[run_id] = incumbent

    if not eligible:
        return {
            "status": "no_donor",
            "snapshot_id": None,
            "path": None,
            "excluded": excluded,
        }

    # Rule 4; the frontier list itself is stored in selection order.
    eligible.sort(key=lambda entry: (entry["final_best_score"], entry["run_id"]))
    chosen = eligible[0]
    incumbent = incumbents[chosen["run_id"]]
    train_path = run_dir / "candidates" / chosen["run_id"] / "train.py"
    selected = {
        "run_id": chosen["run_id"],
        "score": chosen["final_best_score"],
        "params": json.loads(_canonical_json(incumbent["params"])),
        "param_schema": json.loads(_canonical_json(incumbent["param_schema"])),
        "entrypoint_revision": {
            "path": train_path.relative_to(run_dir).as_posix(),
            "sha256": incumbent["train_sha256"],
        },
        "tune_report_digest": incumbent["tune_report_sha256"],
    }
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "kind": SNAPSHOT_KIND,
        "selection_rule": SELECTION_RULE,
        "eligible": eligible,
        "selected": selected,
    }
    snapshot_id = snapshot_id_for(payload)
    snapshot = {"snapshot_id": snapshot_id, **payload}
    path = donors_dir(run_dir) / f"{snapshot_id}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                snapshot,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)
    return {
        "status": "ok",
        "snapshot_id": snapshot_id,
        "path": path.relative_to(run_dir).as_posix(),
        "snapshot": snapshot,
        "excluded": excluded,
    }
