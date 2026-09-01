#!/usr/bin/env python3
"""Bout journal and keep/revert adjudication for the rewrite loop.

A rewrite bout edits <candidate>/train.py in place; this module is the
deterministic trust anchor around that edit:

- snapshot()/revert() — byte-exact pre-edit copy and rollback;
- bouts.jsonl — append-only journal, one entry per adjudicated bout:
  {bout, attempt_id, score, outcome, summary, basis, snapshot};
- current_best() — the measured reference point: the imported baseline score
  versus every kept bout's score (lower is better);
- classify() — an improvement only counts when it clears the noise margin;
- the `finalize` CLI — classify one bout, roll back byte-exactly unless kept,
  and append the journal entry.

Usage:
    python tools/rewrite_bout.py finalize --candidate <dir> --bout <N> \
        [--score S | --nonfinite] --noise-margin E [--attempt-id A] \
        --summary "..." --basis "..."
    python tools/rewrite_bout.py snapshot --candidate <dir> --bout <N>
    python tools/rewrite_bout.py revert --candidate <dir> --snapshot <path>
    python tools/rewrite_bout.py changed --candidate <dir> --snapshot <path>
    python tools/rewrite_bout.py current-best --candidate <dir>
    python tools/rewrite_bout.py journal --candidate <dir> --bout <N> \
        --outcome noop|reverted_crash --summary "..." --basis "..."

finalize stdout: {"outcome", "best"} — best is the post-finalize reference
point; a negative --noise-margin is rejected. snapshot prints the snapshot
path — when the bout's snapshot already exists (the previous attempt at this
bout was interrupted mid-edit) it restores train.py from the snapshot instead
of re-snapshotting the possibly dirty file;
changed exits 0 when train.py is byte-identical to the snapshot and 1
otherwise (a missing file or snapshot is exit 1 with a message);
current-best prints the float (fails when there is no finite reference);
journal appends a non-scored entry (score/attempt_id null) for noop and
preflight-class-failure bouts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REWRITE_DIR = "_rewrite"
TRAIN_PY = "train.py"


def _rewrite_dir(candidate_dir) -> Path:
    return Path(candidate_dir) / REWRITE_DIR


def _snapshot_path(candidate_dir: Path, bout: int) -> Path:
    return _rewrite_dir(candidate_dir) / f"bout-{bout:03d}.pre.py"


def snapshot(candidate_dir, bout: int) -> Path:
    """Byte-exact copy of train.py before bout ``bout``'s edit.

    An already-existing snapshot means a previous attempt at this bout was
    interrupted mid-edit (e.g. the driver was killed): train.py may carry an
    unverified edit, so restore it from the snapshot instead of re-snapshotting
    the dirty file — an unverified edit must never become the baseline.
    """
    candidate_dir = Path(candidate_dir)
    target = _snapshot_path(candidate_dir, bout)
    if target.is_file():
        revert(candidate_dir, target)
        return target
    target.parent.mkdir(exist_ok=True)
    target.write_bytes((candidate_dir / TRAIN_PY).read_bytes())
    return target


def revert(candidate_dir, snap) -> None:
    """Restore train.py byte-exactly from a snapshot (atomic replace)."""
    candidate_dir = Path(candidate_dir)
    target = candidate_dir / TRAIN_PY
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_bytes(Path(snap).read_bytes())
    tmp.replace(target)


def changed(candidate_dir, snap) -> bool:
    """True when train.py differs byte-wise from the snapshot.

    The loop's receipt/file agreement check: the editor's ``edited`` claim
    is only trusted when the bytes say the same thing.
    """
    candidate_dir = Path(candidate_dir)
    target = candidate_dir / TRAIN_PY
    snap = Path(snap)
    if not target.is_file():
        raise ValueError(f"train.py missing: {target}")
    if not snap.is_file():
        raise ValueError(f"snapshot missing: {snap}")
    return target.read_bytes() != snap.read_bytes()


def _bouts_path(candidate_dir) -> Path:
    return _rewrite_dir(candidate_dir) / "bouts.jsonl"


def load_bouts(candidate_dir) -> list[dict]:
    path = _bouts_path(candidate_dir)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def append_bout(candidate_dir, entry: dict) -> None:
    path = _bouts_path(candidate_dir)
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def current_best(candidate_dir) -> float:
    """min(_import.json baseline_score, every kept bout's score).

    Raises ValueError when neither a finite baseline nor any kept score
    exists — adjudication without a measured reference is meaningless.
    """
    candidate_dir = Path(candidate_dir)
    best = None
    manifest_path = candidate_dir / "_import.json"
    if manifest_path.is_file():
        baseline = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "baseline_score"
        )
        if _finite(baseline):
            best = float(baseline)
    for entry in load_bouts(candidate_dir):
        if entry.get("outcome") != "kept":
            continue
        score = entry.get("score")
        if _finite(score):
            best = float(score) if best is None else min(best, float(score))
    if best is None:
        raise ValueError(
            f"{candidate_dir}: no finite baseline_score in _import.json "
            "and no kept bout score"
        )
    return best


def classify(best: float, score: float | None, margin: float) -> str:
    """Adjudicate one measured score against the incumbent best.

    Scores are lower-is-better; a crash is any non-finite score. Only an
    improvement exceeding the noise margin is real and survives.
    """
    if not _finite(score):
        return "reverted_crash"
    if best - score > margin:
        return "kept"
    if best - score > 0:
        return "reverted_marginal"
    return "reverted_worse"


def consecutive_non_kept(bouts: list[dict]) -> int:
    """Trailing run of non-kept outcomes (the loop's stall signal)."""
    count = 0
    for entry in reversed(bouts):
        if entry.get("outcome") == "kept":
            break
        count += 1
    return count


def finalize(
    candidate_dir,
    bout: int,
    score: float | None,
    noise_margin: float,
    attempt_id: str | None,
    summary: str,
    basis: str,
) -> dict:
    """Adjudicate one bout: roll back unless kept, then journal it.

    Returns {"outcome", "best"} with the post-finalize reference point (the
    new score when kept, else the unchanged incumbent best).
    """
    if noise_margin < 0:
        raise ValueError(f"noise_margin must be >= 0, got {noise_margin}")
    candidate_dir = Path(candidate_dir)
    best = current_best(candidate_dir)
    outcome = classify(best, score, noise_margin)
    snap = _snapshot_path(candidate_dir, bout)
    if outcome != "kept":
        # An edit that was not verified better must never survive.
        if not snap.is_file():
            raise ValueError(
                f"cannot revert bout {bout}: pre-edit snapshot missing: {snap}"
            )
        revert(candidate_dir, snap)
    stored_score = float(score) if _finite(score) else None
    append_bout(
        candidate_dir,
        {
            "bout": bout,
            "attempt_id": attempt_id,
            "score": stored_score,
            "outcome": outcome,
            "summary": summary,
            "basis": basis,
            "snapshot": snap.name,
        },
    )
    if outcome == "kept":
        best = stored_score
    return {"outcome": outcome, "best": best}


def journal(
    candidate_dir,
    bout: int,
    outcome: str,
    summary: str,
    basis: str,
) -> dict:
    """Append a non-scored bout entry (noop or preflight-class failure).

    Same entry shape as finalize's with score/attempt_id null; these bouts
    never reached evaluation, so there is nothing to adjudicate.
    """
    if outcome not in ("noop", "reverted_crash"):
        raise ValueError(f"journal outcome must be noop|reverted_crash, got {outcome!r}")
    candidate_dir = Path(candidate_dir)
    entry = {
        "bout": bout,
        "attempt_id": None,
        "score": None,
        "outcome": outcome,
        "summary": summary,
        "basis": basis,
        "snapshot": _snapshot_path(candidate_dir, bout).name,
    }
    append_bout(candidate_dir, entry)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fin = subparsers.add_parser("finalize", help="adjudicate one measured bout")
    fin.add_argument("--candidate", required=True, type=Path)
    fin.add_argument("--bout", required=True, type=int)
    score_group = fin.add_mutually_exclusive_group(required=True)
    score_group.add_argument("--score", type=float)
    score_group.add_argument(
        "--nonfinite",
        action="store_true",
        help="the evaluation crashed or returned no finite score",
    )
    fin.add_argument("--noise-margin", required=True, type=float)
    fin.add_argument("--attempt-id", default=None,
                     help="omit when the evaluation's attempt id could not "
                          "be recovered; journaled as null")
    fin.add_argument("--summary", required=True)
    fin.add_argument("--basis", required=True)
    snap_p = subparsers.add_parser(
        "snapshot", help="byte-exact pre-edit copy of train.py; prints its path"
    )
    snap_p.add_argument("--candidate", required=True, type=Path)
    snap_p.add_argument("--bout", required=True, type=int)
    rev = subparsers.add_parser("revert", help="restore train.py from a snapshot")
    rev.add_argument("--candidate", required=True, type=Path)
    rev.add_argument("--snapshot", required=True, type=Path)
    chg = subparsers.add_parser(
        "changed",
        help="exit 0 when train.py is byte-identical to the snapshot, "
             "1 otherwise",
    )
    chg.add_argument("--candidate", required=True, type=Path)
    chg.add_argument("--snapshot", required=True, type=Path)
    best_p = subparsers.add_parser(
        "current-best", help="print the current reference score"
    )
    best_p.add_argument("--candidate", required=True, type=Path)
    jrnl = subparsers.add_parser(
        "journal", help="append a non-scored bout entry (noop / reverted_crash)"
    )
    jrnl.add_argument("--candidate", required=True, type=Path)
    jrnl.add_argument("--bout", required=True, type=int)
    jrnl.add_argument("--outcome", required=True, choices=["noop", "reverted_crash"])
    jrnl.add_argument("--summary", required=True)
    jrnl.add_argument("--basis", required=True)
    args = parser.parse_args()
    if args.command == "finalize":
        score = None if args.nonfinite else args.score
        try:
            result = finalize(
                args.candidate,
                args.bout,
                score,
                args.noise_margin,
                args.attempt_id,
                args.summary,
                args.basis,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(json.dumps(result))
        return 0
    if args.command == "snapshot":
        print(snapshot(args.candidate, args.bout))
        return 0
    if args.command == "revert":
        revert(args.candidate, args.snapshot)
        return 0
    if args.command == "changed":
        try:
            differs = changed(args.candidate, args.snapshot)
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        return 1 if differs else 0
    if args.command == "current-best":
        try:
            print(current_best(args.candidate))
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        return 0
    if args.command == "journal":
        print(json.dumps(journal(
            args.candidate, args.bout, args.outcome, args.summary, args.basis
        )))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
