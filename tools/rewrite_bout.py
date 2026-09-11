#!/usr/bin/env python3
"""Bout journal and keep/revert adjudication for the rewrite loop.

A rewrite bout edits <candidate>/train.py in place; this module is the
deterministic trust anchor around that edit:

- snapshot()/revert() — byte-exact pre-edit copy and rollback;
- bouts.jsonl — append-only journal, one entry per adjudicated bout:
  {bout, attempt_id, score, outcome, summary, basis, snapshot};
- current_best() — the measured reference point: the imported baseline score
  (or the caller's `--reference`, the ledger score in the experiment loop),
  moved by every kept bout and by each kept bout's confirmation re-eval;
- classify() — an improvement only counts when it clears the noise margin;
- the `finalize` CLI — classify one bout, roll back byte-exactly unless kept,
  and append the journal entry;
- the `confirm` CLI — after a keep, fold one confirmation score into the
  reference (mean of the two) so a lucky single sample does not anchor it;
- the `params-equal` CLI — reject pure tuner moves, while allowing tuner
  parameters and declarations to change alongside implementation code.

Usage:
    python tools/rewrite_bout.py finalize --candidate <dir> --bout <N> \
        [--score S | --nonfinite] --noise-margin E [--attempt-id A] \
        [--reference R] --summary "..." --basis "..."
    python tools/rewrite_bout.py confirm --candidate <dir> --bout <N> \
        [--score S | --nonfinite] [--attempt-id A]
    python tools/rewrite_bout.py snapshot --candidate <dir> --bout <N>
    python tools/rewrite_bout.py revert --candidate <dir> --snapshot <path>
    python tools/rewrite_bout.py changed --candidate <dir> --snapshot <path>
    python tools/rewrite_bout.py params-equal --candidate <dir> --snapshot <path>
    python tools/rewrite_bout.py current-best --candidate <dir>
    python tools/rewrite_bout.py journal --candidate <dir> --bout <N> \
        --outcome noop|reverted_crash|reverted_params --summary "..." --basis "..."

finalize stdout: {"outcome", "best"} — best is the post-finalize reference
point; a negative --noise-margin is rejected. snapshot prints the snapshot
path — when the bout's snapshot already exists (the previous attempt at this
bout was interrupted mid-edit) it restores train.py from the snapshot instead
of re-snapshotting the possibly dirty file;
changed exits 0 when train.py is byte-identical to the snapshot and 1
otherwise (a missing file or snapshot is exit 1 with a message);
params-equal exits 1 when a search-space declaration or space-named
BASE_PARAMS value changed without implementation code changing, 0 otherwise
or when a needed piece is unparseable (the
literal-form failure belongs to the evaluation stage's repair path);
current-best prints the float (fails when there is no finite reference);
journal appends a non-scored entry (score/attempt_id null) for noop,
preflight-class-failure, and params-contract-violation bouts.
"""

from __future__ import annotations

import argparse
import ast
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


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None


CONTRACT_NAMES = frozenset(("PARAM_SCHEMA", "SEARCH_SPACE", "BASE_PARAMS"))


def _literal_dict(tree: ast.Module, name: str) -> dict | None:
    """The module-level ``name`` assignment as a plain dict, or None when it
    is missing or not a pure literal dict (same acceptance set as
    rewrite_eval.read_base_params)."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0] if len(node.targets) == 1 else None
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            target = None
        if not (isinstance(target, ast.Name) and target.id == name):
            continue
        if not isinstance(node.value, ast.Dict):
            return None
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, dict) else None
    return None


def _code_shape(tree: ast.Module) -> str:
    """AST shape with contract declarations removed.

    Formatting and comments do not count as implementation movement. This
    lets a rewrite change a tuner value when it also changes code. This is
    a structural check; the editor remains responsible for the change's merit.
    """
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = {target.id for target in node.targets
                     if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = {node.target.id}
        else:
            names = set()
        if names & CONTRACT_NAMES:
            continue
        body.append(node)
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def params_equal(candidate_dir, snap) -> bool | None:
    """Allow implementation changes, reject edits confined to tuning.

    Parameters outside the declared space may move independently. A change
    to tuner values or declarations needs an accompanying code AST change.
    Returns None for unreadable BASE_PARAMS, which the params repair path
    handles. Without a readable space, compare the whole BASE_PARAMS dict.
    """
    before = _parse(Path(snap))
    after = _parse(Path(candidate_dir) / TRAIN_PY)
    if before is None or after is None:
        return None
    before_params = _literal_dict(before, "BASE_PARAMS")
    after_params = _literal_dict(after, "BASE_PARAMS")
    if before_params is None or after_params is None:
        return None
    if _code_shape(before) != _code_shape(after):
        return True
    params_changed = before_params != after_params
    before_space = _literal_dict(before, "SEARCH_SPACE")
    after_space = _literal_dict(after, "SEARCH_SPACE")
    if before_space is None or after_space is None:
        # Without a readable space, ownership is unknowable: require an
        # implementation change for any parameter-only edit.
        contract_changed = params_changed or before_space != after_space
    else:
        tuner_keys = set(before_space) | set(after_space)
        missing = object()
        tuner_params_changed = any(
            before_params.get(key, missing) != after_params.get(key, missing)
            for key in tuner_keys
        )
        contract_changed = (
            before_space != after_space
            or _literal_dict(before, "PARAM_SCHEMA") != _literal_dict(
                after, "PARAM_SCHEMA")
            or tuner_params_changed
        )
    return not contract_changed


def _bouts_path(candidate_dir) -> Path:
    return _rewrite_dir(candidate_dir) / "bouts.jsonl"


CONFIRMATION_KIND = "confirmation"


def load_entries(candidate_dir) -> list[dict]:
    """Every journal line in order: bout entries and confirmation entries."""
    path = _bouts_path(candidate_dir)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_bouts(candidate_dir) -> list[dict]:
    """Bout entries only (confirmation re-evals are not bouts)."""
    return [
        entry for entry in load_entries(candidate_dir)
        if entry.get("kind") != CONFIRMATION_KIND
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


def current_best(candidate_dir, baseline: float | None = None) -> float:
    """The reference point: min(baseline, every kept bout's score), where a
    kept bout's confirmation entry resets the reference to its mean, in
    journal order.

    ``baseline`` defaults to `_import.json`'s baseline_score. Raises
    ValueError when no finite reference exists — adjudication without a
    measured reference is meaningless.
    """
    candidate_dir = Path(candidate_dir)
    best = float(baseline) if _finite(baseline) else None
    manifest_path = candidate_dir / "_import.json"
    if best is None and manifest_path.is_file():
        imported = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "baseline_score"
        )
        if _finite(imported):
            best = float(imported)
    for entry in load_entries(candidate_dir):
        if entry.get("kind") == CONFIRMATION_KIND:
            if _finite(entry.get("reference")):
                best = float(entry["reference"])
            continue
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
    reference: float | None = None,
) -> dict:
    """Adjudicate one bout: roll back unless kept, then journal it.

    ``reference`` overrides the journal-derived incumbent (the experiment
    loop passes the ledger's current score). Returns {"outcome", "best"} with
    the post-finalize reference point (the new score when kept, else the
    unchanged incumbent best).
    """
    if noise_margin < 0:
        raise ValueError(f"noise_margin must be >= 0, got {noise_margin}")
    candidate_dir = Path(candidate_dir)
    best = float(reference) if _finite(reference) else current_best(candidate_dir)
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


def confirm(
    candidate_dir,
    bout: int,
    score: float | None,
    attempt_id: str | None,
) -> dict:
    """Fold a kept bout's confirmation re-eval into the reference.

    The reference becomes the mean of the kept score and the confirmation
    score; a non-finite confirmation leaves it at the kept score. The keep
    itself is never undone here — a real improvement lost to one unlucky
    sample costs more than a noisy keep.
    """
    candidate_dir = Path(candidate_dir)
    kept = next(
        (
            entry for entry in load_bouts(candidate_dir)
            if entry.get("bout") == bout and entry.get("outcome") == "kept"
        ),
        None,
    )
    if kept is None or not _finite(kept.get("score")):
        raise ValueError(f"bout {bout} has no kept score to confirm")
    kept_score = float(kept["score"])
    confirmed = float(score) if _finite(score) else None
    reference = kept_score if confirmed is None else (kept_score + confirmed) / 2
    entry = {
        "kind": CONFIRMATION_KIND,
        "bout": bout,
        "attempt_id": attempt_id,
        "score": confirmed,
        "reference": reference,
    }
    append_bout(candidate_dir, entry)
    return entry


def journal(
    candidate_dir,
    bout: int,
    outcome: str,
    summary: str,
    basis: str,
) -> dict:
    """Append a non-scored bout entry (noop, preflight-class failure, or
    params-contract violation).

    Same entry shape as finalize's with score/attempt_id null; these bouts
    never reached evaluation, so there is nothing to adjudicate.
    """
    if outcome not in ("noop", "reverted_crash", "reverted_params"):
        raise ValueError(
            f"journal outcome must be noop|reverted_crash|reverted_params, "
            f"got {outcome!r}")
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
    fin.add_argument("--reference", type=float, default=None,
                     help="reference score to adjudicate against (default: "
                          "the journal-derived incumbent)")
    conf = subparsers.add_parser(
        "confirm", help="fold a kept bout's confirmation re-eval into the reference"
    )
    conf.add_argument("--candidate", required=True, type=Path)
    conf.add_argument("--bout", required=True, type=int)
    conf_group = conf.add_mutually_exclusive_group(required=True)
    conf_group.add_argument("--score", type=float)
    conf_group.add_argument("--nonfinite", action="store_true")
    conf.add_argument("--attempt-id", default=None)
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
    peq = subparsers.add_parser(
        "params-equal",
        help="exit 1 for tuner-only edits without implementation changes, "
             "0 otherwise or when params need repair",
    )
    peq.add_argument("--candidate", required=True, type=Path)
    peq.add_argument("--snapshot", required=True, type=Path)
    best_p = subparsers.add_parser(
        "current-best", help="print the current reference score"
    )
    best_p.add_argument("--candidate", required=True, type=Path)
    jrnl = subparsers.add_parser(
        "journal", help="append a non-scored bout entry "
                        "(noop / reverted_crash / reverted_params)"
    )
    jrnl.add_argument("--candidate", required=True, type=Path)
    jrnl.add_argument("--bout", required=True, type=int)
    jrnl.add_argument("--outcome", required=True,
                      choices=["noop", "reverted_crash", "reverted_params"])
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
                reference=args.reference,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(json.dumps(result))
        return 0
    if args.command == "confirm":
        try:
            entry = confirm(
                args.candidate,
                args.bout,
                None if args.nonfinite else args.score,
                args.attempt_id,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        print(json.dumps(entry))
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
    if args.command == "params-equal":
        return 0 if params_equal(args.candidate, args.snapshot) is not False else 1
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
