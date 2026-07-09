#!/usr/bin/env python3
"""Deterministically write a candidate's BASE_PARAMS (create or rewrite).

Replaces the manual Edit the loop used to do by hand — the single most
consequential and error-prone step, because BASE_PARAMS is the experiment
surface and whatever lands here is inherited by every candidate later branched
from this one. AST-locates the module-level `BASE_PARAMS = {...}` assignment and
replaces ONLY its value node (rewrite), or inserts a new one after SEARCH_SPACE
(create), so SEARCH_SPACE / make_model / imports cannot be clobbered by a fuzzy
string edit.

Two modes (chosen automatically):
- **rewrite** — BASE_PARAMS already exists; replace its value. The new params'
  keys must exactly match the existing BASE_PARAMS keys.
- **create** — no BASE_PARAMS yet (`warmstart_eval` step 1 creates it from a
  warm-start config). Inserts `BASE_PARAMS = {...}` right after SEARCH_SPACE
  (else PARAM_SCHEMA, else the last assignment, else before the first def). If a
  literal SEARCH_SPACE is present, the new keys must equal its key set.

Guards (hard-reject with nonzero exit, never a partial write):
- A rewritten BASE_PARAMS must be a *pure literal dict*. A computed default such
  as `"seed": RANDOM_SEED` is rejected — splicing would silently drop the
  reference.
- Key-match as described per mode.
- The rewritten file must still parse (re-checked before writing).

This tool intentionally does NOT bound-check the incoming values. A tuned best
can land outside the current SEARCH_SPACE range (e.g. a warm-start config the
LLM proposed beyond it that scored best) — that value is kept. The key-match
guard keeps the contract's key set consistent; the range is reconciled by
`check-search-space`, which widens SEARCH_SPACE to bracket the configs (so
lint-contract's out-of-bounds check, still a hard error, is satisfied by
construction, not by clamping the value).
Inline comments *inside* the BASE_PARAMS literal are not preserved (only that
block is written); surrounding code and comments are untouched.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _line_starts(source: str) -> list[int]:
    starts, total = [0], 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        starts.append(total)
    return starts


def _find_base_params(tree: ast.Module):
    """Return the ast.Dict value node of the module-level BASE_PARAMS, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "BASE_PARAMS":
                return node.value
    return None


def _find_assignment(tree: ast.Module, name: str):
    """Return the module-level Assign/AnnAssign *node* for `name = ...`, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                return node
    return None


def _insertion_anchor(tree: ast.Module):
    """Where to insert a created BASE_PARAMS: right after SEARCH_SPACE if present,
    else after PARAM_SCHEMA, else after the last module-level assignment, else
    before the first def/class."""
    for name in ("SEARCH_SPACE", "PARAM_SCHEMA"):
        node = _find_assignment(tree, name)
        if node is not None:
            return node
    last_assign = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            last_assign = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return last_assign or node
    return last_assign


def _format_dict(params: dict, key_order: list[str]) -> str:
    ordered = list(key_order) + [k for k in params if k not in key_order]
    lines = ["{"]
    for key in ordered:
        lines.append(f"    {key!r}: {params[key]!r},")
    lines.append("}")
    return "\n".join(lines)


def apply(candidate_path: Path, params: dict) -> dict:
    source = candidate_path.read_text()
    tree = ast.parse(source)
    starts = _line_starts(source)
    value_node = _find_base_params(tree)

    if value_node is not None:
        # ----- rewrite an existing BASE_PARAMS -----
        if not isinstance(value_node, ast.Dict):
            raise SystemExit("BASE_PARAMS value is not a dict literal")
        try:
            current = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            raise SystemExit(
                "BASE_PARAMS is not a pure literal dict (a default is a computed "
                "expression); refusing to splice"
            )
        if set(current) != set(params):
            raise SystemExit(
                f"key mismatch: BASE_PARAMS has {sorted(current)}, params has {sorted(params)}"
            )
        start = starts[value_node.lineno - 1] + value_node.col_offset
        end = starts[value_node.end_lineno - 1] + value_node.end_col_offset
        new_source = source[:start] + _format_dict(params, list(current.keys())) + source[end:]
        mode = "rewrote"
    else:
        # ----- create mode: no BASE_PARAMS yet (warmstart_eval step 1) -----
        space_node = _find_assignment(tree, "SEARCH_SPACE")
        if space_node is not None and isinstance(space_node.value, ast.Dict):
            try:
                space = ast.literal_eval(space_node.value)
                if set(space) != set(params):
                    raise SystemExit(
                        f"key mismatch vs SEARCH_SPACE: space {sorted(space)}, "
                        f"params {sorted(params)} (BASE_PARAMS keys must equal SEARCH_SPACE keys)"
                    )
            except (ValueError, SyntaxError):
                pass  # SEARCH_SPACE not a pure literal — skip the key cross-check
        anchor = _insertion_anchor(tree)
        if anchor is None:
            raise SystemExit("cannot find an insertion anchor (no assignment/def in module)")
        insert_at = starts[anchor.end_lineno]  # start of the line after the anchor
        block = "\nBASE_PARAMS = " + _format_dict(params, list(params.keys())) + "\n"
        new_source = source[:insert_at] + block + source[insert_at:]
        mode = "created"

    ast.parse(new_source)  # guarantee the result still parses before writing
    candidate_path.write_text(new_source)
    return {"applied": True, "mode": mode, "keys": sorted(params),
            "candidate_path": str(candidate_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--params-json", required=True, type=Path,
                        help="JSON file with the params dict to write into BASE_PARAMS")
    args = parser.parse_args()
    params = json.loads(args.params_json.read_text())
    print(json.dumps(apply(args.candidate_path, params)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
