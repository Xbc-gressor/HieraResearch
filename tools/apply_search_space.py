#!/usr/bin/env python3
"""Deterministically rewrite a candidate's SEARCH_SPACE with a finalized space.

The data-driven sibling of `apply_base_params.py`: where that writes BASE_PARAMS,
this writes SEARCH_SPACE. Used after the inducer proposes a space and
`tune_tools.py check-search-space` validates + expands it — the finalized space
lands here, never by a fuzzy string edit. AST-locates the module-level
`SEARCH_SPACE = {...}` assignment and replaces ONLY its value node, so
BASE_PARAMS / make_model / imports cannot be clobbered.

Guards (hard-reject with nonzero exit, never a partial write):
- SEARCH_SPACE must exist as a module-level assignment whose value is a *pure
  literal dict* (anything computed is rejected — splicing would drop it).
- The new space's keys must exactly match the existing SEARCH_SPACE keys. The
  key set is the schema (which params are tunable, fixed by the extractor);
  ranges/options may change, the key set may not.
- Each entry must be a valid `(kind, ...)` tuple (float/int/categorical).
- The rewritten file must still parse (re-checked before writing).

Entries are written as tuples (JSON delivers them as lists; this restores the
tuple form the tuner contract expects). Inline comments inside the SEARCH_SPACE
literal are not preserved; surrounding code and comments are untouched.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tuners"))
from tune_tools import valid_space_entry  # noqa: E402  (pure stdlib helper)


def _line_starts(source: str) -> list[int]:
    starts, total = [0], 0
    for line in source.splitlines(keepends=True):
        total += len(line)
        starts.append(total)
    return starts


def _find_search_space(tree: ast.Module):
    """Return the ast.Dict value node of the module-level SEARCH_SPACE, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "SEARCH_SPACE":
                return node.value
    return None


def _as_entry(value):
    """JSON delivers SEARCH_SPACE entries as lists; restore the tuple form.
    The categorical option list stays a list: ('categorical', ['a', 'b'])."""
    return tuple(value)


def _format_space(space: dict, key_order: list[str]) -> str:
    ordered = list(key_order) + [k for k in space if k not in key_order]
    lines = ["{"]
    for key in ordered:
        lines.append(f"    {key!r}: {_as_entry(space[key])!r},")
    lines.append("}")
    return "\n".join(lines)


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
    """Where to insert a created SEARCH_SPACE: right after PARAM_SCHEMA if present,
    else after the last module-level assignment, else before the first def/class."""
    schema = _find_assignment(tree, "PARAM_SCHEMA")
    if schema is not None:
        return schema
    last_assign = None
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            last_assign = node
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return last_assign or node
    return last_assign


def apply(candidate_path: Path, space: dict) -> dict:
    bad = [k for k, v in space.items() if not valid_space_entry(_as_entry(v))]
    if bad:
        raise SystemExit(f"invalid SEARCH_SPACE entries for keys: {sorted(bad)}")

    source = candidate_path.read_text()
    tree = ast.parse(source)
    starts = _line_starts(source)
    value_node = _find_search_space(tree)

    if value_node is not None:
        # ----- rewrite an existing SEARCH_SPACE -----
        if not isinstance(value_node, ast.Dict):
            raise SystemExit("SEARCH_SPACE value is not a dict literal")
        try:
            current = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            raise SystemExit(
                "SEARCH_SPACE is not a pure literal dict (an entry is a computed "
                "expression); refusing to splice"
            )
        if set(current) != set(space):
            raise SystemExit(
                f"key mismatch: SEARCH_SPACE has {sorted(current)}, new has {sorted(space)} "
                "(the key set is the schema and may not change)"
            )
        start = starts[value_node.lineno - 1] + value_node.col_offset
        end = starts[value_node.end_lineno - 1] + value_node.end_col_offset
        new_source = source[:start] + _format_space(space, list(current.keys())) + source[end:]
        mode = "rewrote"
    else:
        # ----- create mode: no SEARCH_SPACE yet (step 0, only PARAM_SCHEMA) -----
        schema_node = _find_assignment(tree, "PARAM_SCHEMA")
        if schema_node is not None and isinstance(schema_node.value, ast.Dict):
            try:
                schema = ast.literal_eval(schema_node.value)
                if set(schema) != set(space):
                    raise SystemExit(
                        f"key mismatch vs PARAM_SCHEMA: schema {sorted(schema)}, "
                        f"space {sorted(space)} (the key set is the schema and may not change)"
                    )
            except (ValueError, SyntaxError):
                pass  # schema not a pure literal — skip the key cross-check
        anchor = _insertion_anchor(tree)
        if anchor is None:
            raise SystemExit("cannot find an insertion anchor (no assignment/def in module)")
        insert_at = starts[anchor.end_lineno]  # start of the line after the anchor
        block = "\nSEARCH_SPACE = " + _format_space(space, list(space.keys())) + "\n"
        new_source = source[:insert_at] + block + source[insert_at:]
        mode = "created"

    ast.parse(new_source)  # guarantee the result still parses before writing
    candidate_path.write_text(new_source)
    return {"applied": True, "mode": mode, "keys": sorted(space),
            "candidate_path": str(candidate_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-path", required=True, type=Path)
    parser.add_argument("--space-json", required=True, type=Path,
                        help="JSON file with the finalized SEARCH_SPACE to write")
    args = parser.parse_args()
    space = json.loads(args.space_json.read_text())
    print(json.dumps(apply(args.candidate_path, space)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
