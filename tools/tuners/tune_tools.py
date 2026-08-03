#!/usr/bin/env python3
"""Deterministic helpers the tuner-orchestrator calls instead of doing the
work by hand. Each subcommand is a pure computation/check — no judgment — so
the choices stay stable across long autonomous runs.

Subcommands:
- lint-contract   : candidate train.py -> {ok, n_dims, keys, make_model_defined,
                    make_model_called, errors[]}; checks the contract's
                    relational invariants (key-set match, in-bounds defaults,
                    tuple shapes, make_model is a def) by AST. Exit 1 on any
                    hard error. The gate before a candidate may be tuned.
- lint-schema     : candidate train.py -> {ok, keys, kinds, make_model_defined,
                    make_model_called, errors[]}; schema-mode check (make_model +
                    PARAM_SCHEMA) for step 0, before SEARCH_SPACE/BASE_PARAMS exist.
- select-method   : candidate SEARCH_SPACE -> {n_dims, method, fallback}
- phase-c-action  : candidate + tune_report -> deterministic resume action
                    (run method, finalize, or stop on exhausted allocation)
- select-best     : tune_report.json -> global best (minimum)
                    {best_params, best_score, source} over selectable
                    warm+phase_c trials; fidelity controls are observations,
                    never incumbents
- validate-params : a params dict's keys/bounds vs the candidate SEARCH_SPACE
                    -> {ok, violations}; exit 1 on any violation
- summarize       : tune_report.json -> stored tuning summary {best_warm_score,
                    final_best_score, trials_completed, trials_attempted,
                    elapsed_seconds}
- check-search-space : proposed SEARCH_SPACE + proposed warm configs -> {ok,
                    finalized_space, expansions, errors}; validates kinds vs the
                    full schema and widens ranges to bracket every validated
                    config. On ok, the
                    proposed --space-json is overwritten in place with finalized_space
                    (apply_search_space.py then writes it into train.py); exit 1 on a
                    hard error.
- lineage-evidence : run_dir + parent run_ids -> {per_parent} — per parent its
                    idea, best config+score, searched space, explored ranges, and
                    a few whole trials (top-by-score + farthest-point diverse, so
                    hyperparameter interactions survive). Feeds proposer/inducer.
- build-inheritance : replace warm config 0 with the primary parent's applied
                    incumbent projected exactly onto the child's compatible
                    schema, and persist a stale-detecting transfer receipt.
- render-failure  : frozen receipt by default; exact full/ranged traceback only
                    when explicitly requested.

Pure stdlib (ast/argparse/json), so it needs **no uv environment**: SEARCH_SPACE
is read by AST literal_eval rather than importing the candidate. Scores are
**always lower-is-better** (minimize), so no optimization direction is read or
passed — the task's eval fn must conform.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/ for run_cfg

from failure_artifacts import render_failure
from evaluation_budget import budget_status, find_run_dir  # noqa: E402
from run_cfg import read_framework_cfg  # noqa: E402
from semantic_evidence import unbound_primary_descendants  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMETER_TRANSFER_FILENAME = "_parameter_transfer.json"


# ---------- SEARCH_SPACE via AST (no candidate import) ----------


def _read_search_space(train_path: Path) -> dict:
    """SEARCH_SPACE dict via AST literal_eval — no import, no uv env. The tuner
    contract requires a module-level dict literal; anything else errors here
    (lint-contract's job to explain why)."""
    try:
        return _read_literal_mapping(train_path, "SEARCH_SPACE")
    except (OSError, SyntaxError, ValueError) as exc:
        raise SystemExit(f"invalid SEARCH_SPACE contract: {exc}") from None


def _read_param_schema(train_path: Path) -> dict:
    """PARAM_SCHEMA dict via AST literal_eval — the schema the extractor writes
    before SEARCH_SPACE exists. Each value is a kind declaration:
      "int" | "float" | ("float", "log") | ("categorical", [opt, ...])."""
    try:
        return _read_literal_mapping(train_path, "PARAM_SCHEMA")
    except (OSError, SyntaxError, ValueError) as exc:
        raise SystemExit(f"invalid PARAM_SCHEMA contract: {exc}") from None


def _schema_kind(entry) -> str | None:
    """The kind string of a PARAM_SCHEMA entry, or ``None`` when malformed."""
    if isinstance(entry, str):
        return entry
    if (
        isinstance(entry, (tuple, list))
        and entry
        and isinstance(entry[0], str)
    ):
        return entry[0]
    return None


def _valid_categorical_value(value) -> bool:
    """Whether a categorical value is stable across AST, JSON, and samplers."""
    if value is None or type(value) in {bool, int, str}:
        return True
    return type(value) is float and math.isfinite(value)


def _safe_repr(value, *, limit: int = 500) -> str:
    """Bounded diagnostic rendering that cannot fail on enormous integers."""
    try:
        rendered = repr(value)
    except (OverflowError, ValueError):
        if type(value) is int:
            return f"<int bit_length={value.bit_length()}>"
        return f"<{type(value).__name__}>"
    if len(rendered) > limit:
        return rendered[: limit - 3] + "..."
    return rendered


def _categorical_value_equal(left, right) -> bool:
    """Categorical membership is type-exact (``True`` is not integer ``1``)."""
    return type(left) is type(right) and left == right


def _categorical_contains(options, value) -> bool:
    return any(_categorical_value_equal(option, value) for option in options)


def _valid_categorical_options(options) -> bool:
    if not isinstance(options, (list, tuple)) or not options:
        return False
    seen: list = []
    for value in options:
        if not _valid_categorical_value(value):
            return False
        # Reject both exact duplicates and equality collisions across primitive
        # types (notably True/1 and False/0), which categorical samplers cannot
        # represent unambiguously.
        if any(value == previous for previous in seen):
            return False
        seen.append(value)
    return True


def _valid_schema_entry(entry) -> bool:
    """A PARAM_SCHEMA value: "int" / "float" / ("float","log") / ("categorical",[opts])."""
    if entry in ("int", "float"):
        return True
    if isinstance(entry, (tuple, list)) and len(entry) >= 1:
        kind = entry[0]
        if kind == "float":
            return len(entry) == 2 and entry[1] == "log"
        if kind == "int":
            return len(entry) == 1
        if kind == "categorical":
            return len(entry) == 2 and _valid_categorical_options(entry[1])
    return False


def _space_schema_mismatch(schema_entry, space_entry) -> str | None:
    """Explain why a valid SEARCH_SPACE entry disagrees with PARAM_SCHEMA."""
    schema_kind = _schema_kind(schema_entry)
    if space_entry[0] != schema_kind:
        return f"kind {space_entry[0]!r} != schema kind {schema_kind!r}"
    if schema_kind == "float":
        schema_log = (
            isinstance(schema_entry, (tuple, list))
            and len(schema_entry) == 2
            and schema_entry[1] == "log"
        )
        space_log = len(space_entry) == 4 and space_entry[3] == "log"
        if schema_log != space_log:
            return (
                f"log mode {space_log!r} != schema log mode {schema_log!r}"
            )
    if schema_kind == "categorical":
        invalid = [
            value
            for value in space_entry[1]
            if not _schema_accepts_value(schema_entry, value)
        ]
        if invalid:
            return (
                "contains options outside PARAM_SCHEMA: "
                f"{_safe_repr(invalid)}"
            )
    return None


# ---------- method selection ----------

# Single source of truth for the dimensionality -> Phase C method map. Trial-cap
# args (n_trials etc.) are NOT here — they are caller-overridable defaults owned
# by each search script. `fallback` is the method to try if the chosen one
# rejects (e.g. grid size exceeds --max-trials).
def select_method(n_dims: int) -> dict:
    # Data-driven thresholds (dev_plan/hpo-benchmark-report.md): multivariate TPE
    # (method "bo") wins at BOTH mid and high dims; cmaes/cmaes+ were *worst* at
    # high dims, so the old "cmaes for n_dims>=16" tier is dropped. grid stays for
    # the tiny (<=2) exhaustive regime.
    if n_dims <= 2:
        return {"n_dims": n_dims, "method": "grid", "fallback": ["bo"]}
    return {"n_dims": n_dims, "method": "bo", "fallback": ["cmaes"]}


# ---------- bounds checking ----------


def _value_in_bounds(value, entry) -> bool:
    if not valid_space_entry(entry):
        return False
    kind = entry[0]
    if kind == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        try:
            numeric = float(value)
            low = float(entry[1])
            high = float(entry[2])
        except (OverflowError, TypeError, ValueError):
            return False
        return (
            math.isfinite(numeric)
            and math.isfinite(low)
            and math.isfinite(high)
            and low <= numeric <= high
        )
    if kind == "int":
        if type(value) is int:
            return int(entry[1]) <= value <= int(entry[2])
        if type(value) is not float:
            return False
        if not math.isfinite(value) or not value.is_integer():
            return False
        return int(entry[1]) <= int(value) <= int(entry[2])
    if kind == "categorical":
        return _categorical_contains(entry[1], value)
    return False


def _bounds_violations(params: dict, search_space: dict) -> list[dict]:
    """Per-key violations: missing key, extra key, or out-of-bounds value.
    Enforces exact key match (apply requires it)."""
    violations = []
    space_keys, param_keys = set(search_space), set(params)
    for key in sorted(space_keys - param_keys):
        violations.append({"key": key, "value": None, "reason": "missing from params"})
    for key in sorted(param_keys - space_keys):
        violations.append({"key": key, "value": params[key], "reason": "not in SEARCH_SPACE"})
    for key in sorted(space_keys & param_keys):
        if not _value_in_bounds(params[key], search_space[key]):
            rendered_value = _safe_repr(params[key])
            value = (
                params[key]
                if not rendered_value.startswith("<int bit_length=")
                else rendered_value
            )
            violations.append({
                "key": key,
                "value": value,
                "reason": f"outside {_safe_repr(search_space[key])}",
            })
    return violations


# ---------- contract lint ----------

# The tuner contract's *relational* invariants — the ones that silently poison
# downstream tuning if an LLM writes the contract slightly wrong. `_common` only
# checks the three symbols exist (hasattr); this checks they agree. Pure AST so
# it runs as a gate before any uv env is touched. Errors carry the offending key
# and its source line so the fix is mechanical.

CONTRACT_SYMBOLS = ("PARAM_SCHEMA", "SEARCH_SPACE", "BASE_PARAMS", "make_model")


def _module_bindings(tree: ast.Module) -> dict[str, list[ast.AST]]:
    """Return every module-level binding, in runtime order.

    Python uses the final binding while older tuner helpers inspected the first.
    Keeping the complete list lets every deterministic boundary reject that
    split-brain state instead of validating or rewriting a different contract
    from the one Python imports.
    """
    out: dict[str, list[ast.AST]] = {}

    def _append(name: str, value: ast.AST) -> None:
        out.setdefault(name, []).append(value)

    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            _append(node.targets[0].id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                _append(node.target.id, node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _append(node.name, node)
    return out


def _module_symbols(tree: ast.Module) -> dict[str, ast.AST]:
    """Runtime-effective module-level bindings (the final binding wins).

    Contract readers must additionally require exactly one binding; this helper
    exists for diagnostics that need to keep inspecting a malformed module.
    """
    return {name: nodes[-1] for name, nodes in _module_bindings(tree).items()}


def _duplicate_binding_errors(
    bindings: dict[str, list[ast.AST]],
    names: tuple[str, ...] = CONTRACT_SYMBOLS,
) -> list[dict]:
    errors = []
    for name in names:
        nodes = bindings.get(name, [])
        if len(nodes) > 1:
            errors.append(
                {
                    "code": "duplicate_symbol",
                    "detail": (
                        f"{name} has {len(nodes)} module-level bindings at lines "
                        f"{[getattr(node, 'lineno', 0) for node in nodes]}"
                    ),
                    "line": getattr(nodes[1], "lineno", 0),
                }
            )
    return errors


def _target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        return {
            name
            for element in target.elts
            for name in _target_names(element)
        }
    return set()


def _contract_declaration_errors(tree: ast.Module) -> list[dict]:
    """Contract declarations must be one direct ``NAME = literal`` target."""
    errors: list[dict] = []
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            names = {
                name
                for target in statement.targets
                for name in _target_names(target)
            }
            affected = names & set(CONTRACT_SYMBOLS)
            simple = (
                len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            )
            if affected and not simple:
                for name in sorted(affected):
                    errors.append({
                        "code": "non_simple_declaration",
                        "symbol": name,
                        "detail": (
                            f"{name} must use one simple module-level target; "
                            "chained and destructuring assignments are forbidden"
                        ),
                        "line": getattr(statement, "lineno", 0),
                    })
        elif isinstance(statement, ast.AnnAssign):
            names = _target_names(statement.target) & set(CONTRACT_SYMBOLS)
            if names and not isinstance(statement.target, ast.Name):
                for name in sorted(names):
                    errors.append({
                        "code": "non_simple_declaration",
                        "symbol": name,
                        "detail": f"{name} must use one simple module-level target",
                        "line": getattr(statement, "lineno", 0),
                    })
    return errors


def _contract_mutation_errors(tree: ast.Module) -> list[dict]:
    """Reject module-scope mutations outside the one literal declaration.

    Assignments nested in a top-level ``if`` still execute in module scope, and
    ``SEARCH_SPACE.update(...)`` can otherwise change the imported contract
    after the AST linter has approved its literal.
    """
    errors: list[dict] = []
    contract_mappings = {"PARAM_SCHEMA", "SEARCH_SPACE", "BASE_PARAMS"}

    def _root_name(node: ast.AST) -> str | None:
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    def _globals_contract_name(node: ast.AST) -> str | None:
        """Contract key targeted through ``globals()[...]``, if statically known."""
        if not isinstance(node, ast.Subscript):
            return None
        owner = node.value
        if not (
            isinstance(owner, ast.Call)
            and isinstance(owner.func, ast.Name)
            and owner.func.id == "globals"
            and not owner.args
            and not owner.keywords
        ):
            return None
        key = node.slice
        if (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and key.value in CONTRACT_SYMBOLS
        ):
            return key.value
        return None

    def _contains_mapping_reference(node: ast.AST | None) -> bool:
        return node is not None and any(
            isinstance(child, ast.Name)
            and isinstance(child.ctx, ast.Load)
            and child.id in contract_mappings
            for child in ast.walk(node)
        )

    def _record_binding(name: str | None, statement: ast.AST) -> None:
        if name in CONTRACT_SYMBOLS:
            errors.append({
                "code": "contract_mutation",
                "symbol": name,
                "detail": (
                    f"{name} is rebound outside its one direct module-level "
                    "declaration"
                ),
                "line": getattr(statement, "lineno", 0),
            })

    def _pattern_names(pattern: ast.pattern) -> set[str]:
        names = {
            child.name
            for child in ast.walk(pattern)
            if isinstance(child, (ast.MatchAs, ast.MatchStar))
            and child.name is not None
        }
        names.update(
            child.rest
            for child in ast.walk(pattern)
            if isinstance(child, ast.MatchMapping)
            and child.rest is not None
        )
        return names

    class _NamedExprVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):  # noqa: N802
            return

        def visit_AsyncFunctionDef(self, node):  # noqa: N802
            return

        def visit_ClassDef(self, node):  # noqa: N802
            return

        def visit_Lambda(self, node):  # noqa: N802
            return

        def visit_NamedExpr(self, node):  # noqa: N802
            for name in _target_names(node.target):
                _record_binding(name, node)
            self.generic_visit(node.value)

    def _walk_statements(statements: list[ast.stmt], *, nested: bool) -> None:
        for statement in statements:
            targets: list[ast.AST] = []
            if isinstance(statement, ast.Assign):
                targets = list(statement.targets)
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            elif isinstance(statement, (ast.AugAssign, ast.Delete)):
                targets = (
                    [statement.target]
                    if isinstance(statement, ast.AugAssign)
                    else list(statement.targets)
                )
            for target in targets:
                root = _root_name(target)
                dynamic_root = _globals_contract_name(target)
                if dynamic_root is not None:
                    errors.append({
                        "code": "contract_mutation",
                        "symbol": dynamic_root,
                        "detail": (
                            f"{dynamic_root} is rebound through globals(); "
                            "dynamic contract writes are forbidden"
                        ),
                        "line": getattr(statement, "lineno", 0),
                    })
                    continue
                if root not in CONTRACT_SYMBOLS:
                    continue
                if nested or isinstance(target, (ast.Subscript, ast.Attribute)) \
                        or isinstance(statement, (ast.AugAssign, ast.Delete)):
                    errors.append({
                        "code": "contract_mutation",
                        "symbol": root,
                        "detail": (
                            f"{root} is mutated outside its one direct "
                            "module-level declaration"
                        ),
                        "line": getattr(statement, "lineno", 0),
                    })

            # A second name pointing at a contract mapping permits mutations the
            # AST literal readers cannot see. Reject the escape at its source.
            assignment_value = (
                statement.value
                if isinstance(statement, (ast.Assign, ast.AnnAssign))
                else None
            )
            assignment_names = {
                name
                for target in targets
                for name in _target_names(target)
            }
            if (
                assignment_value is not None
                and _contains_mapping_reference(assignment_value)
                and assignment_names - set(CONTRACT_SYMBOLS)
            ):
                errors.append({
                    "code": "contract_alias_escape",
                    "detail": (
                        "a module-scope assignment exposes a contract mapping "
                        f"through {sorted(assignment_names - set(CONTRACT_SYMBOLS))}"
                    ),
                    "line": getattr(statement, "lineno", 0),
                })

            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and _root_name(statement.value.func.value) in contract_mappings
            ):
                root = _root_name(statement.value.func.value)
                errors.append({
                    "code": "contract_mutation",
                    "symbol": root,
                    "detail": f"{root}.{statement.value.func.attr}(...) mutates or obscures the literal contract",
                    "line": getattr(statement, "lineno", 0),
                })

            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
                if (
                    isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Call)
                    and isinstance(call.func.value.func, ast.Name)
                    and call.func.value.func.id == "globals"
                    and call.func.attr
                    in {"update", "__setitem__", "setdefault", "pop", "clear"}
                ):
                    errors.append({
                        "code": "contract_mutation",
                        "detail": (
                            "module globals are mutated dynamically; tuner "
                            "contract identity cannot be proven"
                        ),
                        "line": getattr(statement, "lineno", 0),
                    })
                escaped = any(
                    _contains_mapping_reference(value)
                    for value in [
                        *call.args,
                        *(keyword.value for keyword in call.keywords),
                    ]
                )
                if escaped:
                    errors.append({
                        "code": "contract_alias_escape",
                        "detail": (
                            "a module-scope call receives a contract mapping and "
                            "could mutate it outside the literal declaration"
                        ),
                        "line": getattr(statement, "lineno", 0),
                    })

            if isinstance(statement, (ast.For, ast.AsyncFor)):
                for name in _target_names(statement.target):
                    _record_binding(name, statement)
            if isinstance(statement, (ast.With, ast.AsyncWith)):
                for item in statement.items:
                    if item.optional_vars is not None:
                        for name in _target_names(item.optional_vars):
                            _record_binding(name, statement)
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                for alias in statement.names:
                    bound = alias.asname
                    if bound is None:
                        bound = (
                            alias.name
                            if isinstance(statement, ast.ImportFrom)
                            else alias.name.split(".", 1)[0]
                        )
                    _record_binding(bound, statement)
            if isinstance(statement, ast.Try):
                for handler in statement.handlers:
                    _record_binding(handler.name, handler)
            if isinstance(statement, ast.Match):
                for case in statement.cases:
                    for name in _pattern_names(case.pattern):
                        _record_binding(name, statement)

            _NamedExprVisitor().visit(statement)

            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if nested or isinstance(statement, ast.ClassDef):
                    _record_binding(statement.name, statement)
                # A function executes later, but it still must not mutate or
                # expose the module's tuning contract. Otherwise import-time
                # equality is only a transient truth and subsequent configs can
                # run under a different search space/default mapping.
                _walk_statements(statement.body, nested=True)
                continue
            child_lists = []
            for field in ("body", "orelse", "finalbody"):
                value = getattr(statement, field, None)
                if isinstance(value, list):
                    child_lists.append(value)
            if isinstance(statement, ast.Try):
                child_lists.extend(handler.body for handler in statement.handlers)
            if isinstance(statement, ast.Match):
                child_lists.extend(case.body for case in statement.cases)
            for children in child_lists:
                _walk_statements(children, nested=True)

    _walk_statements(tree.body, nested=False)
    return errors


def _dict_literal(node: ast.AST) -> tuple[dict, dict]:
    """(evaluated dict, {key: source line}) for a module-level dict literal.
    Raises ValueError if it is not a pure string-keyed dict literal."""
    if not isinstance(node, ast.Dict):
        raise ValueError("is not a dict literal")
    key_lines = {}
    for key_node in node.keys:
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise ValueError("has a non-string key")
        if key_node.value in key_lines:
            raise ValueError(
                f"has duplicate key {key_node.value!r} at lines "
                f"{key_lines[key_node.value]} and {key_node.lineno}"
            )
        key_lines[key_node.value] = key_node.lineno
    try:
        value = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        raise ValueError("is not a pure literal")
    return value, key_lines


def valid_space_entry(entry) -> bool:
    """A SEARCH_SPACE value must be a (kind, ...) tuple the search scripts can
    sample: float (lo, hi[, "log"]), int (lo, hi), or categorical [opts]."""
    if not isinstance(entry, (tuple, list)) or len(entry) < 2:
        return False
    kind = entry[0]
    if kind == "float":
        if len(entry) not in (3, 4) or (len(entry) == 4 and entry[3] != "log"):
            return False
        lo, hi = entry[1], entry[2]
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   for x in (lo, hi)):
            return False
        try:
            low, high = float(lo), float(hi)
        except (OverflowError, TypeError, ValueError):
            return False
        if not math.isfinite(low) or not math.isfinite(high):
            return False
        return low <= high and (len(entry) != 4 or low > 0)
    if kind == "int":
        if len(entry) != 3:
            return False
        lo, hi = entry[1], entry[2]
        return all(isinstance(x, int) and not isinstance(x, bool) for x in (lo, hi)) and lo <= hi
    if kind == "categorical":
        return len(entry) == 2 and _valid_categorical_options(entry[1])
    return False


def lint_contract(train_path: Path, *, require_base_params: bool = True) -> dict:
    """Check the tuner contract's relational invariants by AST. Returns
    {ok, n_dims, keys, make_model_defined, make_model_called, errors[]} where each
    error is {code, detail, line}. `ok` covers the hard invariants; the
    make_model_called field is advisory (best-effort) and never flips `ok`."""
    src = Path(train_path).read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"ok": False, "n_dims": None, "keys": [], "make_model_defined": False,
                "make_model_called": False,
                "errors": [{"code": "syntax_error", "detail": str(exc), "line": exc.lineno or 0}]}

    bindings = _module_bindings(tree)
    syms = {name: nodes[-1] for name, nodes in bindings.items()}
    errors: list = [
        *_contract_declaration_errors(tree),
        *_duplicate_binding_errors(bindings),
        *_contract_mutation_errors(tree),
    ]

    def _read(name, *, required: bool = True):
        if name not in syms:
            if required:
                errors.append({"code": "missing_symbol", "detail": f"no module-level {name}", "line": 0})
            return None, {}
        try:
            return _dict_literal(syms[name])
        except ValueError as exc:
            errors.append({"code": "not_dict_literal", "detail": f"{name} {exc}",
                           "line": getattr(syms[name], "lineno", 0)})
            return None, {}

    schema, schema_lines = _read("PARAM_SCHEMA")
    search_space, space_lines = _read("SEARCH_SPACE")
    base_params, base_lines = _read("BASE_PARAMS", required=require_base_params)

    mm = syms.get("make_model")
    make_model_defined = isinstance(mm, ast.FunctionDef)
    if mm is None:
        errors.append({"code": "missing_symbol", "detail": "no module-level make_model", "line": 0})
    elif not make_model_defined:
        errors.append({"code": "make_model_not_func",
                       "detail": "make_model must be a synchronous def",
                       "line": getattr(mm, "lineno", 0)})

    if isinstance(schema, dict):
        for key in sorted(schema):
            if not _valid_schema_entry(schema[key]):
                errors.append({
                    "code": "bad_schema_entry",
                    "detail": (
                        f"PARAM_SCHEMA['{key}']={_safe_repr(schema[key])} "
                        "is invalid"
                    ),
                    "line": schema_lines.get(key, 0),
                })

    if isinstance(schema, dict) and isinstance(search_space, dict):
        schema_keys, space_keys = set(schema), set(search_space)
        for key in sorted(schema_keys - space_keys):
            errors.append({
                "code": "key_mismatch",
                "detail": f"{key} in PARAM_SCHEMA but not SEARCH_SPACE",
                "line": schema_lines.get(key, 0),
            })
        for key in sorted(space_keys - schema_keys):
            errors.append({
                "code": "key_mismatch",
                "detail": f"{key} in SEARCH_SPACE but not PARAM_SCHEMA",
                "line": space_lines.get(key, 0),
            })
        for key in sorted(schema_keys & space_keys):
            entry = search_space[key]
            if valid_space_entry(entry) and _valid_schema_entry(schema[key]):
                mismatch = _space_schema_mismatch(schema[key], entry)
                if mismatch:
                    errors.append({
                        "code": "schema_mismatch",
                        "detail": f"SEARCH_SPACE['{key}'] {mismatch}",
                        "line": space_lines.get(key, 0),
                    })

    if isinstance(search_space, dict):
        space_keys = set(search_space)
        base_keys = set(base_params) if isinstance(base_params, dict) else set()
        if isinstance(base_params, dict):
            for key in sorted(space_keys - base_keys):
                errors.append({"code": "key_mismatch",
                               "detail": f"{key} in SEARCH_SPACE but not BASE_PARAMS",
                               "line": space_lines.get(key, 0)})
            for key in sorted(base_keys - space_keys):
                errors.append({"code": "key_mismatch",
                               "detail": f"{key} in BASE_PARAMS but not SEARCH_SPACE",
                               "line": base_lines.get(key, 0)})
        elif not require_base_params:
            base_keys = set()
        for key in sorted(space_keys):
            if not valid_space_entry(search_space[key]):
                errors.append({"code": "bad_tuple",
                               "detail": f"SEARCH_SPACE['{key}']={_safe_repr(search_space[key])} is not a valid (kind, ...) entry",
                               "line": space_lines.get(key, 0)})
        for key in sorted(space_keys & base_keys):
            entry = search_space[key]
            if valid_space_entry(entry) and not _value_in_bounds(base_params[key], entry):
                errors.append({"code": "out_of_bounds",
                               "detail": f"BASE_PARAMS['{key}']={_safe_repr(base_params[key])} outside {_safe_repr(list(entry))}",
                               "line": base_lines.get(key, 0)})

    if isinstance(base_params, dict):
        for key in sorted(base_params):
            if isinstance(base_params[key], tuple):
                errors.append({"code": "base_param_tuple",
                               "detail": f"BASE_PARAMS['{key}'] is a tuple; must be a concrete value",
                               "line": base_lines.get(key, 0)})
        if isinstance(schema, dict):
            for key in sorted(set(base_params) & set(schema)):
                if (
                    _valid_schema_entry(schema[key])
                    and not _schema_accepts_value(schema[key], base_params[key])
                ):
                    errors.append({
                        "code": "base_schema_mismatch",
                        "detail": (
                            f"BASE_PARAMS['{key}']={_safe_repr(base_params[key])} is "
                            f"incompatible with PARAM_SCHEMA {_safe_repr(schema[key])}"
                        ),
                        "line": base_lines.get(key, 0),
                    })

    make_model_called = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "make_model"
        for n in ast.walk(tree)
    )

    return {
        "ok": not errors,
        "n_dims": len(search_space) if isinstance(search_space, dict) else None,
        "keys": sorted(search_space) if isinstance(search_space, dict) else [],
        "make_model_defined": make_model_defined,
        "make_model_called": make_model_called,
        "errors": errors,
    }


def lint_schema(train_path: Path) -> dict:
    """Schema-mode contract lint (step 0, before SEARCH_SPACE/BASE_PARAMS exist):
    PARAM_SCHEMA is a valid kind/options declaration and make_model is a function.
    Returns {ok, keys, kinds, make_model_defined, make_model_called, errors[]}."""
    src = Path(train_path).read_text(errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"ok": False, "keys": [], "kinds": {}, "make_model_defined": False,
                "make_model_called": False,
                "errors": [{"code": "syntax_error", "detail": str(exc), "line": exc.lineno or 0}]}
    bindings = _module_bindings(tree)
    syms = {name: nodes[-1] for name, nodes in bindings.items()}
    errors: list = [
        *_contract_declaration_errors(tree),
        *_duplicate_binding_errors(bindings),
        *_contract_mutation_errors(tree),
    ]

    for stray in ("SEARCH_SPACE", "BASE_PARAMS"):
        if stray in syms:
            errors.append({
                "code": "stray_symbol",
                "detail": f"{stray} must be absent during schema extraction",
                "line": getattr(syms[stray], "lineno", 0),
            })

    schema, schema_lines = None, {}
    if "PARAM_SCHEMA" not in syms:
        errors.append({"code": "missing_symbol", "detail": "no module-level PARAM_SCHEMA", "line": 0})
    else:
        try:
            schema, schema_lines = _dict_literal(syms["PARAM_SCHEMA"])
        except ValueError as exc:
            errors.append({"code": "not_dict_literal", "detail": f"PARAM_SCHEMA {exc}",
                           "line": getattr(syms["PARAM_SCHEMA"], "lineno", 0)})
    if isinstance(schema, dict):
        for key in sorted(schema):
            if not _valid_schema_entry(schema[key]):
                errors.append({"code": "bad_schema_entry", "key": key,
                               "detail": f"PARAM_SCHEMA['{key}']={_safe_repr(schema[key])} is not a valid kind/options",
                               "line": schema_lines.get(key, 0)})

    mm = syms.get("make_model")
    make_model_defined = isinstance(mm, ast.FunctionDef)
    if mm is None:
        errors.append({"code": "missing_symbol", "detail": "no module-level make_model", "line": 0})
    elif not make_model_defined:
        errors.append({"code": "make_model_not_func",
                       "detail": "make_model must be a synchronous def",
                       "line": getattr(mm, "lineno", 0)})
    make_model_called = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "make_model"
        for n in ast.walk(tree)
    )
    return {
        "ok": not errors,
        "keys": sorted(schema) if isinstance(schema, dict) else [],
        "kinds": {
            k: kind
            for k, value in schema.items()
            if (kind := _schema_kind(value)) is not None
        } if isinstance(schema, dict) else {},
        "make_model_defined": make_model_defined,
        "make_model_called": make_model_called,
        "errors": errors,
    }


# ---------- best-trial selection ----------


def finite_warm_incumbent_rows(rows) -> list[dict]:
    """Return finite warm rows that may become the candidate incumbent.

    ``inherited_control`` is a fidelity observation on child code. Letting that
    row win would turn reproduction/evaluation variance into a semantic
    candidate improvement and corrupt the lineage base.
    """
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("role") != "inherited_control"
        and isinstance(row.get("params"), dict)
        and _is_finite_score(row.get("score"))
    ]


def _iter_trials(report: dict, *, include_fidelity_controls: bool = False):
    """Yield selectable trials, optionally including scored fidelity controls.

    The inclusive view is accounting-only. Incumbent/final selection always
    uses the default view, which excludes inherited config 0 and any Phase-C
    duplicate of that exact parameter vector.
    """
    phase_a = report.get("phase_a", {})
    inherited_params_sha256: set[str] = set()
    for warm in phase_a.get("warm_start_configs", []):
        if (
            isinstance(warm, dict)
            and warm.get("role") == "inherited_control"
            and isinstance(warm.get("params"), dict)
        ):
            try:
                inherited_params_sha256.add(_json_sha256(warm["params"]))
            except (OverflowError, TypeError, ValueError):
                pass
        if (
            _is_finite_score(warm.get("score"))
            and (
                include_fidelity_controls
                or warm.get("role") != "inherited_control"
            )
        ):
            yield ("warm_start", warm["params"], float(warm["score"]))
    for stage in report.get("phase_c", {}).get("stages", []):
        method = stage.get("method", "phase_c")
        for trial in stage.get("trials", []):
            is_inherited_duplicate = False
            if (
                inherited_params_sha256
                and isinstance(trial.get("params"), dict)
            ):
                try:
                    is_inherited_duplicate = (
                        _json_sha256(trial["params"])
                        in inherited_params_sha256
                    )
                except (OverflowError, TypeError, ValueError):
                    pass
            if (
                _is_finite_score(trial.get("score"))
                and (
                    include_fidelity_controls
                    or not is_inherited_duplicate
                )
            ):
                yield (method, trial["params"], float(trial["score"]))


def _is_finite_score(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def select_best(report: dict) -> dict:
    candidates = list(_iter_trials(report))
    if not candidates:
        return {"best_params": None, "best_score": None, "source": None}
    best = min(candidates, key=lambda c: c[2])
    return {"best_params": best[1], "best_score": best[2], "source": best[0]}


# ---------- Phase-C completion / finalization guard ----------


def validated_phase_a_incumbent(report: dict) -> dict:
    """Return the declared Phase-A incumbent only when observations prove it."""
    if not isinstance(report, dict):
        raise ValueError("tune report must be an object")
    phase_a = report.get("phase_a")
    if not isinstance(phase_a, dict) or phase_a.get("status") != "ok":
        raise ValueError("phase_a.status must be 'ok'")
    warm_rows = phase_a.get("warm_start_configs")
    if not isinstance(warm_rows, list):
        raise ValueError("phase_a.warm_start_configs must be a list")
    finite_warm = finite_warm_incumbent_rows(warm_rows)
    if not finite_warm:
        raise ValueError("phase_a has no finite selectable warm observation")
    warm_best = min(finite_warm, key=lambda row: float(row["score"]))
    if phase_a.get("best_warm_params") != warm_best["params"]:
        raise ValueError(
            "phase_a.best_warm_params does not match its finite warm best"
        )
    if (
        not _is_finite_score(phase_a.get("best_warm_score"))
        or float(phase_a["best_warm_score"]) != float(warm_best["score"])
    ):
        raise ValueError(
            "phase_a.best_warm_score does not match its finite warm best"
        )
    return {
        "params": warm_best["params"],
        "score": float(warm_best["score"]),
    }


_TERMINAL_STAGE_STATUSES = {
    "ok",
    "failed",
    "rejected",
    "budget_exhausted",
    "no_search_needed",
    "time_exhausted",
}
# Every terminal status except `rejected` can close a candidate.  A rejected
# stage is a method the deterministic chain never ran, so it carries no
# observation to finalize; the exhausted-chain path below handles a report whose
# stages are all rejected.
_FINALIZABLE_STAGE_STATUSES = _TERMINAL_STAGE_STATUSES - {"rejected"}


def finalizable_tuning_result(report: dict, *, require_applied: bool = False) -> dict:
    """Return the global best once Phase C reached a terminal state.

    Stage status records *how the search ended*; it does not gate *what the
    search observed*.  Every finite trial row in every stage is admissible
    evidence regardless of that status, because two other mechanisms already
    bind those rows to the candidate on disk:

    * ``deep_tune_time_budget`` calls ``validate_candidate_execution_revision``
      on every invocation before a trial can be appended, so all rows in a
      stage share one pinned candidate/evaluator revision; and
    * ``validate_report_trial_rows`` bounds-checks every row against the live
      ``PARAM_SCHEMA``/``SEARCH_SPACE`` and rejects a finite score carrying a
      non-ok status.

    A killed search's rows are therefore proven observations, not suspect ones.
    Discarding them used to lose real, already-paid-for evaluations whenever the
    last invocation happened to complete nothing — an interrupted resume, a
    budget death, or a run of crashing configs.

    ``running`` is deliberately *not* terminal: a live process may still append
    to that stage, so it must be closed first (see ``close_exhausted_stage``).
    Earlier stages may be ``rejected`` when the deterministic fallback chain
    selected another method.

    Bouts: stages are grouped by ``bout_index`` (legacy stages: bout 0). The
    per-bout prefix/chain rules mirror the single-pass rules; finalization is
    valid when every stage is terminal and the LAST bout's final stage is
    finalizable (or that bout's whole chain was rejected). The best spans all
    bouts, so a continuation close can never regress the score.
    """
    from _common import stages_by_bout

    if not isinstance(report, dict):
        raise ValueError("tuning report is not finalizable: report must be an object")
    errors: list[str] = []
    phase_a = report.get("phase_a")
    if not isinstance(phase_a, dict) or phase_a.get("status") != "ok":
        errors.append("phase_a.status must be 'ok'")
        phase_a = {}

    warm_rows = phase_a.get("warm_start_configs")
    if not isinstance(warm_rows, list):
        errors.append("phase_a.warm_start_configs must be a list")
        warm_rows = []
    finite_warm = finite_warm_incumbent_rows(warm_rows)
    warm_best = (
        min(finite_warm, key=lambda row: float(row["score"]))
        if finite_warm
        else None
    )
    if warm_best is None:
        errors.append("phase_a has no finite selectable warm observation")
    else:
        if phase_a.get("best_warm_params") != warm_best["params"]:
            errors.append(
                "phase_a.best_warm_params does not match its finite warm best"
            )
        if (
            not _is_finite_score(phase_a.get("best_warm_score"))
            or float(phase_a["best_warm_score"]) != float(warm_best["score"])
        ):
            errors.append(
                "phase_a.best_warm_score does not match its finite warm best"
            )

    phase_c = report.get("phase_c")
    stages = phase_c.get("stages") if isinstance(phase_c, dict) else None
    if not isinstance(stages, list) or not stages:
        errors.append("phase_c.stages must be a non-empty list")
        stages = []

    statuses: list[str | None] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            errors.append(f"phase_c.stages[{index}] must be an object")
            statuses.append(None)
            continue
        method = stage.get("method")
        if method not in {"grid", "bo", "cmaes"}:
            errors.append(f"phase_c.stages[{index}].method is invalid")
        status = stage.get("status")
        statuses.append(status if isinstance(status, str) else None)
        if status not in _TERMINAL_STAGE_STATUSES:
            errors.append(
                f"phase_c.stages[{index}].status is not terminal "
                f"(got {status!r})"
            )
        if (
            status == "rejected"
            and isinstance(stage, dict)
            and stage.get("trials") not in (None, [])
        ):
            errors.append(
                f"phase_c.stages[{index}] is rejected but contains trial rows"
            )

    # A non-dict stage already failed above; grouping it would crash here, so
    # the per-bout checks simply yield to that collected error.
    bouts = (
        stages_by_bout(stages)
        if all(isinstance(stage, dict) for stage in stages)
        else []
    )
    offset = 0
    for bout in bouts:
        for index, stage in enumerate(bout[:-1]):
            if stage.get("status") != "rejected":
                errors.append(
                    f"phase_c.stages[{offset + index}] precedes another stage "
                    "of its bout but is not rejected"
                )
        offset += len(bout)

    search_space = phase_a.get("search_space")
    expected_methods: list[str] = []
    if isinstance(search_space, dict) and search_space:
        expected = select_method(len(search_space))
        expected_methods = [expected["method"], *expected["fallback"]]
        for bout in bouts:
            bout_methods = [stage.get("method") for stage in bout]
            if bout_methods != expected_methods[:len(bout_methods)]:
                errors.append(
                    f"Phase-C method chain {bout_methods!r} does not match "
                    f"deterministic chain {expected_methods!r}"
                )

    # A chain is exhausted only when the LAST bout rejected every method —
    # earlier bouts closed on their own finalizable stage and prove nothing
    # about the continuation's chain.
    last_bout = bouts[-1] if bouts else []
    last_methods = [stage.get("method") for stage in last_bout]
    last_statuses = [stage.get("status") for stage in last_bout]
    exhausted_rejections = bool(
        expected_methods
        and last_methods == expected_methods
        and last_statuses == ["rejected"] * len(expected_methods)
    )
    if statuses and (
        statuses[-1] not in _FINALIZABLE_STAGE_STATUSES
        and not exhausted_rejections
    ):
        errors.append(
            "final Phase-C stage must end with status "
            + ", ".join(repr(s) for s in sorted(_FINALIZABLE_STAGE_STATUSES))
            + ", unless every method in the deterministic fallback chain "
            "was rejected"
        )

    # The "final stage" the terminal-status receipt checks below bind to is the
    # final stage of the LAST bout — the same object as stages[-1].
    final_stage = stages[-1] if stages else {}
    final_status = statuses[-1] if statuses else None
    final_trials = (
        final_stage.get("trials", [])
        if isinstance(final_stage, dict)
        else []
    )
    if not isinstance(final_trials, list):
        errors.append("final Phase-C stage trials must be a list")
        final_trials = []
    finite_phase_c_rows = [
        (str(stage.get("method")), row)
        for stage in stages
        if isinstance(stage, dict) and isinstance(stage.get("trials"), list)
        for row in stage["trials"]
        if isinstance(row, dict)
        and isinstance(row.get("params"), dict)
        and _is_finite_score(row.get("score"))
    ]

    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or stage.get("status") != "ok":
            continue
        ok_trials = stage.get("trials")
        if not isinstance(ok_trials, list) or not any(
            isinstance(row, dict) and _is_finite_score(row.get("score"))
            for row in ok_trials
        ):
            errors.append(
                f"phase_c.stages[{index}] is ok but must contain a finite trial"
            )
    if exhausted_rejections and any(
        stage.get("trials") not in (None, [])
        for stage in last_bout
    ):
        errors.append(
            "an exhausted rejected method chain cannot contain trial rows"
        )
    if final_status == "time_exhausted":
        time_limit = final_stage.get("time_limit_seconds")
        elapsed_rows = [
            stage.get("elapsed_seconds")
            for stage in stages
            if isinstance(stage, dict)
            and _is_finite_score(stage.get("elapsed_seconds"))
            and float(stage["elapsed_seconds"]) >= 0
        ]
        candidate_elapsed = sum(float(value) for value in elapsed_rows)
        if not (
            final_stage.get("early_stop_reason") == "time_budget"
            and _is_finite_score(final_stage.get("elapsed_seconds"))
            and float(final_stage["elapsed_seconds"]) >= 0
            and _is_finite_score(time_limit)
            and float(time_limit) > 0
            and candidate_elapsed >= float(time_limit) - 0.1
        ):
            errors.append(
                "time_exhausted requires a cumulative elapsed/time-limit receipt"
            )
    if final_status == "no_search_needed":
        effective_space = final_stage.get("effective_search_space", {})
        fixed = (
            isinstance(effective_space, dict)
            and effective_space
            and all(
                valid_space_entry(entry)
                and (
                    (entry[0] in {"float", "int"} and entry[1] == entry[2])
                    or (
                        entry[0] == "categorical"
                        and len(entry[1]) == 1
                    )
                )
                for entry in effective_space.values()
            )
        )
        fixed_params = final_stage.get("fixed_incumbent_params")
        fixed_score = final_stage.get("fixed_incumbent_score")
        if not (
            final_stage.get("method") == "cmaes"
            and final_stage.get("fixed_search_space") is True
            and final_stage.get("early_stop_reason") == "fixed_search_space"
            and final_stage.get("trials_attempted") == 0
            and final_stage.get("trials_completed") == 0
            and final_trials == []
            and fixed
            and warm_best is not None
            and fixed_params == warm_best["params"]
            and _is_finite_score(fixed_score)
            and float(fixed_score) == float(warm_best["score"])
            and not _bounds_violations(fixed_params, effective_space)
        ):
            errors.append(
                "no_search_needed requires a fixed CMA-ES space bound to the "
                "finite Phase-A incumbent"
            )

    eligible_rows = []
    if warm_best is not None:
        eligible_rows.append(("warm_start", warm_best))
    # Every finite Phase-C row across ALL bouts competes with the Phase-A
    # incumbent, whatever terminal status its stage carries (see the docstring
    # for why those rows are already proven). A later bout is not trusted to
    # beat earlier ones, so the argmin spans the whole history. No status
    # filter is needed here: `rejected` stages are validated above to carry no
    # trial rows, so they contribute nothing.
    eligible_rows.extend(finite_phase_c_rows)
    if eligible_rows:
        source, best_row = min(
            eligible_rows, key=lambda item: float(item[1]["score"])
        )
        best = {
            "best_params": best_row["params"],
            "best_score": float(best_row["score"]),
            "source": source,
        }
    else:
        best = {"best_params": None, "best_score": None, "source": None}
        errors.append("no finite finalizable best is available")

    closing_fields = {
        "final_best_params": report.get("final_best_params"),
        "final_best_score": report.get("final_best_score"),
        "applied_to_base_params": report.get("applied_to_base_params"),
    }
    closing_present = any(value is not None for value in closing_fields.values())
    if closing_present or require_applied:
        if closing_fields["final_best_params"] != best.get("best_params"):
            errors.append("final_best_params does not match the report's global best")
        if not _is_finite_score(closing_fields["final_best_score"]) or (
            _is_finite_score(best.get("best_score"))
            and float(closing_fields["final_best_score"]) != float(best["best_score"])
        ):
            errors.append("final_best_score does not match the report's global best")
        if closing_fields["applied_to_base_params"] is not True:
            errors.append("applied_to_base_params must be true")

    if errors:
        raise ValueError("tuning report is not finalizable: " + "; ".join(errors))

    return {
        **best,
        # The applied observation's own provenance, not the stage's status: a
        # Phase-C row that wins carries its method however the stage ended.
        "phase_c_method": (
            None if best.get("source") in (None, "warm_start") else best["source"]
        ),
        "stage_statuses": statuses,
    }


def last_finalized_stage_index(report: dict) -> int | None:
    """Stage index the last finalize close covered (legacy closes: absent)."""
    if not isinstance(report, dict):
        return None
    value = report.get("last_finalized_stage_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def has_applied_close(report: dict) -> bool:
    """Whether a finalize close (any bout) applied its incumbent to BASE_PARAMS.

    The closing fields must prove consistent with the stages they cover —
    i.e. the stage prefix up to ``last_finalized_stage_index`` (legacy closes
    without the field covered every stage, which is all a one-shot report can
    have)."""
    if not isinstance(report, dict) or report.get("applied_to_base_params") is not True:
        return False
    stages = report.get("phase_c", {}).get("stages", [])
    last = last_finalized_stage_index(report)
    if last is None:
        last = len(stages) - 1
    phase_c = report.get("phase_c", {})
    prefix_report = {
        **report,
        "phase_c": {**phase_c, "stages": stages[: last + 1]},
    }
    finalizable_tuning_result(prefix_report, require_applied=True)
    return True


def has_validated_applied_close(report: dict) -> bool:
    """Whether the applied close is CURRENT: it covers every stage in the report."""
    if not has_applied_close(report):
        return False
    stages = report.get("phase_c", {}).get("stages", [])
    last = last_finalized_stage_index(report)
    return last is None or last == len(stages) - 1


def _unresumable_budget_scope(candidate_path: Path) -> tuple[str | None, str]:
    """Prove whether no Phase-C reservation can still be admitted for a candidate.

    Returns ``(scope, detail)``: ``scope`` names the exhausted budget and is
    ``None`` while any reservation remains admissible; ``detail`` records the
    numbers behind the decision for refusals and receipts.  A candidate
    outside a run directory has no evaluation budget to prove against, so it
    always reads as resumable.
    """
    run_dir = find_run_dir(candidate_path)
    if run_dir is None:
        return None, "no enclosing run directory"
    run_id = Path(candidate_path).parent.name
    status_view = budget_status(run_dir)
    deep = status_view.get("deep_tune") or {}
    per_candidate = {
        str(row.get("run_id")): row.get("evals", 0)
        for row in deep.get("per_candidate", [])
        if isinstance(row, dict)
    }
    candidate_used = per_candidate.get(run_id, 0)
    candidate_cap = deep.get("per_candidate_cap")
    detail = (
        f"global remaining={status_view.get('remaining')!r}, "
        f"deep-tune remaining={deep.get('remaining')!r}, "
        f"candidate {run_id} used {candidate_used}/{candidate_cap!r}"
    )
    if isinstance(status_view.get("remaining"), int) and status_view["remaining"] <= 0:
        return "global", detail
    if isinstance(deep.get("remaining"), int) and deep["remaining"] <= 0:
        return "deep_tune_total", detail
    if isinstance(candidate_cap, int) and candidate_used >= candidate_cap:
        return f"deep_tune_candidate:{run_id}", detail
    return None, detail


def phase_c_action(report: dict, candidate_path: Path) -> dict:
    """Return the one legal resume action for a candidate's Phase-C state."""
    applied_close = has_validated_applied_close(report)
    validate_phase_a_candidate_state(
        report,
        candidate_path,
        require_warm_base_applied=not has_applied_close(report),
    )
    search_space = _read_search_space(candidate_path)

    selected = select_method(len(search_space))
    method_chain = [selected["method"], *selected["fallback"]]
    common = {
        "n_dims": selected["n_dims"],
        "method_chain": method_chain,
    }
    phase_c = report.get("phase_c")
    if phase_c is not None and not isinstance(phase_c, dict):
        raise ValueError("phase_c must be an object")
    if phase_c is None or "stages" not in phase_c:
        stages = []
    else:
        stages = phase_c["stages"]
    if stages == []:
        return {
            **common,
            "action": "run",
            "method": method_chain[0],
            "reason": "phase_c_not_started",
            "bout_index": 0,
        }
    if not isinstance(stages, list) or not all(
        isinstance(stage, dict) for stage in stages
    ):
        raise ValueError("phase_c.stages must be a list of objects")
    from _common import stages_by_bout

    bouts = stages_by_bout(stages)
    for bout in bouts:
        bout_methods = [stage.get("method") for stage in bout]
        if (
            len(bout) > len(method_chain)
            or bout_methods != method_chain[:len(bout)]
        ):
            raise ValueError(
                f"Phase-C method chain {bout_methods!r} does not match "
                f"{method_chain!r}"
            )
        if any(
            stage.get("status") != "rejected"
            or stage.get("trials") not in (None, [])
            for stage in bout[:-1]
        ):
            raise ValueError(
                "every Phase-C stage before its bout's active/final stage "
                "must be an empty rejected stage"
            )

    current = bouts[-1]
    bout_index = len(bouts) - 1
    methods = [stage.get("method") for stage in current]
    final_stage = current[-1]
    final_status = final_stage.get("status")
    if final_status == "running":
        scope, _detail = _unresumable_budget_scope(candidate_path)
        if scope is not None:
            # The budget proves no invocation can ever resume this stage, so
            # "run" advice cannot succeed; the deterministic close is the only
            # legal move (prompt-only routing here stranded durable trials).
            return {
                **common,
                "action": "close_exhausted_stage",
                "method": methods[-1],
                "reason": "evaluation_budget_reached",
                "budget_scope": scope,
                "bout_index": bout_index,
            }
        return {
            **common,
            "action": "run",
            "method": methods[-1],
            "reason": "resume_interrupted_stage",
            "bout_index": bout_index,
        }
    if final_status == "rejected":
        if final_stage.get("trials") not in (None, []):
            raise ValueError("a rejected Phase-C stage cannot contain trials")
        if len(current) < len(method_chain):
            return {
                **common,
                "action": "run",
                "method": method_chain[len(current)],
                "reason": "run_deterministic_fallback",
                "bout_index": bout_index,
            }
        if applied_close:
            return _start_new_bout(common, method_chain, bout_index)
        result = finalizable_tuning_result(report)
        return {
            **common,
            "action": "finalize",
            "method": None,
            "reason": "method_chain_exhausted",
            "best_score": result["best_score"],
            "bout_index": bout_index,
        }
    if applied_close:
        return _start_new_bout(common, method_chain, bout_index)
    result = finalizable_tuning_result(report)
    return {
        **common,
        "action": "finalize",
        "method": None,
        "reason": f"terminal_{final_status}",
        "best_score": result["best_score"],
        "bout_index": bout_index,
    }


def _start_new_bout(common: dict, method_chain: list, bout_index: int) -> dict:
    """The previous bout is closed and finalized; begin the next one."""
    return {
        **common,
        "action": "run",
        "method": method_chain[0],
        "reason": "start_new_bout",
        "bout_index": bout_index + 1,
    }


def close_exhausted_stage(candidate_path: Path, report_path: Path) -> dict:
    """Close a `running` Phase-C stage the evaluation budget can never resume.

    An interrupted stage stays `running` until some invocation closes it, but a
    candidate at its deep-tune cap is never selected again, so no invocation
    ever comes.  Its durable trials — real, already-charged evaluations — would
    be stranded forever.  This is the deterministic close for that state.

    Admissible only when the reservation ledger proves no further Phase-C
    reservation can be admitted for this candidate: the global cap, the
    deep-tune total cap, or the per-candidate cap is spent.  Under that proof no
    new trial can be appended, because `reserve_evaluation` would raise before
    `score_fn`.  The residual risk is a straggler trial from an already-paid
    reservation landing after the close, which `trials_at_close` makes
    detectable rather than silent.

    Idempotent: closing an already-terminal stage is a no-op receipt.
    """
    from _common import read_tune_report, set_stage_meta

    candidate_path = Path(candidate_path)
    report_path = Path(report_path)
    report = read_tune_report(report_path)
    stages = report.get("phase_c", {}).get("stages") if isinstance(report, dict) else None
    if not isinstance(stages, list) or not stages:
        raise ValueError("phase_c.stages must be a non-empty list to close")
    final_stage = stages[-1]
    if not isinstance(final_stage, dict):
        raise ValueError("the final Phase-C stage must be an object")
    method = final_stage.get("method")
    status = final_stage.get("status")
    trials = final_stage.get("trials") or []

    if status != "running":
        return {
            "action": "noop",
            "reason": f"final stage is already terminal (status={status!r})",
            "method": method,
            "status": status,
            "trials_at_close": len(trials),
        }

    run_dir = find_run_dir(candidate_path)
    if run_dir is None:
        raise ValueError(
            f"{candidate_path} is not inside a run directory; "
            "no evaluation budget can prove this stage unresumable"
        )
    scope, detail = _unresumable_budget_scope(candidate_path)
    if scope is None:
        raise ValueError(
            "refusing to close a running Phase-C stage while the budget still "
            f"admits a reservation ({detail})"
        )

    set_stage_meta(
        report_path,
        method,
        status="budget_exhausted",
        closed_reason="evaluation_budget_reached",
        budget_scope=scope,
        trials_at_close=len(trials),
        closed_without_invocation=True,
    )
    return {
        "action": "closed",
        "reason": "evaluation_budget_reached",
        "method": method,
        "status": "budget_exhausted",
        "budget_scope": scope,
        "trials_at_close": len(trials),
    }


# ---------- primary-parent parameter inheritance ----------


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _json_sha256(value) -> str:
    return _sha256_bytes(_canonical_json(value))


def _json_native(value):
    """Canonical JSON round-trip so persisted receipts compare after reload."""
    return json.loads(_canonical_json(value))


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _resolve_receipt_path(value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_literal_mapping(train_path: Path, name: str) -> dict:
    tree = ast.parse(Path(train_path).read_text(errors="replace"))
    declaration_errors = [
        error
        for error in _contract_declaration_errors(tree)
        if error.get("symbol") == name
    ]
    if declaration_errors:
        raise ValueError(declaration_errors[0]["detail"])
    bindings = _module_bindings(tree)
    nodes = bindings.get(name, [])
    if not nodes:
        raise ValueError(f"no module-level {name} found in {train_path}")
    if len(nodes) != 1:
        raise ValueError(
            f"{name} must have exactly one module-level binding in {train_path}; "
            f"found {len(nodes)} at lines "
            f"{[getattr(node, 'lineno', 0) for node in nodes]}"
        )
    node = nodes[0]
    try:
        value, _ = _dict_literal(node)
    except ValueError as exc:
        raise ValueError(f"{name} in {train_path} {exc}") from exc
    return value


def _candidate_structure_sha256(candidate_path: Path) -> str:
    """Hash strategy-bearing code while ignoring materialized tuning literals.

    SEARCH_SPACE is absent when transfer is first built and BASE_PARAMS is
    absent until warm evaluation.  Removing both assignments makes the same
    receipt valid across those deterministic materialization steps, while any
    edit to PARAM_SCHEMA or executable candidate code invalidates it.
    """
    tree = ast.parse(Path(candidate_path).read_text(errors="replace"))
    normalized_body = []
    for node in tree.body:
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        names = {
            target.id
            for target in targets
            if isinstance(target, ast.Name)
        }
        if names & {"BASE_PARAMS", "SEARCH_SPACE"}:
            continue
        normalized_body.append(node)
    tree.body = normalized_body
    normalized = ast.dump(
        tree,
        annotate_fields=True,
        include_attributes=False,
    ).encode()
    return _sha256_bytes(normalized)


def _candidate_execution_revision(candidate_path: Path) -> dict:
    """Bind reusable score rows to candidate code and its evaluation surface."""
    candidate_path = Path(candidate_path).resolve()
    search_space = _read_literal_mapping(candidate_path, "SEARCH_SPACE")
    prepare_path = candidate_path.parent / "prepare.py"
    if not prepare_path.is_file():
        raise ValueError(
            f"candidate execution revision requires {prepare_path}"
        )

    task_name = None
    parts = candidate_path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "runs" and index + 1 < len(parts):
            task_name = parts[index + 1]
            break

    evaluation_contract = {
        "task_name": task_name,
        "declaration": {
            "score_fn": "evaluate_config",
        },
        "task_toml_sha256": None,
    }
    if task_name is not None:
        task_toml = REPO_ROOT / "tasks" / task_name / "task.toml"
        if task_toml.is_file():
            try:
                task_doc = tomllib.loads(task_toml.read_text())
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise ValueError(
                    f"cannot read evaluation contract {task_toml}: {exc}"
                ) from exc
            declaration = task_doc.get("evaluation", {})
            if not isinstance(declaration, dict):
                raise ValueError(
                    f"{task_toml} [evaluation] must be a table"
                )
            evaluation_contract = {
                "task_name": task_name,
                "declaration": _json_native(declaration),
                "task_toml_sha256": _file_sha256(task_toml),
            }

    revision = {
        "schema_version": 3,
        "structure_sha256": _candidate_structure_sha256(candidate_path),
        "search_space": _json_native(search_space),
        # Mapping order drives deterministic grid/CMA encodings. Dict equality
        # and canonical JSON hashes intentionally ignore it, so carry the key
        # sequence as an explicit part of the execution revision.
        "search_space_keys": list(search_space),
        "search_space_sha256": _json_sha256(search_space),
        "prepare_sha256": _file_sha256(prepare_path),
        "evaluation_contract": evaluation_contract,
    }
    revision["revision_sha256"] = _json_sha256(revision)
    return revision


def validate_candidate_execution_revision(
    report: dict,
    candidate_path: Path,
) -> dict:
    """Prove that report scores belong to the candidate/evaluator now on disk."""
    if not isinstance(report, dict):
        raise ValueError("tune report must be an object")
    phase_a = report.get("phase_a")
    if not isinstance(phase_a, dict):
        raise ValueError("tune report requires a phase_a object")
    recorded = phase_a.get("candidate_code_revision")
    current = _candidate_execution_revision(candidate_path)
    if recorded != current:
        raise ValueError(
            "phase_a candidate execution revision does not match the current "
            "candidate/evaluator; rerun warm evaluation before tuning or "
            "finalization"
        )
    return current


def validate_phase_a_candidate_state(
    report: dict,
    candidate_path: Path,
    *,
    require_warm_base_applied: bool = True,
) -> dict:
    """Bind Phase-A observations and incumbent to the candidate on disk."""
    candidate_path = Path(candidate_path)
    contract = lint_contract(candidate_path)
    if not contract["ok"]:
        raise ValueError(
            "candidate tuning contract is invalid: "
            + json.dumps(contract["errors"], ensure_ascii=False)
        )
    incumbent = validated_phase_a_incumbent(report)
    validate_candidate_execution_revision(report, candidate_path)
    schema = _read_param_schema(candidate_path)
    search_space = _read_search_space(candidate_path)
    current_space = json.loads(json.dumps(search_space, allow_nan=False))
    phase_a = report["phase_a"]
    if phase_a.get("search_space") != current_space:
        raise ValueError(
            "phase_a.search_space does not match the current SEARCH_SPACE"
        )
    validate_report_trial_rows(
        report,
        candidate_path,
        schema=schema,
        search_space=search_space,
    )

    _validate_schema_values(
        incumbent["params"],
        schema,
        label="phase_a incumbent",
    )
    violations = _bounds_violations(incumbent["params"], search_space)
    if violations:
        raise ValueError(
            "phase_a incumbent violates SEARCH_SPACE: "
            + json.dumps(violations, ensure_ascii=False)
        )
    if (
        require_warm_base_applied
        and _read_literal_mapping(candidate_path, "BASE_PARAMS")
        != incumbent["params"]
    ):
        raise ValueError(
            "candidate BASE_PARAMS do not equal the proven Phase-A incumbent"
        )
    return incumbent


def validate_report_trial_rows(
    report: dict,
    candidate_path: Path,
    *,
    schema: dict | None = None,
    search_space: dict | None = None,
) -> None:
    """Validate every persisted params/score row against one candidate."""
    if not isinstance(report, dict):
        raise ValueError("tune report must be an object")
    if schema is None:
        schema = _read_param_schema(candidate_path)
    if search_space is None:
        search_space = _read_search_space(candidate_path)

    def _validate_rows(
        rows,
        *,
        label: str,
        require_score: bool,
    ) -> None:
        if not isinstance(rows, list):
            raise ValueError(f"{label} must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("params"), dict):
                raise ValueError(f"{label}[{index}] must contain params object")
            params = row["params"]
            _validate_schema_values(
                params,
                schema,
                label=f"{label}[{index}].params",
            )
            violations = _bounds_violations(params, search_space)
            if violations:
                raise ValueError(
                    f"{label}[{index}] violates SEARCH_SPACE: "
                    + json.dumps(violations, ensure_ascii=False)
                )
            if require_score and "score" not in row:
                raise ValueError(f"{label}[{index}] must contain score")
            if (
                "score" in row
                and row["score"] is not None
                and not _is_finite_score(row["score"])
            ):
                raise ValueError(
                    f"{label}[{index}].score must be finite or null"
                )
            if require_score:
                score = row["score"]
                status = row.get("status")
                if _is_finite_score(score) and status not in (None, "ok"):
                    raise ValueError(
                        f"{label}[{index}] finite score cannot have "
                        f"status {status!r}"
                    )
                if score is None and status not in {
                    "failed",
                    "preflight_rejected",
                }:
                    raise ValueError(
                        f"{label}[{index}] null score requires failed or "
                        "preflight_rejected status"
                    )

    phase_a = report.get("phase_a")
    if not isinstance(phase_a, dict):
        raise ValueError("tune report requires a phase_a object")
    _validate_rows(
        phase_a.get("warm_start_configs"),
        label="phase_a.warm_start_configs",
        require_score=True,
    )
    _validate_rows(
        phase_a.get("deferred_configs", []),
        label="phase_a.deferred_configs",
        require_score=False,
    )

    phase_c = report.get("phase_c")
    if phase_c is None:
        return
    if not isinstance(phase_c, dict):
        raise ValueError("phase_c must be an object")
    stages = phase_c.get("stages", [])
    if not isinstance(stages, list):
        raise ValueError("phase_c.stages must be a list")
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(
                f"phase_c.stages[{stage_index}] must be an object"
            )
        _validate_rows(
            stage.get("trials", []),
            label=f"phase_c.stages[{stage_index}].trials",
            require_score=True,
        )


def _schema_accepts_value(entry, value) -> bool:
    kind = _schema_kind(entry)
    if kind == "int":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        )
    if kind == "float":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError):
            valid = False
            numeric = float("nan")
        valid = valid and math.isfinite(numeric)
        if (
            valid
            and isinstance(entry, (tuple, list))
            and len(entry) == 2
            and entry[1] == "log"
        ):
            valid = numeric > 0
        return valid
    if kind == "categorical":
        return (
            isinstance(entry, (tuple, list))
            and len(entry) == 2
            and _valid_categorical_options(entry[1])
            and _valid_categorical_value(value)
            and _categorical_contains(entry[1], value)
        )
    return False


def _validate_schema_values(
    values: dict,
    schema: dict,
    *,
    label: str,
) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{label} must be an object")
    bad_schema = [
        key
        for key, entry in schema.items()
        if not isinstance(key, str) or not _valid_schema_entry(entry)
    ]
    if bad_schema:
        raise ValueError(f"{label} schema has invalid entries for {bad_schema}")
    if set(values) != set(schema):
        raise ValueError(
            f"{label} keys {sorted(values)} do not match schema keys "
            f"{sorted(schema)}"
        )
    invalid = [
        key
        for key, value in values.items()
        if not _schema_accepts_value(schema[key], value)
    ]
    if invalid:
        raise ValueError(
            f"{label} has schema-incompatible values for {sorted(invalid)}"
        )


def _read_candidate_brief(candidate_path: Path) -> tuple[Path, dict]:
    brief_path = Path(candidate_path).parent / "_candidate_brief.json"
    try:
        brief = json.loads(brief_path.read_text())
    except OSError as exc:
        raise ValueError(f"missing candidate brief {brief_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid candidate brief {brief_path}: {exc}") from exc
    if not isinstance(brief, dict):
        raise ValueError("candidate brief must be an object")
    return brief_path, brief


def _primary_parent_from_brief(
    candidate_path: Path,
) -> tuple[Path, dict, dict]:
    brief_path, brief = _read_candidate_brief(candidate_path)
    if brief.get("schema_version") != 4:
        raise ValueError("parameter inheritance requires a schema-4 candidate brief")
    if brief.get("run_id") != Path(candidate_path).resolve().parent.name:
        raise ValueError("candidate brief run_id does not match its directory")
    source_run_ids = brief.get("source_run_ids")
    if (
        not isinstance(source_run_ids, list)
        or not source_run_ids
        or any(
            not isinstance(value, str) or not value.isdigit()
            for value in source_run_ids
        )
    ):
        raise ValueError("parameter inheritance requires non-empty source_run_ids")
    primary = brief.get("primary_parent")
    if not isinstance(primary, dict) or primary.get("schema_version") != 1:
        raise ValueError("schema-4 non-fresh candidate requires primary_parent schema 1")
    primary_run_id = source_run_ids[0]
    if primary.get("run_id") != primary_run_id:
        raise ValueError("primary_parent.run_id must equal source_run_ids[0]")
    source = brief.get("implementation_source")
    if (
        not isinstance(source, dict)
        or source.get("kind") != "primary_parent_snapshot"
        or source.get("parent_run_id") != primary_run_id
        or source.get("path") != primary.get("path")
        or source.get("sha256") != primary.get("sha256")
    ):
        raise ValueError(
            "non-fresh candidate implementation_source must pin the primary parent snapshot"
        )
    parent_train = _resolve_receipt_path(
        primary.get("path"),
        field="primary_parent.path",
    )
    expected_parent = (
        Path(candidate_path).resolve().parent.parent
        / primary_run_id
        / Path(candidate_path).name
    ).resolve()
    if parent_train != expected_parent:
        raise ValueError(
            "primary_parent.path must identify source_run_ids[0] in this run"
        )
    if not parent_train.is_file():
        raise ValueError(f"primary parent entrypoint does not exist: {parent_train}")
    if primary.get("sha256") != _file_sha256(parent_train):
        raise ValueError("primary parent entrypoint no longer matches its candidate brief")
    return brief_path, brief, {
        "run_id": primary_run_id,
        "train_path": parent_train,
    }


def authoritative_parent_incumbent(
    parent_train: Path,
    parent_run_id: str,
) -> dict:
    """Resolve only an incumbent that is demonstrably applied to the parent.

    A running or interrupted Phase C may already contain attractive trial rows.
    Those rows are deliberately ignored until the report has valid closing
    fields and ``applied_to_base_params=true``.  Before that boundary, the
    applied Phase-A best remains authoritative.
    """
    parent_train = Path(parent_train).resolve()
    candidate_dir = parent_train.parent
    if candidate_dir.name != parent_run_id:
        raise ValueError(
            "primary-parent run_id does not match its candidate directory"
        )
    run_dir = candidate_dir.parent.parent
    report_path = candidate_dir / "tune_report.json"
    ledger_path = run_dir / "ledger.json"
    try:
        report = json.loads(report_path.read_text())
    except OSError as exc:
        raise ValueError(f"missing primary-parent tune report {report_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid primary-parent tune report {report_path}: {exc}") from exc

    validate_candidate_execution_revision(report, parent_train)
    closing_keys = (
        "final_best_params",
        "final_best_score",
        "applied_to_base_params",
    )
    closing_present = any(
        key in report and report.get(key) is not None
        for key in closing_keys
    )
    if closing_present:
        try:
            final = finalizable_tuning_result(report, require_applied=True)
        except ValueError as exc:
            raise ValueError(
                f"primary parent {parent_run_id} has untrusted closing state: {exc}"
            ) from exc
        params = final["best_params"]
        score = float(final["best_score"])
        source = "finalized_phase_c"
    else:
        try:
            phase_a_best = validated_phase_a_incumbent(report)
        except ValueError as exc:
            raise ValueError(
                f"primary parent {parent_run_id} has invalid Phase-A best: {exc}"
            ) from exc
        params = phase_a_best["params"]
        score = phase_a_best["score"]
        source = "applied_phase_a"

    contract = lint_contract(parent_train)
    if not contract["ok"]:
        raise ValueError(
            f"primary parent {parent_run_id} has an invalid tuner contract: "
            + json.dumps(contract["errors"], ensure_ascii=False)
        )
    parent_schema = _read_param_schema(parent_train)
    parent_base = _read_literal_mapping(parent_train, "BASE_PARAMS")
    _validate_schema_values(
        parent_base,
        parent_schema,
        label=f"primary parent {parent_run_id} BASE_PARAMS",
    )
    if parent_base != params:
        raise ValueError(
            f"primary parent {parent_run_id} BASE_PARAMS do not equal its "
            f"{source} incumbent"
        )

    try:
        ledger_doc = json.loads(ledger_path.read_text())
    except OSError as exc:
        raise ValueError(f"missing parent ledger {ledger_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid parent ledger {ledger_path}: {exc}") from exc
    matching = [
        record
        for record in ledger_doc.get("records", [])
        if isinstance(record, dict) and record.get("run_id") == parent_run_id
    ]
    if len(matching) != 1:
        raise ValueError(
            f"ledger must contain exactly one primary-parent record {parent_run_id}"
        )
    ledger_record = matching[0]
    ledger_score = ledger_record.get("final_best_score")
    if not _is_finite_score(ledger_score) or float(ledger_score) != score:
        raise ValueError(
            f"primary parent {parent_run_id} ledger score does not match "
            f"its applied incumbent"
        )
    applied_snapshot = ledger_record.get("applied_incumbent")
    # A Phase-A snapshot pins the report bytes that established the applied
    # incumbent.  Phase C deliberately grows that same report while the
    # Phase-A BASE_PARAMS remain authoritative, so comparing the snapshot with
    # the live report hash would make ordinary Phase-C progress invalidate
    # child inheritance.  Finalized Phase C is closed and must still match its
    # complete live report exactly.
    pinned_report_hash = (
        applied_snapshot.get("tune_report_sha256")
        if source == "applied_phase_a" and isinstance(applied_snapshot, dict)
        else _file_sha256(report_path)
    )
    if (
        not isinstance(pinned_report_hash, str)
        or not pinned_report_hash.startswith("sha256:")
        or len(pinned_report_hash) != len("sha256:") + 64
    ):
        raise ValueError(
            f"primary parent {parent_run_id} has an invalid applied report hash"
        )
    expected_snapshot = {
        "schema_version": 1,
        "source": source,
        "score": score,
        "params": params,
        "params_sha256": _json_sha256(params),
        "param_schema": _json_native(parent_schema),
        "param_schema_sha256": _json_sha256(parent_schema),
        "entrypoint_sha256": _file_sha256(parent_train),
        "tune_report_sha256": pinned_report_hash,
    }
    if applied_snapshot != expected_snapshot:
        raise ValueError(
            f"primary parent {parent_run_id} ledger applied-incumbent snapshot "
            "does not match its files and report"
        )

    return {
        "run_id": parent_run_id,
        "source": source,
        "params": params,
        "score": score,
        "param_schema": parent_schema,
        "train_path": parent_train,
        "train_sha256": _file_sha256(parent_train),
        "tune_report_path": report_path,
        "tune_report_sha256": pinned_report_hash,
        "ledger_path": ledger_path,
        "ledger_record_sha256": _json_sha256(ledger_record),
    }


def _receipt_without_self_hash(receipt: dict) -> dict:
    payload = copy.deepcopy(receipt)
    payload.pop("receipt_sha256", None)
    return payload


def _valid_self_hashed_receipt(receipt: dict) -> bool:
    if not isinstance(receipt, dict) or not isinstance(
        receipt.get("receipt_sha256"), str
    ):
        return False
    try:
        expected = _json_sha256(_receipt_without_self_hash(receipt))
    except (OverflowError, TypeError, ValueError):
        return False
    return receipt["receipt_sha256"] == expected


def build_parameter_transfer(
    candidate_path: Path,
    child_defaults: dict,
) -> dict:
    """Build, but do not persist, the exact primary-parent projection receipt."""
    candidate_path = Path(candidate_path).resolve()
    brief_path, brief, primary = _primary_parent_from_brief(candidate_path)
    child_schema = _read_param_schema(candidate_path)
    _validate_schema_values(
        child_defaults,
        child_schema,
        label="child pre-projection warm config 0",
    )
    incumbent = authoritative_parent_incumbent(
        primary["train_path"],
        primary["run_id"],
    )
    parent_params = incumbent["params"]
    parent_schema = incumbent["param_schema"]

    projected: dict = {}
    copied: list[dict] = []
    reset: list[dict] = []
    new: list[dict] = []
    dropped: list[dict] = []
    for key in child_schema:
        if key not in parent_schema:
            projected[key] = child_defaults[key]
            new.append(
                {"key": key, "value": child_defaults[key], "reason": "child_only"}
            )
            continue
        child_kind = _schema_kind(child_schema[key])
        parent_kind = _schema_kind(parent_schema[key])
        parent_value = parent_params[key]
        reason = None
        if child_kind != parent_kind:
            reason = "kind_changed"
        elif not _schema_accepts_value(child_schema[key], parent_value):
            reason = (
                "categorical_value_removed"
                if child_kind == "categorical"
                else "parent_value_incompatible"
            )
        if reason is None:
            projected[key] = parent_value
            copied.append({"key": key, "value": parent_value})
        else:
            projected[key] = child_defaults[key]
            reset.append(
                {
                    "key": key,
                    "parent_value": parent_value,
                    "child_value": child_defaults[key],
                    "reason": reason,
                }
            )
    for key in parent_schema:
        if key not in child_schema:
            dropped.append(
                {
                    "key": key,
                    "value": parent_params[key],
                    "reason": "parent_only",
                }
            )

    _validate_schema_values(
        projected,
        child_schema,
        label="projected inherited control",
    )
    receipt = {
        "schema_version": 2,
        "kind": "primary_parent_parameter_transfer",
        "candidate": {
            "run_id": brief.get("run_id"),
            "path": _display_path(candidate_path),
            "brief_path": _display_path(brief_path),
            "brief_sha256": _file_sha256(brief_path),
            "structure_sha256": _candidate_structure_sha256(candidate_path),
            "param_schema": _json_native(child_schema),
            "param_schema_sha256": _json_sha256(child_schema),
            "defaults": child_defaults,
            "defaults_sha256": _json_sha256(child_defaults),
        },
        "primary_parent": {
            "run_id": incumbent["run_id"],
            "path": _display_path(incumbent["train_path"]),
            "entrypoint_sha256": incumbent["train_sha256"],
            "tune_report_path": _display_path(incumbent["tune_report_path"]),
            "tune_report_sha256": incumbent["tune_report_sha256"],
            "ledger_path": _display_path(incumbent["ledger_path"]),
            "ledger_record_sha256": incumbent["ledger_record_sha256"],
            "incumbent_source": incumbent["source"],
            "incumbent_score": incumbent["score"],
            "incumbent_params": parent_params,
            "incumbent_params_sha256": _json_sha256(parent_params),
            "param_schema": _json_native(parent_schema),
            "param_schema_sha256": _json_sha256(parent_schema),
        },
        "projection": {
            "params": projected,
            "params_sha256": _json_sha256(projected),
            "copied": copied,
            "reset": reset,
            "new": new,
            "dropped": dropped,
        },
        # A parent-parameter projection controls inner-loop tuning quality, but
        # it does not isolate the authored semantic code change.  Until the
        # harness evaluates a same-child-code control/treatment pair, this
        # receipt must remain uncertainty-only evidence.
        "semantic_control": {
            "status": "unverified",
            "reason": "no_same_child_code_control_treatment_pair",
        },
    }
    receipt["receipt_sha256"] = _json_sha256(receipt)
    return receipt


def materialize_parameter_transfer(
    candidate_path: Path,
    configs_path: Path,
    receipt_path: Path | None = None,
) -> dict:
    """Write inherited control at index 0 plus its complete transfer receipt."""
    candidate_path = Path(candidate_path).resolve()
    configs_path = Path(configs_path).resolve()
    if receipt_path is None:
        receipt_path = candidate_path.parent / PARAMETER_TRANSFER_FILENAME
    receipt_path = Path(receipt_path).resolve()
    configs_tmp = configs_path.with_suffix(configs_path.suffix + ".tmp")
    receipt_tmp = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    all_paths = {
        candidate_path,
        configs_path,
        receipt_path,
        configs_tmp.resolve(),
        receipt_tmp.resolve(),
    }
    if len(all_paths) != 5:
        raise ValueError(
            "candidate, configs, receipt, and their temporary paths must all "
            "be distinct"
        )
    try:
        configs = json.loads(configs_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read warm configs {configs_path}: {exc}") from exc
    if not isinstance(configs, list) or not configs or not isinstance(configs[0], dict):
        raise ValueError("warm configs must be a non-empty list of objects")

    child_defaults = configs[0]
    current_brief_path, current_brief = _read_candidate_brief(candidate_path)
    current_run_id = current_brief.get("run_id")
    current_identity = {
        "run_id": current_run_id,
        "path": _display_path(candidate_path),
        "brief_path": _display_path(current_brief_path),
        "brief_sha256": _file_sha256(current_brief_path),
    }
    previous_path = receipt_path
    if not previous_path.exists() and receipt_tmp.exists():
        # Recover the safe half of an interrupted receipt-first materialization.
        previous_path = receipt_tmp
    if previous_path.exists():
        try:
            previous = json.loads(previous_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot trust existing parameter-transfer receipt "
                f"{previous_path}: {exc}"
            ) from exc
        if not _valid_self_hashed_receipt(previous):
            raise ValueError(
                f"existing parameter-transfer receipt {previous_path} has an "
                "invalid self hash"
            )
        previous_candidate = previous.get("candidate")
        if (
            previous.get("schema_version") != 2
            or previous.get("kind") != "primary_parent_parameter_transfer"
            or not isinstance(previous_candidate, dict)
            or not isinstance(previous_candidate.get("defaults"), dict)
        ):
            raise ValueError(
                "existing parameter-transfer receipt has an invalid contract "
                "or lacks original defaults"
            )
        previous_identity = {
            key: previous_candidate.get(key)
            for key in current_identity
        }
        if previous_identity != current_identity:
            raise ValueError(
                "existing parameter-transfer receipt belongs to a different "
                "candidate identity"
            )
        current_schema = _read_param_schema(candidate_path)
        current_schema_hash = _json_sha256(current_schema)
        if (
            previous_candidate.get("param_schema_sha256") == current_schema_hash
            and previous_candidate.get("param_schema") == _json_native(current_schema)
        ):
            # Rebuilding after a code fix must not mistake the already-projected
            # control for the child's original fallback defaults.
            child_defaults = previous_candidate["defaults"]
        else:
            previous_projection = previous.get("projection", {}).get("params")
            if child_defaults == previous_projection:
                raise ValueError(
                    "PARAM_SCHEMA changed but warm config 0 is still the prior "
                    "projection; materialize explicit child defaults for the new "
                    "schema before rebuilding inheritance"
                )
            _validate_schema_values(
                child_defaults,
                current_schema,
                label="child defaults after PARAM_SCHEMA change",
            )

    receipt = build_parameter_transfer(candidate_path, child_defaults)
    updated_configs = list(configs)
    updated_configs[0] = receipt["projection"]["params"]

    configs_tmp.write_text(json.dumps(updated_configs, indent=2) + "\n")
    receipt_tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    # Receipt first: it preserves the original child defaults. If the configs
    # replace is interrupted, warmstart fails closed on the mismatch and rerun
    # can deterministically finish from this receipt.
    receipt_tmp.replace(receipt_path)
    configs_tmp.replace(configs_path)
    return receipt


def validate_parameter_transfer(
    candidate_path: Path,
    configs: list,
    receipt: dict,
) -> dict:
    """Fail closed when parent, child structure, config 0, or receipt is stale."""
    if not _valid_self_hashed_receipt(receipt):
        raise ValueError("parameter-transfer receipt hash is invalid")
    if (
        not isinstance(configs, list)
        or not configs
        or not isinstance(configs[0], dict)
    ):
        raise ValueError("warm configs must contain object config 0")
    candidate = receipt.get("candidate")
    if not isinstance(candidate, dict) or not isinstance(
        candidate.get("defaults"), dict
    ):
        raise ValueError("parameter-transfer receipt lacks child defaults")
    expected = build_parameter_transfer(candidate_path, candidate["defaults"])
    if receipt != expected:
        raise ValueError(
            "parameter-transfer receipt is stale relative to parent or child"
        )
    projected = receipt.get("projection", {}).get("params")
    if configs[0] != projected:
        raise ValueError(
            "warm config 0 does not match the inherited-control receipt"
        )
    if receipt["projection"].get("params_sha256") != _json_sha256(configs[0]):
        raise ValueError("inherited-control params hash is invalid")
    return receipt


# ---------- tuning summary ----------


def summarize(report: dict) -> dict:
    """Reduce one candidate's tune_report.json to its stored tuning summary — a
    pure read over the report (which the search scripts stamp with per-stage
    elapsed_seconds/status)."""
    phase_a = report.get("phase_a", {})
    final = select_best(report)["best_score"]
    best_warm = phase_a.get("best_warm_score")
    if not _is_finite_score(best_warm):
        best_warm = None

    elapsed = 0.0
    if isinstance(phase_a.get("elapsed_seconds"), (int, float)):
        elapsed += phase_a["elapsed_seconds"]
    for stage in report.get("phase_c", {}).get("stages", []):
        if isinstance(stage.get("elapsed_seconds"), (int, float)):
            elapsed += stage["elapsed_seconds"]

    # Completed trials are finite observations; attempted trials additionally
    # include failed score_fn calls. The run-level budget uses attempted calls so
    # crashes cannot disappear from accounting. Injected warm priors are reused
    # and never appended to a Phase-C stage, so neither count double-counts them.
    warm_trials = phase_a.get("warm_start_configs", [])
    phase_a_attempted = phase_a.get("trials_attempted")
    if not isinstance(phase_a_attempted, int) or isinstance(phase_a_attempted, bool) \
            or phase_a_attempted < 0:
        phase_a_attempted = len(warm_trials)
    else:
        phase_a_attempted = max(phase_a_attempted, len(warm_trials))
    phase_c_trials = [
        trial
        for stage in report.get("phase_c", {}).get("stages", [])
        for trial in stage.get("trials", [])
    ]
    phase_c_attempted = sum(
        trial.get("status") != "preflight_rejected"
        for trial in phase_c_trials
    )
    preflight_attempts = report.get("preflight", {}).get("attempts", [])
    if not isinstance(preflight_attempts, list):
        preflight_attempts = []
    preflight_failures = sum(
        attempt.get("status") == "failed"
        for attempt in preflight_attempts
        if isinstance(attempt, dict)
    )
    feasibility_rejections = sum(
        trial.get("status") == "preflight_rejected"
        for trial in phase_c_trials
    )
    trials_completed = sum(
        1 for _ in _iter_trials(report, include_fidelity_controls=True)
    )
    trials_attempted = max(
        phase_a_attempted + phase_c_attempted,
        trials_completed,
    )
    return {
        "best_warm_score": best_warm,
        "final_best_score": final,
        "trials_completed": trials_completed,
        "trials_attempted": trials_attempted,
        "preflight_attempts": len(preflight_attempts),
        "preflight_failures": preflight_failures,
        "feasibility_rejections": feasibility_rejections,
        "phase_c_attempted": phase_c_attempted,
        "elapsed_seconds": round(elapsed, 1),
    }


def tuning_record(report: dict) -> dict:
    """Every ledger tuning field, derived from one tune_report.json. Imported by
    `ledger.py set-tuning --from-report` so the values flow report -> ledger by
    code with no LLM transcription."""
    summary = summarize(report)
    phase_a = report.get("phase_a", {})
    stages = report.get("phase_c", {}).get("stages", [])
    warm_configs = phase_a.get("warm_start_configs", [])
    search_space = phase_a.get("search_space") or {}
    transfer_receipt = phase_a.get("parameter_transfer")
    parameter_transfer = None
    if isinstance(transfer_receipt, dict):
        semantic_control = transfer_receipt.get("semantic_control")
        if (
            isinstance(semantic_control, dict)
            and semantic_control.get("status") == "paired"
        ):
            raise ValueError(
                "report-authored paired semantic controls are not admissible; "
                "no deterministic paired evaluator owns this evidence yet"
            )
        parameter_transfer = {
            "receipt": transfer_receipt,
            "inherited_control": phase_a.get("inherited_control"),
            "warm_start_observations": [
                row
                for row in warm_configs
                if isinstance(row, dict)
                and row.get("role")
                in {"inherited_control", "semantic_treatment"}
                and _is_finite_score(row.get("score"))
            ],
        }
    # Pre-close view only: until a candidate is finalized its scores still come
    # from Phase A, so a Phase-C observation that was never applied must not
    # claim a method or "tuned" depth here.  `finalized_tuning_record` recomputes
    # both fields at close: the method from the applied row's provenance, the
    # depth graded from cumulative Phase-C attempts.
    phase_c_method = next(
        (
            stage.get("method")
            for stage in stages
            if stage.get("status") == "ok"
            and any(
                isinstance(trial, dict)
                and _is_finite_score(trial.get("score"))
                for trial in stage.get("trials", [])
            )
        ),
        None,
    )
    return {
        # ledger tuning fields
        "best_warm_score": summary["best_warm_score"],
        "final_best_score": summary["final_best_score"],
        # Depth of the evaluation behind this record's scores: "tuned" when a
        # Phase-C observation backs them, "screening" otherwise (in this
        # pre-close view the same criterion as phase_c_method; the finalized
        # record switches to cumulative Phase-C attempts, graded at
        # tuner.tuned_threshold).
        # Only tuned children ground contradiction-grade semantic findings: a
        # screening rejection measures the hypothesis at one parameter point
        # and must not prune a space element (run 0730-ds-ex100-1: MTP rejected
        # twice at screening while 011, worse than its own control at
        # screening, won only after Phase C).
        "evaluation_depth": "tuned" if phase_c_method is not None else "screening",
        "n_dims": len(search_space) if search_space else None,
        "warm_start_K": len(warm_configs) if warm_configs else None,
        "warm_percentile": None,   # Phase B retired; gate moved to select-candidate
        "phase_b_decision": None,  # kept as ledger schema fields, always None now
        "phase_c_method": phase_c_method,
        "trials_completed": summary["trials_completed"],
        "trials_attempted": summary["trials_attempted"],
        "preflight_attempts": summary["preflight_attempts"],
        "preflight_failures": summary["preflight_failures"],
        "feasibility_rejections": summary["feasibility_rejections"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "applied": report.get("applied_to_base_params"),
        # Phase-A/pre-close view: no bout has closed yet, so the progressive
        # fields carry their pre-tuning values; `finalized_tuning_record`
        # recomputes both from the closed report.
        "tuning_bouts": 0,
        "last_bout_improved": None,
        # Additive, durable comparator evidence.  The full self-hashed receipt
        # stays beside the compact control pointer and its actually scored row;
        # downstream code need not trust a best-score coincidence.
        "parameter_transfer": parameter_transfer,
    }


# Phase-C attempts at which evaluation_depth becomes "tuned". Lives beside its
# consumer: the §15 selection constants further down load too late for the
# defaulted `finalized_tuning_record` parameter.
DEFAULT_TUNED_THRESHOLD = 16


def _evaluation_depth(phase_c_attempts: int, tuned_threshold: int) -> str:
    """Graded evaluation depth: 0 attempts screening, 1..threshold-1 lightly
    tuned, >=threshold fully tuned."""
    if phase_c_attempts <= 0:
        return "screening"
    return "tuned" if phase_c_attempts >= tuned_threshold else "tuned_lightly"


def _last_bout_improved(report: dict) -> bool | None:
    """Whether the last bout produced a trial strictly better than its
    pre-bout incumbent (warm best plus every earlier bout). None when the
    report has no Phase-C stages."""
    from _common import stages_by_bout

    stages = report.get("phase_c", {}).get("stages", [])
    bouts = stages_by_bout(stages)
    if not bouts:
        return None
    phase_a = report.get("phase_a", {})
    warm_best = phase_a.get("best_warm_score")
    prior_best = float(warm_best) if _is_finite_score(warm_best) else None
    for bout in bouts[:-1]:
        for trial in bout:
            for row in trial.get("trials", []):
                if isinstance(row, dict) and _is_finite_score(row.get("score")):
                    score = float(row["score"])
                    if prior_best is None or score < prior_best:
                        prior_best = score
    current_best = None
    for stage in bouts[-1]:
        for row in stage.get("trials", []):
            if isinstance(row, dict) and _is_finite_score(row.get("score")):
                score = float(row["score"])
                if current_best is None or score < current_best:
                    current_best = score
    if current_best is None:
        return False
    return prior_best is None or current_best < prior_best


def load_tuned_threshold(ledger_path: Path) -> int:
    """tuner.tuned_threshold for the run owning this ledger (default 16)."""
    return int(
        _run_cfg(Path(ledger_path), "tuner").get(
            "tuned_threshold", DEFAULT_TUNED_THRESHOLD
        )
    )


def finalized_tuning_record(
    report: dict,
    *,
    tuned_threshold: int = DEFAULT_TUNED_THRESHOLD,
) -> dict:
    """Ledger-ready tuning fields after the fail-closed completion check."""
    final = finalizable_tuning_result(report, require_applied=True)
    summary = summarize(report)
    stages = report.get("phase_c", {}).get("stages", [])
    from _common import stages_by_bout

    return {
        **tuning_record(report),
        "final_best_score": final["best_score"],
        # Provenance of the applied observation only: None when the Phase-A
        # incumbent wins the argmin, however much Phase C ran.
        "phase_c_method": final["phase_c_method"],
        # Depth is cumulative Phase-C evaluation EFFORT across bouts, not the
        # applied row's provenance: a candidate whose warm incumbent still
        # wins keeps the depth its admitted attempts earned. Graded:
        # screening -> tuned_lightly -> tuned at tuner.tuned_threshold.
        "evaluation_depth": _evaluation_depth(
            summary["phase_c_attempted"], tuned_threshold
        ),
        "tuning_bouts": len(stages_by_bout(stages)),
        "last_bout_improved": _last_bout_improved(report),
    }


# ---------- search-space induction (check + expand) ----------

# When a survived config sits outside the inducer's proposed range, extend past
# it by this fraction of the gap so the value lands *interior*, not on the new
# edge — the search needs room beyond known-good points. Single source of truth.
MARGIN_FRAC = 0.25


def _as_tuple(entry):
    """JSON delivers SEARCH_SPACE entries as lists; restore the tuple form
    (the categorical option list stays a list)."""
    return tuple(entry)


def _expand_space_entry(entry, values):
    """Widen one `(kind, ...)` entry to include all of `values` (the survived
    configs' values for this key), with margin for numerics or by union for
    categoricals. Returns (new_entry_tuple, reasons[]). Only expands where a
    value is outside; degenerate ranges get a min-width floor."""
    entry = _as_tuple(entry)
    kind = entry[0]
    reasons: list = []
    if kind == "categorical":
        opts = list(entry[1])
        for v in values:
            if not _categorical_contains(opts, v):
                opts.append(v)
                reasons.append(f"added option {v!r}")
        return ("categorical", opts), reasons

    original_lo, original_hi = float(entry[1]), float(entry[2])
    lo, hi = original_lo, original_hi
    tail = list(entry[3:])  # e.g. ["log"] for float
    nums = [float(v) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    observed_lo = min(nums) if nums else original_lo
    observed_hi = max(nums) if nums else original_hi

    if "log" in tail:
        # Apply margin in log space so positive distributions stay positive and
        # equal evidence yields the same box regardless of config order.
        log_lo, log_hi = math.log(original_lo), math.log(original_hi)
        if observed_lo < original_lo:
            observed_log_lo = math.log(observed_lo)
            lo = math.exp(
                observed_log_lo
                - MARGIN_FRAC * (log_hi - observed_log_lo)
            )
            reasons.append(f"widened low to include {observed_lo}")
        if observed_hi > original_hi:
            observed_log_hi = math.log(observed_hi)
            hi = math.exp(
                observed_log_hi
                + MARGIN_FRAC * (observed_log_hi - log_lo)
            )
            reasons.append(f"widened high to include {observed_hi}")
    else:
        if observed_lo < original_lo:
            lo = observed_lo - MARGIN_FRAC * (original_hi - observed_lo)
            reasons.append(f"widened low to include {observed_lo}")
        if observed_hi > original_hi:
            hi = observed_hi + MARGIN_FRAC * (observed_hi - original_lo)
            reasons.append(f"widened high to include {observed_hi}")

    if kind == "int":
        lo, hi = int(math.floor(lo)), int(math.ceil(hi))
        return ("int", lo, hi), reasons

    return ("float", lo, hi, *tail), reasons


def check_search_space(authoritative_schema: dict, proposed: dict, configs: list) -> dict:
    """Validate proposed SEARCH_SPACE and configs against the full PARAM_SCHEMA.

    Configs at this boundary are proposals, not observations.  They may widen a
    numeric range only after their exact keys, types, log positivity, and
    categorical membership are proven against the frozen schema.

    Returns
    {ok, finalized_space, expansions[], errors[]}. Hard errors (kind/key
    mismatch, bad tuple) leave finalized_space None and ok False; otherwise the
    finalized space is the proposed one widened to contain all configs."""
    errors: list = []
    if not isinstance(authoritative_schema, dict):
        errors.append({
            "code": "bad_schema",
            "key": None,
            "detail": "PARAM_SCHEMA must be an object",
        })
        authoritative_schema = {}
    if not isinstance(proposed, dict):
        errors.append({
            "code": "bad_space",
            "key": None,
            "detail": "proposed SEARCH_SPACE must be an object",
        })
        proposed = {}
    if not isinstance(configs, list) or not configs:
        errors.append({
            "code": "bad_configs",
            "key": None,
            "detail": "configs must be a non-empty list of objects",
        })
        configs = []

    akeys, pkeys = set(authoritative_schema), set(proposed)
    for key in sorted(akeys - pkeys):
        errors.append({"code": "missing_key", "key": key, "detail": "in schema but not proposed"})
    for key in sorted(pkeys - akeys):
        errors.append({"code": "extra_key", "key": key, "detail": "proposed but not in schema"})
    for key in sorted(akeys & pkeys):
        raw_entry = proposed[key]
        entry = (
            _as_tuple(raw_entry)
            if isinstance(raw_entry, (tuple, list))
            else raw_entry
        )
        if not valid_space_entry(entry):
            errors.append({"code": "bad_tuple", "key": key,
                           "detail": f"{_safe_repr(proposed[key])} is not a valid (kind, ...) entry"})
        elif not _valid_schema_entry(authoritative_schema[key]):
            errors.append({
                "code": "bad_schema_entry",
                "key": key,
                "detail": f"{_safe_repr(authoritative_schema[key])} is not a valid PARAM_SCHEMA entry",
            })
        else:
            mismatch = _space_schema_mismatch(authoritative_schema[key], entry)
            if mismatch:
                errors.append({
                    "code": "schema_mismatch",
                    "key": key,
                    "detail": mismatch,
                })

    for index, config in enumerate(configs):
        if not isinstance(config, dict):
            errors.append({
                "code": "bad_config",
                "key": None,
                "config_index": index,
                "detail": "config must be an object",
            })
            continue
        config_keys = set(config)
        if config_keys != akeys:
            errors.append({
                "code": "config_key_mismatch",
                "key": None,
                "config_index": index,
                "detail": (
                    f"config keys {sorted(config_keys)} != schema keys "
                    f"{sorted(akeys)}"
                ),
            })
            continue
        for key in sorted(akeys):
            schema_entry = authoritative_schema[key]
            if (
                _valid_schema_entry(schema_entry)
                and not _schema_accepts_value(schema_entry, config[key])
            ):
                errors.append({
                    "code": "config_value_invalid",
                    "key": key,
                    "config_index": index,
                    "detail": (
                        f"value {_safe_repr(config[key])} is incompatible with "
                        f"PARAM_SCHEMA {_safe_repr(schema_entry)}"
                    ),
                })
    if errors:
        return {"ok": False, "finalized_space": None, "expansions": [], "errors": errors}

    finalized, expansions = {}, []
    for key in authoritative_schema:  # preserve schema key order
        values = [config[key] for config in configs]
        schema_entry = authoritative_schema[key]
        if _schema_kind(schema_entry) == "categorical":
            observed = values
            values = [
                option
                for option in schema_entry[1]
                if _categorical_contains(observed, option)
            ]
        try:
            new_entry, reasons = _expand_space_entry(proposed[key], values)
        except (OverflowError, TypeError, ValueError) as exc:
            errors.append({
                "code": "expansion_error",
                "key": key,
                "detail": f"cannot safely expand numeric range: {exc}",
            })
            continue
        if not valid_space_entry(new_entry):
            errors.append({
                "code": "expansion_error",
                "key": key,
                "detail": f"expanded entry {_safe_repr(new_entry)} is invalid or non-finite",
            })
            continue
        finalized[key] = list(new_entry)  # JSON-friendly (lists, not tuples)
        if reasons:
            expansions.append({"key": key, "reasons": reasons})
    if errors:
        return {
            "ok": False,
            "finalized_space": None,
            "expansions": [],
            "errors": errors,
        }
    return {"ok": True, "finalized_space": finalized, "expansions": expansions, "errors": []}


# ---------- lineage evidence (parent candidates -> hyperparam->performance) ----------

# How many whole configs to surface per parent: the best TOP_K by score plus
# DIVERSE_K farthest-point picks. Whole configs (not per-key marginals) so the
# joint structure / hyperparameter interactions survive; the diverse picks vary
# the combinations so the LLM can read interactions, not just the winning point.
TOP_K = 3
DIVERSE_K = 4


def _config_distance(a: dict, b: dict, ranges: dict) -> float:
    """Normalized distance between two param dicts over shared keys. Numeric:
    |a-b| / explored-span; non-numeric (categorical): 0 if equal else 1."""
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        va, vb = a[key], b[key]
        numeric = (isinstance(va, (int, float)) and not isinstance(va, bool)
                   and isinstance(vb, (int, float)) and not isinstance(vb, bool))
        if numeric:
            lo, hi = ranges.get(key, (0.0, 1.0))
            total += abs(va - vb) / ((hi - lo) or 1.0)
        else:
            total += 0.0 if va == vb else 1.0
    return total / len(keys)


def _select_trials(trials: list, ranges: dict) -> list:
    """From a parent's trials (`[{params, score}]`), keep TOP_K by score (deduped
    by params) + DIVERSE_K farthest-point configs. Deterministic; keeps all when
    there are few. Whole configs preserve hyperparameter interactions."""
    deduped, seen = [], set()
    for trial in sorted(trials, key=lambda t: t["score"]):  # best (lowest) first
        key = json.dumps(trial["params"], sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            deduped.append(trial)
    if len(deduped) <= TOP_K + DIVERSE_K:
        return deduped
    selected, pool = deduped[:TOP_K], deduped[TOP_K:]
    for _ in range(DIVERSE_K):
        if not pool:
            break
        # the pool config farthest from everything already selected
        far_i = max(range(len(pool)),
                    key=lambda i: min(_config_distance(pool[i]["params"], s["params"], ranges)
                                      for s in selected))
        selected.append(pool.pop(far_i))
    return selected


def lineage_evidence(run_dir: Path, source_run_ids: list) -> dict:
    """Mine the parent candidates for hyperparameter->performance evidence,
    grouped **per parent** (so scores stay on one comparable scale and parent
    identity survives for crossover). For each parent: its `idea`, best config +
    score, the space it searched, per-key explored ranges, and a small set of
    whole `trials` (TOP_K best + DIVERSE_K farthest). Scores are the tuning-oracle
    scores from the tune_report (lower-is-better). Pure data plumbing."""
    run_dir = Path(run_dir)
    ledger = {}
    ledger_path = run_dir / "ledger.json"
    if ledger_path.exists():
        records = json.loads(ledger_path.read_text()).get("records", [])
        ledger = {r.get("run_id"): r for r in records}

    per_parent: dict = {}
    for pid in source_run_ids:
        rec = ledger.get(pid, {})
        entry = {"idea": rec.get("idea"), "best_params": None, "best_score": None,
                 "search_space": None, "explored": {}, "trials": []}
        report_path = run_dir / "candidates" / pid / "tune_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            incumbent = authoritative_parent_incumbent(
                run_dir / "candidates" / pid / "train.py",
                pid,
            )
            entry["best_params"] = incumbent["params"]
            entry["best_score"] = incumbent["score"]
            entry["incumbent_source"] = incumbent["source"]
            entry["search_space"] = report.get("phase_a", {}).get("search_space")
            trials = [{"params": params, "score": score}
                      for _, params, score in _iter_trials(report)
                      if isinstance(params, dict)]
            ranges: dict = {}
            for trial in trials:
                for key, value in trial["params"].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        lo, hi = ranges.get(key, (value, value))
                        ranges[key] = (min(lo, value), max(hi, value))
            entry["explored"] = {key: [lo, hi] for key, (lo, hi) in ranges.items()}
            entry["trials"] = _select_trials(trials, ranges)
        per_parent[pid] = entry
    return {"per_parent": per_parent}


# ---------- candidate selection (decoupled tuning, design §15) ----------

# step-2 is decoupled from idea proposal: every idea stops at step 0+1, and the
# tuner picks ONE candidate from the whole population to deep-tune per round.
# Selection is greedy on `best_warm_score` (the step-1 selectable screening
# score; inherited fidelity controls are excluded) behind a promotion gate.
# NO headroom/spread term: warm-start
# configs are referenced from heterogeneous historical tasks, so their spread
# reflects the reference quality, not the landscape — it is not comparable across
# candidates' different methods. Pure read over the ledger; lower score is better.
DEFAULT_BOUT_TRIALS = 8       # one progressive tuning bout's objective-attempt budget
DEFAULT_REWARM_PROPOSALS = 3  # max LLM-proposed configs a continuation bout may start from
DEFAULT_N_MIN = 5             # P=80's smallest non-empty top-tier population
DEFAULT_TOP_PERCENTILE = 80   # eligible iff the best untuned candidate is in the top (100-P)%


def _has_unresolved_primary_descendant(ledger: dict, parent_run_id: str) -> bool:
    """Whether tuning this parent would race or invalidate a primary child.

    Delegates to the shared lineage predicate so this eligibility gate and
    ``ledger._preserve_descendant_bindings`` cannot disagree about which
    children block a parent mutation.
    """
    return bool(unbound_primary_descendants(ledger, parent_run_id))


def select_candidate(
    ledger: dict,
    *,
    n_min: int = DEFAULT_N_MIN,
    top_percentile: float = DEFAULT_TOP_PERCENTILE,
    budget_allocation: dict | None = None,
) -> dict:
    """Which candidate to deep-tune next (§15.4), or none. Eligible iff the
    population (non-crash, has best_warm_score) is >= n_min AND the best untuned
    candidate has no unresolved primary descendants and its best_warm_score ranks
    in the top (100-top_percentile)% across all candidates. Greedy on
    best_warm_score. Returns {run_id|None, reason, ...}."""
    if (
        not isinstance(n_min, int)
        or isinstance(n_min, bool)
        or n_min <= 0
    ):
        raise ValueError("n_min must be a positive integer")
    if (
        not isinstance(top_percentile, (int, float))
        or isinstance(top_percentile, bool)
        or not math.isfinite(float(top_percentile))
        or not 0 <= float(top_percentile) < 100
    ):
        raise ValueError("top_percentile must be finite and in [0, 100)")

    cands = [r for r in ledger.get("records", [])
             if r.get("status") != "crash" and _is_finite_score(r.get("best_warm_score"))]
    n = len(cands)
    allocation_receipt = None
    per_candidate_deep: dict[str, int] = {}
    per_candidate_cap = None
    if budget_allocation is not None:
        global_remaining = budget_allocation.get("remaining")
        deep = budget_allocation.get("deep_tune", {})
        deep_remaining = deep.get("remaining") if isinstance(deep, dict) else None
        per_candidate_cap = (
            deep.get("per_candidate_cap") if isinstance(deep, dict) else None
        )
        if isinstance(deep, dict):
            per_candidate_deep = {
                str(row.get("run_id")): int(row.get("evals", 0))
                for row in deep.get("per_candidate", [])
                if isinstance(row, dict)
                and isinstance(row.get("run_id"), str)
                and isinstance(row.get("evals"), int)
            }
        allocation_receipt = {
            "global_remaining": global_remaining,
            "deep_tune_remaining": deep_remaining,
            "deep_tune_total_cap": (
                deep.get("total_cap") if isinstance(deep, dict) else None
            ),
            "deep_tune_per_candidate_cap": per_candidate_cap,
            "deep_tune_time_limit_seconds": (
                deep.get("time_limit_seconds") if isinstance(deep, dict) else None
            ),
        }
        if isinstance(global_remaining, int) and global_remaining <= 0:
            return {
                "run_id": None,
                "n_candidates": n,
                "reason": "evaluation_budget_reached",
                "budget_allocation": allocation_receipt,
            }
        if isinstance(deep_remaining, int) and deep_remaining <= 0:
            return {
                "run_id": None,
                "n_candidates": n,
                "reason": "deep_tune_budget_exhausted",
                "budget_allocation": allocation_receipt,
            }

    if n < n_min:
        return {"run_id": None, "n_candidates": n,
                "reason": f"population {n} < n_min {n_min} (breadth first)"}
    all_raw_untuned = [record for record in cands if not record.get("tune")]
    raw_untuned = [
        record
        for record in all_raw_untuned
        if (
            not isinstance(per_candidate_cap, int)
            or per_candidate_deep.get(str(record.get("run_id")), 0)
            < per_candidate_cap
        )
    ]
    untuned = [
        record
        for record in raw_untuned
        if not _has_unresolved_primary_descendant(
            ledger, str(record.get("run_id"))
        )
    ]
    if not untuned:
        if all_raw_untuned and not raw_untuned:
            return {
                "run_id": None,
                "n_candidates": n,
                "reason": "all untuned candidates reached deep-tune per-candidate cap",
                **(
                    {"budget_allocation": allocation_receipt}
                    if allocation_receipt is not None
                    else {}
                ),
            }
        if raw_untuned:
            return {
                "run_id": None,
                "n_candidates": n,
                "reason": (
                    "all untuned candidates have unresolved primary descendants"
                ),
            }
        return {"run_id": None, "n_candidates": n, "reason": "all candidates already tuned"}
    best = min(untuned, key=lambda r: r["best_warm_score"])
    value = best["best_warm_score"]
    # percentile = fraction of OTHER candidates strictly worse (higher score);
    # high percentile = among the best (matches ledger.py percentile semantics).
    worse = sum(1 for r in cands if r is not best and r["best_warm_score"] > value)
    pct = 100.0 * worse / (n - 1) if n > 1 else 100.0
    common = {"best_warm_score": value, "percentile": round(pct), "n_candidates": n}
    if allocation_receipt is not None:
        candidate_used = per_candidate_deep.get(str(best.get("run_id")), 0)
        caps = [
            value
            for value in (
                allocation_receipt["global_remaining"],
                allocation_receipt["deep_tune_remaining"],
                (
                    per_candidate_cap - candidate_used
                    if isinstance(per_candidate_cap, int)
                    else None
                ),
            )
            if isinstance(value, int)
        ]
        allocation_receipt = {
            **allocation_receipt,
            "candidate_attempts": candidate_used,
            "trial_cap": min(caps) if caps else None,
        }
        common["budget_allocation"] = allocation_receipt
    if pct >= top_percentile:
        return {"run_id": best.get("run_id"),
                "reason": f"best untuned in top {100 - top_percentile:.0f}% (percentile {pct:.0f} >= {top_percentile:.0f})",
                **common}
    return {"run_id": None,
            "reason": f"best untuned percentile {pct:.0f} < {top_percentile:.0f} (top {100 - top_percentile:.0f}% already tuned)",
            **common}


# ---------- CLI ----------


def cmd_lint_contract(args) -> int:
    result = lint_contract(args.candidate_path)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_select_method(args) -> int:
    print(json.dumps(select_method(len(_read_search_space(args.candidate_path)))))
    return 0


def cmd_phase_c_action(args) -> int:
    report = json.loads(Path(args.tune_report_json).read_text())
    try:
        result = phase_c_action(report, args.candidate_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result))
    return 0


def cmd_close_exhausted_stage(args) -> int:
    try:
        result = close_exhausted_stage(args.candidate_path, args.tune_report_json)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result))
    return 0


def cmd_select_best(args) -> int:
    report = json.loads(Path(args.tune_report_json).read_text())
    try:
        result = finalizable_tuning_result(report)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result))
    return 0


def cmd_validate_params(args) -> int:
    search_space = _read_search_space(args.candidate_path)
    params = json.loads(Path(args.params_json).read_text())
    violations = _bounds_violations(params, search_space)
    print(json.dumps({"ok": not violations, "violations": violations}))
    return 0 if not violations else 1


def cmd_summarize(args) -> int:
    report = json.loads(Path(args.tune_report_json).read_text())
    print(json.dumps(summarize(report)))
    return 0


def cmd_render_failure(args) -> int:
    line_range = None
    if args.view == "lines":
        if args.line_range is None:
            raise SystemExit("--line-range START:END is required for --view lines")
        try:
            start, end = (int(value) for value in args.line_range.split(":", 1))
        except (TypeError, ValueError):
            raise SystemExit("--line-range must be START:END") from None
        line_range = (start, end)
    elif args.line_range is not None:
        raise SystemExit("--line-range is only valid with --view lines")

    result = render_failure(
        args.tune_report_json,
        args.failure_id,
        view=args.view,
        line_range=line_range,
    )
    if isinstance(result, dict):
        print(json.dumps(result, separators=(",", ":"), ensure_ascii=False))
    else:
        sys.stdout.write(result)
    return 0


def cmd_lint_schema(args) -> int:
    result = lint_schema(args.candidate_path)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_check_search_space(args) -> int:
    schema = _read_param_schema(args.candidate_path)
    proposed = json.loads(Path(args.space_json).read_text())
    configs = json.loads(Path(args.configs_json).read_text())
    result = check_search_space(schema, proposed, configs)
    if result["ok"]:
        # Persist the finalized (expanded) space back to the SAME artifact so the
        # caller never hand-extracts finalized_space from stdout — writing the whole
        # verdict dict there would corrupt the contract apply_search_space reads.
        # On error, leave the proposed file untouched for the agent's self-fix loop;
        # stdout carries the verdict (ok / expansions / errors) either way.
        space_path = Path(args.space_json)
        tmp_path = space_path.with_suffix(space_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(result["finalized_space"], indent=2) + "\n")
        tmp_path.replace(space_path)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_lineage_evidence(args) -> int:
    ids = [x.strip() for x in args.source_run_ids.split(",") if x.strip()]
    print(json.dumps(lineage_evidence(args.run_dir, ids), indent=2))
    return 0


def cmd_build_inheritance(args) -> int:
    receipt_path = (
        args.receipt_json
        if args.receipt_json is not None
        else args.candidate_path.parent / PARAMETER_TRANSFER_FILENAME
    )
    try:
        result = materialize_parameter_transfer(
            args.candidate_path,
            args.configs_json,
            receipt_path,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, indent=2))
    return 0


def _run_cfg(ledger_path: Path, section: str) -> dict:
    """Per-run framework overrides from `<run_dir>/framework_cfg.json` (stdlib only,
    so select-candidate keeps needing no uv env). Shape `{"tuner": {...}, "got": {...}}`.
    A cfg file that exists but cannot be parsed raises RunConfigError instead of
    silently reverting to module defaults."""
    p = Path(ledger_path).parent / "framework_cfg.json"
    if p.is_file():
        return dict(read_framework_cfg(p).get(section, {}))
    return {}


def cmd_select_candidate(args) -> int:
    led = Path(args.ledger)
    ledger = json.loads(led.read_text())
    rc = _run_cfg(led, "tuner")  # explicit flag wins; else framework_cfg.json; else module default
    top_p = args.top_percentile if args.top_percentile is not None else float(rc.get("top_percentile", DEFAULT_TOP_PERCENTILE))
    # n_min DERIVES from top_percentile: the smallest population for which the
    # top-(100-P)% gate can contain >=1 candidate, i.e. ceil(100/(100-P))
    # (P=80→5, 70→4, 90→10). Explicit --n-min / framework_cfg.tuner.n_min overrides.
    if args.n_min is not None:
        n_min = args.n_min
    elif rc.get("n_min") is not None:
        n_min = int(rc["n_min"])
    else:
        n_min = math.ceil(100.0 / (100.0 - top_p)) if top_p < 100 else DEFAULT_N_MIN
    try:
        result = select_candidate(
            ledger,
            n_min=n_min,
            top_percentile=top_p,
            budget_allocation=budget_status(led.parent),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    lc = sub.add_parser("lint-contract", help="Check a candidate's tuner-contract relational invariants by AST.")
    lc.add_argument("--candidate-path", required=True, type=Path)
    lc.set_defaults(func=cmd_lint_contract)

    ls = sub.add_parser("lint-schema",
                        help="Schema-mode lint: make_model + PARAM_SCHEMA (step 0, before SEARCH_SPACE exists).")
    ls.add_argument("--candidate-path", required=True, type=Path)
    ls.set_defaults(func=cmd_lint_schema)

    sm = sub.add_parser("select-method", help="Pick the Phase C method from a candidate's SEARCH_SPACE.")
    sm.add_argument("--candidate-path", required=True, type=Path)
    sm.set_defaults(func=cmd_select_method)

    pa = sub.add_parser(
        "phase-c-action",
        help=(
            "Validate candidate/report state and return the deterministic "
            "Phase-C resume, finalize, or stop action."
        ),
    )
    pa.add_argument("--candidate-path", required=True, type=Path)
    pa.add_argument("--tune-report-json", required=True, type=Path)
    pa.set_defaults(func=cmd_phase_c_action)

    ce = sub.add_parser(
        "close-exhausted-stage",
        help=(
            "Close a running Phase-C stage the evaluation budget can never "
            "resume, so its durable trials stay finalizable."
        ),
    )
    ce.add_argument("--candidate-path", required=True, type=Path)
    ce.add_argument("--tune-report-json", required=True, type=Path)
    ce.set_defaults(func=cmd_close_exhausted_stage)

    sb = sub.add_parser(
        "select-best",
        help="Finalizable global best; partial/interrupted Phase C is rejected.",
    )
    sb.add_argument("--tune-report-json", required=True, type=Path)
    sb.set_defaults(func=cmd_select_best)

    vp = sub.add_parser("validate-params", help="Check a params dict's keys/bounds vs SEARCH_SPACE.")
    vp.add_argument("--candidate-path", required=True, type=Path)
    vp.add_argument("--params-json", required=True, type=Path)
    vp.set_defaults(func=cmd_validate_params)

    sz = sub.add_parser("summarize", help="Reduce one tune_report.json to its tuning summary.")
    sz.add_argument("--tune-report-json", required=True, type=Path)
    sz.set_defaults(func=cmd_summarize)

    rf = sub.add_parser("render-failure", help="Read a frozen failure receipt or retrieve its traceback.")
    rf.add_argument("--tune-report-json", required=True, type=Path)
    rf.add_argument("--failure-id", required=True)
    rf.add_argument("--view", choices=("receipt", "full", "lines"), default="receipt")
    rf.add_argument("--line-range", help="1-based inclusive START:END; only for --view lines")
    rf.set_defaults(func=cmd_render_failure)

    cs = sub.add_parser("check-search-space",
                        help="Validate + expand a proposed SEARCH_SPACE against the full schema and proposed warm configs; on ok, overwrite --space-json in place with the finalized space.")
    cs.add_argument("--candidate-path", required=True, type=Path,
                    help="train.py whose PARAM_SCHEMA gives the authoritative kinds")
    cs.add_argument("--space-json", required=True, type=Path,
                    help="JSON of the proposed SEARCH_SPACE {key: [kind, ...]}; overwritten in place with the finalized (expanded) space on ok")
    cs.add_argument("--configs-json", required=True, type=Path,
                    help="JSON list of proposed warm-config param dicts")
    cs.set_defaults(func=cmd_check_search_space)

    le = sub.add_parser("lineage-evidence",
                        help="Assemble per-hyperparam (value, score) evidence from parent candidates.")
    le.add_argument("--run-dir", required=True, type=Path)
    le.add_argument("--source-run-ids", required=True,
                    help="comma-separated parent run_ids (e.g. 003,005)")
    le.set_defaults(func=cmd_lineage_evidence)

    bi = sub.add_parser(
        "build-inheritance",
        help=(
            "Project the applied primary-parent incumbent into warm config 0 "
            "and persist its transfer receipt."
        ),
    )
    bi.add_argument(
        "--candidate-path",
        required=True,
        type=Path,
        help="schema-4 non-fresh child train.py",
    )
    bi.add_argument(
        "--configs-json",
        required=True,
        type=Path,
        help="_warm_configs.json; index 0 is replaced with the inherited control",
    )
    bi.add_argument(
        "--receipt-json",
        type=Path,
        default=None,
        help=(
            "output receipt path (default: sibling "
            f"{PARAMETER_TRANSFER_FILENAME})"
        ),
    )
    bi.set_defaults(func=cmd_build_inheritance)

    sc = sub.add_parser("select-candidate",
                        help="Pick which candidate to deep-tune next (decoupled tuning, design §15), or none.")
    sc.add_argument("--ledger", required=True, type=Path, help="Path to ledger.json")
    sc.add_argument("--n-min", type=int, default=None,
                    help=f"min population before any tuning (default {DEFAULT_N_MIN}; "
                         f"else framework_cfg.json tuner.n_min)")
    sc.add_argument("--top-percentile", type=float, default=None,
                    help=f"eligibility gate: best untuned in top (100-P)%% (default {DEFAULT_TOP_PERCENTILE}; "
                         f"else framework_cfg.json tuner.top_percentile)")
    sc.set_defaults(func=cmd_select_candidate)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
