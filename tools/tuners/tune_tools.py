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
- select-best     : tune_report.json -> global best (minimum)
                    {best_params, best_score, source} over base+warm+phase_c
- validate-params : a params dict's keys/bounds vs the candidate SEARCH_SPACE
                    -> {ok, violations}; exit 1 on any violation
- summarize       : tune_report.json -> stored tuning summary {best_warm_score,
                    final_best_score, trials_completed, trials_attempted,
                    elapsed_seconds}
- check-search-space : proposed SEARCH_SPACE + survived configs -> {ok,
                    finalized_space, expansions, errors}; validates kinds vs the
                    schema and widens ranges to bracket every config. On ok, the
                    proposed --space-json is overwritten in place with finalized_space
                    (apply_search_space.py then writes it into train.py); exit 1 on a
                    hard error.
- lineage-evidence : run_dir + parent run_ids -> {per_parent} — per parent its
                    idea, best config+score, searched space, explored ranges, and
                    a few whole trials (top-by-score + farthest-point diverse, so
                    hyperparameter interactions survive). Feeds proposer/inducer.
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
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # tools/ for run_cfg

from failure_artifacts import render_failure
from run_cfg import read_framework_cfg  # noqa: E402


# ---------- SEARCH_SPACE via AST (no candidate import) ----------


def _read_search_space(train_path: Path) -> dict:
    """SEARCH_SPACE dict via AST literal_eval — no import, no uv env. The tuner
    contract requires a module-level dict literal; anything else errors here
    (lint-contract's job to explain why)."""
    tree = ast.parse(Path(train_path).read_text(errors="replace"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "SEARCH_SPACE":
                if not isinstance(node.value, ast.Dict):
                    raise SystemExit("SEARCH_SPACE is not a dict literal (tuner contract violation)")
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    raise SystemExit("SEARCH_SPACE is not a pure literal (tuner contract violation)")
    raise SystemExit("no module-level SEARCH_SPACE found in the candidate")


def _read_param_schema(train_path: Path) -> dict:
    """PARAM_SCHEMA dict via AST literal_eval — the schema the extractor writes
    before SEARCH_SPACE exists. Each value is a kind declaration:
      "int" | "float" | ("float", "log") | ("categorical", [opt, ...])."""
    tree = ast.parse(Path(train_path).read_text(errors="replace"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "PARAM_SCHEMA":
                if not isinstance(node.value, ast.Dict):
                    raise SystemExit("PARAM_SCHEMA is not a dict literal (schema contract violation)")
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    raise SystemExit("PARAM_SCHEMA is not a pure literal (schema contract violation)")
    raise SystemExit("no module-level PARAM_SCHEMA found in the candidate")


def _schema_kind(entry) -> str:
    """The kind string of a PARAM_SCHEMA entry ("int" / "float" / "categorical")."""
    return entry if isinstance(entry, str) else entry[0]


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
            return len(entry) == 2 and isinstance(entry[1], (list, tuple)) and len(entry[1]) >= 1
    return False


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
    kind = entry[0]
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool) \
            and float(entry[1]) <= float(value) <= float(entry[2])
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
            return False
        return int(entry[1]) <= int(value) <= int(entry[2])
    if kind == "categorical":
        return value in list(entry[1])
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
            violations.append({"key": key, "value": params[key],
                               "reason": f"outside {list(search_space[key])}"})
    return violations


# ---------- contract lint ----------

# The tuner contract's *relational* invariants — the ones that silently poison
# downstream tuning if an LLM writes the contract slightly wrong. `_common` only
# checks the three symbols exist (hasattr); this checks they agree. Pure AST so
# it runs as a gate before any uv env is touched. Errors carry the offending key
# and its source line so the fix is mechanical.

CONTRACT_SYMBOLS = ("BASE_PARAMS", "SEARCH_SPACE", "make_model")


def _module_symbols(tree: ast.Module) -> dict:
    """Module-level name -> defining AST node. For BASE_PARAMS/SEARCH_SPACE the
    node is the assigned value; for make_model it is the FunctionDef itself (so a
    non-function `make_model = ...` is detectable). First binding wins."""
    out: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out.setdefault(node.target.id, node.value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.setdefault(node.name, node)
    return out


def _dict_literal(node: ast.AST) -> tuple[dict, dict]:
    """(evaluated dict, {key: source line}) for a module-level dict literal.
    Raises ValueError if it is not a pure string-keyed dict literal."""
    if not isinstance(node, ast.Dict):
        raise ValueError("is not a dict literal")
    key_lines = {}
    for key_node in node.keys:
        if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
            raise ValueError("has a non-string key")
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
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (lo, hi)) and lo <= hi
    if kind == "int":
        if len(entry) != 3:
            return False
        lo, hi = entry[1], entry[2]
        return all(isinstance(x, int) and not isinstance(x, bool) for x in (lo, hi)) and lo <= hi
    if kind == "categorical":
        return len(entry) == 2 and isinstance(entry[1], (list, tuple)) and len(entry[1]) >= 1
    return False


def lint_contract(train_path: Path) -> dict:
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

    syms = _module_symbols(tree)
    errors: list = []

    def _read(name):
        if name not in syms:
            errors.append({"code": "missing_symbol", "detail": f"no module-level {name}", "line": 0})
            return None, {}
        try:
            return _dict_literal(syms[name])
        except ValueError as exc:
            errors.append({"code": "not_dict_literal", "detail": f"{name} {exc}",
                           "line": getattr(syms[name], "lineno", 0)})
            return None, {}

    search_space, space_lines = _read("SEARCH_SPACE")
    base_params, base_lines = _read("BASE_PARAMS")

    mm = syms.get("make_model")
    make_model_defined = isinstance(mm, (ast.FunctionDef, ast.AsyncFunctionDef))
    if mm is None:
        errors.append({"code": "missing_symbol", "detail": "no module-level make_model", "line": 0})
    elif not make_model_defined:
        errors.append({"code": "make_model_not_func",
                       "detail": "make_model is not a def (must be a function)",
                       "line": getattr(mm, "lineno", 0)})

    if isinstance(search_space, dict) and isinstance(base_params, dict):
        space_keys, base_keys = set(search_space), set(base_params)
        for key in sorted(space_keys - base_keys):
            errors.append({"code": "key_mismatch",
                           "detail": f"{key} in SEARCH_SPACE but not BASE_PARAMS",
                           "line": space_lines.get(key, 0)})
        for key in sorted(base_keys - space_keys):
            errors.append({"code": "key_mismatch",
                           "detail": f"{key} in BASE_PARAMS but not SEARCH_SPACE",
                           "line": base_lines.get(key, 0)})
        for key in sorted(space_keys):
            if not valid_space_entry(search_space[key]):
                errors.append({"code": "bad_tuple",
                               "detail": f"SEARCH_SPACE['{key}']={search_space[key]!r} is not a valid (kind, ...) entry",
                               "line": space_lines.get(key, 0)})
        for key in sorted(space_keys & base_keys):
            entry = search_space[key]
            if valid_space_entry(entry) and not _value_in_bounds(base_params[key], entry):
                errors.append({"code": "out_of_bounds",
                               "detail": f"BASE_PARAMS['{key}']={base_params[key]!r} outside {list(entry)}",
                               "line": base_lines.get(key, 0)})

    if isinstance(base_params, dict):
        for key in sorted(base_params):
            if isinstance(base_params[key], tuple):
                errors.append({"code": "base_param_tuple",
                               "detail": f"BASE_PARAMS['{key}'] is a tuple; must be a concrete value",
                               "line": base_lines.get(key, 0)})

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
    syms = _module_symbols(tree)
    errors: list = []

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
                               "detail": f"PARAM_SCHEMA['{key}']={schema[key]!r} is not a valid kind/options",
                               "line": schema_lines.get(key, 0)})

    mm = syms.get("make_model")
    make_model_defined = isinstance(mm, (ast.FunctionDef, ast.AsyncFunctionDef))
    if mm is None:
        errors.append({"code": "missing_symbol", "detail": "no module-level make_model", "line": 0})
    elif not make_model_defined:
        errors.append({"code": "make_model_not_func",
                       "detail": "make_model is not a def (must be a function)",
                       "line": getattr(mm, "lineno", 0)})
    make_model_called = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "make_model"
        for n in ast.walk(tree)
    )
    return {
        "ok": not errors,
        "keys": sorted(schema) if isinstance(schema, dict) else [],
        "kinds": {k: _schema_kind(v) for k, v in schema.items()} if isinstance(schema, dict) else {},
        "make_model_defined": make_model_defined,
        "make_model_called": make_model_called,
        "errors": errors,
    }


# ---------- best-trial selection ----------


def _iter_trials(report: dict):
    """Yield (source, params, score) over base + warm-start + every phase_c
    stage trial. Mirrors _common.read_prior_trials but numpy-free so this stays
    env-free."""
    phase_a = report.get("phase_a", {})
    for warm in phase_a.get("warm_start_configs", []):
        if _is_finite_score(warm.get("score")):
            yield ("warm_start", warm["params"], float(warm["score"]))
    for stage in report.get("phase_c", {}).get("stages", []):
        method = stage.get("method", "phase_c")
        for trial in stage.get("trials", []):
            if _is_finite_score(trial.get("score")):
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
    trials_completed = sum(1 for _ in _iter_trials(report))
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
    # The method whose stage actually ran (status ok); None when Phase B stopped.
    phase_c_method = next((s.get("method") for s in stages if s.get("status") == "ok"), None)
    return {
        # ledger tuning fields
        "best_warm_score": summary["best_warm_score"],
        "final_best_score": summary["final_best_score"],
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
            if v not in opts:
                opts.append(v)
                reasons.append(f"added option {v!r}")
        return ("categorical", opts), reasons

    lo, hi = float(entry[1]), float(entry[2])
    tail = list(entry[3:])  # e.g. ["log"] for float
    nums = [float(v) for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for v in nums:
        if v < lo:
            lo = v - MARGIN_FRAC * (hi - v)
            reasons.append(f"widened low to include {v}")
        elif v > hi:
            hi = v + MARGIN_FRAC * (v - lo)
            reasons.append(f"widened high to include {v}")

    if kind == "int":
        lo, hi = int(math.floor(lo)), int(math.ceil(hi))
        if hi <= lo:
            hi = lo + 1
            reasons.append("min-width floor (int)")
        return ("int", lo, hi), reasons

    if "log" in tail and lo <= 0:
        # Log-scale distributions require a strictly positive lower bound. Widening
        # (or a non-positive survived value, e.g. reg_alpha=0.0) must never push it
        # to <= 0, or Optuna's log FloatDistribution rejects the space and BO crashes.
        positives = [x for x in ([float(entry[1])] + nums) if x > 0]
        lo = min(positives) if positives else (hi * 1e-3 if hi > 0 else 1e-6)
        reasons.append("clamped log lower bound to > 0")
    if hi <= lo:  # float degenerate range
        pad = abs(lo) * MARGIN_FRAC or MARGIN_FRAC
        lo, hi = lo - pad, hi + pad
        reasons.append("min-width floor (float)")
    return ("float", lo, hi, *tail), reasons


def check_search_space(authoritative_kinds: dict, proposed: dict, configs: list) -> dict:
    """Validate the inducer's proposed SEARCH_SPACE against the schema kinds and
    expand it to bracket every survived config. Returns
    {ok, finalized_space, expansions[], errors[]}. Hard errors (kind/key
    mismatch, bad tuple) leave finalized_space None and ok False; otherwise the
    finalized space is the proposed one widened to contain all configs."""
    errors: list = []
    akeys, pkeys = set(authoritative_kinds), set(proposed)
    for key in sorted(akeys - pkeys):
        errors.append({"code": "missing_key", "key": key, "detail": "in schema but not proposed"})
    for key in sorted(pkeys - akeys):
        errors.append({"code": "extra_key", "key": key, "detail": "proposed but not in schema"})
    for key in sorted(akeys & pkeys):
        entry = _as_tuple(proposed[key])
        if not valid_space_entry(entry):
            errors.append({"code": "bad_tuple", "key": key,
                           "detail": f"{proposed[key]!r} is not a valid (kind, ...) entry"})
        elif entry[0] != authoritative_kinds[key]:
            errors.append({"code": "kind_mismatch", "key": key,
                           "detail": f"proposed kind {entry[0]!r} != schema kind {authoritative_kinds[key]!r}"})
    if errors:
        return {"ok": False, "finalized_space": None, "expansions": [], "errors": errors}

    finalized, expansions = {}, []
    for key in authoritative_kinds:  # preserve schema key order
        values = [c[key] for c in configs if isinstance(c, dict) and key in c]
        new_entry, reasons = _expand_space_entry(proposed[key], values)
        finalized[key] = list(new_entry)  # JSON-friendly (lists, not tuples)
        if reasons:
            expansions.append({"key": key, "reasons": reasons})
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
            best = select_best(report)
            entry["best_params"] = best["best_params"]
            entry["best_score"] = best["best_score"]
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
# Selection is greedy on `best_warm_score` (the step-1 fidelity score, comparable
# across candidates) behind a promotion gate. NO headroom/spread term: warm-start
# configs are referenced from heterogeneous historical tasks, so their spread
# reflects the reference quality, not the landscape — it is not comparable across
# candidates' different methods. Pure read over the ledger; lower score is better.
DEFAULT_N_MIN = 10            # build breadth before any tuning (cold-start, inverted vs old Phase B)
DEFAULT_TOP_PERCENTILE = 80   # eligible iff the best untuned candidate is in the top (100-P)%


def select_candidate(ledger: dict, *, n_min: int = DEFAULT_N_MIN,
                     top_percentile: float = DEFAULT_TOP_PERCENTILE) -> dict:
    """Which candidate to deep-tune next (§15.4), or none. Eligible iff the
    population (non-crash, has best_warm_score) is >= n_min AND the best untuned
    candidate's best_warm_score ranks in the top (100-top_percentile)% across all
    candidates. Greedy on best_warm_score. Returns {run_id|None, reason, ...}."""
    cands = [r for r in ledger.get("records", [])
             if r.get("status") != "crash" and _is_finite_score(r.get("best_warm_score"))]
    n = len(cands)
    if n < n_min:
        return {"run_id": None, "n_candidates": n,
                "reason": f"population {n} < n_min {n_min} (breadth first)"}
    untuned = [r for r in cands if not r.get("tune")]
    if not untuned:
        return {"run_id": None, "n_candidates": n, "reason": "all candidates already tuned"}
    best = min(untuned, key=lambda r: r["best_warm_score"])
    value = best["best_warm_score"]
    # percentile = fraction of OTHER candidates strictly worse (higher score);
    # high percentile = among the best (matches ledger.py percentile semantics).
    worse = sum(1 for r in cands if r is not best and r["best_warm_score"] > value)
    pct = 100.0 * worse / (n - 1) if n > 1 else 100.0
    common = {"best_warm_score": value, "percentile": round(pct), "n_candidates": n}
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


def cmd_select_best(args) -> int:
    report = json.loads(Path(args.tune_report_json).read_text())
    print(json.dumps(select_best(report)))
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
    kinds = {k: _schema_kind(v) for k, v in _read_param_schema(args.candidate_path).items()}
    proposed = json.loads(Path(args.space_json).read_text())
    configs = json.loads(Path(args.configs_json).read_text())
    result = check_search_space(kinds, proposed, configs)
    if result["ok"]:
        # Persist the finalized (expanded) space back to the SAME artifact so the
        # caller never hand-extracts finalized_space from stdout — writing the whole
        # verdict dict there would corrupt the contract apply_search_space reads.
        # On error, leave the proposed file untouched for the agent's self-fix loop;
        # stdout carries the verdict (ok / expansions / errors) either way.
        Path(args.space_json).write_text(
            json.dumps(result["finalized_space"], indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_lineage_evidence(args) -> int:
    ids = [x.strip() for x in args.source_run_ids.split(",") if x.strip()]
    print(json.dumps(lineage_evidence(args.run_dir, ids), indent=2))
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
    print(json.dumps(select_candidate(ledger, n_min=n_min, top_percentile=top_p), indent=2))
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

    sb = sub.add_parser("select-best", help="Global-best trial over base+warm+phase_c.")
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
                        help="Validate + expand a proposed SEARCH_SPACE against the schema and survived configs; on ok, overwrite --space-json in place with the finalized space.")
    cs.add_argument("--candidate-path", required=True, type=Path,
                    help="train.py whose PARAM_SCHEMA gives the authoritative kinds")
    cs.add_argument("--space-json", required=True, type=Path,
                    help="JSON of the proposed SEARCH_SPACE {key: [kind, ...]}; overwritten in place with the finalized (expanded) space on ok")
    cs.add_argument("--configs-json", required=True, type=Path,
                    help="JSON list of survived config param dicts")
    cs.set_defaults(func=cmd_check_search_space)

    le = sub.add_parser("lineage-evidence",
                        help="Assemble per-hyperparam (value, score) evidence from parent candidates.")
    le.add_argument("--run-dir", required=True, type=Path)
    le.add_argument("--source-run-ids", required=True,
                    help="comma-separated parent run_ids (e.g. 003,005)")
    le.set_defaults(func=cmd_lineage_evidence)

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
