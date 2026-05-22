#!/usr/bin/env python3
"""Validate task package structure and task.toml metadata."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$")
REQUIRED_TOP_LEVEL = {"name", "description"}
REQUIRED_SECTIONS = {
    "env": {"type", "project"},
    "run": {"working_dir", "command", "timeout_seconds", "log_template"},
    "result": {"metric", "lower_is_better", "parser", "required_patterns", "results_file"},
    "constraints": {"editable_files", "readonly_files", "allow_dependencies"},
}


def strip_comment(line: str) -> str:
    in_quote = False
    quote = ""
    for i, char in enumerate(line):
        if char in {"'", '"'} and (i == 0 or line[i - 1] != "\\"):
            if not in_quote:
                in_quote = True
                quote = char
            elif quote == char:
                in_quote = False
        elif char == "#" and not in_quote:
            return line[:i]
    return line


def parse_value(value: str):
    value = value.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def parse_task_toml(path: Path) -> dict:
    data: dict = {}
    current = data
    for line_no, raw_line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        line = strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            if not section:
                raise ValueError(f"line {line_no}: empty section")
            current = data.setdefault(section, {})
            continue
        if "=" not in line:
            raise ValueError(f"line {line_no}: expected key = value")
        key, value = line.split("=", 1)
        current[key.strip()] = parse_value(value)
    return data


def validate_task(task_dir: Path) -> list[str]:
    errors: list[str] = []
    for filename in ("TASK.md", "task.toml", "pyproject.toml"):
        if not (task_dir / filename).exists():
            errors.append(f"{task_dir}: missing {filename}")

    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return errors

    try:
        data = parse_task_toml(task_toml)
    except ValueError as exc:
        errors.append(f"{task_toml}: {exc}")
        return errors

    missing = sorted(REQUIRED_TOP_LEVEL - data.keys())
    if missing:
        errors.append(f"{task_toml}: missing top-level fields: {', '.join(missing)}")

    name = data.get("name")
    if name != task_dir.name:
        errors.append(f"{task_toml}: name '{name}' does not match folder '{task_dir.name}'")
    if not isinstance(name, str) or not NAME_RE.match(name):
        errors.append(f"{task_toml}: invalid task name '{name}'")

    for section, required_keys in REQUIRED_SECTIONS.items():
        section_data = data.get(section)
        if not isinstance(section_data, dict):
            errors.append(f"{task_toml}: missing [{section}] section")
            continue
        missing_keys = sorted(required_keys - section_data.keys())
        if missing_keys:
            errors.append(f"{task_toml}: [{section}] missing fields: {', '.join(missing_keys)}")

    env = data.get("env", {})
    if isinstance(env, dict):
        project = env.get("project")
        if project and not (ROOT / project).is_dir():
            errors.append(f"{task_toml}: env.project does not exist: {project}")
        if env.get("type") != "uv":
            errors.append(f"{task_toml}: env.type must be 'uv'")

    run = data.get("run", {})
    if isinstance(run, dict):
        working_dir = run.get("working_dir")
        if working_dir and not (ROOT / working_dir).is_dir():
            errors.append(f"{task_toml}: run.working_dir does not exist: {working_dir}")
        if not isinstance(run.get("command"), str) or not run.get("command"):
            errors.append(f"{task_toml}: run.command must be a non-empty string")
        if not isinstance(run.get("timeout_seconds"), int):
            errors.append(f"{task_toml}: run.timeout_seconds must be an integer")
        log_template = run.get("log_template")
        if not isinstance(log_template, str) or "{run_id}" not in log_template:
            errors.append(f"{task_toml}: run.log_template must include '{{run_id}}'")

    result = data.get("result", {})
    if isinstance(result, dict):
        if not isinstance(result.get("metric"), str) or not result.get("metric"):
            errors.append(f"{task_toml}: result.metric must be a non-empty string")
        if not isinstance(result.get("lower_is_better"), bool):
            errors.append(f"{task_toml}: result.lower_is_better must be a boolean")
        parser = result.get("parser")
        if not isinstance(parser, str):
            errors.append(f"{task_toml}: result.parser must be a string")
        elif not (ROOT / parser).is_file():
            errors.append(f"{task_toml}: result.parser does not exist: {parser}")
        patterns = result.get("required_patterns")
        if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
            errors.append(f"{task_toml}: result.required_patterns must be a list of strings")
        else:
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    errors.append(f"{task_toml}: invalid result.required_patterns entry {pattern!r}: {exc}")
        if not isinstance(result.get("results_file"), str) or not result.get("results_file"):
            errors.append(f"{task_toml}: result.results_file must be a non-empty string")

    constraints = data.get("constraints", {})
    if isinstance(constraints, dict):
        for key in ("editable_files", "readonly_files"):
            value = constraints.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                errors.append(f"{task_toml}: constraints.{key} must be a list of strings")
                continue
            for item in value:
                if not (task_dir / item).exists():
                    errors.append(f"{task_toml}: constraints.{key} entry does not exist: {item}")
        if not isinstance(constraints.get("allow_dependencies"), bool):
            errors.append(f"{task_toml}: constraints.allow_dependencies must be a boolean")

    candidate = data.get("candidate")
    if candidate is not None:
        if not isinstance(candidate, dict):
            errors.append(f"{task_toml}: [candidate] must be a table")
        else:
            if not isinstance(candidate.get("enabled"), bool):
                errors.append(f"{task_toml}: candidate.enabled must be a boolean")
            root_template = candidate.get("root_template")
            if not isinstance(root_template, str) or not root_template:
                errors.append(f"{task_toml}: candidate.root_template must be a non-empty string")
            else:
                for placeholder in ("{task_name}", "{tag}", "{run_id}"):
                    if placeholder not in root_template:
                        errors.append(
                            f"{task_toml}: candidate.root_template must include {placeholder}"
                        )
            copy_files = candidate.get("copy_files")
            if not isinstance(copy_files, list) or not all(isinstance(item, str) for item in copy_files):
                errors.append(f"{task_toml}: candidate.copy_files must be a list of strings")
            else:
                for item in copy_files:
                    if not (task_dir / item).is_file():
                        errors.append(f"{task_toml}: candidate.copy_files entry does not exist: {item}")
            entrypoint = candidate.get("entrypoint")
            if not isinstance(entrypoint, str) or not entrypoint:
                errors.append(f"{task_toml}: candidate.entrypoint must be a non-empty string")
            elif isinstance(copy_files, list) and entrypoint not in copy_files:
                errors.append(f"{task_toml}: candidate.entrypoint must be listed in copy_files")
            for key in ("editable_files", "readonly_files"):
                value = candidate.get(key)
                if value is None:
                    continue
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    errors.append(f"{task_toml}: candidate.{key} must be a list of strings")

    return errors


def main() -> int:
    if not TASKS_DIR.exists():
        print(f"missing tasks directory: {TASKS_DIR}", file=sys.stderr)
        return 1

    task_dirs = sorted(path for path in TASKS_DIR.iterdir() if path.is_dir())
    if not task_dirs:
        print("no tasks found", file=sys.stderr)
        return 1

    errors: list[str] = []
    for task_dir in task_dirs:
        errors.extend(validate_task(task_dir))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(task_dirs)} task package(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
