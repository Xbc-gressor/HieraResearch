#!/usr/bin/env python3
"""Validate repo-local .claude/skills package structure."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / ".claude" / "skills"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$")
FORBIDDEN_DOCS = {
    "README.md",
    "CHANGELOG.md",
    "INSTALL.md",
    "INSTALLATION.md",
    "QUICK_REFERENCE.md",
}


def parse_frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text(errors="replace")
    if not text.startswith("---\n"):
        raise ValueError("missing YAML frontmatter")
    end = text.find("\n---\n", 4)
    if end == -1:
        raise ValueError("unterminated YAML frontmatter")
    block = text[4:end]
    data: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or raw_line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data


def validate_skill(skill_dir: Path) -> list[str]:
    errors: list[str] = []
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return [f"{skill_dir}: missing SKILL.md"]

    try:
        frontmatter = parse_frontmatter(skill_md)
    except ValueError as exc:
        return [f"{skill_md}: {exc}"]

    name = frontmatter.get("name", "")
    description = frontmatter.get("description", "")

    if name != skill_dir.name:
        errors.append(f"{skill_md}: name '{name}' does not match folder '{skill_dir.name}'")
    if not NAME_RE.match(name):
        errors.append(f"{skill_md}: invalid skill name '{name}'")
    if len(description.split()) < 8:
        errors.append(f"{skill_md}: description is too short to be a useful trigger")

    for path in skill_dir.rglob("*"):
        if path.is_file() and path.name in FORBIDDEN_DOCS:
            errors.append(f"{path}: auxiliary docs do not belong inside skill packages")

    return errors


def main() -> int:
    if not SKILLS_DIR.exists():
        print(f"missing skills directory: {SKILLS_DIR}", file=sys.stderr)
        return 1

    skill_dirs = sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if not skill_dirs:
        print("no skills found", file=sys.stderr)
        return 1

    errors: list[str] = []
    for skill_dir in skill_dirs:
        errors.extend(validate_skill(skill_dir))

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(skill_dirs)} skill package(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

