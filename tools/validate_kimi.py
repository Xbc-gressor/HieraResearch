#!/usr/bin/env python3
"""Validate the kimi-cli runtime port (.kimi/) without invoking a model.

Checks the structural contracts the harness relies on:

- all 8 agents exist as <name>.yaml + <name>.md and the YAML parses;
- every referenced tool is a real kimi_cli tool;
- the orchestrator's spawn closure is exactly the six role agents;
- per-role tool restrictions survived the extend chain (no child has the
  Agent tool, candidate-writer has no Shell, only background-researcher has
  web tools);
- prompt bodies carry no Claude-isms or unknown template variables;
- the hook fragment and the launcher's merged config are well-formed.

Run: python tools/validate_kimi.py
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
AGENTS_DIR = ROOT / ".kimi" / "agents"

# Every tool kimi-cli 1.43 ships (from kimi_cli/agents/default/agent.yaml plus
# the optional Think/SendDMail), keyed by the path form used in agent files.
KNOWN_TOOLS = {
    "kimi_cli.tools.agent:Agent",
    "kimi_cli.tools.ask_user:AskUserQuestion",
    "kimi_cli.tools.todo:SetTodoList",
    "kimi_cli.tools.shell:Shell",
    "kimi_cli.tools.background:TaskList",
    "kimi_cli.tools.background:TaskOutput",
    "kimi_cli.tools.background:TaskStop",
    "kimi_cli.tools.file:ReadFile",
    "kimi_cli.tools.file:ReadMediaFile",
    "kimi_cli.tools.file:Glob",
    "kimi_cli.tools.file:Grep",
    "kimi_cli.tools.file:WriteFile",
    "kimi_cli.tools.file:StrReplaceFile",
    "kimi_cli.tools.web:SearchWeb",
    "kimi_cli.tools.web:FetchURL",
    "kimi_cli.tools.plan:ExitPlanMode",
    "kimi_cli.tools.plan.enter:EnterPlanMode",
    "kimi_cli.tools.think:Think",
    "kimi_cli.tools.dmail:SendDMail",
}

# The built-in default agent's tool list (what `extend: default` starts from).
DEFAULT_TOOLS = KNOWN_TOOLS - {
    "kimi_cli.tools.think:Think",
    "kimi_cli.tools.dmail:SendDMail",
}

EXPECTED_AGENTS = {
    "autoresearch-experiment",
    "autoresearch-hillclimb",
    "background-researcher",
    "candidate-writer",
    "experience-extractor",
    "idea-generator",
    "tunable-contract-extractor",
    "tuner-orchestrator",
}

CHILD_AGENTS = {
    "background-researcher",
    "candidate-writer",
    "experience-extractor",
    "idea-generator",
    "tunable-contract-extractor",
    "tuner-orchestrator",
}

WEB_TOOLS = {"kimi_cli.tools.web:SearchWeb", "kimi_cli.tools.web:FetchURL"}
AGENT_TOOL = "kimi_cli.tools.agent:Agent"
SHELL_TOOL = "kimi_cli.tools.shell:Shell"

KNOWN_TEMPLATE_VARS = {
    "KIMI_NOW",
    "KIMI_WORK_DIR",
    "KIMI_WORK_DIR_LS",
    "KIMI_AGENTS_MD",
    "KIMI_SKILLS",
    "KIMI_ADDITIONAL_DIRS_INFO",
}

# Patterns that indicate an unconverted Claude-ism in a prompt body.
CLAUDE_ISMS = [
    (r"claude --agent", "claude CLI launch command"),
    (r"\bWebSearch\b|\bWebFetch\b", "Claude web tool name"),
    (r"(?<!`)`?\bBash\b(?!`)`?", "Claude Bash tool name (```bash fences excepted)"),
    (r"Skill\(", "Claude Skill tool invocation"),
    (r"\.claude/", "reference to .claude/ path"),
    (r"CLAUDE\.md|CLAUDE_PROJECT_DIR", "CLAUDE.md / CLAUDE_PROJECT_DIR reference"),
    (r"claude-webfetch", "claude-webfetch backend label"),
]


def load_agent(name: str) -> tuple[dict | None, str | None]:
    path = AGENTS_DIR / f"{name}.yaml"
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        return None, f"{path}: {exc}"
    if not isinstance(data, dict) or data.get("version") != 1 or "agent" not in data:
        return None, f"{path}: expected `version: 1` and an `agent:` mapping"
    return data["agent"], None


def effective_tools(name: str, _seen: set[str] | None = None) -> set[str]:
    """Resolve the extend chain: start from the parent's tools, drop excludes.

    `allowed_tools`, when present, is treated as the authoritative final set
    (the belt-and-suspenders pattern the built-in subagents use).
    """
    _seen = _seen or set()
    if name in _seen:
        raise ValueError(f"extend cycle at {name}")
    _seen.add(name)
    agent, err = load_agent(name)
    if err:
        raise ValueError(err)
    extend = agent.get("extend", "default")
    if extend == "default":
        tools = set(DEFAULT_TOOLS)
    else:
        parent = str(extend).removeprefix("./").removesuffix(".yaml")
        tools = effective_tools(parent, _seen)
    tools -= set(agent.get("exclude_tools") or [])
    allowed = agent.get("allowed_tools")
    if allowed:
        tools &= set(allowed)
    return tools


def main() -> int:
    errors: list[str] = []

    # 1. All agents present, YAML parses, prompt file exists and matches.
    specs: dict[str, dict] = {}
    for name in sorted(EXPECTED_AGENTS):
        yaml_path = AGENTS_DIR / f"{name}.yaml"
        if not yaml_path.is_file():
            errors.append(f"missing {yaml_path}")
            continue
        agent, err = load_agent(name)
        if err:
            errors.append(err)
            continue
        specs[name] = agent
        prompt = agent.get("system_prompt_path")
        if not prompt:
            errors.append(f"{name}.yaml: no system_prompt_path")
        elif not (AGENTS_DIR / prompt).is_file():
            errors.append(f"{name}.yaml: system_prompt_path {prompt} does not exist")
        elif Path(prompt).stem != name:
            errors.append(f"{name}.yaml: prompt file {prompt} should be ./{name}.md")

    # 2. Tool references are real; exclusions hold after extend resolution.
    for name, agent in specs.items():
        for field in ("tools", "exclude_tools", "allowed_tools"):
            for ref in agent.get(field) or []:
                if ref not in KNOWN_TOOLS:
                    errors.append(f"{name}.yaml: unknown tool {ref} in {field}")
        try:
            tools = effective_tools(name)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if name in CHILD_AGENTS and AGENT_TOOL in tools:
            errors.append(f"{name}: child agent must not keep the Agent tool")
        if name == "candidate-writer" and SHELL_TOOL in tools:
            errors.append("candidate-writer: must not keep the Shell tool")
        if name != "background-researcher" and tools & WEB_TOOLS:
            errors.append(f"{name}: only background-researcher may keep web tools")
        if name == "background-researcher" and not tools & WEB_TOOLS:
            errors.append("background-researcher: lost its web tools")

    # 3. Spawn closure: the orchestrator declares exactly the six children.
    orchestrator = specs.get("autoresearch-experiment", {})
    subagents = orchestrator.get("subagents") or {}
    if set(subagents) != CHILD_AGENTS:
        errors.append(
            "autoresearch-experiment subagents must be exactly "
            f"{sorted(CHILD_AGENTS)}, got {sorted(subagents)}"
        )
    for child, entry in subagents.items():
        path = entry.get("path") if isinstance(entry, dict) else None
        if not path or not (AGENTS_DIR / path).is_file():
            errors.append(f"subagent {child}: bad path {path!r}")
        if not (isinstance(entry, dict) and entry.get("description")):
            errors.append(f"subagent {child}: missing routing description")
    for name, agent in specs.items():
        if name != "autoresearch-experiment" and agent.get("subagents"):
            errors.append(f"{name}: only the orchestrator may declare subagents")
    if AGENT_TOOL not in effective_tools("autoresearch-experiment"):
        errors.append("autoresearch-experiment: lost the Agent tool")
    if AGENT_TOOL in effective_tools("autoresearch-hillclimb"):
        errors.append("autoresearch-hillclimb: must not keep the Agent tool")

    # 4. Prompt bodies: no Claude-isms, no unknown ${VARS}.
    for name in sorted(EXPECTED_AGENTS):
        md_path = AGENTS_DIR / f"{name}.md"
        if not md_path.is_file():
            continue
        body = re.sub(r"```bash\n", "", md_path.read_text())  # fences are fine
        for pattern, label in CLAUDE_ISMS:
            for match in re.finditer(pattern, body):
                line = body[: match.start()].count("\n") + 1
                errors.append(f"{md_path.name}:{line}: {label}")
        for var in re.findall(r"\$\{([A-Z_]+)\}", body):
            if var not in KNOWN_TEMPLATE_VARS:
                errors.append(f"{md_path.name}: unknown template var ${{{var}}}")
        if re.search(r"\{%|\{\{", body):
            errors.append(f"{md_path.name}: Jinja2 control/interpolation hazard")

    # 5. Hook fragment + launcher merge.
    try:
        sys.path.insert(0, str(ROOT / "tools"))
        import kimi_run

        merged = kimi_run.build_merged_config()
        hook_cmds = [
            h.get("command", "") for h in merged.get("hooks", [])
            if h.get("event") == "PreToolUse" and h.get("matcher") == "Agent"
        ]
        guard = [c for c in hook_cmds if "harness_guard.py" in c]
        if not guard:
            errors.append("merged config: no PreToolUse/Agent harness_guard hook")
        else:
            script = next(
                (tok for tok in guard[0].split() if tok.endswith("harness_guard.py")),
                None,
            )
            if "${HIERA_ROOT}" in guard[0] or not script or not Path(script).is_file():
                errors.append(
                    f"merged config: unexpanded or missing guard path: {guard[0]}"
                )
            if "--allowed-subagents" not in guard[0]:
                errors.append(
                    "merged config: guard hook lacks --allowed-subagents "
                    "(spawn closure would not be enforced)"
                )
        json.dumps(merged)  # must serialize for --config
    except Exception as exc:  # noqa: BLE001 - report any merge failure
        errors.append(f"launcher merge failed: {exc}")

    # 6. Skill + rules present with required shape.
    skill = ROOT / ".kimi" / "skills" / "crash-diagnosis" / "SKILL.md"
    if not skill.is_file():
        errors.append(f"missing {skill}")
    else:
        head = skill.read_text().split("---", 2)
        if len(head) < 3 or "name: crash-diagnosis" not in head[1]:
            errors.append(f"{skill}: bad frontmatter")
    for extra in (ROOT / ".kimi" / "rules" / "ledger.md", ROOT / "AGENTS.md"):
        if not extra.is_file():
            errors.append(f"missing {extra}")

    if errors:
        print("validate_kimi: FAILED")
        for err in errors:
            print(f"  - {err}")
        return 1
    print(f"validate_kimi: OK ({len(EXPECTED_AGENTS)} agents, spawn closure + tool scoping verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
