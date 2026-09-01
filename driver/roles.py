"""Role registry: prompt file, capability set, receipt schema, postconditions.

The positive capability set mirrors each retired agent's frontmatter
``tools:`` list minus Agent/Task/Skill; ``disallowed`` is defense in depth
under the fail-closed PreToolUse hook (session.py). Postconditions are
driver-side checks run after a session returns; each returns an error
string or None.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

Postcondition = Callable[["InvocationContext"], "str | None"]


@dataclass
class InvocationContext:
    task: str
    tag: str
    run_dir: Path
    invocation_id: int
    run_id: str | None = None
    round_no: int | None = None
    extra: dict = field(default_factory=dict)
    resume_session_id: str | None = None
    # A verbatim bounded payload for tool-free roles (e.g. the slate judge's
    # prepared prompt text). None keeps the historical message byte-identical.
    inline_payload: str | None = None

    def user_message(self) -> str:
        """Paths and compact ids first; an inline payload follows a fixed delimiter."""
        lines = [
            f"task: {self.task}",
            f"tag: {self.tag}",
            f"run_dir: {self.run_dir}",
        ]
        if self.run_id is not None:
            lines.append(f"run_id: {self.run_id}")
        if self.round_no is not None:
            lines.append(f"round: {self.round_no}")
        for key, value in self.extra.items():
            lines.append(f"{key}: {value}")
        message = "\n".join(lines)
        if self.inline_payload is not None:
            message += "\n\n---\n\n" + self.inline_payload
        return message

    def postcondition_checklist(self, role: "RoleDefinition") -> str:
        items = "\n".join(f"- {check.__doc__}" for check in role.postconditions)
        return (
            "The driver will verify ALL of the following after you return. "
            "Complete every item, then call mcp__receipts__submit_receipt:\n"
            f"{items or '- (receipt only)'}"
        )


@dataclass(frozen=True)
class RoleDefinition:
    name: str
    prompt_file: str  # relative to driver/prompts/
    tools: tuple[str, ...]
    disallowed: tuple[str, ...]
    receipt_schema: dict
    postconditions: tuple[Postcondition, ...] = ()
    corrective_attempts: int = 3
    # Hard per-session turn cap (SDK max_turns). None = unbounded. A role
    # whose job is bounded judgment (e.g. read-only diagnosis) gets one so a
    # model that keeps "confirming" instead of submitting a receipt is cut
    # off and surfaces as InvocationFailed to the caller.
    max_turns: int | None = None
    # When non-empty, Bash calls must start with one of these prefixes
    # (enforced by the PreToolUse hook in session.py).
    bash_patterns: tuple[str, ...] = ()
    # Long objective commands are driver-owned. These substrings keep an agent
    # from bypassing the typed job handoff and orphaning a GPU process.
    forbidden_bash_substrings: tuple[str, ...] = ()


def driver_job_handoff_problem(role_name: str, receipt: dict) -> str | None:
    """Validate the XOR between an intermediate job handoff and a terminal receipt."""
    job = receipt.get("driver_job")
    if not isinstance(job, dict):
        return None
    if role_name == "tunable-contract-extractor":
        if (
            receipt.get("status") == "driver_job"
            and receipt.get("ledger_updated") is False
            and job.get("kind") == "warmstart"
        ):
            return None
        return (
            "extractor driver_job requires status='driver_job', "
            "ledger_updated=false, and kind='warmstart'"
        )
    if role_name == "tuner-orchestrator":
        if (
            receipt.get("tuned") is False
            and receipt.get("ledger_updated") is False
            and job.get("kind") == "phase_c"
        ):
            return None
        return (
            "tuner driver_job requires tuned=false, ledger_updated=false, "
            "and kind='phase_c'"
        )
    return f"role {role_name} may not submit driver_job"


# --- helpers shared by postconditions -------------------------------------
#
# `ledger.py brief` deliberately omits per-record data (its docstring: "omit
# records and large experience text"), so anything that needs a specific
# record's status reads `ledger.json` directly. Reading is allowed; only
# hand-editing is banned — `tools/ledger.py` remains the sole mutator.


def ledger_brief(run_dir: Path) -> dict:
    out = subprocess.run(
        [sys.executable, "tools/ledger.py", "brief",
         "--ledger", str(run_dir / "ledger.json")],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _ledger_records(run_dir: Path) -> list[dict]:
    path = run_dir / "ledger.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [r for r in data.get("records", []) if isinstance(r, dict)]


def record_status(run_dir: Path, run_id: str) -> str | None:
    for record in _ledger_records(run_dir):
        if record.get("run_id") == run_id:
            return record.get("status")
    return None


# --- postconditions ---------------------------------------------------------


def background_artifacts_exist(ctx: InvocationContext) -> str | None:
    """background.md and background_retrieval.json exist in run_dir."""
    missing = [
        name
        for name in ("background.md", "background_retrieval.json")
        if not (ctx.run_dir / name).exists()
    ]
    return f"missing background artifacts: {missing}" if missing else None


def candidate_train_py_exists(ctx: InvocationContext) -> str | None:
    """candidates/<run_id>/train.py exists."""
    path = ctx.run_dir / "candidates" / str(ctx.run_id) / "train.py"
    return None if path.exists() else f"missing candidate file: {path}"


def record_is_terminal(ctx: InvocationContext) -> str | None:
    """the ledger record for run_id is keep/discard/crash, never pending."""
    status = record_status(ctx.run_dir, str(ctx.run_id))
    if status in ("keep", "discard", "crash"):
        return None
    return f"record {ctx.run_id} status is {status!r}, expected keep/discard/crash"


def refresh_flag_cleared(ctx: InvocationContext) -> str | None:
    """ledger brief reports experience_refresh_required == false."""
    brief = ledger_brief(ctx.run_dir)
    if brief.get("experience_refresh_required"):
        return "experience_refresh_required still true after refresh"
    return None


def actions_admitted(ctx: InvocationContext) -> str | None:
    """every action run_id named in extra['action_run_ids'] has a ledger record."""
    admitted = {r.get("run_id") for r in _ledger_records(ctx.run_dir)}
    missing = [r for r in ctx.extra.get("action_run_ids", []) if r not in admitted]
    return f"actions not admitted to ledger: {missing}" if missing else None


def editor_train_py_exists(ctx: InvocationContext) -> str | None:
    """the hillclimb working copy train.py exists in run_dir."""
    return None if (ctx.run_dir / "train.py").exists() else "missing working copy train.py"


def rewrite_train_py_exists(ctx: InvocationContext) -> str | None:
    """the rewrite candidate's train.py exists in candidate_dir."""
    path = Path(ctx.extra["candidate_dir"]) / "train.py"
    return None if path.exists() else f"missing candidate file: {path}"


# --- registry -----------------------------------------------------------------

_BASE_DISALLOWED = ("Agent", "Task", "Skill")

ROLES: dict[str, RoleDefinition] = {
    "background-researcher": RoleDefinition(
        name="background-researcher",
        prompt_file="background-researcher.md",
        tools=("Read", "Write", "Bash", "Glob", "WebSearch", "WebFetch"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={
            "status": ("enum", "ok"),
            "background": "str",
            "retrieval_manifest": "str",
        },
        postconditions=(background_artifacts_exist,),
    ),
    "idea-generator": RoleDefinition(
        name="idea-generator",
        prompt_file="idea-generator.md",
        tools=("Read", "Write", "Bash", "Glob"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"actions": "list"},
        postconditions=(actions_admitted,),
    ),
    "candidate-writer": RoleDefinition(
        name="candidate-writer",
        prompt_file="candidate-writer.md",
        tools=("Read", "Write", "Edit", "Glob"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={
            "status": ("enum", "written", "existing"),
            "wrote": "bool",
            "candidate_dir": "str",
        },
        postconditions=(candidate_train_py_exists,),
    ),
    "tunable-contract-extractor": RoleDefinition(
        name="tunable-contract-extractor",
        prompt_file="tunable-contract-extractor.md",
        tools=("Read", "Edit", "Write", "Bash", "Glob"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={
            "run_id": "str",
            "status": ("enum", "keep", "discard", "crash", "unevaluated",
                       "driver_job"),
            "ledger_updated": "bool",
            "driver_job": "?dict",
        },
        postconditions=(record_is_terminal,),
        forbidden_bash_substrings=("warmstart_eval.py", "nohup "),
    ),
    "tuner-orchestrator": RoleDefinition(
        name="tuner-orchestrator",
        prompt_file="tuner-orchestrator.md",
        tools=("Read", "Write", "Edit", "Bash", "Glob"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={
            "tuned_run_id": "str",
            "tuned": "bool",
            "ledger_updated": "bool",
            "driver_job": "?dict",
        },
        forbidden_bash_substrings=(
            "tools/tuners/grid_search.py",
            "tools/tuners/bo_search.py",
            "tools/tuners/cmaes_search.py",
            "nohup ",
        ),
    ),
    "experience-extractor": RoleDefinition(
        name="experience-extractor",
        prompt_file="experience-extractor.md",
        tools=("Read", "Write", "Bash"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={
            "search_space_state_revision": "int",
            "decision_ids": "list",
        },
        postconditions=(refresh_flag_cleared,),
    ),
    "crash-diagnosis": RoleDefinition(
        name="crash-diagnosis",
        prompt_file="crash-diagnosis.md",
        tools=("Read", "Bash", "Glob", "Grep"),
        disallowed=_BASE_DISALLOWED + ("Edit", "Write"),
        receipt_schema={
            "verdict": ("enum", "config_invalid", "code_incompatible", "abandon"),
            "summary": "str",
            "evidence": "list",
        },
        # the diagnosis methodology needs exactly one Bash command family
        bash_patterns=("python tools/tuners/tune_tools.py render-failure",),
        # observed unbounded "confirm the bug" loops at 55-301 turns; a
        # bounded read-only diagnosis needs far fewer
        max_turns=40,
    ),
    "hillclimb-editor": RoleDefinition(
        name="hillclimb-editor",
        prompt_file="hillclimb-editor.md",
        tools=("Read", "Write", "Edit", "Bash", "Glob"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"edited": "bool", "summary": "str"},
        postconditions=(editor_train_py_exists,),
    ),
    # rewrite loop: one long-lived session per imported candidate, one
    # in-place improvement edit per bout; basis names the intelligence the
    # edit consumed. No Bash — preflight/evaluation are driver-side.
    "rewrite-editor": RoleDefinition(
        name="rewrite-editor",
        prompt_file="rewrite-editor.md",
        tools=("Read", "Write", "Edit", "Glob", "Grep"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"edited": "bool", "summary": "str", "basis": "str"},
        postconditions=(rewrite_train_py_exists,),
    ),
    # judged_slate arm: a tool-free listwise judge. The bounded payload
    # arrives as the invocation context's inline_payload; the receipt's
    # ranking is the only decision input (rationale is audit-only). The one
    # corrective chance is a resume of this rollout's own session owned by
    # the driver's _invoke_slate_judge, so the generic repair loop is off.
    "slate-judge": RoleDefinition(
        name="slate-judge",
        prompt_file="slate-judge.md",
        tools=(),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"ranking": "list", "rationale": "str"},
        corrective_attempts=0,
        max_turns=8,
    ),
    # judged_slate arm: writes the PLAN for exactly one frozen slate seat.
    # Read-only: the driver persists the receipt as plans/slot-N.json; the
    # ledger admission helper owns every durable state change.
    "slate-plan-writer": RoleDefinition(
        name="slate-plan-writer",
        prompt_file="slate-plan-writer.md",
        tools=("Read",),
        disallowed=_BASE_DISALLOWED + ("Write", "Edit", "Bash"),
        receipt_schema={
            "slot": "int",
            "idea": "str",
            "change": "str",
            "candidate_name": "str",
            "route_provenance": "?dict",
        },
    ),
}
