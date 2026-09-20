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
    # The run whose time budget bounds this session. None = derive from
    # run_dir (itself, or the enclosing runs/<task>/<tag> when the session's
    # storage directory is a per-candidate sub-directory such as
    # candidates/<id>/_hebo_llm). Never rendered into the message.
    budget_run_dir: Path | None = None

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
    # When true, Bash runs only the retrieval-adapter commands
    # (adapter_command_verdict in session.py): one allowlisted adapter
    # invocation per call, no shell operators, and --manifest confined to the
    # run's own directory.
    bash_adapter_only: bool = False
    # Early corrective denies for consecutive identical (tool, input) calls
    # BEFORE the hard repetition trip: #2..#LIMIT-1 are denied with an
    # explicit count while the invocation stays alive. Reserved for roles
    # with an observed read-loop attractor (slate-plan-writer, 2026-09-16:
    # 75/79 sessions lost to byte-identical Reads the CLI could not dissuade).
    early_repeat_correct: bool = False
    # Long objective commands are driver-owned. These substrings keep an agent
    # from bypassing the typed job handoff and orphaning a GPU process.
    forbidden_bash_substrings: tuple[str, ...] = ()
    # Backstop wall-clock bound for one invocation (seconds; None = only the
    # run deadline bounds it). Set far above the observed maximum: a trip is
    # a hang signal to inspect, not a trimming threshold. Checked per
    # streamed message, so it fires at the next message after the bound.
    wall_limit_seconds: float | None = None
    # At the wall limit, roles whose partial result is itself valid (an edit,
    # a draft, notes) get one "submit now" turn with a short grace; terminal
    # roles (ledger writers, judges, diagnosis) fail outright, because a
    # coerced receipt would be a fabricated verdict.
    soft_rescue: bool = False
    # Longest legal silence between two streamed messages (tool execution
    # produces none). Guards against a hung CLI only; None disables.
    idle_timeout_seconds: float | None = 900.0


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


def tuner_target_matches_handoff(ctx: InvocationContext) -> str | None:
    """A pinned tune invocation's receipt names the selected candidate."""
    handoff = ctx.extra.get("scheduler_selection")
    if not handoff:
        return None
    from .receipts import ReceiptStore  # local import: receipts reads roles
    path = ReceiptStore(ctx.run_dir).receipt_path(
        "tuner-orchestrator", ctx.invocation_id)
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    selected = (json.loads(handoff) or {}).get("run_id")
    executed = receipt.get("tuned_run_id", "none")
    if executed not in (selected, "none"):
        return (
            f"tuned_run_id {executed!r} must echo the scheduler selection "
            f"{selected!r} or be 'none'; a pinned invocation cannot name "
            "another candidate"
        )
    return None


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


def _own_receipt(ctx: InvocationContext, role_name: str) -> dict | None:
    from .receipts import ReceiptStore  # local import: receipts reads roles
    path = ReceiptStore(ctx.run_dir).receipt_path(role_name, ctx.invocation_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def record_is_terminal(ctx: InvocationContext) -> str | None:
    """the ledger record for run_id is keep/discard/crash (written via
    record-run); the one exception is the zero-attempt budget-exhausted path,
    where the receipt says status=unevaluated, ledger_updated=false and the
    record stays pending for the driver to resolve."""
    status = record_status(ctx.run_dir, str(ctx.run_id))
    if status in ("keep", "discard", "crash"):
        return None
    receipt = _own_receipt(ctx, "tunable-contract-extractor") or {}
    if receipt.get("status") == "unevaluated" \
            and receipt.get("ledger_updated") is False \
            and status in ("pending", "unevaluated"):
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
        tools=("Read", "Write", "Bash", "Glob"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={
            "status": ("enum", "ok"),
            "background": "str",
            "retrieval_manifest": "str",
        },
        postconditions=(background_artifacts_exist,),
        bash_adapter_only=True,
        # Runs once per run; healthy cells observed 1066-1259s, degraded
        # retrieval pushed one session past 2164s (2026-09-19 calibration).
        wall_limit_seconds=3600.0,
        soft_rescue=True,
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
        # Observed max 874s across three MLE cells (2026-09-19 calibration);
        # 960s left only 9% headroom, violating the far-above-maximum rule.
        wall_limit_seconds=1800.0,
        soft_rescue=True,
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
        wall_limit_seconds=900.0,
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
        postconditions=(tuner_target_matches_handoff,),
        forbidden_bash_substrings=(
            "tools/tuners/grid_search.py",
            "tools/tuners/bo_search.py",
            "tools/tuners/cmaes_search.py",
            "nohup ",
        ),
        wall_limit_seconds=900.0,
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
        wall_limit_seconds=900.0,
        soft_rescue=True,
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
        wall_limit_seconds=600.0,
    ),
    "hillclimb-editor": RoleDefinition(
        name="hillclimb-editor",
        prompt_file="hillclimb-editor.md",
        tools=("Read", "Write", "Edit", "Bash", "Glob"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"edited": "bool", "summary": "str"},
        postconditions=(editor_train_py_exists,),
        # Backstop for pathologies the repetition breaker cannot fingerprint
        # (varied junk turns). Healthy invocations stay far below: a few
        # reads, one edit, one receipt. The incident run burned 894 turns.
        max_turns=120,
        wall_limit_seconds=900.0,
        soft_rescue=True,
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
        wall_limit_seconds=900.0,
        soft_rescue=True,
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
        wall_limit_seconds=600.0,
    ),
    # background faithfulness audit: a tool-free judge over a prepared
    # payload of claim↔receipt mappings. corrective_attempts=0: the driver's
    # audit gate owns the single fresh-session retry.
    "background-faithfulness-judge": RoleDefinition(
        name="background-faithfulness-judge",
        prompt_file="background-faithfulness-judge.md",
        tools=(),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"verdicts": "list", "rationale": "str"},
        corrective_attempts=0,
        max_turns=8,
        wall_limit_seconds=600.0,
    ),
    # judged_slate arm: writes the PLAN for exactly one frozen slate seat.
    # Read-only: the driver persists the receipt as plans/slot-N.json; the
    # ledger admission helper owns every durable state change.
    "slate-plan-writer": RoleDefinition(
        name="slate-plan-writer",
        prompt_file="slate-plan-writer.md",
        tools=("Read",),
        disallowed=_BASE_DISALLOWED + ("Write", "Edit", "Bash"),
        early_repeat_correct=True,
        receipt_schema={
            "slot": "int",
            "idea": "str",
            "change": "str",
            "candidate_name": "str",
            "route_provenance": "?dict",
        },
        wall_limit_seconds=600.0,
        soft_rescue=True,
    ),
}
