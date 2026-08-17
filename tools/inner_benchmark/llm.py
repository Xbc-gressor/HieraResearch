"""LLM bout-session layer for the inner-tuner benchmark (PLAN §四).

This module is the session/context infrastructure consumed by the four
LLM-using arms (Current's rewarm step, LLM hillclimb, active-set, pool
proposer). It owns:

- ``LLMConfig`` — the manifest record for every LLM call in a cell
  (PLAN §5.3: one pinned model id; production exposes NO decoding knobs,
  so ``decoding`` is recorded as null). The model id is ALWAYS supplied by
  the caller (cell wiring); nothing here hardcodes a default model.
- ``BoutSession`` — one instance = one bout-scoped persistent session.
  A bout keeps ONE session: the first ``ask`` starts fresh, later asks
  chain ``resume_session_id`` from this bout's previous invocation
  (persisted/loaded via ReceiptStore, the production pattern of
  driver/loops/hillclimb.py ``_editor_session``). A new bout builds a NEW
  BoutSession from structured state — never resume another bout's
  transcript.
- ``BENCH_ROLES`` — benchmark-local role registry (deliberately NOT the
  production ``driver.roles.ROLES``; ``runner.run`` accepts any
  RoleDefinition). Postcondition choice: none — receipt shape is already
  enforced by the schema validator, and benchmark legality (bounds,
  schema, duplicates) is the runner's deterministic preflight, not a
  driver postcondition.
- Structured context builders — the §四 standard view as plain-text
  blocks that arms place into invocation extras.

Import-safety contract: this module imports ONLY stdlib plus
``driver.roles`` / ``driver.receipts`` at module level (both stdlib-only
themselves; ``driver.session`` needs ``anyio`` and the SDK, so it appears
only under TYPE_CHECKING). ``import llm`` and all formatter tests work
SDK-free.

Extras-key contract (invocation context ``extra`` values are strings):

- First call of a bout: the keys produced by ``first_message_blocks`` —
  ``task``, ``items``, ``search_space``, ``candidate``, ``incumbent``, ``history``,
  ``protocol``, ``budget``, plus ``evidence`` when provided.
- Later calls: arm-specific incremental blocks. Standardized keys:
  ``outcome`` (``outcome_message`` after each evaluation) and
  ``feasible_set`` (active-set's filtered (parameter, step) joint set,
  formatted by that arm). ``bench-hillclimb-editor`` additionally
  receives ``working_copy`` (absolute path of the file it may Read/Edit).

``evidence`` policy (PLAN §四): today only same-candidate history may be
attached as read-only text blocks; cross-candidate (parent/sibling)
retrieval will later enter through the same ``evidence`` parameter — old
sessions of other candidates are never resumed.

Usage capture: claude-agent-sdk 0.2.130's ``ResultMessage.usage`` is a
free-form ``dict[str, Any] | None`` (the CLI's raw usage object —
``input_tokens`` / ``output_tokens`` / ``cache_*_input_tokens`` when the
CLI reports them). The production runner emits it on ``session_end``
events (one per query turn, so corrective follow-ups add rows for the
same invocation). BoutSession re-reads ``<run_dir>/driver_events.jsonl``
after each call and sums the numeric usage fields across that
invocation's ``session_end`` rows. A runner that emits nothing (fakes)
yields zero-token entries — call counts are always recorded.
"""

from __future__ import annotations

import json
import math
import sys
import tomllib
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

# Repo root makes ``driver.*`` importable (same convention as the sibling
# modules inserting tools/ paths). Both imported driver modules are
# stdlib-only at module level.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from driver.receipts import ReceiptStore  # noqa: E402
from driver.roles import (  # noqa: E402
    _BASE_DISALLOWED,
    InvocationContext,
    RoleDefinition,
)

if TYPE_CHECKING:  # annotations only; keeps runtime imports stdlib-only
    import checkpoint as _checkpoint
    import space as _space
    from driver.session import SessionRunner

# File name of the production events log (driver/events.py EventsLog).
EVENTS_FILENAME = "driver_events.jsonl"

# Standardized extras keys (see module docstring).
BLOCK_KEYS = ("task", "items", "search_space", "candidate", "incumbent", "history",
              "protocol", "budget")
TASK_KEY = "task"
EVIDENCE_KEY = "evidence"
OUTCOME_KEY = "outcome"
FEASIBLE_SET_KEY = "feasible_set"
# bench-hillclimb-editor only: absolute path of the file it may Read/Edit.
WORKING_COPY_KEY = "working_copy"

SDK_DISTRIBUTION = "claude-agent-sdk"


# --- LLMConfig ---------------------------------------------------------------


def detect_sdk_version() -> str:
    """Installed claude-agent-sdk version, or "not-installed" (SDK-free envs)."""
    try:
        return importlib_metadata.version(SDK_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


@dataclass(frozen=True)
class LLMConfig:
    """What every LLM call in a cell pins to (PLAN §5.3).

    ``model`` is always caller-supplied. ``sdk_version`` auto-detects at
    construction; ``from_manifest`` restores the recorded value verbatim
    (the manifest records what WAS used, it never re-detects).
    """

    model: str
    sdk_version: str = field(default_factory=detect_sdk_version)
    extra: dict = field(default_factory=dict)

    def to_manifest(self) -> dict:
        return {
            "model": self.model,
            "sdk_version": self.sdk_version,
            # Production exposes no decoding parameters; the model id is the
            # only LLM knob. Recorded explicitly as null.
            "decoding": None,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_manifest(cls, data: dict) -> "LLMConfig":
        return cls(
            model=data["model"],
            sdk_version=data.get("sdk_version", "unknown"),
            extra=dict(data.get("extra", {})),
        )


# --- benchmark roles ----------------------------------------------------------

# Benchmark-local registry — NOT merged into production driver.roles.ROLES.
# All roles mirror production ``_BASE_DISALLOWED`` (Agent/Task/Skill denied)
# and keep the default corrective_attempts=3. No postconditions: receipt
# shape is schema-enforced; legality is the benchmark runner's deterministic
# preflight (see module docstring).
BENCH_ROLES: dict[str, RoleDefinition] = {
    # Current's rewarm step: propose up to 3 configs to continue a bout.
    "bench-rewarm-proposer": RoleDefinition(
        name="bench-rewarm-proposer",
        prompt_file="bench-rewarm-proposer.md",
        tools=(),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"configs": "list", "rationale": "?str"},
    ),
    # Active-set: pick ONE (parameter, step) out of the provided feasible set.
    "bench-active-set": RoleDefinition(
        name="bench-active-set",
        prompt_file="bench-active-set.md",
        tools=(),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"parameter": "str", "step": "float", "rationale": "?str"},
    ),
    # Pool proposer: one call returns POOL=5 distinct configs + self-ranking.
    "bench-pool-proposer": RoleDefinition(
        name="bench-pool-proposer",
        prompt_file="bench-pool-proposer.md",
        tools=(),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"configs": "list", "order": "list", "rationale": "?str"},
    ),
    # Same-pool shadow experiment: one fresh, tool-free session judges one
    # pair and is then discarded. A/B are opaque positions, not proposer
    # ranks; the standalone runner maps the verdict back to pool indexes.
    "bench-pool-pairwise-judge": RoleDefinition(
        name="bench-pool-pairwise-judge",
        prompt_file="bench-pool-pairwise-judge.md",
        tools=(),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={
            "winner": ("enum", "A", "B"),
            "reasoning": "?str",
        },
    ),
    # Param-only hillclimb editor (production hillclimb-editor narrowed to
    # SEARCH_SPACE parameter values; one change per invocation).
    "bench-hillclimb-editor": RoleDefinition(
        name="bench-hillclimb-editor",
        prompt_file="bench-hillclimb-editor.md",
        tools=("Read", "Edit"),
        disallowed=_BASE_DISALLOWED,
        receipt_schema={"edited": "bool", "summary": "str"},
    ),
}


# --- BoutSession --------------------------------------------------------------


class BoutSession:
    """One bout-scoped LLM session (PLAN §四).

    The first ``ask`` starts a fresh SDK session (no resume); each later
    ``ask`` chains ``resume_session_id`` from THIS bout's previous
    successful invocation via ReceiptStore — the production hillclimb
    chaining pattern. Construction starts nothing (lazy). Only successful
    calls advance the resume chain: after ``InvocationFailed`` the next
    ``ask`` resumes from the last good invocation. Instances hold no
    global state; multiple bouts (even over one run_dir/store) chain
    independently — invocation ids come from the shared store, session
    files are keyed by (role, invocation_id).
    """

    def __init__(
        self,
        *,
        role: RoleDefinition,
        runner: "SessionRunner",
        run_dir,
        task: str,
        tag: str,
        first_extras: dict | None = None,
    ):
        self.role = role
        self.runner = runner
        self.run_dir = Path(run_dir)
        self.task = task
        self.tag = tag
        self.first_extras = dict(first_extras or {})
        self._store = ReceiptStore(self.run_dir)
        self._last_invocation_id: int | None = None
        self.usage_log: list[dict] = []
        # Token accounting reads the events log the runner WRITES. If the two
        # disagree the usage silently reads zero, byte-identical to a legitimate
        # fake-runner cell. Real runners expose events.path, so check it.
        events_path = getattr(getattr(runner, "events", None), "path", None)
        if events_path is not None and Path(events_path).parent != self.run_dir:
            raise ValueError(
                f"run_dir {self.run_dir} does not hold the runner's events log "
                f"({events_path}); token usage would silently read zero"
            )

    @property
    def call_count(self) -> int:
        """Number of ``ask`` invocations, INCLUDING failed ones.

        A failed invocation still ran the model (its corrective follow-ups are
        exactly what failed), so it counts for §九 cost accounting.
        """
        return len(self.usage_log)

    def ask(self, extra: dict | None = None) -> dict:
        """Run one invocation in this bout's session; return the accepted receipt.

        The first call merges ``first_extras`` under ``extra`` (``extra``
        wins key collisions); later calls send only ``extra``.
        ``InvocationFailed`` propagates to the caller (arms map it to
        ArmError per their own rules).
        """
        invocation_id = self._store.next_invocation_id()
        resume = None
        if self._last_invocation_id is not None:
            resume = self._store.load_session_id(
                self.role.name, self._last_invocation_id
            )
        message_extras = dict(extra or {})
        if self._last_invocation_id is None:
            message_extras = {**self.first_extras, **message_extras}
        ctx = InvocationContext(
            task=self.task,
            tag=self.tag,
            run_dir=self.run_dir,
            invocation_id=invocation_id,
            extra=message_extras,
            resume_session_id=resume,
        )
        receipt = None
        try:
            receipt = self.runner.run(self.role, ctx)
            return receipt
        finally:
            # Accounted in finally: an InvocationFailed still burned tokens
            # (that is what the corrective follow-ups are), and dropping them
            # would make an arm that retries a lot look CHEAPER than one that
            # gets it right the first time — exactly backwards for §九.
            self.usage_log.append(self._capture_usage(invocation_id))
            if receipt is not None:
                self._last_invocation_id = invocation_id

    def totals(self) -> dict:
        """Aggregate under the runner's arm_state keys
        (arm_api.AGGREGATE_ARM_STATE_KEYS): llm_calls / llm_input_tokens /
        llm_output_tokens."""
        return {
            "llm_calls": len(self.usage_log),
            "llm_input_tokens": sum(e["input_tokens"] for e in self.usage_log),
            "llm_output_tokens": sum(e["output_tokens"] for e in self.usage_log),
        }

    def _capture_usage(self, invocation_id: int) -> dict:
        """Sum this invocation's ``session_end`` usage rows from the events log.

        The production runner emits one session_end per query turn
        (initial + corrective follow-ups), carrying the SDK ResultMessage's
        ``usage`` dict; numeric fields are summed key-wise.

        ``events_log`` marks how the entry was obtained: "read" when the log
        existed, "missing" when it did not (fake runners, or a run_dir that is
        not the one the SessionRunner writes to). Both yield zeros, and a
        genuine zero-cost invocation is indistinguishable from a misrouted
        run_dir without this flag.
        """
        entry = {
            "invocation_id": invocation_id,
            "session_ends": 0,
            "num_turns": 0,
            "total_cost_usd": 0.0,
            "usage": {},
            "input_tokens": 0,
            "output_tokens": 0,
            "events_log": "read",
        }
        path = self.run_dir / EVENTS_FILENAME
        if not path.exists():
            entry["events_log"] = "missing"
            return entry
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") != "session_end":
                continue
            if (row.get("role") != self.role.name
                    or row.get("invocation_id") != invocation_id):
                continue
            entry["session_ends"] += 1
            if isinstance(row.get("num_turns"), (int, float)):
                entry["num_turns"] += int(row["num_turns"])
            if isinstance(row.get("total_cost_usd"), (int, float)):
                entry["total_cost_usd"] += float(row["total_cost_usd"])
            usage = row.get("usage")
            if isinstance(usage, dict):
                for key, value in usage.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        entry["usage"][key] = entry["usage"].get(key, 0) + value
        entry["input_tokens"] = int(entry["usage"].get("input_tokens", 0))
        entry["output_tokens"] = int(entry["usage"].get("output_tokens", 0))
        return entry


def make_bout_session_factory(
    *,
    runner: "SessionRunner",
    run_dir,
    task: str,
    tag: str,
) -> Callable[..., BoutSession]:
    """The factory cell wiring places into ``ctx.extras`` for LLM arms.

    ``factory(role_name, first_extras=None) -> BoutSession`` — always a
    FRESH bout session (never a resume of another bout's transcript).
    ``role_name`` must be a key of BENCH_ROLES.

    INVARIANT — ``run_dir`` must belong to exactly one cell. Invocation ids
    come from scanning receipt filenames in that directory, so two cells
    sharing a run_dir can mint the same id; ``persist_session_id`` keys files
    by (role, invocation_id), and a collision would let one bout resume
    ANOTHER bout's transcript — precisely what PLAN §四 forbids. Within a cell,
    reusing one run_dir across bouts is safe (ids keep increasing).
    """

    def factory(role_name: str, first_extras: dict | None = None) -> BoutSession:
        return BoutSession(
            role=BENCH_ROLES[role_name],
            runner=runner,
            run_dir=run_dir,
            task=task,
            tag=tag,
            first_extras=first_extras,
        )

    return factory


# --- structured context builders (the §四 standard view) ----------------------


def _compact_params(params: dict) -> str:
    return json.dumps(params, separators=(",", ":"), ensure_ascii=False, default=str)


def format_search_space(contract: "_space.CandidateContract") -> str:
    """Per-dimension name, kind, bounds/options, log flag, base value, and
    fixed-vs-tunable status (degenerate single-value ranges are FIXED)."""
    lines = [
        f"Search space: {len(contract.dimensions)} dimension(s) in "
        "declaration order. Scores are lower-is-better.",
    ]
    for dim in contract.dimensions:
        base = contract.base_params.get(dim.name)
        if dim.kind == "categorical":
            desc = "categorical, options " + json.dumps(list(dim.options or ()))
            fixed = len(dim.options or ()) <= 1
        else:
            desc = f"{dim.kind} in [{dim.lo}, {dim.hi}]"
            if dim.log:
                desc += ", log-scale"
            fixed = dim.lo == dim.hi
        status = ("FIXED (degenerate single-value range; never propose a "
                  "different value)") if fixed else "tunable"
        lines.append(
            f"- {dim.name}: {desc}; base={json.dumps(base, default=str)}; {status}"
        )
    lines.append(
        "PARAM_SCHEMA (declared parameter types): "
        + json.dumps(contract.param_schema, ensure_ascii=False, default=str)
    )
    return "\n".join(lines)


def _row_params(row) -> dict:
    # state.Trial carries .config; checkpoint.HistoryRow carries .params.
    config = getattr(row, "config", None)
    if config is not None:
        return config
    return getattr(row, "params")


def _row_origin(row) -> str:
    # state.Trial carries .source; checkpoint.HistoryRow carries .origin/.role.
    origin = getattr(row, "source", None) or getattr(row, "origin", None) or "unknown"
    if getattr(row, "role", None) == "inherited_control":
        origin += " (inherited_control)"
    return origin


# Prepended to the first message's history block (prompt-v2, failure-mode
# M1/M3): the noise scale (remeasure spread of the same config is ~0.003)
# and the confounded-attribution warning. Facts only; no protocol change.
HISTORY_READING_NOTES = (
    "Reading this history: adjacent rows usually change MANY parameters at "
    "once — never attribute a row-to-row score difference to a single axis. "
    "Scores are noisy: re-evaluating the SAME config can differ by ~0.003; "
    "differences below ~0.005 carry no directional information. Treat "
    "sub-noise \"improvements\" as ties, not as gradients to follow.\n"
)


def format_history(
    trials: Iterable,
    *,
    incumbent_score_at_each_eval: float,
) -> str:
    """The structured trial view — one line per EXECUTED trial, oldest first.

    Columns: index, origin, compact params, score (or CRASH), and whether
    the trial strictly improved on the incumbent AT THE TIME. At-the-time
    flags are recomputed from the sequence: the running incumbent starts at
    ``incumbent_score_at_each_eval`` and tightens on every strict improvement.
    Crash rows never update it. Preflight-rejected rows are not executions and
    are skipped.

    The reference score is REQUIRED, with no default: it is the score a row
    had to beat AT THE TIME, and getting it wrong inverts the flag column.
    For an in-bout sequence that is the bout's starting incumbent (a trial
    that ties it did not improve). For a full checkpoint trajectory it is
    ``math.inf`` — the recorded rows start from the candidate's own tuning
    start, so replaying from +inf reconstructs the at-the-time incumbent
    sequence; the checkpoint's final incumbent would flag every row "no
    improvement" (it is the best any row achieved). ``first_message_blocks``
    applies exactly this convention.
    """
    lines = []
    incumbent = float(incumbent_score_at_each_eval)
    for row in trials:
        status = getattr(row, "status", None)
        if status not in ("ok", "crash"):
            continue
        index = len(lines) + 1
        head = f"#{index} [{_row_origin(row)}] {_compact_params(_row_params(row))}"
        score = getattr(row, "score", None)
        if status == "crash" or score is None:
            lines.append(f"{head} -> CRASH (score +inf, worst)")
            continue
        improved = score < incumbent
        if improved:
            incumbent = score
        flag = "IMPROVED (became the incumbent at the time)" if improved else "no improvement"
        lines.append(f"{head} -> score={score} {flag}")
    if not lines:
        return "(no executed trials yet)"
    return "\n".join(lines)


def _markdown_section(path: Path, heading: str) -> str | None:
    """Body of ``## <heading>`` up to the next ``## `` heading; None when the
    file or the section is missing."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    marker = f"## {heading}"
    start = next(
        (i for i, line in enumerate(lines) if line.strip() == marker), None
    )
    if start is None:
        return None
    body = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        body.append(line)
    return "\n".join(body).strip() or None


def _toml_description(path: Path) -> str | None:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    description = data.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()
    return None


def _task_card(project: str | None) -> str:
    """The one prose channel in the tuning context: the task's own
    ``## Goal`` section from ``tasks/<task>/TASK.md`` — what the score means,
    the improvement target, and fixed constraints such as the training-time
    budget. Verbatim from the task package so it cannot drift from what the
    generation layer reads. Falls back to the task.toml one-line
    description, then to an honest placeholder."""
    name = project.rsplit("/", 1)[-1] if project else ""
    header = f"task: {name or 'unknown'}"
    if name:
        goal = _markdown_section(_REPO_ROOT / "tasks" / name / "TASK.md", "Goal")
        if goal:
            return f"{header} — the task's own goal statement:\n{goal}"
        description = _toml_description(_REPO_ROOT / "tasks" / name / "task.toml")
        if description:
            return f"{header} — task.toml description: {description}"
    return f"{header} (no TASK.md goal statement available)"


def _experiment_items(checkpoint: "_checkpoint.Checkpoint") -> str:
    baseline = checkpoint.items.get("task_baseline")
    if not isinstance(baseline, dict):
        return "task_baseline: unavailable"
    metric = baseline.get("metric", "score")
    value = float(baseline["value"])
    lines = [
        "Frozen run-global observations (read-only):",
        f"task_baseline.metric: {metric}",
        f"task_baseline.value: {value}",
        f"task_baseline.direction: {baseline.get('direction', 'minimize')}",
    ]
    relative = checkpoint.task.relative_improvement_over_baseline
    if relative is not None:
        target = value * (1.0 - relative)
        lines.extend([
            f"required_relative_improvement: {relative}",
            f"required_target_score: {target}",
        ])
    return "\n".join(lines)


def first_message_blocks(
    checkpoint: "_checkpoint.Checkpoint",
    contract: "_space.CandidateContract",
    *,
    protocol: str,
    budget_remaining: int,
    trials: Iterable | None = None,
    live_incumbent: tuple[dict, float] | None = None,
    evidence: list | None = None,
    candidate_kind: str | None = None,
) -> dict:
    """Assemble the §四 first-call checklist as invocation extras (str values).

    Keys: task (the task's own ``## Goal`` statement — the one prose
    channel), items (frozen run-global observations and derived target),
    search_space (+ parameter semantics via PARAM_SCHEMA), candidate
    (kind fresh/improve/crossover/provided-baseline when known, regime/
    stratum, inherited-control flag), incumbent (params + score), history
    (executed trials; defaults to the checkpoint's history), protocol (the
    arm's own protocol description slot), budget (remaining objective
    evaluations). ``live_incumbent`` overrides the frozen checkpoint
    incumbent for a replayed or later in-bout factual snapshot. ``evidence``:
    extra read-only text blocks appended under
    one ``evidence`` key when provided — currently restricted to
    same-candidate history (see module docstring).

    Candidate kind comes from the checkpoint object when recorded
    (``source["kind"]``, written by freeze; ``extra["kind"]`` as a fallback),
    or from an explicit ``candidate_kind`` override. Anything else renders
    honestly as "unknown".
    """
    source = checkpoint.source or {}
    kind = (
        candidate_kind
        or source.get("kind")
        or checkpoint.extra.get("kind")
        or "unknown"
    )
    candidate = "\n".join([
        f"checkpoint_id: {checkpoint.checkpoint_id}",
        f"regime: {checkpoint.regime} (first | continuation | deep)",
        f"stratum: {checkpoint.stratum}",
        f"candidate kind: {kind} (fresh | improve | crossover | "
        "provided-baseline; 'unknown' = not recorded at freeze time)",
        f"source candidate_id: {source.get('candidate_id', 'unknown')}",
        f"incumbent is the inherited control: "
        f"{checkpoint.incumbent_is_inherited_control}",
    ])
    if live_incumbent is None:
        incumbent_params = checkpoint.incumbent.params
        incumbent_score = checkpoint.incumbent.score
        incumbent_heading = "Checkpoint incumbent"
    else:
        incumbent_params, incumbent_score = live_incumbent
        incumbent_heading = "Current incumbent"
    incumbent = "\n".join([
        f"{incumbent_heading} — the config to beat; only a strictly lower "
        "score improves it:",
        f"params: {_compact_params(incumbent_params)}",
        f"score: {incumbent_score}",
    ])
    blocks = {
        "task": _task_card(checkpoint.task.project),
        "items": _experiment_items(checkpoint),
        "search_space": format_search_space(contract),
        "candidate": candidate,
        "incumbent": incumbent,
        "history": HISTORY_READING_NOTES
        + format_history(
            checkpoint.history if trials is None else trials,
            # Replay from the trajectory's own start: the recorded history
            # begins at the candidate's own tuning start, so a +inf
            # reference reconstructs the at-the-time incumbent sequence
            # (PLAN §四: 相对当时 incumbent 的结果). The checkpoint incumbent
            # is the trajectory's FINAL best — using it as the reference
            # would flag every historical row "no improvement", exactly
            # backwards. The first row's IMPROVED flag is the trivial
            # best-so-far seeding.
            incumbent_score_at_each_eval=math.inf,
        ),
        "protocol": protocol,
        "budget": (
            f"Remaining objective-evaluation budget for this bout: "
            f"{budget_remaining}. Each executed proposal consumes exactly 1; "
            "preflight-rejected proposals consume none."
        ),
    }
    if evidence:
        blocks[EVIDENCE_KEY] = "\n\n---\n\n".join(str(block) for block in evidence)
    return blocks


def outcome_message(
    params: dict,
    *,
    status: str,
    score: float | None = None,
    incumbent_score: float,
    budget_remaining: int,
) -> str:
    """Per-evaluation append (PLAN §四: only the authoritative result is
    appended to the bout session): config, score/crash, strict
    improvement-vs-current-incumbent, remaining budget."""
    if status == "crash":
        result = "CRASH (score = +inf, the worst outcome; the budget was still consumed)"
        verdict = "did NOT improve the current incumbent"
    elif status == "ok":
        if score is None or not math.isfinite(score):
            raise ValueError(f"ok outcomes need a finite score, got {score!r}")
        improved = score < incumbent_score
        verdict = ("IMPROVED the current incumbent (strictly lower score)"
                   if improved else "did NOT improve the current incumbent")
        result = f"score = {score}"
    else:
        raise ValueError(f"unknown outcome status {status!r}; expected 'ok' or 'crash'")
    return "\n".join([
        f"config: {_compact_params(params)}",
        f"result: {result}",
        f"verdict: {verdict} (current incumbent score = {incumbent_score})",
        f"remaining budget: {budget_remaining} objective evaluations",
    ])


__all__ = [
    "BENCH_ROLES",
    "BLOCK_KEYS",
    "BoutSession",
    "EVIDENCE_KEY",
    "FEASIBLE_SET_KEY",
    "LLMConfig",
    "OUTCOME_KEY",
    "TASK_KEY",
    "WORKING_COPY_KEY",
    "detect_sdk_version",
    "first_message_blocks",
    "format_history",
    "format_search_space",
    "make_bout_session_factory",
    "outcome_message",
]
