---
name: idea-proposer
description: Use at step 2 of the experiment loop in program.md, when the next candidate's experimental idea has not yet been chosen. This skill stays in the main Claude's context so signals from the ongoing conversation (user-directed pivots, recent observations not yet in the ledger) are preserved. The output is a single short proposal that the caller hands to the candidate-writer subagent for implementation. The skill enforces outer-loop diversity by reading idea_log.md and forcing each new idea onto a primary axis distinct from the recent ones.
metadata:
  short-description: Propose one diverse autoresearch candidate idea
---

# Idea Proposer

Methodology for choosing **one** experimental idea before each new candidate.
Stays in the main Claude's context — do not spawn a subagent for this step.
Conversation history matters here: the user's recent pivots, the things they
have shown interest in or pushed back on, and observations the main Claude
made while reading the last few logs all feed in.

This skill drives the **outer loop** of a two-layer search: each idea opens
up a different region of the task's search space, and the `tuner-orchestrator`
agent refines within whatever region the idea picked. Outer loop optimizes
for **coverage**; inner loop optimizes for **depth**. The two together avoid
both clustering on one direction and skipping past a direction that needed
better hyperparameters before being judged.

The skill is task-agnostic. Concrete axis names, heuristic priorities, and
search directions are owned by each task's `TASK.md`; this skill only
enforces the diversity rule and the proposal format.

## Inputs To Read

Before proposing, read these task-local artifacts:

1. **`runs/<task>/<tag>/idea_log.md`** — required. The structured record of
   prior ideas and their primary axes. Read this first; it drives the
   diversity constraint below. If the file does not exist yet (this is the
   first non-baseline run), treat the recent-axes set as empty.
2. `runs/<task>/<tag>/results.tsv` — the full ledger. If the task's parser
   records per-component scores (per-dataset, per-subtask, per-suite), note
   which component is currently dragging the headline metric.
3. `runs/<task>/<tag>/loop_state.md` — the current best candidate id, last
   status, and the one-line search direction.
4. The current best candidate's editable file(s) — what we are trying to beat.
5. The most recent 1–2 candidates' editable files (kept or discarded) —
   what was just tried.
6. `tasks/<task-name>/TASK.md` — the task's high-level scope and any
   suggested starting directions. Suggested directions are an **invitation,
   not a whitelist**: ideas outside the list are welcome as long as they
   fit the task's domain. If the task documents an axis taxonomy, it lives
   here.
7. `tasks/<task-name>/task.toml` — `result.metric`, `result.lower_is_better`,
   and the constraint sections. Together with `TASK.md`'s high-level scope
   these are the **only hard boundaries**.

Skip reading anything else. The agent that implements the idea will read the
readonly files and the source editable file itself.

## Diversity Constraint (Hard)

Every new idea must have a **primary axis** distinct from the last
`min(10, len(idea_log))` entries.

The primary axis is the dimension along which this experiment varies. It is
**task-defined**:

- If `TASK.md` documents an axis taxonomy, pick one tag from there.
- If not, invent a concise `lowercase_with_underscores` tag that names the
  dimension (e.g. `optimizer_swap`, `attention_variant`, `feature_pipeline`,
  `loss_function_change`) and **reuse the exact tag** when a future idea
  exercises the same dimension. The skill checks textual match against
  `idea_log.md`; inconsistent tagging defeats the diversity rule.

Same-axis hyperparameter refinements are the **inner loop's** job (handled
by `tuner-orchestrator`), not the outer loop's — do not spend an outer-loop
slot on them. If the heuristics below produced a candidate idea that lands
on a recently-used axis, replace it with one that lands on an unused axis.

When `idea_log.md` has fewer than 10 entries, constrain only against the
entries that exist.

## Heuristics For Picking An Axis

Use these to decide *which* unused axis to pick. They are tie-breakers
under the diversity constraint above, not standalone rules.

1. **Untried region of the task's domain.** Pick a direction that fits the
   task's high-level scope (and respects `task.toml`) and has not yet
   appeared in the ledger. `TASK.md`'s suggested directions are a starting
   set; directions outside that list are encouraged when they make sense
   for the task and have not been explored. The point of an outer-loop
   search is breadth — do not constrain yourself to the literal list.
2. **Component-specific gap.** If the task reports per-component scores and
   one component is dragging the headline metric, prefer an axis whose
   typical strengths target that component's failure mode.
3. **Combine two near-misses.** If two recent `discard` candidates scored
   decently in complementary regimes, a combination/ensemble move can be a
   strong choice — but only if the combination axis is itself unused.
4. **Simplify.** If the current best is complex and a much simpler axis has
   not been tried, propose the simplification.

Hard rules:

- **One idea per candidate.** No bundling.
- **No re-runs of discarded ideas verbatim.** A retry only counts if the
  axis is the same but the inner-loop hyperparameter range fundamentally
  differs — and even then, prefer letting `tuner-orchestrator` re-explore
  rather than spending an outer-loop slot.
- **Respect `task.toml`.** No proposals requiring readonly edits, metric
  changes, or violations of the task's `constraints` section.
- **Allow dependencies only when permitted.** If
  `constraints.allow_dependencies = false`, do not propose adding new
  packages.

## Output Format

Produce a short proposal block in this exact shape:

```text
idea:                <one sentence: what to change, in concrete terms>
primary_axis:        <a concise lowercase_with_underscores tag from TASK.md
                      or invented and reused consistently>
rationale:           <one or two sentences: which ledger/conversation signal
                      drove this, and why this axis was unused or stale>
scope:               <what files/areas of the editable file(s) the
                      implementation should touch>
risks:               <one short line; "none notable" is allowed>
candidate_name_hint: <lowercase_with_underscores, identifies the experiment>
tune:                true
```

`tune` is a boolean signalling whether the candidate should be tuned. It
is currently hard-locked to `true` for every idea — the two-layer split
(this skill for outer-loop diversity, the `tuner-orchestrator` agent for
inner-loop depth) means even a structural idea benefits from one round of
tuning before being judged. Method selection (LLM warm-start / grid / BO /
CMA-ES) belongs to the orchestrator, not this skill.

## Append To idea_log.md (Required)

After producing the block, append this entry to
`runs/<task>/<tag>/idea_log.md` (create the file if missing):

```text
## run <next_run_id>
- idea:         <copy from block>
- primary_axis: <copy from block>
- tune:         true
- candidate_name_hint: <copy from block>
```

This file is the **only** authoritative record of axis history; the next
round depends on it. Do not skip the append, and do not edit prior entries.

After Phase C the `tuner-orchestrator` agent will extend this same entry
with additional fields (`baseline_score`, `best_warm_score`,
`final_best_score`, `n_dims`, `warm_start_K`, `warm_percentile`,
`phase_b_decision`, `phase_c_method`, `trials_completed`,
`elapsed_seconds`, `applied`). Do not anticipate or pre-fill those — they
are the orchestrator's responsibility.

## Boundaries

- Do not write code. The proposal is text only.
- Do not create the candidate directory or copy files. That is the caller's
  job before invoking the `candidate-writer` agent.
- Do not start the experiment. Hand the proposal back to the caller.
- Do not skip the diversity constraint to fit a heuristic — pick a
  different axis instead.
