# autoresearch-automl

Multi-task autonomous experimentation harness. The supported interactive
runtimes are Claude Code (`.claude/`) and OpenCode (`.opencode/`); deterministic
state, graph search, evaluation, and tuning live in `tools/` and are shared.

The loop's current bar is beating `autoresearch-hillclimb` — the deliberately
simple edit→run→keep/revert baseline — at matched evaluation budget. It does not
yet. Until it does, prefer diagnosing current run artifacts and simplifying the
loop over extending it.

## Start an experiment with OpenCode

Run from this repository root:

```bash
opencode --agent autoresearch-experiment \
  --model moonshotai/kimi-k3 --auto
```

Then provide `task_name`, `tag`, and optionally `max_evaluations` and
`timeout` (the hard limit in seconds for each evaluation). For a non-interactive
session:

```bash
opencode run --agent autoresearch-experiment \
  --model moonshotai/kimi-k3 --auto \
  "task_name=<task> tag=<tag> max_evaluations=<n> timeout=<seconds>"
```

`autoresearch-hillclimb` is the deliberately simple comparison baseline and is
started with the same commands using `--agent autoresearch-hillclimb`.

The same controls can be set or changed deterministically before a run:

```bash
python tools/init_run.py <task> <tag> \
  --max-evaluations <n> --timeout <seconds>
```

They are persisted as `max_evaluations` and `per_runtime_limit` in the run's
`framework_cfg.json`; explicit initialization values override the copied
template.

OpenCode primary and subagents inherit `moonshotai/kimi-k3` from the launch
command. `--auto` approves permission requests that are not explicitly denied;
the project agents still enforce their hard role boundaries. Every experiment
agent explicitly allows doom-loop recovery so an unattended run does not pause
for that prompt.

## Authoritative inputs

Before experiment work, read:

1. `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml`.
2. `.opencode/rules/ledger.md` only when ledger schema detail is needed.

OpenCode injects `AGENTS.md` and the selected agent prompt automatically. Do
not read either one again from inside the agent session.

The ledger contract is intentionally not injected globally through
`opencode.json`; most role agents need only a narrow helper-rendered view.

Read on demand, not by default: `docs/search-space.md` (the formal model the P2
helpers implement), `docs/background-research.md` (search-space contract,
evidence and scope semantics), `docs/dimension-induction.md` (only for the
`llm_induced` strategy), and `docs/observability.md` (`harness_watch.py`).

## Runtime layout

```text
.opencode/agents/                 project-local primary/subagents
.opencode/skills/                 inline capability skills
.opencode/rules/                  on-demand state contracts
.opencode/plugins/hiera-guard.js  delegation and receipt guard
contracts/                        versioned shared contracts (dimension catalog)
docs/                             search-space, background, observability notes
tools/                            shared deterministic machinery
tests/                            pytest suite over tools/; tests/fixtures.py
                                  holds the shared toy search space
tasks/<task-name>/                independent uv task projects
runs/<task-name>/<tag>/           local experiment artifacts (gitignored)
```

`.claude/` and `.kimi/` mirror `.opencode/` for other runtimes, but
`.opencode/` is canonical and the mirrors may drift.

The experiment primary agent may invoke exactly these six subagents through
OpenCode's `Task` tool:

| subagent | role | cadence |
|---|---|---|
| `background-researcher` | freezes `background.md` + `background_retrieval.json`: the run's semantic search space | once, before the loop (required) |
| `idea-generator` | graph `SELECT` via `got_select`, then semantic point choice and record/receipt persistence | per round |
| `candidate-writer` | implements one candidate's `train.py` from its own ledger record | per candidate |
| `tunable-contract-extractor` | step 0+1: `PARAM_SCHEMA` refactor, warm configs, `SEARCH_SPACE`, screening evaluation | per candidate |
| `tuner-orchestrator` | step 2: progressive tuning gate, then at most one tuning bout (first or continuation) in place | once per round |
| `experience-extractor` | regenerates the bounded belief snapshot and requests state transitions | per completed non-empty round |

Each agent's prompt is authoritative for its own contract. The allow-list is
encoded in the primary agent's native `permission.task` map. Every child has
`task: deny`; `candidate-writer` also has `bash: deny`.
`.opencode/plugins/hiera-guard.js` rejects the known writer/evaluation boundary
collapse and replaces rich child output with compact receipts before it returns
to the primary context.

Step 2 is decoupled from step 0+1 (design §15): every candidate stops at step
0+1, then `tuner-orchestrator` runs once for the whole round and runs at most
one tuning bout — a first bout on a gated untuned candidate or a continuation
bout on a responder. A `none` selection is a valid no-op.

## Context and state discipline

- Pass paths and compact identifiers to child agents. Durable artifacts are the
  payload; child responses are receipts, not copies of code, logs, or ledgers.
- Never hand-edit `runs/**/ledger.json`; use `tools/ledger.py`.
- Experience refresh reads `got_graph.py render --incremental` with fixed
  Top/Bottom anchors. Do not inject the unbounded full ledger or global DAG.
- Dimension/hypothesis beliefs cite `background_contract.py target-evidence`
  only; `ledger.py apply-space-state` owns every append-only
  `search_space_state` transition. The frozen registry never carries runtime
  pruning state.
- Retrieve a full record, source, or log only when a compact view identifies a
  specific missing field or bottleneck.
- Do not collapse role boundaries to save time. An evaluation budget does not
  authorize combining writer, evaluation, or tuning contexts.
- Run-level and task-owned candidate preflights are no-score engineering
  checks. Their failures are diagnosed inline but never counted as objective
  evaluations.
- `evaluation_attempts.jsonl` is the strict objective-call admission log. A
  tuner must reserve there immediately before `score_fn`; aggregate ledger
  fields remain per-candidate summaries.
- The outer loop searches semantic candidates; step 0+1 / step 2 tunes numeric
  parameters inside one candidate. Keep those search levels distinct.

## Task and run boundaries

- `tasks/<task>/prepare.py` is the fixed evaluation surface.
- A candidate entrypoint declared in `[seed].provided` is copied into run `000`
  and evaluated first at the all-baselines point; seedless tasks bootstrap with
  normal `fresh` candidates.
- Experiments modify candidate copies under `runs/`, never task-source
  `train.py` in place.
- Do not commit anything under `runs/`.
- Add dependencies only when `constraints.allow_dependencies = true`.
- Each task is its own uv project; use `uv --directory tasks/<task> ...`.

## Narrow checks

```bash
python -m pytest tests -q             # the suite; fast, no GPU, no network
python tools/validate_tasks.py        # task contracts
python tools/validate_background.py   # background round trip, shape
                                      # neutrality, retrieval, lifecycle
python tools/validate_got.py          # graph/ledger invariants
python tools/validate_search_backends.py
```

Add only the check implied by the touched contract. Do not add required-wording
or forbidden-wording checks over agent prompts, rules, or docs. OpenCode runtime
wiring is checked with the real CLI (`opencode debug agent <name>` and
`opencode agent list`) rather than another repository-specific validator.
