# Live harness observability

`tools/harness_watch.py` attributes token use and exposes hidden delegation and
run-lifecycle drift without changing a run. It has no external service
dependency.

## OpenCode

Run from the OpenCode repository root:

```bash
python tools/harness_watch.py \
  --run-dir runs/<task>/<tag> --watch 2
```

The default source is OpenCode's local SQLite store, opened read-only. The
monitor selects the latest root session associated with the run and traverses
all descendants. If multiple runs share a project, pin one with
`--session <root-or-child-session-id>`. Use `--json` for machine-readable
snapshots and `--all-sessions` for the complete tree.

The display separates fresh input, cache reads/writes, output, reasoning,
processed input, and recorded cost. It also shows per-agent totals, current
tools, latest context size, run evaluations, pending candidates, and alerts.
Exit status is 2 when a one-shot report contains alerts; this is intentional for
CI or shell checks.

Default alerts cover:

- candidate-writer assignments that include contract/evaluation work;
- recursive or unexpected delegation;
- root context at or above 150k tokens;
- child fresh input at or above 50k tokens;
- an idle root while the run remains active;
- pending candidates without an active worker.

Tune thresholds with `--max-root-context` and `--max-session-input`.

The project-local `.opencode/plugins/hiera-guard.js` adds enforcement:

- `tool.execute.before` rejects the observed writer/step-0+1 role collapse;
- `tool.execute.after` replaces relevant child output with a validated receipt
  before parent reinjection, regardless of whether the child followed its
  output prompt;
- `session.idle` warns in the TUI when the run still has work remaining.

Child agents also have `task: deny`, so they cannot recursively delegate. The
candidate writer additionally has no shell tool; `new_candidate.py` materializes
its narrow `_candidate_brief.json` before launch.

## Claude Code

`.claude/settings.json` installs a main status line, subagent status lines, a
PreToolUse delegation guard, and a runtime Bash guard. The Bash guard denies
direct `train.py` execution and auto-approves only named HieraResearch
control-plane scripts; arbitrary Python is forced back to an explicit approval
even if a broad interpreter allow rule exists. The background researcher adds a
lifecycle-scoped hook that narrows Bash further to retrieval/validation commands
and denies raw network or machine inspection. The status line is local and does
not invoke a model. For historical
or full-session attribution, point the monitor at a main JSONL transcript:

```bash
python tools/harness_watch.py --source claude \
  --transcript ~/.claude/projects/<project>/<session>.jsonl \
  --run-dir runs/<task>/<tag>
```

The monitor deduplicates streamed assistant updates by message id and includes
discoverable subagent transcript directories. Claude's main status-line payload
provides current context and cost; subagent status lines expose each task's
reported token count and flag 50k or more. Historical Claude attribution is less
complete than OpenCode when transcripts have been removed or moved.

Claude hooks can deny a bad delegation before it starts, but do not offer the
same reliable post-tool result rewriting used by the OpenCode plugin. Compact
child receipts therefore remain prompt-plus-schema discipline on Claude, while
the independent monitor exposes violations after the fact.

## Artifact-only fallback

When a session database or transcript is unavailable, lifecycle checks still
derive from `ledger.json`, `framework_cfg.json`, and `loop_state.md`:

```bash
python tools/ledger.py brief --ledger runs/<task>/<tag>/ledger.json
python tools/harness_watch.py --check-run-idle --run-dir runs/<task>/<tag>
```

`ledger.py set-phase` is the only supported lifecycle transition. It refuses
premature completion and requires a concrete reason for blocked runs:

```bash
python tools/ledger.py set-phase --ledger runs/<task>/<tag>/ledger.json \
  --phase blocked --stop-condition "<reason>"
```

An explicit evaluation budget passed at completion is persisted in the ledger,
so later monitors can still derive the correct phase without the original
conversation.
