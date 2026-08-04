# Live harness observability

`tools/harness_watch.py` attributes token use and exposes hidden delegation and
run-lifecycle drift without changing a run. It has no external service
dependency.

## Claude Code

`.claude/settings.json` installs a main status line, subagent status lines, and a
PreToolUse delegation guard. The status line is local and does not invoke a
model. For historical or full-session attribution, point the monitor at a main
JSONL transcript:

```bash
python tools/harness_watch.py --source claude \
  --transcript ~/.claude/projects/<project>/<session>.jsonl \
  --run-dir runs/<task>/<tag>
```

The monitor deduplicates streamed assistant updates by message id and includes
discoverable subagent transcript directories. Claude's main status-line payload
provides current context and cost; subagent status lines expose each task's
reported token count and flag 50k or more. Historical attribution is incomplete
when transcripts have been removed or moved.

Claude hooks can deny a bad delegation before it starts, but do not rewrite
post-tool results. Compact child receipts therefore remain prompt-plus-schema
discipline, while the independent monitor exposes violations after the fact.

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

## Deprecated: OpenCode

The `--source opencode` path still exists in `tools/harness_watch.py` but is
unmaintained, as are the `.opencode/` and `.kimi/` runtime mirrors. It reads
OpenCode's local SQLite store read-only, selects the latest root session for the
run, and traverses all descendants; `--session <root-or-child-session-id>` pins
one when several runs share a project. Keep it only for reading old sessions.

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
