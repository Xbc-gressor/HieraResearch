# Live harness observability

`tools/harness_watch.py` attributes Claude Code transcript usage and projects
run lifecycle state from durable artifacts without changing a run. It has no
external service dependency.

## Claude Code transcripts and status lines

For historical or full-session attribution, point the monitor at a main Claude
Code JSONL transcript:

```bash
python tools/harness_watch.py \
  --transcript ~/.claude/projects/<project>/<session>.jsonl \
  --run-dir runs/<task>/<tag>
```

The monitor deduplicates streamed assistant updates by message id and includes
discoverable subagent transcript directories. Use `--watch 2` to refresh every
two seconds, `--json` for machine-readable snapshots, and
`--max-session-input` to change the default 50k fresh-input alert threshold for
child transcripts. A one-shot report exits with status 2 when it contains an
alert, which is intentional for shell and CI checks.

The display separates fresh input, cache reads/writes, output, reasoning,
processed input, recorded cost, run evaluations, pending candidates, and
alerts. `.claude/settings.json` also installs local main and subagent status
lines plus a `PreToolUse` delegation guard. These status lines consume Claude
Code's status payload and do not invoke a model.

The supported `hieraresearch` path does not delegate lifecycle control to
interactive child agents. Candidate briefs are materialized by deterministic
Python, and bounded Agent SDK calls enforce their own tool and path policies in
`ModelGateway`.

## Artifact-only fallback

When a transcript is unavailable, lifecycle checks still
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
