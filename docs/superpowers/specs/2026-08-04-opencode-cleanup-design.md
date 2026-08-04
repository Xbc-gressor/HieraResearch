# OpenCode Cleanup Design

## Goal

Remove OpenCode as a supported HieraResearch runtime while preserving the
Python-owned coordinator, the Claude Agent SDK and Messages API boundaries, the
Claude Code hillclimb comparison baseline, and useful Claude/artifact
observability.

The cleanup must remove executable OpenCode support rather than only deleting
visible names. It must not rewrite completed or in-progress run artifacts.

## Non-goals

- Do not change the `hieraresearch` lifecycle, model boundaries, objective
  accounting, or semantic-selection policy.
- Do not remove the Claude Code `autoresearch-hillclimb` comparison baseline.
- Do not remove Claude transcript reporting, status lines, or artifact-derived
  run-idle checks.
- Do not add compatibility shims for removed OpenCode commands or flags.
- Do not edit anything under `runs/`.

## Repository surface

Delete the OpenCode-owned configuration and runtime files:

- `opencode.json`
- `.opencode/agents/autoresearch-hillclimb.md`
- `.opencode/plugins/hiera-guard.js`
- `.opencode/rules/ledger.md`

The Claude hillclimb agent and Claude project settings remain supported.

## Monitoring and guard behavior

`tools/harness_watch.py` becomes Claude-and-artifact-only:

- remove the OpenCode SQLite dependency and reader;
- remove OpenCode session discovery, tree traversal, role alerts, and rendering;
- remove `--source`, `--project-dir`, `--session`, `--db`,
  `--all-sessions`, and `--max-root-context`;
- require `--transcript` for a full transcript report;
- preserve `--run-dir`, `--watch`, `--json`, `--max-session-input`, the Claude
  status-line modes, and `--check-run-idle`;
- keep artifact snapshots available to Claude reports and idle checks.

`tools/harness_guard.py` keeps its Claude `PreToolUse` entrypoint. Remove the
OpenCode-only `--opencode-check` interface and the receipt-compaction interface
used only by the deleted OpenCode plugin. Delete implementation and tests that
have no remaining caller. The remaining Claude hook continues to reject the
specific role-collapse assignment it already guards.

Removed command-line flags fail through normal `argparse` unknown-argument
handling. No deprecated aliases or silent fallbacks are introduced.

## Active background instructions

Rewrite `docs/agent-resources/background-researcher/retrieval.md` for the sole
live SDK runtime:

- refer to `WebSearch` and `WebFetch` only;
- record external fetched evidence as `claude-webfetch` only;
- remove the `opencode-webfetch` alternative.

This file is part of the background author's declared immutable inputs. Its
content change therefore changes the input revision. Existing frozen runs with
valid background artifacts remain authoritative and are not rewritten.
Incomplete authoring state bound to the previous revision must be surfaced as
stale or invalid by the existing revision checks, not silently restamped.

## Documentation

Rewrite `docs/observability.md` around the supported paths:

- Claude transcript reports;
- Claude main and subagent status lines;
- artifact-only run-idle checks.

Remove OpenCode setup, SQLite, plugin, and cross-runtime comparison text. Keep
the explanation that the Python coordinator does not delegate lifecycle
control to interactive agents.

## Tests and verification

Adjust `tests/test_harness_controls.py` to remove coverage for deleted OpenCode
and receipt-compaction interfaces while retaining distinct coverage for:

- the Claude delegation hook;
- Claude transcript usage deduplication;
- run snapshot and budget projection;
- artifact-only idle checks and status-line behavior where already covered.

Do not add a permanent forbidden-wording test. Use a repository search as a
one-time cleanup verification instead.

Verification sequence:

1. Run the targeted harness-control tests.
2. Run `python tools/validate_background.py` because an active background
   instruction resource changes.
3. Run `python -m pytest tests -q` at the integration point.
4. Search tracked files for `opencode` and `open code`, excluding this design
   record and its implementation plan. There must be no remaining matches in
   supported source, tools, tests, runtime configuration, or user-facing
   documentation.
5. Inspect `git diff` and `git status` to ensure unrelated user changes and
   `runs/` artifacts were not modified.

## Expected result

`python -m hieraresearch` remains unchanged as the sole run-level coordinator.
The repository retains its Claude hillclimb comparison and Claude/artifact
observability, but no longer advertises, configures, parses, or documents an
OpenCode runtime.
