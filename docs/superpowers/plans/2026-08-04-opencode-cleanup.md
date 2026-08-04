# OpenCode Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove OpenCode runtime support while preserving the Python coordinator, Claude hillclimb baseline, Claude transcript/status reporting, and artifact-only lifecycle checks.

**Architecture:** Delete the OpenCode-owned configuration surface, then simplify the shared harness utilities to their remaining Claude and artifact responsibilities. Update the active background instruction resource and user-facing observability documentation without changing coordinator code or durable run artifacts.

**Tech Stack:** Python 3.10+, `argparse`, `unittest`/pytest, Claude Code project settings, Markdown documentation.

## Global Constraints

- Do not change the `hieraresearch` lifecycle, model boundaries, objective accounting, or semantic-selection policy.
- Preserve `.claude/agents/autoresearch-hillclimb.md` and the Claude status-line and transcript workflows.
- Do not edit `runs/` or restamp durable revisions.
- Do not add compatibility shims for removed flags.
- Preserve unrelated working-tree changes, including current edits under `src/hieraresearch/`, `tests/`, and `tools/tuners/`.
- Do not stage or commit; repository policy requires separate explicit authorization.

---

### Task 1: Remove OpenCode interfaces from harness utilities

**Files:**
- Modify: `tests/test_harness_controls.py`
- Modify: `tools/harness_watch.py`
- Modify: `tools/harness_guard.py`

**Interfaces:**
- Consumes: Claude transcript JSONL paths, optional run directories, Claude hook JSON on stdin.
- Produces: `claude_report(transcript: Path, run_dir: Path | None, max_session_input: int) -> dict[str, Any]`, status-line output, `check_run_idle(run_dir: Path) -> int`, and the existing Claude `PreToolUse` hook behavior.

- [ ] **Step 1: Add failing CLI-removal tests**

Add these tests to `DelegationGuardTests` and `UsageAndLifecycleTests`:

```python
def test_rejects_removed_opencode_guard_flag(self) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "harness_guard.py"),
         "--opencode-check", "candidate-writer"],
        input="Write train.py only.", capture_output=True, text=True,
    )
    self.assertEqual(result.returncode, 2)
    self.assertIn("unrecognized arguments", result.stderr)

def test_rejects_removed_opencode_watch_source(self) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "harness_watch.py"),
         "--source", "opencode"],
        capture_output=True, text=True,
    )
    self.assertEqual(result.returncode, 2)
    self.assertIn("unrecognized arguments", result.stderr)
```

- [ ] **Step 2: Run the new tests and confirm they fail for the old interfaces**

Run:

```bash
python -m pytest \
  tests/test_harness_controls.py::DelegationGuardTests::test_rejects_removed_opencode_guard_flag \
  tests/test_harness_controls.py::UsageAndLifecycleTests::test_rejects_removed_opencode_watch_source -q
```

Expected: both tests fail because the current utilities still accept the removed OpenCode interfaces or enter the OpenCode path.

- [ ] **Step 3: Simplify `tools/harness_guard.py`**

Make the module Claude-only:

```python
#!/usr/bin/env python3
"""Deterministic delegation boundary checks for Claude agent calls."""
```

Delete `RECEIPT_FIELDS`, `RECEIPT_ENUMS`, `_field_name`, `compact_task_result`, the `--compact-result` branch, and the `--opencode-check` branch. Keep `delegation_violation`, `_claude_hook`, and a no-option `main()` that uses `argparse.ArgumentParser(description=__doc__).parse_args()` before calling `_claude_hook()`. Remove imports no longer used after those deletions.

- [ ] **Step 4: Simplify `tools/harness_watch.py`**

Delete the `sqlite3` and `os` imports, `KNOWN_AGENTS`, `_default_opencode_db`, and the entire `OpenCodeReader` class. Keep `_run_snapshot`, Claude transcript aggregation, report printing, status lines, and idle checking.

Change `build_parser()` so its report-related public options are exactly:

```python
parser.add_argument("--run-dir", type=Path)
parser.add_argument("--transcript", type=Path, help="Claude Code main transcript JSONL")
parser.add_argument("--watch", nargs="?", const=2.0, type=float, help="Refresh every N seconds")
parser.add_argument("--json", action="store_true")
parser.add_argument("--max-session-input", type=int, default=50_000)
```

Retain the three suppressed Claude/status flags. In `main()`, require `--transcript` after handling status-line and idle-check modes, then call `claude_report(...)` directly. Simplify `_print_report` to the Claude transcript rendering branch and remove the `all_sessions` parameter.

- [ ] **Step 5: Remove tests for deleted receipt-compaction behavior**

Delete every `DelegationGuardTests` method that calls `harness_guard.compact_task_result`. Retain the two `delegation_violation` tests, the new removed-flag test, Claude transcript tests, run snapshot tests, and other unrelated harness-control coverage.

- [ ] **Step 6: Run the harness-control test file**

Run:

```bash
python -m pytest tests/test_harness_controls.py -q
```

Expected: all remaining tests pass.

- [ ] **Step 7: Inspect the task diff without staging or committing**

Run:

```bash
git diff -- tools/harness_guard.py tools/harness_watch.py tests/test_harness_controls.py
```

Expected: only OpenCode and callerless receipt-compaction code is removed; Claude and artifact behavior remains.

---

### Task 2: Remove runtime configuration and update supported documentation

**Files:**
- Delete: `opencode.json`
- Delete: `.opencode/agents/autoresearch-hillclimb.md`
- Delete: `.opencode/plugins/hiera-guard.js`
- Delete: `.opencode/rules/ledger.md`
- Modify: `docs/agent-resources/background-researcher/retrieval.md`
- Modify: `docs/observability.md`

**Interfaces:**
- Consumes: Claude SDK tools `WebSearch` and `WebFetch`; Claude transcript paths; run artifact paths.
- Produces: Claude-only background retrieval instructions and observability commands.

- [ ] **Step 1: Record the pre-cleanup search evidence**

Run:

```bash
rg -n -i --hidden --glob '!.git/**' --glob '!runs/**' \
  --glob '!docs/superpowers/specs/2026-08-04-opencode-cleanup-design.md' \
  --glob '!docs/superpowers/plans/2026-08-04-opencode-cleanup.md' \
  'opencode|open[ _-]?code' .
```

Expected: matches in the four runtime/configuration files, the two harness utilities, the retrieval resource, and observability documentation.

- [ ] **Step 2: Delete the OpenCode-owned runtime surface**

Delete the four listed tracked files. Do not delete or modify their Claude counterparts.

- [ ] **Step 3: Make the active retrieval instructions Claude-only**

Replace the runtime fallback section with instructions that name only `WebSearch`, `WebFetch`, and this backend value:

```bash
python tools/search_backends.py record-visit \
  --manifest <run_dir>/background_retrieval.json --lane grounding \
  --backend claude-webfetch --view page --status success \
  --content-file <temporary-fetched-content> --url <url>
```

Keep the existing rules about retaining exact fetched content, recording failures, and not treating search snippets as evidence.

- [ ] **Step 4: Rewrite observability documentation around supported paths**

Start with the Claude transcript command:

```bash
python tools/harness_watch.py \
  --transcript ~/.claude/projects/<project>/<session>.jsonl \
  --run-dir runs/<task>/<tag>
```

Document `--watch`, `--json`, and `--max-session-input`; retain the Claude status-line explanation and the artifact-only `ledger.py brief` and `--check-run-idle` commands. Remove SQLite, OpenCode plugin, session-tree, and cross-runtime comparison claims.

- [ ] **Step 5: Run the targeted background validator**

Run:

```bash
python tools/validate_background.py
```

Expected: exit 0 with the background contract checks passing.

- [ ] **Step 6: Verify supported files no longer reference OpenCode**

Run the search from Step 1 again.

Expected: no matches outside the excluded design and implementation records.

- [ ] **Step 7: Inspect the task diff without staging or committing**

Run:

```bash
git diff -- .opencode opencode.json docs/agent-resources/background-researcher/retrieval.md docs/observability.md
```

Expected: the OpenCode surface is deleted and the two retained documents are Claude-only.

---

### Task 3: Integration verification and ownership audit

**Files:**
- Verify only; do not modify additional files unless a test exposes a cleanup-owned defect.

**Interfaces:**
- Consumes: the complete current working tree.
- Produces: evidence that supported execution and validation paths pass without OpenCode support.

- [ ] **Step 1: Run the complete local test suite**

Run:

```bash
python -m pytest tests -q
```

Expected: zero failures.

- [ ] **Step 2: Verify the supported CLI imports and reports help**

Run:

```bash
python -m hieraresearch --help
python tools/harness_watch.py --help
python tools/harness_guard.py --help
```

Expected: all commands exit 0; harness help exposes no removed OpenCode flags.

- [ ] **Step 3: Run the final repository search**

Run the Task 2 search once more.

Expected: no matches outside the design and implementation records.

- [ ] **Step 4: Audit working-tree ownership and scope**

Run:

```bash
git status --short
git diff --stat
git diff --check
```

Expected: no whitespace errors; cleanup files are limited to the plan/spec, OpenCode deletions, harness utilities/tests, and two documentation resources. Pre-existing unrelated modifications remain untouched and are reported separately.

- [ ] **Step 5: Report completion without staging or committing**

Summarize removed surfaces, retained Claude behavior, exact verification commands and results, and any failures attributable to unrelated pre-existing worktree changes. Do not stage or commit.
