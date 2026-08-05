# Progressive Tuning Implementation Plan

> **STATUS: COMPLETED AND HISTORICAL — DO NOT EXECUTE.** This plan landed in
> `45d5e69`…`ad2fa7d`. Its unchecked `- [ ]` boxes are stale bookkeeping, not
> pending work, and parts of it are now actively wrong: it targets `.opencode/`
> throughout and at one point names `.opencode` canonical, whereas `.claude/` is
> the canonical and only maintained runtime and `.opencode/`/`.kimi/` are
> deprecated mirrors. Read it only as a record of what was decided on
> 2026-08-03. For current mechanics use the code, `.claude/`, and the spec at
> `docs/superpowers/specs/2026-08-03-progressive-tuning-design.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one-shot deep tuning with progressive, fixed-size tuning bouts: candidates accumulate Phase-C attempts across rounds, tuned candidates re-enter the selection pool, evidence depth becomes graded, and continuation bouts may start from LLM-proposed (deterministically validated) configs.

**Architecture:** Spec: `docs/superpowers/specs/2026-08-03-progressive-tuning-design.md`. Phase-C stages gain a `bout_index`; stage admission restarts the method chain per bout once the previous bout is finalized; finalization closes per-bout (best-so-far over ALL stages) and writes `tuning_bouts`, `last_bout_improved`, and a three-level `evaluation_depth` to the ledger; `select-candidate` re-admits tuned candidates using a like-for-like ranking key; a new `validate-proposals` subcommand admits orchestrator-proposed re-warm configs into `phase_c.pending_proposals`, which the search scripts consume first.

**Tech Stack:** Python 3 stdlib-only tools (`tools/`), pytest-style unittest suite (`tests/`), OpenCode agent prompts (`.opencode/`), JSON contracts.

## Global Constraints

- Additive schema evolution only (workspace AGENTS.md): never rename established ids/fields; old artifacts must stay readable (normalize-on-read).
- Scores are always lower-is-better; crashes are worst.
- Never rank an untuned screening score against a tuning-lowered score (like-for-like invariant).
- Deterministic mechanisms (validation, selection, transitions, receipts) stay in `tools/`, outside LLM prose. LLM proposes, deterministic machinery disposes.
- `.claude` is the canonical runtime; `.opencode`/`.kimi/` mirrors may drift (user decision, overrides repo AGENTS.md mirror rule).
- Tests minimum: extend existing test files; no new validators; no required/forbidden-wording checks over prompts or docs.
- `python -m pytest tests -q` must stay fast: no GPU, no network.
- Never commit anything under `runs/`.
- Commit message style: `feat:` / `fix:` / `docs:` imperative subjects (see `git log`).
- In tests, phase-C trial rows: successful rows carry `{"params", "score"}` (no `status` key); failed rows carry `status: "failed"`; preflight rejections carry `status: "preflight_rejected"` and never count as objective attempts.

## Task Map

| # | Task | Touches |
|---|---|---|
| 1 | Workspace guide update | `/home/woden/spark/AGENTS.md`, `/home/woden/spark/CLAUDE.md` |
| 2 | Config knobs | `tools/run_cfg.py`, `tasks/framework_cfg.example.json`, `tools/tuners/tune_tools.py` (constants), `tests/test_run_cfg.py` |
| 3 | Ledger schema: `tuning_bouts`, `last_bout_improved`, graded depth docs | `tools/ledger.py`, `.opencode/rules/ledger.md`, `tests/test_tuning_finalization.py` |
| 4 | Bout-aware stage machinery | `tools/tuners/_common.py`, `tools/tuners/tune_tools.py`, `tests/test_deep_tune_governance.py`, `tests/test_tuner_patience.py` |
| 5 | Cross-bout finalization | `tools/tuners/tune_tools.py`, `tools/ledger.py`, `tools/finalize_tuning.py`, `tests/test_tuning_finalization.py` |
| 6 | Progressive select-candidate | `tools/tuners/tune_tools.py`, `tests/test_deep_tune_governance.py` |
| 7 | `validate-proposals` + search-script consumption | `tools/tuners/tune_tools.py`, `tools/tuners/_common.py`, `tools/tuners/grid_search.py`, `tools/tuners/bo_search.py`, `tools/tuners/cmaes_search.py`, `tests/test_deep_tune_governance.py` |
| 8 | Depth-aware mechanical evidence gates (plan amendment — added after Task 5 review found the binary depth split in code) | `tools/semantic_evidence.py`, `tools/search_space_state.py`, `tools/ledger.py` (comment), `.opencode/rules/ledger.md`, `tests/test_semantic_evidence.py`, `tests/test_search_space_state.py` |
| 9 | Agent prompts + docs | `.opencode/agents/tuner-orchestrator.md`, `.opencode/agents/experience-extractor.md`, `AGENTS.md`, `README_ZH.md`, `docs/search-space.md` |
| 10 | Full verification | — |

Interfaces between tasks (exact names later tasks rely on):

- Task 4 produces, in `tools/tuners/_common.py`: `stage_bout_index(stage: dict) -> int`, `stages_by_bout(stages: list) -> list[list[dict]]` (raises `ValueError` on regressing/skipping bout order), and `prior_patience_state(report_path, bout_index: int | None = None)`.
- Task 4 produces, in `tools/tuners/tune_tools.py`: `has_applied_close(report: dict) -> bool`, extended `has_validated_applied_close(report: dict) -> bool` (currency-aware), and `deep_tune_time_budget(...)` return dicts gain `"bout_index": int`.
- Task 5 produces: `tune_tools.DEFAULT_TUNED_THRESHOLD = 16`, `tune_tools.load_tuned_threshold(ledger_path: Path) -> int`, `finalized_tuning_record(report, *, tuned_threshold: int = DEFAULT_TUNED_THRESHOLD)`, ledger fields `tuning_bouts` / `last_bout_improved` (from Task 3), report field `last_finalized_stage_index`.
- Task 6 produces: `select_candidate(ledger, *, n_min, top_percentile, bout_trials=None, budget_allocation=None)` returning `bout_index` / `is_continuation`.
- Task 7 produces: `tune_tools.validate_proposals(candidate_path, report_path, proposals) -> {ok, accepted, rejected}`, CLI `tune_tools.py validate-proposals`, `_common.read_pending_proposals(report_path) -> list[dict]`.

---

### Task 1: Workspace guide update

**Files:**
- Modify: `/home/woden/spark/AGENTS.md` (the mirror-sync bullet under "Change Discipline")
- Modify: `/home/woden/spark/CLAUDE.md` (same content — the file's own rule: apply edits to both)

**Interfaces:** none (documentation only; unblocks later tasks that leave `.claude`/`.kimi` mirrors stale).

- [ ] **Step 1: Edit `/home/woden/spark/AGENTS.md`**

In the "Change Discipline" section, replace:

```markdown
- Keep mirrored runtime contracts synchronized when a shared agent protocol
  changes.
```

with:

```markdown
- `.opencode/` is the canonical runtime for agent contracts; `.claude/` and
  `.kimi/` mirrors are allowed to drift (mirror sync is no longer a
  priority). Step 2 tuning is progressive: fixed-size bouts, re-selection by
  score, graded evidence depth — see
  `HieraResearch/docs/superpowers/specs/2026-08-03-progressive-tuning-design.md`.
```

- [ ] **Step 2: Apply the identical edit to `/home/woden/spark/CLAUDE.md`**

Run: `grep -n "canonical runtime" /home/woden/spark/AGENTS.md /home/woden/spark/CLAUDE.md`
Expected: one match in each file.

- [ ] **Step 3: Commit (only if the workspace root is a git repo)**

```bash
git -C /home/woden/spark rev-parse --is-inside-work-tree 2>/dev/null && \
  git -C /home/woden/spark add AGENTS.md CLAUDE.md && \
  git -C /home/woden/spark commit -m "docs: name .opencode canonical; note progressive tuning" || \
  echo "workspace root is not a git repo — no commit"
```

---

### Task 2: Config knobs (`bout_trials`, `tuned_threshold`, `rewarm_proposals`)

**Files:**
- Modify: `tools/run_cfg.py` (`_validate_tuner_config`, the `_validate_positive_int_override` loop, ~line 105)
- Modify: `tools/tuners/tune_tools.py` (module constants near `DEFAULT_N_MIN`, line 2972)
- Modify: `tasks/framework_cfg.example.json` (`tuner` section + `_keys`)
- Test: `tests/test_run_cfg.py`

**Interfaces:**
- Produces: `tune_tools.DEFAULT_BOUT_TRIALS = 8`, `tune_tools.DEFAULT_TUNED_THRESHOLD = 16`, `tune_tools.DEFAULT_REWARM_PROPOSALS = 3`; run_cfg rejects non-positive-int values for `tuner.bout_trials` / `tuner.tuned_threshold` / `tuner.rewarm_proposals`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_cfg.py` (match the file's existing import/style; `read_framework_cfg` and `RunConfigError` live in `tools/run_cfg.py`):

```python
class ProgressiveTunerKnobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg_path = Path(self.tmp.name) / "framework_cfg.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, tuner: dict) -> None:
        self.cfg_path.write_text(json.dumps({"tuner": tuner}))

    def test_valid_progressive_knobs_parse(self):
        self._write({"bout_trials": 8, "tuned_threshold": 16, "rewarm_proposals": 3})
        cfg = read_framework_cfg(self.cfg_path)
        self.assertEqual(cfg["tuner"]["bout_trials"], 8)
        self.assertEqual(cfg["tuner"]["tuned_threshold"], 16)
        self.assertEqual(cfg["tuner"]["rewarm_proposals"], 3)

    def test_invalid_progressive_knobs_rejected(self):
        for key in ("bout_trials", "tuned_threshold", "rewarm_proposals"):
            for bad in (0, -1, 2.5, "8", True):
                with self.subTest(key=key, bad=bad):
                    self._write({key: bad})
                    with self.assertRaises(RunConfigError):
                        read_framework_cfg(self.cfg_path)
```

If `tests/test_run_cfg.py` does not already import `json`, `tempfile`, `unittest`, `Path`, `read_framework_cfg`, `RunConfigError`, add the missing imports to match.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_cfg.py -q -k ProgressiveTunerKnobs`
Expected: FAIL — `RunConfigError` not raised for bad values (validation does not know the keys yet).

- [ ] **Step 3: Add validation in `tools/run_cfg.py`**

In `_validate_tuner_config`, extend the `_validate_positive_int_override` loop's key tuple from:

```python
    for key in (
        "bo_n_trials",
        "bo_patience_cap",
        "bo_patience_floor",
        "deep_tune_per_candidate_cap",
    ):
```

to:

```python
    for key in (
        "bo_n_trials",
        "bo_patience_cap",
        "bo_patience_floor",
        "deep_tune_per_candidate_cap",
        "bout_trials",
        "tuned_threshold",
        "rewarm_proposals",
    ):
```

- [ ] **Step 4: Add module defaults in `tools/tuners/tune_tools.py`**

Immediately above `DEFAULT_N_MIN = 5` (line 2972), insert:

```python
DEFAULT_BOUT_TRIALS = 8       # one progressive tuning bout's objective-attempt budget
DEFAULT_TUNED_THRESHOLD = 16  # Phase-C attempts at which evaluation_depth becomes "tuned"
DEFAULT_REWARM_PROPOSALS = 3  # max LLM-proposed configs a continuation bout may start from
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_run_cfg.py -q`
Expected: PASS (whole file).

- [ ] **Step 6: Update `tasks/framework_cfg.example.json`**

In the `tuner` object, after `"deep_tune_per_candidate_cap": 20` add (keep JSON valid, mind the comma):

```json
    "deep_tune_per_candidate_cap": 20,
    "bout_trials": 8,
    "tuned_threshold": 16,
    "rewarm_proposals": 3
```

In `_keys`, update the `tuner.deep_tune_per_candidate_cap` text and add three entries:

```json
    "tuner.deep_tune_per_candidate_cap": "inner budget policy: maximum admitted Phase-C objective calls for one candidate across all its bouts and retries (default 20).",
    "tuner.bout_trials": "inner: objective-attempt budget of ONE progressive tuning bout (default 8). The orchestrator's per-invocation trial cap is min(bout_trials, per-candidate-cap remaining, budget remaining).",
    "tuner.tuned_threshold": "inner: cumulative Phase-C attempts at which a candidate's evaluation_depth becomes 'tuned' (default 16 = two full bouts); 1..threshold-1 attempts is 'tuned_lightly', 0 is 'screening'.",
    "tuner.rewarm_proposals": "inner: maximum number of warm configs the tuner-orchestrator may PROPOSE at a continuation bout (default 3). Proposals pass tune_tools.py validate-proposals (in-space, schema-compatible, deduped) and consume bout budget like any trial.",
```

Also update the `_README` array's `READ BY` line: append " · tune_tools.py validate-proposals -> tuner.rewarm_proposals" before the closing period.

- [ ] **Step 7: Verify example JSON parses and passes validation**

Run: `python -c "import json; from pathlib import Path; import sys; sys.path.insert(0, 'tools'); from run_cfg import read_framework_cfg; read_framework_cfg(Path('tasks/framework_cfg.example.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git add tools/run_cfg.py tools/tuners/tune_tools.py tasks/framework_cfg.example.json tests/test_run_cfg.py
git commit -m "feat: add progressive tuning config knobs"
```

---

### Task 3: Ledger schema — `tuning_bouts`, `last_bout_improved`, graded depth contract

**Files:**
- Modify: `tools/ledger.py` (`RECORD_FIELDS` ~line 104, `TUNING_FIELDS` ~line 107, `_new_record` ~line 236, `_load_ledger` ~line 172)
- Modify: `.opencode/rules/ledger.md` (field table ~line 97, depth definition ~lines 175-188, close-path paragraph ~lines 37-46)
- Test: `tests/test_tuning_finalization.py`

**Interfaces:**
- Produces: ledger record fields `tuning_bouts` (int, default 0) and `last_bout_improved` (bool | null, default null); `_load_ledger` normalizes legacy records (`tune: true` with no `tuning_bouts` → 1).
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Write the failing test**

Append a new test class to `tests/test_tuning_finalization.py` (the file already imports `json`, `tempfile`, `unittest`, `Path`; add `import ledger` to the `sys.path.insert(0, str(ROOT / "tools"))` import block):

```python
class LedgerProgressiveFieldsTest(unittest.TestCase):
    def test_new_record_carries_progressive_defaults(self):
        record = ledger._new_record("001")
        self.assertEqual(record["tuning_bouts"], 0)
        self.assertIsNone(record["last_bout_improved"])

    def test_legacy_records_normalize_on_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            legacy = {
                "run_id": "001",
                "semantic_point": {},
                "policy_receipt": {},
                "status": "keep",
                "tune": True,
                "final_best_score": 0.9,
            }
            fresh = {
                "run_id": "002",
                "semantic_point": {},
                "policy_receipt": {},
                "status": "keep",
                "tune": False,
                "final_best_score": 1.1,
            }
            ledger_path.write_text(json.dumps({
                "task": "autoresearch-baseline",
                "tag": "test",
                "metric": "val_bpb",
                "search_space_state": empty_search_space_state(),
                "records": [legacy, fresh],
            }))
            data = ledger._load_ledger(ledger_path)
            by_id = {r["run_id"]: r for r in data["records"]}
            self.assertEqual(by_id["001"]["tuning_bouts"], 1)
            self.assertIsNone(by_id["001"]["last_bout_improved"])
            self.assertEqual(by_id["002"]["tuning_bouts"], 0)
            self.assertIsNone(by_id["002"]["last_bout_improved"])
```

(`empty_search_space_state` is already imported at the top of the file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tuning_finalization.py -q -k LedgerProgressiveFields`
Expected: FAIL — `KeyError: 'tuning_bouts'` on the first test.

- [ ] **Step 3: Extend the schema in `tools/ledger.py`**

In `RECORD_FIELDS` (ends with `"dag_revision"`), append two fields after it:

```python
    "dag_revision",      # last score/status revision visible to the development DAG
    "tuning_bouts",      # int: completed progressive-tuning bouts (0 = never tuned)
    "last_bout_improved",  # bool | null: last bout beat its pre-bout incumbent
)
```

In `TUNING_FIELDS`, append the same two names before the closing paren:

```python
    "applied_incumbent",
    "tuning_bouts",
    "last_bout_improved",
)
```

In `_new_record`, extend the literal override:

```python
def _new_record(run_id: str) -> dict:
    return {field: None for field in RECORD_FIELDS} | {
        "run_id": run_id,
        "tune": False,
        "tuning_bouts": 0,
        "last_bout_improved": None,
        "status": "pending",
    }
```

In `_load_ledger`, immediately after `data.setdefault("records", [])` (and before the `search_space_state` check), insert:

```python
        for record in data["records"]:
            if not isinstance(record, dict):
                continue
            # Progressive-tuning fields, additive: a legacy one-shot-tuned
            # record counts as one completed bout with unknown response.
            record.setdefault(
                "tuning_bouts", 1 if record.get("tune") else 0
            )
            record.setdefault("last_bout_improved", None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tuning_finalization.py -q`
Expected: PASS (whole file — the new fields flow through existing close paths as `None`/`0` via the `TUNING_FIELDS` projection, which uses `.get()`).

- [ ] **Step 5: Update `.opencode/rules/ledger.md`**

In the field table, replace the `tune` row and add rows (keep table format):

```markdown
| `tune` | derived bool: the candidate completed at least one tuning bout (`evaluation_depth` is not `screening`) |
| `tuning_bouts` | completed progressive-tuning bouts; 0 for screening-only or legacy untuned records, 1 for legacy one-shot-tuned records |
| `last_bout_improved` | whether the last bout produced a trial strictly better than its pre-bout incumbent; null when unknown/never tuned |
```

Replace the depth definition sentence in the comparator section (currently "A direct edge is **tuned** when the child record's `evaluation_depth` is `tuned` (it has a scored Phase-C trial); screening-depth and legacy direct edges remain evidence but cannot drive contradiction gates.") with:

```markdown
`evaluation_depth` is graded by cumulative Phase-C objective attempts:
`screening` (0), `tuned_lightly` (1 to `tuner.tuned_threshold`−1, default 15),
or `tuned` (≥ threshold, default 16). A direct edge is **tuned** when the
child record's `evaluation_depth` is `tuned` and **lightly tuned** at
`tuned_lightly`; screening-depth and legacy direct edges remain evidence but
cannot drive contradiction gates. Contradiction-grade transitions require at
least two direct tuned edges or at least three direct edges at
`tuned_lightly` or deeper.
```

In the mutation-command list, replace the `finalize_tuning.py` bullet with:

```markdown
- `finalize_tuning.py` for a completed tuning-bout close: it validates
  terminal Phase C for the current bout, applies the global best (warm plus
  every Phase-C trial across all bouts), and writes score/status/tuning
  metadata (`tuning_bouts`, `last_bout_improved`, graded `evaluation_depth`)
  together. A finalized candidate stays eligible for later bouts.
  `set-tuning --mark-tuned` is disabled, so there is no second close path;
```

- [ ] **Step 6: Commit**

```bash
git add tools/ledger.py .opencode/rules/ledger.md tests/test_tuning_finalization.py
git commit -m "feat: add progressive tuning fields to ledger records"
```

---

### Task 4: Bout-aware stage machinery

**Files:**
- Modify: `tools/tuners/_common.py` (`append_trial` :1028, `set_stage_meta` :1043, `_deep_tune_time_budget_locked` :460, `prior_patience_state` :1275; new `stage_bout_index` / `stages_by_bout`)
- Modify: `tools/tuners/tune_tools.py` (`has_validated_applied_close` :1330, `phase_c_action` :1374; new `has_applied_close` / `last_finalized_stage_index`)
- Test: `tests/test_deep_tune_governance.py`, `tests/test_tuner_patience.py`

**Interfaces:**
- Produces: `stage_bout_index(stage) -> int`, `stages_by_bout(stages) -> list[list[dict]]` (both in `_common.py`); `has_applied_close(report) -> bool` and currency-aware `has_validated_applied_close(report) -> bool` (in `tune_tools.py`); `deep_tune_time_budget` return dict gains `"bout_index": int`; `phase_c_action` run-actions gain `"bout_index": int` and the new `"start_new_bout"` reason; `prior_patience_state(report_path, bout_index=None)` — streak counted over the given (default: current/max) bout only.
- Consumes: nothing from Tasks 2-3 at code level (ledger fields unused here).

Key facts established during planning (do not re-verify, rely on them):
- `_candidate_structure_sha256` strips `BASE_PARAMS`/`SEARCH_SPACE` assignments before hashing, so a finalize-applied `BASE_PARAMS` rewrite does NOT change the execution revision.
- Stage admission currently calls `validate_phase_a_candidate_state(report, path)` with the default `require_warm_base_applied=True`, which fails after a Phase-C winner is applied; the relaxation must be `not has_applied_close(report)` (mirrors `phase_c_action`).
- `_TERMINAL_STAGE_STATUSES` lives in `tune_tools.py`; `_common.py` already lazily imports from `tune_tools` inside functions.

- [ ] **Step 1: Write the failing tests (patience scoping)**

In `tests/test_tuner_patience.py`, extend `_write_report` so stages may carry `bout_index`, and add a test class:

```python
def _write_report(path: Path, warm_scores, stage_trials, bout_index: int | None = None) -> None:
    stage = {
        "method": "bo",
        "trials": [
            {"params": {"x": float(i)}, "score": s, "status": st}
            for i, (s, st) in enumerate(stage_trials)
        ],
    }
    if bout_index is not None:
        stage["bout_index"] = bout_index
    write_tune_report(
        path,
        {
            "phase_a": {
                "warm_start_configs": [
                    {"params": {"x": float(i)}, "score": s}
                    for i, s in enumerate(warm_scores)
                ]
            },
            "phase_c": {"stages": [stage] if stage_trials or bout_index is not None else []},
        }
    )


def _write_two_bout_report(path: Path) -> None:
    write_tune_report(
        path,
        {
            "phase_a": {
                "warm_start_configs": [{"params": {"x": 1.0}, "score": 1.07}]
            },
            "phase_c": {
                "stages": [
                    {
                        "method": "bo",
                        "trials": [
                            {"params": {"x": 2.0}, "score": 1.05},
                            {"params": {"x": 3.0}, "score": None, "status": "failed"},
                        ],
                    },
                    {
                        "method": "bo",
                        "bout_index": 1,
                        "trials": [
                            {"params": {"x": 4.0}, "score": 1.06},
                        ],
                    },
                ]
            },
        }
    )


class BoutPatienceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.report_path = Path(self.tmp.name) / "tune_report.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_streak_scoped_to_current_bout_best_is_global(self):
        _write_two_bout_report(self.report_path)
        best, streak = prior_patience_state(self.report_path)
        self.assertEqual(best, 1.05)
        self.assertEqual(streak, 1)  # only bout 1's non-improving trial counts

    def test_explicit_earlier_bout_replays_that_bout(self):
        _write_two_bout_report(self.report_path)
        best, streak = prior_patience_state(self.report_path, bout_index=0)
        self.assertEqual(best, 1.05)
        self.assertEqual(streak, 1)  # bout 0: improvement then failed trial
```

Update the existing call in the file's header comment if needed; existing tests must keep passing unchanged (default `bout_index=None` = current bout = old behavior for single-bout reports).

- [ ] **Step 2: Write the failing tests (bout admission)**

In `tests/test_deep_tune_governance.py`, add a test class. Reuse `DeepTuneGovernanceTest._fixture` patterns: the fixture below mirrors it (train.py with `PARAM_SCHEMA`/`SEARCH_SPACE`/`BASE_PARAMS`, `prepare.py`, phase-a report keyed by `_candidate_execution_revision`); adjust to the exact fixture helpers in the file if they differ:

```python
class BoutAdmissionTest(unittest.TestCase):
    def _fixture(self, root: Path):
        run_dir = root / "run"
        candidate_dir = run_dir / "candidates" / "001"
        candidate_dir.mkdir(parents=True)
        candidate = candidate_dir / "train.py"
        candidate.write_text(
            "PARAM_SCHEMA = {'x': 'float'}\n"
            "SEARCH_SPACE = {'x': ('float', 0.0, 2.0)}\n"
            "BASE_PARAMS = {'x': 1.0}\n"
            "def make_model(params):\n"
            "    return params\n"
        )
        (candidate_dir / "prepare.py").write_text(
            "def evaluate_config(make_model, params):\n"
            "    return float(params['x'])\n"
        )
        report_path = candidate_dir / "tune_report.json"
        report = {
            "phase_a": {
                "status": "ok",
                "warm_start_configs": [{"params": {"x": 1.0}, "score": 1.0}],
                "best_warm_params": {"x": 1.0},
                "best_warm_score": 1.0,
                "trials_attempted": 1,
                "elapsed_seconds": 1.0,
                "search_space": {"x": ["float", 0.0, 2.0]},
            },
            "preflight": {"attempts": []},
        }
        report["phase_a"]["candidate_code_revision"] = (
            _candidate_execution_revision(candidate)
        )
        report_path.write_text(json.dumps(report))
        return candidate, report_path

    def _close_bout_zero(self, candidate, report_path, *, improved: bool) -> None:
        """Simulate a finalized bout 0 (warm incumbent stays best)."""
        report = json.loads(report_path.read_text())
        report["phase_c"] = {
            "stages": [
                {
                    "method": "grid",
                    "status": "ok",
                    "trials": [
                        {"params": {"x": 0.5 if improved else 1.5},
                         "score": 0.5 if improved else 1.5}
                    ],
                    "elapsed_seconds": 1.0,
                }
            ]
        }
        best = 0.5 if improved else 1.0
        report["final_best_params"] = {"x": best}
        report["final_best_score"] = best
        report["applied_to_base_params"] = True
        report["last_finalized_stage_index"] = 0
        report_path.write_text(json.dumps(report))
        # finalize would have rewritten BASE_PARAMS to the applied winner
        source = candidate.read_text().replace(
            "BASE_PARAMS = {'x': 1.0}", f"BASE_PARAMS = {{'x': {best}}}"
        )
        candidate.write_text(source)

    def test_new_bout_admitted_after_finalized_bout(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            self._close_bout_zero(candidate, report_path, improved=False)
            budget = deep_tune_time_budget(candidate, report_path, "grid")
            try:
                self.assertEqual(budget["bout_index"], 1)
                report = json.loads(report_path.read_text())
                stages = report["phase_c"]["stages"]
                self.assertEqual(len(stages), 2)
                self.assertEqual(stages[1].get("bout_index"), 1)
                self.assertEqual(stages[1]["status"], "running")
            finally:
                budget["_phase_c_lock_handle"].close()

    def test_new_bout_refused_when_previous_bout_not_finalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "ok",
                        "trials": [{"params": {"x": 1.5}, "score": 1.5}],
                        "elapsed_seconds": 1.0,
                    }
                ]
            }
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(
                DeepTuneStageAdmissionError, "must be finalized"
            ):
                deep_tune_time_budget(candidate, report_path, "grid")

    def test_terminal_rerun_within_same_bout_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            report = json.loads(report_path.read_text())
            report["phase_c"] = {
                "stages": [
                    {
                        "method": "grid",
                        "status": "ok",
                        "trials": [{"params": {"x": 1.5}, "score": 1.5}],
                        "elapsed_seconds": 1.0,
                    },
                    {
                        "method": "grid",
                        "bout_index": 1,
                        "status": "rejected",
                        "trials": [],
                    },
                    {
                        "method": "bo",
                        "bout_index": 1,
                        "status": "ok",
                        "trials": [{"params": {"x": 1.4}, "score": 1.4}],
                        "elapsed_seconds": 1.0,
                    },
                ]
            }
            report_path.write_text(json.dumps(report))
            # bo is the active final stage of bout 1 and terminal; re-admitting
            # it is a same-bout rerun (only the PRIMARY method of a terminal
            # bout can start a new bout, and that requires a finalized close).
            with self.assertRaisesRegex(DeepTuneStageAdmissionError, "cannot be rerun"):
                deep_tune_time_budget(candidate, report_path, "bo")
```

Note for the implementer: the first test asserts on the new-bout path where the method (grid) IS present in the closed bout — this is the chain-restart case and only works if new-bout detection precedes the within-bout stage lookup. Corollary: re-admitting a terminal bout's PRIMARY method without a finalized close now fails with "must be finalized" (the new-bout guard), not the old "cannot be rerun" message.

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_tuner_patience.py tests/test_deep_tune_governance.py -q -k "Bout"`
Expected: FAIL (`prior_patience_state` has no `bout_index` param; admission rejects/ignores bouts).

- [ ] **Step 4: Add bout helpers to `tools/tuners/_common.py`**

Near `read_tune_report` (:1043), add:

```python
def stage_bout_index(stage: dict) -> int:
    """0-based bout a Phase-C stage belongs to (legacy unstamped stages: 0)."""
    value = stage.get("bout_index", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def stages_by_bout(stages: list) -> list[list[dict]]:
    """Group ordered stages into bouts. Bout indices must start at 0, be
    contiguous, and never regress along the list."""
    bouts: list[list[dict]] = []
    for index, stage in enumerate(stages):
        bout = stage_bout_index(stage)
        if bout < len(bouts) - 1:
            raise ValueError(
                f"phase_c.stages[{index}] bout_index {bout} regresses below an "
                "earlier bout"
            )
        if bout > len(bouts):
            raise ValueError(
                f"phase_c.stages[{index}] bout_index {bout} skips a bout"
            )
        if bout == len(bouts):
            bouts.append([])
        bouts[-1].append(stage)
    return bouts
```

- [ ] **Step 5: Make stage addressing last-match in `tools/tuners/_common.py`**

In both `append_trial` (:1035) and `set_stage_meta` (:1053), change the lookup line from:

```python
    stage = next((s for s in stages if s.get("method") == method), None)
```

to:

```python
    stage = next((s for s in reversed(stages) if s.get("method") == method), None)
```

Rationale (update each docstring's first line): the same method recurs across bouts; the active stage is always the LAST stage with that method. Also give `set_stage_meta` a keyword-only `bout_index: int | None = None` parameter; when it CREATES the stage and `bout_index` is a positive int, stamp it:

```python
    if stage is None:
        stage = {"method": method, "trials": []}
        if isinstance(bout_index, int) and not isinstance(bout_index, bool) and bout_index > 0:
            stage["bout_index"] = bout_index
        stages.append(stage)
```

- [ ] **Step 6: Scope `prior_patience_state` to the current bout**

Replace the function in `tools/tuners/_common.py` (:1275) with:

```python
def prior_patience_state(
    report_path: Path,
    bout_index: int | None = None,
) -> tuple[float | None, int]:
    """Replay persisted trials into (best_score, patience streak) for
    PatienceMonitor seeding. The improvement bar is GLOBAL (warm screening
    plus every bout's trials); the patience streak is scoped to one bout
    (default: the current/max bout), so a continuation bout starts a fresh
    window while still having to beat the run's best to reset.

    Only Phase-C rows of the scoped bout increment the streak. Warm screening
    is a deliberate spread over distinct numeric regimes, not a stalled
    optimizer. The inherited config-0 fidelity control is not an incumbent, so
    it cannot set ``best`` or reset patience; a Phase-C duplicate of its exact
    params counts as a spent non-improving trial.
    """
    report = read_tune_report(report_path)
    phase_a = report.get("phase_a", {})
    stages = report.get("phase_c", {}).get("stages", [])
    if bout_index is None:
        bout_index = max((stage_bout_index(s) for s in stages), default=0)

    best: float | None = None
    inherited_param_ids: set[str] = set()
    for trial in phase_a.get("warm_start_configs", []):
        if (
            trial.get("role") == "inherited_control"
            and isinstance(trial.get("params"), dict)
        ):
            inherited_param_ids.add(params_identity(trial["params"]))
            continue
        score = trial.get("score")
        if is_finite_score(score) and (best is None or float(score) < best):
            best = float(score)

    def absorbs(trial: dict) -> bool:
        """Whether the trial improves the running best (reset) or not."""
        nonlocal best
        is_inherited_duplicate = (
            isinstance(trial.get("params"), dict)
            and params_identity(trial["params"]) in inherited_param_ids
        )
        score = trial.get("score")
        if (
            not is_inherited_duplicate
            and is_finite_score(score)
            and (best is None or float(score) < best)
        ):
            best = float(score)
            return True
        return False

    # Earlier bouts only move the global bar; they never seed the streak.
    for stage in stages:
        if stage_bout_index(stage) >= bout_index:
            continue
        for trial in stage.get("trials", []):
            absorbs(trial)

    streak = 0
    for stage in stages:
        if stage_bout_index(stage) != bout_index:
            continue
        for trial in stage.get("trials", []):
            if absorbs(trial):
                streak = 0
            else:
                streak += 1
    return best, streak
```

- [ ] **Step 7: Add close-state predicates to `tools/tuners/tune_tools.py`**

Replace `has_validated_applied_close` (:1330-1335) with:

```python
def last_finalized_stage_index(report: dict) -> int | None:
    """Stage index the last finalize close covered (legacy closes: absent)."""
    if not isinstance(report, dict):
        return None
    value = report.get("last_finalized_stage_index")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def has_applied_close(report: dict) -> bool:
    """Whether a finalize close (any bout) applied its incumbent to BASE_PARAMS.

    The closing fields must prove consistent with the stages they cover —
    i.e. the stage prefix up to ``last_finalized_stage_index`` (legacy closes
    without the field covered every stage, which is all a one-shot report can
    have)."""
    if not isinstance(report, dict) or report.get("applied_to_base_params") is not True:
        return False
    stages = report.get("phase_c", {}).get("stages", [])
    last = last_finalized_stage_index(report)
    if last is None:
        last = len(stages) - 1
    phase_c = report.get("phase_c", {})
    prefix_report = {
        **report,
        "phase_c": {**phase_c, "stages": stages[: last + 1]},
    }
    finalizable_tuning_result(prefix_report, require_applied=True)
    return True


def has_validated_applied_close(report: dict) -> bool:
    """Whether the applied close is CURRENT: it covers every stage in the report."""
    if not has_applied_close(report):
        return False
    stages = report.get("phase_c", {}).get("stages", [])
    last = last_finalized_stage_index(report)
    return last is None or last == len(stages) - 1
```

- [ ] **Step 8: Rework stage admission in `_deep_tune_time_budget_locked` (`tools/tuners/_common.py`)**

Three edits inside the function:

(a) Extend the lazy import and relax the warm-base check. Change:

```python
    from tune_tools import (
        _read_search_space,
        select_method,
        validate_candidate_execution_revision,
        validate_phase_a_candidate_state,
    )

    try:
        validate_phase_a_candidate_state(report, Path(ref_path))
```

to:

```python
    from tune_tools import (
        _TERMINAL_STAGE_STATUSES,
        _read_search_space,
        has_applied_close,
        has_validated_applied_close,
        select_method,
        validate_candidate_execution_revision,
        validate_phase_a_candidate_state,
    )

    try:
        bouts = stages_by_bout(stages)
        validate_phase_a_candidate_state(
            report,
            Path(ref_path),
            require_warm_base_applied=not has_applied_close(report),
        )
```

(keep the rest of the `try` body — `candidate_execution_revision` and `search_space` assignments — unchanged, and the `except (SystemExit, ValueError)` clause unchanged, so a malformed bout order surfaces as `DeepTuneStageAdmissionError`.)

(b) Replace the block from `methods = [stage.get("method") for stage in stages]` through the end of the stage find/append `else` clause (the whole "duplicate methods / expected prefix / stage is None / else" region) with bout-scoped logic operating on the current bout:

```python
    current = bouts[-1] if bouts else []
    methods = [stage.get("method") for stage in current]
    if not all(isinstance(value, str) for value in methods):
        raise DeepTuneStageAdmissionError(
            f"phase_c stage methods must be strings: {methods!r}"
        )
    if len(methods) != len(set(methods)):
        raise DeepTuneStageAdmissionError(
            f"phase_c bout contains duplicate method stages: {methods!r}"
        )
    expected_prefix = method_chain[:position]
    actual_prefix = methods[:position]
    if actual_prefix != expected_prefix or any(
        current[index].get("status") != "rejected"
        for index in range(min(position, len(current)))
    ):
        raise DeepTuneStageAdmissionError(
            f"method {method!r} requires rejected prefix {expected_prefix!r}; "
            f"found methods/statuses "
            f"{[(stage.get('method'), stage.get('status')) for stage in current]!r}"
        )

    current_terminal = bool(current) and all(
        item.get("status") in _TERMINAL_STAGE_STATUSES for item in current
    )
    stage = None
    bout_index = len(bouts) - 1 if bouts else 0
    if current_terminal and method == method_chain[0]:
        # The previous bout's chain closed out (finalizable or exhausted). A
        # validated applied close lets a NEW bout restart the chain — the new
        # stage reuses the method under the next bout_index.
        if not has_validated_applied_close(report):
            raise DeepTuneStageAdmissionError(
                "the previous bout must be finalized "
                "(tools/finalize_tuning.py) before a new bout starts"
            )
        bout_index = len(bouts)
        stage = {"method": method, "trials": [], "bout_index": bout_index}
        stages.append(stage)
    else:
        stage = next(
            (item for item in current if item.get("method") == method),
            None,
        )
        if stage is None:
            if len(current) != position:
                raise DeepTuneStageAdmissionError(
                    f"method {method!r} is not the next Phase-C stage of its bout"
                )
            stage = {"method": method, "trials": []}
            if bout_index > 0:
                stage["bout_index"] = bout_index
            stages.append(stage)
        else:
            if current.index(stage) != position or len(current) != position + 1:
                raise DeepTuneStageAdmissionError(
                    f"method {method!r} is not the active final Phase-C stage "
                    "of its bout"
                )
            if stage.get("status") != "running":
                raise DeepTuneStageAdmissionError(
                    f"terminal Phase-C method {method!r} cannot be rerun "
                    f"within its bout (status={stage.get('status')!r})"
                )
```

(c) Add `"bout_index": bout_index` to the returned budget dict.

- [ ] **Step 9: Rework `phase_c_action` in `tools/tuners/tune_tools.py`**

Replace the whole function (:1374-1473) with:

```python
def phase_c_action(report: dict, candidate_path: Path) -> dict:
    """Return the one legal resume action for a candidate's Phase-C state."""
    applied_close = has_validated_applied_close(report)
    validate_phase_a_candidate_state(
        report,
        candidate_path,
        require_warm_base_applied=not has_applied_close(report),
    )
    search_space = _read_search_space(candidate_path)

    selected = select_method(len(search_space))
    method_chain = [selected["method"], *selected["fallback"]]
    common = {
        "n_dims": selected["n_dims"],
        "method_chain": method_chain,
    }
    phase_c = report.get("phase_c")
    if phase_c is not None and not isinstance(phase_c, dict):
        raise ValueError("phase_c must be an object")
    if phase_c is None or "stages" not in phase_c:
        stages = []
    else:
        stages = phase_c["stages"]
    if stages == []:
        return {
            **common,
            "action": "run",
            "method": method_chain[0],
            "reason": "phase_c_not_started",
            "bout_index": 0,
        }
    if not isinstance(stages, list) or not all(
        isinstance(stage, dict) for stage in stages
    ):
        raise ValueError("phase_c.stages must be a list of objects")
    from _common import stages_by_bout

    bouts = stages_by_bout(stages)
    for bout in bouts:
        bout_methods = [stage.get("method") for stage in bout]
        if (
            len(bout) > len(method_chain)
            or bout_methods != method_chain[:len(bout)]
        ):
            raise ValueError(
                f"Phase-C method chain {bout_methods!r} does not match "
                f"{method_chain!r}"
            )
        if any(
            stage.get("status") != "rejected"
            or stage.get("trials") not in (None, [])
            for stage in bout[:-1]
        ):
            raise ValueError(
                "every Phase-C stage before its bout's active/final stage "
                "must be an empty rejected stage"
            )

    current = bouts[-1]
    bout_index = len(bouts) - 1
    methods = [stage.get("method") for stage in current]
    final_stage = current[-1]
    final_status = final_stage.get("status")
    if final_status == "running":
        scope, _detail = _unresumable_budget_scope(candidate_path)
        if scope is not None:
            # The budget proves no invocation can ever resume this stage, so
            # "run" advice cannot succeed; the deterministic close is the only
            # legal move (prompt-only routing here stranded durable trials).
            return {
                **common,
                "action": "close_exhausted_stage",
                "method": methods[-1],
                "reason": "evaluation_budget_reached",
                "budget_scope": scope,
                "bout_index": bout_index,
            }
        return {
            **common,
            "action": "run",
            "method": methods[-1],
            "reason": "resume_interrupted_stage",
            "bout_index": bout_index,
        }
    if final_status == "rejected":
        if final_stage.get("trials") not in (None, []):
            raise ValueError("a rejected Phase-C stage cannot contain trials")
        if len(current) < len(method_chain):
            return {
                **common,
                "action": "run",
                "method": method_chain[len(current)],
                "reason": "run_deterministic_fallback",
                "bout_index": bout_index,
            }
        if applied_close:
            return _start_new_bout(common, method_chain, bout_index)
        result = finalizable_tuning_result(report)
        return {
            **common,
            "action": "finalize",
            "method": None,
            "reason": "method_chain_exhausted",
            "best_score": result["best_score"],
            "bout_index": bout_index,
        }
    if applied_close:
        return _start_new_bout(common, method_chain, bout_index)
    result = finalizable_tuning_result(report)
    return {
        **common,
        "action": "finalize",
        "method": None,
        "reason": f"terminal_{final_status}",
        "best_score": result["best_score"],
        "bout_index": bout_index,
    }


def _start_new_bout(common: dict, method_chain: list, bout_index: int) -> dict:
    """The previous bout is closed and finalized; begin the next one."""
    return {
        **common,
        "action": "run",
        "method": method_chain[0],
        "reason": "start_new_bout",
        "bout_index": bout_index + 1,
    }
```

- [ ] **Step 10: Run the new tests plus the full governance/patience files**

Run: `python -m pytest tests/test_tuner_patience.py tests/test_deep_tune_governance.py -q`
Expected: PASS. If `test_wrong_fallback_and_terminal_rerun_are_rejected_before_admission` fails on the message regex, update its expected text: re-admitting a bout's terminal primary method after a finalized close is now refused with "must be finalized" when no applied close exists, and "cannot be rerun within its bout" inside an open bout — read that test and align the regex with the new messages (the rejection-before-mutation invariant is unchanged).

- [ ] **Step 11: Commit**

```bash
git add tools/tuners/_common.py tools/tuners/tune_tools.py tests/test_deep_tune_governance.py tests/test_tuner_patience.py
git commit -m "feat: admit progressive tuning bouts into the Phase-C stage model"
```

---

### Task 5: Cross-bout finalization and graded depth

**Files:**
- Modify: `tools/tuners/tune_tools.py` (`finalizable_tuning_result` :1062, `summarize` :2469, `tuning_record` :2537, `finalized_tuning_record` :2621; new `_evaluation_depth`, `_last_bout_improved`, `load_tuned_threshold`)
- Modify: `tools/ledger.py` (`_finalized_tuning_record_from_report` :738)
- Modify: `tools/finalize_tuning.py` (close stamping, ~line 218)
- Test: `tests/test_tuning_finalization.py`

**Interfaces:**
- Consumes: `stages_by_bout` / `stage_bout_index` (Task 4, `_common.py`); ledger fields `tuning_bouts` / `last_bout_improved` (Task 3); `DEFAULT_TUNED_THRESHOLD` (Task 2).
- Produces: `finalized_tuning_record(report, *, tuned_threshold=DEFAULT_TUNED_THRESHOLD)` returning graded `evaluation_depth`, `tuning_bouts`, `last_bout_improved`; `load_tuned_threshold(ledger_path) -> int`; reports stamped with `last_finalized_stage_index` at every close.

- [ ] **Step 1: Write the failing tests**

In `tests/test_tuning_finalization.py`, add to `TuningFinalizationTests` (uses the existing `_fixture`, which writes one grid stage with one trial scoring 0.8 vs warm 1.0):

```python
    def test_first_bout_close_records_progressive_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            self.assertEqual(result["final_best_score"], 0.8)
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["tuning_bouts"], 1)
            self.assertIs(record["last_bout_improved"], True)
            self.assertEqual(record["evaluation_depth"], "tuned_lightly")
            self.assertIs(record["tune"], True)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["last_finalized_stage_index"], 0)

    def test_second_bout_close_recovers_best_across_bouts(self) -> None:
        """A continuation bout closes against every stage, not just its own."""
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"].append(
                {
                    "method": "grid",
                    "bout_index": 1,
                    "status": "ok",
                    "trials": [{"params": {"x": 0.7}, "score": 0.7}],
                    "elapsed_seconds": 1.0,
                }
            )
            report_path.write_text(json.dumps(report, indent=2))
            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            self.assertEqual(result["final_best_score"], 0.7)
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["tuning_bouts"], 2)
            self.assertIs(record["last_bout_improved"], True)
            self.assertEqual(record["evaluation_depth"], "tuned_lightly")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["last_finalized_stage_index"], 1)
            self.assertIn("'x': 0.7", candidate_path.read_text())

    def test_non_improving_bout_marks_non_responder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"].append(
                {
                    "method": "grid",
                    "bout_index": 1,
                    "status": "ok",
                    "trials": [{"params": {"x": 1.9}, "score": 1.9}],
                    "elapsed_seconds": 1.0,
                }
            )
            report_path.write_text(json.dumps(report, indent=2))
            result = finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            self.assertEqual(result["final_best_score"], 0.8)  # bout 0's best
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["tuning_bouts"], 2)
            self.assertIs(record["last_bout_improved"], False)

    def test_depth_becomes_tuned_at_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path, report_path, ledger_path = self._fixture(
                Path(tmp), stage_status="ok"
            )
            report = json.loads(report_path.read_text())
            report["phase_c"]["stages"][0]["trials"] = [
                {"params": {"x": 1.5 + i / 100.0}, "score": 1.5 + i / 100.0}
                for i in range(16)
            ]
            report_path.write_text(json.dumps(report, indent=2))
            finalize_tuning.finalize(
                candidate_path=candidate_path,
                report_path=report_path,
                ledger_path=ledger_path,
                run_id="001",
                task_name="autoresearch-baseline",
            )
            record = json.loads(ledger_path.read_text())["records"][0]
            self.assertEqual(record["evaluation_depth"], "tuned")
```

Also UPDATE the existing `test_phase_c_losing_to_warm_incumbent_still_records_tuned_depth` (:237): with one scored Phase-C trial the depth is now `tuned_lightly`, not `tuned`. Change the final assertion to `self.assertEqual(record["evaluation_depth"], "tuned_lightly")` and adjust the docstring's last sentence to: depth is now graded — one scored Phase-C trial promotes screening to `tuned_lightly`; `tuned` requires `tuner.tuned_threshold` attempts.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tuning_finalization.py -q`
Expected: FAIL — new keys missing (`tuning_bouts` etc.), old depth assertion fails.

- [ ] **Step 3: Rework `finalizable_tuning_result` for bouts**

In `tools/tuners/tune_tools.py`, inside `finalizable_tuning_result`:

(a) Delete the global "every stage before the final must be rejected" check:

```python
    if statuses:
        for index, status in enumerate(statuses[:-1]):
            if status != "rejected":
                errors.append(
                    f"phase_c.stages[{index}] precedes another stage but is not rejected"
                )
```

Replace with per-bout checks (insert where the deleted block stood; `stages_by_bout` is lazily imported at the top of the function: `from _common import stages_by_bout`):

```python
    bouts = stages_by_bout(stages)
    offset = 0
    for bout in bouts:
        for index, stage in enumerate(bout[:-1]):
            if stage.get("status") != "rejected":
                errors.append(
                    f"phase_c.stages[{offset + index}] precedes another stage "
                    "of its bout but is not rejected"
                )
        offset += len(bout)
```

(b) Replace the method-chain check:

```python
        if methods != expected_methods[:len(methods)]:
            errors.append(
                f"Phase-C method chain {methods!r} does not match deterministic "
                f"chain {expected_methods!r}"
            )
```

with per-bout chain checks:

```python
        for bout in bouts:
            bout_methods = [stage.get("method") for stage in bout]
            if bout_methods != expected_methods[:len(bout_methods)]:
                errors.append(
                    f"Phase-C method chain {bout_methods!r} does not match "
                    f"deterministic chain {expected_methods!r}"
                )
```

(c) Scope `exhausted_rejections` to the LAST bout (a chain rejected within one bout, not the whole history):

```python
    last_bout = bouts[-1] if bouts else []
    last_methods = [stage.get("method") for stage in last_bout]
    last_statuses = [stage.get("status") for stage in last_bout]
    exhausted_rejections = bool(
        expected_methods
        and last_methods == expected_methods
        and last_statuses == ["rejected"] * len(expected_methods)
    )
```

(d) Widen the eligible rows from "final stage only" to every stage. Replace the `finite_final` construction and the `eligible_rows.extend(...)` block with:

```python
    finite_phase_c_rows = [
        (str(stage.get("method")), row)
        for stage in stages
        if isinstance(stage, dict)
        for row in stage.get("trials", [])
        if isinstance(row, dict)
        and isinstance(row.get("params"), dict)
        and _is_finite_score(row.get("score"))
    ]
```

and:

```python
    eligible_rows = []
    if warm_best is not None:
        eligible_rows.append(("warm_start", warm_best))
    # Every finite Phase-C row across ALL bouts competes with the Phase-A
    # incumbent, whatever terminal status its stage carries (see the docstring
    # for why those rows are already proven). A later bout is not trusted to
    # beat earlier ones, so the argmin spans the whole history.
    eligible_rows.extend(finite_phase_c_rows)
```

Caution: the removed `final_trials`/`finite_final` locals may still be referenced by the checks in the compressed regions (e.g. the `time_exhausted` receipt check, the `no_search_needed` validation). Re-point any surviving reference at `stages` / `finite_phase_c_rows`; every check that meant "the final stage" now means "the final stage of the last bout" (`bouts[-1][-1]`), which is the same object as `stages[-1]`.

Keep the docstring but append one paragraph:

```python
    Bouts: stages are grouped by ``bout_index`` (legacy stages: bout 0). The
    per-bout prefix/chain rules mirror the single-pass rules; finalization is
    valid when every stage is terminal and the LAST bout's final stage is
    finalizable (or that bout's whole chain was rejected). The best spans all
    bouts, so a continuation close can never regress the score.
```

Also update the "ok-stage-needs-finite-trial" check (in the compressed region) so it applies to EVERY stage with status `ok`, not only the final stage:

```python
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict) or stage.get("status") != "ok":
            continue
        if not any(
            isinstance(row, dict) and _is_finite_score(row.get("score"))
            for row in stage.get("trials", [])
        ):
            errors.append(f"phase_c.stages[{index}] is ok but has no finite trial")
```

- [ ] **Step 4: Export `phase_c_attempted` from `summarize`**

In `summarize` (:2469), add one key to the returned dict (additive; consumers pick keys):

```python
        "feasibility_rejections": feasibility_rejections,
        "phase_c_attempted": phase_c_attempted,
        "elapsed_seconds": round(elapsed, 1),
```

- [ ] **Step 5: Graded depth and bout fields in the record reducers**

In `tune_tools.py`, add (near `finalized_tuning_record`):

```python
def _evaluation_depth(phase_c_attempts: int, tuned_threshold: int) -> str:
    """Graded evaluation depth: 0 attempts screening, 1..threshold-1 lightly
    tuned, >=threshold fully tuned."""
    if phase_c_attempts <= 0:
        return "screening"
    return "tuned" if phase_c_attempts >= tuned_threshold else "tuned_lightly"


def _last_bout_improved(report: dict) -> bool | None:
    """Whether the last bout produced a trial strictly better than its
    pre-bout incumbent (warm best plus every earlier bout). None when the
    report has no Phase-C stages."""
    from _common import stages_by_bout

    stages = report.get("phase_c", {}).get("stages", [])
    bouts = stages_by_bout(stages)
    if not bouts:
        return None
    phase_a = report.get("phase_a", {})
    warm_best = phase_a.get("best_warm_score")
    prior_best = float(warm_best) if _is_finite_score(warm_best) else None
    for bout in bouts[:-1]:
        for trial in bout:
            for row in trial.get("trials", []):
                if isinstance(row, dict) and _is_finite_score(row.get("score")):
                    score = float(row["score"])
                    if prior_best is None or score < prior_best:
                        prior_best = score
    current_best = None
    for stage in bouts[-1]:
        for row in stage.get("trials", []):
            if isinstance(row, dict) and _is_finite_score(row.get("score")):
                score = float(row["score"])
                if current_best is None or score < current_best:
                    current_best = score
    if current_best is None:
        return False
    return prior_best is None or current_best < prior_best


def load_tuned_threshold(ledger_path: Path) -> int:
    """tuner.tuned_threshold for the run owning this ledger (default 16)."""
    return int(
        _run_cfg(Path(ledger_path), "tuner").get(
            "tuned_threshold", DEFAULT_TUNED_THRESHOLD
        )
    )
```

In `tuning_record`'s returned dict, add the two new TUNING_FIELDS keys with phase-A values (the phase-A path is always pre-tuning):

```python
        "applied": report.get("applied_to_base_params"),
        "tuning_bouts": 0,
        "last_bout_improved": None,
```

Replace `finalized_tuning_record` (:2621-2647) with:

```python
def finalized_tuning_record(
    report: dict,
    *,
    tuned_threshold: int = DEFAULT_TUNED_THRESHOLD,
) -> dict:
    """Ledger-ready tuning fields after the fail-closed completion check."""
    final = finalizable_tuning_result(report, require_applied=True)
    summary = summarize(report)
    stages = report.get("phase_c", {}).get("stages", [])
    from _common import stages_by_bout

    return {
        **tuning_record(report),
        "final_best_score": final["best_score"],
        # Provenance of the applied observation only: None when the Phase-A
        # incumbent wins the argmin, however much Phase C ran.
        "phase_c_method": final["phase_c_method"],
        # Depth is cumulative Phase-C evaluation EFFORT across bouts, not the
        # applied row's provenance: a candidate whose warm incumbent still
        # wins keeps the depth its admitted attempts earned. Graded:
        # screening -> tuned_lightly -> tuned at tuner.tuned_threshold.
        "evaluation_depth": _evaluation_depth(
            summary["phase_c_attempted"], tuned_threshold
        ),
        "tuning_bouts": len(stages_by_bout(stages)),
        "last_bout_improved": _last_bout_improved(report),
    }
```

- [ ] **Step 6: Plumb the threshold through `tools/ledger.py`**

In `_finalized_tuning_record_from_report` (:738), change the import and the `finalized_tuning_record` call:

```python
    from tune_tools import (  # noqa: E402
        finalized_tuning_record,
        load_tuned_threshold,
        validate_report_trial_rows,
    )
```

and:

```python
    ledger_guess = report_path.parent.parent.parent / "ledger.json"
    fields = finalized_tuning_record(
        report,
        tuned_threshold=load_tuned_threshold(ledger_guess),
    )
```

(The report lives at `<run_dir>/candidates/<run_id>/tune_report.json`, so `parent.parent.parent` is the run dir; `load_tuned_threshold` only reads `framework_cfg.json` beside that ledger path and falls back to the default when absent.)

- [ ] **Step 7: Stamp `last_finalized_stage_index` in `tools/finalize_tuning.py`**

In `finalize()`, where `closed_report` is built (after `closed_report["applied_to_base_params"] = True`), add:

```python
    closed_report["last_finalized_stage_index"] = len(
        closed_report.get("phase_c", {}).get("stages", [])
    ) - 1
```

- [ ] **Step 8: Run the full finalization file**

Run: `python -m pytest tests/test_tuning_finalization.py -q`
Expected: PASS. If legacy-close tests fail on the new report key, note: `has_applied_close` treats a missing `last_finalized_stage_index` as "covers all stages", so one-shot reports stay valid; genuine failures are more likely stale assertions about `evaluation_depth` — grep the file for `"tuned"` assertions and align each with graded semantics.

- [ ] **Step 9: Commit**

```bash
git add tools/tuners/tune_tools.py tools/ledger.py tools/finalize_tuning.py tests/test_tuning_finalization.py
git commit -m "feat: finalize tuning bouts against cross-bout best with graded depth"
```

---

### Task 6: Progressive select-candidate

**Files:**
- Modify: `tools/tuners/tune_tools.py` (`select_candidate` :2986, `cmd_select_candidate` :3287)
- Test: `tests/test_deep_tune_governance.py` (new `ProgressiveSelectCandidateTest`)

**Interfaces:**
- Consumes: ledger fields `tuning_bouts` / `last_bout_improved` (Task 3), `DEFAULT_BOUT_TRIALS` (Task 2).
- Produces: `select_candidate(..., bout_trials=None, ...)` returning `bout_index`, `is_continuation`, `tuning_bouts`, `last_bout_improved`, `final_best_score` alongside the legacy fields; null reasons enumerated below.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_deep_tune_governance.py` (imports need `select_candidate` from `tune_tools` — add it to the existing import block):

```python
def _candidate_record(
    run_id: str,
    warm: float,
    *,
    tune: bool = False,
    bouts: int = 0,
    improved: bool | None = None,
    final: float | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "status": "keep",
        "best_warm_score": warm,
        "tune": tune,
        "tuning_bouts": bouts,
        "last_bout_improved": improved,
        "final_best_score": final if final is not None else warm,
    }


class ProgressiveSelectCandidateTest(unittest.TestCase):
    def _ledger(self, records) -> dict:
        return {"records": records}

    def test_first_bout_still_requires_percentile_gate(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)
        self.assertEqual(result["bout_index"], 0)

    def test_continuation_selected_when_fresh_fails_gate(self):
        # Fresh candidates carry the WORST warm scores, so the best fresh
        # warm percentile is 50 (< 80) and the gate refuses a first bout.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00),
            # tuned responder: continuation pools ignore warm rank
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=True, final=0.80),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "005")
        self.assertIs(result["is_continuation"], True)
        self.assertEqual(result["bout_index"], 1)

    def test_fresh_first_bout_beats_continuation(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.40, tune=True, bouts=1,
                              improved=True, final=0.70),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)

    def test_continuation_ranks_fewest_bouts_then_tuned_score(self):
        # Fresh warm scores are worst (best fresh percentile 50 < 80), so the
        # choice falls to continuations: fewer bouts beats better tuned score.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00, tune=True, bouts=2,
                              improved=True, final=0.60),
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=True, final=0.90),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "005")  # fewer bouts wins over better score

    def test_non_responders_never_selected(self):
        # Fresh gate fails (percentile 50) and the only tuned candidate did
        # not improve in its last bout.
        ledger = self._ledger([
            _candidate_record("001", 1.30),
            _candidate_record("002", 1.20),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.00),
            _candidate_record("005", 0.95, tune=True, bouts=1,
                              improved=False, final=0.70),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertIsNone(result["run_id"])
        self.assertIn("non-responder", result["reason"])

    def test_trial_cap_includes_bout_trials(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
        ])
        allocation = {
            "remaining": 100,
            "deep_tune": {
                "remaining": None,
                "total_cap": None,
                "per_candidate_cap": 20,
                "per_candidate": [],
                "time_limit_seconds": None,
            },
        }
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0,
            bout_trials=8, budget_allocation=allocation,
        )
        self.assertEqual(result["run_id"], "001")
        self.assertEqual(result["budget_allocation"]["trial_cap"], 8)
```

Note: `_candidate_record` records deliberately lack `semantic_point`/`policy_receipt`; `select_candidate` must not touch them. If `_has_unresolved_primary_descendant` → `unbound_primary_descendants` requires fields these stubs lack, check its implementation and add the minimal fields it reads (likely `source_run_ids: []`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_deep_tune_governance.py -q -k ProgressiveSelectCandidate`
Expected: FAIL — `select_candidate` got an unexpected keyword `bout_trials`, and continuations are never selected.

- [ ] **Step 3: Rewrite `select_candidate`**

Replace the whole function (:2986-3134) with:

```python
def select_candidate(
    ledger: dict,
    *,
    n_min: int = DEFAULT_N_MIN,
    top_percentile: float = DEFAULT_TOP_PERCENTILE,
    bout_trials: int | None = None,
    budget_allocation: dict | None = None,
) -> dict:
    """Which candidate receives the next tuning bout (progressive §15), or none.

    Eligible: non-crash, finite best_warm_score, lifetime Phase-C attempts
    below the per-candidate cap, no unresolved primary descendant. First
    bouts additionally require the legacy gate: population >= n_min AND the
    best untuned candidate in the top (100-top_percentile)% by warm score.
    Continuations (tuning_bouts >= 1) skip the percentile gate but require
    `last_bout_improved` not False — a non-responder is never re-tuned.

    Ranking is like-for-like: fresh candidates (0 bouts, warm scores) always
    precede continuations; continuations order by (tuning_bouts,
    final_best_score) — fewest bouts first (evidence coverage), then best
    tuned score. Warm and tuned scores are never compared against each
    other. Returns {run_id|None, reason, bout_index, is_continuation, ...}.
    """
    if (
        not isinstance(n_min, int)
        or isinstance(n_min, bool)
        or n_min <= 0
    ):
        raise ValueError("n_min must be a positive integer")
    if (
        not isinstance(top_percentile, (int, float))
        or isinstance(top_percentile, bool)
        or not math.isfinite(float(top_percentile))
        or not 0 <= float(top_percentile) < 100
    ):
        raise ValueError("top_percentile must be finite and in [0, 100)")
    if bout_trials is not None and (
        not isinstance(bout_trials, int)
        or isinstance(bout_trials, bool)
        or bout_trials <= 0
    ):
        raise ValueError("bout_trials must be a positive integer or None")

    cands = [r for r in ledger.get("records", [])
             if r.get("status") != "crash" and _is_finite_score(r.get("best_warm_score"))]
    n = len(cands)
    allocation_receipt = None
    per_candidate_deep: dict[str, int] = {}
    per_candidate_cap = None
    if budget_allocation is not None:
        global_remaining = budget_allocation.get("remaining")
        deep = budget_allocation.get("deep_tune", {})
        deep_remaining = deep.get("remaining") if isinstance(deep, dict) else None
        per_candidate_cap = (
            deep.get("per_candidate_cap") if isinstance(deep, dict) else None
        )
        if isinstance(deep, dict):
            per_candidate_deep = {
                str(row.get("run_id")): int(row.get("evals", 0))
                for row in deep.get("per_candidate", [])
                if isinstance(row, dict)
                and isinstance(row.get("run_id"), str)
                and isinstance(row.get("evals"), int)
            }
        allocation_receipt = {
            "global_remaining": global_remaining,
            "deep_tune_remaining": deep_remaining,
            "deep_tune_total_cap": (
                deep.get("total_cap") if isinstance(deep, dict) else None
            ),
            "deep_tune_per_candidate_cap": per_candidate_cap,
            "deep_tune_time_limit_seconds": (
                deep.get("time_limit_seconds") if isinstance(deep, dict) else None
            ),
        }
        if isinstance(global_remaining, int) and global_remaining <= 0:
            return {
                "run_id": None,
                "n_candidates": n,
                "reason": "evaluation_budget_reached",
                "budget_allocation": allocation_receipt,
            }
        if isinstance(deep_remaining, int) and deep_remaining <= 0:
            return {
                "run_id": None,
                "n_candidates": n,
                "reason": "deep_tune_budget_exhausted",
                "budget_allocation": allocation_receipt,
            }

    if n < n_min:
        return {"run_id": None, "n_candidates": n,
                "reason": f"population {n} < n_min {n_min} (breadth first)"}

    def cap_ok(record: dict) -> bool:
        return (
            not isinstance(per_candidate_cap, int)
            or per_candidate_deep.get(str(record.get("run_id")), 0)
            < per_candidate_cap
        )

    def eligible(record: dict) -> bool:
        return cap_ok(record) and not _has_unresolved_primary_descendant(
            ledger, str(record.get("run_id"))
        )

    fresh = [r for r in cands if not r.get("tune") and eligible(r)]
    continuations = [
        r
        for r in cands
        if r.get("tune")
        and r.get("last_bout_improved") is not False
        and _is_finite_score(r.get("final_best_score"))
        and eligible(r)
    ]
    non_responders = [
        r
        for r in cands
        if r.get("tune") and r.get("last_bout_improved") is False and eligible(r)
    ]

    selected = None
    is_continuation = False
    pct = None
    if fresh:
        best_fresh = min(fresh, key=lambda r: r["best_warm_score"])
        value = best_fresh["best_warm_score"]
        # percentile = fraction of OTHER candidates strictly worse (higher
        # score); high percentile = among the best (warm-vs-warm, including
        # tuned candidates' historical warm scores, as before).
        worse = sum(1 for r in cands if r is not best_fresh and r["best_warm_score"] > value)
        pct = 100.0 * worse / (n - 1) if n > 1 else 100.0
        if pct >= top_percentile:
            selected = best_fresh
    if selected is None and continuations:
        selected = min(
            continuations,
            key=lambda r: (
                int(r.get("tuning_bouts") or 1),
                float(r["final_best_score"]),
            ),
        )
        is_continuation = True

    if selected is None:
        if fresh and pct is not None and non_responders and not continuations:
            reason = (
                f"best untuned percentile {pct:.0f} < {top_percentile:.0f} and "
                "all tuned candidates are non-responders (last bout improved "
                "nothing)"
            )
        elif fresh and pct is not None:
            reason = (
                f"best untuned percentile {pct:.0f} < {top_percentile:.0f} "
                f"(top {100 - top_percentile:.0f}% already tuned)"
            )
        elif non_responders and not continuations:
            reason = (
                "all tuned candidates are non-responders (last bout improved "
                "nothing); no first-bout candidate available"
            )
        elif any(not cap_ok(r) for r in cands) and not any(
            cap_ok(r) for r in cands
        ):
            reason = "all candidates reached deep-tune per-candidate cap"
        elif cands:
            reason = "all candidates have unresolved primary descendants"
        else:
            reason = "no candidates"
        result = {"run_id": None, "n_candidates": n, "reason": reason}
        if pct is not None:
            result["percentile"] = round(pct)
        if allocation_receipt is not None:
            result["budget_allocation"] = allocation_receipt
        return result

    tuning_bouts = int(
        selected.get("tuning_bouts") or (1 if selected.get("tune") else 0)
    )
    if is_continuation:
        reason = (
            f"continuation: responder with fewest bouts ({tuning_bouts}) and "
            "best tuned score"
        )
    else:
        reason = (
            f"best untuned in top {100 - top_percentile:.0f}% "
            f"(percentile {pct:.0f} >= {top_percentile:.0f})"
        )
    result = {
        "run_id": selected.get("run_id"),
        "reason": reason,
        "is_continuation": is_continuation,
        "bout_index": tuning_bouts,
        "tuning_bouts": tuning_bouts,
        "last_bout_improved": selected.get("last_bout_improved"),
        "best_warm_score": selected.get("best_warm_score"),
        "final_best_score": selected.get("final_best_score"),
        "percentile": round(pct) if pct is not None else None,
        "n_candidates": n,
    }
    if allocation_receipt is not None:
        candidate_used = per_candidate_deep.get(str(selected.get("run_id")), 0)
        caps = [
            value
            for value in (
                allocation_receipt["global_remaining"],
                allocation_receipt["deep_tune_remaining"],
                (
                    per_candidate_cap - candidate_used
                    if isinstance(per_candidate_cap, int)
                    else None
                ),
                bout_trials,
            )
            if isinstance(value, int)
        ]
        result["budget_allocation"] = {
            **allocation_receipt,
            "candidate_attempts": candidate_used,
            "bout_trials": bout_trials,
            "trial_cap": min(caps) if caps else None,
        }
    return result
```

- [ ] **Step 4: Pass `bout_trials` through `cmd_select_candidate`**

In `cmd_select_candidate` (:3287), after the `top_p`/`n_min` resolution, add:

```python
    bout = int(rc.get("bout_trials", DEFAULT_BOUT_TRIALS))
```

and change the `select_candidate(...)` call to pass `bout_trials=bout`:

```python
        result = select_candidate(
            ledger,
            n_min=n_min,
            top_percentile=top_p,
            bout_trials=bout,
            budget_allocation=budget_status(led.parent),
        )
```

- [ ] **Step 5: Run the new tests plus the full file**

Run: `python -m pytest tests/test_deep_tune_governance.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/tuners/tune_tools.py tests/test_deep_tune_governance.py
git commit -m "feat: reselect tuned candidates for progressive bouts like-for-like"
```

---

### Task 7: `validate-proposals` + re-warm consumption in search scripts

**Files:**
- Modify: `tools/tuners/tune_tools.py` (new `validate_proposals`, `cmd_validate_proposals`, parser entry ~:3307)
- Modify: `tools/tuners/_common.py` (new `read_pending_proposals`)
- Modify: `tools/tuners/grid_search.py` (:210-255 intake), `tools/tuners/bo_search.py` (:494-544 enqueue), `tools/tuners/cmaes_search.py` (:335-376 intake, :606 deferred loop)
- Test: `tests/test_deep_tune_governance.py`

**Interfaces:**
- Consumes: `_bounds_violations`, `_schema_accepts_value`, `_read_search_space`, `_read_param_schema` (existing, `tune_tools.py`); `attempted_config_identities`, `cast_params_to_search_space`, `params_identity`, `split_configs_by_space`, `deduplicate_configs` (existing, `_common.py`).
- Produces: `validate_proposals(candidate_path, report_path, proposals) -> {"ok": bool, "proposed_count": int, "accepted": list, "rejected": list}`; CLI `python tools/tuners/tune_tools.py validate-proposals --candidate-path <p> --tune-report-json <r> --proposals-json <f>` (exit 0 iff ≥1 accepted; on acceptance it writes `phase_c.pending_proposals` into the report); `_common.read_pending_proposals(report_path) -> list[dict]`.
- Boundary: proposals are validated against the DECLARED `SEARCH_SPACE`. The preflight-clamped box is enforced at consumption time via the existing `split_configs_by_space` skip (same as deferred configs), with `rewarm_skipped_outside_space` receipts — `validate-proposals` does not re-run the clamp.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_deep_tune_governance.py` (imports: add `validate_proposals` to the `tune_tools` import block; `BoutAdmissionTest._fixture` from Task 4 is reused — place these tests as methods on `BoutAdmissionTest` or a sibling class with the same fixture):

```python
    def test_validate_proposals_accepts_valid_and_rejects_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            proposals = [
                {"x": 0.5},                    # valid, novel
                {"x": 1.0},                    # already attempted (warm row)
                {"x": 9.0},                    # outside SEARCH_SPACE
                {"x": 0.5},                    # duplicate of the accepted one
                "not-a-dict",                  # malformed
            ]
            result = validate_proposals(candidate, report_path, proposals)
            self.assertTrue(result["ok"])
            self.assertEqual(result["proposed_count"], 5)
            self.assertEqual(result["accepted"], [{"x": 0.5}])
            reasons = [r["reason"] for r in result["rejected"]]
            self.assertEqual(
                reasons,
                ["already_attempted", "out_of_space", "already_attempted",
                 "params_must_be_object"],
            )

    def test_validate_proposals_all_rejected_is_not_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            result = validate_proposals(candidate, report_path, [{"x": 1.0}])
            self.assertFalse(result["ok"])
            self.assertEqual(result["accepted"], [])
```

For the CLI write-through, also add:

```python
    def test_validate_proposals_cli_writes_pending_proposals(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate, report_path = self._fixture(Path(tmp))
            proposals_path = Path(tmp) / "proposals.json"
            proposals_path.write_text(json.dumps([{"x": 0.5}, {"x": 9.0}]))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "tuners" / "tune_tools.py"),
                    "validate-proposals",
                    "--candidate-path", str(candidate),
                    "--tune-report-json", str(report_path),
                    "--proposals-json", str(proposals_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(report_path.read_text())
            self.assertEqual(
                report["phase_c"]["pending_proposals"], [{"x": 0.5}]
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_deep_tune_governance.py -q -k "validate_proposals"`
Expected: FAIL — `validate_proposals` does not exist.

- [ ] **Step 3: Add `read_pending_proposals` to `tools/tuners/_common.py`**

Next to `read_deferred_configs` (:1133):

```python
def read_pending_proposals(report_path: Path) -> list[dict]:
    """LLM re-warm configs admitted by `tune_tools.py validate-proposals` for a
    continuation bout. Search scripts attempt them FIRST (before deferred
    configs); they consume the bout's trial budget like any other trial. The
    list is overwritten wholesale by the next validate-proposals call, so
    unconsumed leftovers never leak into a later bout."""
    phase_c = read_tune_report(report_path).get("phase_c", {})
    if not isinstance(phase_c, dict):
        return []
    proposals = phase_c.get("pending_proposals", [])
    if not isinstance(proposals, list):
        return []
    return [p for p in proposals if isinstance(p, dict)]
```

- [ ] **Step 4: Add `validate_proposals` + CLI to `tools/tuners/tune_tools.py`**

New function (place near `check_search_space`):

```python
def validate_proposals(
    candidate_path: Path,
    report_path: Path,
    proposals: list,
) -> dict:
    """Validate LLM re-warm proposals for a continuation tuning bout.

    Deterministic disposal of LLM-proposed configs: each proposal must be a
    param dict with the exact SEARCH_SPACE key set, in-bounds values, and a
    novel config identity (against every attempted config and earlier
    accepted proposals). PARAM_SCHEMA compatibility is implied by the
    in-bounds check for numeric kinds but checked explicitly for
    categoricals. Returns {ok, proposed_count, accepted, rejected}."""
    from _common import (
        attempted_config_identities,
        cast_params_to_search_space,
        params_identity,
    )

    candidate_path = Path(candidate_path)
    report_path = Path(report_path)
    search_space = _read_search_space(candidate_path)
    schema = _read_param_schema(candidate_path)
    seen = attempted_config_identities(report_path, search_space)
    accepted: list = []
    rejected: list = []
    if not isinstance(proposals, list):
        proposals = []
    for index, params in enumerate(proposals):
        if not isinstance(params, dict):
            rejected.append({"index": index, "reason": "params_must_be_object"})
            continue
        violations = _bounds_violations(params, search_space)
        if violations:
            rejected.append(
                {"index": index, "reason": "out_of_space", "violations": violations}
            )
            continue
        schema_bad = [
            key
            for key in search_space
            if key in schema
            and _valid_schema_entry(schema[key])
            and not _schema_accepts_value(schema[key], params[key])
        ]
        if schema_bad:
            rejected.append(
                {"index": index, "reason": "schema_incompatible", "keys": schema_bad}
            )
            continue
        identity = params_identity(
            cast_params_to_search_space(dict(params), search_space)
        )
        if identity in seen:
            rejected.append({"index": index, "reason": "already_attempted"})
            continue
        seen.add(identity)
        accepted.append(params)
    return {
        "ok": bool(accepted),
        "proposed_count": len(proposals),
        "accepted": accepted,
        "rejected": rejected,
    }


def cmd_validate_proposals(args) -> int:
    proposals = json.loads(Path(args.proposals_json).read_text())
    result = validate_proposals(
        args.candidate_path, args.tune_report_json, proposals
    )
    if result["accepted"]:
        from _common import read_tune_report, write_tune_report

        report = read_tune_report(args.tune_report_json)
        report.setdefault("phase_c", {}).setdefault("stages", [])
        report["phase_c"]["pending_proposals"] = result["accepted"]
        write_tune_report(args.tune_report_json, report)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1
```

In `build_parser`, after the `check-search-space` block, add:

```python
    vpr = sub.add_parser(
        "validate-proposals",
        help=(
            "Validate LLM re-warm configs for a continuation bout; on any "
            "acceptance, write the accepted list to phase_c.pending_proposals."
        ),
    )
    vpr.add_argument("--candidate-path", required=True, type=Path)
    vpr.add_argument("--tune-report-json", required=True, type=Path)
    vpr.add_argument("--proposals-json", required=True, type=Path)
    vpr.set_defaults(func=cmd_validate_proposals)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_deep_tune_governance.py -q -k "validate_proposals"`
Expected: PASS.

- [ ] **Step 6: Consume proposals in `grid_search.py`**

In the intake block (:210-255), after `attempted_identities = ...`, insert proposal intake BEFORE the deferred dedup and chain the `seen` set:

```python
    proposals_in_space, proposals_outside = split_configs_by_space(
        read_pending_proposals(args.tune_report_json), search_space
    )
    proposals = [
        cast_params_to_search_space(dict(p), search_space)
        for p in proposals_in_space
    ]
    proposals, proposals_skipped_seen, seen = deduplicate_configs(
        proposals,
        seen=attempted_identities,
    )
```

Then change the deferred dedup to consume `seen=seen` (instead of `seen=attempted_identities`), keep the grid dedup as-is (it already chains), and change:

```python
    param_dicts = deferred + grid_configs
```

to:

```python
    param_dicts = proposals + deferred + grid_configs
```

Extend the `set_stage_meta(..., "grid", status="running", ...)` call with:

```python
                   rewarm_proposals_enqueued=len(proposals),
                   rewarm_skipped_outside_space=len(proposals_outside),
                   rewarm_skipped_already_seen=proposals_skipped_seen,
```

Add `read_pending_proposals` to the `_common` import list in `grid_search.py`.

- [ ] **Step 7: Consume proposals in `bo_search.py`**

Add `read_pending_proposals` to the `_common` imports. After the deferred `split_configs_by_space` (:494) and BEFORE the `_enqueue_unique_deferred(study, deferred_in_space, ...)` call, insert:

```python
    proposals_in_space, proposals_outside = split_configs_by_space(
        read_pending_proposals(args.tune_report_json), search_space
    )
```

Enqueue proposals first by changing the deferred enqueue call to pass the concatenation:

```python
        n_enqueued = _enqueue_unique_deferred(
            study,
            proposals_in_space + deferred_in_space,
            search_space,
            distributions,
        )
```

(Ordering inside `_enqueue_unique_deferred` is preserved, so proposals become the first WAITING trials; both consume `n_trials` via the existing `n_trials = n_trials + n_enqueued`, i.e. bout budget.) Extend the running `set_stage_meta` call with:

```python
        rewarm_proposals_enqueued=len(proposals_in_space),
        rewarm_skipped_outside_space=len(proposals_outside),
```

- [ ] **Step 8: Consume proposals in `cmaes_search.py`**

Add `read_pending_proposals` to the `_common` imports. In the intake (:335-376), BEFORE the deferred dedup, add proposal intake and chain the `seen` set through the deferred dedup so a proposal equal to a deferred config is not attempted twice:

```python
    proposals_in_space, proposals_outside = split_configs_by_space(
        read_pending_proposals(args.tune_report_json), search_space
    )
    proposals_in_space = [
        cast_params_to_search_space(dict(params), search_space)
        for params in proposals_in_space
    ]
    proposals_in_space, proposals_skipped_seen, seen = deduplicate_configs(
        proposals_in_space,
        seen=attempted_identities,
    )
```

Then change the existing deferred dedup from `seen=attempted_identities` to `seen=seen`, and change the deferred evaluation loop header (:606) from:

```python
    for d_params in deferred_in_space:
```

to:

```python
    for d_params in [*proposals_in_space, *deferred_in_space]:
```

(One shared loop: proposals are attempted first, then deferred; both are EXTRA to the CMA-ES evals budget, exactly like deferred today.) Extend the running-stage meta stamp (the `set_stage_meta` call that records `deferred_skipped_outside_space`) with:

```python
        rewarm_proposals_enqueued=len(proposals_in_space),
        rewarm_skipped_outside_space=len(proposals_outside),
        rewarm_skipped_already_seen=proposals_skipped_seen,
```

- [ ] **Step 9: Stamp `bout_index` on every `set_stage_meta` call in the three search scripts**

The admission return (`time_budget = deep_tune_time_budget(...)`) now carries `"bout_index"` (Task 4). In `grid_search.py`, `bo_search.py`, and `cmaes_search.py`, pass `bout_index=time_budget["bout_index"]` as a keyword to EVERY `set_stage_meta(...)` call (the parameter exists since Task 4 Step 5 and only takes effect when a stage is created — i.e. the rejected-before-running path). Enumerate the call sites first:

Run: `grep -n "set_stage_meta" tools/tuners/grid_search.py tools/tuners/bo_search.py tools/tuners/cmaes_search.py`
Expected: every call gains the keyword; none missed.

- [ ] **Step 10: Run the full governance file plus a grid resume smoke test**

Run: `python -m pytest tests/test_deep_tune_governance.py tests/test_tuner_patience.py -q`
Expected: PASS. In particular `test_resumed_grid_does_not_re_evaluate_deferred_or_grid_history` must still pass with the extended intake chain.

- [ ] **Step 11: Commit**

```bash
git add tools/tuners/tune_tools.py tools/tuners/_common.py tools/tuners/grid_search.py tools/tuners/bo_search.py tools/tuners/cmaes_search.py tests/test_deep_tune_governance.py
git commit -m "feat: admit validated LLM re-warm proposals into continuation bouts"
```

---

### Task 8: Depth-aware mechanical evidence gates (plan amendment)

**Why this task exists:** the Task 5 review found that the mechanical evidence
layer still encodes the binary tuned/screening split, which would make the
spec's "≥3 lightly-tuned edges" path unreachable no matter what the prompts
say. This task makes the mechanics depth-aware BEFORE the prompts (Task 9)
describe them.

**Files:**
- Modify: `tools/semantic_evidence.py` (`COVERAGE_KEYS` :58, `_coverage_category` ~:1600, `mechanical_gain_direction` :1636, `evaluation_state` computation ~:1870, three acquisition-role derivations :356/:513/:1807)
- Modify: `tools/search_space_state.py` (transition gates ~:490-511; decision schema version handling :44-50)
- Modify: `tools/ledger.py` (stale field comment :91)
- Modify: `.opencode/rules/ledger.md` (`comparator_covered` definition sentence ~:176)
- Test: `tests/test_semantic_evidence.py`, `tests/test_search_space_state.py`

**Interfaces:**
- Consumes: graded `evaluation_depth` values on ledger records (Task 5: `screening` / `tuned_lightly` / `tuned`).
- Produces: coverage bucket `direct_lightly_tuned_edges`; `comparator_covered` = ≥2 direct edges at `tuned_lightly` or deeper; contradiction-grade bar = ≥2 `direct_tuned_edges` OR ≥3 direct edges at `tuned_lightly`+; decision receipts normalize the new key to 0 for legacy artifacts.

- [ ] **Step 1: Write the failing tests (evidence)**

In `tests/test_semantic_evidence.py` (uses `tests/fixtures.py`; `fixtures.record(...)` already accepts a `depth` parameter — pass `depth="tuned_lightly"` for lightly-tuned children; `fixtures.belief_ledger` builds the ledger). Add tests covering:

```python
    def test_lightly_tuned_edges_have_their_own_bucket(self):
        # a direct comparator whose child is tuned_lightly increments
        # direct_lightly_tuned_edges, not direct_tuned_edges and not
        # direct_noncrash_edges

    def test_comparator_covered_accepts_two_lightly_tuned_edges(self):
        # evaluation_state == "comparator_covered" with 2 lightly-tuned direct
        # edges (and with 1 tuned + 1 lightly-tuned)

    def test_mechanical_gain_direction_depth_bar(self):
        # 2 agreeing tuned controls orient (legacy behavior);
        # 3 agreeing lightly-tuned (or mixed tuned/lightly) controls orient;
        # 2 agreeing lightly-tuned controls abstain

    def test_legacy_coverage_receipts_normalize(self):
        # 4-key and 3-key coverage shapes read forward with
        # direct_lightly_tuned_edges == 0
```

These are skeletons: mirror the existing test setups in the file for building direct-comparator edges (they use `fixtures.record` + `fixtures.attach_matched_transfer`). Each assertion names the exact bucket/state/direction string above.

- [ ] **Step 2: Write the failing tests (transition gates)**

In `tests/test_search_space_state.py`, add gate tests at the transition level (mirror the file's existing deprioritize/prune test setups):

```python
    def test_deprioritize_allows_three_lightly_tuned_edges(self):
        # 0 tuned + 3 lightly-tuned direct edges, unpromising/med+ ->
        # deprioritized accepted

    def test_prune_allows_two_tuned_or_three_lightly(self):
        # prune accepted with 2 tuned (legacy) and with 3 lightly-tuned at
        # high confidence

    def test_two_lightly_tuned_edges_do_not_transition(self):
        # exactly 2 lightly-tuned (0 tuned): comparator_covered but the
        # contradiction bar is unmet -> recommendation forced to stay active
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_semantic_evidence.py tests/test_search_space_state.py -q -k "lightly or depth_bar or normalize"`
Expected: FAIL (unknown bucket name / gates still tuned-only).

- [ ] **Step 4: Implement the depth-aware mechanics in `tools/semantic_evidence.py`**

(a) `COVERAGE_KEYS` (:58) — insert the new bucket after `direct_tuned_edges`:

```python
COVERAGE_KEYS = (
    "direct_tuned_edges",
    "direct_lightly_tuned_edges",
    "direct_noncrash_edges",
    "confounded_noncrash_edges",
    "crash_edges",
)
```

Keep `LEGACY_COVERAGE_KEYS` as-is and extend `normalize_coverage` (:74) so both
legacy shapes (3-key and 4-key) read forward with
`"direct_lightly_tuned_edges": 0` — follow the function's existing
schema-1→schema-2 normalization pattern exactly.

Add next to `MIN_EDGES_PER_TARGET` (:52):

```python
# Contradiction-grade depth bar: >=MIN_EDGES_PER_TARGET direct tuned edges, or
# >=DEPTH_BAR_LIGHT_MIN direct edges at tuned_lightly or deeper.
DEPTH_BAR_LIGHT_MIN = 3


def _contradiction_depth_bar(coverage: dict) -> bool:
    """Whether a coverage receipt clears the contradiction-grade depth bar."""
    tuned = coverage.get("direct_tuned_edges", 0)
    light = coverage.get("direct_lightly_tuned_edges", 0)
    return tuned >= MIN_EDGES_PER_TARGET or (tuned + light) >= DEPTH_BAR_LIGHT_MIN
```

(b) `_coverage_category` (~:1600-1615) — replace the depth check:

```python
        child = records.get(str(receipt.get("child_run_id")), {})
        if child.get("evaluation_depth") == "tuned":
            return "direct_tuned_edges"
        return "direct_noncrash_edges"
```

with:

```python
        child = records.get(str(receipt.get("child_run_id")), {})
        depth = child.get("evaluation_depth")
        if depth == "tuned":
            return "direct_tuned_edges"
        if depth == "tuned_lightly":
            return "direct_lightly_tuned_edges"
        return "direct_noncrash_edges"
```

and update the comment above it: a direct comparator is contradiction-grade
when the child was deep-tuned, intermediate at `tuned_lightly`; legacy
records without `evaluation_depth` read as screening and fail closed.

(c) `mechanical_gain_direction` (:1636) — replace the tuned-only filter:

```python
        # Only tuned-child controls orient a direction: screening-depth
        # controls measure one parameter point and abstain here.
        child = records.get(str(receipt.get("child_run_id")), {})
        if child.get("evaluation_depth") != "tuned":
            continue
```

with depth-aware collection, and the decision rule:

```python
        # Tuned-child controls orient a direction at full weight;
        # tuned_lightly controls orient only in numbers (the depth bar).
        # Screening-depth controls measure one parameter point and abstain.
        child = records.get(str(receipt.get("child_run_id")), {})
        depth = child.get("evaluation_depth")
        if depth not in {"tuned", "tuned_lightly"}:
            continue
```

Collect `oriented_effects` as `(delta, is_tuned)` pairs, then replace the
`len(oriented_effects) < MIN_EDGES_PER_TARGET` gate with:

```python
    strong = [effect for effect, is_tuned in oriented_effects if is_tuned]
    effects = [effect for effect, _ in oriented_effects]
    if not (
        len(strong) >= MIN_EDGES_PER_TARGET
        or len(effects) >= DEPTH_BAR_LIGHT_MIN
    ):
        return "none"
    if all(effect < -1e-12 for effect in effects):
        return "positive"
    if all(effect > 1e-12 for effect in effects):
        return "negative"
    return "none"
```

(d) `evaluation_state` computation (:1870) — replace:

```python
    if coverage["direct_tuned_edges"] >= 2:
        return "comparator_covered"
```

with:

```python
    if (
        coverage["direct_tuned_edges"] + coverage["direct_lightly_tuned_edges"]
        >= MIN_EDGES_PER_TARGET
    ):
        return "comparator_covered"
```

(e) The three acquisition-role derivations (:356-359, :513-516, :1807-1810) —
each computes `comparator_gain` from `coverage["direct_tuned_edges"] >=
MIN_EDGES_PER_TARGET`; replace that conjunct at all three sites with
`_contradiction_depth_bar(coverage)` (at :1807 the variable is
`expected_coverage`).

- [ ] **Step 5: Update the transition gates in `tools/search_space_state.py`**

At :495 and :509, replace:

```python
        and coverage["direct_tuned_edges"] >= 2
```

with the shared bar — import `_contradiction_depth_bar` from
`semantic_evidence` if the module already imports from it (check the import
block first; it is the same dependency direction as the coverage
normalization); otherwise replicate the predicate inline with a
cross-reference comment naming `semantic_evidence._contradiction_depth_bar`
(this mirrors the pre-existing gate-duplication pattern — do not restructure
imports to avoid it).

Bump `DECISION_SCHEMA_VERSION` from 2 to 3, add 3 to
`READABLE_DECISION_SCHEMA_VERSIONS`, and extend the schema comment (:46-50):
schema 3 carries the five-key `comparator_coverage` that split
`direct_lightly_tuned_edges` out of the direct bucket; schema-1/2 receipts
stay valid and normalize forward on read. Make the normalization fill
`direct_lightly_tuned_edges: 0` for 3-key and 4-key receipts.

- [ ] **Step 6: Contract text and stale comment**

(a) `tools/ledger.py:91` — replace the stale comment:

```python
    "evaluation_depth",  # screening | tuned: has a scored Phase-C trial
```

with:

```python
    "evaluation_depth",  # screening | tuned_lightly | tuned: graded by cumulative Phase-C attempts
```

(b) `.opencode/rules/ledger.md` (~:176) — replace the `comparator_covered`
definition fragment "(at least two direct tuned edges)" with "(at least two
direct edges at `tuned_lightly` or deeper)". Do NOT touch
`.claude/rules/ledger.md` (mirror intentionally drifts).

- [ ] **Step 7: Run the new tests plus both full files, then the whole suite**

Run: `python -m pytest tests/test_semantic_evidence.py tests/test_search_space_state.py -q`
Expected: PASS. Then `python -m pytest tests -q` → all green (watch
`test_dag_incremental.py` and any snapshot fixtures that pin coverage key
lists — align stale fixtures, never weaken production guards to accommodate
them).

- [ ] **Step 8: Commit**

```bash
git add tools/semantic_evidence.py tools/search_space_state.py tools/ledger.py .opencode/rules/ledger.md tests/test_semantic_evidence.py tests/test_search_space_state.py
git commit -m "feat: make mechanical evidence gates depth-aware"
```

---

### Task 9: Agent prompts and docs

**Files:**
- Modify: `.opencode/agents/tuner-orchestrator.md`
- Modify: `.opencode/agents/experience-extractor.md`
- Modify: `AGENTS.md`
- Modify: `README_ZH.md` (§5.7)
- Modify: `docs/search-space.md`

**Interfaces:** documentation only; must reflect Tasks 2-8 exactly. No mirror sync (`.claude`/`.kimi` untouched, per Task 1).

- [ ] **Step 1: Rewrite `.opencode/agents/tuner-orchestrator.md` for bouts**

Targeted replacements (keep the rest of the file):

(a) Frontmatter `description:` — replace with:

```yaml
description: Run the deterministic progressive-tuning gate once per round, run at most one
  tuning bout (first bout or continuation) on its selected candidate, apply the best config,
  and persist the tuned score/metadata. A null selection is a valid no-op. Never warm-start,
  select by hand, or run a second bout.
```

(b) The "You are the **decoupled tuning step**" paragraph (:30-35) — replace with:

```text
You are the **decoupled tuning step** of the loop (design §15, progressive).
Once per round you pick **one** candidate from the whole population and run
**one tuning bout** on it in place: a fixed slice of `tuner.bout_trials`
objective attempts (default 8). A first bout deep-tunes a promising untuned
candidate; a continuation bout resumes a tuned candidate that responded to
its last bout. After each bout the candidate is finalized (best-so-far
applied, ledger updated) and stays eligible for later bouts until its
lifetime `tuner.deep_tune_per_candidate_cap` is spent or a bout improves
nothing. **One invocation = at most one bout** (often zero — a valid no-op).
```

(c) Phase S section (:76-113) — replace the description of the output and the gate with:

```text
### Phase S — Select the candidate (the progressive gate)

Pick which candidate gets the next bout — over the **whole population**:

```
python tools/tuners/tune_tools.py select-candidate --ledger <run_dir>/ledger.json
```

It prints `{run_id, reason, is_continuation, bout_index, tuning_bouts,
last_bout_improved, best_warm_score, final_best_score, percentile,
n_candidates, budget_allocation}`. First bouts require the legacy gate:
population (non-crash, has `best_warm_score`) ≥ `N_min` (derived as 5 for
P=80) **and** the best untuned candidate in the top (100−`P`)% by
`best_warm_score`. Continuations skip the percentile gate but require the
candidate's last bout to have improved on its pre-bout incumbent
(`last_bout_improved`); a non-responder is never re-tuned. Fresh first bouts
outrank continuations (evidence coverage); continuations rank by fewest
bouts, then best tuned score — warm and tuned scores are never compared
against each other. A candidate with an unresolved primary descendant is
temporarily ineligible. `budget_allocation.trial_cap` is
`min(tuner.bout_trials, per-candidate cap remaining, budget remaining)`.

- **`run_id` is `null`** → no candidate is eligible this round (below
  `N_min`; the top tier is tuned and no continuation responded; every tuned
  candidate is a non-responder; or cap/budget exhaustion). Emit the Output
  Format with `tuned_run_id: none` and `selection_reason` = the printed
  `reason`, then **stop**. This is a valid no-op.
- **`run_id` is a candidate** → that is the bout you run. Derive its paths:

| value | how |
|---|---|
| `run_id` | from select-candidate |
| `candidate_dir` | `<run_dir>/candidates/<run_id>` |
| `candidate_path` | `<candidate_dir>/train.py` |
| `<candidate_dir>/tune_report.json` | has `phase_a` plus any earlier bouts' `phase_c.stages` |

There is **no Phase B** here — the percentile gate moved into
`select-candidate`, which judges the whole population once.
```

(d) Insert a new section between Phase S and Phase C:

```text
### Phase R — Re-warm proposals (continuation bouts only)

When `is_continuation` is true, BEFORE launching the search, read the
candidate's `tune_report.json` trial history and propose up to
`tuner.rewarm_proposals` (default 3) configs that your read of the evidence
says are most promising (near the incumbent's best region unless trials say
it is exhausted; justify each in one line in your working notes). Write them
to a scratch JSON file and validate:

```
python tools/tuners/tune_tools.py validate-proposals \
  --candidate-path <candidate_path> \
  --tune-report-json <candidate_dir>/tune_report.json \
  --proposals-json <scratch proposals.json>
```

Exit 0 → the accepted configs were written to `phase_c.pending_proposals`
and the search scripts attempt them FIRST, inside the bout's `trial_cap`
(they displace search trials, never add to them). Exit 1 → every proposal
was rejected (out-of-space, schema-incompatible, or already attempted);
proceed with the plain search, noting the rejection reasons in your receipt.
Never hand-edit `pending_proposals` or the report yourself. First bouts
never get proposals — they consume the deferred-config supply from step 0+1.
```

(e) Phase C state-machine note — wherever the prompt describes running
`phase-c-action`, add: the action receipt now carries `bout_index`; run the
search with `--n-trials`/`--max-trials` clamped to
`budget_allocation.trial_cap`, not the method defaults, when the cap is
smaller. After a finalized bout, `phase-c-action` returns
`{"action": "run", "reason": "start_new_bout"}` — that is how the NEXT
invocation recognizes a continuation.

(f) Finalize section (:213-248) — replace the sentence "It is idempotent and
prints the receipt fields below." region's description of what the close
does with:

```text
This command first proves that Phase A succeeded and every Phase-C stage is
terminal for the current bout. It then selects the global best across the
warm incumbent AND every Phase-C trial of EVERY bout, AST-rewrites
`BASE_PARAMS`, stamps `last_finalized_stage_index`, closes the report, and
writes the score, keep/discard status, tuning metadata, strict attempt
count, `tuning_bouts`, `last_bout_improved`, graded `evaluation_depth`, and
`tune: true` together through the ledger helper. It is idempotent per bout:
retrying the same close is a no-op, and a later bout's close supersedes it.
```

(g) Boundaries (:297-319) — replace the first two bullets with:

```text
- **One bout per round, chosen by `select-candidate`.** Never override its
  choice, tune a candidate it did not pick, or run a second bout. `null` →
  no-op. A tuned candidate stays eligible: `last_bout_improved` responders
  re-enter the pool, and an ancestor remains eligible after child bindings
  are captured (children stay bound to historical revisions).
- **No warm-start here.** You do not propose or evaluate step-0+1 warm
  configs and do not write `phase_a`. Continuation re-warm proposals (Phase
  R) go only through `validate-proposals`, never by hand.
```

- [ ] **Step 2: Update `.opencode/agents/experience-extractor.md` for graded depth**

(a) Replace the `comparator_covered` paragraph (:107-112) with:

```text
Empty `summary` is a valid abstention. Never call a target `promising` or
`unpromising` unless its mechanical state is `comparator_covered` and it
clears the depth bar: at least two direct **tuned** edges, or at least three
direct edges at `tuned_lightly` or deeper (a `tuned_lightly` child has 1 to
`tuner.tuned_threshold`−1 Phase-C attempts — real but shallow tuning
evidence; screening-only children measure one parameter point and stay
weaker evidence). All weaker/confounded evidence is `mixed` or `unknown`.
```

(b) In the recommendation gates (:189-204), replace every occurrence of the
phrase "at least two direct tuned edges" with "the depth bar (≥2 direct
tuned edges, or ≥3 direct edges at `tuned_lightly` or deeper)". There are
three occurrences: the `deprioritized` bullet, the `pruned` bullet, and the
"Every `promising` or `unpromising` assessment" bullet.

- [ ] **Step 3: Update `AGENTS.md`**

In the subagent table, replace the `tuner-orchestrator` row with:

```markdown
| `tuner-orchestrator` | step 2: progressive tuning gate, then at most one tuning bout (first or continuation) in place | once per round |
```

In the "Step 2 is decoupled" paragraph below the table, replace "every
candidate stops at step 0+1, then `tuner-orchestrator` runs once for the
whole round and picks at most one candidate. A `none` selection is a valid
no-op." with "every candidate stops at step 0+1, then `tuner-orchestrator`
runs once for the whole round and runs at most one tuning bout — a first
bout on a gated untuned candidate or a continuation bout on a responder. A
`none` selection is a valid no-op."

- [ ] **Step 4: Rewrite `README_ZH.md` §5.7**

Replace the whole §5.7 subsection with:

```markdown
### 5.7 tuner-orchestrator

Step 2（解耦渐进式调优，设计 §15）：**每轮在整个运行上运行一次**，每次至多运行**一个调优 bout**（`tuner.bout_trials` 次客观评估，默认 8）。**无热启动**——`phase_a` 是 step 0+1（extractor）输出用作输入。

流程：

1. 选择候选方案：运行 `tools/tuners/tune_tools.py select-candidate`——首个 bout 门控：种群 ≥ `N_min`（P=80 时推导为 5）且按 `best_warm_score` 的最佳未调优候选在前 20%；继续 bout 跳过百分位门控但要求上一 bout 有改进（`last_bout_improved`），无响应者不再调优。首个 bout 优先于继续 bout（证据覆盖优先）；继续 bout 之间按 bout 数最少、再按调优后最佳分数排序（warm 分与调优分互不比较）→ 选择**一个** bout
2. Phase R（仅继续 bout）：orchestrator 依据已有 trial 历史提出至多 `tuner.rewarm_proposals`（默认 3）个配置，经 `tune_tools.py validate-proposals` 确定性校验（在空间内、schema 兼容、去重）后写入 `phase_c.pending_proposals`，由搜索脚本在 bout 预算内**优先**评估
3. Phase C：按维度确定性选择方法（`grid`/`bo`/`cmaes`），以全部历史 trial 为先验续搜
4. Finalize：运行 `tools/finalize_tuning.py`；它验证当前 bout 的 Phase C 已终止，在 warm  incumbent 与**所有 bout 的全部 trial** 上取全局最佳、写回 `BASE_PARAMS`、关闭 report，并一次性更新 ledger（`tuning_bouts`、`last_bout_improved`、分级 `evaluation_depth`；**无重新运行**；可按 bout 安全重试）。若搜索进程被杀死或 report 非终态，则不应用参数且不更新 ledger。

资格不足（种群太小、顶层已调优且无响应继续、或预算/上限耗尽）返回 `none`——有效的无操作。
```

(Note: this also fixes the stale `N_min=10` in the old text — it is derived as 5 for P=80.)

- [ ] **Step 5: Update `docs/search-space.md`**

Read the inner-loop paragraphs (:47-61 and :153-189). Where they describe the inner loop as "extractor warm-start + orchestrator deep tuning" approximating `F(s)` in a one-shot deep tune, adjust to progressive semantics: the inner loop now approximates `F(s)` incrementally — each bout tightens the upper bound `f(x) ≥ F(π(x))`, and `evaluation_depth` (screening / tuned_lightly / tuned) records how tight the bound is, which is what lets lightly-tuned candidates count as intermediate semantic evidence. Keep the two-level model and the parameter-transfer/config-0-control content untouched; this is a framing edit of a few sentences, not a rewrite.

- [ ] **Step 6: Verify no stale wording remains in the touched canonical files**

Run: `grep -rn "at most one candidate\|one-shot" .opencode/agents/tuner-orchestrator.md AGENTS.md README_ZH.md docs/search-space.md`
Expected: no matches (or deliberate historical references only — inspect each).

- [ ] **Step 7: Commit**

```bash
git add .opencode/agents/tuner-orchestrator.md .opencode/agents/experience-extractor.md AGENTS.md README_ZH.md docs/search-space.md
git commit -m "docs: rewrite step-2 contracts for progressive tuning"
```

---

### Task 10: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `python -m pytest tests -q`
Expected: PASS, fast, no GPU/network. Pay special attention to suites that consume ledger records or `evaluation_depth` — `test_semantic_evidence.py`, `test_dag_incremental.py`, `test_parameter_inheritance.py`, `test_tuning_efficacy.py`, `test_search_space_state.py`: a legitimately changed expectation (e.g. depth vocabulary) must be aligned with graded semantics, never deleted to force green.

- [ ] **Step 2: Contract validators**

Run each, expect exit 0:

```bash
python tools/validate_got.py
python tools/validate_search_backends.py
python tools/validate_tasks.py
python tools/validate_background.py
```

- [ ] **Step 3: Drift grep**

Run: `grep -rn "evaluation_depth" docs .opencode/rules AGENTS.md README_ZH.md | grep -v "tuned_lightly" | grep -v specs/ | grep -v plans/`
Expected: every remaining mention is compatible with graded depth (inspect each hit; the spec/plan docs in `docs/superpowers/` are excluded).

- [ ] **Step 4: Final commit (if any alignment fixes were needed)**

```bash
git add -A && git commit -m "fix: align remaining contracts with progressive tuning" || echo "nothing to commit"
```
