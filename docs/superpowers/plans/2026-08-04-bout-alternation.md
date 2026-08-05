# Bout Alternation + Knob Retune Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Guarantee responders a follow-up bout (fresh/responder alternation) and retune progressive-tuning knobs (bout 10, cap 40, threshold 20).

**Architecture:** Per `docs/superpowers/specs/2026-08-04-bout-alternation-design.md`. One new optional input to `select_candidate`, derived by the CLI from `evaluation_attempts.jsonl`; default-value changes in two files; doc/reference updates.

**Tech Stack:** Python 3 stdlib; unittest under pytest.

## Global Constraints

- Scores lower-is-better. No ledger contract change. No new dependencies.
- `.claude/` is canonical; workspace `AGENTS.md`/`CLAUDE.md` edits apply to both.
- Unchanged on purpose: the percentile gate, the strict any-improvement responder rule.

---

### Task 1: Alternation rule in select_candidate

**Files:**
- Modify: `tools/tuners/tune_tools.py` (`select_candidate` line 3229, `cmd_select_candidate` line 3626)
- Test: `tests/test_deep_tune_governance.py` (`ProgressiveSelectCandidateTest`, `_candidate_record` helper at ~line 1255)

**Interfaces:**
- Produces: `select_candidate(..., last_bout_was_first: bool | None = None)`;
  `_last_bout_was_first(run_dir: Path, ledger: dict) -> bool | None` (new helper).

- [ ] **Step 1: Write the failing tests**

Add to `ProgressiveSelectCandidateTest` in `tests/test_deep_tune_governance.py`:

```python
    def test_alternation_responder_follows_first_bout(self):
        # After a first bout, the waiting responder wins even though the best
        # fresh candidate passes the percentile gate.
        ledger = self._ledger([
            _candidate_record("001", 0.90),                              # fresh, passes gate
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=1, improved=True, final=0.94),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=True
        )
        self.assertEqual(result["run_id"], "006")
        self.assertIs(result["is_continuation"], True)
        self.assertIn("alternation", result["reason"])

    def test_alternation_fresh_follows_continuation(self):
        # Same population, but the last bout was a continuation: the
        # gate-passing fresh candidate wins.
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=2, improved=True, final=0.94),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=False
        )
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)

    def test_alternation_none_preserves_legacy_order(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=1, improved=True, final=0.94),
        ])
        result = select_candidate(ledger, n_min=5, top_percentile=80.0)
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)

    def test_alternation_without_responder_falls_back_to_fresh(self):
        ledger = self._ledger([
            _candidate_record("001", 0.90),
            _candidate_record("002", 1.00),
            _candidate_record("003", 1.10),
            _candidate_record("004", 1.20),
            _candidate_record("005", 1.30),
            _candidate_record("006", 0.95, tune=True, bouts=1, improved=False, final=0.95),
        ])
        result = select_candidate(
            ledger, n_min=5, top_percentile=80.0, last_bout_was_first=True
        )
        self.assertEqual(result["run_id"], "001")
        self.assertIs(result["is_continuation"], False)
```

And a derivation test class at module level:

```python
class LastBoutWasFirstTest(unittest.TestCase):
    def _attempts(self, root: Path, rows: list[dict]) -> Path:
        path = root / "evaluation_attempts.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_last_finalized_first_bout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._attempts(root, [
                {"kind": "score_attempt", "phase": "phase_a", "run_id": "001"},
                {"kind": "score_attempt", "phase": "phase_c", "run_id": "001"},
            ])
            ledger = {"records": [
                {"run_id": "001", "status": "keep", "tune": True, "tuning_bouts": 1},
            ]}
            self.assertIs(_last_bout_was_first(root, ledger), True)

    def test_last_finalized_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._attempts(root, [
                {"kind": "score_attempt", "phase": "phase_c", "run_id": "001"},
            ])
            ledger = {"records": [
                {"run_id": "001", "status": "keep", "tune": True, "tuning_bouts": 2},
            ]}
            self.assertIs(_last_bout_was_first(root, ledger), False)

    def test_in_flight_bout_yields_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._attempts(root, [
                {"kind": "score_attempt", "phase": "phase_c", "run_id": "001"},
            ])
            ledger = {"records": [
                {"run_id": "001", "status": "keep"},  # no tune flag yet
            ]}
            self.assertIsNone(_last_bout_was_first(root, ledger))

    def test_missing_attempts_file_yields_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_last_bout_was_first(Path(tmp), {"records": []}))
```

Add `_last_bout_was_first` to the `from tune_tools import (...)` list.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_deep_tune_governance.py -q -k "alternation or LastBoutWasFirst"`
Expected: FAIL (`TypeError: select_candidate got an unexpected keyword argument` / `ImportError`).

- [ ] **Step 3: Implement**

In `tools/tuners/tune_tools.py`:

1. New helper before `select_candidate`:

```python
def _last_bout_was_first(run_dir: Path, ledger: dict) -> bool | None:
    """Whether the run's last finalized bout was a first bout, if knowable.

    Derived from the last phase_c score attempt in evaluation_attempts.jsonl
    plus the ledger's tuning_bouts. None when no bout has run, the attempts
    file is missing, or the last bout has not finalized (in flight).
    """
    path = Path(run_dir) / "evaluation_attempts.jsonl"
    if not path.is_file():
        return None
    last_run_id = None
    try:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(row, dict)
                and row.get("kind") == "score_attempt"
                and row.get("phase") == "phase_c"
                and isinstance(row.get("run_id"), str)
            ):
                last_run_id = row["run_id"]
    except OSError:
        return None
    if last_run_id is None:
        return None
    record = next(
        (
            item
            for item in ledger.get("records", [])
            if isinstance(item, dict) and str(item.get("run_id")) == last_run_id
        ),
        None,
    )
    if not isinstance(record, dict) or not record.get("tune"):
        return None  # bout still in flight, or pre-progressive record
    return int(record.get("tuning_bouts") or 1) <= 1
```

2. `select_candidate` signature: add `last_bout_was_first: bool | None = None`
   after `budget_allocation`. Update the docstring's ranking paragraph:

```
    Ranking is like-for-like and alternates: when the run's last finalized
    bout was a first bout (last_bout_was_first=True), a waiting responder is
    selected before the fresh gate runs; otherwise fresh candidates (0 bouts,
    warm scores) precede continuations. Continuations order by
    (tuning_bouts, final_best_score) — fewest bouts first, then best tuned
    score. Warm and tuned scores are never compared against each other.
```

3. In the body, replace the selection block (currently `selected = None /
   is_continuation = False / pct = None / if fresh: ... / if selected is None
   and continuations: ...`) with:

```python
    selected = None
    is_continuation = False
    pct = None
    alternation = False
    if last_bout_was_first and continuations:
        # Alternation: a completed first bout guarantees a waiting responder
        # the next bout before any new first bout starts.
        selected = min(
            continuations,
            key=lambda r: (
                int(r.get("tuning_bouts") or 1),
                float(r["final_best_score"]),
            ),
        )
        is_continuation = True
        alternation = True
    if selected is None and fresh:
        best_fresh = min(fresh, key=lambda r: r["best_warm_score"])
        value = best_fresh["best_warm_score"]
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
```

   (The moved `fresh` block is unchanged; it is only relocated under the
   alternation check. The existing `if selected is None and continuations:`
   block is unchanged.)

4. In the `is_continuation` reason branch, prepend the alternation receipt:

```python
    if is_continuation:
        reason = (
            f"continuation: responder with fewest bouts ({tuning_bouts}) and "
            "best tuned score"
        )
        if alternation:
            reason = "alternation: responder follows last round's first bout; " + reason
```

5. `cmd_select_candidate`: pass the derived value — after
   `budget_allocation=budget_status(led.parent),` add:

```python
            last_bout_was_first=_last_bout_was_first(led.parent, ledger),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_deep_tune_governance.py -q`
Expected: PASS (whole file).

- [ ] **Step 5: Commit**

```bash
git -C /home/woden/spark/HieraResearch add tools/tuners/tune_tools.py tests/test_deep_tune_governance.py
git -C /home/woden/spark/HieraResearch commit -m "feat: alternate first bouts with responder continuations"
```

---

### Task 2: Knob defaults (bout 10, cap 40, threshold 20)

**Files:**
- Modify: `tools/tuners/tune_tools.py` (`DEFAULT_BOUT_TRIALS` line 3213, `DEFAULT_TUNED_THRESHOLD` line 2726)
- Modify: `tools/evaluation_budget.py` (per-candidate cap default, line ~121)
- Modify: `tasks/framework_cfg.example.json` (values + `_keys`)
- Test: `tests/test_run_cfg.py`, any test asserting 8/16/20 defaults

- [ ] **Step 1: Find every default reference**

Run: `cd /home/woden/spark/HieraResearch && grep -rn 'DEFAULT_BOUT_TRIALS\|DEFAULT_TUNED_THRESHOLD\|deep_tune_per_candidate_cap", 20\|deep_tune_per_candidate_cap.: 20' tools/ tasks/ tests/ | grep -v test_`
Expected: `tune_tools.py` two constants, `evaluation_budget.py` fallback `20`, template three values.

- [ ] **Step 2: Change defaults**

- `DEFAULT_BOUT_TRIALS = 10` (comment: one bout clears TPE's startup regime)
- `DEFAULT_TUNED_THRESHOLD = 20` (two full 10-trial bouts)
- `evaluation_budget.py`: `tuner.get("deep_tune_per_candidate_cap", 40)`
- Template: `"deep_tune_per_candidate_cap": 40`, `"bout_trials": 10`,
  `"tuned_threshold": 20`; update the three `_keys` entries' "(default …)"
  texts (`tuned_threshold`: "default 20 = two full bouts").

- [ ] **Step 3: Update stale test expectations**

Run: `python3 -m pytest tests/ -q 2>&1 | grep FAILED`
Fix each failure that asserts an old default (expected: `test_run_cfg.py`
lines ~34/146/175-178 — `deep_tune_per_candidate_cap: 20` → 40,
`bout_trials: 8` → 10, `tuned_threshold: 16` → 20; possibly
`test_tuning_finalization.py` threshold-boundary cases — those set explicit
values and should be unaffected).

- [ ] **Step 4: Full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C /home/woden/spark/HieraResearch add tools/tuners/tune_tools.py tools/evaluation_budget.py tasks/framework_cfg.example.json tests/
git -C /home/woden/spark/HieraResearch commit -m "feat: retune progressive knobs — bout 10, per-candidate cap 40, tuned threshold 20"
```

---

### Task 3: Prompts, rules, README

**Files:**
- Modify: `.claude/agents/tuner-orchestrator.md` (bout size line ~18, ranking paragraph in Phase S ~lines 74-84)
- Modify: `.claude/rules/ledger.md` (threshold defaults, line ~193)
- Modify: `README_ZH.md` (lines ~145, ~249)

- [ ] **Step 1: tuner-orchestrator.md**

- Intro: "(default 8)" → "(default 10)".
- Phase S ranking text: replace "Fresh first bouts outrank continuations
  (evidence coverage); continuations rank by fewest bouts, then best tuned
  score —" with "First bouts and continuations alternate: after a first
  bout, a waiting responder is selected before the fresh gate runs;
  continuations rank by fewest bouts, then best tuned score —".

- [ ] **Step 2: rules/ledger.md**

"`tuned_lightly` (1 to `tuner.tuned_threshold`−1, default 15)" → "default 19";
"or `tuned` (≥ threshold, default 16)" → "default 20".

- [ ] **Step 3: README_ZH.md**

Update the bout-size and threshold mentions (默认 8 → 默认 10; any 16 → 20).

- [ ] **Step 4: Verify + commit**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/ -q` — PASS.
Run: `grep -rn 'bout_trials.*8\|default 8\|默认 8\|threshold.*16\|默认 16' .claude/ README_ZH.md docs/search-space.md docs/background-research.md` — review each hit is intentional history (dated specs/plans keep their numbers).

```bash
git -C /home/woden/spark/HieraResearch add .claude/ README_ZH.md
git -C /home/woden/spark/HieraResearch commit -m "docs: bout alternation and retuned progressive-tuning defaults"
git -C /home/woden/spark/HieraResearch add docs/superpowers/specs/2026-08-04-bout-alternation-design.md docs/superpowers/plans/2026-08-04-bout-alternation.md
git -C /home/woden/spark/HieraResearch commit -m "docs: bout-alternation design and plan"
```
