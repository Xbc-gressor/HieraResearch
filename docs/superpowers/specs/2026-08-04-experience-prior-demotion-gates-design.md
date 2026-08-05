# Deterministic experience prior + degraded demotion gates

Date: 2026-08-04
Status: approved in brainstorming; pre-plan

## Motivation (run evidence)

Two autopsied runs on `autoresearch-baseline` (budget 125 each) show the same
failure: **experience is recorded but never consumed.**

- remoteW (`0803-sonnet-ex125-1`): z-loss tried 3× (003, 008, 016),
  AdamW-only 3×, LAWA 2×, CUDA-graph 3×. ~32/125 evals (~26%) re-probed
  already-negative mechanisms. Run 016 re-tried z-loss after a clean isolated
  negative (008) without acknowledging it; run 010 gave an already-negative
  LAWA a 22-eval deep tune (18% of budget).
- remoteV (`0804-v4f-pr125-1`): schedule-free + constant-LR tried 3×
  (002: 2.65, 014: 2.62, 015: 2.92 vs baseline 1.11). 014's code header
  literally cites 002's config.

Mechanism chain (all verified in code):

1. Selection runs the `coverage` policy everywhere (template
   `tasks/framework_cfg.example.json` pins it; the coordinator hardcodes
   `--policy coverage`). Under coverage no predictions load, so every
   receipt's experience components are null
   (`tools/semantic_search.py:1218`, `:1274-1293`, `:1337-1341`).
2. Demotion is unreachable: `direct_comparator_capability: unavailable`
   (`tools/semantic_evidence.py:137-142`) makes every production edge
   "confounded", so `evaluation_state` never reaches `comparator_covered`,
   and `tools/background_contract.py:2291-2327` *rejects* any
   `deprioritized`/`pruned` recommendation the extractor authors.
3. Therefore `derive_experience_transitions`
   (`tools/search_space_state.py:749`) always returns `[]` and
   `search_space_state` stays `{revision: 0, decisions: []}` — confirmed in
   both runs.

## Approved shape (from brainstorming)

- **Directional prior, not magnitude.** Experience contributes a
  deterministic, count-based signed signal per hypothesis — no LLM-authored
  numeric gain estimates.
- **Asymmetric gates.** Demotion (deprioritize/prune) no longer requires
  `comparator_covered`; promotion (`promising`) keeps the strict comparator
  gates. Crash-alone still cannot contradict a mechanism.
- **Old LLM gain-prediction machinery stays but dormant** (`predictions.json`
  path, `gain`/`gain_uncertainty` policies, conditioning validation). The new
  policy becomes the default; the old policies remain selectable.

## Design

### 1. Hypothesis carriers (deterministic, in `tools/semantic_evidence.py`)

For each non-baseline hypothesis `h`, derived fresh from ledger records +
persisted semantic edges at selection time (never trusted from prose):

- **Negative carrier**: an edge whose child point adds `h` relative to its
  parent, the child has a finite score, and the child is strictly worse than
  the parent at like-for-like depth (warm-vs-warm when both exist, else
  final-vs-final; never warm-vs-tuned). If neither comparable pair exists,
  the edge does not count as a carrier in either direction.
- **Positive carrier**: same shape, child strictly better.
- **Independence**: group carriers by parent `run_id`; each distinct parent
  is one independent context.
- **Crash edges never count** (a crash falsifies the implementation, not the
  mechanism). Edges with missing/non-finite scores never count.

Signal per hypothesis: `(n_negative_contexts, n_positive_contexts)`.

### 2. New default selection policy: `coverage_experience`

In `tools/semantic_search.py`:

- `POLICIES` gains `coverage_experience`; it becomes the default in
  `tasks/framework_cfg.example.json`, in `cmd_select`'s fallback
  (`semantic_search.py:1482`), and in the coordinator's hardcoded
  `--policy` (`.claude/agents/autoresearch-experiment.md`).
- Score: `coverage + experience_prior`, where per point
  `experience_prior = min(n_pos, POS_CAP) * pos_weight
  - min(n_neg, NEG_CAP) * neg_weight`, summed over the point's non-baseline
  hypotheses.
- Default weights (config knobs under `semantic_search` in
  `framework_cfg.json`, validated like the existing policy config):
  `pos_weight = 0.05`, `POS_CAP = 2`, `neg_weight = 0.2`, `NEG_CAP = 3`.
  Deliberately asymmetric: re-probing a deadend wastes budget; a missed
  exploration costs nothing measurable. These are explicit policy numbers —
  recorded as such, tunable per run.
- The receipt keeps separate components per point: coverage, per-hypothesis
  carrier counts, prior value. Selection stays fully auditable.
- The prior is computed fresh at `select` time, so it acts immediately —
  it does not wait for the experience-extractor refresh cycle.

### 3. Degraded demotion gates

`tools/background_contract.py::_validate_target_evidence`:

- `deprioritized` becomes valid when **either** the existing strict path
  passes **or** the deterministic carrier rule holds:
  `n_negative_contexts >= 2 and n_positive_contexts == 0`. The validator
  recomputes the counts from edges (like `evaluation_state` today — derived,
  not trusted). `reopen_when` still required.
- `pruned` requires the carrier rule at `n_negative_contexts >= 3 and
  n_positive_contexts == 0` (plus the existing two-stage staging).
- `promising`/`unpromising` assessment gates unchanged.
- `unevaluated`/`failed` forcing unchanged; crash-only targets stay
  `failed` → forced `active` (crash-alone cannot contradict).

`tools/search_space_state.py::_effective_recommendation` mirrors the same
relaxed rule so `derive_experience_transitions` can finally emit
`active→deprioritized` and (with advancing evidence)
`deprioritized→pruned` transitions. Reopen semantics unchanged.

This amends the workspace invariant "strong support or contradiction
requires comparator coverage": **contradiction may now rest on consistent
negative carriers across independent contexts**; support still requires
comparator coverage. Workspace `AGENTS.md`/`CLAUDE.md` get a one-line
update to say so.

### 4. Collapse lane scheduling into the prior

Deprioritized currently means a separate budget lane force-scheduled every
`deprioritized_budget_interval`-th selection
(`semantic_search.py:1297-1322`). With the prior, that is a second, cruder
soft penalty over the same signal.

- Lane scheduling is removed from selection. `pruned` = excluded (hard,
  unchanged, still enforced in `_eligible_hypotheses`/`_add_point`).
  `deprioritized` = eligible; its negative carriers already penalize it via
  the prior.
- `budget_lane` stays in the proposal-set schema for backward readability
  (always `"active"` going forward); it no longer affects selection.

### 5. Dormant machinery (kept, not wired)

`gain` / `gain_uncertainty` / `gain_uncertainty_nocost` policies, the
`predictions.json` validation path, `build_gain_context`, and
`validate_conditioned_adjustment` remain in the tree and tested, but no
default or prompt routes to them. No new LLM-authored numeric input is
introduced anywhere.

### 6. Prerequisites and flagged issues

- **Schema-pin fix**: `semantic_search.py:129`
  (`_experience_snapshot_receipt`) hard-requires experience
  `schema_version == 3` while the extractor writes schema 4
  (`background_contract.py:2087`). Accept `{3, 4}` (the `READABLE` set).
  Latent hard-fail for any path that pins the snapshot.
- **Server code skew**: remoteV's receipts cite schema-4 experience
  generations that this checkout's `select` would reject, so the server ran
  different code. Confirm which checkout the server runs before the next
  experiment, or run diagnosis stays confounded.

### 7. Prompts, rules, docs to update

- `.claude/agents/idea-generator.md`: default policy becomes
  `coverage_experience`; drop the "if absent use coverage" fallback
  wording; keep the dormant gain-context section but mark it non-default.
- `.claude/agents/autoresearch-experiment.md`: replace hardcoded
  `--policy coverage`.
- `.claude/agents/experience-extractor.md` + `.claude/rules/ledger.md`:
  demotion recommendations are now reachable via the carrier rule; document
  the rule (counts, independence, crash exclusion) and that the validator
  recomputes them.
- `tasks/framework_cfg.example.json`: default policy + new weight knobs.
- Workspace `AGENTS.md` / `CLAUDE.md`: the amended contradiction invariant.

### 8. Tests (extend existing files only)

- `tests/test_semantic_evidence.py`: carrier counting (positive/negative,
  parent-grouping independence, crash exclusion, like-for-like depth
  pairing, missing scores).
- `tests/test_search_space_state.py`: demotion transitions fire on the
  carrier rule; mixed-evidence (any positive) blocks demotion; reopen still
  needs advancing evidence; two-stage staging still enforced.
- `tests/test_semantic_policy_default.py`: new default is
  `coverage_experience`; prior appears in receipts with components;
  deprioritized points are penalized, pruned still excluded; dormant
  policies still pass their existing tests.
- `tests/test_background_contract.py`: validator accepts/rejects demotion
  recommendations per the recomputed carrier rule.

## Non-goals

- Continuation scheduling and bout size (remoteV's tuning-ROI questions) —
  deferred, separate change.
- Noise estimation / materiality thresholds beyond "strictly worse" — no
  noise measurements exist yet; counts + zero-contradiction carry the rule.
- Removing the dormant LLM gain machinery.
- Dynamic search-space expansion (P3/P4, still deferred).
