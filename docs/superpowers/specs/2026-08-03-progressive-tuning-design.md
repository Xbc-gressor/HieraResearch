# Progressive tuning — design

Date: 2026-08-03
Status: approved in brainstorming; pending spec review

## Motivation

The current step-2 schema deep-tunes at most one candidate per round in a
single one-shot bout (method chain + patience, lifetime cap 20 attempts).
Three structural problems:

1. **Evidence starvation.** `experience-extractor` only grades a semantic
   target when it is `comparator_covered` with ≥2 direct *tuned* edges, and
   screening-only children count as weak evidence. At one deep tune per
   round, strong semantic evidence accumulates too slowly to steer the outer
   search.
2. **One-shot waste.** The full per-candidate budget is committed to
   whichever candidate `best_warm_score` ranks highest at one moment.
   Patience limits losses inside a bout but cannot redirect budget to a
   candidate that is actually responding to tuning.
3. **No use of gathered tuning information.** Later tuning rounds cannot
   exploit what earlier trials learned; the search restarts from the same
   deferred-config supply every time.

## Decisions (from brainstorming)

| question | decision |
|---|---|
| Promotion driver | Score-based re-selection: tuned candidates re-enter the eligibility pool; a deterministic gate picks the next bout |
| Evidence representation | Graded depth levels: `screening` → `tuned_lightly` → `tuned` |
| Bout size | Fixed from config (`tuner.bout_trials`, default 8); lifetime per-candidate cap retained |
| LLM startup configs | Full re-warm per continuation bout, proposed inline by `tuner-orchestrator`, admitted only through the deterministic validation path, consuming bout budget |
| Mirror sync | `.claude` is canonical; `.opencode`/`.kimi` mirrors may drift (workspace AGENTS.md updated first) |
| Light-evidence bar for pruning | ≥2 edges at `tuned`, or ≥3 edges at `tuned_lightly`+ |

## Design

### 1. Core model

The one-shot deep tune is replaced by **bouts**: fixed-size slices of Phase-C
tuning, `tuner.bout_trials` objective attempts each (default 8), up to the
existing lifetime `tuner.deep_tune_per_candidate_cap` (default 20). After
each bout the candidate is finalized-for-now — best-so-far applied to
`BASE_PARAMS`, ledger updated, keep/discard re-derived — but stays eligible
for future bouts.

Step 0+1 (screening, warm configs, deferred supply) is untouched; only step
2 changes. Cadence unchanged: one `tuner-orchestrator` invocation per round,
at most one bout per invocation, `none` still a valid no-op.

### 2. Selection and eligibility

`select-candidate` (`tools/tuners/tune_tools.py`) stops filtering on the
boolean `tune`. A candidate is eligible when it is non-crash, has a finite
best-known score, lifetime Phase-C attempts < cap, and no unresolved primary
descendant (existing rule, unchanged).

Admission gates per pool:

- **First bout (0 completed bouts):** unchanged from today — population
  `n >= n_min`, and the best untuned candidate must sit in the top (100−P)%
  by `best_warm_score` over the population (warm-vs-warm; the denominator
  includes tuned candidates' historical warm scores, as today).
- **Continuation (≥1 completed bout):** attempts < cap plus the
  non-responder rule below. No percentile re-check; pool membership was
  earned at first-bout admission.

Ranking among eligible candidates is pool-ordered, fresh pool first:

- A continuation whose last completed bout produced no trial strictly
  better than its pre-bout incumbent is a **non-responder** and is excluded
  from selection entirely — never re-selected, never re-tuned. Never-tuned
  candidates are responders by definition.
- Fresh candidates (0 completed bouts) rank by `best_warm_score`, best
  first.
- Continuations rank by `(n_bouts, score)`, ascending: fewest completed
  bouts first, then best `final_best_score` within the same bout count.

**Like-for-like guarantee** (workspace AGENTS.md invariant): score
comparisons only ever occur inside an equal `(non_responder, n_bouts)`
class — warm-vs-warm among untuned candidates, tuned-vs-tuned among
candidates with the same bout count. Cross-class ordering uses only
`n_bouts` and the non-responder flag, never scores. An untuned screening
score is never ranked against a tuning-lowered score.

Consequences, all deliberate:

- The whole percentile-admitted cohort receives a first bout before any
  second bout happens (evidence coverage; motivation 1).
- Continuations favor responders with the best tuned scores (exploitation).
- Non-responders are never re-selected; budget is redirected to candidates
  that respond to tuning (motivation 2).

`budget_allocation.trial_cap` becomes
`min(bout_trials, cap_remaining, global_remaining)`.

Null (no-selection) reasons stay explicit for observability; "all
candidates already tuned" is replaced by granular reasons: all eligible
candidates at the lifetime cap, all continuations non-responding while no
first-bout candidate clears the percentile gate, budget reached.

### 3. Report and stage model

`phase_c.stages` stays one append-only list; each stage gains a
`bout_index: int` (default 0 for legacy reports). The "terminal method
cannot be rerun" rule relaxes to *within the same bout* — a new bout
restarts the deterministic method chain. Stage admission
(`deep_tune_time_budget`) accepts `bout_index` and validates that the stage
is the next in chain within its bout.

This is intentionally minimal: the search scripts already resume with full
trial history as priors and dedup by attempted-config identities, so a
continuation bout is mechanically "run the same script again with enlarged
history, new proposals enqueued first." Patience starts fresh per bout
(existing per-stage semantics); with 8-trial bouts, the gate's
non-responder rule replaces patience as the cross-bout early-stop
mechanism.

### 4. Finalization and ledger

`finalize_tuning.py` remains the single close path, now per-bout: it closes
when a new terminal stage exists since the last finalize, rather than
treating every repeat close as an idempotent no-op. The report records
`last_finalized_stage_index` so the finalize helper can distinguish
"nothing new to close" (true no-op) from "new terminal bout to close."
Per-bout close
recomputes best-so-far over the Phase-A warm incumbent plus all Phase-C
trials across all bouts (today's `finalizable_tuning_result` semantics
extended over bouts), applies `BASE_PARAMS`, updates the ledger, and
re-derives keep/discard.

Ledger changes (additive only, per workspace contract rules — no renames,
old fields stay readable):

- New record field `tuning_bouts: int` (default 0).
- `evaluation_depth` extended: `screening` (0 Phase-C attempts) →
  `tuned_lightly` (1 to `tuner.tuned_threshold`−1, default 15) →
  `tuned` (≥ threshold, default 16 = two bouts).
- `tune` retained as a derived boolean (`depth != screening`) for legacy
  readers.
- `phase_c_method` stays scalar — the latest bout's method (method choice
  is deterministic on `n_dims`, so it is stable across bouts).
- `applied_incumbent` snapshot updates per bout; the inheritance-authority
  rule is unchanged (latest finalized applied incumbent; a running bout is
  never an authority). Children bind to historical revisions via the
  existing `_preserve_descendant_bindings`, so parent re-tuning after
  bindings are captured needs no new mechanism.

Old runs normalize on read: `tune: true` records map to depth by their
existing attempt counts; legacy reports default `bout_index` to 0. Run
artifacts are local and disposable (workspace guide), but they must stay
readable.

### 5. Evidence grading

`experience-extractor` re-calibrates on depth:

- `comparator_covered` accepts edges at `tuned_lightly` or deeper (today:
  deep-tuned only — under the old binary, one scored Phase-C trial already
  qualified, so this is a relabeling plus a stronger tier above it).
- `promising`/`unpromising` verdicts and `pruned`/`deprioritized`
  transitions require **≥2 edges at `tuned`, or ≥3 edges at
  `tuned_lightly`+** (deep edges count toward the light quota).
- The belief snapshot states the depth distribution with cited edge ids;
  transitions remain reversible through the `search_space_state` overlay
  and strengthen as deep evidence arrives.

A first bout (8 trials) now upgrades a candidate's evidence grade — this is
what delivers motivation 1.

### 6. Re-warm proposals

Continuation bouts only: the orchestrator reads the candidate's tune
report and proposes `tuner.rewarm_proposals` configs (default 3) with brief
rationale, informed by trial history. A new `tune_tools.py
validate-proposals` subcommand runs them through the same gauntlet as warm
configs — in-space, clamped to the preflight-feasible box, deduped against
attempted-config identities — and enqueues survivors first in the bout.
Rejections carry per-config reason receipts recorded on the bout's first
stage.

Proposals consume bout budget: they displace search trials inside
`trial_cap`, never add to it. First bouts still consume the deferred-config
supply from step 0+1. LLM proposes, deterministic machinery disposes; the
orchestrator's permission closure is unchanged. The orchestrator never
proposes configs for first bouts and never edits `SEARCH_SPACE` or
`make_model`.

### 7. Budget accounting

No structural change. `phase_c` reservations in `evaluation_attempts.jsonl`
are unchanged; the lifetime per-candidate cap already accumulates across
invocations and now simply spans bouts; the optional run-level
`deep_tune_budget_fraction` stays as-is (default null). `bout_trials` only
shapes `budget_allocation.trial_cap` per invocation. `got_select`'s
screening-admission cap is untouched.

New config knobs (`tasks/framework_cfg.example.json`, `tuner` section):
`bout_trials: 8`, `tuned_threshold: 16`, `rewarm_proposals: 3`.

### 8. Contracts, docs, tests

**Step 0 (before any implementation):** update `/home/woden/spark/AGENTS.md`
and its twin `/home/woden/spark/CLAUDE.md` (its own rule: apply edits to
both) — relax the "keep mirrored runtime contracts synchronized" line to
name `.claude` as canonical, and note progressive tuning as the step-2
schema so the workspace guide cannot contradict the work.

Agent contracts (`.claude` canonical; `.opencode`/`.kimi` intentionally not
re-synced):

- `tuner-orchestrator.md`: rewritten around bouts — "one invocation = at
  most one bout"; Phase S returns `{run_id, bout_index, is_continuation,
  trial_cap}`; continuation context includes the prior-bout summary and the
  re-warm proposal step via `validate-proposals`; per-bout finalize;
  boundaries per §6.
- `experience-extractor.md`: depth-aware grading per §5.
- Unchanged: `tunable-contract-extractor.md`, `idea-generator.md`, the
  primary loop contract.

Docs: repo `AGENTS.md` (subagent-table row and any depth-flag wording),
`README_ZH.md` §5.7, `docs/search-space.md` inner-loop paragraph,
`rules/ledger.md` in the canonical runtime (new fields).

Tests (extend existing files; no new validators, per workspace change
discipline):

- `tests/test_deep_tune_governance.py`: bout re-admission (terminal method
  rerunnable in a new bout, still refused within the same bout), lifetime
  cap spanning bouts, `close_exhausted_stage` at cap still finalizes.
- `tests/test_tuning_finalization.py`: per-bout delta close, depth
  transitions (`screening` → `tuned_lightly` → `tuned`), derived `tune`
  flag, legacy normalize-on-read, non-responder ranking,
  `validate-proposals` (in-space/clamp/dedup/rejection receipts), proposals
  consuming bout budget.
- `tests/test_tuner_patience.py`: patience resets per bout.

### 9. Out of scope

No change to screening (`K`, `K_eval`), no multi-bout rounds, no adaptive
bout sizing, no LLM steering of the numeric search itself, no mirror
re-sync, no changes to the hillclimb baseline.

## Verification

- `python -m pytest tests -q` green (fast, no GPU, no network).
- `python tools/validate_got.py` and `python tools/validate_search_backends.py`
  (ledger/graph and tuner contracts are touched).
- `python tools/validate_tasks.py` / `validate_background.py` only if their
  contracts are touched.
- Empirical effect (evidence coverage per round, score vs. hillclimb) is
  judged later on real runs under the workspace phase goal — it is not a
  gate for this refactor landing.
