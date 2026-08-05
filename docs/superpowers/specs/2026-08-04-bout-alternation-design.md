# Bout alternation + retuned progressive-tuning knobs

Date: 2026-08-04
Status: approved in brainstorming; pre-plan

## Motivation (remoteV evidence)

remoteV (`0804-v4f-pr125-1`, first bout-loop run) showed two allocation
defects:

1. **Continuation starvation.** `select_candidate` lets the best untuned
   candidate that passes the top-percentile gate always outrank
   continuations. Candidates 006/010/011 each improved their first bout and
   never received a continuation, while 003 — still responding after two
   bouts (−0.034 total) — was cut off by the per-candidate cap arithmetic
   (cap 20 ≈ 2.5 bouts at bout size 8).
2. **Shallow bouts.** 8 search trials sit at TPE's startup regime; first
   bouts are rescued by the deferred-config supply (10–11 attempts in
   practice), continuation bouts are not.

What this change does NOT claim: remoteV's final best came from a warm score
(013: 1.0584, zero tuning gain, and its own deferred configs also failed to
beat it — a flat neighborhood, not a too-short bout). Candidate luck and
warm-start quality remain the dominant open question; this change fixes the
evidenced scheduling waste, not the win condition by itself. Deliberately
unchanged: the percentile gate, and the strict any-improvement responder
definition (no underived noise floor; the cap bounds noise-riding).

## Approved shape

- **Strict alternation**: after any first bout, the next bout goes to the
  best waiting responder if one exists; fresh first bouts resume when no
  responder waits.
- **Knobs**: `bout_trials` 8 → 10 (a continuation bout clears TPE startup
  alone), `deep_tune_per_candidate_cap` 20 → 40 (four full bouts),
  `tuned_threshold` 16 → 20 (still exactly two full bouts).

## Design

### 1. Alternation in `select_candidate` (`tools/tuners/tune_tools.py`)

New optional parameter `last_bout_was_first: bool | None = None`:

- `None` (no prior bout, or last bout not yet finalized) → current ordering
  (fresh gate, then continuations).
- `True` → if any responder exists (`tune`, `last_bout_improved` not False,
  finite final score, cap-eligible, no unresolved primary descendant),
  select the best responder (fewest `tuning_bouts`, then best
  `final_best_score`) **before** the fresh gate runs.
- `False` → fresh gate first; continuations only when no fresh candidate
  passes the percentile gate.

The CLI (`cmd_select_candidate`) derives the value from the run's
`evaluation_attempts.jsonl` (sibling of the ledger): take the last
`phase_c` score-attempt row's `run_id`; if that record has no finalized
bout (`tune` falsy — bout in flight), yield `None`; otherwise
`tuning_bouts <= 1` → `True`, `>= 2` → `False`. Missing file → `None`.
No ledger contract change, no new persisted state.

Receipt: `select_candidate`'s printed reason names the alternation when it
fires (e.g. `alternation: responder follows last round's first bout`).

### 2. Knob defaults

- `DEFAULT_BOUT_TRIALS` 8 → 10 (`tune_tools.py`)
- `deep_tune_per_candidate_cap` default 20 → 40 (`evaluation_budget.py`)
- `DEFAULT_TUNED_THRESHOLD` 16 → 20 (`tune_tools.py`)
- `tasks/framework_cfg.example.json` values + `_keys` docs
- Prompt/rules/README references: `.claude/agents/tuner-orchestrator.md`
  (bout size, ranking description), `.claude/rules/ledger.md` (threshold
  defaults), `README_ZH.md` (both)

### 3. Tests

- `tests/test_deep_tune_governance.py`: alternation priority (responder
  beats a gate-passing fresh candidate after a first bout; fresh wins after
  a continuation; `None` preserves legacy order; no responder → fresh as
  usual), the `evaluation_attempts.jsonl` derivation (last-phase_c lookup,
  in-flight bout → `None`), new knob defaults.
- `tests/test_run_cfg.py`: new default expectations.
- Threshold label boundary (`tuned` at 20 attempts) wherever asserted.
