---
name: crash-diagnosis
description: Diagnose ONE autoresearch candidate failure from its traceback and decide how to recover. Read inline by `tunable-contract-extractor` when a no-score candidate preflight or warm config fails, or by the main thread for a legacy official-run crash. Classifies the failure into config-invalid / code-incompatible / abandon and returns the minimal recovery action, preferring a code fix whenever the config is a legitimate hyperparameter value.
metadata:
  short-description: Diagnose one crash → {config_invalid | code_incompatible | abandon} + minimal fix
---

# Crash Diagnosis

Methodology for diagnosing **one** candidate crash and deciding how to recover.
You follow this in your CURRENT context (you are not a separate agent) — the
caller is the context that just ran the candidate and will apply the fix:

- **`tunable-contract-extractor`, segment ③** — a warm config's no-score
  preflight or admitted eval-K call raised; you have the config + its frozen
  `failure_receipt`. A preflight failure consumed no objective slot.
- **the main thread / `autoresearch-experiment` (official run)** — the candidate
  crashed; you have the run log.

One invocation = one crash = one verdict + action.

## What You Read

- The failed config's **`failure_receipt`**, or the last ~80 lines of an official
  run log. Start from the receipt; do not read the full `tune_report.json`.
- The candidate's **`train.py`** in full.
- The **offending config** (the param dict), when diagnosing an eval-K crash.
- `prepare.py` only if the traceback points into it or you need its exports.
- `TASK.md`'s `## Evaluation Contract` + `task.toml` `[constraints]` — whether a
  fix is allowed (readonly files, dependencies) depends on the contract, not
  guesswork.

If the receipt is ambiguous or would force `confidence: low`, retrieve only the
missing evidence using:

```bash
python tools/tuners/tune_tools.py render-failure \
  --tune-report-json <candidate_dir>/tune_report.json \
  --failure-id <failure_id> --view lines --line-range <start:end>
```

Each receipt frame's `traceback_line` is the 1-based line used by
`--line-range`. **Hard gate:** before returning `config_invalid` or `abandon`,
if `failure_receipt.omitted_traceback_lines > 0`, you MUST retrieve the complete
traceback with `--view full` and reconsider the verdict from that evidence.

Outside that hard gate, use `--view full` only when no bounded range can answer
the question. A legacy failed entry with inline `error_traceback` remains
readable, but never regenerate or rewrite a receipt for a failure that already
has `failure_ref`.

## The Verdict (classify into exactly one)

| verdict | meaning | recovery |
|---|---|---|
| **`config_invalid`** | the config **value itself** is intrinsically bad — no reasonable code should make it work (e.g. `max_depth=0`/negative count, an option the algorithm cannot accept). | **change the config** to a valid value (the caller edits `_warm_configs.json`); the code is fine. |
| **`code_incompatible`** | the config is a **legitimate hyperparameter** a practitioner would use, but the code/library can't handle it *as written* (a branch not implemented, a hardcoded size, a dtype/shape mismatch, an option `make_model` doesn't route). | **minimally fix `train.py`** so it handles this value — this is **preferred**: making the code adapt grows the candidate's usable search space. |
| **`abandon`** | the only fix would violate the contract — edit a `readonly_files`, add a forbidden dependency, need a GPU/budget the task doesn't allow, or the idea is fundamentally incompatible. | give up on this candidate. |

**Decision rule — prefer the code fix.** Ask: *is this config value something a
competent practitioner might reasonably try?*
- **Yes** → it should work; the code is the problem → `code_incompatible`, fix the
  code (unless fixing would violate the contract → `abandon`).
- **No, the value is intrinsically nonsensical** → `config_invalid`, fix the
  config.

## Recovery Discipline

- **Minimal fix.** A `code_incompatible` fix handles the crashing value **without
  changing what the candidate does on configs that already worked** and without
  changing the candidate's strategy — add a guard/branch/clamp, don't swap the
  model or rewrite the approach.
- **Respect the contract.** Any fix that needs to edit `prepare.py` or another
  `readonly_files` entry, or add a forbidden dependency, is `abandon` — never do
  it.
- **A `config_invalid` fix** picks the nearest valid value for that key (or drops
  to a safe default), keeping the rest of the config.

## Output (return to your own next step — act on it)

```text
verdict:     config_invalid | code_incompatible | abandon
root_cause:  <one sentence: the actual cause>
evidence:    <one short quote from the traceback>
action:      <config_invalid: key=old → new value
              code_incompatible: the minimal train.py change (1–3 lines)
              abandon: one sentence why no in-contract fix exists>
confidence:  high | medium | low
```

`confidence: low` when the traceback is truncated/ambiguous or the candidate's
intent isn't clear. Use it honestly rather than guessing a fix that may not work.
