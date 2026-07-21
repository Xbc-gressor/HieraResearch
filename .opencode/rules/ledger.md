
# ledger.json Rules

`ledger.json` is the single structured ledger for one
`runs/<task-name>/<tag>/` directory. It replaces the older split between
`idea_log.md` (the per-candidate "why" + tuning summary) and `results.tsv`
(numeric scores + keep/discard/crash). One JSON record per candidate holds
both halves, so there is no cross-file sync to drift.

## Never Hand-Edit

Do **not** edit `ledger.json` by hand and do not write it from an agent with
Write/Edit. It is written only by `tools/ledger.py`, which owns the schema,
the keep/discard/crash decision, `next_run_id`, the Phase B percentile, and
the derived `loop_state.md`. Hand-edits defeat the determinism that keeps a
long autonomous run stable.

Mutate it only through the helper:

- `python tools/ledger.py add-record ... --op <fresh|improve|crossover>` —
  idea-generator creates a candidate record (idea fields + `op` +
  `source_run_ids`; scores/tuning `null`, `status: pending`, `tune: false`).
- `python tools/ledger.py set-tuning ...` — writes tuning **metadata** (only the
  fields passed). The extractor calls it at step 0+1 **without** `--mark-tuned`;
  only the deep-tuner passes `--mark-tuned` (which sets `tune: true`).
- `python tools/ledger.py record-run --final-best-score <v> ...` — owns
  `final_best_score` + `status` (`keep`/`discard`/`crash`). The extractor (step
  0+1, `v = best_warm_score`) and the tuner (deep-tune, `v = tuned best`) call it
  with the `config → score` best — there is no `parse_result` / run log.
- `python tools/ledger.py percentile ...` — read-only cross-idea percentile
  (`best_warm_score`), used by `tune_tools.py select-candidate`'s gate.
- `python tools/ledger.py loop-state ...` — regenerate `loop_state.md`.
- `python tools/ledger.py show ...` — read a record or the whole ledger.

## Shape

```json
{
  "task": "<task-name>",
  "tag": "<tag>",
  "metric": "<result.metric>",
  "dag_revision": 0,
  "records": [ { "<record>" }, ... ],
  "experience": { "<bounded derived snapshot + dag_revision cursor>" }
}
```

Records are ordered by `run_id`. Each record carries every field below;
unavailable fields are `null`, never omitted.

| field | meaning |
|---|---|
| `run_id` | zero-padded id, matches the candidate dir name |
| `kind` | always `optimization` (the seed phase is retired) |
| `op` | `fresh` / `improve` / `crossover` (the S-GoT operation) |
| `idea` | why this candidate exists |
| `source_run_ids` | parent run_ids (`improve`/`crossover`); a `tf-*` direction tag for `fresh`; `[]` for explore |
| `candidate_name` | stable name (hint at add-record; log's `best_model` after the run) |
| `description` | short summary |
| `metric` | `result.metric` |
| `tune` | whether the candidate went through the tuner |
| `status` | `pending` / `keep` / `discard` / `crash` |
| `best_warm_score` | best step-1 warm-start score |
| `final_best_score` | the `config → score` best (= `best_warm_score` at step 0+1, lowered in place when deep-tuned); `+inf` on crash |
| `n_dims` | number of `SEARCH_SPACE` dims |
| `warm_start_K` | warm-start configs evaluated |
| `warm_percentile` | cross-idea percentile of `best_warm_score` |
| `phase_b_decision` | `continue` / `stop` |
| `phase_c_method` | `grid` / `bo` / `cmaes` |
| `trials_completed` | total warm-start + Phase C trials |
| `elapsed_seconds` | tuner wall-clock |
| `applied` | whether tuned params were applied to `BASE_PARAMS` |
| `dag_revision` | helper-owned revision when this node became graph-visible or its score/status changed |

`keep` means `final_best_score` is strictly **smaller** than the best previous
`keep` (scores are always lower-is-better; a crash scores `+inf`). The helper is
the source of truth; if `loop_state.md` disagrees, regenerate it with
`ledger.py loop-state`.

## Direction evidence

The optional top-level `experience.direction_evidence` array joins external
hypotheses from `background.md` to run-local evidence using the stable `tf-*`
tag stored on each fresh root. It keeps two axes separate:

- `literature_credibility` is copied from `background.md` and describes the
  inspected external evidence (`unverified` / `preliminary` / `corroborated` /
  `replicated` / `contested`).
- `run_status` is regenerated from this ledger (`untested` / `inconclusive` /
  `supported_here` / `contradicted_here` / `mixed`).
- Registry v2 also stores `claim_coverage` (`none` / `partial` / `direct`),
  `comparison_runs`, and `missing_comparisons`. A decisive run status requires
  direct coverage of the direction's named comparisons by at least two scored
  non-crash runs; otherwise the status remains `inconclusive`.
- `direct_runs`, `descendant_runs`, and `combination_runs` are bounded
  representative receipts (at most five each), not copies of every historical
  run id. Exhaustive ancestry stays in the immutable records.

`ledger.dag_revision` advances only when a node first receives a result or an
existing node's score/status changes. `ledger.py set-experience` copies the
current revision into `experience.dag_revision` after a validated snapshot is
stored. `got_graph.py render --incremental` uses that cursor to emit only changed
nodes and affected edges plus fixed-size Top/Bottom anchors.

Registry v2 external guidance and direction scopes use the same typed axes.
Selection status is derived by `background_contract.py directions`; a prose
Pitfall cannot block a direction, and scope-mismatched guidance cannot change its
priority. Direct `supported_here` or `mixed` evidence may reopen an external
exclusion without rewriting the external credibility stamp.

Before storing a revised experience snapshot, validate it with
`tools/background_contract.py validate-experience`. This checks that every
direction appears once, the external stamp was not rewritten, and the recorded
direct, descendant, combination, and comparison run ids match scored DAG
records. It cannot infer whether free-form candidate code implements the claimed
arm, so the extractor must check the record semantics. The experience block
remains advisory; candidate records and scores are the hard truth.
