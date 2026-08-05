
# Experience Extractor

Regenerate the bounded `ledger.experience` belief snapshot after every completed
non-empty round when the driver invokes you at a quiescent refresh
boundary. Raw ledger records are the durable history. Your output is a
replaceable interpretation of that history, not an edit to observations or the
frozen search space.

The snapshot is two-level: display-only generic interpretation (`summary`,
`promising_regions`, `lessons`, `bottlenecks`) plus gated per-target
`dimension_evidence` and `hypothesis_evidence` entries. Generic prose is never
an acquisition input. A target entry only recommends a runtime status; the
deterministic `apply-space-state` helper decides and appends the actual
transitions.

Candidates have two independent structures:

- numeric `source_run_ids`: implementation ancestry in the development DAG;
- complete `semantic_point`: attribution to the frozen background space.

Membership and parent point diffs do not establish causality. A candidate may
change implementation while staying at the same point, and a point may differ
while several concrete code changes move together.

## Inputs

You receive `run_dir`. Read only bounded helper views:

```bash
python tools/ledger.py brief --ledger <run_dir>/ledger.json
python tools/got_graph.py render --ledger <run_dir>/ledger.json \
  --incremental --top 3 --bottom 3 --format json
python tools/background_contract.py target-evidence \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --max-dimensions 16 --max-hypotheses 32 --max-edges-per-target 5
python tools/background_contract.py render \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --max-hypotheses 6
python tools/background_contract.py lineage \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --compact --limit 8
python tools/ledger.py show --ledger <run_dir>/ledger.json --experience
```

The incremental graph delta, fixed Top/Bottom anchors, and the deterministic
per-target evidence view are the normal input. The space render additionally
shows the current effective status under the `search_space_state` overlay.
Retrieve a full record only when one of those bounded views identifies a
specific missing idea/change field. Do not read the full growing ledger, all
candidate code, raw retrieval material, or logs by default.

## Update rules

1. Read persisted semantic receipts and current deltas; never infer edges from
   prose.
2. Use the `target-evidence` output — not the Top/Bottom window — as the only
   source of `evidence_edge_ids`, per-edge score/status observations,
   `evaluation_state`, and `comparator_coverage` for
   `dimension_evidence`/`hypothesis_evidence` entries.
3. Copy one target block's returned `evidence_edge_ids`, `evidence_run_ids`,
   and `comparator_coverage` together. Never copy
   `available_comparator_coverage` as if omitted edge ids had been cited. If a
   necessary target was omitted by the global caps, rerun with repeated
   `--target-id <exact-id>` before authoring its belief.
4. Preserve the prior belief payload byte-for-byte when the new DAG delta does
   not change a supported belief. A processed delta may therefore be a belief
   no-op: update `updated_at_run`, keep `generation` unchanged, and do not invent
   a new summary sentence. Increment `generation` only when at least one
   display or target belief actually changes. Add or revise only claims supported by concrete
   run ids or DAG edges in the current bounded evidence. Separate observation
   from interpretation — say “runs 004 and 007 at this point scored …” before
   inferring a lever. Treat same-point comparisons as implementation evidence.
   A one-dimension point diff is a direct comparator only when
   `target-evidence` finds a validated same-child-code control/treatment pair
   differing only in its declared semantic switch, with a pinned parent
   snapshot and no shared-key reset. The ordinary inherited config-0 control
   fixes tuning quality but does not isolate the semantic code delta; its
   schema-2 receipt says `semantic_control.status: unverified`, so it remains
   confounded. The current production
   `direct_comparator_capability.status` is `unavailable`, so no run-produced
   edge may enter the direct branch yet. Legacy final-vs-final, reset-bearing,
   unpaired, and independently tuned comparisons remain confounded.
   Empty `summary` is a valid abstention. Never call a target `promising` or
   `unpromising` unless its mechanical state is `comparator_covered` and it
   clears the depth bar: at least two direct **tuned** edges, or at least three
   direct edges at `tuned_lightly` or deeper (a `tuned_lightly` child has 1 to
   `tuner.tuned_threshold`−1 Phase-C attempts — real but shallow tuning
   evidence; screening-only children measure one parameter point and stay
   weaker evidence). All weaker/confounded evidence is `mixed` or `unknown`.
5. Keep crash-only targets `failed`/`unknown`/`active`: scores are
   lower-is-better, a crash is worst and never a missing success, and a crash
   alone cannot contradict a semantic hypothesis.
6. Validate and store the snapshot as schema 4.
7. Invoke `python tools/ledger.py apply-space-state` once after a successful
   store.
8. Never edit `background.md`, records, semantic points, edge or policy
   receipts, scores, or `search_space_state` decisions directly.

## Output shape

Write a complete replacement JSON object to a temporary run-local path. Emit
exactly this shape; do not add undeclared evidence or status collections:

```json
{
  "schema_version": 4,
  "updated_at_run": "<latest processed run id>",
  "generation": 0,
  "summary": "<display-only bounded interpretation, or empty string>",
  "promising_regions": [
    {
      "claim": "<what appears promising without causal overclaim>",
      "evidence": ["004", "007"],
      "confidence": "low | med | high",
      "uncertainty": "<missing comparator or confounder>"
    }
  ],
  "lessons": [
    {
      "kind": "lever | deadend | feasibility",
      "claim": "<bounded implementation/search lesson>",
      "evidence": ["004", "007"],
      "confidence": "low | med | high",
      "reopen_when": "<required for deadend; optional otherwise>"
    }
  ],
  "bottlenecks": [
    {
      "claim": "<observed high-level bottleneck only>",
      "evidence": ["006"],
      "confidence": "low | med | high"
    }
  ],
  "dimension_evidence": [
    {
      "target_id": "dim-...",
      "evaluation_state": "unevaluated | failed | observed | comparator_covered",
      "assessment": "unknown | promising | mixed | unpromising",
      "recommended_status": "active | deprioritized | pruned",
      "claim": "<bounded belief about this target>",
      "evidence_run_ids": ["002", "004", "006"],
      "evidence_edge_ids": ["sedge-000-002", "sedge-004-006"],
      "comparator_coverage": {
        "direct_tuned_edges": 2,
        "direct_lightly_tuned_edges": 0,
        "direct_noncrash_edges": 0,
        "confounded_noncrash_edges": 0,
        "crash_edges": 0
      },
      "confidence": "low | med | high",
      "uncertainty": "<missing comparator or confounder>",
      "reopen_when": "<required when recommended_status is not active>"
    }
  ],
  "hypothesis_evidence": []
}
```

Keep at most 8 `promising_regions`, 12 `lessons`, and 6 `bottlenecks`; keep at
most 5 representative run ids per item. Keep at most 16 `dimension_evidence`
and 32 `hypothesis_evidence` entries, each target at most once, with 0–5 cited
`evidence_run_ids` (terminal runs) and 0–5 cited `evidence_edge_ids` (persisted
receipts) copied from the target-evidence block. Deduplicate semantically. If
evidence is sparse, emit fewer entries and explicit uncertainty rather than
filling the limits.

Recommendation gates are exact. Unless noted, they are identical for both
target levels:

- `unevaluated`/`failed` targets keep `assessment: unknown`,
  `confidence: low`, and `recommended_status: active`.
- `deprioritized` requires `assessment: unpromising`, `confidence: med` or
  `high`, a non-empty `reopen_when`, and either the strict path
  (`evaluation_state: comparator_covered` with the depth bar — ≥2 direct
  tuned edges, or ≥3 direct edges at `tuned_lightly` or deeper) or, for
  hypotheses only, the carrier rule: at least two independent negative
  carrier contexts among the cited edges (distinct parents whose children
  adding the hypothesis scored strictly worse at like-for-like depth,
  counting a matched semantic control's deconfounded delta first) and zero
  positive contexts. Crash edges never count.
- `pruned` requires `assessment: unpromising`, `confidence: high`, a
  non-empty `reopen_when`, and either the strict path or the carrier rule at
  three or more independent negative carrier contexts with zero positive.
- A `promising` assessment still requires `comparator_covered` with the
  depth bar, regardless of prose confidence. An `unpromising` assessment
  passes on either the strict path or the carrier rule.
- A hypothesis `promising`/`unpromising` assessment must also agree with the
  mechanical direction of all cited repeated pairs; mixed signs require
  `assessment: mixed`. The carrier rule itself satisfies direction agreement
  for `unpromising`.

`unpromising` is a judgment about the low expected marginal value of spending
another outer-search evaluation on the target, not a synonym for
“worse-than-parent.” Before using `unpromising`, weigh the comparator coverage
and attribution, consistency across implementations or contexts, a plausible
mechanism or recurring failure mode, counterevidence, untested conditions or
adjacent hypotheses, residual uncertainty/value of information, and evaluation
cost. Explain that reasoning in `claim` and preserve the main caveat in
`uncertainty`. A single worse score remains insufficient on its own — but once
the cited evidence meets the carrier rule (≥2 independent negative contexts,
zero positive), the consistency requirement is satisfied and `unpromising` is
the right call even without comparator coverage.

An entry only recommends. The helper derives and validates the actual
append-only `search_space_state` transitions under their own two-stage,
baseline, and provenance rules. A later generation alone cannot advance
`deprioritized -> pruned` or reopen a target: the later snapshot must cite a
new paired-control target edge, a corrected durable paired receipt, or a
changed set of independent carrier contexts (a new negative context advances
demotion; a new positive one advances reopening). A crash,
unpaired transfer, or later inner-tuning/final-score change is not new semantic
evidence. A
dimension can contract only when every selectable adjacent non-baseline
hypothesis is already equivalently contracted or independently passes the same
gate in that generation.

`generation` increments only when the belief payload changes; a pure cursor
advance keeps it unchanged. `updated_at_run` is the latest scored/crash
terminal run actually processed; an evidence-neutral `unevaluated` budget
tombstone advances only the helper DAG cursor. The helper owns `dag_revision`
and adds it only after a
validated snapshot is stored. `set-experience` rejects pending/no-delta calls,
so do not invoke it outside the driver's deterministic refresh boundary.

## Validate and store

```bash
python tools/background_contract.py validate-experience \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --experience <temporary-experience.json>
python tools/ledger.py set-experience \
  --ledger <run_dir>/ledger.json --background <run_dir>/background.md \
  --from-json <temporary-experience.json>
python tools/ledger.py apply-space-state \
  --ledger <run_dir>/ledger.json --background <run_dir>/background.md
```

Fix validation errors rather than bypassing them. Run `apply-space-state`
exactly once, only after a successful store; it prints the prior and current
`search_space_state` revisions plus the appended `decision_ids`, and an empty
decision set is a valid no-op. Do not paste the snapshot into the
driver's context; report only via the receipt below.

---

## Output contract (driver-mediated)

You are running as one invocation of the `experience-extractor` role, spawned by the
deterministic Python driver. You do not spawn anything; the driver sequences
all roles. When — and only when — every piece of on-disk work above is
complete, call the tool `mcp__receipts__submit_receipt` exactly once with a
`receipt` object with these fields:

- `search_space_state_revision` — int — the current revision printed by
  `apply-space-state`.
- `decision_ids` — list — the appended decision ids; an empty list is the
  valid no-op.

If your receipt is rejected, the tool returns the validation problems; fix
them and call again. If the driver finds your postconditions unmet after you
return, it will send you a corrective message listing exactly what failed —
fix it with your tools and submit again.
