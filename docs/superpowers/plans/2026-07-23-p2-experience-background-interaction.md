# P2 Experience–Background Interaction Implementation Plan

> **STATUS: COMPLETED AND HISTORICAL — DO NOT EXECUTE.** P2 landed; see the
> Current Phase section of the workspace guide. Its mirror-synchronization
> mandate and every `.opencode/` target below are now wrong: `.claude/` is the
> canonical and only maintained runtime, and `.opencode/`/`.kimi/` are
> deprecated mirrors that must not be synced. Read this only as a record of
> what was decided on 2026-07-23.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add evidence-preserving semantic pruning, two-level dimension/hypothesis belief extraction, and persistent semantic DAG edge receipts without mutating the frozen background registry or conflating observations, beliefs, and selection policy.

**Architecture:** Keep `background.md` schema 3 as the immutable definition of the run's semantic search space. Persist mechanically derived semantic edge receipts on candidate records, keep `ledger.experience` as a bounded replaceable belief snapshot, and add an append-only top-level `search_space_state` decision overlay that controls future point eligibility. `semantic_search.py` consumes the overlay at a revision recorded in every proposal and policy receipt; historical points remain valid against the state revision at which they were selected.

**Tech Stack:** Python 3 standard library, JSON/Markdown run contracts, `unittest`, existing HieraResearch deterministic CLI helpers, mirrored Claude/OpenCode agent prompts.

## Global Constraints

- Modify only `HieraResearch/`; never edit benchmark/reference checkouts or anything under `runs/` by hand.
- `background.md` remains frozen, content-addressed, and schema version 3. P2 never changes its dimensions, hypotheses, relations, provenance, literature credibility, or `space_revision`.
- `search_space` remains the immutable background receipt. New runtime eligibility lives only in `search_space_state`.
- Raw records and semantic edge receipts are durable observations/attribution. `experience` remains a replaceable derived belief. Pruning decisions are append-only selection-policy receipts.
- Scores remain lower-is-better. A crash is `+inf`, is distinct from an unevaluated target, and cannot by itself contradict or prune a semantic element.
- Keep the states distinct: external guidance may be `active`, `deprioritized`, or `excluded`; runtime control may be `active`, `deprioritized`, or `pruned`; belief coverage may be `unevaluated`, `failed`, `observed`, or `comparator_covered`.
- Pruning never deletes an id, record, point, observation, or prior decision. Reopening appends a new transition.
- A runtime-pruned hypothesis remains valid in historical points but is unavailable to new proposals. A runtime-pruned dimension is pinned to its explicit baseline in new proposals; it is not removed from point assignments.
- Baseline hypotheses and `baseline_only` dimensions cannot be runtime-deprioritized or pruned.
- Strong support/contradiction and a `pruned` recommendation require comparator coverage. At least two completed, non-crash, single-dimension semantic edges touching the target are required.
- A dimension may be pruned only when every currently selectable non-baseline hypothesis in it is already runtime-pruned or has its own high-confidence, comparator-covered `pruned` recommendation. Evidence against one hypothesis must not ban adjacent mechanisms.
- `active → pruned` is forbidden. Automated pruning is two-stage: `active → deprioritized` in one experience generation, then `deprioritized → pruned` in a later generation.
- `deprioritized|pruned → active` is allowed only from a later experience generation whose DAG cursor advanced or whose evidence contains an edge absent from the prior decision receipt.
- Runtime decisions never override externally scoped `excluded` guidance. An excluded hypothesis remains present and traceable but receives no runtime transition.
- `ledger.dag_revision` continues to track graph-visible score/status changes only. `search_space_state.revision` independently tracks pruning-policy changes.
- Admission is round-serial: experience/state refresh completes before the next idea-generator call, and propose → select → `add-record` completes before any candidate implementation starts. No extractor overlaps an in-flight candidate action.
- P3 intermediate-log retrieval, P4 space expansion, and convergence/regret claims are out of scope.
- Existing local run artifacts are disposable. Reject P1 records/experience/policy receipts rather than adding a migration path.
- Do not add dependencies or launch benchmark preparation, GPU jobs, live retrieval, or autonomous experiments for validation.
- Keep `.claude` and `.opencode` runtime contracts semantically synchronized.

---

## Contract Decisions

### Persistent semantic edge receipt

Every non-fresh candidate record stores one receipt per numeric parent in `semantic_edges`. The helper derives the receipts from the exact parent and child points; the model never authors them.

```json
{
  "schema_version": 1,
  "edge_id": "sedge-000-001",
  "space_revision": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "parent_run_id": "000",
  "child_run_id": "001",
  "change_class": "single_dimension",
  "changes": [
    {
      "dimension_id": "dim-data-curation",
      "operation": "hypothesis_changed",
      "from_hypothesis_id": "hyp-data-raw",
      "to_hypothesis_id": "hyp-data-filtered"
    }
  ]
}
```

Exact `operation` values are:

- `hypothesis_changed`: both hypothesis ids are strings;
- `dimension_activated`: `from_hypothesis_id` is `null` and `to_hypothesis_id` is a string;
- `dimension_deactivated`: `from_hypothesis_id` is a string and `to_hypothesis_id` is `null`.

`change_class` is `same_point`, `single_dimension`, or `multi_dimension`, based only on the number of changed assignments. It is an attribution-strength category, not a causal claim. Fresh records store `semantic_edges: []`.

### Bounded per-target evidence view

The extractor never reconstructs comparator coverage from the Top/Bottom graph window. A deterministic command scans the persisted ledger and returns bounded candidate evidence for each target:

```bash
python tools/background_contract.py target-evidence \
  --background runs/hard-interactions/p2-fixture/background.md \
  --ledger runs/hard-interactions/p2-fixture/ledger.json \
  --max-dimensions 16 --max-hypotheses 32 --max-edges-per-target 5
```

The output contains at most 16 dimensions and 32 hypotheses. Repeated `--target-id` selects exact known targets for a smaller follow-up view and bypasses only the dimension/hypothesis target-count caps; the per-target edge cap remains 2–5. Targets with new DAG evidence come first, then targets preserved in the prior experience snapshot, then other targets with comparator evidence, all with registry order as the stable tie-break.

The top-level shape is exact:

```json
{
  "schema_version": 1,
  "space_revision": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "dag_revision": 12,
  "experience_dag_revision": 8,
  "bounds": {
    "max_dimensions": 16,
    "max_hypotheses": 32,
    "max_edges_per_target": 5,
    "max_runs_per_target": 5
  },
  "dimension_targets": [],
  "hypothesis_targets": [],
  "omitted_target_ids": {
    "dimensions": [],
    "hypotheses": []
  }
}
```

`experience_dag_revision` is `0` when no prior snapshot exists. `omitted_target_ids` makes the target-count cap explicit and supplies exact ids for `--target-id` follow-up calls.

Each returned target has this exact mechanical block:

```json
{
  "target_id": "hyp-data-filtered",
  "evaluation_state": "comparator_covered",
  "evidence_run_ids": ["000", "001", "002", "003"],
  "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
  "comparator_coverage": {
    "direct_noncrash_edges": 2,
    "confounded_noncrash_edges": 0,
    "crash_edges": 0
  },
  "available_comparator_coverage": {
    "direct_noncrash_edges": 2,
    "confounded_noncrash_edges": 0,
    "crash_edges": 0
  },
  "omitted_edge_counts": {
    "direct_noncrash_edges": 0,
    "confounded_noncrash_edges": 0,
    "crash_edges": 0
  },
  "edges": [
    {
      "edge_id": "sedge-000-001",
      "change_class": "single_dimension",
      "parent_run_id": "000",
      "child_run_id": "001",
      "parent_status": "keep",
      "child_status": "discard",
      "parent_score": 0.4,
      "child_score": 0.5,
      "delta": 0.1
    },
    {
      "edge_id": "sedge-002-003",
      "change_class": "single_dimension",
      "parent_run_id": "002",
      "child_run_id": "003",
      "parent_status": "keep",
      "child_status": "discard",
      "parent_score": 0.41,
      "child_score": 0.52,
      "delta": 0.11
    }
  ]
}
```

`comparator_coverage` is computed from exactly `evidence_edge_ids` and is the object copied into experience. `available_comparator_coverage` and `omitted_edge_counts` disclose bounded-view loss but are never copied as if their hidden edge ids had been cited. Edge selection is deterministic: take up to two newest direct non-crash edges, then one newest confounded non-crash edge, then one newest crash edge, and fill remaining slots in direct/confounded/crash order. Returned run ids are unique endpoints of returned edges followed by recent target-bearing terminal runs, capped at five. Thus the default view exposes two direct comparators whenever they exist anywhere in history, even when neither edge appears in the incremental Top/Bottom render.

### Replaceable experience schema 3

Keep the existing bounded `summary`, `promising_regions`, `lessons`, and `bottlenecks`, then add bounded target-specific collections. Each target may appear at most once in its collection.

```json
{
  "schema_version": 3,
  "updated_at_run": "007",
  "generation": 2,
  "summary": "The current bounded interpretation of completed observations.",
  "promising_regions": [],
  "lessons": [],
  "bottlenecks": [],
  "dimension_evidence": [
    {
      "target_id": "dim-data-curation",
      "evaluation_state": "comparator_covered",
      "assessment": "unpromising",
      "recommended_status": "pruned",
      "claim": "Matched changes in data curation have repeatedly worsened the score.",
      "evidence_run_ids": ["000", "001", "004", "007"],
      "evidence_edge_ids": ["sedge-000-001", "sedge-004-007"],
      "comparator_coverage": {
        "direct_noncrash_edges": 2,
        "confounded_noncrash_edges": 0,
        "crash_edges": 0
      },
      "confidence": "high",
      "uncertainty": "Both comparisons still include implementation-level changes.",
      "reopen_when": "A new matched comparison improves after filtering is revised."
    }
  ],
  "hypothesis_evidence": []
}
```

The validator recomputes `comparator_coverage` from the cited edge ids and requires an exact match. The mechanical evaluation states are:

- `unevaluated`: no cited terminal runs or semantic edges;
- `failed`: cited target-bearing runs/edges are crash outcomes and there is no non-crash target observation;
- `observed`: at least one non-crash target observation, but fewer than two direct non-crash edges;
- `comparator_covered`: at least two direct non-crash edges.

`assessment` is `unknown`, `promising`, `mixed`, or `unpromising`. `recommended_status` is `active`, `deprioritized`, or `pruned`. The positive recommendation gates are exact:

- `active` is the only recommendation allowed for `unevaluated` or `failed`; those states also require `assessment: unknown` and `confidence: low`.
- `deprioritized` requires `assessment: unpromising`, `confidence: med | high`, `evaluation_state: observed | comparator_covered`, at least one direct non-crash edge in `comparator_coverage`, and a non-empty `reopen_when`. A non-comparative fresh observation cannot deprioritize a target.
- `pruned` requires `assessment: unpromising`, `confidence: high`, `evaluation_state: comparator_covered`, at least two direct non-crash edges, and a non-empty `reopen_when`.

These gates apply identically to dimension and hypothesis entries; the additional all-adjacent-hypotheses rule still applies before a dimension decision advances to runtime `pruned`.

### Append-only search-space state

The ledger gains this top-level object from its first P2 record onward:

```json
{
  "search_space_state": {
    "schema_version": 1,
    "revision": 1,
    "decisions": [
      {
        "schema_version": 1,
        "decision_id": "sdec-000001",
        "revision": 1,
        "target": {
          "kind": "hypothesis",
          "dimension_id": "dim-data-curation",
          "id": "hyp-data-filtered"
        },
        "from_status": "active",
        "to_status": "deprioritized",
        "experience_generation": 2,
        "experience_dag_revision": 8,
        "assessment": "unpromising",
        "confidence": "med",
        "claim": "The hypothesis has a direct but not yet replicated negative comparison.",
        "uncertainty": "One more matched comparison is required before pruning.",
        "reopen_when": "A later matched comparison shows an improvement.",
        "evidence_edge_ids": ["sedge-000-001"],
        "comparator_coverage": {
          "direct_noncrash_edges": 1,
          "confounded_noncrash_edges": 0,
          "crash_edges": 0
        },
        "evidence_observations": [
          {
            "edge_id": "sedge-000-001",
            "parent_status": "keep",
            "child_status": "discard",
            "parent_score": 0.4,
            "child_score": 0.5,
            "delta": 0.1
          }
        ]
      }
    ]
  }
}
```

Decision ids and revisions are helper-owned and contiguous. The helper copies the belief and current edge observations into the decision, so an older decision remains auditable after `experience` is regenerated or a candidate is deep-tuned.

Effective hypothesis status is composed with this precedence:

```text
guidance excluded
  > runtime hypothesis/dimension pruned
  > guidance or runtime hypothesis/dimension deprioritized
  > active
```

`excluded` and `pruned` remain distinguishable in every render and receipt.

### Selection revision

Proposal schema 2 and policy-receipt schema 2 both carry `search_space_state_revision`. `ledger.py add-record` accepts a candidate only when the receipt revision equals the current overlay revision and the selected point is eligible at that revision. Full-ledger validation replays the overlay at each historical receipt revision, so later pruning does not invalidate earlier records.

Strict equality relies on an explicit round-serial admission invariant. At a refresh boundary, every previously admitted record is terminal and no `idea-generator`, candidate writer, evaluator, or tuner is in flight. The coordinator runs `experience-extractor` and `apply-space-state` to completion before spawning `idea-generator`. Inside one `idea-generator` call, each action performs propose → select → `add-record` without yielding to another extractor; the pending ledger record is admitted before its candidate directory is created or implementation work begins. Therefore a conforming run cannot lose completed candidate work to a revision change: no candidate work exists before admission, and the state cannot advance during admission.

Both orchestrator prompts and harness tests enforce this ordering. A stale receipt is treated as a protocol violation and rejected. If concurrent extraction/admission is introduced later, it must first add separate selection/admission revision semantics; silently re-stamping the selection receipt would destroy the audit of what the acquisition policy actually saw and is not part of P2.

## File Responsibility Map

- Create `tools/semantic_evidence.py`: build, validate, index, and observe semantic edge receipts; compute comparator coverage and bounded per-target evidence views.
- Create `tools/search_space_state.py`: validate/replay append-only decisions, compose effective statuses, and validate point eligibility at a revision.
- Create `tests/test_semantic_evidence.py`: receipt, comparator-coverage, and bounded target-view unit tests.
- Create `tests/p2_fixtures.py`: shared four-record comparator ledger and schema-3 belief builders for focused P2 tests.
- Create `tests/test_search_space_state.py`: transition, replay, precedence, and historical-validity unit tests.
- Modify `tools/semantic_space.py`: expose normalized assignment changes, render persisted semantic lineage, and retain structural point validation.
- Modify `tools/background_contract.py`: validate ledger receipts/state, validate experience schema 3, render effective status, and expose the `target-evidence` CLI.
- Modify `tools/ledger.py`: add record fields, initialize the overlay, derive receipts, apply experience-backed transitions, and expose compact state.
- Modify `tools/semantic_search.py`: filter/order proposals through the current effective state and stamp state revisions.
- Modify `tools/got_graph.py`: include persisted semantic receipts beside current score deltas in edge views.
- Modify `tools/validate_background.py`: exercise the complete P2 contract through real CLIs.
- Modify `tests/test_background_contract.py`: focused schema-3 belief validation.
- Modify `tests/test_dag_incremental.py`: persisted semantic edge rendering and unchanged DAG-cursor behavior.
- Modify `tests/test_harness_controls.py`: compact receipt fields for the experience-extractor state-decision contract.
- Modify `tools/harness_guard.py`: compact the experience extractor's new state-decision receipt fields.
- Modify `.claude/agents/experience-extractor.md` and `.opencode/agents/experience-extractor.md`: author two-level belief, then request deterministic state transitions.
- Modify `.claude/agents/idea-generator.md` and `.opencode/agents/idea-generator.md`: consume effective status and state-stamped point receipts.
- Modify `.claude/agents/autoresearch-experiment.md` and `.opencode/agents/autoresearch-experiment.md`: recognize belief refresh plus optional policy-state advancement.
- Modify `.claude/rules/ledger.md` and `.opencode/rules/ledger.md`: document exact observation/belief/policy boundaries and schemas.
- Modify `docs/background-research.md`, `docs/search-space.md`, `AGENTS.md`, `CLAUDE.md`, and `README_ZH.md`: replace the P1 boundary with the implemented P2 behavior while keeping P3/P4 deferred.

---

### Task 1: Mechanical Semantic Edge Receipts and Target Views

**Files:**
- Create: `tools/semantic_evidence.py`
- Create: `tests/p2_fixtures.py`
- Create: `tests/test_semantic_evidence.py`
- Modify: `tools/semantic_space.py:point_diff`
- Modify: `tools/background_contract.py:build_parser,cmd_target_evidence`

**Interfaces:**
- Consumes: structurally valid parent/child `semantic_point` objects and ordered ledger records.
- Produces: `build_semantic_edges(prior_records: list[dict], child_record: dict) -> list[dict]`, `validate_semantic_edges(prior_records: list[dict], child_record: dict, registry: dict) -> list[str]`, `edge_index(ledger: dict) -> dict[str, dict]`, `edge_observation(ledger: dict, edge_id: str) -> dict`, `comparator_coverage(ledger: dict, edge_ids: list[str], *, target_kind: str, target_id: str) -> dict[str, int]`, `target_evaluation_state(ledger: dict, *, target_kind: str, target_id: str, evidence_run_ids: list[str], evidence_edge_ids: list[str]) -> str`, and `render_target_evidence(registry: dict, ledger: dict, *, max_dimensions: int = 16, max_hypotheses: int = 32, max_edges_per_target: int = 5, target_ids: list[str] | None = None) -> dict`.

- [ ] **Step 1: Write the shared comparator fixture and failing receipt/view tests**

Create `tests/p2_fixtures.py` with this no-I/O helper before importing it from the tests:

```python
def belief_ledger(registry: dict) -> dict:
    baseline = complete_point(registry)
    filtered = complete_point(
        registry, {"dim-data-curation": "hyp-data-filtered"}
    )
    records = [
        {
            "run_id": "000", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.40,
            "dag_revision": 1,
        },
        {
            "run_id": "001", "source_run_ids": ["000"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.50, "dag_revision": 2,
        },
        {
            "run_id": "002", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.41,
            "dag_revision": 3,
        },
        {
            "run_id": "003", "source_run_ids": ["002"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.52, "dag_revision": 4,
        },
    ]
    records[1]["semantic_edges"] = build_semantic_edges(records[:1], records[1])
    records[3]["semantic_edges"] = build_semantic_edges(records[:3], records[3])
    return {"records": records, "dag_revision": 4}
```

Add tests that construct baseline, one-dimension, activated-dimension, and multi-dimension points from `fixture_registry()` and assert exact receipts:

```python
def test_builds_one_persistent_receipt_per_parent(self) -> None:
    registry = fixture_registry()
    baseline = complete_point(registry)
    changed = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    parent = {"run_id": "000", "semantic_point": baseline}
    child = {
        "run_id": "001",
        "source_run_ids": ["000"],
        "semantic_point": changed,
    }

    self.assertEqual(
        build_semantic_edges([parent], child),
        [{
            "schema_version": 1,
            "edge_id": "sedge-000-001",
            "space_revision": changed["space_revision"],
            "parent_run_id": "000",
            "child_run_id": "001",
            "change_class": "single_dimension",
            "changes": [{
                "dimension_id": "dim-data-curation",
                "operation": "hypothesis_changed",
                "from_hypothesis_id": "hyp-data-raw",
                "to_hypothesis_id": "hyp-data-filtered",
            }],
        }],
    )
```

Also assert: fresh produces `[]`; unchanged points produce `same_point`; activation uses `dimension_activated`; deactivation uses `dimension_deactivated`; crossover receipt order follows `source_run_ids`; a forged operation, id, revision, or change list is rejected.

Add a bounded-view test whose ledger has the two direct filtered edges from `belief_ledger()` followed by at least six newer unrelated same-point edges. Assert that a one-node incremental graph view omits `sedge-000-001` while `render_target_evidence(..., target_ids=["hyp-data-filtered"])` still returns both old direct edge ids, exact selected coverage `2/0/0`, full edge observations, and zero omitted direct edges. Add cap tests for 16 dimensions, 32 hypotheses, five edges, two-to-five edge-limit validation, deterministic category balancing, and unknown `--target-id` rejection.

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run: `python -m unittest tests.test_semantic_evidence -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'semantic_evidence'`.

- [ ] **Step 3: Normalize point differences in `semantic_space.py`**

Replace the current string-sentinel diff with exact nullable assignments while keeping registry order:

```python
def point_diff(parent: dict[str, Any], child: dict[str, Any]) -> list[dict[str, Any]]:
    parent_by_dim = {
        item["dimension_id"]: item
        for item in parent.get("assignments", [])
        if isinstance(item, dict) and isinstance(item.get("dimension_id"), str)
    }
    changes: list[dict[str, Any]] = []
    for current in child.get("assignments", []):
        if not isinstance(current, dict) or not isinstance(current.get("dimension_id"), str):
            continue
        previous = parent_by_dim.get(current["dimension_id"], {})
        before = previous.get("hypothesis_id") if previous.get("state") == "selected" else None
        after = current.get("hypothesis_id") if current.get("state") == "selected" else None
        if before == after:
            continue
        operation = (
            "dimension_activated" if before is None
            else "dimension_deactivated" if after is None
            else "hypothesis_changed"
        )
        changes.append({
            "dimension_id": current["dimension_id"],
            "operation": operation,
            "from_hypothesis_id": before,
            "to_hypothesis_id": after,
        })
    return changes
```

Update the existing lineage assertions to the normalized shape in the same change.

- [ ] **Step 4: Implement `semantic_evidence.py`**

Define exact constants and deterministic builders:

```python
EDGE_SCHEMA_VERSION = 1
CHANGE_CLASSES = {"same_point", "single_dimension", "multi_dimension"}
OPERATIONS = {"hypothesis_changed", "dimension_activated", "dimension_deactivated"}
EDGE_ID_RE = re.compile(r"^sedge-([0-9]+)-([0-9]+)$")


def _change_class(changes: list[dict[str, Any]]) -> str:
    if not changes:
        return "same_point"
    return "single_dimension" if len(changes) == 1 else "multi_dimension"


def build_semantic_edges(
    prior_records: list[dict[str, Any]], child_record: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {str(record["run_id"]): record for record in prior_records}
    child_id = str(child_record["run_id"])
    child_point = child_record["semantic_point"]
    receipts = []
    for parent_id in child_record.get("source_run_ids", []):
        parent_point = by_id[str(parent_id)]["semantic_point"]
        changes = point_diff(parent_point, child_point)
        receipts.append({
            "schema_version": EDGE_SCHEMA_VERSION,
            "edge_id": f"sedge-{parent_id}-{child_id}",
            "space_revision": child_point["space_revision"],
            "parent_run_id": str(parent_id),
            "child_run_id": child_id,
            "change_class": _change_class(changes),
            "changes": changes,
        })
    return receipts
```

Implement validation by rebuilding the expected list and requiring exact equality. Implement observation lookup with `delta = child_score - parent_score` only when both records are terminal and non-crash; otherwise use `null`. Compute comparator coverage only from cited receipts that touch the requested dimension or hypothesis. Count `single_dimension` non-crash receipts as direct, `multi_dimension` non-crash receipts as confounded, and any receipt incident to a crash as crash evidence. Implement `target_evaluation_state` once in this module using the four mechanical state rules; both the bounded renderer and the later experience validator call this same function so their state labels cannot drift.

- [ ] **Step 5: Implement the bounded target-evidence renderer and CLI**

In `semantic_evidence.py`, scan the complete persisted edge index for each known target, classify every touching receipt, and build the selected/available/omitted coverage objects from the contract section above. Mark a target as having new DAG evidence when a touching edge has a parent or child record whose `dag_revision` exceeds `experience.dag_revision`; then apply the documented target-priority order. Use numeric child id descending within each edge category. Select up to two direct, then one confounded, then one crash edge, and fill unused slots in direct/confounded/crash order without duplicates. Compute `evaluation_state` from only the returned run/edge evidence, and include complete `edge_observation` fields plus `change_class`, `parent_run_id`, and `child_run_id`.

In `background_contract.py`, add a `target-evidence` subcommand with required `--background` and `--ledger`, repeated `--target-id`, integer `--max-dimensions`/`--max-hypotheses`, and integer `--max-edges-per-target`. Reuse `_validated_inputs` before rendering, require target caps in `[1, 16]` and `[1, 32]`, require the edge cap in `[2, 5]`, and print the JSON view. The CLI reads the full ledger deterministically; it never reads candidate code or logs.

Add a real-CLI subprocess assertion to `tools/validate_background.py` and require the returned `hyp-data-filtered` block's `comparator_coverage` to equal `comparator_coverage(ledger, block["evidence_edge_ids"], target_kind="hypothesis", target_id="hyp-data-filtered")` exactly.

- [ ] **Step 6: Run receipt and target-evidence tests**

Run: `python -m unittest tests.test_semantic_evidence -v`

Expected: all semantic-receipt, observation, and comparator-coverage tests PASS.

- [ ] **Step 7: Run the existing semantic contract tests**

Run: `python -m unittest tests.test_background_contract tests.test_dag_incremental -v`

Expected: PASS after updating only the exact `point_diff` assertions affected by the normalized shape.

- [ ] **Step 8: Commit the semantic receipt and evidence-view primitive**

```bash
git add tools/semantic_space.py tools/semantic_evidence.py tools/background_contract.py tests/p2_fixtures.py tests/test_semantic_evidence.py tests/test_dag_incremental.py tools/validate_background.py
git commit -m "feat: define semantic DAG evidence receipts"
```

---

### Task 2: Persist and Render Semantic Edges

**Files:**
- Modify: `tools/ledger.py:RECORD_FIELDS,cmd_add_record`
- Modify: `tools/background_contract.py:validate_ledger`
- Modify: `tools/semantic_space.py:derive_semantic_lineage`
- Modify: `tools/got_graph.py:_edge_view`
- Modify: `tools/validate_background.py:record,main`
- Modify: `tests/test_dag_incremental.py`

**Interfaces:**
- Consumes: Task 1's receipt builders/validators.
- Produces: helper-authored `record.semantic_edges`; graph edge views containing `semantic_edge`; ledger validation that rejects missing or forged receipts.

- [ ] **Step 1: Write failing persistence tests**

Extend the real-CLI fixture to assert that an improve record contains an exact persisted receipt and that graph rendering reuses it:

```python
stored = json.loads(ledger_path.read_text())
edge = stored["records"][1]["semantic_edges"][0]
assert edge["edge_id"] == "sedge-000-001"
assert edge["parent_run_id"] == "000"
assert edge["child_run_id"] == "001"

view = render_incremental(stored, top=1, bottom=1)
rendered = next(item for item in view["delta_edges"] if item["child"] == "001")
assert rendered["semantic_edge"] == edge
```

Add negative assertions for a removed `semantic_edges` field and a forged `change_class`.

- [ ] **Step 2: Run the affected tests and verify failure**

Run: `python -m unittest tests.test_dag_incremental -v`

Expected: FAIL because records and graph views do not yet contain `semantic_edges`.

- [ ] **Step 3: Persist helper-derived receipts at record creation**

Add `semantic_edges` immediately after `semantic_point` in `RECORD_FIELDS`. In `cmd_add_record`, populate it only after ancestry and point validation:

```python
from semantic_evidence import build_semantic_edges

record.update(
    kind=args.kind,
    idea=args.idea,
    change=args.change,
    source_run_ids=source_run_ids,
    op=args.op,
    semantic_point=semantic_point,
    semantic_edges=[],
    policy_receipt=policy_receipt,
    candidate_name=args.candidate_name_hint,
    description=args.description or args.idea,
    metric=data["metric"],
)
record["semantic_edges"] = build_semantic_edges(data["records"], record)
```

No CLI argument is added: the model must not author semantic receipts.

- [ ] **Step 4: Validate stored receipts and consume them in lineage/rendering**

In `background_contract.validate_ledger`, call `validate_semantic_edges` before adding the child id to `known`. In `semantic_space.derive_semantic_lineage`, expose each record's stored receipts instead of recomputing `parent_diffs`. In `_edge_view`, locate the child receipt by `parent_run_id` and add it under `semantic_edge`; retain current `delta` and free-text `change` as separate fields.

The exact graph edge shape becomes:

```python
{
    "child": child,
    "parent": parent,
    "delta": delta,
    "change": _change_for_parent(recs.get(child, {}).get("change"), parent),
    "semantic_edge": semantic_receipt,
}
```

- [ ] **Step 5: Update fixtures to be P2-only**

Change `tools/validate_background.py:record` to call `build_semantic_edges` after its parents are available. Do not accept records without the field; local P1 ledgers are intentionally unsupported.

- [ ] **Step 6: Run persistence and validator tests**

Run: `python -m unittest tests.test_semantic_evidence tests.test_dag_incremental tests.test_background_contract -v`

Expected: PASS, including missing/forged receipt rejection.

- [ ] **Step 7: Commit edge persistence**

```bash
git add tools/ledger.py tools/background_contract.py tools/semantic_space.py tools/got_graph.py tools/validate_background.py tests/test_dag_incremental.py
git commit -m "feat: persist semantic receipts on DAG edges"
```

---

### Task 3: Append-Only Search-Space State Contract

**Files:**
- Create: `tools/search_space_state.py`
- Create: `tests/test_search_space_state.py`
- Modify: `tools/background_contract.py:validate_ledger`
- Modify: `tools/ledger.py:_load_ledger,cmd_add_record,cmd_brief`

**Interfaces:**
- Consumes: background registry ids, static guidance selection, semantic edge index, and ledger top-level state.
- Produces: `empty_search_space_state() -> dict`, `validate_search_space_state(registry: dict, ledger: dict) -> list[str]`, `replay_search_space_state(registry: dict, state: dict, *, revision: int | None = None) -> dict`, `compose_effective_selection(registry: dict, guidance: dict, runtime: dict) -> dict`, and `validate_point_eligibility(point: dict, registry: dict, effective: dict) -> list[str]`.

- [ ] **Step 1: Write failing replay and precedence tests**

Cover empty state, contiguous transitions, baseline protection, external exclusion precedence, dimension pinning, historical replay, and reopening:

```python
def decision(revision: int, target_id: str, before: str, after: str) -> dict:
    return {
        "schema_version": 1,
        "decision_id": f"sdec-{revision:06d}",
        "revision": revision,
        "target": {
            "kind": "hypothesis",
            "dimension_id": "dim-data-curation",
            "id": target_id,
        },
        "from_status": before,
        "to_status": after,
        "experience_generation": revision,
        "experience_dag_revision": revision + 2,
        "assessment": "unpromising" if after != "active" else "mixed",
        "confidence": "high",
        "claim": "Fixture belief copied into an immutable decision receipt.",
        "uncertainty": "Concrete implementations remain a possible confounder.",
        "reopen_when": "A later direct comparison contradicts this decision.",
        "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
        "comparator_coverage": {
            "direct_noncrash_edges": 2,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        },
        "evidence_observations": [
            {
                "edge_id": "sedge-000-001",
                "parent_status": "keep",
                "child_status": "discard",
                "parent_score": 0.4,
                "child_score": 0.5,
                "delta": 0.1,
            },
            {
                "edge_id": "sedge-002-003",
                "parent_status": "keep",
                "child_status": "discard",
                "parent_score": 0.42,
                "child_score": 0.51,
                "delta": 0.09,
            },
        ],
    }


def test_replay_preserves_pruned_identity_and_allows_reopen(self) -> None:
    registry = fixture_registry()
    state = {
        "schema_version": 1,
        "revision": 2,
        "decisions": [
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "deprioritized", "pruned"),
        ],
    }
    at_one = replay_search_space_state(registry, state, revision=1)
    at_two = replay_search_space_state(registry, state, revision=2)
    self.assertEqual(at_one["hypotheses"]["hyp-data-filtered"], "deprioritized")
    self.assertEqual(at_two["hypotheses"]["hyp-data-filtered"], "pruned")
    self.assertIn("hyp-data-filtered", at_two["hypotheses"])
```

Add a test that a pruned dimension leaves its baseline eligible and makes every non-baseline hypothesis in that dimension `pruned` for selection.

- [ ] **Step 2: Run and confirm the missing-module failure**

Run: `python -m unittest tests.test_search_space_state -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'search_space_state'`.

- [ ] **Step 3: Implement state replay and exact validation**

Start the module with these constants and legal transitions:

```python
STATE_SCHEMA_VERSION = 1
DECISION_SCHEMA_VERSION = 1
RUNTIME_STATUSES = {"active", "deprioritized", "pruned"}
LEGAL_TRANSITIONS = {
    ("active", "deprioritized"),
    ("deprioritized", "active"),
    ("deprioritized", "pruned"),
    ("pruned", "active"),
}
DECISION_ID_RE = re.compile(r"^sdec-[0-9]{6}$")


def empty_search_space_state() -> dict[str, Any]:
    return {"schema_version": 1, "revision": 0, "decisions": []}
```

Replay dimensions and hypotheses separately, defaulting every searchable target to `active`. Validate exact fields, sequential `revision`, derived `decision_id`, known target ownership, legal `from_status`, and legal transitions. Reject decisions targeting a baseline hypothesis or a `baseline_only` dimension.

- [ ] **Step 4: Implement effective-status composition**

Return all component statuses rather than a single lossy label:

```python
{
    "hyp-data-filtered": {
        "guidance_status": "active",
        "dimension_runtime_status": "active",
        "hypothesis_runtime_status": "pruned",
        "effective_status": "pruned",
        "binding_guidance": [],
        "matched_guidance": [],
    }
}
```

`validate_point_eligibility` checks only selection-time policy eligibility. Keep structural `validate_point` unchanged so historical observations remain structurally valid.

- [ ] **Step 5: Initialize and validate top-level state**

On the first P2 `add-record`, set:

```python
data["search_space_state"] = data.get("search_space_state") or empty_search_space_state()
```

Make `validate_ledger` require and validate this object whenever records exist. Add `search_space_state_revision` and counts of runtime `deprioritized`/`pruned` dimensions and hypotheses to `ledger.py brief`; do not include the full decision log.

- [ ] **Step 6: Run state and background tests**

Run: `python -m unittest tests.test_search_space_state tests.test_background_contract -v`

Expected: PASS for legal replay, rejection cases, status precedence, and baseline protection.

- [ ] **Step 7: Commit the state contract**

```bash
git add tools/search_space_state.py tests/test_search_space_state.py tools/background_contract.py tools/ledger.py
git commit -m "feat: add append-only semantic pruning state"
```

---

### Task 4: Two-Level Experience Schema and Comparator Gates

**Files:**
- Modify: `tools/background_contract.py:validate_experience,validate_experience_replacement`
- Modify: `tests/test_background_contract.py`
- Modify: `tools/validate_background.py`
- Modify: `.claude/rules/ledger.md`
- Modify: `.opencode/rules/ledger.md`

**Interfaces:**
- Consumes: Task 1 comparator coverage, terminal run ids, persisted semantic edge ids, and the background registry.
- Produces: validated experience schema 3 with bounded `dimension_evidence` and `hypothesis_evidence`.

- [ ] **Step 1: Replace the P1 rejection test with failing schema-3 tests**

Add table-driven tests for valid `unevaluated`, `failed`, `observed`, and `comparator_covered` entries. Include a valid one-direct-edge `deprioritized` entry. Add rejection tests for an unknown target, an edge that does not touch the target, forged comparator counts, crash-only contradiction, high confidence without coverage, deprioritization with no direct edge, deprioritization without `unpromising`/`med|high`/`reopen_when`, prune without high-confidence contradiction, duplicate target, missing reopen condition, more than 16 dimension entries, and more than 32 hypothesis entries.

Use this core positive assertion:

```python
def test_accepts_comparator_covered_hypothesis_belief(self) -> None:
    registry = fixture_registry()
    ledger = belief_ledger(registry)
    experience = {
        "schema_version": 3,
        "updated_at_run": "003",
        "generation": 0,
        "summary": "Two direct comparisons make the filtered hypothesis eligible for a conservative recommendation.",
        "promising_regions": [],
        "lessons": [],
        "bottlenecks": [],
        "dimension_evidence": [],
        "hypothesis_evidence": [{
            "target_id": "hyp-data-filtered",
            "evaluation_state": "comparator_covered",
            "assessment": "unpromising",
            "recommended_status": "pruned",
            "claim": "Both direct comparisons were worse than their matched baseline parents.",
            "evidence_run_ids": ["000", "001", "002", "003"],
            "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
            "comparator_coverage": {
                "direct_noncrash_edges": 2,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            },
            "confidence": "high",
            "uncertainty": "Implementation differences remain confounded with each semantic change.",
            "reopen_when": "A later direct comparison improves over its parent.",
        }],
    }
    self.assertEqual(validate_experience(experience, registry, ledger), [])
```

- [ ] **Step 2: Run and verify schema-version failure**

Run: `python -m unittest tests.test_background_contract -v`

Expected: FAIL because `validate_experience` still requires schema version 2 and rejects the two-level fields.

- [ ] **Step 3: Implement the target-entry validator**

Import Task 1's `comparator_coverage` and `target_evaluation_state`; do not reimplement their classification rules in `background_contract.py`. Implement `_validate_target_evidence(items, *, field, target_kind, registry, ledger, limit) -> list[str]` with these exact phases: require a list no longer than `limit`; require the exact common fields plus optional `reopen_when`; require unique known targets of `target_kind`; require 0–5 unique terminal target-related run ids and 0–5 unique target-touching edge ids; treat a run as target-related when its point bears the target or it is an endpoint of a cited target-touching edge; recompute and compare the three comparator counts; call `target_evaluation_state` and compare its result to the authored `evaluation_state`; validate the four assessment values, three recommendation values, and three confidence values; enforce crash-only/unevaluated conservatism; enforce the full `deprioritized` gate (`unpromising`, `med|high`, `observed|comparator_covered`, direct non-crash count at least one, non-empty `reopen_when`); require comparator coverage for high-confidence `promising` or `unpromising`; and enforce the full prune gate plus `reopen_when`. Each failed condition emits a target-qualified error string used by the negative tests from Step 1.

- [ ] **Step 4: Advance the complete snapshot to schema 3**

Require both collections, retain current bounds for generic collections, and preserve helper ownership of `dag_revision`:

```python
allowed_top = {
    "schema_version", "updated_at_run", "generation", "summary",
    "promising_regions", "lessons", "bottlenecks",
    "dimension_evidence", "hypothesis_evidence", "dag_revision",
}
```

Remove the deferred P1-field rejection. Keep replacement freshness rules unchanged except for the new schema version.

- [ ] **Step 5: Document the exact schema in both ledger rules**

State explicitly that target evidence is replaceable belief, cited edge ids are attribution evidence, and comparator coverage does not prove causality. Document all enum values, limits, pruning gates, and the separation from `search_space_state` decisions.

- [ ] **Step 6: Run belief validation**

Run: `python -m unittest tests.test_background_contract -v`

Expected: PASS for every evaluation state and conservative pruning gate.

- [ ] **Step 7: Run the deterministic background validator**

Run: `python tools/validate_background.py`

Expected: final line `P2 semantic background, edge, belief, and policy-state checks passed.` after updating the script's final message and fixtures to schema 3.

- [ ] **Step 8: Commit two-level belief extraction contract**

```bash
git add tools/background_contract.py tests/test_background_contract.py tools/validate_background.py .claude/rules/ledger.md .opencode/rules/ledger.md
git commit -m "feat: validate two-level semantic experience"
```

---

### Task 5: Deterministic Experience-to-Pruning Transitions

**Files:**
- Modify: `tools/search_space_state.py`
- Modify: `tools/ledger.py:cmd_set_experience,build_parser`
- Modify: `tests/test_search_space_state.py`
- Modify: `tools/validate_background.py`

**Interfaces:**
- Consumes: current schema-3 experience, current `search_space_state`, semantic edge observations, and background target ownership.
- Produces: `derive_experience_transitions(registry: dict, ledger: dict) -> list[dict]`, `append_experience_transitions(registry: dict, ledger: dict) -> list[dict]`, and CLI `ledger.py apply-space-state`.

- [ ] **Step 1: Write failing transition-policy tests**

Test the exact state machine:

```python
class ExperienceTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.ledger = belief_ledger(self.registry)
        self.ledger["search_space_state"] = empty_search_space_state()

    def experience(self, generation: int) -> dict:
        return {
            "schema_version": 3,
            "updated_at_run": "003",
            "generation": generation,
            "dag_revision": self.ledger["dag_revision"],
            "summary": "Repeated direct comparisons are worse than matched baselines.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [{
                "target_id": "hyp-data-filtered",
                "evaluation_state": "comparator_covered",
                "assessment": "unpromising",
                "recommended_status": "pruned",
                "claim": "Both direct comparisons were worse.",
                "evidence_run_ids": ["000", "001", "002", "003"],
                "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
                "comparator_coverage": {
                    "direct_noncrash_edges": 2,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
                "confidence": "high",
                "uncertainty": "Implementation changes remain confounded.",
                "reopen_when": "A later direct comparison improves.",
            }],
        }

    def test_pruning_requires_two_generations(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(first[0]["from_status"], "active")
        self.assertEqual(first[0]["to_status"], "deprioritized")

        self.ledger["experience"] = self.experience(generation=2)
        second = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(second[0]["from_status"], "deprioritized")
        self.assertEqual(second[0]["to_status"], "pruned")
```

Import `belief_ledger` from `tests.p2_fixtures`; do not duplicate the comparator ledger or import helpers from another test module.

Also assert: repeated application of one generation is a no-op; low confidence cannot prune; one direct edge can only deprioritize; crash-only evidence makes no transition; a baseline target is rejected; an externally excluded target produces no runtime transition; reopening appends `pruned → active`; decision observations retain the score values present when the transition was appended; and a dimension cannot move to `pruned` while any selectable non-baseline hypothesis lacks its own comparator-covered prune recommendation or prior prune decision.

- [ ] **Step 2: Run and verify missing-interface failures**

Run: `python -m unittest tests.test_search_space_state -v`

Expected: FAIL because experience-backed transition functions do not exist.

- [ ] **Step 3: Implement deterministic transition derivation**

For each belief entry in registry order, derive at most one transition:

```python
def _recommended_transition(current: str, belief: dict[str, Any], last: dict | None) -> str | None:
    recommendation = belief["recommended_status"]
    if current == "active" and recommendation in {"deprioritized", "pruned"}:
        return "deprioritized"
    if current == "deprioritized" and recommendation == "pruned":
        if last is not None and belief["experience_generation"] > last["experience_generation"]:
            return "pruned"
        return None
    if current in {"deprioritized", "pruned"} and recommendation == "active":
        if _has_reopening_evidence(belief, last):
            return "active"
    return None
```

Pass `experience_generation` into the normalized belief before calling this helper. `_has_reopening_evidence` returns true only when the experience DAG cursor is newer than the prior decision cursor or at least one cited edge id was absent from that decision.

Before returning a dimension's `deprioritized → pruned` transition, enumerate its non-baseline hypotheses. Require each non-excluded hypothesis either to replay as `pruned` already or to have a same-generation hypothesis belief that independently satisfies the high-confidence comparator-covered prune gate. This scoped check prevents evidence about one mechanism from excluding adjacent mechanisms.

- [ ] **Step 4: Build immutable decision receipts**

Set `revision = state["revision"] + 1` and `decision_id = f"sdec-{revision:06d}"`. Copy the validated belief fields, exact comparator coverage, and current `edge_observation` for every cited edge. Append decisions in registry dimension order, with a dimension before its hypotheses. Increment only `search_space_state.revision`; do not touch `dag_revision`.

- [ ] **Step 5: Add the atomic ledger command**

Add:

```bash
python tools/ledger.py apply-space-state \
  --ledger runs/hard-interactions/p2-fixture/ledger.json \
  --background runs/hard-interactions/p2-fixture/background.md
```

The command loads and validates the background, ledger, experience, and current state; derives transitions; validates the resulting ledger; saves once; and prints:

```json
{"ok": true, "prior_revision": 1, "revision": 2, "decision_ids": ["sdec-000002"]}
```

An empty transition set is a successful no-op. Keep `set-experience` as a separate replaceable-belief write so a failed policy transition never corrupts or hides the new belief.

- [ ] **Step 6: Run transition and real-CLI tests**

Run: `python -m unittest tests.test_search_space_state -v`

Expected: PASS for two-stage pruning, idempotence, reopening, baseline protection, and immutable evidence snapshots.

Run: `python tools/validate_background.py`

Expected: PASS through `set-experience` followed by `apply-space-state`.

- [ ] **Step 7: Commit deterministic pruning transitions**

```bash
git add tools/search_space_state.py tools/ledger.py tests/test_search_space_state.py tools/validate_background.py
git commit -m "feat: apply evidence-backed semantic pruning"
```

---

### Task 6: State-Aware Proposal and Selection Receipts

**Files:**
- Modify: `tools/semantic_search.py`
- Modify: `tools/background_contract.py:_validate_policy_receipt,validate_ledger,render_space`
- Modify: `tools/ledger.py:cmd_add_record`
- Modify: `tests/test_search_space_state.py`
- Modify: `tools/validate_background.py`

**Interfaces:**
- Consumes: effective statuses and point eligibility at a requested state revision.
- Produces: proposal schema 2, policy receipt schema 2, state-aware filtering/ordering, and historical revision validation.

- [ ] **Step 1: Write failing selection lifecycle tests**

Cover this complete lifecycle in one test:

1. Revision 0 proposes a point containing `hyp-data-filtered`.
2. A later state decision prunes that hypothesis.
3. Revision 1 proposals omit it.
4. A revision-0 historical record remains ledger-valid.
5. `add-record` rejects a deliberately forged/stale revision-0 receipt after the state reaches revision 1; the test creates no candidate directory or implementation because admission precedes all candidate work.
6. Reopening at revision 2 makes the hypothesis proposal-eligible again.
7. A structural `requires` completion that would force a pruned hypothesis is discarded rather than leaking an ineligible point into the proposal set.

Assert exact state stamps:

```python
self.assertEqual(proposals["schema_version"], 2)
self.assertEqual(proposals["search_space_state_revision"], 1)
self.assertEqual(receipt["schema_version"], 2)
self.assertEqual(receipt["search_space_state_revision"], 1)
```

- [ ] **Step 2: Run and verify schema/filter failures**

Run: `python -m unittest tests.test_search_space_state -v`

Expected: FAIL because proposals still use guidance-only eligibility and receipts have no state revision.

- [ ] **Step 3: Make proposal construction state-aware**

Change `_eligible_hypotheses` to accept the ledger and return both selectable ids and effective metadata. Treat a missing bootstrap ledger or absent top-level state as `empty_search_space_state()` only while building the first proposal; persisted P2 ledgers must contain the validated object. Exclude `excluded` and `pruned`, retain `deprioritized`, and always retain a protected baseline. When a dimension is runtime-pruned, expose only its baseline.

Change `_add_point` to run both structural `validate_point` and revision-current `validate_point_eligibility`. This second gate is required because deterministic completion of a `requires` relation can select a hypothesis that was not present in the sparse overrides; filtering only the input choice lists is insufficient.

Change proposal sorting so a proposal with runtime-deprioritized content sorts after active proposals at equal coverage. Preserve the current deterministic tie-breaks and proposal cap.

- [ ] **Step 4: Advance proposal and policy receipt schemas**

Add `search_space_state_revision` to the proposal-set digest input and output. Require the selection command's ledger revision to match the proposal revision. Write the same integer into policy receipt schema 2. Keep gain, uncertainty, cost, and coverage separate and otherwise unchanged.

- [ ] **Step 5: Validate each record at its historical selection revision**

In `validate_ledger`:

```python
record_revision = record["policy_receipt"]["search_space_state_revision"]
runtime = replay_search_space_state(registry, ledger["search_space_state"], revision=record_revision)
effective = compose_effective_selection(registry, derive_hypothesis_selection(registry), runtime)
errors.extend(validate_point_eligibility(record["semantic_point"], registry, effective))
```

Require revisions to be integers in `[0, search_space_state.revision]`. In `cmd_add_record`, additionally require equality with the current revision to reject stale proposal artifacts.

- [ ] **Step 6: Render all status components**

For every hypothesis, replace the P1 `selection` object with:

```json
{
  "guidance_status": "active",
  "dimension_runtime_status": "active",
  "hypothesis_runtime_status": "pruned",
  "effective_status": "pruned",
  "binding_guidance": [],
  "matched_guidance": []
}
```

Add the current `search_space_state_revision` and each dimension's runtime status to the bounded render. Keep literature credibility separate.

- [ ] **Step 7: Run state-aware search tests**

Run: `python -m unittest tests.test_search_space_state tests.test_background_contract -v`

Expected: PASS for filtering, deprioritized ordering, stale-receipt rejection, historical validity, and reopening.

Run: `python tools/validate_background.py`

Expected: PASS through the end-to-end lifecycle.

- [ ] **Step 8: Commit state-aware acquisition**

```bash
git add tools/semantic_search.py tools/background_contract.py tools/ledger.py tests/test_search_space_state.py tools/validate_background.py
git commit -m "feat: select points against revisioned pruning state"
```

---

### Task 7: Experience Extractor and Runtime Integration

**Files:**
- Modify: `.claude/agents/experience-extractor.md`
- Modify: `.opencode/agents/experience-extractor.md`
- Modify: `.claude/agents/idea-generator.md`
- Modify: `.opencode/agents/idea-generator.md`
- Modify: `.claude/agents/autoresearch-experiment.md`
- Modify: `.opencode/agents/autoresearch-experiment.md`
- Modify: `tools/harness_guard.py`
- Modify: `tests/test_harness_controls.py`

**Interfaces:**
- Consumes: bounded graph edges with semantic receipts, the deterministic bounded per-target evidence view, bounded effective-space render, current experience, and current state revision.
- Produces: schema-3 belief snapshots, deterministic `apply-space-state` invocation, compact state-decision receipts, and synchronized runtime behavior.

- [ ] **Step 1: Update the experience-extractor workflow**

Keep the existing bounded brief, incremental graph, space render, compact lineage, and prior-experience inputs. Add this deterministic input immediately after the graph render:

```bash
python tools/background_contract.py target-evidence \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json \
  --max-dimensions 16 --max-hypotheses 32 --max-edges-per-target 5
```

Require the extractor to:

1. Read persisted semantic receipts and current deltas, never infer edges from prose.
2. Use the target-evidence output—not the Top/Bottom window—as the only source of `evidence_edge_ids`, per-edge score/status observations, `evaluation_state`, and `comparator_coverage` for dimension/hypothesis entries.
3. Copy one target block's returned `evidence_edge_ids`, `evidence_run_ids`, and `comparator_coverage` together. Never copy `available_comparator_coverage` as if omitted edge ids had been cited. If a necessary target was omitted by the global caps, rerun with repeated `--target-id <exact-id>` before authoring its belief.
4. Preserve/revise generic beliefs and emit bounded target entries.
5. Keep crash-only targets `failed`/`unknown`/`active`.
6. Validate and store schema 3.
7. Invoke `ledger.py apply-space-state` once after a successful store.
8. Never edit background, records, points, receipts, scores, or decisions directly.

Change the compact output to:

```text
updated_at_run: 007
generation: 2
evidence_runs: 5
search_space_state_revision: 3
decision_ids: sdec-000003
ledger: runs/hard-interactions/p2-fixture/ledger.json
```

Use `decision_ids: none` for a valid no-op.

- [ ] **Step 2: Update idea-generator and coordinator language**

State that proposal/select helpers own effective eligibility and state revision checks. The idea generator must not override a pruning decision or hand-author a receipt. It must finish propose → select → `add-record` for each action without an intervening extractor, and the record must exist before the coordinator creates a candidate directory or spawns candidate implementation.

In both orchestrators, define a refresh boundary as: all records from the prior generation are terminal, decoupled tuning for that round has returned or no-op'd, and no idea-generator/writer/extractor/tuner child is active. Only then may the coordinator run the experience extractor; it waits for belief storage plus `apply-space-state` to return before spawning the next idea generator. The coordinator accepts the extractor's compact receipt but does not interpret belief or apply state itself. State explicitly that parallelizing these phases requires a future admission-revision contract and is prohibited by the current strict-equality protocol.

- [ ] **Step 3: Advance receipt compaction**

Change the experience-extractor contract in `harness_guard.py` to require:

```python
("updated_at_run", "generation", "evidence_runs", "search_space_state_revision", "decision_ids", "ledger")
```

Add a test for both a concrete decision id and `decision_ids: none`.

- [ ] **Step 4: Run runtime contract tests**

Run: `python -m unittest tests.test_harness_controls -v`

Expected: PASS for the updated compact-receipt contract.

- [ ] **Step 5: Resolve the OpenCode agents**

Run: `opencode debug agent experience-extractor`

Run: `opencode debug agent idea-generator`

Expected: both mirrored agents resolve without error; manually confirm the resolved prompts reflect the updated workflows.

- [ ] **Step 6: Commit runtime integration**

```bash
git add .claude/agents/experience-extractor.md .opencode/agents/experience-extractor.md .claude/agents/idea-generator.md .opencode/agents/idea-generator.md .claude/agents/autoresearch-experiment.md .opencode/agents/autoresearch-experiment.md tools/harness_guard.py tests/test_harness_controls.py
git commit -m "feat: connect semantic experience to pruning policy"
```

---

### Task 8: Documentation and Cross-Shape Acceptance

**Files:**
- Modify: `docs/background-research.md`
- Modify: `docs/search-space.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `README_ZH.md`
- Modify: `tools/semantic_space.py` module/status documentation
- Modify: `tools/semantic_search.py` module documentation
- Modify: `tools/background_contract.py` module/CLI help and error text
- Modify: `tools/ledger.py` P1-specific helper/error text
- Modify: `tools/validate_background.py`
- Modify: `tests/fixtures/semantic-space-coverage.json` only if an existing mapping is incorrect; do not add benchmark-specific state rules.

**Interfaces:**
- Consumes: all implemented P2 contracts.
- Produces: an authoritative maintainer description and network/GPU-free acceptance evidence for MLE-bench-shaped and PostTrainBench-shaped spaces.

- [ ] **Step 1: Add failing acceptance assertions**

In `validate_background.py`, retain the concrete fixture objects from the real-CLI lifecycle and add exact assertions that: `target-evidence` returns an old direct comparator omitted from a one-node graph window and its selected counts equal validator recomputation; `validate_registry(registry, ledger=ledger)` returns no errors for the pre-pruning record after later decisions; no revision-current proposal selects the pruned hypothesis; every active assignment for a runtime-pruned dimension names its `baseline_hypothesis_id`; applying a crash-only belief returns no decision ids; applying a later active recommendation appends a reopen decision and makes the hypothesis proposal-eligible; the saved `dag_revision` equals its value before policy transitions; and `search_space_state.revision` equals the number of saved decisions and is greater than zero.

Run the same state/proposal helpers over the existing `mle_bench_shaped` and `posttrain_bench_shaped` ownership maps to prove there is no estimator-specific branch.

- [ ] **Step 2: Run and verify the documentation/acceptance gap**

Run: `python tools/validate_background.py`

Expected: FAIL until every P2 lifecycle assertion is satisfied.

- [ ] **Step 3: Update architectural documentation**

Document this exact flow:

```text
background schema 3 (immutable S)
  -> semantic point selection at search_space_state revision r
  -> candidate + persistent semantic edge receipts
  -> score/crash observations
  -> bounded experience schema 3 (replaceable belief)
  -> deterministic validated decision transition
  -> append-only search_space_state revision r+1
  -> next proposal set filters/orders against r+1
```

Explain that a “dimension added/removed” edge in the roadmap is represented inside a fixed run as conditional `dimension_activated`/`dimension_deactivated`; changing registry membership is P4 expansion and remains deferred. Explain why dimension pruning pins the explicit baseline instead of changing point arity or `space_revision`.

Document `background_contract.py target-evidence` as the extractor's authoritative bounded source for target edge ids, observations, evaluation state, and cited comparator counts. Also document the strict round-serial admission boundary: extraction/state application happens only at quiescent round boundaries, while propose → select → `add-record` completes before candidate implementation; concurrent admission is deferred until it has an explicit revision contract.

- [ ] **Step 4: Update repository guides**

Replace statements that P2 is deferred with the implemented boundaries. Keep the terminology warning that current `experience` is derived belief, while raw records are durable history. Do not describe LLM confidence as calibrated probability.

Update stale P1-only module docstrings, CLI help, and user-facing validation errors in the four touched tool modules. Retain historical phrases such as “pre-P1 flat registry” only where they still describe an intentionally rejected artifact.

Update the workspace-root `AGENTS.md` Current-to-Target Bridge to the implemented P2 contracts (`semantic_edges` receipts, experience schema 3, `search_space_state` overlay) in a separate commit in the root checkout; it sits outside this repository and outside every commit list in this plan.

- [ ] **Step 5: Run the focused suite**

Run: `python -m unittest tests.test_semantic_evidence tests.test_search_space_state tests.test_background_contract tests.test_dag_incremental tests.test_harness_controls -v`

Expected: PASS.

- [ ] **Step 6: Run deterministic repository validators**

Run: `python tools/validate_background.py`

Expected: `P2 semantic background, edge, belief, and policy-state checks passed.`

Run: `python tools/validate_got.py`

Expected: all graph-selection regression groups PASS; P2 does not alter PUCB ownership.

Run: `python tools/validate_skills.py`

Expected: PASS.

Run: `python tools/validate_tasks.py`

Expected: PASS.

- [ ] **Step 7: Verify worktree scope and whitespace**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only the P2 files listed in this plan are modified; nothing under `runs/`, `tasks/*/prepare.py`, benchmark checkouts, or reference repositories appears.

- [ ] **Step 8: Commit P2 documentation and acceptance coverage**

```bash
git add docs/background-research.md docs/search-space.md AGENTS.md CLAUDE.md README_ZH.md tools/semantic_space.py tools/semantic_search.py tools/background_contract.py tools/ledger.py tools/validate_background.py tests/fixtures/semantic-space-coverage.json
git commit -m "docs: define P2 semantic belief and pruning loop"
```

---

## Final Review Gate

Before declaring P2 complete, review the combined diff against these questions:

- Can a reader identify which bytes are space definition, observations, derived belief, and selection policy?
- Can every semantic edge, belief claim, and pruning transition be traced to stable run/edge ids?
- Can the extractor obtain each target's cited edge ids, score/status observations, and exact comparator counts from `target-evidence` without relying on the Top/Bottom window?
- Can state refresh occur only at a quiescent round boundary, with no admitted candidate work at risk from strict revision equality?
- Does later pruning leave earlier records and points valid at their recorded state revisions?
- Are external `excluded`, runtime `pruned`, `deprioritized`, `failed`, and `unevaluated` visibly distinct?
- Can a target be reopened without deleting or rewriting its prior decision?
- Can a crash inform feasibility without becoming contradiction evidence?
- Does dimension pruning pin a baseline without changing point arity or the frozen background revision?
- Does the incremental experience path remain bounded by DAG delta, fixed Top/Bottom anchors, compact lineage, and bounded target collections?
- Are both runtime mirrors synchronized, and do both MLE-bench-shaped and PostTrainBench-shaped fixtures exercise the same generic machinery?

Only after every answer is yes should the implementation be assessed for commit/merge readiness.
