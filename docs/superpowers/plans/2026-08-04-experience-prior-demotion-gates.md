# Experience Prior + Demotion Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the loop consume its own experience: a deterministic carrier-count prior in candidate selection, and degraded demotion gates so repeated disasters actually reach the pruning overlay.

**Architecture:** Per `docs/superpowers/specs/2026-08-04-experience-prior-demotion-gates-design.md`. A new deterministic `hypothesis_carriers()` helper counts independent negative/positive contexts per hypothesis from raw ledger edges. It feeds three consumers: a new default selection policy `coverage_experience` (score = coverage + prior), relaxed demotion gates in the experience validator, and relaxed transition gating in the search-space-state overlay. The old LLM gain-prediction policies stay in the tree, dormant.

**Tech Stack:** Python 3 stdlib only; unittest-style tests run under pytest.

## Global Constraints

- Scores are always **lower-is-better**; a positive child−parent delta is a *negative* carrier.
- Additive schema evolution: bump schema versions, keep old versions readable, never rename established ids/fields.
- `.claude/` is the canonical runtime; `.opencode/` is stale — do not update it.
- Workspace-root `AGENTS.md` and `CLAUDE.md` are twins: apply any edit to both.
- No new dependencies. Crash edges never falsify a mechanism.
- Deterministic mechanisms stay outside LLM prose; the prior is computed from ledger edges, never authored.
- Tests extend existing files; no new test files, no new validators beyond what tasks specify.
- Do not commit anything under `runs/`.

---

### Task 1: `hypothesis_carriers` in semantic_evidence.py

**Files:**
- Modify: `tools/semantic_evidence.py` (insert after `_finite_score`, line 816)
- Test: `tests/test_semantic_evidence.py`

**Interfaces:**
- Produces: `hypothesis_carriers(ledger: dict, *, target_id: str) -> dict` returning
  `{"negative": int, "positive": int, "negative_contexts": list[str], "positive_contexts": list[str]}`.
  Tasks 2, 3, 4 all consume exactly this shape. `negative_contexts`/`positive_contexts`
  are sorted parent run-id lists (one entry per independent context).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_semantic_evidence.py`. Add `hypothesis_carriers` to the
`from semantic_evidence import (...)` list at the top of the file. Add this
local helper and test class (the existing `_append` in this file calls
`attach_matched_transfer`, which carriers must be independent of — so use a
plain builder):

```python
def _append_plain(
    records: list[dict],
    run_id: str,
    parents: list[str],
    point: dict,
    *,
    status: str,
    score: float | None,
    dag_revision: int,
    warm: float | None = None,
) -> dict:
    record = {
        "run_id": run_id,
        "source_run_ids": parents,
        "semantic_point": point,
        "status": status,
        "final_best_score": score,
        "evaluation_depth": "screening",
        "dag_revision": dag_revision,
    }
    if warm is not None:
        record["best_warm_score"] = warm
    record["semantic_edges"] = build_semantic_edges(records, record)
    records.append(record)
    return record


class TestHypothesisCarriers(unittest.TestCase):
    def _registry_points(self):
        registry = fixture_registry()
        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        return registry, baseline, filtered

    def test_two_independent_negative_contexts(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="discard", score=0.50, dag_revision=2)
        _append_plain(records, "002", [], baseline, status="keep", score=0.41, dag_revision=3)
        _append_plain(records, "003", ["002"], filtered, status="discard", score=0.52, dag_revision=4)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 2)
        self.assertEqual(result["positive"], 0)
        self.assertEqual(result["negative_contexts"], ["000", "002"])
        self.assertEqual(result["positive_contexts"], [])

    def test_crash_child_never_counts(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="crash", score=None, dag_revision=2)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)

    def test_positive_context_reported(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="keep", score=0.38, dag_revision=2)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 1)

    def test_mixed_context_counts_neither(self):
        # Two children of the SAME parent adding the hypothesis, one better,
        # one worse: the single context is contested and counts neither way.
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="discard", score=0.50, dag_revision=2)
        _append_plain(records, "002", ["000"], filtered, status="keep", score=0.38, dag_revision=3)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)

    def test_equal_scores_do_not_count(self):
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        _append_plain(records, "001", ["000"], filtered, status="keep", score=0.40, dag_revision=2)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)

    def test_warm_scores_pair_with_warm_not_final(self):
        # Warm-vs-warm must win over final-vs-final: warm delta is negative
        # (better) here while the final delta is positive (worse).
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1, warm=0.45)
        _append_plain(records, "001", ["000"], filtered, status="keep", score=0.50, dag_revision=2, warm=0.44)
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 1)

    def test_missing_pair_does_not_count(self):
        # Parent has only a final, child has only a warm: no like-for-like pair.
        registry, baseline, filtered = self._registry_points()
        records: list[dict] = []
        _append_plain(records, "000", [], baseline, status="keep", score=0.40, dag_revision=1)
        child = _append_plain(records, "001", ["000"], filtered, status="keep", score=None, dag_revision=2, warm=0.60)
        del child["final_best_score"]  # no terminal final either
        result = hypothesis_carriers({"records": records}, target_id="hyp-data-filtered")
        self.assertEqual(result["negative"], 0)
        self.assertEqual(result["positive"], 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_semantic_evidence.py::TestHypothesisCarriers -q`
Expected: FAIL with `ImportError` / `cannot import name 'hypothesis_carriers'`.

- [ ] **Step 3: Implement**

In `tools/semantic_evidence.py`, immediately after `_finite_score` (ends line 816), insert:

```python
def _like_for_like_delta(
    parent: dict[str, Any], child: dict[str, Any]
) -> float | None:
    """child − parent at matched evaluation depth, else ``None``.

    Warm scores pair with warm scores; terminal finals pair with finals.  A
    warm score is never compared against a tuning-lowered final.
    """
    parent_warm = _finite_score(parent.get("best_warm_score"))
    child_warm = _finite_score(child.get("best_warm_score"))
    if parent_warm is not None and child_warm is not None:
        return child_warm - parent_warm
    parent_final = _terminal_score(parent)
    child_final = _terminal_score(child)
    if parent_final is not None and child_final is not None:
        return child_final - parent_final
    return None


def hypothesis_carriers(ledger: dict[str, Any], *, target_id: str) -> dict[str, Any]:
    """Count independent contexts where adding ``target_id`` hurt or helped.

    A carrier edge's child point adds the hypothesis relative to the edge's
    parent.  Contexts group by parent run id: a context is negative when
    every carrier delta in it is strictly worse (positive — scores are
    lower-is-better) and positive when every delta is strictly better.
    Mixed, zero-delta, crash, and depth-unpaired edges never count.
    """
    records = _records_by_id(ledger)
    contexts: dict[str, list[float]] = {}
    for record in ledger.get("records", []):
        if not isinstance(record, dict):
            continue
        if record.get("status") not in NONCRASH_TERMINAL_STATUSES:
            continue
        receipts = record.get("semantic_edges")
        if not isinstance(receipts, list):
            continue
        for receipt in receipts:
            if not isinstance(receipt, dict):
                continue
            changes = receipt.get("changes")
            if not isinstance(changes, list):
                continue
            adds = any(
                isinstance(change, dict)
                and change.get("to_hypothesis_id") == target_id
                and change.get("from_hypothesis_id") != target_id
                for change in changes
            )
            if not adds:
                continue
            parent = records.get(str(receipt.get("parent_run_id")))
            if (
                parent is None
                or parent.get("status") not in NONCRASH_TERMINAL_STATUSES
            ):
                continue
            delta = _like_for_like_delta(parent, record)
            if delta is None or abs(delta) <= 1e-12:
                continue
            contexts.setdefault(str(receipt.get("parent_run_id")), []).append(delta)
    negative = sorted(
        parent_id
        for parent_id, deltas in contexts.items()
        if all(delta > 0 for delta in deltas)
    )
    positive = sorted(
        parent_id
        for parent_id, deltas in contexts.items()
        if all(delta < 0 for delta in deltas)
    )
    return {
        "negative": len(negative),
        "positive": len(positive),
        "negative_contexts": negative,
        "positive_contexts": positive,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_semantic_evidence.py -q`
Expected: PASS (whole file, including pre-existing tests).

- [ ] **Step 5: Commit**

```bash
git -C /home/woden/spark/HieraResearch add tools/semantic_evidence.py tests/test_semantic_evidence.py
git -C /home/woden/spark/HieraResearch commit -m "feat: count independent hypothesis carrier contexts from ledger edges"
```

---

### Task 2: Relax demotion gates in background_contract.py

**Files:**
- Modify: `tools/background_contract.py` (`_validate_target_evidence`, gates at lines 2291–2339; import block)
- Test: `tests/test_background_contract.py`

**Interfaces:**
- Consumes: `hypothesis_carriers` from Task 1.
- Produces: validator accepts `recommended_status: deprioritized` when
  `assessment == "unpromising"`, `confidence in {"med","high"}`, non-empty
  `reopen_when`, and (strict comparator path **or** `negative >= 2 and
  positive == 0`); `pruned` needs confidence high and (strict path **or**
  `negative >= 3 and positive == 0`). `promising` unchanged (strict only).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_background_contract.py`. The file already defines
`_base_experience()` (line 472) and `_comparator_covered_entry()` (line 486)
and imports `belief_ledger`, `fixture_registry`, `complete_point`,
`build_semantic_edges`, `validate_experience` — mirror those patterns:

```python
def _confounded_ledger(registry: dict) -> dict:
    """belief_ledger's shape WITHOUT matched transfers: confounded edges only.

    Both children adding hyp-data-filtered are worse than their baseline
    parents (0.50 > 0.40, 0.52 > 0.41) — two independent negative carrier
    contexts, zero comparator coverage.
    """
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = [
        {
            "run_id": "000", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.40,
            "evaluation_depth": "screening", "dag_revision": 1,
        },
        {
            "run_id": "001", "source_run_ids": ["000"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.50,
            "evaluation_depth": "screening", "dag_revision": 2,
        },
        {
            "run_id": "002", "source_run_ids": [], "semantic_point": baseline,
            "semantic_edges": [], "status": "keep", "final_best_score": 0.41,
            "evaluation_depth": "screening", "dag_revision": 3,
        },
        {
            "run_id": "003", "source_run_ids": ["002"], "semantic_point": filtered,
            "status": "discard", "final_best_score": 0.52,
            "evaluation_depth": "screening", "dag_revision": 4,
        },
    ]
    records[1]["semantic_edges"] = build_semantic_edges(records[:1], records[1])
    records[3]["semantic_edges"] = build_semantic_edges(records[:3], records[3])
    return {"records": records, "dag_revision": 4}


def _carrier_demote_entry() -> dict:
    return {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "observed",
        "assessment": "unpromising",
        "recommended_status": "deprioritized",
        "claim": "Two independent confounded contexts were both strictly worse.",
        "evidence_run_ids": ["000", "001", "002", "003"],
        "evidence_edge_ids": ["sedge-000-001", "sedge-002-003"],
        "comparator_coverage": {
            "direct_tuned_edges": 0,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 2,
            "crash_edges": 0,
        },
        "confidence": "med",
        "uncertainty": "Implementation drift is an alternative explanation.",
        "reopen_when": "Any independent context where adding it improves the parent.",
    }


class TestCarrierDemotionGates(unittest.TestCase):
    def _validate(self, ledger, entry):
        experience = _base_experience()
        experience["hypothesis_evidence"] = [entry]
        return validate_experience(experience, ledger=ledger, registry=fixture_registry())

    def test_deprioritize_accepted_on_two_negative_contexts(self):
        registry = fixture_registry()
        errors = self._validate(_confounded_ledger(registry), _carrier_demote_entry())
        self.assertEqual(errors, [])

    def test_prune_rejected_on_only_two_negative_contexts(self):
        registry = fixture_registry()
        entry = _carrier_demote_entry()
        entry["recommended_status"] = "pruned"
        entry["confidence"] = "high"
        errors = self._validate(_confounded_ledger(registry), entry)
        self.assertTrue(any("pruned" in error for error in errors), errors)

    def test_positive_context_blocks_demotion(self):
        registry = fixture_registry()
        ledger = _confounded_ledger(registry)
        ledger["records"][3]["final_best_score"] = 0.39  # 003 now BEATS parent 002
        errors = self._validate(ledger, _carrier_demote_entry())
        self.assertTrue(any("deprioritized" in error for error in errors), errors)

    def test_promising_still_requires_comparator_coverage(self):
        registry = fixture_registry()
        entry = _carrier_demote_entry()
        entry["assessment"] = "promising"
        entry["recommended_status"] = "active"
        errors = self._validate(_confounded_ledger(registry), entry)
        self.assertTrue(any("promising" in error for error in errors), errors)
```

NOTE for the implementer: check the exact call signature of `validate_experience`
used by neighboring tests in this file (positional vs keyword argument order)
and match it; also check whether `_base_experience()` needs `schema_version: 3`
vs 4 for these tests — copy what the neighboring `recommended_status` tests
(lines ~491–826) do verbatim.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_background_contract.py::TestCarrierDemotionGates -q`
Expected: FAIL — `test_deprioritize_accepted_on_two_negative_contexts` reports gate errors.

- [ ] **Step 3: Implement**

In `tools/background_contract.py`:

1. Add `hypothesis_carriers` to the existing `from semantic_evidence import ...` import list.
2. In `_validate_target_evidence`, replace the gate block at lines 2291–2339
   (the `if recommended == "deprioritized" ...`, `if recommended == "pruned" ...`,
   `if assessment in {"promising", "unpromising"} ...`, and
   `if target_kind == "hypothesis" and assessment in {...}` blocks) with:

```python
        carriers = (
            hypothesis_carriers(ledger, target_id=target_id)
            if target_kind == "hypothesis"
            else {"negative": 0, "positive": 0}
        )
        carrier_demote = carriers["negative"] >= 2 and carriers["positive"] == 0
        carrier_prune = carriers["negative"] >= 3 and carriers["positive"] == 0
        strict_demote = state == "comparator_covered" and depth_bar
        if recommended == "deprioritized" and not (
            assessment == "unpromising"
            and confidence in {"med", "high"}
            and (strict_demote or carrier_demote)
            and _nonempty(item.get("reopen_when"))
        ):
            errors.append(
                f"{target}.recommended_status deprioritized requires assessment "
                "unpromising, confidence med or high, a non-empty reopen_when, "
                "and either comparator_covered evaluation_state with the depth "
                "bar, or at least two independent negative carrier contexts "
                "with no positive context"
            )
        if recommended == "pruned" and not (
            assessment == "unpromising"
            and confidence == "high"
            and (strict_demote or carrier_prune)
            and _nonempty(item.get("reopen_when"))
        ):
            errors.append(
                f"{target}.recommended_status pruned requires assessment "
                "unpromising, confidence high, a non-empty reopen_when, and "
                "either comparator_covered evaluation_state with the depth "
                "bar, or at least three independent negative carrier contexts "
                "with no positive context"
            )
        if assessment == "promising" and not strict_demote:
            errors.append(
                f"{target} assessment promising requires comparator_covered "
                "evaluation_state with at least two direct tuned edges, or at "
                "least three direct edges at tuned_lightly or deeper"
            )
        if assessment == "unpromising" and not (strict_demote or carrier_demote):
            errors.append(
                f"{target} assessment unpromising requires comparator_covered "
                "evaluation_state with the depth bar, or at least two "
                "independent negative carrier contexts with no positive context"
            )
        if (
            target_kind == "hypothesis"
            and assessment in {"promising", "unpromising"}
            and not (assessment == "unpromising" and carrier_demote)
        ):
            required_direction = (
                "positive" if assessment == "promising" else "negative"
            )
            if mechanical_direction != required_direction:
                errors.append(
                    f"{target} assessment {assessment} conflicts with the "
                    "direction of its repeated matched semantic-control pairs"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_background_contract.py -q`
Expected: PASS (whole file — including the pre-existing strict-gate tests, which
exercise comparator-covered ledgers and must be unaffected).

- [ ] **Step 5: Commit**

```bash
git -C /home/woden/spark/HieraResearch add tools/background_contract.py tests/test_background_contract.py
git -C /home/woden/spark/HieraResearch commit -m "feat: admit carrier-rule demotion alongside comparator gates"
```

---

### Task 3: Carrier plumbing in search_space_state.py

**Files:**
- Modify: `tools/search_space_state.py` (`DECISION_SCHEMA_VERSION` line 46,
  `READABLE_DECISION_SCHEMA_VERSIONS` line 53, `DECISION_FIELDS`, validation
  ~line 328–334, `_effective_recommendation` line 475, `_normalized_beliefs`
  lines 586–613, `_has_advancing_evidence` line 617, `_decision_receipt` line 713)
- Test: `tests/test_search_space_state.py`

**Interfaces:**
- Consumes: `hypothesis_carriers` from Task 1.
- Produces: `_effective_recommendation(..., carrier_demote: bool = False,
  carrier_prune: bool = False)`; normalized beliefs carry `"_carriers"` (the
  full `hypothesis_carriers` result); decision receipts carry
  `"carrier_contexts": {"negative": [...], "positive": [...]}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_search_space_state.py`. Reuse the file's existing
imports/patterns for registry fixtures (check its header; mirror how existing
`append_experience_transitions` tests at ~lines 541–1062 build their
registry/ledger/experience). Core test code:

```python
def _carrier_ledger_and_experience(registry, *, prune=False):
    """Confounded ledger with 2 (or 3) negative contexts + a demoting belief."""
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    rows = [
        ("000", [], baseline, "keep", 0.40),
        ("001", ["000"], filtered, "discard", 0.50),
        ("002", [], baseline, "keep", 0.41),
        ("003", ["002"], filtered, "discard", 0.52),
    ]
    if prune:
        rows += [
            ("004", [], baseline, "keep", 0.42),
            ("005", ["004"], filtered, "discard", 0.53),
        ]
    records = []
    for revision, (run_id, parents, point, status, score) in enumerate(rows, 1):
        record = {
            "run_id": run_id, "source_run_ids": parents,
            "semantic_point": point, "status": status,
            "final_best_score": score, "evaluation_depth": "screening",
            "dag_revision": revision,
        }
        record["semantic_edges"] = build_semantic_edges(records, record)
        records.append(record)
    pairs = [("000", "001"), ("002", "003")] + ([("004", "005")] if prune else [])
    edge_ids = [f"sedge-{parent}-{child}" for parent, child in pairs]
    run_ids = [row[0] for row in rows]
    entry = {
        "target_id": "hyp-data-filtered",
        "assessment": "unpromising",
        "confidence": "high" if prune else "med",
        "recommended_status": "pruned" if prune else "deprioritized",
        "claim": "Repeated independent negative contexts.",
        "uncertainty": "Implementation drift possible.",
        "reopen_when": "Any independent improving context.",
        "evidence_run_ids": run_ids,
        "evidence_edge_ids": edge_ids,
    }
    experience = {
        "schema_version": 4,
        "updated_at_run": run_ids[-1],
        "generation": 1,
        "dag_revision": len(rows),
        "summary": "test",
        "promising_regions": [],
        "lessons": [],
        "bottlenecks": [],
        "dimension_evidence": [],
        "hypothesis_evidence": [entry],
    }
    return {"records": records, "dag_revision": len(rows),
            "experience": experience,
            "search_space_state": empty_search_space_state()}
```python
class TestCarrierTransitions(unittest.TestCase):
    def test_deprioritize_fires_on_carrier_rule(self):
        registry = fixture_registry()
        ledger = _carrier_ledger_and_experience(registry)
        transitions = derive_experience_transitions(registry, ledger)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["target"]["id"], "hyp-data-filtered")
        self.assertEqual(transitions[0]["from_status"], "active")
        self.assertEqual(transitions[0]["to_status"], "deprioritized")
        self.assertEqual(transitions[0]["schema_version"], 4)
        self.assertEqual(
            transitions[0]["carrier_contexts"]["negative"], ["000", "002"]
        )

    def test_staging_still_applies_to_prune_recommendation(self):
        # Even with 3 negative contexts, active->pruned is forbidden: the
        # first transition is still a deprioritization.
        registry = fixture_registry()
        ledger = _carrier_ledger_and_experience(registry, prune=True)
        transitions = derive_experience_transitions(registry, ledger)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["to_status"], "deprioritized")

    def test_advance_to_pruned_on_new_negative_context(self):
        registry = fixture_registry()
        # First: 2 negative contexts, deprioritize lands in the overlay.
        ledger = _carrier_ledger_and_experience(registry)
        append_experience_transitions(registry, ledger)
        # Second: a third negative context appears; a new generation
        # recommends prune.
        baseline = complete_point(registry)
        filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        records = ledger["records"]
        for run_id, parents, point, status, score, revision in (
            ("004", [], baseline, "keep", 0.42, 5),
            ("005", ["004"], filtered, "discard", 0.53, 6),
        ):
            record = {
                "run_id": run_id, "source_run_ids": parents,
                "semantic_point": point, "status": status,
                "final_best_score": score, "evaluation_depth": "screening",
                "dag_revision": revision,
            }
            record["semantic_edges"] = build_semantic_edges(records, record)
            records.append(record)
        experience = ledger["experience"]
        experience["generation"] = 2
        experience["dag_revision"] = 6
        experience["updated_at_run"] = "005"
        entry = experience["hypothesis_evidence"][0]
        entry["recommended_status"] = "pruned"
        entry["confidence"] = "high"
        entry["evidence_run_ids"] = ["000", "001", "002", "003", "004", "005"]
        entry["evidence_edge_ids"] = ["sedge-000-001", "sedge-002-003", "sedge-004-005"]
        transitions = derive_experience_transitions(registry, ledger)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["from_status"], "deprioritized")
        self.assertEqual(transitions[0]["to_status"], "pruned")
```

(Implementer: confirm the exact import surface of this test file — it already
imports `derive_experience_transitions`/`append_experience_transitions`/
`empty_search_space_state` for its existing transition tests; add
`build_semantic_edges`, `complete_point`, `fixture_registry` if missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_search_space_state.py::TestCarrierTransitions -q`
Expected: FAIL — no transitions derived (gates reject the recommendation).

- [ ] **Step 3: Implement**

In `tools/search_space_state.py`:

1. Line 46: `DECISION_SCHEMA_VERSION = 4`. Line 53:
   `READABLE_DECISION_SCHEMA_VERSIONS = {1, 2, 3, 4}`.
   Add `"carrier_contexts"` to `DECISION_FIELDS`. In
   `validate_search_space_state` (~line 331), make the missing-field check
   tolerate pre-4 decisions:

```python
        missing = sorted(DECISION_FIELDS - set(decision))
        if (
            decision.get("schema_version") in {1, 2, 3}
            and "carrier_contexts" in missing
        ):
            missing.remove("carrier_contexts")
```

2. Add `hypothesis_carriers` to the existing `from semantic_evidence import ...` list.

3. Replace `_effective_recommendation` (lines 475–514) with:

```python
def _effective_recommendation(
    belief: dict[str, Any],
    evaluation_state: str,
    coverage: dict[str, int],
    *,
    target_kind: str,
    mechanical_direction: str,
    carrier_demote: bool = False,
    carrier_prune: bool = False,
) -> str | None:
    """Gate the authored recommendation on mechanically recomputed evidence.

    Demotion passes either the strict comparator path or the carrier rule
    (repeated independent negative contexts, zero positive); a ``pruned``
    recommendation that only meets the deprioritize gate is carried out as a
    deprioritization; a recommendation whose gates fail yields no transition
    at all (``None``), never a reopening.
    """
    recommended = belief.get("recommended_status")
    if recommended == "active":
        return "active"
    if recommended not in {"deprioritized", "pruned"}:
        return None
    strict = evaluation_state == "comparator_covered" and _contradiction_depth_bar(
        coverage
    )
    deprioritize_ok = (
        belief.get("assessment") == "unpromising"
        and belief.get("confidence") in {"med", "high"}
        and (strict or carrier_demote)
        and (
            target_kind != "hypothesis"
            or mechanical_direction == "negative"
            or carrier_demote
        )
        and _nonempty(belief.get("reopen_when"))
    )
    if not deprioritize_ok:
        return None
    if recommended == "deprioritized":
        return "deprioritized"
    prune_ok = belief.get("confidence") == "high" and (strict or carrier_prune)
    return "pruned" if prune_ok else "deprioritized"
```

4. In `_normalized_beliefs`, inside the per-item loop just before
   `beliefs[(target_kind, target_id)] = {...}` (line 586), compute carriers
   and thread them through:

```python
            carriers = (
                hypothesis_carriers(ledger, target_id=target_id)
                if target_kind == "hypothesis"
                else {
                    "negative": 0,
                    "positive": 0,
                    "negative_contexts": [],
                    "positive_contexts": [],
                }
            )
```

   Add to the belief dict (after `"comparator_coverage": coverage,`):
   `"_carriers": carriers,`
   and change the `_effective_recommendation(...)` call to add:

```python
                    carrier_demote=(
                        carriers["negative"] >= 2 and carriers["positive"] == 0
                    ),
                    carrier_prune=(
                        carriers["negative"] >= 3 and carriers["positive"] == 0
                    ),
```

5. In `_decision_receipt`, add to the returned dict (after
   `"evidence_observations": observations,`):

```python
        "carrier_contexts": {
            "negative": list(belief["_carriers"]["negative_contexts"]),
            "positive": list(belief["_carriers"]["positive_contexts"]),
        },
```

6. Replace the tail of `_has_advancing_evidence` (from `direct_current = [` at
   line 634 to the end of the function) with:

```python
    direct_current = [
        item
        for item in belief.get("_direct_observations", [])
        if isinstance(item, dict)
        and {
            item.get("parent_status"),
            item.get("child_status"),
        }.issubset(TERMINAL_STATUSES)
    ]
    if any(item.get("edge_id") not in prior_edge_ids for item in direct_current):
        return True
    prior_by_edge = {
        item.get("edge_id"): item
        for item in prior_observations
        if isinstance(item, dict) and isinstance(item.get("edge_id"), str)
    }
    if any(
        isinstance(item, dict)
        and isinstance(item.get("edge_id"), str)
        and prior_by_edge.get(item["edge_id"]) != item
        for item in direct_current
    ):
        return True
    # Carrier path: a changed set of independent contexts (a new negative
    # context advances demotion; a new positive one advances reopening).
    last_carriers = last.get("carrier_contexts")
    if isinstance(last_carriers, dict):
        current_carriers = belief.get("_carriers") or {}
        return (
            list(current_carriers.get("negative_contexts") or [])
            != list(last_carriers.get("negative") or [])
            or list(current_carriers.get("positive_contexts") or [])
            != list(last_carriers.get("positive") or [])
        )
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_search_space_state.py -q`
Expected: PASS (whole file — existing staging/reopen tests must be unaffected).

- [ ] **Step 5: Commit**

```bash
git -C /home/woden/spark/HieraResearch add tools/search_space_state.py tests/test_search_space_state.py
git -C /home/woden/spark/HieraResearch commit -m "feat: drive pruning-overlay transitions from carrier evidence"
```

---

### Task 4: `coverage_experience` policy in semantic_search.py

**Files:**
- Modify: `tools/semantic_search.py` (constants lines 77–94,
  `_experience_snapshot_receipt` line 129, new `_carrier_priors` before
  `select_proposal`, config validation ~lines 1182–1215, prediction block
  ~1217–1227, scoring ~1242–1268, components ~1269–1293, lane block
  ~1297–1322, rationale ~1337–1341, receipt ~1343–1364, `cmd_select` default
  line 1482)
- Modify: `tools/semantic_evidence.py` (`matched_inherited_control` receipt
  schema check, line 1517)
- Test: `tests/test_semantic_policy_default.py`

**Interfaces:**
- Consumes: `hypothesis_carriers` from Task 1.
- Produces: policy `"coverage_experience"`; config knobs
  `carrier_pos_weight` (0.05), `carrier_pos_cap` (2), `carrier_neg_weight`
  (0.2), `carrier_neg_cap` (3); `POLICY_RECEIPT_SCHEMA_VERSION = 7`; receipt
  `components` gains `experience_prior` and `carriers` keys.

- [ ] **Step 1: Write the failing tests**

In `tests/test_semantic_policy_default.py`:

1. Find `test_default_policy_is_pure_coverage_in_template_and_cli` (~line 581)
   and rewrite it to assert the new default — same structure, new name/value:

```python
    def test_default_policy_is_coverage_experience_in_template_and_cli(self):
        # template default
        template = json.loads(
            (ROOT / "tasks" / "framework_cfg.example.json").read_text()
        )
        self.assertEqual(
            template["semantic_search"]["policy"], "coverage_experience"
        )
        # CLI fallback default (no --policy, no configured policy)
        ...  # keep the existing test's CLI assertion mechanics, expecting
        ...  # policy "coverage_experience" instead of "coverage"
```

(Implementer: read the existing test first and preserve its exact CLI-driving
mechanics; only the expected policy string changes. If the file references
`Path` / `json` differently, match its conventions.)

2. Append a new test class. Mirror the file's existing helpers for building a
registry/ledger/proposal set (check its imports for `build_proposal_set`,
`select_proposal`, `complete_point`, `fixture_registry`,
`build_semantic_edges`, `selected_assignments`):

```python
def _two_negative_ledger(registry):
    baseline = complete_point(registry)
    filtered = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
    records = []
    for revision, (run_id, parents, point, status, score) in enumerate([
        ("000", [], baseline, "keep", 0.40),
        ("001", ["000"], filtered, "discard", 0.50),
        ("002", [], baseline, "keep", 0.41),
        ("003", ["002"], filtered, "discard", 0.52),
    ], 1):
        record = {
            "run_id": run_id, "source_run_ids": parents,
            "semantic_point": point, "status": status,
            "final_best_score": score, "evaluation_depth": "screening",
            "dag_revision": revision,
        }
        record["semantic_edges"] = build_semantic_edges(records, record)
        records.append(record)
    return {"records": records, "dag_revision": 4}


class TestCoverageExperiencePolicy(unittest.TestCase):
    def test_repeated_negative_point_loses_to_clean_point(self):
        registry = fixture_registry()
        ledger = _two_negative_ledger(registry)
        proposals = build_proposal_set(registry, ledger, op="fresh", parents=[])
        point, receipt = select_proposal(
            proposals, policy="coverage_experience", ledger=ledger
        )
        self.assertNotEqual(
            selected_assignments(point).get("dim-data-curation"),
            "hyp-data-filtered",
        )
        self.assertEqual(receipt["schema_version"], 7)
        self.assertEqual(receipt["policy"]["name"], "coverage_experience")
        self.assertEqual(receipt["budget"]["fallback"], "lanes_removed")

    def test_zero_evidence_matches_coverage_choice(self):
        registry = fixture_registry()
        ledger = {"records": []}
        proposals = build_proposal_set(registry, ledger, op="fresh", parents=[])
        point_cov, _ = select_proposal(proposals, policy="coverage", ledger=ledger)
        point_exp, receipt = select_proposal(
            proposals, policy="coverage_experience", ledger=ledger
        )
        self.assertEqual(point_cov["point_id"], point_exp["point_id"])
        self.assertEqual(receipt["components"]["experience_prior"], 0.0)

    def test_components_expose_prior_and_carriers(self):
        registry = fixture_registry()
        ledger = _two_negative_ledger(registry)
        proposals = build_proposal_set(registry, ledger, op="fresh", parents=[])
        _, receipt = select_proposal(
            proposals, policy="coverage_experience", ledger=ledger
        )
        carriers = receipt["components"]["carriers"]
        self.assertIsInstance(carriers, dict)
        for detail in carriers.values():
            self.assertIn("negative", detail)
            self.assertIn("positive", detail)

    def test_schema4_experience_accepted(self):
        registry = fixture_registry()
        ledger = _two_negative_ledger(registry)
        ledger["experience"] = {
            "schema_version": 4,
            "updated_at_run": "003",
            "generation": 1,
        }
        proposals = build_proposal_set(registry, ledger, op="fresh", parents=[])
        _, receipt = select_proposal(
            proposals, policy="coverage_experience", ledger=ledger
        )
        self.assertEqual(receipt["experience"]["generation"], 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_semantic_policy_default.py -q`
Expected: FAIL — `policy must be one of [...]` for `coverage_experience`, and the
template assertion fails.

- [ ] **Step 3: Implement**

In `tools/semantic_search.py`:

1. Line 83:
   `POLICIES = {"coverage", "coverage_experience", "gain", "gain_uncertainty", "gain_uncertainty_nocost"}`
2. `DEFAULT_POLICY_CONFIG` (lines 84–90) — add four knobs:

```python
DEFAULT_POLICY_CONFIG = {
    "coverage_weight": 0.10,
    "cost_weight": 0.20,
    "uncertainty_weight": 0.50,
    "deprioritized_budget_interval": 5,
    "llm_intelligence_score": 100.0,
    "carrier_pos_weight": 0.05,
    "carrier_pos_cap": 2,
    "carrier_neg_weight": 0.20,
    "carrier_neg_cap": 3,
}
```

3. Add `hypothesis_carriers` to the `from semantic_evidence import ...` list,
   and ensure `selected_assignments` is imported (it is already used at line 498).
4. `_experience_snapshot_receipt` line 129: change
   `experience.get("schema_version") != 3` to
   `experience.get("schema_version") not in {3, 4}` and update the error
   message to "a valid schema-3/4 snapshot".
5. Insert before `select_proposal` (line 1156):

```python
def _carrier_priors(
    proposal_set: dict[str, Any],
    ledger: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, tuple[float, dict[str, dict[str, int]]]]:
    """Deterministic per-point experience prior from hypothesis carriers."""
    cache: dict[str, dict[str, Any]] = {}
    priors: dict[str, tuple[float, dict[str, dict[str, int]]]] = {}
    for proposal in proposal_set["proposals"]:
        prior = 0.0
        detail: dict[str, dict[str, int]] = {}
        for hypothesis_id in sorted(
            set(selected_assignments(proposal["point"]).values())
        ):
            if hypothesis_id not in cache:
                cache[hypothesis_id] = hypothesis_carriers(
                    ledger, target_id=hypothesis_id
                )
            carriers = cache[hypothesis_id]
            prior += min(carriers["positive"], int(cfg["carrier_pos_cap"])) * float(
                cfg["carrier_pos_weight"]
            ) - min(carriers["negative"], int(cfg["carrier_neg_cap"])) * float(
                cfg["carrier_neg_weight"]
            )
            detail[hypothesis_id] = {
                "negative": carriers["negative"],
                "positive": carriers["positive"],
            }
        priors[proposal["point_id"]] = (round(prior, 10), detail)
    return priors
```

6. In `select_proposal`'s config-validation loop (~line 1182), add cap
   validation before the generic non-negative-number branch:

```python
            if key in {"carrier_pos_cap", "carrier_neg_cap"}:
                if (
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 1 <= value <= 100
                ):
                    raise ContractError(
                        f"semantic policy config {key} must be an integer in [1, 100]"
                    )
                cfg[key] = value
                continue
```

7. After the `if policy != "coverage":` prediction block (~line 1227), add:

```python
    carrier_priors: dict[str, tuple[float, dict[str, dict[str, int]]]] = {}
    if policy == "coverage_experience":
        if not isinstance(ledger, dict):
            raise ContractError("coverage_experience selection requires the ledger")
        carrier_priors = _carrier_priors(proposal_set, ledger, cfg)
```

8. Scoring (~lines 1242–1268): insert a branch before `elif policy == "gain":`

```python
        elif policy == "coverage_experience":
            score = coverage + carrier_priors[point_id_value][0]
```

9. Components (~line 1269): add two keys:

```python
            "experience_prior": (
                None
                if policy != "coverage_experience"
                else carrier_priors[point_id_value][0]
            ),
            "carriers": (
                None
                if policy != "coverage_experience"
                else carrier_priors[point_id_value][1]
            ),
```

10. Replace the lane-scheduling block (~lines 1297–1322, from
    `interval = cfg["deprioritized_budget_interval"]` through
    `score, selected_id, selected, components = selected_item`) with:

```python
    selected_item = ranked[0]
    base_rank = 1
    final_ranked = ranked
    score, selected_id, selected, components = selected_item
```

    (`budget_lane` is still computed in `build_proposal_set` for backward
    readability; it no longer affects selection.)

11. Rationale (~lines 1337–1341): restructure to

```python
    if prediction is not None:
        rationale = prediction["experience_rationale"]
    elif policy == "coverage_experience":
        rationale = (
            "deterministic carrier prior over recorded edges; "
            "no model-scored experience"
        )
    else:
        rationale = "coverage policy does not use model-scored experience"
```

    and set `"rationale": rationale` in `experience_receipt`.

12. Line 82: `POLICY_RECEIPT_SCHEMA_VERSION = 7`. In the receipt's `budget`
    block (~line 1355):

```python
        "budget": {
            "selection_index": selection_index,
            "deprioritized_interval": None,
            "scheduled_lane": None,
            "selected_lane": None,
            "fallback": "lanes_removed",
            "base_rank": base_rank,
        },
```

13. Line 1509 (`cmd_select` llm-score freeze check): change
    `prior_receipt.get("schema_version") == 6` to
    `prior_receipt.get("schema_version") in {6, 7}`.
14. Line 1482 (`cmd_select` fallback): `"coverage"` → `"coverage_experience"`.

In `tools/semantic_evidence.py`:

15. Line 1517 (`matched_inherited_control`): change
    `child["policy_receipt"].get("schema_version") != 6` to
    `child["policy_receipt"].get("schema_version") not in {6, 7}`.

Then grep for any other schema-6 receipt checks and fix them the same way:

Run: `grep -rn 'schema_version.*[!=]=.*6\|"schema_version": 6' /home/woden/spark/HieraResearch/tools/ /home/woden/spark/HieraResearch/tests/`
Expected: only test fixtures intentionally building schema-6 receipts (leave
those; they test the readable-old-version path) and the updated `in {6, 7}`
checks.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/test_semantic_policy_default.py tests/test_semantic_evidence.py tests/test_search_space_state.py tests/test_background_contract.py -q`
Expected: PASS. Then run the full suite: `python3 -m pytest tests/ -q` — PASS.
(Existing lane-scheduling tests, if any, will fail against the removed lanes;
update those tests to the lane-less receipt shape — that is an intended
behavior change, not a regression.)

- [ ] **Step 5: Commit**

```bash
git -C /home/woden/spark/HieraResearch add tools/semantic_search.py tools/semantic_evidence.py tests/test_semantic_policy_default.py
git -C /home/woden/spark/HieraResearch commit -m "feat: select points by coverage plus deterministic carrier prior"
```

---

### Task 5: Template, prompts, and docs

**Files:**
- Modify: `tasks/framework_cfg.example.json` (semantic_search section line 27–34, `_keys`)
- Modify: `.claude/agents/autoresearch-experiment.md:172`
- Modify: `.claude/agents/idea-generator.md:117-126`
- Modify: `.claude/agents/experience-extractor.md:180-210`
- Modify: `.claude/rules/ledger.md` (gate rules copy, ~lines 176–211)
- Modify: `/home/woden/spark/AGENTS.md` and `/home/woden/spark/CLAUDE.md` (twin rule)

**Interfaces:**
- Consumes: Tasks 1–4 merged behavior.
- Produces: no code interfaces; every consumer of the contracts sees the new defaults.

- [ ] **Step 1: Update the config template**

In `tasks/framework_cfg.example.json`:

```json
  "semantic_search": {
    "policy": "coverage_experience",
    "coverage_weight": 0.1,
    "cost_weight": 0.2,
    "uncertainty_weight": 0.5,
    "deprioritized_budget_interval": 5,
    "llm_intelligence_score": 100,
    "carrier_pos_weight": 0.05,
    "carrier_pos_cap": 2,
    "carrier_neg_weight": 0.2,
    "carrier_neg_cap": 3
  },
```

Update `_keys`:

- `semantic_search.policy`: replace the description with:
  "outer semantic point policy: coverage_experience (default; deterministic coverage plus the carrier prior — per-hypothesis independent negative/positive context counts from ledger edges, no LLM scores), coverage (coverage only), gain, gain_uncertainty, or gain_uncertainty_nocost. The last three require per-proposal rubric inputs and are currently dormant."
- `semantic_search.deprioritized_budget_interval`: append "LEGACY: lane scheduling was removed; the key is still accepted but ignored — deprioritized content is penalized via the carrier prior instead."
- Add four entries:

```json
    "semantic_search.carrier_pos_weight": "experience prior: acquisition bonus per independent positive carrier context for a hypothesis (default 0.05).",
    "semantic_search.carrier_pos_cap": "experience prior: maximum positive contexts counted per hypothesis (default 2).",
    "semantic_search.carrier_neg_weight": "experience prior: acquisition penalty per independent negative carrier context for a hypothesis (default 0.2 — deliberately larger than the positive weight: re-probing a deadend wastes budget, a missed exploration costs nothing measurable).",
    "semantic_search.carrier_neg_cap": "experience prior: maximum negative contexts counted per hypothesis (default 3)."
```

- [ ] **Step 2: Update the coordinator's hardcoded policy**

In `.claude/agents/autoresearch-experiment.md:172`, change
`--policy coverage` to `--policy coverage_experience`. Then
`grep -rn -- '--policy' .claude/` from the repo root and update any other
hardcoded occurrence the same way.

- [ ] **Step 3: Update idea-generator policy section**

In `.claude/agents/idea-generator.md` (lines 117–126): change "If absent, use
`coverage`." to "If absent, use `coverage_experience`." and add a bullet at
the top of the policy list:

```markdown
- `coverage_experience` (default): coverage plus a deterministic per-hypothesis
  carrier prior computed from ledger edges — independent contexts where adding
  a hypothesis made its parent strictly worse penalize every point that selects
  it, improving contexts give a smaller bonus;
```

- [ ] **Step 4: Update the experience-extractor gates**

In `.claude/agents/experience-extractor.md`, replace the gate bullets at
lines 184–198 with:

```markdown
- `deprioritized` requires `assessment: unpromising`, `confidence: med` or
  `high`, a non-empty `reopen_when`, and either the strict path
  (`evaluation_state: comparator_covered` with the depth bar) or the carrier
  rule: at least two independent negative carrier contexts (distinct parents
  whose children adding the hypothesis scored strictly worse at like-for-like
  depth) and zero positive contexts. Crash edges never count.
- `pruned` requires `assessment: unpromising`, `confidence: high`, a
  non-empty `reopen_when`, and either the strict path or the carrier rule at
  three or more independent negative carrier contexts with zero positive.
- `promising` still requires `comparator_covered` with the depth bar
  (≥2 direct tuned edges, or ≥3 direct edges at `tuned_lightly` or deeper),
  regardless of prose confidence. `unpromising` passes on the carrier rule.
- A hypothesis `promising`/`unpromising` assessment must agree with the
  mechanical direction of all cited repeated pairs; mixed signs require
  `assessment: mixed`. The carrier rule itself satisfies direction agreement
  for `unpromising`.
```

Apply the matching edit to the rules copy in `.claude/rules/ledger.md`
(~lines 176–211 — find the same gate bullets and mirror the new text).

- [ ] **Step 5: Amend the workspace invariant (twin files)**

In BOTH `/home/woden/spark/AGENTS.md` and `/home/woden/spark/CLAUDE.md`,
find the invariant "Strong support or contradiction requires comparator
coverage, not merely a good or bad descendant score. Record uncertainty when
attribution is weak." and append to that bullet:

```
  Demotion is the exception: consistent negative carrier contexts (≥2
  independent parents, zero positive) suffice to deprioritize, ≥3 to prune;
  promotion still requires comparator coverage.
```

- [ ] **Step 6: Verify**

Run: `cd /home/woden/spark/HieraResearch && python3 -c "import json; json.load(open('tasks/framework_cfg.example.json'))"` — no output (valid JSON).
Run: `cd /home/woden/spark/HieraResearch && python3 -m pytest tests/ -q` — PASS.
Run: `grep -rn '"coverage"' .claude/ tasks/framework_cfg.example.json | grep -v coverage_experience | grep -v gain` — review each hit is intentional (describing the still-available `coverage` policy, not the default).

- [ ] **Step 7: Commit**

```bash
git -C /home/woden/spark/HieraResearch add tasks/framework_cfg.example.json .claude/
git -C /home/woden/spark/HieraResearch commit -m "docs: default to coverage_experience and document carrier-rule demotion"
git -C /home/woden/spark add AGENTS.md CLAUDE.md
git -C /home/woden/spark commit -m "docs: amend contradiction invariant for carrier-rule demotion"
```

---

## Deferred (not in this plan)

- Continuation scheduling and bout-size questions from remoteV.
- Server-code-skew confirmation (remoteV receipts cite schema-4 experience
  this checkout's old `select` would reject) — confirm which checkout the
  server runs before the next experiment; not a code change here.
- Removing the dormant LLM gain-prediction machinery.
