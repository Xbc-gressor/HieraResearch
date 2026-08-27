from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from background_contract import (  # noqa: E402
    ContractError,
    derive_hypothesis_selection,
    render_space,
    validate_experience,
    validate_ledger,
)
from ledger import (  # noqa: E402
    _load_ledger,
    cmd_add_record,
    cmd_brief,
    cmd_set_experience,
    cmd_set_phase,
    record_run,
    resolve_unevaluated,
)
from search_space_state import (  # noqa: E402
    append_experience_transitions,
    compose_effective_selection,
    derive_experience_transitions,
    empty_search_space_state,
    replay_search_space_state,
    runtime_status_counts,
    validate_point_eligibility,
    validate_search_space_state,
)
from semantic_evidence import (  # noqa: E402
    build_semantic_edges,
    target_evaluation_state,
)
from semantic_search import (  # noqa: E402
    build_proposal_set,
    cmd_select,
    select_proposal,
)
from semantic_space import (  # noqa: E402
    complete_point,
    digest,
    selected_assignments,
    space_receipt,
    validate_point,
)
from tests.fixtures import (  # noqa: E402
    attach_matched_transfer,
    background_text,
    belief_ledger,
    fixture_registry,
    policy_receipt,
    record,
)


def current_policy_receipt(*args, **kwargs) -> dict:
    """The fixture receipt at the current admission schema."""
    return policy_receipt(*args, schema_version=6, **kwargs)


def empty_experience(run_id: str, *, generation: int = 0) -> dict:
    return {
        "schema_version": 3,
        "updated_at_run": run_id,
        "generation": generation,
        "summary": "No comparator-qualified belief changed in this DAG delta.",
        "promising_regions": [],
        "lessons": [],
        "bottlenecks": [],
        "dimension_evidence": [],
        "hypothesis_evidence": [],
    }


def _scope(intervention: str) -> dict:
    return {
        "model_families": ["toy-models"],
        "data_regimes": ["toy-data"],
        "metrics": ["validation-loss"],
        "interventions": [intervention],
        "evaluation_protocols": ["heldout-split"],
    }


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
            "direct_tuned_edges": 2,
            "direct_lightly_tuned_edges": 0,
            "direct_noncrash_edges": 0,
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


def hypothesis_decision(
    revision: int, dimension_id: str, target_id: str, before: str, after: str
) -> dict:
    value = decision(revision, target_id, before, after)
    value["target"] = {"kind": "hypothesis", "dimension_id": dimension_id, "id": target_id}
    return value


def dimension_decision(revision: int, dimension_id: str, before: str, after: str) -> dict:
    value = decision(revision, dimension_id, before, after)
    value["target"] = {"kind": "dimension", "dimension_id": dimension_id, "id": dimension_id}
    return value


def state_with(*decisions: dict) -> dict:
    return {
        "schema_version": 1,
        "revision": len(decisions),
        "decisions": list(decisions),
    }


def ledger_with_state(state: dict | None, *, records: list | None = None) -> dict:
    ledger: dict = {"records": records if records is not None else []}
    if state is not None:
        ledger["search_space_state"] = state
    return ledger


class SearchSpaceStateValidationTests(unittest.TestCase):
    def test_empty_state_is_valid_and_replays_all_active(self) -> None:
        registry = fixture_registry()
        state = empty_search_space_state()
        self.assertEqual(
            validate_search_space_state(registry, ledger_with_state(state)), []
        )
        replayed = replay_search_space_state(registry, state)
        self.assertEqual(set(replayed["dimensions"].values()), {"active"})
        self.assertEqual(set(replayed["hypotheses"].values()), {"active"})
        self.assertIn("hyp-data-filtered", replayed["hypotheses"])

    def test_validate_accepts_contiguous_transitions_and_reopen(self) -> None:
        registry = fixture_registry()
        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "deprioritized", "pruned"),
            decision(3, "hyp-data-filtered", "pruned", "active"),
            dimension_decision(4, "dim-ensemble", "active", "deprioritized"),
            dimension_decision(5, "dim-ensemble", "deprioritized", "pruned"),
        )
        self.assertEqual(
            validate_search_space_state(registry, ledger_with_state(state)), []
        )

    def test_replay_is_addressable_by_revision_and_preserves_identity(self) -> None:
        registry = fixture_registry()
        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "deprioritized", "pruned"),
            decision(3, "hyp-data-filtered", "pruned", "active"),
        )

        def status_at(revision: int | None) -> str:
            replayed = replay_search_space_state(registry, state, revision=revision)
            # A pruned element stays present; only its eligibility changed.
            self.assertIn("hyp-data-filtered", replayed["hypotheses"])
            return replayed["hypotheses"]["hyp-data-filtered"]

        self.assertEqual(status_at(1), "deprioritized")
        self.assertEqual(status_at(2), "pruned")
        self.assertEqual(status_at(None), "active")

    def test_validate_rejects_unauditable_or_forged_decisions(self) -> None:
        """The overlay alone gates eligibility, so every field is mechanical."""
        registry = fixture_registry()

        def mutate(*decisions: dict, **changes: object) -> dict:
            state = state_with(*decisions)
            for key, value in changes.items():
                if key == "decision":
                    state["decisions"][0].update(value)
                elif key == "drop_field":
                    del state["decisions"][0][value]
                else:
                    state[key] = value
            return state

        one = decision(1, "hyp-data-filtered", "active", "deprioritized")
        cases = [
            # A contraction must pass through deprioritized, never skip it.
            ("transition", mutate(decision(1, "hyp-data-filtered", "active", "pruned"))),
            # Each decision must start where the previous one left the target.
            (
                "from_status",
                mutate(one, decision(2, "hyp-data-filtered", "active", "deprioritized")),
            ),
            # Revision must equal the append-only decision count.
            ("revision", mutate(one, revision=0)),
            # decision_id is derived from the revision, not chosen.
            ("decision_id", mutate(one, decision={"decision_id": "sdec-000009"})),
            # Free prose is not auditable evidence.
            ("unknown fields", mutate(one, decision={"note": "free text"})),
            ("unknown fields", mutate(one, state_note="not in the contract")),
            # A reversible decision must say what would reopen it.
            ("reopen_when", mutate(one, drop_field="reopen_when")),
            # Targets must resolve in the frozen registry, under their owner.
            ("target", mutate(decision(1, "hyp-not-real", "active", "deprioritized"))),
            (
                "dimension_id",
                mutate(
                    hypothesis_decision(
                        1,
                        "dim-model-architecture",
                        "hyp-data-filtered",
                        "active",
                        "deprioritized",
                    )
                ),
            ),
            ("target", mutate(dimension_decision(1, "dim-not-real", "active", "deprioritized"))),
            # A baseline is the comparison floor and can never be contracted.
            ("baseline", mutate(decision(1, "hyp-data-raw", "active", "deprioritized"))),
            (
                "baseline_only",
                mutate(
                    dimension_decision(
                        1, "dim-initialization-adaptation", "active", "deprioritized"
                    )
                ),
            ),
        ]
        for needle, state in cases:
            with self.subTest(needle=needle):
                errors = validate_search_space_state(registry, ledger_with_state(state))
                self.assertTrue(any(needle in error for error in errors), errors)


class EffectiveSelectionTests(unittest.TestCase):
    def test_compose_reports_all_component_statuses(self) -> None:
        registry = fixture_registry()
        guidance = derive_hypothesis_selection(registry)
        runtime = {
            "dimensions": {"dim-data-curation": "active"},
            "hypotheses": {"hyp-data-filtered": "pruned"},
        }
        effective = compose_effective_selection(registry, guidance, runtime)
        self.assertEqual(
            effective["hyp-data-filtered"],
            {
                "guidance_status": "active",
                "dimension_runtime_status": "active",
                "hypothesis_runtime_status": "pruned",
                "effective_status": "pruned",
                "binding_guidance": [],
                "matched_guidance": [{"id": "g-01", "effect": "caution"}],
            },
        )
        self.assertEqual(
            effective["hyp-data-raw"]["effective_status"],
            "active",
        )

    def test_excluded_guidance_beats_runtime_pruned(self) -> None:
        registry = fixture_registry()
        registry["guidance"].append(
            {"id": "g-90", "effect": "exclude", "scope": _scope("filtering")}
        )
        guidance = derive_hypothesis_selection(registry)
        self.assertEqual(
            guidance["hyp-data-filtered"]["selection_status"], "excluded"
        )
        runtime = {
            "dimensions": {},
            "hypotheses": {"hyp-data-filtered": "pruned"},
        }
        entry = compose_effective_selection(registry, guidance, runtime)[
            "hyp-data-filtered"
        ]
        self.assertEqual(entry["effective_status"], "excluded")
        self.assertEqual(entry["hypothesis_runtime_status"], "pruned")
        self.assertEqual(entry["guidance_status"], "excluded")

    def test_runtime_pruned_beats_any_deprioritized(self) -> None:
        registry = fixture_registry()
        registry["guidance"].append(
            {"id": "g-91", "effect": "deprioritize", "scope": _scope("filtering")}
        )
        guidance = derive_hypothesis_selection(registry)
        deprioritized_only = compose_effective_selection(
            registry, guidance, {"dimensions": {}, "hypotheses": {}}
        )
        self.assertEqual(
            deprioritized_only["hyp-data-filtered"]["effective_status"], "deprioritized"
        )
        runtime = {
            "dimensions": {"dim-data-curation": "deprioritized"},
            "hypotheses": {"hyp-data-filtered": "pruned"},
        }
        entry = compose_effective_selection(registry, guidance, runtime)[
            "hyp-data-filtered"
        ]
        self.assertEqual(entry["effective_status"], "pruned")
        self.assertEqual(entry["dimension_runtime_status"], "deprioritized")

    def test_dimension_status_propagates_to_non_baseline_only(self) -> None:
        registry = fixture_registry()
        guidance = derive_hypothesis_selection(registry)
        runtime = {
            "dimensions": {"dim-data-curation": "deprioritized"},
            "hypotheses": {},
        }
        effective = compose_effective_selection(registry, guidance, runtime)
        self.assertEqual(
            effective["hyp-data-filtered"]["effective_status"], "deprioritized"
        )
        self.assertEqual(effective["hyp-data-raw"]["effective_status"], "active")

    def test_pruned_dimension_keeps_baseline_eligible(self) -> None:
        registry = fixture_registry()
        state = state_with(
            dimension_decision(1, "dim-data-curation", "active", "deprioritized"),
            dimension_decision(2, "dim-data-curation", "deprioritized", "pruned"),
        )
        self.assertEqual(
            validate_search_space_state(registry, ledger_with_state(state)), []
        )
        runtime = replay_search_space_state(registry, state)
        self.assertEqual(runtime["dimensions"]["dim-data-curation"], "pruned")
        guidance = derive_hypothesis_selection(registry)
        effective = compose_effective_selection(registry, guidance, runtime)
        self.assertEqual(effective["hyp-data-filtered"]["effective_status"], "pruned")
        self.assertEqual(effective["hyp-data-raw"]["effective_status"], "active")

        baseline_point = complete_point(registry)
        self.assertEqual(
            validate_point_eligibility(baseline_point, registry, effective), []
        )
        filtered_point = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        errors = validate_point_eligibility(filtered_point, registry, effective)
        self.assertTrue(any("pruned" in error for error in errors), errors)


class PointEligibilityTests(unittest.TestCase):
    """Only exclusion and pruning bar selection; deprioritizing just rebudgets."""

    @staticmethod
    def _effective(registry: dict, runtime: dict | None = None) -> dict:
        return compose_effective_selection(
            registry,
            derive_hypothesis_selection(registry),
            runtime or {"dimensions": {}, "hypotheses": {}},
        )

    def test_eligible_points_pass(self) -> None:
        registry = fixture_registry()
        filtered = {"dim-data-curation": "hyp-data-filtered"}
        cases = [
            ("baseline", None, {}),
            # Deprioritizing changes the admission budget, not eligibility.
            ("deprioritized hypothesis", {"hypotheses": {"hyp-data-filtered": "deprioritized"}}, filtered),
            ("deprioritized dimension", {"dimensions": {"dim-data-curation": "deprioritized"}}, filtered),
            (
                "sibling of deprioritized dimension",
                {"dimensions": {"dim-data-curation": "deprioritized"}},
                {"dim-validation-selection": "hyp-valid-cv"},
            ),
        ]
        for name, runtime, selection in cases:
            with self.subTest(case=name):
                effective = self._effective(
                    registry, {"dimensions": {}, "hypotheses": {}, **(runtime or {})}
                )
                point = complete_point(registry, selection) if selection else complete_point(registry)
                self.assertEqual(
                    validate_point_eligibility(point, registry, effective), []
                )

    def test_excluded_and_pruned_points_are_ineligible(self) -> None:
        excluded_registry = fixture_registry()
        excluded_registry["guidance"].append(
            {"id": "g-90", "effect": "exclude", "scope": _scope("filtering")}
        )
        pruned_registry = fixture_registry()

        for needle, registry, runtime in (
            ("excluded", excluded_registry, None),
            (
                "pruned",
                pruned_registry,
                {"dimensions": {}, "hypotheses": {"hyp-data-filtered": "pruned"}},
            ),
        ):
            with self.subTest(needle=needle):
                effective = self._effective(registry, runtime)
                point = complete_point(
                    registry, {"dim-data-curation": "hyp-data-filtered"}
                )
                # Ineligible for new selection, yet still a structurally valid
                # point, so historical observations at it stay readable.
                self.assertEqual(validate_point(point, registry), [])
                errors = validate_point_eligibility(point, registry, effective)
                self.assertTrue(any(needle in error for error in errors), errors)


class ExperienceTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.ledger = belief_ledger(self.registry)
        self.ledger["search_space_state"] = empty_search_space_state()
        self.additional_evidence_run_ids: list[str] = []
        self.additional_evidence_edge_ids: list[str] = []

    def advance_dag_cursor(self) -> None:
        next_id = max(int(item["run_id"]) for item in self.ledger["records"]) + 1
        parent_id = f"{next_id:03d}"
        child_id = f"{next_id + 1:03d}"
        parent = {
            "run_id": parent_id,
            "source_run_ids": [],
            "semantic_point": complete_point(self.registry),
            "semantic_edges": [],
            "status": "keep",
            "final_best_score": 0.42,
            "dag_revision": self.ledger["dag_revision"] + 1,
        }
        child = {
            "run_id": child_id,
            "source_run_ids": [parent_id],
            "semantic_point": complete_point(
                self.registry, {"dim-data-curation": "hyp-data-filtered"}
            ),
            "status": "discard",
            "final_best_score": 0.54,
            "dag_revision": self.ledger["dag_revision"] + 2,
        }
        child["semantic_edges"] = build_semantic_edges(
            [*self.ledger["records"], parent], child
        )
        attach_matched_transfer(parent, child)
        self.ledger["records"].extend([parent, child])
        self.ledger["dag_revision"] += 2
        self.additional_evidence_run_ids.extend([parent_id, child_id])
        self.additional_evidence_edge_ids.append(
            f"sedge-{parent_id}-{child_id}"
        )

    def experience(self, generation: int) -> dict:
        return {
            "schema_version": 3,
            "updated_at_run": (
                self.additional_evidence_run_ids[-1]
                if self.additional_evidence_run_ids
                else "003"
            ),
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
                "evidence_run_ids": [
                    "000",
                    "001",
                    "002",
                    "003",
                    *self.additional_evidence_run_ids,
                ][-5:],
                "evidence_edge_ids": [
                    "sedge-000-001",
                    "sedge-002-003",
                    *self.additional_evidence_edge_ids,
                ][-5:],
                "comparator_coverage": {
                    "direct_tuned_edges": 2
                    + len(self.additional_evidence_edge_ids),
                    "direct_lightly_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
                "confidence": "high",
                "uncertainty": "Implementation changes remain confounded.",
                "reopen_when": "A later direct comparison improves.",
            }],
        }

    def test_pruning_requires_later_generation_with_new_evidence(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(first[0]["from_status"], "active")
        self.assertEqual(first[0]["to_status"], "deprioritized")

        self.ledger["experience"] = self.experience(generation=2)
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])

        # An unrelated global cursor advance is not target evidence.
        self.ledger["dag_revision"] += 1
        self.ledger["experience"] = self.experience(generation=3)
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])

        # A newly cited but still-pending target edge is not an observation.
        pending = {
            "run_id": "004",
            "source_run_ids": ["000"],
            "semantic_point": complete_point(
                self.registry, {"dim-data-curation": "hyp-data-filtered"}
            ),
            "status": "pending",
            "final_best_score": None,
            "dag_revision": self.ledger["dag_revision"],
        }
        pending["semantic_edges"] = build_semantic_edges(
            self.ledger["records"], pending
        )
        self.ledger["records"].append(pending)
        experience = self.experience(generation=4)
        entry = experience["hypothesis_evidence"][0]
        entry["evidence_edge_ids"].append("sedge-000-004")
        self.ledger["experience"] = experience
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])

        # Once that same edge becomes terminal, it is genuinely new target
        # evidence and can complete the second stage.
        pending["status"] = "discard"
        pending["final_best_score"] = 0.53
        attach_matched_transfer(self.ledger["records"][0], pending)
        self.ledger["dag_revision"] += 1
        pending["dag_revision"] = self.ledger["dag_revision"]
        experience = self.experience(generation=5)
        entry = experience["hypothesis_evidence"][0]
        entry["evidence_run_ids"].append("004")
        entry["evidence_edge_ids"].append("sedge-000-004")
        entry["comparator_coverage"]["direct_noncrash_edges"] = 1
        self.ledger["experience"] = experience
        second = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(second[0]["from_status"], "deprioritized")
        self.assertEqual(second[0]["to_status"], "pruned")

    def _regrade_lightly(self, *run_ids: str) -> None:
        """Re-grade fixture comparator children as lightly tuned in place."""
        for entry in self.ledger["records"]:
            if entry["run_id"] in run_ids:
                entry["evaluation_depth"] = "tuned_lightly"

    def _third_lightly_edge(self) -> None:
        """Add a third matched pair and grade its child tuned_lightly."""
        self.advance_dag_cursor()
        self._regrade_lightly(self.additional_evidence_run_ids[-1])

    def test_deprioritize_allows_three_lightly_tuned_edges(self) -> None:
        # Zero tuned + three lightly-tuned agreeing direct edges clear the
        # contradiction-grade depth bar: unpromising/med deprioritizes.
        self._regrade_lightly("001", "003")
        self._third_lightly_edge()
        experience = self.experience(generation=1)
        entry = experience["hypothesis_evidence"][0]
        entry["recommended_status"] = "deprioritized"
        entry["confidence"] = "med"
        entry["comparator_coverage"] = {
            "direct_tuned_edges": 0,
            "direct_lightly_tuned_edges": 3,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        }
        self.ledger["experience"] = experience
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["from_status"], "active")
        self.assertEqual(first[0]["to_status"], "deprioritized")
        self.assertEqual(first[0]["schema_version"], 4)
        self.assertEqual(
            validate_search_space_state(self.registry, self.ledger), []
        )

    def test_prune_allows_two_tuned_or_three_lightly(self) -> None:
        # Legacy: two tuned direct edges still carry the two-stage prune.
        self.ledger["experience"] = self.experience(generation=1)
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(first[0]["to_status"], "deprioritized")
        self.advance_dag_cursor()
        self.ledger["experience"] = self.experience(generation=2)
        second = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(second[0]["from_status"], "deprioritized")
        self.assertEqual(second[0]["to_status"], "pruned")

        # Depth-aware: three lightly-tuned direct edges (zero tuned) at high
        # confidence carry the same two-stage prune.
        self.setUp()
        self._regrade_lightly("001", "003")
        self._third_lightly_edge()
        self.ledger["experience"] = self.experience(generation=1)
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(first[0]["to_status"], "deprioritized")
        self._third_lightly_edge()
        self.ledger["experience"] = self.experience(generation=2)
        second = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(second[0]["from_status"], "deprioritized")
        self.assertEqual(second[0]["to_status"], "pruned")

    def test_two_lightly_tuned_edges_deprioritize_via_carrier_rule(self) -> None:
        # Exactly two lightly-tuned direct edges (zero tuned) stay below the
        # strict contradiction-grade depth bar, but they are two independent
        # negative carrier contexts — under the degraded demotion gates that
        # suffices to deprioritize.
        self._regrade_lightly("001", "003")
        self.assertEqual(
            target_evaluation_state(
                self.ledger,
                target_kind="hypothesis",
                target_id="hyp-data-filtered",
                evidence_run_ids=["000", "001", "002", "003"],
                evidence_edge_ids=["sedge-000-001", "sedge-002-003"],
            ),
            "comparator_covered",
        )
        experience = self.experience(generation=1)
        entry = experience["hypothesis_evidence"][0]
        entry["comparator_coverage"] = {
            "direct_tuned_edges": 0,
            "direct_lightly_tuned_edges": 2,
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        }
        self.ledger["experience"] = experience
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["to_status"], "deprioritized")
        self.assertEqual(self.ledger["search_space_state"]["revision"], 1)
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(
            replayed["hypotheses"]["hyp-data-filtered"], "deprioritized"
        )

    def test_mixed_matched_control_directions_cannot_contract_target(self) -> None:
        # Keep two direct controls but make the second one favor the selected
        # hypothesis.  Mere comparator count is not directional evidence.
        attach_matched_transfer(
            self.ledger["records"][2],
            self.ledger["records"][3],
            control_score=0.30,
        )
        experience = self.experience(generation=1)
        self.ledger["experience"] = experience
        errors = validate_experience(experience, self.registry, self.ledger)
        self.assertTrue(
            any("direction of its repeated matched" in error for error in errors),
            errors,
        )
        self.assertEqual(
            derive_experience_transitions(self.registry, self.ledger),
            [],
        )

    def test_crash_edge_cannot_advance_deprioritized_target_to_pruned(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(first[0]["to_status"], "deprioritized")

        crash = {
            "run_id": "004",
            "source_run_ids": ["000"],
            "semantic_point": complete_point(
                self.registry, {"dim-data-curation": "hyp-data-filtered"}
            ),
            "status": "crash",
            "final_best_score": float("inf"),
            "dag_revision": self.ledger["dag_revision"] + 1,
        }
        crash["semantic_edges"] = build_semantic_edges(
            self.ledger["records"], crash
        )
        self.ledger["records"].append(crash)
        self.ledger["dag_revision"] += 1
        experience = self.experience(generation=2)
        entry = experience["hypothesis_evidence"][0]
        entry["evidence_run_ids"] = ["000", "001", "002", "003", "004"]
        entry["evidence_edge_ids"].append("sedge-000-004")
        entry["comparator_coverage"]["crash_edges"] = 1
        self.ledger["experience"] = experience

        self.assertEqual(
            append_experience_transitions(self.registry, self.ledger),
            [],
        )
        self.assertEqual(
            replay_search_space_state(
                self.registry, self.ledger["search_space_state"]
            )[
                "hypotheses"
            ]["hyp-data-filtered"],
            "deprioritized",
        )

    def test_repeated_generation_is_no_op(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(len(first), 1)
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        self.assertEqual(self.ledger["search_space_state"]["revision"], 1)
        self.assertEqual(validate_search_space_state(self.registry, self.ledger), [])

    def test_derive_does_not_mutate_state(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        derived = derive_experience_transitions(self.registry, self.ledger)
        self.assertEqual(len(derived), 1)
        self.assertEqual(derived[0]["decision_id"], "sdec-000001")
        self.assertEqual(self.ledger["search_space_state"]["revision"], 0)
        self.assertEqual(self.ledger["search_space_state"]["decisions"], [])

    def test_weak_evidence_never_contracts_the_space(self) -> None:
        """Belief prose cannot move the overlay past its evidence gates."""
        no_beliefs = self.experience(generation=1)
        no_beliefs["hypothesis_evidence"] = []

        low_confidence = self.experience(generation=1)
        low_confidence["hypothesis_evidence"][0]["confidence"] = "low"

        single_edge = self.experience(generation=1)
        single_edge["hypothesis_evidence"][0].update(
            {
                "evaluation_state": "observed",
                "evidence_run_ids": ["000", "001"],
                "evidence_edge_ids": ["sedge-000-001"],
                "comparator_coverage": {
                    "direct_tuned_edges": 1,
                    "direct_lightly_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }
        )

        # A baseline is the comparison floor, so it is protected even though
        # the recomputed comparator coverage would otherwise qualify.
        baseline_target = self.experience(generation=1)
        baseline_target["hypothesis_evidence"][0].update(
            {
                "target_id": "hyp-data-raw",
                "claim": "Both direct comparisons moved away from the raw baseline.",
            }
        )

        for name, experience, target in (
            ("no beliefs", no_beliefs, "hyp-data-filtered"),
            ("low confidence", low_confidence, "hyp-data-filtered"),
            ("one direct edge", single_edge, "hyp-data-filtered"),
            ("baseline target", baseline_target, "hyp-data-raw"),
        ):
            with self.subTest(case=name):
                ledger = copy.deepcopy(self.ledger)
                ledger["experience"] = experience
                self.assertEqual(
                    append_experience_transitions(self.registry, ledger), []
                )
                self.assertEqual(ledger["search_space_state"]["revision"], 0)
                replayed = replay_search_space_state(
                    self.registry, ledger["search_space_state"]
                )
                self.assertEqual(replayed["hypotheses"][target], "active")

    def test_crash_only_evidence_makes_no_transition(self) -> None:
        crash_record = {
            "run_id": "004",
            "source_run_ids": ["002"],
            "semantic_point": complete_point(
                self.registry, {"dim-data-curation": "hyp-data-filtered"}
            ),
            "status": "crash",
            "final_best_score": None,
            "dag_revision": 5,
        }
        crash_record["semantic_edges"] = build_semantic_edges(
            self.ledger["records"], crash_record
        )
        self.ledger["records"].append(crash_record)
        self.ledger["dag_revision"] = 5
        experience = self.experience(generation=1)
        entry = experience["hypothesis_evidence"][0]
        entry.update(
            {
                "evaluation_state": "failed",
                "assessment": "unpromising",
                "recommended_status": "deprioritized",
                "claim": "The only cited comparison attempt crashed before a score.",
                "evidence_run_ids": ["004"],
                "evidence_edge_ids": ["sedge-002-004"],
                "comparator_coverage": {
                    "direct_tuned_edges": 0,
                    "direct_lightly_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 1,
                },
                "confidence": "med",
            }
        )
        self.ledger["experience"] = experience
        # A crash alone cannot contradict or prune a semantic element, even
        # when the belief prose asks for a transition.
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])

    def test_excluded_target_makes_no_transition(self) -> None:
        self.registry["guidance"].append(
            {"id": "g-90", "effect": "exclude", "scope": _scope("filtering")}
        )
        self.ledger["experience"] = self.experience(generation=1)
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        self.assertEqual(self.ledger["search_space_state"]["revision"], 0)

    def test_reopening_appends_pruned_to_active(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        append_experience_transitions(self.registry, self.ledger)
        self.advance_dag_cursor()
        self.ledger["experience"] = self.experience(generation=2)
        append_experience_transitions(self.registry, self.ledger)
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["hypotheses"]["hyp-data-filtered"], "pruned")

        # A repeated recommendation without newer evidence cannot reopen.
        experience = self.experience(generation=3)
        entry = experience["hypothesis_evidence"][0]
        entry.update(
            {
                "assessment": "promising",
                "recommended_status": "active",
                "claim": "The matched comparisons now look favorable.",
            }
        )
        self.ledger["experience"] = experience
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])

        # A later generation whose DAG cursor advanced and which cites an edge
        # absent from the prior decision appends a reopening decision.
        next_id = max(int(item["run_id"]) for item in self.ledger["records"]) + 1
        parent_id = f"{next_id:03d}"
        child_id = f"{next_id + 1:03d}"
        parent = {
            "run_id": parent_id,
            "source_run_ids": [],
            "semantic_point": complete_point(self.registry),
            "semantic_edges": [],
            "status": "keep",
            "final_best_score": 0.39,
            "dag_revision": self.ledger["dag_revision"] + 1,
        }
        child = {
            "run_id": child_id,
            "source_run_ids": [parent_id],
            "semantic_point": complete_point(
                self.registry, {"dim-data-curation": "hyp-data-filtered"}
            ),
            "status": "keep",
            "final_best_score": 0.38,
            "dag_revision": self.ledger["dag_revision"] + 2,
        }
        child["semantic_edges"] = build_semantic_edges(
            [*self.ledger["records"], parent], child
        )
        attach_matched_transfer(parent, child)
        self.ledger["records"].extend([parent, child])
        self.ledger["dag_revision"] += 2
        experience = self.experience(generation=4)
        experience["updated_at_run"] = child_id
        entry = experience["hypothesis_evidence"][0]
        existing_edges = list(entry["evidence_edge_ids"])
        existing_runs = list(entry["evidence_run_ids"])
        entry.update(
            {
                "assessment": "promising",
                "recommended_status": "active",
                "claim": "A later direct comparison improved over its matched parent.",
                "evidence_run_ids": [*existing_runs[-3:], parent_id, child_id],
                "evidence_edge_ids": [
                    *existing_edges,
                    f"sedge-{parent_id}-{child_id}",
                ][-5:],
                "comparator_coverage": {
                    "direct_tuned_edges": len(existing_edges) + 1,
                    "direct_lightly_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
            }
        )
        self.ledger["experience"] = experience
        (reopened,) = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(reopened["from_status"], "pruned")
        self.assertEqual(reopened["to_status"], "active")
        self.assertEqual(reopened["experience_generation"], 4)
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["hypotheses"]["hyp-data-filtered"], "active")
        self.assertEqual(validate_search_space_state(self.registry, self.ledger), [])

    def test_decision_observations_keep_scores_at_append_time(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        (decision,) = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(
            decision["evidence_observations"],
            [
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
                    "parent_score": 0.41,
                    "child_score": 0.52,
                    "delta": 0.11,
                },
            ],
        )
        # Later deep-tuned final-score changes never rewrite the inherited
        # control or masquerade as new semantic evidence.
        for item in self.ledger["records"]:
            if item["run_id"] == "001":
                item["final_best_score"] = 0.99
        stored = self.ledger["search_space_state"]["decisions"][0]
        self.assertEqual(stored["evidence_observations"][0]["child_score"], 0.5)
        self.assertEqual(stored["evidence_observations"][0]["delta"], 0.1)
        # A genuinely new matched control can advance the staged decision.
        self.advance_dag_cursor()
        self.ledger["experience"] = self.experience(generation=2)
        (second,) = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(second["evidence_observations"][0]["child_score"], 0.5)
        self.assertEqual(second["evidence_observations"][0]["delta"], 0.1)
        self.assertEqual(
            second["evidence_observations"][-1]["edge_id"],
            self.additional_evidence_edge_ids[-1],
        )
        self.assertEqual(second["evidence_observations"][-1]["delta"], 0.12)
        self.assertEqual(validate_search_space_state(self.registry, self.ledger), [])

    def dimension_belief(self, generation: int, *, with_hypothesis: bool) -> dict:
        experience = self.experience(generation)
        hypothesis_entry = experience["hypothesis_evidence"][0]
        experience["dimension_evidence"] = [{
            "target_id": "dim-data-curation",
            "evaluation_state": "comparator_covered",
            "assessment": "unpromising",
            "recommended_status": "pruned",
            "claim": "Every matched data-curation change worsened the score.",
            "evidence_run_ids": list(hypothesis_entry["evidence_run_ids"]),
            "evidence_edge_ids": list(hypothesis_entry["evidence_edge_ids"]),
            "comparator_coverage": dict(
                hypothesis_entry["comparator_coverage"]
            ),
            "confidence": "high",
            "uncertainty": "Implementation changes remain confounded.",
            "reopen_when": "A later direct comparison improves.",
        }]
        if not with_hypothesis:
            experience["hypothesis_evidence"] = []
        return experience

    def test_dimension_prune_requires_scoped_hypothesis_evidence(self) -> None:
        self.ledger["experience"] = self.dimension_belief(1, with_hypothesis=False)
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["dimensions"]["dim-data-curation"], "active")

    def test_dimension_prunes_after_covered_hypotheses_in_order(self) -> None:
        self.ledger["experience"] = self.dimension_belief(1, with_hypothesis=True)
        first = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(
            [(d["target"]["kind"], d["to_status"]) for d in first],
            [("dimension", "deprioritized"), ("hypothesis", "deprioritized")],
        )
        self.advance_dag_cursor()
        self.ledger["experience"] = self.dimension_belief(2, with_hypothesis=True)
        second = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(
            [(d["target"]["kind"], d["to_status"]) for d in second],
            [("dimension", "pruned"), ("hypothesis", "pruned")],
        )
        self.assertEqual(validate_search_space_state(self.registry, self.ledger), [])
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["dimensions"]["dim-data-curation"], "pruned")
        self.assertEqual(replayed["hypotheses"]["hyp-data-filtered"], "pruned")

    def test_dimension_prunes_once_hypothesis_already_pruned(self) -> None:
        self.ledger["experience"] = self.experience(generation=1)
        append_experience_transitions(self.registry, self.ledger)
        self.advance_dag_cursor()
        self.ledger["experience"] = self.experience(generation=2)
        append_experience_transitions(self.registry, self.ledger)
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["hypotheses"]["hyp-data-filtered"], "pruned")

        self.ledger["experience"] = self.dimension_belief(3, with_hypothesis=False)
        (third,) = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(third["to_status"], "deprioritized")
        self.advance_dag_cursor()
        self.ledger["experience"] = self.dimension_belief(4, with_hypothesis=False)
        (fourth,) = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(fourth["from_status"], "deprioritized")
        self.assertEqual(fourth["to_status"], "pruned")
        self.assertEqual(validate_search_space_state(self.registry, self.ledger), [])


class LedgerIntegrationTests(unittest.TestCase):
    def test_policy_receipt_budget_matches_historical_lane_and_index(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        entry = record("000", "fresh", [], baseline, score=0.5, status="keep")
        ledger = {
            "search_space": space_receipt(registry),
            "records": [entry],
            "search_space_state": empty_search_space_state(),
        }

        wrong_lane = copy.deepcopy(ledger)
        budget = wrong_lane["records"][0]["policy_receipt"]["budget"]
        budget["selected_lane"] = "deprioritized"
        budget["fallback"] = "no_active_proposals"
        errors = validate_ledger(registry, wrong_lane)
        self.assertTrue(
            any("selected_lane must match" in error for error in errors), errors
        )

        wrong_index = copy.deepcopy(ledger)
        receipt = wrong_index["records"][0]["policy_receipt"]
        receipt["policy"]["config"]["deprioritized_budget_interval"] = 2
        budget = receipt["budget"]
        budget.update(
            {
                "selection_index": 2,
                "deprioritized_interval": 2,
                "scheduled_lane": "deprioritized",
                "selected_lane": "active",
                "fallback": "no_deprioritized_proposals",
            }
        )
        errors = validate_ledger(registry, wrong_index)
        self.assertTrue(
            any("one-based admission index 1" in error for error in errors), errors
        )

    def test_validate_ledger_requires_state_once_records_exist(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        entry = record("000", "fresh", [], baseline, score=0.5, status="keep")
        ledger = {"search_space": space_receipt(registry), "records": [entry]}
        errors = validate_ledger(registry, ledger)
        self.assertTrue(any("search_space_state" in error for error in errors), errors)

        ledger["search_space_state"] = empty_search_space_state()
        self.assertEqual(validate_ledger(registry, ledger), [])

        ledger["search_space_state"] = state_with(
            decision(1, "hyp-data-filtered", "active", "pruned")
        )
        errors = validate_ledger(registry, ledger)
        self.assertTrue(any("search_space_state" in error for error in errors), errors)

    def test_loader_does_not_rebuild_missing_state_for_record_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps({"records": [{"run_id": "000"}]}))

            with self.assertRaisesRegex(ValueError, "requires search_space_state"):
                _load_ledger(ledger_path)

    def test_runtime_status_counts_and_brief_summary(self) -> None:
        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "deprioritized", "pruned"),
            hypothesis_decision(
                3, "dim-validation-selection", "hyp-valid-cv", "active", "deprioritized"
            ),
            dimension_decision(4, "dim-ensemble", "active", "deprioritized"),
        )
        counts = runtime_status_counts(state)
        self.assertEqual(
            counts,
            {
                "dimensions": {"deprioritized": 1, "pruned": 0},
                "hypotheses": {"deprioritized": 1, "pruned": 1},
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(
                json.dumps({"records": [], "search_space_state": state})
            )
            args = types.SimpleNamespace(ledger=str(ledger_path), budget=None)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                cmd_brief(args)
            brief = json.loads(output.getvalue())
        self.assertEqual(brief["search_space_state_revision"], 4)
        self.assertEqual(brief["runtime_deprioritized_dimensions"], 1)
        self.assertEqual(brief["runtime_pruned_dimensions"], 0)
        self.assertEqual(brief["runtime_deprioritized_hypotheses"], 1)
        self.assertEqual(brief["runtime_pruned_hypotheses"], 1)
        self.assertNotIn("decisions", brief)

    def test_brief_requires_experience_refresh_for_terminal_dag_delta(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        entry = record("000", "fresh", [], baseline, score=0.5, status="keep")
        ledger = {
            "records": [entry],
            "dag_revision": 1,
            "experience": {
                "generation": 0,
                "updated_at_run": None,
                "dag_revision": 0,
            },
            "search_space_state": empty_search_space_state(),
        }

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            args = types.SimpleNamespace(ledger=str(ledger_path), budget=None)

            def brief() -> dict:
                ledger_path.write_text(json.dumps(ledger))
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    cmd_brief(args)
                return json.loads(output.getvalue())

            stale = brief()
            self.assertEqual(stale["experience_dag_delta"], 1)
            self.assertTrue(stale["semantic_admission_blocked"])
            self.assertTrue(stale["experience_refresh_required"])

            ledger["records"][0]["status"] = "pending"
            self.assertTrue(brief()["semantic_admission_blocked"])
            self.assertFalse(brief()["experience_refresh_required"])

            ledger["records"][0]["status"] = "keep"
            ledger["experience"]["dag_revision"] = 1
            current = brief()
            self.assertEqual(current["experience_dag_delta"], 0)
            self.assertFalse(current["semantic_admission_blocked"])
            self.assertFalse(current["experience_refresh_required"])

            ledger["experience"]["dag_revision"] = 2
            with self.assertRaisesRegex(
                SystemExit, "cannot exceed ledger.dag_revision"
            ):
                brief()

    def test_noop_belief_refresh_advances_cursor_without_generation(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        entry = record("000", "fresh", [], baseline, score=0.5, status="keep")
        entry["dag_revision"] = 1
        prior = empty_experience("000", generation=0)
        prior["dag_revision"] = 0
        ledger = {
            "task": "hard-interactions",
            "tag": "noop-refresh",
            "metric": "validation_loss",
            "search_space": space_receipt(registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 1,
            "records": [entry],
            "experience": prior,
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            background_path = tmp_path / "background.md"
            ledger_path = tmp_path / "ledger.json"
            replacement_path = tmp_path / "experience.json"
            background_path.write_text(background_text(registry))
            ledger_path.write_text(json.dumps(ledger))
            replacement = empty_experience("000", generation=0)
            replacement_path.write_text(json.dumps(replacement))
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                background=str(background_path),
                catalog=None,
                from_json=replacement_path,
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cmd_set_experience(args), 0)
            result = json.loads(output.getvalue())
            stored = json.loads(ledger_path.read_text())

            self.assertFalse(result["belief_changed"])
            self.assertEqual(stored["experience"]["generation"], 0)
            self.assertEqual(stored["experience"]["dag_revision"], 1)

            before = ledger_path.read_bytes()
            with self.assertRaisesRegex(
                SystemExit, "no unprocessed terminal DAG delta"
            ):
                cmd_set_experience(args)
            self.assertEqual(ledger_path.read_bytes(), before)

    def test_pending_record_rejects_experience_refresh_without_mutation(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        entry = record("000", "fresh", [], baseline, score=0.5, status="pending")
        ledger = {
            "task": "hard-interactions",
            "tag": "pending-refresh",
            "metric": "validation_loss",
            "search_space": space_receipt(registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 1,
            "records": [entry],
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            background_path = tmp_path / "background.md"
            ledger_path = tmp_path / "ledger.json"
            replacement_path = tmp_path / "experience.json"
            background_path.write_text(background_text(registry))
            ledger_path.write_text(json.dumps(ledger))
            replacement_path.write_text(json.dumps(empty_experience("000")))
            before = ledger_path.read_bytes()
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                background=str(background_path),
                catalog=None,
                from_json=replacement_path,
            )

            with self.assertRaisesRegex(SystemExit, "record is still pending"):
                cmd_set_experience(args)
            self.assertEqual(ledger_path.read_bytes(), before)

    def test_final_completion_requires_terminal_delta_refresh(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        entry = record("000", "fresh", [], baseline, score=0.5, status="keep")
        entry["dag_revision"] = 1
        ledger = {
            "task": "hard-interactions",
            "tag": "final-refresh",
            "metric": "validation_loss",
            "search_space": space_receipt(registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 1,
            "records": [entry],
            "experience": {
                **empty_experience("000"),
                "dag_revision": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps(ledger))
            before = ledger_path.read_bytes()
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                phase="completed",
                stop_condition=None,
                budget=0,
            )

            with self.assertRaisesRegex(
                SystemExit, "terminal DAG delta remains unprocessed"
            ):
                cmd_set_phase(args)
            self.assertEqual(ledger_path.read_bytes(), before)

            ledger["experience"]["dag_revision"] = 1
            ledger_path.write_text(json.dumps(ledger))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_set_phase(args), 0)
            stored = json.loads(ledger_path.read_text())
            self.assertEqual(stored["run_state"]["phase"], "completed")

    def test_final_completion_rejects_terminal_delta_with_pending_sibling(
        self,
    ) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        terminal = record(
            "000", "fresh", [], baseline, score=0.5, status="keep"
        )
        terminal["dag_revision"] = 1
        pending = record(
            "001",
            "improve",
            ["000"],
            baseline,
            score=0.5,
            status="pending",
            prior_records=[terminal],
        )
        ledger = {
            "task": "hard-interactions",
            "tag": "pending-final-refresh",
            "metric": "validation_loss",
            "search_space": space_receipt(registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 1,
            "records": [terminal, pending],
            "experience": {
                **empty_experience("000"),
                "dag_revision": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.json"
            ledger_path.write_text(json.dumps(ledger))
            before = ledger_path.read_bytes()
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                phase="completed",
                stop_condition=None,
                budget=0,
            )
            with self.assertRaisesRegex(
                SystemExit, "pending records must be resolved"
            ):
                cmd_set_phase(args)
            self.assertEqual(ledger_path.read_bytes(), before)

    def test_budget_exhaustion_resolves_zero_attempt_candidate_without_evidence(
        self,
    ) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        terminal = record(
            "000", "fresh", [], baseline, score=0.5, status="keep"
        )
        terminal["dag_revision"] = 1
        pending = record(
            "001",
            "improve",
            ["000"],
            baseline,
            score=float("inf"),
            status="pending",
            prior_records=[terminal],
        )
        pending["final_best_score"] = None
        ledger = {
            "task": "hard-interactions",
            "tag": "unevaluated-resolution",
            "metric": "validation_loss",
            "search_space": space_receipt(registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 1,
            "records": [terminal, pending],
            "experience": {
                **empty_experience("000"),
                "dag_revision": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            ledger_path = run_dir / "ledger.json"
            background_path = run_dir / "background.md"
            replacement_path = run_dir / "experience.json"
            (run_dir / "framework_cfg.json").write_text(
                json.dumps({"max_evaluations": 1})
            )
            (run_dir / "evaluation_attempts.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "score_attempt",
                        "attempt_id": "eval-000001",
                        "run_id": "000",
                        "phase": "phase_a",
                        "method": "warmstart",
                        "params": {"x": 0},
                    }
                )
                + "\n"
            )
            ledger_path.write_text(json.dumps(ledger))
            background_path.write_text(background_text(registry))

            resolved = resolve_unevaluated(
                ledger_path,
                "hard-interactions",
                "001",
            )
            self.assertEqual(resolved["status"], "unevaluated")
            self.assertIsNone(resolved["final_best_score"])
            self.assertNotIn("attempt_log_sha256", resolved["unevaluated_receipt"])
            self.assertEqual(
                resolved["unevaluated_receipt"][
                    "candidate_objective_attempts"
                ],
                0,
            )
            with self.assertRaisesRegex(ValueError, "terminal unevaluated"):
                record_run(
                    ledger_path,
                    "hard-interactions",
                    "001",
                    final_best_score=0.1,
                )
            stored = json.loads(ledger_path.read_text())
            self.assertEqual(stored["dag_revision"], 2)

            replacement_path.write_text(
                json.dumps(empty_experience("000", generation=0))
            )
            refresh_args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                background=str(background_path),
                catalog=None,
                from_json=replacement_path,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_set_experience(refresh_args), 0)
            refreshed = json.loads(ledger_path.read_text())
            self.assertEqual(refreshed["experience"]["dag_revision"], 2)
            self.assertEqual(refreshed["experience"]["updated_at_run"], "000")

            phase_args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                phase="completed",
                stop_condition=None,
                budget=None,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_set_phase(phase_args), 0)
            completed = json.loads(ledger_path.read_text())
            self.assertEqual(completed["run_state"]["phase"], "completed")


class StateAwareSelectionLifecycleTests(unittest.TestCase):
    """Proposals and selection receipts track the revisioned pruning overlay.

    Automated pruning is two-stage (``active -> deprioritized -> pruned``), so
    the pruned overlay is revision 2 and reopening lands at revision 3; the
    deprioritized stage at revision 1 exercises the carrier-prior penalty.
    """

    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.filtered = complete_point(
            self.registry, {"dim-data-curation": "hyp-data-filtered"}
        )

    def _proposals(self, ledger: dict) -> dict:
        return build_proposal_set(
            self.registry, ledger, op="fresh", parents=[], max_points=64
        )

    def test_deprioritized_without_carriers_is_penalized(self) -> None:
        # A deprioritized hypothesis with no carrier history (e.g. external
        # guidance) still earns one negative-context weight: demotion must
        # not be selection-inert now that lanes are gone.
        from semantic_search import DEFAULT_POLICY_CONFIG, _carrier_priors

        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized")
        )
        ledger = {"records": [], "search_space_state": state}
        proposals = self._proposals(ledger)
        priors = _carrier_priors(proposals, ledger, dict(DEFAULT_POLICY_CONFIG))
        deprioritized = [
            proposal
            for proposal in proposals["proposals"]
            if "hyp-data-filtered" in proposal["deprioritized_hypotheses"]
        ]
        self.assertTrue(deprioritized)
        for proposal in deprioritized:
            prior, detail = priors[proposal["point_id"]]
            self.assertEqual(prior, -0.2)
            self.assertEqual(
                detail["hyp-data-filtered"], {"negative": 1, "positive": 0}
            )
        clean = [
            proposal
            for proposal in proposals["proposals"]
            if not proposal["deprioritized_hypotheses"]
        ]
        self.assertTrue(clean)
        for proposal in clean:
            prior, _ = priors[proposal["point_id"]]
            self.assertEqual(prior, 0.0)

    @staticmethod
    def _selects_filtered(proposal: dict) -> bool:
        return (
            selected_assignments(proposal["point"]).get("dim-data-curation")
            == "hyp-data-filtered"
        )

    def test_add_record_cleanly_rejects_missing_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            background_path = tmp_path / "background.md"
            ledger_path = tmp_path / "ledger.json"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "policy.json"
            background_path.write_text(background_text(self.registry))
            point_path.write_text(json.dumps(self.filtered))
            receipt_path.write_text(
                json.dumps(
                    current_policy_receipt(
                        "improve",
                        ["999"],
                        self.filtered,
                    )
                )
            )
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                run_id="001",
                kind="optimization",
                op="improve",
                source_run_ids="999",
                background=str(background_path),
                catalog=None,
                semantic_point=str(point_path),
                policy_receipt=str(receipt_path),
                idea="A candidate whose claimed parent does not exist.",
                change="attempt to improve a missing parent",
                candidate_name_hint="fixture_missing_parent",
                description=None,
                route_provenance=None,
            )
            with self.assertRaises(SystemExit) as raised:
                cmd_add_record(args)
        self.assertIn(
            "record 001 parent 999 is not an earlier record",
            str(raised.exception),
        )

    def test_add_record_rejects_unprocessed_terminal_delta_before_admission(
        self,
    ) -> None:
        baseline = complete_point(self.registry)
        existing = record(
            "000", "fresh", [], baseline, score=0.5, status="keep"
        )
        existing["dag_revision"] = 1
        pending = record(
            "001",
            "improve",
            ["000"],
            baseline,
            score=0.5,
            status="pending",
            prior_records=[existing],
        )
        ledger = {
            "task": "hard-interactions",
            "tag": "stale-dag",
            "metric": "validation_loss",
            "search_space": space_receipt(self.registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 1,
            # A sibling already admitted in the same batch may still be
            # pending, but the terminal delta closes further admission.
            "records": [existing, pending],
            "experience": {
                **empty_experience("000"),
                "dag_revision": 0,
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ledger_path = tmp_path / "ledger.json"
            background_path = tmp_path / "background.md"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "policy.json"
            ledger_path.write_text(json.dumps(ledger))
            background_path.write_text(background_text(self.registry))
            point_path.write_text(json.dumps(self.filtered))
            # The cadence gate must fire before the deliberately invalid
            # selection receipt can be considered.
            receipt_path.write_text("{}")
            before = ledger_path.read_bytes()
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                run_id="002",
                kind="optimization",
                op="fresh",
                source_run_ids="",
                background=str(background_path),
                catalog=None,
                semantic_point=str(point_path),
                policy_receipt=str(receipt_path),
                idea="A candidate that must wait for the belief cursor.",
                change="from scratch after an unprocessed terminal delta",
                candidate_name_hint="fixture_stale_dag",
                description=None,
                route_provenance=None,
            )

            with self.assertRaisesRegex(
                SystemExit, "terminal DAG evidence must be refreshed"
            ):
                cmd_add_record(args)
            self.assertEqual(ledger_path.read_bytes(), before)

    def test_persisted_schema6_conditioning_tamper_is_rejected(self) -> None:
        baseline = complete_point(self.registry)
        first = record(
            "000", "fresh", [], baseline, score=0.5, status="keep"
        )
        first["dag_revision"] = 1
        second = record(
            "001",
            "improve",
            ["000"],
            self.filtered,
            score=0.6,
            status="discard",
            prior_records=[first],
        )
        second["dag_revision"] = 2
        second["policy_receipt"] = current_policy_receipt(
            "improve",
            ["000"],
            self.filtered,
            selection_index=2,
        )
        attach_matched_transfer(first, second)
        experience = {
            "schema_version": 3,
            "updated_at_run": "001",
            "generation": 0,
            "summary": "One direct comparison remains uncertainty-only.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [{
                "target_id": "hyp-data-filtered",
                "evaluation_state": "observed",
                "assessment": "mixed",
                "recommended_status": "active",
                "claim": "One matched edge cannot establish direction.",
                "evidence_run_ids": ["000", "001"],
                "evidence_edge_ids": ["sedge-000-001"],
                "comparator_coverage": {
                    "direct_tuned_edges": 1,
                    "direct_lightly_tuned_edges": 0,
                    "direct_noncrash_edges": 0,
                    "confounded_noncrash_edges": 0,
                    "crash_edges": 0,
                },
                "confidence": "low",
                "uncertainty": "A second direct comparison is absent.",
            }],
            "dag_revision": 2,
        }
        ledger = {
            "task": "hard-interactions",
            "tag": "conditioning-receipt",
            "metric": "validation_loss",
            "search_space": space_receipt(self.registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 2,
            "records": [first, second],
            "experience": experience,
        }
        proposals = self._proposals(ledger)
        filtered_proposal = next(
            proposal
            for proposal in proposals["proposals"]
            if self._selects_filtered(proposal)
        )
        predictions = {
            "schema_version": 3,
            "proposal_set_revision": proposals["proposal_set_revision"],
            "experience": {
                "generation": 0,
                "updated_at_run": "001",
                "revision": digest(experience),
            },
            "predictions": [],
        }
        for proposal in proposals["proposals"]:
            selected = proposal["point_id"] == filtered_proposal["point_id"]
            predictions["predictions"].append(
                {
                    "point_id": proposal["point_id"],
                    "prior_gain": 0.9 if selected else 0.1,
                    "experience_gain_adjustment": 0.0,
                    "predicted_gain": 0.9 if selected else 0.1,
                    "prior_uncertainty": 0.3,
                    "experience_uncertainty_adjustment": 0.05 if selected else 0.0,
                    "uncertainty": 0.35 if selected else 0.3,
                    "experience_run_ids": ["000", "001"] if selected else [],
                    "experience_edge_ids": (
                        ["sedge-000-001"] if selected else []
                    ),
                    "experience_target_ids": (
                        ["hyp-data-filtered"] if selected else []
                    ),
                    "experience_rationale": (
                        "The single comparator increases uncertainty only."
                    ),
                    "evidence": ["schema-6 conditioning receipt fixture"],
                }
            )
        point, receipt = select_proposal(
            proposals,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            selection_index=3,
            experience=experience,
            ledger=ledger,
        )
        self.assertEqual(point["point_id"], filtered_proposal["point_id"])
        third = record(
            "002", "fresh", [], point, score=0.55, status="discard"
        )
        third["policy_receipt"] = receipt
        third["dag_revision"] = 3
        ledger["records"].append(third)
        ledger["dag_revision"] = 3
        self.assertEqual(validate_ledger(self.registry, ledger), [])

        tampered = copy.deepcopy(ledger)
        conditioning = tampered["records"][-1]["policy_receipt"]["experience"][
            "conditioning"
        ][0]
        conditioning["acquisition_role"] = "comparator_gain"
        conditioning["gain_direction"] = "negative"
        errors = validate_ledger(self.registry, tampered)
        self.assertTrue(
            any("acquisition_role is not mechanically derived" in error for error in errors),
            errors,
        )

    def test_add_record_rejects_stale_experience_receipt(self) -> None:
        baseline = complete_point(self.registry)
        existing = record(
            "000", "fresh", [], baseline, score=0.5, status="keep"
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "000",
            "generation": 0,
            "summary": "The first terminal run supplies the current bounded belief.",
            "promising_regions": [
                {
                    "claim": "The first point is provisionally promising.",
                    "evidence": ["000"],
                    "confidence": "low",
                    "uncertainty": "Only one implementation has been observed.",
                }
            ],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            background_path = tmp_path / "background.md"
            ledger_path = tmp_path / "ledger.json"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "policy.json"
            background_path.write_text(background_text(self.registry))
            ledger_path.write_text(
                json.dumps(
                    {
                        "task": "hard-interactions",
                        "tag": "stale-experience",
                        "metric": "validation_loss",
                        "search_space": space_receipt(self.registry),
                        "search_space_state": empty_search_space_state(),
                        "dag_revision": 0,
                        "records": [existing],
                        "experience": experience,
                    }
                )
            )
            point_path.write_text(json.dumps(self.filtered))
            # This otherwise-current schema-6 receipt falsely claims that no
            # experience snapshot existed at admission.
            receipt_path.write_text(
                json.dumps(
                    current_policy_receipt(
                        "fresh",
                        [],
                        self.filtered,
                        selection_index=2,
                    )
                )
            )
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                run_id="001",
                kind="optimization",
                op="fresh",
                source_run_ids="",
                background=str(background_path),
                catalog=None,
                semantic_point=str(point_path),
                policy_receipt=str(receipt_path),
                idea="A candidate selected with a stale belief receipt.",
                change="from scratch at a stale belief revision",
                candidate_name_hint="fixture_stale_experience",
                description=None,
                route_provenance=None,
            )
            with self.assertRaises(SystemExit) as raised:
                cmd_add_record(args)
            self.assertIn(
                "stale policy receipt: experience", str(raised.exception)
            )

            forged_receipt = current_policy_receipt(
                "fresh",
                [],
                self.filtered,
                selection_index=2,
            )
            forged_receipt["experience"].update(
                {
                    "generation": 0,
                    "updated_at_run": "000",
                    "revision": digest(experience),
                    "evidence_run_ids": ["999"],
                }
            )
            receipt_path.write_text(json.dumps(forged_receipt))
            with self.assertRaises(SystemExit) as forged:
                cmd_add_record(args)
        self.assertIn(
            "coverage policy must not cite experience", str(forged.exception)
        )

    def test_add_record_accepts_revision_pinned_empty_evidence_snapshot(self) -> None:
        baseline = complete_point(self.registry)
        existing = record(
            "000", "fresh", [], baseline, score=0.5, status="keep"
        )
        experience = {
            "schema_version": 3,
            "updated_at_run": "000",
            "generation": 0,
            "summary": "The current bounded extraction found no evidence-bearing belief.",
            "promising_regions": [],
            "lessons": [],
            "bottlenecks": [],
            "dimension_evidence": [],
            "hypothesis_evidence": [],
        }
        ledger = {
            "task": "hard-interactions",
            "tag": "empty-experience",
            "metric": "validation_loss",
            "search_space": space_receipt(self.registry),
            "search_space_state": empty_search_space_state(),
            "dag_revision": 0,
            "records": [existing],
            "experience": experience,
        }
        proposals = self._proposals(ledger)
        predictions = {
            "schema_version": 3,
            "proposal_set_revision": proposals["proposal_set_revision"],
            "experience": {
                "generation": 0,
                "updated_at_run": "000",
                "revision": digest(experience),
            },
            "predictions": [
                {
                    "point_id": proposal["point_id"],
                    "prior_gain": 0.5,
                    "experience_gain_adjustment": 0.0,
                    "predicted_gain": 0.5,
                    "prior_uncertainty": 0.3,
                    "experience_uncertainty_adjustment": 0.0,
                    "uncertainty": 0.3,
                    "experience_run_ids": [],
                    "experience_edge_ids": [],
                    "experience_target_ids": [],
                    "experience_rationale": (
                        "The pinned snapshot carries no conditioning evidence."
                    ),
                    "evidence": ["background prior only"],
                }
                for proposal in proposals["proposals"]
            ],
        }
        point, receipt = select_proposal(
            proposals,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            selection_index=2,
            experience=experience,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            background_path = tmp_path / "background.md"
            ledger_path = tmp_path / "ledger.json"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "policy.json"
            background_path.write_text(background_text(self.registry))
            ledger_path.write_text(json.dumps(ledger))
            point_path.write_text(json.dumps(point))
            receipt_path.write_text(json.dumps(receipt))
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                run_id="001",
                kind="optimization",
                op="fresh",
                source_run_ids="",
                background=str(background_path),
                catalog=None,
                semantic_point=str(point_path),
                policy_receipt=str(receipt_path),
                idea="A candidate with a revision-pinned empty belief snapshot.",
                change="from scratch without usable experience evidence",
                candidate_name_hint="fixture_empty_experience",
                description=None,
                route_provenance=None,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cmd_add_record(args), 0)
            stored = json.loads(ledger_path.read_text())
        self.assertEqual(stored["records"][-1]["run_id"], "001")
        self.assertEqual(
            stored["records"][-1]["policy_receipt"]["experience"]["evidence_run_ids"],
            [],
        )

    def test_prune_select_stale_reject_and_reopen_lifecycle(self) -> None:
        registry = self.registry

        # 1. Revision 0 proposes a point containing hyp-data-filtered.
        ledger: dict = {"records": [], "search_space_state": empty_search_space_state()}
        proposals = self._proposals(ledger)
        self.assertEqual(proposals["schema_version"], 4)
        self.assertEqual(proposals["search_space_state_revision"], 0)
        self.assertTrue(any(self._selects_filtered(p) for p in proposals["proposals"]))
        _, receipt = select_proposal(proposals, policy="coverage")
        self.assertEqual(receipt["schema_version"], 7)
        self.assertEqual(receipt["search_space_state_revision"], 0)

        # A revision-0 historical record selects the hypothesis to be pruned.
        ledger["records"].append(
            record("000", "fresh", [], self.filtered, score=0.5, status="discard")
        )
        ledger["search_space"] = space_receipt(registry)

        # 2. Later state decisions prune that hypothesis (two-stage).  The
        #    revision-1 deprioritized stage keeps the content eligible but
        #    assigns it to the limited admission-budget lane.
        state = state_with(decision(1, "hyp-data-filtered", "active", "deprioritized"))
        ledger["search_space_state"] = state
        proposals = self._proposals(ledger)
        self.assertEqual(proposals["schema_version"], 4)
        self.assertEqual(proposals["search_space_state_revision"], 1)
        filtered_proposals = [
            item for item in proposals["proposals"] if self._selects_filtered(item)
        ]
        self.assertTrue(filtered_proposals)
        for item in filtered_proposals:
            self.assertEqual(item["deprioritized_hypotheses"], ["hyp-data-filtered"])
            self.assertEqual(item["budget_lane"], "deprioritized")
        positions = {
            item["point_id"]: index
            for index, item in enumerate(proposals["proposals"])
        }
        mixed_pairs = [
            (active, dep)
            for active in proposals["proposals"]
            if not active["deprioritized_hypotheses"]
            for dep in proposals["proposals"]
            if dep["deprioritized_hypotheses"]
            and active["coverage"] == dep["coverage"]
        ]
        self.assertTrue(mixed_pairs)  # the ordering assertion is non-vacuous
        for active, dep in mixed_pairs:
            self.assertLess(positions[active["point_id"]], positions[dep["point_id"]])
        _, receipt = select_proposal(proposals, policy="coverage")
        self.assertEqual(receipt["schema_version"], 7)
        self.assertEqual(receipt["search_space_state_revision"], 1)
        self.assertIsNone(receipt["budget"]["selected_lane"])
        self.assertEqual(receipt["budget"]["fallback"], "lanes_removed")

        # 3. Revision 2 proposals omit the pruned hypothesis.
        state["decisions"].append(
            decision(2, "hyp-data-filtered", "deprioritized", "pruned")
        )
        state["revision"] = 2
        proposals = self._proposals(ledger)
        self.assertEqual(proposals["schema_version"], 4)
        self.assertEqual(proposals["search_space_state_revision"], 2)
        self.assertFalse(any(self._selects_filtered(p) for p in proposals["proposals"]))
        _, receipt = select_proposal(proposals, policy="coverage")
        self.assertEqual(receipt["schema_version"], 7)
        self.assertEqual(receipt["search_space_state_revision"], 2)

        # 4. The revision-0 historical record remains ledger-valid.
        self.assertEqual(validate_ledger(registry, ledger), [])

        # 5. add-record rejects the stale revision-0 receipt at revision 2.
        #    Admission precedes all candidate work, so no candidate directory
        #    or implementation is created and the ledger stays untouched.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            background_path = tmp_path / "background.md"
            ledger_path = tmp_path / "ledger.json"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "policy.json"
            background_path.write_text(background_text(registry))
            ledger_path.write_text(json.dumps(ledger))
            point_path.write_text(json.dumps(self.filtered))
            receipt_path.write_text(
                json.dumps(
                    current_policy_receipt(
                        "fresh",
                        [],
                        self.filtered,
                        state_revision=0,
                        selection_index=2,
                    )
                )
            )
            args = types.SimpleNamespace(
                ledger=str(ledger_path),
                task="hard-interactions",
                run_id="001",
                kind="optimization",
                op="fresh",
                source_run_ids="",
                background=str(background_path),
                catalog=None,
                semantic_point=str(point_path),
                policy_receipt=str(receipt_path),
                idea="A forged late arrival at the pruned point.",
                change="from scratch at the pruned point",
                candidate_name_hint="fixture_stale",
                description=None,
                route_provenance=None,
            )
            with self.assertRaises(SystemExit) as raised:
                cmd_add_record(args)
            self.assertIn("stale", str(raised.exception))
            # A revision-current receipt still cannot admit a pruned point:
            # the selected point must be eligible at its receipt revision.
            receipt_path.write_text(
                json.dumps(
                    current_policy_receipt(
                        "fresh",
                        [],
                        self.filtered,
                        state_revision=2,
                        selection_index=2,
                    )
                )
            )
            with self.assertRaises(SystemExit) as raised:
                cmd_add_record(args)
            self.assertIn("pruned", str(raised.exception))
            stored = json.loads(ledger_path.read_text())
        self.assertEqual([item["run_id"] for item in stored["records"]], ["000"])

        # 6. Reopening appends a new decision; revision 3 is eligible again.
        state["decisions"].append(decision(3, "hyp-data-filtered", "pruned", "active"))
        state["revision"] = 3
        proposals = self._proposals(ledger)
        self.assertEqual(proposals["search_space_state_revision"], 3)
        self.assertTrue(any(self._selects_filtered(p) for p in proposals["proposals"]))
        self.assertEqual(validate_ledger(registry, ledger), [])

        # 7. A structural `requires` completion that would force the pruned
        #    hyp-valid-cv is discarded rather than leaking an ineligible
        #    point (stacking always completes to cross-validation).
        cv_state = state_with(
            hypothesis_decision(
                1, "dim-validation-selection", "hyp-valid-cv", "active", "deprioritized"
            ),
            hypothesis_decision(
                2, "dim-validation-selection", "hyp-valid-cv", "deprioritized", "pruned"
            ),
        )
        cv_proposals = self._proposals(
            {"records": [], "search_space_state": cv_state}
        )
        self.assertEqual(cv_proposals["search_space_state_revision"], 2)
        self.assertTrue(cv_proposals["proposals"])
        for item in cv_proposals["proposals"]:
            selected = selected_assignments(item["point"])
            self.assertNotEqual(selected.get("dim-validation-selection"), "hyp-valid-cv")
            self.assertNotEqual(selected.get("dim-ensemble"), "hyp-ensemble-stacking")

    def test_budget_lane_field_no_longer_schedules(self) -> None:
        # Lane scheduling was removed: budget_lane stays on proposals for
        # backward readability but selection ranks by score alone.
        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized")
        )
        proposals = self._proposals(
            {"records": [], "search_space_state": state}
        )
        predictions = {
            "schema_version": 1,
            "proposal_set_revision": proposals["proposal_set_revision"],
            "predictions": [],
        }
        for proposal in proposals["proposals"]:
            predictions["predictions"].append(
                {
                    "point_id": proposal["point_id"],
                    "predicted_gain": (
                        1.0 if proposal["budget_lane"] == "deprioritized" else 0.0
                    ),
                    "uncertainty": 0.0,
                    "cost": 0.0,
                    "evidence": ["budget-lane regression fixture"],
                }
            )

        self.assertTrue(
            any(
                item["budget_lane"] == "deprioritized"
                for item in proposals["proposals"]
            )
        )
        for selection_index in (1, 5):
            point, receipt = select_proposal(
                proposals,
                policy="gain",
                predictions=predictions,
                selection_index=selection_index,
            )
            self.assertIsNone(receipt["budget"]["scheduled_lane"])
            self.assertIsNone(receipt["budget"]["selected_lane"])
            self.assertEqual(receipt["budget"]["fallback"], "lanes_removed")
            self.assertEqual(receipt["budget"]["base_rank"], 1)
            # The highest-scoring proposal wins at every selection index:
            # no lane reservation diverts the fifth slot.
            self.assertEqual(
                selected_assignments(point).get("dim-data-curation"),
                "hyp-data-filtered",
            )

    def test_empty_proposal_lane_state_records_no_fallback(self) -> None:
        proposals = self._proposals(
            {"records": [], "search_space_state": empty_search_space_state()}
        )
        self.assertTrue(
            all(item["budget_lane"] == "active" for item in proposals["proposals"])
        )
        _point, receipt = select_proposal(
            proposals, policy="coverage", selection_index=5
        )
        self.assertIsNone(receipt["budget"]["scheduled_lane"])
        self.assertIsNone(receipt["budget"]["selected_lane"])
        self.assertEqual(receipt["budget"]["fallback"], "lanes_removed")

    def test_pruned_dimension_proposes_only_its_baseline(self) -> None:
        state = state_with(
            dimension_decision(1, "dim-data-curation", "active", "deprioritized"),
            dimension_decision(2, "dim-data-curation", "deprioritized", "pruned"),
        )
        proposals = self._proposals({"records": [], "search_space_state": state})
        self.assertEqual(proposals["search_space_state_revision"], 2)
        chosen = {
            selected_assignments(item["point"]).get("dim-data-curation")
            for item in proposals["proposals"]
        }
        self.assertEqual(chosen, {"hyp-data-raw"})

    def test_select_command_requires_matching_ledger_revision(self) -> None:
        proposals = self._proposals(
            {"records": [], "search_space_state": empty_search_space_state()}
        )
        state = state_with(decision(1, "hyp-data-filtered", "active", "deprioritized"))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            proposals_path = tmp_path / "proposals.json"
            ledger_path = tmp_path / "ledger.json"
            point_path = tmp_path / "point.json"
            receipt_path = tmp_path / "receipt.json"
            proposals_path.write_text(json.dumps(proposals))
            ledger_path.write_text(
                json.dumps({"records": [], "search_space_state": state})
            )
            args = types.SimpleNamespace(
                proposals=proposals_path,
                predictions=None,
                ledger=ledger_path,
                policy="coverage",
                cfg=None,
                point_output=point_path,
                receipt_output=receipt_path,
            )
            with self.assertRaises(ContractError) as raised:
                cmd_select(args)
            self.assertIn("stale", str(raised.exception))

            current = self._proposals({"records": [], "search_space_state": state})
            proposals_path.write_text(json.dumps(current))
            self.assertEqual(cmd_select(args), 0)
            written = json.loads(receipt_path.read_text())
        self.assertEqual(written["schema_version"], 7)
        self.assertEqual(written["search_space_state_revision"], 1)

    def test_llm_weight_is_auditable_and_fixed_midrun(self) -> None:
        proposals = self._proposals(
            {"records": [], "search_space_state": empty_search_space_state()}
        )
        predictions = {
            "schema_version": 1,
            "proposal_set_revision": proposals["proposal_set_revision"],
            "predictions": [
                {
                    "point_id": proposal["point_id"],
                    "predicted_gain": 0.6,
                    "uncertainty": 0.2,
                    "evidence": ["receipt reliability-prior regression fixture"],
                }
                for proposal in proposals["proposals"]
            ],
        }
        point, receipt = select_proposal(
            proposals,
            policy="gain_uncertainty_nocost",
            predictions=predictions,
            config={"llm_intelligence_score": 44},
        )
        entry = record("000", "fresh", [], point, score=0.5, status="keep")
        entry["policy_receipt"] = receipt
        ledger = {
            "search_space": space_receipt(self.registry),
            "search_space_state": empty_search_space_state(),
            "records": [entry],
        }
        self.assertEqual(validate_ledger(self.registry, ledger), [])

        wrong_weight = copy.deepcopy(ledger)
        wrong_weight["records"][0]["policy_receipt"]["components"][
            "llm_judgment_weight"
        ] = 0.61
        errors = validate_ledger(self.registry, wrong_weight)
        self.assertTrue(
            any("must equal policy.config.llm_intelligence_score / 100" in error for error in errors),
            errors,
        )

        wrong_score = copy.deepcopy(ledger)
        wrong_score["records"][0]["policy_receipt"]["acquisition_score"] += 0.01
        errors = validate_ledger(self.registry, wrong_score)
        self.assertTrue(
            any("acquisition_score does not match" in error for error in errors),
            errors,
        )

        second_proposals = build_proposal_set(
            self.registry,
            ledger,
            op="fresh",
            parents=[],
            max_points=64,
        )
        second_predictions = {
            "schema_version": 1,
            "proposal_set_revision": second_proposals["proposal_set_revision"],
            "predictions": [
                {
                    "point_id": proposal["point_id"],
                    "predicted_gain": 0.6,
                    "uncertainty": 0.2,
                    "evidence": ["frozen reliability-prior regression fixture"],
                }
                for proposal in second_proposals["proposals"]
            ],
        }
        second_point, second_receipt = select_proposal(
            second_proposals,
            policy="gain_uncertainty_nocost",
            predictions=second_predictions,
            config={"llm_intelligence_score": 61},
            selection_index=2,
        )
        second_entry = record(
            "001", "fresh", [], second_point, score=0.6, status="discard"
        )
        second_entry["policy_receipt"] = second_receipt
        changed_midrun = copy.deepcopy(ledger)
        changed_midrun["records"].append(second_entry)
        errors = validate_ledger(self.registry, changed_midrun)
        self.assertTrue(
            any("must stay fixed at 44" in error for error in errors),
            errors,
        )

    def test_validate_ledger_replays_each_record_at_its_receipt_revision(self) -> None:
        registry = self.registry
        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "deprioritized", "pruned"),
        )
        ledger = {
            "search_space": space_receipt(registry),
            "records": [],
            "search_space_state": state,
        }
        admitted_at_one = record(
            "000", "fresh", [], self.filtered, score=0.5, status="keep"
        )
        admitted_at_one["policy_receipt"]["search_space_state_revision"] = 1
        admitted_at_one["policy_receipt"]["budget"].update(
            {
                "selected_lane": "deprioritized",
                "fallback": "no_active_proposals",
            }
        )
        ledger["records"].append(admitted_at_one)
        # Replayed at revision 1 the point was only deprioritized: still valid.
        self.assertEqual(validate_ledger(registry, ledger), [])

        forged = copy.deepcopy(admitted_at_one)
        forged["run_id"] = "001"
        forged["policy_receipt"]["search_space_state_revision"] = 2
        ledger["records"].append(forged)
        errors = validate_ledger(registry, ledger)
        self.assertTrue(any("pruned" in error for error in errors), errors)

    def test_validate_ledger_rejects_receipt_revision_beyond_state(self) -> None:
        registry = self.registry
        entry = record("000", "fresh", [], self.filtered, score=0.5, status="keep")
        entry["policy_receipt"]["search_space_state_revision"] = 1
        ledger = {
            "search_space": space_receipt(registry),
            "records": [entry],
            "search_space_state": empty_search_space_state(),
        }
        errors = validate_ledger(registry, ledger)
        self.assertTrue(
            any("search_space_state_revision must be in [0," in error for error in errors),
            errors,
        )

    def test_render_space_reports_all_status_components(self) -> None:
        registry = self.registry
        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "deprioritized", "pruned"),
            dimension_decision(3, "dim-ensemble", "active", "deprioritized"),
        )
        rendered = render_space(
            registry, {"records": [], "search_space_state": state}, max_hypotheses=8
        )
        self.assertEqual(rendered["search_space_state_revision"], 3)
        dimensions = {item["id"]: item for item in rendered["dimensions"]}
        self.assertEqual(dimensions["dim-ensemble"]["runtime_status"], "deprioritized")
        self.assertEqual(dimensions["dim-data-curation"]["runtime_status"], "active")
        hypotheses = {
            item["id"]: item for item in dimensions["dim-data-curation"]["hypotheses"]
        }
        self.assertEqual(
            hypotheses["hyp-data-filtered"]["selection"],
            {
                "guidance_status": "active",
                "dimension_runtime_status": "active",
                "hypothesis_runtime_status": "pruned",
                "effective_status": "pruned",
                "binding_guidance": [],
                "matched_guidance": [{"id": "g-01", "effect": "caution"}],
            },
        )
        # Literature credibility stays a separate, unaffected field.
        self.assertEqual(
            hypotheses["hyp-data-filtered"]["literature_credibility"], "preliminary"
        )
        self.assertEqual(
            hypotheses["hyp-data-raw"]["selection"]["effective_status"], "active"
        )
        ensemble = {
            item["id"]: item for item in dimensions["dim-ensemble"]["hypotheses"]
        }
        stacking = ensemble["hyp-ensemble-stacking"]["selection"]
        self.assertEqual(stacking["dimension_runtime_status"], "deprioritized")
        self.assertEqual(stacking["effective_status"], "deprioritized")
        self.assertEqual(
            ensemble["hyp-ensemble-identity"]["selection"]["effective_status"], "active"
        )


if __name__ == "__main__":
    unittest.main()



def _carrier_ledger_and_experience(registry: dict, *, prune: bool = False) -> dict:
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


class CarrierTransitionTests(unittest.TestCase):
    def test_deprioritize_fires_on_carrier_rule(self) -> None:
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

    def test_staging_still_applies_to_prune_recommendation(self) -> None:
        registry = fixture_registry()
        ledger = _carrier_ledger_and_experience(registry, prune=True)
        transitions = derive_experience_transitions(registry, ledger)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["to_status"], "deprioritized")

    def test_advance_to_pruned_on_new_negative_context(self) -> None:
        registry = fixture_registry()
        ledger = _carrier_ledger_and_experience(registry)
        append_experience_transitions(registry, ledger)
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
        entry["evidence_edge_ids"] = [
            "sedge-000-001", "sedge-002-003", "sedge-004-005"
        ]
        transitions = derive_experience_transitions(registry, ledger)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[0]["from_status"], "deprioritized")
        self.assertEqual(transitions[0]["to_status"], "pruned")

    def test_no_transition_when_evidence_unchanged(self) -> None:
        registry = fixture_registry()
        ledger = _carrier_ledger_and_experience(registry)
        append_experience_transitions(registry, ledger)
        # Same evidence, but a later generation now recommends prune: without
        # advancing evidence the stage cannot advance.
        experience = ledger["experience"]
        experience["generation"] = 2
        experience["dag_revision"] = 5
        entry = experience["hypothesis_evidence"][0]
        entry["recommended_status"] = "pruned"
        entry["confidence"] = "high"
        self.assertEqual(derive_experience_transitions(registry, ledger), [])
