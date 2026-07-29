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
    validate_ledger,
)
from ledger import _load_ledger, cmd_add_record, cmd_brief  # noqa: E402
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
from semantic_evidence import build_semantic_edges  # noqa: E402
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
from tests.p2_fixtures import belief_ledger  # noqa: E402
from validate_background import (  # noqa: E402
    background_text,
    fixture_registry,
    policy_receipt,
    record,
)


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
        self.assertTrue(replayed["dimensions"])
        self.assertTrue(replayed["hypotheses"])
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

        reopened = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "deprioritized", "pruned"),
            decision(3, "hyp-data-filtered", "pruned", "active"),
        )
        at_three = replay_search_space_state(registry, reopened)
        self.assertEqual(at_three["hypotheses"]["hyp-data-filtered"], "active")
        self.assertEqual(
            replay_search_space_state(registry, reopened, revision=2)["hypotheses"][
                "hyp-data-filtered"
            ],
            "pruned",
        )

    def test_validate_rejects_active_to_pruned(self) -> None:
        registry = fixture_registry()
        state = state_with(decision(1, "hyp-data-filtered", "active", "pruned"))
        errors = validate_search_space_state(registry, ledger_with_state(state))
        self.assertTrue(any("transition" in error for error in errors), errors)

    def test_validate_rejects_from_status_mismatch(self) -> None:
        registry = fixture_registry()
        state = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(2, "hyp-data-filtered", "active", "deprioritized"),
        )
        errors = validate_search_space_state(registry, ledger_with_state(state))
        self.assertTrue(any("from_status" in error for error in errors), errors)

    def test_validate_rejects_nonsequential_revision_and_underived_id(self) -> None:
        registry = fixture_registry()
        skipped = state_with(
            decision(1, "hyp-data-filtered", "active", "deprioritized"),
            decision(3, "hyp-data-filtered", "deprioritized", "pruned"),
        )
        skipped["revision"] = 2
        errors = validate_search_space_state(registry, ledger_with_state(skipped))
        self.assertTrue(any("revision" in error for error in errors), errors)

        bad_id = state_with(decision(1, "hyp-data-filtered", "active", "deprioritized"))
        bad_id["decisions"][0]["decision_id"] = "sdec-000009"
        errors = validate_search_space_state(registry, ledger_with_state(bad_id))
        self.assertTrue(any("decision_id" in error for error in errors), errors)

        bad_revision = state_with(decision(1, "hyp-data-filtered", "active", "deprioritized"))
        bad_revision["revision"] = 0
        errors = validate_search_space_state(registry, ledger_with_state(bad_revision))
        self.assertTrue(any("revision" in error for error in errors), errors)

    def test_validate_rejects_unknown_and_missing_fields(self) -> None:
        registry = fixture_registry()
        extra = state_with(decision(1, "hyp-data-filtered", "active", "deprioritized"))
        extra["decisions"][0]["note"] = "free text is not auditable"
        errors = validate_search_space_state(registry, ledger_with_state(extra))
        self.assertTrue(any("unknown fields" in error for error in errors), errors)

        missing = state_with(decision(1, "hyp-data-filtered", "active", "deprioritized"))
        del missing["decisions"][0]["reopen_when"]
        errors = validate_search_space_state(registry, ledger_with_state(missing))
        self.assertTrue(any("reopen_when" in error for error in errors), errors)

        extra_top = state_with(decision(1, "hyp-data-filtered", "active", "deprioritized"))
        extra_top["state_note"] = "not part of the contract"
        errors = validate_search_space_state(registry, ledger_with_state(extra_top))
        self.assertTrue(any("unknown fields" in error for error in errors), errors)

    def test_validate_rejects_unknown_targets(self) -> None:
        registry = fixture_registry()
        unknown_hypothesis = state_with(
            decision(1, "hyp-not-real", "active", "deprioritized")
        )
        errors = validate_search_space_state(registry, ledger_with_state(unknown_hypothesis))
        self.assertTrue(any("target" in error for error in errors), errors)

        wrong_owner = state_with(
            hypothesis_decision(
                1, "dim-model-architecture", "hyp-data-filtered", "active", "deprioritized"
            )
        )
        errors = validate_search_space_state(registry, ledger_with_state(wrong_owner))
        self.assertTrue(any("dimension_id" in error for error in errors), errors)

        unknown_dimension = state_with(
            dimension_decision(1, "dim-not-real", "active", "deprioritized")
        )
        errors = validate_search_space_state(registry, ledger_with_state(unknown_dimension))
        self.assertTrue(any("target" in error for error in errors), errors)

    def test_validate_rejects_baseline_and_baseline_only_targets(self) -> None:
        registry = fixture_registry()
        baseline_hypothesis = state_with(
            decision(1, "hyp-data-raw", "active", "deprioritized")
        )
        errors = validate_search_space_state(registry, ledger_with_state(baseline_hypothesis))
        self.assertTrue(any("baseline" in error for error in errors), errors)

        baseline_only_dimension = state_with(
            dimension_decision(1, "dim-initialization-adaptation", "active", "deprioritized")
        )
        errors = validate_search_space_state(
            registry, ledger_with_state(baseline_only_dimension)
        )
        self.assertTrue(any("baseline_only" in error for error in errors), errors)


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
    def test_baseline_point_is_eligible(self) -> None:
        registry = fixture_registry()
        guidance = derive_hypothesis_selection(registry)
        effective = compose_effective_selection(
            registry, guidance, {"dimensions": {}, "hypotheses": {}}
        )
        point = complete_point(registry)
        self.assertEqual(validate_point_eligibility(point, registry, effective), [])

    def test_excluded_hypothesis_is_ineligible(self) -> None:
        registry = fixture_registry()
        registry["guidance"].append(
            {"id": "g-90", "effect": "exclude", "scope": _scope("filtering")}
        )
        guidance = derive_hypothesis_selection(registry)
        effective = compose_effective_selection(
            registry, guidance, {"dimensions": {}, "hypotheses": {}}
        )
        point = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        errors = validate_point_eligibility(point, registry, effective)
        self.assertTrue(any("excluded" in error for error in errors), errors)

    def test_pruned_hypothesis_is_ineligible_but_structurally_valid(self) -> None:
        registry = fixture_registry()
        guidance = derive_hypothesis_selection(registry)
        runtime = {
            "dimensions": {},
            "hypotheses": {"hyp-data-filtered": "pruned"},
        }
        effective = compose_effective_selection(registry, guidance, runtime)
        point = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        # Historical observations stay structurally valid; only new selection is barred.
        self.assertEqual(validate_point(point, registry), [])
        errors = validate_point_eligibility(point, registry, effective)
        self.assertTrue(any("pruned" in error for error in errors), errors)

    def test_deprioritized_dimension_remains_eligible(self) -> None:
        registry = fixture_registry()
        guidance = derive_hypothesis_selection(registry)
        runtime = {
            "dimensions": {"dim-data-curation": "deprioritized"},
            "hypotheses": {},
        }
        effective = compose_effective_selection(registry, guidance, runtime)
        # Deprioritization changes the admission budget, not structural eligibility.
        filtered_point = complete_point(
            registry, {"dim-data-curation": "hyp-data-filtered"}
        )
        self.assertEqual(
            validate_point_eligibility(filtered_point, registry, effective), []
        )
        self.assertEqual(
            effective["hyp-data-filtered"]["dimension_runtime_status"],
            "deprioritized",
        )
        baseline_point = complete_point(registry)
        self.assertEqual(
            validate_point_eligibility(baseline_point, registry, effective), []
        )
        cv_point = complete_point(
            registry, {"dim-validation-selection": "hyp-valid-cv"}
        )
        self.assertEqual(validate_point_eligibility(cv_point, registry, effective), [])

    def test_deprioritized_hypothesis_remains_eligible(self) -> None:
        registry = fixture_registry()
        guidance = derive_hypothesis_selection(registry)
        runtime = {
            "dimensions": {},
            "hypotheses": {"hyp-data-filtered": "deprioritized"},
        }
        effective = compose_effective_selection(registry, guidance, runtime)
        point = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})
        self.assertEqual(validate_point_eligibility(point, registry, effective), [])


class ExperienceTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = fixture_registry()
        self.ledger = belief_ledger(self.registry)
        self.ledger["search_space_state"] = empty_search_space_state()

    def advance_dag_cursor(self) -> None:
        self.ledger["records"][-1]["final_best_score"] += 0.001
        self.ledger["dag_revision"] += 1
        self.ledger["records"][-1]["dag_revision"] = self.ledger["dag_revision"]

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
        self.ledger["dag_revision"] += 1
        pending["dag_revision"] = self.ledger["dag_revision"]
        experience = self.experience(generation=5)
        entry = experience["hypothesis_evidence"][0]
        entry["evidence_run_ids"].append("004")
        entry["evidence_edge_ids"].append("sedge-000-004")
        entry["comparator_coverage"]["direct_noncrash_edges"] = 3
        self.ledger["experience"] = experience
        second = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(second[0]["from_status"], "deprioritized")
        self.assertEqual(second[0]["to_status"], "pruned")

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

    def test_empty_beliefs_make_no_transition(self) -> None:
        experience = self.experience(generation=1)
        experience["hypothesis_evidence"] = []
        self.ledger["experience"] = experience
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        self.assertEqual(self.ledger["search_space_state"]["revision"], 0)

    def test_low_confidence_cannot_prune(self) -> None:
        experience = self.experience(generation=1)
        experience["hypothesis_evidence"][0]["confidence"] = "low"
        self.ledger["experience"] = experience
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["hypotheses"]["hyp-data-filtered"], "active")

    def test_one_direct_edge_cannot_deprioritize(self) -> None:
        def one_edge(generation: int) -> dict:
            experience = self.experience(generation)
            entry = experience["hypothesis_evidence"][0]
            entry["evaluation_state"] = "observed"
            entry["evidence_run_ids"] = ["000", "001"]
            entry["evidence_edge_ids"] = ["sedge-000-001"]
            entry["comparator_coverage"] = {
                "direct_noncrash_edges": 1,
                "confounded_noncrash_edges": 0,
                "crash_edges": 0,
            }
            return experience

        self.ledger["experience"] = one_edge(generation=1)
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        self.ledger["experience"] = one_edge(generation=2)
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["hypotheses"]["hyp-data-filtered"], "active")

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

    def test_baseline_target_is_rejected(self) -> None:
        experience = self.experience(generation=1)
        entry = experience["hypothesis_evidence"][0]
        entry["target_id"] = "hyp-data-raw"
        entry["claim"] = "Both direct comparisons moved away from the raw baseline."
        self.ledger["experience"] = experience
        # The comparator gate passes on the recomputed coverage; only baseline
        # protection blocks the transition.
        self.assertEqual(append_experience_transitions(self.registry, self.ledger), [])
        replayed = replay_search_space_state(
            self.registry, self.ledger["search_space_state"]
        )
        self.assertEqual(replayed["hypotheses"]["hyp-data-raw"], "active")

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
        parent = {
            "run_id": "004",
            "source_run_ids": [],
            "semantic_point": complete_point(self.registry),
            "semantic_edges": [],
            "status": "keep",
            "final_best_score": 0.39,
            "dag_revision": 5,
        }
        child = {
            "run_id": "005",
            "source_run_ids": ["004"],
            "semantic_point": complete_point(
                self.registry, {"dim-data-curation": "hyp-data-filtered"}
            ),
            "status": "keep",
            "final_best_score": 0.38,
            "dag_revision": 6,
        }
        child["semantic_edges"] = build_semantic_edges(
            [*self.ledger["records"], parent], child
        )
        self.ledger["records"].extend([parent, child])
        self.ledger["dag_revision"] = 6
        experience = self.experience(generation=4)
        experience["updated_at_run"] = "005"
        entry = experience["hypothesis_evidence"][0]
        entry.update(
            {
                "assessment": "promising",
                "recommended_status": "active",
                "claim": "A later direct comparison improved over its matched parent.",
                "evidence_run_ids": ["004", "005"],
                "evidence_edge_ids": ["sedge-000-001", "sedge-002-003", "sedge-004-005"],
                "comparator_coverage": {
                    "direct_noncrash_edges": 3,
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
        # Later score changes never rewrite an appended decision's snapshot.
        for record in self.ledger["records"]:
            if record["run_id"] == "001":
                record["final_best_score"] = 0.99
        stored = self.ledger["search_space_state"]["decisions"][0]
        self.assertEqual(stored["evidence_observations"][0]["child_score"], 0.5)
        self.assertEqual(stored["evidence_observations"][0]["delta"], 0.1)
        # The next decision snapshots the current observation values instead.
        self.advance_dag_cursor()
        self.ledger["experience"] = self.experience(generation=2)
        (second,) = append_experience_transitions(self.registry, self.ledger)
        self.assertEqual(second["evidence_observations"][0]["child_score"], 0.99)
        self.assertEqual(second["evidence_observations"][0]["delta"], 0.59)
        self.assertEqual(validate_search_space_state(self.registry, self.ledger), [])

    def dimension_belief(self, generation: int, *, with_hypothesis: bool) -> dict:
        experience = self.experience(generation)
        experience["dimension_evidence"] = [{
            "target_id": "dim-data-curation",
            "evaluation_state": "comparator_covered",
            "assessment": "unpromising",
            "recommended_status": "pruned",
            "claim": "Every matched data-curation change worsened the score.",
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

    def test_historical_policy_receipt_schema_2_remains_readable(self) -> None:
        registry = fixture_registry()
        baseline = complete_point(registry)
        entry = record("000", "fresh", [], baseline, score=0.5, status="keep")
        receipt = entry["policy_receipt"]
        receipt["schema_version"] = 2
        receipt.pop("budget")
        receipt.pop("experience")
        receipt["policy"]["config"].pop("deprioritized_budget_interval")
        for field in (
            "prior_gain",
            "experience_gain_adjustment",
            "prior_uncertainty",
            "experience_uncertainty_adjustment",
        ):
            receipt["components"].pop(field)
        ledger = {
            "search_space": space_receipt(registry),
            "records": [entry],
            "search_space_state": empty_search_space_state(),
        }
        self.assertEqual(validate_ledger(registry, ledger), [])

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


class StateAwareSelectionLifecycleTests(unittest.TestCase):
    """Proposals and selection receipts track the revisioned pruning overlay.

    Automated pruning is two-stage (``active -> deprioritized -> pruned``), so
    the pruned overlay is revision 2 and reopening lands at revision 3; the
    deprioritized stage at revision 1 exercises the deterministic budget lane.
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
                    policy_receipt(
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
            )
            with self.assertRaises(SystemExit) as raised:
                cmd_add_record(args)
        self.assertIn(
            "record 001 parent 999 is not an earlier record",
            str(raised.exception),
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
            # This otherwise-current schema-4 receipt falsely claims that no
            # experience snapshot existed at admission.
            receipt_path.write_text(
                json.dumps(
                    policy_receipt(
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
            )
            with self.assertRaises(SystemExit) as raised:
                cmd_add_record(args)
            self.assertIn(
                "stale policy receipt: experience", str(raised.exception)
            )

            forged_receipt = policy_receipt(
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
            "experience evidence ids must be cited", str(forged.exception)
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
            "schema_version": 2,
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
        self.assertEqual(proposals["schema_version"], 3)
        self.assertEqual(proposals["search_space_state_revision"], 0)
        self.assertTrue(any(self._selects_filtered(p) for p in proposals["proposals"]))
        _, receipt = select_proposal(proposals, policy="coverage")
        self.assertEqual(receipt["schema_version"], 4)
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
        self.assertEqual(proposals["schema_version"], 3)
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
        self.assertEqual(receipt["schema_version"], 4)
        self.assertEqual(receipt["search_space_state_revision"], 1)
        self.assertEqual(receipt["budget"]["selected_lane"], "active")

        # 3. Revision 2 proposals omit the pruned hypothesis.
        state["decisions"].append(
            decision(2, "hyp-data-filtered", "deprioritized", "pruned")
        )
        state["revision"] = 2
        proposals = self._proposals(ledger)
        self.assertEqual(proposals["schema_version"], 3)
        self.assertEqual(proposals["search_space_state_revision"], 2)
        self.assertFalse(any(self._selects_filtered(p) for p in proposals["proposals"]))
        _, receipt = select_proposal(proposals, policy="coverage")
        self.assertEqual(receipt["schema_version"], 4)
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
                    policy_receipt(
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
            )
            with self.assertRaises(SystemExit) as raised:
                cmd_add_record(args)
            self.assertIn("stale", str(raised.exception))
            # A revision-current receipt still cannot admit a pruned point:
            # the selected point must be eligible at its receipt revision.
            receipt_path.write_text(
                json.dumps(
                    policy_receipt(
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

    def test_deprioritized_lane_receives_only_its_budget_slot(self) -> None:
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

        active_point, active_receipt = select_proposal(
            proposals,
            policy="gain",
            predictions=predictions,
            selection_index=1,
        )
        self.assertEqual(active_receipt["budget"]["scheduled_lane"], "active")
        self.assertEqual(active_receipt["budget"]["selected_lane"], "active")
        self.assertEqual(active_receipt["budget"]["fallback"], "none")
        self.assertNotEqual(
            selected_assignments(active_point).get("dim-data-curation"),
            "hyp-data-filtered",
        )

        deprioritized_point, deprioritized_receipt = select_proposal(
            proposals,
            policy="gain",
            predictions=predictions,
            selection_index=5,
        )
        self.assertEqual(
            deprioritized_receipt["budget"]["scheduled_lane"], "deprioritized"
        )
        self.assertEqual(
            deprioritized_receipt["budget"]["selected_lane"], "deprioritized"
        )
        self.assertEqual(deprioritized_receipt["budget"]["fallback"], "none")
        self.assertEqual(
            selected_assignments(deprioritized_point).get("dim-data-curation"),
            "hyp-data-filtered",
        )
        self.assertGreater(deprioritized_receipt["budget"]["base_rank"], 0)

    def test_empty_scheduled_lane_records_deterministic_fallback(self) -> None:
        proposals = self._proposals(
            {"records": [], "search_space_state": empty_search_space_state()}
        )
        self.assertTrue(
            all(item["budget_lane"] == "active" for item in proposals["proposals"])
        )
        _point, receipt = select_proposal(
            proposals, policy="coverage", selection_index=5
        )
        self.assertEqual(receipt["budget"]["scheduled_lane"], "deprioritized")
        self.assertEqual(receipt["budget"]["selected_lane"], "active")
        self.assertEqual(
            receipt["budget"]["fallback"], "no_deprioritized_proposals"
        )

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
        registry = self.registry
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
        self.assertEqual(written["schema_version"], 4)
        self.assertEqual(written["search_space_state_revision"], 1)

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
