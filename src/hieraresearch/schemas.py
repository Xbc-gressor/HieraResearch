"""JSON schemas for the small judgment-only Claude calls."""

from __future__ import annotations

from typing import Any


IDEA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "idea": {"type": "string"},
        "change": {"type": "string"},
        "candidate_name_hint": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["idea", "change", "candidate_name_hint", "description"],
    "additionalProperties": False,
}


DEBUG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["config_invalid", "code_incompatible", "abandon"],
        },
        "rationale": {"type": "string"},
        "corrected_config": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "key": {"type": "string"},
                    "value_json": {"type": "string"},
                },
                "required": ["key", "value_json"],
                "additionalProperties": False,
            },
        },
        "repair_instructions": {"type": "string"},
    },
    "required": ["verdict", "rationale", "corrected_config", "repair_instructions"],
    "additionalProperties": False,
}


def prediction_schema(*, include_cost: bool) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "point_id": {"type": "string"},
        "prior_gain": {"type": "number"},
        "experience_gain_adjustment": {"type": "number"},
        "predicted_gain": {"type": "number"},
        "prior_uncertainty": {"type": "number"},
        "experience_uncertainty_adjustment": {"type": "number"},
        "uncertainty": {"type": "number"},
        "experience_target_ids": {"type": "array", "items": {"type": "string"}},
        "experience_run_ids": {"type": "array", "items": {"type": "string"}},
        "experience_edge_ids": {"type": "array", "items": {"type": "string"}},
        "experience_rationale": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    }
    if include_cost:
        properties["cost"] = {"type": "number"}
    required = list(properties)
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer", "enum": [3]},
            "proposal_set_revision": {"type": "string"},
            "experience": {
                "type": "object",
                "properties": {
                    "generation": {"type": ["integer", "null"]},
                    "updated_at_run": {"type": ["string", "null"]},
                    "revision": {"type": ["string", "null"]},
                },
                "required": ["generation", "updated_at_run", "revision"],
                "additionalProperties": False,
            },
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        },
        "required": ["schema_version", "proposal_set_revision", "experience", "predictions"],
        "additionalProperties": False,
    }


def _belief_item_schema(*, lesson: bool = False, bottleneck: bool = False) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "claim": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "med", "high"]},
    }
    required = ["claim", "evidence", "confidence"]
    if lesson:
        properties["kind"] = {
            "type": "string",
            "enum": ["lever", "deadend", "feasibility"],
        }
        properties["reopen_when"] = {"type": "string"}
        required.append("kind")
    elif not bottleneck:
        properties["uncertainty"] = {"type": "string"}
        required.append("uncertainty")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _target_evidence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "target_id": {"type": "string"},
            "evaluation_state": {
                "type": "string",
                "enum": ["unevaluated", "failed", "observed", "comparator_covered"],
            },
            "assessment": {
                "type": "string",
                "enum": ["unknown", "promising", "mixed", "unpromising"],
            },
            "recommended_status": {
                "type": "string",
                "enum": ["active", "deprioritized", "pruned"],
            },
            "claim": {"type": "string"},
            "evidence_run_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_edge_ids": {"type": "array", "items": {"type": "string"}},
            "comparator_coverage": {
                "type": "object",
                "properties": {
                    "direct_tuned_edges": {"type": "integer"},
                    "direct_noncrash_edges": {"type": "integer"},
                    "confounded_noncrash_edges": {"type": "integer"},
                    "crash_edges": {"type": "integer"},
                },
                "required": [
                    "direct_tuned_edges",
                    "direct_noncrash_edges",
                    "confounded_noncrash_edges",
                    "crash_edges",
                ],
                "additionalProperties": False,
            },
            "confidence": {"type": "string", "enum": ["low", "med", "high"]},
            "uncertainty": {"type": "string"},
            "reopen_when": {"type": "string"},
        },
        "required": [
            "target_id",
            "evaluation_state",
            "assessment",
            "recommended_status",
            "claim",
            "evidence_run_ids",
            "evidence_edge_ids",
            "comparator_coverage",
            "confidence",
            "uncertainty",
        ],
        "additionalProperties": False,
    }


EXPERIENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "integer", "enum": [4]},
        "updated_at_run": {"type": "string"},
        "generation": {"type": "integer"},
        "summary": {"type": "string"},
        "promising_regions": {"type": "array", "items": _belief_item_schema()},
        "lessons": {"type": "array", "items": _belief_item_schema(lesson=True)},
        "bottlenecks": {"type": "array", "items": _belief_item_schema(bottleneck=True)},
        "dimension_evidence": {"type": "array", "items": _target_evidence_schema()},
        "hypothesis_evidence": {"type": "array", "items": _target_evidence_schema()},
    },
    "required": [
        "schema_version",
        "updated_at_run",
        "generation",
        "summary",
        "promising_regions",
        "lessons",
        "bottlenecks",
        "dimension_evidence",
        "hypothesis_evidence",
    ],
    "additionalProperties": False,
}
