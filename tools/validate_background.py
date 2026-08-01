#!/usr/bin/env python3
"""End-to-end contract checks for the semantic search space.

Keep this validator small. It covers only what the pytest suite cannot:

* the ``background.md`` round trip through the real loader, and refusal of the
  legacy flat schema;
* benchmark-shape neutrality — the MLE-bench-shaped and PostTrainBench-shaped
  ownership maps must drive the shared helpers identically, so any
  estimator-specific branch shows up as a divergent signature;
* retrieval provenance gating, where head/brief metadata is triage only;
* one real-subprocess run of the round lifecycle (admit, score, refresh
  belief, apply state, re-propose), using an explicit fixture-only capability
  to exercise the dormant downstream paired-comparator contract without
  claiming production can create those receipts.

Per-rule schema, belief-gate, acquisition, and state-transition behaviour is
unit-tested in ``tests/``; do not restate it here.

Run: ``python3 tools/validate_background.py`` (exit code 0 means all passed).
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from background_contract import (  # noqa: E402
    ContractError,
    EXPERIENCE_SCHEMA_VERSION,
    derive_hypothesis_selection,
    load_registry,
    validate_background_markdown,
    validate_registry,
)
from search_backends import add_visit, new_manifest  # noqa: E402
from search_space_state import (  # noqa: E402
    compose_effective_selection,
    derive_experience_transitions,
    empty_search_space_state,
    replay_search_space_state,
)
from semantic_evidence import (  # noqa: E402
    DIRECT_COMPARATOR_CAPABILITY_KEY,
    DIRECT_COMPARATOR_FIXTURE_CAPABILITY,
    comparator_coverage,
)
from semantic_search import (  # noqa: E402
    build_proposal_set,
    select_proposal,
    validate_proposal_set,
)
from semantic_space import (  # noqa: E402
    complete_point,
    digest,
    load_catalog,
    selected_assignments,
    validate_point,
)
from tests.fixtures import (  # noqa: E402
    attach_matched_transfer,
    background_text,
    fixture_registry,
    policy_receipt,
    shape_registry,
)

COVERAGE_FIXTURE = ROOT / "tests" / "fixtures" / "semantic-space-coverage.json"


def check_markdown_round_trip(registry: dict) -> None:
    """The frozen registry survives a render/load cycle; legacy is refused."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "background.md"
        path.write_text(background_text(registry))
        assert load_registry(path) == registry
        assert validate_background_markdown(path, registry) == []

        # The human-readable half must keep naming every registered hypothesis.
        path.write_text(background_text(registry).replace("`hyp-data-filtered`", "filtered", 2))
        errors = validate_background_markdown(path, registry)
        assert any("hyp-data-filtered" in error for error in errors), errors

        legacy = Path(tmp) / "legacy.md"
        legacy.write_text(
            '# old\n\n## Direction registry\n```json\n{"schema_version":2,"directions":[]}\n```\n'
        )
        try:
            load_registry(legacy)
        except ContractError as exc:
            assert "legacy flat" in str(exc)
        else:
            raise AssertionError("legacy flat background was silently accepted")


def check_shape_neutrality() -> None:
    """Both benchmark shapes must exercise the helpers identically.

    Each shape owns every catalog dimension through its own names. If the
    resulting statuses, proposal counts, or revision stamps differ, a helper
    has grown a benchmark-specific branch.
    """
    catalog = load_catalog()
    catalog_ids = {item["id"] for item in catalog["dimensions"]}
    coverage = json.loads(COVERAGE_FIXTURE.read_text())

    signatures: dict[str, dict] = {}
    for shape_name, mapping in coverage.items():
        # No catch-all dimension may absorb an unmapped material choice.
        assert set(mapping.values()) == catalog_ids, (shape_name, mapping)
        assert len(mapping) == len(catalog_ids), (shape_name, mapping)
        assert all("misc" not in value for value in mapping.values())

        shape = shape_registry(catalog, mapping, f"toy-{shape_name}")
        assert validate_registry(shape) == [], validate_registry(shape)
        state = empty_search_space_state()
        runtime = replay_search_space_state(shape, state)
        effective = compose_effective_selection(
            shape, derive_hypothesis_selection(shape), runtime
        )
        ledger = {"records": [], "search_space_state": state}
        proposals = build_proposal_set(
            shape, ledger, op="fresh", parents=[], max_points=16
        )
        assert validate_proposal_set(proposals) == []
        assert proposals["proposals"]
        point, receipt = select_proposal(proposals, policy="coverage")
        assert validate_point(point, shape) == []
        assert derive_experience_transitions(shape, ledger) == []
        signatures[shape_name] = {
            "dimension_statuses": list(runtime["dimensions"].values()),
            "hypothesis_statuses": list(runtime["hypotheses"].values()),
            "effective_statuses": [
                entry["effective_status"] for entry in effective.values()
            ],
            "n_proposals": len(proposals["proposals"]),
            "proposal_state_revision": proposals["search_space_state_revision"],
            "receipt_state_revision": receipt["search_space_state_revision"],
        }

    assert set(signatures) == {"mle_bench_shaped", "posttrain_bench_shaped"}
    first, *rest = signatures.values()
    assert all(other == first for other in rest), signatures


def check_retrieval_provenance(registry: dict) -> None:
    """Every literature source needs an inspected primary-source visit."""
    manifest = new_manifest()
    manifest["coverage_exemptions"] = [
        {
            "dimension_id": dimension["id"],
            "rationale": "Synthetic contract fixture does not run literature search.",
        }
        for dimension in registry["dimensions"]
        if dimension["mode"] == "searchable"
    ]

    head_only = copy.deepcopy(manifest)
    add_visit(
        head_only,
        url=registry["sources"][0]["url"],
        lane="grounding",
        backend="deepxiv",
        view="head",
        status="success",
        content='{"abstract":"metadata only","sections":{"Method":{"token_count":100}}}',
    )
    errors = validate_registry(registry, retrieval_manifest=head_only)
    assert any("head/brief metadata is triage only" in error for error in errors), errors

    add_visit(
        manifest,
        url=registry["sources"][0]["url"],
        lane="grounding",
        backend="fixture",
        view="full_text",
        status="success",
        content="inspected primary source" * 30,
    )
    assert validate_registry(registry, retrieval_manifest=manifest) == []


class Run:
    """Drives the real CLIs against one temporary run directory."""

    def __init__(self, tmp: Path, registry: dict) -> None:
        self.registry = registry
        self.background = tmp / "background.md"
        self.ledger = tmp / "ledger.json"
        self.point = tmp / "point.json"
        self.policy = tmp / "policy.json"
        self.experience = tmp / "experience.json"
        self.proposals = tmp / "proposals.json"
        self.background.write_text(background_text(registry))

    def _cli(self, tool: str, *args: str, expect_failure: bool = False):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "tools" / tool), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if expect_failure:
            assert completed.returncode != 0, completed.stdout
        else:
            assert completed.returncode == 0, completed.stderr
        return completed

    def _stored(self) -> dict:
        return json.loads(self.ledger.read_text()) if self.ledger.exists() else {}

    def add_record(self, run_id: str, op: str, parents: list[str], point: dict, **kwargs):
        """Admit one candidate, pinning the receipt to current durable state."""
        stored = self._stored()
        receipt = policy_receipt(
            op,
            parents,
            point,
            selection_index=len(stored.get("records", [])) + 1,
            schema_version=6,
            **kwargs,
        )
        experience = stored.get("experience")
        if isinstance(experience, dict) and experience:
            receipt["experience"].update(
                {
                    "generation": experience["generation"],
                    "updated_at_run": experience["updated_at_run"],
                    "revision": digest(experience),
                }
            )
        self.point.write_text(json.dumps(point))
        self.policy.write_text(json.dumps(receipt))
        return self._cli(
            "ledger.py", "add-record",
            "--ledger", str(self.ledger),
            "--task", "hard-interactions",
            "--run-id", run_id,
            "--op", op,
            "--source-run-ids", ",".join(parents),
            "--background", str(self.background),
            "--semantic-point", str(self.point),
            "--policy-receipt", str(self.policy),
            "--idea", f"Complete fixture solution {run_id} at the selected point.",
            "--change", f"fixture change for {op} run {run_id}",
            "--candidate-name-hint", f"fixture_{run_id}",
            **kwargs.pop("cli", {}),
        )

    def stale_add_record_is_refused(self, run_id: str, point: dict, *, state_revision: int) -> None:
        self.point.write_text(json.dumps(point))
        self.policy.write_text(
            json.dumps(
                policy_receipt(
                    "fresh", [], point, state_revision=state_revision, schema_version=6
                )
            )
        )
        before = self.ledger.read_bytes()
        completed = self._cli(
            "ledger.py", "add-record",
            "--ledger", str(self.ledger),
            "--task", "hard-interactions",
            "--run-id", run_id,
            "--op", "fresh",
            "--source-run-ids", "",
            "--background", str(self.background),
            "--semantic-point", str(self.point),
            "--policy-receipt", str(self.policy),
            "--idea", "A forged late arrival at the pruned point.",
            "--change", "from scratch at the pruned point",
            "--candidate-name-hint", "fixture_stale",
            expect_failure=True,
        )
        assert "stale" in completed.stderr, completed.stderr
        # Admission precedes candidate work, so nothing durable may change.
        assert self.ledger.read_bytes() == before

    def record_run(self, run_id: str, score: float | None = None) -> None:
        """Report a result, attaching the matched control a real tuner would."""
        if score is not None:
            stored = self._stored()
            child = next(r for r in stored["records"] if r["run_id"] == run_id)
            parents = child.get("source_run_ids") or []
            if parents:
                parent = next(
                    r for r in stored["records"] if r["run_id"] == parents[0]
                )
                attach_matched_transfer(parent, child, control_score=float(score))
                # The lifecycle's prune/reopen beliefs need contradiction-grade
                # comparators, which require tuned children. No runtime tuning
                # report exists in this CLI fixture, so mark depth directly —
                # same fixture posture as the capability patch below.
                child["evaluation_depth"] = "tuned"
                # This validator exercises the dormant downstream paired
                # contract. Production ledger writers emit only the explicit
                # unavailable capability; the fixture-only gate cannot be
                # produced by a runtime tuning report.
                stored[DIRECT_COMPARATOR_CAPABILITY_KEY] = dict(
                    DIRECT_COMPARATOR_FIXTURE_CAPABILITY
                )
                self.ledger.write_text(json.dumps(stored))
        args = ["--ledger", str(self.ledger), "--task", "hard-interactions", "--run-id", run_id]
        if score is not None:
            args += ["--final-best-score", str(score)]
        self._cli("ledger.py", "record-run", *args)

    def set_experience(
        self,
        generation: int,
        updated_at_run: str,
        beliefs: list[dict] | None = None,
        dimension_beliefs: list[dict] | None = None,
    ) -> None:
        self.experience.write_text(
            json.dumps(
                {
                    "schema_version": EXPERIENCE_SCHEMA_VERSION,
                    "updated_at_run": updated_at_run,
                    "generation": generation,
                    "summary": (
                        "Comparator evidence against the filtered hypothesis."
                        if beliefs or dimension_beliefs
                        else ""
                    ),
                    "promising_regions": [],
                    "lessons": [],
                    "bottlenecks": [],
                    "dimension_evidence": dimension_beliefs or [],
                    "hypothesis_evidence": beliefs or [],
                }
            )
        )
        self._cli(
            "ledger.py", "set-experience",
            "--ledger", str(self.ledger),
            "--task", "hard-interactions",
            "--background", str(self.background),
            "--from-json", str(self.experience),
        )

    def apply_space_state(self) -> dict:
        """Apply pruning decisions; this must never advance the DAG cursor."""
        before = self._stored().get("dag_revision")
        applied = json.loads(
            self._cli(
                "ledger.py", "apply-space-state",
                "--ledger", str(self.ledger),
                "--background", str(self.background),
            ).stdout
        )
        assert self._stored().get("dag_revision") == before
        return applied

    def propose(self) -> dict:
        self._cli(
            "semantic_search.py", "propose",
            "--background", str(self.background),
            "--ledger", str(self.ledger),
            "--op", "fresh",
            "--output", str(self.proposals),
        )
        return json.loads(self.proposals.read_text())

    def target_evidence(self, target_id: str) -> dict:
        return json.loads(
            self._cli(
                "background_contract.py", "target-evidence",
                "--background", str(self.background),
                "--ledger", str(self.ledger),
                "--target-id", target_id,
            ).stdout
        )


def _belief(runs: list[str], edges: list[str], **overrides) -> dict:
    belief = {
        "target_id": "hyp-data-filtered",
        "evaluation_state": "comparator_covered",
        "assessment": "unpromising",
        "recommended_status": "pruned",
        "claim": (
            "Repeated matched comparisons show the filtering mechanism removes "
            "useful signal without reducing downstream cost."
        ),
        "evidence_run_ids": runs,
        "evidence_edge_ids": edges,
        "comparator_coverage": {
            "direct_tuned_edges": len(edges),
            "direct_noncrash_edges": 0,
            "confounded_noncrash_edges": 0,
            "crash_edges": 0,
        },
        "confidence": "high",
        "uncertainty": (
            "Matched transfer controls isolate tuned state, but candidate-code "
            "fidelity can still vary."
        ),
        "reopen_when": "A later direct comparison improves over its parent.",
    }
    belief.update(overrides)
    return belief


def check_round_lifecycle(registry: dict) -> None:
    """One real-CLI pass: two-stage pruning, stale refusal, then reopening."""
    baseline = complete_point(registry)
    target = complete_point(registry, {"dim-data-curation": "hyp-data-filtered"})

    def selects_target(proposal: dict) -> bool:
        return (
            selected_assignments(proposal["point"]).get("dim-data-curation")
            == "hyp-data-filtered"
        )

    with tempfile.TemporaryDirectory() as tmp:
        run = Run(Path(tmp), registry)

        # Two matched comparisons against a fresh baseline parent, both worse.
        run.add_record("000", "fresh", [], baseline)
        run.record_run("000", 0.45)
        run.set_experience(0, "000")
        run.add_record("001", "improve", ["000"], target)
        run.record_run("001", 0.60)
        run.set_experience(0, "001")
        run.add_record("002", "improve", ["000"], target)
        run.record_run("002", 0.62)

        # Stage one: comparator-covered contradiction deprioritizes the target.
        edges = ["sedge-000-001", "sedge-000-002"]
        run.set_experience(1, "002", [_belief(["000", "001", "002"], edges)])
        assert run.apply_space_state() == {
            "ok": True, "prior_revision": 0, "revision": 1,
            "decision_ids": ["sdec-000001"],
        }

        # A deprioritized target still gets its reserved budget lane, and that
        # new direct comparison is the fresh evidence stage two requires.
        run.add_record(
            "003", "improve", ["000"], target,
            state_revision=1, selected_lane="deprioritized", deprioritized_interval=2,
        )
        run.record_run("003", 0.61)
        edges.append("sedge-000-003")
        run.set_experience(2, "003", [_belief(["000", "001", "002", "003"], edges)])
        assert run.apply_space_state() == {
            "ok": True, "prior_revision": 1, "revision": 2,
            "decision_ids": ["sdec-000002"],
        }

        stored = json.loads(run.ledger.read_text())
        assert validate_registry(registry, ledger=stored) == []
        # Pruning changes eligibility only: earlier records stay valid points.
        for item in stored["records"]:
            if selected_assignments(item["semantic_point"]).get("dim-data-curation") == (
                "hyp-data-filtered"
            ):
                assert validate_point(item["semantic_point"], registry) == []

        # Proposals track the overlay, and a stale receipt cannot slip through.
        proposals = run.propose()
        assert proposals["search_space_state_revision"] == 2
        assert proposals["proposals"]
        assert not any(selects_target(item) for item in proposals["proposals"])
        run.stale_add_record_is_refused("004", target, state_revision=0)

        # `target-evidence` is the extractor's bounded source: it recovers the
        # target's own comparators and its counts match the validator exactly.
        block, = run.target_evidence("hyp-data-filtered")["hypothesis_targets"]
        assert set(edges) <= set(block["evidence_edge_ids"])
        assert block["comparator_coverage"] == comparator_coverage(
            stored,
            block["evidence_edge_ids"],
            target_kind="hypothesis",
            target_id="hyp-data-filtered",
        )

        # A later matched control that removes the target contradicts the
        # earlier signal and reopens it; inner re-tuning never would.
        run.add_record("004", "improve", ["003"], baseline, state_revision=2)
        run.record_run("004", 0.70)
        run.set_experience(
            3,
            "004",
            [
                _belief(
                    ["000", "001", "002", "003", "004"],
                    [*edges, "sedge-003-004"],
                    assessment="mixed",
                    recommended_status="active",
                    claim="A new matched removal control contradicts the earlier signal.",
                    confidence="med",
                    uncertainty="The new control conflicts with earlier ones.",
                )
            ],
        )
        assert run.apply_space_state() == {
            "ok": True, "prior_revision": 2, "revision": 3,
            "decision_ids": ["sdec-000003"],
        }
        assert any(selects_target(item) for item in run.propose()["proposals"])

        final = json.loads(run.ledger.read_text())
        state = final["search_space_state"]
        assert state["revision"] == len(state["decisions"]) == 3
        assert validate_registry(registry, ledger=final) == []
        assert (
            replay_search_space_state(registry, state)["hypotheses"]["hyp-data-filtered"]
            == "active"
        )


def main() -> int:
    registry = fixture_registry()
    assert validate_registry(registry) == [], validate_registry(registry)

    check_markdown_round_trip(registry)
    check_shape_neutrality()
    check_retrieval_provenance(registry)
    check_round_lifecycle(registry)

    print("Semantic background, shape-neutrality, retrieval, and lifecycle checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
