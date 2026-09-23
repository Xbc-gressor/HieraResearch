from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from background_contract import load_registry, validate_registry
from semantic_space import (complete_point, coverage_from_records, derive_semantic_lineage,
                            load_catalog, point_diff, space_receipt, space_revision)
from tests.fixtures import background_text, fixture_registry, hypothesis, dimension, retrieval_hit_manifest


def synthesis():
    value = hypothesis("hyp-model-transfer", "transferred representation",
                       kind="synthesis_probe", intervention="transfer")
    value.update(evidence=[], literature_credibility="unverified",
                 provenance=[{"kind": "agent_synthesis", "ref": "Runtime residual pattern"}])
    return value


def initial_run(tmp_path):
    registry = fixture_registry()
    background = tmp_path / "background.md"
    background.write_text(background_text(registry))
    (tmp_path / "background_retrieval.json").write_text(json.dumps(retrieval_hit_manifest()))
    return background, registry


def test_append_only_expansion_preserves_history_and_parent_choice(tmp_path):
    import space_revisions as revisions
    from semantic_search import build_proposal_set
    background, old = initial_run(tmp_path)
    point = complete_point(old)
    ledger = {"records": [{"run_id": "000", "semantic_point": point, "status": "keep",
                           "source_run_ids": [], "final_best_score": .4}]}
    delta = {"hypotheses": {"dim-model-architecture": [synthesis()]}}
    new, catalog = revisions.apply_expansion(old, delta)
    assert validate_registry(new) == []
    history = {space_revision(old): old}
    coverage = coverage_from_records(new, ledger["records"], registry_history=history)
    assert coverage["n_valid_records"] == 1
    assert coverage["invalid_records"] == []
    assert derive_semantic_lineage(new, ledger, registry_history=history)["runs"][0]["run_id"] == "000"
    proposals = build_proposal_set(new, ledger, op="improve", parents=["000"], registry_history=history)
    assert any(p["point"]["point_id"] != point["point_id"] for p in proposals["proposals"])
    assert ledger["records"][0]["semantic_point"] == point
    assert space_receipt(old) != space_receipt(new)
    assert catalog == load_catalog()


def test_new_dimension_is_undeclared_in_history_and_not_a_direct_change():
    import space_revisions as revisions
    from semantic_evidence import build_semantic_edges
    old = fixture_registry()
    catalog = load_catalog()
    new_id = next(d["id"] for d in catalog["dimensions"] if d["id"] not in {d["id"] for d in old["dimensions"]})
    new_dim = dimension(catalog, new_id, hypothesis("hyp-extra-baseline", "identity",
                        kind="baseline", intervention="identity-extra"), mode="baseline_only")
    new, _ = revisions.apply_expansion(old, {"dimensions": [new_dim]})
    parent = {"run_id": "000", "semantic_point": complete_point(old)}
    child = {"run_id": "001", "source_run_ids": ["000"], "semantic_point": complete_point(new)}
    changes = point_diff(parent["semantic_point"], child["semantic_point"])
    assert changes[0]["operation"] == "dimension_declared"
    assert build_semantic_edges([parent], child)[0]["change_class"] == "space_extension"
    coverage = coverage_from_records(new, [parent], registry_history={space_revision(old): old})
    assert next(d for d in coverage["dimensions"] if d["dimension_id"] == new_id)["hypotheses"][0]["count"] == 0


def test_expansion_rejects_old_meaning_changes_and_catalog_escape():
    import space_revisions as revisions
    old = fixture_registry()
    with pytest.raises(ValueError):
        revisions.apply_expansion(old, {"dimensions": [copy.deepcopy(old["dimensions"][0])]})
    with pytest.raises(ValueError):
        revisions.apply_expansion(old, {"catalog_dimensions": [{"id": "dim-unknown"}]})
    bad_relation = copy.deepcopy(old["relations"][0])
    bad_relation["id"] = "rel-retroactive"
    with pytest.raises(ValueError):
        revisions.apply_expansion(old, {"relations": [bad_relation]})


def test_publish_binds_admission_review_and_probe_without_editing_background(tmp_path):
    import space_revisions as revisions
    background, old = initial_run(tmp_path)
    original = background.read_bytes()
    delta = {"hypotheses": {"dim-model-architecture": [synthesis()]}}
    new, _ = revisions.apply_expansion(old, delta)
    review = {
        "decision": "expand", "reason": "Current routes appear insufficient for the goal",
        "basis": ["background.md"], "base_revision": space_revision(old), "delta": delta,
        "probe": {"point": complete_point(new, {"dim-model-architecture": "hyp-model-transfer"}),
                  "op": "fresh", "parents": [], "implementation_seconds": 20,
                  "screening_seconds": 30, "expected_observation": "A valid transfer score"},
    }
    with pytest.raises(ValueError):
        revisions.publish_expansion(background, review, ledger={"records": []}, admission={})
    assert load_registry(background) == old
    receipt = revisions.publish_expansion(background, review, ledger={"records": []},
                                         admission={"reservation_id": "r-1"})
    assert receipt["space"] == space_receipt(new)
    assert receipt["pending_probe"]["op"] == "fresh"
    assert load_registry(background) == new
    assert revisions.load_registry_history(background)[space_revision(old)] == old
    assert background.read_bytes() == original
    with pytest.raises(ValueError):
        revisions.publish_expansion(background, review, ledger={"records": []},
                                     admission={"reservation_id": "r-2"})

    next_hypothesis = synthesis()
    next_hypothesis["id"] = "hyp-model-distilled"
    next_hypothesis["scope"]["interventions"] = ["distillation"]
    next_delta = {"hypotheses": {"dim-model-architecture": [next_hypothesis]}}
    next_registry, _ = revisions.apply_expansion(new, next_delta)
    next_review = dict(review, base_revision=space_revision(new), delta=next_delta,
                       probe=dict(review["probe"], point=complete_point(next_registry,
                                  {"dim-model-architecture": "hyp-model-distilled"})))
    ledger = {"records": [{"run_id": "000", "semantic_point": review["probe"]["point"],
                           "status": "pending"}]}
    with pytest.raises(ValueError, match="previous probe is still pending"):
        revisions.publish_expansion(background, next_review, ledger=ledger,
                                     admission={"reservation_id": "r-2"})
    ledger["records"][0]["status"] = "aborted"
    second = revisions.publish_expansion(background, next_review, ledger=ledger,
                                          admission={"reservation_id": "r-2"})
    assert second["space"] == space_receipt(next_registry)
    assert load_registry(background) == next_registry
    assert len(revisions.load_registry_history(background)) == 3


def test_review_material_includes_unregistered_research_failures_and_live_experience(tmp_path):
    from space_review import build_review_material, validate_retrieval_request
    background, registry = initial_run(tmp_path)
    background.write_text(background.read_text().replace("## Search space registry",
        "## Deferred research\nA spectral transfer route was not registered.\n\n## Search space registry"))
    ledger = {"records": [{"run_id": str(i), "status": "crash" if i % 2 else "keep",
                           "semantic_point": complete_point(registry), "final_best_score": None if i % 2 else .4}
                          for i in range(30)],
              "experience": {"lessons": [{"claim": "Check compute before discarding transfer", "evidence": ["1"]}]}}
    (tmp_path / "evaluation_attempts.jsonl").write_text(json.dumps({
        "kind": "score_completion", "run_id": "29", "duration_seconds": 12.5,
    }) + "\n")
    packet = build_review_material(background, ledger=ledger, limit=4)
    assert "spectral transfer" in json.dumps(packet["research"])
    assert packet["search"]["total_records"] == 30
    assert packet["search"]["status_counts"] == {"keep": 15, "crash": 15}
    assert len(packet["search"]["recent"]) == 4
    assert packet["coverage"]["n_valid_records"] == 30
    assert packet["experience"]["lessons"][0]["evidence"] == ["1"]
    assert packet["execution"]["completed_eval_seconds"] == 12.5
    assert validate_retrieval_request({"knowledge_gap": "Need adaptation cost",
         "decision_impact": "Determines whether the probe fits", "queries": ["generic transfer inference cost"]}) == []
    assert validate_retrieval_request({"queries": ["anything"]})


def test_ledger_validation_uses_each_records_registry_and_keeps_runtime_status():
    from background_contract import validate_ledger
    from search_space_state import empty_search_space_state, replay_search_space_state
    from space_revisions import apply_expansion
    from tests.fixtures import record
    old = fixture_registry()
    parent = record("000", "fresh", [], complete_point(old), score=.4, status="keep")
    state = empty_search_space_state()
    ledger = {"records": [parent], "search_space": space_receipt(old), "search_space_state": state}
    assert validate_ledger(old, ledger) == []
    new, _ = apply_expansion(old, {"hypotheses": {"dim-model-architecture": [synthesis()]}})
    ledger["search_space"] = space_receipt(new)  # in-memory fixture; production writes belong to ledger.py
    child = record("001", "improve", ["000"], complete_point(new, {"dim-model-architecture": "hyp-model-transfer"}),
                   score=.3, status="keep", prior_records=[parent])
    ledger["records"].append(child)
    assert validate_ledger(new, ledger, registry_history={space_revision(old): old}) == []
    assert validate_ledger(new, ledger)  # missing snapshots must not silently drop or rebase history
    state["revision"] = 1
    state["decisions"] = [{"revision": 1, "target": {"kind": "hypothesis", "id": "hyp-data-filtered"},
                            "to_status": "deprioritized"}]
    runtime = replay_search_space_state(new, state)
    assert runtime["hypotheses"]["hyp-data-filtered"] == "deprioritized"
    assert runtime["hypotheses"]["hyp-model-transfer"] == "active"


def test_induced_catalog_expands_together_with_registry():
    from semantic_space import catalog_receipt
    from space_revisions import apply_expansion
    old = fixture_registry()
    catalog = copy.deepcopy(load_catalog())
    used = {d["id"] for d in old["dimensions"]}
    catalog["dimensions"] = [d for d in catalog["dimensions"] if d["id"] in used]
    # Induced catalog order is the same as the registry order.
    by_id = {d["id"]: d for d in catalog["dimensions"]}
    catalog["dimensions"] = [by_id[d["id"]] for d in old["dimensions"]]
    old["catalog"] = catalog_receipt(catalog)
    addition = {"id": "dim-new-representation", "definition": "Representation family", "boundary": "Feature representation only"}
    next_catalog = copy.deepcopy(catalog)
    next_catalog["dimensions"].append(addition)
    new_dim = dimension(next_catalog, addition["id"], hypothesis("hyp-representation-identity", "identity",
                        kind="baseline", intervention="identity-representation"), mode="baseline_only")
    new, resolved = apply_expansion(old, {"catalog_dimensions": [addition], "dimensions": [new_dim]},
                                    catalog=catalog, dimension_strategy="llm_induced")
    assert len(new["dimensions"]) == len(old["dimensions"]) + 1
    assert validate_registry(new, catalog=resolved, dimension_strategy="llm_induced") == []


def test_source_free_synthesis_numbers_are_not_literature_claims(tmp_path):
    from space_revisions import apply_expansion, publish_expansion
    from space_review import validate_review
    background, registry = initial_run(tmp_path)
    route=synthesis()
    route.update(claim='Hypothesis: mask 50% of representation inputs',
                 credibility_rationale='Unverified synthesis; run loss remains at 0.4',
                 reopen_when='Reconsider when loss falls below 0.3')
    delta={'hypotheses':{'dim-model-architecture':[route]}}
    expanded,_=apply_expansion(registry,delta)
    review={'decision':'expand','reason':'Test a quantitative synthesized mechanism',
        'basis':['run observations'], 'base_revision':space_revision(registry), 'delta':delta,
        'probe':{'point':complete_point(expanded,{'dim-model-architecture':route['id']}),
            'op':'fresh','parents':[],'implementation_seconds':10,'screening_seconds':20,
            'expected_observation':'A finite score'}}
    ledger={'records':[]}
    assert validate_review(review,registry,ledger=ledger)==[]
    publish_expansion(background,review,ledger=ledger,admission={'reservation_id':'reserved'})
    # Adding literature support restores the existing source-number contract.
    cited=copy.deepcopy(expanded)
    target=next(d for d in cited['dimensions'] if d['id']=='dim-model-architecture')
    target['hypotheses'][-1]['evidence']=copy.deepcopy(next(h['evidence'] for d in registry['dimensions'] for h in d['hypotheses'] if h['evidence']))
    errors=validate_registry(cited,retrieval_manifest=json.loads((tmp_path/'background_retrieval.json').read_text()),
        manifest_dir=tmp_path,manifest_path=tmp_path/'background_retrieval.json',number_gate=True)
    assert any('number 50%' in error for error in errors)
