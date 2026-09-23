import copy
import json

import pytest

from tests.test_slate import _lane, _proposal, _pset
from tests.fixtures import fixture_registry
import slate
from semantic_space import complete_point, space_receipt


def test_frozen_generation_replays_after_space_publication(tmp_path):
    from tests.test_slate_replay import build_admitted_generation, run_replay
    from tests.test_space_expansion import initial_run, synthesis
    from semantic_space import space_revision
    from space_revisions import apply_expansion, publish_expansion

    build_admitted_generation(tmp_path)
    background, old = initial_run(tmp_path)
    delta = {"hypotheses": {"dim-model-architecture": [synthesis()]}}
    new, _ = apply_expansion(old, delta)
    review = {
        "decision": "expand", "reason": "Current routes appear insufficient for the goal",
        "basis": ["background.md"], "base_revision": space_revision(old), "delta": delta,
        "probe": {"point": complete_point(new, {"dim-model-architecture": "hyp-model-transfer"}),
                  "op": "fresh", "parents": [], "implementation_seconds": 20,
                  "screening_seconds": 30, "expected_observation": "A valid transfer score"},
    }
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    publish_expansion(background, review, ledger=ledger, admission={"reservation_id": "r-1"})
    code, report = run_replay(tmp_path)
    assert code == 0, report
    from types import SimpleNamespace
    with pytest.raises(slate.ContractError, match="frozen"):
        slate.cmd_construct(SimpleNamespace(pool_output=tmp_path / ".semantic/gen-0001/pool.json"))
    approved = {"space": space_receipt(new), "review": review, "probe": review["probe"],
                "admission": {"reservation_id": "r-1"}}
    assert slate.validate_approved_probe_binding(tmp_path, approved) == []
    approved["admission"]["reservation_id"] = "forged"
    assert slate.validate_approved_probe_binding(tmp_path, approved)


def test_published_probe_constructs_admits_and_replays(tmp_path):
    from tests.test_slate_replay import build_admitted_generation, run_replay
    from tests.test_space_expansion import initial_run, synthesis
    from semantic_space import space_revision
    from space_revisions import apply_expansion, publish_expansion

    def publish(run_dir, ledger):
        background, old = initial_run(run_dir)
        delta = {"hypotheses": {"dim-model-architecture": [synthesis()]}}
        new, _ = apply_expansion(old, delta)
        review = {
            "decision": "expand", "reason": "Current routes appear insufficient for the goal",
            "basis": ["background.md"], "base_revision": space_revision(old), "delta": delta,
            "probe": {"point": complete_point(new, {"dim-model-architecture": "hyp-model-transfer"}),
                      "op": "improve", "parents": ["000"], "implementation_seconds": 20,
                      "screening_seconds": 30, "expected_observation": "A valid transfer score"},
        }
        publish_expansion(background, review, ledger=ledger, admission={"reservation_id": "r-1"})
        ledger["search_space"] = space_receipt(new)

    manifest = build_admitted_generation(tmp_path, prepare_run=publish)
    probe = manifest["slate"][0]
    assert probe["seat_type"] == "space_probe"
    assert probe["carrier"]["op"] == "improve"
    assert probe["carrier"]["parents"] == ["000"]
    assert len(manifest["slate"]) == 2
    code, report = run_replay(tmp_path)
    assert code == 0, report


@pytest.mark.parametrize("ordinary_contains_probe", [False, True])
def test_approved_probe_keeps_carrier_and_seat_despite_last_rank(ordinary_contains_probe):
    registry = fixture_registry()
    points = [complete_point(registry, choices) for choices in (
        {}, {"dim-data-curation": "hyp-data-filtered"},
        {"dim-validation-selection": "hyp-valid-cv"},
    )]
    probe = {"op": "fresh", "parents": [], "point": points[-1]}
    approved = {"probe": probe, "space": space_receipt(registry),
                "review": {"decision": "expand", "probe": probe},
                "admission": {"reservation_id": "review-1"}}
    pool = slate.build_pool(
        [_lane("lane-00", "fresh", [], None)],
        {"lane-00": _pset("ordinary", [_proposal(p, 1.0) for p in
                                       (points if ordinary_contains_probe else points[:-1])])},
        registry, 3, approved_probe=approved,
    )
    pool.update(budget={"admission_cap": 2}, approved_probe=approved,
                generation_seed="seed", gen_no=1, ledger_snapshot={}, pool_size=3,
                lanes_digest="lanes", pool_digest="pool")
    target = next(e for e in pool["pool"] if e["point_id"] == points[-1]["point_id"])
    assert len(pool["pool"]) == 3
    assert target["carrier"]["lane_id"] == "space-probe"
    ranking = [e["label"] for e in pool["pool"] if e is not target] + [target["label"]]
    stages = {stage: {"stage": stage, "gen_no": 1, "status": "valid", "ranking": ranking,
                      "pool_digest": "pool", "context_digest": "context",
                      "presented_order": slate.presented_order(pool, "seed", stage)}
              for stage in slate.REGULAR_STAGES}
    decision, called = slate.decide_aggregation(pool, stages, "context")
    assert decision["slate"] == [target["label"], ranking[0]]
    judge = {"aggregation": decision, "context_digest": "context"}
    manifest = slate.build_manifest(pool, judge, pool["budget"], ["005", "006"])
    assert manifest["slate"][0]["seat_type"] == "space_probe"
    assert manifest["slate"][0]["space_probe_binding"]["admission"] == approved["admission"]
    assert "space_probe_binding" not in target["summary"]
    broken = copy.deepcopy(pool)
    broken["budget"]["admission_cap"] = 1
    with pytest.raises(slate.ContractError, match="two"):
        slate.decide_aggregation(broken, stages, "context")
