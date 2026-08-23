from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "tuners"))
sys.path.insert(0, str(ROOT / "tools" / "inner_benchmark"))

import checkpoint as checkpoint_mod  # noqa: E402
import freeze  # noqa: E402
import space as space_mod  # noqa: E402

# Mirrors the production literal formats exactly (tuples included), same as
# tests/test_inner_benchmark_objective.py.
TRAIN_TEMPLATE = '''
PARAM_SCHEMA = {
    "depth": "int",
    "lr": ("float", "log"),
    "dropout": "float",
    "mode": ("categorical", ["fast", "slow"]),
}
SEARCH_SPACE = {
    "depth": ("int", 1, 8),
    "lr": ("float", 0.0001, 0.1, "log"),
    "dropout": ("float", 0.0, 0.5),
    "mode": ("categorical", ["fast", "slow"]),
}
BASE_PARAMS = %r
def make_model(params):
    return dict(params)
'''

# Cheap deterministic score over the make_model stub (same as the objective
# tests): depth * 10 + lr * 100 + dropout + (0 if fast else 1).
PREPARE_SOURCE = '''
def _weighted_sum(model):
    return (
        model["depth"] * 10
        + model["lr"] * 100
        + model["dropout"]
        + (0 if model["mode"] == "fast" else 1)
    )

def evaluate_config(make_model, params):
    return _weighted_sum(make_model(params))

def preflight_config(make_model, params):
    return {"status": "ok"}
'''

CONTROL = {"depth": 2, "lr": 0.002, "dropout": 0.2, "mode": "slow"}
WARM_A = {"depth": 4, "lr": 0.001, "dropout": 0.1, "mode": "fast"}
WARM_B = {"depth": 6, "lr": 0.005, "dropout": 0.3, "mode": "fast"}
DEFERRED_1 = {"depth": 1, "lr": 0.01, "dropout": 0.4, "mode": "slow"}
DEFERRED_2 = {"depth": 8, "lr": 0.02, "dropout": 0.05, "mode": "fast"}
E1 = {"depth": 3, "lr": 0.003, "dropout": 0.15, "mode": "fast"}
E2 = {"depth": 5, "lr": 0.007, "dropout": 0.25, "mode": "slow"}
E3 = {"depth": 7, "lr": 0.008, "dropout": 0.35, "mode": "fast"}
E4 = {"depth": 2, "lr": 0.004, "dropout": 0.45, "mode": "fast"}
REJECTED = {"depth": 1, "lr": 0.09, "dropout": 0.0, "mode": "slow"}
F1 = {"depth": 4, "lr": 0.006, "dropout": 0.2, "mode": "slow"}
F2 = {"depth": 6, "lr": 0.009, "dropout": 0.1, "mode": "slow"}
F3 = {"depth": 8, "lr": 0.0015, "dropout": 0.3, "mode": "slow"}
F4 = {"depth": 3, "lr": 0.02, "dropout": 0.05, "mode": "fast"}


def _warm_row(params, score, role=None):
    row = {"proposed_index": 0, "params": params, "score": score}
    if role is not None:
        row["role"] = role
    return row


def _trial(params, score):
    return {"params": params, "score": score}


def _crash_trial(params):
    return {"params": params, "score": None, "status": "failed"}


def _rejected_trial(params):
    return {"params": params, "score": None, "status": "preflight_rejected"}


def make_report(*, control_score=1.00, bout0_scores=None, bout1_scores=None):
    """Two-bout tune_report mirroring the production shapes (0805+ era).

    Defaults: control 1.00 < warm A 1.20 < warm B 1.10, so the inherited
    control is the Phase-A incumbent; bout 0 improves (0.90 < 1.00), bout 1 improves
    (0.88 < 0.90). bout_trials=4, two deferred extras attempted in bout 0, so
    bout 0's nominal size is 6 and bout 1's is 4.
    """
    if bout0_scores is None:
        bout0_scores = {
            "d1": 1.30,
            "d2": 0.90,
            "e1": 1.15,
            "e2": 1.25,
            "e4": 1.05,
        }
    if bout1_scores is None:
        bout1_scores = {"f1": 0.95, "f2": 0.88, "f3": 1.00, "f4": 0.99}
    warm_rows = [
        _warm_row(CONTROL, control_score, role="inherited_control"),
        _warm_row(WARM_A, 1.20),
        _warm_row(WARM_B, 1.10),
    ]
    best_warm = min(warm_rows, key=lambda r: r["score"])
    bout0_trials = [
        _trial(DEFERRED_1, bout0_scores["d1"]),
        _trial(DEFERRED_2, bout0_scores["d2"]),
        _trial(E1, bout0_scores["e1"]),
        _trial(E2, bout0_scores["e2"]),
        _crash_trial(E3),
        _trial(E4, bout0_scores["e4"]),
        _rejected_trial(REJECTED),
    ]
    bout1_trials = [
        _trial(F1, bout1_scores["f1"]),
        _trial(F2, bout1_scores["f2"]),
        _trial(F3, bout1_scores["f3"]),
        _trial(F4, bout1_scores["f4"]),
    ]
    # The production close fields must equal the global best over every finite
    # warm and Phase-C row, regardless of provenance role.
    candidates = [(best_warm["params"], best_warm["score"])] + [
        (t["params"], t["score"])
        for t in bout0_trials + bout1_trials
        if t["score"] is not None
    ]
    final_params, final_score = min(candidates, key=lambda item: item[1])
    return {
        "phase_a": {
            "status": "ok",
            "warm_start_configs": warm_rows,
            "best_warm_score": best_warm["score"],
            "best_warm_params": best_warm["params"],
            "deferred_configs": [{"params": DEFERRED_1}, {"params": DEFERRED_2}],
            "warm_config_selection": {"selected_indices": [0, 2]},
        },
        "phase_c": {
            "stages": [
                {
                    "method": "bo",
                    "status": "ok",
                    "trials": bout0_trials,
                    "preflight_rejections": 1,
                    "early_stopped": False,
                    "budget_exhausted": False,
                },
                {
                    "method": "bo",
                    "bout_index": 1,
                    "status": "ok",
                    "trials": bout1_trials,
                    "preflight_rejections": 0,
                    "early_stopped": False,
                    "budget_exhausted": False,
                },
            ],
            # Transient state that must never reach a checkpoint.
            "pending_proposals": [{"params": F1}],
            "pending_proposals_bout_index": 2,
        },
        "final_best_score": final_score,
        "final_best_params": final_params,
        "applied_to_base_params": True,
        "last_finalized_stage_index": 1,
    }


def make_run(
    tmp_path,
    *,
    report,
    base_params,
    candidate_id="007",
    records=None,
    inner_policy_id=None,
):
    """Fake source run dir under runs/<task>/<tag>/ with production layout."""
    run_dir = tmp_path / "runs" / "autoresearch-baseline" / "0000-toy-1"
    candidate_dir = run_dir / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "train.py").write_text(TRAIN_TEMPLATE % base_params)
    (candidate_dir / "prepare.py").write_text(PREPARE_SOURCE)
    (candidate_dir / "tune_report.json").write_text(json.dumps(report))
    (candidate_dir / "_candidate_brief.json").write_text(
        json.dumps({"run_id": candidate_id, "op": "improve"})
    )
    tuner = {"bout_trials": 4}
    if inner_policy_id is not None:
        tuner["inner_policy"] = inner_policy_id
    (run_dir / "framework_cfg.json").write_text(
        json.dumps({"tuner": tuner, "per_runtime_limit": 60})
    )
    if records is None:
        records = [
            {"run_id": candidate_id, "op": "improve", "candidate_name": "toy-op"}
        ]
    (run_dir / "ledger.json").write_text(json.dumps({
        "records": records,
        "items": {
            "task_baseline": {
                "schema_version": 1,
                "kind": "observed_metric",
                "metric": "val_bpb",
                "value": 1.2,
                "direction": "minimize",
                "source": {
                    "role": "task_provided_baseline",
                    "run_id": "000",
                    "stage": "screening",
                },
            }
        },
    }))
    return run_dir


@pytest.fixture()
def two_bout_run(tmp_path):
    """2-bout source whose train.py BASE_PARAMS sits at the bout-1 close (F2)."""
    return make_run(tmp_path, report=make_report(), base_params=F2)


def read_base_params(candidate_train: Path) -> dict:
    return space_mod.read_contract(candidate_train).base_params


def origins(history):
    return [row.origin for row in history]


# --- 1. inspect ---------------------------------------------------------------


def test_inspect_two_bout_candidate(two_bout_run):
    rows = freeze.inspect_run(two_bout_run)

    assert len(rows) == 1
    row = rows[0]
    assert row["candidate_id"] == "007"
    assert row["kind"] == "improve"
    assert row["phase_a_ok"] is True
    assert row["completed_bouts"] == 2
    assert [b["improved"] for b in row["bouts"]] == [True, True]
    assert row["bouts"][0]["best_score"] == pytest.approx(0.90)
    assert row["bouts"][0]["objective_rows"] == 6  # preflight_rejected excluded
    assert row["bouts"][0]["nominal"] == 6  # 4 bout_trials + 2 deferred extras
    assert row["bouts"][1]["nominal"] == 4
    assert row["finite_unique_by_boundary"] == [3, 8, 12]
    assert row["eligible_regimes"] == ["initial", "deep"]


def test_inspect_provided_baseline_kind(tmp_path):
    run_dir = make_run(
        tmp_path,
        report=make_report(),
        base_params=F2,
        records=[{"run_id": "007", "op": "fresh", "candidate_name": "provided_baseline"}],
    )

    (row,) = freeze.inspect_run(run_dir)
    assert row["kind"] == "provided-baseline"


def test_inspect_cli_prints_table(two_bout_run, capsys):
    assert freeze.main(["inspect", "--run-dir", str(two_bout_run)]) == 0
    out = capsys.readouterr().out
    assert "007" in out and "initial,deep" in out


def test_source_policy_does_not_change_initial_deep_taxonomy(tmp_path):
    run_dir = make_run(
        tmp_path,
        report=make_report(),
        base_params=F2,
        inner_policy_id="hebo24-hebo20",
    )

    (row,) = freeze.inspect_run(run_dir)
    assert row["eligible_regimes"] == ["initial", "deep"]

    initial_out = tmp_path / "ckpt-initial"
    deep_out = tmp_path / "ckpt-deep"
    freeze.create_checkpoint(run_dir, "007", 0, initial_out)
    freeze.create_checkpoint(run_dir, "007", 1, deep_out)

    initial = checkpoint_mod.load_checkpoint(initial_out)
    deep = checkpoint_mod.load_checkpoint(deep_out)
    assert (initial.regime, initial.stratum) == ("initial", "initial")
    assert (deep.regime, deep.stratum) == ("deep", "deep")


# --- 2. create N=0 (INITIAL regime) --------------------------------------------


def test_create_initial_regime(two_bout_run, tmp_path):
    out = tmp_path / "ckpt-initial"
    summary = freeze.create_checkpoint(two_bout_run, "007", 0, out)

    ckpt = checkpoint_mod.load_checkpoint(out)  # loads cleanly (schema v2)
    assert ckpt.regime == "initial" and ckpt.stratum == "initial"
    # History = phase_a evaluated rows only, control role tag preserved.
    assert len(ckpt.history) == 3
    assert set(origins(ckpt.history)) == {"phase_a"}
    roles = {row.role for row in ckpt.history}
    assert roles == {"inherited_control", None}
    # BASE_PARAMS = phase-a best finite row, including the control when it wins.
    assert read_base_params(ckpt.candidate_path) == CONTROL
    assert summary["base_params_restored"] == CONTROL
    # deferred carried as recorded (no bouts included).
    assert [dict(d) for d in ckpt.deferred_configs] == [DEFERRED_1, DEFERRED_2]
    # The checkpoint incumbent matches production: control 1.00 wins.
    assert ckpt.incumbent.params == CONTROL
    assert ckpt.incumbent_is_inherited_control is True
    # Provisional source scores; nothing re-measured yet.
    assert ckpt.extra["remeasure"]["remeasured_identities"] == []
    assert ckpt.extra["incumbents"]["production"]["params"] == CONTROL
    # Task wiring from the repo task.toml + source framework_cfg.
    assert ckpt.task.score_fn == "evaluate_config"
    assert ckpt.task.preflight_fn == "preflight_config"
    assert ckpt.task.per_runtime_limit == 60
    # The environment-partition pin: task.toml's own [env].project.
    assert ckpt.task.project == "tasks/autoresearch-baseline"
    assert ckpt.task.relative_improvement_over_baseline == pytest.approx(0.075)
    assert ckpt.items["task_baseline"]["value"] == pytest.approx(1.2)
    # Candidate dir holds train.py + prepare.py ONLY.
    assert sorted(p.name for p in (out / "candidate").iterdir()) == [
        "prepare.py",
        "train.py",
    ]


# --- 3. create N=1 on a 2-bout source -------------------------------------------


def test_create_deep_restores_boundary_base_params(two_bout_run, tmp_path):
    out = tmp_path / "ckpt-deep"
    freeze.create_checkpoint(two_bout_run, "007", 1, out)

    ckpt = checkpoint_mod.load_checkpoint(out)
    assert ckpt.regime == "deep" and ckpt.stratum == "deep"
    # BASE_PARAMS restored to the bout-1 applied value (global best after bout
    # 0 = DEFERRED_2 at 0.90), NOT the source file's bout-2 value (F2 0.88).
    assert read_base_params(two_bout_run / "candidates" / "007" / "train.py") == F2
    assert read_base_params(ckpt.candidate_path) == DEFERRED_2
    # History: 3 phase_a rows + 6 bout_0 objective rows (crash kept, preflight
    # rejected excluded); no bout_1 rows.
    assert len(ckpt.history) == 9
    assert origins(ckpt.history).count("phase_a") == 3
    assert origins(ckpt.history).count("bout_0") == 6
    crash = [row for row in ckpt.history if row.status == "crash"]
    assert len(crash) == 1 and crash[0].params == E3 and crash[0].score is None
    assert all(row.params != REJECTED for row in ckpt.history)
    # Attempted deferred configs removed.
    assert list(ckpt.deferred_configs) == []
    # Transient state never carried: no pending_proposals anywhere in output.
    assert "pending_proposals" not in (out / "checkpoint.json").read_text()
    # Source run dir untouched (read-only contract).
    report = json.loads(
        (two_bout_run / "candidates" / "007" / "tune_report.json").read_text()
    )
    assert report["phase_c"]["pending_proposals"] == [{"params": F1}]


# --- 4. inherited_control incumbent guard ---------------------------------------


def test_create_inherited_control_incumbent(tmp_path):
    # Control row is the global best (0.80 < everything else).
    run_dir = make_run(
        tmp_path, report=make_report(control_score=0.80), base_params=F2
    )
    ckpt = checkpoint_mod.load_checkpoint(
        freeze.create_checkpoint(run_dir, "007", 0, tmp_path / "ckpt")["out"]
    )
    assert ckpt.incumbent.params == CONTROL
    assert ckpt.incumbent.score == pytest.approx(0.80)
    assert ckpt.incumbent_is_inherited_control is True
    assert "inherited_control_excluded" not in ckpt.extra
    assert read_base_params(ckpt.candidate_path) == CONTROL


def test_create_inherited_control_out_of_space_is_excluded(tmp_path):
    control_oob = dict(CONTROL, depth=99)  # outside ("int", 1, 8)
    report = make_report(control_score=0.80)
    report["phase_a"]["warm_start_configs"][0]["params"] = control_oob
    report["phase_a"]["best_warm_params"] = control_oob
    report["final_best_params"] = control_oob
    run_dir = make_run(tmp_path, report=report, base_params=F2)

    out = tmp_path / "ckpt"
    freeze.create_checkpoint(run_dir, "007", 0, out)
    ckpt = checkpoint_mod.load_checkpoint(out)
    # Excluded from the benchmark incumbent: next best finite row wins.
    assert ckpt.incumbent.params == WARM_B
    assert ckpt.incumbent_is_inherited_control is False
    excluded = ckpt.extra["inherited_control_excluded"]
    assert "SEARCH_SPACE" in excluded["reason"]
    assert excluded["rows"][0]["params"] == control_oob
    assert excluded["rows"][0]["violations"][0]["key"] == "depth"
    # The excluded row stays in history as an evaluated observation.
    assert any(row.params == control_oob for row in ckpt.history)


# --- 5. stratum -------------------------------------------------------------------


def test_deep_classification_does_not_depend_on_following_segment(tmp_path):
    # At bouts=1, bout 0 is inside the checkpoint and bout 1 is only factual
    # future context. Its outcome must not create a CONTINUE subtype.
    improved_run = make_run(tmp_path / "a", report=make_report(), base_params=F2)
    ckpt = checkpoint_mod.load_checkpoint(
        freeze.create_checkpoint(improved_run, "007", 1, tmp_path / "a" / "ckpt")["out"]
    )
    assert (ckpt.regime, ckpt.stratum) == ("deep", "deep")
    assert ckpt.extra["last_bout"]["improved_production"] is True
    assert ckpt.extra["bout_after_boundary"]["bout_index"] == 1
    assert ckpt.extra["bout_after_boundary"]["improved_production"] is True

    # Variant: bout 0 improves but bout 1 never beats it (all rows >= 0.90).
    report = make_report(bout1_scores={"f1": 0.95, "f2": 0.99, "f3": 1.00, "f4": 0.90})
    flat_run = make_run(tmp_path / "b", report=report, base_params=DEFERRED_2)
    ckpt = checkpoint_mod.load_checkpoint(
        freeze.create_checkpoint(flat_run, "007", 1, tmp_path / "b" / "ckpt")["out"]
    )
    assert (ckpt.regime, ckpt.stratum) == ("deep", "deep")
    assert ckpt.extra["last_bout"]["improved_production"] is True  # bout 0 did
    after = ckpt.extra["bout_after_boundary"]
    assert after["improved_production"] is False
    assert after["best_score"] == pytest.approx(0.90)
    assert after["starting_incumbent_score_production"] == pytest.approx(0.90)
    # Boundary BASE_PARAMS is the bout-0 incumbent, unaffected by bout 1.
    assert read_base_params(ckpt.candidate_path) == DEFERRED_2


def test_create_deep_needs_no_following_segment(tmp_path):
    # One-bout run: the boundary after INITIAL is already a valid DEEP start.
    report = make_report()
    report["phase_c"]["stages"] = report["phase_c"]["stages"][:1]
    report["last_finalized_stage_index"] = 0
    # The close fields must match the surviving global best (bout 0's 0.90).
    report["final_best_params"] = DEFERRED_2
    report["final_best_score"] = 0.90
    run_dir = make_run(tmp_path, report=report, base_params=DEFERRED_2)

    deep = checkpoint_mod.load_checkpoint(
        freeze.create_checkpoint(run_dir, "007", 1, tmp_path / "ckpt")["out"]
    )
    assert (deep.regime, deep.stratum) == ("deep", "deep")
    freeze.create_checkpoint(run_dir, "007", 0, tmp_path / "ckpt-initial")


def test_create_deep_regime(two_bout_run, tmp_path):
    out = tmp_path / "ckpt-deep"
    freeze.create_checkpoint(two_bout_run, "007", 2, out)
    ckpt = checkpoint_mod.load_checkpoint(out)
    assert ckpt.regime == "deep" and ckpt.stratum == "deep"
    assert origins(ckpt.history).count("bout_1") == 4
    # Last-bout improvement evidence preserved in extra for analysis.
    assert ckpt.extra["last_bout"]["bout_index"] == 1
    assert ckpt.extra["last_bout"]["improved_production"] is True


# --- 6. nonexistent boundary -------------------------------------------------------


def test_create_refuses_nonexistent_boundary(two_bout_run, tmp_path):
    with pytest.raises(ValueError, match="boundary after 3 completed bout"):
        freeze.create_checkpoint(two_bout_run, "007", 3, tmp_path / "ckpt")


def test_create_accepts_bout_short_of_nominal_rows(tmp_path):
    # A bout can consume n_trials slots without producing objective rows
    # (preflight rejection / exact duplicate early returns, patience stop).
    # Production calls such a bout finished, so freeze must too — the row
    # shortfall is reported descriptively, not as an incompleteness.
    report = make_report()
    report["phase_c"]["stages"][1]["trials"] = report["phase_c"]["stages"][1]["trials"][:3]
    run_dir = make_run(tmp_path, report=report, base_params=F2)

    freeze.create_checkpoint(run_dir, "007", 2, tmp_path / "ckpt")
    row = freeze.inspect_run(run_dir)[0]
    assert row["completed_bouts"] == 2
    assert row["bouts"][1]["objective_rows"] == 3
    assert row["bouts"][1]["rows_below_nominal"] == 1


def test_create_refuses_nonempty_out_dir(two_bout_run, tmp_path):
    out = tmp_path / "ckpt"
    freeze.create_checkpoint(two_bout_run, "007", 0, out)
    with pytest.raises(ValueError, match="non-empty out dir"):
        freeze.create_checkpoint(two_bout_run, "007", 0, out)


def test_create_refuses_deep_below_warmup(tmp_path):
    report = make_report()
    # Crash four of six bout-0 rows: 3 warm + e4 = 4 finite unique < WARMUP=8.
    trials = report["phase_c"]["stages"][0]["trials"]
    for index in (0, 1, 2, 3):
        trials[index] = _crash_trial(trials[index]["params"])
    run_dir = make_run(tmp_path, report=report, base_params=F2)

    with pytest.raises(ValueError, match="WARMUP"):
        freeze.create_checkpoint(run_dir, "007", 1, tmp_path / "ckpt")
    # INITIAL has no WARMUP guard.
    freeze.create_checkpoint(run_dir, "007", 0, tmp_path / "ckpt-initial")


# --- 7. remeasure -----------------------------------------------------------------


def _fake_eval(scores, crashes=(), calls=None):
    """eval_fn seam: local scores keyed by params json; crashes by key."""

    def key(params):
        return json.dumps(params, sort_keys=True)

    def eval_fn(params):
        if calls is not None:
            calls.append(params)
        k = key(params)
        if k in {key(dict(p)) for p in crashes}:
            return SimpleNamespace(status="crash", score=None, detail="boom")
        return SimpleNamespace(status="ok", score=scores[k], detail=None)

    return eval_fn


def _local_scores():
    # Local scores that FLIP the incumbent: CONTROL (inherited_control) becomes
    # best locally, while the source benchmark incumbent is DEFERRED_2 (0.90).
    table = [
        (CONTROL, 2.0),
        (E4, 2.5),
        (E1, 3.0),
        (WARM_A, 4.0),
        (DEFERRED_1, 5.0),
        (WARM_B, 6.0),
        (E2, 7.0),
        (DEFERRED_2, 8.0),
        (E3, 7.0),  # only used when not crashed
    ]
    return {json.dumps(params, sort_keys=True): score for params, score in table}


def test_remeasure_updates_scores_and_flips_incumbent(two_bout_run, tmp_path):
    out = tmp_path / "ckpt"
    freeze.create_checkpoint(two_bout_run, "007", 1, out)
    before = checkpoint_mod.load_checkpoint(out)
    assert before.incumbent.params == DEFERRED_2  # source benchmark incumbent

    summary = freeze.remeasure_checkpoint(out, eval_fn=_fake_eval(_local_scores(), crashes=[E3]))

    assert summary["evaluated"] == 9  # unique configs incl. the crashed row
    assert summary["skipped"] == 0
    assert summary["crashed"] == 1
    assert summary["complete"] is True
    assert summary["verdict"] == "ok"
    ckpt = checkpoint_mod.load_checkpoint(out)
    by_key = {json.dumps(dict(r.params), sort_keys=True): r for r in ckpt.history}
    assert by_key[json.dumps(WARM_A, sort_keys=True)].score == pytest.approx(4.0)
    crash_row = by_key[json.dumps(E3, sort_keys=True)]
    assert crash_row.status == "crash" and crash_row.score is None
    # Incumbent recomputed from LOCAL scores: control (2.0) flips vs source.
    assert ckpt.incumbent.params == CONTROL
    assert ckpt.incumbent.score == pytest.approx(2.0)
    assert ckpt.incumbent_is_inherited_control is True
    assert summary["incumbent_before"] == pytest.approx(0.90)
    assert summary["incumbent_after"] == pytest.approx(2.0)
    state = ckpt.extra["remeasure"]
    assert state["complete"] is True and state["valid"] is True
    assert state["finite_unique_local"] == 8  # 9 unique - 1 local crash


def test_remeasure_is_idempotent(two_bout_run, tmp_path):
    out = tmp_path / "ckpt"
    freeze.create_checkpoint(two_bout_run, "007", 1, out)
    freeze.remeasure_checkpoint(out, eval_fn=_fake_eval(_local_scores()))

    def forbidden(params):  # a second run must evaluate nothing
        raise AssertionError("eval_fn called on an already-measured config")

    summary = freeze.remeasure_checkpoint(out, eval_fn=forbidden)
    assert summary["evaluated"] == 0
    assert summary["skipped"] == 9
    assert summary["verdict"] == "ok"


def test_remeasure_eval_limit_stops_and_resumes(two_bout_run, tmp_path):
    out = tmp_path / "ckpt"
    freeze.create_checkpoint(two_bout_run, "007", 1, out)
    calls = []
    eval_fn = _fake_eval(_local_scores(), calls=calls)

    first = freeze.remeasure_checkpoint(out, eval_fn=eval_fn, eval_limit=4)
    assert first["evaluated"] == 4 and first["remaining"] == 5
    assert first["complete"] is False and first["verdict"] == "incomplete"
    # Persisted mid-run state still loads through the Task-3 loader.
    checkpoint_mod.load_checkpoint(out)

    second = freeze.remeasure_checkpoint(out, eval_fn=eval_fn, eval_limit=100)
    assert second["evaluated"] == 5 and second["complete"] is True
    assert len(calls) == 9  # every unique config evaluated exactly once
    assert second["verdict"] == "ok"


def test_remeasure_warmup_guard_invalidates_deep(two_bout_run, tmp_path):
    out = tmp_path / "ckpt"
    freeze.create_checkpoint(two_bout_run, "007", 1, out)
    # Crash three configs locally: finite unique drops to 6 < WARMUP=8.
    summary = freeze.remeasure_checkpoint(
        out, eval_fn=_fake_eval(_local_scores(), crashes=[E3, DEFERRED_2, E2])
    )

    assert summary["verdict"] == "invalid"
    assert "WARMUP" in summary["invalid_reason"]
    ckpt = checkpoint_mod.load_checkpoint(out)  # still loadable for inspection
    assert ckpt.extra["remeasure"]["valid"] is False
    assert ckpt.extra["remeasure"]["finite_unique_local"] == 6
    assert ckpt.incumbent.score == pytest.approx(2.0)  # local, finite


def test_remeasure_initial_regime_has_no_warmup_guard(two_bout_run, tmp_path):
    out = tmp_path / "ckpt"
    freeze.create_checkpoint(two_bout_run, "007", 0, out)
    summary = freeze.remeasure_checkpoint(out, eval_fn=_fake_eval(_local_scores()))

    assert summary["finite_unique_local"] == 3  # below WARMUP...
    assert summary["verdict"] == "ok"  # ...but INITIAL has no guard


def test_remeasure_default_eval_fn_runs_objective(two_bout_run, tmp_path):
    """CLI default path: real objective.evaluate subprocesses on the toy task."""
    out = tmp_path / "ckpt"
    freeze.create_checkpoint(two_bout_run, "007", 0, out)

    summary = freeze.remeasure_checkpoint(out)  # no eval_fn injection

    assert summary["evaluated"] == 3 and summary["verdict"] == "ok"
    ckpt = checkpoint_mod.load_checkpoint(out)
    expected = {
        json.dumps(p, sort_keys=True): p["depth"] * 10 + p["lr"] * 100 + p["dropout"] + (0 if p["mode"] == "fast" else 1)
        for p in (CONTROL, WARM_A, WARM_B)
    }
    for row in ckpt.history:
        assert row.score == pytest.approx(expected[json.dumps(dict(row.params), sort_keys=True)])
    # CONTROL: 21.4 < WARM_A 40.2 < WARM_B 60.8.
    assert ckpt.incumbent.params == CONTROL
    assert ckpt.incumbent.score == pytest.approx(21.4)
