from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driver.jobs import DriverJobError, build_driver_job  # noqa: E402
from driver.loops.experiment import _invoke_with_driver_jobs  # noqa: E402
from driver.receipts import ReceiptStore  # noqa: E402
from driver.roles import InvocationContext  # noqa: E402
from driver.session import FakeSessionRunner  # noqa: E402


def _fixture(tmp_path: Path) -> tuple[Path, InvocationContext]:
    repo = tmp_path / "repo"
    task = repo / "tasks" / "toy"
    task.mkdir(parents=True)
    (task / "task.toml").write_text(
        '[env]\ntype = "uv"\nproject = "tasks/toy"\n'
    )
    run_dir = repo / "runs" / "toy" / "r1"
    candidate = run_dir / "candidates" / "007"
    candidate.mkdir(parents=True)
    (candidate / "train.py").write_text("# candidate\n")
    (candidate / "_warm_configs.json").write_text('[{"x": 1}, {"x": 2}]')
    (candidate / "tune_report.json").write_text(json.dumps({"phase_a": {}}))
    ctx = InvocationContext(
        task="toy", tag="r1", run_dir=run_dir,
        invocation_id=3, run_id="007",
    )
    return repo, ctx


def test_warmstart_job_argv_is_foreground_and_driver_derived(tmp_path: Path) -> None:
    repo, ctx = _fixture(tmp_path)
    argv, log, run_id = build_driver_job(
        "tunable-contract-extractor",
        ctx,
        {"kind": "warmstart", "run_id": "007", "k_eval": 2},
        repo_root=repo,
    )
    joined = " ".join(argv)
    assert run_id == "007" and log.name == "_warmstart.log"
    assert "warmstart_eval.py" in joined and "--k-eval 2" in joined
    assert "nohup" not in joined and "&" not in joined
    # No donor binding in the invocation (old policies / no_donor) -> no flag.
    assert "--donor-snapshot" not in argv


def test_warmstart_job_passes_the_bound_donor_snapshot(tmp_path: Path) -> None:
    repo, ctx = _fixture(tmp_path)
    ctx.extra["donor_binding"] = "bound"
    ctx.extra["donor_snapshot"] = str(
        ctx.run_dir / ".scheduler" / "donors" / "donor-abc123.json"
    )
    argv, _, _ = build_driver_job(
        "tunable-contract-extractor",
        ctx,
        {"kind": "warmstart", "run_id": "007", "k_eval": 2},
        repo_root=repo,
    )
    assert "--donor-snapshot" in argv
    assert argv[argv.index("--donor-snapshot") + 1] == ctx.extra["donor_snapshot"]


def test_warmstart_job_no_donor_binding_adds_no_flag(tmp_path: Path) -> None:
    repo, ctx = _fixture(tmp_path)
    ctx.extra["donor_binding"] = "no_donor"
    argv, _, _ = build_driver_job(
        "tunable-contract-extractor",
        ctx,
        {"kind": "warmstart", "run_id": "007", "k_eval": 2},
        repo_root=repo,
    )
    assert "--donor-snapshot" not in argv


def test_phase_c_job_must_match_deterministic_action(tmp_path: Path) -> None:
    repo, ctx = _fixture(tmp_path)
    request = {
        "kind": "phase_c", "run_id": "007", "method": "bo", "trial_cap": 10
    }
    with mock.patch(
        "driver.jobs._phase_c_action",
        return_value={"action": "run", "method": "bo", "bout_trials": 10,
                      "sampler": "tpe"},
    ):
        argv, _, _ = build_driver_job(
            "tuner-orchestrator", ctx, request, repo_root=repo
        )
    assert "--n-trials" in argv and argv[argv.index("--n-trials") + 1] == "10"
    assert "nohup" not in argv and "&" not in argv

    with mock.patch(
        "driver.jobs._phase_c_action",
        return_value={"action": "run", "method": "grid"},
    ):
        with pytest.raises(DriverJobError, match="does not match"):
            build_driver_job("tuner-orchestrator", ctx, request, repo_root=repo)


def test_v3_phase_c_job_requires_the_complete_bout_cap(tmp_path: Path) -> None:
    repo, ctx = _fixture(tmp_path)
    (ctx.run_dir / "framework_cfg.json").write_text(json.dumps({
        "max_evaluations": 100,
        "tuner": {"scheduler_policy": "v3_2", "bout_trials": 10},
    }))
    request = {
        "kind": "phase_c", "run_id": "007", "method": "bo", "trial_cap": 5
    }
    with mock.patch(
        "driver.jobs._phase_c_action",
        return_value={"action": "run", "method": "bo", "bout_trials": 10,
                      "sampler": "tpe"},
    ):
        with pytest.raises(DriverJobError, match="complete bout"):
            build_driver_job("tuner-orchestrator", ctx, request, repo_root=repo)


def test_selfrank_job_uses_repo_root_pool_runner(tmp_path: Path) -> None:
    repo, ctx = _fixture(tmp_path)
    request = {
        "kind": "phase_c",
        "run_id": "007",
        "method": "selfrank",
        "trial_cap": 8,
    }
    with mock.patch(
        "driver.jobs._phase_c_action",
        return_value={
            "action": "run",
            "method": "selfrank",
            "bout_trials": 8,
        },
    ):
        argv, log, _ = build_driver_job(
            "tuner-orchestrator", ctx, request, repo_root=repo
        )
    assert argv[:3] == ["uv", "--project", str(repo)]
    assert argv[4:6] == ["python", str(repo / "tools/tuners/selfrank_search.py")]
    assert argv[-2:] == ["--n-evals", "8"]
    assert log.name == "_phase_c_selfrank.log"


@pytest.mark.parametrize(
    ("method", "trial_cap"),
    [("mixup", 24), ("turbo", 10)],
)
def test_mixup_turbo_jobs_use_repo_root_adapter(
    tmp_path: Path, method: str, trial_cap: int
) -> None:
    repo, ctx = _fixture(tmp_path)
    (ctx.run_dir / "framework_cfg.json").write_text(
        json.dumps(
            {
                "max_evaluations": 100,
                "tuner": {
                    "scheduler_policy": "anchor_challenger_v1",
                    "inner_policy": "mixup24-turbo20-v1",
                    "deep_tune_budget_fraction": None,
                    "deep_tune_per_candidate_cap": 44,
                },
            }
        )
    )
    request = {
        "kind": "phase_c",
        "run_id": "007",
        "method": method,
        "trial_cap": trial_cap,
    }
    with mock.patch(
        "driver.jobs._phase_c_action",
        return_value={
            "action": "run",
            "method": method,
            "bout_trials": trial_cap,
        },
    ):
        argv, log, _ = build_driver_job(
            "tuner-orchestrator", ctx, request, repo_root=repo
        )

    assert argv[:3] == ["uv", "--project", str(repo)]
    assert argv[4:6] == [
        "python",
        str(repo / f"tools/tuners/{method}_search.py"),
    ]
    assert argv[-2:] == ["--n-evals", str(trial_cap)]
    assert log.name == f"_phase_c_{method}.log"


def test_driver_job_handoff_waits_then_resumes_same_session(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = ReceiptStore(run_dir)
    runner = FakeSessionRunner([
        {"receipt": {
            "tuned_run_id": "007", "tuned": False, "ledger_updated": False,
            "driver_job": {
                "kind": "phase_c", "run_id": "007", "method": "bo",
                "trial_cap": 10,
            },
        }},
        {"receipt": {
            "tuned_run_id": "007", "tuned": True, "ledger_updated": True,
        }},
    ])
    calls = []

    def job_runner(role, ctx, request, *, repo_root):
        calls.append((role, ctx.invocation_id, request, repo_root))
        return {"kind": "phase_c", "run_id": "007", "returncode": 0}

    receipt, inv_id = _invoke_with_driver_jobs(
        runner, store, "tuner-orchestrator", "toy", "r1", run_dir,
        round_no=2, repo_root=tmp_path, job_runner=job_runner,
    )

    assert receipt["tuned"] is True and inv_id == 2
    assert calls[0][1] == 1
    assert runner.calls[1][1].resume_session_id == "fake-sess-0001"
    assert "driver_job_result" in runner.calls[1][1].extra


def test_driver_job_resume_preserves_original_extra(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    store = ReceiptStore(run_dir)
    runner = FakeSessionRunner([
        {"receipt": {
            "run_id": "007", "status": "driver_job", "ledger_updated": False,
            "driver_job": {"kind": "warmstart", "run_id": "007", "k_eval": 2},
        }},
        {"receipt": {"run_id": "007", "status": "keep", "ledger_updated": True}},
    ])

    def job_runner(role, ctx, request, *, repo_root):
        return {"kind": "warmstart", "run_id": "007", "returncode": 0}

    _invoke_with_driver_jobs(
        runner, store, "tunable-contract-extractor", "toy", "r1", run_dir,
        run_id="007", extra={"candidate_dir": "/candidate", "diagnosis_verdict": "fix"},
        repo_root=tmp_path, job_runner=job_runner,
    )

    resumed = runner.calls[1][1].extra
    assert resumed["candidate_dir"] == "/candidate"
    assert resumed["diagnosis_verdict"] == "fix"
    assert "driver_job_result" in resumed


def _running_record(jobs_dir: Path, pid) -> Path:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    path = jobs_dir / "tuner-orchestrator-0001.json"
    path.write_text(json.dumps({
        "schema_version": 1, "role": "tuner-orchestrator", "invocation_id": 1,
        "run_id": "007", "request": {}, "argv": [], "log": "x.log",
        "status": "running", "started_at": "2026-08-13T00:00:00", "pid": pid,
    }))
    return path


def test_reconcile_marks_orphaned_running_record_dead(tmp_path: Path) -> None:
    from driver.jobs import _reconcile_running_jobs
    # 占用一个肯定已死的 pid：spawn 后立即回收
    import subprocess as sp
    dead = sp.Popen(["true"]); dead.wait()
    path = _running_record(tmp_path / "driver_jobs", dead.pid)
    _reconcile_running_jobs(tmp_path / "driver_jobs")
    assert json.loads(path.read_text())["status"] == "dead"


def test_reconcile_refuses_launch_beside_live_orphan(tmp_path: Path) -> None:
    import subprocess as sp
    from driver.jobs import _reconcile_running_jobs
    live = sp.Popen(["sleep", "30"])
    try:
        _running_record(tmp_path / "driver_jobs", live.pid)
        with pytest.raises(DriverJobError, match="still running"):
            _reconcile_running_jobs(tmp_path / "driver_jobs")
    finally:
        live.kill(); live.wait()
