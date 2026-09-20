"""Final delivery integration without a model, GPU, or paid backend."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]

from driver import __main__ as cli
from driver.session import FakeSessionRunner
from tests.test_driver_experiment import (
    judge_entry,
    write_audited_background,
    write_task,
)
from tools import mlebench_export, mlebench_finalize, mlebench_grade
from tools.search_space_state import empty_search_space_state


@pytest.fixture(autouse=True)
def _isolated_evaluation_domain():
    """cli.main exports EVALUATION_STAGE/FIDELITY for the whole process;
    keep that out of later tests (rounds.status appends them to its calls)."""
    keys = ("EVALUATION_STAGE", "EVALUATION_FIDELITY")
    saved = {k: os.environ.pop(k) for k in keys if k in os.environ}
    yield
    for key in keys:
        if key in saved:
            os.environ[key] = saved[key]
        else:
            os.environ.pop(key, None)


@pytest.mark.parametrize("phase,export_rc,grade_rc", [
    ("completed", 0, 0), ("completed", 3, 0), ("completed", 0, 4), ("blocked", 0, 0),
])
def test_driver_delivers_after_completion_without_watchdog(tmp_path, phase, export_rc, grade_rc):
    run_dir = tmp_path / "runs" / "toy" / "r1"
    run_dir.mkdir(parents=True)
    # The persisted deadline deliberately differs from the CLI budget.
    deadline = time.time() + 30
    (run_dir / "framework_cfg.json").write_text(json.dumps({"deadline": deadline}))
    export = tmp_path / "export.py"
    export.write_text("from pathlib import Path\n"
                      "Path('submission.csv').write_text('Id,Probability\\n1,0.5\\n')\n"
                      f"raise SystemExit({export_rc})\n")
    grade = tmp_path / "grade.py"
    grade.write_text("from pathlib import Path\n"
                     "assert Path('submission.csv').exists()\n"
                     "Path('graded').touch()\n"
                     f"raise SystemExit({grade_rc})\n")
    with mock.patch("driver.roles.REPO_ROOT", tmp_path), \
            mock.patch("driver.session.SDKSessionRunner"), \
            mock.patch("driver.loops.experiment.run_experiment",
                       return_value={"phase": phase}):
        rc = cli.main([
            "run", "toy", "r1", "--loop", "experiment", "--model", "m",
            "--time-budget", "27000", "--data-dir", str(tmp_path),
            "--submission-command", shlex.join([sys.executable, str(export)]),
            "--grader-command", shlex.join([sys.executable, str(grade)]),
        ])
    manifest = run_dir / "finalization-manifest.json"
    if phase == "blocked":
        assert rc != 0
        assert not manifest.exists()
        assert not (run_dir / "submission.csv").exists()
        return
    assert rc == (export_rc or grade_rc)
    data = json.loads(manifest.read_text())
    assert data["deadline_unix"] == deadline
    assert (run_dir / "graded").exists() == (export_rc == 0)
    if export_rc == 0:
        assert data["submission"]["ended_at_unix"] <= deadline
        assert data["grader"]["started_at_unix"] >= data["submission"]["ended_at_unix"]
        assert data["grader"]["counts_toward_submission_budget"] is False


def test_early_stop_run_finalizes_in_the_same_process(tmp_path, capsys):
    """REPORT-finalize-early-stop-gap: a zero-progress early stop must reach
    automatic finalize in the same driver process. Everything between the
    loop entry and the delivery artifacts is real — the real completion
    guard (ledger set-phase), the real status derivation, the real CLI gate;
    only the paid role sessions and the export/grading commands are
    replaced."""
    repo = tmp_path
    shutil.copytree(ROOT / "tools", repo / "tools",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "contracts", repo / "contracts")
    write_task(repo)
    run_dir = repo / "runs" / "fake-task" / "t1"
    subprocess.run(
        [sys.executable, "tools/init_run.py", "fake-task", "t1",
         "--time-budget", "3600", "--dimension-strategy", "catalog_subset",
         "--semantic-policy", "coverage_attempt"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    (run_dir / "ledger.json").write_text(json.dumps({
        "records": [{"run_id": "000", "status": "keep", "op": "fresh",
                     "final_best_score": 0.2}],
        "search_space_state": empty_search_space_state(),
    }))
    # A real background that passes the real validator (catalog-matching
    # registry); its claim mappings route through one faithfulness judge
    # session, scripted as all-faithful.
    write_audited_background(run_dir)
    (run_dir / "evaluation_attempts.jsonl").write_text("")
    cfg = json.loads((run_dir / "framework_cfg.json").read_text())
    deadline = cfg["deadline"]
    # The cutoff (deadline − reserve) is comfortably in the future.
    assert deadline - time.time() > 2 * cfg.get("final_reserve_seconds", 0)

    export = repo / "export.py"
    export.write_text("from pathlib import Path\n"
                      "Path('submission.csv').write_text('Id,Probability\\n1,0.5\\n')\n")
    grade = repo / "grade.py"
    grade.write_text("from pathlib import Path\n"
                     "assert Path('submission.csv').exists()\n"
                     "Path('graded').touch()\n")

    # Two zero-progress rounds (idea-generator proposes nothing, tuner has
    # no bout) quiesce the loop while the budget/clock still read "running".
    runner = FakeSessionRunner([
        judge_entry(set()),
        {"receipt": {"actions": []}},
        {"receipt": {"tuned_run_id": "none", "tuned": False,
                     "ledger_updated": False}},
        {"receipt": {"actions": []}},
        {"receipt": {"tuned_run_id": "none", "tuned": False,
                     "ledger_updated": False}},
    ])

    import driver.loops.experiment as experiment_mod
    saved_default = experiment_mod.run_experiment.__kwdefaults__["repo_root"]
    experiment_mod.run_experiment.__kwdefaults__["repo_root"] = repo
    try:
        with mock.patch("driver.roles.REPO_ROOT", repo), \
                mock.patch("driver.session.SDKSessionRunner",
                           lambda model, events, cli_path=None: runner), \
                mock.patch("driver.loops.common.preflight_env",
                           lambda *args, **kwargs: None):
            rc = cli.main([
                "run", "fake-task", "t1", "--loop", "experiment",
                "--model", "m", "--data-dir", str(repo),
                "--submission-command", shlex.join([sys.executable, str(export)]),
                "--grader-command", shlex.join([sys.executable, str(grade)]),
            ])
    finally:
        experiment_mod.run_experiment.__kwdefaults__["repo_root"] = saved_default

    assert rc == 0
    # The CLI prints the loop's returned status as a multi-line JSON block
    # between single-line "[driver]" progress events.
    block: list[str] = []
    for line in capsys.readouterr().out.splitlines():
        if line == "{":
            block = ["{"]
        elif block:
            block.append(line)
            if line == "}":
                break
    status = json.loads("\n".join(block))
    assert status["phase"] == "completed"
    assert status["stop_condition"] == "quiescent"
    assert status["active_stop_condition"] == "quiescent"
    stored = json.loads((run_dir / "ledger.json").read_text())["run_state"]
    assert stored["phase"] == "completed"
    assert stored["active_stop_condition"] == "quiescent"
    loop_state = (run_dir / "loop_state.md").read_text()
    assert "phase: completed" in loop_state
    assert "active_stop_condition: quiescent" in loop_state
    assert (run_dir / "submission.csv").exists()
    assert (run_dir / "graded").exists()
    manifest = json.loads(
        (run_dir / "finalization-manifest.json").read_text())
    assert manifest["deadline_unix"] == deadline
    assert manifest["submission"]["ended_at_unix"] <= deadline
    assert manifest["grader"]["started_at_unix"] >= \
        manifest["submission"]["ended_at_unix"]
    events = (run_dir / "driver_events.jsonl").read_text()
    assert "quiescent" in events
    assert "finalization_finished" in events


@pytest.mark.parametrize("remaining", [-1, 0.15])
def test_export_deadline_never_runs_grader(tmp_path, remaining):
    export = tmp_path / "export.py"
    export.write_text("import time\ntime.sleep(60)\n")
    grade = tmp_path / "grade.py"
    grade.write_text("from pathlib import Path\nPath('graded').touch()\n")
    rc = mlebench_finalize.main([
        "--task", "toy", "--run-dir", str(tmp_path), "--data-dir", str(tmp_path),
        "--deadline", str(time.time() + remaining),
        "--submission-command", shlex.join([sys.executable, str(export)]),
        "--grader-command", shlex.join([sys.executable, str(grade)]),
    ])
    assert rc == 124
    assert not (tmp_path / "graded").exists()
    data = json.loads((tmp_path / "finalization-manifest.json").read_text())
    assert data["submission"]["status"] == ("deadline_expired" if remaining < 0 else "timeout")


def test_export_uses_selected_implementation_and_applied_params(tmp_path, monkeypatch):
    task = tmp_path / "tasks" / "toy"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('name = "toy"\n')
    (task / "prepare.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def export_submission(make_model, params, output):\n"
        "    assert Path(os.environ['MLEBENCH_PUBLIC_DATA']).is_dir()\n"
        "    output.write_text('Id,Probability\\n1,' + str(make_model(None, params)) + '\\n')\n")
    run = tmp_path / "run"
    (run / "run_input" / "public").mkdir(parents=True)
    candidate = run / "candidates" / "002"
    candidate.mkdir(parents=True)
    (candidate / "train.py").write_text("def make_model(dataset, params): return params['p']\n")
    (run / "ledger.json").write_text(json.dumps({"records": [
        {"run_id": "001", "final_best_score": 0.4},
        {"run_id": "002", "final_best_score": 0.1, "applied_incumbent": {"params": {"p": 0.7}}},
    ]}))
    monkeypatch.setattr(mlebench_export, "__file__", str(tmp_path / "tools/mlebench_export.py"))
    monkeypatch.setenv("MLEBENCH_PUBLIC_DATA", "unused")
    monkeypatch.delitem(sys.modules, "prepare", raising=False)
    with mock.patch.object(sys, "path", list(sys.path)), mock.patch.dict(sys.modules):
        assert mlebench_export.main(["--task", "toy", "--run-dir", str(run)]) == 0
    assert (run / "submission.csv").read_text() == "Id,Probability\n1,0.7\n"


def test_grading_can_follow_smoke_without_deleting_source(tmp_path):
    public = tmp_path / "data" / "birds" / "prepared" / "public"
    public.mkdir(parents=True)
    private = tmp_path / "operator-private"
    private.mkdir()
    (private / "answers.csv").write_text("answers")
    roots = []

    def grade(command, *, stdout, **kwargs):
        root = Path(command[-1])
        roots.append(root)
        assert (root / "birds/prepared/private/answers.csv").read_text() == "answers"
        assert (root / "birds/prepared/public").resolve() == public
        stdout.write('{"valid_submission": true}\n')
        return subprocess.CompletedProcess(command, 0)

    with mock.patch.object(mlebench_grade.subprocess, "run", side_effect=grade):
        for tag in ("smoke", "serial", "parallel"):
            assert mlebench_grade.main([
                "--competition", "birds", "--data-dir", str(tmp_path / "data"),
                "--private-source", str(private), "--submission", str(tmp_path / "submission.csv"),
                "--output-log", str(tmp_path / f"{tag}.log"), "--mlebench", "mlebench",
            ]) == 0
            assert (private / "answers.csv").read_text() == "answers"
    assert len(set(roots)) == 3
    assert all(not root.exists() for root in roots)
