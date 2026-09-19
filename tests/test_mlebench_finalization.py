"""Final delivery integration without a model, GPU, or paid backend."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest import mock

import pytest

from driver import __main__ as cli
from tools import mlebench_export, mlebench_finalize, mlebench_grade


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
