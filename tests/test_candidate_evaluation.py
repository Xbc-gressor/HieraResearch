"""Candidate preparation/settlement at the real tool boundary, without training."""
import json
from pathlib import Path
import subprocess
import sys

from driver import candidate_evaluation
from driver.events import EventsLog
from tests.test_driver_experiment import ExperimentCmd

ROOT = Path(__file__).resolve().parents[1]


def run_tool(args, repo_root, **kwargs):
    return subprocess.run([sys.executable, *map(str, args[1:])], cwd=repo_root,
                          capture_output=True, text=True, check=kwargs.get('check', True))


def test_prepare_reuses_installed_contract_and_is_repeatable(tmp_path):
    candidate = tmp_path / 'candidates/001'
    candidate.mkdir(parents=True)
    (candidate / 'train.py').write_text(
        'PARAM_SCHEMA = {"x": "int"}\nSEARCH_SPACE = {"x": ("int", 1, 2)}\n'
        'BASE_PARAMS = {"x": 1}\ndef make_model(params):\n    return params\n')
    (candidate / '_warm_configs.json').write_text('[{"x": 1}, {"x": 3}]')
    (candidate / '_search_space.json').write_text('{"x": ["int", 1, 3]}')
    (candidate / '_candidate_brief.json').write_text('{"source_run_ids": []}')
    assert candidate_evaluation.prepare(candidate, ROOT, run_tool, {}) == []
    source = (candidate / 'train.py').read_text()
    assert "('int', 1, 3)" in source
    assert candidate_evaluation.prepare(candidate, ROOT, run_tool, {}) == []
    assert (candidate / 'train.py').read_text() == source
    (candidate / '_warm_configs.json').write_text('[{"x": 4}]')
    assert candidate_evaluation.prepare(candidate, ROOT, run_tool, {})


def test_successful_report_settles_without_session_and_is_idempotent(tmp_path):
    cmd = ExperimentCmd(tmp_path)
    run = cmd.run_dir
    candidate = run / 'candidates/008'
    candidate.mkdir(parents=True)
    cmd._save_ledger({'records': [{'run_id': '008', 'status': 'pending'}]})
    (candidate / 'tune_report.json').write_text(json.dumps({
        'phase_a': {'status': 'ok', 'best_warm_score': 0.2161099781414916}}))
    assert candidate_evaluation.settle(run, '008', tmp_path, cmd, EventsLog(run))
    assert cmd._ledger()['records'][0]['status'] == 'keep'
    calls = list(cmd.calls)
    assert candidate_evaluation.settle(run, '008', tmp_path, cmd, EventsLog(run))
    assert cmd.calls == calls
    assert not any('--mark-tuned' in call for call in calls)


def test_settlement_refuses_report_rejected_by_ledger(tmp_path):
    cmd = ExperimentCmd(tmp_path)
    run = cmd.run_dir
    candidate = run / 'candidates/009'
    candidate.mkdir(parents=True)
    cmd._save_ledger({'records': [{'run_id': '009', 'status': 'pending'}]})
    (candidate / 'tune_report.json').write_text(
        '{"phase_a": {"status": "ok", "best_warm_score": 0.30940899106672853}}')
    cmd.fail_payload['set-tuning'] = (1, '', 'execution revision mismatch')
    assert not candidate_evaluation.settle(run, '009', tmp_path, cmd, EventsLog(run))
    assert cmd._ledger()['records'][0]['status'] == 'pending'
    assert not any('record-run' in call for call in cmd.calls)


def test_writer_evaluation_success_needs_no_settlement_session(tmp_path):
    from driver.loops import experiment
    from driver.receipts import ReceiptStore
    from driver.session import FakeSessionRunner
    from tests.test_driver_experiment import write_task
    write_task(tmp_path)
    cmd = ExperimentCmd(tmp_path)
    run = cmd.run_dir
    candidate = run / 'candidates/001'
    candidate.mkdir(parents=True)
    (run / 'framework_cfg.json').write_text('{"round": {"tune_bouts": 0}}')
    cmd._save_ledger({'records': [{'run_id': '001', 'status': 'pending'}]})

    def write(ctx):
        (candidate / 'train.py').write_text('# candidate\n')
        (candidate / '_warm_configs.json').write_text('[{"x": 1}]')
        (candidate / '_search_space.json').write_text('{"x": ["int", 1, 3]}')
        (candidate / '_candidate_brief.json').write_text('{"source_run_ids": []}')

    runner = FakeSessionRunner([{'receipt': {'status': 'written', 'wrote': True,
                                           'candidate_dir': str(candidate)},
                                 'side_effects': write}])
    jobs = []

    def evaluate(role, ctx, request, **kwargs):
        jobs.append((role, request))
        (candidate / 'tune_report.json').write_text(
            '{"phase_a": {"status": "ok", "best_warm_score": 0.2, '
            '"deferred_configs": [{"params": {"x": 3}}]}}')
        return {'kind': 'warmstart', 'run_id': '001', 'returncode': 0}

    experiment._implement_candidate(runner, ReceiptStore(run), 'fake-task', 't1',
                                    run, '001', tmp_path, cmd, EventsLog(run),
                                    job_runner=evaluate)
    assert [role for role, ctx in runner.calls] == ['candidate-writer']
    assert jobs[0][0] == 'driver' and jobs[0][1]['kind'] == 'warmstart'
    assert cmd._ledger()['records'][0]['status'] == 'keep'


def test_preparation_refreshes_inheritance_after_installing_space(tmp_path):
    from tests.test_parameter_inheritance import ParameterInheritanceTests
    from tune_tools import validate_parameter_transfer
    parent, child, configs = ParameterInheritanceTests()._fixture(tmp_path)
    space = {'same': ['int', 1, 10], 'category': ['categorical', ['a', 'c']],
             'changed_kind': ['int', 1, 10], 'new_key': ['float', 0.1, 1.]}
    (child.parent / '_search_space.json').write_text(json.dumps(space))
    assert candidate_evaluation.prepare(child.parent, ROOT, run_tool, {}) == []
    assert json.loads(configs.read_text())[0]['same'] == 7
    # Evaluator's own checker must accept the receipt against installed code.
    validate_parameter_transfer(child, json.loads(configs.read_text()),
        json.loads((child.parent / "_parameter_transfer.json").read_text()))
    assert candidate_evaluation.prepare(child.parent, ROOT, run_tool, {}) == []


def test_completed_artifacts_survive_missing_writer_receipt(tmp_path):
    from driver.loops import experiment
    from driver.receipts import ReceiptStore
    from driver.session import FakeSessionRunner
    from tests.test_driver_experiment import write_task, writer_effect
    write_task(tmp_path)
    cmd = ExperimentCmd(tmp_path)
    run = cmd.run_dir
    candidate = run / 'candidates/001'
    candidate.mkdir(parents=True)
    (run / 'framework_cfg.json').write_text('{}')
    cmd._save_ledger({'records': [{'run_id': '001', 'status': 'pending'}]})
    runner = FakeSessionRunner([
        {'fail': ['no accepted receipt'], 'side_effects': writer_effect},
        {'receipt': {'status': 'written', 'wrote': False, 'candidate_dir': str(candidate)}}])
    experiment._implement_candidate(runner, ReceiptStore(run), 'fake-task', 't1', run,
        '001', tmp_path, cmd, EventsLog(run), job_runner=cmd.evaluate)
    assert cmd._ledger()['records'][0]['status'] == 'keep'


def test_reserving_repair_attempt_preserves_anchor_and_recovery_state(tmp_path):
    from driver.loops import experiment
    state = {'attempts': 2, 'baseline_received': True,
             'completed_invocation_id': 8, 'consecutive_failures': 1}
    (tmp_path / 'writer.attempts.json').write_text(json.dumps(state))
    experiment._register_candidate_writer_attempt(tmp_path)
    assert json.loads((tmp_path / 'writer.attempts.json').read_text()) == {
        **state, 'attempts': 3}


def test_resource_preflight_blocks_without_candidate_observation(tmp_path):
    import pytest
    from driver.loops import experiment
    from driver.receipts import ReceiptStore
    from driver.resources import ResourcePreflightError
    from driver.session import FakeSessionRunner
    from tests.test_driver_experiment import write_task, writer_effect
    write_task(tmp_path)
    cmd = ExperimentCmd(tmp_path)
    run = cmd.run_dir
    candidate = run / 'candidates/000'
    candidate.mkdir(parents=True)
    (candidate / 'train.py').write_text('# supplied baseline\n')
    (run / 'framework_cfg.json').write_text('{}')
    cmd._save_ledger({'records': [{'run_id': '000', 'status': 'pending'}]})
    runner = FakeSessionRunner([{'receipt': {'status': 'existing', 'wrote': False,
        'candidate_dir': str(candidate)}, 'side_effects': writer_effect}])
    def refuse(*args, **kwargs):
        raise ResourcePreflightError('GPU memory below task requirement')
    with pytest.raises(experiment.RunBlocked, match='resource'):
        experiment._implement_candidate(runner, ReceiptStore(run), 'fake-task', 't1',
            run, '000', tmp_path, cmd, EventsLog(run), job_runner=refuse,
            task_toml={'seed': {'provided': 'train.py'}})
    assert cmd._ledger()['records'][0]['status'] == 'pending'
    assert not any('record-run' in call for call in cmd.calls)
