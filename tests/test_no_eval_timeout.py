"""Production budget mode, subprocess boundaries and role task-contract routing."""
import json
import sys
import time
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools/tuners'))
from init_run import initialize_run
from driver.__main__ import build_parser
from evaluation_budget import EvaluationBudgetExhausted, reserve_evaluation
from _common import timed_eval, timed_preflight


def repo_fixture(tmp_path):
    task = tmp_path / 'tasks/unit'
    task.mkdir(parents=True)
    (task / 'task.toml').write_text('[run]\ntimeout_seconds = 0.1\nworking_dir = "tasks/unit"\n'
                                  '[evaluation]\nscore_fn = "evaluate_config"\npreflight_fn = "preflight_config"\n')
    (task / 'TASK.md').write_text('# Task\nScore honestly.\n\n<!-- runtime-budget:start -->\n'
                                'A default budget.\n<!-- runtime-budget:end -->\n\nKeep evaluation fixed.\n')
    return tmp_path


def test_init_stages_effective_contract_and_freezes_mode(tmp_path):
    repo = repo_fixture(tmp_path)
    source = (repo / 'tasks/unit/task.toml').read_bytes()
    run = initialize_run(repo, 'unit', 'uncapped', no_eval_timeout=True,
                         time_budget_seconds=120, final_reserve_seconds=10)
    cfg_path = run / 'framework_cfg.json'
    cfg = json.loads(cfg_path.read_text())
    assert cfg['evaluation_timeout_mode'] == 'run_budget'
    assert cfg['per_runtime_limit'] is None
    staged = run / 'task_contract'
    doc = tomllib.loads((staged / 'task.toml').read_text())
    assert doc['run'] == {'working_dir': 'tasks/unit', 'evaluation_timeout_mode': 'run_budget'}
    assert doc['evaluation'] == tomllib.loads(source.decode())['evaluation']
    budget = json.loads((staged / 'budget.json').read_text())
    assert budget == {'mode': 'run_budget', 'per_runtime_limit': None}
    view = (staged / 'TASK.md').read_bytes()
    initialize_run(repo, 'unit', 'uncapped')
    assert cfg_path.read_text() == json.dumps(cfg, indent=2, ensure_ascii=False) + '\n'
    assert (staged / 'TASK.md').read_bytes() == view
    assert (repo / 'tasks/unit/task.toml').read_bytes() == source
    with pytest.raises(ValueError, match='cannot change.*timeout'):
        initialize_run(repo, 'unit', 'uncapped', per_runtime_limit=5)
    fixed = initialize_run(repo, 'unit', 'fixed', per_runtime_limit=2)
    fixed_doc = tomllib.loads((fixed / 'task_contract/task.toml').read_text())
    assert fixed_doc['run']['timeout_seconds'] == 2
    assert json.loads((fixed / 'task_contract/budget.json').read_text()) == {
        'mode': 'fixed', 'per_runtime_limit': 2}
    with pytest.raises(ValueError, match='cannot change.*timeout'):
        initialize_run(repo, 'unit', 'fixed', no_eval_timeout=True, time_budget_seconds=120)


def test_cli_timeout_switches_are_exclusive():
    parser = build_parser()
    args = parser.parse_args(['run', 'unit', 'tag', '--loop', 'experiment', '--no-eval-timeout'])
    assert args.no_eval_timeout
    with pytest.raises(SystemExit):
        parser.parse_args(['run', 'unit', 'tag', '--loop', 'experiment', '--no-eval-timeout', '--timeout', '10'])


def runtime_fixture(tmp_path, *, mode='run_budget', remaining=20):
    repo = repo_fixture(tmp_path)
    candidate = repo / 'runs/mle-statoil-iceberg/tag/candidates/001/train.py'
    candidate.parent.mkdir(parents=True)
    candidate.write_text('PARAM_SCHEMA = {"x": "int"}\nSEARCH_SPACE = {"x": ("int", 1, 2)}\n'
                         'BASE_PARAMS = {"x": 1}\ndef make_model(params):\n    return params\n')
    (candidate.parent / 'prepare.py').write_text(
        'import time\ndef evaluate_config(make_model, params):\n    time.sleep(1.4)\n    return 0.25\n'
        'def preflight_config(make_model, params):\n    time.sleep(1.4)\n    return {"status": "ok"}\n')
    run = candidate.parent.parent.parent
    (run / 'framework_cfg.json').write_text(json.dumps({
        'evaluation_timeout_mode': mode, 'per_runtime_limit': None if mode == 'run_budget' else 0.1,
        'preflight_runtime_limit': None if mode == 'run_budget' else 0.1, 'deadline': time.time() + remaining,
        'final_reserve_seconds': 0, 'max_evaluations': 5}))
    return run, candidate


def test_running_eval_crosses_phase_quota_but_next_attempt_is_refused(tmp_path):
    run, candidate = runtime_fixture(tmp_path)
    state = run / '.scheduler/round_state.json'
    state.parent.mkdir()
    state.write_text(json.dumps({'phase': 'optimize', 'phase_deadline': time.time() + 0.2}))
    assert timed_eval(None, None, {'x': 1}, candidate) == 0.25
    with pytest.raises(EvaluationBudgetExhausted) as caught:
        reserve_evaluation(candidate, params={'x': 2}, phase='phase_a', method='warmstart')
    assert caught.value.scope == 'round_quota'
    assert len([r for r in (run / 'evaluation_attempts.jsonl').read_text().splitlines()
                if json.loads(r)['kind'] == 'score_attempt']) == 1


def test_preflight_uses_run_budget_and_deadline_is_not_infeasible(tmp_path, monkeypatch):
    monkeypatch.setattr("_common.DEFAULT_PREFLIGHT_LIMIT", 0.1)
    run, candidate = runtime_fixture(tmp_path)
    assert timed_preflight({'x': 1}, candidate)['status'] == 'ok'
    cfg = json.loads((run / 'framework_cfg.json').read_text())
    cfg['deadline'] = time.time() + 5.2
    cfg['final_reserve_seconds'] = 5
    (run / 'framework_cfg.json').write_text(json.dumps(cfg))
    # Longer than the legacy one-second floor too; the global cutoff must win.
    (candidate.parent / 'prepare.py').write_text(
        'import time\ndef preflight_config(make_model, params):\n    time.sleep(3)\n    return {}\n')
    with pytest.raises(EvaluationBudgetExhausted) as caught:
        timed_preflight({'x': 1}, candidate)
    assert caught.value.scope == 'time_cutoff'


def test_fixed_mode_still_times_out(tmp_path):
    _, candidate = runtime_fixture(tmp_path, mode='fixed')
    with pytest.raises(TimeoutError):
        timed_eval(None, None, {'x': 1}, candidate)


def test_roles_and_prepared_judge_use_the_frozen_task_contract(tmp_path, monkeypatch):
    from driver.roles import InvocationContext, RoleDefinition
    from driver.session import admit_session
    from driver.loops import experiment

    repo = repo_fixture(tmp_path)
    run = initialize_run(repo, 'unit', 'r', no_eval_timeout=True, time_budget_seconds=120)
    role = RoleDefinition(name='test', prompt_file='test.md', receipt_schema={}, tools=[], disallowed=[],
                          postconditions=lambda ctx, payload: [])
    for directory in (run, run / 'candidates/001/_hebo_llm'):
        directory.mkdir(parents=True, exist_ok=True)
        ctx = admit_session(role, InvocationContext(task='unit', tag='r', run_dir=directory, invocation_id=1))
        assert Path(ctx.extra['task_contract_dir']) == run / 'task_contract'
        assert json.loads(ctx.extra['effective_evaluation_budget']) == {
            'mode': 'run_budget', 'per_runtime_limit': None}

    def prepare_judge(_run, _repo, _cmd, _events, args, _label):
        brief = Path(args[args.index('--task-brief') + 1])
        assert brief == run / 'task_contract/TASK.md'
        Path(args[args.index('--output') + 1]).write_text(json.dumps({'prompt_text': brief.read_text()}))

    def invoke(*args, **kwargs):
        assert kwargs['inline_payload'] == (run / 'task_contract/TASK.md').read_text()
        return {}, 1

    from types import SimpleNamespace
    monkeypatch.setattr(experiment, '_slate_cmd', prepare_judge)
    monkeypatch.setattr(experiment, '_invoke', invoke)
    monkeypatch.setattr(experiment, '_validate_judge_stage', lambda *a, **kw: {'status': 'valid'})
    store = SimpleNamespace(receipt_path=lambda *a: None, load_session_id=lambda *a: None)
    experiment._invoke_slate_judge(None, store, 'unit', 'r', run, run / 'gen', 'regular',
                                  round_no=1, model='m', repo_root=repo, cmd=None, events=None)


def test_task_contract_preserves_non_budget_fields_for_installed_tasks(tmp_path):
    from task_contract import stage_task_contract
    for task in (ROOT / 'tasks').iterdir():
        if not (task / 'task.toml').is_file() or not (task / 'TASK.md').is_file():
            continue
        run = tmp_path / task.name
        run.mkdir()
        (run / 'framework_cfg.json').write_text(json.dumps({
            'evaluation_timeout_mode': 'run_budget', 'per_runtime_limit': None,
            'deadline': time.time() + 100, 'preflight_runtime_limit': None}))
        staged = stage_task_contract(ROOT, run, task.name)
        source = tomllib.loads((task / 'task.toml').read_text())
        actual = tomllib.loads((staged / 'task.toml').read_text())
        source.setdefault('run', {}).pop('timeout_seconds', None)
        source['run']['evaluation_timeout_mode'] = 'run_budget'
        assert actual == source, task.name


def test_global_eval_cutoff_and_standalone_preflight_are_budget_events(tmp_path):
    import subprocess
    run, candidate = runtime_fixture(tmp_path, remaining=0.2)
    with pytest.raises(EvaluationBudgetExhausted) as caught:
        timed_eval(None, None, {'x': 1}, candidate)
    assert caught.value.scope == 'time_cutoff'
    rows = [json.loads(line) for line in (run / 'evaluation_attempts.jsonl').read_text().splitlines()]
    assert rows[-1]['time_cutoff'] is True
    result = subprocess.run([sys.executable, str(ROOT / 'tools/preflight_candidate.py'),
                             '--candidate-path', str(candidate)], capture_output=True, text=True)
    assert result.returncode == 4
    assert json.loads(result.stdout)['status'] == 'budget_exhausted'


@pytest.mark.parametrize('engine', ['hebo', 'local_tr'])
def test_checkpoint_deadline_closes_stage_without_crash(tmp_path, monkeypatch, capsys, engine):
    import importlib
    import inner_policy
    from tests.test_inner_policy import _fixture, FLOAT3, BASE3

    search = importlib.import_module(engine + '_search')
    policy = inner_policy.HEBO_TURBO_POLICY_ID if engine == 'hebo' else inner_policy.LOCAL_TR_POLICY_ID
    candidate, report_path = _fixture(tmp_path, space=FLOAT3, base=BASE3, inner_policy_id=policy)
    cfg_path = tmp_path / 'framework_cfg.json'
    cfg = json.loads(cfg_path.read_text())
    cfg.update(evaluation_timeout_mode='run_budget', per_runtime_limit=None,
               preflight_runtime_limit=None, deadline=time.time() + 120)
    cfg_path.write_text(json.dumps(cfg))
    build = search._build_checkpoint

    def expire_before_checkpoint(**kwargs):
        cfg['deadline'] = time.time() - 1
        cfg_path.write_text(json.dumps(cfg))
        return build(**kwargs)

    monkeypatch.setattr(search, '_build_checkpoint', expire_before_checkpoint)
    monkeypatch.setattr(sys, 'argv', [engine + '_search.py', '--candidate-path', str(candidate),
                                    '--tune-report-json', str(report_path), '--n-evals', '2'])
    assert search.main() == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt['status'] == 'budget_exhausted'
    assert receipt['budget_exhausted_scope'] == 'time_cutoff'
    assert receipt['trials_attempted'] == 0
    stage = json.loads(report_path.read_text())['phase_c']['stages'][-1]
    assert stage['status'] == 'budget_exhausted'
    assert stage['budget_exhausted_scope'] == 'time_cutoff'
    assert not stage['trials']
    assert not (tmp_path / 'evaluation_attempts.jsonl').exists()


@pytest.mark.parametrize('loop', ['hillclimb', 'baseline-tune', 'rewrite'])
def test_existing_run_budget_config_cannot_enter_other_loop(tmp_path, monkeypatch, loop):
    from driver import __main__ as cli, roles, metadata
    repo = repo_fixture(tmp_path)
    initialize_run(repo, 'unit', 'uncapped', no_eval_timeout=True, time_budget_seconds=120)
    monkeypatch.setattr(roles, 'REPO_ROOT', repo)

    def unexpected_model_resolution(*args):
        raise AssertionError('unsupported mode must be rejected before launching the loop')

    monkeypatch.setattr(metadata, 'resolve_model', unexpected_model_resolution)
    with pytest.raises(SystemExit) as caught:
        cli.main(['run', 'unit', 'uncapped', '--loop', loop, '--model', 'unused'])
    assert caught.value.code == 2
