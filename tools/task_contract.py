"""Run-local task presentation derived from the effective evaluation budget.

Only presentation is staged: evaluators, environments and data retain their
original task paths. This is not a filesystem sandbox.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

try:
    from .run_cfg import read_framework_cfg
except ImportError:  # Direct tools/init_run.py execution.
    from run_cfg import read_framework_cfg

START = '<!-- runtime-budget:start -->'
END = '<!-- runtime-budget:end -->'


def timeout_policy(config: dict) -> dict:
    return {'mode': config.get('evaluation_timeout_mode', 'fixed'),
            'per_runtime_limit': config.get('per_runtime_limit')}


def render_budget(policy: dict, cost_signal: bool = True) -> str:
    if policy['mode'] == 'run_budget' and cost_signal:
        rule = ('No admitted evaluation is cut off by a per-evaluation time limit or by its '
                'optimization-phase quota (only the run search deadline interrupts it), but its '
                'wall-clock time counts against the run budget and against the candidate itself: for the same '
                'wall-clock time, a slower candidate gets fewer evaluations. The scheduler checks '
                'the phase quota before starting the next evaluation. Inner resampling (k-fold, '
                'calibration folds, multiplied augmentation) multiplies evaluation cost; use it '
                'only when the mechanism itself depends on it.')
    elif policy['mode'] == 'run_budget':
        rule = ('There is no independent per-evaluation time limit. An admitted evaluation may run '
                'until the run search deadline, including past its optimization-phase quota. '
                'The scheduler checks the phase quota before starting the next evaluation.')
    elif policy['per_runtime_limit'] is not None:
        rule = f"Each evaluation has a {policy['per_runtime_limit']:g}-second wall-clock limit."
    else:
        rule = 'No per-evaluation limit is configured; the run and phase budgets still apply.'
    return ('## Effective runtime budget\n\n' + rule + '\n'
            'Agent calls, setup, training and inference all consume the run budget. '
            'Preserve time for final refitting and submission. The driver supplies the current '
            'remaining search time; framework_cfg.json records the absolute deadline and final reserve.\n')


def _present_toml(source: str, policy: dict) -> str:
    # Preserve every non-budget TOML field, without introducing a TOML writer
    # dependency. Only the ordinary [run] table's two budget keys are replaced.
    table = re.search(r'(?m)^\[run\][ \t]*(?:#.*)?\n', source)
    if table:
        tail = re.search(r'(?m)^\[', source[table.end():])
        end = table.end() + tail.start() if tail else len(source)
        body = source[table.end():end]
        body = re.sub(r'(?m)^[ \t]*(?:timeout_seconds|evaluation_timeout_mode)[ \t]*=.*\n?', '', body)
        # Comments in [run] may justify the task's default cap; the run's
        # effective policy supersedes that default.
        body = re.sub(r'(?m)^[ \t]*#.*\n?', '', body)
        prefix, suffix = source[:table.end()], source[end:]
    else:
        prefix, body, suffix = source.rstrip() + '\n\n[run]\n', '', ''
    budget = f'evaluation_timeout_mode = "{policy["mode"]}"\n'
    if policy['per_runtime_limit'] is not None:
        budget += f'timeout_seconds = {policy["per_runtime_limit"]}\n'
    return prefix + budget + body.rstrip() + '\n\n' + suffix


def stage_task_contract(repo_root: Path, run_dir: Path, task: str) -> Path | None:
    source = Path(repo_root) / 'tasks' / task
    if not (source / 'TASK.md').is_file() or not (source / 'task.toml').is_file():
        return None  # Standalone helpers/test repositories need not install tasks.
    config = read_framework_cfg(Path(run_dir) / 'framework_cfg.json')
    policy = timeout_policy(config)
    target = Path(run_dir) / 'task_contract'
    receipt = target / 'budget.json'
    if receipt.is_file():
        if json.loads(receipt.read_text()) != policy:
            raise ValueError('cannot change evaluation timeout policy after task contract staging')
        return target
    task_text = (source / 'TASK.md').read_text()
    budget_text = render_budget(policy, config.get('cost_signals', {}).get('writer', True))
    if START in task_text:
        before, rest = task_text.split(START, 1)
        _, after = rest.split(END, 1)
        task_text = before + budget_text + after
    else:
        task_text = task_text.rstrip() + '\n\n' + budget_text
    target.mkdir(parents=True, exist_ok=True)
    (target / 'TASK.md').write_text(task_text)
    (target / 'task.toml').write_text(_present_toml((source / 'task.toml').read_text(), policy))
    receipt.write_text(json.dumps(policy, indent=2) + '\n')
    return target


def task_brief_path(repo_root: Path, run_dir: Path, task: str) -> Path:
    staged = Path(run_dir) / 'task_contract/TASK.md'
    return staged if staged.is_file() else Path(repo_root) / 'tasks' / task / 'TASK.md'


def contract_context(run_dir: Path) -> dict:
    """Shared context for SDK and fake runners, including nested tuner sessions."""
    target = Path(run_dir) / 'task_contract'
    if not (target / 'budget.json').is_file():
        return {}
    policy = timeout_policy(read_framework_cfg(Path(run_dir) / 'framework_cfg.json'))
    if json.loads((target / 'budget.json').read_text()) != policy:
        raise ValueError('cannot change evaluation timeout policy after task contract staging')
    return {'task_contract_dir': str(target.resolve()),
            'effective_evaluation_budget': json.dumps(policy)}
