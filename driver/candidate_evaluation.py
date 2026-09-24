"""Driver-owned candidate preparation and Phase-A settlement.

Tools validate and materialize the contract; this module only sequences them.
Neither successful preparation nor settlement needs a model receipt.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .roles import record_status


def prepare(candidate_dir: Path, repo_root: Path, cmd, extra: dict) -> list[str]:
    train = candidate_dir / 'train.py'
    configs = candidate_dir / '_warm_configs.json'
    space = candidate_dir / '_search_space.json'
    missing = [str(p) for p in (train, configs, space) if not p.is_file()]
    if missing:
        return ['missing candidate artifacts: ' + ', '.join(missing)]
    brief_path = candidate_dir / '_candidate_brief.json'
    brief = json.loads(brief_path.read_text())
    inherit = ['python', 'tools/tuners/tune_tools.py', 'build-inheritance',
               '--candidate-path', train, '--configs-json', configs]
    commands = [
        ['python', 'tools/tuners/tune_tools.py', 'lint-schema',
         '--candidate-path', train, '--allow-materialized'],
    ]
    if brief.get('source_run_ids'):
        commands.append(inherit)
    commands += [
        ['python', 'tools/tuners/tune_tools.py', 'check-search-space',
         '--candidate-path', train, '--space-json', space, '--configs-json', configs],
        ['python', 'tools/apply_search_space.py', '--candidate-path', train,
         '--space-json', space, '--replace-schema'],
    ]
    if brief.get('source_run_ids'):
        commands.append(inherit)  # bind the installed space, not its previous revision
    if extra.get('donor_binding'):
        donor = ['python', 'tools/tuners/tune_tools.py', 'inject-global-donor',
                 '--candidate-path', train, '--configs-json', configs]
        if extra.get('donor_binding') == 'bound':
            donor += ['--donor-snapshot', extra['donor_snapshot']]
        commands.append(donor)
    for args in commands:
        result = cmd(args, repo_root, check=False)
        if result.returncode:
            return [f'{args[2]}: {(result.stderr or result.stdout).strip()[-4000:]}']
    return []


def settle(run_dir: Path, run_id: str, repo_root: Path, cmd, events) -> bool:
    """Settle a validated applied warm incumbent; false means no valid result.

    set-tuning owns revision, trial-row, inheritance and applied-params checks.
    It must succeed before record-run may publish the score. Both writes are
    idempotent, including interruption between them.
    """
    if record_status(run_dir, run_id) in ('keep', 'discard'):
        return True
    report_path = run_dir / 'candidates' / run_id / 'tune_report.json'
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text())
    phase = report.get('phase_a') or {}
    score = phase.get('best_warm_score')
    if phase.get('status') != 'ok' or isinstance(score, bool) or not isinstance(
            score, (float, int)) or not math.isfinite(score):
        return False
    result = cmd(['python', 'tools/ledger.py', 'set-tuning',
                  '--ledger', run_dir / 'ledger.json', '--run-id', run_id,
                  '--from-report', report_path], repo_root, check=False)
    if result.returncode:
        events.emit('candidate_settlement_failed', run_id=run_id,
                    detail=(result.stderr or result.stdout).strip()[-2000:])
        return False
    result = cmd(['python', 'tools/ledger.py', 'record-run',
                  '--ledger', run_dir / 'ledger.json', '--run-id', run_id,
                  '--final-best-score', str(score)], repo_root, check=False)
    if result.returncode:
        events.emit('candidate_settlement_failed', run_id=run_id,
                    detail=(result.stderr or result.stdout).strip()[-2000:])
        return False
    events.emit('candidate_settled', run_id=run_id,
                outcome=record_status(run_dir, run_id), best_warm_score=score)
    return True