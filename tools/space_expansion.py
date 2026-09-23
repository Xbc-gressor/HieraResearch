"""Durable review/probe lifecycle; deterministic decisions, no model calls."""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from evaluation_budget import (time_budget, reserve_budget, settle_reservation,
    outstanding_reservations, BudgetReservationDenied, _read_budget_rows)
from scheduler.contract import DEFAULT_K_EVAL
from run_cfg import read_framework_cfg
from semantic_evidence import LIFECYCLE_TERMINAL_STATUSES
from space_revisions import load_revision_state

DEFAULTS = {'enabled': False, 'stall_slates': 2, 'improvement_threshold': 0.0,
            'max_reviews': 2, 'targeted_retrieval': False}
STATE = Path('.semantic/space-review-state.json')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def config(run):
    return {**DEFAULTS, **read_framework_cfg(Path(run)/'framework_cfg.json').get('space_expansion', {})}


def load_state(run):
    path = Path(run)/STATE
    if path.exists():
        return json.loads(path.read_text())
    return {'seen_slates': [], 'best_by_domain': {}, 'stalled_slates': 0, 'reviews': [],
            'cancelled_reservations': [], 'probes': {},
            'initial_search_seconds': time_budget(run)['usable_seconds']}


def initialize(run):
    if config(run)['enabled'] and not (Path(run)/STATE).exists():
        write_json(Path(run)/STATE, load_state(run))


def pending_expansion(run):
    run = Path(run)
    state = load_revision_state(run/'background.md')
    if not state:
        return None
    version = state['versions'][-1]
    admission = version.get('admission')
    if not admission or admission['reservation_id'] in load_state(run)['cancelled_reservations']:
        return None
    probe = version['review']['probe']
    ledger_path = run/'ledger.json'
    records = json.loads(ledger_path.read_text()).get('records', []) if ledger_path.exists() else []
    if any(r.get('semantic_point', {}).get('point_id') == probe['point']['point_id']
           and r.get('status') in LIFECYCLE_TERMINAL_STATUSES for r in records):
        return None
    return {'probe': probe, 'space': version['space'], 'review': version['review'], 'admission': admission}


def observe_boundary(run, ledger):
    run = Path(run)
    cfg, state = config(run), load_state(run)
    groups = {}
    for row in ledger.get('records', []):
        generation = (row.get('policy_receipt') or {}).get('generation_id')
        if generation:
            groups.setdefault(generation, []).append(row)
    unseen = [gid for gid, rows in groups.items() if gid not in state['seen_slates']
              and all(r.get('status') in LIFECYCLE_TERMINAL_STATUSES for r in rows)]
    scores = {}
    for record in ledger.get('records', []):
        if record.get('status') not in {'keep', 'discard'}:
            continue
        score = record.get('final_best_score', record.get('best_warm_score'))
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
            continue
        # The production scalar path uses one fixed task evaluator throughout a
        # run. Explicit alternative domains stay separate; investment is not a
        # semantic explanation for a score delta.
        domain = record.get('evaluation_domain') or {'task': run.parent.name, 'surface': 'fixed-task-evaluator'}
        key = json.dumps(domain, sort_keys=True)
        scores[key] = min(score, scores.get(key, math.inf))
    prior = state['best_by_domain']
    improved = (not prior and bool(scores)) or any(
        key in prior and value < prior[key]-cfg['improvement_threshold'] for key, value in scores.items())
    if improved:
        state['stalled_slates'] = 0
    elif unseen:
        state['stalled_slates'] += len(unseen)
    # Seed observations and optimization progress are retained even between
    # completed slates. A new evaluation domain never counts as improvement.
    state['best_by_domain'] = {**prior, **{key:min(value, prior.get(key, math.inf)) for key,value in scores.items()}}
    state['seen_slates'].extend(unseen)
    state['review_due'] = (cfg['enabled'] and bool(unseen) and state['stalled_slates'] >= cfg['stall_slates']
        and len(state['reviews']) < cfg['max_reviews'] and pending_expansion(run) is None
        and not any(r.get('status') == 'pending' for r in ledger.get('records', [])))
    write_json(run/STATE, state)
    return state


def regular_cost(run):
    """Observed implementation/screening/planning costs, independent of timeout."""
    cfg = read_framework_cfg(run/'framework_cfg.json')
    width = max(2, int(cfg.get('tuner', {}).get('K_eval') or DEFAULT_K_EVAL))
    events = run/'driver_events.jsonl'
    rows = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
    seats = [r['seconds'] for r in rows if r.get('kind') == 'seat_finished' and r.get('seconds', 0) > 0]
    plans = [r['seconds']/2 for r in rows if r.get('kind') == 'slate_planning_finished' and r.get('seconds', 0) > 0]
    durations = [r['duration_seconds'] for r in _read_budget_rows(run)
                 if r['kind'] == 'score_completion' and r.get('duration_seconds', 0) > 0]
    if not seats or not plans or not durations:
        raise BudgetReservationDenied('unknown_cost', 'need observed seat, planning and evaluation durations')
    screening = width*statistics.median(durations)
    implementation = max(statistics.median(seats)-screening, .25*statistics.median(seats))
    return {'implementation_seconds': implementation, 'screening_seconds': screening,
            'planning_seconds': statistics.median(plans), 'evaluations': width,
            'source': 'observed seat/planning/evaluation medians',
            'uncertainty': 'new-route cost may differ; implementation floor is 25% of observed seat time'}


def reserve_review(run):
    state, cost = load_state(run), regular_cost(run)
    initial = state['initial_search_seconds']
    if initial is None:
        raise BudgetReservationDenied('unbounded_search_time', 'review share requires a run deadline')
    allowance = min(cost['implementation_seconds']+cost['screening_seconds'],
                    .1*initial-sum(r['budget_seconds'] for r in state['reviews']))
    if allowance <= 0:
        raise BudgetReservationDenied('review_share_exhausted')
    candidate_seconds = cost['implementation_seconds']+cost['screening_seconds']+cost['planning_seconds']
    reservation = reserve_budget(run, label='space_probe_slate',
        seconds=2*candidate_seconds+allowance, evaluations=2*cost['evaluations'], note=cost['source'])
    settle_reservation(run, reservation['reservation_id'], consumed_seconds=allowance)
    review = {'index':len(state['reviews'])+1, 'budget_seconds':allowance,
              'deadline_epoch':time.time()+allowance, 'reservation_id':reservation['reservation_id'], 'status':'reviewing'}
    state['reviews'].append(review)
    state.update(stalled_slates=0, review_due=False)
    write_json(run/STATE,state)
    return {'review':review, 'slate':reservation, 'regular_cost':cost}


def finish_review(run, index, status, **details):
    state = load_state(run)
    state['reviews'][index-1].update(status=status, **details)
    write_json(run/STATE,state)


def cancel_expansion(run, reservation_id, reason):
    settle_reservation(run, reservation_id, outcome='released', reason=reason)
    state = load_state(run)
    if reservation_id not in state['cancelled_reservations']:
        state['cancelled_reservations'].append(reservation_id)
    state['probes'].setdefault(reservation_id, {}).update(status='cancelled', reason=reason)
    write_json(run/STATE,state)


def confirm_probe_cost(run, admission, probe):
    cost = admission['regular_cost']
    required = (cost['implementation_seconds']+cost['screening_seconds']+2*cost['planning_seconds']
                +probe['implementation_seconds']+probe['screening_seconds'])
    rid = admission['slate']['reservation_id']
    current = next(r for r in outstanding_reservations(run)['rows'] if r['reservation_id']==rid)
    extra = None
    if required > current['seconds']:
        extra = reserve_budget(run, label='space_probe_slate', seconds=required-current['seconds'], note='concrete probe cost')
    # Revalidate existing promises against the clock as well as any top-up.
    holds = outstanding_reservations(run)
    if holds['seconds'] > time_budget(run)['usable_seconds']:
        if extra:
            settle_reservation(run, extra['reservation_id'], outcome='released')
        raise BudgetReservationDenied('insufficient_time', 'concrete slate no longer fits')
    return {'reservation_id':rid, 'additional_reservation_ids':[extra['reservation_id']] if extra else [],
            'estimated_seconds':required, 'regular_cost':cost, 'review_index':admission['review']['index']}


def bind_probe(run, manifest):
    slots = [slot for slot in manifest.get('slate', []) if slot.get('seat_type') == 'space_probe']
    if not slots:
        return None
    slot = slots[0]
    admission = slot['space_probe_binding']['admission']
    state = load_state(run)
    probe = state['probes'].setdefault(admission['reservation_id'],
        {'status':'admitted', 'generations':[], 'aborted_generations':[]})
    probe.setdefault('generations', [])
    probe.setdefault('aborted_generations', [])
    gid = manifest['generation_id']
    if gid not in probe['generations']:
        probe['generations'].append(gid)
    probe['run_id'] = slot['run_id']
    write_json(run/STATE,state)
    return admission


def abort_probe_generation(run, manifest):
    admission = bind_probe(run, manifest)
    if admission is None:
        return
    state = load_state(run)
    rid, gid = admission['reservation_id'], manifest['generation_id']
    probe = state['probes'][rid]
    if gid in probe['aborted_generations'] or rid in state['cancelled_reservations']:
        return
    probe['aborted_generations'].append(gid)
    probe['status'] = 'deferred'
    write_json(run/STATE,state)
    holds = outstanding_reservations(run)
    usable = time_budget(run)['usable_seconds']
    if len(probe['aborted_generations']) >= 2 or (usable is not None and holds['seconds'] > usable):
        reason = 'repeated plan abort' if len(probe['aborted_generations']) >= 2 else 'deferred slate no longer fits'
        for reservation_id in [rid, *admission.get('additional_reservation_ids', [])]:
            cancel_expansion(run, reservation_id, reason)


def reconcile_probes(run):
    """Rebuild generation bindings after an interrupted plan/abort boundary."""
    if not (run/'.semantic/space-revisions.json').exists():
        return
    for path in sorted((run/'.semantic').glob('gen-*/generation.json')):
        manifest = json.loads(path.read_text())
        if (path.parent/'generation.aborted.json').exists():
            abort_probe_generation(run, manifest)
        else:
            bind_probe(run, manifest)
    ledger_path = run/'ledger.json'
    if not ledger_path.exists():
        return
    records = json.loads(ledger_path.read_text()).get('records', [])
    attempts = _read_budget_rows(run)
    state = load_state(run)
    for rid, probe in state['probes'].items():
        if rid in state['cancelled_reservations']:
            continue
        record = next((r for r in records if r.get('run_id') == probe.get('run_id')
            and (r.get('policy_receipt') or {}).get('generation_id') in probe.get('generations', [])), None)
        if record is None:
            continue
        probe['implemented'] = (run/'candidates'/record['run_id']/'train.py').exists()
        probe['objective_attempts'] = sum(r.get('kind') == 'score_attempt' and r.get('run_id') == record['run_id'] for r in attempts)
        probe['valid_result'] = record.get('status') in {'keep', 'discard'}
        if record.get('status') in LIFECYCLE_TERMINAL_STATUSES:
            probe['status'] = 'evaluated' if probe['objective_attempts'] else 'unevaluated'
    write_json(run/STATE,state)


def release_orphan_reservations(run):
    """Settle outstanding reservations no review or published probe still promises.

    The append-only attempt log is the authority for promised capacity; the
    intent state is written after its log row. A crash inside that window
    leaves a reservation nothing will ever drain, so recovery releases it.
    """
    run = Path(run)
    expected = set()
    for review in load_state(run)['reviews']:
        if review.get('status') == 'reviewing':
            expected.add(review['reservation_id'])
            expected.update(review.get('additional_reservation_ids', []))
    pending = pending_expansion(run)
    if pending is not None:
        admission = pending['admission']
        expected.add(admission['reservation_id'])
        expected.update(admission.get('additional_reservation_ids', []))
    orphans = [row['reservation_id'] for row in outstanding_reservations(run)['rows']
               if row['reservation_id'] not in expected]
    for rid in orphans:
        settle_reservation(run, rid, outcome='released', reason='orphaned reservation recovery')
    if orphans:
        state = load_state(run)
        state['cancelled_reservations'].extend(
            rid for rid in orphans if rid not in state['cancelled_reservations'])
        write_json(run/STATE, state)
    return orphans
