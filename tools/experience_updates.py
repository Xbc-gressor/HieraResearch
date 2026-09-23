"""Incremental experience: explanations persist; evidence is a derived view.

No writes here. ledger.py owns attempt bookkeeping and atomic publication.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

if __package__:
    from .ledger_core import experience_receipt, search_space_state_revision
    from .semantic_space import selected_assignments, dimension_map, hypothesis_map
    from .semantic_evidence import (edge_index, render_target_evidence,
                                   comparator_coverage, target_evaluation_state)
else:
    from ledger_core import experience_receipt, search_space_state_revision
    from semantic_space import selected_assignments, dimension_map, hypothesis_map
    from semantic_evidence import (edge_index, render_target_evidence,
                                   comparator_coverage, target_evaluation_state)

COLLECTIONS = ('lessons', 'bottlenecks', 'promising_regions',
               'dimension_evidence', 'hypothesis_evidence')
META = {'id', 'target_ids', 'basis_dag_revision', 'judgment_generation'}
FACTS = {'evaluation_state', 'comparator_coverage'}


def binding(data):
    return {'dag_revision': data.get('dag_revision', 0),
            'experience': experience_receipt(data),
            'search_space_state_revision': search_space_state_revision(data)}


def targets(record):
    point = record.get('semantic_point')
    if not isinstance(point, dict) or not point:
        return set()
    selected = selected_assignments(point)
    return set(selected) | set(selected.values())


def related(item, target_ids, run_ids):
    return (bool(set(item.get('target_ids', [])) & target_ids)
            or item.get('target_id') in target_ids
            or bool(set(item.get('evidence', item.get('evidence_run_ids', []))) & run_ids))


def affected_targets(data, changed):
    """Include removed hypotheses and updated parent/control endpoints."""
    wanted = set().union(*(targets(r) for r in changed)) if changed else set()
    runs = {r['run_id'] for r in changed}
    records = {r['run_id']: r for r in data.get('records', [])}
    for edge in edge_index(data).values():
        if edge.get('parent_run_id') not in runs and edge.get('child_run_id') not in runs:
            continue
        for change in edge.get('changes', []):
            wanted.update(change.get(k) for k in ('dimension_id', 'from_hypothesis_id', 'to_hypothesis_id') if change.get(k))
        for ident in (edge.get('parent_run_id'), edge.get('child_run_id')):
            if ident in records:
                wanted.update(targets(records[ident]))
    return wanted


def related_view(data, point, run_ids=()):
    wanted = targets({'semantic_point': point})
    runs = set(run_ids)
    by_id = {r['run_id']: r for r in data.get('records', [])}
    pending = list(runs)
    while pending:
        for parent in by_id.get(pending.pop(), {}).get('source_run_ids', []):
            if parent not in runs:
                runs.add(parent)
                pending.append(parent)
    view = project_experience(data)
    return [{'collection': field, **item} for field in COLLECTIONS
            for item in view.get(field, []) if related(item, wanted, runs)]


def item_is_current(item, data):
    """An unchanged interpretation cannot be promoted by another item's update."""
    basis = item.get('basis_dag_revision')
    if basis is None:
        return True  # schema-4 helper fixtures
    cited = set(item.get('evidence', item.get('evidence_run_ids', [])))
    watched = set(item.get('target_ids', [])) | {item.get('target_id')}
    index = edge_index(data)
    for edge_id in item.get('evidence_edge_ids', []):
        edge = index.get(edge_id, {})
        cited.update((edge.get('parent_run_id'), edge.get('child_run_id')))
    changed = [r for r in data.get('records', []) if r.get('dag_revision', 0) > basis]
    return not (any(r['run_id'] in cited for r in changed)
                or bool(affected_targets(data, changed) & watched))


def project_experience(data, *, actionable_only=False):
    """Consumer view with latest mechanical facts; authored judgments untouched."""
    experience = copy.deepcopy(data.get('experience') or {})
    for field, kind in [('dimension_evidence', 'dimension'),
                        ('hypothesis_evidence', 'hypothesis')]:
        if actionable_only:
            experience[field] = [item for item in experience.get(field, [])
                                 if item_is_current(item, data)]
        for item in experience.get(field, []):
            item['comparator_coverage'] = comparator_coverage(
                data, item.get('evidence_edge_ids', []), target_kind=kind,
                target_id=item['target_id'])
            item['evaluation_state'] = target_evaluation_state(
                data, target_kind=kind, target_id=item['target_id'],
                evidence_run_ids=item.get('evidence_run_ids', []),
                evidence_edge_ids=item.get('evidence_edge_ids', []))
    return experience


def validation_snapshot(data):
    experience = project_experience(data)
    experience['schema_version'] = 4
    for field in COLLECTIONS:
        experience[field] = [{k: v for k, v in item.items() if k not in META}
                             for item in experience.get(field, [])]
    return experience


def build_context(data, registry, *, target_ids=None, entry_ids=None):
    experience = data.get('experience') or {}
    cursor = experience.get('dag_revision', 0)
    changed = [r for r in data.get('records', [])
               if r.get('dag_revision', 0) > cursor]
    wanted = set(target_ids or [])
    wanted.update(affected_targets(data, changed))
    runs = {r['run_id'] for r in changed}
    entries = []
    for field in COLLECTIONS:
        for item in experience.get(field, []):
            if related(item, wanted, runs) or item.get('id') in (entry_ids or []):
                entries.append({'collection': field, **copy.deepcopy(item)})
    if entry_ids and set(entry_ids) - {e.get('id') for e in entries}:
        raise ValueError('unknown experience entry id')
    definitions = {**dimension_map(registry), **{key: value[1] for key, value in hypothesis_map(registry).items()}}
    evidence = render_target_evidence(registry, data, target_ids=sorted(wanted))
    edge_facts = {}
    for block in evidence['dimension_targets'] + evidence['hypothesis_targets']:
        for edge in block.pop('edges'):
            edge_facts[edge['edge_id']] = edge
    evidence['edges_by_id'] = edge_facts
    return {'snapshot': binding(data), 'since_dag_revision': cursor,
            'target_definitions': [{k: definitions[t].get(k) for k in
                ('id', 'title', 'definition', 'claim', 'boundary', 'testable_expectation')}
                for t in sorted(wanted)],
            'changes': [{k: r.get(k) for k in (
                'run_id', 'dag_revision', 'status', 'source_run_ids', 'semantic_point',
                'final_best_score', 'best_warm_score', 'evaluation_depth',
                'trials_attempted', 'elapsed_seconds')} for r in changed],
            'target_evidence': evidence,
            'entries': entries}


def publication_boundary(run_dir, data):
    records = data.get('records', [])
    if not records or any(r.get('status') not in
                          {'keep', 'discard', 'crash', 'unevaluated', 'aborted'} for r in records):
        return False
    # A provisional pool already pins experience before admission creates rows.
    by_id = {r['run_id']: r for r in records}
    for directory in (Path(run_dir) / '.semantic').glob('gen-*'):
        if (directory / 'generation.aborted.json').exists():
            continue
        manifest = directory / 'generation.json'
        if manifest.exists():
            doc = json.loads(manifest.read_text())
            for seat in doc['slate']:
                record = by_id.get(seat['run_id'])
                receipt = (record or {}).get('policy_receipt', {})
                if receipt.get('generation_id') != doc['generation_id']:
                    return False
        elif any(directory.iterdir()):
            return False
    return True


def merge_patch(data, registry, context, patch):
    """Pure, all-or-nothing merge. Empty updates acknowledge only the cursor."""
    from background_contract import validate_experience
    if context.get('snapshot') != binding(data):
        raise ValueError('experience snapshot changed since input preparation')
    if not isinstance(patch, dict) or set(patch) != {'updates'} or not isinstance(patch['updates'], list):
        raise ValueError('patch must contain an updates list')
    result = copy.deepcopy(data)
    prior = result.get('experience') or {}
    experience = copy.deepcopy(prior) if prior else {
        'schema_version': 5, 'generation': 0, 'dag_revision': 0, 'summary': '',
        'updated_at_run': None, **{field: [] for field in COLLECTIONS}}
    if experience.get('schema_version') != 5:
        raise ValueError('incremental publication requires a new-run schema-5 experience')
    if data['dag_revision'] <= experience['dag_revision']:
        raise ValueError('no unprocessed terminal DAG delta')
    table = {item['id']: (field, item) for field in COLLECTIONS for item in experience[field]}
    next_id = int(result.get('experience_update', {}).get('next_id', 1))
    generation = experience['generation'] + bool(patch['updates'])
    known_targets = set(dimension_map(registry)) | set(hypothesis_map(registry))
    changed_ids = set()
    for update in patch['updates']:
        if not isinstance(update, dict) or update.get('op') not in ('upsert', 'delete'):
            raise ValueError('unknown experience patch operation')
        op = update['op']
        allowed = {'op', 'id'} if op == 'delete' else {'op', 'id', 'collection', 'value'}
        if set(update) - allowed:
            raise ValueError('unknown patch fields')
        ident = update.get('id')
        if ident is not None and not isinstance(ident, str):
            raise ValueError('entry id must be a string')
        if ident is not None and ident not in table:
            raise ValueError('unknown experience entry id')
        if ident in changed_ids:
            raise ValueError('multiple updates for the same entry')
        if op == 'delete':
            if ident is None:
                raise ValueError('delete requires an id')
            field, old = table.pop(ident)
            experience[field].remove(old)
            changed_ids.add(ident)
            continue
        field = update.get('collection')
        item = copy.deepcopy(update.get('value'))
        if field not in COLLECTIONS or not isinstance(item, dict):
            raise ValueError('invalid experience collection/value')
        if field.endswith('_evidence') and not isinstance(item.get('target_id'), str):
            raise ValueError('target entry needs a target_id')
        if (set(item) & (META - {'target_ids'})) or set(item) & FACTS:
            raise ValueError('metadata and mechanical fields are helper-owned')
        for key in ('assessment', 'recommended_status', 'confidence', 'kind'):
            if key in item and not isinstance(item[key], str):
                raise ValueError(f'{key} must be a string')
        for key in ('evidence', 'evidence_run_ids', 'evidence_edge_ids'):
            if key in item and (not isinstance(item[key], list) or any(not isinstance(v, str) for v in item[key])):
                raise ValueError(f'{key} must be a list of string IDs')
        scope = item.get('target_ids')
        if not isinstance(scope, list) or any(not isinstance(t, str) or t not in known_targets for t in scope):
            raise ValueError('target_ids must reference registered targets')
        if ident:
            old_field, old = table[ident]
            if old_field != field:
                raise ValueError('an entry cannot change collection')
            experience[field][experience[field].index(old)] = item
        else:
            ident = f'experience-{next_id}'
            next_id += 1
            experience[field].append(item)
        item.update(id=ident, basis_dag_revision=data['dag_revision'], judgment_generation=generation)
        table[ident] = (field, item)
        changed_ids.add(ident)
    experience.update(dag_revision=data['dag_revision'], generation=generation)
    terminal = [r for r in data['records'] if r.get('status') in {'keep', 'discard', 'crash'}]
    if terminal:
        experience['updated_at_run'] = max(terminal, key=lambda r: r.get('dag_revision', 0))['run_id']
    result['experience'] = experience
    errors = validate_experience(experience, registry, result)
    # Apply conservative judgment gates only to newly authored entries, against
    # current evidence. Untouched explanations remain historical interpretations.
    from background_contract import _validate_target_evidence
    view = validation_snapshot(result)
    for field, kind in [('dimension_evidence','dimension'), ('hypothesis_evidence','hypothesis')]:
        selected = [projected for stored, projected in zip(experience[field], view[field])
                    if stored['id'] in changed_ids]
        errors += _validate_target_evidence(selected, field=field, target_kind=kind,
            registry=registry, ledger=result, limit=len(selected))
    if errors:
        raise ValueError('; '.join(errors))
    result['experience_update'] = {**result.get('experience_update', {}),
        'attempted_dag_revision': data['dag_revision'], 'status': 'published',
        'error': None, 'next_id': next_id}
    return result
