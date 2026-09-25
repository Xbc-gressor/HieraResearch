"""Optional bounded space review at a settled slate boundary."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from driver.session import InvocationFailed
from . import background_audit  # installs tools/ for script-style helper imports
import space_expansion as expansion
from evaluation_budget import BudgetAdmissionDenied, activate_reservation
from space_review import build_review_material, validate_review, validate_retrieval_request
from space_revisions import apply_expansion, publish_expansion, load_registry_history
from background_contract import load_registry, validate_registry
from semantic_space import (resolve_dimension_catalog, resolve_dimension_strategy,
    space_receipt, space_revision)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _release(run, admission, reason):
    for rid in [admission['reservation_id'], *admission.get('additional_reservation_ids', [])]:
        expansion.cancel_expansion(run, rid, reason)


def recover(run, *, cmd, repo_root, events):
    """Finish published registry→ledger synchronization before any admission."""
    if (run/'.semantic/space-revisions.json').exists():
        cmd(['python','tools/ledger.py','sync-space','--ledger',run/'ledger.json',
             '--background',run/'background.md'],repo_root)
    if not (run/expansion.STATE).exists() and not (run/'.semantic/space-revisions.json').exists():
        _release_orphans(run,events)
        return
    expansion.reconcile_probes(run)
    state=expansion.load_state(run)
    for review in state['reviews']:
        if review['status']=='reviewing':
            pending=expansion.pending_expansion(run)
            if pending and pending['admission']['reservation_id']==review['reservation_id']:
                expansion.finish_review(run,review['index'],'expanded')
            else:
                _release(run,review,'interrupted review')
                expansion.finish_review(run,review['index'],'cancelled',reason='interrupted review')
                events.emit('space_review_cancelled',review_index=review['index'],reason='interrupted review')
    _release_orphans(run,events)


def _release_orphans(run,events):
    orphans=expansion.release_orphan_reservations(run)
    if orphans:
        events.emit('space_reservation_orphaned',reservation_ids=orphans)


def activate_pending(run, *, events) -> bool:
    pending=expansion.pending_expansion(run)
    if pending is None:
        return False
    try:
        for rid in [pending['admission']['reservation_id'], *pending['admission'].get('additional_reservation_ids',[])]:
            activate_reservation(run,rid)
    except BudgetAdmissionDenied as exc:
        _release(run,pending['admission'],str(exc))
        events.emit('space_probe_cancelled',reason=str(exc),space=pending['space'])
        return False
    events.emit('space_probe_activated',space=pending['space'],admission=pending['admission'])
    return True


def _audit_delta(registry,new,run,deadline,invoke,runner,store,task,tag,folder):
    prior={background_audit._identity_key(m) for m in background_audit.collect_claim_mappings(registry)}
    mappings=[m for m in background_audit.collect_claim_mappings(new)
              if background_audit._identity_key(m) not in prior]
    if not mappings:
        return
    manifest=_read(run/'background_retrieval.json')
    entries=background_audit.build_audit_entries(new,manifest,run,mappings)
    verdicts=[]
    for batch in background_audit.pack_batches(entries):
        payload=background_audit.build_judge_payload(batch)
        receipt,_=invoke(runner,store,background_audit.JUDGE_ROLE,task,tag,run,
                         inline_payload=payload,deadline_epoch=deadline)
        errors=background_audit.verdict_errors(receipt,[entry['label'] for entry in batch])
        if errors or any(v['verdict']!='faithful' for v in receipt.get('verdicts',[])):
            raise ValueError('new citation faithfulness audit failed: '+str(errors or receipt))
        verdicts.append(receipt)
    expansion.write_json(folder/'faithfulness.json',{'batches':verdicts})


def targeted_retrieval(run, repo_root, request, deadline) -> str:
    """One adapter search using this run's existing retrieval surface."""
    path=run/'background_retrieval.json'
    manifest=_read(path)
    calls=[call for round_ in manifest.get('rounds', []) for call in round_.get('backend_calls', [])]
    names={call.get('backend') for call in calls}
    argv=[sys.executable,str(repo_root/'tools/search_backends.py'),'search',
          '--manifest',str(path),'--max-results','12']
    if 'frozen' in names:
        corpora={call.get('metadata', {}).get('corpus_path') for call in calls
                 if call.get('backend')=='frozen'} - {None}
        if len(corpora)!=1:
            raise ValueError('targeted retrieval needs the retained frozen corpus path')
        argv.extend(['--frozen-corpus',next(iter(corpora))])
    else:
        for backend in sorted(names.intersection({'deepxiv','jina-search'}) or {'deepxiv'}):
            argv.extend(['--backend',backend])
    for query in request['queries']:
        argv.extend(['--query-spec',json.dumps({'text':query,'target_dimension_ids':[],
                                               'evidence_roles':['hypothesis']})])
    result=subprocess.run(argv,cwd=repo_root,capture_output=True,text=True,
        timeout=max(.01,deadline-time.time()),check=True)
    latest=_read(path)['rounds'][-1]
    cards=[{**row,'snippet':str(row.get('snippet',''))[:4000]} for row in latest['results'][:12]]
    return result.stdout+'\n'+json.dumps(cards,ensure_ascii=False,indent=2)


def review_boundary(runner,store,task,tag,run,repo_root,cmd,events,*,invoke,job_runner):
    cfg=expansion.config(run)
    if not cfg['enabled'] or not (run/'ledger.json').exists():
        return
    framework=_read(run/'framework_cfg.json')
    if framework.get('semantic_search',{}).get('policy','judged_slate')!='judged_slate':
        return
    ledger=_read(run/'ledger.json')
    # A frozen but not yet admitted slate must finish before changing its space.
    admitted={(r.get('policy_receipt') or {}).get('generation_id') for r in ledger['records']}
    for path in (run/'.semantic').glob('gen-*/generation.json'):
        if _read(path)['generation_id'] not in admitted and not (path.parent/'generation.aborted.json').exists():
            return
    state=expansion.observe_boundary(run,ledger)
    if not state['review_due']:
        return
    try:
        admission=expansion.reserve_review(run)
    except BudgetAdmissionDenied as exc:
        events.emit('space_review_denied',reason=str(exc))
        return
    index=admission['review']['index']; deadline=admission['review']['deadline_epoch']
    folder=run/'.semantic'/f'review-{index:02d}'; folder.mkdir(parents=True,exist_ok=True)
    output=folder/'review.json'
    background=run/'background.md'
    published=False; binding={'reservation_id':admission['slate']['reservation_id']}
    request=None
    try:
        material=build_review_material(background,ledger=ledger,
            goal=_read(run/'objective_brief.json') if (run/'objective_brief.json').exists() else None,
            remaining_seconds=expansion.time_budget(run)['usable_seconds'])
        material['admitted_review_seconds']=admission['review']['budget_seconds']
        expansion.write_json(folder/'material.json',material)
        extra={'material':str(folder/'material.json'),'review_output':str(output)}
        if cfg['targeted_retrieval']:
            extra['retrieval_request_output']=str(folder/'retrieval-request.json')
        retrieved=False
        for attempt in range(2):
            try:
                receipt,_=invoke(runner,store,'space-reviewer',task,tag,run,extra=extra,deadline_epoch=deadline)
                if receipt.get('retrieval_request'):
                    if retrieved or not cfg['targeted_retrieval']:
                        raise ValueError('targeted retrieval unavailable or already used')
                    path=folder/'retrieval-request.json'
                    if Path(receipt['retrieval_request']).resolve()!=path.resolve():
                        raise ValueError('retrieval request must use supplied output path')
                    retrieval=_read(path)
                    errors=validate_retrieval_request(retrieval)
                    if errors:
                        raise ValueError('; '.join(errors))
                    retrieved=True
                    result=targeted_retrieval(run,repo_root,retrieval,deadline)
                    (folder/'retrieval-results.txt').write_text(result)
                    extra['retrieval_results']=str(folder/'retrieval-results.txt')
                    retrieved=True
                    receipt,_=invoke(runner,store,'space-reviewer',task,tag,run,extra=extra,deadline_epoch=deadline)
                if Path(receipt.get('review') or '').resolve()!=output.resolve():
                    raise ValueError('review receipt must name supplied output path')
                review=_read(output)
                registry=load_registry(background)
                catalog=resolve_dimension_catalog(background); strategy=resolve_dimension_strategy(background)
                errors=validate_review(review,registry,ledger=ledger,catalog=catalog,
                    dimension_strategy=strategy,registry_history=load_registry_history(background))
                if errors:
                    raise ValueError('; '.join(errors))
                if review['decision']=='expand':
                    new,new_catalog=apply_expansion(registry,review['delta'],catalog=catalog,
                        dimension_strategy=strategy,records=ledger['records'])
                    # Mirror sync-space's post-publication check exactly: once
                    # published the revision is immutable and a ledger mismatch
                    # would block the run forever, so it must fail here instead.
                    # The pre-expansion registry is part of the history view —
                    # after publication it becomes versions[-2].
                    errors=validate_registry(new,catalog=new_catalog,dimension_strategy=strategy,
                        ledger={**ledger,'search_space':space_receipt(new)},
                        registry_history={**load_registry_history(background),
                                          space_revision(registry):registry},
                        retrieval_manifest=_read(run/'background_retrieval.json'),manifest_dir=run,
                        manifest_path=run/'background_retrieval.json',number_gate=True)
                    if errors:
                        raise ValueError('; '.join(errors))
                    _audit_delta(registry,new,run,deadline,invoke,runner,store,task,tag,folder)
                break
            except (InvocationFailed, ValueError, TypeError, KeyError) as exc:
                if attempt or time.time()>=deadline:
                    raise
                extra['validation_errors']=str(exc)
        if time.time()>=deadline:
            raise ValueError('admitted review deadline reached')
        if review['decision']=='expand':
            binding=expansion.confirm_probe_cost(run,admission,review['probe'])
            expansion.finish_review(run,index,'reviewing',additional_reservation_ids=binding['additional_reservation_ids'])
            published_receipt=publish_expansion(background,review,ledger=ledger,admission=binding)
            published=True
            cmd(['python','tools/ledger.py','sync-space','--ledger',run/'ledger.json',
                 '--background',background],repo_root)
            expansion.finish_review(run,index,'expanded',space=published_receipt['space'])
            events.emit('space_expanded',review_index=index,**published_receipt)
        else:
            _release(run,binding,'review continued in current space')
            expansion.finish_review(run,index,'continue',reason=review['reason'])
            events.emit('space_review_continued',review_index=index,reason=review['reason'])
    except (InvocationFailed,ValueError,TypeError,KeyError,OSError,subprocess.SubprocessError,BudgetAdmissionDenied) as exc:
        # A publication is immutable. Once written, sync failure belongs to
        # recovery and cannot silently leave the run searching the old receipt.
        if published:
            raise
        _release(run,binding,str(exc))
        expansion.finish_review(run,index,'failed',reason=str(exc))
        events.emit('space_review_failed',review_index=index,reason=str(exc))


def cancel_pending(run, reason):
    pending = expansion.pending_expansion(run)
    if pending:
        _release(run, pending['admission'], reason)
