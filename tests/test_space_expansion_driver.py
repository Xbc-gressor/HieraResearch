"""Runtime review, capacity and retry boundaries; no SDK or network."""
import json
import time
from pathlib import Path

import pytest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))


def run_fixture(tmp_path, **overrides):
    run = tmp_path/'runs'/'task'/'cell'
    run.mkdir(parents=True)
    cfg = {'deadline': time.time()+10000, 'max_evaluations': 20,
           'evaluation_timeout_mode': 'run_budget', 'per_runtime_limit': None,
           'space_expansion': {'enabled': True}, **overrides}
    (run/'framework_cfg.json').write_text(json.dumps(cfg))
    # Seat duration includes implementation and screening, with objective
    # duration recorded separately by the real evaluator.
    (run/'driver_events.jsonl').write_text(json.dumps({'kind':'seat_finished', 'seconds':80})+'\n'+json.dumps({'kind':'slate_planning_finished','seconds':20})+'\n')
    (run/'evaluation_attempts.jsonl').write_text(json.dumps({
        'schema_version':1, 'kind':'score_completion', 'attempt_id':'eval-000001',
        'run_id':'000', 'duration_seconds':10})+'\n')
    return run


def test_completed_slate_progress_includes_optimization_and_separates_domains(tmp_path):
    from space_expansion import observe_boundary
    run = run_fixture(tmp_path)
    ledger = {'records': []}
    def complete(n, score, domain='proxy'):
        for seat in range(2):
            ledger['records'].append({'run_id':f'{n*2+seat:03d}', 'status':'keep',
                'policy_receipt':{'generation_id':f'g{n}'}, 'final_best_score':score,
                'evaluation_domain':{'split': domain}})
        return observe_boundary(run, ledger)
    assert complete(0, .4)['stalled_slates'] == 0
    assert complete(1, .5)['stalled_slates'] == 1
    # The same ledger scalar is updated by production rewrite/tune settlement.
    ledger['records'][0]['final_best_score'] = .3
    assert complete(2, .5)['stalled_slates'] == 0
    assert complete(3, .5)['stalled_slates'] == 1
    assert complete(4, .1, 'other')['review_due']
    assert observe_boundary(run, ledger)['stalled_slates'] == 2


def test_stall_counts_after_first_optimization_and_doubles_after_expansion(tmp_path):
    from space_expansion import observe_boundary, load_state, write_json, STATE
    from scheduler.round_policy import load_round_state, save_round_state
    run = run_fixture(tmp_path, tuner={'scheduler_policy': 'round_v1'})
    ledger = {'records': []}
    def complete(n):
        ledger['records'].append({'run_id':f'{n:03d}', 'op':'fresh', 'status':'keep',
            'policy_receipt':{'generation_id':f'g{n}'}, 'final_best_score':.5})
        return observe_boundary(run, ledger)
    complete(0)
    assert complete(1)['stalled_slates'] == 0
    state = load_round_state(run, for_update=True)
    state['cycle'] = 1
    save_round_state(run, state)
    complete(2)
    assert complete(3)['review_due']
    state = load_state(run)
    state.update(reviews=[{'status':'expanded'}], stalled_slates=0)
    write_json(run/STATE, state)
    assert not any(complete(n)['review_due'] for n in (4, 5, 6))
    assert complete(7)['review_due']
    assert [(row['run_id'], row['source']) for row in load_state(run)['best_history']] == [('000', 'fresh')]


def test_review_share_charges_actual_wall_clock(tmp_path):
    from space_expansion import reserve_review, finish_review, cancel_expansion, load_state
    run = run_fixture(tmp_path, space_expansion={'enabled': True, 'review_seconds': 300})
    first = reserve_review(run)
    assert first['review']['budget_seconds'] == 300
    cancel_expansion(run, first['slate']['reservation_id'], 'continue')
    finish_review(run, 1, 'continue')
    assert load_state(run)['reviews'][0]['elapsed_seconds'] < 300
    assert reserve_review(run)['review']['budget_seconds'] == 300


def test_review_reserves_complete_slate_without_eval_timeout(tmp_path):
    from space_expansion import reserve_review, cancel_expansion
    from evaluation_budget import outstanding_reservations, BudgetReservationDenied
    run = run_fixture(tmp_path)
    admission = reserve_review(run)
    assert admission['slate']['reserved_evaluations'] >= 4
    assert outstanding_reservations(run)['evaluations'] >= 4
    cancel_expansion(run, admission['slate']['reservation_id'], 'invalid proposal')
    assert outstanding_reservations(run)['evaluations'] == 0
    cfg = json.loads((run/'framework_cfg.json').read_text())
    cfg['deadline'] = time.time()+2
    (run/'framework_cfg.json').write_text(json.dumps(cfg))
    with pytest.raises(BudgetReservationDenied):
        reserve_review(run)


def proposal(run):
    from background_contract import load_registry
    from space_revisions import apply_expansion
    from semantic_space import complete_point, space_revision
    from tests.test_space_expansion import synthesis
    old = load_registry(run/'background.md')
    delta = {'hypotheses': {'dim-model-architecture': [synthesis()]}}
    new, _ = apply_expansion(old, delta)
    return {'decision':'expand', 'base_revision':space_revision(old),
        'reason':'Existing routes have stopped closing the goal gap', 'basis':['runtime observations'],
        'delta':delta, 'probe':{'point':complete_point(new, {'dim-model-architecture':'hyp-model-transfer'}),
            'op':'fresh', 'parents':[], 'implementation_seconds':20, 'screening_seconds':30,
            'expected_observation':'A measured transfer score'}}


def test_review_publication_flows_through_normal_judged_generation(tmp_path, monkeypatch):
    from driver.loops import experiment, space_reviews
    from driver.events import EventsLog
    from driver.receipts import ReceiptStore
    from driver.session import FakeSessionRunner
    from tests.test_driver_judged_slate import (JudgedCmd, _ledger_data, write_judged_task,
        judge_entry, plan_entry, writer_entry, TASK, TAG)
    from tests.test_space_expansion import initial_run
    from tools.ledger import sync_space_revision
    from evaluation_budget import reserve_evaluation, record_evaluation_completion, outstanding_reservations
    from space_expansion import pending_expansion, load_state
    from space_review import build_review_material
    from tests.test_slate_replay import run_replay
    write_judged_task(tmp_path)
    run = tmp_path/'runs'/TASK/TAG
    run.mkdir(parents=True)
    template = run_fixture(tmp_path/'observations')
    for name in ('framework_cfg.json','driver_events.jsonl','evaluation_attempts.jsonl'):
        (run/name).write_bytes((template/name).read_bytes())
    _, registry = initial_run(run)
    ledger = _ledger_data(registry)
    (run/'ledger.json').write_text(json.dumps(ledger))
    cfg=json.loads((run/'framework_cfg.json').read_text())
    cfg.update(semantic_search={'policy':'judged_slate'}, max_evaluations=100)
    (run/'framework_cfg.json').write_text(json.dumps(cfg))
    monkeypatch.setattr(space_reviews.expansion,'observe_boundary',lambda *a:{'review_due':True})
    def invoke(*args, **kw):
        output=Path(kw['extra']['review_output'])
        output.write_text(json.dumps(proposal(run)))
        return {'review':str(output)}, 1
    class Cmd(JudgedCmd):
        def __call__(self,args,*a,**kw):
            if 'sync-space' in args:
                sync_space_revision(run/'ledger.json',run/'background.md')
                return None
            return super().__call__(args,*a,**kw)
    cmd=Cmd(tmp_path)
    events=EventsLog(run)
    store=ReceiptStore(run)
    space_reviews.review_boundary(None,store,TASK,TAG,run,tmp_path,cmd,events,invoke=invoke,job_runner=None)
    assert pending_expansion(run)
    assert outstanding_reservations(run)['evaluations'] >= 4
    runner=FakeSessionRunner([judge_entry(),judge_entry(),plan_entry(),plan_entry(),writer_entry(),writer_entry()])
    actions=experiment._evaluate_judged_generation(runner,store,TASK,TAG,run,0,tmp_path,cmd,events,'m',cmd.evaluate)
    assert len(actions)==2
    manifest=json.loads((run/'.semantic/gen-0001/generation.json').read_text())
    probe=next(slot for slot in manifest['slate'] if slot.get('seat_type')=='space_probe')
    assert outstanding_reservations(run)['evaluations']==0
    assert pending_expansion(run) is None
    # The fake evaluation backend does not append objective attempts. This is
    # deliberately not reported as a fulfilled probe merely because activation
    # and a synthetic score exist.
    rid=probe['space_probe_binding']['admission']['reservation_id']
    assert load_state(run)['probes'][rid]['status']=='unevaluated'
    ref=run/'candidates'/probe['run_id']/'train.py'
    receipt=reserve_evaluation(ref,params={},phase='warm',method='test')
    record_evaluation_completion(ref,attempt_id=receipt['attempt_id'],duration_seconds=1)
    space_reviews.expansion.reconcile_probes(run)
    assert load_state(run)['probes'][rid]['status']=='evaluated'
    packet=build_review_material(run/'background.md',ledger=json.loads((run/'ledger.json').read_text()))
    assert probe['run_id'] in json.dumps(packet['search'])
    code,report=run_replay(run)
    assert code==0,report


def test_probe_plan_abort_defers_once_and_recovery_is_idempotent(tmp_path):
    from tests.test_space_expansion import initial_run
    from space_revisions import publish_expansion
    from space_expansion import (reserve_review, confirm_probe_cost, abort_probe_generation,
        reconcile_probes, pending_expansion, load_state)
    from evaluation_budget import outstanding_reservations
    run=run_fixture(tmp_path)
    initial_run(run)
    ledger={'records':[]}
    (run/'ledger.json').write_text(json.dumps(ledger))
    review=proposal(run)
    admission=confirm_probe_cost(run,reserve_review(run),review['probe'])
    publish_expansion(run/'background.md',review,ledger=ledger,admission=admission)
    rid=admission['reservation_id']
    for n in (1,2):
        manifest={'generation_id':f'g{n}','slate':[{'run_id':'005','seat_type':'space_probe',
            'space_probe_binding':{'admission':admission}}]}
        folder=run/'.semantic'/f'gen-{n:04d}';folder.mkdir()
        (folder/'generation.json').write_text(json.dumps(manifest))
        (folder/'generation.aborted.json').write_text('{}')
        abort_probe_generation(run,manifest)
        reconcile_probes(run)
        assert len(load_state(run)['probes'][rid]['aborted_generations'])==n
        if n==1:
            assert pending_expansion(run) and outstanding_reservations(run)['evaluations'] > 0
    assert pending_expansion(run) is None
    assert outstanding_reservations(run)['evaluations']==0
    assert load_state(run)['probes'][rid]['status']=='cancelled'


@pytest.mark.parametrize('malformed',[False,True])
def test_optional_review_failure_is_bounded_without_publication(tmp_path,monkeypatch,malformed):
    from driver.loops import space_reviews
    from driver.events import EventsLog
    from driver.session import InvocationFailed
    from tests.test_space_expansion import initial_run
    from space_expansion import load_state
    from evaluation_budget import outstanding_reservations
    run=run_fixture(tmp_path)
    initial_run(run)
    (run/'ledger.json').write_text(json.dumps({'records':[]}))
    monkeypatch.setattr(space_reviews.expansion,'observe_boundary',lambda *a:{'review_due':True})
    attempts=[]
    def fail(*a,**kw):
        attempts.append(kw['deadline_epoch'])
        if malformed:
            out=Path(kw['extra']['review_output']);out.write_text('{"decision":"nonsense"}')
            return {'review':str(out)},1
        raise InvocationFailed('space-reviewer',['bad proposal'])
    space_reviews.review_boundary(None,None,'task','cell',run,Path('.'),None,EventsLog(run),invoke=fail,job_runner=None)
    assert len(attempts)==2 and attempts[0]==attempts[1]
    assert load_state(run)['reviews'][0]['status']=='failed'
    assert outstanding_reservations(run)['evaluations']==0
    assert not (run/'.semantic/space-revisions.json').exists()


def test_targeted_retrieval_uses_retained_backend_and_real_adapter(tmp_path):
    from driver.loops.space_reviews import targeted_retrieval
    from search_backends import new_manifest
    root=Path(__file__).resolve().parents[1]
    corpus=tmp_path/'prior.json'
    corpus.write_text(json.dumps({'schema_version':1,'corpus_id':'prior','cutoff':'2020-01-01',
        'created_at':'2019-01-01','provenance':'generic prior','prepared_before_task_ids':True,
        'items':[{'url':'https://example.org/transfer','title':'Transfer representations',
                  'abstract':'Transfer representation learning can improve limited label models.'}]}))
    manifest=new_manifest()
    # Bootstrap through the real adapter, keeping its exact frozen backend receipt.
    import subprocess
    path=tmp_path/'background_retrieval.json'
    path.write_text(json.dumps(manifest))
    subprocess.run([sys.executable,str(root/'tools/search_backends.py'),'search','--manifest',str(path),
        '--frozen-corpus',str(corpus),'--query-spec',json.dumps({'text':'representation',
        'target_dimension_ids':[],'evidence_roles':['hypothesis']})],check=True,capture_output=True)
    result=targeted_retrieval(tmp_path,root,{'queries':['transfer representations']},time.time()+30)
    assert 'Transfer representations' in result
    stored=json.loads(path.read_text())
    assert len(stored['rounds'])==2
    assert stored['rounds'][-1]['queries'][0]['evidence_roles']==['hypothesis']
    assert stored['rounds'][-1]['backend_calls'][0]['backend']=='frozen'



def test_expansion_cli_and_init_persist_independent_switch(tmp_path):
    from driver.__main__ import build_parser
    from init_run import initialize_run
    from tests.test_init_run import InitRunDimensionStrategyTests
    from space_expansion import config
    args=build_parser().parse_args(['run','toy','cell','--loop','experiment','--space-expansion','--no-eval-timeout'])
    assert args.space_expansion and args.no_eval_timeout
    InitRunDimensionStrategyTests()._repo(tmp_path)
    run=initialize_run(tmp_path,'toy','cell',space_expansion=True,no_eval_timeout=True,time_budget_seconds=2000)
    assert config(run)['enabled'] and not config(run)['targeted_retrieval']
    initialize_run(tmp_path,'toy','cell',space_expansion=False)
    assert not config(run)['enabled']


def test_promised_probe_bypasses_round_quota(tmp_path,monkeypatch):
    from driver.loops import experiment
    from driver.events import EventsLog
    from space_expansion import reserve_review, confirm_probe_cost
    from space_revisions import publish_expansion
    from tests.test_space_expansion import initial_run
    from scheduler.round_policy import load_round_state, save_round_state
    run=run_fixture(tmp_path)
    initial_run(run)
    ledger={'records':[]}
    (run/'ledger.json').write_text(json.dumps(ledger))
    review=proposal(run)
    admission=confirm_probe_cost(run,reserve_review(run),review['probe'])
    publish_expansion(run/'background.md',review,ledger=ledger,admission=admission)
    state=load_round_state(run,for_update=True)
    state.update(phase='optimize',phase_deadline=time.time()-1)
    save_round_state(run,state)
    monkeypatch.setattr(experiment.rounds,'status',lambda *a:{'generate':False,'state':state})
    def generate(*a,**kw):
        from evaluation_budget import phase_quota_remaining
        assert phase_quota_remaining(run) is None
        return [{'run_id':'001'}]
    monkeypatch.setattr(experiment,'_evaluate_generation',generate)
    actions,optimized=experiment._round_step(None,None,'task','cell',run,1,{},tmp_path,None,EventsLog(run),None,'m',ledger_exists=True)
    assert actions and not optimized


def test_review_rejects_runtime_pruned_assignments_before_publication(tmp_path):
    from tests.test_space_expansion import initial_run
    from background_contract import load_registry
    from space_review import validate_review
    from semantic_space import complete_point
    from space_revisions import apply_expansion
    from search_space_state import empty_search_space_state
    run=run_fixture(tmp_path)
    initial_run(run)
    review=proposal(run)
    registry=load_registry(run/'background.md')
    expanded,_=apply_expansion(registry,review['delta'])
    review['probe']['point']=complete_point(expanded,{'dim-model-architecture':'hyp-model-transfer','dim-data-curation':'hyp-data-filtered'})
    runtime=empty_search_space_state()
    runtime.update(revision=1,decisions=[{'revision':1,'target':{'kind':'hypothesis','id':'hyp-data-filtered'},'to_status':'pruned'}])
    errors=validate_review(review,registry,ledger={'records':[],'search_space_state':runtime})
    assert any('runtime-pruned' in error for error in errors)


def test_crash_between_reservation_and_state_leaves_no_orphan_hold(tmp_path):
    from driver.loops import space_reviews
    from driver.events import EventsLog
    from space_expansion import load_state
    from evaluation_budget import reserve_budget, outstanding_reservations
    run=run_fixture(tmp_path)
    # Crash inside reserve_review/confirm_probe_cost: durable reservation rows
    # exist, but the intent state referencing them was never written.
    main=reserve_budget(run,label='space_probe_slate',seconds=100,evaluations=6)
    top=reserve_budget(run,label='space_probe_slate',seconds=50,note='concrete probe cost')
    assert outstanding_reservations(run)['evaluations']==6
    space_reviews.recover(run,cmd=None,repo_root=tmp_path,events=EventsLog(run))
    assert outstanding_reservations(run)['evaluations']==0
    assert set(load_state(run)['cancelled_reservations'])=={
        main['reservation_id'],top['reservation_id']}


def test_recovery_keeps_published_probe_promise_and_releases_orphans(tmp_path):
    from driver.loops import space_reviews
    from driver.events import EventsLog
    from tests.test_space_expansion import initial_run
    from space_expansion import reserve_review, confirm_probe_cost, pending_expansion, load_state
    from space_revisions import publish_expansion
    from evaluation_budget import reserve_budget, outstanding_reservations
    from search_space_state import empty_search_space_state
    from tools.ledger import sync_space_revision
    run=run_fixture(tmp_path)
    initial_run(run)
    ledger={'records':[],'search_space_state':empty_search_space_state()}
    (run/'ledger.json').write_text(json.dumps(ledger))
    review=proposal(run)
    admission=confirm_probe_cost(run,reserve_review(run),review['probe'])
    publish_expansion(run/'background.md',review,ledger=ledger,admission=admission)
    orphan=reserve_budget(run,label='space_probe_slate',seconds=100,evaluations=4)
    def cmd(args,*a,**kw):
        assert 'sync-space' in args
        sync_space_revision(run/'ledger.json',run/'background.md')
    space_reviews.recover(run,cmd=cmd,repo_root=tmp_path,events=EventsLog(run))
    holds=outstanding_reservations(run)
    assert pending_expansion(run) is not None
    assert holds['evaluations']>0
    assert all(row['reservation_id']!=orphan['reservation_id'] for row in holds['rows'])
    assert orphan['reservation_id'] in load_state(run)['cancelled_reservations']


def test_ledger_incompatible_expansion_fails_before_publication(tmp_path,monkeypatch):
    from driver.loops import space_reviews
    from driver.events import EventsLog
    from tests.test_space_expansion import initial_run
    from tests.fixtures import record
    from semantic_space import complete_point, space_receipt
    from search_space_state import empty_search_space_state
    from space_expansion import load_state
    from evaluation_budget import outstanding_reservations
    run=run_fixture(tmp_path)
    _,registry=initial_run(run)
    parent=record('000','fresh',[],complete_point(registry),score=.4,status='keep')
    # A stale space_revision survives apply_expansion's choice projection but
    # fails the full-ledger check sync-space runs after the immutable publish.
    parent['semantic_point']['space_revision']='stale-registry'
    ledger={'records':[parent],'search_space':space_receipt(registry),
            'search_space_state':empty_search_space_state()}
    (run/'ledger.json').write_text(json.dumps(ledger))
    monkeypatch.setattr(space_reviews.expansion,'observe_boundary',lambda *a:{'review_due':True})
    def invoke(*a,**kw):
        out=Path(kw['extra']['review_output']);out.write_text(json.dumps(proposal(run)))
        return {'review':str(out)},1
    space_reviews.review_boundary(None,None,'task','cell',run,Path('.'),None,
        EventsLog(run),invoke=invoke,job_runner=None)
    assert load_state(run)['reviews'][0]['status']=='failed'
    assert not (run/'.semantic/space-revisions.json').exists()
    assert outstanding_reservations(run)['evaluations']==0
