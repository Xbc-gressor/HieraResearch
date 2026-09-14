from pathlib import Path
from tools.task_evaluator import EvaluationContract, EvaluationRunner

class E:
    @staticmethod
    def run(artifact, params, **kwargs):
        return {'expected_holdout_ids':['a','b'], 'holdout_predictions':{'a':1,'b':2}, 'score':.5}

def test_runner_emits_record():
    c=EvaluationContract('toy','1','loss','min',{'run':'run'}, {})
    r=EvaluationRunner(c,evaluator_module=E).evaluate(Path('.'),candidate_id='c',input_revision='i',stage='run')
    assert r.score == .5 and r.output_artifact_digest

def test_runner_persists_operator_output(tmp_path):
    c=EvaluationContract('toy','1','loss','min',{'run':'run'}, {})
    r=EvaluationRunner(c,evaluator_module=E,reporting_root=tmp_path).evaluate(Path('.'),candidate_id='c',input_revision='i',stage='run')
    assert Path(r.output_artifact).is_file()

def test_official_record_is_hidden(tmp_path):
    c=EvaluationContract('toy','1','loss','min',{'official':'run'}, {})
    r=EvaluationRunner(c,evaluator_module=E).evaluate(Path('.'),candidate_id='c',input_revision='i',stage='official')
    assert r.selection_visible is False

def test_legacy_adapter_requires_explicit_make_model():
    class Legacy:
        @staticmethod
        def run(make_model, params): return 1.0
    c=EvaluationContract('toy','1','loss','min',{'run':'run'}, {})
    r=EvaluationRunner(c,evaluator_module=Legacy).evaluate(Path('.'),candidate_id='c',input_revision='i',stage='run',params={'make_model':lambda *x:None,'expected_holdout_ids':[],'holdout_predictions':{}})
    assert r.score == 1.0
