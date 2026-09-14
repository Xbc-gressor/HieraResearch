from tools.mlebench_official import validate_submission, run_official_grade

def test_submission_validation(tmp_path):
    p=tmp_path/'candidate'/'submission.csv'; p.parent.mkdir(); p.write_text('id,label\na,x\n')
    assert validate_submission(p)['rows']==1

def test_missing_grader_is_structured_official_failure(tmp_path):
    p=tmp_path/'candidate'/'submission.csv'; p.parent.mkdir(); p.write_text('id,label\na,x\n')
    r=run_official_grade(task_id='t',candidate_id='c',submission=p,grader=['missing-grader'],reporting_root=tmp_path/'reports',input_revision='i')
    assert r.stage=='official' and r.selection_visible is False and r.failure_kind=='preflight'
