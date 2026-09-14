import pytest
from tools.evaluation_records import EvaluationRecord, append_record, read_records, selection_view

def test_record_roundtrip_and_same_domain_projection(tmp_path):
    path = tmp_path / "evaluation.jsonl"
    rec = EvaluationRecord(task_id="toy", candidate_id="c1", input_revision="r1", output_artifact_digest="o1", contract_version="1", stage="proxy", fidelity="fast", metric_name="loss", score=0.2)
    digest = append_record(path, rec)
    rows = read_records(path)
    assert rows[0]["record_digest"] == digest
    assert selection_view(rows, stage="proxy", fidelity="fast")[0]["score"] == 0.2
    assert selection_view(rows, stage="protocol", fidelity="full") == []

def test_official_is_operator_only():
    with pytest.raises(ValueError):
        EvaluationRecord(task_id="t", candidate_id="c", input_revision="r", output_artifact_digest="o", contract_version="1", stage="official", fidelity="full", metric_name="x", score=1)

def test_scheduler_domain_projection_does_not_mix_stages():
    from tools.scheduler.state import candidate_score
    rows = [EvaluationRecord(task_id='t', candidate_id='c', input_revision='r', output_artifact_digest='o', contract_version='1', stage='proxy', fidelity='fast', metric_name='loss', score=.4).to_dict(), EvaluationRecord(task_id='t', candidate_id='c', input_revision='r2', output_artifact_digest='o2', contract_version='1', stage='official', fidelity='full', metric_name='loss', score=.1, selection_visible=False).to_dict()]
    record = {'evaluation_records': rows, 'best_warm_score': .4, 'final_best_score': .1, 'tune': True}
    assert candidate_score(record, stage='proxy', fidelity='fast') == .4
    assert candidate_score(record, stage='protocol', fidelity='full') is None

def test_legacy_score_record_preserves_proxy_domain(tmp_path):
    from tools.evaluation_records import legacy_score_record
    out=tmp_path/'score.json'; out.write_text('{"score": 1}\n')
    r=legacy_score_record(task_id='t',candidate_id='c',input_revision='i',score=1,metric_name='loss',output_artifact=out)
    assert r.stage=='proxy' and r.output_artifact_digest

def test_legacy_bridge_writes_the_record_in_the_evaluator(tmp_path):
    import json, os
    from tools.tuners._common import timed_eval
    run_dir = tmp_path / "runs" / "unit" / "bridge"
    candidate = run_dir / "candidates" / "001" / "train.py"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("# candidate\n")
    (run_dir / "framework_cfg.json").write_text(json.dumps({"max_evaluations": 4}))
    (run_dir / "ledger.json").write_text(json.dumps(
        {"records": [{"run_id": "001"}], "search_space_state": {}}))
    # The candidate process must never be handed the record path.
    assert "EVALUATION_RECORD_PATH" not in os.environ
    score = timed_eval(lambda make_model, params: 0.25, object(), {"x": 1},
                       candidate, phase="phase_a", method="warmstart")
    assert score == 0.25
    rows = read_records(run_dir / ".evaluation_records" / "001.jsonl")
    assert selection_view(rows, stage="proxy", fidelity="fast")[0]["score"] == 0.25
    indexed = json.loads((run_dir / "ledger.json").read_text())["records"][0]
    assert indexed["evaluation_record_digests"] == [rows[0]["record_digest"]]


def test_materialize_reads_each_record_file_once(tmp_path, monkeypatch):
    import tools.evaluation_records as er
    from tools.scheduler.state import materialize_evaluation_records

    record_path = tmp_path / "records.jsonl"
    rows = [
        EvaluationRecord(task_id="t", candidate_id="001",
                         input_revision=f"r{i}", output_artifact_digest=f"o{i}",
                         contract_version="1", stage="proxy", fidelity="fast",
                         metric_name="loss", score=0.1 * i).to_dict()
        for i in range(3)
    ]
    for row in rows:
        append_record(record_path, row)
    digests = [row["record_digest"] for row in rows]
    ledger = {"records": [{
        "run_id": "001",
        "evaluation_record_digests": digests,
        "evaluation_record_paths": {d: str(record_path) for d in digests},
    }]}
    real_read = er.read_records
    calls = []

    def counting_read(path):
        calls.append(str(path))
        return real_read(path)

    monkeypatch.setattr(er, "read_records", counting_read)
    materialize_evaluation_records(ledger)
    assert calls == [str(record_path)]
    assert [r["record_digest"] for r in ledger["records"][0]["evaluation_records"]] == digests
