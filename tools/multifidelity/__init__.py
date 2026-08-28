"""Offline benchmark harness for the independent-fidelity recall experiment.

Serves EXPERIMENT-independent-fidelity-recall-v1: frozen manifests and job
digests (manifest.py), the task-specific evaluation child
(autoresearch_eval_one.py), the wave-barrier matrix runner with job-level
resume (run_matrix.py), qualification/pool freezing (qualify_pools.py), and
the deterministic Recall@2 analyzer (analyze_recall.py).

Not a production path: nothing here reads or writes ledger.json, and no score
produced here enters semantic evidence.
"""
