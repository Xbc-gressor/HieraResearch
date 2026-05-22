# autoresearch

This is an experiment to have the LLM do its own research.

The repo is organized as a multi-task harness. Each task under `tasks/` is an
independent uv project with its own `pyproject.toml`, `uv.lock`, task code, and
task metadata. The original autoresearch task lives at
`tasks/autoresearch-baseline/`.

## Setup

To set up a new experiment, work with the user to:

1. **Choose a task**: default to `tasks/autoresearch-baseline` unless the user
   names another task. Read the task's `TASK.md` and `task.toml`.
2. **Agree on a run tag**: propose a tag based on today's date and purpose
   (e.g. `may8-baseline`). The directory `runs/<task-name>/<tag>` must not
   already exist.
3. **Create the run directory**:
   `mkdir -p runs/<task-name>/<tag>`. This run directory is the durable record
   for the experiment session. Do not create or switch git branches just to
   manage runs.
4. **Read the in-scope files**:
   - `README.md` — repository context.
   - `CLAUDE.md` — project layout, skills, task package conventions.
   - `tasks/<task-name>/TASK.md` — task-specific context.
   - `tasks/<task-name>/task.toml` — run command, metric, and file boundaries.
   - `tasks/<task-name>/prepare.py` — fixed data preparation and evaluation.
     Do not modify when present.
   - `tasks/<task-name>/train.py` — the normal file to modify for autoresearch
     tasks.
5. **Verify the task environment**: run
   `uv --directory tasks/<task-name> sync` if dependencies are not already
   installed.
6. **Verify task assets**: read `TASK.md` and `task.toml` for data, tokenizer,
   model, benchmark, or fixture requirements. If the task declares a
   `run.prepare_command`, use it when required assets are missing.
7. **Initialize results.tsv**: use the exact header declared by the task parser
   or task documentation. If the parser creates the file on first append, it is
   acceptable to leave `results.tsv` absent until the first parsed run. Do not
   hand-write the generic six-column header when the task parser uses
   task-specific columns. The baseline will be recorded after the first run.
8. **Confirm and go**: confirm setup looks good.
9. **Baseline preparation** (run 000; before entering the experiment loop):
   The baseline is treated as the first **fully tuned** candidate. The
   task's `train.py` does not have to be contract-compliant — the setup
   adapts it. Five sub-steps:

   **9a. Copy template**:
   ```bash
   python tools/new_candidate.py <task-name> <tag> 000
   ```
   Creates `runs/<task>/<tag>/candidates/000/` with `prepare.py` and
   `train.py` copied from the task root.

   **9b. Summarize baseline as idea_log entry** (main Claude, no
   subagent): read `candidates/000/train.py`. Append the entry to
   `runs/<task>/<tag>/idea_log.md` (create the file if missing):
   ```markdown
   ## run 000
   - idea: baseline of the task's default model — <one-sentence summary
           of the model class and default hyperparameters>
   - primary_axis: baseline_calibration
   - tune: true
   - candidate_name_hint: <from CANDIDATE_NAME in train.py>
   ```

   **9c. Adapt baseline to tunable contract**: spawn `candidate-writer`
   (Task tool, `subagent_type: "candidate-writer"`) with:
   - `idea`: "Refactor candidates/000/train.py to expose `BASE_PARAMS` /
     `SEARCH_SPACE` / `make_model`. Preserve the baseline's behavior
     under `BASE_PARAMS` — default values must match the original code's
     behavior. Declare a conservative `SEARCH_SPACE` around those
     defaults that the task author would consider reasonable."
   - `source_train_py`: `candidates/000/train.py` (the just-copied
     baseline)
   - `target_candidate_dir`: `candidates/000/`
   - readonly `prepare.py`, plus task constraints from `task.toml`.

   candidate-writer edits `candidates/000/train.py` in place to add the
   contract without changing baseline behavior.

   **9d. Tune baseline**: spawn `tuner-orchestrator` (Task tool,
   `subagent_type: "tuner-orchestrator"`) on `candidates/000/` exactly
   as in experiment-loop step 2d. The orchestrator runs Phase A, Phase B
   (cold-start: always continue with `n_prior < 10`), Phase C, applies
   tuned `BASE_PARAMS` to `train.py` in place, writes
   `tune_report.json`, and extends the `## run 000` entry in
   `idea_log.md` with the 11 tuning summary fields.

   **9e. Run baseline end-to-end + parse to ledger**: same as
   experiment-loop steps 4-7. Produces `run-000.log`, the run 000 row in
   `results.tsv` (with `baseline_score / best_warm_score /
   final_best_score` populated), and initializes `loop_state.md` via
   `parse_result.py`.

   After step 9 completes, enter the experiment loop with
   `next_run_id: 001` (the loop's standard 2a–2d flow applies to every
   run from 001 onward).

Once you get confirmation and step 9 produces a recorded run 000, kick
off the experimentation loop.

## Experimentation

Each experiment runs according to `tasks/<task-name>/task.toml`. Use the task's
`run.command`, `run.working_dir`, `run.timeout_seconds`, and `result` section as
the source of truth.

For uv tasks, prefer running from the repo root with `uv --directory
tasks/<task-name> ...` so each task uses its own environment.

## Candidate Directories

For tasks that use file-based candidate experiments, each proposal gets its own
subdirectory under the run directory:

```text
runs/<task-name>/<tag>/
  candidates/
    000/
      prepare.py
      train.py
    001/
      prepare.py
      train.py
  run-000.log
  run-001.log
  results.tsv
```

Use this model whenever the task `TASK.md`, `task.toml`, or the human requests
per-proposal folders.

Rules:

- The first run (`000`) is the baseline candidate. It is prepared during
  Setup step 9: copied from the task root, adapted to the tunable
  contract by `candidate-writer`, tuned by `tuner-orchestrator`, then run
  end-to-end. From the loop's perspective, run 000 is already a
  contract-compliant tuned candidate when the loop starts at run 001.
- Each new proposal gets a new candidate directory, usually
  `candidates/<run_id>/`.
- Copy `prepare.py` from the task root every time so the candidate records the
  fixed evaluation snapshot.
- Copy `train.py` from the current best candidate when continuing an
  improvement line; otherwise copy it from the task root.
- Modify only the candidate directory's `train.py` for that proposal.
- Treat the candidate directory's `prepare.py` as readonly. It is a snapshot for
  reproducibility, not a new evaluation surface.
- Do not overwrite or delete old candidate directories. A discarded idea still
  stays in the run record.

When available, use the helper:

```bash
python tools/new_candidate.py <task-name> <tag> <run_id> --from-candidate <best_run_id>
```

For uv tasks, run the candidate entrypoint with the task environment, for
example:

```bash
uv --directory tasks/<task-name> run python "$(pwd)/runs/<task-name>/<tag>/candidates/<run_id>/train.py" > runs/<task-name>/<tag>/run-<run_id>.log 2>&1
```

Because candidate code under `runs/` is intentionally not committed, pass
`--commit worktree` to the task parser unless the candidate code has been
captured in a committed location.

**What you CAN do:**
- Modify files listed in `tasks/<task-name>/task.toml` under
  `constraints.editable_files`.
- For autoresearch-style training tasks, the task's editable training file is
  the normal experiment surface.
- When using candidate directories, apply the editable/readonly boundary to the
  copied files inside the candidate directory. Do not edit the task root
  `train.py` for ordinary candidate proposals.

**What you CANNOT do:**
- Modify files listed in `constraints.readonly_files`.
- Install new packages or add dependencies unless `task.toml` allows dependency
  changes or the human explicitly asks for environment work.
- Modify the task's evaluation harness or metric implementation unless the
  human explicitly asks for framework or benchmark work.

**The goal is simple: improve the configured task metric.** Read
`result.metric` and `result.lower_is_better` from `task.toml`. Everything in the
editable files is fair game. The code must run without crashing and finish
within the task budget.

**Resource use** is a soft constraint unless the task says otherwise. Some
increase is acceptable for meaningful metric gains, but it should not blow up
dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small
improvement that adds ugly complexity is not worth it. Conversely, removing
something and getting equal or better results is a great outcome — that's a
simplification win. When evaluating whether to keep a change, weigh the
complexity cost against the improvement magnitude. A tiny metric improvement
that adds 20 lines of hacky code? Probably not worth it. A tiny improvement
from deleting code? Definitely keep. An improvement of ~0 but much simpler code?
Keep.

**The first run**: Your very first run should always be to establish the
baseline, so you will run the task as is. If the task uses candidate
directories, create `candidates/000/` first and run that copied baseline.

## Output format

Read `tasks/<task-name>/TASK.md` and `tasks/<task-name>/task.toml` for the
task-specific output format, metric, parser, and required log patterns. Parse
the run log according to that task contract and append a normalized row to
`runs/<task-name>/<tag>/results.tsv`.

For example, extract the configured metric from a run log:

```
grep "<metric-pattern>" runs/<task-name>/<tag>/run-<id>.log
```

## Logging results

When an experiment is done, log it to `runs/<task-name>/<tag>/results.tsv`
(tab-separated, NOT comma-separated — commas break in descriptions). The exact
metric and any extra resource columns are task-specific and may be produced by
the parser declared in `task.toml`.

Use this normalized minimum shape only when the task parser does not declare a
more specific schema:

```
run_id	commit	metric	final_best_score	status	description
```

1. run id inside this session (e.g. `000`, `001`, `002`)
2. git commit hash (short, 7 chars) for traceability, or `worktree` if the
   code is intentionally uncommitted
3. metric name from `result.metric`
4. final score after the tuner-orchestrator's Phase C (or after Phase B
   stop) — use a task-defined sentinel for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Task parsers may add extra columns, such as `baseline_score`,
`best_warm_score`, `best_model`, memory, or throughput, when useful. In
that case, the parser's header is the source of truth.

Example (tabular-model-search parser schema):

```
run_id	commit	metric	baseline_score	best_warm_score	final_best_score	best_model	status	description
000	a1b2c3d	mean_test_accuracy	0.668000	n/a	0.668000	extra_trees_baseline	keep	baseline
001	b2c3d4e	mean_test_accuracy	0.685000	0.701000	0.713000	xgboost_strong_reg	keep	xgboost + tuner
002	c3d4e5f	mean_test_accuracy	0.660000	0.665000	0.665000	rf_balanced	discard	random forest
003	d4e5f6g	mean_test_accuracy	0.000000	0.000000	0.000000	none	crash	bad import
```

For each experiment, also copy or write useful artifacts into the run directory,
for example:

```
runs/<task-name>/<tag>/
  results.tsv
  loop_state.md
  candidates/000/train.py
  candidates/001/train.py
  run-000.log
  run-001.log
  notes.md
```

## Loop State

Maintain `runs/<task-name>/<tag>/loop_state.md` throughout the experiment. This
file is the durable short-term memory for the loop; do not rely on conversation
context for active state.

Create or update it after setup and after every run with at least:

```text
task: <task-name>
tag: <tag>
phase: setup|running|blocked
next_run_id: <next id>
best_run_id: <best kept run id or none>
best_score: <best metric value or none>
metric: <result.metric>
lower_is_better: <true|false>
candidate_mode: <true|false>
best_candidate_dir: <path or none>
last_run_id: <last run id or none>
last_status: keep|discard|crash|none
last_score: <metric value or none>
active_stop_condition: none|<hard stop reason>
notes: <one-line current search direction>
```

Before creating each new candidate, refresh the protocol and state by re-reading
`program.md` sections `The experiment loop` and `NEVER STOP`, the task's
`task.toml`, the task's `TASK.md`, and `loop_state.md`. Use
`results.tsv` as the source of truth if `loop_state.md` and the ledger disagree,
then repair `loop_state.md`.

## The experiment loop

The experiment is tracked by the run directory, e.g.
`runs/autoresearch-baseline/may8-baseline`. Do not create a dedicated git branch
for the run unless the human explicitly asks for branch-based work.

LOOP FOREVER:

1. Refresh protocol and state: re-read the loop-critical parts of `program.md`,
   `tasks/<task-name>/task.toml`, `tasks/<task-name>/TASK.md`, the current run
   directory, `results.tsv`, and `loop_state.md` if present. Repair
   `loop_state.md` from `results.tsv` if needed.
2. Propose and implement the next candidate. **All four sub-steps below
   are mandatory; do not skip any. Especially do not skip 2d.** Run 000
   is handled separately in Setup step 9 (baseline preparation); these
   sub-steps apply from run 001 onward.

   **2a. Outer-loop idea (`idea-proposer` skill, main loop)**
   Apply the `idea-proposer` skill in the main Claude context to produce
   one experimental idea grounded in the ledger, the current best
   candidate, and the ongoing conversation. The skill outputs an idea
   block (idea / primary_axis / rationale / scope / risks /
   candidate_name_hint / `tune: true`) AND appends the entry to
   `runs/<task>/<tag>/idea_log.md`. Confirm the append happened before
   continuing — this file is the diversity record the next round depends
   on.

   **2b. Candidate directory**
   Create the candidate directory: copy `prepare.py` from the task root
   and `train.py` from the current best candidate (e.g. via
   `tools/new_candidate.py`).

   **2c. Code generation (`candidate-writer` subagent)**
   Spawn `candidate-writer` (Task tool,
   `subagent_type: "candidate-writer"`) with the idea text, the source
   `train.py` path, the target candidate dir, the readonly `prepare.py`
   path, and the task's `editable_files` / `readonly_files` lists from
   `task.toml`. Apply the returned `train.py` and sanity-check the diff.

   **2d. Inner-loop tuning (`tuner-orchestrator` subagent) — REQUIRED**
   After `candidate-writer` returns, **you must** spawn
   `tuner-orchestrator` (Task tool,
   `subagent_type: "tuner-orchestrator"`) with the candidate train.py
   path, the task dir, the run dir, the repo root, and the run id.
   The orchestrator runs Phase A (warm-start), Phase B (percentile
   continue/stop), Phase C (one method by SEARCH_SPACE dim), writes
   `tune_report.json` to the candidate dir, edits `BASE_PARAMS` in
   `train.py` to the tuned values, and extends the candidate's
   `## run <run_id>` entry in `idea_log.md` with tuning summary fields
   (3 scores, method, percentile, decision, trials, elapsed, applied).
   **Verification before step 3**: confirm `tune_report.json` exists in
   the candidate dir AND the idea_log entry now has the tuner-added
   fields. If either is missing, the orchestrator did not run — go back
   and spawn it before proceeding.

3. Record enough provenance for the run: candidate directory path, git diff if
   committed files changed, and a short description in the run directory.
   Commit code changes only when you want a durable checkpoint; do not use
   branches as the run ledger.
4. Run the experiment using `run.command` and `run.working_dir`, redirecting all
   output to the run directory. When using candidate directories, adapt only the
   Python entrypoint path so the task environment still comes from
   `tasks/<task-name>`.
5. Read out the results using `result.required_patterns` from `task.toml`.
6. If the grep output is empty, the run crashed. Spawn the `crash-diagnoser`
   subagent (Task tool, `subagent_type: "crash-diagnoser"`) and pass it the
   crashed `runs/<task-name>/<tag>/run-<run_id>.log` plus the candidate
   directory. Act on its returned verdict:
   - `recommendation: fix` → apply the suggested patch to the candidate's
     `train.py` and re-run. If two consecutive fix attempts still crash, treat
     the idea as failed and move on.
   - `recommendation: abandon` → log the run as `crash` and try a different
     idea.
   Only fall back to inlining `tail -n 50` and debugging in the main loop if
   the subagent is unavailable or its verdict is `confidence: low` and you
   need to read the traceback yourself.
7. Record the result in `runs/<task-name>/<tag>/results.tsv` (NOTE: do not
   commit run artifacts). Update `loop_state.md` immediately after recording
   the row.
8. If the configured metric improved, mark that candidate as the current best
   and continue from its `train.py`.
9. If the configured metric is equal or worse, mark the candidate as discarded
   and continue from the best known candidate. Do not delete the candidate
   folder, run log, or TSV row. If the experiment modified committed task files,
   revert only that unrelated committed-file change.

The idea is that you are a completely autonomous researcher trying things out.
If they work, keep. If they don't, discard. The run directory is the complete
record of what happened. If you feel like you're getting stuck in some way, you
can rewind code changes, but you should probably do this very very sparingly
(if ever).

**Timeout**: Use `run.timeout_seconds` from `task.toml`. If a run exceeds the
timeout, kill it and treat it as a failure (discard and revert the code change).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If
it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and
re-run. If the idea itself is fundamentally broken, just skip it, log `crash` as
the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do
not stop at a progress summary, do not send a final answer as a stopping point,
and do not ask the human whether to continue. Do not ask "should I keep going?"
or "is this a good stopping point?". Keep creating, running, parsing, and
recording new candidates indefinitely until one of these hard stop conditions
occurs:

1. The human explicitly interrupts, stops, pauses, or redirects you.
2. A required permission, credential, dependency download, or unavailable
   external resource blocks further progress.
3. A tool/runtime limit prevents additional commands from being executed.
4. The task contract is internally inconsistent and continuing would corrupt the
   benchmark or results ledger.

Repeated `discard` results, lack of obvious ideas, no recent improvement, or
multiple crashes are not stop conditions. If an idea fails, log it and try a
different one. If you run out of ideas, think harder: inspect the best and worst
candidate diffs, re-read the task files, vary model families, simplify previous
near-misses, combine promising changes, or try a more radical direction. The
loop continues until a hard stop condition is reached.

As an example use case, a user might leave you running while they sleep. The
user then wakes up to experimental results, all completed by you while they
slept!
