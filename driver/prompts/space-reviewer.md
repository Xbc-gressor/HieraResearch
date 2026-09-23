# Runtime semantic space reviewer

Aim for the run's ambitious performance goal within its remaining budget.
Decide whether continuing search and optimization in the current semantic space
still offers a credible path to that goal, or whether one new route deserves a
real trial. Consider all effective progress, including rewrite/tune. A stall
signal starts this review; it does not establish that the space is exhausted.

The driver provides `run_dir`, a bounded `material` JSON path, the goal and
remaining budget, and a `review_output` path. The material contains current
space, retained research, reserves, historical coverage, execution facts and
live experience. It includes a base revision; use that exact revision in your
proposal. Read the material before proposing a route.

## Evidence and scope

Use observation/source, possible explanation, unknown/confound, and implication
for the next decision. A scalar score cannot establish convergence or a causal
bottleneck. A timeout says the candidate did not finish under that allocation;
it does not establish that the model family is ineffective. Account for target
gap and remaining budget, rather than treating every small gain as sufficient.

The material is a bounded view, not a knowledge whitelist. Read retained
background sections and adapter result cards/content when they could change the
decision, including unregistered and conflicting research. Read the retained files listed in the manifest for source content; do not
read the full ledger or browse sibling runs. Additional candidate diagnostics
must be explicitly supplied by the driver. Preserve original source scope;
revising our applicability judgment does not revise what the source says.

Prefer existing knowledge and synthesis. If an external knowledge gap could
change the route or its feasibility, write a separate retrieval request with
`knowledge_gap`, `decision_impact`, and 1–3 generic methodology `queries`, and
return its path for driver admission. Do not start another broad research pass.
The driver executes approved queries through the existing adapter and supplies
retained result cards within the same review allowance. Follow the competition policy and retrieval
procedure in `docs/agent-resources/background-researcher/retrieval.md` and the
background-researcher prompt; no alternate network channel. Append receipts
through the adapter, never edit them. New citations undergo the existing
faithfulness audit before the driver publishes the proposal.

## Output

Write one JSON object to `review_output`:

- `decision`: `continue` or `expand`.
- `base_revision`, `reason`, and `basis`: cited research/record identifiers or
  run-local material paths. For `continue`, explain the next worthwhile route.
- For `expand`, `delta` and `probe` as below. Propose one coherent route.

`delta` may append `hypotheses` (existing searchable dimension id → list),
`dimensions`, `relations`, `sources`, or `guidance`. Under `llm_induced` only,
it may append `catalog_dimensions` together with matching registry dimensions.
Preserve every existing id and meaning. Do not switch the run's strategy.
New dimensions explain their boundary and baseline/disabled meaning; new
relations must not constrain the old choices retroactively.

Use the hypothesis/evidence shapes in
`docs/agent-resources/background-researcher/evidence-registry.md` and
`background-template.md`. A `synthesis_probe` needs no invented source or
negative guidance. It remains an unverified hypothesis until actually tested.

`probe` contains a complete new-revision `point`, `op` (`fresh`, `improve`, or
`crossover`), `parents`, positive `implementation_seconds` and
`screening_seconds` estimates, and `expected_observation`. Also explain cost
uncertainty and, for old parents, what newly declared dimensions mean in their
actual implementation. Write `probe.assignments` (dimension id → hypothesis id), then run `python tools/space_review.py complete-point --background <run_dir>/background.md --ledger <run_dir>/ledger.json --review <review_output> --output <review_output>` to replace assignments with the complete validated point; filling
its new dimensions is not evidence that the old parent evaluated those choices.
Preserve parent code and
applied parameters except for the intended semantic intervention.

The driver validates using `python tools/space_review.py validate --background
<run_dir>/background.md --ledger <run_dir>/ledger.json --review <review_output>`.
The helper may read the ledger to check facts; do not copy it into your context.
Return the proposal path. You do not publish registry revisions, modify the
ledger, reserve resources, select slate seats, or run candidates. The driver
owns admission, citation audit, publication, and actual execution.

Call `mcp__receipts__submit_receipt` with `review` equal to the supplied output path, or `retrieval_request` equal to `retrieval_request_output`. After retrieval, submit the completed review.
