# Evidence-aware background research

HieraResearch uses background research to create external `tf-*` hypotheses for
fresh candidates. This stage must optimize for actionable, testable directions,
not report length or the number of retrieved URLs.

## Why the previous path was insufficient

The original `background-researcher` directly used generic `WebSearch` and
`WebFetch`, then synthesized a Markdown table and source list. That left several
important decisions implicit:

- no explicit decomposition into independent research questions;
- no distinction between retrieving a claim and validating it;
- no progressive paper-reading or counterevidence pass;
- no machine-readable source-to-claim relationship;
- no way to connect a `tf-*` external hypothesis back to later run evidence;
- no validation of stable ids, citations, or empirical attribution.

In particular, an arXiv upload is only a distribution event. It does not imply
peer review, sound baselines, reproducibility, or applicability to the current
task.

A comparative run exposed a second, more consequential failure. A negative
result for one learner, balancing mechanism, data regime, metric, and evaluation
protocol was rewritten as a general warning against the whole intervention
family. The resulting registry collapsed around the favored learner and omitted
a neighboring ensemble mechanism outside the paper's tested scope. A less
literature-constrained comparison retained that mechanism and found it
promising.

This does not show that the paper was wrong in its tested setting. It shows that
the harness discarded the paper's scope before using the claim to shape the
direction registry. Once a plausible family is absent from that registry,
`idea-generator` has no `tf-*` identity from which to propose it as a fresh
candidate.

The intervention order follows that causal path:

1. Record the exact studied scope at source ingestion. This prevents the first
   lossy conversion from paper result to bare claim.
2. Put every consequential Pitfall or Deprioritize claim in structured guidance.
   Unregistered negative Markdown has no selection authority and fails the v2
   contract.
3. Compare guidance and direction scopes deterministically before changing
   priority or eligibility. Credibility and applicability remain distinct.
4. Preserve an out-of-scope `scope_probe` for each binding negative item so a
   neighboring mechanism cannot disappear with the scoped mechanism.
5. Require direct, paired run evidence before experience can support, contradict,
   or declare a scoped dead end, and let direct local support reopen an external
   exclusion.

## What was adapted from sibling projects

`InternAgent/` demonstrates useful orchestration mechanisms:

- its deep-research workflow uses a planner-generated task graph, concurrent
  execution, an optional coordinator that adds follow-up work, and a final
  synthesizer (`internagent/mas/agents/dr_agents/workflow/main.py`);
- `config_complex.yaml` makes breadth, iteration count, tool-call budgets, and
  the coordinator explicit rather than leaving research depth accidental;
- its global reference manager deduplicates URLs and preserves reference type
  across parallel tasks (`utils/reference_manager.py`).

Those mechanisms motivated bounded question decomposition, a coverage-gap pass,
deduplication, and retained provenance here. The implementation was not copied:
InternAgent's reference type and URL tracking do not judge claim credibility,
and importing its full multi-agent runtime would be disproportionate for one
HieraResearch setup stage.

`deepxiv_sdk/` provides the complementary retrieval mechanism:

- metadata/brief triage before expensive reads;
- a section map with token counts;
- selective method, results, and limitations reads;
- explicit service-failure handling and fallback behavior.

HieraResearch can use the DeepXiv CLI, including the workspace sibling checkout,
but does not depend on it. The official CLI automatically obtains a free
anonymous API token on first use and persists it to `~/.env`; HieraResearch
permits that one-time provider bootstrap but never records the token in run
artifacts. DeepXiv is a remote-service client, its default corpus is
preprint-oriented, and its rank, TLDR, citations, and popularity are not
credibility judgments. The background agent therefore uses it for discovery and
progressive access only, and falls back to targeted primary-source web retrieval
while recording the coverage limitation.

`Arbor/` contributes the retrieval plumbing around those sources:

- `src/core/tools/web/backends.py` hides heterogeneous providers behind one
  query-to-results interface and degrades when a backend is unavailable;
- `src/core/tools/web/search.py` fans out queries, canonicalizes URLs, merges
  repeated results, counts distinct-query support, and balances the final set
  across queries;
- `src/coordinator/tools/_agent_recover.py` cross-checks cited URLs against URLs
  actually passed to visit tools;
- `docs/search.md` separates a small novelty-audit reading budget from a larger
  grounded-ideation budget and stores the two contexts separately.

HieraResearch adapts those mechanisms in `tools/search_backends.py` rather than
importing Arbor's coordinator or agent runtime. Its backend interface is local
and open, but DeepXiv and Jina remain hosted external services. They are never a
correctness dependency and must be selected explicitly. Claude-native
WebSearch/WebFetch remain an explicit fallback because they cannot be invoked
from Python; successful fallback visits must still be recorded in the manifest.

The preferred reproducible publication condition is the `frozen` backend: a
pinned local JSON corpus is searched lexically and replayed without network
access. Its corpus id, cutoff, path, and SHA-256 are retained. DeepXiv is an
optional open-world scholarly condition; Jina is an optional live-web
fallback/ablation rather than a default. The CLI has no implicit backend: it
fails closed when neither a frozen corpus nor an explicit `--backend` is
supplied.

A minimal frozen corpus is:

```json
{
  "schema_version": 1,
  "corpus_id": "tabular-prior-2025-01",
  "cutoff": "2025-01-31",
  "created_at": "2025-02-01T00:00:00Z",
  "provenance": "Generic scholarly corpus prepared before benchmark task identities.",
  "prepared_before_task_ids": true,
  "items": [
    {
      "url": "https://arxiv.org/abs/0000.00000",
      "external_id": "0000.00000v2",
      "title": "Example retained paper",
      "abstract": "Retained abstract used for local lexical search.",
      "text": "Retained inspected sections used for offline grounding."
    }
  ]
}
```

The conventional task-local path is
`tasks/<task-name>/background_corpus.json`. It is optional for ordinary
open-world development, but required when that task is evaluated under the
frozen, network-disabled condition. An explicit alternative path may be used
when the corpus is maintained outside the repository; its hash is still saved.
For strict benchmark claims, task-local placement does not establish clean
provenance: the generic corpus must have been frozen before task identities were
revealed and must exclude task-specific discussions, notebooks, repositories,
and partial or complete solutions.

The main-condition commands are then:

```bash
python tools/search_backends.py search \
  --manifest <run_dir>/background_retrieval.json \
  --frozen-corpus <corpus.json> \
  --query "<question one>" --query "<question two>"
python tools/search_backends.py visit \
  --manifest <run_dir>/background_retrieval.json \
  --frozen-corpus <corpus.json> --lane grounding --url <selected-url>
```

## Retrieval manifest

`<run_dir>/background_retrieval.json` is the mechanical trace paired with the
human-readable background brief. It records:

- the decomposed questions and their retrieval lane;
- backend failures without discarding surviving results;
- canonical, cross-query deduplicated candidates;
- distinct-query and distinct-backend support;
- a balanced selected set with shared results first and per-query coverage
  afterward;
- successful and failed visits, content depth, and token budget.
- raw backend responses, timestamps, backend/client versions when available,
  frozen-corpus identity, and response/content SHA-256 hashes.

The `grounding` lane has a 6000-token reading budget. The separate `novelty`
lane has a 2048-token budget. A novelty-only visit cannot support a Direction
registry claim: the source must be revisited in the grounding lane. This keeps
sources that shaped a hypothesis separate from later overlap checking.

The visited-source guarantee is strongest for visits made through
`search_backends.py`, which writes receipts itself. Claude WebFetch or OpenCode
webfetch fallback is recorded explicitly after a successful tool response; this
provides deterministic artifact consistency, although it cannot independently
inspect the runtime's private tool transcript the way Arbor's integrated runtime
can.

From the workspace layout used during development, the no-install smoke test is:

```bash
PYTHONPATH=.. python -m deepxiv_sdk.deepxiv_sdk.cli search \
  "tabular machine learning benchmark" --limit 1 --format json
```

The first call may create the free token in `~/.env`. This command is an
optional provider check, not part of deterministic repository validation.

## Two evidence axes

Every direction has one stable `tf-*` id and two deliberately separate views:

1. `literature_credibility` lives in `background.md` and describes inspected
   external evidence: `unverified`, `preliminary`, `corroborated`, `replicated`,
   or `contested`.
2. `run_status` lives in `ledger.json`'s incrementally revised experience snapshot and
   describes only the current task/run: `untested`, `inconclusive`,
   `supported_here`, `contradicted_here`, or `mixed`.

Keeping them separate preserves cases such as a replicated method that fails on
this dataset, or a preliminary idea that works locally. A later selection policy
may combine the axes, but storage must not collapse them into one truth score.

The Direction registry also labels every source relationship as `supports`,
`contradicts`, or `context`. A `replicated` direction requires an independently
reproduced supporting source; `contested` requires explicit contradicting
evidence.

## Scope is a separate contract

Registry schema v2 makes transfer scope explicit instead of hiding it in a
credibility rationale. Sources, external negative guidance, and directions use
the same five typed axes: `model_families`, `data_regimes`, `metrics`,
`interventions`, and `evaluation_protocols`. Values are exact lowercase tags;
the wildcard `*` is permitted only when the retained evidence genuinely covers
the whole axis. Directions separately retain a human-readable claim boundary,
required local comparisons, and a reopening condition.

`tools/background_contract.py` derives the relationship instead of trusting an
agent-authored applicability label:

- `direct`: every direction tag is contained by the guidance on every axis;
- `mismatch`: at least one axis is disjoint;
- `partial`: all axes overlap, but the guidance does not contain the full
  direction;
- `unknown`: either scope is malformed.

Only `direct` negative guidance can change selection. `caution` annotates,
`deprioritize` moves a direction behind active directions, and `exclude` removes
only a directly matched direction while retaining an exclusion receipt. The
validator permits binding guidance only from preliminary-or-stronger,
non-withdrawn primary empirical evidence. `unverified` and `contested` negatives
can only caution. Exclusion additionally needs corroborated/replicated evidence,
two directly scoped supporting sources, and directly scoped independent
reproduction. Canonically duplicate sources are rejected so one work cannot
stand in for corroboration. A broad guidance scope that is not contained by at
least one eligible supporting source is invalid.

This distinction matters because credible evidence can still transfer poorly.
For example, a binary linear-model comparison of class weighting against global
random undersampling says nothing decisive about component-level sampling inside
an ensemble, multiclass behavior, or an aggregate metric over several datasets.
Those are separate mechanisms/scopes, not synonyms for “resampling.”

Consequential Pitfalls and Deprioritize entries are structured `g-*` guidance
inside the registry. The Markdown validator requires each bullet to start with a
matching `g-*`, `task-constraint`, or `operational` marker and requires every
structured guidance item to appear in its declared section. Unregistered prose
is both nonbinding and a contract error. This closes the old gap where a
universal-sounding bullet could silently determine which `tf-*` directions
existed even though no consumer could inspect its scope.

Directions are either `evidence_prior` or `scope_probe`. Every binding external
`deprioritize` or `exclude` item must have at least one plausible direction whose
typed scope is outside that guidance. The probe names its `g-*` boundary in
`probe_for`, keeps a stable `tf-*` id, and participates in the ordinary priority
list; it does not consume a hard-coded bootstrap slot. Direct task constraints
still exclude illegal approaches, and the graph search still owns allocation.

At fresh-candidate selection time, active directions sort before directly
deprioritized directions. An excluded direction remains visible in a top-level
receipt. The compact view separates all `matched_guidance` from the
`binding_guidance` subset so a caution cannot be reinterpreted as a block. If
previous direct task evidence records `supported_here` or `mixed`, the selector
reopens an external exclusion. Scope mismatch and partial transfer never reduce
eligibility.

Schema v1 registries remain readable for run resumption and are emitted to the
idea generator as `legacy_unspecified`. Their absent scope metadata must not be
used to block neighboring mechanisms. When every legacy direction is consumed,
the orchestrator's small `background_contract.py preflight` command returns
`action: refresh_background`. The orchestrator reruns background research,
preserves existing `tf-*` meanings, migrates them to v2, appends scope probes,
refreshes experience, and only then invokes idea generation. Background
maintenance never appears as an idea-agent status.

Run-local status has a matching guard. Each v2 `direction_evidence` entry records
`claim_coverage`, `comparison_runs`, and `missing_comparisons`.
`supported_here`, `contradicted_here`, and `mixed` require direct coverage by at
least two scored non-crash runs and no missing required comparator. Structural
validation cannot prove that free-form candidate code implements the right arm,
so the experience extractor must still check the persisted candidate records;
the schema makes that judgment visible and prevents a single unpaired result or
crash from becoming a decisive status.

To keep long runs bounded, exhaustive ancestry stays mechanically derivable from
the immutable ledger. The experience snapshot stores at most five representative
direct/descendant/combination run receipts per direction. Periodic extraction
consumes only the helper-revisioned DAG delta, compact lineage delta, and fixed
Top/Bottom anchors.

The same boundary applies to reusable `deadend` lessons. Under a v2 registry, a
dead end needs two scored non-crash runs and must state the exact tested scope
and a reopening condition. One weak implementation remains a hypothesis; it
cannot blacklist a model or method family for later ideation.

## Feedback path

```text
task contract
    -> background-researcher
    -> background_retrieval.json: frozen/open condition + raw search + visits
    -> background.md: stable tf-* claims + literature credibility
    -> fresh candidate records: source_run_ids=[tf-*]
    -> candidate DAG and scores
    -> experience-extractor
    -> ledger.experience.direction_evidence: run-local status
    -> idea-generator reads both axes
```

`background.md` has one writer: `background-researcher`. The
`experience-extractor` reads it but writes only the ledger experience block.
The separate write surfaces keep external claims distinct from internal
observations.

Attribution is conservative. `tools/background_contract.py lineage` separates:

- the direct fresh implementation of a direction;
- descendants with exactly one originating direction;
- multi-origin descendants, whose crossover success cannot be credited to every
  ancestor.

A single crash does not contradict a direction, and one strong raw score does
not support it automatically. Status rationales must cite actual run ids or DAG
edges.

## Validation

```bash
python tools/search_backends.py validate \
  --manifest <run_dir>/background_retrieval.json
python tools/background_contract.py validate \
  --background <run_dir>/background.md \
  --retrieval-manifest <run_dir>/background_retrieval.json
python tools/background_contract.py preflight \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json
python tools/background_contract.py lineage \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json
python tools/background_contract.py validate-experience \
  --background <run_dir>/background.md --ledger <run_dir>/ledger.json
python tools/validate_background.py
python tools/validate_search_backends.py
```

The contract validator checks registry structure, stable and contiguous `tf-*`
ids, source relationships, studied-scope fields, credibility prerequisites,
source-to-guidance scope containment, exclusion evidence thresholds,
Markdown-to-guidance alignment, out-of-scope probe coverage for binding
guidance, unknown ledger tags, lineage attribution, copied literature stamps,
claim coverage, and scored comparison runs. The frozen contract fixture
exercises the general failure mode: guidance for one learner and global
intervention directly deprioritizes an exact replication, while a neighboring
ensemble mechanism is a scope mismatch and remains active.

Semantic judgments—whether the retained source text justifies its scope tags, a
candidate implements the named comparator, or two experiments are genuinely
independent—remain agent responsibilities. Exact tags intentionally fail open
when inconsistent: they may preserve too much exploration, but cannot silently
broaden a negative claim. Those residual judgments should be surfaced as
uncertainty rather than hidden behind the schema.
