# Retrieval mechanics for the background researcher

Read this resource immediately before the first retrieval action, after the
evidence-search plan exists. It defines the retrieval condition, the adapter
command forms, manifest receipts, lane budgets, progressive reading,
source-quality checks, and grounding-visit requirements.

Contents: retrieval condition · search dispatch · visits and progressive
reading · runtime web fallback and record-visit · novelty lane isolation ·
backend failure handling · source-quality checks · grounding-visit
requirement.

The run-local retrieval manifest at `<run_dir>/background_retrieval.json`
records which questions ran, which resolved dimensions and evidence roles they
target, which backends answered, how duplicate results merged, and which
sources were opened.

## Retrieval condition

Choose the retrieval condition explicitly. For a reproducible benchmark or
publication comparison, use the task-provided
`tasks/<task>/background_corpus.json` (a pinned object with `corpus_id`,
`cutoff`, `created_at`, `provenance`, `prepared_before_task_ids`, and retained
`items`) or an explicitly supplied equivalent path;
pass it as `--frozen-corpus <path>`, which implies the fully local `frozen`
backend. For strict benchmark comparisons, confirm that this is a generic
prior frozen before task identities were exposed and that it excludes
task-specific discussions, notebooks, repositories, and solutions; the
task-local path is storage, not evidence of compliant provenance. For an
exploratory open-world run, opt into `--backend deepxiv`. Use `--backend jina`
only as a separately noted live-web fallback/ablation, not an invisible
default. Do not mix frozen and live results in the main reproducible
condition.

## Search dispatch

Dispatch the planned questions together through the local adapter. Replace
each placeholder dimension with an exact id from the resolved registry:

```bash
# Reproducible condition:
python tools/search_backends.py search \
  --manifest <run_dir>/background_retrieval.json --lane grounding \
  --query-spec '{"text":"Which mechanisms address the dominant task failure under its declared constraints?","target_dimension_ids":["<exact-resolved-dim-id>"],"evidence_roles":["hypothesis","counterevidence"]}' \
  --query-spec '{"text":"Which results are the standard strong comparators for this problem class?","target_dimension_ids":[],"evidence_roles":["baseline"]}' \
  --query-spec '{"text":"Which numeric ranges are stable for the task-declared parameters?","target_dimension_ids":[],"evidence_roles":["inner_hpo_prior"]}' \
  --coverage-exemption '{"dimension_id":"<exact-uncovered-dim-id>","rationale":"<why applicable literature evidence is unavailable>"}' \
  --frozen-corpus <pinned-corpus.json>

# Or, explicitly, an open-world condition:
python tools/search_backends.py search \
  --manifest <run_dir>/background_retrieval.json --lane grounding \
  --query-spec '{"text":"<bounded evidence question>","target_dimension_ids":["<exact-resolved-dim-id>"],"evidence_roles":["hypothesis","baseline"]}' \
  --backend deepxiv
```

Omit `--coverage-exemption` when every searchable dimension is targeted.
The adapter fans each query across every explicitly selected usable backend
(local frozen corpus, DeepXiv, and/or keyless Jina search), drops
unavailable/failing backends, canonicalizes URLs, deduplicates across backends
and queries, ranks by the number of **distinct** supporting queries, then
balances the selected set so one broad query cannot erase the others. Backend
and query agreement are retrieval signals, not scientific corroboration.

## Visits and progressive reading

Read selected sources through the adapter so visits are recorded. Background
research always uses the `grounding` lane (6000-token budget):

```bash
python tools/search_backends.py visit \
  --manifest <run_dir>/background_retrieval.json --lane grounding \
  --url <url> --view <head|section|preview|full_text|auto> --section <name>
```

Omit `--section` unless `--view section` is used. For papers, triage metadata
and section maps first, then read relevant method, results, or limitations
sections. In the frozen condition, append `--frozen-corpus <pinned-corpus.json>`
to replay retained content without network access. Outside that condition,
`auto` uses DeepXiv for arXiv and a direct HTTP fetch for other sources; Jina
visiting is explicit via `--visit-backend jina` and remains an optional
live-web ablation.

## Runtime web fallback and record-visit

If the local backends miss an evidence class, use targeted web search
(`WebSearch` in Claude, `websearch` in OpenCode) for later versions,
independent reproductions, official repositories, benchmark records, and
primary artifacts. Use web fetch (`WebFetch` / `webfetch`) only after triage.
After every successful fetch that will appear in the search-space registry,
write the returned content to a temporary run-local file and append a receipt
that retains and hashes that exact content, with the backend name matching
your runtime (`claude-webfetch` or `opencode-webfetch`):

```bash
python tools/search_backends.py record-visit \
  --manifest <run_dir>/background_retrieval.json --lane grounding \
  --backend <claude-webfetch|opencode-webfetch> --view page --status success \
  --content-file <temporary-fetched-content> --url <url>
```

Record failures too, with `--status failed --error "<reason>"`. Never claim a
source merely because it appeared in a search snippet.
Delete the temporary file after the receipt is stored.

## Novelty lane isolation

Keep post-hoc novelty search isolated from grounding. The adapter defines a
smaller `novelty` lane (2048 tokens), but novelty-only visits do not qualify a
source to support a background claim. Revisit any useful novelty result in the
grounding lane before using it in a hypothesis.

## Backend failure handling

If all specialized backends fail, continue with your runtime's web-tool
fallback and record the coverage limitation. Never silently replace missing
primary evidence with a generic blog summary. The official DeepXiv CLI may
auto-register its free anonymous token in `~/.env` on first use; never expose
it in logs or run files. Do not install the package merely to obtain the CLI.

## Source-quality checks while reading

General blogs, forums, social posts, and SEO summaries are high-noise. Assume
such content may be AI-generated: LLM summaries fabricate method details and
numeric values while mimicking authoritative technical prose, and never count
as corroboration. They may serve as leads or first-hand operational reports,
but consequential method and numeric claims need a paper or primary artifact
when possible. Deduplicate the underlying work: five summaries of one preprint
are one source, not corroboration.

For each promising claim, inspect enough of the actual source to assess:

- publication status (an arXiv upload alone is `preprint_only`, not validation),
- strength and tuning of baselines, datasets/seeds, variance, and ablations,
- code/data availability and correspondence to the claimed method,
- independent support, reproduction, contradiction, or retraction,
- similarity between the reported setting and this task.

## Grounding-visit requirement

Every search-space source must have a successful grounding-lane visit in
`background_retrieval.json`. A search hit, snippet, generated TLDR, or
novelty-only visit is insufficient.
