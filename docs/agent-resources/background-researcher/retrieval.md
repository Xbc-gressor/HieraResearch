# Retrieval mechanics for the background researcher

Read this resource immediately before the first retrieval action. It defines
the adapter commands, the append-only manifest, result cards, the status
dashboard, progressive reading, and source-quality checks.

Contents: manifest · backends · search rounds · result cards and status ·
visits and progressive reading · continuation reads · backend failures ·
source-quality checks · receipts.

The run-local retrieval manifest at `<run_dir>/background_retrieval.json`
records which questions ran, which resolved dimensions and evidence roles they
targeted, which backends answered, and which sources were opened. It is
append-only and written only through `tools/search_backends.py` — never
hand-edit it or the `retrieval/` content files it points to.

## Manifest

Every `search` call appends one round (`r-01`, `r-02`, …) holding that call's
queries, backend calls, and merged results; query ids (`q-01`, `q-02`, …)
keep counting up across rounds. Nothing is ever rebuilt or deleted: a
follow-up round adds to the record instead of rerunning earlier questions.
Visits append globally, independent of rounds. A successful visit's content
lives in a file under `retrieval/` next to the manifest; the manifest keeps
the pointer and the character count.

## Backends

Select search backends explicitly:

- `frozen` — a pinned local JSON corpus for reproducible, network-disabled
  work. Pass `--frozen-corpus <path>`; it implies the frozen backend when
  `--backend` is omitted, and the adapter refuses any other backend alongside
  it. For strict benchmark comparisons, confirm the corpus is a generic prior
  frozen before task identities were exposed and that it excludes
  task-specific discussions, notebooks, repositories, and solutions; the
  task-local path is storage, not evidence of compliant provenance.
- `deepxiv` — open-world scholarly retrieval.
- `jina-search` — live web search; requires `JINA_API_KEY` in the environment,
  without it every call fails.

Repeat `--backend` to fan a round across several. A backend that is
unavailable or errors records per-query failed calls and does not block the
others. Never mix frozen and live backends in one run's evidence.

## Search rounds

```bash
python tools/search_backends.py search \
  --manifest <run_dir>/background_retrieval.json \
  --query-spec '{"text":"<bounded evidence question>","target_dimension_ids":["<exact-resolved-dim-id>"],"evidence_roles":["hypothesis","counterevidence"]}' \
  --query-spec '{"text":"<problem-class comparator question>","target_dimension_ids":[],"evidence_roles":["baseline"]}' \
  --backend deepxiv --backend jina-search
```

Each `--query-spec` is a JSON object with `text`, optional
`target_dimension_ids` (exact `dim-*` ids the question informs right now; an
intent record, empty allowed), and a non-empty `evidence_roles` list drawn
from `hypothesis`, `baseline`, `failure_mode`, `counterevidence`, `relation`.
`--max-results` caps hits per query per backend (default 50).

Results deduplicate by canonical URL / arXiv work and rank by the number of
distinct supporting queries. Backend and query agreement are retrieval
signals, not scientific corroboration.

## Result cards and status

`search` prints a summary line — round id, query count, hits, this round's
`max_results`, empty and failed query ids, manifest validation state — then
one result card per new hit:

```text
[q-01 rank 1] <title>
  <url> | authors: … | date: … | citations: …
  tldr: …
  snippet: …
```

Cards are the triage surface: they distinguish an actionable method report
from a pure-theory title before you spend a visit. Backend calls are
three-state: `success`, `empty` (answered, zero hits), `failed`. An `empty`
call is a diagnostic failure — check credentials/quota or widen the query. A
round that produced no hits at all exits nonzero.

`search` prints only the round it just appended. To re-browse a past round's
cards without touching the record:

```bash
python tools/search_backends.py results \
  --manifest <run_dir>/background_retrieval.json [--round r-NN]
```

`--round` takes `r-NN` or a bare number and defaults to the latest round.

Between rounds, survey the whole record:

```bash
python tools/search_backends.py status \
  --manifest <run_dir>/background_retrieval.json
```

The dashboard reports round/query/result/visit totals, verification tier
counts, per-dimension result and visit counts (from query targets, past or
present), high-rank hits not yet visited, and every query that came back
empty or failed. Use it to pick the next round's questions and the next
visits.

## Visits and progressive reading

```bash
python tools/search_backends.py visit \
  --manifest <run_dir>/background_retrieval.json \
  --url <url> [--view auto|brief|head|preview|section|full_text] \
  [--section <name>] [--visit-backend auto|direct|jina-read] \
  [--frozen-corpus <pinned-corpus.json>]
```

- For arXiv sources, use the default `--view auto`: the adapter records a
  `head` triage receipt, ranks up to three available body sections against
  the source's retrieval questions, fetches them, and falls back to `preview`
  when no section body can be obtained. Metadata with no body is a failed
  visit.
- For other web sources, the adapter reads through the jina reader with a
  direct HTTP fallback; `--visit-backend direct` skips the reader. Error
  pages, empty pages, and tiny responses are recorded as failed visits with
  the reason.
- `--view section` requires `--section <name>`; in the frozen condition,
  `--frozen-corpus` replays retained corpus content without network access.

`visit` stdout is a bounded head view: the first screen of content, a section
map with character ranges, and the exact continuation command. The full text
(up to a generous cap) is already stored on disk. Head metadata is triage;
before a source supports a claim, read enough of the body to state its
studied scope.

## Continuation reads

`read` pages through already-stored visit content; it appends no receipt:

```bash
python tools/search_backends.py read \
  --manifest <run_dir>/background_retrieval.json \
  [--visit N | --url <url> [--view <view>]] \
  [--section <name> | --offset N [--length L]]
```

With no selector it prints the first 8000-character page of the latest
successful visit for the URL and the offset for the next page. `--section`
matches a markdown heading from the visit's section map; a miss lists the
available names. `--visit N` selects a receipt by index.

## Backend failures

When a backend fails or answers empty, switch backends, rephrase, or widen
the query. Never silently replace missing primary evidence with a generic
blog summary; declare the gap under unresolved evidence instead. The official
DeepXiv CLI may auto-register its free anonymous token in `~/.env` on first
use; never expose it in logs or run files. Do not install the package merely
to obtain the CLI.

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

## Receipts

A source enters the registry only from this record: a search hit (a snippet
receipt) or a successful visit. From the record, each source derives a
verification tier — `snippet_only`, `preview`, `section`, or `full_text` —
surfaced in `status`. Read deep enough that the claims you register match the
tier behind them; before the space freezes, citations are spot-checked
against the recorded content, and one the record does not carry comes back
for repair or removal.

Finish with:

```bash
python tools/search_backends.py validate \
  --manifest <run_dir>/background_retrieval.json
```
