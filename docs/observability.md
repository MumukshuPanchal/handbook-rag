# Observability: reading a trace

When a search is slow or returns the wrong passage, the first question is which stage is responsible. A trace answers it.

- A **trace** is the record of one request.
- It is made of **spans**: named, timed steps that nest inside each other. A trace reads like a call tree with a duration on every line.

The pipeline is instrumented with OpenTelemetry, the vendor-neutral standard ([ADR 0008](adr/0008-opentelemetry-tracing.md)). Tracing is off by default.

## Try it

```bash
make traces                                   # start the trace backend (Jaeger) in Docker
HANDBOOK_RAG_TRACING=true uv run handbook-rag search "how do I expense a laptop"
```

The command prints the id of its trace on the last line. Open <http://localhost:16686>, choose the service `handbook-rag`, and open the trace, or paste the id into the lookup box.

![A reranked search as a trace in Jaeger: ten spans, with the reranker taking 208 of 336 milliseconds](images/trace-reranked-search.png)

This is one search with reranking switched on. The same trace as text:

```
search                           336 ms
  retrieve.reranked              335 ms
    retrieve.hybrid              127 ms
      retrieve.dense              92 ms
        embed.query               30 ms   the embedding model turns the question into a vector
        vector_store.query        61 ms   Qdrant finds the 30 nearest chunks
      retrieve.sparse             34 ms
        vector_store.query        33 ms   Qdrant finds the 30 best keyword matches
      fuse                       0.3 ms   the two lists are merged by rank
    rerank                       208 ms   the cross-encoder reads the question with each of 30 chunks
```

## What each span records

| Span | What it covers | Worth looking at |
|---|---|---|
| `search` | The whole request | `mode`, and the ids, pages and scores of what was returned |
| `retrieve.reranked` | Search plus reranking, when reranking is on | The final order |
| `retrieve.hybrid` | Both searches and the merge | `candidates`: how many each search was asked for |
| `retrieve.dense` | Search by meaning | The chunks it found, before merging |
| `embed.query` | The embedding model | `model` |
| `retrieve.sparse` | Search by keyword | `query.terms`: how many words were left after stop words were removed |
| `vector_store.query` | One call to Qdrant | `vector` (dense or sparse) and the collection |
| `fuse` | Reciprocal rank fusion | `fuse.in_both`: how many chunks both searches returned |
| `rerank` | The cross-encoder | `model` and `candidates` |

Every retrieval span carries `results.count`, `results.chunk_ids`, `results.doc_ids` and `results.scores`. Our own attributes start with `handbook_rag.`. The database spans also use the standard `db.*` names, which is why Jaeger shows them with a database icon.

## What is not recorded

A trace holds ids, scores, sizes and timings. It does not hold text.

- **The question** is recorded as its length only. Questions can contain personal or confidential details, and traces are usually kept longer and read by more people than the application's own data. `HANDBOOK_RAG_TRACE_CONTENT=true` adds the question's text, for debugging on your own machine.
- **Chunk text** is never copied into a trace. A chunk id leads back to the text in the index.

## Using a trace to debug

**"It was slow."** Find the longest bar. In the trace above the reranker takes 208 of 336 ms. Without reranking the same search takes about 36 ms, of which the embedding model takes 17 and the two Qdrant calls 11 and 8.

**"The right passage was not returned."** Look for its page among `results.doc_ids`, stage by stage.

- It is in `retrieve.dense` or `retrieve.sparse` but not in the final list: the merge or the reranker pushed it out. Compare its rank in each list.
- It is in neither: no search found it within its 30 candidates. That is a question of wording or of how the page was chunked, and no setting of the merge will fix it.
- `query.terms` is 0: the question contained only stop words, so keyword search had nothing to match on.
- `fuse.in_both` is low: the two searches disagree, and each result rests on one of them. In the trace above they share 6 of their 30 candidates.

**"It failed."** The failing span is marked as an error, with the exception attached, and so is every span around it. A search against a collection that was never built fails in `vector_store.query` with "Collection ... does not exist yet".

## What the first traces showed

Reading a handful of traces turned up three things that the summary numbers had not:

1. **The two searches run one after the other.** Neither needs the other's result. Run side by side, a hybrid search should take about as long as the slower one: an estimated 28 ms instead of 36.
2. **Every call to Qdrant is two round trips.** The store asks whether the collection exists before it queries it. That check takes about 3 ms, and a hybrid search pays it twice.
3. **Reranking is 62% of a reranked request.** That is the cost [ADR 0011](adr/0011-hybrid-retrieval-by-default.md) weighed against its gain.

The first two are improvements to make. Both are small, and neither would have been found by looking at one total.

## When no trace backend is running

With tracing switched on and nothing listening, a search still works and still exits with success. The trace is lost: the exporter retries for a few seconds, then gives up and says that the spans could not be exported. In one measurement the command took about five seconds longer.

## Which backend

The local backend is [Jaeger](https://www.jaegertracing.io/): one container of 57 MB that uses about 20 MB of memory and accepts OTLP directly. Spans are sent over OTLP, so another backend needs a different `HANDBOOK_RAG_OTLP_ENDPOINT` and no change to the code.

Backends built for language-model applications show token counts and cost per request. They differ from Jaeger only on spans of model calls, and the pipeline has none yet. The comparison that ADR 0008 asks for is therefore made when the answering step is traced.

## Not traced yet

- **Answer generation**: the model call, its token counts and whether the assistant declined. These arrive with the answering step.
- **Evaluation runs and indexing.** An evaluation run records per-question latency in its own files.
