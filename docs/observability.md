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

The command starts a new process each time, so its trace shows a cold start: the first run of the embedding model alone took about 400 ms when measured. The traces below are warm. Each is the last of six searches made in one process, which is how a running service behaves.

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

## Through the HTTP API

A request to the API is traced the same way. Its response carries the trace id in `trace_id` and in the `X-Trace-Id` header, and the header is also on a failed request, which is the one a caller most needs an id for. The web framework adds its own spans around ours: the trace starts with `POST /search`, which records the method, the route and the status code, and our `search` span sits inside it.

Requests to `/health` are not traced. Something asks for it every few seconds, and those traces would bury the ones made by people.

Start the service with `HANDBOOK_RAG_TRACING=true make serve`.

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

Every `search` and `retrieve.*` span carries `results.count`, `results.chunk_ids`, `results.doc_ids` and `results.scores`. A `vector_store.query` span carries the count only, because the span around it has the rest. Our own attributes start with `handbook_rag.`. The database spans also use the standard `db.*` names, which is why Jaeger shows them with a database icon.

## What is not recorded

A trace holds ids, scores, sizes and timings, and the names of things: the collection, the models, the handbook section a search was limited to. It does not hold what was asked or what was found.

- **The question** is recorded as its length only. Questions can contain personal or confidential details, and traces are usually kept longer and read by more people than the application's own data.
- **Chunk text** is never copied into a trace. A chunk id leads back to the text in the index.
- **An error** is recorded as its kind only, for example `handbook_rag.index.store.VectorStoreError`. An error message can quote the input that caused it, and a stack trace holds file paths from the machine it ran on.

`HANDBOOK_RAG_TRACE_CONTENT=true` adds the question's text and the full message and stack trace of an error, for debugging on your own machine.

## Using a trace to debug

**"It was slow."** Find the longest bar. In the trace above the reranker takes 208 of 336 ms.

One trace is one measurement, and on a busy laptop they vary: the search part of that trace took 127 ms, and the same question without reranking took 36 ms a moment earlier. For typical figures, 40 warm searches without reranking had a median of 43 ms, of which the embedding model took 18 and the two Qdrant calls 13 and 11. These were measured before the fix described below.

**"The right passage was not returned."** Look for its page among `results.doc_ids`, stage by stage.

- It is in `retrieve.dense` or `retrieve.sparse` but not in the final list: the merge or the reranker pushed it out. Compare its rank in each list.
- It is in neither: no search found it within its 30 candidates. That is a question of wording or of how the page was chunked. Asking each search for more candidates (`HANDBOOK_RAG_FUSION_CANDIDATES`) is the only setting that could bring it in, and the merge constant cannot.
- `query.terms` is 0: the question contained only stop words, so keyword search had nothing to match on.
- `fuse.in_both` is low: the two searches disagree, and each result rests on one of them. In the trace above they share 6 of their 30 candidates.

**"It failed."** The failing span is marked as an error with the kind of exception, and so is every span around it. A search against a collection that was never built fails in `vector_store.query` with a `CollectionMissingError`; the command line prints the message, "Collection ... does not exist yet". The span of a failed store call also records, in `error.type`, the kind of error the store's client raised underneath.

## What the first traces showed

Reading a handful of traces turned up three things that the summary numbers had not:

1. **The two searches run one after the other.** Neither needs the other's result. Run side by side, a hybrid search should take about as long as the slower of the two, which would save roughly the time of the keyword search: an estimated 8 to 11 ms.
2. **Every call to Qdrant was two round trips.** The store asked whether the collection exists before it queried it, and a hybrid search paid for that twice.
3. **Reranking is most of a reranked request.** It is 62% of the trace above, in which the search part was unusually slow. By the medians recorded in [ADR 0011](adr/0011-hybrid-retrieval-by-default.md), 247 ms with reranking against 33 without, it is nearer 87%. That is the cost the decision record weighed against its gain.

Neither of the first two would have been found by looking at one total.

**The second is fixed.** The store now asks whether the collection exists only after a query has failed, to explain the failure. Measuring the gain needed some care: the time of a whole run varies by ten milliseconds from one run to the next on a busy laptop, which hides a difference of a few. So both versions were timed back to back on the same 300 queries, in alternating order, and compared pair by pair, the same way two search variants are compared in the [evaluation guide](evaluation.md).

| | Median time of one store query |
|---|---|
| Asking first | 18.1 ms |
| Asking only after a failure | 12.7 ms |

A query got 5.9 ms faster on average (95% interval 4.8 to 7.0), and was faster in 256 of the 300 pairs. That was on a busy laptop, and only dense queries were timed. On a quieter run the check took about 3 ms on its own, so the size of the saving depends on how loaded the machine is. What does not depend on it: every query now makes one round trip where it made two.

**The first is left as it is.** Running the two searches side by side would save an estimated 8 to 11 ms and add threads to the search path. Once answers are generated, a request will spend a second or more in the language model, and nobody would notice the difference. It is written down so that it is a decision and not an oversight.

## When no trace backend is running

With tracing switched on and nothing listening, a search still works and still exits with success. The trace is lost: the exporter retries for up to three seconds, which is its time limit, then says that the spans could not be exported. The command still prints a trace id, which then leads nowhere.

## Which backend

The local backend is [Jaeger](https://www.jaegertracing.io/): one container, a download of 53 MB, that used 20 to 35 MB of memory here and accepts OTLP directly. Spans are sent over OTLP, so another backend needs a different `HANDBOOK_RAG_OTLP_ENDPOINT` and no change to the code. A hosted backend also wants a credential, which the exporter reads from the standard `OTEL_EXPORTER_OTLP_TRACES_HEADERS` variable.

Backends built for language-model applications show token counts and cost per request. What sets them apart is how they present spans of model calls, and the pipeline has none yet. The comparison that ADR 0008 asks for is therefore made when the answering step is traced.

## Not traced yet

- **Answer generation**: the model call, its token counts and whether the assistant declined. These arrive with the answering step.
- **Evaluation runs and indexing.** An evaluation run records per-question latency in its own files.
