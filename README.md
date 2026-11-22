# Production RAG Pipeline

A knowledge assistant that answers questions from a company handbook, built the way a production team would build it: answers grounded in retrieved passages and cited, quality measured by an evaluation suite, and every request traced for latency and cost.

**Status: retrieval is built and measured; answering is next.** The handbook is indexed, hybrid search is the default, and every change is judged on a checked set of test questions. Answering with citations is written and still has to be verified against the model API. This README is updated as each milestone lands.

## Documents

| Document | What it covers |
|---|---|
| [Project charter](docs/charter.md) | The problem, the users, success metrics and non-goals |
| [Architecture](docs/architecture.md) | How the system works, with diagrams and a glossary |
| [Decision records](docs/adr/) | Why each major technical choice was made |
| [Roadmap](docs/roadmap.md) | Milestones and how work flows |
| [Evaluation](docs/evaluation.md) | The golden set, the metrics and how to read a run |
| [Experiments](docs/experiments/) | What was tried, with numbers, including what did not work |

## Planned scope

- **Retrieval and generation**: ingest documents, retrieve the relevant passages, and answer with citations
- **Evaluation**: a curated test set with retrieval and answer-quality metrics, run in CI so regressions are caught
- **Observability**: traces, latency, and cost for every request
- **Production engineering**: tested code, CI, containerized services, and written design decisions

## Results so far

Retrieval on 132 test questions over 38,881 chunks of the handbook. The table is printed by `make eval-table` from the runs in [`evals/runs/retrieval`](evals/runs/retrieval).

| Run | What it is | Questions | Hit@5 (95% interval) | Hit@1 | MRR@10 | Median time |
|---|---|---|---|---|---|---|
| baseline | Dense retrieval only (bge-small-en-v1.5). | 132 | 0.856 (0.795 to 0.909) | 0.583 | 0.705 | 26 ms |
| v1 | Keyword search only (BM25). | 132 | 0.932 (0.886 to 0.970) | 0.727 | 0.813 | 10 ms |
| v2 | Dense and BM25 merged with reciprocal rank fusion. | 132 | 0.962 (0.924 to 0.992) | 0.697 | 0.807 | 33 ms |
| v3 | Hybrid, then the top 30 reranked by a cross-encoder (ms-marco-MiniLM-L6-v2). | 132 | 0.939 (0.894 to 0.977) | 0.773 | 0.852 | 247 ms |

- **Hybrid search is the default.** It clearly beats dense search alone: +0.106 on Hit@5 (95% interval +0.053 to +0.159), with 14 questions gained and none lost. The reasoning is in [ADR 0011](docs/adr/0011-hybrid-retrieval-by-default.md).
- **Reranking is off by default.** It brings no clear gain at five results, and it makes search about seven times slower.
- **Two experiments did not help**, and are written up anyway: half-size chunks, and other settings of the rank fusion ([experiment log](docs/experiments/2026-11-22-chunk-size-and-fusion.md)).
- **Targets.** The [charter](docs/charter.md) asks for Recall@5 of at least 0.85 and MRR@10 of at least 0.70. Hybrid search reaches 0.962 and 0.807. Answer quality is not measured yet.

**How far to trust these numbers.** The test questions were made by one model session and checked by another, over several rounds. No person has verified them one by one, and most were written by a model that could see the passage. The [dataset card](evals/golden/README.md) says how the set was made, what the checks found, and what its limits are.

## Data

The assistant is built on the [GitLab Handbook](https://handbook.gitlab.com/), which GitLab publishes under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) licence. The handbook text is not stored in this repository. `make snapshot` downloads its Markdown pages at a pinned commit (4,206 pages, 43 MB) and writes a manifest that records the source, commit and a checksum of what was fetched. The reasoning is in [ADR 0002](docs/adr/0002-handbook-corpus.md).

`make chunks` cleans the pages and splits them along their headings into 38,881 chunks with a median of 225 tokens, none longer than the embedding model can read. How and why is in [ADR 0009](docs/adr/0009-structure-aware-chunking.md).

The test questions in [`evals/golden/`](evals/golden/README.md) quote short passages from the handbook. That folder is therefore licensed CC BY-SA 4.0 and credits GitLab, separately from the code, which is under the MIT licence.

This project is not affiliated with or endorsed by GitLab.

## Development

You need [uv](https://docs.astral.sh/uv/), `make`, `git` and Docker.

```bash
make install    # install dependencies and git hooks
make check      # lint, type check and fast tests: everything CI runs
make snapshot   # download the handbook at the pinned commit into data/
make chunks     # clean and chunk it, and print a summary
make qdrant     # start the Qdrant vector store in Docker
make index      # embed the chunks and store them; safe to run again
make search q="how do I expense a laptop"
uv run handbook-rag search "what is compa-ratio" --rerank
make golden-check    # verify every test question against the handbook
make eval-retrieval  # score the default search on the test questions
make eval-table      # print the results table of every committed run
```

The workflow and conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).
