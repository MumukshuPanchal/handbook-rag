# Production RAG Pipeline

A Retrieval-Augmented Generation (RAG) system built the way a production team would build it: answers grounded in source documents and cited, quality measured by an evaluation suite, and every request traced for latency and cost.

**Status: planning.** The use case, architecture, and roadmap are being defined. This README is updated as each milestone lands.

## Planned scope

- **Retrieval and generation**: ingest documents, retrieve the relevant passages, and answer with citations
- **Evaluation**: a curated test set with retrieval and answer-quality metrics, run in CI so regressions are caught
- **Observability**: traces, latency, and cost for every request
- **Production engineering**: tested code, CI, containerized services, and written design decisions
