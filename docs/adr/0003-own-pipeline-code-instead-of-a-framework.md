# ADR 0003: Write the pipeline ourselves instead of adopting a RAG framework

- Status: Proposed
- Date: 2026-11-21

## Context

RAG frameworks such as LangChain, LlamaIndex and Haystack ship ready-made document loaders, text splitters, retrievers and answer chains. They get a demo running quickly.

This project has two goals that pull the other way. First, every stage has to be measured and tuned separately, which needs direct access to what goes into and comes out of each stage. Second, the pipeline has to be fully explainable by the people who built it.

## Decision

The pipeline is plain Python: one module per stage, each behind a small interface (`Embedder`, `VectorStore`, `Retriever`, `Reranker`, `Generator`). A stage can be replaced without touching the others.

Focused libraries do the heavy lifting inside those stages: the Qdrant client, sentence-transformers, the Anthropic SDK, FastAPI and OpenTelemetry. None of them owns the control flow.

## Alternatives considered

- **LangChain or LlamaIndex.** Faster start and a large catalogue of integrations. Rejected because the prompts, scoring and ranking steps we need to measure sit inside framework abstractions, and a framework upgrade can change behaviour underneath an evaluation.
- **Haystack.** A cleaner pipeline model, with the same underlying trade-off for this project.

## Consequences

- Every prompt, score and ranking step is in this repository, readable and unit-tested.
- Fewer dependencies to pin and audit.
- We write the glue ourselves: Markdown loading, chunking, rank fusion. That is a few hundred lines we must test and maintain.
- No ready-made integrations. Supporting another vector store or model provider means implementing the matching interface.
