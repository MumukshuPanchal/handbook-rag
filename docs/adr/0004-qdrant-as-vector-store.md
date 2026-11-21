# ADR 0004: Qdrant as the vector store

- Status: Accepted
- Date: 2026-11-21

## Context

Chunks of the handbook must be stored with their vectors and metadata, and searched at query time. The store has to support:

- dense vector search (search by meaning);
- sparse vector search (keyword-style search), so the two can be combined into hybrid retrieval;
- filtering on metadata such as handbook section;
- running locally with no managed service or account;
- use in tests without starting a server.

## Decision

Use Qdrant.

One collection holds, for each chunk, a dense vector, a sparse vector and a payload (text, source path, heading path, URL, character offsets). The collection name includes the embedding model and chunker version, so different configurations can sit side by side during experiments.

The running service talks to a Qdrant container. Tests and CI use the client's embedded in-memory mode, which exposes the same API without a server.

A spike against `qdrant-client` 1.20 confirmed the pieces we rely on: dense search, sparse search with IDF weighting, metadata filters, and fusing both result lists.

## Alternatives considered

- **PostgreSQL with pgvector.** A strong choice for a team that already runs Postgres: one database, SQL filters, transactions. Not chosen here because keyword search needs a separate full-text setup, and tests would need a running database.
- **FAISS.** A vector search library, not a database. Payloads, filtering and persistence would be ours to build.
- **Chroma.** A simple embedded store. Not evaluated in depth; Qdrant already met every requirement above in the spike.
- **A managed service such as Pinecone.** Needs an account and network access, which breaks one-command local reproduction.

## Consequences

- Hybrid retrieval comes from a single store.
- Tests are fast and need no containers.
- One more container to run alongside the API.
- The embedded mode is not the server engine. Latency numbers must be measured against the server, not in tests.
- Dependence on Qdrant is limited to one module behind the `VectorStore` interface.
