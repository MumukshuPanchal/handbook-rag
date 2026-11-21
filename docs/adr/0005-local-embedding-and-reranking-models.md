# ADR 0005: Local open-weight models for embedding and reranking

- Status: Accepted
- Date: 2026-11-21

## Context

Retrieval needs three models: one that turns text into dense vectors, a way to build sparse keyword vectors, and a cross-encoder that reranks candidates. They can be called through a hosted API or run locally.

Requirements:

- anyone can reproduce the results without creating extra accounts;
- retrieval costs nothing per query, so the retrieval evaluation can run on every change and in CI;
- indexing the whole handbook takes minutes, so chunking and model experiments stay practical.

We measured embedding speed on the development laptop (8-core Apple M1 Pro) with 240 synthetic passages of about 350 tokens. It is a rough guide, not a rigorous benchmark.

| Library | Model | Device | Passages per second |
|---|---|---|---|
| fastembed (ONNX Runtime) | `BAAI/bge-small-en-v1.5` | CPU | 14 |
| fastembed (ONNX Runtime) | `snowflake/snowflake-arctic-embed-xs` | CPU | 28 |
| fastembed (ONNX Runtime) | `BAAI/bge-base-en-v1.5` | CPU | 4 |
| sentence-transformers (PyTorch) | `BAAI/bge-small-en-v1.5` | CPU | 38 |
| sentence-transformers (PyTorch) | `BAAI/bge-small-en-v1.5` | Apple GPU | 84 |
| sentence-transformers (PyTorch) | `BAAI/bge-base-en-v1.5` | CPU | 17 |
| sentence-transformers (PyTorch) | `BAAI/bge-base-en-v1.5` | Apple GPU | 34 |

Batch size, worker processes and the CoreML provider made no measurable difference to the fastembed numbers. Both libraries produced identical vectors for the same model (cosine similarity 1.00000 on 20 passages).

## Decision

Run open-weight models locally.

- **Dense embeddings and reranking: sentence-transformers.** It was the fastest option on our hardware, on CPU and on GPU, and it uses a GPU when one is present without needing one.
- **Baseline embedding model: `BAAI/bge-small-en-v1.5`** (384 dimensions). Larger models are compared in milestone M3.
- **Reranking: a cross-encoder** loaded through the same library. The specific model is chosen in M3 by measurement.
- **Sparse vectors: BM25**, built with fastembed's BM25 tokenizer. Qdrant applies the IDF weighting. No neural model is involved.

Model names are configuration, and the models sit behind the `Embedder` and `Reranker` interfaces.

## Alternatives considered

- **Hosted embedding API (Voyage, OpenAI, Cohere).** Typically stronger models and fast bulk indexing. Rejected for now: a second paid vendor, an account for anyone reproducing the work, and network calls inside tests.
- **fastembed for everything.** Much lighter to install, and it was our first plan. Rejected after the measurement above: indexing was 3 to 6 times slower on our hardware, which would make every chunking or model experiment slow. Because the vectors are identical, it remains an option for a lighter serving image later, with no re-indexing.

## Consequences

- Retrieval is free to run and works offline once the models are downloaded.
- Any embedding or reranking model published for sentence-transformers can be tried by changing configuration.
- The same library supports fine-tuning an embedding model on our own data, should we attempt it later.
- PyTorch is a heavy dependency. CI and the container image must install the CPU-only build explicitly, or they pull several gigabytes of GPU libraries.
- Speed depends on hardware. The numbers above describe one laptop; the container runs on CPU and must be measured separately.
- A stronger hosted model might retrieve better. The interface keeps that option open, and the evaluation would show the difference.
