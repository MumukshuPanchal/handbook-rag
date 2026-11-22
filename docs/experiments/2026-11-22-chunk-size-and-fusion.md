# Experiment log: chunk size and fusion settings

- Date: 2026-11-22
- Golden set: 132 answerable questions, checked by a maker-checker process (see the [dataset card](../../evals/golden/README.md))
- Issues: #12 (chunking and embedding experiments), #10 (hybrid retrieval)

## Why these experiments

Reading the retrieval misses of the first runs, on the first draft of the golden set, gave two hypotheses.

1. **Facts get buried.** Most of those misses were single-sentence facts inside a chunk that is mostly about something else. Smaller chunks should surface them.
2. **Fusion buries what only one search finds.** In several of hybrid's misses, keyword search ranked the right chunk first, the embedding model did not have it at all, and the merge pushed it out of the top five. A smaller constant in reciprocal rank fusion, or more candidates, should help.

The experiments were first run on that draft. The questions were then checked and partly reworded, and everything was run again. The numbers below are from the checked set.

## Setup

- Each variant is compared with its counterpart on the same questions (`make eval-compare`). A change is called clear when the 95% interval of the per-question difference excludes zero.
- Chunk size: a second index built with `HANDBOOK_RAG_CHUNK_MAX_TOKENS=160`, `HANDBOOK_RAG_CHUNK_MIN_TOKENS=40` and `HANDBOOK_RAG_CHUNK_OVERLAP_TOKENS=20`. Everything else is unchanged.
- Fusion: `HANDBOOK_RAG_RRF_K` set to 20, 10 and 5, and `HANDBOOK_RAG_FUSION_CANDIDATES` set to 100, on the existing index.

## Results

All runs, as printed by `make eval-table`:

| Run | What it is | Questions | Hit@5 (95% interval) | Hit@1 | Hit@10 | MRR@10 | Median time |
|---|---|---|---|---|---|---|---|
| baseline | Dense retrieval only (bge-small-en-v1.5). | 132 | 0.856 (0.795 to 0.909) | 0.583 | 0.909 | 0.705 | 26 ms |
| v1 | Keyword search only (BM25). | 132 | 0.932 (0.886 to 0.970) | 0.727 | 0.977 | 0.813 | 10 ms |
| v2 | Dense and BM25 merged with reciprocal rank fusion. | 132 | 0.962 (0.924 to 0.992) | 0.697 | 0.985 | 0.807 | 33 ms |
| v3 | Hybrid, then the top 30 reranked by a cross-encoder (ms-marco-MiniLM-L6-v2). | 132 | 0.939 (0.894 to 0.977) | 0.773 | 0.970 | 0.852 | 247 ms |
| v4 | Dense, with 160-token chunks. | 132 | 0.826 (0.758 to 0.894) | 0.614 | 0.864 | 0.698 | 32 ms |
| v5 | Keyword, with 160-token chunks. | 132 | 0.871 (0.811 to 0.924) | 0.705 | 0.939 | 0.783 | 6 ms |
| v6 | Hybrid, with 160-token chunks. | 132 | 0.917 (0.871 to 0.962) | 0.652 | 0.962 | 0.756 | 33 ms |
| v7 | Hybrid with reranking, with 160-token chunks. | 132 | 0.932 (0.879 to 0.970) | 0.742 | 0.962 | 0.818 | 147 ms |
| v8 | Hybrid with the fusion constant k lowered from 60 to 20. | 132 | 0.955 (0.917 to 0.985) | 0.697 | 0.985 | 0.807 | 38 ms |
| v9 | Hybrid with the fusion constant k lowered from 60 to 10. | 132 | 0.962 (0.924 to 0.992) | 0.697 | 0.985 | 0.810 | 35 ms |
| v10 | Hybrid with the fusion constant k lowered from 60 to 5. | 132 | 0.955 (0.917 to 0.985) | 0.697 | 0.985 | 0.813 | 35 ms |
| v11 | Hybrid with 100 candidates from each search instead of 30. | 132 | 0.947 (0.902 to 0.985) | 0.697 | 0.985 | 0.804 | 46 ms |

### Half-size chunks against the current size

| Variant | Before | After | Change in Hit@5 (95% interval) | Better / worse | Reading |
|---|---|---|---|---|---|
| Dense | 0.856 | 0.826 | -0.030 (-0.091 to +0.030) | 6 / 10 | unclear |
| Keyword | 0.932 | 0.871 | -0.061 (-0.106 to -0.015) | 1 / 9 | clear |
| Hybrid | 0.962 | 0.917 | -0.045 (-0.098 to +0.000) | 3 / 9 | unclear |
| Hybrid with reranking | 0.939 | 0.932 | -0.008 (-0.045 to +0.030) | 3 / 4 | unclear |

### The same amount of text: ten small chunks against five large ones

| Search | Hit@5 with five 320-token chunks | Hit@10 with ten 160-token chunks | Difference (95% interval) | Reading |
|---|---|---|---|---|
| Dense | 0.856 | 0.864 | +0.008 (-0.053 to +0.068) | unclear |
| Keyword | 0.932 | 0.939 | +0.008 (-0.038 to +0.053) | unclear |
| Hybrid | 0.962 | 0.962 | +0.000 (-0.045 to +0.045) | unclear |
| Hybrid with reranking | 0.939 | 0.962 | +0.023 (-0.008 to +0.061) | unclear |

### Fusion settings against the current hybrid

| Variant | Before | After | Change in Hit@5 (95% interval) | Better / worse | Reading |
|---|---|---|---|---|---|
| Constant 20 | 0.962 | 0.955 | -0.008 (-0.023 to +0.000) | 0 / 1 | unclear |
| Constant 10 | 0.962 | 0.962 | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear |
| Constant 5 | 0.962 | 0.955 | -0.008 (-0.023 to +0.000) | 0 / 1 | unclear |
| 100 candidates | 0.962 | 0.947 | -0.015 (-0.038 to +0.000) | 0 / 2 | unclear |

## What we learned

1. **Half-size chunks do not help, and they make keyword search clearly worse.** Keyword search lost 9 questions and gained 1. Hybrid lost 9 and gained 3. Its interval ends exactly at zero, so by our rule that difference is not clear, but it points the same way. On the first draft of the golden set the loss for hybrid was clear.
2. **Smaller chunks rescue some questions, at a price.** With the embedding model alone, 6 questions were found only with 160-token chunks and 10 only with 320-token chunks. A small chunk has less text around the fact to match on.
3. **With the same amount of text, the two sizes are level.** No difference in the second table is clear.
4. **Half-size chunks cost more.** 71,011 chunks against 38,881, and 255 MB against 193 MB on disk in Qdrant.
5. **The fusion constant and the number of candidates have no measurable effect.** At most 2 of the 132 questions changed, and none for the better. With equal weights, a constant of 60 and lists of 30, any chunk that both searches return outranks a chunk that only one returns, even at rank 1: first place in one list scores 1/61 = 0.0164, and thirtieth place in both scores 2/90 = 0.0222. A smaller constant weakens that, and it did not help.

Both hypotheses were wrong in the form we tested them. The observations behind them still stand.

## Decision

No default changes: `chunk_max_tokens` stays 320, `rrf_k` stays 60 and `fusion_candidates` stays 30.

## Next

- Weighted fusion, with the weight chosen on held-out questions.
- Keep the heading of a merged section in the chunk's context header.
- Small-to-big retrieval: match on a small unit and return the larger chunk around it.
- A larger set of questions written without the passage (#43), because the checked set still consists mostly of questions written from their passage.

## How to reproduce

```bash
# Half-size chunks: build a second index (about 7 minutes), then score a variant on it.
export HANDBOOK_RAG_CHUNK_MAX_TOKENS=160 HANDBOOK_RAG_CHUNK_MIN_TOKENS=40 HANDBOOK_RAG_CHUNK_OVERLAP_TOKENS=20
uv run handbook-rag index
uv run handbook-rag eval-retrieval --variant v6 --mode hybrid --description "Hybrid, with 160-token chunks."
unset HANDBOOK_RAG_CHUNK_MAX_TOKENS HANDBOOK_RAG_CHUNK_MIN_TOKENS HANDBOOK_RAG_CHUNK_OVERLAP_TOKENS

# Fusion constant, on the normal index.
HANDBOOK_RAG_RRF_K=10 uv run handbook-rag eval-retrieval --variant v9 --mode hybrid --description "Hybrid with the fusion constant k lowered from 60 to 10."

make eval-compare base=v2 candidate=v6
make eval-table
```
