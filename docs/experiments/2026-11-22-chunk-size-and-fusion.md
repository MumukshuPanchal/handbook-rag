# Experiment log: chunk size and fusion settings

- Date: 2026-11-22
- Status: provisional. The golden set is a draft that no person has reviewed yet, so nothing here changes a default.
- Issues: #12 (chunking and embedding experiments), #10 (hybrid retrieval)

## Why these experiments

Reading the 24 retrieval misses of the first runs gave two hypotheses.

1. **Facts get buried.** 17 of the misses were single-sentence facts inside a chunk that is mostly about something else. Smaller chunks should surface them.
2. **Fusion buries what only one search finds.** In 3 of hybrid's 6 misses, keyword search ranked the right chunk 1st, 4th and 1st, the embedding model did not have it at all, and the merge pushed it out of the top five. A smaller constant in reciprocal rank fusion, or more candidates, should help.

## Setup

- 123 answerable questions of the draft golden set, on the full index of the pinned handbook snapshot.
- Each variant is compared with its counterpart on the same questions (`make eval-compare`). A change is called clear when the 95% interval of the per-question difference excludes zero.
- Chunk size: a second index built with `HANDBOOK_RAG_CHUNK_MAX_TOKENS=160`, `HANDBOOK_RAG_CHUNK_MIN_TOKENS=40` and `HANDBOOK_RAG_CHUNK_OVERLAP_TOKENS=20`. Everything else is unchanged.
- Fusion: `HANDBOOK_RAG_RRF_K` set to 20, 10 and 5, and `HANDBOOK_RAG_FUSION_CANDIDATES` set to 100, on the existing index.

## Results

All runs, as printed by `make eval-table`:

| Run | What it is | Questions | Hit@5 (95% interval) | Hit@1 | Hit@10 | MRR@10 | Median time |
|---|---|---|---|---|---|---|---|
| baseline | Dense retrieval only (bge-small-en-v1.5). | 123 | 0.837 (0.772 to 0.902) | 0.610 | 0.919 | 0.719 | 32 ms |
| v1 | Keyword search only (BM25). | 123 | 0.943 (0.902 to 0.984) | 0.780 | 0.984 | 0.852 | 5 ms |
| v2 | Dense and BM25 merged with reciprocal rank fusion. | 123 | 0.951 (0.911 to 0.984) | 0.740 | 0.984 | 0.831 | 38 ms |
| v3 | Hybrid, then the top 30 reranked by a cross-encoder (ms-marco-MiniLM-L6-v2). | 123 | 0.943 (0.902 to 0.984) | 0.780 | 0.967 | 0.856 | 221 ms |
| v4 | Dense, with 160-token chunks. | 123 | 0.821 (0.748 to 0.886) | 0.634 | 0.862 | 0.708 | 29 ms |
| v5 | Keyword, with 160-token chunks. | 123 | 0.911 (0.854 to 0.959) | 0.740 | 0.951 | 0.813 | 10 ms |
| v6 | Hybrid, with 160-token chunks. | 123 | 0.894 (0.837 to 0.943) | 0.675 | 0.951 | 0.776 | 38 ms |
| v7 | Hybrid with reranking, with 160-token chunks. | 123 | 0.919 (0.870 to 0.959) | 0.772 | 0.959 | 0.838 | 144 ms |
| v8 | Hybrid with the fusion constant k lowered from 60 to 20. | 123 | 0.935 (0.886 to 0.976) | 0.740 | 0.984 | 0.831 | 42 ms |
| v9 | Hybrid with the fusion constant k lowered from 60 to 10. | 123 | 0.951 (0.911 to 0.984) | 0.740 | 0.984 | 0.833 | 30 ms |
| v10 | Hybrid with the fusion constant k lowered from 60 to 5. | 123 | 0.943 (0.902 to 0.984) | 0.740 | 0.984 | 0.835 | 29 ms |
| v11 | Hybrid with 100 candidates from each search instead of 30. | 123 | 0.943 (0.902 to 0.984) | 0.740 | 0.976 | 0.829 | 50 ms |

### Half-size chunks against the current size

| Variant | Before | After | Change in Hit@5 (95% interval) | Better / worse | Reading |
|---|---|---|---|---|---|
| Dense | 0.837 | 0.821 | -0.016 (-0.081 to +0.049) | 8 / 10 | unclear |
| Keyword | 0.943 | 0.911 | -0.033 (-0.065 to -0.008) | 0 / 4 | clear |
| Hybrid | 0.951 | 0.894 | -0.057 (-0.106 to -0.016) | 1 / 8 | clear |
| Hybrid with reranking | 0.943 | 0.919 | -0.024 (-0.065 to +0.016) | 2 / 5 | unclear |

### Fusion settings against the current hybrid

| Variant | Before | After | Change in Hit@5 (95% interval) | Better / worse | Reading |
|---|---|---|---|---|---|
| Constant 20 | 0.951 | 0.935 | -0.016 (-0.041 to +0.000) | 0 / 2 | unclear |
| Constant 10 | 0.951 | 0.951 | +0.000 (-0.024 to +0.024) | 1 / 1 | unclear |
| Constant 5 | 0.951 | 0.943 | -0.008 (-0.041 to +0.024) | 1 / 2 | unclear |
| 100 candidates | 0.951 | 0.943 | -0.008 (-0.024 to +0.000) | 0 / 1 | unclear |

## What we learned

1. **Half-size chunks make keyword and hybrid retrieval clearly worse.** Hybrid lost 8 questions and gained 1.
2. **They do rescue buried facts, at a price.** Of the 15 buried facts the embedding model missed at 320 tokens, 7 were found at 160. But 10 other questions that were found at 320 were missed at 160. A small chunk has less text around the fact to match on.
3. **With the same amount of text, the two sizes are level.** Ten 160-token chunks against five 320-token chunks: hybrid 0.951 against 0.951, dense 0.862 against 0.837, keyword 0.951 against 0.943. None of these differences is clear.
4. **Half-size chunks cost more.** 71,011 chunks against 38,881, and 255 MB against 193 MB on disk in Qdrant.
5. **The fusion constant and the number of candidates have no measurable effect.** With equal weights, a constant of 60 and lists of 30, any chunk that both searches return outranks a chunk that only one returns, even at rank 1: first place in one list scores 1/61 = 0.0164, and thirtieth place in both scores 2/90 = 0.0222. A smaller constant weakens that, and it helped and hurt in equal measure.

Both hypotheses were wrong in the form we tested them. The observations behind them still stand.

## Decision

No default changes: `chunk_max_tokens` stays 320, `rrf_k` stays 60 and `fusion_candidates` stays 30.

## Next

- Weighted fusion, with the weight chosen on held-out questions.
- Keep the heading of a merged section in the chunk's context header.
- Small-to-big retrieval: match on a small unit and return the larger chunk around it.
- Repeat on the reviewed golden set, with hand-written questions, before any default changes.

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
