# ADR 0011: Hybrid retrieval by default, reranking off

- Status: Accepted
- Date: 2026-11-22

## Context

The pipeline can retrieve by meaning (dense vectors), by keyword (BM25), or by both merged with reciprocal rank fusion (hybrid). It can also rerank the candidates with a cross-encoder. Until now the default was dense retrieval, for no better reason than that it was built first. The roadmap says defaults are chosen from measurements.

The measurements are on the golden set: 132 questions the handbook answers, checked by a maker-checker process and described in its [dataset card](../../evals/golden/README.md). They were made on the full index of 38,881 chunks. The answer model reads five chunks, so Hit@5 is the number that bounds answer quality.

| Variant | Hit@1 | Hit@5 (95% interval) | MRR@10 | Median time |
|---|---|---|---|---|
| Dense | 0.583 | 0.856 (0.795 to 0.909) | 0.705 | 26 ms |
| Keyword (BM25) | 0.727 | 0.932 (0.886 to 0.970) | 0.813 | 10 ms |
| Hybrid | 0.697 | 0.962 (0.924 to 0.992) | 0.807 | 33 ms |
| Hybrid with reranking | 0.773 | 0.939 (0.894 to 0.977) | 0.852 | 247 ms |

Compared on the same questions:

| Comparison | Change in Hit@5 (95% interval) | Better / worse | Reading |
|---|---|---|---|
| Hybrid against dense | +0.106 (+0.053 to +0.159) | 14 / 0 | clear |
| Keyword against dense | +0.076 (+0.015 to +0.144) | 15 / 5 | clear |
| Hybrid against keyword | +0.030 (-0.008 to +0.076) | 6 / 2 | unclear |
| Reranking against hybrid | -0.023 (-0.061 to +0.015) | 2 / 5 | unclear |

Two experiments on other settings found nothing to change: half-size chunks did not help, and neither did other values of the fusion constant or more candidates ([experiment log](../experiments/2026-11-22-chunk-size-and-fusion.md)).

## Decision

- `retrieval_mode` defaults to `hybrid`.
- `rerank` stays `false`. Reranking remains available per command and in the settings.
- `chunk_max_tokens` (320), `rrf_k` (60) and `fusion_candidates` (30) stay as they are.

## Alternatives considered

- **Keep dense retrieval as the default.** Hybrid is clearly better at five results, and it lost no question that dense retrieval had found.
- **Keyword search alone.** It is the fastest, and on this set its difference from hybrid is not clear. It was not chosen for two reasons. Most of the questions were written by a model looking at the passage, which may favour matching on words. And on the ten questions written without the handbook, keyword search was the weakest single search at rank 1 (0.40 against 0.70 for dense). Ten questions do not make that clear either, so this choice rests on judgement more than on measurement.
- **Hybrid with reranking.** At five results it does not help. At rank 1 it gains 0.076, with an interval that just touches zero, and it makes search about seven times slower. It may still earn its place: on the ten blind questions it lifted Hit@1 from 0.60 to 0.90, and the reranker tested is small and from 2021.

## Consequences

- Every search now runs two queries and needs the keyword encoder as well as the embedding model. The median time rises from 26 ms to 33 ms.
- The index already stores both kinds of vector, so nothing has to be re-indexed.
- The runs named `baseline` in `evals/runs/retrieval` remain the dense baseline that later variants are compared against.
- The evidence has limits. The golden set was checked by models and not verified by a person, and 122 of its 132 answerable questions were written from their passage. The choice between hybrid and keyword-only is to be looked at again when the larger set of blind questions exists (#43).
- The fusion gives both searches equal weight. A weighted fusion is the next experiment.
- Reranking is to be measured again with a newer model, and once answers are evaluated, because rank 1 matters more if fewer chunks are passed to the model.
