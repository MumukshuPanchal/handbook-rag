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
- **Keyword search alone.** This is the close call, and the measurements do not settle it. Keyword search is the fastest (10 ms against 33 ms). At five results hybrid is ahead by 0.030, which is 6 questions gained and 2 lost, with an interval that includes zero. At rank 1 keyword search is ahead by 0.030 (0.727 against 0.697), also with an interval that includes zero (-0.098 to +0.038 for hybrid against keyword). Hybrid was chosen for three reasons, and they are judgement more than measurement:
  - The answer model reads five chunks, so the number that matters most is Hit@5, and that is where hybrid has the higher value: it missed 5 questions and keyword search missed 9.
  - Hybrid depends less on how a question is worded. Dense search found 5 of the 9 questions keyword search missed, and keyword search found 15 of the 19 that dense search missed. People will ask in their own words, and which of the two searches copes with a given wording cannot be known in advance.
  - On the ten questions written without the handbook, hybrid was ahead of keyword search at rank 1 (0.60 against 0.40: 2 questions better, none worse, interval +0.00 to +0.50).

  That last point is weak evidence and must not be read as more. Ten questions cannot make a difference clear. All four variants found every one of those ten answers within five results, so at five results the blind questions do not separate them at all. And on the same ten questions hybrid was behind dense search at rank 1 (0.60 against 0.70), so they do not show that hybrid is the best choice, only that keyword search alone was the weakest there. The other 122 questions were written by a model looking at the passage, which might favour matching on words. A simple count of shared words does not show that (35% for those questions, 41% for the blind ones), so it stays a suspicion.
- **Hybrid with reranking.** At five results it does not help: -0.023, with 2 questions gained and 5 lost. At rank 1 it gains 0.076, with an interval that just touches zero (+0.000 to +0.152), and it makes search about seven times slower (247 ms against 33 ms). It may still earn its place. On the ten blind questions it lifted Hit@1 from 0.60 to 0.90, but that is 3 questions, with an interval from +0.00 to +0.60: a hint, not a finding. The reranker tested is also a small, early model, so this says little about newer ones.

## Consequences

- Every search now runs two queries and needs the keyword encoder as well as the embedding model. The median time rises from 26 ms to 33 ms.
- The index already stores both kinds of vector, so nothing has to be re-indexed.
- The run named `baseline` in `evals/runs/retrieval` remains the dense baseline that later variants are compared against. Because the default mode is no longer dense, `eval-retrieval` without `--variant` now writes a scratch run named `current`, which git ignores. A run has to be named to be kept, so the committed baseline cannot be replaced by accident.
- The evidence has limits. The golden set was checked by models and not verified by a person, and 122 of its 132 answerable questions were written from their passage. The choice between hybrid and keyword-only is to be looked at again when the larger set of blind questions exists (#43).
- The fusion gives both searches equal weight. A weighted fusion is the next experiment.
- Reranking is to be measured again with a newer model, and once answers are evaluated, because rank 1 matters more if fewer chunks are passed to the model.
