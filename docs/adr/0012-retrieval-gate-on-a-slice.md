# ADR 0012: A retrieval gate in CI, on a slice of the handbook

- Status: Accepted
- Date: 2026-11-22

## Context

[ADR 0007](0007-evaluation-strategy.md) says CI must fail when retrieval quality drops. Neither of the checks that existed can do that.

- **The unit tests** use stand-in models. They show that the code is wired correctly, and cannot see a change that makes the real models retrieve worse.
- **The full evaluation** needs all 38,881 chunks embedded and a running vector store. A CI machine has no GPU. It indexed between 11.5 and 14.2 chunks a second when measured, which puts the whole handbook at 45 to 56 minutes.

This record settles how point 7 of ADR 0007, the regression gate, is built. That point spoke of "a small fixed corpus"; what was built is a fixed slice of the real one.

Two more constraints shaped the design. The handbook text is not stored in this repository ([ADR 0002](0002-handbook-corpus.md)), so a test corpus cannot simply be committed. And a check that fails for reasons unrelated to the change gets ignored, so the gate has to give the same result every time.

> **Note, 2026-11-23.** Some handbook text is stored after all, as the note in ADR 0002 says: short quotes in the test questions and labelled answers, and what saved replies repeat, in one case five whole passages. Those files carry the handbook's licence and credit. The snapshot itself is still not stored, and nothing the gate could index is, so the constraint above still holds.

## Decision

A second CI job, the retrieval gate, runs `handbook-rag eval-gate` on every pull request and on every push to `main`.

1. It takes the handbook snapshot at the pinned commit. CI downloads it and keeps it in a cache between runs.
2. It keeps the pages that hold evidence for a golden set question: 134 pages, 3,280 chunks, 8% of the handbook.
3. It indexes them in memory with the real embedding model and the real keyword encoder. No vector store server is needed.
4. It scores the default retriever on all 132 answerable questions.
5. It fails if a mean score is below its minimum in [`evals/gate.json`](../../evals/gate.json), if any question could not be searched, or if nothing was scored.

The minimums are Hit@5 of 0.97 and MRR@10 of 0.84. The gate measures 0.992 and 0.867.

## What the gate can and cannot tell

A slice is easier to search than the whole handbook, so the gate's scores are not quality numbers. They are only compared with themselves.

| Retriever | Hit@5 on the slice | MRR@10 on the slice | Gate | Hit@5 on the whole handbook |
|---|---|---|---|---|
| Hybrid, the default | 0.992 | 0.867 | passes | 0.962 |
| Dense only | 0.939 | 0.775 | fails on both | 0.856 |
| Keyword only | 0.962 | 0.877 | fails on Hit@5 | 0.932 |

So the gate fails when either half of hybrid search is lost. The hybrid and the dense-only rows were each measured both on a laptop and in CI, with the same result. For dense only, the CI run is [pull request 46](https://github.com/MumukshuPanchal/production-rag-pipeline/pull/46): the gate job was made to search by meaning alone, the unit tests passed, and the gate failed. The keyword-only row was measured on a laptop.

It does not detect small changes. The Hit@5 minimum allows three misses in 132 questions. Hybrid search has one, so a change that costs one or two more questions passes. Keyword search alone has five, which is two too many. Deciding whether a change is an improvement still needs the full evaluation and a paired comparison ([evaluation guide](../evaluation.md)).

## Alternatives considered

- **Run the full evaluation in CI.** Most of an hour of embedding for every pull request, plus a vector store service. It could become a scheduled job later.
- **Commit a small corpus as a test fixture**, which is what the issue first proposed. The handbook's text is deliberately kept out of this repository, and pages written by hand for a test would not behave like real ones. That was learned the hard way: 193 over-long chunks appeared only on the real data.
- **Commit the vectors, and embed only the questions.** Fast, but the gate would no longer exercise chunking or document embedding, and the committed vectors would go stale without anyone noticing.
- **Keep the vectors between CI runs, keyed by model and text.** This would cut the job from about six minutes to under two. It was rejected for now because of one failure it invites: a change to the embedding code, or an upgraded library, would be scored against vectors made by the old code unless the cache key covers every file involved. A gate that is blind in exactly that case is worse than a slow one. It is the first thing to revisit if six minutes becomes a burden, with a check that re-embeds a sample of the cached vectors on every run.
- **Cut each page down to the chunks around the evidence.** Keeping eight chunks either side leaves 1,555 chunks, half the work. But the gate then loses most of its grip. On Hit@5, dense only scores 0.977 and keyword only 0.985, against 0.992 for hybrid: three questions and one question apart. MRR@10 still separates dense only (0.800 against 0.886), but keyword only (0.884) can no longer be told from hybrid at all. The other chunks of the same page are what make the slice hard, so whole pages stay.
- **Add pages without evidence as distractors.** Measured with 0, 100 and 300 extra pages: hybrid's Hit@5 was 0.992 every time, and its MRR@10 was 0.867, 0.878 and 0.873. Dense only lost one more question (0.939 to 0.932), and keyword only did not move. For any one question the other 133 pages already are distractors. Extra pages would add 37% to 99% more chunks to embed for next to no sensitivity, so the setting exists and is 0.
- **Compare with `main` instead of with fixed minimums.** That means running the gate twice. Because the gate gives the same result every time, a fixed floor does the same job, and it is simpler to explain.

## Consequences

- Every pull request waits for the gate. Its first three runs took between 4 minutes 35 seconds and 5 minutes 47 seconds. Nearly all of that is indexing, which is mostly the embedding model: 231 to 284 seconds. On a laptop with a GPU the same command takes just over 2 minutes.
- The scores were the same on the laptop and in CI, to the three decimals that are printed and down to which questions are missed. The minimums therefore need no allowance for the machine, only for changes that are meant to pass.
- A pull request that changes retrieval on purpose also changes `evals/gate.json`, with the full evaluation as its evidence. The minimums are a floor to be raised, not a target.
- The slice follows the golden set. A pull request that changes the questions changes the pages, and has to measure the gate again.
- With cold caches the job needs the network: the handbook's host for the snapshot and the model hub for the models. Once both are restored the gate step fetches nothing, so an outage of either cannot fail it. A cache is saved only by a run that passes, and other branches can use it only once a run on `main` has saved it.
- The models are named, not pinned to a revision. If a model's files changed upstream, the gate would notice only after its cache was rebuilt.
- `make_retriever` can now be given a store, an embedder and a keyword encoder that already exist. `build_corpus` is split into loading the snapshot and chunking pages.
