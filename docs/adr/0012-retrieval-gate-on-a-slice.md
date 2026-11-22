# ADR 0012: A retrieval gate in CI, on a slice of the handbook

- Status: Accepted
- Date: 2026-11-22

## Context

[ADR 0007](0007-evaluation-strategy.md) says CI must fail when retrieval quality drops. Neither of the checks that existed can do that.

- **The unit tests** use stand-in models. They show that the code is wired correctly, and cannot see a change that makes the real models retrieve worse.
- **The full evaluation** needs all 38,881 chunks embedded and a running vector store. A CI machine has no GPU. It embedded 11.5 chunks a second when measured, which puts the whole handbook at about 56 minutes.

Two more constraints shaped the design. The handbook text is not stored in this repository ([ADR 0002](0002-handbook-corpus.md)), so a test corpus cannot simply be committed. And a check that fails for reasons unrelated to the change gets ignored, so the gate has to give the same result every time.

## Decision

A second CI job, the retrieval gate, runs `handbook-rag eval-gate` on every pull request and on every push to `main`.

1. It takes the handbook snapshot at the pinned commit. CI downloads it once and keeps it between runs.
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

So the gate fails when either half of hybrid search is lost. It was also run for real with search forced to dense only, and CI went red (see the pull request that added it).

It does not detect small changes. Keyword search alone misses the Hit@5 minimum by one question, and a change that costs one or two questions passes. Deciding whether a change is an improvement still needs the full evaluation and a paired comparison ([evaluation guide](../evaluation.md)).

## Alternatives considered

- **Run the full evaluation in CI.** About an hour of embedding for every pull request, plus a vector store service. It could become a scheduled job later.
- **Commit a small corpus as a test fixture**, which is what the issue first proposed. The handbook's text is deliberately kept out of this repository, and pages written by hand for a test would not behave like real ones. That was learned the hard way: 193 over-long chunks appeared only on the real data.
- **Commit the vectors, and embed only the questions.** Fast, but the gate would no longer exercise chunking or document embedding, and the committed vectors would go stale without anyone noticing.
- **Keep the vectors between CI runs, keyed by model and text.** This would cut the job from about six minutes to under two. It was rejected for now because of one failure it invites: a change to the embedding code, or an upgraded library, would be scored against vectors made by the old code unless the cache key covers every file involved. A gate that is blind in exactly that case is worse than a slow one. It is the first thing to revisit if six minutes becomes a burden, with a check that re-embeds a sample of the cached vectors on every run.
- **Cut each page down to the chunks around the evidence.** Keeping eight chunks either side leaves 1,555 chunks, half the work. But the gate then cannot tell the retrievers apart: dense only scores 0.977 on Hit@5 and keyword only 0.985, against 0.992 for hybrid. The other chunks of the same page are what make the slice hard, so whole pages stay.
- **Add pages without evidence as distractors.** Measured with 0, 100 and 300 extra pages: Hit@5 was 0.992 every time, and MRR@10 was 0.867, 0.878 and 0.873. For any one question the other 133 pages already are distractors. Extra pages would add 37% to 99% more chunks to embed and no sensitivity, so the setting exists and is 0.
- **Compare with `main` instead of with fixed minimums.** That means running the gate twice. Because the gate gives the same result every time, a fixed floor does the same job, and it is simpler to explain.

## Consequences

- Every pull request waits for the gate. The first run took 5 minutes 47 seconds, of which 284 seconds were spent embedding. On a laptop with a GPU the same command takes just over 2 minutes.
- The scores were identical on the laptop and in CI, down to the one question that is missed. The minimums therefore need no allowance for the machine, only for changes that are meant to pass.
- A pull request that changes retrieval on purpose also changes `evals/gate.json`, with the full evaluation as its evidence. The minimums are a floor to be raised, not a target.
- The slice follows the golden set. A pull request that changes the questions changes the pages, and has to measure the gate again.
- With cold caches the job needs the network: the handbook's host for the snapshot and the model hub for the models. Once the model files are restored it runs offline, so an outage there cannot fail it.
- `make_retriever` can now be given a store, an embedder and a keyword encoder that already exist. `build_corpus` is split into loading the snapshot and chunking pages.
