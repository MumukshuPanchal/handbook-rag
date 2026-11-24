# Search variants on the two-passage questions

The default search brings both passages of a two-passage question into its first five results for 12 of the 33 confirmed questions ([the card](README.md) has that measurement). This file reports what the changes to the search that need no request to a model do with those 33 questions, and with the 132 questions of the [golden set](../golden/README.md) that the default search was chosen on. It is the third step of [issue 78](https://github.com/MumukshuPanchal/handbook-rag/issues/78).

No default was changed. Every variant is an option of `eval-retrieval` that is off unless a run asks for it, and the retrieval gate has the minimums it had. What was decided from these results is in [ADR 0021](../../docs/adr/0021-two-passage-questions-and-the-search-as-it-is.md): none of the variants is adopted.

> **Status: made by one model session, and not read by a person.** The 33 questions were written by a model that had both passages in front of it, fourteen variants were tried on those same 33, and nothing here was measured on answers. Read the tables as "is this change worth another test", not as "is this search better".

## What was tried

The default search takes 30 candidates from the embedding search and 30 from the keyword search, and merges them by reciprocal rank fusion with the constant 60. Each variant changes one thing about it.

- **Spreading the results over pages.** At most m chunks of one page among the first k results of the merged list. A chunk over the limit moves down to just below the first k, and none is dropped ([the rule](../../src/handbook_rag/retrieve/spread.py)). The idea was that chunks of one page fill the first five results and keep a second page out. Tried with m = 1 and 2, for k = 5, which is what the model is given, and for k = 10. The first five results are the same for both k, so the two runs with k = 10 say something new only about ranks 6 to 10.
- **Another number of candidates**: 10, 20, 50 and 100 from each search. [Issue 51](https://github.com/MumukshuPanchal/handbook-rag/issues/51) asks the same of the golden set.
- **Another fusion constant**: 20, 10 and 5. With 60 and 30 candidates, a chunk that both searches return always comes before a chunk that only one of them returns, even one it ranks first: 2/90 is more than 1/61. A lower constant lets the first results of one search through, which could matter when each search finds a different one of the two passages.
- **The reranker**, an option the project has and had measured only on questions with one passage: a cross-encoder reads the question with each of the 30 merged results and orders them again. It is a local model and needs no request. It takes time: 216 ms for the question in the middle of the golden set's run, against 30 ms.

These twelve were written down before the first run. Two were added after the first results had been seen, and are marked "added later": m = 3 and m = 4 for k = 5. With m = 1 and m = 2 the rule had pushed the passage itself out of the first five, and the idea only needs one place kept free for a second page. No two changes were combined. No rule that reads the wording of a question to split it was tried: many of these questions were written in two parts on purpose, and such a rule would be fitted to how they were written. Splitting a question needs a model, and belongs to the step that asks a model before the search ([issue 79](https://github.com/MumukshuPanchal/handbook-rag/issues/79)).

## The 33 two-passage questions

"Both in the top 5" is the score All@5: both passages are among the first five results. The change is the paired comparison that `eval-compare` makes with the run of the default search: the mean difference per question with its 95% interval, the questions that got better and worse, and "clear" when the interval excludes zero.

| Variant | Run | Both in the top 5 | Change against the default (95% interval) | Better / worse | Reading | At least one in the top 5 | Both in the top 10 |
|---|---|---|---|---|---|---|---|
| The default: 30 candidates from each search, constant 60 | [`hybrid`](../runs/two-passage/hybrid) | 12 of 33 (0.364) |  |  |  | 26 of 33 (0.788) | 17 of 33 (0.515) |
| At most 1 chunk of a page among the first 5 | [`spread-1-in-5`](../runs/two-passage/spread-1-in-5) | 8 of 33 (0.242) | -0.121 (-0.273 to +0.030) | 1 / 5 | unclear | 20 of 33 (0.606) | 17 of 33 (0.515) |
| At most 2 chunks of a page among the first 5 | [`spread-2-in-5`](../runs/two-passage/spread-2-in-5) | 13 of 33 (0.394) | +0.030 (+0.000 to +0.091) | 1 / 0 | unclear | 25 of 33 (0.758) | 17 of 33 (0.515) |
| At most 3 chunks of a page among the first 5 (added later) | [`spread-3-in-5`](../runs/two-passage/spread-3-in-5) | 12 of 33 (0.364) | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear | 26 of 33 (0.788) | 17 of 33 (0.515) |
| At most 4 chunks of a page among the first 5 (added later) | [`spread-4-in-5`](../runs/two-passage/spread-4-in-5) | 12 of 33 (0.364) | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear | 26 of 33 (0.788) | 17 of 33 (0.515) |
| At most 1 chunk of a page among the first 10 | [`spread-1-in-10`](../runs/two-passage/spread-1-in-10) | 8 of 33 (0.242) | -0.121 (-0.273 to +0.030) | 1 / 5 | unclear | 20 of 33 (0.606) | 9 of 33 (0.273) |
| At most 2 chunks of a page among the first 10 | [`spread-2-in-10`](../runs/two-passage/spread-2-in-10) | 13 of 33 (0.394) | +0.030 (+0.000 to +0.091) | 1 / 0 | unclear | 25 of 33 (0.758) | 16 of 33 (0.485) |
| 10 candidates from each search | [`candidates-10`](../runs/two-passage/candidates-10) | 10 of 33 (0.303) | -0.061 (-0.152 to +0.000) | 0 / 2 | unclear | 24 of 33 (0.727) | 16 of 33 (0.485) |
| 20 candidates from each search | [`candidates-20`](../runs/two-passage/candidates-20) | 13 of 33 (0.394) | +0.030 (+0.000 to +0.091) | 1 / 0 | unclear | 26 of 33 (0.788) | 16 of 33 (0.485) |
| 50 candidates from each search | [`candidates-50`](../runs/two-passage/candidates-50) | 12 of 33 (0.364) | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear | 29 of 33 (0.879) | 17 of 33 (0.515) |
| 100 candidates from each search | [`candidates-100`](../runs/two-passage/candidates-100) | 12 of 33 (0.364) | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear | 29 of 33 (0.879) | 18 of 33 (0.545) |
| Fusion constant 20 | [`rrf-k-20`](../runs/two-passage/rrf-k-20) | 12 of 33 (0.364) | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear | 26 of 33 (0.788) | 18 of 33 (0.545) |
| Fusion constant 10 | [`rrf-k-10`](../runs/two-passage/rrf-k-10) | 12 of 33 (0.364) | +0.000 (-0.091 to +0.091) | 1 / 1 | unclear | 25 of 33 (0.758) | 19 of 33 (0.576) |
| Fusion constant 5 | [`rrf-k-5`](../runs/two-passage/rrf-k-5) | 10 of 33 (0.303) | -0.061 (-0.182 to +0.061) | 1 / 3 | unclear | 24 of 33 (0.727) | 19 of 33 (0.576) |
| The 30 merged results reranked by the cross-encoder | [`rerank`](../runs/two-passage/rerank) | 12 of 33 (0.364) | +0.000 (-0.182 to +0.152) | 4 / 4 | unclear | 26 of 33 (0.788) | 16 of 33 (0.485) |

No variant finds both passages for clearly more questions than the default. The counts run from 8 to 13 of 33, and none of the fourteen differences is clear. The variants with 13 are one question ahead of the default: `t007` with two chunks of a page allowed, and `t021` with 20 candidates. The reranker changes the most: 4 questions better and 4 worse. It is also the only variant that finds both passages of a `term-and-rule` question in its first five, for 1 of the 8.

## The 132 questions of the golden set

Each of these questions rests on one passage, and Hit@5 says whether it is among the first five results.

| Variant | Run | Hit@5 | Change against the default (95% interval) | Better / worse | Reading | MRR@10 | Change of MRR@10 (95% interval) | Reading of MRR@10 | Recall@5 |
|---|---|---|---|---|---|---|---|---|---|
| The default: 30 candidates from each search, constant 60 | [`hybrid`](../runs/search-variants-golden/hybrid) | 127 of 132 (0.962) |  |  |  | 0.807 |  |  | 0.962 |
| At most 1 chunk of a page among the first 5 | [`spread-1-in-5`](../runs/search-variants-golden/spread-1-in-5) | 106 of 132 (0.803) | -0.159 (-0.227 to -0.098) | 1 / 22 | clear | 0.771 | -0.036 (-0.055 to -0.017) | clear | 0.803 |
| At most 2 chunks of a page among the first 5 | [`spread-2-in-5`](../runs/search-variants-golden/spread-2-in-5) | 118 of 132 (0.894) | -0.068 (-0.114 to -0.030) | 0 / 9 | clear | 0.799 | -0.007 (-0.014 to -0.002) | clear | 0.894 |
| At most 3 chunks of a page among the first 5 (added later) | [`spread-3-in-5`](../runs/search-variants-golden/spread-3-in-5) | 123 of 132 (0.932) | -0.030 (-0.061 to -0.008) | 0 / 4 | clear | 0.805 | -0.002 (-0.004 to +0.000) | unclear | 0.932 |
| At most 4 chunks of a page among the first 5 (added later) | [`spread-4-in-5`](../runs/search-variants-golden/spread-4-in-5) | 125 of 132 (0.947) | -0.015 (-0.038 to +0.000) | 0 / 2 | unclear | 0.806 | -0.001 (-0.002 to +0.000) | unclear | 0.947 |
| At most 1 chunk of a page among the first 10 | [`spread-1-in-10`](../runs/search-variants-golden/spread-1-in-10) | 106 of 132 (0.803) | -0.159 (-0.227 to -0.098) | 1 / 22 | clear | 0.745 | -0.062 (-0.091 to -0.035) | clear | 0.803 |
| At most 2 chunks of a page among the first 10 | [`spread-2-in-10`](../runs/search-variants-golden/spread-2-in-10) | 118 of 132 (0.894) | -0.068 (-0.114 to -0.030) | 0 / 9 | clear | 0.787 | -0.020 (-0.033 to -0.008) | clear | 0.894 |
| 10 candidates from each search | [`candidates-10`](../runs/search-variants-golden/candidates-10) | 125 of 132 (0.947) | -0.015 (-0.045 to +0.015) | 1 / 3 | unclear | 0.803 | -0.004 (-0.010 to +0.001) | unclear | 0.947 |
| 20 candidates from each search | [`candidates-20`](../runs/search-variants-golden/candidates-20) | 126 of 132 (0.955) | -0.008 (-0.023 to +0.000) | 0 / 1 | unclear | 0.808 | +0.001 (-0.002 to +0.004) | unclear | 0.955 |
| 50 candidates from each search | [`candidates-50`](../runs/search-variants-golden/candidates-50) | 127 of 132 (0.962) | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear | 0.804 | -0.002 (-0.006 to +0.000) | unclear | 0.962 |
| 100 candidates from each search | [`candidates-100`](../runs/search-variants-golden/candidates-100) | 125 of 132 (0.947) | -0.015 (-0.038 to +0.000) | 0 / 2 | unclear | 0.804 | -0.002 (-0.006 to +0.000) | unclear | 0.947 |
| Fusion constant 20 | [`rrf-k-20`](../runs/search-variants-golden/rrf-k-20) | 126 of 132 (0.955) | -0.008 (-0.023 to +0.000) | 0 / 1 | unclear | 0.807 | +0.001 (-0.001 to +0.002) | unclear | 0.955 |
| Fusion constant 10 | [`rrf-k-10`](../runs/search-variants-golden/rrf-k-10) | 127 of 132 (0.962) | +0.000 (+0.000 to +0.000) | 0 / 0 | unclear | 0.810 | +0.003 (+0.000 to +0.008) | clear | 0.962 |
| Fusion constant 5 | [`rrf-k-5`](../runs/search-variants-golden/rrf-k-5) | 126 of 132 (0.955) | -0.008 (-0.023 to +0.000) | 0 / 1 | unclear | 0.813 | +0.006 (-0.006 to +0.019) | unclear | 0.955 |
| The 30 merged results reranked by the cross-encoder | [`rerank`](../runs/search-variants-golden/rerank) | 124 of 132 (0.939) | -0.023 (-0.061 to +0.015) | 2 / 5 | unclear | 0.852 | +0.045 (-0.002 to +0.093) | unclear | 0.939 |

Every limit on the chunks of a page costs the golden set passages: Hit@5 is 0.803, 0.894, 0.932 and 0.947 with 1, 2, 3 and 4 chunks allowed, against 0.962. The first three differences are clear. No other variant is clearly different from the default on Hit@5, and none is ahead of it: they find the passage for 124 to 127 of the 132 questions, where the default finds it for 127.

## What a limit for each page moves

The rule cannot tell which chunk of a page holds the passage. In the default run of the golden set, the passage is among the first five results for 127 questions, and for 23 of those another chunk of the same page stands above it. With one chunk of a page allowed, 22 questions lose their passage, and all 22 are among those 23.

The table counts passages. The 33 questions have 66 of them, of which the default search has 38 in its first five results. The golden questions have 138, of which it has 129.

| Chunks of a page allowed among the first 5 | Of the 66: left the first five | Of the 66: came into them | Of the 138: left the first five | Of the 138: came into them |
|---|---|---|---|---|
| 1 | 12 | 2 | 23 | 2 |
| 2 | 2 | 2 | 9 | 0 |
| 3 | 1 | 1 | 4 | 0 |
| 4 | 0 | 0 | 2 | 0 |

With four allowed, which is the rule at its mildest, no passage of the 33 questions enters or leaves the first five. The golden questions that lose their passage are the same ones each time: the 2 questions that lose it with four allowed are among the 4 that lose it with three, those are among the 9, and those among the 22.

## Depth: what the default search holds further down

This is not a variant. It is read from the two saved runs of the default search, which hold the rank of every passage among the first ten results.

| In the first | Both passages, of the 33 | At least one, of the 33 | The passage, of the 132 golden questions |
|---|---|---|---|
| 5 | 12 of 33 (0.364) | 26 of 33 (0.788) | 127 of 132 (0.962) |
| 8 | 16 of 33 (0.485) | 29 of 33 (0.879) | 129 of 132 (0.977) |
| 10 | 17 of 33 (0.515) | 30 of 33 (0.909) | 130 of 132 (0.985) |

The first ten results hold both passages for 5 questions more than the first five: +0.152 (+0.030 to +0.303). The difference can only go that way, because more results never lose a passage. What giving the model eight or ten passages would do to its answers is not known. It would read more text for every question, which costs more and takes longer, and it could answer differently where it answers well today. That takes real requests to measure, and nothing here guesses at it.

## Fourteen variants on the same questions

"Clear" means that the 95% interval of one comparison excludes zero. When many comparisons are made, some come out clear by chance. Twelve different comparisons were made here on "both in the top 5": fourteen variants, of which the two with k = 10 repeat two others on the first five results. If none of twelve variants changed anything, the chance that at least one comparison came out clear would be up to 1 - 0.95^12 = 46%, not 5%.

On that score no comparison is clear, so there is no winner to discount. If one variant had looked good, the claim it could carry would have been "worth testing on new questions", and not "better".

The clear readings are in the table of the golden set, which holds 28 comparisons, and they are not all worth the same:

- **One chunk of a page**: 22 questions worse and 1 better. An exact sign test gives p = 0.000006. No allowance for fourteen variants changes that.
- **Two chunks**: 9 worse and none better, p = 0.004. A correction for the fourteen comparisons of Hit@5 asks for less than 0.0036, so this one stands at its edge.
- **Three chunks**: 4 worse and none better. The interval excludes zero, and the sign test gives p = 0.125: the rule that `eval-compare` applies is generous when few questions change ([issue 80](https://github.com/MumukshuPanchal/handbook-rag/issues/80)). By itself this reading would show nothing.
- **Fusion constant 10, on MRR@10**: +0.003, with 5 questions better and 1 worse, p = 0.219. Among 28 comparisons a reading like this is what chance produces. The committed run [`v9`](../runs/retrieval/v9) had the same figures.

What makes the loss from the page limit believable is not one of these tests. It is that the loss grows as the limit tightens (2, 4, 9 and 22 questions), that the same questions are lost each time, and that the section above says why.

## What this shows, and what it does not

1. None of the fourteen variants brings both passages into the first five results for clearly more of the 33 questions than the default search does: the best are one question ahead, 13 against 12, and no difference is clear.
2. Spreading the results over pages is the one change whose effect stands up, and the effect is a loss: on the golden set it pushes the passage itself out of the first five results (Hit@5 0.803 with one chunk of a page allowed and 0.894 with two, against 0.962), because the rule cannot tell which chunk of a page holds the passage.
3. Another number of candidates, a lower fusion constant and the reranker move a few questions in either direction on both sets, which is all that 33 and 132 questions can say of them, and the reranker takes about 7 times as long for a question.
4. What the default search does hold lies further down: both passages are among its first eight results for 16 of the 33 questions and among its first ten for 17, against 12 among the first five, but whether a model answers better from eight or ten passages, and at what cost, can only be measured with real requests.
5. None of this says what would happen with other questions: these 33 were written by a model that saw both passages, fourteen variants were tried on them, and for 16 of them a passage is not among the default's first ten results at all, where no reordering of those results can reach it.

## How the runs were made

```bash
# A variant on both sets: here, at most 2 chunks of a page among the first 5 results
uv run handbook-rag eval-retrieval --dataset evals/two-passage/questions.jsonl --reviewed-only --output evals/runs/two-passage --mode hybrid --no-rerank --variant spread-2-in-5 --spread-per-page 2 --spread-within 5
uv run handbook-rag eval-retrieval --output evals/runs/search-variants-golden --mode hybrid --no-rerank --variant spread-2-in-5 --spread-per-page 2 --spread-within 5

# The paired comparison with the default search, on each set
uv run handbook-rag eval-compare hybrid spread-2-in-5 --runs evals/runs/two-passage
uv run handbook-rag eval-compare hybrid spread-2-in-5 --runs evals/runs/search-variants-golden
```

The other runs differ in their last options: `spread-M-in-K` has `--spread-per-page M --spread-within K`, `candidates-N` has `--fusion-candidates N`, `rrf-k-K` has `--rrf-k K`, and `rerank` has `--rerank` in place of `--no-rerank`. The run `hybrid` has none of them. Each run's `summary.json` records what it was made with under `config`.

- **The golden set's runs are in a directory of their own**, [`evals/runs/search-variants-golden`](../runs/search-variants-golden), so that the committed runs of the golden set in `evals/runs/retrieval` stay as they are.
- **Six of the fifteen runs of the golden set measure a search that a committed run had measured before**: `hybrid` is [`v2`](../runs/retrieval/v2), `rerank` is [`v3`](../runs/retrieval/v3), `rrf-k-20` is [`v8`](../runs/retrieval/v8), `rrf-k-10` is [`v9`](../runs/retrieval/v9), `rrf-k-5` is [`v10`](../runs/retrieval/v10) and `candidates-100` is [`v11`](../runs/retrieval/v11). All 924 scores of each are the same in the new run. So the default search returns what it returned before the options were added, and the options for the constant, the candidates and the reranker do what the settings did.
- **The runs with 10 and 20 candidates give the figures that issue 51 quotes** from two runs that were made and never committed: Hit@5 0.947 and 0.955 on the golden set.
- **The run of the default search on the 33 questions is the one the card reports.** It was made again with this code into a scratch directory, and every row came out the same. That copy is not in the repository, and the card has the command that repeats it.
- **Ten results.** A run asks the search for ten results, as in the card, so where a passage would have come further down is not recorded.
