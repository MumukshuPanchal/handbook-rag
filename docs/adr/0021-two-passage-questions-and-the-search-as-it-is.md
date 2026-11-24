# ADR 0021: Two-passage questions are measured, and the search stays as it is

- Status: Accepted
- Date: 2026-11-24

## Context

**The test questions rest on one passage.** Of the 132 questions of the golden set that the handbook answers, 129 have one piece of evidence. Two others list passages on more than one page, any one of which answers the question, and one needs two passages of one page. None needs passages on two pages. Such questions are easy to ask: how a benefit differs between two countries, a rule on one page and its exception on another, two steps of one process. For such a question Hit@5 is already 1 when one of the two passages is found. The model then answers from half of what it needs, and no score says so ([issue 78](https://github.com/MumukshuPanchal/handbook-rag/issues/78)).

**What was built to see it.**

- **A set of questions that need two passages on two different pages** ([the card](../../evals/two-passage/README.md)). 34 questions of four kinds, each with a quote on both pages, a reference answer, and a reason for each passage: what a reader with only that passage gets wrong or leaves out. One model session wrote them, with both passages in front of it. The set is a file of its own, so the saved runs of the golden set stay comparable.
- **A second reading.** Each question was shown three ways, with both passages, with the first alone and with the second alone, each way to a different one of three reader sessions of the same model. The readers saw none of the maker's notes. They confirm 33 of the 34: complete with both passages, and not with either alone. The one they dispute, `t025`, stays in the file and is counted in nothing. A model's reading is not a person's review, and no person has read the questions.
- **All@k beside Hit@k.** All@k is 1 when the first k results hold every passage the answer needs. For a question with one passage it is Hit@k. No score the golden set had changed. In its runs made with the new code All@k equals Hit@k for every question: its one question that needs two passages, `b15`, has both in one chunk. A run also records the rank at which each passage first came.

**What the default search does with the 33.** No language model is involved. The runs are [`evals/runs/two-passage/hybrid`](../../evals/runs/two-passage/hybrid) and, for the golden set, [`evals/runs/search-variants-golden/hybrid`](../../evals/runs/search-variants-golden/hybrid), which repeats the committed run `v2` score for score.

| In the first | Both passages, of the 33 | At least one, of the 33 | The passage, of the 132 golden questions |
|---|---|---|---|
| 5 | 12 of 33 (0.364) | 26 of 33 (0.788) | 127 of 132 (0.962) |
| 8 | 16 of 33 (0.485) | 29 of 33 (0.879) | 129 of 132 (0.977) |
| 10 | 17 of 33 (0.515) | 30 of 33 (0.909) | 130 of 132 (0.985) |

- **Both passages are among the five results the model is given for 12 of the 33 questions**: 0.364, with a 95% interval of 0.212 to 0.515. At least one is there for 26. So 14 questions count as a hit while the model would be given half of the evidence.
- **The kinds differ.** Both passages are found for 8 of the 12 comparisons, which name both of their subjects, and for none of the 8 questions that describe a case without the term its rule is written for.
- **For 16 of the 33 questions a passage is not among the first ten results at all**, and for 3 of them neither is.

**Fourteen changes to the search were tried that need no request to a model** ([the results](../../evals/two-passage/search-variants.md)). Each changes one thing about the default, and each was run on the 33 questions and on the golden set. Twelve were written down before the first run. Two were added after the first results had been seen, and are marked. That order is the account of the session that made the runs, and the repository cannot show it.

| Variant | Both in the top 5, of the 33 | Hit@5 on the golden set, of 132 |
|---|---|---|
| The default: 30 candidates from each search, constant 60 | 12 | 127 |
| At most 1 chunk of a page among the first 5 | 8 | 106 |
| At most 2 chunks of a page among the first 5 | 13 | 118 |
| At most 3 chunks of a page among the first 5 (added later) | 12 | 123 |
| At most 4 chunks of a page among the first 5 (added later) | 12 | 125 |
| At most 1 chunk of a page among the first 10 | 8 | 106 |
| At most 2 chunks of a page among the first 10 | 13 | 118 |
| 10 candidates from each search | 10 | 125 |
| 20 candidates from each search | 13 | 126 |
| 50 candidates from each search | 12 | 127 |
| 100 candidates from each search | 12 | 125 |
| Fusion constant 20 | 12 | 126 |
| Fusion constant 10 | 12 | 127 |
| Fusion constant 5 | 10 | 126 |
| The 30 merged results reranked by the cross-encoder | 12 | 124 |

Compared with the default question by question, no variant differs clearly on "both in the top 5". On the golden set the limits of one, two and three chunks for each page differ clearly on Hit@5, and each is a loss; the limits of one and two chunks are losses on MRR@10 too. The only clear reading in a variant's favour is of the fusion constant 10, on MRR@10 on the golden set, and it is discussed below.

## Decision

- **The default search stays as it is**: hybrid, 30 candidates from each search, the constant 60, no reranking, and five passages for the model. None of the fourteen variants is adopted.
- **The questions, the second reading and All@k are kept**, as a second set beside the golden set. `eval-retrieval --reviewed-only` scores it without the disputed question.
- **The variants stay as options of `eval-retrieval`, for one run each**, so that the same measurement can be made on other questions. The limit for each page is a rule of its own (`handbook_rag.retrieve.spread`): no setting turns it on, and no other command has it. The number of candidates and the fusion constant could already be set for every command, by `HANDBOOK_RAG_FUSION_CANDIDATES` and `HANDBOOK_RAG_RRF_K`, and reranking by `HANDBOOK_RAG_RERANK`. Their defaults are as they were.
- **The next step asks a model before the search**, to split a question that asks for two things, and belongs with the request that [issue 79](https://github.com/MumukshuPanchal/handbook-rag/issues/79) plans for follow-up questions.

### Why none of the fourteen

**Nothing found both passages clearly more often.** The counts run from 8 to 13 of 33, where the default has 12. The best variants are one question ahead, and no interval excludes zero.

**The page limit is the one firm result, and it is a loss.** The idea was that chunks of one page fill the first five results and keep a second page out. The rule that follows from it cannot tell which chunk of a page holds the passage. In the default run of the golden set the passage is among the first five results for 127 questions, and for 23 of those another chunk of the same page stands above it. With one chunk of a page allowed, 22 questions lose their passage, all 22 among those 23, and 1 other question gains it. Hit@5 on the golden set is 0.803, 0.894, 0.932 and 0.947 with 1, 2, 3 and 4 chunks allowed, against 0.962. The loss grows as the limit tightens (2, 4, 9 and 22 questions), and the same questions are lost each time. On the 33 questions the same limits find both passages for 8, 13, 12 and 12, where the default finds them for 12.

**The other changes move a few questions in either direction.** With 50 or 100 candidates at least one passage is found for 29 of the 33 questions where the default finds one for 26, and both for the same 12. With 10 and 20 candidates both are found for 10 and 13. The fusion constants 20, 10 and 5 find both for 12, 12 and 10. The reranker finds both for 12, with 4 questions better and 4 worse. It took 216 ms for the question in the middle of the golden set's run, against 30 ms. The committed runs of the same two searches measured 247 ms and 33 ms.

**How far a reading of "clear" goes here.** Twelve different comparisons were made on "both in the top 5", so the chance that one of them reads clear when nothing differs is up to 46%, and not 5%. None does. The clear readings are on the golden set, and they are not all worth the same. On Hit@5 an exact sign test gives p = 0.000006 for the limit of one chunk and p = 0.004 for the limit of two. For the limit of three it gives p = 0.125, on 4 questions worse and none better: the rule that `eval-compare` applies is generous when few questions change ([issue 80](https://github.com/MumukshuPanchal/handbook-rag/issues/80)). The fusion constant 10 reads as clear on MRR@10, +0.003 with 5 questions better and 1 worse, and the sign test gives p = 0.219: among the 28 comparisons on the golden set, that is what chance produces. What makes the loss from the page limit believable is not one test. It is that it grows with the limit, loses the same questions, and has a reason.

### Why the next step is a model step

For 16 of the 33 questions a passage is not among the first ten results, and for 3 of them neither is. No reordering of the first ten can bring it in, and none of the fourteen variants brings both passages into the first ten for more than 2 of those 16. A question that asks for two things is put to the search as one question. Searching for each half by itself needs the question to be split, and a rule that splits at "and" or "compared with" would be fitted to how these questions were written. So it takes a model, and one more request before the search. [Issue 79](https://github.com/MumukshuPanchal/handbook-rag/issues/79) plans such a request, to rewrite a follow-up question before the search, as the rewriting step of [issue 22](https://github.com/MumukshuPanchal/handbook-rag/issues/22). Splitting a question is a step of the same kind.

## Alternatives considered

- **Adopt the limit of two chunks for each page.** It is one question ahead on the 33, 13 against 12, which is nothing with this many questions. It loses 9 of the golden questions and gains none.
- **Adopt the mildest limit, four chunks.** That is the idea as it was stated: keep one place free for a second page. No passage of the 33 questions enters or leaves the first five with it, and 2 golden questions lose theirs.
- **More candidates.** They raise "at least one", which is not what was missing. With 100 the golden set loses 2 questions.
- **Adopt the fusion constant 10.** Its one clear reading is the gain of 0.003 on MRR@10 above. It changes no golden question's Hit@5, and on the 33 it finds both passages for 1 question more and 1 less. The [experiment log](../experiments/2026-11-22-chunk-size-and-fusion.md) measured the same in run `v9` and kept 60.
- **The reranker.** It is the only variant that finds both passages of a question that describes a case without its term, for 1 of the 8. It does not find both more often overall, and it takes about 7 times as long.
- **Give the model eight or ten passages now.** The first eight results hold both passages for 16 questions and the first ten for 17. Nothing says what that does to the answers. The model would read more text for every question, which costs more and takes longer, and it could answer differently where it answers well today. Finding out means the answer evaluation again at each number of passages. For the golden set that is 165 answers and, if as many of them are judged as in the first run, 256 verdicts of the two judges: about 420 requests each time, on a key that allows 500 a day, and more when attempts have to be repeated ([the guide](../evaluation.md#results)). The 33 questions come on top, at five passages too, because none of them has been answered. It was not done.
- **A rule that splits a question by its wording.** Not tried, for the reason above.
- **Count any chunk of the right page as finding the passage.** That is too generous: most chunks of a page do not hold the fact. The card gives it as an upper bound beside the count by quote, 15 against 12 for the first five results.
- **Put the two-passage questions into the golden set.** Every saved run of the golden set would then be of other questions, and the gate's minimums with them.

## What is still missing

- **Answers on these questions.** No answer was written for them. Whether the model uses both passages when it is given both, and says that something is missing when it is given one, is not measured. Neither judge was checked on an answer that needs two passages.
- **A person's reading.** One model session wrote the questions and three sessions of the same model read them again.
- **Questions written without sight of the passages.** These were written by a model that had both in front of it, and they share words with their evidence. They are likely easier for a search than real questions are, so the shares here are more likely too high than too low.
- **New questions for anything that looks good later.** Fourteen variants were tried on the same 33 questions. A variant chosen on them has to be measured on questions that did not choose it.
- **More passages for the model, with answers measured.** See above.
- **Where a missing passage would have come.** A run asks the search for ten results.
- **A sounder rule for "clear".** [Issue 80](https://github.com/MumukshuPanchal/handbook-rag/issues/80).
- **Questions that need three passages, or two of one page.**

## Consequences

- There is a second number beside Hit@5: for a question that needs two passages, the search gives the model both for 12 of 33. It is on questions that are likely easier than real ones.
- The golden set's scores are as they were. Six runs of it made with the new code repeat committed runs score for score, all 924 scores of each.
- The code holds a rule that nothing uses by default (`handbook_rag.retrieve.spread`). The answer cache pins the code that decides what the search finds, and its test was brought up to date without raising the cache's version: with no option given the search returns what it returned before, and a test holds that to rankings written down from the code before the change.
- `eval-retrieval` refuses a variant it could not measure: one of the two numbers of a page limit without the other, a variant of hybrid search in another mode, a page limit with reranking, and fewer candidates than the results the search is asked for.
- Issue 78 stays open. What remains of it is the answers on these questions, and the model step.
