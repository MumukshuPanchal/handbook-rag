# Golden set

The test questions this project is measured with. Each question has a reference answer and the handbook passage that supports it, or is marked as one the handbook cannot answer. The format is described in the [evaluation guide](../../docs/evaluation.md).

> **Status: draft.** Every question was drafted by a model and none has been reviewed by a person yet (`"review": "pending"`). Numbers measured with this set are provisional until the review is done.

## What is in it

| | Count |
|---|---|
| Questions the handbook answers | 123 |
| Questions the handbook cannot answer | 34 |
| of which near-miss (topic covered, detail missing) | 19 |
| of which out-of-scope (a handbook would never hold it) | 12 |
| of which false-premise (assumes something the company does not have) | 3 |
| Handbook sections covered by answerable questions | 30 of 34 |
| Written by a person | 0 |
| Reviewed by a person | 0 |

The questions are tied to the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8`. `make golden-check` verifies every evidence quote against that snapshot.

## How it was built

1. **Sampling.** `handbook-rag golden-candidates --count 180 --seed 7` drew 180 passages, at most one per page. Each handbook section's share grows with the square root of its size, so large sections get more questions without crowding out small ones. Passages that are mostly tables or code were not sampled.
2. **Drafting.** Claude (`claude-fable-5-2`) wrote one question per passage, or skipped it, following the [drafting guidelines](drafting-guidelines.md). It wrote 136 questions and skipped 44 passages: lists of links, templates, diagram fragments, and passages whose content is repeated elsewhere in the handbook.
3. **Automatic check.** Every evidence quote must appear word for word in its page, and on no other page.
4. **Independent audit of the answerable questions.** A second pass by the same model, in a fresh session that saw only the question, the answer, the quote and the passage, checked each question against the whole corpus. It looked for answers the passage does not support, questions that only make sense next to their page, and above all for a second passage that answers the same question. It removed 12 of the 136: 11 because a second passage answers the same question, and one whose evidence sits on a leftover backup page. It found no reference answer that the passage does not support, and no quote that misses its answer. 12 of the remaining questions carry a `note` with a softer concern for the reviewer, mostly that general knowledge could answer them.
5. **Unanswerable questions.** 50 were drafted, each searched for in the handbook by keyword before it was kept. Each was then put to the project's strongest retriever (hybrid search with reranking), and an auditor read the six passages it returned and searched further. No question turned out to be answered by the handbook. 16 were set aside because the handbook partly answers them or corrects their premise, which leaves 34.
6. **Check of the misses.** Four retrieval variants were scored on the set. For each of the 25 questions that at least one variant missed, a reviewer (the same model, in a fresh session) read the passages returned instead. One question was removed because a passage from another page answers it. The other 24 misses are real.
7. **Review by a person.** Not done yet. `make golden-review` writes a page that shows each question beside its passage with the evidence marked.

## Known limits

Read these before trusting a number.

- **Written by a model that could see the passage.** Such questions tend to reuse the words of their source: on average a question shares 36% of its meaningful words with its evidence. This favours keyword search over search by meaning. Questions typed by people who have not seen the page are needed to measure that gap, and there are none yet.
- **One model family drafted, audited and will answer.** Shared blind spots are possible. Review by a person is the remedy.
- **One recorded passage per question.** The evaluation credits only that passage. If another passage also answers the question and a retriever returns it, the retriever is scored as wrong. The audit and the check of the misses removed the cases they found. The audit searched by keyword, and the check looked only at what four retrieval variants returned, so neither can have found them all. Retrieval scores are therefore lower bounds, and the misses of a new variant need the same check before they are believed.
- **Small.** With about 123 questions, a 95% confidence interval on Hit@5 is roughly 5 points either side. Smaller differences between two systems cannot be told apart; `make eval-compare` reports this.
- **No question needs two pages.** Every answer sits in one passage, so nothing here tests combining information.
- **Yes-or-no questions lean one way.** About one answerable question in six can be answered yes or no, and the reference answer is usually "no". For answer evaluation this makes guessing pay.
- **Few false-premise questions.** Most drafted ones were set aside: the handbook names what the company really uses, so the right behaviour is to correct the premise, not to decline. That behaviour deserves its own test (#37).
- **A snapshot in time.** Budgets, dates and tool names in the handbook change. The questions are valid for the pinned commit only.
- **English only**, and prose only: facts that live in tables are under-represented.

## Licence and attribution

The evidence quotes are excerpts from the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence. The questions and reference answers were written for this project from those passages.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../docs/adr/0002-handbook-corpus.md).
