# Golden set

The test questions this project is measured with. Each question has a reference answer and the handbook passage that supports it, or is marked as one the handbook cannot answer. The format is described in the [evaluation guide](../../docs/evaluation.md).

> **Status: checked by independent model sessions, not verified by a person.** Every question was made by one model session and checked by another that had not seen how it was made. The project's author read through the set and asked for this maker-checker review in place of checking each question by hand. Every question is therefore marked `"review": "checked"`, and none is marked `"verified"`.

## What is in it

| | Count |
|---|---|
| Questions the handbook answers, written from a passage (`model-drafted`) | 122 |
| Questions the handbook answers, written without seeing the handbook (`model-blind`) | 10 |
| Questions the handbook cannot answer | 33 |
| of which near-miss (topic covered, detail missing) | 19 |
| of which out-of-scope (a handbook would never hold it) | 11 |
| of which false-premise (assumes something the company does not have) | 3 |
| Checked by an independent model session | 165 |
| Verified by a person | 0 |

The questions are tied to the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8`. `make golden-check` verifies every evidence quote against that snapshot.

## How it was made: makers and checkers

Nobody checked their own work. A maker produced each item, and a checker that had not seen how it was made looked for the maker's mistakes. Makers and checkers ran as separate sessions, and from round 5 on the checker was a different model from the maker.

| Round | Role | Model | What happened |
|---|---|---|---|
| 1 | Sampling | script | 180 passages drawn with a fixed seed, at most one per page, spread across handbook sections |
| 2 | Maker | claude-fable-5-2 | Wrote 136 questions from those passages and skipped 44. Wrote 50 questions the handbook should not be able to answer |
| 3 | Checker | claude-fable-5-2 | Searched the whole handbook for a second passage that answers the same question. 12 questions removed. 16 unanswerable questions set aside because the handbook partly answers them |
| 4 | Checker | claude-fable-5-2 | Read what the searches returned for every missed question. One more second source found, and that question removed |
| 5 | Checker | claude-opus-5-5 | Reviewed all 157 remaining items as a strict reviewer. 127 accepted, 30 sent back |
| 6 | Maker | claude-fable-5-2 | Repaired the 27 answerable questions that were sent back |
| 7 | Checker | claude-opus-5-5 | Checked the repairs. 26 accepted, 1 sent back |
| 8 | Maker | claude-sonnet-5-5 | Wrote 36 blind questions from a list of section names and page titles, with no handbook text |
| 9 | Maker | claude-sonnet-5-5 | Searched the handbook for each blind question. Clear evidence found for 19. The other 17 were ambiguous or only partly answered |
| 10 | Checker | claude-opus-5-5 | Checked those 19 labels. 4 accepted, 15 sent back, mostly because the quote was too short to support the answer |
| 11 | Maker | claude-opus-5-5 | Repaired 12 labels and 1 question. 3 blind questions dropped because their wording did not match what the handbook says |
| 12 | Checker | claude-fable-5-2 | Last check of those repairs. 10 accepted, 1 accepted with the checker's exact correction, 2 rejected |
| 13 | Checker | script | After every round: each quote must be in its page word for word, and on no other page. This rejected one repair |
| 14 | Checker | claude-opus-5-5 | Read what the searches returned for the 12 missed questions that were new or reworded. 7 real misses. 1 incomplete label, completed with three more passages. 4 blind questions removed as ambiguous |

What the strict review in round 5 sent back:

| Finding | Items |
|---|---|
| A yes-or-no question whose wording gives the answer away | 20 |
| The reference answer leaves out part of what the passage says | 5 |
| Unanswerable question with the wrong kind, or partly answered | 3 |
| The quote does not cover the whole answer | 2 |
| Reads like a quiz, not like an employee's question | 2 |
| Wording copied from the passage | 1 |

Some items had two findings. In most of the yes-or-no questions the answer was "no", so an assistant that always said "no" would have scored well on them. They are now open questions. The earlier check in round 3 had not looked for this.

Two rules shaped the blind questions:

- **A blind question is never reworded by anyone who has seen the handbook.** When its wording did not fit what the handbook says, it was dropped.
- **Not finding an answer never makes a question "unanswerable".** A blind question with no clear evidence was left out.

Of the 36 blind questions, 10 remain. 17 had no single clear answer in the handbook, 5 did not fit the handbook's wording or could not be labelled correctly, and 4 turned out to be ambiguous once the searches' results were read: the handbook gives a different valid answer elsewhere, for example a rule that applies to one team.

## How the two kinds of question behave

Measured on the full index, on the runs in [`evals/runs/retrieval`](../runs/retrieval).

| Search | From a passage (122): Hit@1 | Hit@5 | Blind (10): Hit@1 | Hit@5 |
|---|---|---|---|---|
| Embedding model | 0.574 | 0.844 | 0.70 | 1.00 |
| Keyword (BM25) | 0.754 | 0.926 | 0.40 | 1.00 |
| Hybrid | 0.705 | 0.959 | 0.60 | 1.00 |
| Hybrid with reranking | 0.762 | 0.934 | 0.90 | 1.00 |

- **The blind questions cannot separate the searches at five results.** Every search finds the evidence in its top five. That is partly selection: only questions whose evidence a search could find were kept, and ambiguous ones were removed.
- **At rank 1 the order is different.** Keyword search is the best single search on passage-written questions and the worst on blind ones, and reranking helps most on blind ones. With ten questions none of this is statistically clear: for keyword search against the embedding model at rank 1, the difference is -0.30 with a 95% interval from -0.70 to +0.20. It is a reason to grow the blind subset, not a result.
- **Word overlap does not explain it.** By a simple count, blind questions share 41% of their meaningful words with their evidence and passage-written ones 35%. Blind questions are short and name the thing they ask about. Passage-written ones were paraphrased on purpose.

One warning from building this. Before round 14, keyword search appeared to do much worse than the embedding model on blind questions at five results (0.64 against 0.86). Reading what the searches had returned showed that this came from one incomplete label and four ambiguous questions. A result that rests on three questions has to be checked this way before anyone believes it.

## Known limits

Read these before trusting a number.

- **Checked by models, not verified by a person.** The makers and checkers are different sessions and partly different models, but all are Claude models and can share blind spots.
- **Mostly written by a model that could see the passage.** 122 of the 132 answerable questions were written from their passage.
- **The blind subset is small and selected.** Ten questions, kept only when evidence was found and the answer was unambiguous. Absolute scores on it are optimistic.
- **One recorded passage for most questions.** The evaluation credits only recorded passages. If another passage also answers a question and a search returns it, the search is scored as wrong. The checks removed or completed the cases they found, and cannot have found them all, so retrieval scores are lower bounds. The misses of any new search variant need the same check before they are believed.
- **Small.** With 132 answerable questions, a 95% interval on Hit@5 is roughly 5 points either side. Smaller differences cannot be told apart; `make eval-compare` reports this.
- **Almost no question needs two passages.** One does.
- **A snapshot in time.** Budgets, dates and tool names in the handbook change. The questions are valid for the pinned commit only.
- **English only**, and prose only: facts that live in tables are under-represented.
- **Questions with a false assumption that the handbook corrects are not in the set.** The right behaviour there is to correct the assumption, not to decline, and that needs its own test (issue 37).

## Licence and attribution

The evidence quotes are excerpts from the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence. The questions and reference answers were written for this project from those passages.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../docs/adr/0002-handbook-corpus.md).
