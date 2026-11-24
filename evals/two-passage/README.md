# Two-passage questions

Questions whose answer needs two passages on two different pages of the handbook. [`questions.jsonl`](questions.jsonl) holds 34 of them, in the format of the [golden set](../golden/README.md), and [`why-two.jsonl`](why-two.jsonl) says for each one why it needs both passages. The task is [issue 78](https://github.com/MumukshuPanchal/handbook-rag/issues/78).

Almost every golden question rests on one passage. A search that brings one of two needed passages counts as a hit there, and nothing shows that half of the evidence is missing. These questions are for measuring that.

> **Status: written by one model session, not yet read by anyone else.** Every question, reference answer, quote and reason was written by a Claude model (`claude-fable-5-2`) in one session, with both passages in front of it. A second reading by another model session, which sees the questions and the passages and not the reasons, is planned and has not happened. No person has read the questions. Every question is therefore marked `"review": "pending"`. A model's check is not a person's review.

## What is in it

| Kind | What the two passages are | Questions |
|---|---|---|
| `comparison` | the same thing in two countries, teams or programmes, each on its own page | 12 |
| `term-and-rule` | one page says what a thing is or which class it falls in, and a rule on another page is written for that term or class | 8 |
| `rule-and-exception` | a rule on one page, and its exception, its condition or a local version of it on another | 7 |
| `process` | two steps of one process, on two pages | 7 |
| All | | 34 |

- **Every question has two pieces of evidence, on two pages.** Each piece is a page and an exact quote of 48 to 286 characters.
- **58 pages from 18 handbook sections** hold the 68 pieces. 8 pages are used by more than one question: four country pages, the page of company-wide benefits, and three security pages (see [known limits](#known-limits)).
- **In 16 questions the two pages are in two different sections**, for example a country page and the expenses policy. In the other 18 they are two pages of one section.
- **The questions are 14 to 39 words long**, 22 in the middle. 12 of them ask for two things in one sentence, joined by "and". They are two-part questions on purpose: the two parts belong to one subject, and each part is answered on its own page.
- **The questions are tied to the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8`**, as the golden set is.

Examples, one of each kind:

| Id | Kind | Question | First page | Second page |
|---|---|---|---|---|
| `t008` | comparison | Is the recruiter's screening call the same length for someone applying to Support Engineering as for someone applying to Infrastructure? | interviewing for Customer Support: 40 minutes | interviewing for Infrastructure: 30 minutes |
| `t014` | term-and-rule | We want to change how our Donut integration is set up. Do we have to follow the change management policy for that? | critical system tiers: Donut is Tier 4 Administrative | change management policy: Tier 4 is not in its scope |
| `t023` | rule-and-exception | I'm based in Israel and rent a desk in a hotel's business lounge every week. Can I get the transportation allowance for travelling there? | Israel: the space must meet the global co-working policy | expenses policy: co-working in a hotel is not accepted |
| `t031` | process | I'm bringing in a contractor who will work with Orange data. What has to be finished before they can be onboarded, and how much earlier than usual should the Zip request go in? | temporary service providers: what must be complete | tips for a Zip request: 10 days earlier |

## How it was made

One model session did all of it. It followed the golden set's [drafting guidelines](../golden/drafting-guidelines.md) where they apply to a question with two passages: a question a person would ask, in their words and not the page's; quotes copied exactly; nothing that needs knowledge from outside the handbook.

1. **Finding pairs of pages.** Three ways were used. Pages with the same structure were compared heading by heading: the pages of country and entity benefits, of two interview processes, of two mentorship programmes, of two shadow programmes. Sentences that point to another page were collected from the snapshot, by its links and by words such as "exception", "unless", "in line with the" and "per the", and the page pointed to was read. And pages that define classes were followed to the rules written for those classes: the critical system tiers, the data classification levels and the security incident severities.
2. **Writing.** For each pair kept: the question, a reference answer that uses both passages, a quote on each page, and the two reasons (see below).
3. **The test of "needs two".** For each question the session wrote down what a reader with only the first passage gets wrong or leaves out, and the same for the second. A candidate for which one side could not be written was set aside or reworded. By the maker's own count that was some 20 candidate pairs set aside and 7 questions reworded. The repository does not hold the candidates, so it cannot show this. The most common reason: the page with the exception also repeats the rule, so that page alone answers the question.
4. **Checks by code.** Each quote was copied from the cleaned page by a script, never typed. `golden-check` then verified that each quote is in its page word for word and on no other page, and that every question has two passages on two pages and a reason for each.
5. **A search for the same fact elsewhere.** For each fact, a pattern was searched for in all 3,846 pages, and within the fact's own page. This finds a fact that is repeated in similar words. It does not find a paraphrase.
6. **Reading the misses.** After the first run, the chunks the default search returned were read for every question with a passage missing from the first ten results. One question was dropped (its page states the rule twice, in two wordings) and two were reworded so that they ask for what only the quoted passage says.

### The reasons

`why-two.jsonl` has one line for each question:

```json
{"id": "t013", "kind": "term-and-rule", "only_first": "Figma's tier is known (Tier 3 Business Operational), but not how long an approval lasts for any tier.", "only_second": "The approval windows are listed by Critical System Tier, but nothing says which tier Figma is, so the right row cannot be picked."}
```

`only_first` is what a reader who has only the first passage gets wrong or leaves out, and `only_second` the same for the second passage. "First" and "second" are the order of the evidence in `questions.jsonl`. The reasons are the maker's claim, written to be checked: a second reader who is shown the question and one passage should find that same gap.

## What the search does with them

Measured on the full index, on the runs in [`evals/runs/two-passage`](../runs/two-passage). No language model is involved. "Both" is the score All@k: both passages are among the first k results. "At least one" is Hit@k, the score the golden set is reported with.

| Search | Run | Both in the top 5 (95% interval) | At least one in the top 5 | Both in the top 10 | At least one in the top 10 |
|---|---|---|---|---|---|
| Hybrid, the default | [`hybrid`](../runs/two-passage/hybrid) | 0.353 (0.206 to 0.529) | 0.794 | 0.500 | 0.912 |
| Embedding model | [`dense`](../runs/two-passage/dense) | 0.206 (0.088 to 0.353) | 0.794 | 0.500 | 0.853 |
| Keyword (BM25) | [`sparse`](../runs/two-passage/sparse) | 0.235 (0.088 to 0.382) | 0.706 | 0.294 | 0.824 |

The default search, as counts of the 34 questions:

| In the first | Both found | One found | None found | A chunk of both pages |
|---|---|---|---|---|
| 3 | 5 | 17 | 12 | 10 |
| 5 | 12 | 15 | 7 | 16 |
| 10 | 17 | 14 | 3 | 21 |

- **The default search brings both passages into its first five results for 12 of the 34 questions.** It brings at least one for 27. So 15 questions count as a hit today while the model would be given half of what the answer needs.
- **Looking further down helps a little.** The rank at which both passages were there:

| Rank at which both were found | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Not in the first 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Questions | 3 | 2 | 4 | 3 | 1 | 2 | 1 | 0 | 1 | 17 |

  For 5 questions the later of the two passages came at ranks 6 to 10. For 17 questions a passage is not among the first ten results at all, and for 3 of those neither is. The run asks for ten results, so it cannot say where the missing passages would have come.
- **The kinds differ.** In the first five results of the default search:

| Kind of question | Questions | Both found | One found | None found |
|---|---|---|---|---|
| `comparison` | 12 | 8 | 4 | 0 |
| `term-and-rule` | 8 | 0 | 5 | 3 |
| `rule-and-exception` | 7 | 1 | 5 | 1 |
| `process` | 7 | 3 | 1 | 3 |

  A comparison names both of its subjects, and the search finds both for 8 of 12. A `term-and-rule` question describes a case and does not use the term that the rule is written for, and the search finds both for none of the 8: in 7 of them the passage that defines the term or gives the class is not in the first ten results. With 7 to 12 questions in a kind, this is a pattern to look into, not a result.
- **"A chunk of both pages" is an upper bound.** A passage is credited by its quote. A page sometimes says the same thing in a second place, and a search that returns that place is counted as not finding it. The last column of the second table counts any chunk of the right page as finding the passage, whatever the chunk says. That is too generous: most other chunks of a page do not hold the fact. For the first five results the truth lies between 12 and 16 of 34.
- **Hybrid is ahead of keyword search, and cannot be told apart from the embedding model.** Hybrid has both passages in its first five for 12 questions, keyword search for 8 and the embedding model for 7. Compared question by question, as `eval-compare` does it, hybrid against keyword search is +0.118 (95% interval +0.029 to +0.235): 4 questions better and 0 worse. Hybrid against the embedding model is +0.147 (-0.029 to +0.324): 8 questions better and 3 worse, which chance could explain. The interval on a single share is about 16 points either side.

### What the runs cannot show

- **These questions are easier for a search than real ones.** They were written by a model that had both passages in front of it. The comparisons name both of their subjects, and the questions share words with their evidence: 25% of a question's meaningful words on average are also in the quote it shares most words with, as `golden-check` counts it. The golden set's card says the same of its questions. The shares above are therefore more likely too high than too low.
- **Nothing here is about answers.** Whether the model uses both passages when it is given both, and says so when it is given one, is a later step of the issue.
- **Nothing here is about a change to the search.** These runs are the starting point for the changes the issue lists.

## How the golden set's scores are affected

They are not. The evaluation gained three scores, All@3, All@5 and All@10, beside the ones it had, and every row of a run now also records the rank at which each passage first came. No existing score changed its meaning. For a question with one passage, All@k is the same as Hit@k.

The golden set was run again in hybrid mode with the new code, into a scratch directory. All 924 existing scores of its 132 questions were the same as in the committed run [`retrieval/v2`](../runs/retrieval/v2), and so were the seven means with their intervals. That run is not in the repository. It can be repeated with the command below and compared with `v2`.

## Format

`questions.jsonl` is a golden set file: `id`, `question`, `reference_answer`, `evidence` (two entries, each a `doc_id` and a `quote`), `type` (`answerable`), `origin` (`model-drafted`) and `review` (`pending`). Every command that reads a golden set reads it.

`why-two.jsonl` holds what the golden format has no place for: `id`, `kind`, `only_first` and `only_second`.

```bash
# Check every quote against the snapshot, and the file against the rules for two-passage questions
uv run handbook-rag golden-check --dataset evals/two-passage/questions.jsonl --reasons evals/two-passage/why-two.jsonl

# The three saved runs
uv run handbook-rag eval-retrieval --dataset evals/two-passage/questions.jsonl --output evals/runs/two-passage --variant hybrid --mode hybrid --no-rerank
uv run handbook-rag eval-retrieval --dataset evals/two-passage/questions.jsonl --output evals/runs/two-passage --variant dense --mode dense --no-rerank
uv run handbook-rag eval-retrieval --dataset evals/two-passage/questions.jsonl --output evals/runs/two-passage --variant sparse --mode sparse --no-rerank

# The golden set with the same code, to compare with the committed run v2
uv run handbook-rag eval-retrieval --variant current --mode hybrid --no-rerank
```

A run's `summary.json` holds the counts of the tables above under `needs_several_passages`, and each row of `results.jsonl` holds `evidence_ranks` (the rank of the first chunk that covers each quote) and `page_ranks` (the rank of the first chunk of each quote's page).

## Known limits

Read these before trusting a number.

- **Not checked by anyone.** One model session wrote the questions and decided that each needs two passages. The second reading has not happened.
- **Written by a model that could see both passages.** See [what the runs cannot show](#what-the-runs-cannot-show). There are no questions written without sight of the handbook.
- **Small.** 34 questions, and 7 to 12 of a kind.
- **Fewer than planned.** The issue's plan was about 40. Candidates that did not pass the test from both sides were dropped, and no weaker ones were added to make up the number.
- **Some questions could be answered well enough from one passage.** These are the ones the maker is least sure of:
  - `t017`: a reader of the communication plan alone may assume that a customer-facing service being down is Severity 1, and be right.
  - `t020`: a reader of the assessment rule alone may guess that personal data is Orange and not Red, because the question says that no customer data is involved.
  - `t018`, `t028`, `t029` and the other two-part questions: each part is answered by one passage. The whole answer needs both, and a reader who is shown one passage and not told that there is a second may still call it complete.
  - `t006`: the page of the CFO programme says that it mirrors the CEO programme. A reader may take that to mean the same length.
- **Some facts are stated in a second place.** The evaluation credits the quoted passage only, so these questions can be scored as a miss when the search found the fact:
  - on another page: weekly one-to-ones (`t005`, on several pages about managing), the length of the CSMA programme (`t007`, on the CSM onboarding page), the data level Glean is approved for (`t016`, in the table of AI tools for Support), the examples of a Severity 1 incident (`t017`, on the engineering incident page), what must be done before a contractor starts (`t031`, on an engineering page about hiring contractors);
  - on the same page, in another passage: the length of a CEO shadow rotation (`t006`), the two business days of a legal review (`t012`), the three workdays of job abandonment (`t018`).
  The column "a chunk of both pages" bounds the second group. Nothing bounds the first.
- **A few pages carry several questions.** The page of critical system tiers is used by 3 questions, and so is the page of Dutch benefits. 16 of the 68 pieces of evidence are on security pages and 15 on pages of total rewards.
- **Three passages are also evidence of a golden question**: the hire dates of the compensation review (`t024`, golden `q060`), the referral bonus (`t028`, golden `b16`) and the skip-level meetings (`t005`, golden `q103`, the sentence before). The two sets are not independent there.
- **The kinds are not equally hard, and not equally many.** A share over all 34 questions depends on the mix.
- **Ten results.** A run asks the search for ten results. Where a passage would have come further down is not recorded.
- **A snapshot in time, English only, and prose and short lists only**, as in the golden set.

## Licence and attribution

The evidence quotes are excerpts from the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence. The questions, reference answers and reasons were written for this project from those passages.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../docs/adr/0002-handbook-corpus.md).
