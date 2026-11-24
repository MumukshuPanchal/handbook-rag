# Two-passage questions

Questions whose answer needs two passages on two different pages of the handbook. [`questions.jsonl`](questions.jsonl) holds 34 of them, in the format of the [golden set](../golden/README.md). [`why-two.jsonl`](why-two.jsonl) says for each one why it needs both passages, and [`second-reading.jsonl`](second-reading.jsonl) holds what three readers said who were shown the questions without those reasons. The readers confirm 33 of the 34, and every figure in this card is of those 33. The task is [issue 78](https://github.com/MumukshuPanchal/handbook-rag/issues/78).

Almost every golden question rests on one passage. A search that brings one of two needed passages counts as a hit there, and nothing shows that half of the evidence is missing. These questions are for measuring that.

> **Status: written by one model session, and read again by three reader sessions of the same model that did not see its notes. Not read by a person.** One session of `claude-fable-5-2` wrote every question, reference answer, quote and reason, with both passages in front of it. Three reader sessions of the same model then read every question, each with both passages or with one of them, and with none of the maker's notes. They confirm that 33 of the 34 questions need both passages. Those 33 are marked `"review": "checked"`. The one they do not confirm, `t025`, stays `"review": "pending"`, carries a note, and is left out of every figure. All four sessions are models, and a model's check is not a person's review.

## What is in it

| Kind | What the two passages are | Questions | Confirmed |
|---|---|---|---|
| `comparison` | the same thing in two countries, teams or programmes, each on its own page | 12 | 12 |
| `term-and-rule` | one page says what a thing is or which class it falls in, and a rule on another page is written for that term or class | 8 | 8 |
| `rule-and-exception` | a rule on one page, and its exception, its condition or a local version of it on another | 7 | 6 |
| `process` | two steps of one process, on two pages | 7 | 7 |
| All | | 34 | 33 |

"Confirmed" counts the questions that [the second reading](#the-second-reading) found to need both passages. The counts that follow are of those 33.

- **Every question has two pieces of evidence, on two pages.** Each piece is a page and an exact quote of 48 to 286 characters.
- **56 pages from 18 handbook sections** hold the 66 pieces. 8 pages are used by more than one question: four country pages, the page of company-wide benefits, and three security pages (see [known limits](#known-limits)).
- **In 16 questions the two pages are in two different sections**, for example a country page and the expenses policy. In the other 17 they are two pages of one section.
- **The questions are 14 to 38 words long**, 22 in the middle. 11 of them ask for two things in one sentence, joined by "and", and 6 comparisons join their two sides by "compared with". They are two-part questions on purpose: the two parts belong to one subject, and each part is answered on its own page. The readers remarked on it (see [known limits](#what-the-readers-said-about-the-questions)).
- **The questions are tied to the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8`**, as the golden set is.
- **The disputed question, `t025`**, is a `rule-and-exception` question about what Support may treat as an emergency. It is in the file with `"review": "pending"` and a `note`, and no figure counts it.

Examples, one of each kind:

| Id | Kind | Question | First page | Second page |
|---|---|---|---|---|
| `t008` | comparison | Is the recruiter's screening call the same length for someone applying to Support Engineering as for someone applying to Infrastructure? | interviewing for Customer Support: 40 minutes | interviewing for Infrastructure: 30 minutes |
| `t014` | term-and-rule | We want to change how our Donut integration is set up. Do we have to follow the change management policy for that? | critical system tiers: Donut is Tier 4 Administrative | change management policy: Tier 4 is not in its scope |
| `t023` | rule-and-exception | I'm based in Israel and rent a desk in a hotel's business lounge every week. Can I get the transportation allowance for travelling there? | Israel: the space must meet the global co-working policy | expenses policy: co-working in a hotel is not accepted |
| `t031` | process | I'm bringing in a contractor who will work with Orange data. What has to be finished before they can be onboarded, and how much earlier than usual should the Zip request go in? | temporary service providers: what must be complete | tips for a Zip request: 10 days earlier |

## How it was made

One model session wrote all 34 questions. It followed the golden set's [drafting guidelines](../golden/drafting-guidelines.md) where they apply to a question with two passages: a question a person would ask, in their words and not the page's; quotes copied exactly; nothing that needs knowledge from outside the handbook.

1. **Finding pairs of pages.** Three ways were used. Pages with the same structure were compared heading by heading: the pages of country and entity benefits, of two interview processes, of two mentorship programmes, of two shadow programmes. Sentences that point to another page were collected from the snapshot, by its links and by words such as "exception", "unless", "in line with the" and "per the", and the page pointed to was read. And pages that define classes were followed to the rules written for those classes: the critical system tiers, the data classification levels and the security incident severities.
2. **Writing.** For each pair kept: the question, a reference answer that uses both passages, a quote on each page, and the two reasons (see below).
3. **The test of "needs two".** For each question the session wrote down what a reader with only the first passage gets wrong or leaves out, and the same for the second. A candidate for which one side could not be written was set aside or reworded. By the maker's own count that was some 20 candidate pairs set aside and 7 questions reworded. The repository does not hold the candidates, so it cannot show this. The most common reason: the page with the exception also repeats the rule, so that page alone answers the question.
4. **Checks by code.** Each quote was copied from the cleaned page by a script, never typed. `golden-check` then verified that each quote is in its page word for word and on no other page, and that every question has two passages on two pages and a reason for each.
5. **A search for the same fact elsewhere.** For each fact, a pattern was searched for in all 3,846 pages, and within the fact's own page. This finds a fact that is repeated in similar words. It does not find a paraphrase.
6. **Reading the misses.** After the first run, the chunks the default search returned were read for every question with a passage missing from the first ten results. One question was dropped (its page states the rule twice, in two wordings) and two were reworded so that they ask for what only the quoted passage says.
7. **A second reading**, by three sessions that had seen none of this. It is the next section.

### The reasons

`why-two.jsonl` has one line for each question:

```json
{"id": "t013", "kind": "term-and-rule", "only_first": "Figma's tier is known (Tier 3 Business Operational), but not how long an approval lasts for any tier.", "only_second": "The approval windows are listed by Critical System Tier, but nothing says which tier Figma is, so the right row cannot be picked.", "second_reading": {"both": "complete", "first": "partial", "second": "partial"}, "confirmed": true}
```

`only_first` is what a reader who has only the first passage gets wrong or leaves out, and `only_second` the same for the second passage. "First" and "second" are the order of the evidence in `questions.jsonl`. The reasons are the maker's claim, written before anyone else had read a question. `second_reading` and `confirmed` were added afterwards: what the second readers said of the question shown with both passages, with the first only and with the second only, and whether that confirms the claim.

## The second reading

This section is about the questions: does each one need both of its passages? It says nothing about the search, which has a section of its own further down.

**What was done.** The 34 questions were dealt over three forms, so that each form holds every question once: with both passages on one form, with the first passage only on another, and with the second only on the third. Each form has 11 or 12 questions of each of the three showings. A passage was shown as the search would return it: the chunk that holds the quote, with its page title and the headings above it. An item had a neutral name and nothing else: no reason, no reference answer, no kind, no word of which items belong together.

Three reader sessions of `claude-fable-5-2`, the model that wrote the questions, each got one form and nothing but its form. A reader was asked whether the question can be answered in full from the passages shown and from nothing else: `complete`, `partial` with what is missing, or `none`. It also wrote the answer it could give, and said how sure it was. So no reader met a question twice, and none could tell from a second showing what the first had held back.

**Controls.** If every item with one passage were half of a two-passage question, the number of passages would give the label away. So each form also has 4 questions of the golden set that rest on one passage: 8 over the three forms are shown with that passage alone, and 4 with that passage and a neighbouring passage of its page.

| What the reader was shown | Showings | `complete` | `partial` | `none` |
|---|---|---|---|---|
| Both passages | 34 | 34 | 0 | 0 |
| The first passage only | 34 | 1 | 33 | 0 |
| The second passage only | 34 | 0 | 34 | 0 |
| Control: a golden question's one passage | 8 | 8 | 0 | 0 |
| Control: that passage and its neighbour | 4 | 4 | 0 | 0 |

| Reader | Items | `complete` | `partial` | Sure: `high` | `medium` | `low` |
|---|---|---|---|---|---|---|
| A | 38 | 16 | 22 | 32 | 5 | 1 |
| B | 38 | 16 | 22 | 32 | 6 | 0 |
| C | 38 | 15 | 23 | 34 | 4 | 0 |

What it showed:

- **33 of the 34 questions are confirmed.** A question is confirmed when it was `complete` with both passages and not `complete` with either one alone.
- **Both passages were enough for all 34 questions.** The maker then compared the answer each reader wrote from both passages with the reference answer: all 34 state the same facts. No code checks that, and it is the maker reading its own reference answers.
- **One passage was not enough in 67 of the 68 showings**, and in each of the 67 the reader named what was missing. For the 33 confirmed questions that is all 66 showings.
- **All 12 controls came out `complete`**, each with high confidence. So no reader called an item partial because it showed one passage, or doubted an item because it showed two.
- **The readers were sure of most labels.** 98 of the 114 labels came with high confidence, 15 with medium and 1 with low. Of the 16 below high, 10 are showings of a first passage alone, 5 of a second passage alone and 1 of both passages (`t001`). What they turn on is under [known limits](#what-the-readers-said-about-the-questions).
- **The question they dispute is `t025`**: "A customer's GitLab instance is running, but they say the problem will land them in immediate trouble with an audit. Does that meet the strict definition of an emergency, and if not, can Support still treat it as one?" The maker's claim was that the first passage gives the strict definition and says that exceptions exist, and that only the second passage names an immediate audit problem as a ground for one. The reader who saw only the first passage called the question `complete`, with medium confidence. Its answer: Support can still treat the case as an emergency, "because triage also asks whether the request fits the criteria for exceptions and whether the problem is causing a significant impact to the customer's business (a deliberately flexible test), and finally what the right thing to do for the customer is". The reader who saw only the second passage called it `partial`, also with medium confidence: the strict definition was missing. No third reader was asked. The question keeps its place in the file, is marked as disputed, and is counted in nothing.

**What the labels rest on.** Each showing was read by one reader, and no showing was read twice, so a label is one session's judgement. A reader had the passage and not its page, and not the rest of the handbook. So "needs both" means both of these two passages. It does not mean that no other passage could stand in for one of them (see [known limits](#known-limits)). The readers are sessions of the model that wrote the questions, and can share its blind spots.

**What the repository holds of it.** [`second-reading.jsonl`](second-reading.jsonl) has every line the readers wrote, 114 of them, each with the question it belongs to and what its item showed. The forms themselves are not in the repository, and neither is what a reader was told: that is the account of the session that ran the reading. A further reading can be set up from the files: show a reader the question and the chunk that holds each quote, and nothing of `why-two.jsonl`.

## What the search does with them

This section is about the search, measured on the 33 confirmed questions. The readers had no part in it, and a question's place in it says nothing about whether the question is a good one.

Measured on the full index, on the runs in [`evals/runs/two-passage`](../runs/two-passage). No language model is involved. "Both" is the score All@k: both passages are among the first k results. "At least one" is Hit@k, the score the golden set is reported with.

| Search | Run | Both in the top 5 (95% interval) | At least one in the top 5 | Both in the top 10 | At least one in the top 10 |
|---|---|---|---|---|---|
| Hybrid, the default | [`hybrid`](../runs/two-passage/hybrid) | 0.364 (0.212 to 0.515) | 0.788 | 0.515 | 0.909 |
| Embedding model | [`dense`](../runs/two-passage/dense) | 0.182 (0.061 to 0.333) | 0.788 | 0.485 | 0.848 |
| Keyword (BM25) | [`sparse`](../runs/two-passage/sparse) | 0.242 (0.091 to 0.394) | 0.697 | 0.303 | 0.818 |

The default search, as counts of the 33 questions:

| In the first | Both found | One found | None found | A chunk of both pages |
|---|---|---|---|---|
| 3 | 5 | 16 | 12 | 9 |
| 5 | 12 | 14 | 7 | 15 |
| 10 | 17 | 13 | 3 | 20 |

- **The default search brings both passages into its first five results for 12 of the 33 questions.** It brings at least one for 26. So 14 questions count as a hit today while the model would be given half of what the answer needs.
- **Looking further down helps a little.** The rank at which both passages were there:

| Rank at which both were found | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Not in the first 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Questions | 3 | 2 | 4 | 3 | 1 | 2 | 1 | 0 | 1 | 16 |

  For 5 questions the later of the two passages came at ranks 6 to 10. For 16 questions a passage is not among the first ten results at all, and for 3 of those neither is. The run asks for ten results, so it cannot say where the missing passages would have come.
- **The kinds differ.** In the first five results of the default search:

| Kind of question | Questions | Both found | One found | None found |
|---|---|---|---|---|
| `comparison` | 12 | 8 | 4 | 0 |
| `term-and-rule` | 8 | 0 | 5 | 3 |
| `rule-and-exception` | 6 | 1 | 4 | 1 |
| `process` | 7 | 3 | 1 | 3 |

  A comparison names both of its subjects, and the search finds both for 8 of 12. A `term-and-rule` question describes a case and does not use the term that the rule is written for, and the search finds both for none of the 8: in 7 of them the passage that defines the term or gives the class is not in the first ten results. Neither of the other two searches finds both passages of a `term-and-rule` question in its first five either. With 6 to 12 questions in a kind, this is a pattern to look into, not a result.
- **"A chunk of both pages" is an upper bound.** A passage is credited by its quote. A page sometimes says the same thing in a second place, and a search that returns that place is counted as not finding it. The last column of the second table counts any chunk of the right page as finding the passage, whatever the chunk says. That is too generous: most other chunks of a page do not hold the fact. For the first five results the truth lies between 12 and 15 of 33.
- **Hybrid finds both passages for more questions than either other search, and neither lead is firm.** Hybrid has both passages in its first five for 12 questions, keyword search for 8 and the embedding model for 6. Compared question by question, as `eval-compare` does it, hybrid against keyword search is +0.121 (95% interval +0.030 to +0.242): 4 questions better and 0 worse. By the rule `eval-compare` applies, that difference reads as clear. A count of 4 against 0 cannot carry the word: an exact sign test on it gives p = 0.125, and the rule is generous when few questions change ([issue 80](https://github.com/MumukshuPanchal/handbook-rag/issues/80)). So hybrid found both passages for more questions than keyword search did, and these runs do not settle that it does so in general. Hybrid against the embedding model is +0.182 (+0.000 to +0.364): 8 questions better and 2 worse. That interval reaches zero, so by the same rule the difference is not clear. The interval on a single share is about 15 points either side.
- **With the disputed question counted, the figures were these.** Before the second reading the three runs scored all 34 questions: 12, 7 and 8 of the 34 had both passages in the first five (hybrid, embedding model, keyword), and 27, 27 and 24 had at least one. Those runs are in the repository's history, at commit `f880d4b` ("feat: questions that need two passages, and whether the search finds both").

### What the runs cannot show

- **These questions are easier for a search than real ones.** They were written by a model that had both passages in front of it. The comparisons name both of their subjects, and the questions share words with their evidence: 25% of a question's meaningful words on average are also in the quote it shares most words with, as `golden-check` counts it. The golden set's card says the same of its questions. The shares above are therefore more likely too high than too low.
- **Nothing here is about answers.** Whether the model uses both passages when it is given both, and says so when it is given one, is a later step of the issue.
- **Nothing here is about a change to the search.** These runs are the starting point for the changes the issue lists. Fourteen changes that need no request to a model were measured afterwards, and none was adopted: [`search-variants.md`](search-variants.md) has the results, and [ADR 0021](../../docs/adr/0021-two-passage-questions-and-the-search-as-it-is.md) the decision.

## How the golden set's scores are affected

They are not. The evaluation gained three scores, All@3, All@5 and All@10, beside the ones it had, and every row of a run now also records the rank at which each passage first came. No existing score changed its meaning. For a question with one passage, All@k is the same as Hit@k. The option `--reviewed-only`, which leaves the disputed question out of the runs above, does nothing unless it is asked for, and the golden set has no question whose review is pending.

The golden set was run again in hybrid mode with this code, into a scratch directory. All 924 existing scores of its 132 questions were the same as in the committed run [`retrieval/v2`](../runs/retrieval/v2), and so were the seven means with their intervals. That run is not in the repository. It can be repeated with the command below and compared with `v2`. A later run of the same search, made with the code of the search variants, is in the repository: [`search-variants-golden/hybrid`](../runs/search-variants-golden/hybrid), which repeats `v2` score for score too, as [the results of the variants](search-variants.md) say.

## Format

`questions.jsonl` is a golden set file: `id`, `question`, `reference_answer`, `evidence` (two entries, each a `doc_id` and a `quote`), `type` (`answerable`), `origin` (`model-drafted`) and `review`: `checked` for the 33 questions the second reading confirmed, and `pending` for the disputed one, which also has a `note`. Every command that reads a golden set reads the file.

`why-two.jsonl` holds what the golden format has no place for: `id`, `kind`, `only_first`, `only_second`, `second_reading` (the label of each of the three showings) and `confirmed`. The loader refuses a `confirmed` that the labels contradict, and `golden-check` refuses a `review` that does not fit it.

`second-reading.jsonl` has one line for each of the 114 items the readers were shown: `item` (the name the reader knew it by; its letter is the form), `question_id`, `shown` (`both`, `first`, `second`, `control` or `control-and-neighbour`), `label`, `confidence`, `answer` and `missing`. The item's name and the last four are the reader's own, unchanged. The question and the showing were joined on from the key of the forms. A test holds the labels in `why-two.jsonl` to these lines.

```bash
# Check every quote against the snapshot, and the file against the rules for two-passage questions
uv run handbook-rag golden-check --dataset evals/two-passage/questions.jsonl --reasons evals/two-passage/why-two.jsonl

# The three saved runs, on the questions whose review is not pending
uv run handbook-rag eval-retrieval --dataset evals/two-passage/questions.jsonl --output evals/runs/two-passage --variant hybrid --mode hybrid --no-rerank --reviewed-only
uv run handbook-rag eval-retrieval --dataset evals/two-passage/questions.jsonl --output evals/runs/two-passage --variant dense --mode dense --no-rerank --reviewed-only
uv run handbook-rag eval-retrieval --dataset evals/two-passage/questions.jsonl --output evals/runs/two-passage --variant sparse --mode sparse --no-rerank --reviewed-only

# The golden set with the same code, to compare with the committed run v2
uv run handbook-rag eval-retrieval --variant current --mode hybrid --no-rerank
```

A run's `summary.json` holds the counts of the tables above under `needs_several_passages`, and names the question it left out under `questions_left_out`. Each row of `results.jsonl` holds `evidence_ranks` (the rank of the first chunk that covers each quote) and `page_ranks` (the rank of the first chunk of each quote's page).

## Known limits

Read these before trusting a number.

- **Checked by models, not by a person.** One model session wrote the questions, and three sessions of the same model read them again. See [what the labels rest on](#the-second-reading).
- **Written by a model that could see both passages.** See [what the runs cannot show](#what-the-runs-cannot-show). There are no questions written without sight of the handbook.
- **Small.** 33 questions, and 6 to 12 of a kind.
- **Fewer than planned.** The issue's plan was about 40. Candidates that did not pass the test from both sides were dropped, and no weaker ones were added to make up the number.
- **The maker's doubts were not the readers'.** Before the second reading the maker named `t017`, `t020` and `t006` as the questions a reader might answer well enough from one passage. The readers called each of them `partial` from either passage alone. The one question they dispute, `t025`, was not on the maker's list.
- **Some facts are stated in a second place.** The evaluation credits the quoted passage only, so these questions can be scored as a miss when the search found the fact:
  - on another page: weekly one-to-ones (`t005`, on several pages about managing), the length of the CSMA programme (`t007`, on the CSM onboarding page), the data level Glean is approved for (`t016`, in the table of AI tools for Support), the examples of a Severity 1 incident (`t017`, on the engineering incident page), what must be done before a contractor starts (`t031`, on an engineering page about hiring contractors);
  - on the same page, in another passage: the length of a CEO shadow rotation (`t006`), the two business days of a legal review (`t012`), the three workdays of job abandonment (`t018`).
  The column "a chunk of both pages" bounds the second group. Nothing bounds the first. The second reading does not touch this limit: a reader saw the two passages only.
- **A few pages carry several questions.** The page of critical system tiers is used by 3 questions, and so is the page of Dutch benefits. 16 of the 66 pieces of evidence are on security pages and 15 on pages of total rewards.
- **Three passages are also evidence of a golden question**: the hire dates of the compensation review (`t024`, golden `q060`), the referral bonus (`t028`, golden `b16`) and the skip-level meetings (`t005`, golden `q103`, the sentence before). The two sets are not independent there.
- **The kinds are not equally hard, and not equally many.** A share over all 33 questions depends on the mix.
- **Ten results.** A run asks the search for ten results. Where a passage would have come further down is not recorded.
- **A snapshot in time, English only, and prose and short lists only**, as in the golden set.

### What the readers said about the questions

Besides its labels, each reader gave the session that ran the reading some remarks on the questions. The remarks themselves are not in `second-reading.jsonl`, which holds the readers' lines only. Each one below was checked by the maker against those lines and against the questions, and what the check found is said with it.

- **Several questions are two facts joined, and not one question.** The readers named eleven items, which are five questions: `t009` and `t012` join two facts by "compared with", and `t028`, `t029` and `t034` join two by "and". That is so, and it is so of more than those five: 11 of the 33 questions join two parts by "and", and 6 comparisons join two sides by "compared with". In every showing of one passage of those five questions, the reader answered one half in full and named the other half as missing. Such a question shows whether a search brings two passages. It does not show that a person would ask for both in one breath.
- **Some comparisons set unlike things side by side.** A share of salary against a share of a "pension base" (`t001`, the one question a reader was less than sure of with both passages in hand). A multiple of annual base salary against a multiple of annual CTC (`t002`: "though the pay base differs", its reader wrote). A cap for an eye test and glasses together against an amount for glasses (`t010`). Days of sick pay by law against a number of paid sick days (`t027`: "they are not added together", its reader wrote). The maker adds `t003`, whose two caps are in two currencies. The reference answers give both figures as the pages give them. One of them, `t001`, also says which is larger, and that holds only if the pension base is not much larger than the salary, which neither passage says.
- **A few questions are worded loosely.** The readers named three items, each a showing of the second passage of a `process` question: `t032` ("What do we do with the report now"), `t031` ("What has to be finished before they can be onboarded") and `t033` ("What do I need to request"). In each the first half is open, and the reader of the second passage could not tell what a full answer to it would hold. The label of `t033` is the only one given with low confidence, and that of `t031` came with medium.
- **With only the linking fact, `partial` was close to `none`.** A reader who is shown only the passage that gives a system's tier, a data class or a severity has a fact the answer is built on, and nothing of what the question asks. In 7 of the 8 such showings of `term-and-rule` questions the reader chose `partial` with medium confidence, and 6 of those answers begin "The passage only establishes", "The passage only shows" or "The passage says only". The eighth is `t018`, whose first passage holds half of what the question asks. Two showings of `rule-and-exception` questions (`t021` and `t023`) were read the same way. The readers said that under a stricter reading these would be `none`. That does not weaken "needs two": `none` says more firmly than `partial` that the passage is not enough. It does mean that `partial` in the file covers two things: half of the answer, and a fact without which the other passage cannot be applied. No showing was labelled `none`.
- **A fact that was missing in one item sometimes stood in another item of the same form.** Four passages are each the evidence of two questions: the table of system tiers, the approval windows, the examples of Orange data and the company-wide life cover. So in 8 showings of one passage, the passage that was held back stood on the same form under another question (`t021` and `t020` on form A, `t013`, `t014` and `t020` on form B, `t013`, `t016` and `t022` on form C). All 8 were labelled `partial`, 6 of them with high confidence. So no reader carried a fact from one item to another, which is also what the readers said. A further reading should not put two questions that share a passage on one form in showings that give each other away.

## Licence and attribution

The evidence quotes are excerpts from the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence. The questions, reference answers and reasons were written for this project from those passages, and the readers' answers in `second-reading.jsonl` restate them.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../docs/adr/0002-handbook-corpus.md).
