# ADR 0023: A second version of the support judge, for the limit an answer leaves out

- Status: Accepted
- Date: 2026-11-25

## Context

**The fault.** An answer can give a rule from its passage and leave out the limit that the passage puts on it: who the rule is for, where or when it holds, how much, on what condition, with what exception. Nothing in such a statement is wrong, and every word of it can be in the passage. Three real answers of the first answer run did this. The readers of the first labelled answers split on them, and the judge of "supported by its sources" passed all three ([issue 59](https://github.com/MumukshuPanchal/handbook-rag/issues/59)).

**A labelled set for this fault alone** ([the card](../../evals/judge/limits/README.md)). 120 answers on 40 passages of the handbook, three for each passage: one keeps the limit, one leaves it out, and one leaves out something that does not limit the rule, so that a judge which faults every omission is not rewarded. One model session wrote and labelled them. Three reader sessions labelled them again from the passage and the rubric alone, each a third of the answers, without seeing those labels, by the account of how they were briefed. The two labellings agree on 196 of 197 statements. The one they split on is labelled `unclear` and not counted, so a check compares 39 statements that leave a limit out and 157 that are supported. No person has read the set.

**The judge on it.** The judge of every score in the results tables caught 13 of the 39 limits left out: 0.333 (95% interval 0.206 to 0.490). It raised no false alarm, and it caught none of the 3 limits that stand only in a passage's title ([`judge/limits`](../../evals/runs/judge/limits)).

**Why it could not simply be changed.** Every verdict is saved with a hash of the judge's instructions and of the layout of its requests. Changed instructions make another judge: its agreement with the labels has to be measured again, and no saved run could be continued or quoted as a run of the judge the code builds.

## Decision

- **A second version of the support judge, beside the first** ([`judge_v2.py`](../../src/handbook_rag/evals/judge_v2.py)), chosen with `--judge-version 2` on `eval-judge` and `eval-support`. It is told [the rubric](../../evals/judge/rubric.md) word for word, with the two rules that version 1 was never told ([issue 58](https://github.com/MumukshuPanchal/handbook-rag/issues/58)). It is told to start a second check from the passage: find what the passage and its title say about who a rule is for, where and when it holds, how much, what it covers, on what condition and with what exception, then see whether the answer gives each. Its reply names the limits it found before its reason and its verdict. It is still one request for each answer.
- **Version 1 stays the judge of "supported by its sources", unchanged.** It gives every score in the results tables and in a run's summary, also at the next measurement of the answers. The default does not change.
- **Version 2 is kept as a second instrument for one fault.** It is the judge to run where the question is whether a limit was left out. Its verdicts on an answer run stay in files of its own beside those of version 1, and its score is reported beside the score of version 1, never in its place.
- **Each version keeps its verdicts on an answer run in files of its own.** Version 1 keeps `support.jsonl`, `support.json` and `support_check.json`. Version 2 writes `support-v2.jsonl`, `support-v2.json` and `support_check-v2.json`, and never reads, continues or discards the files of version 1. A later version of any judge is refused unless its file names carry its number. The run's summary, `eval-table`, `eval-compare` and the README read the files of version 1.

**How the decision came about.** At the start the plan was that version 1 stays the default in this change, and that the next measurement of the answers is judged by version 2, where the default would change. The plan changed when version 2's check on the first labelled answers and its verdicts on the answers of the `baseline` run came in, below. On the first labelled answers it is no better than version 1 by any figure of agreement, and it reads more than twice the tokens. In the 129 saved answers it finds, by the labels, nothing that version 1 missed, and it faults two statements that both readers called supported and one that the readers split on. A decision is allowed to follow the evidence, and this one did. No figure, label or rule of scoring was changed after it was seen.

### What version 2 did on the set

| On the limit answers | Version 1 | Version 2 |
|---|---|---|
| Limits left out that it caught (recall) | 13 of 39, 0.333 (0.206 to 0.490) | 33 of 39, 0.846 (0.703 to 0.928) |
| False alarms | 0 of 157 | 1 of 157 |
| Agreement beyond chance (Cohen's kappa) | 0.445 | 0.882 |
| Input tokens for each answer, on average | 1,159 | 2,916 |
| Median time of a verdict | 1.7 s | 2.0 s |
| Run | [`judge/limits`](../../evals/runs/judge/limits) | [`judge/limits-v2`](../../evals/runs/judge/limits-v2) |

- **Version 2 caught every limit that version 1 caught, and 20 more**, the 3 that stand only in a title among them.
- **It missed six.** In two of them its reason names the very limit that the statement leaves out, and still calls the statement supported.
- **Its one false alarm** faulted a statement for not naming the teams that its question names, though its own instructions say that a limit the question asks about is kept.
- **It costs no more requests**: one for each answer, when none fails. It reads more than twice the input tokens, because its instructions are almost three times as long. On the free tier the allowance is counted in requests.

### What version 2 did on the first labelled answers, and on the real answers

- **Version 2 on the first labelled answers.** It caught 40 of the 41 faults: 0.976 (0.874 to 0.996). It raised 2 false alarms on the 218 statements labelled supported or without a claim, where version 1 raised 1 ([`judge/v2`](../../evals/runs/judge/v2)).
- **There it is worse than version 1.** It missed a planted citation of the wrong passage, and both its false alarms fault a real answer for leaving out a limit, where both readers called the statement supported. Kappa is 0.957 against 0.986, and the same verdict on the whole answer 166 of 169 against 168.
- **Version 2's score of the `baseline` answers.** It calls 125 of the 129 answers supported: 0.969 (0.938 to 0.992). Version 1 calls 127 of them supported ([the verdicts](../../evals/runs/answers/baseline/support-v2.jsonl)).
- **The four statements it faults there.** One cites no passage, and version 1 faulted it too. Each of the other three it faults for a limit left out. The labels call two of those supported and the third unclear. By the labels, version 2 found no fault in the real answers that version 1 missed.

## Alternatives considered

- **Change the instructions of version 1.** Every saved run of the support judge records them, so the changed judge would be another judge under the old name: no saved run could be continued, and the published score would rest on a judge that no saved check measured.
- **Make version 2 the default, now or at the next measurement of the answers.** That was the plan at the start. The runs above do not show version 2 to be better at what the score measures: on the first labelled answers it is no better than version 1 by any figure of agreement, and in the answers of the `baseline` run it finds, by the labels, no fault that version 1 missed. It also reads more than twice the tokens.
- **A second request for each answer, for limits alone.** It would double the requests of every check and every scoring, on a key that allows 500 a day.
- **Change the answering instructions first.** That changes every answer, so the answer evaluation would have to be run again: about 460 requests of the day's 500. It belongs with the hardening against planted instructions, which changes the instructions too. And without a judge that sees the fault, the change could not be measured.
- **A judge of another model family.** [ADR 0014](0014-answer-evaluation-in-stages.md) names it as the better design. It needs a paid key ([issue 17](https://github.com/MumukshuPanchal/handbook-rag/issues/17)).

## What is still missing

What issue 59 still needs, as plans:

- **How often the assistant's own answers leave out a limit.** The 129 saved answers hold too few such cases to show a change: three statements, all disputed. Each of the 40 passages of the limit answers states a limit. Asking the assistant their questions for real, and judging those answers with version 2, with the disputed ones read, would give a rate that a change to the answering instructions can be measured against, before and after.
- **The change to the answering instructions**, and the answers measured again, with the hardening against planted instructions ([issue 66](https://github.com/MumukshuPanchal/handbook-rag/issues/66)). No answer was changed. This decision keeps a judge that catches most of the limits left out of answers written to carry them. On real answers none of its three alarms for a limit is confirmed by the labels, and of the three real cases the readers split on it faults one.
- **A person's reading** of the three disputed statements, statement 1 of `q062`, statement 2 of `q131` and statement 8 of `q104`, and of the labels in general.
- **A judge that catches dropped limits and faults no statement the labels call supported.** Version 2 catches far more dropped limits than version 1, and on the real answers of the first labelled set it faulted two statements that both readers called supported. Neither version measured here does both.

What the figures here do not show:

- **Faults that a model makes by itself.** The 39 were written by a model to carry the fault. The three real cases stay labelled `unclear` in the first labelled answers, and they are the one test of whether a result here carries over to real answers.
- **Another author, and a person.** One model wrote and labelled the set, and three sessions of the same model labelled it again. No person has read it. 39 faults is a small sample, and the answers share passages, three to each, so an interval worked out as if they were independent is too narrow.
- **A set that did not guide version 2.** Its instructions were written after the result of version 1 on this set was known, without the set's answers in view. They share more with the set's card than its seven kinds of limit: their step 3, which asks for a person, a place, a time, an amount or a case that the answer covers and the passage does not, is the card's own test for building the set, and their list of what may be left out is the card's four kinds of harmless omission. No phrase of four or more words of their check for a limit left out is in a passage, a question or a statement of the set. They were written with the first labelled answers in view, their labels, the reasons of version 1 and the words that mark a limit in the passages of the real answers among them, so that check is not held out from version 2 either. That would favour version 2, and it did no better there. A judge has to be checked on answers it was not written around.
- **Version 2 is not steady on the answers it is unsure of.** On the first labelled answers it decides the same statement two ways in two pairs, a real answer and a copy that differs from it only in another statement's citation: `b09` and `b09-v1`, `q104` and `q104-v1`. It runs at the model's default sampling, and each check was run once.
- **The misses and the false alarms.** Version 2 was not changed after its runs. A change would make a third version, to be checked again.

## Consequences

- Every score in the results tables and in a run's summary is as it was, read from the files of version 1.
- The next measurement of the answers will be scored by version 1, as every score in the results tables is. Where it asks whether limits were left out, version 2 runs beside it, and its score is reported beside that of version 1.
- `eval-support --judge-version 2` writes the files of version 2 only. It prints version 2's score beside the run's other scores, and leaves the run's summary as it was.
- Version 2 reads more than twice the input tokens of version 1, for the same number of requests.
- A later version of any judge must name its files by its number, or it is refused.
- Issue 59 stays open.
