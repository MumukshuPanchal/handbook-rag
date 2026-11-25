# ADR 0023: A second version of the support judge, for the limit an answer leaves out

- Status: Accepted
- Date: 2026-11-25

## Context

**The fault.** An answer can give a rule from its passage and leave out the limit that the passage puts on it: who the rule is for, where or when it holds, how much, on what condition, with what exception. Nothing in such a statement is wrong, and every word of it can be in the passage. Three real answers of the first answer run did this. The readers of the first labelled answers split on them, and the judge of "supported by its sources" passed all three ([issue 59](https://github.com/MumukshuPanchal/handbook-rag/issues/59)).

**A labelled set for this fault alone** ([the card](../../evals/judge/limits/README.md)). 120 answers on 40 passages of the handbook, three for each passage: one keeps the limit, one leaves it out, and one leaves out something that does not limit the rule, so that a judge which faults every omission is not rewarded. One model session wrote and labelled them. Three reader sessions labelled them again from the passage and the rubric alone, each a third of the answers, without seeing those labels. The two labellings agree on 196 of 197 statements. The one they split on is labelled `unclear` and not counted, so a check compares 39 statements that leave a limit out and 157 that are supported. No person has read the set.

**The judge on it.** The judge of every published score caught 13 of the 39 limits left out: 0.333 (95% interval 0.206 to 0.490). It raised no false alarm, and it caught none of the 3 limits that stand only in a passage's title ([`judge/limits`](../../evals/runs/judge/limits)).

**Why it could not simply be changed.** Every verdict is saved with a hash of the judge's instructions and of the layout of its requests. Changed instructions make another judge: its agreement with the labels has to be measured again, and no saved run could be continued or quoted as a run of the judge the code builds.

## Decision

- **A second version of the support judge, beside the first** ([`judge_v2.py`](../../src/handbook_rag/evals/judge_v2.py)), chosen with `--judge-version 2` on `eval-judge` and `eval-support`. It is told [the rubric](../../evals/judge/rubric.md) word for word, with the two rules that version 1 was never told ([issue 58](https://github.com/MumukshuPanchal/handbook-rag/issues/58)). It is told to start a second check from the passage: find what the passage and its title say about who a rule is for, where and when it holds, how much, what it covers, on what condition and with what exception, then see whether the answer gives each. Its reply names the limits it found before its reason and its verdict. It is still one request for each answer.
- **Version 1 stays the default, unchanged.** Every published score is read from its verdicts.
- **Each version keeps its verdicts on an answer run in files of its own.** Version 1 keeps `support.jsonl`, `support.json` and `support_check.json`. Version 2 writes `support-v2.jsonl`, `support-v2.json` and `support_check-v2.json`, and never reads, continues or discards the files of version 1. A later version of any judge is refused unless its file names carry its number. The run's summary, `eval-table`, `eval-compare` and the README read the files of version 1.
- **The next measurement of the answers is to be judged by version 2.** When the answers are measured again, with new instructions against planted instructions ([issue 66](https://github.com/MumukshuPanchal/handbook-rag/issues/66)), they are to be judged by version 2 and compared with version 2's score of the `baseline` answers. The default changes there, when the README's figures are rebuilt. This is a plan, not something done.

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
- **It costs no more requests**: one for each answer, when none fails. It reads two and a half times the input tokens, because the rubric is in its instructions. On the free tier the allowance is counted in requests.

## Alternatives considered

- **Change the instructions of version 1.** Every saved run of the support judge records them, so the changed judge would be another judge under the old name: no saved run could be continued, and the published score would rest on a judge that no saved check measured.
- **Make version 2 the default now.** The published "supported" would move to another judge with no change to the answers. The README's figures are rebuilt with the next measurement of the answers, and the default changes with them.
- **A second request for each answer, for limits alone.** It would double the requests of every check and every scoring, on a key that allows 500 a day.
- **Change the answering instructions first.** That changes every answer, so the answer evaluation would have to be run again: about 460 requests of the day's 500. It belongs with the hardening against planted instructions, which changes the instructions too. And without a judge that sees the fault, the change could not be measured.
- **A judge of another model family.** [ADR 0014](0014-answer-evaluation-in-stages.md) names it as the better design. It needs a paid key ([issue 17](https://github.com/MumukshuPanchal/handbook-rag/issues/17)).

## What is still missing

- **Version 2 on the first labelled answers.** It is not yet checked on the first labelled answers, the 169 that version 1 was checked on, where it must not start faulting good answers. Until it is, no score of version 2 is quoted.
- **Version 2's score of the `baseline` answers.** The next measurement is to be compared with it, and it is not yet recorded.
- **Answers that keep their limits.** No answer was changed. Issue 59 stays open: this decision makes the fault measurable and gives a judge that sees it, and it does not remove the fault from the answers.
- **Faults that a model makes by itself.** The 39 were written by a model to carry the fault. The three real cases stay labelled `unclear` in the first labelled answers, and they are the one test of whether a result here carries over to real answers.
- **Another author, and a person.** One model wrote and labelled the set, and three sessions of the same model labelled it again. No person has read it. 39 faults is a small sample, and the answers share passages, three to each, so an interval worked out as if they were independent is too narrow.
- **A set that did not guide version 2.** Version 2 was written after the result of version 1 on this set was known, by a session that was not shown the set's answers. The repository does not record what else that session read: the set's card and the run of version 1 were in it, and the instructions of version 2 list the same seven kinds of limit that the card sorts the set by. A judge written with a set's result in view has to be checked on answers it was not written around.
- **The six misses and the false alarm.** Version 2 was not changed after its run. A change would make a third version, to be checked again.

## Consequences

- Every published score is as it was, read from the files of version 1.
- `eval-support --judge-version 2` writes the files of version 2 only. It prints version 2's score beside the run's other scores, and leaves the run's summary as it was.
- Version 2 reads two and a half times the input tokens of version 1, for the same number of requests.
- A later version of any judge must name its files by its number, or it is refused.
- Issue 59 stays open.
