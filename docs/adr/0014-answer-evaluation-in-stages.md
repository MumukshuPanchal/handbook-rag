# ADR 0014: Evaluate answers in stages, within a daily allowance of requests

- Status: Accepted
- Date: 2026-11-22

## Context

[ADR 0007](0007-evaluation-strategy.md) decided how answers would be measured: a judge model scores each answer on three yes/no criteria, and the judge is checked against labelled examples before its scores are trusted. That was written before the project had an answer model it could run.

It has one now. [ADR 0013](0013-gemini-free-tier-with-marker-citations.md) put answering on a free Gemini key, and the limits of that key decide what an evaluation can be. They were read in Google AI Studio on 2026-11-22, for the Google project the key belongs to. Google does not guarantee them.

| Model | Requests a minute | Requests a day |
|---|---|---|
| `gemini-3.5-flash-lite`, the answer model | 15 | 500 |
| Gemini 3.6 Flash, 3.8 Flash and 4 Hydrogen | 5 | 20 |

Four facts follow from this, and from what was seen while building the answering step:

- **One run is a large part of a day.** The golden set has 165 questions, so asking them all is 165 requests. A judge adds one request for every answer and every criterion it scores.
- **A retried request is counted again.** The client that serves `ask` tries a failed request up to three times. One question to an overloaded model showed as three requests in the usage table, with three times the tokens.
- **Only one free model can judge.** The larger models allow 20 requests a day, which is not enough to judge 165 answers.
- **The same question gets a different answer each time.** Three runs of one question on 2026-11-22 gave three answers: 79, 121 and 127 tokens long, one of them stating a rule for one country as if it held everywhere.

## Decision

1. **Stages, each reading what the one before saved.**

   | Stage | What it does | Requests |
   |---|---|---|
   | Ask | Puts every golden question through the pipeline and saves each reply | 165 |
   | Score in code | Scores the saved replies on what a program can decide | none |
   | Judge: supported | Asks a judge whether each statement is backed by the passage it cites | one for each answer |
   | Judge: correct | Asks a judge whether the answer agrees with the reference answer | one for each answer to an answerable question |

   A stage can run on another day than the one before it. A judge can be changed, or run again, without asking the questions again.

2. **"Right to answer or to decline" is computed, not judged.** ADR 0007 listed it as a criterion for the judge. No judge is needed: the generator already decides by a fixed rule whether a reply is an answer ([ADR 0010](0010-search-result-blocks-and-declining.md), [ADR 0013](0013-gemini-free-tier-with-marker-citations.md)), and the golden set says which questions the handbook answers. Code compares the two. That is free, and the same every time.

3. **Three more scores come from code**, because the replies carry their citations as passage numbers and the golden set carries its evidence as quotes:

   | Score | Question it answers |
   |---|---|
   | Evidence among the passages | Did the search give the model the passage that holds the answer? This is the search's Hit@5, kept beside the answer scores to tell a search miss from a wrong decline |
   | Every statement cited | Does every statement in an answer name a passage? |
   | Cites the evidence | Does an answer that was given the evidence name the passage that holds it? |

4. **A run is careful with requests.**
   - It sends at most 10 a minute by default, under the limit of 15, so the key stays usable for other things while a run is going.
   - The client is built with one attempt for each request, so no request is sent a second time unseen. The run counts each question before it sends it, and tries a failed question once more itself, after the others.
   - Each reply is saved as it arrives. A run that stops continues where it left off.
   - A run records its settings, including a hash of the instructions given to the model, and refuses to continue with other settings.
   - Three failures in a row stop the run. By then the cause is almost certainly the day's allowance, the network or the key, and each further question would spend a request to learn the same thing.
   - A failure is recorded as a failure. It is never scored as a wrong answer.

5. **The judge is the answer model, and it has to earn trust first.** `gemini-3.5-flash-lite` is the only free model with enough requests. A model judging its own output tends to be lenient, so before any of its scores is reported it is run on labelled examples, among them answers with a known fault, and its agreement with the labels is reported beside every score it produces. The labels are made by one model session and checked by a separate one that has not seen how they were made, both of another model family than the judge. They are recorded as `checked`. No person has labelled them, and nothing may say otherwise.

6. **One reply for each question, at the settings the product uses.** The evaluation measures the assistant as it runs, so it does not lower the sampling temperature to make replies repeat. Google's guidance for this model family, read on 2026-11-22, is also to leave the sampling settings at their defaults. Every mean is reported with a 95% interval over the questions.

## Alternatives considered

- **Judge in the same step as asking.** Simpler, and it would tie every change to the judge to 165 new answers. Saving the replies costs one file.
- **A stronger free model as the judge.** At 20 requests a day it could judge a sample of twenty. That may still be worth doing later, to audit the cheaper judge on a small sample.
- **A paid key, or a judge from another provider.** Either would give a stronger judge from another model family, which is the better design: a judge should not share the blind spots of what it judges. Both need an account or money this project does not have yet. The judge sits behind a setting, so changing it later is one variable.
- **Temperature 0 for evaluation only.** Replies would repeat, and the evaluation would measure a system that nobody uses.
- **Several replies for each question.** This is the right way to measure how much replies vary. Three replies for each question would be 495 requests, a whole day's allowance with nothing left for a judge.
- **RAGAS or DeepEval.** Not adopted, for the reasons in ADR 0007.

## Consequences

- A full evaluation takes most of a day's allowance. On the first run that is 169 attempts to ask, then 129 requests to judge whether the answers are supported and 127 to judge whether they are correct: about 425 of the 500. Checking a judge against its labelled examples takes one more request for each example. So on a day when a judge is first checked, not everything fits, and what is left waits for the allowance to reset at midnight Pacific time.
- The first numbers, from code alone, are available as soon as the questions have been asked.
- A reply can be read again at any time: `answers.jsonl` holds every reply with the passages it cited, by chunk id. The passages themselves are not copied. The replies are the model's own words, and one of them may repeat a sentence of a passage, which is the kind of short, attributed excerpt [ADR 0002](0002-handbook-corpus.md) allows.
- The reply is kept as statements and citations, not as the text the model wrote. When an answer comes back without a citation, the saved record cannot show whether the model left the marker out or the parser lost it.
- "Right call" is 1 or 0 by a fixed rule. A reply that gives a related fact and then says the handbook lacks the detail asked for counts as an answer, and so as a wrong call on an unanswerable question. The first run had two such replies.
- Scores from one run do not show how much replies vary from run to run. The interval reported with each mean covers which questions were asked, not how the model would answer them a second time. This is a stated limit.
- Until a stronger judge is available, every judged score depends on a small model judging itself. The agreement figure reported beside it is the measure of how far to trust it.

## Update, 2026-11-22: the judge was checked, and the first judged score recorded

What decision 5 asked for was done the same day. The numbers are in the [evaluation guide](../evaluation.md#the-judged-score-supported-by-its-sources), and the labelled answers have a [card](../../evals/judge/README.md) of their own.

- **The labels.** 169 answers with 264 statements: the 129 real answers of the first run, and 40 copies with one planted fault. Three model sessions of another family than the judge labelled them, the second without seeing the first one's labels. They agreed on 262 statements, and the second found all 40 planted faults without being told that any were planted.
- **The check.** The judge caught 41 of 41 statements labelled unsupported and raised one false alarm in 42.
- **The score.** 0.984 of the 129 answers are supported by their sources.

Doing it changed four things in the design:

1. **Checking the judge and scoring a run share their verdicts.** A verdict is saved under a hash of what the judge read, so the 129 real answers were judged once, in the check, and scoring the run cost no request. The day's count was 169 attempts to ask and 201 to judge. The second judged criterion waits for the next day's allowance after all, because its own labelled examples need judging too.
2. **A failed request is followed by a rest.** Thirty-two attempts at judging failed, 31 of them because the model was overloaded. The first run of the check stopped itself after three in a row, as designed, although a short wait would have outlasted the overload. The loop now rests for 30 seconds after each failure, which costs no request.
3. **A plain share gets a Wilson interval.** The bootstrap gave "41 of 41" an interval from 1.000 to 1.000. The Wilson interval runs from 0.914, which is what 41 cases can show.
4. **Three verdicts, and room for "unclear".** A statement that only says what the handbook does not cover asserts nothing to check, so it has a verdict of its own. And where a third reader stood against two, the statement is labelled unclear and left out of the agreement figures, instead of a verdict being forced.

Two weaknesses were found, and neither is fixed:

- **Limits and exceptions that an answer leaves out.** The three unclear statements are all of this kind, the rubric had to be reworded for it twice, and the judge passed all three. It is also the fault seen by eye on the first day: a rule for one country given as if it held everywhere. Nothing in this evaluation measures it reliably yet.
- **The judge can lose count in a long answer.** In one answer of nine statements its reasons slipped by one from the fourth statement on, and the last statement was never judged. Asking the judge to repeat the first words of each statement, and refusing a reply where they do not match, would catch this.
