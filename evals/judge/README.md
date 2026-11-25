# Labelled answers for checking the judges

Answers whose verdicts are known. They are what a judge is checked against before any score it gives is reported: `eval-judge` gives the judge these answers and counts how often it reaches the labelled verdict. There are two sets, one for each thing a judge decides. The first is described from here on. The second, for whether an answer agrees with its reference answer, has [a section of its own](#the-second-set-answers-against-their-reference-answers) further down, with [a control](#a-control-the-reference-answers-themselves) beside it: the reference answers themselves, given as the answer. A further file for the first judge tests one fault alone, the limit an answer leaves out. It has [a card of its own](limits/README.md), which gives its status. The reasoning is in [ADR 0014](../../docs/adr/0014-answer-evaluation-in-stages.md), and the commands are in the [evaluation guide](../../docs/evaluation.md).

> **Status of both sets: labelled by independent model sessions, not by a person.** One model session labelled every statement, a second labelled them again without seeing the first one's labels, and a third settled the cases they left open. No person has labelled any of them. All three sessions were Claude models, which is another model family than the judge, a Gemini model.

## The first set: statements against the passages they cite

### What is in it

[`support/cases.jsonl`](support/cases.jsonl) holds 169 answers, one on each line.

| | Count |
|---|---|
| Real answers: every reply of the first answer run that answered | 129 |
| Copies of a real answer with one planted fault | 40 |
| Statements in all | 264 |
| labelled `supported` | 215 |
| labelled `unsupported` | 41 |
| labelled `no_claim` (the statement asserts nothing about the handbook) | 3 |
| labelled `unclear`, and not counted | 5 |

The planted faults are what make the check possible. In the 129 real answers the readers found one statement that is unsupported: a correct answer that names no passage. A judge cannot be shown to catch faults from one example.

Each of the 40 copies has exactly one altered statement, five of each kind:

| Kind | What was changed |
|---|---|
| `number` | A number, date or amount became another plausible one |
| `name` | A person, role, team or tool was replaced by another |
| `added_detail` | A plausible detail was added that the passage does not contain |
| `scope` | A limit in the passage was dropped or widened |
| `negation` | A condition or obligation was reversed, or its strength changed |
| `uncited` | The statement is unchanged, and now cites nothing |
| `wrong_passage` | The statement is unchanged, and now cites a passage that does not back it |
| `other_question` | The statement was replaced by one about the same topic that the passage does not contain |

A statement names its passages by number. A passage is recorded by chunk id, with a hash of its text, and the text itself is fetched from the handbook snapshot when the file is loaded. If a passage is no longer in the corpus, or reads differently, loading stops: a label is about exact words. The labels are tied to the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8` and to the chunking `v1-max320-min60-overlap40`.

### How it was made: three readers

The readers decided by [the rubric](rubric.md). The judge's instructions give the same rules except two that have been missing from them since they were first committed, as the [evaluation guide](../../docs/evaluation.md#what-the-judge-decides) describes.

| Step | Reader | Model | What happened |
|---|---|---|---|
| 1 | First | claude-opus-5-5 | Labelled all 204 statements of the 129 real answers: 199 supported, 2 unsupported, 3 without a claim. Marked 7 as borderline. Then wrote the 40 copies with a planted fault, each from a different answer |
| 2 | Second | claude-sonnet-5-5 | Labelled all 264 statements. It was not shown the first reader's labels, and was not told which answers had been altered or how many |
| 3 | Third | claude-opus-5-5 | Settled 8 statements: the 2 on which the first two disagreed, and 6 that one of them had marked as borderline. It was shown both earlier verdicts without being told whose they were |

What the comparison showed:

- **The first two readers agreed on 262 of 264 statements** (0.992, Cohen's kappa 0.974).
- **The second reader marked all 40 planted faults as unsupported**, without knowing they were planted. That is the evidence that each planted fault is real and can be found by reading.
- **On one of the two disagreements, both readers were wrong.** The statement says what the handbook does not specify and adds one fact, which its cited passage backs. One reader wrote that it cites nothing, the other that it only notes what is missing. The third reader checked both claims against the statement and labelled it supported.
- **Three statements are labelled `unclear`.** On each, the third reader ruled `unsupported` where both earlier readers had said `supported`: an answer that left out "except in the circumstances set out below", one that left out that a pension plan is for full-time staff, and one that applied a rule about one kind of meeting to another. The three did not read the same wording of the rule about limits and exceptions: it was being reworded as the labels were made, and only the third reader had the final one. The split may lie between the wordings as much as between the readers, so no verdict is forced. These statements are in the file with all three verdicts, and are left out of every agreement figure. Two of them also appear in a copy, which makes five lines.

The rubric changed while the labels were made. The first reader's notes showed where two careful readers would split, and rules were added for limits and exceptions, for permission and obligation, for a name written in two ways, and for a passage that can be read in two ways. Only the eight statements the third reader settled were read under the final wording. With their copies those are 11 of the 264 lines. The other 253 were labelled under an earlier wording, and on each of them two readers agreed.

### Known limits

- **No person has labelled anything.** Two models of one family can share a blind spot, and where they do, their agreement proves nothing.
- **The planted faults were written by a model.** They may be easier or harder to catch than the faults a model makes by itself. The check says how well the judge catches these, not every kind.
- **One real fault.** Among real answers only one statement is labelled unsupported, so the check cannot say how the judge does on faults that arise naturally.
- **Dropped limits are where the rubric still leaves room.** All three unclear statements are of that kind, and so was the rule that had to be written twice.
- **The set is tied to one snapshot and one chunking.** After a change to either, the passages have to be found again and the labels checked.
- **The real answers come from one run of one model.** A judge that does well on them has not been shown to do well on another model's answers.

### A further file for one fault: the limit an answer leaves out

[`limits/cases.jsonl`](limits/cases.jsonl) holds 120 answers written to test the fault this set could not settle: 40 that keep the limit their passage states, 40 that leave it out, and 40 that leave out something harmless. It is in the format of this set and has [a card of its own](limits/README.md). **Its status is its own: one model session wrote and labelled it, three reader sessions labelled it again without seeing those labels, by the account of how they were briefed, each on a third of the answers, and no person has read it.** The two labellings agree on 196 of its 197 statements, and the one they split on is labelled `unclear` and not counted. The status line at the top of this page does not cover it. Version 1 of the judge caught 13 of its 39 limits left out, and version 2, written for this fault, caught 33 ([its card](limits/README.md#what-the-judges-did-on-it)).

## The second set: answers against their reference answers

[`correct/cases.jsonl`](correct/cases.jsonl) holds 167 answers, one on each line. Each line holds everything the judge reads: the question, the reference answer from the golden set, and the answer.

| | Count |
|---|---|
| Real answers: every answer of the first answer run to a question that has a reference answer | 127 |
| Copies of a real answer with one planted fault | 40 |
| labelled `correct` | 127 |
| labelled `incorrect` | 35 |
| labelled `partial` (right as far as it goes, with part of what was asked missing) | 5 |
| labelled `unclear` | 0 |

**The readers called all 127 real answers correct.** So here too the planted faults are what makes the check possible: without them there would be nothing for a judge to catch. Each of the 40 copies has one fault, five of each kind:

| Kind | What was changed in the answer | Label |
|---|---|---|
| `number` | A number, date or amount the question asks for became another plausible one | `incorrect` |
| `name` | A person, role, team, tool or document was replaced by another | `incorrect` |
| `negation` | The answer was reversed, or its strength changed: optional became mandatory | `incorrect` |
| `scope` | The answer was given for other people, places or plans than the reference gives it for | `incorrect` |
| `contradiction` | The right answer was kept, and a sentence was added that contradicts the reference | `incorrect` |
| `other_question` | The answer was replaced by a confident answer to another question on the same topic | `incorrect` |
| `no_answer` | The answer was replaced by a reply that gives none of what was asked: "the handbook does not say", "it depends", a referral | `incorrect` |
| `omission` | Part of what the question asks for was removed, and what remains is right | `partial` |

Two of these, `other_question` and `no_answer`, are known negatives that [issue 8](https://github.com/MumukshuPanchal/handbook-rag/issues/8) asks for: a confident answer to another question, and "I don't know". The third it names, an empty answer, never reaches the judge: code counts an empty reply, and the fixed sentence the assistant declines with, as declining.

### How it was made

The readers decided by [the rubric](correct/rubric.md). That file is, word for word, what the judge is instructed with, and a test holds the two together. This is the lesson of the first set, whose judge turned out to have been told two rules fewer than its readers. The rubric was also fixed before the first label was made and not changed afterwards, so every label was made under the same words.

| Step | Reader | Model | What happened |
|---|---|---|---|
| 1 | First | claude-opus-5-5 | Labelled the 127 real answers: all correct, 5 of them marked as borderline. Then wrote the 40 copies with a planted fault, each from a different answer |
| 2 | Second | claude-sonnet-5-5 | Labelled all 167. It was not shown the first reader's labels, and was not told which answers had been altered or how many |
| 3 | Third | claude-opus-5-5 | Read the 8 cases that one of the first two had marked as borderline, with both earlier verdicts shown without names |

- **The first two readers gave the same verdict on all 167** (Cohen's kappa 1.000).
- **The second reader faulted all 40 planted faults**, each with the very verdict the first had intended, without knowing they were planted.
- **The third reader agreed on all 8 borderline cases**, so no case is labelled `unclear`.

### Known limits

- **No person has labelled anything**, and all three readers are models of one family. The two who labelled every answer agreed on every one, and the third agreed on the eight it read. That shows the cases are easy for such readers. It does not show that the labels are right.
- **The planted faults are plain by design.** The readers were asked for faults that a careful reader would certainly call a fault. A fault as subtle as a real slip is not in the set, so the check says nothing about those.
- **No real fault at all.** The readers found every real answer correct. How a judge does on wrong answers that arise naturally cannot be told from this set.
- **"Correct" means "agrees with the reference answer".** Most reference answers were written by a model from the same passage the assistant later quotes, which makes agreement likely once the passage is found. And if a reference answer is wrong, an answer that repeats it counts as correct.
- **Where the rubric leaves room**, by the readers' own notes: an answer that adds a condition or a step the reference does not mention; a reference that gives a general instruction beside the one the question asks about; a role named loosely ("Executive Sponsor" for "E-Group Sponsor"). The eight borderline cases are of these kinds.
- **The kinds are the first reader's own names** for what it changed. A separate review judged three of the five `scope` faults to be faults of another kind (another action, an added item, a wider offering). The verdicts stand, and the counts by kind are approximate.
- **One planted omission is itself doubtful.** Its reference answer introduces its list as "examples", so an answer with one example fewer could be called correct. All three readers, and the judge, called it partial.
- **The real answers come from one run of one model.**

### A control: the reference answers themselves

[`correct/references.jsonl`](correct/references.jsonl) holds 132 answers, one for each answerable question of the golden set, in the format of `correct/cases.jsonl`. Each is of the kind `reference`: its answer is the question's reference answer, word for word, and it is labelled `correct`. [Issue 8](https://github.com/MumukshuPanchal/handbook-rag/issues/8) asks that reference answers pass, and this is that check. A judge that does not call the reference answer correct is out of step with the very measure it applies.

- **No reader labelled them.** The label follows from what "correct" means here: agreeing with the reference answer. So no case records readers.
- **The file cannot drift from the golden set.** The loader refuses a `reference` case whose answer is not its reference answer word for word, or whose label is not `correct`. A test makes the 132 cases again from `evals/golden/questions.jsonl` and compares them with the file.
- **There is nothing to catch.** The set holds no fault, so it has no recall, and no kappa either, since every label makes the same decision; `eval-judge` says both in place of a number. What it measures is how many of the 132 the judge calls correct. Each one it does not is a false alarm, listed in `disagreements.jsonl` with the judge's reason.
- **It is the easiest case there is.** The answer and the reference answer are the same words. Passing it shows only that the judge is not out of step with its measure; it says nothing about answers worded otherwise.

Run with `make eval-judge-references`, which costs 132 requests when none fails. Result: `references`, on 23 November 2026, with `gemini-3.5-flash-lite`: the judge called all 132 reference answers correct, so "Same decision" is 1.000 (95% interval 0.972 to 1.000) and there is no false alarm. The saved files record 145 attempts for the 132 answers. That the other 13 were turned away by an overloaded model and made again was printed by the run, and is not saved. The run is in [`evals/runs/judge-correct/references`](../runs/judge-correct/references).

## Licence and attribution

The statements and answers are replies of this project's assistant, and edits of those replies. Those in `limits/` are neither: a model wrote them for the purpose, as [their card](limits/README.md) says. The questions and reference answers in the second set are those of the [golden set](../golden/README.md). Some repeat a few words of the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence. The file holds no passage of the handbook: passages are named by id and read from the snapshot.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../docs/adr/0002-handbook-corpus.md).
