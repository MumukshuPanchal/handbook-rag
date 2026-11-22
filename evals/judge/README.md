# Labelled answers for checking the judge

Answers whose statements have known verdicts. They are what the judge is checked against before any score it gives is reported: `eval-judge` gives the judge these answers and counts how often it reaches the labelled verdict. The reasoning is in [ADR 0014](../../docs/adr/0014-answer-evaluation-in-stages.md), and the commands are in the [evaluation guide](../../docs/evaluation.md).

> **Status: labelled by independent model sessions, not by a person.** One model session labelled every statement, a second labelled them again without seeing the first one's labels, and a third settled the cases they left open. No person has labelled any of them. All three sessions were Claude models, which is another model family than the judge, a Gemini model.

## What is in it

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

## How it was made: three readers

The readers decided by [the rubric](rubric.md). The judge's instructions give the same rules except two that were lost from them in an edit, as the [evaluation guide](../../docs/evaluation.md#what-the-judge-decides) describes.

| Step | Reader | Model | What happened |
|---|---|---|---|
| 1 | First | claude-opus-5-5 | Labelled all 204 statements of the 129 real answers: 199 supported, 2 unsupported, 3 without a claim. Marked 7 as borderline. Then wrote the 40 copies with a planted fault, each from a different answer |
| 2 | Second | claude-sonnet-5-5 | Labelled all 264 statements. It was not shown the first reader's labels, and was not told which answers had been altered or how many |
| 3 | Third | claude-opus-5-5 | Settled 8 statements: the 2 on which the first two disagreed, and 6 that one of them had marked as borderline. It was shown both earlier verdicts without being told whose they were |

What the comparison showed:

- **The first two readers agreed on 262 of 264 statements** (0.992, Cohen's kappa 0.974).
- **The second reader marked all 40 planted faults as unsupported**, without knowing they were planted. That is the evidence that each planted fault is real and can be found by reading.
- **On one of the two disagreements, both readers were wrong.** The statement says what the handbook does not specify and adds one fact, which its cited passage backs. One reader wrote that it cites nothing, the other that it only notes what is missing. The third reader checked both claims against the statement and labelled it supported.
- **Three statements are labelled `unclear`.** On each, the third reader ruled `unsupported` where both earlier readers had said `supported`: an answer that left out "except in the circumstances set out below", one that left out that a pension plan is for full-time staff, and one that applied a rule about one kind of meeting to another. Readers with the same rule in front of them split, so no verdict is forced. These statements are in the file with all three verdicts, and are left out of every agreement figure. Two of them also appear in a copy, which makes five lines.

The rubric changed while the labels were made. The first reader's notes showed where two careful readers would split, and rules were added for limits and exceptions, for permission and obligation, for a name written in two ways, and for a passage that can be read in two ways. Only the eight statements the third reader settled were read under the final wording. With their copies those are 11 of the 264 lines. The other 253 were labelled under an earlier wording, and on each of them two readers agreed.

## Known limits

- **No person has labelled anything.** Two models of one family can share a blind spot, and where they do, their agreement proves nothing.
- **The planted faults were written by a model.** They may be easier or harder to catch than the faults a model makes by itself. The check says how well the judge catches these, not every kind.
- **One real fault.** Among real answers only one statement is labelled unsupported, so the check cannot say how the judge does on faults that arise naturally.
- **Dropped limits are where the rubric still leaves room.** All three unclear statements are of that kind, and so was the rule that had to be written twice.
- **The set is tied to one snapshot and one chunking.** After a change to either, the passages have to be found again and the labels checked.
- **The real answers come from one run of one model.** A judge that does well on them has not been shown to do well on another model's answers.

## Licence and attribution

The statements are replies of this project's assistant, and edits of those replies. Some repeat a few words of the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence. The file holds no passage of the handbook: passages are named by id and read from the snapshot.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../docs/adr/0002-handbook-corpus.md).
