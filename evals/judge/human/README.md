# A sample of the labelled answers, for one person to label

> **If you are the one who labels, read this after you have labelled.** It says how the batch is made up, which could sway a close call. The steps you need are beside the page, in [`batch-1/HOW-TO-LABEL.md`](batch-1/HOW-TO-LABEL.md).

Every figure of agreement in this project compares a model with labels that model sessions made and checked: the [two labelled sets](../README.md) the judges are checked against, and every reading since. This sample is the first time a person labels any of it. One person labels a small sample drawn from the two sets, without seeing anyone else's label, and `eval-human` reports how far the sets' labels and the judges' saved verdicts agree with that person. The reasoning is in [ADR 0026](../../../docs/adr/0026-one-person-labels-a-sample-of-the-labelled-answers.md), and [issue 8](https://github.com/MumukshuPanchal/handbook-rag/issues/8) is the issue it belongs to.

> **Status: batch 1 waits for the owner's labels.** No person has labelled any item yet, so no figure of agreement with a person exists. This card says what will be counted, and was written before any label existed.

## Who labels

One person: the project's owner. The owner is not an expert on GitLab, and labels from what the page shows and from the rubric, as the model readers did. The labels file records the labeller as "the project's owner" and no name or address. What comes out is one person's labels on a small sample, and it is never called more than that: not a verification of the sets, and not a review by an expert.

## What a batch holds

A batch is 24 judgements: 16 statements (part A) and 8 answers (part B), planned as about half an hour. Each part matches one judge.

| Part | One item is | The owner is shown | The owner chooses |
|---|---|---|---|
| A | One statement of an answer | The question; the whole answer, statement by statement, with the passages each cites and the statement to label marked; the full text of each passage the marked statement cites, with its title path | supported, unsupported, no claim, cannot tell |
| B | One answer | The question, the reference answer and the answer | correct, partial, incorrect, cannot tell |

Each item may take a short note. "Cannot tell" is for an item that what is shown does not let the owner decide.

## How a batch is drawn

`handbook-rag human-sample` draws it from [`support/cases.jsonl`](../support/cases.jsonl) and [`correct/cases.jsonl`](../correct/cases.jsonl) and the judges' saved checks on them, with a fixed seed. It sends no request to any model.

**Disputed items first.** A person's label tells most where the instruments disagree. So a batch first takes, in each part, the statements or answers on which a judge's saved verdict differs from the set's label (version 1 or version 2 of the judge of support, the judge of correctness), on which the set's first two readers gave different labels, or which the set calls `unclear`: as many as there are, up to 6 statements and 3 answers, answers first. Their places spread over these sorts of dispute.

**Then the rest, in groups by the set's label**, half labelled with a fault and half not; an odd place goes to the side of which the disputed items hold fewer. The statements labelled `unsupported` are spread over the kinds of planted fault. The answers not correct hold at least one labelled `partial` and one `incorrect` where the set has them. Each statement labelled `supported` is drawn from a real answer of the same shape, one statement or several, as a statement labelled `unsupported` drawn before it, as far as the set allows. The supported statements and the correct answers come from real answers; statements labelled `unclear` or `no_claim` are drawn only as disputed items.

**Within a group**, each place goes to the kind drawn least so far, counting earlier batches; then to the kind with the most left; then by the seed; and then a question, and one of its statements, by the seed, so that a long answer is not drawn more often for having more statements. **No question is drawn twice**, in either part or in a later batch: seen twice, a question would show a real answer beside its copy with a planted fault, and part B's reference answer would tell more than part A's passages.

**What the sets and the saved checks hold.** 13 statements are disputed, on 10 questions: 4 on which version 1 of the judge of support differs from the set, 4 on which version 2 does (one statement is among both), 2 on which the readers differed (one of them also among version 1's), and 5 the set calls `unclear`. 3 answers are disputed, each on which the judge of correctness differs from the set; the readers differed on none. One question holds a disputed statement and a disputed answer.

## Batch 1

Drawn with the seed 20261128, on the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8` and the chunking `v1-max320-min60-overlap40`.

| Group | Part | In batch 1 |
|---|---|---|
| Disputed statements | A | 6: 2 on which version 1 of the judge of support differs from the set, 1 on which version 2 differs, 1 on which the readers differed, and 2 the set calls `unclear` |
| Statements labelled `unsupported` | A | 5: one of each of 5 kinds of planted fault |
| Statements of real answers labelled `supported` | A | 5 |
| Disputed answers | B | 3, on each of which the judge of correctness differs from the set |
| Answers labelled `partial` or `incorrect` | B | 3: 1 `partial` and 2 `incorrect` |
| Real answers labelled `correct` | B | 2 |

Of the 6 disputed statements the set labels 2 `supported`, 1 `unsupported`, 1 `no_claim` and 2 `unclear`; of the 3 disputed answers it labels 2 `correct` and 1 `partial`. The 5 statements labelled `unsupported` all come from answers of one statement, and so do the 5 labelled `supported` that are matched to them.

**On the 15 items that are not disputed, every judge's saved verdict is the set's label.** So "the owner against a judge" differs from "the owner against the set" only on the disputed items: on 2 for version 1, on 1 for version 2, and on 3 for the judge of correctness. That is why the disputed items come first, and why the report gives them a section of their own.

**The first draw of batch 1 was replaced before any label existed.** It was made on 28 November 2026 by an earlier rule: 8 statements labelled supported and 8 unsupported, 4 answers correct and 4 not. A review found two things. On every one of its 24 items each judge's saved verdict was the set's label, so every comparison with a judge would only have repeated the comparison with the set. And its part B held no answer labelled `partial`, where the judge of correctness is known to be weak. Its key, which gives the label of each of its items, had been committed and pushed with it. So the batch was drawn again by the rule above, with the same seed, and the 24 statements and answers of the first draw are set aside: [`set-aside.json`](set-aside.json) names them by hash, and no batch draws them, so that key gives the label of no item of a batch. 6 questions of batch 1 were in the first draw too, each through another statement or the other part's answer.

## What the owner is shown, and what the code checks

Before a batch is written, the code checks what the owner will see, and a batch that fails is not written:

- **fields:** each item holds only the fields of what is shown, so that no label, verdict, reader, kind or group can ride along in another field;
- **ids:** items are named by their part and place, `A01` to `A16` and `B01` to `B08`;
- **words:** each item shows exactly the words of its case; its question is that of the real answer, down to a space; and no text of it starts or ends with white space;
- **names:** no id of the sets is in the items, the page or the steps (an id such as `b04-v1` names a copy), and no item's text holds a name of a kind, a group or a sort of dispute that no real answer of the sets uses;
- **order:** within a part, the order does not put each group in one run, nor keep every two of a group apart, where each group of the part has at least 3 members; it is shuffled again by the seed until it does neither. A smaller group is left where the shuffle puts it, since rejecting some orders would give its place away. In batch 1 this applies to part A, whose groups have 6, 5 and 5; part B's groups have 3, 3 and 2, so its order is the shuffle's;
- **the page:** its data holds the items, the choices and the rubrics and nothing else; it holds no comment; and neither its text, its script nor the steps hold a word about how a batch is made up, such as "half", "fault", "group" or "kind".

The key, which says which case and statement each item is and the label the set gives it, is not in the batch's folder until the labels are in place. `batch.json` holds only how the batch was drawn (the seed, the plan, the files it was drawn from with their hashes) and the SHA-256 of the key. `eval-human` draws the key again from the seed, the sets and the saved checks, refuses it unless it has that hash, and only then writes it beside the labels.

**What these checks do not cover:**

- **The make-up of the batch.** A person who reads this card, the decision record, the commit or the pull request that added the batch learns how it is made up: that it has disputed items, how many of each label, which kinds of fault. The page and the steps ask the owner to read none of them before the labels are in place.
- **The sets themselves.** Every question, statement and answer of the batch is in the committed sets with its label. Searching an item's text there finds its label. The steps ask the owner not to.
- **The shape of an answer.** 5 of the 6 disputed statements come from answers of several statements, where every other statement of part A comes from an answer of one. So a long answer says "disputed", which is not a label; the disputed statements' labels are mixed. Where in a long answer the marked statement stands is not matched either, and in part B the shape of an answer is not matched at all.
- **What a planted fault looks like.** A statement that cites no passage, or an answer that gives no answer, is decided by the rubric from what is shown alone. Planted faults were written to be plain. Agreement on them is likely, and raises the shares.

## The page

`batch-1/label.html` is one file that works from disk, with no server and no network. Its style and script are inside it, and a security policy in the page forbids it to load or send anything. It shows the rubric in short at the top, with each rubric in full under it, then one item at a time, how many are done, and a strip of the items to jump between them. Keys 1 to 4 choose, N and P move. Every text from the sets and the handbook goes into the page as data and onto the screen as text, never as markup. One passage of batch 1 holds a web address: it is shown as text in that passage, never as a link, and nothing loads it.

The page keeps unsaved labels in the browser's local storage and says so. Saving the file does not clear them: a button does, after asking. When the browser refuses to keep them, or its storage is full, or the page is open in a second tab, the page says that labels are not being kept and warns before it is closed. Labels found from before are shown with how many there are and when they were last changed, and are taken only when the owner chooses to keep them; labels of another batch or of another draw are never taken. Each save makes a new file, named by the batch and the time of the save. A test runs the page's script in node through each of these.

## The labels file

`handbook-rag human-labels <file>` checks the file the page saved and puts it in place as `batch-1/labels.json`: every item of the batch present once, only the choices of its part, the batch's number, the hash of the items drawn and the hash of each item matching, a note of at most 1000 characters, the labeller recorded as "the project's owner", the counts of labels and notes the file states, and no other field. A file that does not fit is refused, with a message that says what to do. So is a file beside which a newer labels file of the same batch lies, unless `--older` is given; the command prints when the file was saved and how many labels and notes it holds. Labels in place are not replaced without `--replace`, and `eval-human` checks them again before it counts them. A note is free text, and the code does not read it.

## What will be reported

`handbook-rag eval-human --variant batch-1` will write the run to `evals/runs/judge/human/batch-1`. It sends no request to any model. For each part:

- **The owner against the sets' labels**, the label each set held when the batch was drawn, as the key records it: the same verdict; the same decision, a fault or not (unsupported in part A; partial or incorrect in part B); Cohen's kappa for that decision; and for each label of the set, how many of its items get the same label from the owner. A statement the set calls `unclear` has no label to compare, and is counted apart.
- **The owner against each judge's saved verdicts**: version 1 of the judge of support, which gives every published score ([`judge/baseline`](../../runs/judge/baseline)), version 2 beside it ([`judge/v2`](../../runs/judge/v2)), and the judge of correctness ([`judge-correct/baseline`](../../runs/judge-correct/baseline)). The same verdict and the same decision, kappa, of the items the owner calls a fault how many the judge faulted, and of the judge's faults how many the owner agrees with. These are the verdicts of the judges' saved checks on the labelled sets, the very ones their published agreement rests on. A verdict counts only if it was given on exactly the words drawn, by the fingerprint of what the judge read; an item with none is counted apart, as having no saved verdict, and never as agreement or disagreement. Every item of batch 1 has a verdict of each judge of its part.
- **The disputed items**: for each judge, of the items on which it differs from the set, how many times the owner sides with the set, with the judge, with neither, or cannot tell; and for the statements the set calls `unclear`, what the owner called them.
- **"Cannot tell" is counted apart**, never as agreement or disagreement.
- **Every share is given with its count and a 95% Wilson interval.** Kappa is left out where one side makes only one decision.
- **In full, for a later look**: every item on which the owner differs from a set or from a judge in `differences.jsonl`, every "cannot tell" in `cannot-tell.jsonl`, and every disputed item in `disputed.jsonl`, each with what was shown, the owner's label and note, the set's label, the kind of case, and each judge's verdict and reason. Neither the owner nor the set is taken to be right there. The run also writes `summary.json` and `run.json`, which says when it was made, by which version of the code, from which labels, key and saved checks, by their hashes.

## What it can carry, and what it cannot

- **One person.** One person's labels say how far the sets and the judges agree with that person. They do not say who is right where they differ, and they are not a verification of the sets.
- **A beginner in the subject.** The owner labels from the passage and the rubric, as the model readers did, and knows no more about GitLab than the page shows.
- **24 judgements.** The intervals are wide: agreement on all 5 statements of one label would still leave an interval that reaches down to 0.566, and on all 16 statements of part A, down to 0.806. The disputed items are 9, so each judge's section rests on 1 to 3 items.
- **Drawn in groups, from sets in which a model planted the faults.** The mix of labels in the batch is chosen, where the real answers hold 1 unsupported statement in 204 and no answer labelled not correct in 127. So no share here is a natural rate, and kappa depends on this mix. The planted faults were written to be plain.
- **The owner knows the project**: that the sets hold copies with planted faults, and their kinds. An owner who has read this card knows the make-up of the batch.
- **The judges' verdicts are those of one saved check each**, made at the model's default sampling.

## What a second labeller or a second pass would need

Nothing of it is built. The labels file records `labeller` and `pass`, and the check accepts only "the project's owner" and pass 1. A second labeller needs another value of `labeller` that names no person, a file of their own beside `labels.json`, the check to accept both, and the report to compare the two people with each other. A second pass by the owner needs pass 2, made after some time, and an order of the items other than the first: the page shows the order of `items.jsonl`, and has no second order.

## Files

| File | What it holds |
|---|---|
| `batch-1/items.jsonl` | The 24 items, one on each line, as the page shows them, each with the hash of its content |
| `batch-1/batch.json` | How the batch was drawn: the seed, the plan, the files it was drawn from with their hashes; the hash of `items.jsonl`, and the SHA-256 of the key |
| `batch-1/label.html` | The page |
| `batch-1/HOW-TO-LABEL.md` | Five steps for the owner |
| `batch-1/labels.json` | The owner's labels, once they are in place. Not there yet |
| `batch-1/key.json` | For each item: its case and statement, the question, its group, its kind, its disputes and the label the set held. Written by `eval-human` once the labels are in place. Not there yet |
| `set-aside.json` | The statements and answers of the first draw of batch 1, by hash, which no batch draws |

## Licence and attribution

`items.jsonl` and `label.html` hold the full text of the handbook passages they show, from the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence; the page says so in words, without a link. The questions and answers are those of the labelled sets. The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab.
