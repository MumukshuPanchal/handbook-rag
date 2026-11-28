# ADR 0026: One person labels a sample of the labelled answers, blind

> **If you are the one who labels, read this after you have labelled.** It says how the batch is made up, which could sway a close call.

- Status: Accepted
- Date: 2026-11-28

## Context

**Every score rests on labels that only models have made and checked.** The judges of "supported by its sources" and "correct by the reference" were checked against [two labelled sets](../../evals/judge/README.md). Model sessions labelled them, a second session labelled them again without seeing the first one's labels, and a third settled the cases left open. All were Claude models. No person has labelled any of it, and the README's first limitation says so. Two models of one family can share a blind spot, and where they do, their agreement shows nothing. [Issue 8](https://github.com/MumukshuPanchal/handbook-rag/issues/8) has one line left open: "Judge agreement with human labels is reported".

**One person can label, for a limited time.** The project's owner will label. The owner is not an expert on GitLab, and labels from the passage and the rubric, as the model readers did. Labelling costs no request to a model, only reading time, so the time is to be spent where a person's label tells most.

**Where the instruments already agree, a person's label adds little.** Of the labelled statements and answers, each judge's saved verdict equals the set's label on all but a few. A sample drawn by the set's label alone would compare the person with the judges only where the judges and the set agree, and the comparison with a judge would repeat the comparison with the set. The disputes are few: 13 statements on 10 questions, and 3 answers ([the card](../../evals/judge/human/README.md#how-a-batch-is-drawn) counts them).

## Decision

- **A small sample, in batches** ([the card](../../evals/judge/human/README.md)). The first batch is 24 judgements, about half an hour: 16 statements against the passages they cite (part A, the judge of support) and 8 answers against their reference answers (part B, the judge of correctness). Later batches are drawn with the same command and the same rules, and the report counts every batch whose labels are in place.
- **Disputed items first.** In each part, the statements or answers on which a judge's saved verdict differs from the set's label, on which the set's two readers differed, or which the set calls `unclear`, as many as there are, up to 6 statements and 3 answers. Their places spread over these sorts of dispute.
- **Then the rest, in groups by the set's label, half with a fault and half without.** The faulted statements spread over the kinds of planted fault; the answers not correct hold at least one `partial` and one `incorrect`; each supported statement comes from a real answer of the same shape, one statement or several, as a faulted statement drawn before it, as far as the set allows. Within a group, a question is chosen before one of its statements. No question is drawn twice, in either part or any later batch.
- **The key waits for the labels.** The key, which says which case and statement each item is and the label the set gives it, is not committed with the batch. The batch's `batch.json` holds the seed, the plan, the files it was drawn from with their hashes, and the SHA-256 of the key. `eval-human` draws the key again, refuses it unless it has that hash, and writes it beside the labels.
- **What the code checks before a batch is written**: each item holds only the fields of what is shown; its id is its part's letter and its place; it shows exactly the words of its case, its question that of the real answer down to a space, with no text starting or ending in white space; no id of the sets is in what the owner is given, and no item's text holds a name of a kind, a group or a sort of dispute that real answers do not use; within a part whose groups have at least 3 members each, the order neither puts each group in one run nor keeps every two of a group apart; and the page's data holds nothing beyond the items, the choices and the rubrics, the page no comment, and the page, its script and the steps no word about how a batch is made up. A batch that fails is not written.
- **One page, from disk, one item at a time** (`label.html`). One item on the screen keeps the eye on the passages the statement cites, and keeps two answers from being read side by side; a strip of the ids shows what is done and jumps to any item. Keys 1 to 4 choose, N and P move. A passage is shown whole, with its title path, as its Markdown source: passages are chunks of a page, short enough to show whole, so none is cut, folded or put in a box of its own that scrolls. The page holds its style and script, and a security policy in the page itself forbids it to load or send anything. Every text from the sets and the handbook goes in as data and onto the screen as text.
- **The page keeps unsaved labels, and says when it cannot.** They stay in the browser's local storage until the owner clears them with a button that asks first; saving the file does not clear them. When storage refuses, is full, or the page is open in a second tab, the page says that labels are not kept and warns before it is closed. Labels found from before are shown with how many and when, and taken only if the owner chooses; labels of another batch or another draw are never taken.
- **Each save is a new file, and the newest is the one put in place.** The file is named by the batch and the time of the save, and holds the time and how many labels and notes it carries. `human-labels` checks it: every item once, only the choices of its part, the batch and every item's hash as drawn, a short note, the labeller as "the project's owner", the counts as stated, nothing else. It prints the time and the counts, and refuses a file beside which a newer one of the same batch lies, unless told `--older`.
- **The report sends no request to any model.** `eval-human` compares the owner's labels with the label each set held when the batch was drawn, and with the verdicts of the judges' saved checks on the labelled sets: version 1 of the judge of support, version 2 beside it, and the judge of correctness. Those are the verdicts the published agreement rests on. A verdict counts only if it was given on exactly the words drawn; an item without one is counted apart, as is "cannot tell" and the set's `unclear`. For the disputed items it says, for each judge, with whom the owner sides: the set, the judge, neither, or cannot tell. Every share comes with its count and a 95% Wilson interval, and every difference, "cannot tell" and disputed item is written in full for a later look.
- **What is reported, and what it cannot carry, was written down before any label existed**, in the card.

### The first draw, replaced before any label existed

The first draw of batch 1, on 28 November 2026, took 8 statements labelled supported and 8 unsupported, and 4 answers correct and 4 not, by the set's label alone. A review found that on every one of its 24 items each judge's saved verdict was the set's label, so the comparison with every judge would have repeated the comparison with the set, and that its part B held no answer labelled `partial`, where the judge of correctness is known to be weak. The review also found that its key had gone into the same change as the page, where the owner could see it. So batch 1 was drawn again, by the rule above, with the same seed, and its key waits for the labels. The first draw's key had been pushed, and it gives the label of each of the first draw's 24 statements and answers; they are set aside (`set-aside.json`, by hash), and no batch draws them. 6 questions of the new batch were in the first draw through another statement or the other part's answer.

### Batch 1

| Part | Disputed | Labelled with a fault | Labelled without |
|---|---|---|---|
| A, 16 statements | 6: version 1 differs from the set on 2, version 2 on 1, the readers on 1, and 2 the set calls `unclear` | 5 `unsupported`, one of each of 5 kinds of planted fault | 5 `supported` |
| B, 8 answers | 3: the judge of correctness differs from the set on each | 3: 1 `partial`, 2 `incorrect` | 2 `correct` |

On the 15 items that are not disputed every judge's saved verdict is the set's label, so the comparisons with the judges differ from the comparison with the set only on the 9 disputed items.

## Alternatives considered

- **A spreadsheet.** It needs no code. But a spreadsheet shows every row at once, and a column of labels or ids is one click away; the order and the ids would have to be kept blind by hand; a passage of 300 tokens reads badly in a cell; and what comes back would be checked by eye. The page keeps the key out of the batch and the check is code.
- **A labelling tool as a dependency.** Tools built for this exist. Each is a server, a database or an account, a new dependency to keep, and a place where the labels live outside the repository. For one person and 24 items the page does what is needed in one file.
- **Several labellers.** Two people would give an agreement between people, the measure a judge should be held to. The project has one person who can do this now. The labels file names its labeller and its pass, so a second labeller can be added later without changing what the first gave.
- **Labelling real answers only.** The real answers hold 1 unsupported statement in 204, and no answer labelled not correct in 127. A sample of them would hold almost no fault for the owner to find, and would say nothing of whether a person sees the faults the judges are checked on.
- **Drawing by the set's label alone.** That was the first draw. It compares the person with the judges only where they agree with the set, as above.
- **Disputed items only.** There are 16 in all, enough for less than one batch. They alone would compare the person with the set only where the instruments disagree, and say nothing of how far the person agrees with the labels everywhere else.
- **The judges asked again on the sample.** It would cost requests, and the verdicts would be new ones, not those the published agreement rests on.
- **The key committed with the batch.** It would be the one file that gives every label, in the change the owner is shown first. Its hash records it as well, before any label exists.

## What is still missing

- **The labels.** Batch 1 waits for the owner's labels. Until they are in place no figure of agreement with a person exists, and issue 8 stays open.
- **A second person**, and with it an agreement between people.
- **A second pass by the same person**, which would need a second order of the items.
- **Real faults.** Every fault in the sample was planted by a model, except where a judge or the readers disputed a real answer.
- **The golden set.** This sample is of the labelled answers. The golden questions and their reference answers have not been read by a person either, and part B takes the reference answers as given.

## Consequences

- Three commands: `human-sample` draws a batch and writes its page, without its key; `human-labels` checks a labels file and puts it in place; and `eval-human` draws the key again, writes it beside the labels, and writes the report to `evals/runs/judge/human/`. None sends a request to a model.
- A batch's key can be drawn again only from the same sets and saved checks. If a set or a saved check changes before the labels are in place, `eval-human` refuses, and `batch.json` holds the hash of each file it was drawn from.
- The batch files hold the full text of the passages they show, under the handbook's licence, as the card says.
- No document gives a label of a person, or a figure from one, until `labels.json` exists. A test waits for it, and once the labels are there it fails until the documents give the figures of a kept report.
