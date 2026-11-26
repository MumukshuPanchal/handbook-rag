# Labelled answers for one fault: the limit an answer leaves out

120 answers that test one decision of the judge and no other. Each cites a handbook passage that states a rule with a limit, an exception or a condition. Some answers keep the limit and some leave it out. The set exists because this is the one kind of fault the answer evaluation found in real answers and could not measure: the judge passed all three real cases, and the readers of [the first set](../README.md) split on them ([issue 59](https://github.com/MumukshuPanchal/handbook-rag/issues/59)). Making these examples is the first step of that issue. Measuring the judge on them is the second: version 1 of the judge caught 13 of the 39 limits left out, and version 2, written for this fault, caught 33 ([what the judges did on it](#what-the-judges-did-on-it)).

> **Status: written and labelled by one model session, and labelled again by three reader sessions that did not see those labels, by the account of how they were briefed, each on a third of the answers. No person has read it.** One session of `claude-fable-5-2` chose the passages, wrote every answer and gave every label, on 24 November 2026. The same day, three reader sessions labelled all 197 statements again, from the passage and the rubric alone. They agree with the first labels on 196. The statement they split on is labelled `unclear` and is not counted. All four sessions are models, and a model's check is not a person's review. Two versions of the judge have been checked on the file, [below](#what-the-judges-did-on-it).

## What is in it

[`cases.jsonl`](cases.jsonl) holds 120 answers, one on each line, in the format of [`support/cases.jsonl`](../support/cases.jsonl), so `eval-judge` reads it as it is. It has no `question_id`, because no question of the golden set stands behind these answers, and it has fields of its own, listed further down.

| | Count |
|---|---|
| Passages of the handbook | 40 |
| Answers, three for each passage | 120 |
| Statements in all | 197 |
| labelled `supported` | 157 |
| labelled `unsupported` | 39 |
| labelled `unclear`, and not counted | 1 |

Each passage has one answer of each kind:

| Kind | What the answer does | Its labels | Answers |
|---|---|---|---|
| `kept` | Gives the rule with its limit | Every statement `supported` | 40 |
| `dropped` | Gives the rule and leaves the limit out. Nothing else in it is wrong | One statement `unsupported`, the others `supported`. In one of the 40 that statement is `unclear`, because the readers split on it | 40 |
| `harmless` | Leaves out something that does not limit what it says | Every statement `supported` | 40 |

The `harmless` answers are there so that a judge which faults every omission is not rewarded. They come in two forms. In 10 the statement gives the limited rule with its limit and leaves out something beside it. In 30 it gives another rule or fact of the passage and does not speak of the limited rule at all. The second form is the sharper control: the passage states a limit, the statement does not mention it, and the statement is still supported.

| What a `harmless` answer leaves out | Answers |
|---|---|
| `example` | 4 |
| `reason` | 3 |
| `detail` | 3 |
| `other_rule` | 30 |

The limits are of seven kinds. The last column is what a check of the judge counts and prints a line for. It is one short for `when`, because the statement the readers split on is not counted.

| Kind of limit | What the passage limits the rule to | Passages | Faults a check counts |
|---|---|---|---|
| `who` | Some people: those who leave by their own choice, managers, four named teams | 5 | 5 |
| `where` | A country or a company entity | 7 | 7 |
| `when` | A period: the first year, onboarding, accounts created after a date | 5 | 4 |
| `how_much` | An amount or a count that decides whether the rule applies at all | 6 | 6 |
| `unless` | An exception: "unless", "exempt", "most, but not all" | 6 | 6 |
| `only_if` | A condition: "as long as", "provided that", "only if", "only in exceptional cases" | 7 | 7 |
| `what` | One product, plan, kind of deal or kind of feature | 4 | 4 |

The passages come from 16 of the handbook's 34 sections:

| Section | Passages |
|---|---|
| `total-rewards` | 6 |
| `finance` | 4 |
| `security` | 4 |
| `engineering` | 3 |
| `marketing` | 3 |
| `people-group` | 3 |
| `sales` | 3 |
| `company` | 2 |
| `entity` | 2 |
| `eta` | 2 |
| `solutions-architects` | 2 |
| `support` | 2 |
| `customer-experience` | 1 |
| `legal` | 1 |
| `people-policies` | 1 |
| `product-development` | 1 |

Three more things vary, because they varied in the real cases:

- **Where the passage states the limit.** In 37 passages the limit's words are in the text. In 3 they are only in the title: the name of an entity, GitLab BV, GitLab GmbH or GitLab Canada Corp. The rubric counts a title as part of the passage. The third of these is the passage behind the reply of 22 November 2026 that gave a notice rule of GitLab Canada Corp as if it held for everyone. Its `dropped` answer was written after that fault. It is not a copy of the reply, whose words are not in the repository. The text of that passage names Canadian provinces, so its country can be guessed without the title. The texts of the other two name no country.
- **Where a `kept` answer carries the limit.** In 36 the limit is in the statement. In 4 it is in the question and the statement does not repeat it ("I took three consecutive days of caregiver sick time. Do I have to provide documentation?"). The rubric reads a statement in the context of its question. For one of those four passages the `dropped` answer is the same statement, word for word, under a question without the limit.
- **How long an answer is.** An answer has one, two or three statements. In a `dropped` answer the statement that leaves out the limit is the first in 33, the second in 6 and the third in 1.

A statement names its passage by number. The passage is recorded by chunk id, with a hash of its text, and the text is fetched from the handbook snapshot when the file is loaded, as in the first set. The labels are tied to the handbook at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8` and to the chunking `v1-max320-min60-overlap40`. None of the 40 passages is cited in the first set.

### The fields of a line

| Field | Where | Meaning |
|---|---|---|
| `case` | Answer | `lim-001` to `lim-120`. The number says nothing of the kind: the answers were shuffled, and the three answers of a passage are at least five lines apart |
| `kind` | Answer | `kept`, `dropped` or `harmless`: what the answer was written to be |
| `section` | Answer | The handbook section of the passage |
| `limit_kind` | Answer | The kind of limit, as in the table above |
| `limit` | Answer | The limit in the passage's own words. The same for the three answers of a passage |
| `limit_where` | Answer | `text` or `title`: where the passage states the limit |
| `kept_in` | `kept` answers | `statement` or `question`: where the answer carries the limit |
| `left_out`, `left_out_kind` | `harmless` answers | Words of the passage that the answer leaves out, and what they are. For `other_rule` they are words of the limited rule |
| `why` | Answer | One sentence by the first reader: why its labels follow from the rubric |
| `second_session` | Answer | 1, 2 or 3: which of the three reader sessions gave this answer its second labels |
| `question`, `passages`, `statements` | Answer | What the judge reads, as in the first set |
| `expected` | Statement | The label a judge is checked against: the one both readers gave, or `unclear` where they differ |
| `readers` | Statement | Both labels: `first`, by the session that wrote the answers, and `second`, by the reader session |
| `second_confidence` | Statement | `high`, `medium` or `low`: how sure the second reader said it was |
| `planted` | The 39 statements labelled `unsupported` | The kind of limit that was left out. `eval-judge` counts what a judge caught by this field, so a check prints a line for each kind of limit |
| `change` | The statement of each `dropped` answer that leaves the limit out, all 40 | What was left out |

## How it was made

The labels follow [the rubric](../rubric.md) as it stood on 24 November 2026, above all two of its rules: limits, conditions and exceptions are part of the fact, and a narrower statement is fine. The rubric was not changed.

1. **Finding passages.** The 38,881 chunks of the snapshot were searched for the words that state a limit: "unless", "except", "only if", "as long as", "provided that", "eligible", amounts, dates, and the names of entities in titles. By the notes of the making, which are not in the repository, about 70 candidates were read in full. 40 were kept. A passage was kept when it states both the rule and the limit, when the rule can be said without the limit in a sentence that still reads well, and when the limit is not the whole point of the passage.
2. **Making sure a limit is a limit.** For each passage there had to be a case that the rule without its limit covers and the passage does not: a person who was dismissed, a free account created in 2020, an opportunity of $3,000. Where no such case could be named, the passage was left out. That is the rubric's own test: a statement "must not make the rule apply where the passage does not".
3. **One limit at a time.** Most passages hold more than one limit. Every answer keeps the others, except the `dropped` answer of `lim-008`, which leaves out two parts of one condition: "new or renewed" and the start date. Where a title limits the whole passage to an entity or a team, and that is not the limit under test, the question names the entity or team in all three answers.
4. **Writing the answers.** Each answer is in the form of the assistant's replies, with a question a person would ask. A `dropped` answer adds no word that widens, such as "any" or "always", unless the passage has it: the limit is simply absent, as in the real cases. Its question does not carry the limit either. Where the limit governs the whole passage, the `dropped` answer is a single statement, so that it is plain which statement carries the fault. In no `kept` answer does the limit come in a later statement than the rule, because the rubric does not say how to read that.
5. **Checks by code.** The limit's words and the left-out words are in the passage, word for word. No `dropped` answer holds the limit's words. No number in a statement is absent from its passage and its question. No statement that leaves out a limit holds a widening word that its passage lacks. A test repeats these four checks. A list of the words of each statement that are not in its passage was also made by code, and read by the same session.
6. **Order.** The 120 answers were shuffled with a fixed seed until the three answers of every passage were at least five lines apart.

### The second labelling: three readers, each on a third

| Step | Reader | What happened |
|---|---|---|
| 1 | First: one session of `claude-fable-5-2` | Chose the passages, wrote the 120 answers and labelled their 197 statements: 157 supported and 40 unsupported. Listed eleven of the 40 as open to dispute |
| 2 | Second: three reader sessions of `claude-fable-5-2`, the model that wrote the answers | Each labelled the 40 answers of its third, and said of every label how sure it was. Their labels and reasons are in [`second-labelling.jsonl`](second-labelling.jsonl). By the account of how they were briefed, a reader was given the question, the passage with its title, the statements and the rubric, and nothing else: no label, no kind, no note, and no word of how many answers leave a limit out. No reader's instructions were saved |

The answers were dealt by their place in the file. Of the three answers of a passage, the first went to reader 1, the second to reader 2 and the third to reader 3. So no reader saw two answers about one passage, and none could find a fault by laying answers side by side. Each reader had answers of all three kinds.

| Reader session | Answers | Statements | Statements given the first reader's label |
|---|---|---|---|
| Reader 1 | 40 | 64 | 63 |
| Reader 2 | 40 | 70 | 70 |
| Reader 3 | 40 | 63 | 63 |

What the comparison showed:

- **The two labellings agree on 196 of 197 statements** (0.995, Cohen's kappa 0.984). No reader gave a statement `no_claim`.

  | The first reader's label | Second reader: `supported` | Second reader: `unsupported` |
  |---|---|---|
  | `supported` | 157 | 0 |
  | `unsupported` | 1 | 39 |

- **The readers found 39 of the 40 statements that leave a limit out**, without being told that any did. That is the evidence that each of the 39 is a fault a reader can find from the passage and the rubric.
- **They raised no alarm on a `kept` or a `harmless` answer.** All 129 statements of those 80 answers got the first reader's label.

  | Kind of answer | Statements | Statements with the same label | Answers | Answers with the same labels throughout |
  |---|---|---|---|---|
  | In `kept` answers | 69 | 69 | 40 | 40 |
  | In `dropped` answers | 68 | 67 | 40 | 39 |
  | In `harmless` answers | 60 | 60 | 40 | 40 |

- **The readers were sure of most labels, and said where they were not.** The ten labels below `high` are listed under the known limits, with what each reader's reason turns on.

  | The second reader's confidence | Statements |
  |---|---|
  | `high` | 187 |
  | `medium` | 9 |
  | `low` | 1 |

- **The statement they split on is statement 1 of `lim-026`**: "Yes. New team members get a stipend of $1,500 USD, or the local currency equivalent, for their home office set up." Its passage says the stipend is "only available during their first year of employment". The first reader labelled it `unsupported`: the passage gives the stipend a period, and the statement gives the stipend without it. The second reader labelled it `supported`, with medium confidence: the statement "keeps 'new team members' but leaves out that the stipend is only available during the first year of employment, which I read as left-out detail because the rule is not given for team members in general". The first reader had itself listed this answer among those a careful reader could call supported. No third reader was asked. As with the disputed statements of the first set, the statement keeps both labels, is labelled `unclear`, and is left out of every agreement figure. Its answer is still given to a judge and counts through its second statement, which both readers call supported.

**What the second labels rest on.** For each answer a reader had the one passage it cites, and the rubric. It did not have the rest of the handbook, the other two answers about the passage, or the first reader's reasons. So a second label says whether a statement is backed by its passage as the rubric defines that, and nothing more. It is also one session's label: no answer was read by two of the three readers.

## How to use it

`make eval-limits` uses the file another way: it asks the assistant itself the question of each passage's `dropped` answer, with that passage, and judges the answers with version 2 of the judge ([the guide](../../../docs/evaluation.md#the-limit-questions-asked-of-the-assistant)). `make eval-judge-limits` checks the first judge on this file and writes to `evals/runs/judge/limits`. It costs 120 real requests when none fails. `make eval-judge-limits-v2` does the same for version 2 of the judge, and writes to `evals/runs/judge/limits-v2`. A check prints the faults caught in all and for each kind of limit. Alarms on `kept` and `harmless` answers are false alarms. `disagreements.jsonl` names the case of each, and the kind of a case is in this file.

Two fixed points for reading a result. The check compares 196 statements and leaves out the one whose label is unclear. A judge that passes everything agrees on 157 of the 196 (0.801) and on 81 of 120 answers, and catches none of the 39. A judge that faults every statement catches all 39, and 39 of its 196 alarms are right (0.199).

**For a further reading**, show only what a judge reads, as was done for the second labelling: `case`, `question`, `passages`, and of each statement `number`, `text` and `cites`. Everything else tells the label or the kind. The three answers of a passage can still be found by their passage, and laying them side by side gives the fault away. So a reader should be given one answer of each passage and no more.

## What the judges did on it

| Judge | Faults caught | False alarms | Kappa | Attempts | Run |
|---|---|---|---|---|---|
| Version 1 | 13 of 39 | 0 of 157 | 0.445 | 172 | [`judge/limits`](../../runs/judge/limits) |
| Version 2 | 33 of 39 | 1 of 157 | 0.882 | 236 | [`judge/limits-v2`](../../runs/judge/limits-v2) |

Version 1 gives every score in the results tables and in a run's summary. Version 2 was written for this fault, after the result of version 1 was known. By the account of how they were written, its instructions were written without these answers in view, but they share with this card its seven kinds of limit, its test of a limit (step 2 of how it was made) and its four kinds of harmless omission ([ADR 0023](../../../docs/adr/0023-a-second-version-of-the-support-judge-for-dropped-limits.md)). It caught every fault that version 1 caught. Of the 3 limits that stand only in a title, version 1 caught none and version 2 all 3. The faults that each caught by kind of limit, the six that version 2 missed and its one false alarm are in [the evaluation guide](../../../docs/evaluation.md#limits-that-an-answer-leaves-out). Both runs met an overloaded model, and their attempts count every failed one.

Version 2 is not steady on the statements the readers split on. On the first labelled answers it decides the same statement two ways in two pairs, a real answer and a copy that differs from it only in another statement's citation: `b09` and `b09-v1`, `q104` and `q104-v1`. Of the three real answers there on which the readers split, it faults one. It runs at the model's default sampling, and each check was run once.

## The assistant's own answers to these questions

`make eval-limits` asked the assistant the question of each passage's `dropped` answer, with that passage, under both versions of the answering instructions, and version 2 of the judge read the answers ([ADR 0025](../../../docs/adr/0025-hardening-against-planted-instructions.md#the-limit-questions-asked-of-the-old-and-the-new-assistant), [the runs](../../runs/limits)). The assistant's answers are not in this file and were labelled by no one: the judge's verdicts are all there is.

| Limit questions | Version 1 of the instructions | Version 2 |
|---|---|---|
| Answers with no statement faulted, of the 40 questions | 32 of 40, 0.800 (0.652 to 0.895) | 39 of 40, 0.975 (0.871 to 0.996) |

7 questions were better under version 2 and none worse, p = 0.016 by an exact two-sided sign test. Each of the eight faulted answers is faulted for a limit left out. Read against the judge's instructions by one model session, two of the faults look wrong, since the answer gives the limit (`lim-108`, `lim-093`), and version 2's answer to `lim-091` was called supported though it leaves out the limit the judge faulted under version 1; without those three verdicts it is 4 against 0, p = 0.125. Version 2's answer to `lim-026` keeps its limit in the weaker word the readers of this file split on, "new team members", and the judge called it supported.

A blind reading of the 80 answers, by one model session told neither the version nor the verdict, labelled 32 under version 1 and 37 under version 2 as keeping their limit: 5 questions better and none worse, p = 0.0625, not a clear difference. It marked `lim-026` under version 2 unsure, for the same weaker word, and `lim-084` under both versions. It agrees with the judge on 74 of the 80 answers ([the packet and the readings](../../runs/limits/reading)). The reading was decided on after the judge's figure was known, and no person has read the answers.

## Known limits

- **No person has read anything.** One model session wrote the answers and labelled them, and three model sessions labelled them again. Where models share a blind spot, their agreement proves nothing.
- **Agreement shows that these readers apply the rubric alike, not that the labels are right.** It says nothing of how hard the 39 faults are for the judge either, which is another model and has passed three real cases of this fault.
- **One second reader for each answer.** Each answer was read by one of the three reader sessions. No answer has two second labels, and no third reader settled the statement they split on.
- **A written fault is not a fault a model makes by itself.** The `dropped` answers were written by a model in order to carry a fault. A model that drops a limit while answering may do it in other ways, and may word the rest of the answer differently as well. Nor are the 40 equally hard: in some the limit sits in the sentence that states the rule, in others in a heading, a line of a list or the title. The check says how a judge does on these answers. The three real cases are not in this file. They stay labelled `unclear` in the first set, and are the one test of whether a result here carries over to real answers. The assistant's own answers to these 40 questions were judged and read once, blind, above, but not labelled: what they show rests on the judge and one reader.
- **Eleven `dropped` answers that the first reader thought a careful reader could call supported.** The second readers called ten of them unsupported and one supported. That one is the statement now labelled `unclear`. They are listed below.
- **Answers labelled `supported` that lean on their context.** The rubric reads a statement in the context of its question and of the statements before it. A reader who does not would fault the four `kept` answers whose limit is in the question only (`lim-060`, `lim-066`, `lim-078`, `lim-116`), three statements that lean on the statement before them for the limit (statement 2 of `lim-015`, `lim-030` and `lim-034`), and the answers whose question names the entity or team of the passage's title (point 3 of how it was made). `lim-050` is doubtful in another way: it gives the length of a trial from a passage that also says who may have one. The second readers called every one of these supported, `lim-066` with low confidence and the rest with high.
- **Two statements have a slip in their wording.** In statement 2 of `lim-037`, "in Proposal stage (Stage-4) and beyond" can be read with one of the two kinds of opportunity before it or with both. Statement 1 of `lim-086` shortens an example of the passage, "existing production data accessed by our users", to "existing production data". Their readers called both supported, with medium confidence. The words were left as the readers read them, because a label is about exact words.
- **Labels a review would dispute, recorded and not changed.** Changing a label now would change the test after its result was seen. Statement 1 of `lim-083` is labelled unsupported for leaving out "for First Order deals", while statement 2 of the same answer, which is the passage's own sentence, says "for new subscriptions": a weaker form of that limit, of the kind that made `lim-026` unclear. The rubric reads a statement with the statements before it and not those after, which favours the label. `lim-102` makes "Submit the request" of "can be submitted", as the readers' doubts below say. Two kinds are open to question: `lim-038` is a `harmless` answer tagged `detail` for leaving out "feature changes that update large datasets in the database", which is a second scope, and the limit of `lim-053`, "within approximately 60 minutes of the host city", is tagged `who` and is nearer `where`.
- **The kinds differ on the surface.** The statements of a `harmless` answer have 17.6 words on average, those of a `kept` answer 21.4 and those of a `dropped` answer 20.9. No `harmless` answer has three statements. A judge or a reader could lean on that.
- **Three answers share each passage.** There are 120 answers but 40 passages, and each kind of limit has 4 to 7 passages. An interval worked out as if the answers were independent is too narrow, and a figure for one kind of limit says little.
- **Not a sample of the handbook.** Sections are not weighted by their size: `engineering` holds about a third of the chunks and 3 of the 40 passages. Pages on benefits by country are over-represented, because that is where the real faults were.
- **One shape of answer.** Every answer cites one passage, and no statement is labelled `no_claim`. One of the three real cases cited two passages, of which one stated the rule without its exception. Nothing here tests that.
- **The questions were written to fit.** A `dropped` answer needs a question that does not name the limit. People often do name their country or their role, and then the same statement is supported.
- **Tied to one snapshot and one chunking.** After a change to either, the passages have to be found again and the labels checked.

### The eleven `dropped` answers the first reader listed as open to dispute

The first two columns were written before the second labelling, and the third after it.

| Case | Why a reader could differ | What the second reader said |
|---|---|---|
| `lim-004`, `lim-057`, `lim-088` | The limit is only in the passage's title. A reader who does not apply the rule that a title is part of the passage finds no limit in the text | Unsupported, all three |
| `lim-077` | The limit is in a heading and the title, not in the sentence that states the rule | Unsupported |
| `lim-014` | The exemption is a sentence of its own, and the statement repeats the sentence before it word for word | Unsupported |
| `lim-103` | The sentence it repeats has no limit of its own. The limit is in the heading and the two sentences before | Unsupported, with medium confidence |
| `lim-090` | It drops "most, but not all". The rubric's rule on hedges speaks of a hedge that a statement adds, not of one it drops | Unsupported |
| `lim-118` | The passage says that managers can submit the request. It does not say that only managers can | Unsupported |
| `lim-026` | It says "new team members", which a reader may take to stand for "during their first year of employment" | Supported, with medium confidence. The statement is now `unclear` |
| `lim-082` | "May be in cash" is not false: cash may be considered, in exceptional cases | Unsupported |
| `lim-110` | It says the additional money "is available" and does not say for which expenses | Unsupported |

### Where the second readers were unsure

Five places where the rubric did not decide a label by itself, as the readers' reasons show.

- **A limit that only the question carries, under a statement worded as a general rule.** The rubric says "Read a statement in its context", and it says that a statement which gives a rule without its limit is "wider than the passage". All three readers followed the first sentence, and their reasons say so: reader 1 in `lim-001`, `lim-003`, `lim-012`, `lim-017` and `lim-051`, reader 2 in `lim-028`, `lim-031`, `lim-060` and `lim-068`, reader 3 in `lim-055`, `lim-066`, `lim-078`, `lim-109`, `lim-110`, `lim-116` and `lim-119`. Only one voiced a doubt: reader 3 gave `lim-066` the one `low` of the set, because its statement "is backed only when read as being about the asker's account". A reader who asked for the limit inside the statement would fault every one of these.
- **An exception that the passage attaches to the sentence next to the one a statement gives.** Statement 2 of `lim-006` and of `lim-114` says that laptop wipes must be performed by scheduling an appointment with an IT analyst. The passage's exception, a wipe "at the request of IT", belongs to its sentence about wiping the laptop yourself. Both readers took it not to reach the sentence about the appointment and called the statement supported, reader 1 with high confidence and reader 3 with medium. The rubric does not say how far an exception reaches.
- **A statement that can itself be read in two ways.** Statement 2 of `lim-037`, as said above. The rubric's rule on two readings speaks of a passage, not of a statement. Reader 2 took the reading that matches the passage.
- **An instruction made from a "can" sentence.** The passage of `lim-102` says that requests "can be submitted" with a template, and the statement says "Submit the request using the `application_integration_request` template." The rubric says that permission is not obligation. Reader 2 read the instruction as the answer to a how-to question, "not as an added obligation".
- **A limit kept in a weaker word.** "New team members" for "only available during their first year of employment", in `lim-026`. This is the statement the readers split on.

The ten labels that came with less than high confidence:

| Case | Statement | Confidence | Second label | What the reader's reason turns on |
|---|---|---|---|---|
| `lim-026` | 1 | medium | supported | "New team members" read as carrying the first-year limit. The readers split here |
| `lim-037` | 2 | medium | supported | "In Proposal stage (Stage-4) and beyond" read as covering both kinds of opportunity before it |
| `lim-040` | 1 | medium | supported | The statement gives when a claim can be made, and no other condition, "without saying that every customer is eligible" |
| `lim-066` | 1 | low | supported | The limit is in the question only |
| `lim-083` | 1 | medium | unsupported | It drops "for First Order deals". The reason states the fault and no doubt |
| `lim-086` | 1 | medium | supported | A shortened example, read "as left-out detail of an example and not as a limit on the rule" |
| `lim-091` | 1 | medium | unsupported | It drops "This page only applies to Zendesk Global". The reason states the fault and no doubt |
| `lim-102` | 1 | medium | supported | An instruction made from "can be submitted" |
| `lim-103` | 1 | medium | unsupported | The limit is in the title and the opening sentences. The reason states the fault and no doubt |
| `lim-114` | 2 | medium | supported | The exception belongs to the other sentence, about wiping the laptop yourself |

### Where the rubric's rule on limits leaves room

Found by the first reader while the set was made. The pages named in brackets were left out because of these questions. No label rests on them, except where a case is named.

- **A limit that comes in a later statement.** The rubric reads a statement in the context of the question and of the statements before it. It does not say how to read an answer whose first statement gives a rule and whose second gives the exemption from it. No answer here is built that way.
- **A requirement in the next sentence.** "You can take breaks during your shift. You need to ensure that you are able to acknowledge a page within 15 minutes and be back at your desk shortly thereafter." Is the second sentence a condition on the first, or a second rule? (On-Call Processes and Policies.)
- **An exception that is not marked as one.** "… with all Analytics Engineers participating in the rotation. New team members are added to the rotation after one to two quarters on the team". And: "a service must meet all the criteria for that level and all previous levels. Some criteria do not apply to all services". (Analytics Engineering Triage Guide; Service Maturity Model.)
- **Some conditions out of several, after "include".** "Selection criteria include at least 6 months' tenure at GitLab, active membership in the TMxG, manager approval, and a minimum one-year commitment." A statement that names two of the four is narrower, which the rubric allows, and it leaves out conditions, which the rubric forbids. (TMxG leadership.)
- **A cap on an amount.** "Up to €140 reimbursement". The rule lists people, places, plans, periods and situations, not amounts. No answer here leaves out a cap: amounts are the limit under test only where they decide whether a rule applies at all.
- **A limit kept in a weaker word.** "Large events" for "with a total cost greater than $1M"; "by default" for "unless there is a specific reason not to". `lim-026` is of this kind.
- **A hedge of the passage that the statement drops.** "Most, but not all", "typically", "generally". `lim-090` and `lim-082` rest on this.
- **A limit on how to use something, not on whether it applies.** "Fast quarantine applies to all stable branches also but should only be used to unblock pipelines until a long term quarantine is merged". (Test Quarantine Process.)
- **Discretion.** "Qualifying educational institutions may receive an Academic Discount", and then "The discount will be applied at the discretion of the account owner". Is discretion a condition? (Public Sector, Academic Discount.)
- **A rule stated twice, once without its exception.** "GitLab Sales must work with incumbent channel partners on renewals unless the customer documents in writing that they no longer want to work with the partner. GitLab cannot offer a direct renewal quote if there is an incumbent partner." (Ecosystem Operations, Incumbency Renewals.)

## Licence and attribution

The questions, statements and notes were written by a model for this project. They are not replies of the assistant. Some repeat words of the [GitLab Handbook](https://handbook.gitlab.com/), © GitLab and its contributors, used under the [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) licence, and the fields `limit` and `left_out` quote it a phrase or a sentence at a time. The file holds no whole passage: passages are named by id and read from the snapshot.

The files in this directory are licensed under CC BY-SA 4.0, separately from the code, which is under the MIT licence. This project is not affiliated with or endorsed by GitLab. The basis for using the handbook is recorded in [ADR 0002](../../../docs/adr/0002-handbook-corpus.md).
