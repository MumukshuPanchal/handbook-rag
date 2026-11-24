# Labelled answers for one fault: the limit an answer leaves out

120 answers that test one decision of the judge and no other. Each cites a handbook passage that states a rule with a limit, an exception or a condition. Some answers keep the limit and some leave it out. The set exists because this is the one kind of fault the answer evaluation found in real answers and could not measure: the judge passed all three real cases, and the readers of [the first set](../README.md) split on them ([issue 59](https://github.com/MumukshuPanchal/handbook-rag/issues/59)). Making these examples is the first step of that issue. Measuring the judge on them is the second, and has not been done.

> **Status: written and labelled by one model session, and read by no one else.** One session of `claude-fable-5-2` chose the passages, wrote every answer and gave every label, on 24 November 2026. A second labelling, by a reader who is not shown these labels, is planned and has not been made. No person has read any of it. The judge has not been run on it.

## What is in it

[`cases.jsonl`](cases.jsonl) holds 120 answers, one on each line, in the format of [`support/cases.jsonl`](../support/cases.jsonl), so `eval-judge` reads it as it is. It has no `question_id`, because no question of the golden set stands behind these answers, and it has fields of its own, listed further down.

| | Count |
|---|---|
| Passages of the handbook | 40 |
| Answers, three for each passage | 120 |
| Statements in all | 197 |
| labelled `supported` | 157 |
| labelled `unsupported` | 40 |

Each passage has one answer of each kind:

| Kind | What the answer does | Its labels | Answers |
|---|---|---|---|
| `kept` | Gives the rule with its limit | Every statement `supported` | 40 |
| `dropped` | Gives the rule and leaves the limit out. Nothing else in it is wrong | One statement `unsupported`, the others `supported` | 40 |
| `harmless` | Leaves out something that does not limit what it says | Every statement `supported` | 40 |

The `harmless` answers are there so that a judge which faults every omission is not rewarded. They come in two forms. In 10 the statement gives the limited rule with its limit and leaves out something beside it. In 30 it gives another rule or fact of the passage and does not speak of the limited rule at all. The second form is the sharper control: the passage states a limit, the statement does not mention it, and the statement is still supported.

| What a `harmless` answer leaves out | Answers |
|---|---|
| `example` | 4 |
| `reason` | 3 |
| `detail` | 3 |
| `other_rule` | 30 |

The limits are of seven kinds:

| Kind of limit | What the passage limits the rule to | Passages |
|---|---|---|
| `who` | Some people: those who leave by their own choice, managers, four named teams | 5 |
| `where` | A country or a company entity | 7 |
| `when` | A period: the first year, onboarding, accounts created after a date | 5 |
| `how_much` | An amount or a count that decides whether the rule applies at all | 6 |
| `unless` | An exception: "unless", "exempt", "most, but not all" | 6 |
| `only_if` | A condition: "as long as", "provided that", "only if", "only in exceptional cases" | 7 |
| `what` | One product, plan, kind of deal or kind of feature | 4 |

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
| `kind` | Answer | `kept`, `dropped` or `harmless` |
| `section` | Answer | The handbook section of the passage |
| `limit_kind` | Answer | The kind of limit, as in the table above |
| `limit` | Answer | The limit in the passage's own words. The same for the three answers of a passage |
| `limit_where` | Answer | `text` or `title`: where the passage states the limit |
| `kept_in` | `kept` answers | `statement` or `question`: where the answer carries the limit |
| `left_out`, `left_out_kind` | `harmless` answers | Words of the passage that the answer leaves out, and what they are. For `other_rule` they are words of the limited rule |
| `why` | Answer | One sentence: why the labels follow from the rubric |
| `question`, `passages`, `statements` | Answer | What the judge reads, as in the first set |
| `expected`, `readers` | Statement | The label, and who gave it: `first` alone so far |
| `planted`, `change` | The `unsupported` statement of a `dropped` answer | The kind of limit that was left out, and what was left out. `eval-judge` counts what a judge caught by `planted`, so a check prints a line for each kind of limit |

## How it was made

The labels follow [the rubric](../rubric.md) as it stood on 24 November 2026, above all two of its rules: limits, conditions and exceptions are part of the fact, and a narrower statement is fine. The rubric was not changed.

1. **Finding passages.** The 38,881 chunks of the snapshot were searched for the words that state a limit: "unless", "except", "only if", "as long as", "provided that", "eligible", amounts, dates, and the names of entities in titles. About 70 candidates were read in full and 40 were kept. A passage was kept when it states both the rule and the limit, when the rule can be said without the limit in a sentence that still reads well, and when the limit is not the whole point of the passage.
2. **Making sure a limit is a limit.** For each passage there had to be a case that the rule without its limit covers and the passage does not: a person who was dismissed, a free account created in 2020, an opportunity of $3,000. Where no such case could be named, the passage was left out. That is the rubric's own test: a statement "must not make the rule apply where the passage does not".
3. **One limit at a time.** Most passages hold more than one limit. Every answer keeps the others. Where a title limits the whole passage to an entity or a team, and that is not the limit under test, the question names the entity or team in all three answers.
4. **Writing the answers.** Each answer is in the form of the assistant's replies, with a question a person would ask. A `dropped` answer adds no word that widens, such as "any" or "always", unless the passage has it: the limit is simply absent, as in the real cases. Its question does not carry the limit either. Where the limit governs the whole passage, the `dropped` answer is a single statement, so that it is plain which statement carries the fault. In no `kept` answer does the limit come in a later statement than the rule, because the rubric does not say how to read that.
5. **Checks by code.** The limit's words and the left-out words are in the passage, word for word. No `dropped` answer holds the limit's words. No number in a statement is absent from its passage and its question. No statement that leaves out a limit holds a widening word that its passage lacks. A test repeats these four checks. A list of the words of each statement that are not in its passage was also made by code, and read by the same session.
6. **Order.** The 120 answers were shuffled with a fixed seed until the three answers of every passage were at least five lines apart.

## How to use it

`make eval-judge-limits` checks the first judge on this file and writes to `evals/runs/judge/limits`. It costs 120 real requests when none fails. It prints the faults caught in all and for each kind of limit. Alarms on `kept` and `harmless` answers are false alarms. `disagreements.jsonl` names the case of each, and the kind of a case is in this file.

Two fixed points for reading a result. A judge that passes everything agrees on 157 of 197 statements (0.797) and on 80 of 120 answers, and catches none of the 40. A judge that faults every statement catches all 40, and 40 of its 197 alarms are right (0.203).

**For a second reader**, show only what a judge reads: `case`, `question`, `passages`, and of each statement `number`, `text` and `cites`. Everything else tells the label or the kind. The three answers of a passage can still be found by their passage, and laying them side by side gives the fault away. So a reader should be told to judge each answer by itself, or be given the answers in three parts with one answer of each passage in each.

## Known limits

- **One model session made all of it.** The same session chose the passages, wrote the answers and labelled them. Its labels cannot disagree with what it meant to write, only with the rubric, and nobody has checked that yet.
- **A written fault is not a fault a model makes by itself.** The `dropped` answers were written by a model in order to carry a fault. A model that drops a limit while answering may do it in other ways, and may word the rest of the answer differently as well. Nor are the 40 equally hard: in some the limit sits in the sentence that states the rule, in others in a heading, a line of a list or the title. The check says how a judge does on these 40. The three real cases are not in this file. They stay labelled `unclear` in the first set, and are the one test of whether a result here carries over to real answers.
- **Eleven `dropped` answers that a careful reader could call supported.** They are labelled `unsupported` because the rubric's rule on limits, as written, covers them. They are listed below with the reason a reader might differ.
- **Answers labelled `supported` that lean on their context.** The rubric reads a statement in the context of its question and of the statements before it. A reader who does not would fault the four `kept` answers whose limit is in the question only (`lim-060`, `lim-066`, `lim-078`, `lim-116`), three statements that lean on the statement before them for the limit (statement 2 of `lim-015`, `lim-030` and `lim-034`), and the answers whose question names the entity or team of the passage's title (point 3 of how it was made). `lim-050` is doubtful in another way: it gives the length of a trial from a passage that also says who may have one.
- **The kinds differ on the surface.** The statements of a `harmless` answer have 17.6 words on average, those of a `kept` answer 21.4 and those of a `dropped` answer 20.9. No `harmless` answer has three statements. A judge or a reader could lean on that.
- **Three answers share each passage.** There are 120 answers but 40 passages, and each kind of limit has 4 to 7 passages. An interval worked out as if the answers were independent is too narrow, and a figure for one kind of limit says little.
- **Not a sample of the handbook.** Sections are not weighted by their size: `engineering` holds about a third of the chunks and 3 of the 40 passages. Pages on benefits by country are over-represented, because that is where the real faults were.
- **One shape of answer.** Every answer cites one passage, and no statement is labelled `no_claim`. One of the three real cases cited two passages, of which one stated the rule without its exception. Nothing here tests that.
- **The questions were written to fit.** A `dropped` answer needs a question that does not name the limit. People often do name their country or their role, and then the same statement is supported.
- **Tied to one snapshot and one chunking.** After a change to either, the passages have to be found again and the labels checked.

### The eleven `dropped` answers a careful reader could call supported

| Case | Why a reader could differ |
|---|---|
| `lim-004`, `lim-057`, `lim-088` | The limit is only in the passage's title. A reader who does not apply the rule that a title is part of the passage finds no limit in the text |
| `lim-077` | The limit is in a heading and the title, not in the sentence that states the rule |
| `lim-014` | The exemption is a sentence of its own, and the statement repeats the sentence before it word for word |
| `lim-103` | The sentence it repeats has no limit of its own. The limit is in the heading and the two sentences before |
| `lim-090` | It drops "most, but not all". The rubric's rule on hedges speaks of a hedge that a statement adds, not of one it drops |
| `lim-118` | The passage says that managers can submit the request. It does not say that only managers can |
| `lim-026` | It says "new team members", which a reader may take to stand for "during their first year of employment" |
| `lim-082` | "May be in cash" is not false: cash may be considered, in exceptional cases |
| `lim-110` | It says the additional money "is available" and does not say for which expenses |

### Where the rubric's rule on limits leaves room

Found while the set was made. The pages named in brackets were left out because of these questions. No label rests on them, except where a case is named.

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
