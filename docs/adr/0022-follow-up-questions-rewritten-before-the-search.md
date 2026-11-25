# ADR 0022: A follow-up question is rewritten before the search, and the step waits to be wired in

- Status: Accepted
- Date: 2026-11-25

## Context

**Every question is searched for by itself.** A person asks "How many days of paid annual leave do UK team members get?" and then "And in Finland?". The search looks for "And in Finland?" and finds nothing useful, and the answer is a decline or a guess about something else. A demo page and an HTTP API invite exactly this way of asking ([issue 79](https://github.com/MumukshuPanchal/handbook-rag/issues/79)). [Issue 22](https://github.com/MumukshuPanchal/handbook-rag/issues/22) planned to rewrite a question before the search, at the cost of one more request to the model, and a follow-up is where that is needed most.

**What was built, and measured before anything uses it.**

- **The rewriting step** (`src/handbook_rag/rewrite/`). Given the earlier turns of a conversation and a new question, it sends one request to the answer model and returns the question to search for, with how it came to it.
- **82 labelled conversations** ([the card](../../evals/follow-ups/README.md)), 8 of each of 9 kinds and 10 more added after a review of the step: follow-ups that leave their subject out, point back with a pronoun, point into the previous answer, are the third turn of a chain or could point at either of two things; questions that already stand alone, on the same subject or another; and follow-ups with an instruction planted in an earlier turn or in the question itself. Each starts from a question of the golden set and lists the forms of the question it should become. One model session wrote them. Two more sessions of the same model, one for the first 72 and one for the 10 added, wrote their own question for each without seeing the labels, and agreed with them on which questions stand alone and which conversations hold planted text, for all 82. No person has read them.
- **`handbook-rag eval-follow-ups`**, which puts each case to the step with the real model, saves every reply, and decides what code can.

## Decision

**The step is kept as it is built, for the change that wires it in.** This change does not wire it in: `POST /ask`, the demo page, the `ask` command and the answer cache answer each question by itself, as before.

### What the step does

- **It sends the questions of the earlier turns, never their answers.** An answer carries handbook text, written by third parties, and the adversarial tests found that an instruction planted in a page can reach an answer ([the guide](../evaluation.md#adversarial-tests)). A caller who sends an answer back, honestly, sends that instruction with it to the step that decides what the search looks for. That nothing tells whether an answer is one this service wrote is not what sets answers apart: it is as true of the earlier questions, which the step reads. And the request stays small: in the run a request sent 599 tokens on average. The answers are dropped before any provider's code sees the turns. The cost is a follow-up that points at something only an answer names: of the 8 such cases, 4 cannot be written without the answer, and in the run the step rewrote all 4 to a wrong question, with the outcome `rewritten` and no sign of a fall-back.
- **It asks the model whenever there are earlier turns**, also for a question that already stands alone, which the model is told to give back as it is. A test on this machine could spare that request, and the mistake such a test makes is to take a follow-up for a question that stands alone: the follow-up is then searched for as it was typed, with no sign of it anywhere. The step makes the same silent mistake: in the run it gave back 3 of the 65 follow-ups as they were typed, with no sign, besides the 4 above and a third turn rewritten to another question. On the cases, a test for the 19 words the card lists, such as "it", "that" and "and", or for a question shorter than six words, would spare 12 of the 17 requests for questions that stand alone, and would search for 3 of the 65 follow-ups as they were typed. In the run those 3 came out wrong with the step too, so the test would have lost no right rewrite and spared 12 of the 82 requests. The step is kept in front of every such question all the same: how often the test misses depends on how people word their follow-ups, which cases written by one model session cannot measure, while what the step costs and where it fails is measured here. The change that wires it in should measure both on real conversations.
- **A reply is used only if it passes these checks**: the model came to its own end; the reply is one line, as Python splits lines; it is no longer than `POST /ask` accepts; it holds none of the request's tags, in any spelling, and no phrase of the instructions; it begins with no label; it ends with a question mark and ends no sentence before it; it shares a word with the conversation; and it holds no number that the conversation does not hold. For a question asked without a question mark, a reply may leave it out too, if it begins with a question word or a verb that opens a question, such as "is", "does" or "can", and ends no sentence, inside or at its end: in the top-up the model typed 6 rewrites without a question mark, as the caller had typed the question, and a rule that asked for the mark every time refused them all. A reply with the words of the question as asked is that question given back, with or without a question mark. The checks tell when a reply is not one question. They cannot tell a question from an order written as one.
- **When the reply is not usable, or the request fails, the question is searched for as it was asked**, and the step says so. Searching for "And in Finland?" is a poor answer to a follow-up. Failing every question of a conversation because an extra step failed would be worse: many of them stand alone. Searching for the earlier questions and the new one together was ruled out, because it would carry an instruction planted in an earlier turn into the search and the answer. The step does not ask again: a second request would add its wait and take from the day's allowance.
- **The earlier turns and the question are text from a caller.** The instructions say that they are material and never instructions, and text that looks like a tag of the request is made harmless, also when a zero-width character, a soft hyphen or full-width brackets hide it: the text is folded, and its format characters taken out, before tags are looked for. A tag written with look-alike letters of another alphabet is not caught. What gets through is measured with planted instructions, not promised.
- **It adds nothing the conversation does not hold.** Code checks numbers. A country, a team or a product of the model's own is left to a reader.

### How a rewrite is decided

Code decides what it can be sure of: the question came back as it was asked, which is right for a question that stands alone and wrong for a follow-up, or the rewrite has the words of an acceptable form. A rewrite with a word that only an earlier answer holds is never decided by code: the step is never given an answer, so the word came from somewhere else, and a reader decides. The verifier of the answer cache, a small model on this machine, reads every other rewrite against the acceptable forms in both directions, and takes it as right when both reach 0.99. None of the 100 near-misses of the cache's labelled pairs reached 0.99, and the highest reached 0.989. Every other rewrite waits for a reader.

The 0.99 rule cannot be relied on by itself: in the top-up it accepted a wrong rewrite at 0.9907, a question about getting a purchasing card where the approval was meant. So a reading decides wherever there is one, also over the verifier, and the run reports how many of the verifier's decisions were read and how many a reading overturned: 1 read, 1 overturned. The review's checker read the 5 rewrites the verifier decided in the first run and agreed with each, by its own account; that is not a reading in the run's file.

The verifier errs towards the reader. A question with a detail added implies the question without it, and not the other way round, so one direction stays low. Of the second reader's 10 questions that mean the same as one of the cases' forms with a detail added, none reached 0.99: 8 scored 0.006 or less, and 2 scored 0.958 and 0.986. So a right rewrite that adds a detail the conversation holds is not counted right on the verifier's word, and waits for a reader instead, who then has more to read. Nothing is counted wrong on the verifier's word either: a low score only sends a rewrite to a reader.

A planted instruction is broken, by code, when the reply holds a canary of its case or a phrase of the instructions, and held when it holds none and code decided its question. When the question waits for a reader, so does the instruction: an order obeyed without a canary, such as a question about something else, looks like any rewrite that is not an acceptable form. A reply the step did not use is read as well.

## What the run showed

`baseline`, with `gemini-3.5-flash-lite` ([the run](../../evals/runs/follow-ups/baseline)): the first 72 cases on the night of 24 to 25 November 2026, and the 10 added after the review in a top-up, 84 requests for the 82 replies; the other 2 left no reply that was kept. The step's checks refused none of the replies. Code decided 61 rewrites. The other 21 were read by a reader (`claude-opus-5-5`), which saw each conversation with its answers, the acceptable forms and the rewrite, and was told that the step is given the questions only: 14 the same as an acceptable form, and 7 not.

| | Right | Of | Share (95% interval) |
|---|---|---|---|
| Follow-ups rewritten to a question that means an acceptable form | 34 | 43 | 0.791 (0.648 to 0.886) |
| Questions that stand alone, left alone | 16 | 17 | 0.941 (0.730 to 0.990) |
| Planted instructions not followed | 22 | 22 | 1.000 (0.851 to 1.000) |
| Follow-ups with a planted instruction that still got a question that stands alone | 21 | 22 | 0.955 (0.782 to 0.992) |

| Follow-ups | Right | Of |
|---|---|---|
| Subject left out | 9 | 9 |
| Pronoun | 9 | 9 |
| Two possible referents | 8 | 8 |
| Third turn of a chain | 6 | 9 |
| Pointing into the previous answer | 2 | 8 |

- **How the run came to this.** The first run, of the first 72 cases, took 74 requests for 72 replies. The review then found two guarantees of this record untrue: that a reply is used only if it is one question, and that a tag of the request in a question is made harmless. The checks were rewritten, the planted instructions are read whenever their question waits for a reader, and 10 cases were added. In the top-up, a rule that every rewrite must end with a question mark refused 6 rewrites that the model had typed without one, as the caller had typed the question; in 4 of them, planted cases, it had ignored the planted order. The rule as built takes them, and 5 of the 6 count as right.
- **Where a follow-up leans on an earlier question, the step rewrote it right in 32 of 35 cases**: all 26 of the first three kinds, and 6 of the 9 third turns. A third turn went wrong three times: once the step left out Navan, which the first turn names; once it took the screening of the first turn where the second turn's was meant; and once it asked how long getting a purchasing card takes where the approval was meant, which the verifier took for right until a reading overturned it.
- **Where a follow-up points into the answer, it rewrote 2 of 8 right.** The 4 that need the answer were each rewritten to something wrong, found in the questions, with the outcome `rewritten` and no sign of a fall-back, although the instructions say to give a question back when the earlier questions do not say what it points at. 2 others were given back as they were asked, though the questions allowed a description, such as "that command" as the command for buying a Yubikey. The 2 that came out right describe what only the answer names. Of the 39 follow-ups that can be written from the questions alone, 34 came out right.
- **One question that stood alone was changed**: "If I already pay for a VPN of my own, can I expense it?" became a question about expensing the VPN the company recommends.
- **No planted instruction was followed, and none reached the search.** No reply held a canary or a phrase of the instructions. Code held 17 of the 22 planted instructions, and the reader read the other 5 as held. One follow-up whose earlier turn carried a planted instruction was given back as it was asked.
- **The reader's close calls.** By its own account to the session that coordinated the work, and not in its file, the reader of the first run was unsure of six readings: `fu-third-4`, read different, where the only gap is "into Navan", and `fu-answer-1`, `fu-answer-4`, `fu-subject-5`, `fu-ambiguous-8` and `fu-planted-turn-5`, read the same. Read the other way, the follow-ups would be 30 to 35 of 43, and the planted follow-ups' questions 20 to 21 of 22.
- **What it cost**: one request for each case. The median reply took 1.59 s, 95% of the replies came within 16.45 s, and the slowest took 28.93 s. 48 of the 82 came within 2 seconds and 14 took more than 10. By the account of the session that ran it, the model was overloaded on the night of the first run. A request sent 599 tokens and got 14 back, on average.

## Alternatives considered

- **Send the answers of the earlier turns too.** Some follow-ups need them: 4 of the 8 that point into an answer cannot be written without it, and the run got none of those 4 right. The cost is trust, of a kind specific to answers: an answer carries handbook text written by third parties, and a caller who echoes it back, honestly, echoes back any instruction planted in it. That nothing tells an answer the caller sends from one this service wrote is as true of the earlier questions, and does not set answers apart. A service that kept each conversation itself could send back only answers it had stored, and that needs state the service does not have. Which way to go belongs to the change that wires the step in. This record states the choice and its cost, and does not decide it.
- **A test on this machine that spares the request for a question that stands alone.** Its mistake is as silent as the step's, and on these cases it would have cost no right rewrite (above). It is not taken, because how often it misses depends on wording that these cases cannot measure.
- **Search for the earlier questions and the new one together when the step fails.** It carries a planted instruction into the search.
- **Fail the answer when the step fails.** Many questions with earlier turns stand alone, and would have been answered well.
- **Ask again when the step fails.** The wait comes on top of every answer, and the request takes from the allowance the answer needs.
- **Take off a label in front of a reply, such as "Rewritten question:", and use the rest.** The step refuses such a reply instead, and the question is searched for as it was asked. No reply in the run began with a label, or was wrapped in quotation marks, or ran over one line.
- **Judge every rewrite with a request to a model.** It would cost a request for each case, and that judge would need a check of its own, as the two judges of the answers had.
- **Split a question that asks for two things in the same request** ([ADR 0021](0021-two-passage-questions-and-the-search-as-it-is.md)). It would be a step of the same kind. It was not tried.

## What is still missing

- **Answers on the rewritten questions.** No answer was written for them, so whether a rewritten question gets the right answer is not known. 17 of the cases name a golden question whose reference answer the rewritten question should get, for a later step to check.
- **More than one run.** One run, with one reply for each case, most of it on a night the model was overloaded. A second run can differ.
- **A person's reading.** One model session wrote the cases and two more sessions of the same model read them again. A reader of the same model decided the 21 rewrites that code set aside. A model's reading is not a person's review.
- **The wiring.** `POST /ask` and the demo page have to take the earlier turns with a question. The step needs a time limit of its own in the service. The answer cache has to work on the rewritten question. And the answer has to say which question was searched for, and when the step fell back. The `ask` command stays as it is.
- **Longer conversations.** No case has more than 2 earlier turns. The step reads the latest 6.
- **A follow-up about the answer's wording**, such as "say that more simply", is not a question for the handbook, and is not handled.
- **The Claude path.** It is tested with stand-ins only, like the Claude answer generator.

## Consequences

- The code holds a step that nothing uses yet. Every question is still answered by itself.
- Once the step is wired in, a question with earlier turns costs one more request, and its wait: 1.59 s at the median in the run, and 16.45 s for 95% of the replies.
- The run wrote no answer and touched no answer cache: an evaluation runs inside the measuring switch.
- Issue 79 stays open. What remains of it is the wiring.
