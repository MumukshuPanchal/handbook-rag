# ADR 0024: Follow-up questions in the service: the caller sends the earlier questions, and the reply names the question searched for

- Status: Accepted
- Date: 2026-11-25

## Context

[ADR 0022](0022-follow-up-questions-rewritten-before-the-search.md) built the step that rewrites a follow-up question into one that stands alone, and measured it on 82 conversations: 35 of 43 follow-ups were rewritten to a question that means an acceptable form, 16 of 17 questions that stand alone were left alone, and none of the 22 planted instructions was followed. It wired the step into nothing. This record is the second and last step of [issue 79](https://github.com/MumukshuPanchal/handbook-rag/issues/79): a person can hold a conversation with the service.

**Held-out cases came first.** The step's rule for a rewrite without a question mark had been written after seeing the six replies it decides. Before the step was wired in, 20 more conversations were written, by a session that had seen none of the step's replies, read a second time, and asked once ([the card](../../evals/follow-ups/README.md#the-held-out-cases)). Under the rule as it stood before them, 9 of the 14 follow-ups came out right, 0.643 (0.388 to 0.837), against 0.814 (0.674 to 0.903) in the first run. The intervals overlap, and 14 cases can show little. Of the 5 misses, 2 were good rewrites that the rule refused, because the model had kept the person's "and" or "but" in front, and 3 were the model's own mistakes, which no check of a reply can mend.

## Decision

**A question sent with the earlier questions of its conversation is rewritten first, and everything after that is done with the question that came out.** `POST /ask`, the demo page and the `ask` command take the earlier questions; the step runs before the search; the lookup in the answer cache, the search and the answer take the rewritten question; and every reply says which question was searched for.

### The service keeps no conversation

The caller holds the conversation and sends the earlier questions with each question: `earlier` in the request, oldest first, and `--earlier` on the command line, once for each. The service stores no conversation, needs no id for one, and treats two requests alike whoever sends them. Up to 10 earlier questions are accepted, and the latest 3 are read (`HANDBOOK_RAG_FOLLOW_UP_EARLIER_QUESTIONS`): the oldest are dropped. The limit on the body of a request was raised to 32 KB, so that a question and 10 earlier questions at their longest fit in it, written in characters of four bytes, as JSON sent in unescaped UTF-8. A client that escapes characters outside ASCII, as Python's own `json` does by default, writes six bytes for each such character and twelve for one outside the Basic Multilingual Plane: with long questions in other scripts it can pass the limit, and gets a 413.

### Questions only, never answers

The rewriting model is given the earlier questions and the new one. An answer quotes the handbook, and handbook text is untrusted: the adversarial tests found that an instruction planted in a page can reach an answer ([the guide](../evaluation.md#adversarial-tests)). So the service never passes an answer to the step, and refuses one sent as an answer. Whatever is sent as an earlier question is treated as a question: an answer's text sent as one reaches the step's prompt as a question would, and comes from the caller of the same request (below). The demo page sends only the questions searched for:

- **Over HTTP an answer sent as an answer is refused, not ignored.** `earlier` is a list of strings. An item that is an object, such as one with a question and an answer, a list, a number or a blank string is refused with a 422, and so is any field the request does not define, such as `answer`, `answers` or `history`. A refusal names the field and does not repeat what was sent. Refused, a caller who meant to send answers learns that they were not used; ignored, the caller would believe they were.
- **The command line** takes `--earlier` strings, and **the entry point** (`answer_question`) takes a sequence of strings, from which it makes turns that hold no answer, and refuses a single string in its place. The step drops any answer a turn holds besides, as it always did.

Tests hold all three. **The known cost, said once:** a follow-up that points at something only an answer named is usually rewritten to a wrong question, with no sign of a fall-back. In the run, 2 of the 8 follow-ups that point into an answer came out right. The reply showing the question searched for is what makes that visible to the person.

### Why the reply checks are not the safety boundary

Every piece of text that reaches the rewriting model comes from the caller of the same request: the question and the earlier questions. An order planted there gets the caller nothing they could not get by asking for it directly, and the reply shows them what was searched for. The checks of a reply (`read_reply`) are there to keep the search from running on an answer or on junk, not to stop an instruction: a question can be an order written as a question, and the checks cannot tell. The evidence about planted instructions is the measurement, 22 of 22 not followed in the first run and 3 of 3 on the held-out cases, not the checks.

### The earlier questions are best sent as they were searched for

Each reply carries the question that was searched for. The demo page keeps it for each turn and sends those as the earlier questions, so that a third follow-up builds on questions that already stand alone. The API takes whatever the caller sends. The measurement sent the earlier questions as they were typed, which is the harder case for the step. Conversations longer than two earlier questions were never measured.

### One place in the code

`answer_question` in `pipeline.py` stays the single entry point. Given earlier questions and the step (`FollowUps`), it runs the step inside the answer's trace; the lookup, the search and the answer model then get the standalone question exactly as if it had been typed. Without earlier questions, or without the step, nothing is asked and the code takes the path it took before: a test runs every path, without a cache, with one that misses or holds the answer, and for a fresh answer, and finds no request to the rewriting model on any of them. An earlier question that holds no letter or digit, such as one of spaces, a zero-width space or a byte order mark, is not one, and costs no request.

### A reply says what was searched for

Every answer has `searched_for`: the question as it was `asked`, the `question` searched for, the `outcome` (`not_asked`, `unchanged`, `rewritten`, `unusable` or `failed`), one fixed word for a failure, how many earlier questions were read, and what the step cost (`requests`, `usage`, `took_ms`). The answer's own `requests` count the answer model's requests only, as before. The reason a reply was not usable is not sent: it can quote the model's reply, and the outcome says enough. The error behind a failure goes to the log, in the API's way: a warning with the failure's word, the trace id and the error's own words, and no question, so that a step set up wrong, with a key the provider refuses, is seen there and not only as one word in each reply. The command puts the same words on its error output.

- **In a stream**, one `event: searched` with the same object comes first, before the first `event: text`. It is sent once the first text, or the whole answer, is there, and not sooner: until then a failure is still an HTTP error with a status, as [ADR 0015](0015-answers-over-http.md) has it. The question searched for is known before the search, so it is there as early as that rule allows, while the answer is written.
- **The `ask` command** prints "Searched for: …" before the answer, or says that the follow-up was searched for as it was typed and why, and adds the step's request, tokens and time to the line of details.
- **The demo page** shows "Searched for: …" above the answer whenever it is not the question typed, and says so when a follow-up could not be rewritten, and when the service did not run the step at all for a follow-up it was sent with earlier questions. A test runs the page's script in node, with a fake page and a stubbed `fetch`, through one conversation. Everything from the API goes on the page as text, and nothing is kept in the browser: the conversation is a variable of the page's script, and a reload or "Start a new conversation" ends it. A test fails if the script names any of the browser's stores.
- **Until now the service never sent a question back.** A reply now names the caller's own question, and the one it was rewritten to, so that the caller can send it with the next question. A stream that fails after its first text has named it before the failure.

### When the step fails, the question is searched for as typed, and the reply says so

One attempt, with a time limit of its own (`HANDBOOK_RAG_FOLLOW_UP_TIMEOUT_SECONDS`, 15 seconds by default), no retry and no fallback model. The limit is the client's, and applies to each phase of the request, connecting, sending and each read of the reply, not to the whole wait. The reply is not streamed: the client asks for it whole, and the limit bounds each wait for more of it, the first of which lasts while the model writes, and not the sum of the waits. No retry: a retry costs another of the free key's 15 requests a minute, which the answer needs. The default comes from the saved runs' waits. In the first run, made on an evening the model was overloaded, the median reply took 1.59 s, 95% came within 16.45 s, and 6 of the 82 took longer than 15 s. On the day of the held-out run none took longer than 7.44 s. **Under load, then**, about one follow-up in fourteen of that evening would have been searched for as it was typed, and the person would have seen that it was. A longer limit would save some of those at the price of a longer wait in front of every follow-up's answer, which then waits on the same overloaded model.

### The answer cache is keyed on the standalone question

The lookup comes after the rewrite, and an answer is stored under the question that was searched for. A follow-up whose standalone form was answered before is served from the cache: one request, for the rewrite, in place of two. The cache's rules do not change. Its version stays as it is, a judgement recorded in `tests/test_answer_cache_version.py`: an entry is still the answer to its own question, by the same instructions and search, and without earlier questions the code takes the path that was pinned before. A wrong rewrite looks up and stores an answer to another question, as a wrong question typed would, and the reply names it.

### Evaluations never run the step, except the one that measures it

An evaluation of answers measures the answer to a question as it was typed. The switch every evaluation runs inside (`measuring.py`) refuses the step at the entry point of an answer, as it refuses the answer cache: that covers the path through `answer_question`, and nothing else. A test reads the source as a second net: no evaluation command, no helper or class one reaches, and no code at the command line's module level, which runs for every command, names anything that builds or runs the step, also under another name it was imported as, and only the `ask` command and the service hand it to the entry point. A call of `standalone_question` or `make_rewriter` made directly from an evaluation is caught by that reading of the source alone: the step itself is not refused while an evaluation runs, because `eval-follow-ups`, which measures it by itself and calls no entry point of an answer, runs inside the same switch.

### Settings

`HANDBOOK_RAG_FOLLOW_UPS` (on by default: a question without earlier questions costs nothing more), `HANDBOOK_RAG_FOLLOW_UP_EARLIER_QUESTIONS` (3, between 1 and the 6 the step can send) and `HANDBOOK_RAG_FOLLOW_UP_TIMEOUT_SECONDS` (15). Switched off, earlier questions are ignored and the reply says that the step did not run. The limit on the length of an earlier question is a question's own, 500 characters. The container runs with the defaults, so `compose.yaml` needs no line for them.

### The MCP server is left as it is

Its client is a model, which can write a question that stands alone itself. A second model request there would cost a request and add nothing.

### Tracing

The step's span, `rewrite`, sits in the answer's trace, under `answer` and before `cache.lookup` and `search`, with its outcome and the number of earlier questions; its request is a span of its own with the model, the tokens and the reason it stopped, as an answer's request is. No question text is recorded unless `HANDBOOK_RAG_TRACE_CONTENT` is on ([observability](../observability.md)).

### The rule for a rewrite without a question mark, changed after the held-out replies were seen

The held-out run tested the rule as it stood: a rewrite without a question mark, for a question typed without one, had to begin with a question word, a verb that opens a question, a negated verb or a preposition followed by a question word. After those replies were seen, at 16:29 on 25 November 2026 by the account of the session that coordinated the work, **the requirement for an opening was taken out.** What stays on that path: it applies only to a question asked without a question mark, and the reply must not end with a full stop, an exclamation mark or an ellipsis, and must hold no question or exclamation mark and no full stop followed by a space, except after a single letter. Every other check stays.

- **Why.** The rule for that path has refused good rewrites twice: 6 replies in the first run's top-up, when it asked for a question mark every time, and 2 of the 10 held-out follow-ups typed without one, under the widened list of openings. In the 102 replies measured, it refused no reply that was not a question: the model did not answer in place of rewriting, also not under the 25 planted orders. A list of openings will keep missing how people type, and the checks are not the safety boundary.
- **The cost.** A reply that is a statement, an answer or an order with no stop at its end is now searched for, and the reply to the caller shows it as the question searched for.
- **What it gives, after the fact.** Read again under the changed rule, with no request, the same 20 replies give 11 of the 14 follow-ups, with two more readings of the two refused replies as if the step had used them, and the 82 replies of the first run give the same outcomes as before. That is not held-out evidence, and the changed rule has no held-out test of its own. The held-out run is kept as the command wrote it, under the rule fixed before it, and the second reading is a variant of its own, `held-out-after`. The code of the rule fixed before it is in no commit: the run's summary records its hash, and a test re-creates its check of how a reply begins on today's code and gets every outcome and figure the run recorded.

## What the real check showed

On 25 November 2026, with the real service on the real index and the free key, on a day the model was slow: 13 requests in all, counted from the replies themselves. 11 were the check as planned, held to a cap of 12, and 2 were one step added afterwards by the session that coordinated the work, the last row of the table. That step was not saved as a file, and is given as that session reported it. The answer cache was empty before the check. It was not emptied between the requests over HTTP and the demo page: the page's two rows were served from the entries the first two requests had stored. It was emptied after the planned steps, and again after the step added afterwards.

| What was asked | Searched for | Outcome | Requests | Whole answer |
|---|---|---|---|---|
| "How many days of paid annual leave do UK team members get?" | the same | `not_asked` | 1 | 11.2 s, answered |
| "and in finland", with that question | "How many days of paid annual leave do Finland team members get?" | `rewritten`, in 6.0 s | 2 | 10.2 s, declined |
| "do they carry over to the next year", with both questions as searched | "Do Finland team members paid annual leave days carry over to the next year?" | `rewritten`, in 6.0 s | 2 | 13.1 s, declined |
| The same, asked again | the same rewrite | `rewritten`, in 3.5 s, served from the cache, `exact` | 1 | 3.5 s, declined |
| "Which personal VPN does GitLab recommend for team members?", with both questions | the same | `unchanged`, in 5.8 s | 2 | 10.7 s, answered |
| On the demo page: the first question | the same | `not_asked`, served from the cache, `exact` | 0 | answered |
| On the demo page: "and in finland" | "How many days of paid annual leave do Finland team members get?" | `rewritten`, in 3.4 s, served from the cache, `exact` | 1 | declined |
| `ask --earlier` with the first question, "and in germany", the step's limit at 1 ms | "and in germany", as typed | `failed`, `timeout` | 2 | declined |
| `ask --earlier` with the first question, "and in ireland", added afterwards, as reported and not saved | "How many days of paid annual leave do ireland team members get?" | `rewritten`, in 6.3 s | 2 | answered, citing GitLab Ireland Ltd's time-off policies; its request 6.2 s, by the command's line |

- **The three follow-ups were rewritten to questions that mean what was asked**, two of them in clumsy English, and a question that stood alone was given back. The page showed "Searched for: …" above the answer, listed both questions of the conversation as they were searched for, and kept nothing in the browser's storage. The command said that the Germany follow-up was searched for as it was typed, and why.
- **The declines were right.** The handbook does not give the paid annual leave of team members in Finland: the golden set holds the question of Finland's statutory vacation days, `u019`, as one the assistant should decline, and, by the account of the session that coordinated the work, a search for the rewritten question, with no request to a model, brings back pages of Ireland, the Netherlands, Australia, Japan and Brazil, and none of Finland. So both rewritten Finland questions got the reply they should, when first asked and from the cache after. The follow-up searched for as typed was declined, as such a follow-up is likely to be. Finland was a poor choice for showing an answer. The one follow-up whose answer the handbook holds, "and in ireland", was rewritten and answered, with a citation.
- **What it does not show.** Three follow-ups got an answer or a right decline, in one sitting. Whether rewritten questions get right answers in general is not measured.
- **The rewrite took 3.4 to 6.3 s that day**, against a median of 1.59 s in the first run, and came before the search each time.
- **The limits of the check.** For the last request the answer's deadline was set to 0.01 s, so that a repeat of the answer request could not pass the cap. It is one try of each case, by hand.

## Alternatives considered

- **A conversation stored by the service.** It could send only answers it had stored itself, and keep a caller from sending a conversation it never had. It needs storage, an id for each conversation, a time after which one ends, and a way to keep one caller from another's conversation, on a service that has no login. Nothing in issue 79 needs it.
- **The answers of the earlier turns sent to the rewriting model.** 4 of the 8 follow-ups that point into an answer cannot be written without it. Before it is done, two things would have to be measured: how often a planted instruction in a handbook page, carried in an answer, moves the rewrite, with planted answers among the cases; and how much it helps on follow-ups into an answer, with more than 8 of them. Answers the service stored itself would also remove the doubt about what a caller sends.
- **The answer model given the whole conversation instead of a rewritten question.** The search would still look for "and in Finland?", and the answer model would get passages for that. The cache, keyed on one question, would need a key for a conversation. And a planted instruction in an earlier turn would reach the answer model itself.
- **A cache of rewrites.** The same conversation asked again would save the rewrite's request. Conversations repeat far less than single questions, the request is small, and a stored rewrite would need its own rules for when it stops being right. Not built.
- **A stream that starts before the step's outcome is known.** It would show the question searched for a moment sooner, and turn every failure before the first text into an event inside a stream that had already said yes.
- **Refusing earlier questions when follow-ups are switched off.** The reply already says that the step did not run, and a caller that always sends its conversation keeps working against either setting.

## What is still missing

- **Real conversations.** Every measured conversation was written by a model session, and no person has read them. How people word follow-ups, and how often they point into an answer, is not known.
- **More than two earlier questions.** No measured conversation has more than two. The service reads 3.
- **The Claude path.** The rewriting step on a Claude model is tested with stand-ins only, like the Claude answer generator.
- **Answers to rewritten questions, scored by the judges.** No answer to a rewritten question was scored in any measurement, so whether a follow-up gets the right answer in general is not known; the real check saw three get an answer or a right decline, in one sitting. 17 of the cases name a golden question whose reference answer the rewritten question should get, for that check.
- **A held-out test of the rule as it stands.** The rule for a rewrite without a question mark was changed after the held-out replies were seen.
- **A 422 names an unknown field.** A request with a field the API does not define is refused with the field's name in the reply, as before this change, though the body may now be four times larger. The value is not repeated.
- **What a production system would add.** A login, and a limit for each caller, before a conversation is worth protecting; the outcome of every rewrite counted, so that a rise in `failed` or `unusable` is seen; a sample of real follow-ups read by a person, against what was searched for; a smaller, faster model for the rewrite, measured against this one; answers from a conversation the service stores, if they are ever sent to the step; and a time limit set from real waits.

## Consequences

- **A follow-up costs one more request**, two in all, or one when its rewritten question is served from the cache. On the free key's 15 requests a minute and 500 a day, a conversation of follow-ups gets about half as many answers.
- **A follow-up waits for the rewrite before its search**: a median of 1.59 s in the first run, up to the 15-second limit.
- **A follow-up that could not be rewritten is searched for as it was typed**, and the person sees that it was.
- **Issue 79 is done**: the step is measured, held-out cases were asked, and it is in the service.
