# ADR 0013: Answer with Gemini on its free tier, citing by numbered markers

- Status: Accepted
- Date: 2026-11-22

## Context

[ADR 0006](0006-claude-with-native-citations-for-generation.md) chose Claude with its built-in citations. That answering step was written and unit-tested, but it had never been run: the Claude API has no free tier, and the project has no key for it yet. A step that has never met its real API is not finished.

Google gives a Gemini API key without a billing account: a new account starts on the free tier, and only the paid tier needs billing set up. Three facts about that free tier shape this decision. All were read on Google's own pages on 2026-11-22:

- **The free limits are not listed in the documentation.** It says they can be viewed in Google AI Studio, that they apply per project and not per key, and that they "are not guaranteed".
- **Google uses free-tier content.** What is sent and what comes back may be used to improve Google's products. Paid use is excluded from this.
- **There is no citation feature for passages we supply.** The API returns citations for Google Search results, for files uploaded into a File Search store that Google manages, and for web pages it fetches itself. It has none for text sent in the request. File Search would cite, but it would replace our own retrieval, which is the part this project builds and measures.

ADR 0006 named the method for a provider like this: number the passages and have the model write the numbers.

## Decision

- **Gemini writes the answers by default, with `gemini-3.5-flash-lite`.** It is the lightest of the four Gemini text models that were free that day, and it reported no reasoning tokens in the requests that were timed. The measurements are in the next section.
- **The model's name selects the provider.** A name starting with `gemini-` is served by the Gemini API, one starting with `claude-` by the Claude API, and the settings reject any other name. There is one setting, not a model and a provider that could disagree.
- **An exact model name, never an alias.** An alias such as `gemini-flash-latest` moves to a new model without notice, and results would stop being comparable. The settings reject a name that ends in `-latest`.
- **Citations by numbered marker.** The passages are numbered in the prompt. The model puts the number of each supporting passage after a statement, such as `[2]`. Our code takes the markers out of the text as it streams and turns them into citations.
- **Three rules decide what a marker is.**
  - Every number in it names a passage that was sent. Brackets around any other number stay in the text exactly as written, so a citation the model made up is visible and not silently removed.
  - It is not inside code. Between backticks, `argv[1]` and `[1, 2, 3]` are part of the answer. Code is recognised in the two forms Markdown writes with backticks: text between two equal runs of backticks on one line, and the lines between two lines that start with three or more. A backtick with no partner is an ordinary character, so it cannot switch citations off for the rest of an answer.
  - It is short: up to ten numbers of up to three digits. Everything else the parser recognises has a largest size too. That keeps the work per piece of streamed text small, however long the text is, and at most 200 characters are ever held back.
- **Passages are data.** Each passage sits in its own tagged block, apart from the instructions and the question. Text inside a passage that looks like one of those tags is made harmless, so a handbook page cannot close its own block and pose as something else.
- **Declining follows [ADR 0010](0010-search-result-blocks-and-declining.md), with one addition.** A reply that is the fixed sentence and nothing else is a decline even when the model attached a marker to it. A model told to cite every statement may do that.
- **A reply counts as an answer only when the model finished it or ran out of room.** After any other stop, such as a safety filter or a stream that ends without a reason, what arrived is kept and shown, and it is not counted as an answer. The rule is the same for both providers.
- **No follow-up requests.** The SDK can send a request again by itself when a reply stops at the API's size limit or the stream drops. That is switched off, because the free tier has a small daily allowance.
- **A rate limit is reported, not retried.** A request that fails because Google's servers are busy or failing, or because the connection fails before the reply starts, is sent up to three times. Those attempts are the only way a question costs more than one request. A rate limit on the free tier lasts a minute or a day, so the request is not repeated. The message says which limit it was when the API reports that, and that the limit belongs to the Google project, not to the key.
- **The Claude generator stays**, behind the same interface. It has still not been run against the Claude API, and the README says so.
- **Two tests send real requests** (`make test-live`). They need a key, so they never run in CI. (Note, 2026-11-23: there are three now. The third asks one question through `POST /ask` as a stream, [ADR 0015](0015-answers-over-http.md).)

## What was measured

On 2026-11-22, with a free key, on questions answered from five retrieved passages (1,513 to 1,910 tokens in):

| Model | First words after | Whole answer after | Notes |
|---|---|---|---|
| `gemini-3.5-flash-lite` | 1.2 to 1.4 s, in the three requests where it was timed | 1.2 to 2.2 s in five requests, 5.9 s in a sixth | All thirteen requests made while this was written succeeded |
| `gemini-3.6-flash` | 7.5 s | 8.2 s | One request. Spent 1,213 tokens reasoning first |
| `gemini-3.8-flash` | no answer | no answer | One request, under a retry rule of three attempts: 503 "high demand" |
| `gemini-4-hydrogen` | no answer | no answer | One request, under a retry rule of three attempts: 503 "high demand" |

These are a handful of requests from one laptop on one afternoon, enough to choose a default and not a benchmark. Three of the questions to the default model were read closely. They gave a cited answer, a cited answer that said which part it could not find, and the fixed sentence for a question the handbook does not cover.

## Alternatives considered

- **Wait for a Claude key.** Keeps the stronger citation method as the only one. Nothing would run in the meantime, and the answer evaluation could not start.
- **Replace Claude with Gemini.** Less code. It would remove the stronger citation method and the second provider, which the evaluation can use to compare answers.
- **Ask Gemini for structured output with quoted evidence**, then check in code that each quote is really in its passage. Closer to what built-in citations guarantee. Not chosen now because structured output cannot be shown to a reader while it streams, and the HTTP API will stream. Worth revisiting if the evaluation finds wrong citations.
- **A marker that cannot be mistaken for anything else**, such as `[S2]`. It would remove the need to tell markers from code. Not chosen for now: the plain form is the common convention, and the answers checked that day used it correctly. Worth revisiting if markers and ordinary brackets get mixed up in practice.
- **Another free key.** Read on each provider's pages on 2026-11-22:
  - OpenRouter allows 50 requests a day on free models, fewer than one evaluation run of 165 questions. Buying 10 dollars of credits raises that to 1,000 a day.
  - Cohere's trial key allows 1,000 calls a month.
  - Groq serves open models, and its billing page speaks of a free tier. The one table of limits that could be read allows 200,000 tokens a day for a model, about 100 of our questions. Whether that table describes the free plan could not be confirmed.
  - Cerebras has no permanent free tier: a one-time credit of 5 dollars that expires after 30 days, and it needs a payment method.
  - GitHub Models was retired on 2026-07-30.
  - OpenAI's first usage tier with published limits starts with a purchase of 5 dollars. Its page names a free tier without saying what that tier allows.
- **A model on the laptop.** Free and private. The 16 GB development machine already runs short of memory with the vector store and the embedding models loaded.

## Consequences

- The assistant answers real questions. `handbook-rag ask` and both live tests ran against the Gemini API on 2026-11-22.
- **A citation from Gemini is weaker than one from Claude.** It names a whole passage, not the words the model relied on, so `cited_text` is empty for it. The evaluation cannot check a quote mechanically. It has to judge whether the passage supports the statement.
- **The model can still cite the wrong passage.** Code guarantees only that a cited passage was really sent.
- **The parser follows Markdown closely, not completely.** A bracketed number that happens to match a passage is taken as a citation unless it stands in code the parser recognises. It does not recognise an indented block, a block fenced with `~~~`, or a block inside a quotation. A line that starts with enough backticks closes a block even when more text follows on it. Code beyond the sizes the parser allows is read as ordinary text too: code inside a line that is longer than 200 characters or broken over two lines, and a block whose first line carries more than 180 characters after its backticks or starts after more than 24 characters. In each of these cases a bracket in code can be taken as a citation, or the citations after it can be lost. None has shown up in a real answer, and they are listed in [issue 52](https://github.com/MumukshuPanchal/handbook-rag/issues/52).
- **Look-alike tags are not covered.** Handbook text cannot write one of the wrapper tags, however it spaces or cases it. A tag spelled with a hyphen or a look-alike character passes through, and whether a model would read that as a tag is not known. The same issue proposes a boundary that a passage cannot forge.
- **Only public text may be sent with a free key.** The handbook is public. A deployment over private documents would need the paid tier or another provider.
- **The limits can change, and a full evaluation run is 165 requests.** Whether one run fits in a day is not known. It has to be measured when the answer evaluation is built, and the run may need pacing.
- **The free models are not equally available.** Two of the four were overloaded when tried. This is a reason for the light model as the default, and a reason not to promise that another one works.
- **Cost.** Nothing on the free tier. At list prices on 2026-11-22, `gemini-3.5-flash-lite` costs 0.30 dollars per million input tokens and 2.50 per million output tokens. Claude Haiku 5.5 costs 0.10 and 0.50. An answer of 1,700 tokens in and 130 out would cost about 0.08 cents on Gemini and about 0.02 cents on Claude, taking the token counts as equal, which they are only roughly. The `ask` command prints no dollar figure for Gemini: the price table holds Claude prices only.
- A Gemini model that reasons by default makes the reader wait several seconds for the first word. The reasoning setting applies to Claude models only for now.
- There are two SDKs to keep current, and the comparison of model tiers planned in ADR 0006 now spans two providers.
