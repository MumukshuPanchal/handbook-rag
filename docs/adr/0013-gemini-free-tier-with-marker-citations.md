# ADR 0013: Answer with Gemini on its free tier, citing by numbered markers

- Status: Accepted
- Date: 2026-11-22

## Context

[ADR 0006](0006-claude-with-native-citations-for-generation.md) chose Claude with its built-in citations. That answering step was written and unit-tested, but it had never been run: the Claude API has no free tier, and the project has no key for it yet. A step that has never met its real API is not finished.

Google gives a Gemini API key without a card. Three facts about that free tier shape this decision. All were read on Google's own pages on 2026-11-22:

- **The limits are not published.** They apply per project, not per key, they are shown only inside Google AI Studio, and the documentation says they "are not guaranteed".
- **Google uses free-tier content.** What is sent and what comes back may be used to improve Google's products. Paid use is excluded from this.
- **There is no citation feature for our own passages.** The API cannot return which of the passages *we* supplied supports a statement.

ADR 0006 named the method for a provider like this: number the passages and have the model write the numbers.

## Decision

- **Gemini writes the answers by default, with `gemini-3.5-flash-lite`.** It is the lightest of the four Gemini text models that were free that day, and it did not reason before answering. The measurements are in the next section.
- **The model's name selects the provider.** A name starting with `gemini-` is served by the Gemini API, one starting with `claude-` by the Claude API, and the settings reject any other name. There is one setting, not a model and a provider that could disagree.
- **An exact model name, never an alias.** An alias such as `gemini-flash-latest` moves to a new model without notice, and results would stop being comparable.
- **Citations by numbered marker.** The passages are numbered in the prompt. The model puts the number of each supporting passage after a statement, such as `[2]`. Our code takes the markers out of the text as it streams and turns them into citations.
- **A marker counts only when it names a passage that was sent.** Brackets around any other number stay in the text exactly as written, so a citation the model made up is visible and not silently removed.
- **Passages are data.** Each passage sits in its own tagged block, apart from the instructions and the question. Text inside a passage that looks like one of those tags is made harmless, so a handbook page cannot close its own block and pose as something else.
- **Declining is unchanged.** The fixed sentence and the rule for recognising it are those of [ADR 0010](0010-search-result-blocks-and-declining.md).
- **The Claude generator stays**, behind the same interface. It has still not been run against the Claude API, and the README says so.
- **A rate limit is reported, not retried.** A request that fails because Google's servers are busy is sent up to three times. A rate limit on a free key lasts a minute or a day, so the message says which, and the request is not repeated.
- **Two tests send real requests** (`make test-live`). They need a key, so they never run in CI.

## What was measured

On 2026-11-22, with a free key, on questions answered from five retrieved passages (1,500 to 1,700 tokens in):

| Model | First words after | Whole answer after | Notes |
|---|---|---|---|
| `gemini-3.5-flash-lite` | 1.2 to 1.4 s, in the three requests where it was timed | 1.2 to 2.0 s in four requests, 5.9 s in a fifth | Answered all eight requests made while this was written |
| `gemini-3.6-flash` | 7.5 s | 8.2 s | One request. Spent 1,213 tokens reasoning first |
| `gemini-3.8-flash` | no answer | no answer | One request, sent three times by the retry rule: 503 "high demand" |
| `gemini-4-hydrogen` | no answer | no answer | One request, sent three times by the retry rule: 503 "high demand" |

These are a handful of requests from one laptop on one afternoon, enough to choose a default and not a benchmark. Three of the questions to the default model were read closely. They gave a cited answer, a cited answer that said which part it could not find, and the fixed sentence for a question the handbook does not cover.

## Alternatives considered

- **Wait for a Claude key.** Keeps the stronger citation method as the only one. Nothing would run in the meantime, and the answer evaluation could not start.
- **Replace Claude with Gemini.** Less code. It would remove the stronger citation method and the second provider, which the evaluation can use to compare answers.
- **Ask Gemini for structured output with quoted evidence**, then check in code that each quote is really in its passage. Closer to what built-in citations guarantee. Not chosen now because structured output cannot be shown to a reader while it streams, and the HTTP API will stream. Worth revisiting if the evaluation finds wrong citations.
- **Another free key.** Read on each provider's pages on 2026-11-22:
  - OpenRouter allows 50 requests a day on free models, fewer than one evaluation run of 165 questions.
  - Cohere's trial key allows 1,000 calls a month.
  - Groq is free for open models, with a daily token allowance on the page read that covers about 100 of our questions. Whether that table described the free plan could not be confirmed.
  - Cerebras no longer has a free tier: it gives a one-time credit of 5 dollars and needs a payment method.
  - GitHub Models was retired on 2026-07-30.
  - OpenAI's first usage tier with published limits starts with a purchase of 5 dollars. Its page names a free tier without saying what that tier allows.
- **A model on the laptop.** Free and private. The 16 GB development machine already runs short of memory with the vector store and the embedding models loaded.

## Consequences

- The assistant answers real questions. `handbook-rag ask` and both live tests ran against the Gemini API on 2026-11-22.
- **A citation from Gemini is weaker than one from Claude.** It names a whole passage, not the words the model relied on, so `cited_text` is empty for it. The evaluation cannot check a quote mechanically. It has to judge whether the passage supports the statement.
- **The model can still cite the wrong passage.** Code guarantees only that a cited passage was really sent.
- **Only public text may be sent with a free key.** The handbook is public. A deployment over private documents would need the paid tier or another provider.
- **The limits can change, and a full evaluation run is 165 requests.** Whether one run fits in a day is not known. It has to be measured when the answer evaluation is built, and the run may need pacing.
- **The free models are not equally available.** Two of the four were overloaded when tried. This is a reason for the light model as the default, and a reason not to promise that another one works.
- **Cost.** Nothing on the free tier. At list prices on 2026-11-22, `gemini-3.5-flash-lite` costs 0.30 dollars per million input tokens and 2.50 per million output tokens. Claude Haiku 5.5 costs 0.10 and 0.50. An answer of 1,700 tokens in and 130 out would cost about 0.08 cents on Gemini and about 0.02 cents on Claude, taking the token counts as equal, which they are only roughly. The `ask` command prints no dollar figure for Gemini: the price table holds Claude prices only.
- A Gemini model that reasons by default makes the reader wait several seconds for the first word. The reasoning setting applies to Claude models only for now.
- There are two SDKs to keep current, and the comparison of model tiers planned in ADR 0006 now spans two providers.
