# ADR 0010: Pass chunks as search result blocks, and decline with a fixed sentence

- Status: Accepted
- Date: 2026-11-21

## Context

[ADR 0006](0006-claude-with-native-citations-for-generation.md) chose Claude's built-in citations. Two details were left open: which block type carries the retrieved text, and how the system knows that the model declined to answer.

The API offers two ways to pass text that can be cited:

- **Document blocks.** With a plain-text source, the API splits the text into sentences itself and returns character offsets. With a custom source, we supply the blocks.
- **Search result blocks.** Made for retrieval: each has a source, a title and a list of text blocks, and the model cites whole blocks.

Knowing whether the model declined matters for two reasons. The evaluation must count correct and incorrect refusals, and a client should show "not found" differently from an answer. Citations cannot be combined with schema-constrained output in one request, so a machine-readable "answered" field is not available.

## Decision

- **One search result block per chunk.** The source is the page URL and the title is the page title followed by the heading path. The content is the chunk split into its Markdown blocks: paragraphs, lists, tables and code, with each heading attached to the block below it.
- **Citations map back by position.** The API returns the index of the cited search result, which is the chunk's position in the list we sent.
- **A fixed sentence for declining.** The instructions restrict the model to the search results and give it one exact sentence to reply with when they do not contain the answer. An answer is recorded as not answered when it contains that sentence and cites nothing, or when the model stops with a refusal.
- **No model call without context.** When retrieval returns nothing, the system declines directly.
- **Retrieved text is separated from instructions.** Passages travel as structured blocks, and the instructions say to treat anything inside them as content to report on, never as commands.

## Alternatives considered

- **Plain-text document blocks.** Finer, sentence-level citations. Not chosen because how the API splits Markdown tables and lists into sentences is outside our control, and document blocks have no field for the source URL.
- **A separate model call that decides whether the question is answerable.** A cleaner signal than detecting a sentence, at the price of a second call for every question. Worth revisiting if the evaluation shows the fixed sentence being missed.
- **A structured "answered" field in the same response.** Not possible together with citations.

## Consequences

- A citation identifies a block within a chunk, not a sentence.
- Detecting a decline relies on the model using the exact sentence. The abstention metrics in milestone M2 measure how often that holds.
- Search result blocks are specific to this provider. Another provider would need the marker-based fallback described in ADR 0006.

## Update, 2026-11-22

That fallback now exists, for Gemini: [ADR 0013](0013-gemini-free-tier-with-marker-citations.md). The fixed sentence for declining, the rule for recognising it and the rule of no model call without context are shared by both generators.
