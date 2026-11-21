# ADR 0006: Claude with native citations for answer generation

- Status: Accepted
- Date: 2026-11-21

## Context

The assistant must write answers that are grounded in the retrieved passages and that point to sources a reader can check.

The common approach is to number the passages in the prompt and ask the model to write markers like `[2]`. Nothing guarantees those markers are right: the model can cite the wrong passage, or cite one that does not support the sentence.

## Decision

Generate answers with Claude through the Anthropic API, using its Citations feature. Each retrieved chunk is sent as a document with citations enabled. The response comes back as text blocks, and each cited block carries the exact quoted text and the document it came from. We map those back to the handbook page and section.

- The model is a configuration value. Development starts on Claude Haiku 5.5, the lowest-cost tier. Whether it stays the default is decided by running our evaluation across tiers and comparing quality, latency and cost, and the result is recorded in a follow-up ADR.
- The generator sits behind a `Generator` interface, so retrieval and evaluation do not depend on the provider.
- Credentials come from the environment and never live in the repository.

## Alternatives considered

- **Prompt-based citation markers with any LLM.** Portable across providers, but the citations are unverified text. It remains the fallback strategy for a provider without a citations feature.
- **A local open-weight model.** No API cost and no data leaves the machine. We expect weaker answers and slower responses on laptop hardware. The interface leaves room to add one later.

## Consequences

- A citation is a quoted span tied to a retrieved passage, which the evaluation can check mechanically.
- The feature is provider-specific. Another provider would need the marker-based fallback.
- Citations cannot be combined with schema-constrained output in the same request. Machine-readable fields, such as whether the assistant abstained, need a separate mechanism.
- Answers cost money per request and need network access. Token usage is recorded for every call so cost is always visible.
