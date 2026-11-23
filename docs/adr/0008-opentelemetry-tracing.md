# ADR 0008: Trace every request with OpenTelemetry

- Status: Accepted
- Date: 2026-11-21

## Context

When an answer is wrong or slow, we need to see what was retrieved, how it was ranked, what the model was sent, and where the time and tokens went. Logs alone do not show the timing of each stage or tie the stages of one request together.

## Decision

The pipeline is instrumented with OpenTelemetry, the vendor-neutral standard for traces.

- One trace per request, with a span for each stage: retrieve, fuse, rerank, generate.
- Spans carry the facts needed to debug an answer: retrieved chunk IDs and scores, the model used, token counts, and the outcome.
- The Anthropic SDK emits its own OpenTelemetry spans for model calls, following the GenAI semantic conventions (`gen_ai.*`, including token usage). They nest under our stage spans, so we do not instrument the model call by hand.
- Spans are exported over OTLP, so any compatible backend can receive them. Docker Compose ships a local backend. Which one is decided in the observability milestone by sending real traces to the candidates.
- Prompt and response text is recorded in traces only in local development. The default records metadata only.

## Alternatives considered

- **A vendor SDK (LangSmith, or Langfuse decorators).** Quick to set up with a UI built for LLM apps, but it ties the instrumentation code to one vendor.
- **Structured logs only.** Simple, but no per-stage timing tree and awkward to correlate across stages.

## Consequences

- The instrumentation is portable: changing backend is a configuration change.
- The same standard is used for ordinary services, so the skill carries over beyond LLM apps.
- The SDK's tracing support is in beta, so its span names and attributes may change. The SDK version is pinned.
- LLM-specific views such as cost per request depend on the chosen backend understanding the GenAI conventions.

## Update, 2026-11-22

The retrieval stages are traced, and the local backend is Jaeger: a small container that accepts OTLP directly. The comparison of backends is still open, on purpose. Backends built for language-model applications differ from Jaeger on spans of model calls, and there are none until the answering step is traced. See [the observability guide](../observability.md).

## Update, 2026-11-23

The model call is traced. The pipeline opens its span itself, for both providers, instead of relying on an SDK's own spans as the decision above planned: the default model is now Gemini ([ADR 0013](0013-gemini-free-tier-with-marker-citations.md)), and one span written here reads the same for both. The span is named and described with the names of OpenTelemetry's semantic conventions for generative AI: the provider, the model asked for and the model that answered, the tokens, and why the model stopped. A test compares each name with the package that defines the conventions, so a renamed one is noticed. A backend built for language-model applications that follows these conventions should therefore read these spans without a mapping. None has been tried, and Jaeger stays the local backend. Facts the conventions have no name for, such as whether the reply counts as an answer and which passages it cites, carry the project's own `handbook_rag.` prefix. Whether the Claude SDK adds a span of its own inside ours has not been seen, because the Claude path has not been run.
