# Roadmap

Work is grouped into milestones. Each milestone ends with something that runs and, from M2 onwards, with numbers that can be compared against the milestone before it. Individual tasks are tracked as GitHub issues under the matching milestone.

## Milestones

| Milestone | Outcome | Done when |
|---|---|---|
| **M0 Foundations** | A repository others can build on | Charter, architecture and decision records are reviewed. Lint, tests and CI pass on the project skeleton. |
| **M1 Baseline pipeline** | A question gets a cited answer end to end, using the simplest approach that works | The handbook snapshot is indexed by a repeatable command. `ask` returns an answer with citations, or abstains. |
| **M2 Evaluation** | Quality has numbers | The golden set is reviewed and signed off. Baseline retrieval and answer metrics are recorded. CI fails when retrieval quality drops. |
| **M3 Retrieval quality** | Better retrieval, shown by the numbers | Hybrid search and reranking are each compared with the baseline. Chunking and embedding experiments are recorded. Defaults are chosen from the results. |
| **M4 Observability** | Every request can be explained | Each request produces a trace with per-stage timing, retrieved chunks, and token usage, viewable in a local trace UI. |
| **M5 Serving** | Usable as a service | An HTTP API with streaming answers. `docker compose up` starts the whole system. |
| **M6 Hardening and release** | Ready to show | Tests cover prompt injection and unanswerable questions. Claude tiers are compared on quality, latency and cost. The README reports results and limitations. `v1.0.0` is tagged. |

## Why this order

- **Baseline before improvements.** The simplest pipeline comes first so there is something to measure.
- **Evaluation before optimisation.** M2 comes before M3 because an improvement that cannot be measured cannot be claimed.
- **Tracing before serving.** Traces make the API easier to debug while it is being built.

## Stretch goals

Taken up only when the milestones above are done:

- A small web page for demos.
- An MCP server that exposes handbook search as a tool for AI assistants.
- Incremental re-indexing when the handbook changes.
- Query rewriting before retrieval.

## How work flows

1. Every task starts as an issue with a "done when" checklist.
2. Work happens on a short-lived branch named `type/short-description`.
3. A pull request links the issue, passes CI, and includes tests and documentation for what it changes.
4. Pull requests are merged by fast-forward, so `main` keeps a linear history in which every commit is a meaningful step.
5. A decision that changes the design gets a decision record in the same pull request.
