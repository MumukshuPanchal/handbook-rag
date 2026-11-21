# Production RAG Pipeline

A knowledge assistant that answers questions from a company handbook, built the way a production team would build it: answers grounded in retrieved passages and cited, quality measured by an evaluation suite, and every request traced for latency and cost.

**Status: milestone M0, foundations.** The design is written and the project skeleton is in place. This README is updated as each milestone lands.

## Documents

| Document | What it covers |
|---|---|
| [Project charter](docs/charter.md) | The problem, the users, success metrics and non-goals |
| [Architecture](docs/architecture.md) | How the system works, with diagrams and a glossary |
| [Decision records](docs/adr/) | Why each major technical choice was made |
| [Roadmap](docs/roadmap.md) | Milestones and how work flows |

## Planned scope

- **Retrieval and generation**: ingest documents, retrieve the relevant passages, and answer with citations
- **Evaluation**: a curated test set with retrieval and answer-quality metrics, run in CI so regressions are caught
- **Observability**: traces, latency, and cost for every request
- **Production engineering**: tested code, CI, containerized services, and written design decisions

## Development

You need [uv](https://docs.astral.sh/uv/) and `make`.

```bash
make install   # install dependencies and git hooks
make check     # lint, type check and tests: everything CI runs
```

The workflow and conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).
