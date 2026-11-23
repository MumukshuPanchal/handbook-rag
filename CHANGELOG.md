# Changelog

What each version of this project contains. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Each item names the pull request or decision record that holds it.

## [Unreleased]

### Added

- `handbook-rag mcp`: handbook search for AI assistants and coding tools, as one read-only tool over the Model Context Protocol, on standard input and output. It runs the same retriever as `POST /search`, checks its arguments with the same limits and returns the same object, and the passages as text too. No tool answers, so a call spends no request of the answer model ([#20], [ADR 0017]).
- A demo page at `/`, served by the API: a question box that shows the answer as the model writes it, then the answer with its sources, the model, the tokens, the timings and the trace id, and beside it what the search found. Three hand-written files with no build step. Everything from the API is shown as text, an address in an answer is never a link, and a strict content security policy is sent with each file. `HANDBOOK_RAG_API_DEMO=false` switches it off ([#19], [ADR 0016]).

### Changed

- The repository is named `handbook-rag`, like the Python package and the command. It was `production-rag-pipeline`. GitHub redirects the old address, and the links in the documents use the new one ([#70]).

### Fixed

- The documents said that measuring the answers again after a change to the instructions costs more than a day of the free allowance. It is about 460 requests of the 500 a day, and more than a day only when attempts have to be repeated ([#69]).

## [1.0.0] - 2026-11-23

The first version: an assistant that answers questions from the GitLab handbook with numbered citations, or declines, on the command line and over HTTP. Its search and its answers are measured by runs committed to the repository, and the [README](README.md#results) reports them.

### Added

#### Search

- Download of the handbook's Markdown pages at a pinned commit, with a manifest of what was fetched ([#26], [ADR 0002]).
- Cleaning of each page, and chunks cut along its headings within a token budget ([#27], [ADR 0009]).
- Search by meaning with a local embedding model, in Qdrant, and an index that can be built again safely ([#28], [ADR 0004], [ADR 0005]).
- Search by keyword (BM25), and hybrid search that merges both lists by reciprocal rank fusion ([#32]).
- Optional reranking with a cross-encoder ([#33]).
- Hybrid search as the default and reranking off, chosen by measurement ([#44], [ADR 0011]).
- Models and the tokenizer read from disk before the model hub is asked ([#54], [#55]).

#### Answers

- `ask`: an answer from the retrieved passages with a numbered citation after each statement, or one fixed sentence when they do not hold the answer ([#29], [ADR 0010], [ADR 0013]).
- Gemini on its free tier as the default answer model, and a model name that selects the provider ([#29], [ADR 0013]).
- A Claude generator with the API's own citations, unit-tested and never run against the Claude API ([#29], [ADR 0006]).
- Passages sent as data in tagged blocks, with text that looks like one of the tags made harmless ([#29], [ADR 0013]).
- Generation errors that say what kind they are, such as a rate limit or a used-up allowance ([#64]).

#### Evaluation

- Retrieval evaluation: Hit@k, Recall@5, MRR@10 and nDCG@10, each with a 95% interval ([#30], [ADR 0007]).
- Tools and guidelines for building the golden set, a page for reviewing it by eye, and a check that refuses evidence another page repeats ([#31], [#35]).
- A comparison of two runs question by question, with a fingerprint of the labels behind each score ([#36]).
- Committed runs kept small ([#38]), and a results table of every run ([#40]).
- The golden set: 165 questions, made and checked by separate model sessions ([#39]).
- Experiments with half-size chunks and other fusion settings, which did not help ([#41]).
- A retrieval gate in CI, on a slice of the handbook with the real models ([#45], [ADR 0012]).
- Answer evaluation in stages, within a daily allowance of requests, with the scores that code can compute ([#56], [ADR 0014]).
- A judge for whether each statement is backed by the passage it cites, checked on 169 labelled answers before its score was used ([#57]).
- A second judge for whether an answer agrees with its reference answer, checked on 167 labelled answers ([#62]).
- A control for the second judge: each reference answer given to it as the answer ([#67]).
- Adversarial tests: 36 cases that plant an instruction in a passage or ask a question outside the handbook ([#67]).

#### Serving

- `POST /search` and `GET /health` over HTTP ([#49]).
- `POST /ask`, as one JSON response or as server-sent events, with fixed words for every failure and a cap on answers in flight ([#64], [ADR 0015]).
- `docker compose up` for the API, Qdrant and the trace backend ([#50]), with the Gemini key passed from `.env` and nothing else ([#64]).

#### Observability

- A trace for every search, with a span for each stage, sent over OTLP to Jaeger ([#48], [ADR 0008]).
- One round trip to the vector store for each query instead of two, found by reading traces ([#49]).
- The whole answer traced, the model call included, under the names of OpenTelemetry's conventions for generative AI ([#64]).

#### Engineering

- The package, settings from the environment, lint, type checks, tests and CI on every pull request ([#23]).
- Decision records reviewed and accepted, among them a pipeline written in the project's own code instead of a framework ([#25], [ADR 0001], [ADR 0003]).
- Pull requests merged by fast-forward, so every commit lands on `main` unchanged ([#25]).
- The container image built in CI on every pull request, and checked to run as a user other than root ([#50]).
- The README rewritten with results and limitations, a test that holds its results to the committed runs, and this changelog ([#18]).

### Fixed

- The evaluation tests no longer really wait between requests, which had made the fast suite take minutes ([#65]).

### Known gaps

These are the [limitations in the README](README.md#limitations).

- No person has verified the test questions, the labels the judges were checked against, or the readings of the adversarial replies ([#8]).
- Most test questions were written by a model that could see the passage; too few were written without it ([#43]).
- Both judges are the answer model itself ([#63]).
- The judges were checked on one model's answers only, and whether a planted instruction moves a judge is not tested.
- The first judge's instructions lack two rules of its rubric, and its numbering slipped in one long answer ([#58]).
- Questions whose false assumption the handbook corrects are not in the test set, so nothing tests whether the assistant corrects the assumption ([#37]).
- Planted instructions got through in 8 of the 20 cases that plant one, and nothing is fixed yet ([#66]).
- Answers that leave out a limit or an exception are not measured reliably ([#59]).
- One reply for each question, on a free tier whose speed is uneven: the time to a complete answer is over the charter's target.
- No authentication and no rate limit for each caller, and no deadline or fallback for a model call ([#42]).
- One pinned snapshot of the handbook ([#21], [#34]).
- The Claude path has never been run, and model tiers were not compared ([#17]).
- Checked on one laptop: the quickstart has not been followed on a clean machine, and a full indexing run inside a container has not been done.

[Unreleased]: https://github.com/MumukshuPanchal/handbook-rag/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/MumukshuPanchal/handbook-rag/releases/tag/v1.0.0
[#8]: https://github.com/MumukshuPanchal/handbook-rag/issues/8
[#17]: https://github.com/MumukshuPanchal/handbook-rag/issues/17
[#18]: https://github.com/MumukshuPanchal/handbook-rag/issues/18
[#19]: https://github.com/MumukshuPanchal/handbook-rag/issues/19
[#20]: https://github.com/MumukshuPanchal/handbook-rag/issues/20
[#21]: https://github.com/MumukshuPanchal/handbook-rag/issues/21
[#23]: https://github.com/MumukshuPanchal/handbook-rag/pull/23
[#25]: https://github.com/MumukshuPanchal/handbook-rag/pull/25
[#26]: https://github.com/MumukshuPanchal/handbook-rag/pull/26
[#27]: https://github.com/MumukshuPanchal/handbook-rag/pull/27
[#28]: https://github.com/MumukshuPanchal/handbook-rag/pull/28
[#29]: https://github.com/MumukshuPanchal/handbook-rag/pull/29
[#30]: https://github.com/MumukshuPanchal/handbook-rag/pull/30
[#31]: https://github.com/MumukshuPanchal/handbook-rag/pull/31
[#32]: https://github.com/MumukshuPanchal/handbook-rag/pull/32
[#33]: https://github.com/MumukshuPanchal/handbook-rag/pull/33
[#34]: https://github.com/MumukshuPanchal/handbook-rag/issues/34
[#37]: https://github.com/MumukshuPanchal/handbook-rag/issues/37
[#35]: https://github.com/MumukshuPanchal/handbook-rag/pull/35
[#36]: https://github.com/MumukshuPanchal/handbook-rag/pull/36
[#38]: https://github.com/MumukshuPanchal/handbook-rag/pull/38
[#39]: https://github.com/MumukshuPanchal/handbook-rag/pull/39
[#40]: https://github.com/MumukshuPanchal/handbook-rag/pull/40
[#41]: https://github.com/MumukshuPanchal/handbook-rag/pull/41
[#42]: https://github.com/MumukshuPanchal/handbook-rag/issues/42
[#43]: https://github.com/MumukshuPanchal/handbook-rag/issues/43
[#44]: https://github.com/MumukshuPanchal/handbook-rag/pull/44
[#45]: https://github.com/MumukshuPanchal/handbook-rag/pull/45
[#48]: https://github.com/MumukshuPanchal/handbook-rag/pull/48
[#49]: https://github.com/MumukshuPanchal/handbook-rag/pull/49
[#50]: https://github.com/MumukshuPanchal/handbook-rag/pull/50
[#54]: https://github.com/MumukshuPanchal/handbook-rag/pull/54
[#55]: https://github.com/MumukshuPanchal/handbook-rag/pull/55
[#56]: https://github.com/MumukshuPanchal/handbook-rag/pull/56
[#57]: https://github.com/MumukshuPanchal/handbook-rag/pull/57
[#58]: https://github.com/MumukshuPanchal/handbook-rag/issues/58
[#59]: https://github.com/MumukshuPanchal/handbook-rag/issues/59
[#62]: https://github.com/MumukshuPanchal/handbook-rag/pull/62
[#63]: https://github.com/MumukshuPanchal/handbook-rag/pull/63
[#64]: https://github.com/MumukshuPanchal/handbook-rag/pull/64
[#65]: https://github.com/MumukshuPanchal/handbook-rag/pull/65
[#66]: https://github.com/MumukshuPanchal/handbook-rag/issues/66
[#67]: https://github.com/MumukshuPanchal/handbook-rag/pull/67
[#69]: https://github.com/MumukshuPanchal/handbook-rag/pull/69
[#70]: https://github.com/MumukshuPanchal/handbook-rag/pull/70
[ADR 0001]: docs/adr/0001-record-architecture-decisions.md
[ADR 0002]: docs/adr/0002-handbook-corpus.md
[ADR 0003]: docs/adr/0003-own-pipeline-code-instead-of-a-framework.md
[ADR 0004]: docs/adr/0004-qdrant-as-vector-store.md
[ADR 0005]: docs/adr/0005-local-embedding-and-reranking-models.md
[ADR 0006]: docs/adr/0006-claude-with-native-citations-for-generation.md
[ADR 0007]: docs/adr/0007-evaluation-strategy.md
[ADR 0008]: docs/adr/0008-opentelemetry-tracing.md
[ADR 0009]: docs/adr/0009-structure-aware-chunking.md
[ADR 0010]: docs/adr/0010-search-result-blocks-and-declining.md
[ADR 0011]: docs/adr/0011-hybrid-retrieval-by-default.md
[ADR 0012]: docs/adr/0012-retrieval-gate-on-a-slice.md
[ADR 0013]: docs/adr/0013-gemini-free-tier-with-marker-citations.md
[ADR 0014]: docs/adr/0014-answer-evaluation-in-stages.md
[ADR 0015]: docs/adr/0015-answers-over-http.md
[ADR 0016]: docs/adr/0016-a-demo-page-without-a-build-step.md
[ADR 0017]: docs/adr/0017-an-mcp-server-for-search.md
