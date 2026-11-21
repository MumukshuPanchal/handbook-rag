# Project charter

## Problem

Company knowledge lives in long handbooks and policy pages. Finding an answer means knowing which page to open and which words its author used. Keyword search fails when a question is phrased differently from the page. A general-purpose chatbot answers confidently without knowing what the company's policy actually says.

## What we are building

A knowledge assistant that answers employee questions from a company handbook.

- Every answer is grounded in handbook passages and cites them: page, section, and the quoted text.
- When the handbook does not contain the answer, the assistant says so instead of guessing.
- Quality, latency and cost are measured on a fixed set of questions, and every change is judged against those numbers.

The handbook used is described in [ADR 0002](adr/0002-handbook-corpus.md).

## Users

| User | Needs |
|---|---|
| Employee | A short, trustworthy answer with a link to the source, in seconds |
| Maintainer | To know whether a change made answers better or worse, and to see why a particular answer came out the way it did |

## Success metrics

Targets are provisional. They state intent before anything is measured, and are revisited once the baseline exists (milestone M2). Any change to a target is recorded here with the reason.

| Goal | Metric | Provisional target |
|---|---|---|
| The right passage is retrieved | Recall@5 on the golden set | at least 0.85 |
| The right passage is ranked high | MRR@10 | at least 0.70 |
| Answers are supported by their sources | Share of answers whose claims are all backed by cited passages | at least 0.90 |
| Answers are correct | Share of answers consistent with the reference answer | at least 0.80 |
| The assistant abstains when it should | Share of unanswerable questions correctly declined | at least 0.90 |
| The assistant does not over-abstain | Share of answerable questions wrongly declined | at most 0.05 |
| Answers arrive quickly | 95th percentile time to a complete answer | at most 10 seconds |
| Cost is known and controlled | Mean model cost per answer | reported for each model tier; target set after the tier comparison |

## Non-goals for the first version

- Multi-turn conversation. Each question is independent.
- Per-user permissions on documents. The corpus is public.
- Fine-tuning any model.
- A polished user interface. The product is the API; a demo page is a stretch goal.
- Languages other than English.
- Live sync with the handbook. We index a pinned snapshot.

## Constraints

- Runs on one laptop. A GPU is used when present and never required. The only paid dependency is the model API.
- Reproducible by anyone: pinned data snapshot, locked dependencies, one command to start.
- No secrets in the repository.

## Risks

| Risk | Mitigation |
|---|---|
| Test questions drafted by a model are easier than real ones, because they reuse the wording of their source passage | Paraphrase during drafting, human review and sign-off, a hand-written subset, and results reported per question origin |
| The model may know a public handbook from its training data and answer from memory | Judge answers against the retrieved passages, and include questions the handbook cannot answer |
| A model judge can be wrong | Separate yes/no criteria, validation against human labels on a sample, agreement reported |
| Corpus licence terms | Recorded in ADR 0002; the corpus is downloaded at build time and not redistributed |
| Scope creep | Milestones with explicit completion criteria; stretch goals kept separate |
