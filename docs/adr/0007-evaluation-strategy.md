# ADR 0007: Evaluation strategy

- Status: Proposed
- Date: 2026-11-21

## Context

Without measurement, a change to chunking or retrieval is a guess. A RAG system can fail in two different places: retrieval returns the wrong passages, or generation writes an answer the passages do not support. The fixes are different, so the two need separate metrics.

An evaluation can also mislead. Typical causes are test questions that are too easy, a grader that is wrong, and infrastructure errors counted as wrong answers.

## Decision

1. **Golden set.** A reviewed set of questions with a reference answer each. Every question is tagged with its origin (hand-written, or drafted by a model and verified by a person) and its type (answerable from the handbook, or deliberately unanswerable).
2. **Labels that survive re-chunking.** The supporting evidence for a question is stored as a document path plus the quoted passage, never as a chunk ID. A retrieved chunk counts as relevant when it overlaps that passage. Chunk sizes can then change without relabelling.
3. **Retrieval metrics in code.** Recall@k, MRR and nDCG@k are computed directly. They are deterministic and cost nothing, so they run on every change.
4. **Answer metrics from a model judge.** A judge model scores each answer on separate yes/no criteria: supported by the cited passages, consistent with the reference answer, and correct to abstain or not. The judge returns schema-constrained output. Before its scores are trusted it is checked against human labels on a sample, and the agreement is reported.
5. **One code path.** The evaluation calls the same entry point as the API.
6. **Honest bookkeeping.** Each run stores per-question results and full traces with the model name, token usage and latency taken from the API response. Timeouts and API errors are recorded separately and never scored as wrong answers.
7. **Regression gate.** CI runs the retrieval evaluation on a small fixed corpus and fails the build if a metric drops below its threshold.

## Alternatives considered

- **RAGAS or DeepEval.** Standard metrics out of the box. Not adopted because the metric prompts are defaults we would still have to validate, and the golden set and judge calibration are needed either way. The few metrics we use are small enough to implement and test directly.
- **Manual spot checks only.** Useful for intuition, but not repeatable and not comparable between changes.

## Consequences

- Evaluation is built straight after the first working pipeline, before any optimisation.
- Every later change is reported with numbers, including the ones that did not help.
- Judge-based metrics cost API calls, so they run on demand, not on every commit.
- Model-drafted questions tend to reuse the wording of their source passage, which flatters keyword search. Results are therefore reported per question origin, and the hand-written subset is the reference point.
