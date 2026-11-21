# Evaluation

How the assistant is measured. The reasoning behind the approach is in [ADR 0007](adr/0007-evaluation-strategy.md).

## The golden set

A golden set is a fixed list of test questions with known answers. It lives in `evals/golden/questions.jsonl`, one JSON object per line:

```json
{
  "id": "q0001",
  "question": "How many weeks of parental leave do new parents get?",
  "reference_answer": "Sixteen weeks, fully paid.",
  "evidence": [
    {"doc_id": "people/leave.md", "quote": "New parents get sixteen weeks of paid leave."}
  ],
  "type": "answerable",
  "origin": "model-drafted",
  "review": "verified"
}
```

| Field | Meaning |
|---|---|
| `evidence` | The page and an exact quote from its cleaned text that answers the question |
| `type` | `answerable`, or `unanswerable` for questions the handbook cannot answer |
| `origin` | `hand-written` or `model-drafted` |
| `review` | `verified` once a person has checked the question, the answer and the evidence |

Evidence is a page and a quote, never a chunk id. Chunk ids change whenever chunking settings change; a quote does not. When the evaluation runs, each quote is located in its page, and a retrieved chunk counts as relevant when it contains at least half of the quoted text.

If a quote cannot be found in its page, the run stops. A label that no longer matches the corpus would make every number that depends on it wrong.

## How the golden set is built

1. **Sample passages.** `handbook-rag golden-candidates` draws a seeded sample of chunks to write questions from. Each handbook section gets a share proportional to the square root of its size, so large sections get more questions without crowding out small ones, and at most one chunk is taken per page.
2. **Draft questions.** Each passage gets one question, a reference answer and an evidence quote, or is skipped, following the [drafting guidelines](../evals/golden/drafting-guidelines.md). Questions the handbook cannot answer are drafted separately and searched for in the corpus before they are kept.
3. **Check.** `handbook-rag golden-check` verifies that every quote is found word for word in its page. It also reports how much each question shares its wording with its evidence, because a question that copies its source is easy for keyword matching and says little about search by meaning.
4. **Review.** A person checks each question and sets `review` to `verified`, or removes it.

## Retrieval metrics

```bash
make eval-retrieval
```

No language model is involved, so a run is free and gives the same result every time.

Each way of retrieving is a variant of the same run, so they can be compared question by question:

```bash
uv run handbook-rag eval-retrieval --variant baseline --mode dense
uv run handbook-rag eval-retrieval --variant v1 --mode sparse --description "Keyword search only (BM25)."
uv run handbook-rag eval-retrieval --variant v2 --mode hybrid --description "Dense and BM25 merged with reciprocal rank fusion."
```

| Metric | Question it answers |
|---|---|
| Hit@k | Did a relevant chunk appear in the top k? The model can only answer from what it is given, so Hit@5 is the ceiling on answer quality when five chunks are passed. |
| Recall@5 | What share of the evidence was covered by the top 5? It differs from Hit@5 only when a question needs more than one passage. |
| MRR@10 | How high was the first relevant chunk? 1 at rank 1, one half at rank 2, and so on. |
| nDCG@10 | How good is the ordering overall, compared with the best possible one? |

Every mean is reported with a 95% confidence interval from a bootstrap over the questions. The interval shows how much the number would move with a different sample of questions. A difference between two variants that is smaller than the interval is not evidence that one is better.

Results are also broken down by question origin, because a question drafted by a model tends to reuse the words of its source passage, which flatters keyword-style matching.

Unanswerable questions have no evidence, so they are not scored here. They matter for the answer evaluation, where the assistant must decline.

## What a run writes

Each run is a variant: `baseline` first, then `v1`, `v2` and so on. A variant directory under `evals/runs/retrieval/` contains:

| File | Content |
|---|---|
| `results.jsonl` | One row per question: scores, latency and the ranked chunks with the evidence each covers |
| `summary.json` | Means with confidence intervals, overall and per group |
| `errors.jsonl` | Questions that could not be scored because the harness failed |
| `traces/` | One readable file per question. Not committed. |

A harness failure, such as the vector store being unreachable, is recorded in `errors.jsonl` and never counted as a miss. A score has to describe the retriever, not the plumbing around it.

## Checks the harness itself must pass

The test suite runs two sanity checks through the whole evaluation: a retriever that always returns the right passage first must score 1.0 on every metric, and one that never returns it must score 0. An evaluation that fails either check is measuring its own bugs.
