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
| `origin` | Who made it: `hand-written`, `model-drafted` (a model that was shown the passage) or `model-blind` (a model that was shown no handbook text; the evidence was found afterwards) |
| `review` | Who checked it: `pending` (nobody), `checked` (an independent checker, a separate model session that did not see how the question was made) or `verified` (a person) |
| `evidence_mode` | `all` (the default): the answer needs every piece of evidence. `any`: each piece answers the question on its own, so finding one of them is enough |
| `unanswerable_kind` | Unanswerable questions only: `near-miss`, `false-premise` or `out-of-scope` |
| `note` | Optional. Anything a reviewer should know, such as why no page answers the question |

Evidence is a page and a quote, never a chunk id. Chunk ids change whenever chunking settings change; a quote does not. When the evaluation runs, each quote is located in its page, and a retrieved chunk counts as relevant when it contains at least half of the quoted text.

If a quote cannot be found in its page, the run stops. A label that no longer matches the corpus would make every number that depends on it wrong.

## How the golden set is built

Nobody checks their own work here. A maker writes each question. A checker that has not seen how it was made looks for the maker's mistakes. The maker fixes what was found, and a fresh checker checks the fixes. The `origin` field records who made a question and the `review` field who checked it. The [drafting guidelines](../evals/golden/drafting-guidelines.md) describe both roles.

The current set, its counts and its known limits are described in its [dataset card](../evals/golden/README.md).

1. **Sample passages.** `handbook-rag golden-candidates` draws a seeded sample of chunks to write questions from. Each handbook section gets a share proportional to the square root of its size, so large sections get more questions without crowding out small ones, and at most one chunk is taken per page.
2. **Draft questions.** Each passage gets one question, a reference answer and an evidence quote, or is skipped, following the [drafting guidelines](../evals/golden/drafting-guidelines.md). Questions the handbook cannot answer are drafted separately and searched for in the corpus before they are kept.
3. **Check.** `handbook-rag golden-check` verifies that every quote is found word for word in its page, and on no other page. The handbook repeats some text on several pages (about 1.8% of chunks are exact copies of text elsewhere), and a quote with two homes would score a retriever as wrong for finding the other copy. It also reports how much each question shares its wording with its evidence, because a question that copies its source is easy for keyword matching and says little about search by meaning.
4. **Audit.** A second pass, which sees the question, the answer and the quote but not the drafter's reasoning, checks each question against the whole corpus. Its main job is to find a second passage that answers the same question, because the evaluation credits only the recorded one. Questions marked unanswerable are put to the strongest retriever, and the passages it returns are read.
5. **Check the misses.** After the first evaluation, the passages returned for every missed question are read, to tell a real miss from a valid passage that nobody recorded.
6. **Review.** A person checks each question and sets `review` to `verified`, or removes it. `make golden-review` writes a page to `data/golden-review.html` that shows every question next to the passage it was written from, with the evidence marked, so the check is a matter of reading.

## Retrieval metrics

```bash
make eval-retrieval
```

This scores the default configuration and writes it as the run named `current`. That name is the default and git ignores it, so trying something out never touches a committed run. A run is kept by giving it a name with `--variant`.

No language model is involved, so a run is free and gives the same result every time.

Each way of retrieving is a variant of the same run, so they can be compared question by question:

```bash
uv run handbook-rag eval-retrieval --variant baseline --mode dense
uv run handbook-rag eval-retrieval --variant v1 --mode sparse --description "Keyword search only (BM25)."
uv run handbook-rag eval-retrieval --variant v2 --mode hybrid --description "Dense and BM25 merged with reciprocal rank fusion."
uv run handbook-rag eval-retrieval --variant v3 --mode hybrid --rerank --description "Hybrid, then reranked by a cross-encoder."
```

Each row records the latency of its retrieval, so a variant's gain in quality can be weighed against the time it adds.

| Metric | Question it answers |
|---|---|
| Hit@k | Did a relevant chunk appear in the top k? The model can only answer from what it is given, so Hit@5 is the ceiling on answer quality when five chunks are passed. |
| Recall@5 | What share of the evidence was covered by the top 5? It differs from Hit@5 only when a question needs more than one passage. |
| MRR@10 | How high was the first relevant chunk? 1 at rank 1, one half at rank 2, and so on. |
| nDCG@10 | How good is the ordering overall, compared with the best possible one? |

Every mean is reported with a 95% confidence interval from a bootstrap over the questions. The interval shows how much the number would move with a different sample of questions.

Results are also broken down by question origin, because a question drafted by a model that could see the passage may reuse its words, which may flatter keyword-style matching. A simple count of shared words does not show this for the current set (see the [dataset card](../evals/golden/README.md)), so it is a precaution, not a finding.

Unanswerable questions have no evidence, so they are not scored here. They matter for the answer evaluation, where the assistant must decline.

## Comparing two variants

Two means with their intervals do not yet say which variant is better. To decide that, compare them question by question:

```bash
make eval-compare base=baseline candidate=v2
```

```text
v2 against baseline, on the same 136 questions

  metric     baseline        v2   difference (95% interval)   better  worse   same
  Hit@5         0.838     0.949   +0.110 (+0.059 to +0.169)       15      0    121  clear
  Hit@1         0.588     0.728   +0.140 (+0.074 to +0.206)       22      3    111  clear
```

(Example output, from a run on the draft golden set.)

Both variants answered the same questions, so each question gives a pair of scores. The comparison takes the difference within each pair and puts a confidence interval on the mean of those differences. This is called a paired comparison.

Pairing matters because most of the variation in a score comes from the questions, not from the system: some questions are easy for every variant and some are hard for every variant. Two separate intervals each carry that variation, so they can overlap even when one variant is reliably better. The difference per question cancels it out.

How to read a row:

- **clear** means the interval of the difference excludes zero. Chance is an unlikely explanation.
- **unclear** means the two variants cannot be told apart with this many questions. That is a finding too: it says not to pay extra latency or complexity for the candidate on this evidence.
- **better** and **worse** count the questions where the candidate scored higher or lower. A gain of 15 against a loss of 0 is more convincing than 40 against 25, even when the means match. The questions that got worse are the ones to read.

The comparison refuses to run if the two variants did not score exactly the same questions with the same labels. Every row in `results.jsonl` carries a fingerprint of its question, reference answer and evidence (and of the evidence mode, when that is not the default), so a run made before the golden set changed cannot be compared by accident with one made after.

To report results, `make eval-table` prints one row for every run it finds on disk as a Markdown table, read from the files each run wrote. Numbers in a write-up are generated, never copied by hand.

## The retrieval gate in CI

The full evaluation is too slow to run on every pull request: a CI machine has no GPU and would need about an hour to embed the handbook. The unit tests are fast but use stand-in models, so they cannot see the real models retrieving worse. The retrieval gate sits between the two.

```bash
make eval-gate
```

It keeps the 134 pages that hold evidence for a golden set question, indexes them in memory with the real models, and scores the default retriever on all 132 answerable questions. It needs the snapshot and the models, and no vector store server. It fails when:

- a mean score is below its minimum in [`evals/gate.json`](../evals/gate.json), currently Hit@5 of 0.97 and MRR@10 of 0.84;
- any question could not be searched. Elsewhere such failures are kept apart from the scores; a gate that passed while part of the set went unmeasured would be worse than no gate;
- nothing was scored at all.

Its scores are not quality numbers. A slice of 8% of the handbook is easier to search than the whole, so hybrid search scores 0.992 on Hit@5 there and 0.962 on the full index. The gate's scores are only compared with themselves. What it shows is that a change broke retrieval: on the same slice, dense search alone scores 0.939 and keyword search alone 0.962, and both fail. The dense-only failure was also produced in CI, in a pull request made for that purpose; the keyword-only number was measured on a laptop. A change that costs one or two questions still passes.

When a pull request changes retrieval on purpose, it changes the minimums in the same pull request, with the full evaluation as its evidence. The minimums are a floor to be raised, not a target. The reasoning, and the designs that were measured and rejected, are in [ADR 0012](adr/0012-retrieval-gate-on-a-slice.md).

## What a run writes

Each kept run is a variant: `baseline` first, then `v1`, `v2` and so on. (`current` is the scratch run.) A variant directory under `evals/runs/retrieval/` contains:

| File | Content |
|---|---|
| `results.jsonl` | One row per question: its scores, latency and label fingerprint. Small, so it is committed and two runs can be compared from a fresh clone |
| `summary.json` | Means with confidence intervals, overall and per group, the configuration that produced them (retrieval mode, models, chunking, handbook commit) and a fingerprint of the labels that were scored |
| `errors.jsonl` | Questions that could not be scored because the harness failed |
| `retrieved.jsonl` | The ranked chunks behind each score, with the evidence each covers. For debugging; not committed |
| `traces/` | One readable file per question. Not committed |

A harness failure, such as the vector store being unreachable, is recorded in `errors.jsonl` and never counted as a miss. A score has to describe the retriever, not the plumbing around it.

## Checks the harness itself must pass

The test suite runs two sanity checks through the whole evaluation: a retriever that always returns the right passage first must score 1.0 on every metric, and one that never returns it must score 0. An evaluation that fails either check is measuring its own bugs.
