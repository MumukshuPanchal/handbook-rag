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

## Answer metrics

Retrieval metrics say whether the right passage was found. They say nothing about what the assistant wrote with it. Answers are measured in stages, and the reasoning is in [ADR 0014](adr/0014-answer-evaluation-in-stages.md). The first stage needs no judge:

```bash
uv run handbook-rag eval-answers --variant baseline
```

It puts every golden question through the same entry point that `ask` uses, saves each reply, and scores what a program can decide. `make eval-answers` does the same for the scratch run `current`.

| Score | Taken over | What it says |
|---|---|---|
| Right call | every question | The assistant answered a question the handbook answers, or declined one it does not |
| Answers when it should | answerable questions | The same score, for the questions that have an answer |
| Declines when it should | unanswerable questions | The same score, for the questions that have none |
| Evidence among the passages | answerable questions | A passage the model was given covers the known evidence. This is the search's Hit@5 for the run |
| Every statement cited | answers | No statement in the answer lacks a passage number |
| Cites the evidence | answers that were given the evidence | A passage the answer names covers the known evidence |

Three things to know when reading them:

- **A wrong call is counted from outside.** When the search misses the evidence and the assistant declines, the assistant did the right thing with what it had, and the person asking still got no answer. That counts as a wrong call. "Evidence among the passages" and the split of "Answers when it should" by whether the evidence was given show where the fault lies.
- **A score is taken only over the questions it applies to**, and every mean is printed with that count. "Cites the evidence" means nothing for a question whose evidence never reached the model, so that question is left out of it.
- **These scores do not say that an answer is true.** An answer can name the right passage and still misstate it. Whether each statement is backed by the passage it cites is the next stage, and it needs a judge.

### One real request for each question

On a free Gemini key the default model allows 15 requests a minute and 500 a day, so a run of 165 questions is a third of a day's allowance. The run is careful with it:

| What the run does | Why |
|---|---|
| Sends at most 10 requests a minute (`HANDBOOK_RAG_EVAL_REQUESTS_PER_MINUTE`) | The key stays usable for `ask` while a run is going |
| Builds its client with one attempt for each request, and tries a failed question once more itself, after the others | The run knows how many requests it sent. A client that retries by itself sends requests nobody counts |
| Saves each reply as it arrives, and continues where it stopped when the command is run again | A run that is interrupted loses nothing it paid for |
| Records its settings, with a hash of the instructions, and refuses to continue with other ones | Replies from two models or two prompts never end up in one run |
| Stops itself after three failures in a row | By then the cause is the day's allowance, the network or the key |
| Asks the questions in a fixed order that mixes the kinds | A run that stops early is still a fair sample |

`--max-requests 20` stops after twenty requests. `--restart` discards everything the run has written. A request that fails is shown as it happens. A question that fails twice is written to `errors.jsonl` and is never scored as a wrong answer.

The count of requests is an upper limit: the run counts every question it puts to the pipeline, before the request is sent, and a question whose search fails or finds nothing never reaches the model. The count starts again when a run is restarted.

Answer runs are compared and tabled like retrieval runs: `uv run handbook-rag eval-compare baseline v1 --runs evals/runs/answers` and `uv run handbook-rag eval-table --runs evals/runs/answers`. A score is compared on the questions that have it in both runs, and the output says how many those are.

### The first run

`baseline`, on 22 November 2026: all 165 golden questions, answered by `gemini-3.5-flash-lite` from five passages found by the default hybrid search. No question was left unanswered by a failure. The run made 169 attempts for its 165 replies: four questions failed once and were answered on the second try.

| Score | Result (95% interval) | Questions |
|---|---|---|
| Right call | 0.958 (0.927 to 0.988) | 165 |
| Answers when it should | 0.962 (0.924 to 0.992) | 132 |
| Declines when it should | 0.939 (0.848 to 1.000) | 33 |
| Evidence among the passages | 0.962 (0.924 to 0.992) | 132 |
| Every statement cited | 0.977 (0.953 to 1.000) | 129 |
| Cites the evidence | 0.992 (0.976 to 1.000) | 127 |

What the replies behind these numbers show:

- **Every answerable question that went unanswered was a search miss.** The assistant answered all 127 questions whose evidence it was given, and declined all 5 whose evidence the search had missed. So "Answers when it should" equals the search's Hit@5 exactly, and the way to raise it is better retrieval, not a better prompt. Declining without the evidence is what the assistant is meant to do.
- **Both wrong calls on unanswerable questions are replies a person would accept.** Each gives a related fact from a passage and then says that the handbook does not give the detail that was asked: "However, the specific monetary amount allocated is not provided in the handbook." Both are near-miss questions, where the handbook covers the topic and not the detail. The score is 1 or 0 by a fixed rule, and a reply that carries a citation counts as an answer. So 0.939 understates how the assistant behaved, and judging such replies fairly needs a reader.
- **Three answers have a statement without a citation.** In two, that statement is the sentence saying what the handbook does not cover, which has no passage to cite. The third answer is correct and names no passage at all. The citation parser cites a reconstruction of that answer correctly wherever the marker is placed, so the model most likely wrote no marker. This cannot be proved, because the reply is not kept as the model wrote it.
- **Citations point at the evidence**: 126 of the 127 answers that were given the evidence name a passage that covers it. The one that does not is the answer without any citation.
- **Speed is uneven.** The median reply took 2.4 s, and 95% arrived within 10.1 s. A third of the replies (56 of 165) took more than 5 s, and the slowest 17.1 s. The time is the model call alone, on the free tier.
- **Size.** A question sends 1,654 tokens on average and gets 36 back. Answers are short: the median answer is 36 tokens long, and 120 of the 129 answers cite exactly one passage.

Against the targets in the [charter](charter.md):

| Target | Measured | |
|---|---|---|
| At least 0.90 of unanswerable questions declined | 0.939 (31 of 33) | Met, on 33 questions: the interval reaches down to 0.848 |
| At most 0.05 of answerable questions declined | 0.038 (5 of 132) | Met. All five are search misses |
| 95% of answers complete within 10 seconds | 10.1 s for the model call | Just missed, on the free tier |
| Answers supported by their sources, and answers correct | not measured yet | They need the judge |

**How far to trust these numbers.** Each question was asked once, and the same question gets a different answer each time, so a second run would not give exactly these numbers. The intervals cover which questions were asked, not how the model would answer them again. The questions were made and checked by model sessions and have not been verified by a person. And none of these scores says that an answer is true to its passage.

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

An answer run under `evals/runs/answers/` has the same files, with two differences:

| File | Content |
|---|---|
| `answers.jsonl` | Every reply: its statements, the passage numbers each one cites, the passages the model was given (by chunk id, with the evidence each covers), tokens and timing. Later stages score from this file, so it is committed. The passages are named, not copied; the statements are the model's own words |
| `run.json` | The settings the run was started with, and how many requests it has sent |

There is no `retrieved.jsonl`: `answers.jsonl` already lists the passages.

## Checks the harness itself must pass

The test suite runs two sanity checks through the whole evaluation: a retriever that always returns the right passage first must score 1.0 on every metric, and one that never returns it must score 0. An evaluation that fails either check is measuring its own bugs.

The answer evaluation has the same two checks: an assistant that cites the right passage and declines what it should must score 1.0 on every score, and one that declines every answerable question and answers the others without a citation must score 0 on every score that is about the assistant.
