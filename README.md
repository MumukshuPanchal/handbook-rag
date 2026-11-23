# Production RAG Pipeline

A knowledge assistant that answers questions from a company handbook, built the way a production team would build it: answers grounded in retrieved passages and cited, quality measured by an evaluation suite, and every request traced for latency and cost.

**Status: the assistant answers real questions with citations, on the command line and over HTTP, and its answers are measured: by code, and by two judges that were themselves checked first. Search and answers are served and traced.**

- The handbook is indexed, and hybrid search is the default, chosen on a checked set of test questions.
- It runs as a service: an HTTP API that answers with citations, in one response or streamed while the model writes, and `docker compose up` for the whole system.
- Every request is traced, stage by stage, the request to the language model included.
- CI fails when retrieval breaks.
- `ask` retrieves passages and has a language model write an answer that cites them, or says the handbook does not cover the question. It runs on Gemini with a free API key, and ran for real on 22 November 2026.
- All 165 test questions were put to it on 22 November 2026. It made the right call, to answer or to decline, on 0.958 of them. Of its answers, 0.984 were fully backed by the passages they cite, and 0.984 agreed with the reference answer.
- Not done yet: no person has checked the labels the judges were tested against, and the API has no authentication.

This README is updated as each milestone lands.

## Documents

| Document | What it covers |
|---|---|
| [Project charter](docs/charter.md) | The problem, the users, success metrics and non-goals |
| [Architecture](docs/architecture.md) | How the system works, with diagrams and a glossary |
| [Decision records](docs/adr/) | Why each major technical choice was made |
| [Roadmap](docs/roadmap.md) | Milestones and how work flows |
| [Evaluation](docs/evaluation.md) | The golden set, the metrics and how to read a run |
| [Observability](docs/observability.md) | How a request is traced, and how to read a trace to debug it |
| [Experiments](docs/experiments/) | What was tried, with numbers, including what did not work |

## Planned scope

- **Retrieval and generation**: ingest documents, retrieve the relevant passages, and answer with citations
- **Evaluation**: a curated test set with retrieval and answer-quality metrics, run in CI so regressions are caught
- **Observability**: traces, latency, and cost for every request
- **Production engineering**: tested code, CI, containerized services, and written design decisions

## Results so far

Retrieval on 132 test questions over 38,881 chunks of the handbook. These are the rows `baseline` to `v3` of what `uv run handbook-rag eval-table --metric hit@5 --metric hit@1 --metric mrr@10` prints from the runs in [`evals/runs/retrieval`](evals/runs/retrieval). The rows after them are the experiments linked below.

| Run | What it is | Questions | Hit@5 (95% interval) | Hit@1 | MRR@10 | Median time |
|---|---|---|---|---|---|---|
| baseline | Dense retrieval only (bge-small-en-v1.5). | 132 | 0.856 (0.795 to 0.909) | 0.583 | 0.705 | 26 ms |
| v1 | Keyword search only (BM25). | 132 | 0.932 (0.886 to 0.970) | 0.727 | 0.813 | 10 ms |
| v2 | Dense and BM25 merged with reciprocal rank fusion. | 132 | 0.962 (0.924 to 0.992) | 0.697 | 0.807 | 33 ms |
| v3 | Hybrid, then the top 30 reranked by a cross-encoder (ms-marco-MiniLM-L6-v2). | 132 | 0.939 (0.894 to 0.977) | 0.773 | 0.852 | 247 ms |

- **Hybrid search is the default.** It clearly beats dense search alone: +0.106 on Hit@5 (95% interval +0.053 to +0.159), with 14 questions gained and none lost. Against keyword search alone the picture is not clear: hybrid is ahead at five results (+0.030, an interval that includes zero) and behind at rank 1. That choice rests partly on judgement, and [ADR 0011](docs/adr/0011-hybrid-retrieval-by-default.md) says why.
- **Reranking is off by default.** It brings no clear gain at five results, and it makes search about seven times slower.
- **Two experiments did not help**, and are written up anyway: half-size chunks, and other settings of the rank fusion ([experiment log](docs/experiments/2026-11-22-chunk-size-and-fusion.md)).
- **CI catches broken retrieval.** Every pull request is scored with the real models on a slice of the handbook, and fails if a score falls below a recorded minimum. That detects breakage, such as losing either half of hybrid search. It does not detect a change that costs one or two questions ([ADR 0012](docs/adr/0012-retrieval-gate-on-a-slice.md)).
- **Targets.** The [charter](docs/charter.md) asks for Recall@5 of at least 0.85 and MRR@10 of at least 0.70. Hybrid search reaches 0.962 and 0.807.

### Answer scores

All 165 test questions, answered by `gemini-3.5-flash-lite` from the five passages hybrid search found, on 22 November 2026. The second, third and fourth scores come from two judges, which are one model under two sets of instructions: the model that wrote the answers. The others are computed by code from the saved replies. The run is [`evals/runs/answers/baseline`](evals/runs/answers/baseline), and [the evaluation guide](docs/evaluation.md#answer-metrics) says what each score means.

| Score | Result (95% interval) | Questions |
|---|---|---|
| Right call: answered what the handbook answers, declined what it does not | 0.958 (0.927 to 0.988) | 165 |
| Supported by its sources: no statement goes beyond the passage it cites | 0.984 (0.961 to 1.000) | 129 |
| Correct by the reference: the answer agrees with the reference answer | 0.984 (0.961 to 1.000) | 127 |
| Gets a correct answer: of the questions the handbook answers, those answered correctly | 0.947 (0.902 to 0.985) | 132 |
| Answers when it should | 0.962 (0.924 to 0.992) | 132 |
| Declines when it should | 0.939 (0.848 to 1.000) | 33 |
| Every statement cited | 0.977 (0.953 to 1.000) | 129 |
| Cites the passage that holds the evidence | 0.992 (0.976 to 1.000) | 127 |

- **Every answerable question that went unanswered was a search miss.** The assistant answered all 127 questions whose evidence it was given, and declined the 5 whose evidence the search had missed. Raising this score means better retrieval, not a better prompt.
- **The two wrong calls on unanswerable questions both say that the detail asked for is missing.** Each gives a related fact first, and the fixed rule counts a reply with a citation as an answer. Read by eye, the score understates here. No person has checked that reading.
- **Speed is uneven on the free tier.** The median reply took 2.4 s, and 95% arrived within 10.1 s. That is the model call alone, without the search, and it is just over the 10 seconds the charter allows for a complete answer. A third took more than 5 s.
- **The judge was checked before its score was used.** It was given [169 labelled answers](evals/judge/README.md), 40 of them with one planted fault. It caught all 41 statements labelled unsupported (95% interval 0.914 to 1.000) and raised one false alarm in 42. The judge is the answer model itself, because no other free model allows enough requests, and the labels were made by models of another family, not by a person ([ADR 0014](docs/adr/0014-answer-evaluation-in-stages.md)).
- **Where the judge is weak: limits that an answer leaves out.** Three real answers drop a limit or an exception that their passage states. Readers split on them, and the judge passed all three. If they count as faults, the score is 0.961.
- **A second judge compares each answer with the reference answer.** It was checked on [167 labelled answers](evals/judge/README.md#the-second-set-answers-against-their-reference-answers), 40 of them with a planted fault: it caught 39 of the 40 (95% interval 0.871 to 0.996) and raised 2 false alarms in 41. It then called 125 of the 127 answers correct, and the two it faulted are those false alarms: answers that leave out a next step the question did not ask for.
- **Seven of the 132 answerable questions did not get a correct answer.** Five are search misses, where the assistant declined, and two are the judge's doubtful `partial`. When the search finds the passage, the assistant answers, and answers correctly.
- **"Correct" means "agrees with the reference answer".** Most reference answers were written by a model from the same passage the assistant later quotes, so this score says the assistant reports what it reads. It says less about hard questions.
- **One run, one reply for each question.** The same question gets a different answer each time, so a second run would not give exactly these numbers.

**How far to trust these numbers.** The test questions were made by one model session and checked by another, over several rounds. No person has verified them one by one, and most were written by a model that could see the passage. The [dataset card](evals/golden/README.md) says how the set was made, what the checks found, and what its limits are.

## HTTP API

`make serve` runs the assistant as a service. It needs the index (`make qdrant`, `make index`), and for answers the Gemini key in `.env` (see [Answers](#answers)).

```bash
curl -s localhost:8000/health
# {"status":"ok","version":"0.1.0","chunks":38881,"answering":true,"detail":null}

curl -s -X POST localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "how do I expense a laptop", "limit": 2}'
```

`POST /search` returns the passages most relevant to a question, best first. Each has its text, its page, the headings above it, a link to the page and a score. The response also says how the passages were found and how long that took, and carries the id of the request's trace when tracing is on.

`POST /ask` answers a question, citing the passages the answer rests on. By default the answer comes as one JSON object:

```bash
curl -s -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query": "how do I expense a laptop"}'
```

Real output from 23 November 2026, laid out for reading. The segments and the texts of the passages are cut short where there is a `…`:

```json
{
  "answer": "All laptops are purchased through GitLab unless you are in a region where GitLab cannot deliver one.[1] If your laptop hasn't arrived by your first day, you may use a personal laptop temporarily and postpone hardware-related security setup tasks until it arrives.[2] \n\nIf you purchased a laptop for work, you can submit your receipts for reimbursement through Navan (the expense system) during your first week.[2] If you need a replacement laptop due to damage, you must reach out to IT via an issue before purchasing a new one.[1]",
  "answered": true,
  "segments": [
    {"text": "All laptops are purchased through GitLab unless…", "sources": [1]},
    {"text": "If your laptop hasn't arrived by your first day,…", "sources": [2]},
    {"text": "If you purchased a laptop for work, you can subm…", "sources": [2]},
    {"text": "If you need a replacement laptop due to damage,…", "sources": [1]}
  ],
  "sources": [
    {"number": 1, "doc_id": "finance/expenses.md", "title": "Global Travel and Expense Policy",
     "url": "https://handbook.gitlab.com/handbook/finance/expenses/",
     "section": "finance", "heading_path": ["Approved WFH Equipment List", "Laptops, insurance and repairs"],
     "text": "### Laptops, insurance and repairs - All laptops are purchased through…"},
    {"number": 2, "doc_id": "people-group/general-onboarding/tanewki-tips.md", "title": "TaNewKi Tips",
     "url": "https://handbook.gitlab.com/handbook/people-group/general-onboarding/tanewki-tips/",
     "section": "people-group", "heading_path": ["Your Onboarding Journey", "New Hire FAQ: Everything You Need to Know", "Equipment and Expenses"],
     "text": "#### Equipment and Expenses **Q: Can I purchase work items before my s…"}
  ],
  "model": "gemini-3.5-flash-lite",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 1767, "output_tokens": 112},
  "took_ms": 4593.4,
  "first_text_ms": 4526.8,
  "trace_id": "2e64d104fd523f20a47f46a29bce18ad"
}
```

The object holds the answer with a numbered marker such as `[1]` after each supported statement, whether it counts as an answer, the answer in segments with the numbers of their sources, the cited passages under those numbers, the model, why it stopped, the tokens, how long the whole answer and its first words took, and the trace id. A reply that is not an answer lists no sources.

With `"stream": true`, the answer comes as server-sent events while the model writes: an `event: text` for each piece of text, then one `event: answer` whose data is exactly the JSON object above. `-N` makes `curl` print each event as it arrives:

```bash
curl -sN -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query": "how do I expense a laptop", "stream": true}'
```

Real output from 23 November 2026, shortened where there is a `…`. The first text arrived after 1.5 seconds and the whole answer after 1.8. It is worded differently from the answer above: the model does not write the same reply twice.

```text
event: text
data: {"text":"L"}

event: text
data: {"text":"aptops are normally purchased directly through the company unless you are in a region where the"}

event: text
data: {"text":" company cannot deliver one. All laptops are provided by the company, and they are excluded from the yearly refresh stipend"}

… (4 more `text` events)

event: answer
data: {"answer":"Laptops are normally purchased directly through the company…","answered":true,"segments":[…],"sources":[…],"model":"gemini-3.5-flash-lite","stop_reason":"end_turn","usage":{"input_tokens":1767,"output_tokens":133},"took_ms":1781.3,"first_text_ms":1471.8,"trace_id":"a2549758b7116cd8ba1b846a279b833f"}
```

| Endpoint | What it does |
|---|---|
| `POST /ask` | `query` (1 to 500 characters), optional `section`, and `stream` (`true` or `false`, the default) |
| `POST /search` | `query` (1 to 500 characters), optional `limit` (1 to 20) and `section` |
| `GET /health` | 200 with the number of chunks in the index, and whether answering is configured. 503 if the vector store cannot be reached, the index has not been built, or it is empty, and it says which |
| `GET /docs` | Interactive documentation, generated from the code |

- **Bad requests.** A request that breaks a limit gets a 422 that names the field, without repeating what was sent. A body over 8 KB gets a 413.
- **A vector store that is down** gets a 503 with a fixed message, not a 500. The cause goes to the service's log, because it can name internal addresses. The service starts without the store and works as soon as the store is there.
- **An answer that cannot be given** gets a 503 with fixed words that say why: the vector store or the answer model is unavailable, the model's per-minute limit was reached, its allowance for the day is used up, answering is not configured, or the service is writing as many answers as it can. A per-minute limit and a full service add a `Retry-After` header when the wait is known. It is 503 and not 429 because the limits are the service's own with its model provider, not the caller's. The reason is in [ADR 0015](docs/adr/0015-answers-over-http.md).
- **Streams fail the same way until the first text.** The service waits for the first piece of text before it starts a stream, so an early failure is still a status with a JSON body. A failure after that ends the stream with one `event: error`.
- **Without a key** the service still starts and searches. `/ask` answers 503 and says that answering is not configured, and `/health` reports `"answering": false`. No request is sent to the model to find that out, so a key that is present but wrong shows up only at the first question.
- **Answers in flight.** The service writes at most four answers at once (`HANDBOOK_RAG_API_MAX_ANSWERS_IN_FLIGHT`), because each holds a thread while the model writes. A caller who leaves in the middle of a stream stops the model call at its next piece of text.
- **Trace id.** With tracing on, every response to `/search`, and every response to `/ask` once its answer has started, carries the id of its trace in the `X-Trace-Id` header, a failed one too. A stream also carries it in its last event.
- **Speed.** The models are loaded and run once when the service starts. Measured on the full index, searches one after another took 77 to 88 ms, and 124 to 169 ms after 20 to 60 idle seconds.
- **The embedding model runs on the CPU here**, also on a machine with a GPU. A service embeds one question at a time, which a CPU does as fast and more steadily: on an Apple GPU, the first search after a few idle seconds took over a second. `HANDBOOK_RAG_EMBEDDING_DEVICE` changes it.
- **Not built for the open internet.** There is no authentication and no rate limit for each caller; only the answers in flight are capped. Anyone who reaches the port can spend the key's allowance. The service listens on this machine only unless `HANDBOOK_RAG_API_HOST` says otherwise.

## Tracing

Every request is one trace, with a span for each stage, so a slow or wrong result can be followed to the stage that caused it.

![One search as a trace: the request, both searches, the embedding model, each call to the vector store, the merge and the reranker, each with its duration](docs/images/trace-reranked-search.png)

The trace of an answer adds the request to the language model, under the names OpenTelemetry's conventions for generative AI give its parts: the model, the tokens, why it stopped. Next to them are whether the reply counts as an answer and which passages it cites.

`make traces` starts the viewer. [docs/observability.md](docs/observability.md) explains how to read a trace, how to use one when an answer is wrong, what is deliberately left out of one, and what the first traces showed.

## Run it with Docker

One command starts the API, the vector store and the trace backend. Only Docker is needed.

```bash
docker compose up --detach --build                  # build the image and start everything
docker compose run --rm api handbook-rag snapshot   # download the handbook (43 MB)
docker compose run --rm api handbook-rag index      # embed it and store it in Qdrant
curl -s localhost:8000/health
```

The first build downloads about half a gigabyte and took four minutes here. The image is 2.2 GB, most of it PyTorch.

| Address | What is there |
|---|---|
| <http://localhost:8000/docs> | The API and its documentation |
| <http://localhost:16686> | Traces of every request |
| <http://localhost:6333/dashboard> | Qdrant's own dashboard |

- **Indexing is the slow step.** In a container the embedding model runs on the CPU: at the speed measured in CI that is 45 to 56 minutes for the whole handbook. Indexing can be stopped and started again; it skips what is already stored.
- **Wait for `Index ready`.** Before the first chunks are stored, `/health` answers 503 and says the index has not been built. It turns 200 as soon as some chunks are stored, while indexing is still running, and searches then see only part of the handbook.
- **Indexed already with `make index`?** The containers use the same Qdrant, so there is nothing to do again, as long as the settings that name the index (the handbook commit, the models, the chunk sizes) are the defaults.
- **Settings.** The application in the container does not read a `.env` file. It uses the defaults, plus what `compose.yaml` lists under `environment:`.
- **The Gemini key** is the one value taken from the project's `.env` file: Compose fills it in when it starts the API container, so it is written neither in `compose.yaml` nor in the image. Without it, `/ask` answers 503 and searches still work. `docker compose config` prints the key in clear text, so do not paste its output anywhere.
- **Port 8000.** The API container holds it, so stop it with `docker compose stop api` before running `make serve`.
- Everything listens on this machine only. Nothing here has authentication.

## Data

The assistant is built on the [GitLab Handbook](https://handbook.gitlab.com/), which GitLab publishes under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) licence. The handbook text is not stored in this repository. `make snapshot` downloads its Markdown pages at a pinned commit (4,206 pages, 43 MB) and writes a manifest that records the source, commit and a checksum of what was fetched. The reasoning is in [ADR 0002](docs/adr/0002-handbook-corpus.md).

`make chunks` cleans the pages and splits them along their headings into 38,881 chunks with a median of 225 tokens, none longer than the embedding model can read. How and why is in [ADR 0009](docs/adr/0009-structure-aware-chunking.md).

The test questions in [`evals/golden/`](evals/golden/README.md) quote short passages from the handbook. That folder is therefore licensed CC BY-SA 4.0 and credits GitLab, separately from the code, which is under the MIT licence.

This project is not affiliated with or endorsed by GitLab.

## Development

You need [uv](https://docs.astral.sh/uv/), `make`, `git` and Docker.

```bash
make install    # install dependencies and git hooks
make check      # lint, type check and fast tests: everything CI runs
make snapshot   # download the handbook at the pinned commit into data/
make chunks     # clean and chunk it, and print a summary
make qdrant     # start the Qdrant vector store in Docker
make index      # embed the chunks and store them; safe to run again
make search q="how do I expense a laptop"
make traces     # start the trace backend; see docs/observability.md
make serve      # run the HTTP API at http://127.0.0.1:8000; POST /ask answers, POST /search searches
uv run handbook-rag search "what is compa-ratio" --rerank
make ask q="how do I expense a laptop"   # answer with citations; needs a free Gemini API key
make test-live       # send three real requests to the answer model, one of them through the API
make golden-check    # verify every test question against the handbook
make eval-retrieval  # score the default search, as a scratch run named "current"
make eval-answers    # answer every test question and score what code can check: 165 real requests
make eval-judge      # check the judge against labelled answers: 169 real requests
make eval-support    # judge the saved answers: is each statement backed by the passage it cites?
make eval-judge-correct  # check the second judge against its labelled answers: 167 real requests
make eval-correct    # judge the saved answers: does each one agree with the reference answer?
make eval-table      # print a results table with one row per run on disk
make eval-gate       # the retrieval check CI runs, on a slice of the handbook
```

### Answers

```console
$ make ask q="How many days of paid time off can I take, and do I need approval?"
uv run handbook-rag ask "How many days of paid time off can I take, and do I need approval?"
GitLab does not set a specific annual limit on the amount of Flexible PTO you can take. However, time off exceeding 25 consecutive calendar days requires special permission from your Manager, People Business Partner, and the Absence Management Team.

You do need to coordinate your planned time off with your manager in advance and make a coverage plan.

Sources
  [1] GitLab Canada Corp > Time Away & Leave > Paid Time Off & Vacation
      https://handbook.gitlab.com/handbook/total-rewards/benefits/general-and-entity-benefits/canada-corp-benefits/
  [2] Time Off Types > Flexible Paid Time Off (PTO) > Overview
      https://handbook.gitlab.com/handbook/people-group/time-off-and-absence/time-off-types/

gemini-3.5-flash-lite | 1,728 tokens in | 79 out | 2.0 s
```

That is real output from 22 November 2026, and all of it: the first line is `make` showing the command it runs, and nothing else is printed. The time on the last line is the model call alone. Each command takes about ten seconds to start here, because it loads PyTorch and the embedding model, which the API server does once. Asking the same question again gives an answer that differs in wording and in detail, because the model chooses its words with some randomness. A question the handbook does not cover gets one fixed sentence, "I could not find this in the handbook.", and no sources.

- **The model.** Answers are written by Gemini (`gemini-3.5-flash-lite`). Copy `.env.example` to `.env` and set `GEMINI_API_KEY`. [Google AI Studio](https://aistudio.google.com/) gives a key free of charge, without setting up billing. The `.env` file is ignored by git, and the key is read only by this application.
- **A free key is for public text only.** On the free tier Google may use what is sent to improve its products. The handbook is public, so that is fine here. It would not be for private documents.
- **Citations.** The passages are numbered, the model names the numbers, and code checks that each number belongs to a passage that was really sent. Brackets in code that the parser recognises between backticks, such as `argv[1]`, are left alone. The forms it does not recognise are listed in the decision record. That check cannot tell whether the passage supports the statement, which is what the answer evaluation is for.
- **Claude instead.** Set `HANDBOOK_RAG_ANSWER_MODEL=claude-haiku-5-5` and `ANTHROPIC_API_KEY`. Claude's API returns the quoted words with every citation, a stronger guarantee. This path is unit-tested and has not been run against the Claude API yet.
- The reasoning for all of this is in [ADR 0013](docs/adr/0013-gemini-free-tier-with-marker-citations.md).

The workflow and conventions are in [CONTRIBUTING.md](CONTRIBUTING.md).
