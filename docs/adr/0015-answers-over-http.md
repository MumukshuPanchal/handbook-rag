# ADR 0015: Answer over HTTP, as one JSON response or as a stream

- Status: Accepted
- Date: 2026-11-23

## Context

The assistant answered questions on the command line only. The service offered `POST /search` and `GET /health`, and milestone M5 asks for an HTTP API with streaming answers.

Five facts shaped how `POST /ask` works:

- **An answer takes seconds.** In the baseline evaluation the median reply took 2.4 s and 95% arrived within 10.1 s, for the model call alone. The first words of the default model came after 1.2 to 1.4 s in the requests where that was timed ([ADR 0013](0013-gemini-free-tier-with-marker-citations.md)). A reader who sees the words as they come does not have to wait for the last one.
- **The generators block.** Each hands over text through a callback, from the thread it runs in, and holds that thread until the model has finished. The web framework serves requests on an event loop, which must never wait on a model.
- **The model's limits are the service's own.** The answer model runs on a free key that allowed 15 requests a minute and 500 a day for the project ([ADR 0014](0014-answer-evaluation-in-stages.md)). When a limit is reached, it is this service that has used up its allowance with its provider, whoever asked.
- **The errors of a generator are written for the person who runs it.** They name settings and files, such as "Check GEMINI_API_KEY in .env", and they quote the provider's own words.
- **A failure can come at any point.** The vector store can be down before anything is found, the model can refuse before it writes, and a connection can drop halfway through an answer.

## Decision

1. **One endpoint, two forms, chosen by a field in the body.** `POST /ask` takes `query`, an optional `section`, and `stream`, a strict boolean that is false by default. The limits are those of `/search`, and an unknown field is refused.
   - Without `stream`, the response is one JSON object: the answer with a numbered marker after each supported statement, whether it counts as an answer, the answer in segments with the numbers of their sources, the cited passages, the model, why it stopped, the tokens, two timings and the trace id.
   - With `stream`, the response is server-sent events. Each piece of text arrives as `event: text` as the model writes it, without the citation markers, which only exist once a statement is complete. The stream ends with one `event: answer` whose data is exactly the JSON object of the other form.
   - A reply that is not an answer lists no sources, as on the command line, and its text carries no markers.
2. **Server-sent events** are plain HTTP: a response that keeps arriving, in a text format that `curl -N` prints as it comes. An answer flows one way only, which is all this format offers.
3. **A failure before the first text is an HTTP status, also for a stream.** The request starts the work, waits for the first thing that happens (a piece of text, the finished answer, or a failure), and only then decides between an error response and a stream. Once the first text has been sent, so have the status and the headers, and a later failure can only be reported inside the stream: one `event: error` with a `detail`, after which the stream ends. Without `stream`, nothing is sent until the answer is complete, so every failure is a status.
4. **503 for every failure of something the service depends on**: the vector store, the model, the model's limits, a missing key, and a service that is writing as many answers as it can. Not 429, which tells a caller that they sent too much. The limits here are the service's own with its provider, and a caller who has asked once may meet them.
   - A per-minute limit gets a `Retry-After` header when the provider says how long to wait, rounded up to whole seconds, and never less than 1 or more than 3600. A wait that is not a finite, non-negative number is treated as unknown, and then there is no header.
   - A used-up daily allowance says so, and has no header: the allowance comes back the next day.
   - A full service gets `Retry-After: 5`.
5. **Fixed words for the caller.** The response says what kind of failure it was, in words chosen in advance, and never the error's own message. For that a generation error now carries a `reason` (`rate_limit`, `quota`, `configuration` or `unavailable`) and, when known, `retry_after` in seconds. Each generator sets them where its code already told the cases apart. The message itself goes to the service's log with the trace id, as for a failed search.
6. **A worker thread for each answer, and a cap on them.** The request runs the pipeline in a thread of its own, which puts each event on a queue that the request reads. The thread starts with a copy of the request's context, so the spans of the answer belong to the request's trace. `HANDBOOK_RAG_API_MAX_ANSWERS_IN_FLIGHT`, 4 by default, caps the answers being written at once, because each one holds a thread for as long as the model writes. A slot is held until the thread ends, and given back on every path: an answer, a failure, a fault in the code (even an exit raised in the worker), a caller who went away, a thread that could not start.
7. **A caller who goes away stops the model call.** When a stream ends, however it ends, the worker is told to stop, and the text callback raises at the next piece of text. That leaves the generator through the path it already has for failures: the Claude SDK's stream is closed as the error leaves its `with` block, and the Gemini stream is dropped as the error passes through the loop that reads it. A model that is still thinking before its first words is stopped when they come.
8. **Without a key the service still starts.** Searching needs no answer model. `/ask` then answers 503 with "Answering is not configured on this server.", and `GET /health` reports `answering: false`. That is found out from the settings, without a request to the model, so a key that is present but wrong is noticed only by the first question.
9. **Docker Compose passes the key and nothing else from `.env`**, by substitution in `compose.yaml`, so the key is written neither in that file nor in the image.

## Alternatives considered

- **A WebSocket.** It carries messages both ways, and an answer needs one way. It would also be a second protocol for proxies, clients and tests, next to the plain HTTP the rest of the API uses.
- **A separate streaming endpoint**, such as `POST /ask/stream`. Two endpoints for one operation, with two descriptions to keep in step and two places to validate the same request. A field keeps one of each.
- **Choosing the form by the `Accept` header.** The usual HTTP way. It was not chosen because the choice is easy to get wrong without noticing: many clients send a catch-all `Accept` by default, and a caller would get one form or the other without having asked. A field in the body is visible in the request and in the documentation.
- **Cancelling through the provider's API**, instead of raising in the callback. It would need each generator to hand its open request to another thread, and code for each provider. Raising in the callback ends the request through code both generators already have. Whether a provider stops generating when the reply is no longer read, and whether it counts the tokens it had already produced, is up to the provider and was not measured.

## What is still missing

- **Authentication.** Anyone who can reach the port can spend the key's allowance. The service listens on this machine only unless `HANDBOOK_RAG_API_HOST` says otherwise.
- **A rate limit for each caller.** The cap limits answers in flight, not how many one caller sends.
- **The timeouts and the fallback of [issue 42](https://github.com/MumukshuPanchal/handbook-rag/issues/42).** A model call is bounded only by the client's time limits for each phase of a request and its attempts, not by a deadline for the whole answer, and nothing falls back to another model when one is overloaded.
- **A caller of the JSON form who goes away is not noticed.** The model finishes that answer. Only a stream stops early.
- **Nothing is sent while the model thinks.** Until the first event the caller has no status line, and a stream sends no keep-alive comments. A proxy with a short idle time limit could cut off a long wait.

## Consequences

- The command line, the API and the evaluation run the same pipeline, and the two forms of `/ask` build their answer with the same code. A test checks byte for byte that the event at the end of a stream is the JSON response.
- A caller of a stream handles three event names, and has to expect an error event after text.
- The status line of a stream arrives with the first words, not before. That is the price of reporting an early failure as a status.
- To a caller, a model at its limit looks like a model that is down, apart from the words and the `Retry-After` header. The service's log says which it was, next to the trace id.
- Each answer in flight is a thread. The default cap is set by what the provider allows, not by what the machine could run.
