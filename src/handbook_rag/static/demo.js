// The demo page's script: it asks POST /ask for a streamed answer and shows the answer
// as it arrives, with its sources. Why the page exists, and why it is built this way,
// is in docs/adr/0016-a-demo-page-without-a-build-step.md.
//
// Why it is written the way it is. The answer is written by a model, from handbook
// pages that anyone at the company can edit, and the adversarial tests showed that a
// link planted in a page reached the answer in three cases of four (docs/evaluation.md,
// issue 66). So everything that comes from the API is shown as text: elements are built
// one by one and filled with textContent. Nothing in this script reads a string as
// markup, runs a string as code, or takes the reader to another page. A test reads the
// whole of this file, these comments included, and fails if one of the names that would
// do so appears in it, which is why the comments do not name them either. Nothing in an
// answer becomes a link. The only links are the sources' pages, made in one function,
// sourceLink, and only for an address that starts with https:// or http://. The page's
// content security policy backs this up: the browser runs no inline script and loads
// nothing from another site.
//
// There is no build step and no library: this file is what the browser runs. What can
// go wrong is kept in small functions that take values and return values, such as
// reading the events of a stream and building the answer from its segments. The rest
// changes the page.

// ---- Reading the events of a stream ---------------------------------------------------

// Splits the text received so far into whole events and the start of the next one. An
// event ends with a blank line. A piece of the stream can end in the middle of an
// event, or hold several: what comes after the last blank line is kept for the next
// piece. A line may end in \r\n, \n or \r, as the standard allows. The service sends \n,
// but something between it and the browser may not. A \r at the very end is held back
// until the stream has ended, because the \n that would make the two one line ending
// may start the next piece.
function takeEvents(received, ended) {
  const held = !ended && received.endsWith("\r") ? "\r" : "";
  const lines = received.slice(0, received.length - held.length).replace(/\r\n?/g, "\n");
  const blocks = lines.split("\n\n");
  const rest = blocks.pop() + held;
  const events = blocks.map(parseEvent).filter((event) => event !== null);
  return { events, rest };
}

// One event from its lines: its name and its data. Comment lines, which start with a
// colon, and fields other than event and data are skipped. Without data it is no event.
function parseEvent(block) {
  let name = "message";
  const data = [];
  for (const line of block.split("\n")) {
    const colon = line.indexOf(":");
    if (colon === 0) {
      continue;
    }
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }
    if (field === "event") {
      name = value;
    } else if (field === "data") {
      data.push(value);
    }
  }
  return data.length === 0 ? null : { name, data: data.join("\n") };
}

// Calls onEvent with each event of a response's body, as its pieces arrive. The decoder
// holds back a character whose bytes are split between two pieces until it is whole.
async function readEvents(body, onEvent) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let received = "";
  for (;;) {
    const { done, value } = await reader.read();
    received += done ? decoder.decode() : decoder.decode(value, { stream: true });
    const { events, rest } = takeEvents(received, done);
    received = rest;
    events.forEach(onEvent);
    if (done) {
      return;
    }
  }
}

// The data of an event as JSON, or null when it is not JSON.
function parseData(data) {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

// The media type of a Content-Type header, without its parameters, in lower case.
function mediaType(contentType) {
  return (contentType ?? "").split(";")[0].trim().toLowerCase();
}

// What a reader is told when the service said yes but sent something other than a
// stream of events: what came instead, by its content type.
function notAStream(contentType) {
  const type = mediaType(contentType);
  if (type === "") {
    return "The reply was not the stream of the answer, and it did not say what it was.";
  }
  const words = `The reply was ${type}, not the stream of the answer (text/event-stream).`;
  return type === "text/html"
    ? `${words} Something between this page and the service may have answered instead.`
    : words;
}

// ---- Building the answer --------------------------------------------------------------

// Whether a value has the shape of the API's answer, as far as this page reads it: down
// to each source and each segment, so that building the answer cannot fail halfway.
function isAnswer(value) {
  return (
    isObject(value) &&
    typeof value.answered === "boolean" &&
    Array.isArray(value.sources) &&
    value.sources.every((source) => isObject(source) && Number.isInteger(source.number)) &&
    Array.isArray(value.segments) &&
    value.segments.every((segment) => isObject(segment) && Array.isArray(segment.sources)) &&
    isObject(value.usage)
  );
}

function isObject(value) {
  return value !== null && typeof value === "object";
}

// A value from the API as text to show: a string as it is, a number in figures, and
// anything else, such as a missing field, as "unknown".
function shown(value) {
  if (typeof value === "string") {
    return value;
  }
  return Number.isFinite(value) ? String(value) : "unknown";
}

// The answer as the pieces to show: the text of each segment, and the numbers of the
// sources it rests on. Only numbers the answer lists are kept, so a number never points
// at nothing. A segment without text adds none.
function answerPieces(answer) {
  const listed = new Set(answer.sources.map((source) => source.number));
  return answer.segments.map((segment) => ({
    text: typeof segment.text === "string" ? segment.text : "",
    numbers: segment.sources.filter((number) => listed.has(number)),
  }));
}

// The facts of a finished request, as pairs of a label and a value.
function answerFacts(answer) {
  const { input_tokens: read, output_tokens: written } = answer.usage;
  const tokens =
    Number.isFinite(read) && Number.isFinite(written)
      ? `${read} read, ${written} written`
      : "unknown";
  const firstText = answer.first_text_ms === null ? "no text came" : seconds(answer.first_text_ms);
  return [
    ["Model", shown(answer.model)],
    ...cacheFacts(answer),
    ...attemptFacts(answer),
    ["Tokens", tokens],
    ["First text after", firstText],
    ["Whole answer after", seconds(answer.took_ms)],
    ["Why the model stopped", shown(answer.stop_reason)],
  ];
}

// What the answer took, as pairs of a label and a value, when that is worth showing:
// more than one request to the model, or a reply by the fallback model
// (docs/adr/0018-time-limits-retries-and-a-fallback.md). Most answers take one request
// to the answer model, and then there is nothing to add. A service older than the two
// fields, requests and fallback, sends neither, so isAnswer does not ask for them: a
// field that is missing, or not of its type, counts as not sent.
function attemptFacts(answer) {
  const requests = Number.isInteger(answer.requests) ? answer.requests : null;
  const fallback = answer.fallback === true;
  if (!fallback && (requests === null || requests <= 1)) {
    return [];
  }
  const facts = requests === null ? [] : [["Requests to the model", String(requests)]];
  const by = fallback ? "the fallback model, after the answer model gave up" : "the answer model";
  facts.push(["Written by", by]);
  return facts;
}

// Where an answer came from when the model did not produce it for this request: the
// answer cache of the service (docs/adr/0020-an-answer-cache-that-matches-by-meaning.md).
// Null when the model produced it now. A service older than the field sends none, so
// isAnswer does not ask for it: a field that is missing, or not of this shape, counts
// as not sent.
function cacheHit(answer) {
  const hit = answer.cached;
  return isObject(hit) && typeof hit.question === "string" ? hit : null;
}

// What a reader is told above an answer that came from the cache, as lines of text.
// The stored question is given word for word. For a match by meaning it is another
// wording than the reader's own, and seeing it is how a wrong match is noticed.
function cacheLines(hit) {
  const how =
    hit.tier === "exact"
      ? "This answer was stored earlier for the same question."
      : "This answer was stored earlier for a question that a second model judged to mean the same as yours. That judgement can be wrong.";
  return [
    `${how} No request was sent to the answer model.`,
    `It was written for the question: ${hit.question}`,
  ];
}

// The same, as pairs of a label and a value for the facts of the request.
function cacheFacts(answer) {
  const hit = cacheHit(answer);
  if (hit === null) {
    return [];
  }
  const facts = [
    ["Served from", hit.tier === "exact" ? "the cache, an exact match" : "the cache, matched by meaning"],
    ["Stored", shown(hit.stored_at)],
  ];
  if (Number.isFinite(hit.similarity) && Number.isFinite(hit.score)) {
    facts.push(["Similarity", hit.similarity.toFixed(3)], ["Second check", hit.score.toFixed(3)]);
  }
  return facts;
}

function seconds(milliseconds) {
  return Number.isFinite(milliseconds) ? `${(milliseconds / 1000).toFixed(1)} s` : "unknown";
}

function plural(count, one, many) {
  return `${count} ${count === 1 ? one : many}`;
}

// ---- Failures in words ----------------------------------------------------------------

// What a reader is told when the service refused a request: from the status, the body
// when it could be read as JSON, and the Retry-After header. A 503 carries the service's
// own fixed words; a 422 names the field and what is wrong with it.
function refusalInWords(status, body, retryAfter) {
  const detail = isObject(body) ? body.detail : undefined;
  if (status === 503) {
    const words = typeof detail === "string" ? detail : "The service is unavailable.";
    const wait = waitInWords(retryAfter);
    return wait === "" ? words : `${words} ${wait}`;
  }
  if (status === 422) {
    const problems = Array.isArray(detail) ? detail.filter(isObject) : [];
    const reasons = problems.map((problem) => {
      const field = Array.isArray(problem.loc) ? problem.loc.at(-1) : "request";
      return `${shown(field)}: ${shown(problem.msg)}.`;
    });
    return ["The service did not accept the question.", ...reasons].join(" ");
  }
  if (status === 413) {
    return "The question is too long to send.";
  }
  if (status >= 500) {
    return `The service failed (status ${status}). Its log says why.`;
  }
  return `The service answered with status ${status}.`;
}

// How long the service asked to wait, in words, from a Retry-After header that holds a
// number of seconds, which is what the service sends. Empty when there is none, or when
// it holds anything else, such as a date.
function waitInWords(retryAfter) {
  if (retryAfter === null || !/^\d+$/.test(retryAfter)) {
    return "";
  }
  const wait = Number(retryAfter);
  if (wait === 0) {
    return "You can ask again now.";
  }
  if (wait < 120) {
    return `Wait ${plural(wait, "second", "seconds")} before asking again.`;
  }
  if (wait < 3600) {
    return `Wait about ${Math.ceil(wait / 60)} minutes before asking again.`;
  }
  return `Wait about ${plural(Math.round(wait / 3600), "hour", "hours")} before asking again.`;
}

// ---- Keys and announcements -----------------------------------------------------------

// Whether a key press in the question box asks the question: Ctrl+Enter, or Cmd+Enter on
// a Mac. Enter alone starts a new line, and an Enter that finishes a character being
// composed, as some keyboards for other scripts do, asks nothing.
function isAskKey(event) {
  return event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing;
}

// The text that puts a message in the status line so that a screen reader announces it.
// The same text set again is not announced, so a repeated message gets a no-break space
// added, which changes the text and not what is read. The next repeat takes it off.
function announcement(previous, words) {
  return previous === words ? `${words} ` : words;
}

// ---- Building the page ----------------------------------------------------------------

const page = {
  form: document.getElementById("ask-form"),
  question: document.getElementById("question"),
  count: document.getElementById("question-count"),
  ask: document.getElementById("ask"),
  stop: document.getElementById("stop"),
  examples: [...document.querySelectorAll("button.example")],
  status: document.getElementById("status"),
  answerArea: document.getElementById("answer-area"),
  stream: document.getElementById("stream"),
  answer: document.getElementById("answer"),
  failure: document.getElementById("failure"),
  sourcesArea: document.getElementById("sources-area"),
  sources: document.getElementById("sources"),
  factsArea: document.getElementById("facts-area"),
  facts: document.getElementById("facts"),
  searchArea: document.getElementById("search-area"),
  searchCount: document.getElementById("search-count"),
  searchState: document.getElementById("search-state"),
  passages: document.getElementById("passages"),
};

const UNREADABLE_ANSWER = "The service sent an answer this page cannot read.";

// An element with a class and its text. The text is set as text, never parsed.
function make(tag, className, text) {
  const node = document.createElement(tag);
  if (className !== "") {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

// The one place that makes a link: to a source's page, from the url field the API sends.
// Only an address that starts with https:// or http:// becomes a link. An address of any
// other kind leaves the title as plain text.
function sourceLink(url, title) {
  const isWebAddress =
    typeof url === "string" && (url.startsWith("https://") || url.startsWith("http://"));
  if (!isWebAddress) {
    const plain = make("span", "", title);
    plain.append(make("span", "address", " (no link: not a web address)"));
    return plain;
  }
  const link = document.createElement("a");
  link.href = url;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  link.textContent = title;
  link.append(make("span", "visually-hidden", " (opens in a new tab)"));
  return link;
}

// A source or a found passage: its label, its title as a link to its page, the address
// of that page as text, the headings above it, and its text folded away until asked for.
// The name says which it is, for the fold's screen reader name.
function passageItem(label, name, passage, extra) {
  const item = make("li", "source");
  const title = shown(passage.title);
  const heading = make("p", "source-title");
  heading.append(make("span", "source-number", label), " ", sourceLink(passage.url, title));
  item.append(heading);
  // The title is written by whoever edits the page, so the address is shown too, and the
  // reader sees where the link leads before following it.
  item.append(make("p", "source-address", shown(passage.url)));
  // The headings joined as the command line joins them.
  if (Array.isArray(passage.heading_path) && passage.heading_path.length > 0) {
    item.append(make("p", "source-headings", passage.heading_path.join(" > ")));
  }
  if (extra !== "") {
    item.append(make("p", "source-headings", extra));
  }
  const fold = make("details", "passage");
  const summary = make("summary", "", "Show the passage");
  // Every fold shows the same words, so a screen reader is also told whose passage it is.
  summary.append(make("span", "visually-hidden", ` of ${name}, ${title}`));
  fold.append(summary, make("p", "passage-text", shown(passage.text)));
  item.append(fold);
  return item;
}

// A number after a piece of the answer, which brings its source into view.
function citeButton(number, title) {
  const button = make("button", "cite", String(number));
  button.type = "button";
  button.setAttribute("aria-label", `Source ${number}: ${title}`);
  button.addEventListener("click", () => showSource(number));
  return button;
}

// Brings a source into view and marks it. The focus stays on the number that was
// pressed, so the reader keeps their place in the answer and can press the next one,
// and the status line says which source is marked and where.
function showSource(number) {
  const wanted = `source-${number}`;
  for (const item of page.sources.children) {
    item.classList.toggle("marked", item.id === wanted);
  }
  const item = document.getElementById(wanted);
  if (item !== null) {
    item.scrollIntoView({ block: "nearest" });
    say(`Source ${number} is marked, in the list of sources under the answer.`);
  }
}

function renderAnswer(answer) {
  const titles = new Map(answer.sources.map((source) => [source.number, shown(source.title)]));
  const box = make("div", answer.answered ? "answer" : "answer decline");
  if (!answer.answered) {
    box.append(make("p", "decline-label", "Not an answer"));
  }
  const hit = cacheHit(answer);
  if (hit !== null) {
    const note = make("div", "cached");
    note.append(make("p", "cached-label", "From the cache"));
    for (const line of cacheLines(hit)) {
      note.append(make("p", "", line));
    }
    box.append(note);
  }
  const text = make("p", "answer-text");
  const pieces = answerPieces(answer);
  for (const piece of pieces) {
    text.append(piece.text);
    for (const number of piece.numbers) {
      text.append(citeButton(number, titles.get(number)));
    }
  }
  if (pieces.length === 0) {
    text.textContent = "The reply holds no text.";
  }
  box.append(text);
  const items = answer.sources.map((source) => {
    const item = passageItem(`[${source.number}]`, `source ${source.number}`, source, "");
    item.id = `source-${source.number}`;
    return item;
  });
  const facts = answerFacts(answer);
  // All is built before the page changes, so an answer that cannot be shown leaves the
  // streamed text in place. The final answer then takes the place of that text, which is
  // emptied and hidden.
  page.stream.replaceChildren();
  page.stream.hidden = true;
  page.answer.replaceChildren(box);
  page.sources.replaceChildren(...items);
  page.sourcesArea.hidden = items.length === 0;
  renderFacts(facts, answer.trace_id);
}

// The facts of the request, and its trace id with a button that copies it.
function renderFacts(rows, traceId) {
  page.facts.replaceChildren();
  for (const [label, value] of rows) {
    page.facts.append(make("dt", "", label), make("dd", "", value));
  }
  if (typeof traceId === "string" && traceId !== "") {
    const value = make("dd", "");
    value.append(make("code", "", traceId), " ", copyButton(traceId));
    page.facts.append(make("dt", "", "Trace id"), value);
  }
  page.factsArea.hidden = page.facts.children.length === 0;
}

function copyButton(text) {
  const button = make("button", "copy", "Copy");
  button.type = "button";
  button.setAttribute("aria-label", "Copy the trace id");
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      say("The trace id is copied.");
    } catch {
      say("The trace id could not be copied. Select it and copy it by hand.");
    }
  });
  return button;
}

function say(words) {
  page.status.textContent = announcement(page.status.textContent, words);
}

// Shows a failure in words. Text that had already arrived stays, marked as unfinished.
function fail(words) {
  const partial = page.stream.textContent !== "";
  page.stream.classList.toggle("unfinished", partial);
  page.failure.textContent = partial
    ? `${words} The text above is only what arrived before the failure.`
    : words;
  say("No answer.");
}

function clearAnswer() {
  page.answerArea.hidden = false;
  page.stream.replaceChildren();
  page.stream.hidden = false;
  page.stream.classList.remove("unfinished");
  page.answer.replaceChildren();
  page.failure.textContent = "";
  page.sources.replaceChildren();
  page.sourcesArea.hidden = true;
  page.facts.replaceChildren();
  page.factsArea.hidden = true;
}

// ---- Asking ---------------------------------------------------------------------------

// The request being answered, or null. One at a time: a second question waits until the
// first is answered or stopped, so nothing of the first can land in the second's answer.
let current = null;

// The buttons are marked unavailable with aria-disabled instead of disabled, so that the
// button a keyboard user just pressed keeps the focus.
function setBusy(busy) {
  page.ask.setAttribute("aria-disabled", String(busy));
  for (const example of page.examples) {
    example.setAttribute("aria-disabled", String(busy));
  }
  page.stop.setAttribute("aria-disabled", String(!busy));
}

async function ask(question) {
  const controller = new AbortController();
  current = controller;
  setBusy(true);
  clearAnswer();
  search(question);
  say("Asking. The answer appears below as the model writes it.");
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, stream: true }),
      signal: controller.signal,
    });
    if (response.ok) {
      await readAnswer(response);
    } else {
      await refused(response);
    }
  } catch (error) {
    if (controller.signal.aborted) {
      stopped();
    } else {
      // A network failure, or a mistake in this script: the console says which.
      console.error(error);
      fail(
        page.stream.textContent === ""
          ? "The service could not be reached. Is it running?"
          : "The connection to the service broke.",
      );
    }
  } finally {
    current = null;
    setBusy(false);
  }
}

async function refused(response) {
  const body = await response.json().catch(() => null);
  fail(refusalInWords(response.status, body, response.headers.get("Retry-After")));
  renderFacts([], response.headers.get("X-Trace-Id"));
}

// Reads the events of the answer: text as it is written, then the answer or an error.
// The text is not announced piece by piece: the status line says that the model is
// writing, and when the answer is complete.
async function readAnswer(response) {
  const traceId = response.headers.get("X-Trace-Id");
  const contentType = response.headers.get("Content-Type");
  if (mediaType(contentType) !== "text/event-stream") {
    fail(notAStream(contentType));
    renderFacts([], traceId);
    return;
  }
  let ended = false;
  await readEvents(response.body, (event) => {
    if (ended) {
      return;
    }
    const data = parseData(event.data);
    if (event.name === "text" && data !== null && typeof data.text === "string") {
      if (page.stream.textContent === "") {
        say("The model is writing.");
      }
      page.stream.append(data.text);
    } else if (event.name === "answer") {
      ended = true;
      showAnswer(data);
    } else if (event.name === "error") {
      ended = true;
      const detail = data !== null && typeof data.detail === "string" ? data.detail : "";
      fail(detail === "" ? "The answer could not be finished." : detail);
      renderFacts([], traceId);
    }
  });
  if (!ended) {
    fail("The answer stopped before it was complete: the stream ended without it.");
    renderFacts([], traceId);
  }
}

// Shows the answer event, or says that it cannot be read. Its shape is checked first,
// and a fault while building it is caught too, so a reader is never told that the
// connection broke when it was the answer that could not be read.
function showAnswer(data) {
  if (!isAnswer(data)) {
    fail(UNREADABLE_ANSWER);
    return;
  }
  try {
    renderAnswer(data);
  } catch (error) {
    console.error(error);
    fail(UNREADABLE_ANSWER);
    return;
  }
  const from = cacheHit(data) === null ? "" : " It came from the cache.";
  say(
    data.answered
      ? `The answer is complete. It cites ${plural(data.sources.length, "source", "sources")}.${from}`
      : `Not an answer: the assistant declined, or its reply did not come to a proper end.${from}`,
  );
}

function stopped() {
  // Stop can be pressed in the moment after the answer or an error event arrived and
  // before the stream closed. The request had ended by then, and its outcome stands.
  if (page.answer.children.length > 0 || page.failure.textContent !== "") {
    return;
  }
  const partial = page.stream.textContent !== "";
  page.stream.classList.toggle("unfinished", partial);
  say(
    partial
      ? "Stopped. The text above is only what arrived before that."
      : "Stopped before the model wrote anything.",
  );
}

// ---- What the search found ------------------------------------------------------------

// The search for the question, run beside the answer. It costs no request to the model.
// A search that comes back after the next question was asked is dropped.
let searches = 0;

async function search(question) {
  const mine = ++searches;
  page.searchArea.hidden = false;
  page.searchCount.textContent = "";
  page.searchState.textContent = "Searching.";
  page.passages.replaceChildren();
  let response;
  let body;
  try {
    response = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question }),
    });
    body = await response.json().catch(() => null);
  } catch (error) {
    console.error(error);
    if (mine === searches) {
      page.searchState.textContent = "The search failed. The service could not be reached.";
    }
    return;
  }
  if (mine !== searches) {
    return;
  }
  if (response.ok && isObject(body) && Array.isArray(body.results)) {
    renderPassages(body.results.filter(isObject));
  } else {
    const words = refusalInWords(response.status, body, response.headers.get("Retry-After"));
    page.searchState.textContent = `The search failed. ${words}`;
  }
}

function renderPassages(results) {
  page.searchCount.textContent = `(${plural(results.length, "passage", "passages")})`;
  page.searchState.textContent = results.length === 0 ? "The search found nothing." : "";
  page.passages.replaceChildren(
    ...results.map((result) => {
      const rank = shown(result.rank);
      const score = Number.isFinite(result.score) ? result.score.toFixed(3) : "unknown";
      return passageItem(`${rank}.`, `found passage ${rank}`, result, `Score ${score}`);
    }),
  );
}

// ---- Wiring ---------------------------------------------------------------------------

function showCount() {
  page.count.textContent = `${page.question.value.length} of ${page.question.maxLength} characters`;
}

function start(question) {
  if (current !== null) {
    say("An answer is still being written. Stop it first, or wait for it.");
    return;
  }
  if (question === "") {
    say("Type a question first.");
    page.question.focus();
    return;
  }
  ask(question);
}

page.form.addEventListener("submit", (event) => {
  event.preventDefault();
  start(page.question.value.trim());
});

// Asking from the keyboard without leaving the box.
page.question.addEventListener("keydown", (event) => {
  if (isAskKey(event)) {
    event.preventDefault();
    start(page.question.value.trim());
  }
});

// An example question is put in the box, so the reader sees what was asked, and asked.
for (const example of page.examples) {
  example.addEventListener("click", () => {
    const question = example.textContent.trim();
    if (current === null) {
      page.question.value = question;
      showCount();
    }
    start(question);
  });
}

page.stop.addEventListener("click", () => {
  if (current === null) {
    say("No answer is being written.");
    return;
  }
  current.abort();
});

page.question.addEventListener("input", showCount);
showCount();
