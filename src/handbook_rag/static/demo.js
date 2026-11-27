// The demo page's script: it asks POST /ask for a streamed answer and shows the
// conversation as it grows. Each question stands as a sign, under it how the answer was
// made runs as the stops of a line, and the answer follows with its sources beside it as
// lines, one colour for each. Why the page exists, and why it is built this way, is in
// docs/adr/0016-a-demo-page-without-a-build-step.md.
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
// A conversation lives in this page's memory and nowhere else. Each answer says which
// question the service searched for, and the page sends those questions with the next
// one, so that a follow-up can be rewritten into a question that stands alone
// (docs/adr/0024-follow-up-questions-in-the-service.md). Nothing is kept in the browser's
// storage: reloading the page starts a new conversation.
//
// There is no build step and no library: this file is what the browser runs. What can
// go wrong is kept in small functions that take values and return values, such as
// reading the events of a stream and building the answer from its segments. The rest
// changes the page. tests/test_demo_behaviour.py runs this file in node, with a small
// fake page, through a conversation.

// ---- Reading the events of a stream ---------------------------------------------------

// Splits the text received so far into whole events and the start of the next one. An
// event ends with a blank line. A piece of the stream can end in the middle of an
// event, or hold several: what comes after the last blank line is kept for the next
// piece. A line may end in \r\n, \n or \r, as the standard allows. The service sends \n,
// but something between it and the browser may not. A \r at the very end is held back
// until the stream has ended, because the \n that would turn the two into one line ending
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

// The notes of the checks in code that changed the reply or marked it, in the fixed words
// the service sends, each with whether its check declined the reply in place of what the
// model wrote (docs/adr/0025-hardening-against-planted-instructions.md). The reply on the
// page is the reply as the checks left it, and a note is the only sign that it is not all
// the model wrote, so every note is shown. A service older than the field sends none, so
// isAnswer does not ask for it: a field that is missing, or not a list, gives no note, and
// an item that is not of the shape this page reads is passed over.
function checkNotes(answer) {
  if (!Array.isArray(answer.checks)) {
    return [];
  }
  return answer.checks
    .filter((check) => isObject(check) && typeof check.note === "string" && check.note !== "")
    .map((check) => ({ note: check.note, declined: check.action === "declined" }));
}

// The numbers of the sources that the answer's text cites, each once. An answer can list
// a source that its text does not cite.
function citedNumbers(answer) {
  return new Set(answerPieces(answer).flatMap((piece) => piece.numbers));
}

// Whether a reply that is not an answer is a decline: the model finished and said that
// the handbook does not hold the answer, or nothing was found to answer from. Any other
// way of stopping means that the reply was withheld or cut off, which is not a decline.
function isDecline(answer) {
  return (
    !answer.answered && (answer.stop_reason === "end_turn" || answer.stop_reason === "no_context")
  );
}

// The facts of a finished request, as pairs of a label and a value: the timetable under
// "All details". A stored answer took no request, so it spent no tokens.
function answerFacts(answer) {
  const { input_tokens: read, output_tokens: written } = answer.usage;
  let tokens =
    Number.isFinite(read) && Number.isFinite(written)
      ? `${read} read, ${written} written`
      : "unknown";
  if (cacheHit(answer) !== null) {
    tokens = "none: no request was sent";
  }
  const firstText = answer.first_text_ms === null ? "no text came" : seconds(answer.first_text_ms);
  return [
    ...searchedFacts(answer),
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

// ---- The question searched for --------------------------------------------------------

// The question the service searched for, from the searched event or the searched_for
// field of an answer, when it has the shape this page reads. Null otherwise. A service
// older than the field sends none, so isAnswer does not ask for it.
function searchedFor(value) {
  return isObject(value) && typeof value.question === "string" && typeof value.outcome === "string"
    ? value
    : null;
}

// What a reader is told of the question searched for, as one line of text, or "" when
// it is the question as asked. A follow-up that could not be rewritten is searched for as
// it was typed, and the reader is told so: it then finds little. So is a follow-up that a
// service did not rewrite at all, because its follow-ups are switched off or it has no
// rewriting step: earlier questions were sent, and the reply says the step did not run.
function searchedLine(searched, followUp) {
  if (followUp && searched.outcome === "not_asked") {
    return "Searched for your question as you typed it: this service did not rewrite the follow-up, so your earlier questions were not used. A follow-up searched for that way may find little.";
  }
  if (searched.outcome === "failed" || searched.outcome === "unusable") {
    const why =
      searched.outcome === "failed"
        ? "the request to rewrite it failed"
        : "the rewritten question could not be used";
    return `Searched for your question as you typed it: ${why}. A follow-up searched for that way may find little.`;
  }
  if (searched.question !== searched.asked) {
    return `Searched for: ${searched.question}`;
  }
  return "";
}

// How the rewriting step came to the question, in words.
function outcomeInWords(outcome) {
  const words = {
    unchanged: "given back as it was asked: it stands alone",
    rewritten: "rewritten to stand alone",
    unusable: "not usable, so searched for as typed",
    failed: "failed, so searched for as typed",
  };
  return Object.hasOwn(words, outcome) ? words[outcome] : shown(outcome);
}

// The facts of the rewriting step, as pairs of a label and a value, when it ran.
function searchedFacts(answer) {
  const searched = searchedFor(answer.searched_for);
  if (searched === null || searched.outcome === "not_asked") {
    return [];
  }
  const facts = [
    ["Searched for", searched.question],
    ["The follow-up was", outcomeInWords(searched.outcome)],
  ];
  if (Number.isInteger(searched.requests) && searched.requests > 0) {
    const usage = isObject(searched.usage) ? searched.usage : {};
    const tokens =
      Number.isFinite(usage.input_tokens) && Number.isFinite(usage.output_tokens)
        ? `, ${usage.input_tokens} tokens read, ${usage.output_tokens} written`
        : "";
    facts.push([
      "Rewriting took",
      `${plural(searched.requests, "request", "requests")}${tokens}, ${seconds(searched.took_ms)}`,
    ]);
  }
  return facts;
}

function seconds(milliseconds) {
  return Number.isFinite(milliseconds) ? `${(milliseconds / 1000).toFixed(1)} s` : "unknown";
}

function plural(count, one, many) {
  return `${count} ${count === 1 ? one : many}`;
}

// A time in seconds, in words, for the stops of a line.
function secondsInWords(milliseconds) {
  return Number.isFinite(milliseconds) ? `${(milliseconds / 1000).toFixed(1)} seconds` : "an unknown time";
}

// ---- The line of stops ----------------------------------------------------------------

// The colour of a source's line, by its number: one of the five line colours, and for a
// source past them, which a setting that gives the model more than five passages can
// bring, the colour of the signs. The number always stands in its bullet, so colour is
// never the only way to tell two sources apart.
function lineClass(number) {
  return Number.isInteger(number) && number >= 1 && number <= 5 ? `line-${number}` : "line-other";
}

// How many passages the page's own search found, in words, for the stop of the search.
function foundInWords(found) {
  if (Number.isInteger(found)) {
    return found === 0 ? "nothing found" : `${plural(found, "passage", "passages")} found`;
  }
  if (found === "failed") {
    return "not counted: the page's own search failed";
  }
  if (found === "not run") {
    return "not counted: the service did not say what it searched for";
  }
  if (found === "running") {
    return "counting the passages found";
  }
  return "";
}

// The stop of the step that rewrites a follow-up, by what it did.
function rewriteStop(searched) {
  if (searched === null) {
    return { name: "Rewriting", detail: "", state: "ahead" };
  }
  const words = {
    rewritten: ["Rewritten", searched.question],
    unchanged: ["Stood alone", "kept as you asked it"],
    failed: ["Not rewritten", "the request failed, so searched as typed"],
    unusable: ["Not rewritten", "the rewrite could not be used, so searched as typed"],
    not_asked: ["Not rewritten", "this service does not rewrite, so searched as typed"],
  };
  const [name, detail] = Object.hasOwn(words, searched.outcome)
    ? words[searched.outcome]
    : ["Rewrite", shown(searched.outcome)];
  return { name, detail, state: "reached" };
}

// What went wrong when a request ended without an answer, by its kind: the request
// itself, a reply the page cannot read, or a stream that ended before the answer came.
const FAILED_IN_WORDS = {
  failed: "the request failed",
  unreadable: "the answer could not be read",
  cut: "the stream ended without the answer",
};

// The stop where a line ended short: a failure, or a stop by the reader.
function endStop(journey) {
  if (journey.ended.kind === "stopped") {
    return {
      name: "Stopped by you",
      detail: journey.text ? "after the first words" : "before the first words",
      state: "end",
    };
  }
  const why = FAILED_IN_WORDS[journey.ended.kind] ?? FAILED_IN_WORDS.failed;
  return journey.text
    ? { name: "Broke off", detail: `after the first words: ${why}`, state: "end" }
    : { name: "Not answered", detail: why, state: "end" };
}

// The stop where a reply that is not an answer ended, named by why the model stopped: held
// back by the provider, cut at the length limit, or stopped for another reason.
const WITHHELD = new Set(["refusal", "safety", "recitation", "blocklist", "prohibited_content", "spii"]);

function stoppedStop(answer) {
  const reason = shown(answer.stop_reason);
  let name = "Stopped early";
  if (WITHHELD.has(reason)) {
    name = "Withheld";
  } else if (reason === "max_tokens") {
    name = "Cut off";
  }
  return { name, detail: `the model stopped: ${reason}`, state: "end" };
}

// What the last stop of an answer says: how long it took and how many sources its text
// cites, and what else is worth knowing about how it was made.
function answeredDetail(answer) {
  const parts = [`in ${secondsInWords(answer.took_ms)}`, `${plural(citedNumbers(answer).size, "source", "sources")} cited`];
  if (cacheHit(answer) !== null) {
    parts.push("no request to the model");
  }
  if (Number.isInteger(answer.requests) && answer.requests > 1) {
    parts.push(`after ${answer.requests} requests`);
  }
  if (answer.fallback === true) {
    parts.push("by the fallback model");
  }
  const notes = checkNotes(answer).length;
  if (notes > 0) {
    parts.push(notes === 1 ? "changed by a check" : `changed by ${notes} checks`);
  }
  return parts.join(", ");
}

// The kind of line that leaves each stop for the next: solid to a stop reached or to
// where the line ended, faint to one not reached yet, dashed to one the line does not
// serve, and none after the last.
function withSegments(stops) {
  return stops.map((stop, index) => {
    const next = stops[index + 1];
    let segment = "none";
    if (next !== undefined) {
      segment = next.state === "off" ? "dashed" : next.state === "reached" || next.state === "end" ? "solid" : "faint";
    }
    return { ...stop, segment };
  });
}

// The stops of a turn's line, from what is known of it so far, first to last: each with a
// name, a line of detail, the state it is in and the kind of line that leaves it. While an
// answer is being written this is the page's progress display: "Asked" is reached at once,
// the search when the service says what it searched for, the first words with the first
// text, and the end with the answer. A stored answer runs on another line, with no search
// and no request. A decline runs on to stops the line does not serve. A line that ended
// short, by a failure or a stop, ends where it really ended.
function routeStops(journey) {
  const answer = journey.answer;
  const searched = journey.searched ?? (answer === null ? null : searchedFor(answer.searched_for));
  const hit = answer === null ? null : cacheHit(answer);
  const asked = journey.earlier === 0 ? "by itself" : `with ${plural(journey.earlier, "earlier question", "earlier questions")}`;
  const stops = [{ name: "Asked", detail: asked, state: "reached" }];
  if (journey.earlier > 0) {
    stops.push(rewriteStop(searched));
  }
  const searchReached = searched !== null || journey.text || answer !== null;
  if (hit !== null) {
    const how = hit.tier === "exact" ? "an answer stored for the same question" : "an answer stored for a question judged to mean the same";
    stops.push({ name: "Found stored", detail: how, state: "reached" });
  } else {
    stops.push({ name: "Searched", detail: searchReached ? foundInWords(journey.found) : "", state: searchReached ? "reached" : "ahead" });
  }
  if (journey.ended !== null) {
    const reached = stops.filter((stop) => stop.state === "reached");
    if (journey.text) {
      reached.push({ name: "First words", detail: "arrived", state: "reached" });
    }
    reached.push(endStop(journey));
    return withSegments(reached);
  }
  if (answer === null) {
    stops.push({ name: "First words", detail: journey.text ? "arriving now" : "", state: journey.text ? "reached" : "ahead" });
    stops.push({ name: "Answered", detail: "", state: "ahead" });
    const next = stops.find((stop) => stop.state !== "reached");
    if (next !== undefined) {
      next.state = "next";
    }
    return withSegments(stops);
  }
  const first = answer.first_text_ms === null ? "no text came" : `after ${secondsInWords(answer.first_text_ms)}`;
  const byCheck = checkNotes(answer).some((note) => note.declined);
  if (answer.answered) {
    if (hit === null) {
      stops.push({ name: "First words", detail: first, state: "reached" });
    }
    stops.push({ name: "Answered", detail: answeredDetail(answer), state: "reached" });
  } else if (byCheck) {
    if (hit === null) {
      stops.push({ name: "First words", detail: first, state: "reached" });
    }
    stops.push({ name: "Declined by a check", detail: "nothing is cited", state: "off" });
  } else if (isDecline(answer)) {
    if (hit === null) {
      stops.push({ name: "No answer in them", detail: first, state: "off" });
    }
    stops.push({ name: "Not answered", detail: "nothing is cited", state: "off" });
  } else {
    if (hit === null && answer.first_text_ms !== null) {
      stops.push({ name: "First words", detail: first, state: "reached" });
    }
    stops.push(stoppedStop(answer));
  }
  return withSegments(stops);
}

// ---- Sources in brief -----------------------------------------------------------------

// The headings above a passage, outermost first, as far as they are text.
function headings(passage) {
  return Array.isArray(passage.heading_path)
    ? passage.heading_path.filter((heading) => typeof heading === "string" && heading !== "")
    : [];
}

// A short piece of a passage for its card: its lines without the Markdown headings, which
// the card shows above it already, joined into one line and cut at a word near the length
// given. It stays text: nothing in it is read as Markdown. The whole passage is folded
// away under the card.
function snippet(text, length = 200) {
  if (typeof text !== "string") {
    return "";
  }
  const words = text
    .split("\n")
    .filter((line) => !line.trimStart().startsWith("#"))
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  if (words.length <= length) {
    return words;
  }
  const cut = words.slice(0, length);
  const space = cut.lastIndexOf(" ");
  return `${(space > length / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;
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

// Whether a key press in the question box asks the question: Enter, as in any chat box,
// and Ctrl+Enter or Cmd+Enter as well. Shift+Enter starts a new line. An Enter that
// finishes a character being composed with an input method, as keyboards for some
// scripts do, asks nothing: the browser says so with isComposing, and Safari, which sends
// that Enter after the composition has ended, with the key code 229.
function isAskKey(event) {
  return event.key === "Enter" && !event.shiftKey && !event.isComposing && event.keyCode !== 229;
}

// The text that puts a message in the status line so that a screen reader announces it.
// The same text set again is not announced, so a repeated message gets a no-break space
// added, which changes the text and not what is read. The next repeat takes it off.
function announcement(previous, words) {
  return previous === words ? `${words} ` : words;
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
  failure: document.getElementById("failure"),
  welcome: document.getElementById("welcome"),
  turns: document.getElementById("turns"),
  conversationArea: document.getElementById("conversation-area"),
  conversationCount: document.getElementById("conversation-count"),
  conversationVerb: document.getElementById("conversation-verb"),
  conversation: document.getElementById("conversation"),
  newConversation: document.getElementById("new-conversation"),
};

const UNREADABLE_ANSWER = "The service sent an answer this page cannot read.";

// What the fold of found passages says about them, after the name of the request. It is
// the same for every turn.
const SEARCH_HINT =
  " returns for the question that was searched for, best first: for a follow-up, the question it was rewritten to. The answer runs this search before the model writes, and with the default settings it takes as many passages, so these are the passages the model was given. An answer that came from the cache ran no search: it was written earlier, from the passages found then for the question it names. Compare them with the sources the answer cites. This search costs no request to the model.";

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
// other kind leaves the title as plain text. The whole address is the link's title, shown
// when the link is pointed at.
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
  link.title = url;
  link.textContent = title;
  link.append(make("span", "visually-hidden", " (opens in a new tab)"));
  return link;
}

// A web address in brief: its host, then the parts of its path as a breadcrumb. When that
// runs long, the parts in the middle are left out: the last two are kept if they fit, and
// otherwise the last one. Only a last part that is long by itself is cut, in its middle.
// The host is always given whole, however long, since it says which site the link leads
// to: it is the host the browser reads, so a name before an @ is not shown as the host,
// and a host in another script shows as the browser holds it, in punycode. The line wraps
// on the page, so nothing of it is hidden. Anything that is not a web address is given
// whole.
function addressInBrief(url, length = 48) {
  const isWebAddress =
    typeof url === "string" && (url.startsWith("https://") || url.startsWith("http://"));
  if (!isWebAddress) {
    return shown(url);
  }
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return url;
  }
  const parts = parsed.pathname.split("/").filter((part) => part !== "");
  const crumbs = (list) => [parsed.host, ...list].join(" › ");
  if (parts.length === 0 || crumbs(parts).length <= length) {
    return crumbs(parts);
  }
  const last = partInBrief(parts.at(-1));
  if (parts.length > 2 && crumbs(["…", parts.at(-2), last]).length <= length) {
    return crumbs(["…", parts.at(-2), last]);
  }
  return crumbs(parts.length > 1 ? ["…", last] : [last]);
}

// One part of an address's path, cut in its middle only when it is long by itself.
function partInBrief(part, longest = 32) {
  const half = Math.floor((longest - 1) / 2);
  return part.length <= longest ? part : `${part.slice(0, half)}…${part.slice(-half)}`;
}

// Where a link leads, under its title, since the title can say anything: the address in
// brief, which is what is shown, and the whole address, which a screen reader reads in
// its place. The whole address is also the link's title, and is written out in full in
// the fold with the passage.
function addressLine(url) {
  const line = make("p", "source-address");
  const brief = make("span", "address-brief", addressInBrief(url));
  brief.setAttribute("aria-hidden", "true");
  line.append(brief, make("span", "visually-hidden", shown(url)));
  return line;
}

// A passage's whole text and the whole address of its page, folded away until asked for.
// Every fold shows the same words, so a screen reader is also told whose passage it is.
function passageFold(name, title, passage) {
  const fold = make("details", "passage");
  const summary = make("summary", "", "Show the passage");
  summary.append(make("span", "visually-hidden", ` of ${name}, ${title}`));
  const address = make("p", "passage-address", shown(passage.url));
  address.setAttribute("aria-hidden", "true");
  fold.append(summary, address, make("p", "passage-text", shown(passage.text)));
  return fold;
}

// A passage the search found: its rank, its title as a link to its page, the address of
// that page as text, the headings above it, its score, and its text folded away.
function passageItem(passage) {
  const rank = shown(passage.rank);
  const title = shown(passage.title);
  const score = Number.isFinite(passage.score) ? passage.score.toFixed(3) : "unknown";
  const item = make("li", "passage-item");
  const heading = make("p", "passage-title");
  heading.append(make("span", "passage-rank", `${rank}.`), " ", sourceLink(passage.url, title));
  item.append(heading);
  // The title is written by whoever edits the page, so the address is shown too, and the
  // reader sees where the link leads before following it.
  item.append(addressLine(passage.url));
  const above = headings(passage);
  // The headings joined as the command line joins them.
  if (above.length > 0) {
    item.append(make("p", "passage-headings", above.join(" > ")));
  }
  item.append(make("p", "passage-headings", `Score ${score}`));
  item.append(passageFold(`found passage ${rank}`, title, passage));
  return item;
}

// A source the answer cites, as one line of the list beside it: its number in the bullet
// of its line colour, the page's title as a link and the headings above the passage, a
// short piece of its text, the address of the page written out, and the whole passage
// folded away. Pointing at the line or moving the focus into it marks the numbers in the
// answer that cite it.
function sourceCard(turn, source) {
  const number = source.number;
  const title = shown(source.title);
  const card = make("li", `source-card ${lineClass(number)}`);
  const body = make("div", "source-body");
  const path = make("p", "source-path");
  path.append(sourceLink(source.url, title));
  body.append(path);
  const above = headings(source);
  if (above.length > 0) {
    const where = make("p", "source-where");
    for (const heading of above) {
      where.append(make("span", "crumb", heading));
    }
    body.append(where);
  }
  const piece = snippet(source.text);
  if (piece !== "") {
    body.append(make("p", "source-snippet", piece));
  }
  // The title is written by whoever edits the page, so the address is shown too, and the
  // reader sees where the link leads before following it.
  body.append(addressLine(source.url));
  body.append(passageFold(`source ${number}`, title, source));
  card.append(make("span", "source-number", String(number)), body);
  card.addEventListener("mouseenter", () => link(turn, number, true));
  card.addEventListener("mouseleave", () => link(turn, number, false));
  card.addEventListener("focusin", () => link(turn, number, true));
  card.addEventListener("focusout", () => link(turn, number, false));
  turn.cards.set(number, card);
  return card;
}

// A source's line, or, if it cannot be built, a line that says so: one source the page
// cannot show does not cost the reader the answer and the other sources. That line stands
// under the source's number, so that pressing the number still brings it into view.
function cardOrNotice(turn, source) {
  try {
    return sourceCard(turn, source);
  } catch (error) {
    console.error(error);
    const card = make("li", `source-card source-unreadable ${lineClass(source.number)}`);
    const body = make("div", "source-body");
    body.append(make("p", "source-path", "This source could not be shown."));
    card.append(make("span", "source-number", shown(source.number)), body);
    turn.cards.set(source.number, card);
    return card;
  }
}

// A passage the search found, or, if it cannot be built, an item that says so.
function passageOrNotice(passage) {
  try {
    return passageItem(passage);
  } catch (error) {
    console.error(error);
    const item = make("li", "passage-item passage-unreadable");
    const heading = make("p", "passage-title");
    heading.append(make("span", "passage-rank", `${shown(passage.rank)}.`), " This passage could not be shown.");
    item.append(heading);
    return item;
  }
}

// A number after a piece of the answer: the bullet of its source's line. Pointing at it or
// focusing it marks the line; pressing it brings the line into view.
function citeButton(turn, number, title) {
  const button = make("button", `cite ${lineClass(number)}`, String(number));
  button.type = "button";
  button.setAttribute("aria-label", `Source ${number}: ${title}`);
  button.addEventListener("click", () => showSource(turn, number));
  button.addEventListener("mouseenter", () => link(turn, number, true));
  button.addEventListener("mouseleave", () => link(turn, number, false));
  button.addEventListener("focus", () => link(turn, number, true));
  button.addEventListener("blur", () => link(turn, number, false));
  const chips = turn.chips.get(number) ?? [];
  chips.push(button);
  turn.chips.set(number, chips);
  return button;
}

// Shows a source's card and every number in the answer that cites it as one, while one
// of them is pointed at or focused.
function link(turn, number, on) {
  for (const chip of turn.chips.get(number) ?? []) {
    chip.classList.toggle("linked", on);
  }
  turn.cards.get(number)?.classList.toggle("linked", on);
}

// Brings a source's card into view and marks it. The focus stays on the number that was
// pressed, so the reader keeps their place in the answer and can press the next one,
// and the status line says which source is marked.
function showSource(turn, number) {
  for (const [each, card] of turn.cards) {
    card.classList.toggle("marked", each === number);
  }
  const card = turn.cards.get(number);
  if (card !== undefined) {
    card.scrollIntoView({ block: "nearest" });
    say(`Source ${number} is marked, in the sources of this answer.`);
  }
}

// A line that stands in for a source until the answer has come.
function placeholderCard() {
  const card = make("li", "source-card placeholder line-other");
  card.setAttribute("aria-hidden", "true");
  const body = make("div", "source-body");
  body.append(make("span", "skeleton-line"), make("span", "skeleton-line"));
  card.append(make("span", "source-number", ""), body);
  return card;
}

// One stop of a line, as an item of the list: its name, its detail, and for a stop that
// is not reached, in words for a screen reader, what state it is in.
const STOP_STATES = {
  reached: "",
  next: " (next stop)",
  ahead: " (not yet)",
  off: " (not served)",
  end: " (the line ends here)",
};

function stopItem(stop) {
  const item = make("li", `station station-${stop.state} seg-${stop.segment}`);
  item.append(make("span", "station-name", stop.name));
  if (stop.detail !== "") {
    item.append(make("span", "station-detail", stop.detail));
  }
  if (STOP_STATES[stop.state] !== "") {
    item.append(make("span", "visually-hidden", STOP_STATES[stop.state]));
  }
  return item;
}

// Draws a turn's line again from what is known of it.
function renderRoute(turn) {
  turn.route.replaceChildren(...routeStops(turn.journey).map(stopItem));
  turn.searchFold.hidden = !turn.searchRun || !searchReached(turn.journey);
}

// Whether a turn's line has come as far as the search: the service has said what it
// searched for, text has come, or the request has ended.
function searchReached(journey) {
  return journey.searched !== null || journey.text || journey.answer !== null || journey.ended !== null;
}

// ---- The turns ------------------------------------------------------------------------

// The turns on the page, oldest first: each question asked since the conversation began,
// with what came back for it. The conversation that is sent is another list, below: it
// holds only the questions of turns that were answered, as they were searched for.
const turns = [];

// A new turn for a question: the question as a sign, the line of stops under it, the
// reply beside the lines of its sources, and what the search found. Until something comes,
// placeholders stand where the text and the sources will be, and the line shows how far
// the answer has come.
function addTurn(question, earlier) {
  const turn = {
    question,
    chips: new Map(),
    cards: new Map(),
    waiting: true,
    foundPassages: null,
    citedKeys: null,
    journey: { earlier, searched: null, found: "waiting", text: false, answer: null, ended: null },
    // Whether the page's own search was run for this turn; its fold shows once the line
    // has come as far as the search.
    searchRun: false,
  };
  const item = make("li", "turn");
  const sign = make("h3", "question-sign");
  sign.append(make("span", "visually-hidden", "You asked: "), question);
  turn.route = make("ol", "route");
  turn.route.setAttribute("aria-label", "How this answer is being made");

  const card = make("div", "answer-card");
  turn.title = make("h4", "card-title visually-hidden", "Answer");
  turn.searchedLine = make("p", "searched");
  turn.searchedLine.hidden = true;
  turn.pending = make("div", "pending");
  turn.pending.setAttribute("aria-hidden", "true");
  turn.pending.append(make("span", "skeleton-line"), make("span", "skeleton-line"));
  turn.pending.append(make("span", "skeleton-line"));
  turn.stream = make("div", "stream");
  turn.stream.hidden = true;
  turn.body = make("div", "answer-body");
  turn.error = make("div", "error-card");
  turn.error.hidden = true;
  turn.errorText = make("p", "error-text");
  const again = make("button", "again", "Ask again");
  again.type = "button";
  again.addEventListener("click", () => start(question));
  turn.error.append(make("p", "error-title", "No answer."), turn.errorText, again);
  turn.note = make("p", "turn-note");
  turn.note.hidden = true;
  turn.made = make("div", "made");
  turn.made.hidden = true;
  turn.factsFold = make("details", "facts-fold");
  turn.facts = make("dl", "facts");
  turn.factsFold.append(make("summary", "", "All details"), turn.facts);
  turn.made.append(turn.factsFold);
  card.append(turn.title, turn.searchedLine, turn.pending, turn.stream, turn.body);
  card.append(turn.error, turn.note, turn.made);

  const side = make("div", "turn-sources");
  turn.side = side;
  turn.sourcesBlock = make("div", "sources-block");
  const sourcesTitle = make("h4", "visually-hidden", "Sources");
  turn.sourcesCount = make("span", "side-count", "");
  sourcesTitle.append(turn.sourcesCount);
  turn.sources = make("ol", "source-cards");
  turn.sources.append(placeholderCard(), placeholderCard());
  turn.sourcesBlock.append(sourcesTitle, turn.sources);
  turn.searchFold = make("details", "search-fold");
  turn.searchFold.hidden = true;
  const summary = make("summary", "");
  turn.searchCount = make("span", "side-count", "");
  summary.append(turn.searchCount);
  const hint = make("p", "hint");
  hint.append("The passages ", make("code", "", "POST /search"), SEARCH_HINT);
  turn.searchState = make("p", "search-state");
  turn.passages = make("ol", "passages");
  turn.searchFold.append(summary, hint, turn.searchState, turn.passages);
  side.append(turn.sourcesBlock, turn.searchFold);

  const cols = make("div", "cols");
  cols.append(card, side);
  item.append(sign, turn.route, cols);
  turn.item = item;
  renderRoute(turn);
  page.turns.append(item);
  turns.push(turn);
  updateChrome();
  // The first turn stands under the page's header already. A later one is brought to the
  // top, with room below it for the answer (the style gives the latest turn that room).
  if (turns.length > 1) {
    item.scrollIntoView({ block: "start" });
  }
  return turn;
}

// Text of the answer as it is written: the placeholder makes way for it, a caret stands at
// its end until the turn ends, and the line reaches its stop of the first words.
function addText(turn, text) {
  turn.pending.hidden = true;
  turn.stream.hidden = false;
  turn.stream.classList.add("streaming");
  turn.stream.append(text);
  if (!turn.journey.text) {
    turn.journey.text = true;
    renderRoute(turn);
  }
}

function renderAnswer(turn, answer) {
  const titles = new Map(answer.sources.map((source) => [source.number, shown(source.title)]));
  const parts = [];
  const hit = cacheHit(answer);
  if (hit !== null) {
    const note = make("div", "cached");
    note.append(make("p", "cached-label", "A stored answer."));
    for (const line of cacheLines(hit)) {
      note.append(make("p", "", line));
    }
    parts.push(note);
  }
  const text = make("p", "answer-text");
  const pieces = answerPieces(answer);
  for (const piece of pieces) {
    text.append(piece.text);
    for (const number of piece.numbers) {
      text.append(citeButton(turn, number, titles.get(number)));
    }
  }
  if (pieces.length === 0) {
    text.textContent = "The reply holds no text.";
  }
  const notes = checkNotes(answer);
  if (answer.answered) {
    parts.push(text);
    if (notes.length > 0) {
      parts.push(checkBox(notes));
    }
  } else {
    parts.push(notice(answer, text, notes));
  }
  const cards = answer.sources.map((source) => cardOrNotice(turn, source));
  if (cards.length === 0) {
    const none = answer.answered ? "This answer cites no source." : "Nothing is cited: this reply is not an answer.";
    cards.push(make("li", "sources-none", none));
  }
  const facts = answerFacts(answer);
  const cited = new Set(answer.sources.map(passageKey));
  // All is built before the page changes, so an answer that cannot be shown leaves the
  // streamed text in place. The final answer then takes the place of that text, which is
  // emptied and hidden.
  turn.waiting = false;
  label(turn, cardLabel(answer));
  turn.stream.replaceChildren();
  turn.stream.hidden = true;
  turn.body.replaceChildren(...parts);
  turn.sources.replaceChildren(...cards);
  turn.side.classList.toggle("no-sources", answer.sources.length === 0);
  turn.sourcesCount.textContent = answer.sources.length === 0 ? "" : ` ${answer.sources.length}`;
  // A follow-up is searched for once the service has said which question it searched
  // for. An answer that came without saying so ran no search on this page.
  if (!turn.searchRun) {
    turn.journey.found = "not run";
  }
  turn.journey.answer = answer;
  renderRoute(turn);
  turn.route.setAttribute("aria-label", answer.answered ? "How this answer was made" : "How this reply was made");
  turn.citedKeys = cited;
  renderUnused(turn);
  renderFacts(turn, facts, answer.trace_id);
  turn.made.hidden = false;
}

// A reply that is not an answer, as a calm notice and not as a failure: "I could not find
// this in the handbook" is the right reply to a question the handbook does not cover, and
// the sign of a line that does not run there stands beside it. A reply that was withheld
// or cut off ends its line, says so, and says why the model stopped. A reply that a check
// declined in place of what the model wrote is not one the handbook left unanswered: its
// notice says it was declined by a check, and the note says why.
function notice(answer, text, notes) {
  const declined = isDecline(answer);
  const byCheck = notes.some((note) => note.declined);
  const box = make("div", declined ? "notice notice-decline" : "notice notice-unfinished");
  const title = byCheck ? "Declined by a check" : declined ? "Not in the handbook" : "Not an answer";
  box.append(make("p", "notice-title", `${title}.`));
  box.append(text);
  const why = declined
    ? "The assistant found nothing in the handbook that answers this, and says so instead of guessing."
    : `The reply was withheld or did not come to a proper end. Why the model stopped: ${shown(answer.stop_reason)}.`;
  if (!byCheck) {
    box.append(make("p", "notice-why", why));
  }
  if (notes.length > 0) {
    box.append(checkBox(notes));
  }
  return box;
}

// The notes of the checks that changed a reply, as text, on the yellow of a notice of a
// change to the service: something was changed here, which is not a failure.
function checkBox(notes) {
  const box = make("div", "check-notes");
  box.append(make("p", "check-label", "Changed by a check."));
  for (const note of notes) {
    box.append(make("p", "check-note", note.note));
  }
  return box;
}

// The facts of the request, and its trace id with a button that copies it, as the
// timetable folded away under "All details".
function renderFacts(turn, rows, traceId) {
  turn.facts.replaceChildren();
  for (const [label, value] of rows) {
    turn.facts.append(make("dt", "", label), make("dd", "", value));
  }
  if (typeof traceId === "string" && traceId !== "") {
    const value = make("dd", "");
    value.append(make("code", "", traceId), " ", copyButton(traceId));
    turn.facts.append(make("dt", "", "Trace id"), value);
  }
  turn.factsFold.hidden = turn.facts.children.length === 0;
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

// Shows a failure in words, in its turn as a card that says what happened and offers to
// ask again, and in the alert region, which a screen reader reads at once. Text that had
// already arrived stays, marked as unfinished.
function fail(turn, words, kind = "failed") {
  const partial = turn.stream.textContent !== "";
  turn.stream.classList.toggle("unfinished", partial);
  const told = partial ? `${words} The text above is only what arrived before the failure.` : words;
  page.failure.textContent = told;
  turn.errorText.textContent = told;
  turn.error.hidden = false;
  label(turn, "Failed");
  turn.journey.ended = { kind };
  renderRoute(turn);
  turn.route.setAttribute("aria-label", "How this request went");
  say("No answer.");
}

// The word over a turn's card, and the class that colours its mark, by how it ended.
function label(turn, word) {
  if (turn.ending !== undefined) {
    turn.item.classList.remove(`ended-${turn.ending}`);
  }
  turn.ending = word.toLowerCase();
  turn.title.textContent = word;
  turn.item.classList.add(`ended-${turn.ending}`);
}

// What a reply is, in a word, for the label over its card.
function cardLabel(answer) {
  if (answer.answered) {
    return "Answer";
  }
  return isDecline(answer) ? "Declined" : "Reply";
}

// A failure that came with a trace id shows it under the reply, for a bug report.
function failedFacts(turn, traceId) {
  renderFacts(turn, [], traceId);
  turn.made.hidden = turn.factsFold.hidden;
}

// Ends the turn's waiting look, however the turn ended: the placeholders go and the caret
// stops. Sources that never came leave no empty list behind.
function settle(turn) {
  turn.pending.hidden = true;
  turn.stream.classList.remove("streaming");
  if (turn.waiting) {
    turn.sources.replaceChildren();
    turn.sourcesBlock.hidden = true;
  }
}

// Shows above the reply, as text, why a follow-up was searched for as it was typed. The
// question a follow-up was rewritten to stands at its stop of the line instead, so this
// line is left empty for it.
function showSearched(turn, searched, followUp) {
  const line = searchedLine(searched, followUp);
  turn.searchedLine.textContent = line;
  turn.searchedLine.hidden = line === "" || searched.outcome === "rewritten";
  turn.searchedLine.classList.toggle("searched-as-typed", line !== "" && !line.startsWith("Searched for:"));
}

// What the page shows around the turns: the welcome while there are none, and the button
// that starts again once there is something to start again from.
function updateChrome() {
  page.welcome.hidden = turns.length > 0;
  page.newConversation.hidden = turns.length === 0 && conversation.length === 0;
}

// Takes every turn off the page, for a new conversation.
function clearThread() {
  turns.length = 0;
  page.turns.replaceChildren();
  page.failure.textContent = "";
  updateChrome();
}

// ---- The conversation -----------------------------------------------------------------

// The questions of this conversation as the service searched for them, oldest first, in
// this page's memory only. The latest are sent with each question, up to the number the
// service accepts.
const conversation = [];
const EARLIER_SENT = 10;

function remember(question) {
  conversation.push(question);
  renderConversation();
}

function forget() {
  conversation.length = 0;
  renderConversation();
}

function renderConversation() {
  page.conversation.replaceChildren(...conversation.map((question) => make("li", "", question)));
  page.conversationArea.hidden = conversation.length === 0;
  const sent = Math.min(conversation.length, EARLIER_SENT);
  page.conversationCount.textContent = plural(sent, "earlier question", "earlier questions");
  page.conversationVerb.textContent = ` ${sent === 1 ? "is" : "are"} sent with your next question`;
  updateChrome();
}

// ---- Asking ---------------------------------------------------------------------------

// The request being answered, or null. One at a time: a second question waits until the
// first is answered or stopped, so nothing of the first can land in the second's answer.
let current = null;

// While an answer is being written, the stop button shows beside Ask, and Ask shows that
// it is busy. Ask and the examples are marked unavailable with aria-disabled instead of
// disabled, so that the button a keyboard user just pressed keeps the focus. The stop
// button goes once the answer has ended; if it held the focus, the focus goes back to the
// question box instead of being lost.
function setBusy(busy) {
  page.ask.setAttribute("aria-disabled", String(busy));
  for (const example of page.examples) {
    example.setAttribute("aria-disabled", String(busy));
  }
  if (!busy && document.activeElement === page.stop) {
    page.question.focus();
  }
  page.stop.hidden = !busy;
  page.form.classList.toggle("busy", busy);
}

async function ask(question) {
  const controller = new AbortController();
  current = controller;
  setBusy(true);
  page.failure.textContent = "";
  const earlier = conversation.slice(-EARLIER_SENT);
  const turn = addTurn(question, earlier.length);
  // A question asked by itself is searched for as it is. A follow-up is searched for once
  // the service has said which question it searched for.
  if (earlier.length === 0) {
    search(turn, question);
  }
  say("Asking. The answer appears in the conversation as the model writes it.");
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question, stream: true, earlier }),
      signal: controller.signal,
    });
    if (response.ok) {
      await readAnswer(turn, response, question, earlier.length > 0);
    } else {
      await refused(turn, response);
    }
  } catch (error) {
    if (controller.signal.aborted) {
      stopped(turn);
    } else {
      // A network failure, or a mistake in this script: the console says which.
      console.error(error);
      fail(
        turn,
        turn.stream.textContent === ""
          ? "The service could not be reached. Is it running?"
          : "The connection to the service broke.",
      );
    }
  } finally {
    settle(turn);
    current = null;
    setBusy(false);
  }
}

async function refused(turn, response) {
  const body = await response.json().catch(() => null);
  fail(turn, refusalInWords(response.status, body, response.headers.get("Retry-After")));
  failedFacts(turn, response.headers.get("X-Trace-Id"));
}

// Reads the events of the answer: the question searched for, text as it is written, then
// the answer or an error. The text is not announced piece by piece: the status line says
// that the model is writing, and when the answer is complete.
async function readAnswer(turn, response, question, followUp) {
  const traceId = response.headers.get("X-Trace-Id");
  const contentType = response.headers.get("Content-Type");
  if (mediaType(contentType) !== "text/event-stream") {
    fail(turn, notAStream(contentType), "unreadable");
    failedFacts(turn, traceId);
    return;
  }
  let ended = false;
  await readEvents(response.body, (event) => {
    if (ended) {
      return;
    }
    const data = parseData(event.data);
    if (event.name === "searched") {
      const searched = searchedFor(data);
      if (searched !== null) {
        showSearched(turn, searched, followUp);
        turn.journey.searched = searched;
        renderRoute(turn);
        if (followUp) {
          search(turn, searched.question);
        }
      }
    } else if (event.name === "text" && data !== null && typeof data.text === "string") {
      if (turn.stream.textContent === "") {
        say("The model is writing.");
      }
      addText(turn, data.text);
    } else if (event.name === "answer") {
      ended = true;
      if (showAnswer(turn, data)) {
        const searched = searchedFor(data.searched_for);
        remember(searched === null ? question : searched.question);
      }
    } else if (event.name === "error") {
      ended = true;
      const detail = data !== null && typeof data.detail === "string" ? data.detail : "";
      fail(turn, detail === "" ? "The answer could not be finished." : detail);
      failedFacts(turn, traceId);
    }
  });
  if (!ended) {
    fail(turn, "The answer stopped before it was complete: the stream ended without it.", "cut");
    failedFacts(turn, traceId);
  }
}

// Shows the answer event, or says that it cannot be read. Its shape is checked first,
// and a fault while building it is caught too, so a reader is never told that the
// connection broke when it was the answer that could not be read. True when it is shown.
function showAnswer(turn, data) {
  if (!isAnswer(data)) {
    fail(turn, UNREADABLE_ANSWER, "unreadable");
    return false;
  }
  try {
    renderAnswer(turn, data);
  } catch (error) {
    console.error(error);
    fail(turn, UNREADABLE_ANSWER, "unreadable");
    return false;
  }
  const from = cacheHit(data) === null ? "" : " It came from the cache.";
  let words = `The answer is complete. It cites ${plural(citedNumbers(data).size, "source", "sources")}.`;
  if (!data.answered) {
    words = isDecline(data)
      ? "Not in the handbook: the assistant declined instead of guessing."
      : "Not an answer: the reply was withheld or did not come to a proper end.";
  }
  const notes = checkNotes(data);
  if (notes.some((note) => note.declined)) {
    words = "Declined by a check, in place of the reply the model wrote.";
  }
  // Each note is read out with the outcome: it is the only sign that the reply is not all
  // the model wrote.
  const noted = notes.map((note) => ` Note: ${note.note}`).join("");
  say(`${words}${from}${noted}`);
  return true;
}

function stopped(turn) {
  // Stop can be pressed in the moment after the answer or an error event arrived and
  // before the stream closed. The request had ended by then, and its outcome stands.
  if (turn.body.children.length > 0 || !turn.error.hidden) {
    return;
  }
  const partial = turn.stream.textContent !== "";
  turn.stream.classList.toggle("unfinished", partial);
  const words = partial
    ? "Stopped. The text above is only what arrived before that."
    : "Stopped before the model wrote anything.";
  turn.note.textContent = words;
  turn.note.hidden = false;
  label(turn, "Stopped");
  turn.journey.ended = { kind: "stopped" };
  renderRoute(turn);
  turn.route.setAttribute("aria-label", "How this request went");
  say(words);
}

// ---- What the search found ------------------------------------------------------------

// The search for the question, run beside the answer. It costs no request to the model.
// Each turn has its own, so a search that comes back late fills the turn it was run for.
// How many passages it found is the detail of the line's stop of the search.
async function search(turn, question) {
  turn.searchRun = true;
  turn.searchCount.textContent = "The page's own search is running.";
  turn.searchFold.classList.add("nothing-more");
  turn.searchState.textContent = "Searching.";
  turn.passages.replaceChildren();
  turn.journey.found = "running";
  renderRoute(turn);
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
    turn.searchState.textContent = "The search failed. The service could not be reached.";
    searchFailed(turn);
    return;
  }
  if (response.ok && isObject(body) && Array.isArray(body.results)) {
    renderPassages(turn, body.results.filter(isObject));
  } else {
    const words = refusalInWords(response.status, body, response.headers.get("Retry-After"));
    turn.searchState.textContent = `The search failed. ${words}`;
    searchFailed(turn);
  }
}

// A search of the page's own that failed: its fold's control says so, and opens on why.
function searchFailed(turn) {
  turn.searchCount.textContent = "The page's own search failed.";
  turn.searchFold.classList.add("nothing-more");
  turn.journey.found = "failed";
  renderRoute(turn);
}

function renderPassages(turn, results) {
  turn.foundPassages = results;
  turn.journey.found = results.length;
  turn.searchState.textContent = results.length === 0 ? "The search found nothing." : "";
  renderRoute(turn);
  renderUnused(turn);
}

// A passage, as the page tells whether the answer cites it: by its address and its text.
function passageKey(passage) {
  return `${shown(passage.url)}\n${shown(passage.text)}`;
}

// The passages the search found and the answer does not cite: stops the line does not
// serve, folded away. Before the answer has come, every passage found is listed.
function renderUnused(turn) {
  const found = turn.foundPassages;
  if (found === null) {
    return;
  }
  const cited = turn.citedKeys;
  const unused = cited === null ? found : found.filter((passage) => !cited.has(passageKey(passage)));
  let count = `The page's own search found ${plural(found.length, "passage", "passages")}.`;
  if (found.length === 0) {
    count = "The page's own search found nothing.";
  } else if (cited !== null && turn.journey.answer !== null) {
    if (unused.length === 0) {
      count = "Every passage found is cited.";
    } else if (turn.journey.answer.answered) {
      count = `${plural(unused.length, "more passage was", "more passages were")} found and not cited.`;
    } else {
      count = `${plural(unused.length, "passage was", "passages were")} found, and none is cited.`;
    }
  }
  turn.searchCount.textContent = count;
  // With nothing left to list, the fold offers only what the search is.
  turn.searchFold.classList.toggle("nothing-more", unused.length === 0);
  turn.passages.replaceChildren(...unused.map(passageOrNotice));
}

// ---- Wiring ---------------------------------------------------------------------------

function showCount() {
  page.count.textContent = `${page.question.value.length} of ${page.question.maxLength} characters`;
}

// Asks a question, unless one is being answered or there is none. True when it is asked.
function start(question) {
  if (current !== null) {
    say("An answer is still being written. Stop it first, or wait for it.");
    return false;
  }
  if (question === "") {
    say("Type a question first.");
    page.question.focus();
    return false;
  }
  ask(question);
  return true;
}

// Asks the question in the box, and empties the box once the question is on its way: the
// conversation shows it as it was asked. A question over the limit is not sent: the box
// holds no more than that, so only a script could have put it there.
function askFromBox() {
  if (page.question.value.length > page.question.maxLength) {
    say(`The question is longer than ${page.question.maxLength} characters.`);
    return;
  }
  if (start(page.question.value.trim())) {
    page.question.value = "";
    showCount();
  }
}

page.form.addEventListener("submit", (event) => {
  event.preventDefault();
  askFromBox();
});

// Asking from the keyboard without leaving the box.
page.question.addEventListener("keydown", (event) => {
  if (isAskKey(event)) {
    event.preventDefault();
    askFromBox();
  }
});

// An example question is asked as it is, and shows in the conversation like any other.
for (const example of page.examples) {
  example.addEventListener("click", () => start(example.textContent.trim()));
}

page.stop.addEventListener("click", () => {
  if (current === null) {
    say("No answer is being written.");
    return;
  }
  current.abort();
});

page.newConversation.addEventListener("click", () => {
  if (current !== null) {
    say("An answer is still being written. Stop it first, or wait for it.");
    return;
  }
  clearThread();
  forget();
  say("A new conversation: the next question is asked by itself.");
  page.question.focus();
});

page.question.addEventListener("input", showCount);
showCount();
setBusy(false);
