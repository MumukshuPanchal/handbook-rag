// Runs the demo page's script in a small fake page, with fetch stubbed, through one
// conversation, and prints what the page did after each step as JSON. The Python test
// tests/test_demo_behaviour.py runs it with node and reads the result. It needs node and
// nothing else: no package, no browser. The fake page has only what the script touches.
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(process.argv[2], "utf8");

function element(id) {
  const node = {
    id,
    textContent: "",
    hidden: false,
    value: "",
    maxLength: 500,
    children: [],
    className: "",
    type: "",
    listeners: {},
    classList: { add() {}, remove() {}, toggle() {} },
    setAttribute() {},
    getAttribute() {
      return null;
    },
    addEventListener(type, handler) {
      node.listeners[type] = handler;
    },
    focus() {},
    scrollIntoView() {},
    append(...items) {
      for (const item of items) {
        node.children.push(item);
        node.textContent += typeof item === "string" ? item : item.textContent || "";
      }
    },
    replaceChildren(...items) {
      node.children = [];
      node.textContent = "";
      node.append(...items);
    },
  };
  return node;
}

const elements = {};
const document = {
  getElementById(id) {
    elements[id] ??= element(id);
    return elements[id];
  },
  querySelectorAll() {
    return [];
  },
  createElement(tag) {
    return element(tag);
  },
};

// What /ask answers, in order: a status, a body and a content type.
const replies = [];
const asked = [];
const searchedFor = [];

async function fetchStub(url, options) {
  const body = JSON.parse(options.body);
  if (url === "/search") {
    searchedFor.push(body.query);
    const empty = { results: [], mode: "hybrid", reranked: false, took_ms: 1, trace_id: null };
    return new Response(JSON.stringify(empty), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  asked.push(body);
  const [status, text, type] = replies.shift();
  return new Response(text, { status, headers: { "Content-Type": type } });
}

const context = vm.createContext({
  document,
  fetch: fetchStub,
  console,
  AbortController,
  TextDecoder,
  Response,
  navigator: {},
});
vm.runInContext(`${source}\n;globalThis.__page = { ask, conversation, page };`, context);
const { ask, conversation, page } = context.__page;

function searched(asked, question, outcome, extra = {}) {
  return {
    asked,
    question,
    outcome,
    failure: null,
    earlier: 0,
    requests: 0,
    usage: { input_tokens: 0, output_tokens: 0 },
    took_ms: 0,
    ...extra,
  };
}

function answer(searchedFor) {
  return {
    answer: "Sixteen weeks.",
    answered: true,
    segments: [{ text: "Sixteen weeks.", sources: [] }],
    sources: [],
    model: "stand-in",
    stop_reason: "end_turn",
    usage: { input_tokens: 1, output_tokens: 1 },
    requests: 1,
    fallback: false,
    took_ms: 1,
    first_text_ms: 1,
    trace_id: null,
    cached: null,
    searched_for: searchedFor,
  };
}

function stream(events) {
  const text = events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`);
  return [200, text.join(""), "text/event-stream; charset=utf-8"];
}

function answered(asked, question, outcome) {
  const found = searched(asked, question, outcome);
  return stream([
    ["searched", found],
    ["text", { text: "Sixteen weeks." }],
    ["answer", answer(found)],
  ]);
}

const steps = [];

async function step(name, question, reply) {
  const searchesBefore = searchedFor.length;
  if (reply !== null) {
    replies.push(reply);
  }
  if (question !== null) {
    await ask(question);
  }
  steps.push({
    name,
    sentEarlier: question === null ? null : asked.at(-1).earlier,
    searches: searchedFor.slice(searchesBefore),
    conversation: [...conversation],
    searchedLine: page.searched.textContent,
    searchedHidden: page.searched.hidden,
    failure: page.failure.textContent,
    listed: page.conversation.children.map((item) => item.textContent),
    listHidden: page.conversationArea.hidden,
  });
}

const FIRST = "How many days of paid annual leave do UK team members get?";
const REWRITTEN = "How many days of paid annual leave do team members in Finland get?";

await step("first", FIRST, answered(FIRST, FIRST, "not_asked"));
await step("rewritten", "and in finland", answered("and in finland", REWRITTEN, "rewritten"));
await step(
  "failed rewrite",
  "and in spain",
  stream([
    ["searched", searched("and in spain", "and in spain", "failed", { failure: "timeout", requests: 1 })],
    ["text", { text: "Sixteen weeks." }],
    ["answer", answer(searched("and in spain", "and in spain", "failed", { failure: "timeout", requests: 1 }))],
  ]),
);
await step("unusable rewrite", "and in italy", answered("and in italy", "and in italy", "unusable"));
await step("given back", "Who approves it?", answered("Who approves it?", "Who approves it?", "unchanged"));
await step(
  "error after text",
  "and in portugal",
  stream([
    ["searched", searched("and in portugal", "How many days in Portugal?", "rewritten")],
    ["text", { text: "Sixteen" }],
    ["error", { detail: "The answer model is unavailable. Try again later." }],
  ]),
);
await step(
  "unreadable answer",
  "and in france",
  stream([
    ["searched", searched("and in france", "How many days in France?", "rewritten")],
    ["text", { text: "Sixteen" }],
    ["answer", { broken: true }],
  ]),
);
await step("refused", "and in ireland", [
  503,
  JSON.stringify({ detail: "The service is busy." }),
  "application/json",
]);
await step("not rewritten", "and in germany", answered("and in germany", "and in germany", "not_asked"));
elements["new-conversation"].listeners.click();
await step("new conversation", null, null);
await step("asked by itself", FIRST, answered(FIRST, FIRST, "not_asked"));

console.log(JSON.stringify(steps));
