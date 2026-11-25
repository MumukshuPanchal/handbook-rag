// Runs the demo page's script in a small fake page, with fetch stubbed, through one
// conversation, and prints what the page did after each step as JSON. The Python test
// tests/test_demo_behaviour.py runs it with node and reads the result. It needs node and
// nothing else: no package, no browser. The fake page has only what the script touches:
// elements with children, classes, the hidden flag and listeners. What the harness reads
// back, it reads from those elements, as a reader would see them, and not from the
// script's own variables.
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(process.argv[2], "utf8");

function element(tag) {
  const node = {
    tagName: String(tag).toUpperCase(),
    nodes: [],
    parent: null,
    hidden: false,
    value: "",
    maxLength: 500,
    type: "",
    listeners: {},
    classes: new Set(),
    attrs: {},
    scrolled: 0,
    get children() {
      return this.nodes.filter((child) => typeof child !== "string");
    },
    get className() {
      return [...this.classes].join(" ");
    },
    set className(value) {
      this.classes = new Set(String(value).split(/\s+/).filter(Boolean));
    },
    get textContent() {
      return this.nodes.map((child) => (typeof child === "string" ? child : child.textContent)).join("");
    },
    set textContent(value) {
      const text = String(value);
      this.replaceChildren(...(text === "" ? [] : [text]));
    },
    setAttribute(name, value) {
      this.attrs[name] = String(value);
    },
    getAttribute(name) {
      return this.attrs[name] ?? null;
    },
    addEventListener(type, handler) {
      (this.listeners[type] ??= []).push(handler);
    },
    focus() {},
    scrollIntoView() {
      this.scrolled += 1;
    },
    append(...items) {
      for (const item of items) {
        if (typeof item !== "string") {
          item.remove();
          item.parent = this;
        }
        this.nodes.push(item);
      }
    },
    replaceChildren(...items) {
      for (const child of this.children) {
        child.parent = null;
      }
      this.nodes = [];
      this.append(...items);
    },
    remove() {
      if (this.parent !== null) {
        this.parent.nodes = this.parent.nodes.filter((child) => child !== this);
        this.parent = null;
      }
    },
  };
  node.classList = {
    add: (...names) => names.forEach((name) => node.classes.add(name)),
    remove: (...names) => names.forEach((name) => node.classes.delete(name)),
    toggle(name, force) {
      const on = force === undefined ? !node.classes.has(name) : Boolean(force);
      if (on) {
        node.classes.add(name);
      } else {
        node.classes.delete(name);
      }
      return on;
    },
    contains: (name) => node.classes.has(name),
  };
  return node;
}

// What a reader can do to an element: point at it, focus it, press it.
function fire(node, type) {
  for (const handler of node.listeners[type] ?? []) {
    handler({ type, target: node, preventDefault() {} });
  }
}

// Every element under a node, depth first, that passes a test.
function all(node, test, found = []) {
  for (const child of node.children) {
    if (test(child)) {
      found.push(child);
    }
    all(child, test, found);
  }
  return found;
}

function byClass(node, name) {
  return node === undefined ? [] : all(node, (child) => child.classes.has(name));
}

// Whether an element is on the page and shown: neither it nor anything above it hidden.
function isShown(node) {
  for (let at = node; at !== null; at = at.parent) {
    if (at.hidden) {
      return false;
    }
  }
  return true;
}

function shownByClass(node, name) {
  return byClass(node, name).filter(isShown);
}

// Whether an element sits inside one with a class.
function inside(node, name) {
  for (let at = node.parent; at !== null; at = at.parent) {
    if (at.classes.has(name)) {
      return true;
    }
  }
  return false;
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

// What /ask answers, in order: a status, a body, a content type and any other headers. A
// body that is held is a stream the harness writes to while the page waits, as the
// service would.
const replies = [];
const asked = [];
const searchedFor = [];
// The passages /search finds, by question. Any other question finds nothing.
const passagesFor = new Map();

async function fetchStub(url, options) {
  const body = JSON.parse(options.body);
  if (url === "/search") {
    searchedFor.push(body.query);
    const results = passagesFor.get(body.query) ?? [];
    const found = { results, mode: "hybrid", reranked: false, took_ms: 1, trace_id: null };
    return new Response(JSON.stringify(found), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }
  asked.push(body);
  if (replies.length === 0) {
    // A question the harness did not expect to be sent: answered, so that the run goes on
    // and the test sees that it was sent.
    return new Response(JSON.stringify({ detail: "Not expected." }), { status: 500 });
  }
  const [status, text, type, headers = {}] = replies.shift();
  if (typeof text === "string") {
    return new Response(text, { status, headers: { "Content-Type": type, ...headers } });
  }
  // A held stream breaks off when the request is stopped, as a real fetch does.
  options.signal.addEventListener("abort", () => text.fail());
  return new Response(text.body, { status, headers: { "Content-Type": type } });
}

const context = vm.createContext({
  document,
  fetch: fetchStub,
  console,
  AbortController,
  TextDecoder,
  Response,
  URL,
  navigator: {},
});
vm.runInContext(
  `${source}\n;globalThis.__page = { ask, conversation, page, isBusy: () => current !== null };`,
  context,
);
const { ask, conversation, page, isBusy } = context.__page;

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

function answer(searchedFor, extra = {}) {
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
    ...extra,
  };
}

function event(name, data) {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

function stream(events) {
  return [200, events.map(([name, data]) => event(name, data)).join(""), "text/event-stream; charset=utf-8"];
}

function answered(asked, question, outcome) {
  const found = searched(asked, question, outcome);
  return stream([
    ["searched", found],
    ["text", { text: "Sixteen weeks." }],
    ["answer", answer(found)],
  ]);
}

// A stream that the harness writes to, piece by piece, and ends or breaks off.
function held() {
  const encoder = new TextEncoder();
  let control;
  const body = new ReadableStream({
    start(controller) {
      control = controller;
    },
  });
  return {
    body,
    send: (name, data) => control.enqueue(encoder.encode(event(name, data))),
    end: () => control.close(),
    fail: () => control.error(new DOMException("The request was stopped.", "AbortError")),
  };
}

// Lets the page's pending work run: a reply read, a search answered.
async function settle() {
  for (let round = 0; round < 10; round += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

// The latest turn on the page, as a reader sees it.
function latestTurn() {
  const item = page.turns.children.at(-1);
  if (item === undefined) {
    return { turnSearchedLine: "", turnSearchedHidden: true };
  }
  const [line] = byClass(item, "searched");
  const [error] = byClass(item, "error-card");
  const cards = byClass(item, "source-card").filter((card) => !card.classes.has("placeholder"));
  const answerText = shownByClass(item, "answer-text");
  const [made] = byClass(item, "made");
  const badgeList = made !== undefined && isShown(made) ? shownByClass(made, "badge") : [];
  const [note] = byClass(item, "turn-note");
  const [madeTitle] = byClass(item, "made-title");
  const [cardTitle] = byClass(item, "card-title");
  const [facts] = byClass(item, "facts");
  return {
    turnSearchedLine: line.textContent,
    turnSearchedHidden: !isShown(line),
    bubble: byClass(item, "bubble").map((bubble) => bubble.textContent),
    // An answer's text, shown as an answer and not inside the notice of a reply that is not.
    answer: answerText.some((text) => !inside(text, "notice")),
    answerText: answerText.map((text) => text.textContent),
    decline: shownByClass(item, "notice-decline").length > 0,
    unfinished: shownByClass(item, "notice-unfinished").length > 0,
    notice: shownByClass(item, "notice").map((box) => box.textContent),
    error: shownByClass(item, "error-card").length > 0,
    errorText: error !== undefined ? byClass(error, "error-text").map((text) => text.textContent).join("") : "",
    pending: shownByClass(item, "pending").length > 0,
    placeholders: shownByClass(item, "placeholder").length,
    caret: shownByClass(item, "stream").some((text) => text.classes.has("streaming")),
    streamed: shownByClass(item, "stream").map((text) => text.textContent).join(""),
    progress: shownByClass(item, "card-progress").length > 0,
    badges: badgeList.map((badge) => badge.textContent),
    cached: shownByClass(item, "cached").map((box) => box.textContent).join(" "),
    note: note !== undefined && isShown(note) ? note.textContent : "",
    madeTitle: made !== undefined && isShown(made) ? madeTitle.textContent : "",
    cardTitle: cardTitle.textContent,
    ended: [...item.classes].filter((name) => name.startsWith("ended-")),
    facts: facts !== undefined && isShown(facts) ? facts.textContent : "",
    chips: shownByClass(item, "cite").map((chip) => chip.textContent),
    cards: cards.filter(isShown).map((card) => {
      const links = all(card, (child) => child.tagName === "A");
      return {
        number: byClass(card, "source-number").map((number) => number.textContent).join(""),
        link: links.length === 0 ? null : links[0].href,
        linkTarget: links.length === 0 ? null : links[0].target,
        linkRel: links.length === 0 ? null : links[0].rel,
        path: byClass(card, "source-path").map((path) => path.textContent).join(""),
        crumbs: byClass(card, "crumb").map((crumb) => crumb.textContent),
        snippet: byClass(card, "source-snippet").map((piece) => piece.textContent).join(""),
        linkTitle: links.length === 0 ? null : (links[0].title ?? null),
        brief: byClass(card, "address-brief").map((brief) => brief.textContent).join(""),
        briefHiddenFromReaders: byClass(card, "address-brief").every((brief) => brief.attrs["aria-hidden"] === "true"),
        // The whole address as a screen reader reads it, beside the brief one.
        address: byClass(card, "source-address")
          .flatMap((line) => byClass(line, "visually-hidden"))
          .map((whole) => whole.textContent)
          .join(""),
        foldAddress: byClass(card, "passage-address").map((whole) => whole.textContent).join(""),
      };
    }),
  };
}

const steps = [];

function record(name, question) {
  steps.push({
    name,
    question,
    sentEarlier: question === null ? null : asked.at(-1).earlier,
    conversation: [...conversation],
    // The line about the question searched for, in the latest turn.
    searchedLine: latestTurn().turnSearchedLine,
    searchedHidden: latestTurn().turnSearchedHidden,
    failure: page.failure.textContent,
    listed: page.conversation.children.map((item) => item.textContent),
    listHidden: page.conversationArea.hidden,
    turnCount: page.turns.children.length,
    stopHidden: page.stop.hidden,
    askDisabled: page.ask.getAttribute("aria-disabled"),
    welcomeHidden: page.welcome.hidden,
    status: page.status.textContent,
    ...latestTurn(),
  });
}

async function step(name, question, reply) {
  const searchesBefore = searchedFor.length;
  if (reply !== null) {
    replies.push(reply);
  }
  if (question !== null) {
    await ask(question);
    await settle();
  }
  record(name, question);
  steps.at(-1).searches = searchedFor.slice(searchesBefore);
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
  { "X-Trace-Id": "4be69ac5bb0e6069eaa37e98ae312974" },
]);
await step("not rewritten", "and in germany", answered("and in germany", "and in germany", "not_asked"));
fire(elements["new-conversation"], "click");
await step("new conversation", null, null);
await step("asked by itself", FIRST, answered(FIRST, FIRST, "not_asked"));

// An answer that cites two sources, one of them twice, and names a third that it does
// not list. The search beside it finds three passages.
const US = "How many weeks of parental leave do new parents get in the US?";
const LEAVE_URL =
  "https://handbook.example.com/handbook/total-rewards/benefits/general-and-entity-benefits/leave-in-the-united-states/";
passagesFor.set(US, [1, 2, 3].map((rank) => ({
  rank,
  score: 0.03,
  doc_id: "people/leave.md",
  title: "Leave",
  url: LEAVE_URL,
  section: "people",
  heading_path: ["Parental leave"],
  text: "New parents get sixteen weeks.",
})));
const cited = searched("and in the us", US, "rewritten", { requests: 1, took_ms: 1200 });
await step(
  "cited",
  "and in the us",
  stream([
    ["searched", cited],
    ["text", { text: "New parents get sixteen weeks. Leave is paid in full." }],
    [
      "answer",
      answer(cited, {
        segments: [
          { text: "New parents get sixteen weeks.", sources: [1] },
          { text: " Leave is paid in full.", sources: [1, 2] },
          { text: " A number it does not list.", sources: [7] },
        ],
        sources: [
          {
            number: 1,
            doc_id: "people/leave.md",
            title: "Leave",
            url: LEAVE_URL,
            section: "people",
            heading_path: ["Parental leave", "United States"],
            text: "## Parental leave\n\nNew parents get sixteen weeks\nof paid leave, in full.",
          },
          {
            number: 2,
            doc_id: "people/faq.md",
            title: "Leave FAQ",
            url: "ftp://handbook.example.com/faq",
            section: "people",
            heading_path: [],
            text: "Paid in full.",
          },
        ],
      }),
    ],
  ]),
);

// What pointing, focusing and pressing do to the numbers in that answer and its cards.
{
  const item = page.turns.children.at(-1);
  const chips = byClass(item, "cite");
  const cards = byClass(item, "source-card").filter((card) => !card.classes.has("placeholder"));
  const numberOf = (card) => byClass(card, "source-number")[0].textContent;
  const chip = (number) => chips.find((each) => each.textContent === number);
  const card = (number) => cards.find((each) => numberOf(each) === number);
  const markedCards = () => cards.filter((each) => each.classes.has("linked")).map(numberOf);
  const markedChips = () => chips.filter((each) => each.classes.has("linked")).map((each) => each.textContent);
  const linking = {};
  fire(chip("2"), "mouseenter");
  linking.pointChip = { cards: markedCards(), chips: markedChips() };
  fire(chip("2"), "mouseleave");
  linking.leaveChip = { cards: markedCards(), chips: markedChips() };
  fire(chip("1"), "focus");
  linking.focusChip = { cards: markedCards(), chips: markedChips() };
  fire(chip("1"), "blur");
  linking.blurChip = { cards: markedCards(), chips: markedChips() };
  fire(card("1"), "mouseenter");
  linking.pointCard = { cards: markedCards(), chips: markedChips() };
  fire(card("1"), "mouseleave");
  linking.leaveCard = { cards: markedCards(), chips: markedChips() };
  fire(card("2"), "focusin");
  linking.focusCard = { cards: markedCards(), chips: markedChips() };
  fire(card("2"), "focusout");
  linking.unfocusCard = { cards: markedCards(), chips: markedChips() };
  const before = card("2").scrolled;
  fire(chip("2"), "click");
  linking.pressChip = {
    scrolled: card("2").scrolled - before,
    marked: cards.filter((each) => each.classes.has("marked")).map(numberOf),
    status: page.status.textContent,
  };
  steps.at(-1).linking = linking;
}

const NOT_FOUND = "I could not find this in the handbook.";
function declined(asked, extra = {}) {
  const found = searched(asked, asked, "unchanged");
  return stream([
    ["searched", found],
    ["text", { text: NOT_FOUND }],
    [
      "answer",
      answer(found, {
        answer: NOT_FOUND,
        answered: false,
        segments: [{ text: NOT_FOUND, sources: [] }],
        ...extra,
      }),
    ],
  ]);
}
const STORED = "Do Finland team members paid annual leave days carry over to the next year?";
await step("declined", "How much is the gym budget?", declined("How much is the gym budget?"));
await step(
  "cached",
  "do they carry over",
  declined("do they carry over", {
    usage: { input_tokens: 0, output_tokens: 0 },
    requests: 0,
    cached: { tier: "exact", question: STORED, similarity: null, score: null, stored_at: "2026-11-25T11:52:07+00:00" },
  }),
);
await step("withheld", "Is that withheld?", declined("Is that withheld?", { stop_reason: "safety" }));

// An answer watched as it comes: before anything, while its text is written, and done.
{
  const body = held();
  replies.push([200, body, "text/event-stream; charset=utf-8"]);
  const asking = ask("Who approves parental leave?");
  await settle();
  record("waiting", "Who approves parental leave?");
  const found = searched("Who approves parental leave?", "Who approves parental leave?", "unchanged");
  body.send("searched", found);
  body.send("text", { text: "Your manager" });
  await settle();
  record("writing", "Who approves parental leave?");
  body.send("answer", answer(found, { segments: [{ text: "Your manager approves it.", sources: [] }] }));
  body.end();
  await asking;
  await settle();
  record("arrived", "Who approves parental leave?");
}

// An answer stopped while its text is written.
{
  const body = held();
  replies.push([200, body, "text/event-stream; charset=utf-8"]);
  const asking = ask("Tell me about leave");
  body.send("searched", searched("Tell me about leave", "Tell me about leave", "unchanged"));
  body.send("text", { text: "Leave is" });
  await settle();
  fire(elements.stop, "click");
  await asking;
  await settle();
  record("stopped", "Tell me about leave");
}

await step(
  "stream cut",
  "and the rest",
  stream([
    ["searched", searched("and the rest", "And the rest of the leave?", "rewritten")],
    ["text", { text: "Part of" }],
  ]),
);

// The keys of the question box. A key press is given to the box's listeners as the
// browser gives it, and what the harness reads is whether a question was sent, and
// whether the key's own effect, such as a new line, was prevented.
function press(init) {
  const event = {
    key: "Enter",
    shiftKey: false,
    ctrlKey: false,
    metaKey: false,
    isComposing: false,
    keyCode: 13,
    prevented: false,
    ...init,
    preventDefault() {
      this.prevented = true;
    },
  };
  for (const handler of page.question.listeners.keydown ?? []) {
    handler(event);
  }
  return event;
}

async function untilIdle() {
  for (let round = 0; round < 500 && isBusy(); round += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  await settle();
}

async function tryKey(name, value, init, reply = null) {
  if (reply !== null) {
    replies.push(reply);
  }
  page.question.value = value;
  const before = asked.length;
  const event = press(init);
  await untilIdle();
  replies.length = 0;
  return { name, sent: asked.length - before, prevented: event.prevented, box: page.question.value };
}

{
  const reply = (question) => answered(question, question, "unchanged");
  const keys = [
    await tryKey("enter", "Who approves leave?", {}, reply("Who approves leave?")),
    await tryKey("ctrl+enter", "Who approves it in Ireland?", { ctrlKey: true }, reply("Who approves it in Ireland?")),
    await tryKey("cmd+enter", "Who approves it in Japan?", { metaKey: true }, reply("Who approves it in Japan?")),
    await tryKey("shift+enter", "A first line", { shiftKey: true }),
    await tryKey("composing", "日本の休暇", { isComposing: true }),
    await tryKey("composing, 229", "日本の休暇", { keyCode: 229 }),
    await tryKey("empty", "   ", {}),
    await tryKey("too long", "x".repeat(501), {}),
  ];
  // Enter while an answer is being written sends nothing, and leaves the box as it is.
  const body = held();
  replies.push([200, body, "text/event-stream; charset=utf-8"]);
  page.question.value = "The first of two";
  press({});
  await settle();
  const busy = { stopHidden: page.stop.hidden, askDisabled: page.ask.getAttribute("aria-disabled") };
  const before = asked.length;
  page.question.value = "The second of two";
  const second = press({});
  await settle();
  keys.push({ name: "while busy", sent: asked.length - before, prevented: second.prevented, box: page.question.value });
  const found = searched("The first of two", "The first of two", "unchanged");
  body.send("searched", found);
  body.send("answer", answer(found));
  body.end();
  await untilIdle();
  record("keys", null);
  Object.assign(steps.at(-1), { keys, busy });
}

console.log(JSON.stringify(steps));
