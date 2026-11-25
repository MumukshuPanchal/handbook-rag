// Runs the demo page's script in a small fake page, with fetch stubbed, through one
// conversation, and prints what the page did after each step as JSON. The Python test
// tests/test_demo_behaviour.py runs it with node and reads the result. It needs node and
// nothing else: no package, no browser. The fake page has only what the script touches:
// elements with children, classes, the hidden flag, attributes and listeners, the element
// that has the focus, the example buttons of the page's markup, and a clipboard. What the
// harness reads back, it reads from those elements, as a reader would see them, and not
// from the script's own variables.
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(process.argv[2], "utf8");

function element(tag, id = "") {
  const node = {
    tagName: String(tag).toUpperCase(),
    id,
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
    focus() {
      document.activeElement = this;
    },
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

// What a reader can do to an element: point at it, focus it, press it. An element that is
// not there takes nothing, and the test then sees that nothing happened.
function fire(node, type) {
  for (const handler of node?.listeners[type] ?? []) {
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

// The texts a screen reader is told and a reader does not see, inside an element.
function spoken(node) {
  return byClass(node, "visually-hidden").map((hidden) => hidden.textContent).join("");
}

// The example buttons of the page's markup, as the script finds them.
const EXAMPLES = [
  "How many weeks of parental leave do I get in the US?",
  "Which personal VPN does GitLab recommend for team members?",
  "How many days of paid annual leave do UK team members get?",
  "How many weeks of paid parental leave does GitHub give its employees?",
];
const examples = EXAMPLES.map((question) => {
  const button = element("button");
  button.className = "example";
  button.textContent = question;
  return button;
});

const elements = {};
const document = {
  activeElement: null,
  getElementById(id) {
    elements[id] ??= element("div", id);
    return elements[id];
  },
  querySelectorAll(selector) {
    return selector === "button.example" ? examples : [];
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
// What /search does, by question: the passages it finds, "network" for a search that
// cannot reach the service, or a status and words for one the service refuses. Any other
// question finds nothing.
const searchFor = new Map();
// What the page put on the clipboard, and whether the clipboard refuses.
const clipboard = { copied: [], refuses: false };

async function fetchStub(url, options) {
  const body = JSON.parse(options.body);
  if (url === "/search") {
    searchedFor.push(body.query);
    const what = searchFor.get(body.query) ?? [];
    if (what === "network") {
      throw new TypeError("Failed to fetch");
    }
    if (!Array.isArray(what)) {
      return new Response(JSON.stringify({ detail: what.detail }), {
        status: what.status,
        headers: { "Content-Type": "application/json" },
      });
    }
    const found = { results: what, mode: "hybrid", reranked: false, took_ms: 1, trace_id: null };
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

// What the script writes to the console as an error: kept, and given to the test. So is a
// promise the script lets fail with no one to catch it, which a browser would only log.
const consoleErrors = [];
process.on("unhandledRejection", (error) => consoleErrors.push(`unhandled: ${error?.message ?? error}`));

const context = vm.createContext({
  document,
  fetch: fetchStub,
  console: { ...console, error: (error) => consoleErrors.push(String(error?.message ?? error)) },
  AbortController,
  TextDecoder,
  Response,
  URL,
  navigator: {
    clipboard: {
      async writeText(text) {
        if (clipboard.refuses) {
          throw new DOMException("Not allowed.", "NotAllowedError");
        }
        clipboard.copied.push(text);
      },
    },
  },
});
vm.runInContext(
  `${source}\n;globalThis.__page = { ask, conversation, page, isBusy: () => current !== null, addressInBrief, snippet };`,
  context,
);
const { ask, conversation, page, isBusy, addressInBrief, snippet } = context.__page;

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

function answered(asked, question, outcome, extra = {}) {
  const found = searched(asked, question, outcome);
  return stream([
    ["searched", found],
    ["text", { text: "Sixteen weeks." }],
    ["answer", answer(found, extra)],
  ]);
}

function refusal(detail, headers = {}) {
  return [503, JSON.stringify({ detail }), "application/json", headers];
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

async function untilIdle() {
  for (let round = 0; round < 500 && isBusy(); round += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
  await settle();
}

// A source's passage, as the API sends it.
function passage(rank, extra = {}) {
  return {
    rank,
    score: 0.03,
    doc_id: "people/leave.md",
    title: "Leave",
    url: "https://handbook.example.com/handbook/people/leave/",
    section: "people",
    heading_path: ["Parental leave"],
    text: "New parents get sixteen weeks.",
    ...extra,
  };
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
  const [progress] = byClass(item, "card-progress");
  const [sourcesBlock] = byClass(item, "sources-block");
  const [searchFold] = byClass(item, "search-fold");
  const fold = (node) => {
    const [folded] = byClass(node, "passage");
    if (folded === undefined) {
      return { foldName: "", foldAddress: "", foldAddressHidden: false };
    }
    const [address] = byClass(folded, "passage-address");
    return {
      foldName: spoken(folded.children[0]),
      foldAddress: address === undefined ? "" : address.textContent,
      foldAddressHidden: address !== undefined && address.attrs["aria-hidden"] === "true",
    };
  };
  const addressOf = (node) => {
    const links = all(node, (child) => child.tagName === "A");
    const briefs = byClass(node, "address-brief");
    return {
      link: links.length === 0 ? null : links[0].href,
      linkTarget: links.length === 0 ? null : links[0].target,
      linkRel: links.length === 0 ? null : links[0].rel,
      linkTitle: links.length === 0 ? null : (links[0].title ?? null),
      brief: briefs.map((brief) => brief.textContent).join(""),
      briefHiddenFromReaders: briefs.length > 0 && briefs.every((brief) => brief.attrs["aria-hidden"] === "true"),
      // The whole address as a screen reader reads it, beside the brief one.
      address: byClass(node, "source-address").map(spoken).join(""),
      // Whether the line is there to be seen: neither it nor anything above it hidden.
      addressShown: byClass(node, "source-address").length === 1 && byClass(node, "source-address").every(isShown),
      ...fold(node),
    };
  };
  return {
    turnSearchedLine: line.textContent,
    turnSearchedHidden: !isShown(line),
    searchedAsTyped: line.classes.has("searched-as-typed"),
    bubble: byClass(item, "bubble").map((bubble) => bubble.textContent),
    scrolled: item.scrolled,
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
    streamUnfinished: byClass(item, "stream").some((text) => text.classes.has("unfinished")),
    progress: shownByClass(item, "card-progress").length > 0,
    progressText: progress.textContent,
    badges: badgeList.map((badge) => badge.textContent),
    cached: shownByClass(item, "cached").map((box) => box.textContent).join(" "),
    note: note !== undefined && isShown(note) ? note.textContent : "",
    madeShown: made !== undefined && isShown(made),
    madeTitle: made !== undefined && isShown(made) ? madeTitle.textContent : "",
    cardTitle: cardTitle.textContent,
    ended: [...item.classes].filter((name) => name.startsWith("ended-")),
    facts: facts !== undefined && isShown(facts) ? facts.textContent : "",
    sourcesShown: isShown(sourcesBlock),
    sourcesNone: shownByClass(item, "sources-none").map((none) => none.textContent).join(""),
    chips: shownByClass(item, "cite").map((chip) => chip.textContent),
    chipLabels: shownByClass(item, "cite").map((chip) => chip.attrs["aria-label"] ?? null),
    cards: cards.filter(isShown).map((card) => ({
      number: byClass(card, "source-number").map((number) => number.textContent).join(""),
      path: byClass(card, "source-path").map((path) => path.textContent).join(""),
      crumbs: byClass(card, "crumb").map((crumb) => crumb.textContent),
      snippet: byClass(card, "source-snippet").map((piece) => piece.textContent).join(""),
      unreadable: card.classes.has("source-unreadable"),
      ...(card.classes.has("source-unreadable") ? {} : addressOf(card)),
    })),
    checkNotes: shownByClass(item, "check-note").map((note) => note.textContent),
    checkLabels: shownByClass(item, "check-label").map((label) => label.textContent),
    checkInNotice: byClass(item, "check-notes").map((box) => inside(box, "notice")),
    bodyOrder: byClass(item, "answer-body").flatMap((body) => body.children.map((part) => part.className)),
    noticeTitle: shownByClass(item, "notice-title").map((title) => title.textContent),
    noticeWhy: shownByClass(item, "notice-why").map((why) => why.textContent),
    searchFoldShown: isShown(searchFold),
    searchCount: byClass(item, "search-fold").flatMap((each) => byClass(each, "side-count")).map((count) => count.textContent).join(""),
    searchState: byClass(item, "search-state").map((state) => state.textContent).join(""),
    passages: byClass(item, "passage-item").map((found) => ({
      rank: byClass(found, "passage-rank").map((rank) => rank.textContent).join(""),
      unreadable: found.classes.has("passage-unreadable"),
      ...(found.classes.has("passage-unreadable") ? {} : addressOf(found)),
    })),
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
    conversationSummary: `${page.conversationCount.textContent}${page.conversationVerb.textContent}`,
    turnCount: page.turns.children.length,
    stopHidden: page.stop.hidden,
    askDisabled: page.ask.getAttribute("aria-disabled"),
    examplesDisabled: examples.map((example) => example.getAttribute("aria-disabled")),
    formBusy: page.form.classes.has("busy"),
    newConversationHidden: page.newConversation.hidden,
    welcomeHidden: page.welcome.hidden,
    status: page.status.textContent,
    focus: document.activeElement === null ? null : document.activeElement.id,
    // The notes of checks on the first turn of the conversation and on the one before the
    // latest, so that a note shown on another turn than its own is seen.
    firstTurnCheckNotes: byClass(page.turns.children[0], "check-note").map((note) => note.textContent),
    previousTurnCheckNotes: byClass(page.turns.children.at(-2), "check-note").map((note) => note.textContent),
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
  // A reply left unused, because a fault kept the question from being sent, is dropped,
  // so that the next step gets its own.
  replies.length = 0;
  record(name, question);
  steps.at(-1).searches = searchedFor.slice(searchesBefore);
}

// Presses a button of a turn, by its class. A button that is not there is not pressed,
// and the test then sees that nothing happened.
function press(item, name) {
  const [button] = byClass(item, name);
  if (button !== undefined) {
    fire(button, "click");
  }
}

const FIRST = "How many days of paid annual leave do UK team members get?";
const REWRITTEN = "How many days of paid annual leave do team members in Finland get?";
const TRACE_ID = "4be69ac5bb0e6069eaa37e98ae312974";

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
await step("refused", "and in ireland", refusal("The service is busy.", { "X-Trace-Id": TRACE_ID }));

// The trace id of that refusal, copied, and copied again when the clipboard refuses.
{
  const item = page.turns.children.at(-1);
  press(item, "copy");
  await settle();
  const copied = { text: clipboard.copied.at(-1) ?? null, status: page.status.textContent };
  clipboard.refuses = true;
  press(item, "copy");
  await settle();
  clipboard.refuses = false;
  steps.at(-1).copy = { ...copied, refusedStatus: page.status.textContent };
}

await step("not rewritten", "and in germany", answered("and in germany", "and in germany", "not_asked"));
fire(elements["new-conversation"], "click");
await step("new conversation", null, null);

// The first question of the new conversation, asked by pressing its example button.
{
  const searchesBefore = searchedFor.length;
  replies.push(answered(FIRST, FIRST, "not_asked"));
  fire(examples.find((example) => example.textContent === FIRST), "click");
  await untilIdle();
  replies.length = 0;
  record("asked by itself", FIRST);
  steps.at(-1).searches = searchedFor.slice(searchesBefore);
}

// An answer that cites two sources, one of them twice, and names a third that it does
// not list. The search beside it finds three passages.
const US = "How many weeks of parental leave do new parents get in the US?";
const LEAVE_URL =
  "https://handbook.example.com/handbook/total-rewards/benefits/general-and-entity-benefits/leave-in-the-united-states/";
searchFor.set(US, [1, 2, 3].map((rank) => passage(rank, { url: LEAVE_URL })));
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
  const numberOf = (card) => byClass(card, "source-number").map((number) => number.textContent).join("");
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
  const before = card("2")?.scrolled ?? 0;
  fire(chip("2"), "click");
  linking.pressChip = {
    scrolled: (card("2")?.scrolled ?? 0) - before,
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
// A search that finds one passage, one the service refuses, and one that cannot reach it.
searchFor.set("How much is the gym budget?", [passage(1)]);
searchFor.set("do they carry over", { status: 503, detail: "The service is busy." });
searchFor.set("Is that withheld?", "network");
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
await step(
  "semantic",
  "how long is leave for new parents",
  answered("how long is leave for new parents", "how long is leave for new parents", "unchanged", {
    usage: { input_tokens: 0, output_tokens: 0 },
    requests: 0,
    cached: { tier: "semantic", question: US, similarity: 0.93, score: 0.95, stored_at: "2026-11-25T11:52:07+00:00" },
  }),
);

// An answer the fallback model wrote after two requests. It lists two sources and its
// text cites one, and that source's passage runs long.
const LONG_TEXT =
  "Team members in the United States get sixteen weeks of paid parental leave, which can be taken at any point in the first year after a birth or an adoption, in one block or in several, with the agreement of their manager and the absence team";
await step(
  "fallback",
  "and who pays for it",
  answered("and who pays for it", "and who pays for it", "unchanged", {
    requests: 2,
    fallback: true,
    segments: [{ text: "GitLab pays for it.", sources: [1] }],
    sources: [
      { number: 1, doc_id: "people/leave.md", title: "Leave", url: LEAVE_URL, section: "people", heading_path: [], text: LONG_TEXT },
      { number: 2, doc_id: "people/faq.md", title: "Leave FAQ", url: LEAVE_URL, section: "people", heading_path: [], text: "Paid." },
    ],
  }),
);

// A follow-up answered by a service that sends no searched event first: no search runs.
await step(
  "no searched event",
  "and in canada",
  stream([
    ["text", { text: "Sixteen weeks." }],
    ["answer", answer(searched("and in canada", "and in canada", "unchanged"))],
  ]),
);

// The next question asked after a failure: the alert is emptied for it.
await step("late failure", "and in japan", refusal("The answer model is unavailable. Try again later."));

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

// An answer stopped while its text is written, from the stop button, which has the focus.
{
  const body = held();
  replies.push([200, body, "text/event-stream; charset=utf-8"]);
  const asking = ask("Tell me about leave");
  body.send("searched", searched("Tell me about leave", "Tell me about leave", "unchanged"));
  body.send("text", { text: "Leave is" });
  await settle();
  elements.stop.focus();
  fire(elements.stop, "click");
  await asking;
  await settle();
  record("stopped", "Tell me about leave");
}

// Stop pressed in the moment after the answer came and before the stream closed.
{
  const body = held();
  replies.push([200, body, "text/event-stream; charset=utf-8"]);
  const asking = ask("Is leave paid?");
  const found = searched("Is leave paid?", "Is leave paid?", "unchanged");
  body.send("searched", found);
  body.send("text", { text: "Yes." });
  body.send("answer", answer(found, { segments: [{ text: "Yes.", sources: [] }] }));
  await settle();
  fire(elements.stop, "click");
  await asking;
  await settle();
  record("stop after answer", "Is leave paid?");
}

await step(
  "stream cut",
  "and the rest",
  stream([
    ["searched", searched("and the rest", "And the rest of the leave?", "rewritten")],
    ["text", { text: "Part of" }],
  ]),
);
const cutTurn = page.turns.children.at(-1);

// The keys of the question box. A key press is given to the box's listeners as the
// browser gives it, and what the harness reads is whether a question was sent, and
// whether the key's own effect, such as a new line, was prevented.
function key(init) {
  const pressed = {
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
    handler(pressed);
  }
  return pressed;
}

async function tryKey(name, value, init, reply = null) {
  if (reply !== null) {
    replies.push(reply);
  }
  page.question.value = value;
  const before = asked.length;
  const pressed = key(init);
  await untilIdle();
  replies.length = 0;
  return { name, sent: asked.length - before, prevented: pressed.prevented, box: page.question.value };
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
  // While an answer is being written: Enter sends nothing and leaves the box as it is,
  // "Ask again" on an earlier failure sends nothing, and a new conversation is not begun.
  const body = held();
  replies.push([200, body, "text/event-stream; charset=utf-8"]);
  page.question.value = "The first of two";
  key({});
  await settle();
  const busy = {
    stopHidden: page.stop.hidden,
    askDisabled: page.ask.getAttribute("aria-disabled"),
    examplesDisabled: examples.map((example) => example.getAttribute("aria-disabled")),
    formBusy: page.form.classes.has("busy"),
    earlierSent: asked.at(-1).earlier.length,
  };
  let before = asked.length;
  page.question.value = "The second of two";
  const second = key({});
  await settle();
  keys.push({ name: "while busy", sent: asked.length - before, prevented: second.prevented, box: page.question.value });
  before = asked.length;
  press(cutTurn, "again");
  await settle();
  busy.askAgainSent = asked.length - before;
  const turnsBefore = page.turns.children.length;
  const conversationBefore = conversation.length;
  fire(elements["new-conversation"], "click");
  busy.newConversation = {
    turnsKept: page.turns.children.length === turnsBefore,
    conversationKept: conversation.length === conversationBefore,
    status: page.status.textContent,
  };
  const found = searched("The first of two", "The first of two", "unchanged");
  body.send("searched", found);
  body.send("answer", answer(found));
  body.end();
  await untilIdle();
  record("keys", null);
  Object.assign(steps.at(-1), { keys, busy });
}

// "Ask again" on that failure, now that nothing is being written: the question is asked
// once more, with the conversation as it stands.
{
  replies.push(answered("and the rest", "And the rest of the leave?", "rewritten"));
  const before = asked.length;
  press(cutTurn, "again");
  await untilIdle();
  record("asked again", "and the rest");
  Object.assign(steps.at(-1), { sent: asked.length - before, query: asked.at(-1).query });
}

// Replies that checks in code changed or marked, each with the notes the service sends in
// fixed words, and replies whose list of checks is not of the shape the page reads.
const DROPPED =
  "Text after the answer's last citation was left out: it cited no passage and held a web or email address.";
const COPIED = "Most of this reply repeats the passages it was given, word for word.";
const UNCITED =
  "The reply the model wrote cited none of the passages, so it is not shown: a reply that rests on no passage cannot be checked against the handbook.";
const UNFINISHED =
  "The reply did not come to a proper end, and its words after the last citation are not shown.";
const SOURCE = { number: 1, doc_id: "people/leave.md", title: "Leave", url: LEAVE_URL, section: "people", heading_path: [], text: "Apply a month ahead." };
const cited1 = { segments: [{ text: "Apply a month ahead.", sources: [1] }], sources: [SOURCE] };
const droppedLine = { check: "address_in_closing_line", action: "dropped_line", note: DROPPED };
await step(
  "check dropped line",
  "and the deadline",
  answered("and the deadline", "and the deadline", "unchanged", { ...cited1, checks: [droppedLine] }),
);
await step(
  "two checks",
  "and the form",
  answered("and the form", "and the form", "unchanged", {
    ...cited1,
    checks: [droppedLine, { check: "mostly_a_copy", action: "marked", note: COPIED }],
  }),
);
await step(
  "check declined",
  "and the pay",
  declined("and the pay", { checks: [{ check: "cites_nothing", action: "declined", note: UNCITED }] }),
);
await step(
  "check cut",
  "and the dates",
  stream([
    ["searched", searched("and the dates", "and the dates", "unchanged")],
    ["text", { text: "Apply a month ahead." }],
    [
      "answer",
      // A reply that is not an answer lists no sources, as the service sends it.
      answer(searched("and the dates", "and the dates", "unchanged"), {
        segments: [{ text: "Apply a month ahead.", sources: [] }],
        sources: [],
        answered: false,
        stop_reason: "max_tokens_or_other",
        checks: [{ check: "unfinished_reply", action: "dropped_words", note: UNFINISHED }],
      }),
    ],
  ]),
);
await step(
  "malformed checks",
  "and the notice",
  answered("and the notice", "and the notice", "unchanged", { checks: "not a list" }),
);
await step(
  "malformed items",
  "and the approval",
  answered("and the approval", "and the approval", "unchanged", {
    checks: [null, 3, "a note", { note: 5 }, { check: "cites_nothing" }, { note: "" }],
  }),
);

// A failure, then a new conversation: the alert keeps nothing of it.
await step("failure before new", "and in brazil", refusal("The answer model is unavailable. Try again later."));
fire(elements["new-conversation"], "click");
await step("new after failure", null, null);

// The address in brief and the piece of a passage, asked of the script's own functions.
const ADDRESSES = [
  "https://handbook.example.com",
  "https://handbook.example.com/",
  `https://${"very-long-host-name-".repeat(4)}example.com/`,
  `https://${"very-long-host-name-".repeat(4)}example.com/handbook/people/leave/`,
  "https://handbook.gitlab.com@evil.example/handbook/people/leave/",
  "https://hаndbook.gitlab.com/handbook/people/leave/",
  "https://handbook.example.com:8443/handbook/people/leave/",
  "https://handbook.gitlab.com/handbook/total-rewards/benefits/general-and-entity-benefits/gitlab-ireland-ltd/",
  `https://handbook.gitlab.com/handbook/${"a-very-long-part-of-a-path-".repeat(3)}end/`,
  "https://handbook.gitlab.com/handbook/finance/expenses/",
  "ftp://handbook.example.com/faq",
];
const brief = (url) => {
  try {
    return addressInBrief(url);
  } catch (error) {
    return `threw: ${error.message}`;
  }
};
steps.at(-1).briefs = Object.fromEntries(ADDRESSES.map((url) => [url, brief(url)]));
steps.at(-1).consoleErrors = consoleErrors;
steps.at(-1).longText = LONG_TEXT;
steps.at(-1).snippet = snippet(LONG_TEXT);

console.log(JSON.stringify(steps));
