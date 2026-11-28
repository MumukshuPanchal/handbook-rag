// Runs the labelling page's script in a small fake page, through a set of scenarios, and
// prints what the page did after each step as JSON. The Python test
// tests/evals/test_label_page_behaviour.py runs it with node and reads the result. It needs
// node and nothing else. The fake page has only what the script touches: elements with
// children, text, classes, attributes, the disabled and hidden flags and listeners; a local
// storage that can work, refuse or be full; a confirm dialog that answers as told; and a
// link whose click is caught as a download. What the harness reads back, it reads from those
// elements and from the storage, as a person and the browser would see them.
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(process.argv[2], "utf8");
const data = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const IDS = [
  "batch-data", "item", "items-nav", "choices", "note", "prev", "next", "save", "clear",
  "keep-found", "drop-found", "found", "found-text", "storage-note", "progress", "status",
  "rubric-support", "rubric-correct",
];
const NOW = "2026-11-29T10:15:30.000Z";

function element(tag, id = "") {
  const node = {
    tagName: String(tag).toUpperCase(), id, nodes: [], parent: null, hidden: false,
    disabled: false, value: "", maxLength: -1, type: "", download: "", href: "",
    listeners: {}, classes: new Set(), attrs: {},
    get children() { return this.nodes.filter((child) => typeof child !== "string"); },
    get className() { return [...this.classes].join(" "); },
    set className(value) { this.classes = new Set(String(value).split(/\s+/).filter(Boolean)); },
    get textContent() {
      return this.nodes.map((child) => (typeof child === "string" ? child : child.textContent)).join("");
    },
    set textContent(value) {
      for (const child of this.children) { child.parent = null; }
      this.nodes = String(value) === "" ? [] : [String(value)];
    },
    setAttribute(name, value) { this.attrs[name] = String(value); },
    getAttribute(name) { return this.attrs[name] ?? null; },
    addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); },
    appendChild(child) {
      if (typeof child !== "string") { child.remove(); child.parent = this; }
      this.nodes.push(child);
      return child;
    },
    remove() {
      if (this.parent !== null) {
        this.parent.nodes = this.parent.nodes.filter((child) => child !== this);
        this.parent = null;
      }
    },
    scrollIntoView() {},
    blur() { if (page.document.activeElement === this) { page.document.activeElement = null; } },
    click() {
      if (this.tagName === "A") {
        page.downloads.push({ name: this.download, text: page.blobs.get(this.href) ?? null });
        return;
      }
      if (!this.disabled) { fire(this, "click"); }
    },
  };
  node.classList = {
    add: (...names) => names.forEach((name) => node.classes.add(name)),
    remove: (...names) => names.forEach((name) => node.classes.delete(name)),
    toggle(name, force) {
      const on = force === undefined ? !node.classes.has(name) : Boolean(force);
      if (on) { node.classes.add(name); } else { node.classes.delete(name); }
      return on;
    },
    contains: (name) => node.classes.has(name),
  };
  return node;
}

function fire(node, type, extra = {}) {
  let prevented = false;
  for (const handler of node?.listeners[type] ?? []) {
    handler({ type, target: node, preventDefault() { prevented = true; }, ...extra });
  }
  return prevented;
}

let page = null;

function storage(mode, initial) {
  const map = new Map(Object.entries(initial ?? {}));
  return {
    map,
    getItem(key) { return map.has(key) ? map.get(key) : null; },
    setItem(key, value) {
      if (mode === "full" && !key.endsWith(" probe")) { throw new Error("QuotaExceededError"); }
      map.set(key, String(value));
    },
    removeItem(key) { map.delete(key); },
  };
}

function open({ mode = "ok", stored = {}, confirm = true } = {}) {
  const store = storage(mode, stored);
  const byId = Object.fromEntries(IDS.map((id) => [id, element(id === "note" ? "textarea" : "div", id)]));
  byId["batch-data"].textContent = JSON.stringify(data);
  for (const id of ["found"]) { byId[id].hidden = true; }
  const windowListeners = {};
  page = {
    byId, store, downloads: [], blobs: new Map(), confirmAnswer: confirm, asked: 0,
    windowListeners,
    document: null,
  };
  const body = element("body");
  const document = {
    activeElement: null, body,
    getElementById: (id) => byId[id] ?? null,
    createElement: (tag) => element(tag),
    createTextNode: (text) => String(text),
    listeners: {},
    addEventListener(type, handler) { (this.listeners[type] ??= []).push(handler); },
  };
  page.document = document;
  const window = {
    get localStorage() {
      if (mode === "refused") { throw new Error("SecurityError"); }
      return store;
    },
    confirm: () => { page.asked += 1; return page.confirmAnswer; },
    setTimeout: () => 0,
    addEventListener(type, handler) { (windowListeners[type] ??= []).push(handler); },
  };
  class FixedDate extends Date {
    constructor(...args) { if (args.length) { super(...args); } else { super(NOW); } }
  }
  class Blob { constructor(parts) { this.text = parts.join(""); } }
  let blobs = 0;
  const URL = {
    createObjectURL(blob) { blobs += 1; const href = `blob:${blobs}`; page.blobs.set(href, blob.text); return href; },
    revokeObjectURL() {},
  };
  vm.runInNewContext(source, { window, document, Date: FixedDate, Blob, URL, JSON, String, Object, isNaN });
  return page;
}

function key(name, target = null) {
  const node = target ?? page.document.body;
  for (const handler of page.document.listeners.keydown ?? []) {
    handler({ key: name, target: node, ctrlKey: false, metaKey: false, altKey: false, preventDefault() {} });
  }
}

function typeNote(text) {
  const note = page.byId.note;
  page.document.activeElement = note;
  note.value = text;
  fire(note, "input");
}

function unloadBlocked() {
  let blocked = false;
  for (const handler of page.windowListeners.beforeunload ?? []) {
    handler({ preventDefault() { blocked = true; }, set returnValue(value) { blocked = true; } });
  }
  return blocked;
}

function snapshot(name) {
  const byId = page.byId;
  const where = byId.item.children.find((child) => child.classes.has("where"))?.textContent ?? "";
  const pressed = byId.choices.children.find((button) => button.getAttribute("aria-pressed") === "true");
  return {
    name,
    item: (where.match(/[AB]\d\d/) ?? [null])[0],
    pressed: pressed ? pressed.textContent.replace(/^\d\s*/, "") : null,
    progress: byId.progress.textContent,
    status: byId.status.textContent,
    storageNote: byId["storage-note"].textContent,
    warning: byId["storage-note"].classes.has("warning"),
    found: !byId.found.hidden,
    foundText: byId["found-text"].textContent,
    saveDisabled: byId.save.disabled,
    clearDisabled: byId.clear.disabled,
    choicesDisabled: byId.choices.children.every((button) => button.disabled),
    note: byId.note.value,
    done: byId["items-nav"].children.filter((button) => button.classes.has("done")).length,
    stored: Object.fromEntries(page.store.map),
    downloads: page.downloads,
    unloadBlocked: unloadBlocked(),
    asked: page.asked,
  };
}

const items = data.items;
const choicesOf = (item) => data.choices[item.part];
const results = [];

// Every item labelled with keys, each with the choice of its place; then saved and cleared.
open();
const shown = [];
items.forEach((item, index) => {
  key(String((index % 3) + 1));
  const now = snapshot("");
  shown.push([now.item, now.pressed]);
  key("n");
});
results.push({ ...snapshot("keys"), shown });
page.byId.save.click();
results.push(snapshot("saved"));
page.confirmAnswer = false;
page.byId.clear.click();
results.push(snapshot("clear refused"));
page.confirmAnswer = true;
page.byId.clear.click();
results.push(snapshot("cleared"));

// The save button pressed with one item unlabelled.
open();
items.slice(0, -1).forEach(() => { key("1"); key("n"); });
page.byId.save.click();
fire(page.byId.save, "click");
results.push(snapshot("early save"));

// Every item labelled with clicks on the choices and on "Next".
open();
items.forEach((item, index) => {
  page.byId.choices.children[index % 2].click();
  page.byId.next.click();
});
page.byId.save.click();
results.push(snapshot("clicks"));

// A note typed with keys that are also choices and moves, then Escape and a key.
open();
key("2", page.byId.note);
key("n", page.byId.note);
typeNote("2 n");
results.push(snapshot("typing"));
key("Escape", page.byId.note);
key("2");
results.push(snapshot("after typing"));

// A label changed after going back.
open();
key("1");
key("n");
key("p");
key("2");
results.push(snapshot("back"));

// Labels kept from before, with some that do not fit, and stores of other draws.
const own = `handbook-rag labels, batch ${data.batch}, ${data.items_sha256}`;
const before = {
  [own]: JSON.stringify({
    batch: data.batch,
    items_sha256: data.items_sha256,
    changed_at: "2026-11-28T09:00:00.000Z",
    labels: {
      A01: { label: "unsupported", note: "kept" },
      A02: { label: "correct", note: "" },
      Z99: { label: "supported", note: "" },
      B01: { label: "partial", note: "" },
    },
  }),
  [`handbook-rag labels, batch ${data.batch}, 0000000000000000`]: JSON.stringify({
    batch: data.batch, items_sha256: "0000000000000000", changed_at: NOW,
    labels: Object.fromEntries(items.map((item) => [item.item, { label: choicesOf(item)[0], note: "" }])),
  }),
  [`handbook-rag labels, batch ${data.batch}`]: JSON.stringify({
    batch: data.batch, items_sha256: "0000000000000000", changed_at: NOW,
    labels: Object.fromEntries(items.map((item) => [item.item, { label: choicesOf(item)[0], note: "" }])),
  }),
  [`handbook-rag labels, batch ${data.batch + 1}, ${data.items_sha256}`]: JSON.stringify({
    batch: data.batch + 1, items_sha256: data.items_sha256, changed_at: NOW,
    labels: Object.fromEntries(items.map((item) => [item.item, { label: choicesOf(item)[0], note: "" }])),
  }),
};
open({ stored: before });
results.push(snapshot("found"));
key("1");
page.byId.save.click();
results.push(snapshot("found, a key pressed"));
page.byId["keep-found"].click();
results.push(snapshot("kept"));
open({ stored: before });
page.byId["drop-found"].click();
results.push(snapshot("started again"));

// A store under this page's own key that says it is of another draw.
open({ stored: { [own]: before[`handbook-rag labels, batch ${data.batch}`] } });
results.push(snapshot("other draw under own key"));

// Storage refused from the start, and storage full.
open({ mode: "refused" });
key("1");
results.push(snapshot("refused"));
open({ mode: "full" });
key("1");
results.push(snapshot("full"));

// The page open in another tab, which writes the same store.
open();
key("1");
key("n");
for (const handler of page.windowListeners.storage ?? []) { handler({ key: own }); }
key("2");
results.push(snapshot("another tab"));

process.stdout.write(JSON.stringify(results));
