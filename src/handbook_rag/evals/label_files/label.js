(function () {
  "use strict";
  var data = JSON.parse(document.getElementById("batch-data").textContent);
  var items = data.items;
  var storageKey = "handbook-rag labels, batch " + data.batch + ", " + data.items_sha256;
  var state = {};
  var current = 0;
  var saved = null;
  var storage = null;
  var problem = null;
  var found = null;

  function byId(id) { return document.getElementById(id); }

  function make(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text !== undefined && text !== null) { node.textContent = String(text); }
    return node;
  }

  function allowed(item, label) {
    return typeof label === "string" && data.choices[item.part].indexOf(label) !== -1;
  }

  function when(moment) {
    var date = new Date(moment);
    return isNaN(date.getTime()) ? "at a time it did not record" : date.toLocaleString();
  }

  try {
    storage = window.localStorage;
    storage.setItem(storageKey + " probe", "1");
    storage.removeItem(storageKey + " probe");
  } catch (error) {
    storage = null;
    problem = "this browser does not let a page that is opened from disk keep anything";
  }

  function readStored() {
    var stored = null;
    var labels = {};
    if (!storage) { return null; }
    try { stored = JSON.parse(storage.getItem(storageKey) || "null"); } catch (error) { return null; }
    if (!stored || typeof stored !== "object" || stored.batch !== data.batch ||
        stored.items_sha256 !== data.items_sha256 || !stored.labels || typeof stored.labels !== "object") {
      return null;
    }
    items.forEach(function (item) {
      var kept = stored.labels[item.item];
      if (!kept || typeof kept !== "object") { return; }
      labels[item.item] = {
        label: allowed(item, kept.label) ? kept.label : null,
        note: typeof kept.note === "string" ? kept.note.slice(0, data.note_limit) : ""
      };
    });
    return { labels: labels, changed: stored.changed_at };
  }

  function counts(labels) {
    var given = 0;
    var notes = 0;
    Object.keys(labels).forEach(function (id) {
      if (labels[id].label !== null) { given += 1; }
      if (labels[id].note.trim()) { notes += 1; }
    });
    return { labels: given, notes: notes };
  }

  function showStorage() {
    var note = byId("storage-note");
    note.classList.toggle("warning", problem !== null);
    note.textContent = problem === null
      ? "Your labels are kept in this browser's local storage as you go, so a reload does not lose them. " +
        "Saving the file does not clear them; the button at the bottom right does."
      : "Your labels are not being kept in this browser: " + problem + ". " +
        "Do not reload or close this page until you have saved the labels file.";
    byId("clear").disabled = !storage || problem !== null || found !== null || !stored();
  }

  function stored() {
    try { return storage !== null && storage.getItem(storageKey) !== null; } catch (error) { return false; }
  }

  function keep() {
    if (!storage || problem !== null || found !== null) { return; }
    try {
      storage.setItem(storageKey, JSON.stringify({
        batch: data.batch,
        items_sha256: data.items_sha256,
        changed_at: new Date().toISOString(),
        labels: state
      }));
    } catch (error) {
      problem = "the browser refused to keep more (its storage may be full)";
    }
    showStorage();
  }

  function entry(item) {
    if (!state[item.item]) { state[item.item] = { label: null, note: "" }; }
    return state[item.item];
  }

  function labelOf(item) { return state[item.item] ? state[item.item].label : null; }

  function labelled() {
    return items.filter(function (item) { return labelOf(item) !== null; }).length;
  }

  function signature() {
    return JSON.stringify(items.map(function (item) {
      var kept = entry(item);
      return [item.item, kept.label, kept.note.trim()];
    }));
  }

  function showFound() {
    var panel = byId("found");
    var seen = found === null ? null : counts(found.labels);
    panel.hidden = found === null;
    if (seen !== null) {
      byId("found-text").textContent = "Labels from before were found in this browser: " + seen.labels +
        " labels and " + seen.notes + " notes, last changed " + when(found.changed) +
        ". Keep them, or start again. Until you choose, nothing is labelled or kept.";
    }
  }

  function showNav() {
    var nav = byId("items-nav");
    nav.textContent = "";
    items.forEach(function (item, index) {
      var button = make("button", labelOf(item) !== null ? "done" : "", item.item);
      button.type = "button";
      button.setAttribute("aria-label", item.item + (labelOf(item) !== null ? ", labelled" : ", not labelled yet"));
      if (index === current) { button.setAttribute("aria-current", "true"); }
      button.addEventListener("click", function () { go(index); });
      nav.appendChild(button);
    });
  }

  function cites(numbers) {
    if (!numbers.length) { return "cites no passage"; }
    return (numbers.length === 1 ? "cites passage " : "cites passages ") + numbers.join(", ");
  }

  function showStatement(view, item) {
    var list = make("ol", "answer");
    view.appendChild(make("h2", "", "Part A. Is the marked statement backed by the passages it cites?"));
    view.appendChild(make("p", "label", "Question"));
    view.appendChild(make("p", "question", item.question));
    view.appendChild(make("p", "label", "The answer, statement by statement"));
    item.statements.forEach(function (statement) {
      var marked = statement.number === item.marked;
      var line = make("li", marked ? "marked" : "");
      if (marked) { line.appendChild(make("span", "tag", "Label this statement")); }
      line.appendChild(make("span", "", statement.text));
      line.appendChild(document.createTextNode(" "));
      line.appendChild(make("span", "cites", "(" + cites(statement.cites) + ")"));
      list.appendChild(line);
    });
    view.appendChild(list);
    view.appendChild(make("p", "label", "The passages the marked statement cites"));
    if (!item.passages.length) {
      view.appendChild(make("p", "none", "The marked statement cites no passage."));
    }
    item.passages.forEach(function (passage) {
      view.appendChild(make("p", "passage-title", "Passage " + passage.number + ": " + passage.title));
      view.appendChild(make("div", "box", passage.text));
    });
  }

  function showAnswer(view, item) {
    view.appendChild(make("h2", "", "Part B. Does the answer tell the asker what the reference answer tells them?"));
    view.appendChild(make("p", "label", "Question"));
    view.appendChild(make("p", "question", item.question));
    view.appendChild(make("p", "label", "Reference answer, taken to be right"));
    view.appendChild(make("div", "box", item.reference));
    view.appendChild(make("p", "label", "The assistant's answer"));
    view.appendChild(make("div", "box", item.answer));
  }

  function showItem() {
    var item = items[current];
    var view = byId("item");
    view.textContent = "";
    view.appendChild(make("p", "where", "Item " + (current + 1) + " of " + items.length + ", " + item.item));
    if (item.part === "support") { showStatement(view, item); } else { showAnswer(view, item); }
  }

  function showChoices() {
    var item = items[current];
    var box = byId("choices");
    box.textContent = "";
    data.choices[item.part].forEach(function (label, index) {
      var button = make("button", "", null);
      button.type = "button";
      button.disabled = found !== null;
      button.setAttribute("aria-pressed", labelOf(item) === label ? "true" : "false");
      button.appendChild(make("kbd", "", String(index + 1)));
      button.appendChild(document.createTextNode(" " + data.names[label]));
      button.addEventListener("click", function () { choose(label); });
      box.appendChild(button);
    });
    byId("note").value = state[item.item] ? state[item.item].note : "";
    byId("note").disabled = found !== null;
    byId("prev").disabled = current === 0;
    byId("next").disabled = current === items.length - 1;
  }

  function showProgress() {
    var done = labelled();
    var status = byId("status");
    byId("progress").textContent = done + " of " + items.length + " items labelled";
    byId("save").disabled = done !== items.length || found !== null;
    if (found !== null) {
      status.textContent = "Choose first: keep the labels found in this browser, or start again.";
    } else if (done !== items.length) {
      status.textContent = "Label all " + items.length + " items to save the file.";
    } else if (saved === null) {
      status.textContent = "Every item has a label. Save the labels file.";
    } else if (saved.signature === signature()) {
      status.textContent = "Saved at " + saved.time + " as " + saved.name +
        ", usually in your Downloads folder. That newest file is the one to put in place.";
    } else {
      status.textContent = "Changed since the file saved at " + saved.time + ". Save again: " +
        "the new file will be the one to put in place.";
    }
  }

  function render() {
    showFound();
    showStorage();
    showNav();
    showItem();
    showChoices();
    showProgress();
  }

  function choose(label) {
    if (found !== null) { return; }
    entry(items[current]).label = label;
    keep();
    showNav();
    showChoices();
    showProgress();
  }

  function go(index) {
    if (index < 0 || index >= items.length) { return; }
    current = index;
    render();
    byId("item").scrollIntoView({ block: "start" });
  }

  function two(number) { return (number < 10 ? "0" : "") + number; }

  function stamp(moment) {
    return moment.getFullYear() + "-" + two(moment.getMonth() + 1) + "-" + two(moment.getDate()) + "-" +
      two(moment.getHours()) + two(moment.getMinutes()) + two(moment.getSeconds());
  }

  function labelsFile(moment) {
    var given = counts(state);
    return {
      format: data.format,
      version: data.version,
      batch: data.batch,
      labeller: data.labeller,
      pass: data.pass,
      items_sha256: data.items_sha256,
      saved_at: moment.toISOString(),
      counted: { labels: given.labels, notes: given.notes },
      labels: items.map(function (item) {
        var kept = entry(item);
        return { item: item.item, item_sha256: item.sha256, label: kept.label, note: kept.note.trim() };
      })
    };
  }

  function save() {
    var link = make("a", "", null);
    var moment = new Date();
    var name = data.file_prefix + stamp(moment) + ".json";
    var blob = null;
    if (labelled() !== items.length || found !== null) { return; }
    items.forEach(entry);
    blob = new Blob([JSON.stringify(labelsFile(moment), null, 2) + "\n"], { type: "application/json" });
    link.download = name;
    link.href = URL.createObjectURL(blob);
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(link.href); }, 60000);
    saved = { name: name, time: moment.toLocaleTimeString(), signature: signature() };
    showProgress();
  }

  function clearCopy() {
    if (!storage || !window.confirm("Clear this browser's copy of your labels? Do this only once the " +
        "saved file is where you want it, or to start again.")) {
      return;
    }
    try { storage.removeItem(storageKey); } catch (error) { problem = "the browser refused to clear it"; }
    showStorage();
  }

  function keepFound() {
    if (found === null) { return; }
    state = found.labels;
    found = null;
    render();
  }

  function dropFound() {
    if (found === null || !window.confirm("Start again, and clear the labels found in this browser?")) { return; }
    found = null;
    state = {};
    try { storage.removeItem(storageKey); } catch (error) { problem = "the browser refused to clear it"; }
    render();
  }

  byId("rubric-support").textContent = data.rubrics.support;
  byId("rubric-correct").textContent = data.rubrics.correct;
  byId("note").maxLength = data.note_limit;
  byId("note").addEventListener("input", function (event) {
    if (found !== null) { return; }
    entry(items[current]).note = event.target.value.slice(0, data.note_limit);
    keep();
    showProgress();
  });
  byId("prev").addEventListener("click", function () { go(current - 1); });
  byId("next").addEventListener("click", function () { go(current + 1); });
  byId("save").addEventListener("click", save);
  byId("clear").addEventListener("click", clearCopy);
  byId("keep-found").addEventListener("click", keepFound);
  byId("drop-found").addEventListener("click", dropFound);
  document.addEventListener("keydown", function (event) {
    var target = event.target;
    var choices = data.choices[items[current].part];
    var index = ["1", "2", "3", "4"].indexOf(event.key);
    if (event.ctrlKey || event.metaKey || event.altKey) { return; }
    if (target && (target.tagName === "TEXTAREA" || target.tagName === "INPUT")) {
      if (event.key === "Escape") { target.blur(); }
      return;
    }
    if (index !== -1 && index < choices.length) {
      choose(choices[index]);
    } else if (event.key === "ArrowRight" || event.key === "n" || event.key === "N") {
      go(current + 1);
    } else if (event.key === "ArrowLeft" || event.key === "p" || event.key === "P") {
      go(current - 1);
    } else {
      return;
    }
    event.preventDefault();
  });
  window.addEventListener("storage", function (event) {
    if (event.key !== storageKey || problem !== null) { return; }
    problem = "this page is open in another tab or window as well, and this tab stopped keeping " +
      "its labels so as not to write over the other's. Close one of them";
    showStorage();
  });
  window.addEventListener("beforeunload", function (event) {
    var unsaved = saved === null || saved.signature !== signature();
    if (problem !== null && labelled() > 0 && unsaved) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  found = readStored();
  if (found !== null && counts(found.labels).labels === 0 && counts(found.labels).notes === 0) {
    found = null;
  }
  render();
}());
