"""The demo page's script, run: one conversation through a small fake page in node.

``tests/test_demo.py`` reads the script's source. This runs it, with ``node`` and
``tests/demo_page_harness.mjs``: a fake page with only what the script touches, and a
stubbed ``fetch`` that answers ``/ask`` as the service does, as a stream with the question
searched for first. The project has no JavaScript toolchain (ADR 0016), and none is added:
node runs the script as it is. Without node the tests are skipped, and say why.

The harness reads back what a reader would see in each turn: the question's bubble, the
answer or the notice or the failure card, the placeholders and the caret while the answer
comes, the badges under it, its source cards, and what pointing at, focusing and pressing
a number in the answer does to them.

A second test plants faults in the script and asks that the first one would have seen
each, so that a check which saw nothing would not pass unnoticed.
"""

import json
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

NODE = shutil.which("node")
HARNESS = Path(__file__).parent / "demo_page_harness.mjs"
FIRST = "How many days of paid annual leave do UK team members get?"
REWRITTEN = "How many days of paid annual leave do team members in Finland get?"
LEAVE_URL = (
    "https://handbook.example.com/handbook/total-rewards/benefits/"
    "general-and-entity-benefits/leave-in-the-united-states/"
)
STORED = "Do Finland team members paid annual leave days carry over to the next year?"
TRACE_ID = "4be69ac5bb0e6069eaa37e98ae312974"
# What the page puts between the parts of an address in brief.
CRUMB = "\u203a"

pytestmark = pytest.mark.skipif(
    NODE is None, reason="node is not installed: the page's script cannot be run here"
)

# The steps in which the page waited on the service, by how the turn ended.
ANSWERED = ("first", "rewritten", "given back", "asked by itself", "cited", "arrived")
DECLINED = ("declined", "cached")
FAILED = ("error after text", "unreadable answer", "refused", "stream cut")
# Steps recorded while an answer was still coming, when its turn had not ended.
UNDER_WAY = ("waiting", "writing")


def script() -> str:
    return files("handbook_rag").joinpath("static", "demo.js").read_text(encoding="utf-8")


def run(source: str, tmp_path: Path) -> dict[str, dict[str, Any]]:
    """The page's state after each step of the harness's conversation, by step name."""
    path = tmp_path / "demo.js"
    path.write_text(source, encoding="utf-8")
    assert NODE is not None
    done = subprocess.run(
        [NODE, str(HARNESS), str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return {step["name"]: step for step in json.loads(done.stdout)}


def conversation_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the page did wrong with the conversation it sends, in words."""
    found = []
    first, rewritten = steps["first"], steps["rewritten"]
    if first["conversation"] != [FIRST] or rewritten["conversation"] != [FIRST, REWRITTEN]:
        found.append("the conversation does not grow by the question searched for")
    if rewritten["sentEarlier"] != [FIRST] or first["sentEarlier"] != []:
        found.append("the earlier questions sent are not the conversation")
    if rewritten["searches"] != [REWRITTEN] or first["searches"] != [FIRST]:
        found.append("the side search is not run on the question searched for")
    if rewritten["searchedLine"] != f"Searched for: {REWRITTEN}" or rewritten["searchedHidden"]:
        found.append("a rewritten follow-up does not say what was searched for")
    for name, words, fault in (
        ("failed rewrite", "the request to rewrite it failed", "failed"),
        ("unusable rewrite", "the rewritten question could not be used", "was not usable"),
    ):
        if words not in steps[name]["searchedLine"] or steps[name]["searchedHidden"]:
            found.append(f"a follow-up whose rewrite {fault} does not say so")
    if steps["unusable rewrite"]["conversation"][-2:] != ["and in spain", "and in italy"]:
        found.append("a follow-up searched for as typed is not kept as typed")
    given_back = steps["given back"]
    if given_back["searchedLine"] != "" or not given_back["searchedHidden"]:
        found.append("a line about the question searched for is left showing")
    if steps["refused"]["searchedLine"] != "" or not steps["refused"]["searchedHidden"]:
        found.append("a refused question shows a line about the search")
    for failed in ("error after text", "unreadable answer", "refused"):
        if steps[failed]["conversation"] != given_back["conversation"]:
            found.append(f"a turn that failed ({failed}) is added to the conversation")
        if steps[failed]["failure"] == "":
            found.append(f"a turn that failed ({failed}) says nothing")
    not_rewritten = steps["not rewritten"]
    if (
        "did not rewrite the follow-up" not in not_rewritten["searchedLine"]
        or (not_rewritten["searchedHidden"])
    ):
        found.append("a follow-up the service did not rewrite says nothing of it")
    if steps["new conversation"]["conversation"] != []:
        found.append('"Start a new conversation" does not forget the conversation')
    if steps["asked by itself"]["sentEarlier"] != []:
        found.append("a question after a new conversation is sent with earlier questions")
    for name, step in steps.items():
        if step["listed"] != step["conversation"]:
            found.append(f"the list on the page is not the conversation ({name})")
        if step["listHidden"] != (step["conversation"] == []):
            found.append(f"the list on the page is hidden or shown wrongly ({name})")
    return found


def turn_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the page did wrong in showing the turns, in words."""
    found = []
    new = steps["new conversation"]
    if new["turnCount"] != 0:
        found.append('"Start a new conversation" leaves the turns on the page')
    if steps["given back"]["turnCount"] != 5 or steps["asked by itself"]["turnCount"] != 1:
        found.append("a question does not add one turn to the page")
    for name, step in steps.items():
        if step["welcomeHidden"] != (step["turnCount"] > 0):
            found.append(f"the welcome is shown or hidden wrongly ({name})")
        if step["question"] is not None and step["bubble"] != [f"You asked: {step['question']}"]:
            found.append("a turn does not show the question asked as its own bubble")
    # How each turn ended, and what it looks like then.
    for name in ANSWERED:
        step = steps[name]
        if not step["answer"] or step["decline"] or step["error"]:
            found.append(f"an answer is not shown as an answer ({name})")
    for name in DECLINED:
        step = steps[name]
        if not step["decline"] or step["error"] or step["answer"]:
            found.append(f"a decline is not shown as a calm notice ({name})")
        if "I could not find this in the handbook." not in " ".join(step["notice"]):
            found.append(f"a decline does not show the reply's own words ({name})")
    withheld = steps["withheld"]
    if not withheld["unfinished"] or withheld["decline"] or withheld["error"]:
        found.append("a reply that was withheld or cut off is shown as a decline or a failure")
    for name in FAILED:
        step = steps[name]
        if not step["error"] or step["answer"]:
            found.append(f"a failure is not shown in its turn ({name})")
        elif step["errorText"] != step["failure"]:
            found.append(f"the failure card does not say what the alert says ({name})")
    stopped = steps["stopped"]
    if stopped["note"] != "Stopped. The text above is only what arrived before that.":
        found.append("a stopped answer does not say so in its turn")
    if stopped["streamed"] != "Leave is" or stopped["error"]:
        found.append("a stopped answer loses its text or is shown as a failure")
    # While an answer comes: placeholders, then the text with a caret.
    waiting, writing = steps["waiting"], steps["writing"]
    if not waiting["pending"] or waiting["placeholders"] != 2 or not waiting["progress"]:
        found.append("a turn that waits shows no placeholders")
    if writing["pending"]:
        found.append("the placeholder stays while the text is written")
    if not writing["caret"] or writing["streamed"] != "Your manager":
        found.append("the text as it arrives has no caret")
    if writing["placeholders"] != 2:
        found.append("the sources' placeholders go before the sources come")
    for name, step in steps.items():
        if name in UNDER_WAY or step["question"] is None:
            continue
        if step["caret"]:
            found.append(f"the caret stays after the turn has ended ({name})")
        if step["pending"] or step["progress"]:
            found.append(f"the waiting look stays after the turn has ended ({name})")
        if step["placeholders"] != 0:
            found.append(f"sources that never came leave placeholders ({name})")
    return found


def badge_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the strip of badges under an answer gets wrong, in words."""
    found = []
    cited = steps["cited"]["badges"]
    for wanted in (
        "Search: 3 passages found",
        "Follow-up: rewritten to stand alone",
        "Cited: 2 sources",
        "Model: stand-in",
        "Time: first words 0.0 s, in all 0.0 s",
        "Tokens: 1 in, 1 out",
    ):
        if wanted not in cited:
            found.append(f"the badges miss {wanted.split(':')[0]!r} or get it wrong")
    if "Search: 3 passages found" not in cited:
        found.append("the badges do not say how many passages the search found")
    if any(badge.startswith("Follow-up") for badge in steps["first"]["badges"]):
        found.append("a question asked by itself has a badge about a follow-up")
    if "Follow-up: rewrite not usable" not in steps["unusable rewrite"]["badges"]:
        found.append("a follow-up searched for as typed has no badge that says so")
    cached = steps["cached"]
    if "From the cache: exact match" not in cached["badges"]:
        found.append("a cached answer has no badge that says so")
    if "Tokens: none: no request" not in cached["badges"]:
        found.append("a cached answer counts tokens it never spent")
    if f"It was written for the question: {STORED}" not in cached["cached"]:
        found.append("a cached answer does not name the question it was written for")
    if any(badge.startswith("From the cache") for badge in steps["first"]["badges"]):
        found.append("an answer written now has a badge that says it came from the cache")
    for name in FAILED:
        if steps[name]["badges"] != []:
            found.append(f"a failure shows badges of an answer it does not have ({name})")
    refused = steps["refused"]
    if refused["madeTitle"] != "About this request" or TRACE_ID not in refused["facts"]:
        found.append("a failure does not show the trace id it came with")
    return found


def source_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the source cards and the numbers that lead to them get wrong, in words."""
    found = []
    cited = steps["cited"]
    if cited["chips"] != ["1", "1", "2"]:
        found.append("the numbers in the answer are not the sources it lists")
    cards = {card["number"]: card for card in cited["cards"]}
    if sorted(cards) != ["1", "2"]:
        found.append("a source the answer lists has no card")
        return found
    leave, faq = cards["1"], cards["2"]
    if (leave["link"], leave["linkTarget"], leave["linkRel"]) != (
        LEAVE_URL,
        "_blank",
        "noopener noreferrer",
    ):
        found.append("a source's card does not link to its page as the one link maker does")
    if faq["link"] is not None or "(no link: not a web address)" not in faq["path"]:
        found.append("a source that is not a web address becomes a link")
    if leave["crumbs"] != ["Parental leave", "United States"] or not leave["path"].startswith(
        "Leave"
    ):
        found.append("a source card does not show the page's title and the headings above")
    if leave["snippet"] != "New parents get sixteen weeks of paid leave, in full.":
        found.append("a source card does not show a piece of its passage")
    if leave["address"] != LEAVE_URL or faq["address"] != "ftp://handbook.example.com/faq":
        found.append("a source card does not give a screen reader the whole address")
    # Shown in brief: the host first and whole, the middle of the path left out, on one line.
    brief = leave["brief"]
    if (
        not brief.startswith(f"handbook.example.com {CRUMB} ")
        or "…" not in brief
        or len(brief) > 42
    ):
        found.append("the address in brief does not start with the host, or is not brief")
    if not (leave["briefHiddenFromReaders"] and faq["briefHiddenFromReaders"]):
        found.append("a screen reader reads the address in brief as well as the whole one")
    if faq["brief"] != "ftp://handbook.example.com/faq":
        found.append("an address that is not a web address is shortened")
    if leave["linkTitle"] != LEAVE_URL:
        found.append("pointing at a source's link does not show its whole address")
    if leave["foldAddress"] != LEAVE_URL or faq["foldAddress"] != "ftp://handbook.example.com/faq":
        found.append("the fold with the passage does not write out the whole address")
    linking = cited["linking"]
    if linking["pointChip"]["cards"] != ["2"]:
        found.append("pointing at a number in the answer does not mark its source card alone")
    if linking["focusChip"] != {"cards": ["1"], "chips": ["1", "1"]}:
        found.append("focusing a number in the answer does not mark its source card")
    if linking["pointCard"]["chips"] != ["1", "1"]:
        found.append("pointing at a source card does not mark the numbers that cite it")
    if linking["focusCard"]["chips"] != ["2"]:
        found.append("focusing inside a source card does not mark the numbers that cite it")
    for moved in ("leaveChip", "blurChip", "leaveCard", "unfocusCard"):
        if linking[moved] != {"cards": [], "chips": []}:
            found.append("a mark stays after the pointer or the focus has left")
    press = linking["pressChip"]
    if press["scrolled"] != 1:
        found.append("pressing a number does not bring its source card into view")
    if press["marked"] != ["2"] or "Source 2 is marked" not in press["status"]:
        found.append("pressing a number does not mark its source card and say so")
    return found


def key_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the keys of the question box and the two buttons beside it get wrong."""
    found = []
    keys = {key["name"]: key for key in steps["keys"]["keys"]}
    for name in ("enter", "ctrl+enter", "cmd+enter"):
        key = keys[name]
        if key["sent"] != 1 or not key["prevented"] or key["box"] != "":
            found.append(f"{name.capitalize()} does not ask the question in the box")
    shift = keys["shift+enter"]
    if shift["sent"] != 0 or shift["prevented"]:
        found.append("Shift+Enter asks, or does not start a new line")
    for name in ("composing", "composing, 229"):
        if keys[name]["sent"] != 0 or keys[name]["prevented"]:
            found.append(f"an Enter that ends a composition asks ({name})")
    if keys["empty"]["sent"] != 0:
        found.append("Enter sends an empty question")
    if keys["too long"]["sent"] != 0:
        found.append("Enter sends a question over the limit")
    busy = keys["while busy"]
    if busy["sent"] != 0 or busy["box"] != "The second of two":
        found.append("Enter asks while an answer is being written")
    # Stop shows only while an answer is being written, and Ask then says it is busy.
    for name, step in steps.items():
        under_way = name in UNDER_WAY
        if step["stopHidden"] == under_way:
            found.append(f"the stop button is shown or hidden wrongly ({name})")
        if step["askDisabled"] != ("true" if under_way else "false"):
            found.append(f"Ask does not say whether an answer is being written ({name})")
    if steps["keys"]["busy"] != {"stopHidden": False, "askDisabled": "true"}:
        found.append("the stop button is shown or hidden wrongly (keys)")
    return found


def label_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """Whether the word over each card fits how its turn ended."""
    found = []
    wanted = {
        **dict.fromkeys(ANSWERED, "Answer"),
        **dict.fromkeys(DECLINED, "Declined"),
        **dict.fromkeys(FAILED, "Failed"),
        "withheld": "Reply",
        "stopped": "Stopped",
    }
    for name, word in wanted.items():
        step = steps[name]
        if step["cardTitle"] != word or step["ended"] != [f"ended-{word.lower()}"]:
            found.append(f"the card of a turn is not labelled by how it ended ({name})")
    # The strip of badges names what it sums up: an answer, or a reply that is not one.
    for name in (*ANSWERED, *DECLINED, "withheld"):
        made = "answer" if name in ANSWERED else "reply"
        if steps[name]["madeTitle"] != f"How this {made} was made":
            found.append(f"the strip under a reply calls it an answer ({name})")
    return found


def faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the page did wrong in the conversation, in words. Empty when nothing."""
    return [
        *conversation_faults(steps),
        *turn_faults(steps),
        *badge_faults(steps),
        *source_faults(steps),
        *key_faults(steps),
        *label_faults(steps),
    ]


def test_a_conversation_on_the_page_does_what_the_documents_say(tmp_path: Path) -> None:
    assert faults(run(script(), tmp_path)) == []


PLANTED_IN_THIS_ROUND = [
    # The keys of the question box.
    (
        "Enter does not ask the question in the box",
        "!event.shiftKey && ",
        "(event.ctrlKey || event.metaKey) && ",
    ),
    ("Shift+Enter asks, or does not start a new line", "!event.shiftKey && ", ""),
    ("an Enter that ends a composition asks (composing)", "!event.isComposing && ", ""),
    ("an Enter that ends a composition asks (composing, 229)", " && event.keyCode !== 229", ""),
    ("Enter sends an empty question", '  if (question === "") {', "  if (false) {"),
    (
        "Enter sends a question over the limit",
        "if (page.question.value.length > page.question.maxLength) {",
        "if (false) {",
    ),
    (
        "Enter asks while an answer is being written",
        '    say("An answer is still being written. Stop it first, or wait for it.");\n'
        "    return false;",
        '    say("An answer is still being written. Stop it first, or wait for it.");',
    ),
    # The buttons beside the box.
    (
        "the stop button is shown or hidden wrongly (first)",
        "  page.stop.hidden = !busy;\n",
        "",
    ),
    (
        "Ask does not say whether an answer is being written (writing)",
        '  page.ask.setAttribute("aria-disabled", String(busy));\n',
        "",
    ),
    # The word over each card.
    (
        "the card of a turn is not labelled by how it ended (declined)",
        "  label(turn, cardLabel(answer));\n",
        "",
    ),
    (
        "the card of a turn is not labelled by how it ended (withheld)",
        'return isDecline(answer) ? "Declined" : "Reply";',
        'return "Declined";',
    ),
    (
        "the card of a turn is not labelled by how it ended (refused)",
        '  label(turn, "Failed");\n',
        "",
    ),
    (
        "the card of a turn is not labelled by how it ended (stopped)",
        '  label(turn, "Stopped");\n',
        "",
    ),
    (
        "the strip under a reply calls it an answer (declined)",
        'answer.answered ? "How this answer was made" : "How this reply was made"',
        '"How this answer was made"',
    ),
    # The address of a source.
    (
        "the address in brief does not start with the host, or is not brief",
        "[parsed.host, ...list]",
        "[...list]",
    ),
    (
        "the address in brief does not start with the host, or is not brief",
        "  if (crumbs(parts).length <= length) {",
        "  if (true) {",
    ),
    (
        "a screen reader reads the address in brief as well as the whole one",
        '  brief.setAttribute("aria-hidden", "true");\n',
        "",
    ),
    (
        "a source card does not give a screen reader the whole address",
        '  line.append(brief, make("span", "visually-hidden", shown(url)));',
        "  line.append(brief);",
    ),
    ("pointing at a source's link does not show its whole address", "  link.title = url;\n", ""),
    (
        "the fold with the passage does not write out the whole address",
        "  fold.append(summary, address, ",
        "  fold.append(summary, ",
    ),
]


@pytest.mark.parametrize(("fault", "old", "new"), PLANTED_IN_THIS_ROUND)
def test_the_run_finds_each_fault_planted_in_the_keys_buttons_labels_and_addresses(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in faults(run(source.replace(old, new), tmp_path))


@pytest.mark.parametrize(
    ("fault", "old", "new"),
    [
        # The conversation that is sent (ADR 0024).
        (
            "the conversation does not grow by the question searched for",
            "if (showAnswer(turn, data)) {",
            "if (showAnswer(turn, data) && false) {",
        ),
        (
            '"Start a new conversation" does not forget the conversation',
            "  forget();\n  say(",
            "  say(",
        ),
        (
            "the side search is not run on the question searched for",
            "search(turn, searched.question);",
            "search(turn, question);",
        ),
        (
            "a line about the question searched for is left showing",
            'turn.searchedLine.hidden = line === "";',
            "turn.searchedLine.hidden = false;",
        ),
        (
            "a refused question shows a line about the search",
            "  turn.searchedLine.hidden = true;\n",
            "",
        ),
        (
            "a turn that failed (error after text) is added to the conversation",
            '} else if (event.name === "error") {\n      ended = true;',
            '} else if (event.name === "error") {\n      ended = true;\n      remember(question);',
        ),
        (
            "a follow-up the service did not rewrite says nothing of it",
            'if (followUp && searched.outcome === "not_asked") {',
            "if (false) {",
        ),
        (
            "the earlier questions sent are not the conversation",
            "const earlier = conversation.slice(-EARLIER_SENT);",
            "const earlier = [];",
        ),
        (
            "a follow-up whose rewrite failed does not say so",
            'if (searched.outcome === "failed" || searched.outcome === "unusable") {',
            "if (false) {",
        ),
        (
            "a follow-up whose rewrite was not usable does not say so",
            'if (searched.outcome === "failed" || searched.outcome === "unusable") {',
            'if (searched.outcome === "failed") {',
        ),
        (
            "a turn that failed (unreadable answer) is added to the conversation",
            "  if (!isAnswer(data)) {\n    fail(turn, UNREADABLE_ANSWER);\n    return false;",
            "  if (!isAnswer(data)) {\n    fail(turn, UNREADABLE_ANSWER);\n    return true;",
        ),
        (
            "the list on the page is not the conversation (rewritten)",
            "  page.conversation.replaceChildren(",
            "  [].push(",
        ),
        (
            "the list on the page is hidden or shown wrongly (new conversation)",
            "page.conversationArea.hidden = conversation.length === 0;",
            "page.conversationArea.hidden = false;",
        ),
        # The turns.
        (
            "a turn does not show the question asked as its own bubble",
            'bubble.append(make("span", "visually-hidden", "You asked: "), question);',
            'bubble.append(make("span", "visually-hidden", "You asked: "));',
        ),
        (
            "the welcome is shown or hidden wrongly (new conversation)",
            "page.welcome.hidden = turns.length > 0;",
            "page.welcome.hidden = true;",
        ),
        (
            '"Start a new conversation" leaves the turns on the page',
            "  clearThread();\n",
            "",
        ),
        (
            "a decline is not shown as a calm notice (declined)",
            'make("div", declined ? "notice notice-decline" : "notice notice-unfinished")',
            'make("div", "error-card")',
        ),
        (
            "a reply that was withheld or cut off is shown as a decline or a failure",
            '(answer.stop_reason === "end_turn" || answer.stop_reason === "no_context")',
            "(true)",
        ),
        (
            "a failure is not shown in its turn (refused)",
            "  turn.error.hidden = false;\n",
            "",
        ),
        (
            "the failure card does not say what the alert says (stream cut)",
            "  turn.errorText.textContent = told;\n",
            "  turn.errorText.textContent = words;\n",
        ),
        (
            "a stopped answer does not say so in its turn",
            "  turn.note.hidden = false;\n",
            "",
        ),
        (
            "a turn that waits shows no placeholders",
            "  turn.sources.append(placeholderCard(), placeholderCard());\n",
            "",
        ),
        (
            "the placeholder stays while the text is written",
            "  turn.pending.hidden = true;\n  turn.stream.hidden = false;",
            "  turn.stream.hidden = false;",
        ),
        (
            "the text as it arrives has no caret",
            '  turn.stream.classList.add("streaming");\n',
            "",
        ),
        (
            "the caret stays after the turn has ended (stream cut)",
            '  turn.stream.classList.remove("streaming");\n',
            "",
        ),
        (
            "the waiting look stays after the turn has ended (refused)",
            "  turn.pending.hidden = true;\n  turn.progress.hidden = true;\n",
            "",
        ),
        (
            "sources that never came leave placeholders (refused)",
            "  if (turn.waiting) {",
            "  if (false) {",
        ),
        # The badges.
        (
            "a cached answer has no badge that says so",
            '    badges.push(["From the cache", tier, "cache"]);\n',
            "",
        ),
        (
            "the badges do not say how many passages the search found",
            'results.length === 0 ? "nothing found" : `${count} found`',
            '"searching"',
        ),
        (
            "a cached answer counts tokens it never spent",
            'hit === null ? tokensInBrief(answer.usage) : "none: no request"',
            "tokensInBrief(answer.usage)",
        ),
        (
            "a failure shows badges of an answer it does not have (refused)",
            "  turn.badges.hidden = true;\n",
            "",
        ),
        (
            "a failure does not show the trace id it came with",
            '  failedFacts(turn, response.headers.get("X-Trace-Id"));\n',
            "",
        ),
        # The sources and the numbers that lead to them.
        (
            "the numbers in the answer are not the sources it lists",
            "numbers: segment.sources.filter((number) => listed.has(number)),",
            "numbers: segment.sources,",
        ),
        (
            "a source that is not a web address becomes a link",
            '  if (!isWebAddress) {\n    const plain = make("span", "", title);',
            '  if (false) {\n    const plain = make("span", "", title);',
        ),
        (
            "a source card does not show the page's title and the headings above",
            '    path.append(make("span", "crumb", heading));\n',
            "",
        ),
        (
            "a source card does not show a piece of its passage",
            '    card.append(make("p", "source-snippet", piece));\n',
            "",
        ),
        (
            "pointing at a number in the answer does not mark its source card alone",
            '  turn.cards.get(number)?.classList.toggle("linked", on);',
            '  for (const card of turn.cards.values()) card.classList.toggle("linked", on);',
        ),
        (
            "pointing at a number in the answer does not mark its source card alone",
            '  button.addEventListener("mouseenter", () => link(turn, number, true));\n',
            "",
        ),
        (
            "focusing a number in the answer does not mark its source card",
            '  button.addEventListener("focus", () => link(turn, number, true));\n',
            "",
        ),
        (
            "pointing at a source card does not mark the numbers that cite it",
            '  card.addEventListener("mouseenter", () => link(turn, number, true));\n',
            "",
        ),
        (
            "focusing inside a source card does not mark the numbers that cite it",
            '  card.addEventListener("focusin", () => link(turn, number, true));\n',
            "",
        ),
        (
            "a mark stays after the pointer or the focus has left",
            '  button.addEventListener("mouseleave", () => link(turn, number, false));\n',
            "",
        ),
        (
            "a mark stays after the pointer or the focus has left",
            '  card.addEventListener("focusout", () => link(turn, number, false));\n',
            "",
        ),
        (
            "pressing a number does not bring its source card into view",
            '    card.scrollIntoView({ block: "nearest" });\n',
            "",
        ),
        (
            "pressing a number does not mark its source card and say so",
            '    card.classList.toggle("marked", each === number);\n',
            "",
        ),
    ],
)
def test_the_run_finds_each_fault_planted_in_the_script(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in faults(run(source.replace(old, new), tmp_path))
