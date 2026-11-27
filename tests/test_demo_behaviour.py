"""The demo page's script, run: one conversation through a small fake page in node.

``tests/test_demo.py`` reads the script's source. This runs it, with ``node`` and
``tests/demo_page_harness.mjs``: a fake page with only what the script touches, and a
stubbed ``fetch`` that answers ``/ask`` as the service does, as a stream with the question
searched for first. The project has no JavaScript toolchain (ADR 0016), and none is added:
node runs the script as it is. Without node the tests are skipped, and say why.

The harness reads back what a reader would see in each turn: the question's sign, the
answer or the notice or the failure card, the placeholders and the caret while the answer
comes, the line of stops that says how it was made, its source cards and found passages,
and what pointing at, focusing and pressing a number in the answer does to them; and
around the turns, the buttons and their states, the focus, the status line and the alert,
the examples, "Ask again", the copy of a trace id, and the count of what is sent.

A second test plants faults in the script and asks that the first one would have seen
each, so that a check which saw nothing would not pass unnoticed.
"""

import json
import shutil
import subprocess
from html.parser import HTMLParser
from importlib.resources import files
from pathlib import Path
from typing import Any, ClassVar

import pytest

from tests.test_demo import (
    ADDRESS,
    AROUND,
    AROUND_IDS,
    HOLDERS,
    css_rules,
    cuts_the_address,
    static,
)

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
ANSWERED = (
    "first",
    "rewritten",
    "given back",
    "asked by itself",
    "cited",
    "semantic",
    "fallback",
    "no searched event",
    "arrived",
    "stop after answer",
    "asked again",
    "check dropped line",
    "two checks",
    "malformed checks",
    "malformed items",
)
DECLINED = ("declined", "cached", "check declined", "form declined")
FAILED = (
    "error after text",
    "unreadable answer",
    "refused",
    "late failure",
    "stream cut",
    "failure before new",
)
# The words of a reply made only of the fixed form, which carried citation markers: the
# service sends it as a decline in those words, with no source (ADR 0025).
FORM = "The passages found do not say how much the backup care benefit pays per year."
# Steps recorded while an answer was still coming, when its turn had not ended.
UNDER_WAY = ("waiting", "searched", "writing", "waiting first")


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
    steps = json.loads(done.stdout)
    # A step with no turn on the page, such as one a fault kept from being asked, has none
    # of a turn's fields: it is given them empty, so that the checks say what is missing
    # instead of failing on a missing field.
    template: dict[str, Any] = next((step for step in steps if "sign" in step), {})
    for step in steps:
        for key, value in template.items():
            step.setdefault(key, None if value is None else type(value)())
    return {step["name"]: step for step in steps}


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
    # The question a follow-up was rewritten to stands at its stop of the line; the line
    # above the reply, which says why a follow-up was searched for as typed, stays hidden.
    rewrite = next((stop for stop in rewritten["stops"] if stop["name"] == "Rewritten"), None)
    if rewrite is None or rewrite["detail"] != REWRITTEN or not rewritten["searchedHidden"]:
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
        if step["question"] is not None and step["sign"] != [f"You asked: {step['question']}"]:
            found.append("a turn does not show the question asked as its own sign")
    # How each turn ended, and what it looks like then.
    for name in ANSWERED:
        step = steps[name]
        if not step["answer"] or step["decline"] or step["error"]:
            found.append(f"an answer is not shown as an answer ({name})")
    for name in DECLINED:
        step = steps[name]
        if not step["decline"] or step["error"] or step["answer"]:
            found.append(f"a decline is not shown as a calm notice ({name})")
        words = FORM if name == "form declined" else "I could not find this in the handbook."
        if words not in " ".join(step["notice"]):
            found.append(f"a decline does not show the reply's own words ({name})")
        if step["chips"] or step["cards"]:
            found.append(f"a decline shows a number or a source card ({name})")
        if step["sourcesNone"] != "Nothing is cited: this reply is not an answer.":
            found.append(f"a decline does not say that nothing is cited ({name})")
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
    if (
        not waiting["pending"]
        or waiting["placeholders"] != 2
        or not any(stop["state"] == "next" for stop in waiting["stops"])
    ):
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
        if step["pending"] or any(stop["state"] in ("next", "ahead") for stop in step["stops"]):
            found.append(f"the waiting look stays after the turn has ended ({name})")
        if step["placeholders"] != 0:
            found.append(f"sources that never came leave placeholders ({name})")
    return found


def stop_named(step: dict[str, Any], name: str) -> dict[str, str] | None:
    """The stop of a turn's line that has this name, or None."""
    return next((stop for stop in step["stops"] if stop["name"] == name), None)


def names(step: dict[str, Any]) -> list[str]:
    return [stop["name"] for stop in step["stops"]]


def detail(step: dict[str, Any], name: str) -> str:
    stop = stop_named(step, name)
    return "" if stop is None else stop["detail"]


def stop_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the line of stops and the timetable under an answer get wrong, in words: what
    the strip of badges said before, now said at the stops and under "All details"."""
    found = []
    cited = steps["cited"]
    if detail(cited, "Searched") != "3 passages found":
        found.append("the line does not say how many passages the search found")
    if (
        detail(cited, "Rewritten")
        != "How many weeks of parental leave do new parents get in the US?"
    ):
        found.append("a rewritten follow-up does not show the question it was searched for")
    if "2 sources cited" not in detail(cited, "Answered"):
        found.append("the line does not say how many sources the answer cites")
    for label in (
        "Modelstand-in",
        "Tokens1 read, 1 written",
        "First text after0.0 s",
        "Whole answer after0.0 s",
    ):
        if label not in cited["facts"]:
            found.append(f"the timetable misses {label!r} or gets it wrong")
    if any(
        stop["name"] in ("Rewritten", "Stood alone", "Not rewritten", "Rewriting")
        for stop in steps["first"]["stops"]
    ):
        found.append("a question asked by itself has a stop about a follow-up")
    if "searched as typed" not in detail(steps["unusable rewrite"], "Not rewritten"):
        found.append("a follow-up searched for as typed does not say so at its stop")
    cached = steps["cached"]
    if detail(cached, "Found stored") != "an answer stored for the same question":
        found.append("a cached answer does not say at its stop that it was stored")
    if "Tokensnone: no request was sent" not in cached["facts"]:
        found.append("a cached answer counts tokens it never spent")
    if f"It was written for the question: {STORED}" not in cached["cached"]:
        found.append("a cached answer does not name the question it was written for")
    if stop_named(steps["first"], "Found stored") is not None:
        found.append("an answer written now says that it came from the store")
    for name in FAILED:
        if stop_named(steps[name], "Answered") is not None:
            found.append(f"a failure has the stop of an answer it does not have ({name})")
    refused = steps["refused"]
    if refused["routeLabel"] != "How this request went" or TRACE_ID not in refused["facts"]:
        found.append("a failure does not show the trace id it came with")
    return found


def route_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """Which stop of the line is reached when, in which words, and where the line ends:
    for an answer, a rewritten follow-up, a stored answer, a decline, a check's decline,
    a reply cut off, a failure and a stop, and while an answer is being written."""
    found = []
    first = steps["first"]
    if names(first) != ["Asked", "Searched", "First words", "Answered"] or {
        stop["state"] for stop in first["stops"]
    } != {"reached"}:
        found.append("an answer's line does not run through its four stops")
    if [stop["segment"] for stop in first["stops"]] != ["solid", "solid", "solid", "none"]:
        found.append("an answer's line is not drawn solid from stop to stop")
    if (
        detail(first, "Asked") != "by itself"
        or detail(steps["rewritten"], "Asked") != "with 1 earlier question"
    ):
        found.append("the stop where a question was asked does not say what went with it")
    if names(steps["rewritten"]) != ["Asked", "Rewritten", "Searched", "First words", "Answered"]:
        found.append("a follow-up's line has no stop of its own for the rewrite")
    for name, wanted in (
        ("failed rewrite", "the request failed"),
        ("not rewritten", "does not rewrite"),
    ):
        if wanted not in detail(steps[name], "Not rewritten"):
            found.append(
                f"a follow-up that was not rewritten does not say why at its stop ({name})"
            )
    for name, how in (("cached", "the same question"), ("semantic", "judged to mean the same")):
        step = steps[name]
        if how not in detail(step, "Found stored") or stop_named(step, "Searched") is not None:
            found.append(f"a stored answer's line is not another one, with no search ({name})")
    if "no request to the model" not in detail(steps["semantic"], "Answered"):
        found.append("a stored answer's line does not say that no request was sent")
    for name in ("declined", "form declined"):
        step = steps[name]
        tail = [(stop["name"], stop["state"]) for stop in step["stops"][-2:]]
        if tail != [("No answer in them", "off"), ("Not answered", "off")]:
            found.append(f"a decline's line does not run on to stops it does not serve ({name})")
        searched = stop_named(step, "Searched")
        if searched is None or searched["segment"] != "dashed":
            found.append(f"a decline's line is not dashed after the search ({name})")
    check = steps["check declined"]["stops"]
    if [(stop["name"], stop["state"]) for stop in check[-2:]] != [
        ("First words", "reached"),
        ("Declined by a check", "off"),
    ]:
        found.append("a check's decline is not where its line ends")
    for name, end, words in (
        ("refused", "Not answered", "the request failed"),
        ("error after text", "Broke off", "after the first words"),
        ("stream cut", "Broke off", "after the first words"),
        ("stopped", "Stopped by you", "after the first words"),
        ("withheld", "Withheld", "the model stopped: safety"),
        ("check cut", "Stopped early", "the model stopped: other"),
        ("cut at the limit", "Cut off", "the model stopped: max_tokens"),
    ):
        last = steps[name]["stops"][-1]
        if last["name"] != end or last["state"] != "end" or words not in last["detail"]:
            found.append(f"a line that ended short does not end where it ended ({name})")
        elif last["told"] != " (the line ends here)":
            found.append(f"a screen reader is not told where a line ends ({name})")
    if names(steps["refused"]) != ["Asked", "Not answered"]:
        found.append("a refused question's line runs past where it ended")
    if stop_named(steps["error after text"], "First words") is None:
        found.append("a line that broke off after its first words does not show them reached")
    # While an answer is being written, the line is the progress display.
    progress = {
        "waiting": [
            ("Asked", "reached"),
            ("Rewriting", "next"),
            ("Searched", "ahead"),
            ("First words", "ahead"),
            ("Answered", "ahead"),
        ],
        "searched": [
            ("Asked", "reached"),
            ("Stood alone", "reached"),
            ("Searched", "reached"),
            ("First words", "next"),
            ("Answered", "ahead"),
        ],
        "writing": [
            ("Asked", "reached"),
            ("Stood alone", "reached"),
            ("Searched", "reached"),
            ("First words", "reached"),
            ("Answered", "next"),
        ],
    }
    for name, stops in progress.items():
        if [(stop["name"], stop["state"]) for stop in steps[name]["stops"]] != stops:
            found.append(f"the line does not show how far the answer has come ({name})")
    told = {stop["state"]: stop["told"] for stop in steps["waiting"]["stops"]}
    if told.get("next") != " (next stop)" or told.get("ahead") != " (not yet)":
        found.append("a screen reader is not told which stop is next and which are not reached")
    if steps["waiting"]["stops"][0]["segment"] != "faint":
        found.append("the line to a stop not reached yet is drawn as if it were reached")
    for name, label in (
        ("waiting", "How this answer is being made"),
        ("first", "How this answer was made"),
        ("declined", "How this reply was made"),
        ("stopped", "How this request went"),
    ):
        if steps[name]["routeLabel"] != label:
            found.append(f"the line is not named for what it shows ({name})")
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
    if not (leave["addressShown"] and faq["addressShown"]):
        found.append("the address under a source is not shown")
    # Shown in brief: the host first and whole, the middle of the path left out, on one line.
    brief = leave["brief"]
    if brief != f"handbook.example.com {CRUMB} … {CRUMB} leave-in-the-united-states":
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
    busy = steps["keys"]["busy"]
    if (busy["stopHidden"], busy["askDisabled"]) != (False, "true"):
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
        "check cut": "Reply",
        "stopped": "Stopped",
    }
    for name, word in wanted.items():
        step = steps[name]
        if step["cardTitle"] != word or step["ended"] != [f"ended-{word.lower()}"]:
            found.append(f"the card of a turn is not labelled by how it ended ({name})")
    # The line names what it shows: an answer, or a reply that is not one.
    for name in (*ANSWERED, *DECLINED, "withheld", "check cut"):
        made = "answer" if name in ANSWERED else "reply"
        if steps[name]["routeLabel"] != f"How this {made} was made":
            found.append(f"the line of a reply calls it an answer ({name})")
    return found


def control_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the controls around the turns get wrong: the examples, "Ask again", the copy of
    a trace id, the new-conversation button, the alert, the count of what is sent."""
    found = []
    by_itself = steps["asked by itself"]
    if by_itself["turnCount"] != 1 or by_itself["sign"] != [f"You asked: {FIRST}"]:
        found.append("an example button does not ask its question")
    busy = steps["keys"]["busy"]
    for name, step in steps.items():
        disabled = "true" if name in UNDER_WAY else "false"
        if step["examplesDisabled"] != [disabled] * 4:
            found.append(
                f"the examples are not marked as unavailable while an answer is written ({name})"
            )
        if step["formBusy"] != (name in UNDER_WAY):
            found.append(f"Ask's busy look is shown or left wrongly ({name})")
        if step["newConversationHidden"] != (step["turnCount"] == 0 and step["conversation"] == []):
            found.append(f"the new conversation button is shown or hidden wrongly ({name})")
    if busy["examplesDisabled"] != ["true"] * 4 or not busy["formBusy"]:
        found.append("the examples are not marked as unavailable while an answer is written (keys)")
    again = steps["asked again"]
    if again["sent"] != 1 or again["query"] != "and the rest":
        found.append("Ask again does not ask the question again")
    if busy["askAgainSent"] != 0:
        found.append("Ask again sends while an answer is being written")
    copy = steps["refused"]["copy"]
    if copy["text"] != TRACE_ID or copy["status"] != "The trace id is copied.":
        found.append("the trace id is not copied, or the copy is not announced")
    if "could not be copied" not in copy["refusedStatus"]:
        found.append("a copy the clipboard refused is not announced")
    started = busy["newConversation"]
    if not (started["turnsKept"] and started["conversationKept"]) or not started[
        "status"
    ].startswith("An answer is still being written"):
        found.append("a new conversation begins while an answer is being written")
    if steps["new after failure"]["failure"] != "":
        found.append("the alert keeps a failure after a new conversation")
    if steps["waiting"]["failure"] != "":
        found.append("the alert keeps the last failure while the next question is answered")
    keys = steps["keys"]
    if len(keys["conversation"]) <= 10 or busy["earlierSent"] != 10:
        found.append("the run does not hold a conversation of more than ten questions")
    if keys["conversationSummary"] != "10 earlier questions are sent with your next question":
        found.append("the page says it sends more earlier questions than it does")
    if (
        steps["first"]["conversationSummary"]
        != "1 earlier question is sent with your next question"
    ):
        found.append("the count of what is sent is not said, or not in good grammar")
    return found


def announcement_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the status line and the words for a screen reader get wrong."""
    found = []
    if (
        steps["declined"]["status"]
        != "Not in the handbook: the assistant declined instead of guessing."
    ):
        found.append("a decline or a withheld reply is announced as the other")
    if not steps["withheld"]["status"].startswith("Not an answer: the reply was withheld"):
        found.append("a decline or a withheld reply is announced as the other")
    if not steps["cached"]["status"].endswith("It came from the cache."):
        found.append("an answer from the cache is not announced as one")
    if steps["fallback"]["status"] != "The answer is complete. It cites 1 source.":
        found.append("the status line counts sources the text does not cite")
    for name in FAILED:
        if steps[name]["status"] != "No answer.":
            found.append(f"a failure is not announced ({name})")
    cited = steps["cited"]
    if cited["chipLabels"] != ["Source 1: Leave", "Source 1: Leave", "Source 2: Leave FAQ"]:
        found.append("a citation is read as a bare number")
    names = [card["foldName"] for card in cited["cards"]]
    names += [found_passage["foldName"] for found_passage in cited["passages"]]
    if names != [" of source 1, Leave", " of source 2, Leave FAQ"] + [
        f" of found passage {rank}, Leave" for rank in (1, 2, 3)
    ]:
        found.append("a fold is not named for its passage")
    if not all(item["foldAddressHidden"] for item in [*cited["cards"], *cited["passages"]]):
        found.append("a screen reader reads the whole address twice")
    return found


def search_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the fold of found passages and the stop of the search get wrong."""
    found = []
    cited = steps["cited"]
    if (
        not cited["searchFoldShown"]
        or cited["searchCount"] != "3 more passages were found and not cited."
    ):
        found.append("the fold of what the search found is not shown")
    passages = cited["passages"]
    if [item["rank"] for item in passages] != ["1.", "2.", "3."]:
        found.append("the fold says how many passages were found and lists none")
    elif any(item["link"] != LEAVE_URL for item in passages):
        found.append("a found passage's title is not a link to its page")
    elif any(
        item["address"] != LEAVE_URL
        or not item["brief"].startswith("handbook.example.com")
        or not item["addressShown"]
        for item in passages
    ):
        found.append("a found passage does not show where its link leads")
    for name, wanted in (
        ("declined", "1 passage found"),
        ("withheld", "not counted: the page's own search failed"),
        ("no searched event", "not counted: the service did not say what it searched for"),
    ):
        if detail(steps[name], "Searched") != wanted:
            found.append(f"the stop of the search does not say what the search did ({name})")
    if steps["withheld"]["searchState"] != "The search failed. The service could not be reached.":
        found.append("a search that could not reach the service does not say so")
    if steps["cached"]["searchState"] != "The search failed. The service is busy.":
        found.append("a search the service refused does not say so")
    return found


def answered_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """Whether what stands out about how an answer was made is said at its last stop, and
    only there: more than one request, the fallback model, the sources its text cites."""
    found = []
    fallback = detail(steps["fallback"], "Answered")
    for name in ANSWERED:
        said = detail(steps[name], "Answered")
        if name != "fallback" and ("requests" in said or "fallback" in said):
            found.append(
                f"an ordinary answer is said to have taken more requests or the fallback ({name})"
            )
    if "after 2 requests" not in fallback or "by the fallback model" not in fallback:
        found.append("an answer of two requests by the fallback model is not said so")
    if "Requests to the model2" not in steps["fallback"]["facts"]:
        found.append("the timetable does not give the requests an answer took")
    if "1 source cited" not in fallback:
        found.append("the line counts sources the text does not cite")
    if "the request failed" not in detail(steps["failed rewrite"], "Not rewritten"):
        found.append("a follow-up whose rewrite failed is said to be something else")
    for name in (*DECLINED, "withheld"):
        if stop_named(steps[name], "Answered") is not None:
            found.append(f"a reply that is not an answer has the stop of an answer ({name})")
    semantic = steps["semantic"]
    if "judged to mean the same" not in detail(semantic, "Found stored"):
        found.append("a match by meaning is said to be something else")
    if "That judgement can be wrong." not in semantic["cached"]:
        found.append("a match by meaning does not warn that it can be wrong")
    return found


def outcome_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What each turn shows once it has ended, besides its card's look."""
    found = []
    after = steps["stop after answer"]
    if after["cardTitle"] != "Answer" or after["note"] != "":
        found.append("Stop pressed after the answer came marks it as stopped")
    if steps["stopped"]["focus"] != "question":
        found.append("the focus is lost when the stop button goes")
    for name in FAILED:
        if steps[name]["sourcesShown"]:
            found.append(f"a failed turn keeps a Sources heading over nothing ({name})")
    if steps["first"]["sourcesNone"] != "This answer cites no source.":
        found.append("an answer with no source says nothing under its Sources heading")
    if steps["error after text"]["madeShown"] or not steps["refused"]["madeShown"]:
        found.append("a failure without a trace id shows an empty About this request")
    if "Why the model stopped: safety." not in " ".join(steps["withheld"]["notice"]):
        found.append("a withheld reply does not say why the model stopped")
    for name in ("error after text", "stream cut", "stopped"):
        if not steps[name]["streamUnfinished"]:
            found.append(
                f"text that arrived before a failure or a stop is not marked unfinished ({name})"
            )
    return found


def look_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the brief address, the piece of a passage and the motion of the page get wrong."""
    found = []
    last = steps["new after failure"]
    briefs = last["briefs"]
    host = "very-long-host-name-" * 4 + "example.com"
    wanted = {
        "https://handbook.example.com": "handbook.example.com",
        "https://handbook.example.com/": "handbook.example.com",
        f"https://{host}/": host,
        f"https://{host}/handbook/people/leave/": f"{host} {CRUMB} … {CRUMB} leave",
        "https://handbook.gitlab.com/handbook/total-rewards/benefits/general-and-entity-benefits/"
        "gitlab-ireland-ltd/": f"handbook.gitlab.com {CRUMB} … {CRUMB} gitlab-ireland-ltd",
        "ftp://handbook.example.com/faq": "ftp://handbook.example.com/faq",
    }
    if any(brief.startswith("threw") for brief in briefs.values()):
        found.append("the address in brief throws")
    elif any(briefs[url] != brief for url, brief in wanted.items()):
        found.append("the address in brief does not keep the host whole and name the page")
    if not briefs["https://handbook.gitlab.com@evil.example/handbook/people/leave/"].startswith(
        "evil.example"
    ) or not briefs["https://h\u0430ndbook.gitlab.com/handbook/people/leave/"].startswith("xn--"):
        found.append("the address in brief does not show the host the browser reads")
    if not briefs["https://handbook.example.com:8443/handbook/people/leave/"].startswith(
        "handbook.example.com:8443 "
    ):
        found.append("the address in brief does not show the host the browser reads")
    long_part = briefs[
        "https://handbook.gitlab.com/handbook/" + "a-very-long-part-of-a-path-" * 3 + "end/"
    ]
    if not long_part.endswith("…t-of-a-path-end") or not long_part.startswith(
        f"handbook.gitlab.com {CRUMB} … {CRUMB} a-very-long-par"
    ):
        found.append("the address in brief does not cut a long last part in its middle")
    piece, text = last["snippet"], last["longText"]
    whole_words = text.startswith(piece[:-1]) and text[len(piece) - 1 : len(piece)] == " "
    if not piece.endswith("…") or len(piece) > 201 or not whole_words:
        found.append("the piece of a passage is not cut short at a word")
    for name, step in steps.items():
        if step["question"] is None or "scrolled" not in step:
            continue
        first_of_conversation = name in (
            "first",
            "asked by itself",
            "waiting first",
            "stopped early",
        )
        if (step["scrolled"] == 0) != first_of_conversation:
            found.append(f"a later turn is not brought into view, or the first is ({name})")
    if detail(steps["writing"], "First words") != "arriving now":
        found.append("the line does not say that the first words are arriving")
    for name, as_typed in (
        ("failed rewrite", True),
        ("unusable rewrite", True),
        ("not rewritten", True),
        ("rewritten", False),
        ("given back", False),
    ):
        if steps[name]["searchedAsTyped"] != as_typed:
            found.append("the warning look of a follow-up searched as typed is wrong")
    if last["consoleErrors"] != ["Failed to fetch"]:
        found.append(f"the script wrote errors no step should cause: {last['consoleErrors']}")
    return found


# The notes of the checks in code, in the fixed words the service sends (ADR 0025).
DROPPED = (
    "Text after the answer's last citation was left out: it cited no passage and held a web "
    "or email address."
)
COPIED = "Most of this reply repeats the passages it was given, word for word."
UNCITED = (
    "The reply the model wrote cited none of the passages, so it is not shown: a reply that "
    "rests on no passage cannot be checked against the handbook."
)
UNFINISHED = (
    "The reply did not come to a proper end, and its words after the last citation are not shown."
)
CHECKED = {
    "check dropped line": [DROPPED],
    "two checks": [DROPPED, COPIED],
    "check declined": [UNCITED],
    "check cut": [UNFINISHED],
}


def check_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """Whether every note of a check is shown in its own turn, where its reply's state
    says it belongs, and read out with the outcome; and whether a list of checks of
    another shape costs nothing."""
    found = []
    dropped, declined, cut = (
        steps["check dropped line"],
        steps["check declined"],
        steps["check cut"],
    )
    if steps["two checks"]["checkNotes"] != CHECKED["two checks"] or dropped["checkNotes"] != [
        DROPPED
    ]:
        found.append("the notes of a check are not shown")
    elif dropped["bodyOrder"] != ["answer-text", "check-notes"] or dropped["checkInNotice"] != [
        False
    ]:
        found.append("the notes of a check do not stand under the answer's text")
    if any(steps[name]["checkLabels"] != ["Changed by a check."] for name in CHECKED):
        found.append("the notes of a check are not marked as a change")
    for step, notes in ((declined, [UNCITED]), (cut, [UNFINISHED])):
        if step["checkNotes"] != notes or step["checkInNotice"] != [True]:
            found.append(
                "the note of a check on a reply that is not an answer is not with its notice"
            )
    if declined["noticeTitle"] != ["Declined by a check."] or declined["noticeWhy"] != []:
        found.append("a reply a check declined is shown as one the handbook does not answer")
    if cut["noticeTitle"] != ["Not an answer."] or not cut["noticeWhy"]:
        found.append("a reply a check cut does not say why the model stopped")
    for name, notes in CHECKED.items():
        if not steps[name]["status"].endswith("".join(f" Note: {note}" for note in notes)):
            found.append(f"the note of a check is not announced with the outcome ({name})")
    if not declined["status"].startswith(
        "Declined by a check, in place of the reply the model wrote."
    ):
        found.append("a reply a check declined is announced as one the handbook does not answer")
    for name, words in (
        ("check dropped line", "changed by a check"),
        ("two checks", "changed by 2 checks"),
    ):
        if words not in detail(steps[name], "Answered"):
            found.append(f"a change by a check is not said at the end of the line ({name})")
    if stop_named(steps["check declined"], "Declined by a check") is None:
        found.append("a check's decline is not said at the end of the line")
    for name in ("malformed checks", "malformed items"):
        step = steps[name]
        if not step["answer"] or step["error"] or step["conversation"][-1:] != [step["question"]]:
            found.append(f"a list of checks of another shape loses the answer ({name})")
        elif step["checkNotes"] or "changed by a check" in detail(step, "Answered"):
            found.append(f"a list of checks of another shape gives a note ({name})")
    for name, step in steps.items():
        if name not in CHECKED and step["checkNotes"]:
            found.append(f"the notes of a check are shown on another turn ({name})")
        if step["firstTurnCheckNotes"]:
            found.append("the notes of a check are shown on another turn (the first)")
    if steps["two checks"]["previousTurnCheckNotes"] != [DROPPED]:
        found.append("the notes of a check leave their own turn")
    return found


def line_end_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """Where a line ends and in which words, in the states the first round left untested:
    a stop before the first words, a failure after the search and before any text, an
    answer that came with no text, and what each kind of failure is called; and what a
    screen reader is told of the stops."""
    found = []
    early = steps["stopped early"]
    if names(early) != ["Asked", "Searched", "Stopped by you"] or (
        detail(early, "Stopped by you") != "before the first words"
    ):
        found.append("a stop before the first words is not said where it ended")
    after = steps["failed after search"]
    if names(after) != ["Asked", "Searched", "Not answered"]:
        found.append("a failure after the search does not keep the search it ran")
    elif detail(after, "Not answered") != "the request failed":
        found.append("a failure after the search does not say what failed")
    for name, words in (
        ("error after text", "after the first words: the request failed"),
        ("unreadable answer", "after the first words: the answer could not be read"),
        ("stream cut", "after the first words: the stream ended without the answer"),
    ):
        if steps[name]["stops"][-1:] and steps[name]["stops"][-1]["detail"] != words:
            found.append(f"a line that broke off does not say what went wrong ({name})")
    if detail(steps["no text"], "No answer in them") != "no text came":
        found.append("a reply that came with no text is said to have had first words")
    for name in ("declined", "no text"):
        told = [stop["told"] for stop in steps[name]["stops"] if stop["state"] == "off"]
        if not told or any(each != " (not served)" for each in told):
            found.append(
                f"a screen reader is not told which stops the line does not serve ({name})"
            )
    writing = {stop["name"]: stop["told"] for stop in steps["writing"]["stops"]}
    if writing.get("First words") != "" or writing.get("Answered") != " (next stop)":
        found.append(
            "a screen reader is told that a stop not reached yet is where the answer is now"
        )
    if any(step.get("stops") and step.get("routeTag") != "OL" for step in steps.values()):
        found.append("the line of stops is not a list")
    return found


def fold_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the control of the fold of the page's own search says, and whether it offers to
    show passages only when it holds some; when the fold shows; and whether the column of
    sources says that it holds none."""
    found = []
    for name, words, nothing_more in (
        ("cited", "3 more passages were found and not cited.", False),
        ("declined", "1 passage was found, and none is cited.", False),
        ("six sources", "Every passage found is cited.", True),
        ("first", "The page's own search found nothing.", True),
        ("form declined", "The page's own search found nothing.", True),
        ("withheld", "The page's own search failed.", True),
        ("cached", "The page's own search failed.", True),
    ):
        step = steps[name]
        if step["searchCount"] != words:
            found.append(f"the fold's control says the wrong thing ({name})")
        if ("nothing-more" in step["searchFoldClasses"]) != nothing_more:
            found.append(
                f"the fold offers passages it does not hold, or hides those it does ({name})"
            )
    if steps["waiting first"]["searchFoldShown"] or not steps["stopped early"]["searchFoldShown"]:
        found.append("the fold of the page's own search shows before the line reaches the search")
    if (
        "no-sources" not in steps["declined"]["sideClasses"]
        or "no-sources" in steps["cited"]["sideClasses"]
    ):
        found.append("the column of sources does not say when it holds none")
    return found


def line_of(number: str) -> list[str]:
    """The line colour of a source's number, as a class: five colours, then the signs'."""
    return [f"line-{number}"] if number in ("1", "2", "3", "4", "5") else ["line-other"]


def colour_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """Colour means source: each citation and each source line has the one colour of its
    number, the same for both, and a source past the fifth has the colour of the signs."""
    found = []
    chips = [chip for step in steps.values() for chip in step.get("chipLines") or []]
    cards = [card for step in steps.values() for card in step.get("cards") or []]
    if any(lines != line_of(number) for number, lines in chips):
        found.append("a citation is not drawn in its source's line colour")
    if any(card.get("lines") != line_of(card["number"]) for card in cards):
        found.append("a source's line is not drawn in the colour of its number")
    six = steps["six sources"]
    numbers = ["1", "2", "3", "4", "5", "6"]
    if [number for number, _ in six["chipLines"]] != numbers or [
        card["number"] for card in six["cards"]
    ] != numbers:
        found.append("an answer with six sources does not show its six citations and lines")
    return found


def faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the page did wrong in the conversation, in words. Empty when nothing."""
    return [
        *conversation_faults(steps),
        *turn_faults(steps),
        *stop_faults(steps),
        *route_faults(steps),
        *source_faults(steps),
        *key_faults(steps),
        *label_faults(steps),
        *control_faults(steps),
        *announcement_faults(steps),
        *search_faults(steps),
        *answered_faults(steps),
        *outcome_faults(steps),
        *look_faults(steps),
        *check_faults(steps),
        *line_end_faults(steps),
        *fold_faults(steps),
        *colour_faults(steps),
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
        "the line of a reply calls it an answer (declined)",
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
        "  if (parts.length === 0 || crumbs(parts).length <= length) {",
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
            'turn.searchedLine.hidden = line === "" || searched.outcome === "rewritten";',
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
            "  if (!isAnswer(data)) {\n"
            '    fail(turn, UNREADABLE_ANSWER, "unreadable");\n    return false;',
            "  if (!isAnswer(data)) {\n"
            '    fail(turn, UNREADABLE_ANSWER, "unreadable");\n    return true;',
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
            "a turn does not show the question asked as its own sign",
            'sign.append(make("span", "visually-hidden", "You asked: "), question);',
            'sign.append(make("span", "visually-hidden", "You asked: "));',
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
            '  turn.pending.hidden = true;\n  turn.stream.classList.remove("streaming");',
            '  turn.stream.classList.remove("streaming");',
        ),
        (
            "sources that never came leave placeholders (refused)",
            "  if (turn.waiting) {",
            "  if (false) {",
        ),
        # The badges.
        (
            "a cached answer does not say at its stop that it was stored",
            '    stops.push({ name: "Found stored", detail: how, state: "reached" });\n',
            "",
        ),
        (
            "the line does not say how many passages the search found",
            '`${plural(found, "passage", "passages")} found`',
            '""',
        ),
        (
            "a cached answer counts tokens it never spent",
            '  if (cacheHit(answer) !== null) {\n    tokens = "none: no request was sent";\n  }\n',
            "",
        ),
        (
            "a failure has the stop of an answer it does not have (refused)",
            "    reached.push(endStop(journey));",
            '    reached.push({ name: "Answered", detail: "", state: "reached" });',
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
            '      where.append(make("span", "crumb", heading));\n',
            "",
        ),
        (
            "a source card does not show a piece of its passage",
            '    body.append(make("p", "source-snippet", piece));\n',
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


# The faults the review of pull request 86 planted, 40 of them, on the lines as they stand
# now, each with the check that finds it. The review found that 37 of them got past the
# tests then; these keep them found.
PLANTED_BY_THE_REVIEW = [
    (
        "a found passage does not show where its link leads",
        "  item.append(addressLine(passage.url));\n",
        "",
    ),
    (
        "a found passage's title is not a link to its page",
        '" ", sourceLink(passage.url, title));',
        '" ", make("span", "", title));',
    ),
    (
        "a fold is not named for its passage",
        '  summary.append(make("span", "visually-hidden", ` of ${name}, ${title}`));\n',
        "",
    ),
    (
        "a citation is read as a bare number",
        '  button.setAttribute("aria-label", `Source ${number}: ${title}`);\n',
        "",
    ),
    (
        "Ask again does not ask the question again",
        '  again.addEventListener("click", () => start(question));\n',
        "",
    ),
    (
        "Ask again sends while an answer is being written",
        "() => start(question));",
        "() => ask(question));",
    ),
    (
        "an example button does not ask its question",
        'example.addEventListener("click", () => start(example.textContent.trim()));',
        'example.addEventListener("click", () => start(""));',
    ),
    (
        "the examples are not marked as unavailable while an answer is written (waiting)",
        '    example.setAttribute("aria-disabled", String(busy));\n',
        "",
    ),
    (
        "the focus is lost when the stop button goes",
        "  if (!busy && document.activeElement === page.stop) {\n    page.question.focus();\n  }\n",
        "",
    ),
    (
        "Stop pressed after the answer came marks it as stopped",
        "  if (turn.body.children.length > 0 || !turn.error.hidden) {\n    return;\n  }\n",
        "",
    ),
    (
        "a failed turn keeps a Sources heading over nothing (refused)",
        "    turn.sourcesBlock.hidden = true;\n",
        "",
    ),
    (
        "an answer with no source says nothing under its Sources heading",
        '    cards.push(make("li", "sources-none", none));\n',
        "",
    ),
    (
        "a decline or a withheld reply is announced as the other",
        '      ? "Not in the handbook: the assistant declined instead of guessing."\n'
        '      : "Not an answer: the reply was withheld or did not come to a proper end.";',
        '      ? "Not an answer: the reply was withheld or did not come to a proper end."\n'
        '      : "Not in the handbook: the assistant declined instead of guessing.";',
    ),
    (
        "an ordinary answer is said to have taken more requests or the fallback (first)",
        "  if (Number.isInteger(answer.requests) && answer.requests > 1) {",
        "  if (Number.isInteger(answer.requests) && answer.requests > 0) {",
    ),
    (
        "an ordinary answer is said to have taken more requests or the fallback (first)",
        "  if (answer.fallback === true) {\n    parts.push(",
        "  if (answer.fallback !== true) {\n    parts.push(",
    ),
    (
        "a follow-up whose rewrite failed is said to be something else",
        '    failed: ["Not rewritten", "the request failed, so searched as typed"],',
        '    failed: ["Rewritten", "to stand alone"],',
    ),
    (
        "a reply that is not an answer has the stop of an answer (declined)",
        "  if (answer.answered) {\n    if (hit === null) {",
        "  if (true) {\n    if (hit === null) {",
    ),
    (
        "the new conversation button is shown or hidden wrongly (first)",
        "  page.newConversation.hidden = turns.length === 0 && conversation.length === 0;",
        "  page.newConversation.hidden = true;",
    ),
    (
        "the page says it sends more earlier questions than it does",
        "  const sent = Math.min(conversation.length, EARLIER_SENT);",
        "  const sent = conversation.length;",
    ),
    (
        "the alert keeps a failure after a new conversation",
        '  page.turns.replaceChildren();\n  page.failure.textContent = "";\n',
        "  page.turns.replaceChildren();\n",
    ),
    (
        "the alert keeps the last failure while the next question is answered",
        '  setBusy(true);\n  page.failure.textContent = "";\n',
        "  setBusy(true);\n",
    ),
    (
        "a withheld reply does not say why the model stopped",
        '  box.append(make("p", "notice-why", why));\n',
        "",
    ),
    (
        "a failure without a trace id shows an empty About this request",
        "  turn.made.hidden = turn.factsFold.hidden;",
        "  turn.made.hidden = false;",
    ),
    (
        "the fold of what the search found is not shown",
        "  turn.searchRun = true;\n",
        "",
    ),
    (
        "the fold says how many passages were found and lists none",
        "  turn.passages.replaceChildren(...unused.map(passageOrNotice));",
        "  turn.passages.replaceChildren();",
    ),
    (
        "the stop of the search does not say what the search did (withheld)",
        '  turn.searchFold.classList.add("nothing-more");\n  turn.journey.found = "failed";\n',
        '  turn.searchFold.classList.add("nothing-more");\n',
    ),
    (
        "a new conversation begins while an answer is being written",
        'page.newConversation.addEventListener("click", () => {\n  if (current !== null) {',
        'page.newConversation.addEventListener("click", () => {\n  if (false) {',
    ),
    (
        "a screen reader reads the whole address twice",
        '  address.setAttribute("aria-hidden", "true");\n',
        "",
    ),
    (
        "a later turn is not brought into view, or the first is (rewritten)",
        '  if (turns.length > 1) {\n    item.scrollIntoView({ block: "start" });\n  }\n',
        "",
    ),
    (
        "the piece of a passage is not cut short at a word",
        "  if (words.length <= length) {\n    return words;\n  }",
        "  if (true) {\n    return words;\n  }",
    ),
    (
        "the piece of a passage is not cut short at a word",
        "  return `${(space > length / 2 ? cut.slice(0, space) : cut).trimEnd()}…`;",
        "  return `${cut}…`;",
    ),
    (
        "the stop of the search does not say what the search did (declined)",
        '    return found === 0 ? "nothing found"',
        '    return found === 1 ? "nothing found"',
    ),
    (
        "the address in brief does not show the host the browser reads",
        "[parsed.host, ...list]",
        "[parsed.hostname, ...list]",
    ),
    (
        "the address in brief does not keep the host whole and name the page",
        '  return crumbs(parts.length > 1 ? ["…", last] : [last]);',
        '  return crumbs(parts.length > 1 ? ["…"] : []);',
    ),
    (
        "Ask's busy look is shown or left wrongly (arrived)",
        '  page.form.classList.toggle("busy", busy);',
        '  page.form.classList.toggle("busy", true);',
    ),
    (
        "the line does not say that the first words are arriving",
        'detail: journey.text ? "arriving now" : ""',
        'detail: ""',
    ),
    (
        "a match by meaning is said to be something else",
        'hit.tier === "exact" ? "an answer stored for the same question"',
        'true ? "an answer stored for the same question"',
    ),
    (
        "the warning look of a follow-up searched as typed is wrong",
        '"searched-as-typed", line !== "" && !line.startsWith("Searched for:"));',
        '"searched-as-typed", false);',
    ),
    (
        "a failure is not announced (refused)",
        '"How this request went");\n  say("No answer.");',
        '"How this request went");',
    ),
    (
        "text that arrived before a failure or a stop is not marked unfinished (error after text)",
        '  turn.stream.classList.toggle("unfinished", partial);\n  const told',
        "  const told",
    ),
]

# And the faults of this round's own fixes.
PLANTED_FOR_THE_FIXES = [
    (
        "the address in brief throws",
        "  if (parts.length === 0 || crumbs(parts).length <= length) {",
        "  if (crumbs(parts).length <= length) {",
    ),
    (
        "the address in brief does not keep the host whole and name the page",
        "  return part.length <= longest ? part : `${part.slice(0, half)}…${part.slice(-half)}`;",
        "  return `${part.slice(0, 7)}…${part.slice(-7)}`;",
    ),
    (
        "the stop of the search does not say what the search did (no searched event)",
        '    turn.journey.found = "not run";\n',
        "",
    ),
    (
        "the line counts sources the text does not cite",
        '`${plural(citedNumbers(answer).size, "source", "sources")} cited`',
        '`${plural(answer.sources.length, "source", "sources")} cited`',
    ),
    (
        "the status line counts sources the text does not cite",
        'It cites ${plural(citedNumbers(data).size, "source", "sources")}.`;',
        'It cites ${plural(data.sources.length, "source", "sources")}.`;',
    ),
    (
        "the count of what is sent is not said, or not in good grammar",
        '` ${sent === 1 ? "is" : "are"} sent with your next question`;',
        '" are sent with your next question";',
    ),
    (
        "the trace id is not copied, or the copy is not announced",
        '      say("The trace id is copied.");\n',
        "",
    ),
    (
        "the address under a source is not shown",
        '"visually-hidden", shown(url)));\n',
        '"visually-hidden", shown(url)));\n  line.hidden = true;\n',
    ),
]


@pytest.mark.parametrize(("fault", "old", "new"), PLANTED_BY_THE_REVIEW + PLANTED_FOR_THE_FIXES)
def test_the_run_finds_each_fault_the_review_planted_and_each_of_the_fixes(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in faults(run(source.replace(old, new), tmp_path))


def broken_sources(source: str) -> str:
    """The script with the card of the second source and the second found passage made to
    throw, as a field the page cannot read would make them."""
    card = "function sourceCard(turn, source) {\n"
    found = "function passageItem(passage) {\n"
    assert source.count(card) == source.count(found) == 1
    throw = '  if ({} === 2) {{\n    throw new TypeError("planted");\n  }}\n'
    return source.replace(card, card + throw.format("source.number")).replace(
        found, found + throw.format("passage.rank")
    )


def fallback_faults(cited: dict[str, Any]) -> list[str]:
    """What the page loses when one source or one found passage cannot be built."""
    found = []
    if not cited["answer"] or cited["error"]:
        found.append("one source that cannot be shown costs the answer")
        return found
    cards = cited["cards"]
    if [card["number"] for card in cards] != ["1", "2"] or [
        card["unreadable"] for card in cards
    ] != [False, True]:
        found.append("the card of a source that could not be shown does not say which")
    elif cards[1]["path"] != "This source could not be shown.":
        found.append("the card of a source that could not be shown does not say so")
    press = cited["linking"]["pressChip"]
    if (
        press["scrolled"] != 1
        or press["marked"] != ["2"]
        or "Source 2 is marked" not in press["status"]
    ):
        found.append("pressing the number of a source that could not be shown does nothing")
    passages = cited["passages"]
    if [item["rank"] for item in passages] != ["1.", "2.", "3."] or [
        item["unreadable"] for item in passages
    ] != [False, True, False]:
        found.append("one found passage that cannot be shown costs the others")
    if (
        cited["conversation"][-1]
        != "How many weeks of parental leave do new parents get in the US?"
    ):
        found.append("a turn whose source could not be shown is not remembered")
    return found


def test_a_source_card_that_cannot_be_built_does_not_cost_the_answer(tmp_path: Path) -> None:
    """Building the answer cannot fail halfway: a card that throws stands as a card that
    says so, under its source's number, so that pressing the number brings it into view,
    marks it and says so; the answer and the other cards are shown, and the turn is
    remembered. A found passage that throws stands in the same way."""
    assert fallback_faults(run(broken_sources(script()), tmp_path)["cited"]) == []


@pytest.mark.parametrize(
    ("fault", "old", "new"),
    [
        (
            "one source that cannot be shown costs the answer",
            "answer.sources.map((source) => cardOrNotice(turn, source));",
            "answer.sources.map((source) => sourceCard(turn, source));",
        ),
        (
            "pressing the number of a source that could not be shown does nothing",
            "    turn.cards.set(source.number, card);\n",
            "",
        ),
        (
            "the card of a source that could not be shown does not say which",
            '    card.append(make("span", "source-number", shown(source.number)), body);',
            "    card.append(body);",
        ),
        (
            "one found passage that cannot be shown costs the others",
            "unused.map(passageOrNotice)",
            "unused.map(passageItem)",
        ),
    ],
)
def test_the_run_finds_each_fault_planted_in_the_fallbacks(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in fallback_faults(
        run(broken_sources(source.replace(old, new)), tmp_path)["cited"]
    )


# The faults of the notes of the checks, each with the check that finds it.
PLANTED_IN_THE_NOTES = [
    (
        "the notes of a check are not shown",
        "    if (notes.length > 0) {\n      parts.push(checkBox(notes));\n    }\n",
        "",
    ),
    (
        "the notes of a check are shown on another turn (the first)",
        "      parts.push(checkBox(notes));",
        "      turns[0].body.append(checkBox(notes));",
    ),
    (
        "a list of checks of another shape loses the answer (malformed checks)",
        "  if (!Array.isArray(answer.checks)) {\n    return [];\n  }\n",
        "",
    ),
    (
        "a list of checks of another shape loses the answer (malformed items)",
        ".filter((check) => isObject(check) && typeof",
        ".filter((check) => typeof",
    ),
    (
        "a list of checks of another shape gives a note (malformed items)",
        'typeof check.note === "string" && check.note !== ""',
        "check.note !== undefined",
    ),
    (
        "the note of a check is not announced with the outcome (check dropped line)",
        "  say(`${words}${from}${noted}`);",
        "  say(`${words}${from}`);",
    ),
    (
        "the note of a check on a reply that is not an answer is not with its notice",
        "  if (notes.length > 0) {\n    box.append(checkBox(notes));\n  }\n  return box;",
        "  return box;",
    ),
    (
        "a reply a check declined is shown as one the handbook does not answer",
        'byCheck ? "Declined by a check" : ',
        "",
    ),
    (
        "a reply a check declined is announced as one the handbook does not answer",
        '    words = "Declined by a check, in place of the reply the model wrote.";\n',
        "",
    ),
    (
        "a change by a check is not said at the end of the line (two checks)",
        "  const notes = checkNotes(answer).length;",
        "  const notes = 0;",
    ),
    (
        "the notes of a check are not marked as a change",
        '  box.append(make("p", "check-label", "Changed by a check."));\n',
        "",
    ),
]


@pytest.mark.parametrize(("fault", "old", "new"), PLANTED_IN_THE_NOTES)
def test_the_run_finds_each_fault_planted_in_the_notes_of_the_checks(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in faults(run(source.replace(old, new), tmp_path))


# The faults of the line of stops, each with the check that finds it.
PLANTED_IN_THE_LINE = [
    (
        "an answer's line does not run through its four stops",
        'const stops = [{ name: "Asked", detail: asked, state: "reached" }];',
        'const stops = [{ name: "Asked", detail: asked, state: "ahead" }];',
    ),
    (
        "the line does not show how far the answer has come (searched)",
        "        turn.journey.searched = searched;\n        renderRoute(turn);\n",
        "",
    ),
    (
        "the line does not show how far the answer has come (writing)",
        "    turn.journey.text = true;\n    renderRoute(turn);",
        "    turn.journey.text = true;",
    ),
    (
        "the line does not show how far the answer has come (waiting)",
        '      next.state = "next";\n',
        "",
    ),
    (
        "a decline's line does not run on to stops it does not serve (declined)",
        'stops.push({ name: "No answer in them", detail: first, state: "off" });',
        'stops.push({ name: "No answer in them", detail: first, state: "reached" });',
    ),
    (
        "a decline's line is not dashed after the search (declined)",
        'segment = next.state === "off" ? "dashed" :',
        'segment = next.state === "off" ? "solid" :',
    ),
    (
        "the line to a stop not reached yet is drawn as if it were reached",
        ': next.state === "reached" || next.state === "end" ? "solid" : "faint";',
        ': "solid";',
    ),
    (
        "a line that ended short does not end where it ended (refused)",
        "  turn.journey.ended = { kind };\n",
        "",
    ),
    (
        "a line that ended short does not end where it ended (stopped)",
        '  turn.journey.ended = { kind: "stopped" };\n',
        "",
    ),
    (
        "a line that ended short does not end where it ended (error after text)",
        '{ name: "Broke off", detail: `after the first words: ${why}`, state: "end" }',
        '{ name: "Not answered", detail: why, state: "end" }',
    ),
    (
        "a line that ended short does not end where it ended (withheld)",
        'return { name, detail: `the model stopped: ${reason}`, state: "end" };',
        'return { name, detail: `the model stopped: ${reason}`, state: "off" };',
    ),
    (
        "a stored answer's line is not another one, with no search (cached)",
        "  if (hit !== null) {\n    const how",
        "  if (false) {\n    const how",
    ),
    (
        "a follow-up's line has no stop of its own for the rewrite",
        "  if (journey.earlier > 0) {\n    stops.push(rewriteStop(searched));\n  }\n",
        "",
    ),
    (
        "a rewritten follow-up does not show the question it was searched for",
        '    rewritten: ["Rewritten", searched.question],',
        '    rewritten: ["Rewritten", searched.asked],',
    ),
    (
        "a check's decline is not where its line ends",
        'stops.push({ name: "Declined by a check", detail: "nothing is cited", state: "off" });',
        'stops.push({ name: "Not answered", detail: "nothing is cited", state: "off" });',
    ),
    (
        "a screen reader is not told which stop is next and which are not reached",
        '  next: " (next stop)",',
        '  next: "",',
    ),
    (
        "a screen reader is not told where a line ends (refused)",
        '  end: " (the line ends here)",',
        '  end: "",',
    ),
    (
        "the line is not named for what it shows (stopped)",
        '  turn.route.setAttribute("aria-label", "How this request went");\n  say(words);',
        "  say(words);",
    ),
    (
        "a stored answer's line does not say that no request was sent",
        '    parts.push("no request to the model");\n',
        "",
    ),
]


@pytest.mark.parametrize(("fault", "old", "new"), PLANTED_IN_THE_LINE)
def test_the_run_finds_each_fault_planted_in_the_line_of_stops(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in faults(run(source.replace(old, new), tmp_path))


# The faults the review of the new look planted that got past every test, and more of each
# kind: each with the check that now finds it.
PLANTED_IN_THE_SECOND_ROUND = [
    (
        "a stop before the first words is not said where it ended",
        'detail: journey.text ? "after the first words" : "before the first words",',
        'detail: "after the first words",',
    ),
    (
        "a failure after the search does not keep the search it ran",
        'const reached = stops.filter((stop) => stop.state === "reached");',
        'const reached = stops.filter((stop) => stop.state === "reached"'
        ' && (journey.text || stop.name !== "Searched"));',
    ),
    (
        "a reply that came with no text is said to have had first words",
        'const first = answer.first_text_ms === null ? "no text came" :'
        " `after ${secondsInWords(answer.first_text_ms)}`;",
        "const first = `after ${secondsInWords(answer.first_text_ms)}`;",
    ),
    (
        "a citation is not drawn in its source's line colour",
        ': "line-other";',
        ': "line-5";',
    ),
    (
        "a citation is not drawn in its source's line colour",
        'make("button", `cite ${lineClass(number)}`, String(number));',
        'make("button", "cite", String(number));',
    ),
    (
        "a source's line is not drawn in the colour of its number",
        'make("li", `source-card ${lineClass(number)}`);',
        'make("li", `source-card ${lineClass(number + 1)}`);',
    ),
    (
        "a screen reader is not told which stops the line does not serve (declined)",
        '  off: " (not served)",',
        '  off: "",',
    ),
    (
        "the fold's control says the wrong thing (six sources)",
        'count = "Every passage found is cited.";',
        'count = "Nothing more.";',
    ),
    (
        "the fold offers passages it does not hold, or hides those it does (six sources)",
        'turn.searchFold.classList.toggle("nothing-more", unused.length === 0);',
        'turn.searchFold.classList.toggle("nothing-more", false);',
    ),
    (
        "the line of stops is not a list",
        'turn.route = make("ol", "route");',
        'turn.route = make("div", "route");',
    ),
    # The words a reader got wrong.
    (
        "a screen reader is told that a stop not reached yet is where the answer is now",
        '  next: " (next stop)",',
        '  next: " (now)",',
    ),
    (
        "a line that ended short does not end where it ended (withheld)",
        '    name = "Withheld";',
        '    name = "Cut off";',
    ),
    (
        "the fold's control says the wrong thing (first)",
        '    count = "The page\'s own search found nothing.";',
        '    count = "Every passage found is cited.";',
    ),
    (
        "the fold's control says the wrong thing (withheld)",
        '  turn.searchCount.textContent = "The page\'s own search failed.";',
        '  turn.searchCount.textContent = "";',
    ),
    (
        "a line that broke off does not say what went wrong (unreadable answer)",
        '  unreadable: "the answer could not be read",',
        '  unreadable: "the request failed",',
    ),
    (
        "a change by a check is not said at the end of the line (two checks)",
        "`changed by ${notes} checks`",
        '"changed by a check"',
    ),
    # And the fold, the column and the stop the review found crowded or at odds.
    (
        "the fold of the page's own search shows before the line reaches the search",
        "  turn.searchFold.hidden = !turn.searchRun || !searchReached(turn.journey);",
        "  turn.searchFold.hidden = !turn.searchRun;",
    ),
    (
        "the column of sources does not say when it holds none",
        '  turn.side.classList.toggle("no-sources", answer.sources.length === 0);\n',
        "",
    ),
    (
        "a failure after the search does not say what failed",
        '  failed: "the request failed",',
        '  failed: "the answer could not be read",',
    ),
]


@pytest.mark.parametrize(("fault", "old", "new"), PLANTED_IN_THE_SECOND_ROUND)
def test_the_run_finds_each_fault_planted_in_the_second_round(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in faults(run(source.replace(old, new), tmp_path))


# The address line under a source and a found passage must never be hidden or cut. The
# check of the style sheet in test_demo.py knows, by class, id and element, what stands
# around the line. Here what stands around it is read from the page the script really
# builds, in every state of the run, with the markup around the list of turns, so that a
# new wrapper cannot be left out of that check: each one is either known to it, or this
# fails, and the sheet is checked against the structure as built.


class Around(HTMLParser):
    """The elements of the page's markup that hold the element with an id."""

    VOID: ClassVar[set[str]] = {"meta", "link", "br", "img", "input", "hr", "source", "wbr"}

    def __init__(self, target: str) -> None:
        super().__init__()
        self.target = target
        self.open: list[tuple[str, str, str]] = []
        self.found: list[tuple[str, str, str]] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        named = dict(attrs)
        here = (tag, named.get("class") or "", named.get("id") or "")
        if here[2] == self.target and self.found is None:
            self.found = [*self.open, here]
        if tag not in self.VOID:
            self.open.append(here)

    def handle_endtag(self, tag: str) -> None:
        while self.open and self.open.pop()[0] != tag:
            pass


def structure_around_the_address(
    steps: dict[str, dict[str, Any]],
) -> tuple[set[str], set[str], set[str], set[str]]:
    """The classes, the elements and the ids around every address line on the page, and the
    classes inside the line, as the script built them in the run, with the markup above."""
    built = steps["new after failure"]["around"]
    markup = Around("turns")
    markup.feed(static("demo.html"))
    assert markup.found is not None
    classes = set(built["aroundClasses"]) | {
        name for _, names, _ in markup.found for name in names.split()
    }
    tags = set(built["aroundTags"]) | set(built["insideTags"]) | {tag for tag, _, _ in markup.found}
    ids = {each for _, _, each in markup.found if each}
    # The whole address for a screen reader is kept in a span that is hidden on screen on
    # purpose, beside the address in brief that the reader sees.
    inside = set(built["insideClasses"]) - {"visually-hidden"}
    return classes, tags | {"*"}, ids, inside


@pytest.fixture(scope="module")
def built(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[set[str], set[str], set[str], set[str]]:
    return structure_around_the_address(run(script(), tmp_path_factory.mktemp("built")))


def test_the_check_of_the_address_line_knows_everything_the_script_puts_around_it(
    built: tuple[set[str], set[str], set[str], set[str]],
) -> None:
    classes, tags, ids, inside = built

    assert classes - AROUND == set(), "a class around the address that the check does not know"
    assert tags - HOLDERS == set(), "an element around the address that the check does not know"
    assert ids - AROUND_IDS == set(), "an id around the address that the check does not know"
    assert inside - ADDRESS == set(), "a class inside the address that the check does not know"
    # What the run must have seen, for the list to mean something: a source's line and a
    # found passage's, each with what holds it.
    assert {"source-card", "source-body", "passage-item", "search-fold", "cols", "turn"} <= classes


def test_a_wrapper_added_around_the_address_is_one_the_check_must_learn(tmp_path: Path) -> None:
    old = "  body.append(addressLine(source.url));\n"
    new = (
        '  const wrap = make("div", "address-wrap");\n'
        "  wrap.append(addressLine(source.url));\n"
        "  body.append(wrap);\n"
    )
    source = script()
    assert source.count(old) == 1

    classes, _, _, _ = structure_around_the_address(run(source.replace(old, new), tmp_path))
    assert classes - AROUND == {"address-wrap"}


def test_no_rule_of_the_style_sheet_cuts_the_address_in_the_structure_the_script_builds(
    built: tuple[set[str], set[str], set[str], set[str]],
) -> None:
    classes, tags, ids, _ = built
    rules = css_rules(static("demo.css"))

    assert [
        problem for sel, d in rules for problem in cuts_the_address(sel, d, classes, tags, ids)
    ] == []


@pytest.mark.parametrize(
    "rule",
    [
        ".source-body { overflow: hidden; max-height: 6em; }",
        ".source-body { display: -webkit-box; -webkit-line-clamp: 4; overflow: hidden; }",
        ".cols { overflow: hidden; }\n.source-body { white-space: nowrap; }",
        ".no-sources { overflow: hidden; }",
        ".marked { white-space: nowrap; }",
        "#turns { contain: paint; }",
    ],
)
def test_the_check_of_the_structure_as_built_finds_each_rule_planted(
    rule: str, built: tuple[set[str], set[str], set[str], set[str]]
) -> None:
    classes, tags, ids, _ = built
    rules = css_rules(f"{static('demo.css')}\n{rule}\n")

    assert [
        problem for sel, d in rules for problem in cuts_the_address(sel, d, classes, tags, ids)
    ] != []
