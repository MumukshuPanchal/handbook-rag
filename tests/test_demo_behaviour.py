"""The demo page's script, run: one conversation through a small fake page in node.

``tests/test_demo.py`` reads the script's source. This runs it, with ``node`` and
``tests/demo_page_harness.mjs``: a fake page with only what the script touches, and a
stubbed ``fetch`` that answers ``/ask`` as the service does, as a stream with the question
searched for first. The project has no JavaScript toolchain (ADR 0016), and none is added:
node runs the script as it is. Without node the tests are skipped, and say why.

The harness reads back what a reader would see in each turn: the question's bubble, the
answer or the notice or the failure card, the placeholders and the caret while the answer
comes, the badges under it, its source cards and found passages, and what pointing at,
focusing and pressing a number in the answer does to them; and around the turns, the
buttons and their states, the focus, the status line and the alert, the examples, "Ask
again", the copy of a trace id, and the count of what is sent.

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
DECLINED = ("declined", "cached", "check declined")
FAILED = (
    "error after text",
    "unreadable answer",
    "refused",
    "late failure",
    "stream cut",
    "failure before new",
)
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
    steps = json.loads(done.stdout)
    # A step with no turn on the page, such as one a fault kept from being asked, has none
    # of a turn's fields: it is given them empty, so that the checks say what is missing
    # instead of failing on a missing field.
    template: dict[str, Any] = next((step for step in steps if "bubble" in step), {})
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
    # The strip of badges names what it sums up: an answer, or a reply that is not one.
    for name in (*ANSWERED, *DECLINED, "withheld", "check cut"):
        made = "answer" if name in ANSWERED else "reply"
        if steps[name]["madeTitle"] != f"How this {made} was made":
            found.append(f"the strip under a reply calls it an answer ({name})")
    return found


def control_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the controls around the turns get wrong: the examples, "Ask again", the copy of
    a trace id, the new-conversation button, the alert, the count of what is sent."""
    found = []
    by_itself = steps["asked by itself"]
    if by_itself["turnCount"] != 1 or by_itself["bubble"] != [f"You asked: {FIRST}"]:
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
    """What the fold of found passages and the Search badge get wrong."""
    found = []
    cited = steps["cited"]
    if not cited["searchFoldShown"] or cited["searchCount"] != " 3 passages":
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
        ("declined", "Search: 1 passage found"),
        ("cached", "Search: failed"),
        ("withheld", "Search: failed"),
        ("no searched event", "Search: not run"),
    ):
        if wanted not in steps[name]["badges"]:
            found.append(f"the Search badge does not say what the search did ({name})")
    if steps["withheld"]["searchState"] != "The search failed. The service could not be reached.":
        found.append("a search that could not reach the service does not say so")
    if steps["cached"]["searchState"] != "The search failed. The service is busy.":
        found.append("a search the service refused does not say so")
    return found


def badge_word_faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """Whether each badge that stands out stands where it should, and only there."""
    found = []
    fallback = steps["fallback"]["badges"]
    for name in (*ANSWERED, *DECLINED, "withheld"):
        badges = steps[name]["badges"]
        if name != "fallback" and any(
            badge.startswith(("Requests", "Written by")) for badge in badges
        ):
            found.append(f"a warning badge stands on an ordinary answer ({name})")
    if "Requests: 2" not in fallback or "Written by: the fallback model" not in fallback:
        found.append("an answer of two requests by the fallback model is not badged so")
    if "Cited: 1 source" not in fallback:
        found.append("the Cited badge counts sources the text does not cite")
    if "Follow-up: rewrite failed" not in steps["failed rewrite"]["badges"]:
        found.append("a follow-up whose rewrite failed is badged as something else")
    for name in (*DECLINED, "withheld"):
        if any(badge.startswith("Cited") for badge in steps[name]["badges"]):
            found.append(f"a reply that is not an answer has a Cited badge ({name})")
    semantic = steps["semantic"]
    if "From the cache: matched by meaning" not in semantic["badges"]:
        found.append("a match by meaning is badged as something else")
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
        first_of_conversation = name in ("first", "asked by itself")
        if (step["scrolled"] == 0) != first_of_conversation:
            found.append(f"a later turn is not brought into view, or the first is ({name})")
    if steps["writing"]["progressText"] != "Writing":
        found.append("the card does not say that the model is writing")
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
    if any(steps[name]["checkLabels"] != ["Changed by a check"] for name in CHECKED):
        found.append("the notes of a check are not marked as a change")
    for step, notes in ((declined, [UNCITED]), (cut, [UNFINISHED])):
        if step["checkNotes"] != notes or step["checkInNotice"] != [True]:
            found.append(
                "the note of a check on a reply that is not an answer is not with its notice"
            )
    if declined["noticeTitle"] != ["Declined by a check"] or declined["noticeWhy"] != []:
        found.append("a reply a check declined is shown as one the handbook does not answer")
    if cut["noticeTitle"] != ["Not an answer"] or not cut["noticeWhy"]:
        found.append("a reply a check cut does not say why the model stopped")
    for name, notes in CHECKED.items():
        if not steps[name]["status"].endswith("".join(f" Note: {note}" for note in notes)):
            found.append(f"the note of a check is not announced with the outcome ({name})")
    if not declined["status"].startswith(
        "Declined by a check, in place of the reply the model wrote."
    ):
        found.append("a reply a check declined is announced as one the handbook does not answer")
    for name, badge in (
        ("check dropped line", "Changed: by a check"),
        ("two checks", "Changed: by 2 checks"),
        ("check declined", "Changed: by a check"),
    ):
        if badge not in steps[name]["badges"]:
            found.append(f"a change by a check has no badge ({name})")
    for name in ("malformed checks", "malformed items"):
        step = steps[name]
        if not step["answer"] or step["error"] or step["conversation"][-1:] != [step["question"]]:
            found.append(f"a list of checks of another shape loses the answer ({name})")
        elif step["checkNotes"] or any(badge.startswith("Changed") for badge in step["badges"]):
            found.append(f"a list of checks of another shape gives a note ({name})")
    for name, step in steps.items():
        if name not in CHECKED and step["checkNotes"]:
            found.append(f"the notes of a check are shown on another turn ({name})")
        if step["firstTurnCheckNotes"]:
            found.append("the notes of a check are shown on another turn (the first)")
    if steps["two checks"]["previousTurnCheckNotes"] != [DROPPED]:
        found.append("the notes of a check leave their own turn")
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
        *control_faults(steps),
        *announcement_faults(steps),
        *search_faults(steps),
        *badge_word_faults(steps),
        *outcome_faults(steps),
        *look_faults(steps),
        *check_faults(steps),
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
        "a warning badge stands on an ordinary answer (first)",
        "  if (requests !== null && requests > 1) {",
        "  if (requests !== null && requests > 0) {",
    ),
    (
        "a warning badge stands on an ordinary answer (first)",
        "  if (answer.fallback === true) {",
        "  if (answer.fallback !== true) {",
    ),
    (
        "a follow-up whose rewrite failed is badged as something else",
        '    failed: "rewrite failed",',
        '    failed: "rewritten to stand alone",',
    ),
    (
        "a reply that is not an answer has a Cited badge (declined)",
        '  if (answer.answered) {\n    badges.push(["Cited"',
        '  if (true) {\n    badges.push(["Cited"',
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
        "  turn.searchFold.hidden = false;\n",
        "",
    ),
    (
        "the fold says how many passages were found and lists none",
        "  turn.passages.replaceChildren(...results.map(passageOrNotice));",
        "  turn.passages.replaceChildren();",
    ),
    (
        "the Search badge does not say what the search did (withheld)",
        'could not be reached.";\n    turn.found.textContent = "failed";',
        'could not be reached.";',
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
        "the Search badge does not say what the search did (declined)",
        'results.length === 0 ? "nothing found" : `${count} found`',
        'results.length === 1 ? "nothing found" : `${count} found`',
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
        "the card does not say that the model is writing",
        '  turn.progress.textContent = "Writing";\n',
        "",
    ),
    (
        "a match by meaning is badged as something else",
        '    const tier = hit.tier === "exact" ? "exact match" : "matched by meaning";',
        '    const tier = "exact match";',
    ),
    (
        "the warning look of a follow-up searched as typed is wrong",
        '"searched-as-typed", line !== "" && !line.startsWith("Searched for:"));',
        '"searched-as-typed", false);',
    ),
    (
        "a failure is not announced (refused)",
        '  label(turn, "Failed");\n  say("No answer.");',
        '  label(turn, "Failed");',
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
        "the Search badge does not say what the search did (no searched event)",
        '    turn.found.textContent = "not run";\n',
        "",
    ),
    (
        "the Cited badge counts sources the text does not cite",
        '    badges.push(["Cited", plural(citedNumbers(answer).size, "source", "sources"), ""]);',
        '    badges.push(["Cited", plural(answer.sources.length, "source", "sources"), ""]);',
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
            '    head.append(make("span", "source-number", shown(source.number)));\n',
            "",
        ),
        (
            "one found passage that cannot be shown costs the others",
            "results.map(passageOrNotice)",
            "results.map(passageItem)",
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
        "a change by a check has no badge (two checks)",
        'badges.push(["Changed", notes.length === 1',
        'badges.push(["Unchanged", notes.length === 1',
    ),
    (
        "the notes of a check are not marked as a change",
        '  box.append(make("p", "check-label", "Changed by a check"));\n',
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
