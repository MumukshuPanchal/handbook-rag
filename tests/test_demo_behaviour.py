"""The demo page's script, run: one conversation through a small fake page in node.

``tests/test_demo.py`` reads the script's source. This runs it, with ``node`` and
``tests/demo_page_harness.mjs``: a fake page with only what the script touches, and a
stubbed ``fetch`` that answers ``/ask`` as the service does, as a stream with the question
searched for first. The project has no JavaScript toolchain (ADR 0016), and none is added:
node runs the script as it is. Without node the tests are skipped, and say why.

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

pytestmark = pytest.mark.skipif(
    NODE is None, reason="node is not installed: the page's script cannot be run here"
)


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


def faults(steps: dict[str, dict[str, Any]]) -> list[str]:
    """What the page did wrong in the conversation, in words. Empty when nothing."""
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
    given_back = steps["given back"]
    if given_back["searchedLine"] != "" or not given_back["searchedHidden"]:
        found.append("a line about the question searched for is left showing")
    if steps["refused"]["searchedLine"] != "" or not steps["refused"]["searchedHidden"]:
        found.append("a refused question leaves the last line about the search showing")
    for failed in ("error after text", "refused"):
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
    return found


def test_a_conversation_on_the_page_does_what_the_documents_say(tmp_path: Path) -> None:
    assert faults(run(script(), tmp_path)) == []


@pytest.mark.parametrize(
    ("fault", "old", "new"),
    [
        (
            "the conversation does not grow by the question searched for",
            "if (showAnswer(data)) {",
            "if (showAnswer(data) && false) {",
        ),
        (
            '"Start a new conversation" does not forget the conversation',
            "  forget();\n  say(",
            "  say(",
        ),
        (
            "the side search is not run on the question searched for",
            "search(searched.question);",
            "search(question);",
        ),
        (
            "a line about the question searched for is left showing",
            'page.searched.hidden = line === "";',
            "page.searched.hidden = false;",
        ),
        (
            "a refused question leaves the last line about the search showing",
            '  page.searched.textContent = "";\n  page.searched.hidden = true;\n',
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
    ],
)
def test_the_run_finds_each_fault_planted_in_the_script(
    fault: str, old: str, new: str, tmp_path: Path
) -> None:
    source = script()
    assert source.count(old) == 1, old

    assert fault in faults(run(source.replace(old, new), tmp_path))
