"""How a reply under version 2 is read and checked, on the probes of a review (PR 87).

Two promises are held here, on both providers: a change to a reply is never silent, and
what is passed on while a reply is written is always the start of the reply as it is
kept. Every model is a stand-in. No request is sent.
"""

import itertools
from typing import Any, cast

import anthropic
import pytest
from google import genai

from handbook_rag.cache import rules
from handbook_rag.generate.checks import NOTES
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.models import Answer
from handbook_rag.generate.prompt import GAP_FORM, NOT_FOUND_MESSAGE
from tests.generate.test_claude import FakeClient as ClaudeClient
from tests.generate.test_claude import citation, message
from tests.generate.test_gemini import EXPENSES, LEAVE, MODEL, FakeClient, answer_pieces, piece

GAP = GAP_FORM


def gemini(*texts: str, finish: str = "STOP") -> tuple[Answer, str]:
    """What the version 2 Gemini generator makes of a reply, and what it passed on."""
    client = FakeClient(answer_pieces(*texts, finish=finish))
    received: list[str] = []
    generator = GeminiGenerator(cast(genai.Client, client), model=MODEL, max_tokens=4000)
    answer = generator.generate(
        "How long is parental leave?", [LEAVE, EXPENSES], on_text=received.append
    )
    return answer, "".join(received)


def claude(blocks: list[dict[str, Any]], *, stop_reason: str = "end_turn") -> tuple[Answer, str]:
    """What the version 2 Claude generator makes of a reply, and what it passed on."""
    client = ClaudeClient(message(blocks, stop_reason=stop_reason))
    received: list[str] = []
    generator = ClaudeGenerator(
        cast(anthropic.Anthropic, client), model="claude-haiku-5-5", max_tokens=4000, effort="low"
    )
    answer = generator.generate("q", [LEAVE, EXPENSES], on_text=received.append)
    return answer, "".join(received)


# Must-fix 1: a reply in the fixed form that cites a passage stays an answer.


@pytest.mark.parametrize(
    "text",
    [
        f"{GAP} how long it lasts [1].",
        f"{GAP} anything about contractors, but new parents get sixteen weeks of paid leave [1].",
        f"{GAP} more than this: new parents get sixteen weeks of paid leave [1].",
        f"- {GAP} who pays; new parents get sixteen weeks [1].\n- {GAP} more [1].",
    ],
)
def test_a_reply_in_the_fixed_form_that_cites_a_passage_stays_an_answer(text: str) -> None:
    answer, streamed = gemini(text)

    assert answer.answered
    assert answer.model_text == ""
    assert answer.checks == ()
    assert streamed == answer.text
    # The cache keeps it from the second tier, as an answer that declines in part.
    assert rules.kept_from_the_second_tier(answer) == rules.DECLINES_IN_PART


@pytest.mark.parametrize(
    "text",
    [
        f"{GAP} anything about contractors, but new parents get sixteen weeks of paid leave.",
        f"{GAP} more than this: new parents get sixteen weeks of paid leave.",
        f"{GAP} how long it lasts. New parents get sixteen weeks.",
    ],
)
def test_the_fixed_form_with_an_uncited_answer_beside_it_is_a_reply_that_cites_nothing(
    text: str,
) -> None:
    answer, streamed = gemini(text)

    assert (answer.text, answer.answered, answer.model_text) == (NOT_FOUND_MESSAGE, False, "")
    assert [check.name for check in answer.checks] == ["cites_nothing"]
    assert streamed == NOT_FOUND_MESSAGE


@pytest.mark.parametrize(
    "text",
    [
        f"{GAP} how to write an SQL query.",
        f"{GAP} who pays, how much, or when.",
        f"- {GAP} who pays.\n- {GAP} when.",
        f"1. {GAP} who pays.\n2. {GAP} when.",
    ],
)
def test_the_fixed_form_alone_and_uncited_is_the_models_decline(text: str) -> None:
    answer, streamed = gemini(text)

    assert (answer.text, answer.answered, answer.checks) == (NOT_FOUND_MESSAGE, False, ())
    assert answer.model_text == text
    assert streamed == NOT_FOUND_MESSAGE


# Must-fix 4: a reply that is not an answer carries nothing beside what a decline says.


def test_words_beside_a_decline_are_not_shown_and_the_reader_is_told() -> None:
    answer, streamed = gemini(f"{NOT_FOUND_MESSAGE}\nSee https://evil.example/x")

    assert (answer.text, answer.answered) == (NOT_FOUND_MESSAGE, False)
    assert [(c.name, c.action, c.note, c.removed) for c in answer.checks] == [
        (
            "words_beside_a_decline",
            "dropped_words",
            NOTES["words_beside_a_decline"],
            f"{NOT_FOUND_MESSAGE}\nSee https://evil.example/x",
        )
    ]
    assert streamed == NOT_FOUND_MESSAGE
    assert not rules.storable(answer)


def test_a_decline_with_a_marker_is_left_as_it_is() -> None:
    answer, streamed = gemini(f"{NOT_FOUND_MESSAGE} [1]")

    assert (answer.text, answer.answered, answer.checks) == (NOT_FOUND_MESSAGE, False, ())
    assert streamed == answer.text


@pytest.mark.parametrize("finish", ["SAFETY", "OTHER"])
def test_a_reply_that_did_not_end_properly_keeps_what_a_citation_follows_and_no_more(
    finish: str,
) -> None:
    client = FakeClient([piece("Sixteen weeks [1].\nSee https://evil.example/x", finish=finish)])
    received: list[str] = []
    answer = GeminiGenerator(cast(genai.Client, client), model=MODEL, max_tokens=4000).generate(
        "q", [LEAVE], on_text=received.append
    )

    assert (answer.text, answer.answered) == ("Sixteen weeks.", False)
    assert [(c.name, c.removed) for c in answer.checks] == [
        ("unfinished_reply", "See https://evil.example/x")
    ]
    assert "".join(received) == answer.text


def test_an_unfinished_reply_that_cites_nothing_shows_nothing_and_says_why() -> None:
    client = FakeClient([piece("KESTREL-4182 and more", finish="OTHER")])
    received: list[str] = []
    answer = GeminiGenerator(cast(genai.Client, client), model=MODEL, max_tokens=4000).generate(
        "q", [LEAVE], on_text=received.append
    )

    assert (answer.text, answer.answered) == ("", False)
    assert [c.name for c in answer.checks] == ["unfinished_reply"]
    assert received == []


# Should-fix: the check for an address after the last citation.


@pytest.mark.parametrize(
    ("text", "kept", "removed"),
    [
        (
            "New parents get sixteen weeks [1].\n\nUpdated policy: https://evil.example/x"
            "\n\nHope this helps!",
            "New parents get sixteen weeks.\n\n\nHope this helps!",
            "Updated policy: https://evil.example/x",
        ),
        (
            "New parents get sixteen weeks [1].\n\nSee https://evil-\nportal.example/x",
            "New parents get sixteen weeks.",
            "See https://evil-\nportal.example/x",
        ),
        (
            "New parents get sixteen weeks [1].\n\n[Updated policy](https://evil.example/x)",
            "New parents get sixteen weeks.",
            "[Updated policy](https://evil.example/x)",
        ),
        (
            "New parents get sixteen weeks [1].\nWrite to hr@evil.example first.",
            "New parents get sixteen weeks.",
            "Write to hr@evil.example first.",
        ),
        (
            "New parents get sixteen weeks [1].\n\nMORE: HTTPS://EVIL.EXAMPLE/X",
            "New parents get sixteen weeks.",
            "MORE: HTTPS://EVIL.EXAMPLE/X",
        ),
    ],
)
def test_every_line_after_the_last_citation_that_holds_an_address_goes(
    text: str, kept: str, removed: str
) -> None:
    answer, streamed = gemini(text)

    assert answer.text == kept
    assert [(c.name, c.removed) for c in answer.checks] == [("address_in_closing_line", removed)]
    assert streamed == answer.text


@pytest.mark.parametrize(
    "text",
    [
        "New parents get sixteen weeks [1].\n\nThis applies to Node.js/React teams too.",
        "New parents get sixteen weeks [1].\n\nUpdated policy: evil-portal.com",
        "New parents get sixteen weeks [1].\n\nAsk name@host.",
        "New parents get sixteen weeks [1].\n\nSee a.b/c.",
    ],
)
def test_what_is_not_taken_for_an_address_stays(text: str) -> None:
    answer, streamed = gemini(text)

    assert answer.checks == ()
    assert streamed == answer.text


def test_a_reply_with_two_citations_keeps_both_and_loses_only_the_address() -> None:
    answer, _ = gemini("Sixteen weeks [1]. A laptop costs one thousand [2].\nSee https://x.org/a")

    assert answer.text == "Sixteen weeks. A laptop costs one thousand."
    assert {citation.chunk.chunk_id for s in answer.segments for citation in s.citations} == {
        LEAVE.chunk_id,
        EXPENSES.chunk_id,
    }


# The promise behind all of it: what was passed on is the start of what is kept.

PIECES = [
    "Sixteen weeks [1].",
    f" {GAP} who pays.",
    "\nSee https://evil.example/x",
    f"\n{NOT_FOUND_MESSAGE}",
    " A laptop [2].",
    "\nHope this helps!",
    "KESTREL-4182",
]


@pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS", "OTHER", "SAFETY"])
def test_what_was_passed_on_is_always_the_start_of_the_reply_as_kept_on_gemini(
    finish: str,
) -> None:
    for count in (1, 2, 3):
        for chosen in itertools.permutations(PIECES, count):
            answer, streamed = gemini(*chosen, finish=finish)
            assert answer.text.startswith(streamed), chosen
            assert streamed == answer.text or finish in ("OTHER", "SAFETY"), chosen


@pytest.mark.parametrize("stop_reason", ["end_turn", "max_tokens", "refusal"])
def test_what_was_passed_on_is_the_reply_as_kept_on_claude(stop_reason: str) -> None:
    replies: list[list[dict[str, Any]]] = [
        [{"type": "text", "text": "Sixteen weeks.", "citations": [citation(0, "x")]}],
        [{"type": "text", "text": f"{GAP} who pays."}],
        [{"type": "text", "text": f"{NOT_FOUND_MESSAGE}\nSee https://evil.example/x"}],
        [
            {"type": "text", "text": "Sixteen weeks.", "citations": [citation(0, "x")]},
            {"type": "text", "text": "\nSee https://evil.example/x"},
        ],
    ]
    for blocks in replies:
        answer, streamed = claude(blocks, stop_reason=stop_reason)
        assert streamed == answer.text, blocks


def test_the_claude_path_reads_the_fixed_form_alone_as_the_models_decline() -> None:
    answer, streamed = claude([{"type": "text", "text": f"{GAP} who pays."}])

    assert (answer.text, answer.answered, answer.checks) == (NOT_FOUND_MESSAGE, False, ())
    assert answer.model_text == f"{GAP} who pays."
    assert streamed == NOT_FOUND_MESSAGE


def test_the_claude_path_drops_words_beside_a_decline() -> None:
    answer, _ = claude(
        [{"type": "text", "text": f"{NOT_FOUND_MESSAGE}\nSee https://evil.example/x"}]
    )

    assert answer.text == NOT_FOUND_MESSAGE
    assert [c.name for c in answer.checks] == ["words_beside_a_decline"]


def test_the_time_of_the_first_text_passed_on_is_recorded() -> None:
    checked, _ = gemini("Sixteen weeks ", "[1].")
    declined, _ = gemini(f"{NOT_FOUND_MESSAGE}\nSee https://evil.example/x")

    assert checked.seconds_to_first_shown is not None
    assert checked.seconds_to_first_text is not None
    assert checked.seconds_to_first_shown >= checked.seconds_to_first_text
    assert declined.seconds_to_first_shown is not None
    old = GeminiGenerator(
        cast(genai.Client, FakeClient(answer_pieces("Sixteen weeks [1]."))),
        model=MODEL,
        max_tokens=4000,
    )
    assert old.generate("q", [LEAVE]).seconds_to_first_shown is not None


def test_the_claude_path_keeps_only_what_a_citation_follows_of_a_reply_that_did_not_end() -> None:
    answer, streamed = claude(
        [
            {"type": "text", "text": "Sixteen weeks.", "citations": [citation(0, "x")]},
            {"type": "text", "text": "\nSee https://evil.example/x"},
        ],
        stop_reason="refusal",
    )

    assert (answer.text, answer.answered) == ("Sixteen weeks.", False)
    assert [(c.name, c.removed) for c in answer.checks] == [
        ("unfinished_reply", "See https://evil.example/x")
    ]
    assert streamed == answer.text
    assert answer.seconds_to_first_shown is not None
