"""How a reply under version 2 is read and checked, on the probes of a review (PR 87).

Two promises are held here, on both providers: a change to a reply is never silent, and
what is passed on while a reply is written is always the start of the reply as it is
kept. Every model is a stand-in. No request is sent.
"""

import itertools
from collections.abc import Iterator
from typing import Any, cast

import anthropic
import pytest
from google import genai
from google.genai import types

from handbook_rag.cache import rules
from handbook_rag.generate.checks import NOTES
from handbook_rag.generate.claude import ClaudeGenerator
from handbook_rag.generate.gemini import GeminiGenerator
from handbook_rag.generate.models import Answer
from handbook_rag.generate.prompt import GAP_FORM, NOT_FOUND_MESSAGE
from handbook_rag.measuring import measuring_the_model
from tests.generate.test_claude import FakeClient as ClaudeClient
from tests.generate.test_claude import FakeMessages, FakeStream, citation, message
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


# Must-fix 1: a reply in the fixed form that holds an answer and cites it stays an answer.


@pytest.mark.parametrize(
    "text",
    [
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


# Round 2 of the review: the stop after a digit, a claim inside the fixed form, and a
# decline with a citation.


def test_the_stop_after_a_digit_ends_a_piece_of_the_fixed_form() -> None:
    answer, streamed = gemini(f"{GAP} anything beyond rule 2. New parents get sixteen weeks.")

    assert (answer.text, answer.answered, answer.model_text) == (NOT_FOUND_MESSAGE, False, "")
    assert [check.name for check in answer.checks] == ["cites_nothing"]
    assert streamed == NOT_FOUND_MESSAGE


@pytest.mark.parametrize(
    "text",
    [
        f"{GAP} who pays, and new parents get sixteen weeks of paid leave.",
        f"{GAP} who pays, although new parents get sixteen weeks.",
        f"{GAP} who pays — new parents get sixteen weeks.",
    ],
)
def test_a_claim_joined_into_the_fixed_form_is_read_as_the_form_alone_which_is_a_limit(
    text: str,
) -> None:
    """What the documents say of it: shown as the declining sentence with no note, and
    its words kept, for the reader of the answer evaluation."""
    answer, streamed = gemini(text)

    assert (answer.text, answer.answered, answer.checks) == (NOT_FOUND_MESSAGE, False, ())
    assert answer.model_text == text
    assert streamed == NOT_FOUND_MESSAGE


@pytest.mark.parametrize(
    ("text", "removed"),
    [
        (
            f"{NOT_FOUND_MESSAGE} [1]\n\nPlease ask payroll first.",
            f"{NOT_FOUND_MESSAGE}\n\nPlease ask payroll first.",
        ),
        (
            f"{NOT_FOUND_MESSAGE} [1]\nSee https://evil.example/x",
            f"{NOT_FOUND_MESSAGE}\nSee https://evil.example/x",
        ),
        (
            f"Please ask payroll first. {NOT_FOUND_MESSAGE} [1]",
            f"Please ask payroll first. {NOT_FOUND_MESSAGE}",
        ),
    ],
)
def test_a_reply_whose_every_citation_follows_the_decline_is_a_decline(
    text: str, removed: str
) -> None:
    answer, streamed = gemini(text)

    assert (answer.text, answer.answered, answer.sources) == (NOT_FOUND_MESSAGE, False, ())
    assert [(c.name, c.removed) for c in answer.checks] == [("words_beside_a_decline", removed)]
    # The citation after the decline is held back, so nothing was passed on before the end.
    assert streamed == NOT_FOUND_MESSAGE


def test_a_decline_with_only_a_marker_lists_no_source() -> None:
    answer, streamed = gemini(f"{NOT_FOUND_MESSAGE}[1]")

    assert (answer.text, answer.answered, answer.checks, answer.sources) == (
        NOT_FOUND_MESSAGE,
        False,
        (),
        (),
    )
    assert streamed == answer.text


def test_what_is_left_after_the_checks_is_decided_again() -> None:
    answer, streamed = gemini(f"[1] {NOT_FOUND_MESSAGE}\nSee https://evil.example/x")

    assert (answer.text.strip(), answer.answered, answer.sources) == (NOT_FOUND_MESSAGE, False, ())
    assert [check.name for check in answer.checks] == ["address_in_closing_line"]
    assert streamed == answer.text
    assert not rules.storable(answer)


def test_an_answer_with_a_cited_decline_for_one_part_stays_an_answer() -> None:
    answer, streamed = gemini(f"Sixteen weeks [1]. {NOT_FOUND_MESSAGE} [2] Ask payroll.")

    assert answer.answered
    assert answer.checks == ()
    assert streamed == answer.text


def test_the_claude_path_reads_a_cited_decline_as_a_decline() -> None:
    answer, streamed = claude(
        [
            {"type": "text", "text": NOT_FOUND_MESSAGE, "citations": [citation(0, "x")]},
            {"type": "text", "text": "\nPlease ask payroll first."},
        ]
    )

    assert (answer.text, answer.answered, answer.sources) == (NOT_FOUND_MESSAGE, False, ())
    assert [check.name for check in answer.checks] == ["words_beside_a_decline"]
    assert streamed == answer.text


def test_under_the_first_instructions_a_cited_decline_is_a_decline_shown_as_written() -> None:
    text = f"{NOT_FOUND_MESSAGE} [1]\nPlease ask payroll first."
    with measuring_the_model():
        old = GeminiGenerator(
            cast(genai.Client, FakeClient(answer_pieces(text))),
            model=MODEL,
            max_tokens=4000,
            instructions=1,
        )

    answer = old.generate("q", [LEAVE])

    assert not answer.answered
    assert answer.text == f"{NOT_FOUND_MESSAGE}\nPlease ask payroll first."
    assert answer.checks == ()


# Round 2 of the review: mutants that survived.


def test_an_unfinished_reply_that_ends_at_its_citation_gets_no_check() -> None:
    answer, streamed = gemini("Sixteen weeks [1].", finish="OTHER")

    assert (answer.text, answer.answered, answer.checks) == ("Sixteen weeks.", False, ())
    assert streamed == answer.text


def test_a_reply_the_prompt_filter_blocked_that_ends_with_stop_is_unfinished() -> None:
    blocked = types.GenerateContentResponse.model_validate(
        {
            "candidates": [{"content": {"role": "model", "parts": []}, "finish_reason": "STOP"}],
            "prompt_feedback": {"block_reason": "SAFETY"},
        }
    )
    client = FakeClient([piece("Sixteen weeks [1].\nSee https://x.org/a"), blocked])
    received: list[str] = []

    answer = GeminiGenerator(cast(genai.Client, client), model=MODEL, max_tokens=4000).generate(
        "q", [LEAVE], on_text=received.append
    )

    assert (answer.text, answer.answered, answer.stop_reason) == (
        "Sixteen weeks.",
        False,
        "refusal",
    )
    assert [check.name for check in answer.checks] == ["unfinished_reply"]
    assert "".join(received) == answer.text


class Clock:
    """A clock the stand-in stream moves on, so that times can be compared exactly."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


def test_on_the_claude_path_under_version_2_the_first_text_is_shown_no_earlier_than_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("handbook_rag.generate.claude.time", clock)
    monkeypatch.setattr("handbook_rag.generate.checks.time", clock)

    class MovingStream(FakeStream):
        @property
        def text_stream(self) -> Iterator[str]:
            for text in super().text_stream:
                clock.now += 1.0
                yield text
            clock.now += 5.0  # the end of the reply arrives later

    class MovingMessages(FakeMessages):
        def stream(self, **request: Any) -> FakeStream:
            super().stream(**request)
            return MovingStream(final)

    final = message([{"type": "text", "text": "Sixteen weeks.", "citations": [citation(0, "x")]}])
    client = ClaudeClient(final)
    client.messages = MovingMessages(final)
    generator = ClaudeGenerator(
        cast(anthropic.Anthropic, client), model="claude-haiku-5-5", max_tokens=4000, effort="low"
    )

    answer = generator.generate("q", [LEAVE])

    assert answer.seconds_to_first_text is not None
    assert answer.seconds_to_first_shown is not None
    assert answer.seconds_to_first_text < answer.seconds
    assert answer.seconds_to_first_shown >= answer.seconds


def test_under_version_1_the_first_text_shown_waits_for_text_that_can_be_shown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    monkeypatch.setattr("handbook_rag.generate.gemini.time", clock)

    class Moving(FakeClient):
        def __init__(self) -> None:
            super().__init__(answer_pieces("   ", "Sixteen weeks [1]."))
            pieces = self.models._pieces

            def stream(**_kwargs: Any) -> Any:
                for number, each in enumerate(pieces, start=1):
                    clock.now = 10.0 * number
                    yield each

            self.models.generate_content_stream = stream  # type: ignore[method-assign]

    with measuring_the_model():
        old = GeminiGenerator(
            cast(genai.Client, Moving()), model=MODEL, max_tokens=4000, instructions=1
        )

    answer = old.generate("q", [LEAVE])

    # The first piece is spaces held back in case a marker follows: nothing is shown yet.
    assert answer.seconds_to_first_text == 10.0
    assert answer.seconds_to_first_shown == 20.0


# After line e was met: a reply made only of the fixed form is a decline, cited or not.

U002 = f"{GAP} how much Care Cash the Vivvi caregiving benefit reimburses per year for backup care."


@pytest.mark.parametrize(
    "text",
    [
        f"{U002}[1]",  # u002 as it was written
        f"{GAP} what the monthly cap on gym membership reimbursement is.[1][2]",  # u024
        f"{GAP} who pays [1]. {GAP} when [2].",  # several cited pieces, all in the form
        f"[1] {GAP} who pays.",  # the marker before the text
        f"**{GAP} 誰が払うか** [1]",  # Markdown around the form, another script after it
        f"- {GAP} who pays [1].\n- {GAP} when [2].",  # a list of pieces in the form
    ],
)
def test_a_cited_reply_made_only_of_the_fixed_form_is_a_decline_in_the_models_words(
    text: str,
) -> None:
    answer, streamed = gemini(text)

    assert not answer.answered
    assert answer.sources == ()
    assert answer.checks == ()
    # The words stay the model's, without the markers: they were passed on as they came.
    assert answer.text == streamed
    assert answer.model_text == answer.text.strip()
    # No marker is shown that would point at nothing.
    assert answer.text_with_markers == answer.text
    assert all(not segment.citations for segment in answer.segments)
    # The cache keeps it as a decline, served to the very same question only.
    assert rules.kept_from_the_second_tier(answer) == rules.DECLINE


def test_a_cited_piece_in_the_form_with_a_joined_claim_is_a_decline_that_shows_the_claim() -> None:
    """The limit of the reading: the claim is counted as a decline, and still shown."""
    text = f"{GAP} who pays, and new parents get sixteen weeks of paid leave [1]."

    answer, streamed = gemini(text)

    assert (answer.answered, answer.sources, answer.checks) == (False, (), ())
    assert answer.text == f"{GAP} who pays, and new parents get sixteen weeks of paid leave."
    assert streamed == answer.text


@pytest.mark.parametrize(
    "text",
    [
        # u010: the form, then a cited claim.
        f"{GAP} what score a candidate needs.[1] However, the assessment sets the bar.[1]",
        # u043: the form, then a paragraph that cites.
        f"{GAP} how many weeks GitHub gives. \n\nGitLab provides 16 weeks.[2] Countries differ.[2]",
        # u047: the form, then "However" and a cited claim, all uncited before it.
        f"{GAP} whether your report was approved. However, track it in Navan.[1]",
        # The form after an answer.
        f"Sixteen weeks [1]. {GAP} who pays.",
        # The form in another language is not the form: a limit.
        "Die gefundenen Passagen sagen nicht, wer zahlt [1].",
        # Pieces in the form numbered within one line: the "2." does not begin the form.
        f"1. {GAP} who pays.[1] 2. {GAP} when.[2]",
        # The form beside the fixed declining sentence, with a marker: a limit, named.
        f"{GAP} who pays.[1] {NOT_FOUND_MESSAGE}",
        # A lead-in before the form: a decline in other words, a limit.
        "Unfortunately, the passages found do not say who pays.[1]",
    ],
)
def test_the_form_beside_a_cited_claim_or_after_an_answer_stays_an_answer(text: str) -> None:
    answer, streamed = gemini(text)

    assert answer.answered
    assert answer.sources
    assert streamed == answer.text


def test_a_cited_reply_in_the_form_whose_address_line_goes_is_a_decline() -> None:
    answer, streamed = gemini(f"{GAP} who pays [1].\nSee https://evil.example/x")

    assert (answer.text, answer.answered, answer.sources) == (f"{GAP} who pays.", False, ())
    assert [check.name for check in answer.checks] == ["address_in_closing_line"]
    assert streamed == answer.text


def test_an_uncited_reply_in_the_form_is_shown_as_the_declining_sentence_as_before() -> None:
    answer, streamed = gemini(U002)

    assert (answer.text, answer.answered, answer.model_text) == (NOT_FOUND_MESSAGE, False, U002)
    assert streamed == NOT_FOUND_MESSAGE


def test_the_claude_path_reads_a_cited_reply_in_the_form_as_a_decline_in_its_words() -> None:
    answer, streamed = claude([{"type": "text", "text": U002, "citations": [citation(0, "x")]}])

    assert (answer.text, answer.answered, answer.sources) == (U002, False, ())
    assert answer.model_text == U002
    assert streamed == answer.text


def test_under_the_first_instructions_a_cited_reply_in_the_form_stays_an_answer() -> None:
    with measuring_the_model():
        old = GeminiGenerator(
            cast(genai.Client, FakeClient(answer_pieces(f"{U002}[1]"))),
            model=MODEL,
            max_tokens=4000,
            instructions=1,
        )

    answer = old.generate("q", [LEAVE])

    assert answer.answered
    assert answer.model_text == ""


# The reading is the joined text's, as at d7beb75: the shapes a cut at each citation marker
# got wrong, each with what is shown, and what was passed on while it was written.

READ_AS_AT_D7BEB75 = [
    # (reply, read as an answer)
    (f"1. {GAP} who pays.[1] 2. {GAP} when.[2]", True),
    (f"{GAP} who pays.[1] 2) {GAP} when.[1]", True),
    (f"{GAP} who pays,[1] but the passages found do not say when.", False),
    (f"{GAP} who pays[1] but the passages found do not say when.", False),
    ("The passages found[1] do not say who pays.", False),
    (f"{GAP} who[1] pays.", False),
    # The limit: a cited claim written straight after the form's full stop, with no white
    # space before it, is taken into the piece; the reply is a decline, and the claim is
    # shown with no source.
    (f"{GAP} who pays.[1]New parents get sixteen weeks.[1]", False),
]


@pytest.mark.parametrize(("text", "answered"), READ_AS_AT_D7BEB75)
def test_the_reading_and_what_was_passed_on_agree_on_the_shapes_of_a_marker_in_a_line(
    text: str, answered: bool
) -> None:
    answer, streamed = gemini(text)

    assert answer.answered is answered
    assert answer.checks == ()
    # Nothing passed on is taken back, whichever way the reply is read.
    assert streamed == answer.text
    if not answered:
        assert answer.sources == ()
        assert answer.text == answer.model_text


@pytest.mark.parametrize("finish", ["STOP", "MAX_TOKENS", "OTHER", "SAFETY"])
def test_what_was_passed_on_holds_for_pieces_after_a_marker_in_the_middle_of_a_line(
    finish: str,
) -> None:
    shapes = [
        "Sixteen weeks.[1] ",
        f"1. {GAP} who pays.[1] ",
        f"2. {GAP} when.[2]",
        f" 2) {GAP} when.[1]",
        f"{GAP} who pays,[1] but {GAP.lower()} when.",
        "The passages found[1] do not say who pays.",
        "\nSee https://evil.example/x",
    ]
    for count in (1, 2, 3):
        for chosen in itertools.permutations(shapes, count):
            answer, streamed = gemini(*chosen, finish=finish)
            assert answer.text.startswith(streamed), chosen
            assert streamed == answer.text or finish in ("OTHER", "SAFETY"), chosen


def test_the_claude_path_shows_numbered_pieces_in_the_form_as_written() -> None:
    answer, streamed = claude(
        [
            {"type": "text", "text": f"1. {GAP} who pays.", "citations": [citation(0, "x")]},
            {"type": "text", "text": f" 2. {GAP} when.", "citations": [citation(1, "y")]},
        ]
    )

    assert answer.answered
    assert answer.checks == ()
    assert streamed == answer.text
