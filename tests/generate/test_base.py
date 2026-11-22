"""The rules every generator shares: what counts as an answer, and declining without a model."""

import pytest

from handbook_rag.generate.base import decline_without_context, is_answered
from handbook_rag.generate.models import Citation, Segment
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from tests.support import make_chunk

LEAVE = make_chunk("New parents get sixteen weeks of paid leave.")
CITED = (Citation(LEAVE),)


def test_a_plain_statement_is_an_answer_even_without_a_citation() -> None:
    assert is_answered([Segment("Parents get sixteen weeks.")], refused=False)


@pytest.mark.parametrize(
    "reply",
    [
        NOT_FOUND_MESSAGE,
        NOT_FOUND_MESSAGE.rstrip("."),  # the full stop left out
        NOT_FOUND_MESSAGE.upper(),
        f"  {NOT_FOUND_MESSAGE}\n",
        f"Sorry. {NOT_FOUND_MESSAGE} Ask the People team.",  # the sentence among other words
    ],
)
def test_the_fixed_sentence_without_a_citation_is_a_decline(reply: str) -> None:
    assert not is_answered([Segment(reply)], refused=False)


@pytest.mark.parametrize("reply", [NOT_FOUND_MESSAGE, NOT_FOUND_MESSAGE.rstrip(".").lower()])
def test_the_fixed_sentence_alone_is_a_decline_even_with_a_citation(reply: str) -> None:
    # A model told to cite every statement may put a marker on this sentence too.
    assert not is_answered([Segment(reply, CITED)], refused=False)


def test_an_answer_that_names_a_gap_and_cites_what_it_found_is_an_answer() -> None:
    segments = [Segment("Parents get sixteen weeks.", CITED), Segment(f" {NOT_FOUND_MESSAGE}")]

    assert is_answered(segments, refused=False)


@pytest.mark.parametrize("reply", ["", "   ", "\n\n"])
def test_a_reply_with_no_words_is_not_an_answer(reply: str) -> None:
    assert not is_answered([Segment(reply)], refused=False)
    assert not is_answered([], refused=False)


def test_a_withheld_reply_is_not_an_answer_whatever_it_holds() -> None:
    assert not is_answered([Segment("Parents get sixteen weeks.", CITED)], refused=True)


def test_declining_without_context_says_so_and_names_the_model() -> None:
    received: list[str] = []

    answer = decline_without_context("anything", "some-model", received.append)

    assert answer.text == NOT_FOUND_MESSAGE
    assert received == [NOT_FOUND_MESSAGE]
    assert not answer.answered
    assert answer.stop_reason == "no_context"
    assert answer.model == "some-model"
    assert answer.question == "anything"
    assert decline_without_context("anything", "some-model", None).text == NOT_FOUND_MESSAGE
