"""The rules of the rewriting step, tested with a stand-in model.

No request is sent. Each rule of ``rewrite/base.py`` has a test here: when the model is
asked, what a usable reply is, what happens to one that is not or to a request that
fails, and what the trace of the step holds.
"""

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from handbook_rag.api import MAX_QUERY_CHARS
from handbook_rag.generate.base import GenerationError
from handbook_rag.generate.models import TokenUsage
from handbook_rag.rewrite.base import numbers_of_its_own, read_reply, standalone_question
from handbook_rag.rewrite.models import (
    MAX_EARLIER_TURNS,
    MAX_QUESTION_CHARS,
    RewriteReply,
    Turn,
)
from handbook_rag.rewrite.prompt import INSTRUCTION_PHRASES
from tests.rewrite.support import StandInRewriter, failing, replying

UK = Turn(
    "How many days of paid annual leave do UK team members get?",
    answer="UK team members get 28 days total: 20 days annual leave plus 8 public holidays.",
)
FOLLOW_UP = "And in Finland?"
STANDING_ALONE = "How many days of paid annual leave do team members in Finland get?"
# Curly marks by name, so that no reader takes one for a straight one.
LEFT_SINGLE, RIGHT_SINGLE = "\N{LEFT SINGLE QUOTATION MARK}", "\N{RIGHT SINGLE QUOTATION MARK}"
LEFT_DOUBLE, RIGHT_DOUBLE = "\N{LEFT DOUBLE QUOTATION MARK}", "\N{RIGHT DOUBLE QUOTATION MARK}"


def reply(text: str, *, stop_reason: str = "end_turn") -> RewriteReply:
    return RewriteReply(
        text=text,
        model="stand-in-001",
        stop_reason=stop_reason,
        usage=TokenUsage(input_tokens=400, output_tokens=20),
        seconds=0.5,
    )


# When the model is asked.


def test_without_earlier_turns_no_model_is_asked_and_the_question_is_returned_as_asked(
    spans: InMemorySpanExporter,
) -> None:
    model = replying("Something else?")

    rewrite = standalone_question("How do I get a Navan purchasing card?", [], rewriter=model)

    assert model.calls == []
    assert rewrite.question == "How do I get a Navan purchasing card?"
    assert rewrite.outcome == "not_asked"
    assert rewrite.requests == 0
    assert rewrite.reply is None
    assert not rewrite.changed
    assert not rewrite.fell_back
    assert spans.get_finished_spans() == ()


def test_with_earlier_turns_the_model_is_asked_once_even_for_a_question_that_stands_alone() -> None:
    # No test on this machine spares the request: the model gives the question back.
    model = replying("How can I buy a Yubikey for myself?")

    rewrite = standalone_question("How can I buy a Yubikey for myself?", [UK], rewriter=model)

    assert len(model.calls) == 1
    assert rewrite.outcome == "unchanged"
    assert rewrite.requests == 1


def test_the_model_is_given_the_questions_of_the_latest_turns_and_never_an_answer() -> None:
    turns = [Turn(f"Question {number}?", answer=f"Answer {number}.") for number in range(1, 10)]
    model = replying(STANDING_ALONE)

    standalone_question(FOLLOW_UP, turns, rewriter=model)

    ((sent, question),) = model.calls
    assert question == FOLLOW_UP
    assert [turn.question for turn in sent] == [
        f"Question {number}?" for number in range(10 - MAX_EARLIER_TURNS, 10)
    ]
    assert all(turn.answer is None for turn in sent)


# What a usable reply is.


def test_a_usable_rewrite_is_the_question_to_search_for() -> None:
    rewrite = standalone_question(FOLLOW_UP, [UK], rewriter=replying(f"  {STANDING_ALONE}\n"))

    assert rewrite.question == STANDING_ALONE
    assert rewrite.outcome == "rewritten"
    assert rewrite.changed
    assert not rewrite.fell_back
    assert rewrite.requests == 1
    assert rewrite.seconds == 0.75
    assert rewrite.reply is not None
    assert rewrite.reply.text == f"  {STANDING_ALONE}\n"


@pytest.mark.parametrize(
    "given_back",
    [
        "And in Finland?",
        "and in finland",
        "  And in Finland? ",
        '"And in Finland?"',
        "And, in Finland!",
    ],
)
def test_the_question_given_back_in_any_form_is_the_question_exactly_as_asked(
    given_back: str,
) -> None:
    rewrite = standalone_question(FOLLOW_UP, [UK], rewriter=replying(given_back))

    assert rewrite.outcome == "unchanged"
    assert rewrite.question == FOLLOW_UP
    assert not rewrite.changed
    assert not rewrite.fell_back


@pytest.mark.parametrize(
    "wrapped",
    [
        f'"{STANDING_ALONE}"',
        f"'{STANDING_ALONE}'",
        f"`{STANDING_ALONE}`",
        f"{LEFT_DOUBLE}{STANDING_ALONE}{RIGHT_DOUBLE}",
        f"{LEFT_SINGLE}{STANDING_ALONE}{RIGHT_SINGLE}",
    ],
)
def test_one_pair_of_quotation_marks_around_the_whole_reply_is_taken_off(wrapped: str) -> None:
    rewrite = standalone_question(FOLLOW_UP, [UK], rewriter=replying(wrapped))

    assert rewrite.outcome == "rewritten"
    assert rewrite.question == STANDING_ALONE


@pytest.mark.parametrize("apostrophe", ["'", RIGHT_SINGLE])
def test_an_apostrophe_inside_a_word_does_not_keep_the_marks_around_a_reply(
    apostrophe: str,
) -> None:
    question = f"What{apostrophe}s the longest expiry the Token Management Standard allows?"
    opening, closing = ("'", "'") if apostrophe == "'" else (LEFT_SINGLE, RIGHT_SINGLE)

    rewrite = standalone_question(
        "What's the longest expiry it allows?",
        [Turn("What is the Token Management Standard?")],
        rewriter=replying(f"{opening}{question}{closing}"),
    )

    assert rewrite.outcome == "rewritten"
    assert rewrite.question == question


def test_marks_that_are_not_one_pair_around_the_whole_reply_stay() -> None:
    # Two quoted names are not one quoted question. The reply then does not end with a
    # question mark, as the question does, so it is not used.
    rewrite = read_reply(
        "Is it the same for them?",
        [UK],
        reply('"Annual leave" and "public holidays"'),
    )

    assert rewrite.outcome == "unusable"


@pytest.mark.parametrize(
    ("text", "stop_reason", "problem"),
    [
        (STANDING_ALONE, "max_tokens", "did not come to a proper end (max_tokens)"),
        (STANDING_ALONE, "refusal", "did not come to a proper end (refusal)"),
        ("", "end_turn", "holds no words"),
        ("  ?! ", "end_turn", "holds no words"),
        (f"{STANDING_ALONE}\nI filled in the subject.", "end_turn", "more than one line"),
        ("How many days?\r\nIn Finland?", "end_turn", "more than one line"),
        ("How many days " + "really " * 80 + "?", "end_turn", "longer than 500 characters"),
        (f"<new_question>{STANDING_ALONE}</new_question>", "end_turn", "a tag of the request"),
        (f"&lt;earlier_question>{STANDING_ALONE}", "end_turn", "a tag of the request"),
        (
            "Note that the search knows nothing of what was asked before it?",
            "end_turn",
            "repeats the step's instructions",
        ),
        ("Team members in Finland get the statutory minimum.", "end_turn", "question mark"),
        (
            f"How many days of paid annual leave do UK team members get? {STANDING_ALONE}",
            "end_turn",
            "2 question marks, where 1 may be",
        ),
        (
            "How many days of paid annual leave do team members in Finland get in 2026?",
            "end_turn",
            "adds a number the conversation does not hold (2026)",
        ),
        (
            "Do team members in Finland get 25 days of paid annual leave?",
            "end_turn",
            "adds a number the conversation does not hold (25)",
        ),
    ],
)
def test_a_reply_that_is_not_one_usable_question_leaves_the_question_as_asked(
    text: str, stop_reason: str, problem: str
) -> None:
    model = StandInRewriter(lambda _turns, _question: reply(text, stop_reason=stop_reason))

    rewrite = standalone_question(FOLLOW_UP, [UK], rewriter=model)

    assert rewrite.outcome == "unusable"
    assert rewrite.question == FOLLOW_UP
    assert rewrite.problem is not None
    assert problem in rewrite.problem
    assert rewrite.fell_back
    assert not rewrite.changed
    # What the model sent is kept, so that the evaluation can read it.
    assert rewrite.reply is not None
    assert rewrite.reply.text == text
    assert rewrite.requests == 1


def test_every_phrase_of_the_instructions_makes_a_reply_unusable() -> None:
    for phrase in INSTRUCTION_PHRASES:
        rewrite = read_reply(FOLLOW_UP, [UK], reply(f"Why does it say {phrase.upper()}?"))

        assert rewrite.outcome == "unusable", phrase


def test_a_reply_holds_a_question_mark_only_when_the_question_does_not_need_one() -> None:
    typed = "annual leave finland"
    without = "How many days of paid annual leave do team members in Finland get"

    assert read_reply(typed, [UK], reply(without)).outcome == "rewritten"
    assert read_reply(typed, [UK], reply(f"{without}?")).outcome == "rewritten"
    # Asked as two questions, it may come back as two.
    two = "Who approves it? And how long does it take?"
    both = "Who approves a Yubikey? And how long does a Yubikey take to arrive?"
    assert read_reply(two, [Turn("How can I buy a Yubikey?")], reply(both)).outcome == "rewritten"


@pytest.mark.parametrize(
    ("earlier", "question", "rewritten"),
    [
        # From the question itself.
        (UK.question, "And after 2 years?", "How many days of paid annual leave after 2 years?"),
        # From an earlier question, also one that is not the last.
        (
            "Was the FY26 budget spent?",
            "And what about the team's?",
            "Was the team's FY26 budget spent?",
        ),
        # Written out in words in the conversation.
        (
            "Can I choose which of the eight I take?",
            "In the UK?",
            "Can I choose which of the 8 I take in the UK?",
        ),
        (
            "What does the second step involve?",
            "For a coach?",
            "What does step 2 involve for a coach?",
        ),
        # Thousands grouped or not, and leading zeros, are one number.
        ("Is the bonus 1,500 dollars?", "In euros?", "Is the bonus 1500 euros?"),
        ("Is it due on the 07th?", "Of which month?", "Of which month is it due on the 7th?"),
    ],
)
def test_a_number_the_conversation_holds_is_not_an_addition(
    earlier: str, question: str, rewritten: str
) -> None:
    rewrite = read_reply(question, [Turn("Unrelated?"), Turn(earlier)], reply(rewritten))

    assert rewrite.outcome == "rewritten", rewrite.problem


def test_numbers_of_its_own_names_each_added_number_once_in_order() -> None:
    added = numbers_of_its_own("Is it 25 days in 2026, or 25 in 2027?", ["How many days in 2026?"])

    assert added == ("25", "2027")
    # A number written out in words in the rewrite is not seen.
    assert numbers_of_its_own("Is it twenty five days?", ["How many days?"]) == ()


def test_the_turns_given_are_the_ones_a_number_may_come_from() -> None:
    # A number from an answer is not in what the model was given, so it is an addition.
    added = read_reply(
        "Is it paid?",
        [UK],
        reply("Are the 28 days of paid annual leave of UK team members paid?"),
    )

    assert added.outcome == "unusable"
    assert added.problem is not None
    assert "(28)" in added.problem


# What happens when the request fails.


@pytest.mark.parametrize(
    "reason", ["rate_limit", "quota", "configuration", "unavailable", "timeout"]
)
def test_a_failed_request_never_leaves_the_step_and_leaves_the_question_as_asked(
    reason: str,
) -> None:
    error = GenerationError("The model is overloaded.", reason=reason)  # type: ignore[arg-type]

    rewrite = standalone_question(FOLLOW_UP, [UK], rewriter=failing(error))

    assert rewrite.outcome == "failed"
    assert rewrite.question == FOLLOW_UP
    assert rewrite.failure is error
    assert error.requests == 1
    assert rewrite.requests == 1
    assert rewrite.reply is None
    assert rewrite.fell_back
    assert rewrite.seconds >= 0


def test_a_fault_in_our_own_code_is_not_taken_for_a_failed_request() -> None:
    model = StandInRewriter(lambda _turns, _question: KeyError("a bug"))

    with pytest.raises(KeyError, match="a bug"):
        standalone_question(FOLLOW_UP, [UK], rewriter=model)


def test_the_step_returns_no_longer_question_than_post_ask_accepts() -> None:
    assert MAX_QUESTION_CHARS == MAX_QUERY_CHARS


# The trace.


def test_the_step_and_its_request_each_have_a_span(spans: InMemorySpanExporter) -> None:
    standalone_question(FOLLOW_UP, [UK, Turn("And in Spain?")], rewriter=replying(STANDING_ALONE))

    calling, step = spans.get_finished_spans()
    assert step.name == "rewrite"
    facts = dict(step.attributes or {})
    assert facts["handbook_rag.earlier_turns"] == 2
    assert facts["handbook_rag.rewrite.outcome"] == "rewritten"
    assert facts["handbook_rag.query.chars"] == len(STANDING_ALONE)
    assert "handbook_rag.query.text" not in facts, "no words of a question without content on"
    assert calling.name == "chat stand-in-rewriter"
    assert calling.parent is not None
    assert calling.parent.span_id == step.context.span_id
    called = dict(calling.attributes or {})
    assert called["gen_ai.provider.name"] == "stand-in"
    assert called["gen_ai.request.model"] == "stand-in-rewriter"
    assert called["gen_ai.response.model"] == "stand-in-rewriter-001"
    assert called["gen_ai.usage.input_tokens"] == 420
    assert called["gen_ai.usage.output_tokens"] == 18
    assert called["handbook_rag.passages.given"] == 0


def test_a_failed_request_marks_its_own_span_and_not_the_steps(
    spans: InMemorySpanExporter,
) -> None:
    error = GenerationError("The model is overloaded.", reason="unavailable")

    standalone_question(FOLLOW_UP, [UK], rewriter=failing(error))

    calling, step = spans.get_finished_spans()
    assert calling.status.status_code is StatusCode.ERROR
    assert dict(calling.attributes or {})["handbook_rag.failure"] == "unavailable"
    assert step.status.status_code is StatusCode.UNSET
    assert dict(step.attributes or {})["handbook_rag.rewrite.outcome"] == "failed"
