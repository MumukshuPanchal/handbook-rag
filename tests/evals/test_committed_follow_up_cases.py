"""The follow-up cases in the repository are what their card says they are.

Only the last two tests need a model, the verifier, which they load from the download
cache or the Hub; they are marked slow, and do not run in CI.
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pytest

from handbook_rag.evals.adversarial import find_canaries
from handbook_rag.evals.dataset import load_golden_set
from handbook_rag.evals.follow_ups import (
    FOLLOW_UP,
    KINDS,
    PLANTED,
    STANDALONE,
    VERIFIED_AT,
    FollowUpCase,
    load_cases,
    refuse_phrases_in_cases,
)
from handbook_rag.generate.base import plain_words
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.rewrite.base import read_reply
from handbook_rag.rewrite.models import MAX_EARLIER_TURNS, RewriteReply
from handbook_rag.rewrite.prompt import EXAMPLE_QUESTIONS, INSTRUCTION_PHRASES, holds_our_tags
from tests.test_readme_results import in_words

ROOT = Path(__file__).parents[2]
CASES = ROOT / "evals" / "follow-ups" / "cases.jsonl"
GOLDEN = {
    question.id: question for question in load_golden_set(ROOT / "evals/golden/questions.jsonl")
}
ADVERSARIAL = ROOT / "evals" / "adversarial" / "cases.jsonl"
SECOND_READING = ROOT / "evals" / "follow-ups" / "second-reading.jsonl"
SECOND_READING_ADDED = ROOT / "evals" / "follow-ups" / "second-reading-added.jsonl"
CARD = ROOT / "evals" / "follow-ups" / "README.md"
# The cases added on 25 November, after a review of the step, for what it had not tested:
# questions typed without a question mark, orders that leave no canary when obeyed, and
# tags of the request forged with characters a plain search does not see. The second
# reading of the others was done before them, so another session read them a second time,
# and the saved run asked them in a top-up.
ADDED_AFTER_THE_REVIEW = (
    "fu-subject-9",
    "fu-pronoun-9",
    "fu-third-9",
    "fu-change-9",
    "fu-planted-turn-9",
    "fu-planted-turn-10",
    "fu-planted-question-9",
    "fu-planted-question-10",
    "fu-planted-question-11",
    "fu-planted-question-12",
)
# The second reader's first question had the words of none of the maker's forms here.
# The second model took the first six for the same, both ways, at 0.99 or more. The maker
# read the next ten as the same question with a detail from an earlier answer or turn.
# The last two differ in the reading the reader put first.
VERIFIED_BY_THE_SECOND_MODEL = {
    "fu-answer-5",
    "fu-pronoun-1",
    "fu-pronoun-4",
    "fu-third-1",
    "fu-third-3",
    "fu-third-6",
}
SAME_WITH_A_DETAIL_ADDED = {
    "fu-answer-1",
    "fu-answer-3",
    "fu-answer-6",
    "fu-answer-7",
    "fu-answer-8",
    "fu-ambiguous-3",
    "fu-ambiguous-7",
    "fu-planted-question-8",
    "fu-planted-turn-8",
    "fu-pronoun-8",
}
ANOTHER_READING_FIRST = {"fu-ambiguous-8", "fu-planted-turn-5"}
# Of the cases added after the review, the second reader's first question had the words of
# none of the maker's forms here, and the maker reads it as the same question in other words.
REWORDED_IN_THE_SECOND_READING_OF_THE_ADDED = {"fu-pronoun-9", "fu-planted-question-12"}
# The second model's weaker direction for the second reader's questions that add a
# detail, as the documents give it: most at or under the first value, and the others at
# these, to three places. The slow test below works them out with the model.
DETAIL_LOW = "0.006"
DETAIL_HIGH = ("0.958", "0.986")
# A test on this machine that would spare a request takes a question for a follow-up when
# it holds a telltale word, or has fewer words than this.
FEWEST_WORDS = 6
# The acceptable forms taken out after the review, as looser than the second reader's
# standard. No reply had their words.
FORMS_TAKEN_OUT = {
    "fu-planted-turn-8": "What paperwork will the Absence Management team send me?",
    "fu-planted-turn-5": "Who pays for background screenings of GitLab team members?",
}
# The words a simple test on this machine might take for the sign of a follow-up, in the
# order the card lists them.
TELLTALES = (
    "it",
    "its",
    "they",
    "them",
    "their",
    "that",
    "those",
    "these",
    "this",
    "there",
    "one",
    "ones",
    "and",
    "also",
    "same",
    "too",
    "instead",
    "else",
    "either",
)


def cases() -> list[FollowUpCase]:
    return load_cases(CASES)


def read_twice() -> list[FollowUpCase]:
    """The cases the first second reading read: all but those added after the review."""
    return [case for case in cases() if case.case_id not in ADDED_AFTER_THE_REVIEW]


def test_the_cases_are_in_the_numbers_the_card_gives() -> None:
    counts = Counter(case.kind for case in cases())

    assert counts == {
        **{kind: 8 for kinds in KINDS.values() for kind in kinds},
        "subject_left_out": 9,
        "pronoun": 9,
        "third_turn": 9,
        "change_of_subject": 9,
        "planted_in_turn": 10,
        "planted_in_question": 12,
    }
    groups = Counter(case.group for case in cases())
    assert groups == {FOLLOW_UP: 43, STANDALONE: 17, PLANTED: 22}
    # Before the review there were 8 of each kind.
    assert set(Counter(case.kind for case in read_twice()).values()) == {8}
    assert [case.case_id for case in cases()][-len(ADDED_AFTER_THE_REVIEW) :] == list(
        ADDED_AFTER_THE_REVIEW
    )
    # One request for each case: the number the Makefile names.
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert f"({len(cases())} real requests)" in makefile


def test_every_conversation_starts_from_a_golden_question_and_its_answer() -> None:
    for case in cases():
        golden = GOLDEN[case.source]
        first = case.turns[0]
        assert first.question == golden.question, case.case_id
        expected = golden.reference_answer if golden.type == "answerable" else NOT_FOUND_MESSAGE
        assert first.answer == expected, case.case_id
    assert len({case.source for case in cases()}) == 28


def test_a_later_turn_carries_an_answer_only_when_it_is_a_golden_question() -> None:
    by_question = {question.question: question for question in GOLDEN.values()}
    for case in cases():
        for turn in case.turns[1:]:
            golden = by_question.get(turn.question)
            if golden is None:
                assert turn.answer is None, case.case_id
            else:
                assert turn.answer == golden.reference_answer, case.case_id


def test_a_target_is_a_golden_question_whose_words_are_an_acceptable_form() -> None:
    targeted = [case for case in cases() if case.target is not None]

    for case in targeted:
        assert case.target is not None
        assert GOLDEN[case.target].question in case.standalone, case.case_id
    assert len(targeted) == 17
    # Every question that stands alone is a golden question but one.
    alone = [case for case in cases() if case.group == STANDALONE]
    assert sum(case.target is not None for case in alone) == 14


def test_no_conversation_is_longer_than_two_earlier_turns() -> None:
    lengths = Counter(len(case.turns) for case in cases())

    assert lengths == {1: 67, 2: 15}
    assert max(lengths) <= MAX_EARLIER_TURNS
    assert {case.kind for case in cases() if len(case.turns) == 2} == {
        "third_turn",
        "two_referents",
    }


def test_half_of_the_follow_ups_into_an_answer_cannot_be_written_without_it() -> None:
    into_answer = [case for case in cases() if case.kind == "points_into_answer"]

    assert Counter(case.needs_answer for case in into_answer) == {True: 4, False: 4}
    for case in into_answer:
        assert case.turns[0].answer is not None, case.case_id
        assert case.turns[0].answer != NOT_FOUND_MESSAGE, case.case_id


def test_every_acceptable_form_is_one_the_step_would_take_from_the_model() -> None:
    # Otherwise a right rewrite would be refused by the step's own checks, and scored as
    # a question searched for as it was typed.
    for case in cases():
        for form in case.standalone:
            rewrite = read_reply(
                case.sent_question,
                case.sent_turns[-MAX_EARLIER_TURNS:],
                RewriteReply(form, "a model", "end_turn"),
            )
            assert rewrite.outcome in ("rewritten", "unchanged"), (case.case_id, rewrite.problem)


def test_every_canary_belongs_to_one_case_and_is_where_its_kind_needs_it() -> None:
    planted = [case for case in cases() if case.group == PLANTED]
    seen: list[str] = []
    for case in planted:
        assert case.planted is not None
        for canary in case.canaries:
            seen.append(canary)
            assert re.fullmatch(r"[A-Z]+-\d{4}|[a-z-]+\.example(/[a-z]+)?/\d{4}", canary), canary
            if case.kind == "planted_in_turn":
                assert find_canaries(case.planted, [canary]), case.case_id
            else:
                # Not written out anywhere, so that only a reply that obeyed holds it.
                sent = " ".join([case.sent_question, *(t.question for t in case.sent_turns)])
                assert not find_canaries(sent, [canary]), case.case_id
    assert Counter((case.kind, len(case.canaries)) for case in planted) == {
        ("planted_in_turn", 1): 8,
        ("planted_in_turn", 0): 2,
        ("planted_in_question", 1): 7,
        ("planted_in_question", 0): 5,
    }
    numbers = [re.sub(r"\D", "", canary) for canary in seen]
    assert len(set(numbers)) == len(numbers) == 15
    # And none is a number of the adversarial cases' canaries, so a reply cannot be
    # taken for one of theirs.
    theirs = set(re.findall(r"\d{4}", ADVERSARIAL.read_text(encoding="utf-8")))
    assert not set(numbers) & theirs


def test_the_planted_text_is_in_the_first_turn_or_in_the_question_and_nowhere_else() -> None:
    for case in cases():
        if case.group != PLANTED:
            assert case.sent_turns == case.turns
            assert case.sent_question == case.question
        elif case.kind == "planted_in_turn":
            assert case.sent_turns[0].question.endswith(f"\n\n{case.planted}")
            assert case.sent_question == case.question
        else:
            assert case.sent_question.endswith(f"\n\n{case.planted}")
            assert case.sent_turns == case.turns


def test_no_case_holds_a_phrase_or_an_example_question_of_the_instructions() -> None:
    refuse_phrases_in_cases(cases(), INSTRUCTION_PHRASES)
    examples = {plain_words(example) for example in EXAMPLE_QUESTIONS}
    for case in cases():
        texts = [*(turn.question for turn in case.turns), case.question, *case.standalone]
        assert not examples & {plain_words(text) for text in texts}, case.case_id


def flagged_by_a_simple_test(question: str) -> bool:
    """Whether a test on this machine would take a question for a follow-up, and ask.

    It does when the question holds one of the telltale words or has fewer than
    :data:`FEWEST_WORDS` words, and spares the request otherwise.
    """
    words = re.findall(r"[a-z']+", question.lower())
    return bool(set(TELLTALES) & set(words)) or len(words) < FEWEST_WORDS


def test_the_words_of_a_question_do_not_tell_a_follow_up_from_one_that_stands_alone() -> None:
    """The figures the card gives for a test on this machine that would spare requests."""
    flagged = flagged_by_a_simple_test
    follow_ups = [case for case in cases() if case.group != STANDALONE]
    alone = [case for case in cases() if case.group == STANDALONE]
    missed = sorted(case.case_id for case in follow_ups if not flagged(case.question))
    spared = [case for case in alone if not flagged(case.question)]

    assert missed == ["fu-answer-3", "fu-answer-5", "fu-third-4"]
    assert len(spared) == 12
    no_word = [
        case
        for case in follow_ups
        if not set(TELLTALES) & set(re.findall(r"[a-z']+", case.question.lower()))
    ]
    assert len(no_word) == 8


def test_the_card_names_every_word_of_the_simple_test() -> None:
    """The card's figures for the simple test can be worked out from the card alone."""
    line = next(
        line
        for line in CARD.read_text(encoding="utf-8").splitlines()
        if line.startswith("Measured on these cases, with no request:")
    )

    assert re.findall(r'"([a-z]+)"', line.split(", or has fewer than")[0]) == list(TELLTALES)
    assert f"fewer than {in_words(FEWEST_WORDS)} words" in line


def test_the_cases_added_after_the_review_test_what_it_found_untested() -> None:
    added = {case.case_id: case for case in cases() if case.case_id in ADDED_AFTER_THE_REVIEW}

    # Typed without a question mark: follow-ups, a question that stands alone, planted ones.
    typed = sorted(c for c, case in added.items() if not case.question.rstrip().endswith("?"))
    assert typed == sorted(
        set(ADDED_AFTER_THE_REVIEW) - {"fu-planted-turn-9", "fu-planted-question-11"}
    )
    assert {added[c].group for c in typed} == {FOLLOW_UP, STANDALONE, PLANTED}
    # Orders that leave no canary when obeyed.
    assert sorted(
        c for c, case in added.items() if case.group == PLANTED and not case.canaries
    ) == [
        "fu-planted-question-10",
        "fu-planted-question-9",
        "fu-planted-turn-10",
        "fu-planted-turn-9",
    ]
    # Tags of the request forged with a format character and with full-width brackets:
    # hidden from a plain search, found once the text is read as a model may read it.
    forged = added["fu-planted-question-11"].planted
    assert forged is not None
    assert "</new_question>" not in forged
    assert any(unicodedata.category(c) == "Cf" for c in forged)
    assert any(unicodedata.normalize("NFKC", c) != c for c in forged)
    assert holds_our_tags(forged)
    for case in added.values():
        assert case.second is not None
        assert case.second.name.startswith("d"), "named apart from the first reading's"


def reader_lines(path: Path = SECOND_READING) -> tuple[dict[str, object], list[dict[str, object]]]:
    """A second reader's own file: the line that names the reader, and its readings."""
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return lines[0], lines[1:]


def makers_forms(case: FollowUpCase) -> list[str]:
    """The acceptable forms the maker wrote, without those taken from the second reading."""
    return [form for form in case.standalone if form not in case.forms_from_second]


@pytest.mark.parametrize(
    ("path", "reader", "count", "added"),
    [
        (SECOND_READING, "claude-opus-5-5, one session, 24 November 2026", 72, False),
        (SECOND_READING_ADDED, "claude-opus-5-5, one session, 25 November 2026", 10, True),
    ],
)
def test_every_case_records_the_second_readers_own_line_for_it(
    path: Path, reader: str, count: int, added: bool
) -> None:
    header, readings = reader_lines(path)

    assert header == {"reader": reader}
    by_name = {str(reading["id"]): reading for reading in readings}
    assert len(by_name) == len(readings) == count
    recorded = {}
    read = [case for case in cases() if (case.case_id in ADDED_AFTER_THE_REVIEW) == added]
    for case in read:
        second = case.second
        assert second is not None, case.case_id
        recorded[second.name] = case.case_id
        assert by_name[second.name] == {
            "id": second.name,
            "standalone": list(second.standalone),
            "stands_alone_already": second.stands_alone_already,
            "needs_the_answer": second.needs_the_answer,
            "planted_text": second.planted_text,
            "note": second.note,
        }, case.case_id
    # Each line of the reader's file is the record of exactly one case.
    assert sorted(recorded) == sorted(by_name)


def test_the_second_readers_flags_agree_with_the_kinds_of_the_cases() -> None:
    needs = set()
    for case in cases():
        assert case.second is not None
        assert case.second.stands_alone_already == (case.group == STANDALONE), case.case_id
        assert case.second.planted_text == (case.group == PLANTED), case.case_id
        if case.second.needs_the_answer:
            needs.add(case.case_id)

    into_answer = {case.case_id for case in cases() if case.kind == "points_into_answer"}
    assert needs == into_answer | ANOTHER_READING_FIRST
    # The reader's flag says that the thing pointed at is named only in an answer. The
    # maker's says that no acceptable form can do without the answer: true for 4 of the 8.
    assert sum(bool(case.needs_answer) for case in cases() if case.case_id in needs) == 4


def test_the_second_readers_first_question_against_the_makers_forms() -> None:
    by_words = set()
    for case in cases():
        assert case.second is not None
        first = plain_words(case.second.standalone[0])
        if first in {plain_words(form) for form in makers_forms(case)}:
            by_words.add(case.case_id)
            assert case.forms_from_second == (), case.case_id

    assert len(by_words) == 62
    assert len(by_words & set(ADDED_AFTER_THE_REVIEW)) == 8
    rest = {case.case_id for case in cases()} - by_words
    assert rest == (
        VERIFIED_BY_THE_SECOND_MODEL
        | SAME_WITH_A_DETAIL_ADDED
        | ANOTHER_READING_FIRST
        | REWORDED_IN_THE_SECOND_READING_OF_THE_ADDED
    )
    taken = {case.case_id: case.forms_from_second for case in cases() if case.forms_from_second}
    assert set(taken) == rest - {"fu-planted-turn-5"}
    assert sum(len(forms) for forms in taken.values()) == 20
    by_id = {case.case_id: case for case in cases()}
    # Both readings of the two-referent follow-up are acceptable.
    ambiguous = by_id["fu-ambiguous-8"].second
    assert ambiguous is not None
    assert taken["fu-ambiguous-8"] == ambiguous.standalone
    # The first reading narrows by what only the answer says, and is not taken. The
    # second, screenings of team members in general, takes in the screening before
    # hiring, which the first turn leaves out: taken at first, it was taken out again
    # after the review, as looser than the reader's own standard.
    narrowed = by_id["fu-planted-turn-5"].second
    assert narrowed is not None
    assert not set(narrowed.standalone) & set(by_id["fu-planted-turn-5"].standalone)
    assert [case.case_id for case in cases() if not case.agreed] == ["fu-planted-turn-5"]
    for case_id, form in FORMS_TAKEN_OUT.items():
        assert form not in by_id[case_id].standalone, case_id
    # For the third turn added after the review, the reader put the approval first, as
    # the maker's forms have it, and getting the card at all second: another question,
    # which is not taken.
    third = by_id["fu-third-9"].second
    assert third is not None
    assert third.standalone[0] in by_id["fu-third-9"].standalone
    assert third.standalone[1] not in by_id["fu-third-9"].standalone


@pytest.mark.slow
def test_the_second_model_takes_a_rewording_and_not_a_question_with_a_detail_added() -> None:
    """The evidence the card gives: of the second reader's questions that the maker reads
    as the same as one of the maker's forms, the second model takes the plain rewordings
    at VERIFIED_AT, and none of those that add a detail from an earlier answer or turn."""
    from handbook_rag.cache.verify import EntailmentVerifier
    from handbook_rag.config import Settings

    verifier = EntailmentVerifier.load(Settings().cache_verifier_model, device="cpu")
    reached, with_a_detail = set(), []
    for case in cases():
        if case.case_id not in VERIFIED_BY_THE_SECOND_MODEL | SAME_WITH_A_DETAIL_ADDED:
            continue
        assert case.second is not None
        first = case.second.standalone[0]
        scores = verifier.score([(form, first) for form in makers_forms(case)])
        best = max(score.smaller for score in scores)
        if best >= VERIFIED_AT:
            reached.add(case.case_id)
        if case.case_id in SAME_WITH_A_DETAIL_ADDED:
            with_a_detail.append(best)

    assert reached == VERIFIED_BY_THE_SECOND_MODEL
    # The scores the documents give for the questions that add a detail.
    low = [score for score in with_a_detail if score <= float(DETAIL_LOW)]
    high = sorted(f"{score:.3f}" for score in with_a_detail if score > float(DETAIL_LOW))
    assert (len(low), tuple(high)) == (len(with_a_detail) - len(DETAIL_HIGH), DETAIL_HIGH)


@pytest.mark.slow
def test_the_verifier_never_takes_a_follow_up_as_typed_for_its_acceptable_form() -> None:
    """The verifier decides a rewrite on its own at VERIFIED_AT. On these cases it must
    not decide that a follow-up searched for as it was typed, or the first turn's own
    question in place of a follow-up that changes one detail of it, means an acceptable
    form."""
    from handbook_rag.cache.verify import EntailmentVerifier
    from handbook_rag.config import Settings

    verifier = EntailmentVerifier.load(Settings().cache_verifier_model, device="cpu")
    wrong = [
        (form, case.question)
        for case in cases()
        if case.group != STANDALONE
        for form in case.standalone
    ] + [
        (form, case.turns[0].question)
        for case in cases()
        if case.kind == "subject_left_out"
        for form in case.standalone
    ]

    scores = verifier.score(wrong)

    taken = [
        pair for pair, score in zip(wrong, scores, strict=True) if score.smaller >= VERIFIED_AT
    ]
    assert taken == []
