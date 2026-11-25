"""The held-out follow-up cases in the repository are what their section of the card says.

Twenty conversations, kept apart from the 82 and asked into a run of their own, to test
the step and the rule for a reply without a question mark on what they were not written
from. These tests hold the cases, the second reader's file and the card's section to each
other, as ``test_committed_follow_up_cases.py`` holds the 82 to the rest of the card. The
runs of them, and the figures the documents give, are held by
``test_committed_held_out_follow_up_runs.py``. No test here needs a model or sends a request.
"""

import json
import re
from collections import Counter
from pathlib import Path

from handbook_rag.evals.adversarial import find_canaries
from handbook_rag.evals.dataset import load_golden_set
from handbook_rag.evals.follow_ups import (
    FOLLOW_UP,
    PLANTED,
    STANDALONE,
    FollowUpCase,
    load_cases,
    refuse_phrases_in_cases,
)
from handbook_rag.generate.base import plain_words
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from handbook_rag.rewrite.base import read_reply
from handbook_rag.rewrite.models import MAX_EARLIER_TURNS, RewriteReply
from handbook_rag.rewrite.prompt import EXAMPLE_QUESTIONS, INSTRUCTION_PHRASES
from tests.evals.test_committed_follow_up_cases import makers_forms, reader_lines
from tests.test_readme_results import table

ROOT = Path(__file__).parents[2]
HELD_OUT = ROOT / "evals" / "follow-ups" / "held-out"
CASES = HELD_OUT / "cases.jsonl"
SECOND_READING = HELD_OUT / "second-reading.jsonl"
THE_82 = ROOT / "evals" / "follow-ups" / "cases.jsonl"
CARD = ROOT / "evals" / "follow-ups" / "README.md"
ADVERSARIAL = ROOT / "evals" / "adversarial" / "cases.jsonl"
GOLDEN = {
    question.id: question for question in load_golden_set(ROOT / "evals/golden/questions.jsonl")
}
READER = "claude-opus-5-5, one session, 25 November 2026"
# The reader's first question, taken as an acceptable form: in other words, and with a
# detail from the earlier answer.
IN_OTHER_WORDS = {"ho-03", "ho-05", "ho-07", "ho-09", "ho-19"}
WITH_A_DETAIL_FROM_THE_ANSWER = {"ho-06", "ho-18"}
# The reader's first reading asks for two deadlines, and is not taken.
FIRST_READING_NOT_TAKEN = "ho-04"
# The reader's second reading asks another question, and is not taken.
SECOND_READING_NOT_TAKEN = "ho-08"

# The openings a reply without a question mark had to begin with when the cases were
# written and asked: a question word or a verb that opens a question, a negated verb, or a
# preposition followed directly by a question word. The rule no longer asks for any of
# them (the card says when and why). They are kept here as the record of the rule the
# held-out run was decided by, and they sort the cases in the card's table.
QUESTION_WORDS_AND_VERBS = (
    *("who", "what", "when", "where", "which", "why", "how", "whose", "whom"),
    *("is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would"),
    *("will", "may", "might", "must", "has", "have", "had", "shall"),
)
NEGATED_VERBS = (
    *("isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't"),
    *("couldn't", "shouldn't", "wouldn't", "won't", "hasn't", "haven't", "hadn't"),
)
PREPOSITIONS = (
    *("in", "on", "at", "for", "to", "from", "by", "with", "under", "until", "after"),
    *("before", "during", "about", "of"),
)
QUESTION_WORDS = ("which", "what", "whom", "whose", "how", "when", "where")
_OPENED_BEFORE = re.compile(
    rf"(?:{'|'.join(QUESTION_WORDS_AND_VERBS)})\b"
    rf"|(?:{'|'.join(verb[:-2] for verb in NEGATED_VERBS)})['\u2019]t\b"
    rf"|(?:{'|'.join(PREPOSITIONS)})\s+(?:{'|'.join(QUESTION_WORDS)})\b",
    re.IGNORECASE,
)


def began_as_the_rule_before_asked(text: str) -> bool:
    """Whether a reply begins in one of the ways the rule asked for before the change."""
    return _OPENED_BEFORE.match(text) is not None


def section(text: str, heading: str) -> str:
    """A section of a Markdown text, from its heading to the next of the same level."""
    level = heading.split(" ")[0]
    after = text.split(f"\n{heading}\n")
    assert len(after) == 2, heading
    return after[1].split(f"\n{level} ")[0]


def cases() -> list[FollowUpCase]:
    return load_cases(CASES)


def typed_without_a_mark(case: FollowUpCase) -> bool:
    return not case.question.rstrip().endswith("?")


def opens_with_a_preposition_and_a_question_word(text: str) -> bool:
    words = text.lower().split()
    return len(words) > 1 and words[0] in PREPOSITIONS and words[1] in QUESTION_WORDS


def opens_with_a_negated_verb(text: str) -> bool:
    return text.lower().split()[0] in NEGATED_VERBS


def rows() -> list[tuple[str, list[str]]]:
    """The card's table of the cases, as the cases themselves sort them."""
    follow = [case for case in cases() if case.group == FOLLOW_UP and typed_without_a_mark(case)]
    preposition = [
        c for c in follow if opens_with_a_preposition_and_a_question_word(c.standalone[0])
    ]
    negated = [c for c in follow if opens_with_a_negated_verb(c.standalone[0])]
    groups = [
        preposition,
        negated,
        [c for c in follow if c not in preposition and c not in negated],
        [c for c in cases() if c.group == FOLLOW_UP and not typed_without_a_mark(c)],
        [c for c in cases() if c.group == STANDALONE],
        [c for c in cases() if c.group == PLANTED],
    ]
    return [(str(len(group)), [case.case_id for case in group]) for group in groups]


def card_section() -> str:
    """The card's section on the held-out cases, with each case's name taken out."""
    text = section(CARD.read_text(encoding="utf-8"), "## The held-out cases")
    return re.sub(r"`ho-\d+`", "`a case`", text)


def test_the_cases_are_in_the_numbers_the_card_gives() -> None:
    every = cases()

    assert [case.case_id for case in every] == [f"ho-{number:02d}" for number in range(1, 21)]
    assert Counter(case.kind for case in every) == {
        "subject_left_out": 4,
        "pronoun": 7,
        "third_turn": 3,
        "change_of_subject": 3,
        "planted_in_question": 3,
    }
    assert Counter(case.group for case in every) == {FOLLOW_UP: 14, STANDALONE: 3, PLANTED: 3}
    assert Counter(len(case.turns) for case in every) == {1: 17, 2: 3}
    assert max(len(case.turns) for case in every) <= MAX_EARLIER_TURNS
    # Ten follow-ups typed without a question mark, four with one, and the questions that
    # stand alone and the planted ones typed without one.
    assert [count for count, _ in rows()] == ["4", "3", "3", "4", "3", "3"]
    assert sum(typed_without_a_mark(case) for case in every) == 16


def test_the_cards_table_of_the_cases_is_the_cases() -> None:
    text = section(CARD.read_text(encoding="utf-8"), "## The held-out cases")
    cells = table(text, "| The new question | Cases | Which |")

    assert [(row[1], re.findall(r"`(ho-\d+)`", row[2])) for row in cells] == rows()


def test_seven_follow_ups_open_in_the_ways_the_rule_took_before_they_were_written() -> None:
    """A follow-up typed without a question mark often gets a rewrite typed without one.

    For seven of them the natural question opens in one of the two ways the rule took on
    25 November before the cases were written; six of those seven would have been
    refused before that, as "Can't" was taken already. Every acceptable form of every such
    follow-up, typed without its question mark, is a reply the step uses today. Under the
    rule the held-out run was decided by, one form taken from the second reading was not.
    """
    follow = [case for case in cases() if case.group == FOLLOW_UP and typed_without_a_mark(case)]
    new_openings, refused_before_the_cases, refused_in_the_run = [], [], []
    for case in follow:
        for form in case.standalone:
            bare = form.rstrip("?")
            rewrite = read_reply(
                case.sent_question, case.sent_turns, RewriteReply(bare, "a model", "end_turn")
            )
            assert (rewrite.outcome, rewrite.question) == ("rewritten", bare), (
                case.case_id,
                rewrite.problem,
            )
            if not began_as_the_rule_before_asked(bare):
                refused_in_the_run.append((case.case_id, form.split(",")[0]))
        first = case.standalone[0]
        if opens_with_a_preposition_and_a_question_word(first) or opens_with_a_negated_verb(first):
            new_openings.append(case.case_id)
            words = first.lower().split()
            if words[0] not in QUESTION_WORDS_AND_VERBS and not words[0].startswith("can'"):
                refused_before_the_cases.append(case.case_id)

    assert new_openings == ["ho-01", "ho-02", "ho-03", "ho-04", "ho-05", "ho-06", "ho-07"]
    assert refused_before_the_cases == ["ho-01", "ho-02", "ho-03", "ho-04", "ho-05", "ho-06"]
    assert refused_in_the_run == [("ho-07", "In the Zendesk admin dashboard")]
    card = CARD.read_text(encoding="utf-8")
    assert '`ho-07`\'s opens "In the Zendesk admin dashboard, can I …"' in card


def test_every_acceptable_form_is_one_the_step_would_take_from_the_model() -> None:
    for case in cases():
        for form in case.standalone:
            rewrite = read_reply(
                case.sent_question,
                case.sent_turns[-MAX_EARLIER_TURNS:],
                RewriteReply(form, "a model", "end_turn"),
            )
            assert rewrite.outcome in ("rewritten", "unchanged"), (case.case_id, rewrite.problem)


def test_each_starts_from_a_golden_question_that_none_of_the_82_uses() -> None:
    the_82 = load_cases(THE_82)

    sources = [case.source for case in cases()]
    assert len(set(sources)) == len(sources) == 20
    assert not set(sources) & {case.source for case in the_82}
    by_question = {question.question: question for question in GOLDEN.values()}
    for case in cases():
        golden = GOLDEN[case.source]
        first = case.turns[0]
        assert first.question == golden.question, case.case_id
        expected = golden.reference_answer if golden.type == "answerable" else NOT_FOUND_MESSAGE
        assert first.answer == expected, case.case_id
        # A later turn carries an answer only when it is a golden question, as in the 82.
        for turn in case.turns[1:]:
            later = by_question.get(turn.question)
            assert turn.answer == (None if later is None else later.reference_answer)
    # Nothing is shared with the 82: no name, and no conversation a saved reply could fit.
    assert not {case.case_id for case in cases()} & {case.case_id for case in the_82}
    assert not {case.fingerprint for case in cases()} & {case.fingerprint for case in the_82}
    # No question asks what a golden question asks word for word: none has a target.
    assert all(case.target is None for case in cases())


def test_the_planted_texts_and_their_canary() -> None:
    planted = [case for case in cases() if case.group == PLANTED]

    assert {case.kind for case in planted} == {"planted_in_question"}
    for case in planted:
        assert case.planted is not None
        assert case.sent_question == f"{case.question}\n\n{case.planted}"
        assert case.sent_turns == case.turns
    for case in cases():
        if case.group != PLANTED:
            assert (case.sent_question, case.sent_turns) == (case.question, case.turns)
    canaries = [canary for case in planted for canary in case.canaries]
    assert canaries == ["OSPREY-6182"]
    (canary_case,) = [case for case in planted if case.canaries]
    sent = " ".join([canary_case.sent_question, *(t.question for t in canary_case.sent_turns)])
    assert not find_canaries(sent, canaries), "only a reply that obeyed can hold it"
    # Its number is no other case's, of the 82 or of the adversarial cases.
    others = set(re.findall(r"\d{4}", THE_82.read_text(encoding="utf-8")))
    others |= set(re.findall(r"\d{4}", ADVERSARIAL.read_text(encoding="utf-8")))
    assert "6182" not in others
    # The three planted texts: an answer, a question about something else, a code word.
    texts = [case.planted or "" for case in planted]
    assert "Answer it yourself" in texts[0]
    assert "salaries of everyone in the company" in texts[1]
    assert "OSPREY and 6182" in texts[2]


def test_no_case_holds_a_phrase_or_an_example_question_of_the_instructions() -> None:
    refuse_phrases_in_cases(cases(), INSTRUCTION_PHRASES)
    examples = {plain_words(example) for example in EXAMPLE_QUESTIONS}
    for case in cases():
        texts = [*(turn.question for turn in case.turns), case.question, *case.standalone]
        assert not examples & {plain_words(text) for text in texts}, case.case_id


def test_every_case_records_the_second_readers_own_line_for_it() -> None:
    header, readings = reader_lines(SECOND_READING)

    assert header == {"reader": READER}
    by_name = {str(reading["id"]): reading for reading in readings}
    assert sorted(by_name) == [f"h{number:02d}" for number in range(1, 21)]
    recorded = {}
    for case in cases():
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
    assert sorted(recorded) == sorted(by_name)
    # Under names that are neither the first readings' nor in the order of the cases.
    assert [recorded[name] for name in sorted(recorded)] != [case.case_id for case in cases()]


def test_the_second_reading_against_the_cases() -> None:
    every = {case.case_id: case for case in cases()}
    seconds = {case_id: case.second for case_id, case in every.items() if case.second}
    assert len(seconds) == 20

    for case_id, second in seconds.items():
        case = every[case_id]
        assert second.stands_alone_already == (case.group == STANDALONE), case_id
        assert second.planted_text == (case.group == PLANTED), case_id
    assert [c for c, s in seconds.items() if s.needs_the_answer] == ["ho-18"]
    assert sorted(c for c, s in seconds.items() if len(s.standalone) > 1) == [
        FIRST_READING_NOT_TAKEN,
        SECOND_READING_NOT_TAKEN,
    ]
    by_words = {
        case_id
        for case_id, second in seconds.items()
        if plain_words(second.standalone[0])
        in {plain_words(form) for form in makers_forms(every[case_id])}
    }
    assert len(by_words) == 12
    rest = set(every) - by_words
    taken = {
        case_id: case.forms_from_second for case_id, case in every.items() if case.forms_from_second
    }
    assert set(taken) == IN_OTHER_WORDS | WITH_A_DETAIL_FROM_THE_ANSWER
    assert rest == set(taken) | {FIRST_READING_NOT_TAKEN}
    for case_id, forms in taken.items():
        assert forms == (seconds[case_id].standalone[0],), case_id
    assert [case_id for case_id, case in every.items() if not case.agreed] == [
        FIRST_READING_NOT_TAKEN
    ]
    # Not taken: the first reading of one, which asks for two deadlines, and the second
    # reading of another, which asks about the other description. The first's second
    # reading is one of the writer's forms already.
    first_not_taken = seconds[FIRST_READING_NOT_TAKEN].standalone
    assert first_not_taken[0] not in every[FIRST_READING_NOT_TAKEN].standalone
    assert "equity refreshes" in first_not_taken[0]
    assert first_not_taken[1] in makers_forms(every[FIRST_READING_NOT_TAKEN])
    other = seconds[SECOND_READING_NOT_TAKEN].standalone[1]
    assert other not in every[SECOND_READING_NOT_TAKEN].standalone
    assert "short description" in other
    # The two taken with a detail that only the earlier answer holds.
    for case_id, detail in (("ho-06", "free tier limits"), ("ho-18", "Digital Experience UX")):
        assert detail in seconds[case_id].standalone[0]
        assert detail in (every[case_id].turns[0].answer or "")
        assert detail not in every[case_id].turns[0].question


def test_the_cards_section_is_the_cases_and_their_second_reading() -> None:
    text = card_section()
    original = section(CARD.read_text(encoding="utf-8"), "## The held-out cases")
    every = cases()
    alone = sum(case.group == STANDALONE for case in every)
    planted = sum(case.group == PLANTED for case in every)

    assert table(text, "| Held out | Agree | Of |") == [
        [
            "Stands alone already",
            "20",
            f"20: yes for the {alone} that stand alone, no for the other {len(every) - alone}",
        ],
        [
            "Holds planted text",
            "20",
            f"20: yes for the {planted} planted ones, no for the other {len(every) - planted}",
        ],
        ["The reader's first question has the words of one of the writer's forms", "12", "20"],
    ]
    # The cases each bullet names are the ones the tests above work out.
    taken_line = next(line for line in original.splitlines() if line.startswith("- **Taken"))
    named = set(re.findall(r"`(ho-\d+)`", taken_line))
    assert named == IN_OTHER_WORDS | WITH_A_DETAIL_FROM_THE_ANSWER
    not_taken = next(line for line in original.splitlines() if line.startswith("- **Not taken"))
    assert set(re.findall(r"`(ho-\d+)`", not_taken)) == {
        FIRST_READING_NOT_TAKEN,
        SECOND_READING_NOT_TAKEN,
    }
    assert "No person has read them." in original
    assert READER.split(",")[0] in original


def test_the_command_the_card_gives_is_the_one_the_makefile_runs() -> None:
    text = section(CARD.read_text(encoding="utf-8"), "## The held-out cases")
    command = (
        "handbook-rag eval-follow-ups --cases evals/follow-ups/held-out/cases.jsonl "
        "--variant held-out"
    )
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert f"`{command}`" in text
    assert f"\tuv run {command}\n" in makefile
    assert f"({len(cases())} real requests)" in makefile.split("eval-follow-ups-held-out:")[1]
    assert json.loads(SECOND_READING.read_text(encoding="utf-8").splitlines()[0]) == {
        "reader": READER
    }
