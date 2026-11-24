"""What an answer cache would hold after a saved answer run, by the cache's own rules.

The measurement of the cache can fill its simulated cache with the replies a model really
wrote. These tests write a small answer run to disk and check how its replies are sorted,
and that a real cache, handed the same replies, decides the same.
"""

import hashlib
import json
from pathlib import Path

import pytest

from handbook_rag.cache import rules
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.evals.answers import (
    ANSWERS_FILE,
    AnswerRecord,
    Passage,
    Statement,
    append_record,
)
from handbook_rag.evals.cache_replies import (
    StoredReplies,
    as_the_cache_is_handed_it,
    stored_replies,
)
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.evals.stage import SavedRunError
from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE
from tests.cache.support import memory_answers
from tests.evals.test_cache_pairs import half_answered
from tests.support import FakeEmbedder

# g0 to g5 are questions the handbook answers, by their label, and g6 to g11 are not.
GOLDEN = {question.id: question for question in half_answered()[0]}
CITED = Statement("Ask your manager, who approves it within a week.", (1,))


def reply(
    question_id: str,
    *statements: Statement,
    answered: bool = True,
    stop_reason: str = "end_turn",
    fingerprint: str | None = None,
) -> AnswerRecord:
    """A saved reply to one golden question, as an answer run writes it."""
    return AnswerRecord(
        question_id=question_id,
        fingerprint=GOLDEN[question_id].fingerprint if fingerprint is None else fingerprint,
        passages=(
            Passage(1, "people/approvals.md#0", "people/approvals.md", "Approvals", ()),
            Passage(2, "finance/expenses.md#3", "finance/expenses.md", "Expenses", ()),
        ),
        statements=statements,
        answered=answered,
        stop_reason=stop_reason,
        model="answer-model",
        input_tokens=1800,
        output_tokens=40,
        seconds=2.0,
    )


DECLINED = (Statement(NOT_FOUND_MESSAGE, ()),)
# One reply of every sort, and three golden questions without a usable one.
REPLIES = [
    # An answer that cites a passage.
    reply("g0", CITED),
    # The handbook answers it, by the label, and the model declined all the same.
    reply("g1", *DECLINED, answered=False),
    # Part of an answer, and the fixed sentence for the rest.
    reply("g2", CITED, Statement(f" {NOT_FOUND_MESSAGE}", ())),
    # Part of an answer, and the gap said in the model's own words.
    reply("g3", CITED, Statement(" The amount is not provided in the handbook.", ())),
    # An answer that rests on no passage.
    reply("g4", Statement("Ask your manager.", ())),
    # Cut off at the length limit.
    reply("g5", CITED, stop_reason="max_tokens"),
    reply("g6", *DECLINED, answered=False),
    # The handbook does not answer it, by the label, and the model answered all the same.
    reply("g7", CITED),
    # A reply to the question as it was labelled before.
    reply("g8", CITED, fingerprint="0" * 16),
    # Nothing was written.
    reply("g9"),
]
HELD = {
    "g0": None,
    "g1": rules.DECLINE,
    "g2": rules.DECLINES_IN_PART,
    "g3": rules.SAYS_SOMETHING_IS_MISSING,
    "g4": rules.CITES_NOTHING,
    "g6": rules.DECLINE,
    "g7": None,
}


def a_run(run_dir: Path, records: list[AnswerRecord] = REPLIES) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        append_record(run_dir, record)
    return run_dir


def test_the_replies_of_a_run_are_sorted_by_what_the_cache_would_do_with_each(
    tmp_path: Path,
) -> None:
    golden = list(GOLDEN.values())

    replies = stored_replies(golden, a_run(tmp_path / "answers" / "a-run"))

    assert replies.held == HELD
    assert replies.not_kept == ("g5", "g9")
    assert replies.without_a_reply == ("g8", "g10", "g11")
    assert replies.may_be_candidates == {
        "g0": True,
        "g1": False,
        "g2": False,
        "g3": False,
        "g4": False,
        "g6": False,
        "g7": True,
    }
    assert replies.source == (tmp_path / "answers" / "a-run").as_posix()
    figures = replies.figures(golden)
    assert "not by the golden set's label" in figures.pop("what")
    assert list(figures["kept_from_the_second_tier"]) == list(rules.REASONS)
    assert figures == {
        "from": replies.source,
        "fingerprint": replies.fingerprint,
        "kept": 7,
        "not_kept": ["g5", "g9"],
        "without_a_reply": ["g8", "g10", "g11"],
        "candidates": 2,
        "kept_from_the_second_tier": {
            rules.DECLINE: ["g1", "g6"],
            rules.DECLINES_IN_PART: ["g2"],
            rules.SAYS_SOMETHING_IS_MISSING: ["g3"],
            rules.CITES_NOTHING: ["g4"],
        },
        # Where the reply and the label part ways.
        "candidates_the_handbook_does_not_answer": ["g7"],
        "no_candidates_though_the_handbook_answers": ["g1", "g2", "g3", "g4"],
    }
    # A saved measurement keeps them as JSON.
    assert json.loads(json.dumps(figures)) == figures


def test_a_real_cache_handed_the_same_replies_keeps_and_serves_the_same(tmp_path: Path) -> None:
    replies = stored_replies(list(GOLDEN.values()), a_run(tmp_path / "a-run"))
    store = memory_answers()
    cache = AnswerCache(store, embedder=FakeEmbedder(), threshold=0.9)

    for record in REPLIES:
        question = GOLDEN[record.question_id]
        cache.store(cache.lookup(question.question), as_the_cache_is_handed_it(question, record))

    kept = {}
    for question in GOLDEN.values():
        entry = store.get(rules.entry_id(rules.comparable(question.question), rules.scope_of(None)))
        if entry is not None:
            kept[question.id] = entry.by_meaning
    # g8 is kept by the real cache, which knows nothing of labels: what the measurement
    # leaves out there is a reply to another test, not a reply the cache would refuse.
    assert kept.pop("g8") is True
    assert kept == replies.may_be_candidates
    # And the store counts what it holds the same way, with g8 beside the others.
    held = store.held()
    assert held.answers == len(replies.held) + 1
    assert held.by_meaning == sum(replies.may_be_candidates.values()) + 1
    assert held.declines == len(
        replies.figures(list(GOLDEN.values()))["kept_from_the_second_tier"][rules.DECLINE]
    )


def test_the_latest_reply_to_a_question_is_the_one_that_counts(tmp_path: Path) -> None:
    golden = list(GOLDEN.values())
    first = stored_replies(
        golden, a_run(tmp_path / "a-run", [reply("g0", *DECLINED, answered=False)])
    )
    assert first.held == {"g0": rules.DECLINE}

    # The run asked the question again, and this time the model answered.
    append_record(tmp_path / "a-run", reply("g0", CITED))
    again = stored_replies(golden, tmp_path / "a-run")

    assert again.held == {"g0": None}
    # The run's file is named by its content, so the two measurements name two files.
    answers = (tmp_path / "a-run" / ANSWERS_FILE).read_bytes()
    assert again.fingerprint == hashlib.sha256(answers).hexdigest()[:16]
    assert first.fingerprint != again.fingerprint
    assert stored_replies(golden, tmp_path / "a-run").fingerprint == again.fingerprint


def test_a_saved_reply_is_handed_over_as_the_pipeline_hands_an_answer_to_the_cache() -> None:
    # It cites a passage the model was given and one it was not, which names nothing.
    record = reply("g0", Statement("Ask your manager.", (2, 7)), Statement(" Then wait.", ()))

    answer = as_the_cache_is_handed_it(GOLDEN["g0"], record)

    assert answer.question == GOLDEN["g0"].question
    assert answer.text == "Ask your manager. Then wait."
    assert [chunk.chunk_id for chunk in answer.sources] == ["finance/expenses.md#3"]
    assert [chunk.doc_id for chunk in answer.sources] == ["finance/expenses.md"]
    assert (answer.answered, answer.model, answer.stop_reason) == (True, "answer-model", "end_turn")
    # An answer run asks once and has no fallback model.
    assert (answer.requests, answer.fallback, answer.cached) == (1, False, None)
    assert rules.storable(answer)
    assert rules.kept_from_the_second_tier(answer) is None


def test_a_run_without_usable_replies_is_refused(tmp_path: Path) -> None:
    golden = list(GOLDEN.values())

    with pytest.raises(DatasetError, match="saved replies not found"):
        stored_replies(golden, tmp_path / "no-such-run")
    # Every reply of this run is one the cache would not keep.
    cut_off = a_run(tmp_path / "cut-off", [reply("g0", CITED, stop_reason="max_tokens")])
    with pytest.raises(DatasetError, match="holds no reply that an answer cache would keep"):
        stored_replies(golden, cut_off)
    # And this one holds no reply to a golden question at all.
    other = a_run(tmp_path / "other", [reply("g0", CITED, fingerprint="0" * 16)])
    with pytest.raises(DatasetError, match="holds no reply that an answer cache would keep"):
        stored_replies(golden, other)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / ANSWERS_FILE).write_text('{"question_id": "g0"}\n', encoding="utf-8")
    with pytest.raises(SavedRunError, match="is not a saved reply"):
        stored_replies(golden, broken)


def test_what_was_held_can_be_given_without_a_run_on_disk() -> None:
    # The figures are of the replies alone, so a test of the measurement can state them.
    golden: list[GoldenQuestion] = list(GOLDEN.values())
    replies = StoredReplies(
        source="evals/runs/answers/a-run",
        fingerprint="0123456789abcdef",
        held={"g0": None, "g6": rules.DECLINE},
        not_kept=(),
        without_a_reply=tuple(f"g{number}" for number in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)),
    )

    figures = replies.figures(golden)

    assert (figures["kept"], figures["candidates"]) == (2, 1)
    assert figures["kept_from_the_second_tier"][rules.DECLINE] == ["g6"]
    assert figures["candidates_the_handbook_does_not_answer"] == []
    assert figures["no_candidates_though_the_handbook_answers"] == []
