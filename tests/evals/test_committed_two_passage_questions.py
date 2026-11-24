"""The two-passage questions in the repository are what their card says they are.

The loader and the check of ``golden-check --reasons`` hold the files to their shape.
These tests hold them to the rest of the card: every count it gives about the questions,
every figure it gives about the second reading, which is counted from the readers' own
lines, and every figure it gives about the saved runs, which is read from the runs' files.

Three files are tied together here. ``questions.jsonl`` says by its ``review`` which
questions were confirmed, ``why-two.jsonl`` holds the labels of the second reading for
each question, and ``second-reading.jsonl`` holds the lines the readers wrote. A label
changed in one of them and not in the others fails.

What no test can hold is the claim the set rests on: that each answer needs both
passages. That is a matter of reading, and the card says who has read the questions.
Nor can a test show what a reader was given: the forms are not in the repository.

Only the last test needs the corpus. It builds it from the handbook snapshot, as the
commands do, so it is marked slow and is skipped where there is no snapshot, as in CI.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.config import Settings
from handbook_rag.evals.compare import compare_runs
from handbook_rag.evals.dataset import GoldenQuestion, load_golden_set
from handbook_rag.evals.golden import check_against_corpus
from handbook_rag.evals.metrics import MIN_COVERAGE
from handbook_rag.evals.retrieval import ALL_CUTOFFS, RETRIEVAL_DEPTH
from handbook_rag.evals.two_passage import (
    KINDS,
    OF_A_QUESTION,
    ReadingLine,
    WhyTwo,
    check_two_passage_set,
    kinds_of,
    load_reasons,
    load_second_reading,
    reading_of,
    second_reading_of,
)
from handbook_rag.ingest.build import build_corpus
from tests.evals.test_committed_search_variants import sign_test
from tests.test_readme_results import table

ROOT = Path(__file__).parents[2]
QUESTIONS = ROOT / "evals" / "two-passage" / "questions.jsonl"
REASONS = ROOT / "evals" / "two-passage" / "why-two.jsonl"
READING = ROOT / "evals" / "two-passage" / "second-reading.jsonl"
CARD = ROOT / "evals" / "two-passage" / "README.md"
RUNS = ROOT / "evals" / "runs" / "two-passage"
GOLDEN = ROOT / "evals" / "golden" / "questions.jsonl"
GOLDEN_HYBRID = ROOT / "evals" / "runs" / "retrieval" / "v2"
SNAPSHOT = ROOT / "data" / "handbook"
MODES = ("hybrid", "dense", "sparse")
DISPUTED = "t025"
# Wording that would point at the handbook as a source, which no asker would use.
BANNED = ("handbook", "according to", "as per", "the passage", "this section", "the page")
# The showings in which the passage that was held back stood on the same form, under
# another question that shares it: the item, its question and what it showed.
GIVEN_AWAY_ELSEWHERE = {
    "a-05": ("t021", "second"),
    "a-30": ("t020", "first"),
    "b-11": ("t013", "first"),
    "b-34": ("t014", "second"),
    "b-35": ("t020", "second"),
    "c-28": ("t013", "second"),
    "c-29": ("t016", "second"),
    "c-37": ("t022", "second"),
}


def questions() -> list[GoldenQuestion]:
    """Every question of the file, the disputed one among them."""
    return load_golden_set(QUESTIONS)


def reasons() -> dict[str, WhyTwo]:
    return load_reasons(REASONS)


def confirmed() -> list[GoldenQuestion]:
    """The questions the second reading confirmed: the ones every figure is of."""
    why = reasons()
    return [question for question in questions() if why[question.id].confirmed]


def reading() -> list[ReadingLine]:
    return load_second_reading(READING)


def of_question(question_id: str) -> dict[str, ReadingLine]:
    """The three lines of one question, by what the item showed."""
    return {line.shown: line for line in reading() if line.question_id == question_id}


def card() -> str:
    return CARD.read_text(encoding="utf-8")


def lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def rows(mode: str) -> dict[str, dict[str, Any]]:
    """The rows of a saved run, by question id."""
    return {row["prompt_id"]: row for row in lines(RUNS / mode / "results.jsonl")}


def summary(mode: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((RUNS / mode / "summary.json").read_text(encoding="utf-8"))
    return loaded


def section(doc_id: str) -> str:
    return doc_id.split("/")[0]


def numbers(cells: list[str]) -> list[int]:
    return [int(cell) for cell in cells]


def split(ranks: list[list[int | None]], k: int) -> dict[str, int]:
    """The questions with all, some or none of their ranks within the first k."""
    counts = {"all": 0, "some": 0, "none": 0}
    for ranks_of_question in ranks:
        found = sum(rank is not None and rank <= k for rank in ranks_of_question)
        if found == len(ranks_of_question):
            counts["all"] += 1
        else:
            counts["some" if found else "none"] += 1
    return counts


# The questions and their reasons


def test_the_file_holds_34_questions_in_the_golden_format_with_a_reason_for_each() -> None:
    found = questions()
    fields = ["id", "question", "reference_answer", "evidence", "type", "origin", "review"]

    assert [question.id for question in found] == [f"t{n:03d}" for n in range(1, 35)]
    assert check_two_passage_set(found, reasons()) == []
    for row in lines(QUESTIONS):
        # Only the disputed question has a note.
        expected = [*fields, "note"] if row["id"] == DISPUTED else fields
        assert list(row) == expected, row["id"]
        assert [list(piece) for piece in row["evidence"]] == [["doc_id", "quote"]] * 2, row["id"]
    for row in lines(REASONS):
        assert list(row) == [
            "id",
            "kind",
            "only_first",
            "only_second",
            "second_reading",
            "confirmed",
        ], row["id"]
        assert list(row["second_reading"]) == list(OF_A_QUESTION), row["id"]
    assert [row["id"] for row in lines(REASONS)] == [question.id for question in found]
    assert "holds 34 of them" in card()


def test_the_review_of_each_question_is_what_the_second_reading_found() -> None:
    found, why = questions(), reasons()

    assert Counter(question.review for question in found) == {"checked": 33, "pending": 1}
    assert [question.id for question in found if question.review == "pending"] == [DISPUTED]
    assert [question.id for question in found if not why[question.id].confirmed] == [DISPUTED]
    assert reading_of(found, why) == {"confirmed": 33, "disputed": 1}
    assert {question.origin for question in found} == {"model-drafted"}
    (disputed,) = [question for question in found if question.id == DISPUTED]
    assert disputed.note.startswith("Disputed in the second reading.")
    assert all(not question.note for question in confirmed())
    text = card()
    assert (
        "**Status: written by one model session, and read again by three reader sessions of "
        "the same model that did not see its notes. Not read by a person.**" in text
    )
    assert "They confirm that 33 of the 34 questions need both passages." in text
    assert 'Those 33 are marked `"review": "checked"`.' in text
    assert 'The one they do not confirm, `t025`, stays `"review": "pending"`' in text
    assert "a model's check is not a person's review" in text
    assert "The readers confirm 33 of the 34, and every figure in this card is of those 33." in text


def test_the_kinds_are_in_the_numbers_the_card_gives() -> None:
    made, kept = kinds_of(questions(), reasons()), kinds_of(confirmed(), reasons())
    found = {row[0].strip("`"): numbers(row[2:]) for row in table(card(), "| Kind | What the two")}

    assert found == {**{kind: [made[kind], kept[kind]] for kind in made}, "All": [34, 33]}
    assert made == {"comparison": 12, "term-and-rule": 8, "rule-and-exception": 7, "process": 7}
    assert kept == {"comparison": 12, "term-and-rule": 8, "rule-and-exception": 6, "process": 7}
    assert set(made) == set(KINDS)
    assert reasons()[DISPUTED].kind == "rule-and-exception"
    assert "**The disputed question, `t025`**, is a `rule-and-exception` question" in card()
    # Kept together in the file, kind by kind, in the order of the card's table.
    order = [reasons()[question.id].kind for question in questions()]
    assert order == [kind for kind in found if kind != "All" for _ in range(made[kind])]


def test_the_pages_and_sections_are_the_ones_the_card_counts() -> None:
    found = confirmed()
    pages = Counter(piece.doc_id for question in found for piece in question.evidence)
    sections = Counter(section(doc_id) for doc_id in pages.elements())
    across = sum(len({section(piece.doc_id) for piece in q.evidence}) == 2 for q in found)

    assert (len(pages), len(sections), sum(pages.values())) == (56, 18, 66)
    assert "**56 pages from 18 handbook sections** hold the 66 pieces." in card()
    assert sum(count > 1 for count in pages.values()) == 8
    assert "8 pages are used by more than one question" in card()
    assert (across, len(found) - across) == (16, 17)
    assert "**In 16 questions the two pages are in two different sections**" in card()
    assert "In the other 17 they are two pages of one section." in card()
    assert (sections["security"], sections["total-rewards"]) == (16, 15)
    assert (
        "16 of the 66 pieces of evidence are on security pages and 15 on pages of total" in card()
    )
    used_three_times = sorted(doc_id.split("/")[-1] for doc_id, n in pages.items() if n == 3)
    assert used_three_times == ["bv-benefits-netherlands.md", "critical-systems.md"]
    assert max(pages.values()) == 3
    # The issue asks for at least eight sections.
    assert len({question.section for question in found}) == 15


def test_every_question_and_quote_follows_the_drafting_guidelines() -> None:
    found = confirmed()
    lengths = [len(question.question.split()) for question in found]
    quotes = [len(piece.quote) for question in found for piece in question.evidence]

    for question in questions():
        assert question.question.endswith("?"), question.id
        assert not any(phrase in question.question.lower() for phrase in BANNED), question.id
        for piece in question.evidence:
            assert 40 <= len(piece.quote) <= 300, question.id
            assert piece.quote == piece.quote.strip(), question.id
            assert piece.doc_id.endswith(".md"), question.id
    assert (min(lengths), statistics.median(lengths), max(lengths)) == (14, 22, 38)
    assert "**The questions are 14 to 38 words long**, 22 in the middle." in card()
    assert (min(quotes), max(quotes)) == (48, 286)
    assert "an exact quote of 48 to 286 characters" in card()
    joined_by_and = [question.id for question in found if ", and " in question.question]
    compared = [question.id for question in found if "compared with" in question.question]
    assert (len(joined_by_and), len(compared)) == (11, 6)
    assert {reasons()[question_id].kind for question_id in compared} == {"comparison"}
    assert (
        '11 of them ask for two things in one sentence, joined by "and", and 6 comparisons '
        'join their two sides by "compared with".' in card()
    )
    assert (
        '11 of the 33 questions join two parts by "and", and 6 comparisons join two sides by '
        '"compared with".' in card()
    )


def test_the_reasons_speak_of_each_passage_by_itself() -> None:
    for reason in reasons().values():
        assert reason.only_first != reason.only_second, reason.id
        assert min(len(reason.only_first), len(reason.only_second)) >= 40, reason.id
    # The example the card prints is a line of the file, word for word.
    example = re.search(r"```json\n(.+)\n```", card())
    assert example is not None
    assert example[1] in REASONS.read_text(encoding="utf-8").splitlines()


def test_no_question_is_a_golden_question_and_no_id_is_used_by_both() -> None:
    golden = load_golden_set(GOLDEN)

    assert not {q.id for q in golden} & {q.id for q in questions()}
    assert not {q.question.lower() for q in golden} & {q.question.lower() for q in questions()}


def test_the_examples_of_the_card_are_confirmed_questions_of_the_file() -> None:
    by_id = {question.id: question for question in confirmed()}
    examples = table(card(), "| Id | Kind | Question |")

    assert [reasons()[row[0].strip("`")].kind for row in examples] == [
        "comparison",
        "term-and-rule",
        "rule-and-exception",
        "process",
    ]
    for row in examples:
        assert by_id[row[0].strip("`")].question == row[2], row[0]


# The second reading


def test_the_readers_lines_are_114_items_on_three_forms_each_with_every_question_once() -> None:
    found = reading()
    ids = [question.id for question in questions()]

    assert len(found) == 114
    for row in lines(READING):
        assert list(row) == [
            "item",
            "question_id",
            "shown",
            "label",
            "confidence",
            "answer",
            "missing",
        ], row["item"]
    # In the order of the forms, each named by its place on its form.
    assert [line.item for line in found] == [
        f"{form}-{n:02d}" for form in "abc" for n in range(1, 39)
    ]
    for form in "abc":
        on_form = [line for line in found if line.item.startswith(form)]
        of_the_set = [line for line in on_form if line.shown in OF_A_QUESTION]
        assert sorted(line.question_id for line in of_the_set) == ids, form
        shown = Counter(line.shown for line in of_the_set)
        assert set(shown.values()) <= {11, 12}, form
        assert len(on_form) - len(of_the_set) == 4, form
    for question_id in ids:
        forms = {line.shown: line.item[0] for line in found if line.question_id == question_id}
        # Shown in each of its three ways, and each way to another reader.
        assert sorted(forms) == sorted(OF_A_QUESTION), question_id
        assert len(set(forms.values())) == 3, question_id
    assert "Each form has 11 or 12 questions of each of the three showings." in card()
    assert "has every line the readers wrote, 114 of them" in card()


def test_the_labels_of_each_question_are_the_ones_its_three_readers_gave() -> None:
    from_the_lines = second_reading_of(reading())

    assert from_the_lines == {
        question_id: reason.second_reading for question_id, reason in reasons().items()
    }
    assert [i for i, labels in from_the_lines.items() if not labels.confirms] == [DISPUTED]


def test_the_controls_are_golden_questions_that_rest_on_one_passage() -> None:
    golden = {question.id: question for question in load_golden_set(GOLDEN)}
    controls = [line for line in reading() if line.shown not in OF_A_QUESTION]

    assert Counter(line.shown for line in controls) == {"control": 8, "control-and-neighbour": 4}
    assert len({line.question_id for line in controls}) == 12
    for line in controls:
        assert golden[line.question_id].type == "answerable", line.item
        assert len(golden[line.question_id].evidence) == 1, line.item
        # Every control came out complete, and the reader was sure of it.
        assert (line.label, line.confidence) == ("complete", "high"), line.item
    assert "**All 12 controls came out `complete`**, each with high confidence." in card()
    assert (
        "8 over the three forms are shown with that passage alone, and 4 with that passage "
        "and a neighbouring passage of its page." in card()
    )


def test_the_table_of_what_was_shown_is_counted_from_the_readers_lines() -> None:
    counted: Counter[tuple[str, str]] = Counter((line.shown, line.label) for line in reading())
    shown = ["both", "first", "second", "control", "control-and-neighbour"]
    found = [numbers(row[1:]) for row in table(card(), "| What the reader was shown |")]

    assert found == [
        [
            sum(count for (was_shown, _), count in counted.items() if was_shown == kind),
            counted[(kind, "complete")],
            counted[(kind, "partial")],
            counted[(kind, "none")],
        ]
        for kind in shown
    ]
    assert found == [[34, 34, 0, 0], [34, 1, 33, 0], [34, 0, 34, 0], [8, 8, 0, 0], [4, 4, 0, 0]]
    text = card()
    assert "**33 of the 34 questions are confirmed.**" in text
    assert "**Both passages were enough for all 34 questions.**" in text
    assert "**One passage was not enough in 67 of the 68 showings**" in text
    assert "For the 33 confirmed questions that is all 66 showings." in text
    assert "No showing was labelled `none`." in text
    # Every reader who called an item partial said what was missing.
    assert all(line.missing for line in reading() if line.label == "partial")


def test_the_table_of_the_readers_is_counted_from_their_lines() -> None:
    found = {row[0]: numbers(row[1:]) for row in table(card(), "| Reader | Items |")}
    counted = {}
    for form in "abc":
        on_form = [line for line in reading() if line.item.startswith(form)]
        labels = Counter(line.label for line in on_form)
        sure = Counter(line.confidence for line in on_form)
        counted[form.upper()] = [
            len(on_form),
            labels["complete"],
            labels["partial"],
            sure["high"],
            sure["medium"],
            sure["low"],
        ]

    assert found == counted
    assert found == {
        "A": [38, 16, 22, 32, 5, 1],
        "B": [38, 16, 22, 32, 6, 0],
        "C": [38, 15, 23, 34, 4, 0],
    }
    below_high = Counter(line.shown for line in reading() if line.confidence != "high")
    assert below_high == {"first": 10, "second": 5, "both": 1}
    assert of_question("t001")["both"].confidence == "medium"
    text = card()
    assert "98 of the 114 labels came with high confidence, 15 with medium and 1 with low." in text
    assert (
        "Of the 16 below high, 10 are showings of a first passage alone, 5 of a second passage "
        "alone and 1 of both passages (`t001`)." in text
    )


def test_the_disputed_question_is_told_as_its_readers_told_it() -> None:
    shown = of_question(DISPUTED)
    (question,) = [question for question in questions() if question.id == DISPUTED]

    assert {name: (line.item, line.label, line.confidence) for name, line in shown.items()} == {
        "both": ("a-14", "complete", "high"),
        "first": ("b-23", "complete", "medium"),
        "second": ("c-26", "partial", "medium"),
    }
    quoted = re.search(
        r"Its answer: Support can still treat the case as an emergency, \"(.+?)\"", card()
    )
    assert quoted is not None
    assert quoted[1] in shown["first"].answer
    assert shown["second"].missing is not None
    assert "the strict definition" in shown["second"].missing
    assert f'**The question they dispute is `t025`**: "{question.question}"' in card()
    assert "No third reader was asked." in card()


def test_what_the_readers_said_about_the_questions_fits_their_lines() -> None:
    by_item = {line.item: line for line in reading()}
    why, text = reasons(), card()
    by_id = {question.id: question for question in questions()}

    # Two facts joined: the eleven items the readers named are five questions.
    named = ["a-03", "a-21", "b-04", "b-07", "b-32", "b-18", "b-38", "c-31", "c-23", "c-15", "c-18"]
    assert {by_item[item].question_id for item in named} == {"t009", "t012", "t028", "t029", "t034"}
    assert all("compared with" in by_id[i].question for i in ("t009", "t012"))
    assert all(", and " in by_id[i].question for i in ("t028", "t029", "t034"))
    for question_id in ("t009", "t012", "t028", "t029", "t034"):
        alone = [of_question(question_id)[shown] for shown in ("first", "second")]
        assert [line.label for line in alone] == ["partial", "partial"], question_id
    assert "The readers named eleven items, which are five questions" in text

    # Unlike things side by side: what two readers wrote, and the one who was less sure.
    assert "though the pay base differs" in of_question("t002")["both"].answer
    assert "they are not added together" in of_question("t027")["both"].answer
    with_both = [line for line in reading() if line.shown == "both"]
    assert [line.question_id for line in with_both if line.confidence != "high"] == ["t001"]

    # Loose wording: three showings of the second passage of a process question.
    loose = {item: by_item[item] for item in ("b-10", "c-19", "a-33")}
    assert {item: line.question_id for item, line in loose.items()} == {
        "b-10": "t032",
        "c-19": "t031",
        "a-33": "t033",
    }
    assert {line.shown for line in loose.values()} == {"second"}
    assert {why[line.question_id].kind for line in loose.values()} == {"process"}
    assert [line.item for line in reading() if line.confidence == "low"] == ["a-33"]
    assert loose["c-19"].confidence == "medium"
    for fragment, question_id in (
        ("What do we do with the report now", "t032"),
        ("What has to be finished before they can be onboarded", "t031"),
        ("What do I need to request", "t033"),
    ):
        assert fragment in by_id[question_id].question
        assert f'"{fragment}"' in text

    # Only the linking fact: the first passage of a term-and-rule question, by itself.
    linking = [
        line
        for line in reading()
        if line.shown == "first" and why[line.question_id].kind == "term-and-rule"
    ]
    unsure = [line for line in linking if (line.label, line.confidence) == ("partial", "medium")]
    only = ("The passage only establishes", "The passage only shows", "The passage says only")
    assert (len(linking), len(unsure)) == (8, 7)
    assert sum(line.answer.startswith(only) for line in unsure) == 6
    (eighth,) = [line for line in linking if line not in unsure]
    assert (eighth.question_id, eighth.label, eighth.confidence) == ("t018", "partial", "high")
    for question_id, shown in (("t021", "first"), ("t023", "second")):
        line = of_question(question_id)[shown]
        assert (line.label, line.confidence) == ("partial", "medium"), question_id
        assert " only " in line.answer.split(".")[0], question_id
    assert "In 7 of the 8 such showings of `term-and-rule` questions" in text
    assert not [line for line in reading() if line.label == "none"]

    # A fact that stood elsewhere on the form: none of the eight showings was carried across.
    for item, (question_id, shown) in GIVEN_AWAY_ELSEWHERE.items():
        assert (by_item[item].question_id, by_item[item].shown) == (question_id, shown), item
        assert by_item[item].label == "partial", item
    assert sum(by_item[item].confidence == "high" for item in GIVEN_AWAY_ELSEWHERE) == 6
    assert "All 8 were labelled `partial`, 6 of them with high confidence." in text


def test_the_makers_doubts_were_not_the_readers() -> None:
    # The three questions the maker was least sure of, and the one the readers dispute.
    for question_id in ("t017", "t020", "t006"):
        shown = of_question(question_id)
        assert [shown[name].label for name in OF_A_QUESTION] == ["complete", "partial", "partial"]
    assert "the maker named `t017`, `t020` and `t006`" in card()
    assert "The one question they dispute, `t025`, was not on the maker's list." in card()


def test_the_card_keeps_the_reading_and_the_search_apart() -> None:
    text = card()
    reading_at = text.index("\n## The second reading\n")
    search_at = text.index("\n## What the search does with them\n")

    assert reading_at < search_at
    assert "This section is about the questions" in text[reading_at:search_at]
    assert "It says nothing about the search" in text[reading_at:search_at]
    assert (
        "This section is about the search, measured on the 33 confirmed questions."
        in text[search_at:]
    )
    assert "The readers had no part in it" in text[search_at:]
    # No figure of the search stands in the section on the reading.
    assert "top 5" not in text[reading_at:search_at]


# The saved runs


def test_each_saved_run_scored_the_confirmed_questions_and_left_the_disputed_one_out() -> None:
    found = confirmed()
    for mode in MODES:
        scored, about = rows(mode), summary(mode)
        assert list(scored) == [question.id for question in found], mode
        assert DISPUTED not in scored
        for question in found:
            assert scored[question.id]["fingerprint"] == question.fingerprint, question.id
            assert scored[question.id]["status"] == "ok"
            assert scored[question.id]["tags"][-1] == "review:checked"
        assert (about["questions_scored"], about["questions_failed"]) == (33, 0), mode
        assert about["questions_left_out"] == [DISPUTED], mode
        assert about["config"]["retrieval_mode"] == mode
        assert about["config"]["rerank"] is False
        assert (RUNS / mode / "errors.jsonl").read_text(encoding="utf-8") == ""
    assert len({summary(mode)["labels_fingerprint"] for mode in MODES}) == 1
    assert "names the question it left out under `questions_left_out`" in card()
    assert card().count("--reviewed-only\n") == 3


def test_the_run_of_the_default_search_has_the_settings_of_the_golden_sets_hybrid_run() -> None:
    golden_run = json.loads((GOLDEN_HYBRID / "summary.json").read_text(encoding="utf-8"))

    # The same index, models, fusion settings and handbook commit: the default search.
    assert summary("hybrid")["config"] == golden_run["config"]
    assert summary("hybrid")["config"]["retrieval_mode"] == Settings().retrieval_mode
    assert Settings().rerank is False
    assert "`e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8`" in card()
    assert summary("hybrid")["config"]["handbook_commit"] in card()


def test_the_counts_of_each_run_are_the_counts_of_its_rows() -> None:
    for mode in MODES:
        scored, about = rows(mode), summary(mode)
        several = about["needs_several_passages"]
        by_quote = [row["evidence_ranks"] for row in scored.values()]
        by_page = [row["page_ranks"] for row in scored.values()]

        assert (several["questions"], several["retrieved"]) == (33, RETRIEVAL_DEPTH)
        for k in ALL_CUTOFFS:
            counts = split(by_quote, k)
            assert several["found_in_top"][str(k)] == counts, (mode, k)
            assert several["pages_in_top"][str(k)] == split(by_page, k), (mode, k)
            # The two scores say the same thing as shares.
            assert about["metrics"][f"all@{k}"]["mean"] == pytest.approx(counts["all"] / 33)
            assert about["metrics"][f"hit@{k}"]["mean"] == pytest.approx(
                (counts["all"] + counts["some"]) / 33
            )
            assert [row["grade"][f"all@{k}"] for row in scored.values()] == [
                float(all(rank is not None and rank <= k for rank in ranks)) for ranks in by_quote
            ]
        complete = Counter(max(ranks) for ranks in by_quote if None not in ranks)
        assert several["all_found_by_rank"] == {
            str(rank): count for rank, count in sorted(complete.items())
        }
        assert several["not_all_retrieved"] == sum(None in ranks for ranks in by_quote)
        # A chunk that covers the quote is a chunk of its page, so a page never comes later.
        for of_quote, of_page in zip(by_quote, by_page, strict=True):
            for quote_rank, page_rank in zip(of_quote, of_page, strict=True):
                assert quote_rank is None or (page_rank is not None and page_rank <= quote_rank)


def test_the_table_of_the_three_searches_gives_the_figures_of_the_saved_runs() -> None:
    found = table(card(), "| Search | Run | Both in the top 5")

    assert [row[1] for row in found] == [
        f"[`{mode}`](../runs/two-passage/{mode})" for mode in MODES
    ]
    for row in found:
        metrics = summary(row[1].split("`")[1])["metrics"]
        both = metrics["all@5"]
        assert row[2:] == [
            f"{both['mean']:.3f} ({both['low']:.3f} to {both['high']:.3f})",
            f"{metrics['hit@5']['mean']:.3f}",
            f"{metrics['all@10']['mean']:.3f}",
            f"{metrics['hit@10']['mean']:.3f}",
        ], row[0]
    assert found[0][0] == "Hybrid, the default"
    assert found[0][2:] == ["0.364 (0.212 to 0.515)", "0.788", "0.515", "0.909"]


def test_the_counts_of_the_default_search_are_the_counts_of_its_run() -> None:
    several = summary("hybrid")["needs_several_passages"]
    found = {int(row[0]): numbers(row[1:]) for row in table(card(), "| In the first | Both found")}

    assert found == {
        k: [
            several["found_in_top"][str(k)]["all"],
            several["found_in_top"][str(k)]["some"],
            several["found_in_top"][str(k)]["none"],
            several["pages_in_top"][str(k)]["all"],
        ]
        for k in ALL_CUTOFFS
    }
    assert found[5] == [12, 14, 7, 15]
    text = card()
    assert "The default search, as counts of the 33 questions:" in text
    assert "for 12 of the 33 questions.** It brings at least one for 26. So 14 questions" in text
    assert "For the first five results the truth lies between 12 and 15 of 33." in text
    in_five = {mode: summary(mode)["needs_several_passages"]["found_in_top"]["5"] for mode in MODES}
    assert [in_five[mode]["all"] for mode in MODES] == [12, 6, 8]
    assert (
        "Hybrid has both passages in its first five for 12 questions, keyword search for 8 "
        "and the embedding model for 6." in text
    )


def test_the_comparison_of_the_searches_is_the_paired_comparison_of_their_runs() -> None:
    def against(base: str) -> str:
        """Hybrid against another search on both-in-the-top-five, as the card words it."""
        change = compare_runs(RUNS / base, RUNS / "hybrid").metrics["all@5"].change
        difference = change.difference
        return (
            f"is {difference.mean:+.3f} ({{}}{difference.low:+.3f} to {difference.high:+.3f}): "
            f"{change.better} questions better and {change.worse} worse"
        )

    text = card()
    assert "hybrid against keyword search " + against("sparse").format("95% interval ") in text
    assert "Hybrid against the embedding model " + against("dense").format("") in text
    # Clear against keyword search, by the interval. The card says what that count can
    # carry: all four questions fell one way, which a fair coin does one time in eight.
    to_keyword = compare_runs(RUNS / "sparse", RUNS / "hybrid").metrics["all@5"].change
    assert to_keyword.is_clear
    assert (
        f"A count of {to_keyword.better} against {to_keyword.worse} cannot carry the word: an "
        f"exact sign test on it gives p = {sign_test(to_keyword.better, to_keyword.worse):.3f}"
        in text
    )
    assert (to_keyword.better, to_keyword.worse) == (4, 0)
    assert "By the rule `eval-compare` applies, that difference reads as clear." in text
    assert "([issue 80](https://github.com/MumukshuPanchal/handbook-rag/issues/80))" in text
    # Against the embedding model the interval reaches zero.
    to_dense = compare_runs(RUNS / "dense", RUNS / "hybrid").metrics["all@5"].change
    assert not to_dense.is_clear
    assert to_dense.difference.low == 0.0
    assert "That interval reaches zero" in text
    assert (
        "**Hybrid finds both passages for more questions than either other search, and "
        "neither lead is firm.**" in text
    )
    hybrid, *others = (
        summary(mode)["needs_several_passages"]["found_in_top"]["5"]["all"] for mode in MODES
    )
    assert hybrid > max(others)


def test_the_ranks_at_which_both_passages_were_found_are_those_of_the_run() -> None:
    several = summary("hybrid")["needs_several_passages"]
    header, counts = (
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in card().splitlines()
        if line.startswith(("| Rank at which both were found", "| Questions |"))
    )
    by_rank = dict(zip(header[1:], numbers(counts[1:]), strict=True))

    expected = {str(rank): several["all_found_by_rank"].get(str(rank), 0) for rank in range(2, 11)}
    assert by_rank == {**expected, "Not in the first 10": several["not_all_retrieved"]}
    assert sum(by_rank.values()) == 33
    late = sum(count for rank, count in several["all_found_by_rank"].items() if int(rank) > 5)
    neither = several["found_in_top"]["10"]["none"]
    assert (late, several["not_all_retrieved"], neither) == (5, 16, 3)
    assert "For 5 questions the later of the two passages came at ranks 6 to 10." in card()
    assert (
        "For 16 questions a passage is not among the first ten results at all, and for 3 of "
        "those neither is." in card()
    )


def test_the_table_by_kind_is_counted_from_the_rows_of_the_default_search() -> None:
    why = reasons()

    def by_kind(mode: str) -> dict[str, list[list[int | None]]]:
        counted: dict[str, list[list[int | None]]] = {}
        for question_id, row in rows(mode).items():
            counted.setdefault(why[question_id].kind, []).append(row["evidence_ranks"])
        return counted

    counted = by_kind("hybrid")
    found = {row[0].strip("`"): numbers(row[1:]) for row in table(card(), "| Kind of question |")}

    assert found == {
        kind: [len(ranks), *split(ranks, 5).values()] for kind, ranks in counted.items()
    }
    assert found == {
        "comparison": [12, 8, 4, 0],
        "term-and-rule": [8, 0, 5, 3],
        "rule-and-exception": [6, 1, 4, 1],
        "process": [7, 3, 1, 3],
    }
    # In a term-and-rule question the first passage is the one that defines the term.
    undefined = sum(ranks[0] is None for ranks in counted["term-and-rule"])
    assert undefined == 7
    assert "in 7 of them the passage that defines the term or gives the class is not" in card()
    for mode in ("dense", "sparse"):
        assert split(by_kind(mode)["term-and-rule"], 5)["all"] == 0, mode
    assert (
        "Neither of the other two searches finds both passages of a `term-and-rule` question "
        "in its first five either." in card()
    )
    assert "With 6 to 12 questions in a kind" in card()


def test_the_figures_with_the_disputed_question_counted_fit_the_runs_without_it() -> None:
    # The runs of all 34 questions are in the history and not in the tree, so the
    # sentence cannot be read back from a file. What can be held: one question more can
    # add one to a count, or nothing.
    sentence = (
        "12, 7 and 8 of the 34 had both passages in the first five (hybrid, embedding model, "
        "keyword), and 27, 27 and 24 had at least one."
    )
    in_five = [summary(mode)["needs_several_passages"]["found_in_top"]["5"] for mode in MODES]

    assert sentence in card()
    for with_it, counts in zip((12, 7, 8), in_five, strict=True):
        assert with_it - counts["all"] in (0, 1)
    for with_it, counts in zip((27, 27, 24), in_five, strict=True):
        assert with_it - counts["all"] - counts["some"] in (0, 1)
    # The first commit of issue 78, which measured all 34. A test cannot read the history,
    # which a shallow clone does not have, so it holds the name that goes with the hash.
    assert (
        'at commit `f880d4b` ("feat: questions that need two passages, and whether the '
        'search finds both")' in card()
    )


def test_the_scores_of_the_golden_set_are_counted_as_the_card_says() -> None:
    golden = [question for question in load_golden_set(GOLDEN) if question.type == "answerable"]
    golden_run = lines(GOLDEN_HYBRID / "results.jsonl")

    # "All 924 existing scores of its 132 questions": seven scores for each question.
    assert len(golden) == len(golden_run) == 132
    assert sum(len(row["grade"]) for row in golden_run) == 924
    assert "All 924 existing scores of its 132 questions" in card()
    # No golden question is pending, so the option that leaves pending ones out is idle.
    assert not [question.id for question in load_golden_set(GOLDEN) if question.review == "pending"]
    assert "the golden set has no question whose review is pending" in card()


def test_every_link_of_the_card_leads_somewhere() -> None:
    text = card()
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("# ").lower()).replace(" ", "-")
        for line in text.splitlines()
        if line.startswith("#")
    }
    targets = re.findall(r"\]\(([^)]+)\)", text)

    assert len(targets) >= 20
    for target in targets:
        if target.startswith("https://"):
            continue
        if target.startswith("#"):
            assert target[1:] in headings, target
        else:
            assert (CARD.parent / target).exists(), target


@pytest.mark.slow
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="needs the handbook snapshot in data/handbook")
def test_every_quote_is_in_its_page_once_and_on_no_other_page_and_one_chunk_holds_it() -> None:
    # Built from the snapshot, as the commands build it.
    corpus = build_corpus(Settings(data_dir=ROOT / "data"))
    pages = {document.doc_id: document for document in corpus.documents}

    report = check_against_corpus(questions(), corpus.documents)
    of_the_confirmed = check_against_corpus(confirmed(), corpus.documents)

    assert report.problems == []
    assert report.high_overlap == []
    # "25% of a question's meaningful words on average"
    assert f"{of_the_confirmed.mean_overlap:.0%}" == "25%"
    assert "25% of a question's meaningful words on average" in card()
    assert f"{len(corpus.documents):,}" == "3,846"
    assert "in all 3,846 pages" in card()
    holder: dict[tuple[str, int], str] = {}
    for question in questions():
        for index, piece in enumerate(question.evidence):
            text = pages[piece.doc_id].text
            assert text.count(piece.quote) == 1, question.id
            start = text.index(piece.quote)
            holders = [
                chunk
                for chunk in corpus.chunks
                if chunk.doc_id == piece.doc_id
                and min(chunk.end_char, start + len(piece.quote)) - max(chunk.start_char, start)
                >= MIN_COVERAGE * len(piece.quote)
            ]
            # One chunk holds each quote, so "found" never depends on which of two
            # overlapping chunks the search returned.
            assert len(holders) == 1, question.id
            holder[(question.id, index)] = holders[0].chunk_id

    # Four passages are each the evidence of two questions, as the card says.
    used_by = Counter(holder.values())
    shared = sorted(
        sorted(question_id for (question_id, _), chunk in holder.items() if chunk == chunk_id)
        for chunk_id, count in used_by.items()
        if count > 1
    )
    assert shared == [["t013", "t014"], ["t013", "t020"], ["t016", "t020"], ["t021", "t022"]]
    assert "Four passages are each the evidence of two questions" in card()
    # So on a form, the passage that one item held back could stand under another
    # question. These are the showings where it did.
    shown_as = {"first": (0,), "second": (1,), "both": (0, 1)}
    given_away = {}
    for line in reading():
        if line.shown not in ("first", "second"):
            continue
        held_back = holder[(line.question_id, 1 if line.shown == "first" else 0)]
        elsewhere = [
            other
            for other in reading()
            if other.item[0] == line.item[0]
            and other.item != line.item
            and other.shown in shown_as
            and held_back in {holder[(other.question_id, i)] for i in shown_as[other.shown]}
        ]
        if elsewhere:
            given_away[line.item] = (line.question_id, line.shown)
    assert given_away == GIVEN_AWAY_ELSEWHERE
    assert "So in 8 showings of one passage" in card()
