"""The two-passage questions in the repository are what their card says they are.

The loader and the check of ``golden-check --reasons`` hold the file to its shape. These
tests hold it to the rest of its card: every count the card gives about the questions,
and every figure it gives about the saved runs, which are read from the runs' own files.

What no test can hold is the claim the set rests on: that each answer needs both
passages. That is a matter of reading. The card says who has read the questions so far.

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
    WhyTwo,
    check_two_passage_set,
    kinds_of,
    load_reasons,
)
from handbook_rag.ingest.build import build_corpus
from tests.test_readme_results import table

ROOT = Path(__file__).parents[2]
QUESTIONS = ROOT / "evals" / "two-passage" / "questions.jsonl"
REASONS = ROOT / "evals" / "two-passage" / "why-two.jsonl"
CARD = ROOT / "evals" / "two-passage" / "README.md"
RUNS = ROOT / "evals" / "runs" / "two-passage"
GOLDEN = ROOT / "evals" / "golden" / "questions.jsonl"
GOLDEN_HYBRID = ROOT / "evals" / "runs" / "retrieval" / "v2"
SNAPSHOT = ROOT / "data" / "handbook"
MODES = ("hybrid", "dense", "sparse")
# Wording that would point at the handbook as a source, which no asker would use.
BANNED = ("handbook", "according to", "as per", "the passage", "this section", "the page")


def questions() -> list[GoldenQuestion]:
    return load_golden_set(QUESTIONS)


def reasons() -> dict[str, WhyTwo]:
    return load_reasons(REASONS)


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
    for of_question in ranks:
        found = sum(rank is not None and rank <= k for rank in of_question)
        counts["all" if found == len(of_question) else "some" if found else "none"] += 1
    return counts


def test_the_file_holds_34_questions_in_the_golden_format_with_a_reason_for_each() -> None:
    found = questions()

    assert [question.id for question in found] == [f"t{n:03d}" for n in range(1, 35)]
    assert check_two_passage_set(found, reasons()) == []
    for row in lines(QUESTIONS):
        assert list(row) == [
            "id",
            "question",
            "reference_answer",
            "evidence",
            "type",
            "origin",
            "review",
        ], row["id"]
        assert [list(piece) for piece in row["evidence"]] == [["doc_id", "quote"]] * 2, row["id"]
    for row in lines(REASONS):
        assert list(row) == ["id", "kind", "only_first", "only_second"], row["id"]
    assert [row["id"] for row in lines(REASONS)] == [question.id for question in found]
    assert "holds 34 of them" in card()


def test_the_status_of_the_card_is_the_status_in_the_file() -> None:
    # Written by one model session, and checked by nobody: not by a second session, and
    # not by a person.
    assert {(question.origin, question.review) for question in questions()} == {
        ("model-drafted", "pending")
    }
    assert "**Status: written by one model session, not yet read by anyone else.**" in card()
    assert '`"review": "pending"`' in card()
    assert "No person has read the questions." in card()


def test_the_kinds_are_in_the_numbers_the_card_gives() -> None:
    made = kinds_of(questions(), reasons())
    found = {row[0].strip("`"): int(row[2]) for row in table(card(), "| Kind | What the two")}

    assert found == {**made, "All": 34}
    assert made == {"comparison": 12, "term-and-rule": 8, "rule-and-exception": 7, "process": 7}
    assert set(made) == set(KINDS)
    # Kept together in the file, kind by kind, in the order of the card's table.
    order = [reasons()[question.id].kind for question in questions()]
    assert order == [kind for kind in found if kind != "All" for _ in range(made[kind])]


def test_the_pages_and_sections_are_the_ones_the_card_counts() -> None:
    found = questions()
    pages = Counter(piece.doc_id for question in found for piece in question.evidence)
    sections = Counter(section(doc_id) for doc_id in pages.elements())
    across = sum(len({section(piece.doc_id) for piece in q.evidence}) == 2 for q in found)

    assert (len(pages), len(sections), sum(pages.values())) == (58, 18, 68)
    assert "**58 pages from 18 handbook sections** hold the 68 pieces." in card()
    assert sum(count > 1 for count in pages.values()) == 8
    assert "8 pages are used by more than one question" in card()
    assert across == 16
    assert "**In 16 questions the two pages are in two different sections**" in card()
    assert "In the other 18 they are two pages of one section." in card()
    assert (sections["security"], sections["total-rewards"]) == (16, 15)
    assert (
        "16 of the 68 pieces of evidence are on security pages and 15 on pages of total" in card()
    )
    used_three_times = sorted(doc_id.split("/")[-1] for doc_id, n in pages.items() if n == 3)
    assert used_three_times == ["bv-benefits-netherlands.md", "critical-systems.md"]
    assert max(pages.values()) == 3
    # The issue asks for at least eight sections.
    assert len({question.section for question in found}) == 16


def test_every_question_and_quote_follows_the_drafting_guidelines() -> None:
    found = questions()
    lengths = [len(question.question.split()) for question in found]
    quotes = [len(piece.quote) for question in found for piece in question.evidence]

    for question in found:
        assert question.question.endswith("?"), question.id
        assert not any(phrase in question.question.lower() for phrase in BANNED), question.id
        for piece in question.evidence:
            assert 40 <= len(piece.quote) <= 300, question.id
            assert piece.quote == piece.quote.strip(), question.id
            assert piece.doc_id.endswith(".md"), question.id
    assert (min(lengths), statistics.median(lengths), max(lengths)) == (14, 22, 39)
    assert "**The questions are 14 to 39 words long**, 22 in the middle." in card()
    assert (min(quotes), max(quotes)) == (48, 286)
    assert "an exact quote of 48 to 286 characters" in card()
    assert sum(", and " in question.question for question in found) == 12
    assert '12 of them ask for two things in one sentence, joined by "and".' in card()


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


def test_the_examples_of_the_card_are_questions_of_the_file() -> None:
    by_id = {question.id: question for question in questions()}
    examples = table(card(), "| Id | Kind | Question |")

    assert [reasons()[row[0].strip("`")].kind for row in examples] == [
        "comparison",
        "term-and-rule",
        "rule-and-exception",
        "process",
    ]
    for row in examples:
        assert by_id[row[0].strip("`")].question == row[2], row[0]


def test_each_saved_run_scored_the_committed_questions_with_its_own_search() -> None:
    found = questions()
    for mode in MODES:
        scored, about = rows(mode), summary(mode)
        assert list(scored) == [question.id for question in found], mode
        for question in found:
            assert scored[question.id]["fingerprint"] == question.fingerprint, question.id
            assert scored[question.id]["status"] == "ok"
        assert (about["questions_scored"], about["questions_failed"]) == (34, 0), mode
        assert about["config"]["retrieval_mode"] == mode
        assert about["config"]["rerank"] is False
        assert (RUNS / mode / "errors.jsonl").read_text(encoding="utf-8") == ""
    assert len({summary(mode)["labels_fingerprint"] for mode in MODES}) == 1


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

        assert (several["questions"], several["retrieved"]) == (34, RETRIEVAL_DEPTH)
        for k in ALL_CUTOFFS:
            counts = split(by_quote, k)
            assert several["found_in_top"][str(k)] == counts, (mode, k)
            assert several["pages_in_top"][str(k)] == split(by_page, k), (mode, k)
            # The two scores say the same thing as shares.
            assert about["metrics"][f"all@{k}"]["mean"] == pytest.approx(counts["all"] / 34)
            assert about["metrics"][f"hit@{k}"]["mean"] == pytest.approx(
                (counts["all"] + counts["some"]) / 34
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
    assert found[5] == [12, 15, 7, 16]
    text = card()
    assert "for 12 of the 34 questions.** It brings at least one for 27. So 15 questions" in text
    assert "For the first five results the truth lies between 12 and 16 of 34." in text
    in_five = {mode: summary(mode)["needs_several_passages"]["found_in_top"]["5"] for mode in MODES}
    assert [in_five[mode]["all"] for mode in MODES] == [12, 7, 8]
    assert (
        "Hybrid has both passages in its first five for 12 questions, keyword search for 8 "
        "and the embedding model for 7." in text
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
    # Clear against keyword search, and not against the embedding model.
    assert compare_runs(RUNS / "sparse", RUNS / "hybrid").metrics["all@5"].change.is_clear
    assert not compare_runs(RUNS / "dense", RUNS / "hybrid").metrics["all@5"].change.is_clear
    assert (
        "**Hybrid is ahead of keyword search, and cannot be told apart from the embedding" in text
    )


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
    assert sum(by_rank.values()) == 34
    late = sum(count for rank, count in several["all_found_by_rank"].items() if int(rank) > 5)
    neither = several["found_in_top"]["10"]["none"]
    assert (late, several["not_all_retrieved"], neither) == (5, 17, 3)
    assert "For 5 questions the later of the two passages came at ranks 6 to 10." in card()
    assert (
        "For 17 questions a passage is not among the first ten results at all, and for 3 of "
        "those neither is." in card()
    )


def test_the_table_by_kind_is_counted_from_the_rows_of_the_default_search() -> None:
    scored, why = rows("hybrid"), reasons()
    counted: dict[str, list[list[int | None]]] = {}
    for question_id, row in scored.items():
        counted.setdefault(why[question_id].kind, []).append(row["evidence_ranks"])
    found = {row[0].strip("`"): numbers(row[1:]) for row in table(card(), "| Kind of question |")}

    assert found == {
        kind: [len(ranks), *split(ranks, 5).values()] for kind, ranks in counted.items()
    }
    assert found["comparison"] == [12, 8, 4, 0]
    assert found["term-and-rule"] == [8, 0, 5, 3]
    # In a term-and-rule question the first passage is the one that defines the term.
    undefined = sum(ranks[0] is None for ranks in counted["term-and-rule"])
    assert undefined == 7
    assert "in 7 of them the passage that defines the term or gives the class is not" in card()


def test_the_wording_figure_and_the_scores_of_the_golden_set_are_as_the_card_says() -> None:
    golden = [question for question in load_golden_set(GOLDEN) if question.type == "answerable"]
    golden_run = lines(GOLDEN_HYBRID / "results.jsonl")

    # "All 924 existing scores of its 132 questions": seven scores for each question.
    assert len(golden) == len(golden_run) == 132
    assert sum(len(row["grade"]) for row in golden_run) == 924
    assert "All 924 existing scores of its 132 questions" in card()


def test_every_link_of_the_card_leads_somewhere() -> None:
    text = card()
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("# ").lower()).replace(" ", "-")
        for line in text.splitlines()
        if line.startswith("#")
    }
    targets = re.findall(r"\]\(([^)]+)\)", text)

    assert len(targets) >= 15
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
    found = questions()

    report = check_against_corpus(found, corpus.documents)

    assert report.problems == []
    assert report.high_overlap == []
    # "25% of a question's meaningful words on average"
    assert f"{report.mean_overlap:.0%}" == "25%"
    assert "25% of a question's meaningful words on average" in card()
    assert f"{len(corpus.documents):,}" == "3,846"
    assert "in all 3,846 pages" in card()
    for question in found:
        for piece in question.evidence:
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
