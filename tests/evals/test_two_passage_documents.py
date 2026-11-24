"""What the documents say of the two-passage questions is what the saved runs hold.

The card of the questions and the results of the search variants have tests of their own
(``test_committed_two_passage_questions.py``, ``test_committed_search_variants.py``).
These tests hold the documents written from them: the decision record (ADR 0021), the
section of the evaluation guide and what the guide says of the new runs elsewhere, the
lines of the changelog and the roadmap, and the architecture page. The README's section
is held by ``tests/test_readme_results.py``, with the README's other results.

Every figure is worked out once, in :func:`figures`, from the saved runs and the files of
the questions. A table is built again from the runs. Every line of prose that holds a
number is compared with the numbers it must hold, in their order, and a line with a
number that no test works out fails (``test_readme_results.assert_numbers``). So a figure
changed in one document and not in another fails, a number added by hand fails, and so
does a run made again without the documents being brought up to date. A few small counts
that the documents write as words are held too.

What no test can hold is the reading of the figures, and what was not done. The documents
say that themselves: no answer was written for these questions, a model wrote them with
both passages in front of it, no person has read them, and fourteen variants were tried
on the same 33.
"""

import functools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from handbook_rag.config import Settings
from handbook_rag.evals.compare import compare_runs
from handbook_rag.evals.dataset import GoldenQuestion, load_golden_set
from handbook_rag.evals.metrics import Comparison, paired_difference
from handbook_rag.evals.retrieval import ALL_CUTOFFS, RETRIEVAL_DEPTH
from handbook_rag.evals.two_passage import load_reasons, load_second_reading
from tests.evals import test_committed_search_variants as variants
from tests.evals.test_committed_two_passage_questions import NAMES_ITS_TERM
from tests.test_readme_results import (
    anchors,
    assert_numbers,
    in_words,
    line_starting,
    numbers,
    table,
)

ROOT = Path(__file__).parents[2]
ADR = ROOT / "docs" / "adr" / "0021-two-passage-questions-and-the-search-as-it-is.md"
GUIDE = ROOT / "docs" / "evaluation.md"
CARD = ROOT / "evals" / "two-passage" / "README.md"
RESULTS = ROOT / "evals" / "two-passage" / "search-variants.md"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"
ROADMAP = ROOT / "docs" / "roadmap.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
GOLDEN = ROOT / "evals" / "golden" / "questions.jsonl"
QUESTIONS = ROOT / "evals" / "two-passage" / "questions.jsonl"
READING = ROOT / "evals" / "two-passage" / "second-reading.jsonl"
ANSWERS = ROOT / "evals" / "runs" / "answers" / "baseline"
TWO, GOLDEN_RUNS, DEFAULT = variants.TWO, variants.GOLDEN, variants.DEFAULT
SEARCHES = {"hybrid": "Hybrid, the default", "dense": "Embedding model", "sparse": "Keyword (BM25)"}
ISSUE = "https://github.com/MumukshuPanchal/handbook-rag/issues/"
# Numbers that are names, and so part of what a line holds: the decision record, and the
# issues of the two-passage questions, of follow-up questions, of the rule for "clear"
# and of query rewriting.
RECORD, QUESTIONS_ISSUE, FOLLOW_UPS, CLEAR, REWRITING = "0021", "78", "79", "80", "22"
# The cutoff in the name of the headline score, "both in the top 5".
TOP = "5"
# The free key's allowance of requests a day, as the README's limitations give it.
A_DAY = 500

SEARCHES_HEADER = [
    "Search",
    "Run",
    "Both in the top 5 (95% interval)",
    "At least one in the top 5",
    "Both in the top 10",
    "At least one in the top 10",
]
GUIDE_VARIANTS_HEADER = [
    "Variant",
    "Both in the top 5, of the 33",
    "Change (95% interval)",
    "Better / worse",
    "Reading",
    "Hit@5 on the golden set, of 132",
    "Its change (95% interval)",
    "Its better / worse",
    "Its reading",
]
ADR_VARIANTS_HEADER = ["Variant", "Both in the top 5, of the 33", "Hit@5 on the golden set, of 132"]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def guide_section() -> str:
    """The guide's section on the two-passage questions, up to the next section."""
    after = read(GUIDE).split("\n## Questions that need two passages\n")
    assert len(after) == 2
    return after[1].split("\n## ")[0]


def header(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def three(value: float) -> str:
    return f"{value:.3f}"


def unsigned(comparison: Comparison) -> list[str]:
    """A paired difference and its interval as a line's numbers read them: without signs."""
    difference = comparison.difference
    return [three(abs(value)) for value in (difference.mean, difference.low, difference.high)]


def p_value(comparison: Comparison, places: int) -> str:
    """The exact sign test on a comparison's better and worse, at the places a line gives."""
    return f"{variants.sign_test(comparison.better, comparison.worse):.{places}f}"


def setting(run: str, name: str) -> str:
    """The one setting a variant's run changes, as its name says it."""
    return str(variants.VARIANTS[run][1][name])


def pages(question: GoldenQuestion) -> int:
    return len({piece.doc_id for piece in question.evidence})


def missing(run: str) -> set[str]:
    """The questions of the 33 with a passage that is not among a run's results."""
    return {
        question
        for question, row in variants.rows(TWO, run).items()
        if None in row["evidence_ranks"]
    }


def of_kind(kind: str) -> list[str]:
    """The questions of the 33 of one kind."""
    why = load_reasons(variants.REASONS)
    return [question for question in variants.rows(TWO, DEFAULT) if why[question].kind == kind]


def both_found(run: str, questions: list[str]) -> int:
    """For how many of these questions a run of the 33 has both passages in its first five."""
    return sum(int(variants.rows(TWO, run)[question]["grade"]["all@5"]) for question in questions)


def lines_of(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def depth(k: int) -> int:
    """For how many of the 33 the default's first k results hold both passages."""
    return int(variants.depth_row(k)[1].split(" of ")[0])


@functools.cache
def figures() -> dict[str, Any]:
    """Every figure the documents give, each worked out once from the files and the runs."""
    answerable = [q for q in load_golden_set(GOLDEN) if q.type == "answerable"]
    several = [q for q in answerable if len(q.evidence) > 1]
    written = load_golden_set(QUESTIONS)
    why = load_reasons(variants.REASONS)
    default = variants.summary(TWO, DEFAULT)
    several_33 = default["needs_several_passages"]
    in_top = several_33["found_in_top"]
    golden = variants.rows(GOLDEN_RUNS, DEFAULT)
    of_the_33 = Counter(why[question].kind for question in variants.rows(TWO, DEFAULT))
    late = {
        int(rank): count for rank, count in several_33["all_found_by_rank"].items() if int(rank) > 5
    }

    def at_least_one(run: str) -> int:
        counts = variants.summary(TWO, run)["needs_several_passages"]["found_in_top"]["5"]
        return int(counts["all"] + counts["some"])

    def under_its_page(row: dict[str, Any]) -> bool:
        """Whether a chunk of a passage's page stands above the passage in the first five."""
        return any(
            variants.in_first(of_quote, 5) and of_page < of_quote
            for of_quote, of_page in zip(row["evidence_ranks"], row["page_ranks"], strict=True)
        )

    return {
        # The golden set
        "answerable": len(answerable),
        "one_piece": len(answerable) - len(several),
        "any_of_several_pages": sum(q.evidence_mode == "any" and pages(q) > 1 for q in several),
        "all_on_one_page": [q.id for q in several if q.evidence_mode == "all" and pages(q) == 1],
        "all_on_two_pages": sum(q.evidence_mode == "all" and pages(q) > 1 for q in several),
        "golden_found": int(sum(variants.scores(GOLDEN_RUNS, DEFAULT, "hit@5"))),
        "golden_hit": {
            run: variants.summary(GOLDEN_RUNS, run)["metrics"]["hit@5"]["mean"]
            for run in variants.ORDER
        },
        "with_the_passage": {q for q, row in golden.items() if row["grade"]["hit@5"] == 1.0},
        "under_its_page": {q for q, row in golden.items() if under_its_page(row)},
        # Seven scores for each question in a committed run of the golden set.
        "scores_repeated": sum(
            len(row["grade"]) for row in variants.rows(variants.COMMITTED, "v2").values()
        ),
        # The questions and their second reading
        "written": len(written),
        "kinds_written": Counter(why[question.id].kind for question in written),
        "confirmed": sum(bool(reason.confirmed) for reason in why.values()),
        # The golden questions mixed into the second reading as controls.
        "controls": len(
            {
                line.question_id
                for line in load_second_reading(READING)
                if line.shown in ("control", "control-and-neighbour")
            }
        ),
        "kind_sizes": (min(of_the_33.values()), max(of_the_33.values())),
        # The default search on the 33
        "both": in_top["5"]["all"],
        "only_one": in_top["5"]["some"],
        "share": three(default["metrics"]["all@5"]["mean"]),
        "interval": [three(default["metrics"]["all@5"][end]) for end in ("low", "high")],
        "hit": three(default["metrics"]["hit@5"]["mean"]),
        "comparisons": both_found(DEFAULT, of_kind("comparison")),
        "not_in_ten": several_33["not_all_retrieved"],
        "neither_in_ten": in_top["10"]["none"],
        "by_page": several_33["pages_in_top"]["5"]["all"],
        "deeper": paired_difference(
            variants.scores(TWO, DEFAULT, "all@5"), variants.scores(TWO, DEFAULT, "all@10")
        ),
        "late": sum(late.values()),
        "late_ranks": (min(late), max(late)),
        # The variants
        "found": {run: int(sum(variants.scores(TWO, run, "all@5"))) for run in variants.ORDER},
        "at_least_one": {run: at_least_one(run) for run in variants.ORDER},
        "lost": {
            limit: variants.lost(GOLDEN_RUNS, run, "hit@5")
            for limit, run in variants.PAGE_LIMITS.items()
        },
        "by_limit": {
            limit: variants.against_default(GOLDEN_RUNS, run, "hit@5")
            for limit, run in variants.PAGE_LIMITS.items()
        },
        # For how many of the questions with a passage outside the default's first ten a
        # variant brings both into its first ten.
        "reached": max(len(missing(DEFAULT) - missing(run)) for run in variants.VARIANTS),
        "reached_by": {run: len(missing(DEFAULT) - missing(run)) for run in variants.VARIANTS},
        # Every difference `eval-compare` calls clear, over all its scores, on both sets.
        "clear": variants.clear_readings(),
        "rerank": variants.against_default(TWO, "rerank", "all@5"),
        "constant_33": variants.against_default(TWO, "rrf-k-10", "all@5"),
        "constant_golden": variants.against_default(GOLDEN_RUNS, "rrf-k-10", "hit@5"),
        "constant_mrr": variants.against_default(GOLDEN_RUNS, "rrf-k-10", "mrr@10"),
        "milliseconds": (
            variants.median_ms(GOLDEN_RUNS, "rerank"),
            variants.median_ms(GOLDEN_RUNS, DEFAULT),
            variants.median_ms(variants.COMMITTED, "v3"),
            variants.median_ms(variants.COMMITTED, "v2"),
        ),
        # What measuring the answers again costs: the replies of the saved answer run, and
        # the verdicts of the two judges on them, one for each answer each judge read.
        "answers": lines_of(ANSWERS / "answers.jsonl"),
        "verdicts": lines_of(ANSWERS / "support.jsonl") + lines_of(ANSWERS / "correct.jsonl"),
    }


# The golden set, which the documents start from


def test_the_golden_set_is_as_the_documents_describe_it() -> None:
    known = figures()
    (two_on_one_page,) = known["all_on_one_page"]
    others = in_words(known["any_of_several_pages"]).capitalize()
    sentence = (
        f"{known['one_piece']} have one piece of evidence. {others} others list passages on "
        "more than one page, any one of which answers the question, and "
        f"{in_words(len(known['all_on_one_page']))} needs two passages of one page. None needs "
        "passages on two pages."
    )

    assert known["all_on_two_pages"] == 0
    assert sentence in read(ADR)
    assert sentence in guide_section()
    # That one question has both passages in one chunk, which every run of the golden set
    # made with the new code finds. So All@k is Hit@k there, for every question.
    for run in variants.ORDER:
        rows = variants.rows(GOLDEN_RUNS, run)
        (rank,) = set(rows[two_on_one_page]["evidence_ranks"])
        assert rank is not None, run
        for question, row in rows.items():
            for k in ALL_CUTOFFS:
                assert row["grade"][f"all@{k}"] == row["grade"][f"hit@{k}"], (run, question, k)
        several = variants.summary(GOLDEN_RUNS, run)["needs_several_passages"]
        assert several["questions"] == len(known["all_on_one_page"]), run
    said = f"its one question that needs two passages, `{two_on_one_page}`, has both in one chunk."
    assert said in read(ADR)
    assert said in guide_section()
    assert (
        f"The golden set has {in_words(len(known['all_on_one_page']))} such question, so its runs "
        "have the block, for that question." in read(GUIDE)
    )


# The decision record


def test_the_tables_of_the_decision_record_are_the_saved_runs() -> None:
    text = read(ADR)

    assert table(text, header(variants.DEPTH_HEADER)) == [
        variants.depth_row(k) for k in variants.DEPTHS
    ]
    assert table(text, header(ADR_VARIANTS_HEADER)) == [
        [
            variants.LABELS[run],
            str(figures()["found"][run]),
            str(int(sum(variants.scores(GOLDEN_RUNS, run, "hit@5")))),
        ]
        for run in variants.ORDER
    ]


def decision_record_numbers() -> dict[str, list[str]]:
    """The numbers of every line of the decision record that holds one, in their order."""
    k = figures()
    both, one = k["both"], k["both"] + k["only_one"]
    written, confirmed, not_in_ten = k["written"], k["confirmed"], k["not_in_ten"]
    found, hit, lost, limits = k["found"], k["golden_hit"], k["lost"], variants.PAGE_LIMITS
    one_chunk, two_chunks, three_chunks = (k["by_limit"][limit] for limit in (1, 2, 3))
    rerank, constant = k["rerank"], k["constant_mrr"]
    slow, fast, slow_before, fast_before = k["milliseconds"]
    config = variants.summary(TWO, DEFAULT)["config"]
    term_and_rule = of_kind("term-and-rule")
    at_five, at_ten, more = (
        k["clear"][(TWO.name, run, metric)]
        for run, metric in (
            ("spread-1-in-5", "hit@5"),
            ("spread-1-in-10", "all@10"),
            ("candidates-100", "ndcg@10"),
        )
    )
    gains = [change for change in k["clear"].values() if change.difference.mean > 0]
    different = 2 * len(variants.VARIANTS) - len(variants.SAME_FIRST_FIVE)
    return {
        "- Date:": ["2026", "11", "24"],
        "**The test questions rest on one passage.**": [
            str(k["answerable"]),
            str(k["one_piece"]),
            "1",
            QUESTIONS_ISSUE,
        ],
        "- **A set of questions that need two passages": [str(written)],
        "- **A second reading.**": [str(confirmed), str(written)],
        "- **All@k beside Hit@k.**": ["1"],
        f"**What the default search does with the {confirmed}.**": [str(confirmed)],
        "- **Both passages are among the five results": [
            str(both),
            str(confirmed),
            k["share"],
            *k["interval"],
            str(one),
            str(k["only_one"]),
        ],
        "- **The kinds differ.**": [
            str(k["comparisons"]),
            str(len(of_kind("comparison"))),
            str(len(term_and_rule)),
            str(len(term_and_rule) - len(NAMES_ITS_TERM)),
        ],
        f"- **For {not_in_ten} of the {confirmed} questions a passage": [
            str(not_in_ten),
            str(confirmed),
            str(k["neither_in_ten"]),
        ],
        "**Fourteen changes to the search were tried": [str(confirmed)],
        "Compared with the default question by question": [
            TOP,
            str(confirmed),
            str(at_five.better),
            str(at_five.worse),
            str(at_ten.better),
            str(at_ten.worse),
            str(max(change.better + change.worse for change in gains)),
            setting("rrf-k-10", "rrf_k"),
            setting("candidates-100", "fusion_candidates"),
            str(confirmed),
        ],
        "- **The default search stays as it is**": [
            str(config["fusion_candidates"]),
            str(config["rrf_k"]),
        ],
        "- **The next step asks a model before the search**": [FOLLOW_UPS],
        "**Nothing found both passages clearly more often.**": [
            str(min(found.values())),
            str(max(found.values())),
            str(confirmed),
            str(found[DEFAULT]),
        ],
        "**The page limit is the one firm result": [
            str(len(k["with_the_passage"])),
            str(len(k["under_its_page"])),
            str(len(lost[1])),
            str(len(lost[1])),
            str(len(k["under_its_page"])),
            str(one_chunk.better),
            *(three(hit[limits[limit]]) for limit in (1, 2, 3, 4)),
            *(setting(limits[limit], "spread_per_page") for limit in (1, 2, 3, 4)),
            three(hit[DEFAULT]),
            *(str(len(lost[limit])) for limit in (4, 3, 2, 1)),
            str(confirmed),
            *(str(found[limits[limit]]) for limit in (1, 2, 3, 4)),
            str(found[DEFAULT]),
        ],
        "**The other changes move a few questions in either direction.**": [
            setting("candidates-50", "fusion_candidates"),
            setting("candidates-100", "fusion_candidates"),
            str(k["at_least_one"]["candidates-50"]),
            str(confirmed),
            str(k["at_least_one"][DEFAULT]),
            str(found["candidates-50"]),
            setting("candidates-10", "fusion_candidates"),
            setting("candidates-20", "fusion_candidates"),
            str(found["candidates-10"]),
            str(found["candidates-20"]),
            *(setting(run, "rrf_k") for run in ("rrf-k-20", "rrf-k-10", "rrf-k-5")),
            *(str(found[run]) for run in ("rrf-k-20", "rrf-k-10", "rrf-k-5")),
            str(found["rerank"]),
            str(rerank.better),
            str(rerank.worse),
            str(slow),
            str(fast),
            str(slow_before),
            str(fast_before),
        ],
        "**How far a reading of": [
            TOP,
            CLEAR,
            p_value(one_chunk, 6),
            p_value(two_chunks, 3),
            p_value(three_chunks, 3),
            str(three_chunks.worse),
            setting("rrf-k-10", "rrf_k"),
            unsigned(constant)[0],
            str(constant.better),
            str(constant.worse),
            p_value(constant, 3),
            setting("candidates-100", "fusion_candidates"),
            unsigned(more)[0],
            str(confirmed),
            str(more.better),
            str(more.worse),
            p_value(more, 3),
            str(different),
        ],
        f"For {not_in_ten} of the {confirmed} questions a passage": [
            str(not_in_ten),
            str(confirmed),
            str(k["neither_in_ten"]),
            str(k["reached"]),
            str(not_in_ten),
            str(k["reached_by"]["rerank"]),
            setting("candidates-100", "fusion_candidates"),
            str(k["reached_by"]["candidates-100"]),
            str(RETRIEVAL_DEPTH),
            FOLLOW_UPS,
            REWRITING,
        ],
        "- **Adopt the limit of two chunks for each page.**": [
            str(confirmed),
            str(found[limits[2]]),
            str(found[DEFAULT]),
            str(len(lost[2])),
        ],
        "- **Adopt the mildest limit, four chunks.**": [str(confirmed), str(len(lost[4]))],
        "- **More candidates.**": [
            setting("candidates-100", "fusion_candidates"),
            str(len(variants.lost(GOLDEN_RUNS, "candidates-100", "hit@5"))),
            unsigned(more)[0],
            str(confirmed),
            str(more.better),
            str(more.worse),
        ],
        "- **Adopt the fusion constant 10.**": [
            setting("rrf-k-10", "rrf_k"),
            unsigned(constant)[0],
            str(constant.better + constant.worse),
            str(confirmed),
            str(k["constant_33"].better),
            str(k["constant_33"].worse),
            str(config["rrf_k"]),
        ],
        "- **The reranker.**": [
            str(both_found("rerank", term_and_rule)),
            str(len(term_and_rule)),
            str(round(slow / fast)),
        ],
        "- **Give the model eight or ten passages now.**": [
            str(depth(8)),
            str(depth(10)),
            str(k["answers"]),
            str(k["verdicts"]),
            str(round(k["answers"] + k["verdicts"], -1)),
            str(A_DAY),
            str(confirmed),
        ],
        "- **Count any chunk of the right page": [str(k["by_page"]), str(both)],
        "- **New questions for anything that looks good later.**": [str(confirmed)],
        "- **A sounder rule for": [CLEAR],
        "- There is a second number beside Hit@5": [str(both), str(confirmed)],
        "- The golden set's scores are as they were.": [str(k["scores_repeated"])],
        "- `eval-retrieval` refuses a variant it could not measure": [str(RETRIEVAL_DEPTH)],
        "- Issue 78 stays open.": [QUESTIONS_ISSUE],
    }


def test_every_number_of_the_decision_record_is_worked_out_from_the_runs() -> None:
    assert_numbers(read(ADR), decision_record_numbers())


def test_what_the_numbers_of_the_decision_record_rest_on() -> None:
    """What the lines of numbers take for granted, checked on the runs themselves."""
    known, text = figures(), read(ADR)
    found, lost, clear = known["found"], known["lost"], known["clear"]
    one_chunk = known["by_limit"][1]
    term_and_rule = of_kind("term-and-rule")

    # Those that lose their passage with one chunk are among those under another chunk of
    # their page, and each limit loses what the milder one loses, and more.
    assert known["under_its_page"] <= known["with_the_passage"]
    assert len(known["with_the_passage"]) == known["golden_found"]
    assert lost[1] <= known["under_its_page"]
    assert lost[4] < lost[3] < lost[2] < lost[1]
    assert one_chunk.worse == len(lost[1])
    # "The best variants are one question ahead", and no difference on the 33 is clear.
    assert max(found.values()) - found[DEFAULT] == 1
    on_both = [run for run in variants.VARIANTS if (TWO.name, run, "all@5") in clear]
    assert f'{in_words(len(on_both))} variant differs clearly on "both in the top 5"' in text
    # Of the two scores of the table only Hit@5 on the golden set reads clear: for these
    # limits for each page, and each is a loss.
    on_hit = {
        limit: clear[(GOLDEN_RUNS.name, run, "hit@5")]
        for limit, run in variants.PAGE_LIMITS.items()
        if (GOLDEN_RUNS.name, run, "hit@5") in clear
    }
    assert all(change.difference.mean < 0 for change in on_hit.values())
    *before, last = (in_words(limit) for limit in on_hit)
    assert (
        f"for the limits of {', '.join(before)} and {last} chunks for each page, each a loss"
        in (text)
    )
    # Every other clear loss is of a limit for each page; in a variant's favour, three.
    losses = [key for key, change in clear.items() if change.difference.mean < 0]
    gains = [key for key, change in clear.items() if change.difference.mean > 0]
    assert {run for _, run, _ in losses} <= {
        *variants.PAGE_LIMITS.values(),
        *variants.SAME_FIRST_FIVE,
    }
    assert sorted(gains) == [
        (GOLDEN_RUNS.name, "rrf-k-10", "mrr@10"),
        (GOLDEN_RUNS.name, "rrf-k-10", "ndcg@10"),
        (TWO.name, "candidates-100", "ndcg@10"),
    ]
    assert f"In a variant's favour `eval-compare` reads {in_words(len(gains))}, each on" in text
    on_mrr, on_ndcg = (clear[(GOLDEN_RUNS.name, "rrf-k-10", m)] for m in ("mrr@10", "ndcg@10"))
    assert on_mrr == known["constant_mrr"]
    assert (unsigned(on_ndcg)[0], on_ndcg.better, on_ndcg.worse) == (
        unsigned(on_mrr)[0],
        on_mrr.better,
        on_mrr.worse,
    )
    assert "gains 0.003 on MRR@10 and on nDCG@10 on the golden set, each with" in text
    # The constant 10 moves no golden question's Hit@5; 100 candidates read clear once.
    moved_by_the_constant = known["constant_golden"].better + known["constant_golden"].worse
    assert f"It changes {in_words(moved_by_the_constant)} golden question's Hit@5" in text
    of_more = [key for key in clear if key[1] == "candidates-100"]
    assert f"Its {in_words(len(of_more))} clear reading, the gain of" in text
    # 50 and 100 candidates bring at least one passage for as many questions, and both for
    # the same questions as the default.
    assert known["at_least_one"]["candidates-50"] == known["at_least_one"]["candidates-100"]
    for run in ("candidates-50", "candidates-100"):
        assert not variants.lost(TWO, run, "all@5") | variants.gained(TWO, run, "all@5")
    # The two limits named as alternatives: two gains no golden question, and four moves
    # no passage of the 33.
    assert not variants.gained(GOLDEN_RUNS, variants.PAGE_LIMITS[2], "hit@5")
    assert variants.moved(TWO, variants.PAGE_LIMITS[4]) == (0, 0)
    # Both passages of a term-and-rule question: none in the default run, and only the
    # reranker finds them, of all fourteen, and for one question.
    assert both_found(DEFAULT, term_and_rule) == 0
    assert [run for run in variants.ORDER if both_found(run, term_and_rule)] == ["rerank"]
    (by_the_reranker,) = [q for q in term_and_rule if both_found("rerank", [q])]
    assert f"`term-and-rule` question, `{by_the_reranker}`, for 1 of the 8" in text
    for question in NAMES_ITS_TERM:
        assert f"(`{question}` names its term)" in text
    # What the default search gives the model, and what the record says stays.
    assert Settings().search_limit == 5
    assert "and five passages for the model" in text
    # The counts it writes as words. Every "fourteen" of the record is the number of
    # variants, and each is one of these.
    later, repeated = len(variants.ADDED_LATER), len(variants.REPEATED)
    on_the_headline = len(variants.VARIANTS) - len(variants.SAME_FIRST_FIVE)
    tried = in_words(len(variants.VARIANTS))
    for phrase in (
        f"**{tried.capitalize()} changes to the search were tried",
        f"None of the {tried} variants is adopted.",
        f"### Why none of the {tried}",
        f"none of the {tried} variants brings both passages into the first ten",
        f"{tried.capitalize()} variants were tried on the same 33 questions.",
    ):
        assert phrase in text, phrase
    assert text.lower().count("fourteen") == 5
    assert (
        f"{in_words(len(variants.VARIANTS) - later).capitalize()} were written down before the "
        f"first run. {in_words(later).capitalize()} were added after the first results" in text
    )
    assert f"{1 - 0.95**on_the_headline:.0%}" == "46%"
    assert f"{in_words(on_the_headline).capitalize()} different comparisons were made" in text
    assert "at least one of the twelve would do so 46% of the time, and not 5%" in text
    assert "That is no upper bound" in text
    assert f"{in_words(repeated).capitalize()} runs of it made with the new code" in text
    # The allowance the cost is weighed against is the one the README states.
    assert f"15 requests a minute and {A_DAY} a day" in read(README)


def test_the_decision_record_says_what_was_not_done() -> None:
    text = read(ADR)

    for sentence in (
        "No answer was written for them.",
        "no person has read the questions",
        "One model session wrote them, with both passages in front of it.",
        "Fourteen variants were tried on the same 33 questions.",
        "A model's reading is not a person's review",
        "Neither judge was checked on an answer that needs two passages.",
        "so the shares here are more likely too high than too low",
        "It was not done.",
    ):
        assert sentence in text, sentence
    for word in ("proves", "safe", "shows that"):
        assert word not in text.lower(), word
    for issue in (QUESTIONS_ISSUE, FOLLOW_UPS, CLEAR, REWRITING):
        assert f"]({ISSUE}{issue})" in text, issue
    for heading in (
        "## Context",
        "## Decision",
        "## Alternatives considered",
        "## What is still missing",
        "## Consequences",
    ):
        assert f"\n{heading}\n" in text, heading


def test_the_settings_are_named_as_the_code_has_them() -> None:
    settings = Settings()
    default = variants.summary(TWO, DEFAULT)["config"]

    # The number of candidates, the constant and reranking are settings for every command,
    # and the default run was made with their defaults. The limit for each page is not one.
    assert (settings.fusion_candidates, settings.rrf_k, settings.rerank) == (
        default["fusion_candidates"],
        default["rrf_k"],
        default["rerank"],
    )
    assert not [name for name in Settings.model_fields if "spread" in name]
    for path, said in (
        (ADR, "no setting turns it on, and no other command has it"),
        (GUIDE, "The limit for each page has no setting, and no other command has it."),
        (CHANGELOG, "which no setting and no other command has"),
    ):
        text = read(path)
        assert said in text, path.name
        for name in ("HANDBOOK_RAG_FUSION_CANDIDATES", "HANDBOOK_RAG_RRF_K"):
            assert name.removeprefix("HANDBOOK_RAG_").lower() in Settings.model_fields
            assert f"`{name}`" in text, (path.name, name)
    assert "rerank" in Settings.model_fields
    assert "`HANDBOOK_RAG_RERANK`" in read(ADR)
    assert "`HANDBOOK_RAG_RERANK`" in read(GUIDE)


# The evaluation guide


def searches_row(mode: str, runs: str) -> list[str]:
    """One of the three searches on the 33 questions, as the card's table gives it."""
    metrics = variants.summary(TWO, mode)["metrics"]
    both = metrics["all@5"]
    return [
        SEARCHES[mode],
        f"[`{mode}`]({runs}/{mode})",
        f"{both['mean']:.3f} ({both['low']:.3f} to {both['high']:.3f})",
        f"{metrics['hit@5']['mean']:.3f}",
        f"{metrics['all@10']['mean']:.3f}",
        f"{metrics['hit@10']['mean']:.3f}",
    ]


def guide_variants_row(run: str) -> list[str]:
    """A variant on both sets: the headline score of each, with its paired comparison."""
    two, golden = variants.two_passage_row(run), variants.golden_row(run)
    return [two[0], *two[2:6], *golden[2:6]]


def test_the_tables_of_the_guide_are_the_saved_runs() -> None:
    section = guide_section()

    assert table(section, header(SEARCHES_HEADER)) == [
        searches_row(mode, "../evals/runs/two-passage") for mode in SEARCHES
    ]
    assert table(section, header(variants.DEPTH_HEADER)) == [
        variants.depth_row(k) for k in variants.DEPTHS
    ]
    assert table(section, header(GUIDE_VARIANTS_HEADER)) == [
        guide_variants_row(run) for run in variants.ORDER
    ]
    # The card has the three searches in the same figures.
    assert table(read(CARD), "| Search | Run | Both in the top 5") == [
        searches_row(mode, "../runs/two-passage") for mode in SEARCHES
    ]


def guide_numbers() -> dict[str, list[str]]:
    """The numbers of every line of the guide's section that holds one, in their order."""
    k = figures()
    both, one = k["both"], k["both"] + k["only_one"]
    written, confirmed, not_in_ten = k["written"], k["confirmed"], k["not_in_ten"]
    found, hit, lost, limits = k["found"], k["golden_hit"], k["lost"], variants.PAGE_LIMITS
    constant, three_chunks = k["constant_mrr"], k["by_limit"][3]
    to_keyword = compare_runs(TWO / "sparse", TWO / DEFAULT).metrics["all@5"].change
    config = variants.summary(TWO, DEFAULT)["config"]
    kinds = ("comparison", "term-and-rule", "rule-and-exception", "process")
    more = k["clear"][(TWO.name, "candidates-100", "ndcg@10")]
    return {
        "Almost every golden question rests on one passage.": [
            str(k["answerable"]),
            str(k["one_piece"]),
            RECORD,
            QUESTIONS_ISSUE,
        ],
        f"- **{written} questions in the golden set's format": [
            str(written),
            *(str(k["kinds_written"][kind]) for kind in kinds),
        ],
        "- **The readers confirm": [str(confirmed), str(written)],
        "On the default search, Hit@5 is": [
            k["hit"],
            k["share"],
            str(both),
            str(confirmed),
            str(one),
            str(k["only_one"]),
        ],
        "A passage is credited by its quote": [str(k["by_page"]), str(confirmed), str(both)],
        "Measured on the full index": [str(confirmed)],
        "The kinds differ.": [
            str(k["comparisons"]),
            str(len(of_kind("comparison"))),
            str(len(of_kind("term-and-rule"))),
            str(len(of_kind("term-and-rule")) - len(NAMES_ITS_TERM)),
            *(str(size) for size in k["kind_sizes"]),
        ],
        "Looking further down finds more.": [str(confirmed)],
        "The first eight results hold both passages": [
            str(depth(8)),
            str(depth(10)),
            str(k["deeper"].better),
            *unsigned(k["deeper"]),
        ],
        f"For {not_in_ten} of the {confirmed} questions a passage": [
            str(not_in_ten),
            str(confirmed),
            str(k["neither_in_ten"]),
            str(k["reached"]),
            str(not_in_ten),
        ],
        "Fourteen changes to the search that need no request": [str(confirmed)],
        "- **Another number of candidates**": [str(config["fusion_candidates"])],
        "- **Another fusion constant**": [str(config["rrf_k"])],
        "- **No variant finds both passages clearly more often.**": [
            str(min(found.values())),
            str(max(found.values())),
            str(confirmed),
            str(found[DEFAULT]),
        ],
        "- **A limit for each page costs the golden set its passages.**": [
            *(three(hit[limits[limit]]) for limit in (1, 2, 3, 4)),
            *(setting(limits[limit], "spread_per_page") for limit in (1, 2, 3, 4)),
            three(hit[DEFAULT]),
            str(len(k["with_the_passage"])),
            str(len(k["under_its_page"])),
            str(len(lost[1])),
            str(len(lost[1])),
            str(len(k["under_its_page"])),
            str(k["by_limit"][1].better),
        ],
        "- **So none is adopted": [RECORD, FOLLOW_UPS, str(RETRIEVAL_DEPTH)],
        "The command refuses a variant that it could not measure": [str(RETRIEVAL_DEPTH)],
        "- **Many comparisons on the same questions.**": [
            TOP,
            "1",
            "0.95",
            str(len(variants.VARIANTS) - len(variants.SAME_FIRST_FIVE)),
        ],
        "- **The rule is generous when few questions change**": [
            CLEAR,
            str(three_chunks.worse),
            p_value(three_chunks, 3),
            setting("rrf-k-10", "rrf_k"),
            str(constant.better),
            str(constant.worse),
            p_value(constant, 3),
            setting("candidates-100", "fusion_candidates"),
            str(confirmed),
            str(more.better),
            str(more.worse),
            p_value(more, 3),
            str(confirmed),
            str(to_keyword.better),
        ],
        "The loss from a limit of one or two chunks": [
            p_value(k["by_limit"][1], 6),
            p_value(k["by_limit"][2], 3),
        ],
        "- **Nothing was tested on new questions**": [str(confirmed)],
        "- **What more passages would do to the answers.**": [
            str(k["late"]),
            str(confirmed),
            *(str(rank) for rank in k["late_ranks"]),
        ],
    }


def test_every_number_of_the_guides_section_is_worked_out_from_the_runs() -> None:
    assert_numbers(guide_section(), guide_numbers())


def test_what_the_guide_says_of_the_runs_elsewhere_is_worked_out_from_them() -> None:
    text = read(GUIDE)
    to_keyword = compare_runs(TWO / "sparse", TWO / DEFAULT).metrics["all@5"].change
    config = variants.summary(TWO, DEFAULT)["config"]
    cutoffs = variants.summary(TWO, DEFAULT)["needs_several_passages"]["found_in_top"]

    # Under "Comparing two variants": the card's comparison, as the example of the rule.
    line = line_starting(text, 'The rule for "clear" is generous when few questions change.')
    assert (to_keyword.is_clear, to_keyword.worse) == (True, 0)
    four = in_words(to_keyword.better).capitalize()
    assert f"{four} questions better and none worse reads as clear" in line
    assert numbers(line) == [p_value(to_keyword, 3), CLEAR]
    # Under "What a run writes".
    counts = line_starting(text, "- **Counts for the questions that need two or more passages**")
    assert numbers(counts) == [str(cutoff) for cutoff in ALL_CUTOFFS]
    assert sorted(map(int, cutoffs)) == list(ALL_CUTOFFS)
    variant = line_starting(text, "- **The variant.**")
    assert numbers(variant) == [str(config["fusion_candidates"]), str(config["rrf_k"])]
    checks = line_starting(text, "The two-passage questions are held to their files by tests")
    assert numbers(checks) == [RECORD]
    repeated = in_words(len(variants.REPEATED)).capitalize()
    assert f"{repeated} of the golden set's runs of the variants" in checks


def test_the_counts_the_guides_section_writes_as_words_are_those_of_the_runs() -> None:
    section, later = guide_section(), len(variants.ADDED_LATER)
    on_the_headline = len(variants.VARIANTS) - len(variants.SAME_FIRST_FIVE)
    gains = [change for change in figures()["clear"].values() if change.difference.mean > 0]
    tried = in_words(len(variants.VARIANTS))

    # Every "fourteen" of the section is the number of variants, and each is one of these.
    for phrase in (
        f"### {tried.capitalize()} variants of the search, and none adopted",
        f"{tried.capitalize()} changes to the search that need no request to a model",
        f"{tried} variants, of which two repeat two others",
        f"{tried} variants were tried on the same 33 questions",
    ):
        assert phrase in section, phrase
    assert section.lower().count("fourteen") == 4
    assert (
        f"{in_words(len(variants.VARIANTS) - later).capitalize()} were written down before the "
        f"first run. {in_words(later).capitalize()} were added after the first results" in section
    )
    controls = in_words(figures()["controls"]).capitalize()
    assert f"{controls} questions of the golden set were mixed in as controls" in section
    assert f"{in_words(on_the_headline).capitalize()} different comparisons were made" in section
    assert f"= {1 - 0.95**on_the_headline:.0%} of the time, and not 5%" in section
    assert "That is no upper bound" in section
    assert f"The {in_words(len(gains))} readings in a variant's favour" in section
    for question in NAMES_ITS_TERM:
        assert f"(`{question}` names its term)" in section


def test_the_guide_names_what_a_run_writes_as_the_runs_have_it() -> None:
    text = read(GUIDE)
    row = next(iter(variants.rows(TWO, DEFAULT).values()))
    about = variants.summary(TWO, DEFAULT)
    spread = variants.summary(TWO, "spread-2-in-5")["config"]

    for name in ("evidence_ranks", "page_ranks"):
        assert name in row
        assert f"`{name}`" in text, name
    for cutoff in ALL_CUTOFFS:
        assert f"all@{cutoff}" in row["grade"]
    assert "All@3, All@5 and All@10" in text
    several = about["needs_several_passages"]
    for name in ("found_in_top", "all_found_by_rank", "not_all_retrieved", "pages_in_top"):
        assert name in several
        assert f"`{name}`" in text, name
    assert several["retrieved"] == RETRIEVAL_DEPTH
    assert "`retrieved` the number of results the run asked for" in text
    assert "`needs_several_passages`" in text
    assert about["questions_left_out"] == ["t025"]
    assert "`questions_left_out`" in text
    assert (spread["spread_per_page"], spread["spread_within"]) == (2, 5)
    assert "`spread_per_page` and `spread_within`" in text
    for option in (
        "--reviewed-only",
        "--reasons",
        "--fusion-candidates",
        "--rrf-k",
        "--spread-per-page",
        "--spread-within",
    ):
        assert option in text, option
    # The committed runs of the golden set were made before any of this.
    committed = json.loads(
        (variants.COMMITTED / "v2" / "results.jsonl").read_text().splitlines()[0]
    )
    assert "evidence_ranks" not in committed
    assert "all@5" not in committed["grade"]


def test_the_guide_says_what_was_not_done() -> None:
    section = guide_section()

    for sentence in (
        "No answer was written for these questions.",
        "No person has read the questions.",
        "with both passages in front of it",
        "fourteen variants were tried on the same 33 questions",
        "A model's check is not a person's review",
        "so the shares here are more likely too high than too low",
    ):
        assert sentence in section, sentence
    for word in ("proves", "safe", "shows that"):
        assert word not in section.lower(), word
    assert f"[issue 80]({ISSUE}80)" in section
    assert f"[issue 80]({ISSUE}80)" in read(GUIDE).split("\n## The retrieval gate in CI\n")[0]


# The card and the results file, which have tests of their own


def test_the_card_and_the_results_give_the_figures_the_other_documents_give() -> None:
    known = figures()
    both, one, confirmed = known["both"], known["both"] + known["only_one"], known["confirmed"]
    hit = known["golden_hit"]

    for wording in (
        f"for {both} of the {confirmed} questions.** It brings at least one for {one}.",
        f"For {known['not_in_ten']} questions a passage is not among the first ten results",
        f"For the first five results it gives {known['by_page']} of the {confirmed} in this run, "
        f"against {both} by quote.",
        f"{known['share']} ({known['interval'][0]} to {known['interval'][1]})",
    ):
        assert wording in read(CARD), wording
    for wording in (
        f"for {both} of the {confirmed} confirmed questions",
        f"for {known['not_in_ten']} of them a passage is not among",
        f"{three(hit['spread-1-in-5'])} with one chunk of a page allowed and "
        f"{three(hit['spread-2-in-5'])} with two",
    ):
        assert wording in read(RESULTS), wording


# The changelog, the roadmap and the architecture page


def test_the_changelog_and_the_roadmap_give_the_figures_of_the_runs() -> None:
    known = figures()
    both, one = known["both"], known["both"] + known["only_one"]
    written, confirmed, hit = known["written"], known["confirmed"], known["golden_hit"]
    changelog = read(CHANGELOG).split("## [Unreleased]")[1].split("\n## [")[0]
    roadmap = read(ROADMAP)
    questions = line_starting(changelog, "- The two-passage questions, `evals/two-passage/`")
    scores = line_starting(changelog, "- `eval-retrieval` scores All@3, All@5 and All@10")
    options = line_starting(changelog, "- Variants of hybrid search for one run")
    line = line_starting(roadmap, "- **Questions that need two passages**")

    assert numbers(questions) == [str(written), str(confirmed), QUESTIONS_ISSUE, RECORD]
    assert numbers(scores) == [str(both), str(confirmed), str(one), QUESTIONS_ISSUE, RECORD]
    assert numbers(options) == [
        str(confirmed),
        three(hit[DEFAULT]),
        three(hit["spread-1-in-5"]),
        three(hit["spread-2-in-5"]),
        QUESTIONS_ISSUE,
        RECORD,
    ]
    assert f"{in_words(len(variants.VARIANTS)).capitalize()} variants were run" in options
    assert "\n[#78]: https://github.com/MumukshuPanchal/handbook-rag/issues/78\n" in read(CHANGELOG)
    assert "\n[ADR 0021]: docs/adr/0021-two-passage-questions-and-the-search-as-it-is.md" in read(
        CHANGELOG
    )
    assert numbers(line) == [
        QUESTIONS_ISSUE,
        str(written),
        str(confirmed),
        str(both),
        str(confirmed),
        str(one),
        RECORD,
        str(known["not_in_ten"]),
        str(confirmed),
        FOLLOW_UPS,
    ]
    assert f"{in_words(len(variants.VARIANTS)).capitalize()} changes to the search" in line
    # It stands among what was started since 1.0 and is not finished: issue 78 stays open.
    assert roadmap.index(line) > roadmap.index("Started since 1.0, and not finished:")
    assert "none was adopted" in line
    assert "no person has read the questions" in line


def test_the_architecture_page_names_the_new_parts() -> None:
    text = read(ARCHITECTURE)

    assert (ROOT / "src" / "handbook_rag" / "retrieve" / "spread.py").is_file()
    assert (ROOT / "src" / "handbook_rag" / "evals" / "two_passage.py").is_file()
    assert "a rule that spreads results over pages,\n" in text
    assert "which only an evaluation run can turn on" in text
    assert "the checks of the two-passage questions" in text
    assert "golden set, the two-passage questions, labelled answers" in text
    assert "- **Two-passage questions.**" in text
    assert "| All@k |" in text


def test_every_link_of_the_new_documents_leads_somewhere() -> None:
    for path in (ADR, GUIDE, ROADMAP, ARCHITECTURE):
        text = read(path)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://")):
                continue
            file, _, anchor = target.partition("#")
            linked = (path.parent / file) if file else path
            assert linked.exists(), (path.name, target)
            if anchor:
                assert anchor in anchors(read(linked)), (path.name, target)
