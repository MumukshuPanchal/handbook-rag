"""The results of the search variants are the saved runs.

``evals/two-passage/search-variants.md`` reports what fourteen variants of hybrid search do
with the questions that need two passages, and with the golden set (issue 78). These
tests read every figure of that file from the runs under ``evals/runs/two-passage`` and
``evals/runs/search-variants-golden``, and every difference from the paired comparison
that ``eval-compare`` makes of two of those runs. A figure changed by hand fails here,
and so does a run made again without the file being brought up to date.

They also hold what makes the runs comparable: each variant differs from the default in
the one setting its name says, every run scored the same questions with the same labels,
and where a committed run of the golden set measured the same search, the new run
repeats it score for score.

What no test can hold is what the figures are worth. The file says that itself: 33
questions, written by a model, and fourteen variants tried on the same 33. Nor can a test
show that the list of variants was fixed before the first run: that is the account of the
session that made the runs.
"""

import functools
import json
import re
import statistics
from math import comb
from pathlib import Path
from typing import Any

from handbook_rag.evals.compare import compare_runs
from handbook_rag.evals.gate import load_gate_config
from handbook_rag.evals.metrics import Comparison, paired_difference
from handbook_rag.evals.two_passage import load_reasons
from tests.test_readme_results import table

ROOT = Path(__file__).parents[2]
RESULTS = ROOT / "evals" / "two-passage" / "search-variants.md"
CARD = ROOT / "evals" / "two-passage" / "README.md"
REASONS = ROOT / "evals" / "two-passage" / "why-two.jsonl"
TWO = ROOT / "evals" / "runs" / "two-passage"
GOLDEN = ROOT / "evals" / "runs" / "search-variants-golden"
COMMITTED = ROOT / "evals" / "runs" / "retrieval"
DEFAULT = "hybrid"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
# Each variant: how the tables name it, and the settings its run must differ in from the
# run of the default search. In the order of the tables.
VARIANTS: dict[str, tuple[str, dict[str, object]]] = {
    "spread-1-in-5": (
        "At most 1 chunk of a page among the first 5",
        {"spread_per_page": 1, "spread_within": 5},
    ),
    "spread-2-in-5": (
        "At most 2 chunks of a page among the first 5",
        {"spread_per_page": 2, "spread_within": 5},
    ),
    "spread-3-in-5": (
        "At most 3 chunks of a page among the first 5 (added later)",
        {"spread_per_page": 3, "spread_within": 5},
    ),
    "spread-4-in-5": (
        "At most 4 chunks of a page among the first 5 (added later)",
        {"spread_per_page": 4, "spread_within": 5},
    ),
    "spread-1-in-10": (
        "At most 1 chunk of a page among the first 10",
        {"spread_per_page": 1, "spread_within": 10},
    ),
    "spread-2-in-10": (
        "At most 2 chunks of a page among the first 10",
        {"spread_per_page": 2, "spread_within": 10},
    ),
    "candidates-10": ("10 candidates from each search", {"fusion_candidates": 10}),
    "candidates-20": ("20 candidates from each search", {"fusion_candidates": 20}),
    "candidates-50": ("50 candidates from each search", {"fusion_candidates": 50}),
    "candidates-100": ("100 candidates from each search", {"fusion_candidates": 100}),
    "rrf-k-20": ("Fusion constant 20", {"rrf_k": 20}),
    "rrf-k-10": ("Fusion constant 10", {"rrf_k": 10}),
    "rrf-k-5": ("Fusion constant 5", {"rrf_k": 5}),
    "rerank": (
        "The 30 merged results reranked by the cross-encoder",
        {"rerank": True, "rerank_model": RERANKER, "rerank_candidates": 30},
    ),
}
ORDER = [DEFAULT, *VARIANTS]
LABELS = {
    DEFAULT: "The default: 30 candidates from each search, constant 60",
    **{run: label for run, (label, _) in VARIANTS.items()},
}
# The list of variants was fixed before the first run. These two were added after the
# first results had been seen, and the file says so wherever it names them.
ADDED_LATER = ("spread-3-in-5", "spread-4-in-5")
# The runs with a limit for each page among the first five, by the limit.
PAGE_LIMITS = {limit: f"spread-{limit}-in-5" for limit in (1, 2, 3, 4)}
# The pairs that have the same first five results by construction: the rule fills the
# first places one by one, so a limit that reaches further fills the first five alike.
SAME_FIRST_FIVE = {"spread-1-in-10": "spread-1-in-5", "spread-2-in-10": "spread-2-in-5"}
# The committed runs of the golden set that measured the same search as a new run.
REPEATED = {
    DEFAULT: "v2",
    "rerank": "v3",
    "rrf-k-20": "v8",
    "rrf-k-10": "v9",
    "rrf-k-5": "v10",
    "candidates-100": "v11",
}
DEPTHS = (5, 8, 10)

TWO_HEADER = [
    "Variant",
    "Run",
    "Both in the top 5",
    "Change against the default (95% interval)",
    "Better / worse",
    "Reading",
    "At least one in the top 5",
    "Both in the top 10",
]
GOLDEN_HEADER = [
    "Variant",
    "Run",
    "Hit@5",
    "Change against the default (95% interval)",
    "Better / worse",
    "Reading",
    "MRR@10",
    "Change of MRR@10 (95% interval)",
    "Reading of MRR@10",
    "Recall@5",
]
MOVED_HEADER = [
    "Chunks of a page allowed among the first 5",
    "Of the 66: left the first five",
    "Of the 66: came into them",
    "Of the 138: left the first five",
    "Of the 138: came into them",
]
DEPTH_HEADER = [
    "In the first",
    "Both passages, of the 33",
    "At least one, of the 33",
    "The passage, of the 132 golden questions",
]


def results() -> str:
    return RESULTS.read_text(encoding="utf-8")


@functools.cache
def summary(flow: Path, run: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((flow / run / "summary.json").read_text(encoding="utf-8"))
    return loaded


@functools.cache
def rows(flow: Path, run: str) -> dict[str, dict[str, Any]]:
    """The rows of a saved run, by question id."""
    lines = (flow / run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["prompt_id"]: row for row in map(json.loads, lines)}


def scores(flow: Path, run: str, metric: str) -> list[float]:
    return [float(row["grade"][metric]) for row in rows(flow, run).values()]


def share(count: int, total: int) -> str:
    return f"{count} of {total} ({count / total:.3f})"


def change(comparison: Comparison) -> str:
    """A paired difference as ``eval-compare`` prints it."""
    difference = comparison.difference
    return f"{difference.mean:+.3f} ({difference.low:+.3f} to {difference.high:+.3f})"


def reading(comparison: Comparison) -> str:
    return "clear" if comparison.is_clear else "unclear"


@functools.cache
def against_default(flow: Path, run: str, metric: str) -> Comparison:
    """One score of a run against the default's, question by question.

    This is the comparison ``eval-compare`` makes, of one score instead of all ten: the
    questions in the order of their ids, and the difference of each. A test below holds
    it to the command's own.
    """
    default, variant = rows(flow, DEFAULT), rows(flow, run)
    questions = sorted(default)
    return paired_difference(
        [default[question]["grade"][metric] for question in questions],
        [variant[question]["grade"][metric] for question in questions],
    )


def sign_test(better: int, worse: int) -> float:
    """The exact two-sided sign test: how likely a split this uneven is from a fair coin."""
    changed = better + worse
    tail = sum(comb(changed, count) for count in range(min(better, worse) + 1))
    # Each of the 2 ** changed ways the questions could have fallen is as likely as another.
    return min(1.0, 2 * tail / (1 << changed))


def in_first(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def found(ranks: list[int | None], k: int) -> int:
    """How many of a question's passages came among the first k results."""
    return sum(in_first(rank, k) for rank in ranks)


def lost(flow: Path, run: str, metric: str) -> set[str]:
    """The questions that scored 1 in the default run and 0 in this one."""
    default, variant = rows(flow, DEFAULT), rows(flow, run)
    return {
        question
        for question in default
        if variant[question]["grade"][metric] < default[question]["grade"][metric]
    }


def gained(flow: Path, run: str, metric: str) -> set[str]:
    default, variant = rows(flow, DEFAULT), rows(flow, run)
    return {
        question
        for question in default
        if variant[question]["grade"][metric] > default[question]["grade"][metric]
    }


def two_passage_row(run: str) -> list[str]:
    """One row of the table of the 33 questions, from the run's files."""
    in_top = summary(TWO, run)["needs_several_passages"]["found_in_top"]
    both, one = in_top["5"]["all"], in_top["5"]["some"]
    cells = [LABELS[run], f"[`{run}`](../runs/two-passage/{run})", share(both, 33)]
    if run == DEFAULT:
        cells += ["", "", ""]
    else:
        compared = against_default(TWO, run, "all@5")
        cells += [change(compared), f"{compared.better} / {compared.worse}", reading(compared)]
    return [*cells, share(both + one, 33), share(in_top["10"]["all"], 33)]


def golden_row(run: str) -> list[str]:
    """One row of the table of the golden set, from the run's files."""
    metrics = summary(GOLDEN, run)["metrics"]
    cells = [
        LABELS[run],
        f"[`{run}`](../runs/search-variants-golden/{run})",
        share(int(sum(scores(GOLDEN, run, "hit@5"))), 132),
    ]
    mrr, recall = f"{metrics['mrr@10']['mean']:.3f}", f"{metrics['recall@5']['mean']:.3f}"
    if run == DEFAULT:
        return [*cells, "", "", "", mrr, "", "", recall]
    of_hit, of_mrr = against_default(GOLDEN, run, "hit@5"), against_default(GOLDEN, run, "mrr@10")
    return [
        *cells,
        change(of_hit),
        f"{of_hit.better} / {of_hit.worse}",
        reading(of_hit),
        mrr,
        change(of_mrr),
        reading(of_mrr),
        recall,
    ]


def passages(flow: Path, run: str) -> list[int | None]:
    """The rank of every passage of every question, in the order of the run's rows."""
    return [rank for row in rows(flow, run).values() for rank in row["evidence_ranks"]]


def moved(flow: Path, run: str) -> tuple[int, int]:
    """The passages that left the first five results against the default, and that came in."""
    pairs = list(zip(passages(flow, DEFAULT), passages(flow, run), strict=True))
    left = sum(in_first(before, 5) and not in_first(after, 5) for before, after in pairs)
    came = sum(in_first(after, 5) and not in_first(before, 5) for before, after in pairs)
    return left, came


def moved_row(limit: int) -> list[str]:
    run = PAGE_LIMITS[limit]
    return [str(limit), *map(str, moved(TWO, run)), *map(str, moved(GOLDEN, run))]


def depth_row(k: int) -> list[str]:
    """What the default search's two saved runs hold in their first k results."""
    two = [row["evidence_ranks"] for row in rows(TWO, DEFAULT).values()]
    golden = [row["evidence_ranks"] for row in rows(GOLDEN, DEFAULT).values()]
    return [
        str(k),
        share(sum(found(ranks, k) == len(ranks) for ranks in two), 33),
        share(sum(found(ranks, k) > 0 for ranks in two), 33),
        share(sum(found(ranks, k) > 0 for ranks in golden), 132),
    ]


def median_ms(flow: Path, run: str) -> int:
    seconds = [float(row["latency_s"]) for row in rows(flow, run).values()]
    return round(1000 * statistics.median(seconds))


# The runs


def test_the_runs_are_the_default_and_the_fourteen_variants_on_both_sets() -> None:
    def saved(flow: Path) -> set[str]:
        return {path.parent.name for path in flow.glob("*/summary.json")}

    assert len(VARIANTS) == 14
    assert set(ADDED_LATER) < set(VARIANTS)
    # The golden set's runs of the variants have a directory of their own. The committed
    # runs of the golden set are elsewhere and are not among them.
    assert saved(GOLDEN) == set(ORDER)
    assert saved(TWO) == {*ORDER, "dense", "sparse"}
    assert GOLDEN != COMMITTED
    for flow, scored in ((TWO, 33), (GOLDEN, 132)):
        for run in ORDER:
            about = summary(flow, run)
            assert (about["questions_scored"], about["questions_failed"]) == (scored, 0), run
            assert (flow / run / "errors.jsonl").read_text(encoding="utf-8") == "", run
    for run in ORDER:
        # The disputed question is left out of every run of the 33, and the golden set
        # has no question to leave out.
        assert summary(TWO, run)["questions_left_out"] == ["t025"], run
        assert "questions_left_out" not in summary(GOLDEN, run), run
    text = results()
    assert "fourteen variants were tried on those same 33" in text
    assert "These twelve were written down before the first run." in text
    assert 'and are marked "added later": m = 3 and m = 4 for k = 5.' in text
    for run in VARIANTS:
        assert ("(added later)" in LABELS[run]) == (run in ADDED_LATER), run


def test_each_variant_differs_from_the_default_in_the_setting_its_name_says() -> None:
    for flow in (TWO, GOLDEN):
        default = summary(flow, DEFAULT)["config"]
        assert (default["fusion_candidates"], default["rrf_k"], default["rerank"]) == (
            30,
            60,
            False,
        )
        assert not [key for key in default if key.startswith("spread")]
        for run, (_, changed) in VARIANTS.items():
            assert summary(flow, run)["config"] == {**default, **changed}, (flow.name, run)
    # Both sets were searched in the same index, with the same models.
    assert summary(TWO, DEFAULT)["config"] == summary(GOLDEN, DEFAULT)["config"]


def test_the_options_the_file_gives_for_a_run_are_the_settings_its_run_recorded() -> None:
    text = results()
    # How the file says a run's name maps to its options, and each option to a setting.
    patterns = {
        r"spread-(\d+)-in-(\d+)": ("spread_per_page", "spread_within"),
        r"candidates-(\d+)": ("fusion_candidates",),
        r"rrf-k-(\d+)": ("rrf_k",),
    }

    assert "`spread-M-in-K` has `--spread-per-page M --spread-within K`" in text
    assert "`candidates-N` has `--fusion-candidates N`" in text
    assert "`rrf-k-K` has `--rrf-k K`" in text
    assert "`rerank` has `--rerank` in place of `--no-rerank`" in text
    assert "The run `hybrid` has none of them." in text
    for run, (_, changed) in VARIANTS.items():
        if run == "rerank":
            assert changed["rerank"] is True
            continue
        (settings, numbers) = next(
            (settings, match.groups())
            for pattern, settings in patterns.items()
            if (match := re.fullmatch(pattern, run))
        )
        assert changed == dict(zip(settings, map(int, numbers), strict=True)), run
    # The command the file shows in full is the one of its run.
    shown = (
        "--mode hybrid --no-rerank --variant spread-2-in-5 --spread-per-page 2 --spread-within 5"
    )
    assert text.count(shown) == 2
    assert "--dataset evals/two-passage/questions.jsonl --reviewed-only" in text
    assert "--output evals/runs/search-variants-golden --mode hybrid" in text


def test_every_run_scored_the_questions_of_the_default_with_the_same_labels() -> None:
    for flow in (TWO, GOLDEN):
        default, about = rows(flow, DEFAULT), summary(flow, DEFAULT)
        for run in VARIANTS:
            scored = rows(flow, run)
            assert list(scored) == list(default), (flow.name, run)
            assert all(
                scored[question]["fingerprint"] == default[question]["fingerprint"]
                and scored[question]["status"] == "ok"
                for question in default
            ), (flow.name, run)
            assert summary(flow, run)["labels_fingerprint"] == about["labels_fingerprint"]


def test_a_golden_run_of_a_search_that_was_measured_before_repeats_the_committed_run() -> None:
    for run, committed in REPEATED.items():
        before, again = rows(COMMITTED, committed), rows(GOLDEN, run)
        assert summary(COMMITTED, committed)["config"] == summary(GOLDEN, run)["config"], run
        assert list(before) == list(again), run
        repeated = 0
        for question, row in before.items():
            assert row["fingerprint"] == again[question]["fingerprint"], (run, question)
            for metric, score in row["grade"].items():
                assert again[question]["grade"][metric] == score, (run, question, metric)
                repeated += 1
        # Seven scores for each of the 132 questions. The committed runs were made before
        # the scores All@k existed, and are as they were.
        assert repeated == 924, run
        assert "evidence_ranks" not in next(iter(before.values())), committed
    text = results()
    assert (
        "**Six of the fifteen runs of the golden set measure a search that a committed run "
        "had measured before**" in text
    )
    for run, committed in REPEATED.items():
        assert f"`{run}` is [`{committed}`](../runs/retrieval/{committed})" in text, run
    assert "All 924 scores of each are the same in the new run." in text


def test_the_runs_with_10_and_20_candidates_give_the_figures_of_issue_51() -> None:
    # As the issue words them, of two runs of the golden set that were not committed.
    quoted = {
        "candidates-10": ["125 of 132 (0.947)", "-0.015 (-0.045 to +0.015)", "1 / 3", "unclear"],
        "candidates-20": ["126 of 132 (0.955)", "-0.008 (-0.023 to +0.000)", "0 / 1", "unclear"],
    }

    for run, figures in quoted.items():
        assert golden_row(run)[2:6] == figures, run
    assert (
        "**The runs with 10 and 20 candidates give the figures that issue 51 quotes** from two "
        "runs that were made and never committed: Hit@5 0.947 and 0.955 on the golden set."
        in results()
    )


def test_the_default_run_of_the_33_is_the_run_the_card_reports() -> None:
    card_row = table(CARD.read_text(encoding="utf-8"), "| Search | Run | Both in the top 5")[0]
    metrics = summary(TWO, DEFAULT)["metrics"]

    assert card_row[1] == "[`hybrid`](../runs/two-passage/hybrid)"
    assert card_row[2].startswith(f"{metrics['all@5']['mean']:.3f} ")
    assert two_passage_row(DEFAULT)[2] == "12 of 33 (0.364)"
    assert two_passage_row(DEFAULT)[6:] == ["26 of 33 (0.788)", "17 of 33 (0.515)"]
    assert golden_row(DEFAULT)[2] == "127 of 132 (0.962)"
    assert "for 12 of the 33 confirmed questions ([the card](README.md)" in results()
    assert (
        "**The run of the default search on the 33 questions is the one the card reports.**"
        in results()
    )


# The tables, and what is said under each


def test_the_table_of_the_33_questions_is_the_saved_runs_and_their_paired_comparisons() -> None:
    header = "| " + " | ".join(TWO_HEADER) + " |"
    assert header in results().splitlines()

    assert table(results(), header) == [two_passage_row(run) for run in ORDER]


def test_what_is_said_of_the_33_questions_is_what_their_table_holds() -> None:
    text = results()
    both = {run: int(sum(scores(TWO, run, "all@5"))) for run in ORDER}
    compared = {run: against_default(TWO, run, "all@5") for run in VARIANTS}

    assert both[DEFAULT] == 12
    assert (min(both.values()), max(both.values())) == (8, 13)
    assert not [run for run, comparison in compared.items() if comparison.is_clear]
    assert (
        "The counts run from 8 to 13 of 33, and none of the fourteen differences is clear." in text
    )
    # The runs with 13: the limit of two for each page, whatever its reach, and 20 candidates.
    assert {run for run, count in both.items() if count == 13} == {
        "spread-2-in-5",
        "spread-2-in-10",
        "candidates-20",
    }
    assert gained(TWO, "spread-2-in-5", "all@5") == {"t007"}
    assert gained(TWO, "candidates-20", "all@5") == {"t021"}
    assert not lost(TWO, "spread-2-in-5", "all@5") | lost(TWO, "candidates-20", "all@5")
    assert (
        "The variants with 13 are one question ahead of the default: `t007` with two chunks "
        "of a page allowed, and `t021` with 20 candidates." in text
    )
    # The reranker changes more questions than any other variant.
    changed = {run: comparison.better + comparison.worse for run, comparison in compared.items()}
    assert (compared["rerank"].better, compared["rerank"].worse) == (4, 4)
    assert max(changed, key=lambda run: changed[run]) == "rerank"
    assert "The reranker changes the most: 4 questions better and 4 worse." in text
    why = load_reasons(REASONS)
    of_a_term = {
        run: sum(
            int(row["grade"]["all@5"])
            for question, row in rows(TWO, run).items()
            if why[question].kind == "term-and-rule"
        )
        for run in ORDER
    }
    assert sum(reason.kind == "term-and-rule" for reason in why.values()) == 8
    assert of_a_term == {**dict.fromkeys(ORDER, 0), "rerank": 1}
    assert (
        "It is also the only variant that finds both passages of a `term-and-rule` question "
        "in its first five, for 1 of the 8." in text
    )


def test_the_table_of_the_golden_set_is_the_saved_runs_and_their_paired_comparisons() -> None:
    header = "| " + " | ".join(GOLDEN_HEADER) + " |"
    assert header in results().splitlines()

    assert table(results(), header) == [golden_row(run) for run in ORDER]


def test_what_is_said_of_the_golden_set_is_what_its_table_holds() -> None:
    text = results()
    hit = {run: summary(GOLDEN, run)["metrics"]["hit@5"]["mean"] for run in ORDER}
    compared = {run: against_default(GOLDEN, run, "hit@5") for run in VARIANTS}

    limited = ", ".join(f"{hit[PAGE_LIMITS[limit]]:.3f}" for limit in (1, 2, 3))
    assert (
        f"Hit@5 is {limited} and {hit[PAGE_LIMITS[4]]:.3f} with 1, 2, 3 and 4 chunks allowed, "
        f"against {hit[DEFAULT]:.3f}." in text
    )
    assert [compared[PAGE_LIMITS[limit]].is_clear for limit in (1, 2, 3, 4)] == [
        True,
        True,
        True,
        False,
    ]
    assert "The first three differences are clear." in text
    others = [run for run in VARIANTS if not run.startswith("spread")]
    hits = {run: int(sum(scores(GOLDEN, run, "hit@5"))) for run in ORDER}
    assert not [run for run in others if compared[run].is_clear]
    assert (min(hits[run] for run in others), max(hits[run] for run in others)) == (124, 127)
    assert hits[DEFAULT] == 127
    assert (
        "No other variant is clearly different from the default on Hit@5, and none is ahead "
        "of it: they find the passage for 124 to 127 of the 132 questions, where the default "
        "finds it for 127." in text
    )


def test_spreading_within_ten_has_the_first_five_results_of_spreading_within_five() -> None:
    at_five = ("hit@1", "hit@3", "hit@5", "all@3", "all@5", "recall@5")
    for flow in (TWO, GOLDEN):
        for wider, narrower in SAME_FIRST_FIVE.items():
            of_wider, of_narrower = rows(flow, wider), rows(flow, narrower)
            for question in of_wider:
                for metric in at_five:
                    assert (
                        of_wider[question]["grade"][metric]
                        == of_narrower[question]["grade"][metric]
                    ), (flow.name, wider, question, metric)
                for ranks in ("evidence_ranks", "page_ranks"):
                    assert [
                        rank if in_first(rank, 5) else None for rank in of_wider[question][ranks]
                    ] == [
                        rank if in_first(rank, 5) else None for rank in of_narrower[question][ranks]
                    ], (flow.name, wider, question)
    for wider, narrower in SAME_FIRST_FIVE.items():
        assert two_passage_row(wider)[2:7] == two_passage_row(narrower)[2:7]
        assert golden_row(wider)[2:6] == golden_row(narrower)[2:6]
    assert (
        "The first five results are the same for both k, so the two runs with k = 10 say "
        "something new only about ranks 6 to 10." in results()
    )


def test_the_table_of_what_a_limit_moves_is_counted_from_the_rank_of_every_passage() -> None:
    header = "| " + " | ".join(MOVED_HEADER) + " |"
    assert header in results().splitlines()
    text = results()

    assert table(text, header) == [moved_row(limit) for limit in PAGE_LIMITS]
    # The passages the table counts, and how many the default has in its first five.
    two, golden = passages(TWO, DEFAULT), passages(GOLDEN, DEFAULT)
    assert (len(two), sum(in_first(rank, 5) for rank in two)) == (66, 38)
    assert (len(golden), sum(in_first(rank, 5) for rank in golden)) == (138, 129)
    assert (
        "The 33 questions have 66 of them, of which the default search has 38 in its first "
        "five results. The golden questions have 138, of which it has 129." in text
    )
    assert moved(TWO, PAGE_LIMITS[4]) == (0, 0)
    assert "no passage of the 33 questions enters or leaves the first five" in text


def test_the_golden_questions_that_lose_their_passage_are_those_the_file_describes() -> None:
    text = results()
    default = rows(GOLDEN, DEFAULT)
    with_the_passage = {q for q, row in default.items() if row["grade"]["hit@5"] == 1.0}
    # A chunk of the passage's own page stands above a passage that is in the first five.
    under_its_page = {
        question
        for question, row in default.items()
        if any(
            in_first(of_quote, 5) and of_page < of_quote
            for of_quote, of_page in zip(row["evidence_ranks"], row["page_ranks"], strict=True)
        )
    }
    losses = {limit: lost(GOLDEN, run, "hit@5") for limit, run in PAGE_LIMITS.items()}

    assert (len(with_the_passage), len(under_its_page)) == (127, 23)
    assert under_its_page <= with_the_passage
    assert (
        "the passage is among the first five results for 127 questions, and for 23 of those "
        "another chunk of the same page stands above it" in text
    )
    assert len(losses[1]) == 22
    assert losses[1] <= under_its_page
    assert (
        "With one chunk of a page allowed, 22 questions lose their passage, and all 22 are "
        "among those 23." in text
    )
    # The same questions each time: each limit loses what the milder one loses, and more.
    assert [len(losses[limit]) for limit in (4, 3, 2, 1)] == [2, 4, 9, 22]
    assert losses[4] < losses[3] < losses[2] < losses[1]
    assert (
        "the 2 questions that lose it with four allowed are among the 4 that lose it with "
        "three, those are among the 9, and those among the 22" in text
    )
    assert "the loss grows as the limit tightens (2, 4, 9 and 22 questions)" in text


def test_the_table_of_depth_is_read_from_the_two_runs_of_the_default_search() -> None:
    header = "| " + " | ".join(DEPTH_HEADER) + " |"
    assert header in results().splitlines()
    text = results()

    assert table(text, header) == [depth_row(k) for k in DEPTHS]
    # The rows are counted from the rank of each passage. At five and at ten results the
    # runs have scores of their own, and the counts are those.
    for k in (5, 10):
        assert depth_row(k)[1:] == [
            share(int(sum(scores(TWO, DEFAULT, f"all@{k}"))), 33),
            share(int(sum(scores(TWO, DEFAULT, f"hit@{k}"))), 33),
            share(int(sum(scores(GOLDEN, DEFAULT, f"hit@{k}"))), 132),
        ]
    deeper = paired_difference(scores(TWO, DEFAULT, "all@5"), scores(TWO, DEFAULT, "all@10"))
    assert (deeper.better, deeper.worse) == (5, 0)
    assert (
        f"The first ten results hold both passages for 5 questions more than the first five: "
        f"{change(deeper)}." in text
    )
    assert "That takes real requests to measure, and nothing here guesses at it." in text


def test_what_is_said_about_fourteen_variants_and_chance_is_counted_from_the_runs() -> None:
    text = results()
    on_the_headline = len(VARIANTS) - len(SAME_FIRST_FIVE)

    assert on_the_headline == 12
    assert f"{1 - 0.95**on_the_headline:.0%}" == "46%"
    assert "Twelve different comparisons were made here" in text
    assert "would be up to 1 - 0.95^12 = 46%, not 5%" in text
    assert not [run for run in VARIANTS if against_default(TWO, run, "all@5").is_clear]
    assert "On that score no comparison is clear, so there is no winner to discount." in text
    # The table of the golden set compares two scores of each variant with the default's.
    clear = {
        (run, metric)
        for run in VARIANTS
        for metric in ("hit@5", "mrr@10")
        if against_default(GOLDEN, run, metric).is_clear
    }
    assert "which holds 28 comparisons" in text
    assert 2 * len(VARIANTS) == 28
    # Every clear reading is of a limit for each page, but one.
    assert clear == {
        *((run, "hit@5") for run in ("spread-1-in-5", "spread-2-in-5", "spread-3-in-5")),
        *((run, "hit@5") for run in ("spread-1-in-10", "spread-2-in-10")),
        *((run, "mrr@10") for run in ("spread-1-in-5", "spread-2-in-5")),
        *((run, "mrr@10") for run in ("spread-1-in-10", "spread-2-in-10")),
        ("rrf-k-10", "mrr@10"),
    }
    one, two, three = (against_default(GOLDEN, PAGE_LIMITS[limit], "hit@5") for limit in (1, 2, 3))
    assert (one.better, one.worse, f"{sign_test(one.better, one.worse):.6f}") == (1, 22, "0.000006")
    assert "22 questions worse and 1 better. An exact sign test gives p = 0.000006." in text
    assert (two.better, two.worse, f"{sign_test(two.better, two.worse):.3f}") == (0, 9, "0.004")
    assert f"{0.05 / len(VARIANTS):.4f}" == "0.0036"
    assert sign_test(two.better, two.worse) > 0.05 / len(VARIANTS)
    assert (
        "9 worse and none better, p = 0.004. A correction for the fourteen comparisons of "
        "Hit@5 asks for less than 0.0036, so this one stands at its edge." in text
    )
    assert (three.better, three.worse, sign_test(three.better, three.worse)) == (0, 4, 0.125)
    assert "4 worse and none better. The interval excludes zero, and the sign test gives" in text
    assert "p = 0.125: the rule that `eval-compare` applies is generous" in text
    constant = against_default(GOLDEN, "rrf-k-10", "mrr@10")
    assert (constant.better, constant.worse) == (5, 1)
    assert f"{constant.difference.mean:+.3f}" == "+0.003"
    assert f"{sign_test(constant.better, constant.worse):.3f}" == "0.219"
    assert "+0.003, with 5 questions better and 1 worse, p = 0.219." in text


def test_the_five_sentences_give_the_figures_of_the_runs() -> None:
    text = results()
    section = text.split("## What this shows, and what it does not\n")[1].split("\n## ")[0]
    sentences = [line for line in section.splitlines() if line.strip()]
    hit = {run: summary(GOLDEN, run)["metrics"]["hit@5"]["mean"] for run in ORDER}
    several = summary(TWO, DEFAULT)["needs_several_passages"]

    assert [line[:3] for line in sentences] == ["1. ", "2. ", "3. ", "4. ", "5. "]
    # Each is one sentence: after its number, its only full stop is its last character.
    for line in sentences:
        assert line.endswith(".")
        assert ". " not in line[3:]
    assert (
        "the best are one question ahead, 13 against 12, and no difference is clear"
        in (sentences[0])
    )
    assert (
        f"(Hit@5 {hit['spread-1-in-5']:.3f} with one chunk of a page allowed and "
        f"{hit['spread-2-in-5']:.3f} with two, against {hit[DEFAULT]:.3f})" in sentences[1]
    )
    # The reranker's time, where the file gives it and where the sentence rounds it.
    slow, fast = median_ms(GOLDEN, "rerank"), median_ms(GOLDEN, DEFAULT)
    assert (slow, fast) == (216, 30)
    assert "216 ms for the question in the middle of the golden set's run, against 30 ms" in text
    assert round(slow / fast) == 7
    assert "the reranker takes about 7 times as long for a question" in sentences[2]
    in_eight, in_ten = (depth_row(k)[1].split(" of ")[0] for k in (8, 10))
    assert (
        f"among its first eight results for {in_eight} of the 33 questions and among its first "
        f"ten for {in_ten}, against 12 among the first five" in sentences[3]
    )
    assert several["not_all_retrieved"] == 16
    assert (
        "for 16 of them a passage is not among the default's first ten results at all"
        in sentences[4]
    )


def test_every_link_of_the_results_leads_somewhere_and_the_card_points_to_them() -> None:
    text = results()
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("# ").lower()).replace(" ", "-")
        for line in text.splitlines()
        if line.startswith("#")
    }
    targets = re.findall(r"\]\(([^)]+)\)", text)

    assert len(targets) >= 2 * len(ORDER)
    for target in targets:
        if target.startswith("https://"):
            continue
        if target.startswith("#"):
            assert target[1:] in headings, target
        else:
            assert (RESULTS.parent / target).exists(), target
    assert "[`search-variants.md`](search-variants.md)" in CARD.read_text(encoding="utf-8")


def test_the_gate_has_the_settings_it_had_before_the_variants_were_measured() -> None:
    gate = load_gate_config(ROOT / "evals" / "gate.json")

    assert (gate.distractor_pages, gate.seed) == (0, 7)
    assert gate.minimums == {"hit@5": 0.97, "mrr@10": 0.84}
    assert "No default was changed." in results()
    assert "the retrieval gate has the minimums it had" in results()


def test_the_comparison_of_the_tables_is_the_one_eval_compare_makes() -> None:
    # The command compares every score of two runs. The tables need one or two of them,
    # and take them the same way: on these runs the two agree to the last digit.
    for flow, run in ((TWO, "spread-1-in-5"), (TWO, "rerank"), (GOLDEN, "spread-2-in-5")):
        of_the_command = compare_runs(flow / DEFAULT, flow / run)
        assert of_the_command.questions == len(rows(flow, DEFAULT))
        for metric in ("hit@5", "all@5", "mrr@10", "recall@5"):
            assert against_default(flow, run, metric) == of_the_command.metrics[metric].change
    assert "the paired comparison that `eval-compare` makes with the run of the default search" in (
        results()
    )
