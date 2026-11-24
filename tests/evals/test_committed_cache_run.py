"""The committed cache measurement is what its rows give, and the documents say what it says.

The first half holds the run under ``evals/runs/cache/baseline`` to the pairs it was made
on and to itself: its figures are worked out again from its rows, with no model. The
second half holds the evaluation guide, the card of the pairs, the roadmap and the
changelog to the run, as ``tests/test_readme_results.py`` holds the README: every table
cell by cell, the sentences that carry a finding number by number, and every other
number in those sections to the set of numbers the run holds.

The last test loads the two real models and measures again, so it is marked slow.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from handbook_rag import wiring
from handbook_rag.config import Settings
from handbook_rag.evals import cache_pairs
from handbook_rag.evals.cache_pairs import (
    CHOOSING,
    COSINE,
    DIFFERENT,
    ENTAILS_MEAN,
    ENTAILS_MIN,
    POLICIES,
    REPORTING,
    RIGHT,
    SAME,
    WRONG,
    Asked,
    CachePair,
    agreed_pairs,
    load_cache_run,
    load_pairs,
    measure,
    outcome,
    served,
    split_pairs,
    summarise,
)
from handbook_rag.evals.dataset import GoldenQuestion, load_golden_set
from tests.test_readme_results import anchors, line_starting, numbers, table, three

ROOT = Path(__file__).parents[2]
RUN = ROOT / "evals" / "runs" / "cache" / "baseline"
GUIDE = ROOT / "docs" / "evaluation.md"
CARD = ROOT / "evals" / "cache" / "README.md"
HEADING = "A semantic cache, measured before it is built"

# How the documents name the three ways to decide, and the five scores.
DECIDED_BY = {
    COSINE: "Cosine alone",
    ENTAILS_MIN: "Second check, weaker direction",
    ENTAILS_MEAN: "Second check, mean of both directions",
}
SCORE_NAMES = {
    COSINE: "Cosine of the two embeddings",
    cache_pairs.ENTAILS_FORWARD: "Entailment, stored question to new",
    cache_pairs.ENTAILS_BACKWARD: "Entailment, new question to stored",
    ENTAILS_MIN: "Entailment, weaker direction",
    ENTAILS_MEAN: "Entailment, mean of both directions",
}


def golden() -> list[GoldenQuestion]:
    return load_golden_set(ROOT / "evals" / "golden" / "questions.jsonl")


def pairs() -> list[CachePair]:
    return load_pairs(ROOT / cache_pairs.PAIRS_PATH, golden())


def saved() -> Any:
    return json.loads((RUN / "summary.json").read_text(encoding="utf-8"))


def rows(name: str) -> dict[str, Any]:
    lines = (RUN / name).read_text(encoding="utf-8").splitlines()
    return {row["pair"]: row for row in map(json.loads, lines)}


def asked() -> list[Asked]:
    return load_cache_run(RUN, pairs())


def figures_again() -> dict[str, Any]:
    config = saved()["config"]
    return summarise(
        asked(),
        stored=len(golden()),
        candidates=config["candidates"],
        floor=config["floor"],
        seed=config["split_seed"],
        other_splits=config["other_splits"],
    )


# --- the run -----------------------------------------------------------------------------


def test_the_run_was_made_on_the_committed_pairs_with_the_settings_in_the_code() -> None:
    config = saved()["config"]
    settings = Settings()

    assert config == {
        "embedding_model": settings.embedding_model,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "embedding_device": "cpu",
        "verifier_model": settings.cache_verifier_model,
        "verifier_device": wiring.VERIFIER_DEVICE,
        "verifier_outputs": config["verifier_outputs"],
        "verifier_entailment_output": config["verifier_outputs"].index("entailment"),
        # A pair or a golden question changed since the run would change these.
        **cache_pairs.data_fingerprints(pairs(), golden()),
        "candidates": cache_pairs.CANDIDATES,
        "floor": cache_pairs.FLOOR,
        "split_seed": cache_pairs.SPLIT_SEED,
        "other_splits": cache_pairs.OTHER_SPLITS,
    }
    assert sorted(config["verifier_outputs"]) == ["contradiction", "entailment", "neutral"]
    assert sorted(path.name for path in RUN.iterdir()) == [
        "pairs.jsonl",
        "questions.jsonl",
        "summary.json",
    ]


def test_the_figures_of_the_run_are_what_its_rows_give() -> None:
    summary, again = saved(), figures_again()

    assert set(summary) == {"description", "config", "timing", *again}
    for part, figures in again.items():
        # Compared as saved: a tuple and a list are the same thing in a file.
        assert summary[part] == json.loads(json.dumps(figures)), part
    assert summary["cache_view"]["stored_questions"] == len(golden()) == 165


def test_every_row_holds_what_its_scores_and_the_chosen_thresholds_give() -> None:
    summary = saved()
    own, questions = rows("pairs.jsonl"), rows("questions.jsonl")
    thresholds = {
        policy: summary["cache_view"]["policies"][policy]["threshold"] for policy in POLICIES
    }

    assert list(own) == list(questions) == [pair.pair for pair in pairs()]
    for question in asked():
        pair = question.pair
        row, pair_row = questions[pair.pair], own[pair.pair]
        assert (pair_row["kind"], pair_row["agreed"]) == (pair.kind, pair.agreed)
        assert (pair_row["how"], pair_row["what_differs"]) == (pair.how, pair.what_differs)
        assert (pair_row["question_id"], pair_row["other_id"]) == (pair.question_id, pair.other_id)
        for scores in (pair_row, *row["candidates"]):
            both = (scores["entails_forward"], scores["entails_backward"])
            assert scores[ENTAILS_MIN] == min(both)
            assert scores[ENTAILS_MEAN] == sum(both) / 2
            assert all(0.0 <= value <= 1.0 for value in both)
            assert -1.0 <= scores[COSINE] <= 1.0
        cosines = [candidate[COSINE] for candidate in row["candidates"]]
        # Nearest first, at most as many as the run took, none below its floor.
        assert cosines == sorted(cosines, reverse=True)
        assert len(cosines) <= summary["config"]["candidates"]
        assert all(cosine >= summary["config"]["floor"] for cosine in cosines)
        assert row["stored_above_floor"] >= len(cosines)
        for candidate in row["candidates"]:
            assert candidate["own"] == (candidate["question_id"] == pair.question_id)
            if candidate["own"]:
                assert candidate[COSINE] == pair_row[COSINE]
        for policy in POLICIES:
            assert row["served"][policy] == served(question, policy, thresholds[policy])
            assert row["outcome"][policy] == outcome(pair, row["served"][policy])


def test_no_golden_question_is_on_both_sides_of_the_split_of_the_run() -> None:
    own = rows("pairs.jsonl")
    split = split_pairs(agreed_pairs(pairs()), seed=saved()["config"]["split_seed"])
    sides: dict[str, set[str]] = {CHOOSING: set(), REPORTING: set()}
    kinds: Counter[tuple[str, str]] = Counter()

    for pair in agreed_pairs(pairs()):
        half = own[pair.pair]["half"]
        assert half == split.half_of(pair.pair)
        sides[half] |= {pair.question_id, *([pair.other_id] if pair.other_id else [])}
        kinds[(half, pair.kind)] += 1

    assert not sides[CHOOSING] & sides[REPORTING]
    assert kinds == {
        (CHOOSING, SAME): 49,
        (CHOOSING, DIFFERENT): 50,
        (REPORTING, SAME): 49,
        (REPORTING, DIFFERENT): 50,
    }
    # 130 golden questions are in an agreed pair, each on one side only. The card counts
    # 131 in any pair: one is only in a disputed pair.
    assert len(sides[CHOOSING]) + len(sides[REPORTING]) == 130


def test_the_disputed_pairs_are_in_no_half_and_in_no_figure() -> None:
    summary, own = saved(), rows("pairs.jsonl")

    assert summary["pairs"]["disputed"] == ["s078", "s100"]
    assert [pair["pair"] for pair in summary["disputed"]] == ["s078", "s100"]
    for pair_id in ("s078", "s100"):
        assert own[pair_id]["half"] is None
        assert own[pair_id]["agreed"] is False
    assert summary["pairs"] == {
        "in_file": 200,
        "agreed": 198,
        SAME: 98,
        DIFFERENT: 100,
        "disputed": ["s078", "s100"],
        "other_is_a_stored_question": 13,
    }
    split = summary["cache_view"]["split"]
    assert split[CHOOSING]["pairs"] + split[REPORTING]["pairs"] == 198
    for policy in POLICIES:
        reported = summary["cache_view"]["policies"][policy][REPORTING]
        chose = summary["cache_view"]["policies"][policy][CHOOSING]
        assert reported[SAME]["pairs"] + chose[SAME]["pairs"] == 98
        assert reported[DIFFERENT]["pairs"] + chose[DIFFERENT]["pairs"] == 100
        # The rule allows nothing wrong on the half it chooses on.
        assert chose[SAME][WRONG] == chose[DIFFERENT][WRONG] == 0


def test_a_question_that_is_a_stored_question_never_meets_its_own_entry() -> None:
    questions = rows("questions.jsonl")
    golden_pairs = [pair for pair in pairs() if pair.other_id is not None]

    assert len(golden_pairs) == 13
    for pair in pairs():
        row = questions[pair.pair]
        assert row["excluded"] == pair.other_id, pair.pair
        found = [candidate["question_id"] for candidate in row["candidates"]]
        assert pair.other_id not in found, pair.pair
        # No candidate is at a cosine of 1: nothing served here is an exact match.
        assert all(candidate[COSINE] < 0.9999 for candidate in row["candidates"]), pair.pair


# --- the documents -----------------------------------------------------------------------


def section(text: str, heading: str) -> str:
    """The part of a Markdown document under one heading, up to the next of its level.

    A line in fenced code is no heading, though a comment there starts with the same sign.
    """
    lines = text.splitlines()
    starts = [n for n, line in enumerate(lines) if re.fullmatch(rf"#+ {re.escape(heading)}", line)]
    assert len(starts) == 1, f"expected one heading {heading!r}, found {len(starts)}"
    level = len(lines[starts[0]].split(" ")[0])
    kept: list[str] = []
    in_code = False
    for line in lines[starts[0] + 1 :]:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        elif not in_code and re.match(rf"#{{1,{level}}} ", line):
            break
        kept.append(line)
    return "\n".join(kept)


def guide() -> str:
    return section(GUIDE.read_text(encoding="utf-8"), HEADING)


def card() -> str:
    return section(CARD.read_text(encoding="utf-8"), "What has been measured with the pairs")


def cell(count: int, of: int, estimate: Any) -> str:
    """A count with its share as the documents print it: 7 of 49: 0.143 (0.071 to 0.267)."""
    share = f"{three(estimate['mean'])} ({three(estimate['low'])} to {three(estimate['high'])})"
    return f"{count} of {of}: {share}"


def shown(threshold: float | None) -> str:
    return "none" if threshold is None else f"{threshold:.4f}"


def at(curve: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    (row,) = [row for row in curve if row["threshold"] == threshold]
    return row


def numbers_of_the_run(summary: Any) -> set[str]:
    """Every number the run holds, in each way a document may print it."""
    found: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, bool) or value is None or isinstance(value, str):
            return
        if isinstance(value, int):
            found.add(str(value))
        elif isinstance(value, float):
            found.update({str(value), f"{value:.3f}", f"{value:.4f}", f"{value:.1f}"})
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        else:
            for item in value:
                walk(item)

    walk(summary)
    return found


def test_the_guides_table_of_the_pair_view_is_the_run() -> None:
    scores = saved()["pair_view"]["scores"]
    found = table(guide(), "| Score | Area under the curve |")

    assert [row[0] for row in found] == list(SCORE_NAMES.values())
    for row, score in zip(found, SCORE_NAMES, strict=True):
        same, different = scores[score]["by_kind"][SAME], scores[score]["by_kind"][DIFFERENT]
        assert row[1:] == [
            three(scores[score]["area_under_curve"]),
            *(three(same[name]) for name in ("min", "median", "max")),
            *(three(different[name]) for name in ("min", "median", "max")),
        ], score


def test_the_guides_table_of_the_reporting_half_is_the_run() -> None:
    policies = saved()["cache_view"]["policies"]
    found = table(guide(), "| Decided by | Threshold | Rewordings served right | Served another ")

    assert [row[0] for row in found] == list(DECIDED_BY.values())
    for row, policy in zip(found, DECIDED_BY, strict=True):
        same, different = policies[policy][REPORTING][SAME], policies[policy][REPORTING][DIFFERENT]
        assert row[1:] == [
            shown(policies[policy]["threshold"]),
            cell(same[RIGHT], same["pairs"], same["right_share"]),
            cell(same[WRONG], same["pairs"], same["wrong_share"]),
            cell(same["missed"], same["pairs"], same["missed_share"]),
            cell(different[WRONG], different["pairs"], different["wrong_share"]),
        ], policy
    # "each of the two ways to read the second check served 10 of 49 rewordings right, so
    # the rule takes the weaker direction, which is named first"
    line = line_starting(guide(), "On the choosing half each of the two ways")
    chose = [policies[policy][CHOOSING][SAME] for policy in (ENTAILS_MIN, ENTAILS_MEAN)]
    assert numbers(line) == [str(chose[0][RIGHT]), str(chose[0]["pairs"])]
    assert chose[0] == chose[1]
    assert saved()["cache_view"]["chosen_policy"] == ENTAILS_MIN
    assert POLICIES.index(ENTAILS_MIN) < POLICIES.index(ENTAILS_MEAN)


def by_how_rows(summary: Any) -> list[list[str]]:
    """The table by ``how``: the guide and the card print the same rows."""
    scores, cache = summary["pair_view"]["scores"], summary["cache_view"]
    reported = cache["policies"][cache["chosen_policy"]]["reporting_by_how"]
    return [
        [
            f"`{how}`",
            str(spread["pairs"]),
            three(spread["median"]),
            three(scores[ENTAILS_MIN]["by_how"][how]["median"]),
            three(scores[ENTAILS_MEAN]["by_how"][how]["median"]),
            str(reported[how]["pairs"]),
            str(reported[how][RIGHT]),
        ]
        for how, spread in scores[COSINE]["by_how"].items()
    ]


def test_the_tables_by_how_in_the_guide_and_in_the_card_are_the_run() -> None:
    expected = by_how_rows(saved())

    assert [row[0] for row in expected] == [f"`{how}`" for how in cache_pairs.HOWS]
    assert table(guide(), "| `how` | Agreed pairs | Median cosine |") == expected
    assert table(card(), "| `how` | Agreed pairs | Median similarity |") == expected


def test_the_cards_table_by_what_differs_is_the_run() -> None:
    summary = saved()
    scores, cache = summary["pair_view"]["scores"], summary["cache_view"]
    reported = cache["policies"][cache["chosen_policy"]]["reporting_by_what_differs"]
    found = table(card(), "| `what_differs` | Agreed pairs |")

    assert [row[0] for row in found] == [f"`{what}`" for what in cache_pairs.WHAT_DIFFERS]
    for row, what in zip(found, cache_pairs.WHAT_DIFFERS, strict=True):
        cosine = scores[COSINE]["by_what_differs"][what]
        weaker = scores[ENTAILS_MIN]["by_what_differs"][what]
        assert row[1:] == [
            str(cosine["pairs"]),
            three(cosine["median"]),
            three(cosine["max"]),
            three(weaker["median"]),
            three(weaker["max"]),
            str(reported[what]["pairs"]),
            str(reported[what][WRONG]),
        ], what


def test_the_guides_table_of_the_other_splits_is_the_run() -> None:
    others = saved()["cache_view"]["other_splits"]
    found = table(guide(), "| Decided by | Splits with a threshold |")

    assert [row[0] for row in found] == list(DECIDED_BY.values())
    for row, policy in zip(found, DECIDED_BY, strict=True):
        figures = others["policies"][policy]
        right = figures["right_share"]
        assert row[1:] == [
            str(others["splits"] - figures["splits_without_a_threshold"]),
            str(figures["chosen"]),
            three(right["min"]),
            three(right["median"]),
            three(right["max"]),
            three(figures["wrong_share_of_different"]["max"]),
            str(figures["splits_with_nothing_wrong"]),
        ], policy
    # "For the cosine, the 183 splits with nothing wrong served are those in which it had no
    # threshold ... In each of the 17 in which it had one, something wrong was served."
    cosine = others["policies"][COSINE]
    line = line_starting(guide(), "For the cosine, the ")
    assert numbers(line) == [
        str(cosine["splits_without_a_threshold"]),
        str(cosine["splits_with_something_wrong"]),
    ]
    assert cosine["splits_with_nothing_wrong"] == cosine["splits_without_a_threshold"]
    assert cosine["splits_with_something_wrong"] == (
        others["splits"] - cosine["splits_without_a_threshold"]
    )


def test_the_guides_table_at_fixed_thresholds_is_the_run() -> None:
    whole = saved()["cache_view"]["whole_set"]
    found = table(guide(), "| Threshold | Cosine alone |")

    def counts(figures: Any) -> str:
        return f"{figures[SAME][RIGHT]} and {figures[DIFFERENT][WRONG]}"

    *fixed, with_all = found
    assert [row[0] for row in fixed] == ["0.8", "0.9", "0.95", "0.975", "0.99", "0.995"]
    for row in fixed:
        assert row[1:] == [
            counts(at(whole["curves"][policy], float(row[0]))) for policy in POLICIES
        ], row[0]
    chosen = whole["chosen_on_all_pairs"]
    assert with_all == [
        "The rule, with every pair in view",
        *(f"{shown(chosen[policy]['threshold'])}: {counts(chosen[policy])}" for policy in POLICIES),
    ]


def test_what_the_guide_says_the_run_shows_is_what_the_run_holds() -> None:
    summary = saved()
    scores, cache = summary["pair_view"]["scores"], summary["cache_view"]
    counted, policies, others = summary["pairs"], cache["policies"], cache["other_splits"]
    curves, timing = cache["whole_set"]["curves"], summary["timing"]
    weaker = policies[ENTAILS_MIN]
    reported = weaker[REPORTING]

    def says(start: str) -> list[str]:
        return numbers(line_starting(guide(), f"- **{start}"))

    cosine = scores[COSINE]["by_kind"]
    at_95 = at(curves[COSINE], 0.95)
    assert says("The cosine alone does not tell a rewording from a near-miss on these pairs.") == [
        three(scores[COSINE]["area_under_curve"]),
        three(cosine[DIFFERENT]["max"]),
        three(cosine[SAME]["max"]),
        three(policies[COSINE]["decided_by"]["score"]),
        "0.95",
        str(at_95[SAME][RIGHT]),
        str(counted[SAME]),
        str(at_95[DIFFERENT][WRONG]),
        str(counted[DIFFERENT]),
    ]
    assert cosine[DIFFERENT]["max"] > cosine[SAME]["max"]
    assert policies[COSINE]["threshold"] is None
    assert policies[COSINE]["decided_by"]["pair"] == "d012"
    assert policies[COSINE][REPORTING][SAME][RIGHT] == 0

    by_what = scores[ENTAILS_MIN]["by_what_differs"]
    assert says("The second check separates the kinds far better.") == [
        three(scores[ENTAILS_MIN]["area_under_curve"]),
        three(scores[ENTAILS_MEAN]["area_under_curve"]),
        three(by_what["number"]["max"]),
        three(by_what["negation"]["max"]),
    ]

    by_how = weaker["reporting_by_how"]
    assert says("It still serves little, because it passes plain rewordings and little else.") == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[SAME]["missed"]),
        str(reported[SAME][RIGHT]),
    ]
    assert by_how["reworded"][RIGHT] + by_how["synonyms"][RIGHT] == reported[SAME][RIGHT]
    assert [by_how[how][RIGHT] for how in ("short", "long", "indirect")] == [0, 0, 0]

    at_90 = at(curves[ENTAILS_MIN], 0.9)
    assert says("One near-miss set the threshold.") == [
        three(weaker["decided_by"]["score"]),
        "0.9",
        str(at_90[SAME][RIGHT]),
        str(counted[SAME]),
        str(at_90[DIFFERENT][WRONG]),
        str(counted[DIFFERENT]),
    ]
    assert (weaker["decided_by"]["pair"], weaker["decided_by"]["group"]) == ("d067", "aspect")
    assert at_90[SAME][WRONG] == 0

    near_misses = str(reported[DIFFERENT]["pairs"])
    wrong_splits = {
        others["policies"][policy]["splits_with_something_wrong"]
        for policy in (ENTAILS_MIN, ENTAILS_MEAN)
    }
    assert says("Clean on one half is not safe.") == [
        near_misses,
        near_misses,
        three(cache["zero_wrong_allows"]["up_to"]),
        str(others["splits"]),
        str(others["chosen_policy"]["splits_with_something_wrong"]),
        near_misses,
        near_misses,
    ]
    assert wrong_splits == {others["chosen_policy"]["splits_with_something_wrong"]}
    assert reported[DIFFERENT][WRONG] == 0
    assert cache["zero_wrong_allows"]["near_misses"] == reported[DIFFERENT]["pairs"]

    spread = others["policies"][ENTAILS_MIN]["right_share"]
    assert says("The split reported is not a lucky one for savings.") == [
        three(spread["median"]),
        three(spread["min"]),
        three(spread["max"]),
        three(reported[SAME]["right_share"]["mean"]),
    ]
    assert reported[SAME]["right_share"]["mean"] < spread["median"]

    with_all = cache["whole_set"]["chosen_on_all_pairs"]
    assert says("Scoring on the pairs a threshold was chosen on flatters, as expected.") == [
        str(with_all[ENTAILS_MIN][SAME][RIGHT]),
        str(counted[SAME]),
    ]
    for policy in (ENTAILS_MIN, ENTAILS_MEAN):
        assert with_all[policy][SAME][RIGHT] == with_all[ENTAILS_MIN][SAME][RIGHT]
        assert with_all[policy][SAME][WRONG] == with_all[policy][DIFFERENT][WRONG] == 0
    assert (
        with_all[ENTAILS_MIN][SAME][RIGHT] / counted[SAME] > reported[SAME]["right_share"]["mean"]
    )

    assert says("What the second check costs.") == [
        str(timing["pairs_verified"]),
        str(timing["verifying_seconds"]),
        str(timing["verifying_ms_per_pair"]),
        str(cache["candidates"]),
        str(timing["questions_embedded"]),
        str(timing["embedding_seconds"]),
        str(timing["embedding_ms_per_question"]),
    ]


def test_how_far_the_guide_says_the_run_goes_is_what_the_run_holds() -> None:
    summary = saved()
    cache = summary["cache_view"]
    curves = cache["whole_set"]["curves"]

    def says(start: str) -> list[str]:
        return numbers(line_starting(guide(), f"- **{start}"))

    def highest_with_another_answer(policy: str) -> str:
        return str(max(row["threshold"] for row in curves[policy] if row[SAME][WRONG]))

    assert says("One embedding model, one verifier, and a cache of 165 questions.") == [
        str(cache["stored_questions"])
    ]
    assert says("A serve of another stored question's answer rests on no label.") == [
        highest_with_another_answer(ENTAILS_MIN),
        highest_with_another_answer(ENTAILS_MEAN),
    ]
    # "None happened on the reporting half", and over all pairs it is one rewording, s073.
    for policy in POLICIES:
        assert [w for w in cache["policies"][policy]["wrong_on_reporting"] if not w["own"]] == []
        assert {row[SAME][WRONG] for row in curves[policy]} <= {0, 1}
    another = [
        (question.pair.pair, candidate.question_id)
        for question in asked()
        if question.pair.agreed and question.pair.kind == SAME
        for candidate in question.candidates
        if candidate.question_id != question.pair.question_id
        and candidate.scores.of(ENTAILS_MEAN) >= cache_pairs.THRESHOLDS[0]
    ]
    assert [pair for pair, _ in another] == ["s073"]


def test_what_the_guide_says_of_the_method_is_the_method_of_the_run() -> None:
    summary = saved()
    counted, cache = summary["pairs"], summary["cache_view"]
    split, above = cache["split"], cache["stored_above_floor"]
    text = guide()

    assert (
        f"about the {counted['agreed']} pairs on which the two agree, {counted[SAME]} `same` and "
        f"{counted[DIFFERENT]} `different`"
    ) in text
    assert "The two disputed pairs, `s078` and `s100`, are in no figure" in text
    assert f"holds all {cache['stored_questions']} golden questions is simulated" in text
    assert (
        f"the {cache['candidates']} nearest above a floor of {cache['floor']}. They are tried "
        "nearest first"
    ) in text
    assert (
        f"In {counted['other_is_a_stored_question']} `different` pairs the other question" in text
    )
    assert (
        f"the floor left between {above['min']:.0f} and {above['max']:.0f} stored questions"
    ) in text
    assert cache["questions_with_fewer_candidates"] == 0
    assert f"no question had fewer than {cache['candidates']} candidates" in text
    assert (
        f"There are {split['groups']} groups, the largest of {split['largest_group']} pairs"
    ) in text
    for half in (CHOOSING, REPORTING):
        assert (split[half]["pairs"], split[half][SAME], split[half][DIFFERENT]) == (99, 49, 50)
    assert "That gives 99 pairs on each side: 49 `same` and 50 `different`." in text
    assert (
        f"`{summary['config']['embedding_model']}` for the embeddings and "
        f"`{summary['config']['verifier_model']}` as the verifier, both on the CPU"
    ) in text
    # 59 golden questions are in more than one pair: counted from the pairs, as the card does.
    uses = Counter(pair.question_id for pair in pairs()) + Counter(
        pair.other_id for pair in pairs() if pair.other_id
    )
    assert f"{sum(count > 1 for count in uses.values())} golden questions are in more than" in text
    assert f"The same rules on {cache['other_splits']['splits']} other splits" in text


def test_what_the_card_says_was_measured_is_what_the_run_holds() -> None:
    summary = saved()
    scores, cache = summary["pair_view"]["scores"], summary["cache_view"]
    counted, policies, others = summary["pairs"], cache["policies"], cache["other_splits"]
    reported = policies[cache["chosen_policy"]][REPORTING]
    split = cache["split"]

    def says(start: str) -> list[str]:
        return numbers(line_starting(card(), f"- **{start}"))

    assert says("A simulated cache.") == [
        str(cache["stored_questions"]),
        str(cache["candidates"]),
        str(counted["other_is_a_stored_question"]),
    ]
    assert says("Two halves.") == [
        str(split[REPORTING]["pairs"]),
        str(split[REPORTING][SAME]),
        str(split[REPORTING][DIFFERENT]),
    ]
    assert split[CHOOSING] == {**split[REPORTING], "groups": split[CHOOSING]["groups"]}
    assert says("The similarity of the embeddings alone does not tell the two kinds apart.") == [
        three(scores[COSINE]["area_under_curve"]),
        "0.5",
        str(counted["agreed"]),
    ]
    by_kind = scores[COSINE]["by_kind"]
    assert by_kind[DIFFERENT]["max"] > by_kind[SAME]["max"]
    assert policies[COSINE]["threshold"] is None
    assert says("The second model separates them far better") == [
        three(scores[ENTAILS_MIN]["area_under_curve"]),
        three(scores[ENTAILS_MEAN]["area_under_curve"]),
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[DIFFERENT]["pairs"]),
    ]
    assert reported[DIFFERENT][WRONG] == 0
    assert says("None of 50 is not safe.") == [
        str(reported[DIFFERENT]["pairs"]),
        three(cache["zero_wrong_allows"]["up_to"]),
        str(others["splits"]),
        str(others["chosen_policy"]["splits_with_something_wrong"]),
    ]
    first, second = summary["disputed"]
    assert says("The two disputed pairs are in no figure.") == [
        *(three(first[score]) for score in (COSINE, "entails_forward", "entails_backward")),
        *(three(second[score]) for score in (COSINE, "entails_forward", "entails_backward")),
    ]
    assert (first["pair"], second["pair"]) == ("s078", "s100")
    for pair in (first, second):
        assert set(pair["served"].values()) == {None}
    # "For the similarity it is d012 ... For the second model it is d067, an aspect pair"
    assert policies[COSINE]["decided_by"]["pair"] == "d012"
    assert policies[ENTAILS_MIN]["decided_by"]["pair"] == "d067"
    assert policies[ENTAILS_MEAN]["decided_by"]["pair"] == "d067"
    assert "`d012`" in card()
    assert "`d067`, an `aspect` pair" in card()
    assert f"`{summary['config']['verifier_model']}`" in card()


def test_the_roadmap_and_the_changelog_quote_the_run() -> None:
    summary = saved()
    cache = summary["cache_view"]
    reported = cache["policies"][cache["chosen_policy"]][REPORTING]
    others = cache["other_splits"]
    served_right = [str(reported[SAME][RIGHT]), str(reported[SAME]["pairs"])]
    near_misses = str(reported[DIFFERENT]["pairs"])
    roadmap = line_starting(
        (ROOT / "docs" / "roadmap.md").read_text(encoding="utf-8"), "- **A semantic cache**"
    )
    changelog = line_starting(
        (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), "- `handbook-rag eval-cache`"
    )

    assert numbers(roadmap) == [
        "24",
        str(summary["pairs"]["agreed"]),
        *served_right,
        near_misses,
        str(others["chosen_policy"]["splits_with_something_wrong"]),
        str(others["splits"]),
    ]
    # The last number is the issue the entry links to.
    assert numbers(changelog) == [
        str(summary["pairs"]["in_file"]),
        *served_right,
        near_misses,
        "24",
    ]
    assert reported[DIFFERENT][WRONG] == 0
    assert cache["policies"][COSINE]["threshold"] is None


def test_every_link_in_what_was_written_about_the_run_leads_somewhere() -> None:
    roadmap, changelog = ROOT / "docs" / "roadmap.md", ROOT / "CHANGELOG.md"
    written = [
        (GUIDE, guide()),
        (CARD, CARD.read_text(encoding="utf-8")),
        (roadmap, line_starting(roadmap.read_text(encoding="utf-8"), "- **A semantic cache**")),
        (
            changelog,
            line_starting(changelog.read_text(encoding="utf-8"), "- `handbook-rag eval-cache`"),
        ),
    ]
    checked = 0
    for document, text in written:
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://")):
                continue
            path, _, anchor = target.partition("#")
            linked = document.parent / path if path else document
            assert linked.exists(), f"{document.name}: {target}"
            if anchor:
                assert anchor in anchors(linked.read_text(encoding="utf-8")), target
            checked += 1
    # The run, the card, the guide's section and the card's known limits are all linked.
    assert checked >= 8


def test_every_number_in_the_cache_sections_is_a_number_of_the_run() -> None:
    """A wider and coarser net than the tests above, for the numbers they do not name.

    A number passes when the run holds it somewhere, printed in one of the ways the
    documents print numbers. The run holds many numbers, so a wrong figure can still
    slip through: this catches a number that comes from nowhere, such as one worked out
    by hand. The tables and the sentences that carry a finding are held exactly, above.
    """
    allowed = numbers_of_the_run(saved())
    allowed |= {
        "24",  # the issue, and the day of the run
        "2026",  # its year
        "59",  # golden questions in more than one pair, held to the pairs above
    }
    readme = section((ROOT / "README.md").read_text(encoding="utf-8"), HEADING)

    for name, text in (("guide", guide()), ("card", card()), ("README", readme)):
        unknown = [number for number in numbers(text) if number not in allowed]
        assert unknown == [], name


# --- the real models ---------------------------------------------------------------------


@pytest.mark.slow
def test_the_real_models_give_the_scores_of_the_committed_run_again() -> None:
    settings = Settings()
    config = saved()["config"]
    measured = measure(
        pairs(),
        golden(),
        embedder=wiring.make_embedder(wiring.settings_for_serving(settings)),
        verifier=wiring.make_verifier(settings),
        candidates=config["candidates"],
        floor=config["floor"],
    )

    for now, then in zip(measured.asked, asked(), strict=True):
        assert now.pair == then.pair
        assert [c.question_id for c in now.candidates] == [c.question_id for c in then.candidates]
        assert (now.above_floor, now.excluded) == (then.above_floor, then.excluded)
        for new, old in zip(
            (now.own, *(c.scores for c in now.candidates)),
            (then.own, *(c.scores for c in then.candidates)),
            strict=True,
        ):
            # Another machine may differ in the last digits, and in nothing a reader sees.
            assert new.cosine == pytest.approx(old.cosine, abs=1e-4)
            assert new.entails.forward == pytest.approx(old.entails.forward, abs=1e-3)
            assert new.entails.backward == pytest.approx(old.entails.backward, abs=1e-3)
