"""The committed cache measurement is what its rows give, and the documents say what it says.

The first part holds the run under ``evals/runs/cache/baseline`` to the pairs it was made
on and to itself: its figures are worked out again from its rows, with no model, and the
report the command prints is checked figure by figure against them. The second part
holds the evaluation guide, the card of the pairs, the roadmap and the changelog to the
run, as ``tests/test_readme_results.py`` holds the README: every table cell by cell, the
sentences that carry a finding number by number and name by name, and every other number
in those sections to the set of numbers the run holds.

The last test loads the two real models and measures again, so it is marked slow.
"""

import datetime
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
    MISSED,
    NOT_SERVED,
    POLICIES,
    REPORTING,
    RIGHT,
    SAME,
    SCORES,
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
    unanswered_questions,
)
from handbook_rag.evals.cache_report import report_lines
from handbook_rag.evals.dataset import GoldenQuestion, load_golden_set
from tests.test_readme_results import anchors, line_starting, numbers, table, three

ROOT = Path(__file__).parents[2]
RUN = ROOT / "evals" / "runs" / "cache" / "baseline"
GUIDE = ROOT / "docs" / "evaluation.md"
CARD = ROOT / "evals" / "cache" / "README.md"
HEADING = "A semantic cache, measured before it is built"
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)

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
# The ways of rewording that change the shape of a question, not only its words.
OTHER_SHAPES = ("short", "long", "indirect")


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
        unanswered=unanswered_questions(golden()),
        seed=config["split_seed"],
        other_splits=config["other_splits"],
    )


def day(summary: Any) -> str:
    """The day of the run as the documents write it: 24 November 2026."""
    made = datetime.date.fromisoformat(summary["measured_on"])
    return f"{made.day} {MONTHS[made.month - 1]} {made.year}"


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

    assert set(summary) == {"description", "measured_on", "config", "rows", "timing", *again}
    for part, figures in again.items():
        # Compared as saved: a tuple and a list are the same thing in a file.
        assert summary[part] == json.loads(json.dumps(figures)), part
    assert summary["cache_view"]["stored_questions"] == len(golden()) == 165
    assert summary["cache_view"]["stored_unanswered"] == len(unanswered_questions(golden())) == 33


def test_the_figures_name_the_two_files_of_rows_they_were_worked_out_from() -> None:
    # A row changed by hand is another file, also where no figure depends on what changed.
    assert saved()["rows"] == cache_pairs.rows_fingerprints(RUN)


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
        "different_by_source": {"golden": 13, "written": 87},
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


# --- the report the command prints -------------------------------------------------------


def serve(found: Any) -> str:
    """A wrong serve as the report words it, worked out here apart from the report's code."""
    if found is None:
        return "none"
    whose = "its own golden question" if found["own"] else "another stored question"
    decline = ", a stored decline" if found["stored_unanswered"] else ""
    return (
        f"{found['pair']} ({found['group']}) at {found['score']:.4f}, {whose} "
        f"{found['served']}{decline}"
    )


def spread(found: Any, *names: str) -> list[str]:
    return [three(found[name]) for name in names or ("min", "median", "max")]


def over_splits(figures: Any) -> list[str]:
    """The figures of one row of the report's tables over the other splits."""
    return [
        *spread(figures["right_share"]),
        *spread(figures["wrong_share_of_different"]),
        str(figures["splits_with_nothing_wrong"]),
    ]


def test_every_figure_the_report_prints_is_the_figure_of_the_run_in_its_place() -> None:
    """The report is printed from the committed run, and read back number by number.

    Each expected line is worked out here from ``summary.json``, in the order the report
    prints its figures. A figure printed in another's place, a bound for a mean or one
    half for the other, makes a line differ.
    """
    summary = saved()
    counted, config, timing = summary["pairs"], summary["config"], summary["timing"]
    view, cache = summary["pair_view"]["scores"], summary["cache_view"]
    policies, others = cache["policies"], cache["other_splits"]
    lines = report_lines(summary)

    def part(start: str, end: str | None) -> list[str]:
        """The lines from the one that starts with ``start`` up to the one starting ``end``."""
        (first,) = [n for n, line in enumerate(lines) if line.strip().startswith(start)]
        if end is None:
            return lines[first:]
        (last,) = [n for n, line in enumerate(lines) if line.strip().startswith(end)]
        assert first < last
        return lines[first:last]

    def figures(found: list[str]) -> list[list[str]]:
        """The numbers of each line that has any, in the order they are printed."""
        return [numbers(line) for line in found if numbers(line)]

    assert figures(part("Cache measurement of", "Pair view")) == [
        [*summary["measured_on"].split("-"), *map(str, (198, 98, 100))],
        [str(timing["questions_embedded"]), f"{timing['embedding_seconds']:.1f}"],
        [str(timing["pairs_verified"]), f"{timing['verifying_seconds']:.1f}"],
        [str(config["verifier_entailment_output"])],
    ]
    assert (counted["agreed"], counted[SAME], counted[DIFFERENT]) == (198, 98, 100)
    assert f"{config['embedding_model']} on {config['embedding_device']}:" in lines[2]
    assert f"{config['verifier_model']} on {config['verifier_device']}:" in lines[3]
    assert f"of {', '.join(config['verifier_outputs'])}, read from" in lines[4]

    # The pair view: each score over the two kinds, the medians by group, the counts.
    expected = [
        [
            three(view[score]["area_under_curve"]),
            *spread(view[score]["by_kind"][SAME]),
            *spread(view[score]["by_kind"][DIFFERENT]),
        ]
        for score in SCORES
    ]
    labels = list(SCORES)
    for kind, by in ((SAME, "by_how"), (DIFFERENT, "by_what_differs"), (DIFFERENT, "by_source")):
        for group, found in view[COSINE][by].items():
            expected.append(
                [str(found["pairs"]), *(three(view[p][by][group]["median"]) for p in POLICIES)]
            )
            labels.append(f"{kind}, {group}")
    expected.append([str(counted[SAME]), str(counted[DIFFERENT])])
    labels.append("pairs at or above a threshold")
    for place, threshold in enumerate(cache_pairs.THRESHOLDS):
        cells = [view[policy]["at_or_above"][place] for policy in POLICIES]
        expected.append(
            [three(threshold), *(str(cell[kind]) for cell in cells for kind in (SAME, DIFFERENT))]
        )
        labels.append(f"at {threshold:.3f}")
    pair_view = [line for line in part("Pair view", "Cache view") if numbers(line)]
    assert figures(pair_view) == expected
    for line, label in zip(pair_view, labels, strict=True):
        assert line.strip().startswith(label), line

    # The cache view: the cache, the candidates and the two halves.
    own, split, above = cache["own_question"], cache["split"], cache["stored_above_floor"]
    expected = [
        [
            str(cache["stored_questions"]),
            str(cache["stored_unanswered"]),
            str(cache["candidates"]),
            str(cache["floor"]),
        ],
        [
            f"{above['min']:.0f}",
            f"{above['max']:.0f}",
            str(cache["questions_with_fewer_candidates"]),
            str(cache["candidates"]),
        ],
        *(
            [
                str(own[SAME][key]),
                str(own[SAME]["pairs"]),
                str(own[DIFFERENT][key]),
                str(own[DIFFERENT]["pairs"]),
            ]
            for key in ("nearest", "among_candidates")
        ),
        [str(split["groups"]), str(split["largest_group"])],
    ]
    for half in (CHOOSING, REPORTING):
        of_half = split[half]
        expected.append(
            [
                str(of_half["pairs"]),
                str(of_half[SAME]),
                str(of_half["groups_of"][SAME]["groups"]),
                str(of_half[DIFFERENT]),
                str(of_half["groups_of"][DIFFERENT]["groups"]),
                str(of_half["groups_of"][DIFFERENT]["largest"]),
                str(of_half["different_by_source"]["golden"]),
            ]
        )
        expected.append(
            [
                str(of_half["candidates"]["named_by_the_other_half"]),
                str(of_half["candidates"]["not_own"]),
            ]
        )
    cache_view = part("Cache view", "Chosen on the choosing half")
    assert figures(cache_view) == expected
    assert [line.strip().split(":")[0] for line in cache_view[5:9]] == [
        CHOOSING,
        CHOOSING,
        REPORTING,
        REPORTING,
    ]
    assert "is the nearest" in cache_view[2]
    assert "is a candidate" in cache_view[3]

    # The thresholds chosen, and what set each.
    chosen = part("Chosen on the choosing half", "On the reporting half")
    for line, policy in zip(chosen[2:5], POLICIES, strict=True):
        found = policies[policy]
        right = f"{found[CHOOSING][SAME][RIGHT]} of {found[CHOOSING][SAME]['pairs']}"
        set_by = serve(found["decided_by"])
        if found["halfway_to"] is not None:
            set_by += f"; halfway to the next score, {found['halfway_to']:.4f}"
        at_threshold = "none" if found["threshold"] is None else f"{found['threshold']:.4f}"
        assert line.split() == [policy, at_threshold, *right.split(), *set_by.split()], policy
    assert chosen[5].strip() == f"the rule takes: {cache['chosen_policy']}"

    # The reporting half: counts, and shares only where a threshold was chosen.
    expected_lines = []
    for policy in POLICIES:
        found = policies[policy]
        same, different = found[REPORTING][SAME], found[REPORTING][DIFFERENT]
        nothing = found["threshold"] is None
        expected_lines.append(
            f"{policy} at none: it serves nothing by the rule, so no rate is estimated"
            if nothing
            else f"{policy} at {found['threshold']:.4f}"
        )
        for label, name in (
            ("same: served right", RIGHT),
            ("same: served another answer", WRONG),
            ("same: missed", MISSED),
        ):
            share = same.get(f"{name}_share")
            expected_lines.append(
                f"{label} {same[name]} of {same['pairs']}"
                + ("" if nothing else f" {three(share['mean'])} ({three(share['low'])} to ")
                + ("" if nothing else f"{three(share['high'])})")
            )
        share = different.get("wrong_share")
        expected_lines.append(
            f"different: served wrongly {different[WRONG]} of {different['pairs']}"
            + ("" if nothing else f" {three(share['mean'])} ({three(share['low'])} to ")
            + ("" if nothing else f"{three(share['high'])})")
        )
        expected_lines += [f"wrong: {serve(wrong)}" for wrong in found["wrong_on_reporting"]]
        expected_lines += [
            "right only because the threshold is halfway to the next score: "
            f"{right['pair']} ({right['group']}) at {right['score']:.4f}"
            for right in found["right_only_by_the_margin"]
        ]
    allows = cache["zero_wrong_allows"]
    expected_lines.append(
        f"What this half cannot show: none wrong of {allows['near_misses']} near-misses would "
        f"still allow a true rate of up to {three(allows['up_to'])}."
    )
    reported = part("On the reporting half", "What each half holds")
    assert [" ".join(line.split()) for line in reported[2:] if line.strip()] == expected_lines

    # What each half holds to test a threshold with.
    expected_lines = []
    for policy in POLICIES:
        for half in (CHOOSING, REPORTING):
            highest = policies[policy]["highest_wrong"][half]
            expected_lines.append(f"{policy}, {half}, a near-miss: {serve(highest[DIFFERENT])}")
            expected_lines.append(
                f"{policy}, {half}, a rewording served another answer: {serve(highest[SAME])}"
            )
    hard = cache["near_misses_above"]
    for policy, found in hard["policies"].items():
        named = ", ".join(
            f"{pair['pair']} at {pair['score']:.4f} ({pair['half']})" for pair in found["pairs"]
        )
        expected_lines.append(
            f"{policy}, near-misses it would serve at {hard['score']}: {named}. All of them are "
            f"on the choosing half in {found['other_splits_with_all_on_the_choosing_half']} of "
            f"the {others['splits']} other splits"
        )
    tested = part("What each half holds", "All agreed pairs at fixed thresholds")
    assert [line.strip() for line in tested[1:] if line.strip()] == expected_lines

    # All pairs at fixed thresholds, and at the rule's threshold with every pair in view.
    whole = cache["whole_set"]
    expected = []
    for policy in POLICIES:
        for row in (*whole["curves"][policy], whole["chosen_on_all_pairs"][policy]):
            same, different = row[SAME], row[DIFFERENT]
            expected.append(
                [
                    policy,
                    "none" if row["threshold"] is None else f"{row['threshold']:.4f}",
                    str(same[RIGHT]),
                    str(same[WRONG]),
                    str(same[MISSED]),
                    str(different["wrong_own"]),
                    str(different["wrong_other"]),
                    str(different[NOT_SERVED]),
                ]
            )
    fixed = part("All agreed pairs at fixed thresholds", "The same rules on")
    assert [line.split()[:8] for line in fixed[3:] if line.strip()] == expected
    assert sum("the rule, with every pair in view" in line for line in fixed) == len(POLICIES)

    # The other splits, and the same apart by where one pair fell.
    expected = [
        [
            str(others["policies"][policy]["chosen"]),
            str(others["policies"][policy]["splits_without_a_threshold"]),
            *over_splits(others["policies"][policy]),
        ]
        for policy in POLICIES
    ]
    expected.append(
        [
            str(others["splits"] - others["chosen_policy"]["splits_without_a_choice"]),
            *over_splits(others["chosen_policy"]),
        ]
    )
    splits = part("The same rules on", "The same, apart by where one pair fell")
    assert numbers(splits[0]) == [str(others["splits"])]
    assert figures(splits[2:]) == expected
    assert [line.split()[0] for line in splits[2:5]] == list(POLICIES)
    expected_lines = []
    for policy in POLICIES:
        fell = others["policies"][policy]["by_where_the_hardest_pair_fell"]
        for half in (CHOOSING, REPORTING):
            expected_lines.append(
                " ".join(
                    [
                        f"{policy}, {fell['pair']} on the {half} half",
                        f"{fell[half]['splits']} splits",
                        *over_splits(fell[half]),
                    ]
                )
            )
    apart = part("The same, apart by where one pair fell", "Disputed s078")
    assert [" ".join(line.split()) for line in apart[1:] if line.strip()] == expected_lines

    # The disputed pairs, which are in no figure.
    disputed = [line.strip() for line in part("Disputed s078", None) if line.strip()]
    assert disputed == [
        f"Disputed {pair['pair']} (maker: {pair['kind']}, second: {pair['second_label']}), in "
        f"no figure: cosine {three(pair[COSINE])}, entailment {three(pair['entails_forward'])} "
        f"forward and {three(pair['entails_backward'])} backward. Served: "
        + ", ".join(f"{policy}: {pair['served'][policy] or 'nothing'}" for policy in POLICIES)
        for pair in summary["disputed"]
    ]


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
    """A count as the documents print it, with its share where the run gives one."""
    if estimate is None:
        return f"{count} of {of}"
    share = f"{three(estimate['mean'])} ({three(estimate['low'])} to {three(estimate['high'])})"
    return f"{count} of {of}: {share}"


def shown(threshold: float | None) -> str:
    return "none" if threshold is None else f"{threshold:.4f}"


def at(curve: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    (row,) = [row for row in curve if row["threshold"] == threshold]
    return row


def named(line: str) -> list[str]:
    """The pairs and stored questions a line names in code spans, in order: `d067`, `u004`."""
    return re.findall(r"`([sdqub]\d{2,3})`", line)


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


def of_other_shapes(by_how: Any) -> tuple[int, int]:
    """Over the three ways that change a question's shape: how many pairs, how many served right."""
    return (
        sum(by_how[how]["pairs"] for how in OTHER_SHAPES),
        sum(by_how[how][RIGHT] for how in OTHER_SHAPES),
    )


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
            cell(same[RIGHT], same["pairs"], same.get("right_share")),
            cell(same[WRONG], same["pairs"], same.get("wrong_share")),
            cell(same["missed"], same["pairs"], same.get("missed_share")),
            cell(different[WRONG], different["pairs"], different.get("wrong_share")),
        ], policy
        # A share is given exactly where a threshold was chosen: a cache that serves
        # nothing by the rule has no rate to estimate.
        assert ("right_share" in same) == (policies[policy]["threshold"] is not None)
        assert (":" in row[2]) == (policies[policy]["threshold"] is not None)
    assert policies[COSINE]["threshold"] is None
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
    chosen = cache["policies"][cache["chosen_policy"]]
    return [
        [
            f"`{how}`",
            str(found["pairs"]),
            three(found["median"]),
            three(scores[ENTAILS_MIN]["by_how"][how]["median"]),
            three(scores[ENTAILS_MEAN]["by_how"][how]["median"]),
            str(chosen["choosing_by_how"][how]["pairs"]),
            str(chosen["choosing_by_how"][how][RIGHT]),
            str(chosen["reporting_by_how"][how]["pairs"]),
            str(chosen["reporting_by_how"][how][RIGHT]),
        ]
        for how, found in scores[COSINE]["by_how"].items()
    ]


def test_the_tables_by_how_in_the_guide_and_in_the_card_are_the_run() -> None:
    expected = by_how_rows(saved())

    assert [row[0] for row in expected] == [f"`{how}`" for how in cache_pairs.HOWS]
    assert table(guide(), "| `how` | Agreed pairs | Median cosine |") == expected
    assert table(card(), "| `how` | Agreed pairs | Median similarity |") == expected
    # The two halves together are the agreed pairs of each way.
    for row in expected:
        assert int(row[5]) + int(row[7]) == int(row[1]), row[0]


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


def test_the_cards_table_by_source_is_the_run() -> None:
    summary = saved()
    scores, split = summary["pair_view"]["scores"], summary["cache_view"]["split"]
    found = table(card(), "| `source` | Agreed pairs |")

    assert [row[0] for row in found] == [f"`{source}`" for source in cache_pairs.SOURCES]
    for row, source in zip(found, cache_pairs.SOURCES, strict=True):
        cosine = scores[COSINE]["by_source"][source]
        assert row[1:] == [
            str(cosine["pairs"]),
            three(cosine["min"]),
            three(cosine["median"]),
            three(cosine["max"]),
            three(scores[ENTAILS_MIN]["by_source"][source]["max"]),
            str(split[CHOOSING]["different_by_source"][source]),
            str(split[REPORTING]["different_by_source"][source]),
        ], source
        assert cosine["pairs"] == summary["pairs"]["different_by_source"][source]


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


def test_the_guides_table_by_where_the_hardest_pair_fell_is_the_run() -> None:
    summary = saved()
    others = summary["cache_view"]["other_splits"]
    with_all = summary["cache_view"]["whole_set"]["chosen_on_all_pairs"]
    found = table(guide(), "| Decided by | The pair | It is on |")

    expected = []
    for policy, name in DECIDED_BY.items():
        fell = others["policies"][policy]["by_where_the_hardest_pair_fell"]
        # The pair is the one that sets the threshold with every pair in view.
        assert fell["pair"] == with_all[policy]["decided_by"]["pair"]
        assert fell[CHOOSING]["splits"] + fell[REPORTING]["splits"] == others["splits"]
        for half in (CHOOSING, REPORTING):
            right = fell[half]["right_share"]
            expected.append(
                [
                    name,
                    f"`{fell['pair']}`",
                    f"the {half} half",
                    str(fell[half]["splits"]),
                    three(right["min"]),
                    three(right["median"]),
                    three(right["max"]),
                    str(fell[half]["splits_with_nothing_wrong"]),
                ]
            )
    assert found == expected


def test_the_guides_table_at_fixed_thresholds_is_the_run() -> None:
    whole = saved()["cache_view"]["whole_set"]
    found = table(guide(), "| Threshold | Cosine alone |")

    def counts(figures: Any) -> str:
        """Rewordings served right, rewordings served another answer, near-misses served."""
        return f"{figures[SAME][RIGHT]}, {figures[SAME][WRONG]}, {figures[DIFFERENT][WRONG]}"

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
    # The cells do not leave out the one rewording that is served another answer.
    assert any(at(whole["curves"][ENTAILS_MIN], float(row[0]))[SAME][WRONG] for row in fixed)


def test_what_the_guide_says_the_reporting_half_tested_is_what_the_run_holds() -> None:
    summary = saved()
    cache = summary["cache_view"]
    policies, others, hard = cache["policies"], cache["other_splits"], cache["near_misses_above"]
    line = line_starting(guide(), "**The reporting half did not test these thresholds.**")
    weaker, mean = (
        policies[policy]["highest_wrong"][REPORTING] for policy in (ENTAILS_MIN, ENTAILS_MEAN)
    )
    above = hard["policies"][ENTAILS_MIN]
    near_misses = str(policies[ENTAILS_MIN][REPORTING][DIFFERENT]["pairs"])

    assert numbers(line) == [
        near_misses,
        three(weaker[DIFFERENT]["score"]),
        three(mean[DIFFERENT]["score"]),
        str(policies[ENTAILS_MIN][REPORTING][DIFFERENT][WRONG]),
        near_misses,
        str(hard["score"]),
        *(three(pair["score"]) for pair in above["pairs"]),
        str(above["other_splits_with_all_on_the_choosing_half"]),
        str(others["splits"]),
        three(weaker[SAME]["score"]),
        three(mean[SAME]["score"]),
    ]
    assert named(line) == [
        weaker[DIFFERENT]["pair"],
        mean[DIFFERENT]["pair"],
        *(pair["pair"] for pair in above["pairs"]),
        weaker[SAME]["pair"],
    ]
    assert "None of its 50 near-misses was served" in line
    assert "scores three near-misses above 0.8 by its weaker direction" in line
    assert len(above["pairs"]) == 3
    assert {pair["half"] for pair in above["pairs"]} == {CHOOSING}
    # By the mean it is the same three pairs, on the same half.
    by_mean = hard["policies"][ENTAILS_MEAN]
    assert [pair["pair"] for pair in by_mean["pairs"]] == [pair["pair"] for pair in above["pairs"]]
    # "below the thresholds chosen too": none of these scores reaches its threshold.
    for policy, highest in ((ENTAILS_MIN, weaker), (ENTAILS_MEAN, mean)):
        for kind in (SAME, DIFFERENT):
            assert highest[kind]["score"] < policies[policy]["threshold"]
        assert mean[SAME]["pair"] == weaker[SAME]["pair"]
        assert policies[policy][REPORTING][DIFFERENT][WRONG] == 0


def test_what_the_guide_says_the_run_shows_is_what_the_run_holds() -> None:
    summary = saved()
    scores, cache = summary["pair_view"]["scores"], summary["cache_view"]
    counted, policies, others = summary["pairs"], cache["policies"], cache["other_splits"]
    curves, timing = cache["whole_set"]["curves"], summary["timing"]
    weaker = policies[ENTAILS_MIN]
    reported = weaker[REPORTING]
    questions = rows("questions.jsonl")
    pair_rows = rows("pairs.jsonl")

    def said(start: str) -> str:
        return line_starting(guide(), f"- **{start}")

    def says(start: str) -> list[str]:
        return numbers(said(start))

    start = "The cosine alone does not tell a rewording from a near-miss on these pairs."
    cosine = scores[COSINE]["by_kind"]
    at_95 = at(curves[COSINE], 0.95)
    assert says(start) == [
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
    assert named(said(start)) == [policies[COSINE]["decided_by"]["pair"]] == ["d012"]
    assert cosine[DIFFERENT]["max"] > cosine[SAME]["max"]
    assert policies[COSINE]["threshold"] is None
    assert policies[COSINE][REPORTING][SAME][RIGHT] == 0

    start = "The second check separates the kinds far better."
    by_what = scores[ENTAILS_MIN]["by_what_differs"]
    assert says(start) == [
        three(scores[ENTAILS_MIN]["area_under_curve"]),
        three(scores[ENTAILS_MEAN]["area_under_curve"]),
        str(by_what["number"]["pairs"]),
        three(by_what["number"]["max"]),
        str(by_what["negation"]["pairs"]),
        three(by_what["negation"]["max"]),
    ]
    assert "By the weaker direction, the highest of the 12 pairs that change a number" in said(
        start
    )
    assert "the 15 `negation` pairs, which ask the opposite of their golden question" in said(start)

    start = "It serves little, and what it served here is mostly plain rewordings."
    there, chose = weaker["reporting_by_how"], weaker["choosing_by_how"]
    assert says(start) == [
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[SAME]["missed"]),
        str(reported[SAME][RIGHT]),
        str(there["reworded"][RIGHT]),
        str(there["reworded"]["pairs"]),
        str(there["synonyms"][RIGHT]),
        str(there["synonyms"]["pairs"]),
        str(of_other_shapes(there)[0]),
        str(of_other_shapes(chose)[0]),
    ]
    assert there["reworded"][RIGHT] + there["synonyms"][RIGHT] == reported[SAME][RIGHT]
    # "None of its 31 questions of the other three kinds was served ... On the choosing
    # half one of the 27 of those kinds was, `s062`, an `indirect` one."
    assert of_other_shapes(there) == (31, 0)
    assert of_other_shapes(chose) == (27, 1)
    assert "None of its 31 questions of the other three kinds was served" in said(start)
    assert "On the choosing half one of the 27 of those kinds was" in said(start)
    served_there = [
        pair
        for pair, row in questions.items()
        if row["half"] == CHOOSING
        and pair_rows[pair]["how"] in OTHER_SHAPES
        and row["outcome"][ENTAILS_MIN] == RIGHT
    ]
    assert named(said(start)) == served_there == ["s062"]
    assert pair_rows["s062"]["how"] == "indirect"
    assert "For most of those the weaker direction of entailment is low" in said(start)

    start = "One near-miss set the threshold, and it is a stored decline."
    at_90 = at(curves[ENTAILS_MIN], 0.9)
    assert says(start) == [
        three(weaker["decided_by"]["score"]),
        "0.9",
        "0.9",
        str(at_90[SAME][RIGHT]),
        str(counted[SAME]),
        str(at_90[DIFFERENT][WRONG]),
        str(counted[DIFFERENT]),
    ]
    by = weaker["decided_by"]
    assert (by["pair"], by["group"], by["served"], by["own"]) == ("d067", "aspect", "u004", True)
    assert at_90[SAME][WRONG] == 0
    # "`d081` and `u046` ... These two are the only near-misses the second check scores above 0.9"
    above_90 = {
        pair: row["question_id"]
        for pair, row in pair_rows.items()
        if row["agreed"] and row["kind"] == DIFFERENT and row[ENTAILS_MIN] >= 0.9
    }
    assert above_90 == {"d067": "u004", "d081": "u046"}
    assert named(said(start)) == ["d067", "u004", "d081", "u046"]
    # Each is a stored decline: a golden question the handbook does not answer.
    assert set(above_90.values()) <= unanswered_questions(golden())
    assert by["stored_unanswered"] is True
    # "by the pair's note the question asked has an answer"
    notes = {pair.pair: pair.note for pair in pairs()}
    assert all(notes[pair].startswith("Differs: this is answerable") for pair in above_90)
    assert "A cache that stores no declines would have met neither." in said(start)
    assert "That variant was not measured" in said(start)

    start = "None wrong on the reporting half says little."
    of_reporting = cache["split"][REPORTING]
    near_misses = str(reported[DIFFERENT]["pairs"])
    assert says(start) == [
        near_misses,
        three(cache["zero_wrong_allows"]["up_to"]),
        near_misses,
        str(of_reporting["groups_of"][DIFFERENT]["groups"]),
        str(of_reporting["groups_of"][DIFFERENT]["largest"]),
    ]
    assert reported[DIFFERENT][WRONG] == 0
    assert cache["zero_wrong_allows"]["near_misses"] == reported[DIFFERENT]["pairs"]

    start = "Where one pair falls decides what a split shows."
    fell = others["policies"][ENTAILS_MIN]["by_where_the_hardest_pair_fell"]
    assert says(start) == [
        str(others["splits"]),
        str(others["chosen_policy"]["splits_with_something_wrong"]),
        str(fell[REPORTING]["splits"]),
        str(fell[CHOOSING]["splits"]),
        three(fell[CHOOSING]["right_share"]["median"]),
        three(fell[CHOOSING]["right_share"]["min"]),
        three(fell[CHOOSING]["right_share"]["max"]),
        str(fell[REPORTING]["splits_with_something_wrong"]),
        three(fell[REPORTING]["right_share"]["median"]),
        three(fell[REPORTING]["right_share"]["min"]),
        three(fell[REPORTING]["right_share"]["max"]),
        str(others["splits"]),
        three(others["policies"][ENTAILS_MIN]["right_share"]["median"]),
        three(reported[SAME]["right_share"]["mean"]),
    ]
    assert named(said(start)) == [fell["pair"], fell["pair"]] == ["d067", "d067"]
    for policy in (ENTAILS_MIN, ENTAILS_MEAN):
        apart = others["policies"][policy]["by_where_the_hardest_pair_fell"]
        # "Those 97 are exactly the splits that put `d067` on the reporting half, by either way"
        assert apart["pair"] == "d067"
        assert apart[REPORTING]["splits_with_nothing_wrong"] == 0
        assert apart[CHOOSING]["splits_with_something_wrong"] == 0
        assert (
            apart[REPORTING]["splits"] == others["policies"][policy]["splits_with_something_wrong"]
        )
    # "as in the split reported": the pair is on the choosing half of the run's own split.
    assert pair_rows["d067"]["half"] == CHOOSING
    low, high = fell[CHOOSING]["right_share"]["min"], fell[CHOOSING]["right_share"]["max"]
    assert low <= reported[SAME]["right_share"]["mean"] <= high

    start = "Choosing with every pair in view would have changed nothing here."
    with_all = cache["whole_set"]["chosen_on_all_pairs"]
    chose_right = weaker[CHOOSING][SAME]
    assert says(start) == [
        f"{with_all[ENTAILS_MIN]['threshold']:.4f}",
        f"{weaker['threshold']:.4f}",
        str(with_all[ENTAILS_MIN][SAME][RIGHT]),
        str(counted[SAME]),
        str(chose_right[RIGHT]),
        str(reported[SAME][RIGHT]),
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(chose_right[RIGHT]),
        str(chose_right["pairs"]),
        str(chose["reworded"]["pairs"]),
        str(scores[COSINE]["by_how"]["reworded"]["pairs"]),
    ]
    assert with_all[ENTAILS_MIN][SAME][RIGHT] == chose_right[RIGHT] + reported[SAME][RIGHT]
    # "serves the same questions": none scores between the two thresholds.
    between = sorted((with_all[ENTAILS_MIN]["threshold"], weaker["threshold"]))
    assert not [
        pair
        for pair, row in pair_rows.items()
        if row["agreed"] and between[0] <= row[ENTAILS_MIN] < between[1]
    ]
    for policy in (ENTAILS_MIN, ENTAILS_MEAN):
        assert with_all[policy][SAME][WRONG] == with_all[policy][DIFFERENT][WRONG] == 0

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
    scores, cache = summary["pair_view"]["scores"], summary["cache_view"]
    curves, others = cache["whole_set"]["curves"], cache["other_splits"]
    by_source = summary["pairs"]["different_by_source"]
    of_reporting = cache["split"][REPORTING]

    def said(start: str) -> str:
        return line_starting(guide(), f"- **{start}")

    def highest_with_another_answer(policy: str) -> str:
        return str(max(row["threshold"] for row in curves[policy] if row[SAME][WRONG]))

    start = "The set is hard on purpose, and real questions are not."
    golden_pairs = scores[COSINE]["by_source"]["golden"]
    assert numbers(said(start)) == [
        str(by_source["written"]),
        str(by_source["golden"]),
        three(golden_pairs["min"]),
        three(golden_pairs["max"]),
        str(of_reporting["different_by_source"]["golden"]),
        str(of_reporting[DIFFERENT]),
        str(others["chosen_policy"]["splits_with_something_wrong"]),
        str(others["splits"]),
    ]
    # The claim that was wrong is gone, and what is said instead is that nothing measures it.
    assert "is stricter than real use needs is not measured" in said(start)
    for document in (GUIDE, CARD, ROOT / "README.md"):
        text = document.read_text(encoding="utf-8")
        assert "safe side" not in text, document.name
        assert "will be stricter than real use needs" not in text, document.name
        for sentence in re.findall(r"[^.]*stricter than real use needs[^.]*\.", text):
            assert "is not measured" in sentence, sentence

    assert numbers(said("One embedding model, one verifier, and a cache of 165 questions.")) == [
        str(cache["stored_questions"])
    ]
    start = "A serve of another stored question's answer rests on no label."
    assert numbers(said(start)) == [
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
    assert named(said(start)) == [pair for pair, _ in another] == ["s073"]


def test_what_the_guide_says_of_the_method_is_the_method_of_the_run() -> None:
    summary = saved()
    counted, cache = summary["pairs"], summary["cache_view"]
    split, above = cache["split"], cache["stored_above_floor"]
    weaker = cache["policies"][ENTAILS_MIN]
    by_source = counted["different_by_source"]
    text = guide()

    assert (
        f"about the {counted['agreed']} pairs on which the two agree, {counted[SAME]} `same` and "
        f"{counted[DIFFERENT]} `different`"
    ) in text
    assert (
        f"Of the near-misses, {by_source['written']} were written to change one decisive detail"
    ) in text
    assert f"The other {by_source['golden']} put two golden questions" in text
    assert "The two disputed pairs, `s078` and `s100`, are in no figure" in text
    assert f"holds all {cache['stored_questions']} golden questions is simulated" in text
    assert (
        f"The {cache['stored_unanswered']} that the handbook does not answer are in it too" in text
    )
    # How a question is served, word for word: the rule the simulation applies.
    assert (
        f"the {cache['candidates']} nearest above a floor of {cache['floor']}. They are tried "
        "nearest first, and the first whose score reaches the threshold is served."
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

    # What a pair meets beyond the questions it names.
    line = line_starting(text, "- **That holds for the questions a pair names")
    met = split[REPORTING]["candidates"]
    assert numbers(line) == [
        str(cache["stored_questions"]),
        str(met["named_by_the_other_half"]),
        str(met["not_own"]),
    ]
    # The halfway: the threshold, what it is between, and the question it lets through.
    line = line_starting(text, "- **The halfway is a choice.**")
    (by_margin,) = weaker["right_only_by_the_margin"]
    assert numbers(line) == [
        f"{weaker['threshold']:.4f}",
        f"{weaker['decided_by']['score']:.4f}",
        f"{weaker['halfway_to']:.4f}",
        f"{by_margin['score']:.4f}",
    ]
    assert named(line) == [by_margin["pair"]] == ["s013"]
    assert weaker["decided_by"]["score"] < weaker["threshold"] <= by_margin["score"]
    assert by_margin["score"] < weaker["halfway_to"]
    assert weaker["threshold"] == (weaker["decided_by"]["score"] + weaker["halfway_to"]) / 2
    # What a reader can check of the seed, and what rests on an account.
    assert "The seed is a constant in the code, and the command has no option for it." in text
    assert "cannot be checked from the repository" in text
    assert "is the maker's account" in text
    assert "By the account of the session that planned this measurement" in text
    assert "No figure of that scratch run is in the repository" in text


def test_what_the_card_says_was_measured_is_what_the_run_holds() -> None:
    summary = saved()
    scores, cache = summary["pair_view"]["scores"], summary["cache_view"]
    counted, policies, others = summary["pairs"], cache["policies"], cache["other_splits"]
    weaker = policies[cache["chosen_policy"]]
    reported, split = weaker[REPORTING], cache["split"]
    hard = cache["near_misses_above"]

    def said(start: str) -> str:
        return line_starting(card(), f"- **{start}")

    def says(start: str) -> list[str]:
        return numbers(said(start))

    assert cache["chosen_policy"] == ENTAILS_MIN
    assert says("A simulated cache.") == [
        str(cache["stored_questions"]),
        str(cache["stored_unanswered"]),
        str(cache["candidates"]),
        str(counted["other_is_a_stored_question"]),
    ]
    assert says("Two halves.") == [
        str(split[REPORTING]["pairs"]),
        str(split[REPORTING][SAME]),
        str(split[REPORTING][DIFFERENT]),
    ]
    for name in ("pairs", SAME, DIFFERENT):
        assert split[CHOOSING][name] == split[REPORTING][name]
    assert says("The similarity of the embeddings alone does not tell the two kinds apart.") == [
        three(scores[COSINE]["area_under_curve"]),
        "0.5",
        str(counted["agreed"]),
    ]
    by_kind = scores[COSINE]["by_kind"]
    assert by_kind[DIFFERENT]["max"] > by_kind[SAME]["max"]
    assert policies[COSINE]["threshold"] is None

    start = "The second model separates them far better"
    assert says(start) == [
        three(scores[ENTAILS_MIN]["area_under_curve"]),
        three(scores[ENTAILS_MEAN]["area_under_curve"]),
        str(reported[SAME][RIGHT]),
        str(reported[SAME]["pairs"]),
        str(reported[DIFFERENT]["pairs"]),
    ]
    assert reported[DIFFERENT][WRONG] == 0
    assert "and none of the 50 `different` ones" in said(start)

    start = "None of 50 was no test of that threshold."
    fell = others["policies"][ENTAILS_MIN]["by_where_the_hardest_pair_fell"]
    near_misses = str(reported[DIFFERENT]["pairs"])
    assert says(start) == [
        near_misses,
        three(weaker["highest_wrong"][REPORTING][DIFFERENT]["score"]),
        str(len(hard["policies"][ENTAILS_MIN]["pairs"])),
        str(hard["score"]),
        near_misses,
        three(cache["zero_wrong_allows"]["up_to"]),
        near_misses,
        str(split[REPORTING]["groups_of"][DIFFERENT]["groups"]),
        str(others["splits"]),
        str(others["chosen_policy"]["splits_with_something_wrong"]),
    ]
    assert {pair["half"] for pair in hard["policies"][ENTAILS_MIN]["pairs"]} == {CHOOSING}
    assert named(said(start)) == [fell["pair"]] == ["d067"]
    assert fell[REPORTING]["splits"] == others["chosen_policy"]["splits_with_something_wrong"]
    assert fell[REPORTING]["splits_with_nothing_wrong"] == 0
    assert fell[CHOOSING]["splits_with_something_wrong"] == 0

    first, second = summary["disputed"]
    assert says("The two disputed pairs are in no figure.") == [
        *(three(first[score]) for score in (COSINE, "entails_forward", "entails_backward")),
        *(three(second[score]) for score in (COSINE, "entails_forward", "entails_backward")),
    ]
    assert (first["pair"], second["pair"]) == ("s078", "s100")
    for pair in (first, second):
        assert set(pair["served"].values()) == {None}

    # "For the similarity it is d012 ... For the second model it is d067, an aspect pair"
    start = "A single pair decides each threshold."
    assert named(said(start)) == [
        policies[COSINE]["decided_by"]["pair"],
        policies[ENTAILS_MIN]["decided_by"]["pair"],
    ]
    assert named(said(start)) == ["d012", "d067"]
    assert policies[ENTAILS_MEAN]["decided_by"]["pair"] == "d067"
    assert "`d067`, an `aspect` pair" in said(start)
    assert weaker["decided_by"]["group"] == "aspect"

    start = "The two `different` pairs the second model scores highest are stored declines."
    above_90 = [pair for pair in hard["policies"][ENTAILS_MIN]["pairs"] if pair["score"] >= 0.9]
    assert says(start) == ["0.9"]
    assert named(said(start)) == [
        *(pair["pair"] for pair in above_90),
        *(pair["served"] for pair in above_90),
    ]
    assert named(said(start)) == ["d067", "d081", "u004", "u046"]
    assert all(pair["stored_unanswered"] and pair["own"] for pair in above_90)
    assert at(cache["whole_set"]["curves"][ENTAILS_MIN], 0.9)[DIFFERENT][WRONG] == len(above_90)
    assert "A cache that stores no declines would have met neither of them." in said(start)
    assert "That was not measured." in said(start)

    start = "The second model passes plain rewordings, and few others."
    there, chose = weaker["reporting_by_how"], weaker["choosing_by_how"]
    assert says(start) == [
        str(there["reworded"][RIGHT]),
        str(there["reworded"]["pairs"]),
        str(there["synonyms"][RIGHT]),
        str(there["synonyms"]["pairs"]),
        str(of_other_shapes(there)[0]),
        str(of_other_shapes(chose)[0]),
    ]
    assert (of_other_shapes(there)[1], of_other_shapes(chose)[1]) == (0, 1)
    assert "and none of the 31 `short`, `long` and `indirect` ones" in said(start)
    assert "it served one of the 27 of those three kinds, `s062`" in said(start)
    assert "Most of them have at least one direction of entailment that is low" in said(start)

    start = "The 13 `golden` pairs are not close."
    by_source = counted["different_by_source"]
    assert says(start) == [
        str(by_source["golden"]),
        str(split[REPORTING]["different_by_source"]["golden"]),
        str(split[REPORTING][DIFFERENT]),
        str(by_source["written"]),
    ]
    assert (
        scores[COSINE]["by_source"]["golden"]["max"] < scores[COSINE]["by_source"]["written"]["max"]
    )
    assert "Whether a threshold chosen here is stricter than real use needs is not measured." in (
        said("The set is hard on purpose.")
    )
    assert f"`{summary['config']['verifier_model']}`" in card()
    # The card says what of the second labelling the repository cannot show.
    whole_card = CARD.read_text(encoding="utf-8")
    assert "The second session's own file of labels and that key are not in the repository." in (
        whole_card
    )
    assert "cannot be checked from the repository" in whole_card


def test_the_words_that_say_which_way_a_finding_points_are_in_the_documents() -> None:
    # A changed number is caught by the tests above. These are the words between the
    # numbers, where "none" for "one" or "choosing" for "reporting" would turn a finding
    # round without changing a figure.
    text = guide()
    assert "on the reporting half, 68 of the 201" in text
    assert "all three are on the choosing half" in text
    assert "nothing wrong was served in any of the 103" in text
    assert "something wrong was served in all 97" in text
    assert "On the choosing half it sets the threshold, or for the cosine leaves none." in text
    card = CARD.read_text(encoding="utf-8")
    assert (
        "The highest score at which a `different` pair of the reporting half would be served is"
        in card
    )
    assert "are all on the choosing half" in card


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
        str(others["splits"]),
        str(others["chosen_policy"]["splits_with_something_wrong"]),
    ]
    # The last number is the issue the entry links to.
    assert numbers(changelog) == [
        str(summary["pairs"]["in_file"]),
        *served_right,
        near_misses,
        "24",
    ]
    # "none of 50 near-misses" is a count of the run too, written as a word.
    assert reported[DIFFERENT][WRONG] == 0
    for line in (roadmap, changelog):
        assert "served 7 of 49 rewordings and none of 50 near-misses" in line
        assert "no near-miss hard enough to test" in line
    assert cache["policies"][COSINE]["threshold"] is None
    # "the 97 that put the hardest near-miss there"
    fell = others["policies"][ENTAILS_MIN]["by_where_the_hardest_pair_fell"]
    assert fell[REPORTING]["splits"] == others["chosen_policy"]["splits_with_something_wrong"]
    assert fell[REPORTING]["splits_with_nothing_wrong"] == 0
    highest = cache["policies"][ENTAILS_MIN]["highest_wrong"][REPORTING][DIFFERENT]["score"]
    assert highest < cache["policies"][ENTAILS_MIN]["threshold"] / 2


def test_the_documents_give_the_day_the_run_records() -> None:
    made = day(saved())
    readme = section((ROOT / "README.md").read_text(encoding="utf-8"), HEADING)

    assert f"`baseline`, on {made}, with" in guide()
    assert f"On {made} `handbook-rag eval-cache` measured" in card()
    assert f"The run, on {made}, is" in readme
    for text in (guide(), card(), readme):
        # No other day is given in what is written about the run.
        assert re.findall(r"\d{1,2} [A-Z][a-z]+ \d{4}", text) == [made]


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
    summary = saved()
    made = datetime.date.fromisoformat(summary["measured_on"])
    chosen = summary["cache_view"]["policies"][summary["cache_view"]["chosen_policy"]]
    allowed = numbers_of_the_run(summary)
    allowed |= {
        "24",  # the issue
        str(made.day),
        str(made.year),
        "59",  # golden questions in more than one pair, held to the pairs above
        # Two sums of counts the run holds: the pairs of the three ways of rewording that
        # change a question's shape, on each half. The tests above work them out.
        str(of_other_shapes(chosen["reporting_by_how"])[0]),
        str(of_other_shapes(chosen["choosing_by_how"])[0]),
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
