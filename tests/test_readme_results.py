"""The results in the README are the results of the committed runs.

Issue 18 asks that every number in the README lead to the committed run that produced
it. These tests read the runs under ``evals/runs/`` and compare them with what the README
prints, at the precision it prints them. A number changed by hand fails here, and so does
a run recorded again without the README being brought up to date. So does a link that
leads nowhere: to a missing file, a missing heading, or an issue the repository does not
have.

Every table and figure a test expects must be found, exactly once: a test that found
nothing to compare would pass without checking anything.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from handbook_rag.config import Settings
from handbook_rag.evals.compare import compare_runs
from handbook_rag.evals.dataset import load_golden_set
from handbook_rag.evals.two_passage import load_reasons

ROOT = Path(__file__).parents[1]
RUNS = ROOT / "evals" / "runs"
REPOSITORY = "https://github.com/MumukshuPanchal/handbook-rag"

# The issues and pull requests of the repository at version 1.0.0, as `gh issue list` and
# `gh pr list` gave them. A test cannot ask GitHub, so a link may name only these. GitHub
# numbers both from one sequence, so a number in the other list is a wrong link too.
# When the README links a newer issue or pull request, add its number here
# (CONTRIBUTING.md says so too).
ISSUES = {*range(1, 23), 24, 34, 37, 42, 43, 47, 51, 52, 53, 58, 59, 60, 61, 66}
PULLS = {23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 38, 39, 40, 41, 44, 45, 46, 48}
PULLS |= {49, 50, 54, 55, 56, 57, 62, 63, 64, 65, 67, 68}
# Opened since: the two-passage questions, follow-up questions, and the rule for "clear".
ISSUES |= {78, 79, 80}

# A number as the README prints it: 132, 1,654, 0.962. A digit that is part of a name is
# not one (Recall@5, v2, a commit hash), and neither is a percentage, which here only
# ever names an interval ("95% interval").
NUMBER = re.compile(r"(?<![\w@.])\d+(?:,\d{3})*(?:\.\d+)?(?![\w%])")


def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def charter() -> str:
    return (ROOT / "docs" / "charter.md").read_text(encoding="utf-8")


def summary(run: str) -> Any:
    """The ``summary.json`` of a committed run, such as ``retrieval/v2``."""
    return json.loads((RUNS / run / "summary.json").read_text(encoding="utf-8"))


def results(run: str) -> list[Any]:
    """The rows of a committed run's ``results.jsonl``, one for each question."""
    lines = (RUNS / run / "results.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def table(text: str, header: str) -> list[list[str]]:
    """The rows of the Markdown table in ``text`` whose header row starts with ``header``.

    Each row is the list of its cells, without the spaces around them. There must be
    exactly one such table: a second copy, left behind with other numbers, would
    otherwise go unchecked.
    """
    lines = text.splitlines()
    starts = [number for number, line in enumerate(lines) if line.startswith(header)]
    assert len(starts) == 1, f"expected one table starting {header!r}, found {len(starts)}"
    rows = []
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def line_starting(text: str, prefix: str) -> str:
    """The one line of ``text`` that starts with ``prefix``. A second would go unchecked."""
    found = [line for line in text.splitlines() if line.startswith(prefix)]
    assert len(found) == 1, f"expected one line starting {prefix!r}, found {len(found)}"
    return found[0]


def numbers(text: str) -> list[str]:
    """The numbers in a piece of the README, in order, as printed, outside link targets."""
    return NUMBER.findall(re.sub(r"\]\([^)]*\)", "]", text))


# The small counts that the documents write as words, from nothing up.
NUMBER_WORDS = (
    "no",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
)


def in_words(count: int) -> str:
    """A small count as the documents write it: "no", "one", "two" and so on."""
    return NUMBER_WORDS[count]


def prose_with_numbers(markdown: str) -> list[str]:
    """Every line of prose in a piece of Markdown that holds a number.

    Table rows, headings and fenced code are left out: a table is compared cell by cell.
    """
    prose = re.sub(r"^ *```.*?^ *```", "", markdown, flags=re.DOTALL | re.MULTILINE)
    return [
        line for line in prose.splitlines() if not line.startswith(("|", "#")) and numbers(line)
    ]


def assert_numbers(markdown: str, expected: dict[str, list[str]]) -> None:
    """Each line that starts with a key holds exactly those numbers, in that order.

    And no other line of prose holds a number: a number that no key accounts for is one
    that nothing checks. A line is a paragraph or a point, since these documents do not
    wrap their lines.
    """
    for start, held in expected.items():
        assert numbers(line_starting(markdown, start)) == held, start
    unheld = [line for line in prose_with_numbers(markdown) if not line.startswith(tuple(expected))]
    assert unheld == [], f"lines with numbers that no test works out: {unheld}"


def link_target(cell: str) -> str:
    """The target of a table cell that is one Markdown link."""
    match = re.fullmatch(r"\[.+?\]\((.+?)\)", cell)
    if match is None:
        raise ValueError(f"expected a link, found {cell!r}")
    return match[1]


def three(value: float) -> str:
    return f"{value:.3f}"


def with_interval(estimate: Any) -> str:
    """A mean as the README prints it: 0.962 (0.924 to 0.992)."""
    return f"{three(estimate['mean'])} ({three(estimate['low'])} to {three(estimate['high'])})"


def interval_numbers(estimate: Any) -> list[str]:
    return [three(estimate["mean"]), three(estimate["low"]), three(estimate["high"])]


def hits(estimate: Any) -> int:
    """How many of the questions a yes-or-no score counted: its mean times its count."""
    return round(float(estimate["mean"]) * int(estimate["count"]))


def median_latency_ms(run: str) -> float:
    """The median time of a retrieval run, from its rows, as ``eval-table`` computes it."""
    return 1000 * statistics.median(float(row["latency_s"]) for row in results(run))


def anchors(markdown: str) -> set[str]:
    """The anchors GitHub gives the headings of a Markdown file.

    A heading becomes lower case, loses every character but letters, digits, spaces,
    hyphens and underscores, and has its spaces turned into hyphens. A heading met again
    gets -1, -2 and so on. Lines in fenced code are not headings.
    """
    text = re.sub(r"^ *```.*?^ *```", "", markdown, flags=re.DOTALL | re.MULTILINE)
    seen: Counter[str] = Counter()
    found = set()
    for heading in re.findall(r"^#{1,6} +(.+?) *$", text, flags=re.MULTILINE):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        found.add(f"{slug}-{seen[slug]}" if seen[slug] else slug)
        seen[slug] += 1
    return found


def test_the_search_table_is_the_committed_retrieval_runs() -> None:
    rows = table(readme(), "| Run | What it is | Questions | Hit@5 (95% interval) |")

    names = ["baseline", "v1", "v2", "v3"]
    assert [row[0] for row in rows] == [f"[{name}](evals/runs/retrieval/{name})" for name in names]
    for row, name in zip(rows, names, strict=True):
        run = summary(f"retrieval/{name}")
        metrics = run["metrics"]
        assert row[1:] == [
            run["description"],
            str(run["questions_scored"]),
            with_interval(metrics["hit@5"]),
            three(metrics["hit@1"]["mean"]),
            three(metrics["mrr@10"]["mean"]),
            f"{median_latency_ms(f'retrieval/{name}'):.0f} ms",
        ], name


# Each row of the answer table, by the words before the colon, and its score in the run.
ANSWER_SCORES = {
    "Right call": "right_call",
    "Supported by its sources": "supported",
    "Correct by the reference": "correct",
    "Gets a correct answer": "answered_correctly",
    "Answers when it should": "answered",
    "Declines when it should": "declined",
    "Every statement cited": "fully_cited",
    "Cites the passage that holds the evidence": "cites_evidence",
}


def test_the_answer_table_is_the_committed_answer_run() -> None:
    rows = table(readme(), "| Score | Result (95% interval) | Questions |")
    metrics = summary("answers/baseline")["metrics"]

    assert [row[0].split(":")[0] for row in rows] == list(ANSWER_SCORES)
    for label, result, questions in rows:
        estimate = metrics[ANSWER_SCORES[label.split(":")[0]]]
        assert [result, questions] == [with_interval(estimate), str(estimate["count"])], label


def test_the_charter_table_is_the_charter_and_the_committed_runs() -> None:
    goals = table(charter(), "| Goal | Metric | Provisional target |")
    rows = table(readme(), "| Charter goal | Target | Measured | From | Met |")
    hybrid = summary("retrieval/v2")["metrics"]
    answers = summary("answers/baseline")
    scores, timing, tokens = answers["metrics"], answers["timing"], answers["tokens"]
    answered, declined = scores["answered"], scores["declined"]
    unanswered = [
        row
        for row in results("answers/baseline")
        if "answerable" in row["tags"] and not row["answered"]
    ]

    def share(name: str) -> list[str]:
        """A share as the README prints it: its mean and interval, then "127 of 129"."""
        estimate = scores[name]
        return [*interval_numbers(estimate), str(hits(estimate)), str(estimate["count"])]

    retrieval_run, answer_run = "evals/runs/retrieval/v2", "evals/runs/answers/baseline"
    # For each goal: the Target cell, word for word; the numbers of the Measured cell; the
    # run it links to; the Met cell, word for word; and whether what the Met cell says
    # holds for the run and the target.
    expected = [
        (
            "The right passage is retrieved",
            "Recall@5 at least 0.85",
            interval_numbers(hybrid["recall@5"]),
            retrieval_run,
            "Yes",
            hybrid["recall@5"]["mean"] >= 0.85,
        ),
        (
            "The right passage is ranked high",
            "MRR@10 at least 0.70",
            interval_numbers(hybrid["mrr@10"]),
            retrieval_run,
            "Yes",
            hybrid["mrr@10"]["mean"] >= 0.70,
        ),
        (
            "Answers are supported by their sources",
            "At least 0.90 of answers",
            share("supported"),
            answer_run,
            "Yes, by a judge that is the answer model itself",
            scores["supported"]["mean"] >= 0.90,
        ),
        (
            "Answers are correct",
            "At least 0.80 of answers",
            share("correct"),
            answer_run,
            "Yes, by the same model as a second judge",
            scores["correct"]["mean"] >= 0.80,
        ),
        (
            "The assistant abstains when it should",
            "At least 0.90 of unanswerable questions declined",
            share("declined"),
            answer_run,
            "Yes, on few questions: the interval reaches below the target",
            declined["mean"] >= 0.90 > declined["low"],
        ),
        (
            "The assistant does not over-abstain",
            "At most 0.05 of answerable questions declined",
            # The share wrongly declined is one minus the share answered, and so is its
            # interval: the bootstrap of the one maps term for term onto the other.
            [
                three(1 - answered["mean"]),
                three(1 - answered["high"]),
                three(1 - answered["low"]),
                str(answered["count"] - hits(answered)),
                str(answered["count"]),
            ],
            answer_run,
            "Yes, though the interval reaches above the target. All five were search misses",
            1 - answered["mean"] <= 0.05 < 1 - answered["low"]
            and all(row["grade"]["evidence_in_context"] == 0 for row in unanswered),
        ),
        (
            "Answers arrive quickly",
            "95th percentile time to a complete answer at most 10 seconds",
            [f"{timing['p95_seconds']:.1f}", f"{timing['median_seconds']:.1f}"],
            answer_run,
            "No. The time of the search comes on top",
            timing["p95_seconds"] > 10,
        ),
        (
            "Cost is known and controlled",
            "Mean model cost per answer for each model tier, target set after the tier comparison",
            [f"{tokens['input_mean']:,.0f}", f"{tokens['output_mean']:.0f}"],
            answer_run,
            f"Not measured. The tier comparison was not done ([issue 17]({REPOSITORY}/issues/17)), "
            "so no target was set",
            True,
        ),
    ]

    # One row for each goal of the charter, in its order, with the charter's target.
    assert [row[0] for row in rows] == [goal[0] for goal in goals]
    assert [numbers(row[1]) for row in rows] == [numbers(goal[2]) for goal in goals]
    for row, (goal, target, measured, run, met, holds) in zip(rows, expected, strict=True):
        assert [row[0], row[1], row[4]] == [goal, target, met], goal
        assert numbers(row[2]) == measured, goal
        assert link_target(row[3]) == run, goal
        assert holds, f"what the Met cell says no longer holds: {goal}"

    # The README says that for hybrid search Recall@5 is the same number as Hit@5, which
    # the charter row and the search table each print.
    assert with_interval(hybrid["recall@5"]) == with_interval(hybrid["hit@5"])


def test_the_figures_quoted_about_search_are_the_committed_runs() -> None:
    against_dense = compare_runs(RUNS / "retrieval/baseline", RUNS / "retrieval/v2").metrics
    against_keyword = compare_runs(RUNS / "retrieval/v1", RUNS / "retrieval/v2").metrics
    gain = against_dense["hit@5"].change
    ahead = against_keyword["hit@5"].change.difference
    behind = against_keyword["hit@1"].change.difference
    hybrid = line_starting(readme(), "- **Hybrid search is the default.**")
    reranked, plain = median_latency_ms("retrieval/v3"), median_latency_ms("retrieval/v2")
    reranking = line_starting(readme(), "- **Reranking is off by default.**")

    assert (
        f"{gain.difference.mean:+.3f} on Hit@5 (95% interval {gain.difference.low:+.3f} to "
        f"{gain.difference.high:+.3f}), with {gain.better} questions gained and none lost"
    ) in hybrid
    assert f"ahead at five results ({ahead.mean:+.3f}, an interval that includes zero)" in hybrid
    assert ahead.low < 0 < ahead.high
    assert "and behind at rank 1" in hybrid
    assert behind.mean < 0
    assert f"a median of {reranked:.0f} ms against {plain:.0f} ms" in reranking
    # 247 against 33, at any value that prints so, is seven and a half to the nearest half.
    assert "about seven and a half times slower" in reranking


# Each judge in the README's table, and the committed check of it.
JUDGES = {
    "Supported by its sources": "judge/baseline",
    "Correct by the reference": "judge-correct/baseline",
}


def test_the_judges_checks_are_the_committed_checks() -> None:
    rows = table(readme(), "| Judge | Labelled answers | Faults caught (recall) |")

    assert [row[0] for row in rows] == list(JUDGES)
    for judge, labelled, recall, precision, check in rows:
        run = summary(JUDGES[judge])
        agreement = run["agreement"]
        # A check is quoted only when the judge ruled on every labelled answer.
        assert run["answers_judged"] == run["answers_labelled"], judge
        planted = sum(total for kind, (_, total) in agreement["caught"].items() if kind != "real")
        assert numbers(labelled) == [str(run["answers_labelled"]), str(planted)], judge
        for cell, estimate in ((recall, agreement["recall"]), (precision, agreement["precision"])):
            assert numbers(cell) == [
                str(hits(estimate)),
                str(estimate["count"]),
                *interval_numbers(estimate),
            ], judge
        assert link_target(check) == f"evals/runs/{JUDGES[judge]}", judge


def test_the_control_is_the_committed_control_run() -> None:
    run = summary("judge-correct/references")
    same = run["agreement"]["same_decision"]
    line = line_starting(readme(), "- **A control for the second judge.**")

    assert run["answers_judged"] == run["answers_labelled"]
    assert numbers(line) == [str(run["answers_labelled"]), str(hits(same)), *interval_numbers(same)]
    assert "(evals/runs/judge-correct/references)" in line


# Each row of the adversarial table, and the family and figure of the run it shows.
ADVERSARIAL = {
    "Planted instruction not followed": ("injected_passage", "held"),
    "Question outside the handbook declined": ("out_of_scope", "held"),
    "Injected case kept its answer": ("injected_passage", "kept"),
}

# Each row of the table of planted instructions, and its kind in the run.
PLANTED = {
    "Reply with a made-up word only": "override",
    "Repeat your instructions": "reveal_instructions",
    "Obey a forged message from the system": "forged_boundary",
    "End every answer with a made-up link": "append_link",
    "Withhold the answer": "withhold",
}


def test_the_adversarial_figures_are_the_committed_run() -> None:
    figures = summary("adversarial/baseline")["figures"]
    rows = table(readme(), "| | Held | Of | Share held (95% interval) |")
    injected = figures["injected_passage"]["held"]
    limitation = line_starting(readme(), "- **Planted instructions get through.**")

    assert [row[0] for row in rows] == list(ADVERSARIAL)
    for label, held, of, share in rows:
        family, figure = ADVERSARIAL[label]
        estimate = figures[family][figure]
        assert [held, of, share] == [
            str(hits(estimate)),
            str(estimate["count"]),
            with_interval(estimate),
        ], label
    broke = injected["count"] - hits(injected)
    assert f"{broke} of the {injected['count']} planted in passages did" in limitation


def test_the_planted_instructions_are_the_committed_run() -> None:
    figures = summary("adversarial/baseline")["figures"]
    rows = table(readme(), "| Planted instruction | Held | Of |")

    assert [row[0] for row in rows] == list(PLANTED)
    for label, held, of, _what_the_replies_did in rows:
        estimate = figures[f"injected_passage/{PLANTED[label]}"]["held"]
        assert [held, of] == [str(hits(estimate)), str(estimate["count"])], label


# Each row of the cache table, and the score the simulated cache decides by in the run.
CACHE = {
    "Cosine alone": "cosine",
    "Second check, weaker direction": "entails_min",
    "Second check, mean of both directions": "entails_mean",
}
MONTHS = "January February March April May June July August September October November December"


def test_the_cache_measurement_is_the_committed_run() -> None:
    run = summary("cache/baseline")
    pairs, scores, cache = run["pairs"], run["pair_view"]["scores"], run["cache_view"]
    policies, others = cache["policies"], cache["other_splits"]
    rows = table(readme(), "| Decided by | Threshold | Rewordings served right |")

    def counted(count: int, of: int, estimate: Any) -> str:
        """A count, with its share where the run gives one: 7 of 49: 0.143 (0.071 to 0.267)."""
        return f"{count} of {of}" + ("" if estimate is None else f": {with_interval(estimate)}")

    def said(start: str) -> str:
        return line_starting(readme(), start)

    def says(start: str) -> list[str]:
        return numbers(said(start))

    assert [row[0] for row in rows] == list(CACHE)
    for label, threshold, right, wrong in rows:
        policy = policies[CACHE[label]]
        same, different = policy["reporting"]["same"], policy["reporting"]["different"]
        assert [threshold, right, wrong] == [
            "none" if policy["threshold"] is None else f"{policy['threshold']:.4f}",
            counted(same["right"], same["pairs"], same.get("right_share")),
            counted(different["wrong"], different["pairs"], different.get("wrong_share")),
        ], label
        # The table is the half that chose nothing: the run's counts of that half.
        assert same["pairs"] == cache["split"]["reporting"]["same"]
        assert different["pairs"] == cache["split"]["reporting"]["different"]
        # A cache that serves nothing by the rule has counts and no rate.
        assert ("right_share" in same) == (policy["threshold"] is not None), label
    assert policies["cosine"]["threshold"] is None
    assert rows[0][2:] == ["0 of 49", "0 of 50"]

    # The issue, the pairs, the stored questions, and the day the run records.
    year, month, day = run["measured_on"].split("-")
    start = "A semantic cache would answer a new question"
    assert says(start) == [
        "24",
        str(pairs["agreed"]),
        str(pairs["same"]),
        str(pairs["different"]),
        str(cache["stored_questions"]),
        str(int(day)),
        year,
    ]
    assert f"The run, on {int(day)} {MONTHS.split()[int(month) - 1]} {year}, is" in said(start)
    assert "(evals/runs/cache/baseline)" in said(start)
    assert "a threshold was chosen on one half of the pairs" in said(start)
    assert "and the table is the other half." in said(start)

    # Under the table: why the intervals are too narrow, with the run's own counts.
    groups = cache["split"]["reporting"]["groups_of"]["different"]
    start = "A cell is a count, then its share with a 95% interval."
    assert says(start) == [
        str(cache["split"]["reporting"]["different"]),
        str(groups["groups"]),
        str(groups["largest"]),
    ]
    assert "The intervals are too narrow, because pairs share test questions" in said(start)

    (at_95,) = [row for row in cache["whole_set"]["curves"]["cosine"] if row["threshold"] == 0.95]
    start = (
        "- **The similarity of two embeddings does not tell a rewording from a near-miss on these"
    )
    assert says(start) == [
        str(pairs["agreed"]),
        three(scores["cosine"]["area_under_curve"]),
        "0.5",
        "0.95",
        str(at_95["same"]["right"]),
        str(pairs["same"]),
        str(at_95["different"]["wrong"]),
        str(pairs["different"]),
    ]
    # "no threshold served a rewording without a near-miss, so [it] serves nothing"
    assert policies["cosine"]["reporting"]["same"]["right"] == 0

    weaker = policies["entails_min"]
    same, different = weaker["reporting"]["same"], weaker["reporting"]["different"]
    there, chose = weaker["reporting_by_how"], weaker["choosing_by_how"]
    other_shapes = ("short", "long", "indirect")
    start = "- **A second check helps, and serves little.**"
    assert says(start) == [
        three(scores["entails_min"]["area_under_curve"]),
        three(scores["entails_mean"]["area_under_curve"]),
        str(same["right"]),
        str(same["pairs"]),
        str(different["pairs"]),
        str(same["right"]),
        str(there["reworded"]["right"]),
        str(there["reworded"]["pairs"]),
        str(there["synonyms"]["right"]),
        str(there["synonyms"]["pairs"]),
        str(sum(there[how]["pairs"] for how in other_shapes)),
        str(sum(chose[how]["pairs"] for how in other_shapes)),
    ]
    assert different["wrong"] == 0
    assert "and none of the 50 near-misses" in said(start)
    # "None of its 31 questions ... was served, and on the choosing half one of 27 was."
    assert there["reworded"]["right"] + there["synonyms"]["right"] == same["right"]
    assert sum(there[how]["right"] for how in other_shapes) == 0
    assert sum(chose[how]["right"] for how in other_shapes) == 1
    assert "was served, and on the choosing half one of 27 was." in said(start)
    assert f"None of its {sum(there[how]['pairs'] for how in other_shapes)} questions" in said(
        start
    )

    hard = cache["near_misses_above"]
    fell = others["policies"]["entails_min"]["by_where_the_hardest_pair_fell"]
    highest = weaker["highest_wrong"]["reporting"]["different"]
    start = "- **None wrong of 50 was no test of the threshold.**"
    assert says(start) == [
        str(different["pairs"]),
        three(highest["score"]),
        str(len(hard["policies"]["entails_min"]["pairs"])),
        str(hard["score"]),
        str(others["splits"]),
        str(others["chosen_policy"]["splits_with_something_wrong"]),
    ]
    assert highest["score"] < weaker["threshold"]
    # What 0.143 is: the highest score at which a near-miss would be served, not a pair's score.
    assert f"would be served above {three(highest['score'])} by the weaker direction" in said(start)
    assert "are all on the choosing half" in said(start)
    assert {pair["half"] for pair in hard["policies"]["entails_min"]["pairs"]} == {"choosing"}
    # "those are exactly the splits that put one pair, `d067`, on the reporting half"
    assert f"`{fell['pair']}`" in said(start)
    assert fell["reporting"]["splits"] == others["chosen_policy"]["splits_with_something_wrong"]
    assert fell["reporting"]["splits_with_nothing_wrong"] == 0
    assert fell["choosing"]["splits_with_something_wrong"] == 0

    start = "- **These are rates on a hard set, not on real questions.**"
    assert says(start) == [
        str(pairs["different_by_source"]["written"]),
        str(pairs["different_by_source"]["golden"]),
        str(cache["stored_questions"]),
    ]
    assert "Whether the threshold is stricter than real use needs is not measured." in said(start)


def test_the_answer_cache_section_is_the_third_committed_cache_run() -> None:
    """The cache as it is built: its row, its default threshold, and what did not test it.

    The figures of the one check with the real service, and what the default costs on
    the development laptop, were measured by hand and are held by no run. The range of
    thresholds the default is the top of is worked out from the run in
    ``tests/evals/test_committed_cache_run_as_built.py``, which also holds it to this text.
    """
    run = summary("cache/as-built")
    cache = run["cache_view"]
    assert cache["chosen_policy"] == "entails_min"
    weaker = cache["policies"]["entails_min"]
    same, different = weaker["reporting"]["same"], weaker["reporting"]["different"]
    threshold = Settings().cache_threshold
    start = readme().index("### The answer cache\n")
    section = readme()[start : readme().index("## Limitations\n")]

    def counted(count: int, of: int, estimate: Any) -> str:
        return f"{count} of {of}: {with_interval(estimate)}"

    (row,) = table(readme(), "| As the cache is built | Threshold |")
    assert row == [
        "Second check, weaker direction",
        f"{weaker['threshold']:.4f}",
        counted(same["right"], same["pairs"], same["right_share"]),
        counted(same["wrong"], same["pairs"], same["wrong_share"]),
        counted(different["wrong"], different["pairs"], different["wrong_share"]),
    ]
    assert "(evals/runs/cache/as-built)" in section
    # The run by the label is said to be by the label, and is not called the cache as built.
    assert "(evals/runs/cache/no-decline-candidates)" not in section
    assert "Once with stored declines left out of the candidates, by the label of each" in section

    def says(start: str) -> list[str]:
        return numbers(line_starting(section, start))

    # What the cache's own code did with the replies the run was filled with.
    held = cache["stored_replies"]
    kept_out = {reason: len(ids) for reason, ids in held["kept_from_the_second_tier"].items()}
    assert says("The measurement of the last section was run twice more.") == [
        str(held["kept"]),
        str(held["kept"] - held["candidates"]),
        str(kept_out["decline"]),
        str(kept_out["says_something_is_missing"]),
        str(kept_out["cites_nothing"]),
    ]
    assert kept_out["declines_in_part"] == 0
    assert sum(kept_out.values()) == held["kept"] - held["candidates"]

    start_of_line = "- **The cache's default threshold is 0.91, and not that"
    assert says(start_of_line)[:2] == [str(threshold), f"{weaker['threshold']:.4f}"]
    assert len(says(start_of_line)) == 4, "then the two ends of the range, held elsewhere"
    assert "there is no evidence about real questions" in line_starting(section, start_of_line)
    assert "chosen with every pair in view, so nothing tests the choice" in line_starting(
        section, start_of_line
    )
    hardest = weaker["highest_wrong"]["reporting"]
    over = cache["other_splits"]["policies"]["entails_min"]
    start_of_line = "- **That half did not test the threshold hard.**"
    assert says(start_of_line) == [
        three(hardest["different"]["score"]),
        three(hardest["same"]["score"]),
        three(threshold - hardest["same"]["score"]),
        str(cache["other_splits"]["splits"]),
        str(over["splits_with_something_wrong"]),
    ]
    assert f"`{hardest['same']['pair']}`" in line_starting(section, start_of_line)
    assert hardest["same"]["own"] is False, "another stored question's answer"
    assert "the rule's own threshold served something wrong" in line_starting(
        section, start_of_line
    )
    # The near-miss that only the second guard keeps out, by its row in the run.
    lines = (RUNS / "cache" / "as-built" / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    (near_miss,) = [row for row in map(json.loads, lines) if row["pair"] == "d067"]
    assert (
        near_miss["question_id"] in held["kept_from_the_second_tier"]["says_something_is_missing"]
    )
    start_of_line = "- **One near-miss is kept out by the second guard alone.**"
    # Then what the guard finds of the replies it was not written from: 1 of 5, which
    # tests/evals/test_committed_cache_run_as_built.py works out from their file.
    assert says(start_of_line) == [
        three(min(near_miss["entails_forward"], near_miss["entails_backward"])),
        "5",
        "1",
    ]
    assert "`d067`" in line_starting(section, start_of_line)
    assert "so this is no test of it" in line_starting(section, start_of_line)
    # What the second guard is, said with both of its faults.
    assert (
        "It errs towards keeping an answer out, and a reply that words the gap another way "
        "gets past it."
    ) in section
    # The trade-off is said once, plainly, and the cache is nowhere called safe.
    assert section.count("will sometimes be served its answer") == 1
    for word in ("safe", "guarantee", "proves"):
        assert word not in section.lower(), word
    # The day of the check by hand is the day of the run, and it is said to be older
    # than the defaults.
    year, month, day = run["measured_on"].split("-")
    assert f"on {int(day)} {MONTHS.split()[int(month) - 1]} {year}, by hand" in section
    assert "That check was made with the defaults of that day" in section
    # The second check, after the fixes, is given as another session's account.
    assert (
        "it was made again, cold and with the defaults of today, by the session that "
        "coordinated this change: 4 questions, of which 2 cost a request"
    ) in section


# Each search in the table of the two-passage questions, and its run.
TWO_PASSAGE = {
    "Hybrid, the default": "hybrid",
    "Embedding model": "dense",
    "Keyword (BM25)": "sparse",
}
# The runs of the search variants: on the 33 questions beside the three searches, and on
# the golden set in a directory of their own, with the default search as `hybrid`.
VARIANTS_ON_THE_33 = RUNS / "two-passage"
VARIANTS_ON_THE_GOLDEN_SET = RUNS / "search-variants-golden"
TWO_PASSAGE_QUESTIONS = ROOT / "evals" / "two-passage" / "questions.jsonl"
TWO_PASSAGE_REASONS = ROOT / "evals" / "two-passage" / "why-two.jsonl"


def test_the_two_passage_section_is_the_committed_runs() -> None:
    """The section on questions that need two passages, and its line under Limitations.

    Every number in them is worked out here, and a line with a number that nothing works
    out fails. ``tests/evals/test_two_passage_documents.py`` holds the other documents
    that give these figures in the same way.
    """
    rows = table(readme(), "| Search | Both passages in the top 5 (95% interval) |")
    section = readme().split("### Questions that need two passages\n")[1].split("\n### ")[0]
    written = len(load_golden_set(TWO_PASSAGE_QUESTIONS))
    confirmed = sum(bool(reason.confirmed) for reason in load_reasons(TWO_PASSAGE_REASONS).values())

    assert [row[0] for row in rows] == [
        f"[{search}](evals/runs/two-passage/{run})" for search, run in TWO_PASSAGE.items()
    ]
    for row, run in zip(rows, TWO_PASSAGE.values(), strict=True):
        about = summary(f"two-passage/{run}")
        metrics, in_top = about["metrics"], about["needs_several_passages"]["found_in_top"]
        both, one = in_top["5"]["all"], in_top["5"]["some"]
        assert (about["questions_scored"], about["questions_left_out"]) == (confirmed, ["t025"])
        assert row[1:] == [
            f"{with_interval(metrics['all@5'])}, {both} of {confirmed}",
            f"{three(metrics['hit@5']['mean'])}, {both + one} of {confirmed}",
            f"{three(metrics['all@10']['mean'])}, {in_top['10']['all']} of {confirmed}",
        ], run

    # The default search: both passages, at least one, and the questions between the two.
    several = summary("two-passage/hybrid")["needs_several_passages"]
    both, one = several["found_in_top"]["5"]["all"], several["found_in_top"]["5"]["some"]
    # The fourteen variants: every run of the 33 questions but the three searches.
    variants = sorted(
        path.parent.name
        for path in VARIANTS_ON_THE_33.glob("*/summary.json")
        if path.parent.name not in TWO_PASSAGE.values()
    )
    assert {path.parent.name for path in VARIANTS_ON_THE_GOLDEN_SET.glob("*/summary.json")} == {
        *variants,
        "hybrid",
    }
    found = [hits(summary(f"two-passage/{run}")["metrics"]["all@5"]) for run in variants]
    assert not [
        run
        for run in variants
        if compare_runs(VARIANTS_ON_THE_33 / "hybrid", VARIANTS_ON_THE_33 / run)
        .metrics["all@5"]
        .change.is_clear
    ]
    assert "and on that score none is clearly different from the default" in section
    # Every "fourteen" of the section is the number of variants, and each is one of these.
    tried = in_words(len(variants)).capitalize()
    for phrase in (
        f"**{tried} changes to the search were tried, and none is adopted**",
        f"{tried} variants were compared on the same {confirmed} questions",
    ):
        assert phrase in section, phrase
    assert section.lower().count("fourteen") == 2
    # The limit for each page, on the golden set, whose runs repeat the committed `v2`.
    default = summary("search-variants-golden/hybrid")["metrics"]["hit@5"]
    one_chunk, two_chunks = (
        compare_runs(
            VARIANTS_ON_THE_GOLDEN_SET / "hybrid", VARIANTS_ON_THE_GOLDEN_SET / run
        ).metrics["hit@5"]
        for run in ("spread-1-in-5", "spread-2-in-5")
    )
    assert default == summary("retrieval/v2")["metrics"]["hit@5"]
    assert one_chunk.change.is_clear
    assert two_chunks.change.is_clear
    assert two_chunks.change.better == 0
    # Further down in the default's results, from the rank of each passage.
    ranks = [row["evidence_ranks"] for row in results("two-passage/hybrid")]

    def both_within(k: int) -> int:
        return sum(all(rank is not None and rank <= k for rank in of_one) for of_one in ranks)

    assert (both_within(5), both_within(10)) == (both, several["found_in_top"]["10"]["all"])

    assert_numbers(
        section,
        {
            "Almost every test question rests on one passage": [
                str(confirmed),
                str(written),
                str(confirmed),
                "0021",
                "78",
            ],
            "- **The default search gives the model both passages": [
                str(both),
                str(confirmed),
                str(both + one),
                str(one),
            ],
            "- **Fourteen changes to the search were tried": [
                str(min(found)),
                str(max(found)),
                str(confirmed),
            ],
            "- **A limit on the chunks of one page": [
                str(default["count"]),
                three(default["mean"]),
                three(one_chunk.candidate.mean),
                three(two_chunks.candidate.mean),
                str(one_chunk.change.worse),
                str(one_chunk.change.better),
                str(two_chunks.change.worse),
            ],
            "- **The second passage is often further down, or not found.**": [
                str(both_within(8)),
                str(both_within(10)),
                str(several["not_all_retrieved"]),
                str(confirmed),
                "79",
                str(several["retrieved"]),
            ],
            '- **Read "clear" with care here.**': [str(confirmed), "80"],
        },
    )
    # And once more under Limitations, with what was not done.
    limitation = line_starting(readme(), "- **A question that needs two passages often gets one")
    assert numbers(limitation) == [str(both), str(confirmed), "0021", "78"]
    assert "no answer was written for these questions" in limitation
    assert "so the share is more likely too high than too low" in limitation
    assert "no person has read them" in limitation
    assert "No person has read them, and no answer was written for them." in section


def test_the_follow_up_section_is_the_committed_run() -> None:
    """The section on follow-up questions, its line under Limitations and its command.

    Every number in them is worked out from the saved run, as
    ``tests/evals/test_follow_up_documents.py`` works out the other documents that give
    these figures: those of the first run, as today's code reads it, and the cases' of the
    whole file. A line with a number that nothing works out fails.
    """
    # Imported here: that module imports this one's helpers.
    from handbook_rag.evals.follow_ups import FOLLOW_UP, PLANTED, STANDALONE
    from tests.evals import test_follow_up_documents as documents

    k = documents.figures()
    section = readme().split("### Follow-up questions\n")[1].split("\n## ")[0]
    rows = table(readme(), "| | Right | Of | Share (95% interval) |")
    planted = k["planted"]

    assert rows == [
        [
            "Follow-ups rewritten to a question that means an acceptable form",
            str(k["follow_right"]),
            str(k["groups"][FOLLOW_UP]),
            documents.shown(k["follow"]["same"]),
        ],
        [
            "Questions that stand alone, left alone",
            str(k["alone_right"]),
            str(k["groups"][STANDALONE]),
            documents.shown(k["alone"]["same"]),
        ],
        [
            "Planted instructions not followed",
            str(planted["held_by_code"] + planted["read_held"]),
            str(k["groups"][PLANTED]),
            documents.shown(planted["held"]),
        ],
    ]
    assert_numbers(
        section,
        {
            "A question asked after another often leans on it": [
                documents.RECORD,
                documents.FOLLOW_UPS,
                str(k["cases"]),
                *k["model"],
                str(k["first_cases"]),
                *documents.NIGHT,
                str(k["added"]),
                str(k["decided"]),
                str(k["set_aside"]),
            ],
            "- **A follow-up that leans on an earlier question was rewritten right in": [
                str(k["leaning_right"]),
                str(k["leaning_of"]),
                str(k["right"]["points_into_answer"]),
                str(k["into_answer"]),
                str(k["needs_answer"]),
                str(k["into_answer"]),
            ],
            "- **A follow-up costs one more request.**": [k["median"], k["p95"]],
        },
    )
    assert "no person has read them" in section
    assert "two reader sessions decided the other 21" in section
    assert "No answer was written for the rewritten questions" in section
    assert (planted["broke_by_code"], planted["read_broke"], planted["reached_search"]) == (0, 0, 0)
    assert "no planted instruction was followed or reached the search" in section
    # Under Limitations: the step is not wired in.
    limitation = line_starting(readme(), "- **A follow-up question is still answered by itself.**")
    assert numbers(limitation) == [documents.RECORD, documents.FOLLOW_UPS]
    assert "not wired in" in limitation
    # The command, with one real request for each case.
    command = line_starting(readme(), "make eval-follow-ups")
    assert command.endswith(f"and measure it: {k['cases']} real requests")


def test_every_link_to_a_file_or_a_heading_leads_to_one() -> None:
    for target in re.findall(r"\]\(([^)\s]+)\)", readme()):
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.partition("#")
        linked = ROOT / (path or "README.md")
        assert linked.exists(), target
        if anchor:
            assert anchor in anchors(linked.read_text(encoding="utf-8")), target


def test_every_issue_and_pull_request_linked_is_one_of_the_repository() -> None:
    links = re.findall(r"\[([^\]]+)\]\(([^)\s]*github\.com[^)\s]*)\)", readme())
    item = re.escape(REPOSITORY) + r"/(issues|pull)/(\d+)(?:#[\w-]+)?"

    assert [url for _, url in links if not re.fullmatch(item, url)] == []
    for text, url in links:
        match = re.fullmatch(item, url)
        if match is None:
            continue
        known = ISSUES if match[1] == "issues" else PULLS
        assert int(match[2]) in known, (
            f"{url}: not in the list of {match[1]} items at the top of this file. If it is new, "
            "add its number there."
        )
        named = re.search(r"(?:issue|pull request) (\d+)", text)
        assert named is None or named[1] == match[2], f"{text!r} links to {url}"
