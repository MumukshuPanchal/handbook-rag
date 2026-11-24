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

from handbook_rag.evals.compare import compare_runs

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
