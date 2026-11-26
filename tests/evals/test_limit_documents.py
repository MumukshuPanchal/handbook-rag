"""What the documents say of the limit answers and of version 2 of the judge is what the runs hold.

The labelled answers have a test of their own (``test_committed_limit_cases.py``). These
tests hold the documents written from the runs of the two versions of the support judge:
their checks on the limit answers and on the first labelled answers, and their verdicts on
the answers of the `baseline` run. The documents are the decision record (ADR 0023), the
section of the evaluation guide, what the two cards of labelled answers say of the checks,
the lines of the changelog and the roadmap, the architecture page, and the make target
that judges the `baseline` answers with version 2. The README's section is held by
``tests/test_readme_results.py``, with the README's other results.

Every figure is worked out once, in ``tests/evals/limit_runs.py``, from the saved runs and
the files of labelled answers. A table is built again from the runs. Every line of prose
that holds a number is compared with the numbers it must hold, in their order, and a line
with a number that no test works out fails (``test_readme_results.assert_numbers``). The
prose names the two versions "version 1" and "version 2", so those are numbers of a line
too. A few counts that the documents write as words are held as well.

What these tests cannot hold is whether a statement truly leaves out a limit. Where version
2 and the labels differ, the documents give both and say that it is the labels' word
against the judge's.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.config import Settings
from handbook_rag.evals.judge import SUPPORT_INSTRUCTIONS
from handbook_rag.evals.judge_v2 import HOW_TO_CHECK
from handbook_rag.evals.judge_v2 import RUBRIC as RUBRIC_TEXT
from handbook_rag.evals.metrics import Estimate
from handbook_rag.evals.support import SUPPORT, SUPPORT_V2
from handbook_rag.wiring import describe_judge
from tests.evals import hardening_runs, limit_runs
from tests.evals.test_committed_limit_cases import COUNTED_FAULTS, LEFT_OUT, LIMIT_KINDS
from tests.test_readme_results import (
    anchors,
    assert_numbers,
    baseline_line_numbers,
    in_words,
    line_starting,
    numbers,
    table,
    three,
    with_interval,
)

ROOT = Path(__file__).parents[2]
ADR = ROOT / "docs" / "adr" / "0023-a-second-version-of-the-support-judge-for-dropped-limits.md"
GUIDE = ROOT / "docs" / "evaluation.md"
LIMITS_CARD = ROOT / "evals" / "judge" / "limits" / "README.md"
FIRST_CARD = ROOT / "evals" / "judge" / "README.md"
README = ROOT / "README.md"
CHANGELOG = ROOT / "CHANGELOG.md"
ROADMAP = ROOT / "docs" / "roadmap.md"
ARCHITECTURE = ROOT / "docs" / "architecture.md"
MAKEFILE = ROOT / "Makefile"
# Numbers that are names: the decision record and another one it cites, and the issues of
# dropped limits, of the rules version 1 lacks, of the hardening against planted
# instructions and of the comparison of model tiers.
RECORD, LAYERS_RECORD = "0023", "0014"
LIMITS, RUBRIC, HARDENING, TIERS = "59", "58", "66", "17"
# The versions of the judge, as the prose names them.
FIRST, SECOND = "1", "2"
# The day of the decision record.
DATE = ["2026", "11", "25"]
# The allowance of the free key, and what measuring the answers again takes, as the README
# gives them.
A_DAY, MEASURING_AGAIN = "500", "460"
# The figures of agreement that a table of a check gives, beside kappa.
AGREEMENT_FIGURES = ("recall", "precision", "same_decision", "same_answer")
# The step of version 2's check that is the card's test of a limit, and the step of the
# card's making that states that test.
TEST_STEP, CARD_STEP = "3", "2"
# The test, in the words of each.
TEST_IN_THE_CHECK = (
    "name a person, a place, a time, an amount or a case that a reader would take the answer "
    "to cover, and that the passage does not cover"
)
TEST_IN_THE_CARD = (
    "there had to be a case that the rule without its limit covers and the passage does not"
)
# What version 2's check says may be left out, as the card's kinds of harmless omission.
MAY_BE_LEFT_OUT = {
    "how something is done": "detail",
    "why it is done": "reason",
    "an example": "example",
    "another rule of the passage": "other_rule",
}
# The words of the passage of `q104` on which version 2's verdict turns, as its reason quotes
# them.
MEETING = "of the 1:1 meeting"
# What version 2 is told to look for, in the order of its instructions, and the kind of
# limit of the labelled answers that each is.
LOOKED_FOR = {
    "who the rule is for": "who",
    "where it holds": "where",
    "when it holds": "when",
    "how much": "how_much",
    "what it covers": "what",
    "on what condition": "only_if",
    "with what exception": "unless",
}


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def section(text: str, heading: str, level: str = "## ") -> str:
    """The part of a document under a heading, up to the next heading of its level."""
    after = text.split(f"\n{level}{heading}\n")
    assert len(after) == 2, heading
    return after[1].split(f"\n{level}")[0]


def guide_section() -> str:
    return section(read(GUIDE), "Limits that an answer leaves out")


def wrote(count: int) -> str:
    """A count as the tables write it: 1,159."""
    return f"{count:,}"


def of(part: int, whole: int) -> str:
    return f"{part} of {whole}"


def share(estimate: dict[str, Any]) -> str:
    """A share as the tables write it: 33 of 39, 0.846 (0.703 to 0.928)."""
    return f"{limit_runs.hits(estimate)} of {estimate['count']}, {with_interval(estimate)}"


def first_set_line_numbers() -> list[str]:
    """The numbers of the line that gives version 2's check on the first labelled answers.

    The line, in the guide and the decision record:
    "- **Version 2 on the first labelled answers.** It caught {caught} of the {faults}
    faults: {mean} ({low} to {high}). It raised {false alarms} false alarms on the
    {statements} statements labelled supported or without a claim, where version 1 raised
    {false alarms of version 1}." With a link to the run, whose target holds no number
    that counts.
    """
    second, first = limit_runs.first_set_check(2), limit_runs.first_set_check(1)
    recall = second["recall"]
    return [
        SECOND,
        str(second["caught"]),
        str(second["faults"]),
        three(recall["mean"]),
        three(recall["low"]),
        three(recall["high"]),
        str(second["false_alarms"]),
        str(second["not_faults"]),
        FIRST,
        str(first["false_alarms"]),
    ]


def lines_of_the_two_runs() -> dict[str, list[str]]:
    """The lines that give version 2's check on the first labelled answers, and its score
    of the `baseline` answers, with their numbers."""
    return {
        limit_runs.FIRST_SET_LINE: first_set_line_numbers(),
        limit_runs.BASELINE_LINE: baseline_line_numbers(),
    }


def on_the_first_set() -> tuple[dict[str, Any], dict[str, Any]]:
    return limit_runs.first_set_check(1), limit_runs.first_set_check(2)


def disputed() -> list[tuple[str, int]]:
    """The statements of the `baseline` answers that version 2 faults and version 1 does
    not, as the documents list them: those both readers called supported, then the one they
    split on."""
    return sorted(
        baseline_counts()["only"],
        key=lambda key: (limit_runs.first_set_statement(key)["expected"] == "unclear", key),
    )


def baseline_counts() -> dict[str, Any]:
    """Version 2's four faulted statements of the `baseline` answers, sorted as the
    documents sort them: the one version 1 faulted too, and the others by their label."""
    faulted = limit_runs.baseline_faulted(2)
    both = [key for key in faulted if key in limit_runs.baseline_faulted(1)]
    only = [key for key in faulted if key not in both]
    labels = Counter(limit_runs.first_set_statement(key)["expected"] for key in only)
    return {"faulted": faulted, "both": both, "only": only, "labels": labels}


def the_set() -> dict[str, Any]:
    """What the documents say of the labelled answers themselves."""
    labelled = limit_runs.cases()
    statements = [statement for case in labelled.values() for statement in case["statements"]]
    return {
        "answers": len(labelled),
        "passages": len({case["passages"][0]["chunk_id"] for case in labelled.values()}),
        "statements": len(statements),
        "same": sum(s["readers"]["first"] == s["readers"]["second"] for s in statements),
        "kinds": Counter(case["kind"] for case in labelled.values()),
        "readers": len({case["second_session"] for case in labelled.values()}),
    }


# The runs


@pytest.mark.parametrize(("version", "criterion"), [(1, SUPPORT), (2, SUPPORT_V2)])
def test_each_check_on_the_limit_answers_is_whole_and_of_a_judge_the_code_builds(
    version: int, criterion: Any
) -> None:
    run = limit_runs.check(version)
    about = run["summary"]
    saved = json.loads((limit_runs.RUNS / run["run"] / "run.json").read_text(encoding="utf-8"))

    assert about["answers_judged"] == about["answers_labelled"] == len(limit_runs.cases())
    assert about["criterion"] == "support"
    assert about["config"] == saved["config"] == describe_judge(Settings(), criterion)
    assert about["attempts"] == saved["attempts"]
    assert about["left_out_as_unclear"] == 1


@pytest.mark.parametrize("version", [1, 2])
def test_what_a_check_saved_is_what_its_verdicts_hold(version: int) -> None:
    run = limit_runs.check(version)
    recall, precision = run["recall"], run["precision"]
    labelled = limit_runs.cases()

    assert limit_runs.hits(recall) == len(run["caught"])
    assert recall["count"] == len(run["faults"])
    assert precision["count"] == len(run["caught"]) + len(run["false_alarms"])
    assert limit_runs.hits(precision) == len(run["caught"])
    assert run["summary"]["agreement"]["statements"] == len(run["faults"]) + len(run["supported"])
    # The faults caught by kind, as the planted kind of each statement counts them.
    planted = {
        (case["case"], s["number"]): s["planted"]
        for case in labelled.values()
        for s in case["statements"]
        if "planted" in s
    }
    assert set(planted) == run["faults"]
    assert run["by_kind"] == {
        kind: (sum(planted[key] == kind for key in run["caught"]), total)
        for kind, total in COUNTED_FAULTS.items()
    }


def test_the_misses_in_which_version_2_found_the_limit_are_those_the_documents_name() -> None:
    second = limit_runs.check(2)
    labelled = limit_runs.cases()
    missed = dict(second["missed"])

    for case, words in limit_runs.FOUND_THE_LIMIT.items():
        assert words in labelled[case]["limit"], case
        assert words in second["reasons"][(case, missed[case])], case
    for case, number in second["missed"]:
        if case not in limit_runs.FOUND_THE_LIMIT:
            limit = labelled[case]["limit"].rstrip(".").casefold()
            assert limit not in second["reasons"][(case, number)].casefold(), case


def test_the_score_of_the_baseline_answers_is_worked_out_as_the_answer_run_works_it_out() -> None:
    # The way the tests work out version 2's score of the `baseline` answers, once its
    # verdicts are there, gives version 1's score exactly as the run's summary holds it.
    saved = limit_runs.summary("answers/baseline")["metrics"]["supported"]

    assert limit_runs.baseline_score(1) == Estimate(**saved)


def test_version_2_is_told_to_look_for_the_seven_kinds_of_limit_of_the_set() -> None:
    # What the decision record and the guide say: the instructions of version 2 list the
    # same seven kinds of limit that the card sorts the set by.
    listed = re.findall(r"^   - (.+?)(?::|$)", HOW_TO_CHECK, flags=re.MULTILINE)

    assert [item for item in listed if item in LOOKED_FOR] == list(LOOKED_FOR)
    assert len(listed) == len(LOOKED_FOR)
    assert sorted(LOOKED_FOR.values()) == sorted(LIMIT_KINDS)
    for text in (read(ADR), guide_section()):
        assert f"than its {in_words(len(LIMIT_KINDS))} kinds of limit" in text
    for text in (read(README), read(LIMITS_CARD)):
        assert f"its {in_words(len(LIMIT_KINDS))} kinds of limit" in text


def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$']+", text.casefold())


def phrases_of_four(text: str) -> set[tuple[str, ...]]:
    found = words_of(text)
    return {tuple(found[i : i + 4]) for i in range(len(found) - 3)}


def test_what_the_check_of_version_2_shares_with_the_card_of_the_set() -> None:
    card = read(LIMITS_CARD)
    steps = dict(re.findall(r"^(\d)\. \*\*(.+?)\*\*", HOW_TO_CHECK, flags=re.MULTILINE))
    card_steps = dict(re.findall(r"^(\d)\. \*\*(.+?)\*\*", card, flags=re.MULTILINE))
    in_the_check = " ".join(HOW_TO_CHECK.split())

    # Its step of deciding is the card's own test for building the set.
    assert steps[TEST_STEP] == "Decide."
    assert TEST_IN_THE_CHECK in in_the_check
    assert card_steps[CARD_STEP] == "Making sure a limit is a limit."
    assert TEST_IN_THE_CARD in card
    # Its list of what may be left out is the card's four kinds of harmless omission.
    for words in MAY_BE_LEFT_OUT:
        assert words in in_the_check, words
    assert sorted(MAY_BE_LEFT_OUT.values()) == sorted(LEFT_OUT)
    # And no phrase of four words of the check is in a question or a statement of the set, or
    # in the words of its passages that the file quotes. The passages themselves are compared
    # by a slow test in test_committed_limit_cases.py, which needs the handbook.
    of_the_check = phrases_of_four(HOW_TO_CHECK)
    for case in limit_runs.cases().values():
        texts = [case["question"], case["limit"], case.get("left_out", "")]
        texts += [statement["text"] for statement in case["statements"]]
        for text in texts:
            assert not phrases_of_four(text) & of_the_check, (case["case"], text)
    # The documents say so, in the decision record, the guide, the card and the README.
    assert f"step {TEST_STEP}, which asks for a person, a place, a time, an amount or a case" in (
        read(ADR)
    )
    assert f"step {TEST_STEP}, which asks for a person" in guide_section()
    assert f"its test of a limit (step {CARD_STEP} of how it was made)" in card
    assert "its test of a limit and its kinds of harmless omission" in read(README)
    for text in (read(ADR), guide_section()):
        assert "the card's four kinds of harmless omission" in text
        assert "No phrase of four or more words of their check for a limit left out" in text


def test_version_2s_instructions_are_almost_three_times_as_long_as_those_of_version_1() -> None:
    longer = (len(RUBRIC_TEXT) + len(HOW_TO_CHECK)) / len(SUPPORT_INSTRUCTIONS)

    assert 2.5 < longer < 3
    for text in (read(README), read(ADR), guide_section()):
        assert "because its instructions are almost three times as long" in text
        assert "because the rubric is in its instructions" not in text


def test_the_alarms_on_real_answers_count_as_left_out_a_limit_that_the_question_names() -> None:
    on_second = limit_runs.first_set_check(2)
    first_set = limit_runs.first_set()
    text = guide_section()
    in_the_check = " ".join(HOW_TO_CHECK.split())

    # The instructions keep a limit that the question asks about, or that another statement
    # gives, and fault a limit once.
    assert "if the question asks about that very group, place, period or case" in in_the_check
    assert "or if another statement of the same answer gives it for this rule" in in_the_check
    assert "Fault a limit once, in the statement that gives the rule it belongs to." in (
        in_the_check
    )
    faulted = {
        case: number
        for case, number in on_second["verdicts"]
        if on_second["verdicts"][(case, number)] == "unsupported"
        and limit_runs.first_set_statement((case, number))["expected"] in ("supported", "unclear")
        and case in limit_runs.QUESTION_NAMES_THE_LIMIT
    }
    assert sorted(faulted) == sorted(limit_runs.QUESTION_NAMES_THE_LIMIT)
    for case, words in limit_runs.QUESTION_NAMES_THE_LIMIT.items():
        named = limit_runs.named_limits(on_second["reasons"][(case, faulted[case])])
        assert words.casefold() in first_set[case]["question"].casefold(), case
        assert words.casefold() in named.casefold(), case
    # The two real ones are labelled supported by both readers; in `q062` the reason says the
    # statement gives the rule for everyone; in `q131` statement 1 gives the limit too.
    for case in ("q062", "q131"):
        readers = limit_runs.first_set_statement((case, faulted[case]))["readers"]
        assert first_set[case]["kind"] == "real"
        assert readers["first"] == readers["second"] == "supported"
    assert "asserts the rule for everyone" in on_second["reasons"][("q062", faulted["q062"])]
    assert f'"{first_set["q062"]["question"]}"' in text
    first_of_q131 = limit_runs.first_set_statement(("q131", 1))
    assert "production environment" in first_of_q131["text"]
    assert on_second["verdicts"][("q131", 1)] == "supported"
    assert faulted["q131"] != 1
    assert 'In `q131` the question names "GitLab\'s production environment" and statement 1' in text
    assert "GitLab's production environment" in first_set["q131"]["question"]
    # Said once, where the false alarm on the limit answers is described.
    assert (
        "The same mistake, a limit that the question itself names counted as left out, is behind "
        "its false alarms on the real answers `q062` and `q131` below, and on the copy `b09-v1`."
    ) in text
    # `q104`: its reason turns on the passage's words, and the copy is passed.
    assert f"calendar invite {MEETING}" in on_second["reasons"][("q104", 8)]
    assert f'the link in the calendar invite "{MEETING}"' in text
    assert "This is what the files say, read in a review by a model session" in text


def test_version_2_decides_the_same_statement_two_ways_in_two_pairs() -> None:
    on_second = limit_runs.first_set_check(2)
    first_set = limit_runs.first_set()
    real = limit_runs.real_unclear_in_the_first_set()
    pairs = []
    for copy, answer in first_set.items():
        original = first_set.get(copy.removesuffix("-v1"))
        if original is None or original is answer:
            continue
        assert original["question"] == answer["question"], copy
        assert original["passages"] == answer["passages"], copy
        for ours, theirs in zip(original["statements"], answer["statements"], strict=True):
            same = (ours["text"], ours["cites"]) == (theirs["text"], theirs["cites"])
            key, copied = (original["case"], ours["number"]), (copy, theirs["number"])
            if same and on_second["verdicts"][key] != on_second["verdicts"][copied]:
                pairs.append((original["case"], copy, ours["number"]))
    split_on = [
        (case, s["number"])
        for case in real
        for s in first_set[case]["statements"]
        if s["expected"] == "unclear"
    ]
    faulted = [key for key in split_on if on_second["verdicts"][key] == "unsupported"]

    assert sorted(pairs) == [("b09", "b09-v1", 2), ("q104", "q104-v1", 8)]
    assert (len(faulted), len(split_on)) == (1, 3)
    for text in (read(ADR), read(LIMITS_CARD)):
        assert "it decides the same statement two ways in two pairs" in text
        assert "`b09` and `b09-v1`, `q104` and `q104-v1`" in text
        assert "It runs at the model's default sampling, and each check was run once." in text
    assert (
        f"Of the {in_words(len(split_on))} real answers there on which the readers split, it "
        f"faults {in_words(len(faulted))}."
    ) in read(LIMITS_CARD)


# The decision record


def versions_rows(runs: str) -> list[list[str]]:
    """The rows of the table of the two versions in the decision record."""
    first, second = limit_runs.check(1), limit_runs.check(2)
    return [
        [
            "Limits left out that it caught (recall)",
            *(
                f"{of(len(k['caught']), len(k['faults']))}, {with_interval(k['recall'])}"
                for k in (first, second)
            ),
        ],
        [
            "False alarms",
            *(of(len(k["false_alarms"]), len(k["supported"])) for k in (first, second)),
        ],
        [
            "Agreement beyond chance (Cohen's kappa)",
            *(f"{k['kappa']:.3f}" for k in (first, second)),
        ],
        [
            "Input tokens for each answer, on average",
            *(wrote(round(k["input_per_answer"])) for k in (first, second)),
        ],
        ["Median time of a verdict", *(f"{k['median_seconds']:.1f} s" for k in (first, second))],
        ["Run", *(f"[`{k['run']}`]({runs}/{k['run']})" for k in (first, second))],
    ]


def worse_on_every_figure_of_agreement(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether the second check is below the first on every figure of agreement it gives."""
    return (
        all(second[name]["mean"] < first[name]["mean"] for name in AGREEMENT_FIGURES)
        and second["kappa"] < first["kappa"]
    )


def decision_record_numbers() -> dict[str, list[str]]:
    """The numbers of every line of the decision record that holds one, in their order."""
    first, second = limit_runs.check(1), limit_runs.check(2)
    on_first, on_second = on_the_first_set()
    known = the_set()
    recall = first["recall"]
    return {
        "- Date:": DATE,
        "**The fault.**": [LIMITS],
        "**A labelled set for this fault alone**": [
            str(known["answers"]),
            str(known["passages"]),
            str(known["same"]),
            str(known["statements"]),
            str(len(first["faults"])),
            str(len(first["supported"])),
        ],
        "**The judge on it.**": [
            str(len(first["caught"])),
            str(len(first["faults"])),
            three(recall["mean"]),
            three(recall["low"]),
            three(recall["high"]),
            str(len(first["in_title"])),
        ],
        "- **A second version of the support judge, beside the first**": [SECOND, FIRST, RUBRIC],
        '- **Version 1 stays the judge of "supported by its sources", unchanged.**': [FIRST],
        "- **Version 2 is kept as a second instrument for one fault.**": [SECOND, FIRST, FIRST],
        "**How the decision came about.**": [
            FIRST,
            SECOND,
            SECOND,
            FIRST,
            str(len({case for case, _ in limit_runs.baseline_verdicts(2)})),
            FIRST,
        ],
        "- **Each version keeps its verdicts on an answer run in files of its own.**": [
            FIRST,
            SECOND,
            FIRST,
            FIRST,
        ],
        "- **Version 2 caught every limit that version 1 caught, and": [
            SECOND,
            FIRST,
            str(len(second["caught"] - first["caught"])),
            str(len(second["in_title"])),
        ],
        "- **Change the instructions of version 1.**": [FIRST],
        "- **Make version 2 the default, now or at the next measurement": [
            SECOND,
            SECOND,
            FIRST,
            FIRST,
        ],
        "- **A second request for each answer, for limits alone.**": [A_DAY],
        "- **Change the answering instructions first.**": [MEASURING_AGAIN, A_DAY],
        "- **A judge of another model family.**": [LAYERS_RECORD, TIERS],
        **lines_of_the_two_runs(),
        "- **There it is worse than version 1.**": [
            FIRST,
            f"{on_second['kappa']:.3f}",
            f"{on_first['kappa']:.3f}",
            str(limit_runs.hits(on_second["same_answer"])),
            str(on_second["same_answer"]["count"]),
            str(limit_runs.hits(on_first["same_answer"])),
        ],
        "- **The four statements it faults there.**": [FIRST, SECOND, FIRST],
        "What issue 59 still needs, as plans:": [LIMITS],
        "- **How often the assistant's own answers leave out a limit.**": [
            str(len({case for case, _ in limit_runs.baseline_verdicts(2)})),
            str(known["passages"]),
            SECOND,
        ],
        "- **The change to the answering instructions**": [HARDENING],
        "- **A person's reading**": [str(number) for _, number in disputed()],
        "- **A judge that catches dropped limits and faults no statement": [SECOND, FIRST],
        "- **Faults that a model makes by itself.**": [str(len(first["faults"]))],
        "- **Another author, and a person.**": [str(len(first["faults"]))],
        "- **A set that did not guide version 2.**": [
            SECOND,
            FIRST,
            TEST_STEP,
            FIRST,
            SECOND,
            SECOND,
        ],
        "- **Version 2 is not steady on the statements the readers split on.**": [SECOND],
        "- **The misses and the false alarms.**": [SECOND],
        "- Every score in the results tables and in a run's summary is as it was": [FIRST],
        "- The next measurement of the answers will be scored by version 1": [
            FIRST,
            SECOND,
            FIRST,
        ],
        "- `eval-support --judge-version 2` writes": [SECOND, SECOND, SECOND],
        "- Version 2 reads more than twice the input tokens": [SECOND, FIRST],
        "- Issue 59 stays open.": [LIMITS],
    }


def test_the_table_of_the_decision_record_is_the_runs() -> None:
    rows = table(read(ADR), "| On the limit answers | Version 1 | Version 2 |")

    assert rows == versions_rows("../../evals/runs")


def test_every_number_of_the_decision_record_is_worked_out_from_the_runs() -> None:
    assert_numbers(read(ADR), decision_record_numbers())


def test_what_the_decision_record_says_in_words_is_what_the_runs_hold() -> None:
    text = read(ADR)
    first, second = limit_runs.check(1), limit_runs.check(2)
    real = limit_runs.real_unclear_in_the_first_set()
    readme = read(README)

    assert first["caught"] <= second["caught"]
    assert not first["false_alarms"]
    assert not first["caught_in_title"]
    assert "It raised no false alarm, and it caught none of the" in text
    assert f"**It missed {in_words(len(second['missed']))}.**" in text
    assert f"In {in_words(len(limit_runs.FOUND_THE_LIMIT))} of them its reason names" in text
    assert f"**Its {in_words(len(second['false_alarms']))} false alarm**" in text
    assert f"{in_words(len(real)).capitalize()} real answers of the first answer run did" in text
    assert f"The {in_words(len(real))} real cases stay labelled `unclear`" in text
    assert all(2 < ratio < 3 for ratio in limit_runs.ratios_of_input_tokens())
    assert text.count("more than twice the input tokens") == 2
    # On the first labelled answers, and on the real answers of the `baseline` run.
    on_first, on_second = on_the_first_set()
    counts = baseline_counts()
    assert worse_on_every_figure_of_agreement(on_first, on_second)
    assert "It missed a planted citation of the wrong passage" in text
    assert "both its false alarms fault a real answer for leaving out a limit" in text
    assert f"**The {in_words(len(counts['faulted']))} statements it faults there.**" in text
    assert f"Each of the other {in_words(len(counts['only']))} it faults for a limit" in text
    assert (
        f"The labels call {in_words(counts['labels']['supported'])} of those supported and the "
        "third unclear."
    ) in text
    assert counts["labels"] == {"supported": 2, "unclear": 1}
    assert "By the labels, version 2 found no fault in the real answers that version 1 missed" in (
        text
    )
    # The allowance, and the cost of measuring the answers again as the README gives it now
    # that the adversarial cases have been asked under the new instructions (ADR 0025).
    assert f"{A_DAY} a day" in readme
    assert "about 420 requests, of the free key's 500 a day" in readme
    # What was not done is said, and so is what is a plan.
    assert "- Status: Accepted" in text
    assert "The default does not change." in text
    assert "its score is reported beside the score of version 1, never in its place" in text
    assert "A decision is allowed to follow the evidence, and this one did." in text
    assert "No figure, label or rule of scoring was changed after it was seen." in text
    assert "That was the plan at the start." in text
    # The three statements of the plans are the disputed ones, as the files have them.
    assert "three statements, all disputed" in text
    assert len(disputed()) == 3
    for case, number in disputed():
        assert f"statement {number} of `{case}`" in text
    assert "No person has read the set." in text
    assert "without the set's answers in view" in text
    # No sentence says who or what wrote the judge's instructions; who wrote and labelled
    # the data is said, as the project's documents always say it.
    assert "by a session" not in text
    assert "what else that session read" not in text
    assert "so that check is not held out from version 2 either" in text
    assert "Issue 59 stays open" in text


# The guide


def guide_numbers() -> dict[str, list[str]]:
    """The numbers of every line of the guide's section that holds one, in their order."""
    first, second = limit_runs.check(1), limit_runs.check(2)
    on_first, _ = on_the_first_set()
    real_unclear, copy_unclear = unclear_of_the_six()
    baseline = {case for case, _ in limit_runs.baseline_verdicts(2)}
    (other_fault,) = [
        key for key in limit_runs.baseline_faulted(1) if key not in limit_runs.baseline_faulted(2)
    ]
    known = the_set()
    faults = len(first["faults"])
    return {
        "An answer can give a rule from its passage": [LIMITS, RECORD],
        "[`evals/judge/limits/`]": [
            str(known["answers"]),
            str(known["passages"]),
            str(known["same"]),
            str(known["statements"]),
            str(known["statements"] - 1),
            str(faults),
            str(len(first["supported"])),
        ],
        "Version 2 differs from version 1": [SECOND, FIRST],
        "- **The whole rubric.**": [FIRST, RUBRIC],
        "It is still one request for each answer": [FIRST],
        "Its instructions were written after the result of version 1": [
            FIRST,
            TEST_STEP,
            FIRST,
            SECOND,
            SECOND,
            FIRST,
        ],
        "- **Version 2 caught every fault that version 1 caught": [
            SECOND,
            FIRST,
            str(len(second["caught"] - first["caught"])),
        ],
        "- **It missed six.**": [SECOND],
        "- **In two of the misses it found the limit.**": [
            case.split("-")[1] for case in limit_runs.FOUND_THE_LIMIT
        ],
        "- **Its one false alarm**": [
            str(second["false_alarms"][0][1]),
            second["false_alarms"][0][0].split("-")[1],
            SECOND,
        ],
        "Version 2 reads more than twice the input tokens": [
            SECOND,
            FIRST,
            FIRST,
            wrote(len(SUPPORT_INSTRUCTIONS)),
            SECOND,
            wrote(len(RUBRIC_TEXT)),
            wrote(len(HOW_TO_CHECK)),
            SECOND,
        ],
        "This is the check that version 1 passed": [
            FIRST,
            str(on_first["labelled"]),
            str(sum(case["kind"] == "real" for case in limit_runs.first_set().values())),
            str(sum(case["kind"] != "real" for case in limit_runs.first_set().values())),
            SECOND,
            SECOND,
        ],
        "- **On these answers version 2 is worse than version 1": [SECOND, FIRST],
        "- **Its two false alarms are real answers**": [FIRST, SECOND],
        "- **Two of the six are labelled `unclear`": [
            str(real_unclear[1]),
            SECOND,
            *numbers(MEETING),
            str(copy_unclear[1]),
            SECOND,
        ],
        "`make eval-support-v2` judged the": [str(len(baseline)), SECOND, FIRST],
        "- **One is the statement that version 1 faulted too.**": [FIRST],
        "- **The other three are faulted for a limit left out**": [
            SECOND,
            FIRST,
            SECOND,
            SECOND,
            FIRST,
        ],
        "- **The other fault of version 1**": [FIRST, str(other_fault[1]), SECOND],
        '- **Version 1 stays the judge of "supported by its sources".**': [FIRST],
        "- **Each version keeps its verdicts on an answer run in files of its own**": [
            SECOND,
            FIRST,
            SECOND,
            SECOND,
            SECOND,
        ],
        "- **Version 2 is kept as a second instrument for one fault.**": [
            SECOND,
            FIRST,
            SECOND,
            HARDENING,
            RECORD,
        ],
        **lines_of_the_two_runs(),
        "- **Answers that keep their limits.**": [LIMITS],
        "- **Real faults.**": [str(faults)],
        "- **Another author, and a person.**": [str(faults)],
    }


def unclear_of_the_six() -> tuple[tuple[str, int], tuple[str, int]]:
    """Of the statements the two versions decide differently, the two labelled unclear:
    the one of a real answer, and the one of a copy."""
    unclear = [
        key
        for key in limit_runs.decided_differently()
        if limit_runs.first_set_statement(key)["expected"] == "unclear"
    ]
    (real,) = [key for key in unclear if limit_runs.first_set()[key[0]]["kind"] == "real"]
    (copy,) = [key for key in unclear if key != real]
    return real, copy


def test_the_tables_of_the_guide_are_the_runs() -> None:
    text = guide_section()
    first, second = limit_runs.check(1), limit_runs.check(2)
    both = (first, second)
    labelled = limit_runs.cases()

    assert table(text, "| On the limit answers | Version 1 | Version 2 |") == [
        ["Limits left out that it caught (recall)", *(share(k["recall"]) for k in both)],
        ["Alarms that were right (precision)", *(share(k["precision"]) for k in both)],
        [
            "Same decision as the label, unsupported or not",
            *(share(k["same_decision"]) for k in both),
        ],
        ["Agreement beyond chance (Cohen's kappa)", *(f"{k['kappa']:.3f}" for k in both)],
        ["Same verdict on the whole answer", *(share(k["same_answer"]) for k in both)],
        ["Run", *(f"[`{k['run']}`](../evals/runs/{k['run']})" for k in both)],
    ]
    assert table(text, "| Kind of limit | Faults | Version 1 | Version 2 |") == [
        *(
            [f"`{kind}`", str(total), *(str(k["by_kind"][kind][0]) for k in both)]
            for kind, total in ((kind, COUNTED_FAULTS[kind]) for kind in LIMIT_KINDS)
        ),
        [
            "In the title only",
            str(len(first["in_title"])),
            *(str(len(k["caught_in_title"])) for k in both),
        ],
    ]
    assert table(text, "| Case | Kind of limit |") == [
        [
            f"`{case}`",
            f"`{labelled[case]['limit_kind']}`",
            labelled[case]["limit"],
            limit_runs.named_limits(second["reasons"][(case, number)]),
        ]
        for case, number in second["missed"]
    ]
    assert table(text, "| What it costs | Version 1 | Version 2 |") == [
        [
            "Input tokens for each answer, on average",
            *(wrote(round(k["input_per_answer"])) for k in both),
        ],
        [
            "Output tokens for each answer, on average",
            *(wrote(round(k["output_per_answer"])) for k in both),
        ],
        ["Median time of a verdict", *(f"{k['median_seconds']:.1f} s" for k in both)],
        [f"Attempts for the {first['answers']} answers", *(str(k["attempts"]) for k in both)],
    ]
    assert first["answers"] == second["answers"]


def test_the_guides_tables_of_the_first_labelled_answers_and_of_the_baseline_are_the_runs() -> None:
    text = guide_section()
    on_first, on_second = on_the_first_set()
    both = (on_first, on_second)
    by_v1, by_v2 = limit_runs.baseline_verdicts(1), limit_runs.baseline_verdicts(2)

    def label(key: tuple[str, int]) -> str:
        return f"`{limit_runs.first_set_statement(key)['expected']}`"

    assert table(text, "| On the first labelled answers | Version 1 | Version 2 |") == [
        ["Faults caught (recall)", *(share(k["recall"]) for k in both)],
        ["Alarms that were right (precision)", *(share(k["precision"]) for k in both)],
        [
            "Same decision as the label, unsupported or not",
            *(share(k["same_decision"]) for k in both),
        ],
        ["Agreement beyond chance (Cohen's kappa)", *(f"{k['kappa']:.3f}" for k in both)],
        ["Same verdict on the whole answer", *(share(k["same_answer"]) for k in both)],
        [
            "Input tokens for each answer, on average",
            *(wrote(round(k["input_per_answer"])) for k in both),
        ],
        ["Median time of a verdict", *(f"{k['median_seconds']:.1f} s" for k in both)],
        [f"Attempts for the {on_first['judged']} answers", *(str(k["attempts"]) for k in both)],
        ["Run", *(f"[`{k['run']}`](../evals/runs/{k['run']})" for k in both)],
    ]
    assert on_first["judged"] == on_second["judged"] == on_first["labelled"]
    assert table(text, "| Answer | Statement | Label | Version 1 | Version 2 |") == [
        [
            f"`{case}`",
            str(number),
            label((case, number)),
            f"`{on_first['verdicts'][(case, number)]}`",
            f"`{on_second['verdicts'][(case, number)]}`",
        ]
        for case, number in limit_runs.decided_differently()
    ]
    assert table(text, "| Answer | Statement | Version 1 | Label in the first set |") == [
        [
            f"`{case}`",
            str(number),
            f"`{by_v1[(case, number)][0]}`",
            label((case, number)),
            limit_runs.named_limits(by_v2[(case, number)][1]),
        ]
        for case, number in limit_runs.baseline_faulted(2)
    ]


def test_what_the_guide_says_of_the_first_labelled_answers_and_of_the_baseline() -> None:
    text = guide_section()
    on_first, on_second = on_the_first_set()
    differ = limit_runs.decided_differently()
    real = limit_runs.real_unclear_in_the_first_set()
    real_unclear, copy_unclear = unclear_of_the_six()
    counts = baseline_counts()
    first_set = limit_runs.first_set()

    # Worse by every figure of agreement, and where: the statements the two decide
    # differently are those of the table.
    assert worse_on_every_figure_of_agreement(on_first, on_second)
    assert "version 2 is worse than version 1 by every figure of agreement in the table" in text
    assert f"unsupported or not, on {in_words(len(differ))} statements" in text
    # Its one miss: a planted citation of the wrong passage, judged by a passage that the
    # statement does not cite.
    (missed,) = on_second["missed"]
    statement = limit_runs.first_set_statement(missed)
    named = {int(n) for n in re.findall(r"passage (\d+)", on_second["reasons"][missed])}
    assert statement["planted"] == "wrong_passage"
    assert named
    assert not named & set(statement["cites"])
    assert f"the citation of the wrong passage planted in `{missed[0]}`" in text
    # Its false alarms: real answers, each faulted for the limit its reason names, where both
    # readers called the statement supported.
    alarms = on_second["false_alarm_statements"]
    assert f"**Its {in_words(len(alarms))} false alarms are real answers**" in text
    for key in alarms:
        readers = limit_runs.first_set_statement(key)["readers"]
        assert first_set[key[0]]["kind"] == "real", key
        assert readers["first"] == readers["second"] == "supported", key
        assert f'"{limit_runs.named_limits(on_second["reasons"][key])}"' in text, key
    # Version 1's one false alarm, which version 2 calls as it is labelled.
    (alarm_of_version_1,) = on_first["false_alarm_statements"]
    labelled = limit_runs.first_set_statement(alarm_of_version_1)["expected"]
    assert on_second["verdicts"][alarm_of_version_1] == labelled
    assert (
        f"The one false alarm of version 1, in `{alarm_of_version_1[0]}`, version 2 calls "
        f"`{labelled}`, as labelled."
    ) in text
    # The two of the six that are labelled unclear: a real answer the readers split on, and a
    # copy of another, with the same statement, faulted in the copy and passed in the real one.
    assert f"**Two of the {in_words(len(differ))} are labelled `unclear`" in text
    assert real_unclear[0] in real
    assert (
        f"Statement {real_unclear[1]} of `{real_unclear[0]}` is one of the "
        f"{in_words(len(real))} real answers that the readers split on"
    ) in text
    original = copy_unclear[0].removesuffix("-v1")
    in_original = limit_runs.first_set_statement((original, copy_unclear[1]))
    in_copy = limit_runs.first_set_statement(copy_unclear)
    assert first_set[original]["question"] == first_set[copy_unclear[0]]["question"]
    assert first_set[original]["passages"] == first_set[copy_unclear[0]]["passages"]
    assert (in_original["text"], in_original["cites"]) == (in_copy["text"], in_copy["cites"])
    assert [s["cites"] for s in first_set[original]["statements"]] != [
        s["cites"] for s in first_set[copy_unclear[0]]["statements"]
    ]
    assert on_second["verdicts"][copy_unclear] == "unsupported"
    assert on_second["verdicts"][(original, copy_unclear[1])] == "supported"
    assert (
        f"Statement {copy_unclear[1]} of `{copy_unclear[0]}` is the same statement as in the "
        f"real answer `{original}`"
    ) in text
    # The `baseline` answers: real answers of the first set, every verdict from the check.
    state = json.loads((limit_runs.BASELINE / SUPPORT_V2.state_file).read_text(encoding="utf-8"))
    read_by_the_check = {
        row["fingerprint"]
        for row in limit_runs.rows(limit_runs.FIRST_SET_CHECK_V2 / "judgements.jsonl")
    }
    verdict_rows = limit_runs.rows(limit_runs.BASELINE_VERDICTS[2])
    assert state["attempts"] == 0
    assert all(row["fingerprint"] in read_by_the_check for row in verdict_rows)
    assert all(first_set[row["case"]]["kind"] == "real" for row in verdict_rows)
    assert "so each verdict was taken from the check above, and no request was sent" in text
    # The four statements, and what the labels say of each.
    assert f"These are the {in_words(len(counts['faulted']))} statements it calls unsupported" in (
        text
    )
    (both_key,) = counts["both"]
    assert limit_runs.first_set_statement(both_key)["cites"] == []
    assert limit_runs.first_set_statement(both_key)["expected"] == "unsupported"
    assert f"In `{both_key[0]}` it cites no passage, and the labels call it unsupported." in text
    assert f"**The other {in_words(len(counts['only']))} are faulted for a limit left out**" in (
        text
    )
    assert counts["labels"] == {"supported": 2, "unclear": 1}
    assert "The label, supported by both readers, follows the rubric." in text
    assert "The label is supported, by both readers." in text
    (unclear_key,) = [
        key
        for key in counts["only"]
        if limit_runs.first_set_statement(key)["expected"] == "unclear"
    ]
    assert unclear_key[0] in real
    # Version 2 passes the statements of the other real answers the readers split on.
    split_on = [
        (case, s["number"])
        for case in real
        for s in first_set[case]["statements"]
        if s["expected"] == "unclear" and case != unclear_key[0]
    ]
    assert [on_second["verdicts"][key] for key in split_on] == ["supported"] * len(split_on)
    assert (
        f"It passes the other {in_words(len(split_on))} real answers the readers split on" in text
    )
    # The fault of version 1 that version 2 does not raise.
    (other,) = [key for key in limit_runs.baseline_faulted(1) if key not in counts["faulted"]]
    assert limit_runs.first_set_statement(other)["expected"] == "no_claim"
    assert (
        f"statement {other[1]} of `{other[0]}`, which the labels call `no_claim`, version 2 does "
        "not raise"
    ) in text


def test_every_number_of_the_guides_section_is_worked_out_from_the_runs() -> None:
    assert_numbers(guide_section(), guide_numbers())


def test_what_the_guides_section_says_in_words_is_what_the_runs_hold() -> None:
    text = guide_section()
    second = limit_runs.check(2)
    labelled = limit_runs.cases()
    ((alarm_case, _),) = second["false_alarms"]
    alarm = labelled[alarm_case]
    named = limit_runs.named_limits(second["reasons"][second["false_alarms"][0]])

    assert f"**It missed {in_words(len(second['missed']))}.**" in text
    assert f"**In {in_words(len(limit_runs.FOUND_THE_LIMIT))} of the misses it found" in text
    assert f"**Its {in_words(len(second['false_alarms']))} false alarm**" in text
    # The false alarm: a harmless answer, faulted for a limit that its question names.
    assert alarm["kind"] == "harmless"
    assert f'a `{alarm["kind"]}` answer to "{alarm["question"]}"' in text
    assert f'for leaving out "{named}"' in text
    assert named.casefold() in alarm["question"].casefold()
    assert "that a limit the question asks about is kept" in text
    # Its instructions do say so.
    assert "A limit is kept if the statement gives it, in any words; if the question asks" in (
        " ".join(HOW_TO_CHECK.split())
    )
    assert f"In the first labelled answers the judge passed the {in_words(3)} real cases" in text
    assert len(limit_runs.real_unclear_in_the_first_set()) == 3
    assert f"holds {len(labelled)} answers on" in text
    assert f"{in_words(3)} for each" in text
    assert set(Counter(c["passages"][0]["chunk_id"] for c in labelled.values()).values()) == {3}
    assert f"{in_words(the_set()['readers']).capitalize()} reader sessions labelled" in text
    assert "No person has read the set." in text
    assert "The runs do not record why an attempt failed" in text
    assert "This section measures the judge on answers written to test that fault alone" in text


def test_the_guide_names_the_commands_and_the_files_as_they_are() -> None:
    text = guide_section()
    makefile = read(MAKEFILE)
    commands = re.findall(r"^make ([a-z0-9-]+) ", text, flags=re.MULTILINE)

    assert commands == [
        "eval-judge-limits",
        "eval-judge-limits-v2",
        "eval-judge-v2",
        "eval-support-v2",
    ]
    for command in commands:
        assert re.search(rf"^{command}: ##", makefile, flags=re.MULTILINE), command
    # The files of version 2 in an answer run, as the code names them.
    table_of_files = read(GUIDE).split("\n## What a run writes\n")[1]
    files = (SUPPORT_V2.rows_file, SUPPORT_V2.state_file, SUPPORT_V2.check_file)
    assert f"| `{files[0]}`, `{files[1]}`, `{files[2]}` |" in table_of_files


# The cards


def test_the_card_of_the_limit_answers_gives_the_checks_as_the_runs_hold_them() -> None:
    text = read(LIMITS_CARD)
    first, second = limit_runs.check(1), limit_runs.check(2)
    results = section(text, "What the judges did on it")

    assert table(results, "| Judge | Faults caught |") == [
        [
            f"Version {version}",
            of(len(k["caught"]), len(k["faults"])),
            of(len(k["false_alarms"]), len(k["supported"])),
            f"{k['kappa']:.3f}",
            str(k["attempts"]),
            f"[`{k['run']}`](../../runs/{k['run']})",
        ]
        for version, k in ((1, first), (2, second))
    ]
    assert_numbers(
        results,
        {
            "Version 1 gives every score in the results tables and in a run's summary.": [
                FIRST,
                SECOND,
                FIRST,
                CARD_STEP,
                RECORD,
                FIRST,
                str(len(first["in_title"])),
                FIRST,
                SECOND,
                str(len(second["caught_in_title"])),
                SECOND,
            ],
            "Version 2 is not steady on the statements the readers split on.": [SECOND],
        },
    )
    assert first["caught"] <= second["caught"]
    assert not first["caught_in_title"]
    assert "version 1 caught none and version 2 all" in results
    assert f"the {in_words(len(second['missed']))} that version 2 missed" in results
    assert f"its {in_words(len(second['false_alarms']))} false alarm" in results
    # Where the card said that no judge had been run on the file, it now says what was.
    intro = line_starting(text, f"{the_set()['answers']} answers that test one decision")
    assert numbers(intro) == [
        str(the_set()["answers"]),
        LIMITS,
        FIRST,
        str(len(first["caught"])),
        str(len(first["faults"])),
        SECOND,
        str(len(second["caught"])),
    ]
    assert "has not been done" not in text
    assert "The judge has not been run on the file." not in text
    assert "Two versions of the judge have been checked on the file" in text
    assert (
        "`make eval-judge-limits-v2` does the same for version 2 of the judge, and writes to "
        f"`evals/runs/{limit_runs.CHECKS[2]}`"
    ) in text


def test_the_card_of_the_first_set_says_what_the_judges_did_on_the_further_file() -> None:
    first, second = limit_runs.check(1), limit_runs.check(2)
    known = the_set()
    line = line_starting(read(FIRST_CARD), "[`limits/cases.jsonl`](limits/cases.jsonl) holds")

    assert numbers(line) == [
        str(known["answers"]),
        str(known["kinds"]["kept"]),
        str(known["kinds"]["dropped"]),
        str(known["kinds"]["harmless"]),
        str(known["same"]),
        str(known["statements"]),
        FIRST,
        str(len(first["caught"])),
        str(len(first["faults"])),
        SECOND,
        str(len(second["caught"])),
    ]
    assert "The judge has not been checked on it." not in line


# The changelog, the roadmap, the architecture page and the Makefile


def test_the_changelog_and_the_roadmap_give_the_figures_of_the_runs() -> None:
    first, second = limit_runs.check(1), limit_runs.check(2)
    on_first, on_second = on_the_first_set()
    by_second, by_first = limit_runs.baseline_score(2), limit_runs.baseline_score(1)
    # "it calls 125 of the 129 answers of the `baseline` run supported, where version 1
    # calls 127"
    scores = [
        str(round(by_second.mean * by_second.count)),
        str(by_second.count),
        FIRST,
        str(round(by_first.mean * by_first.count)),
    ]
    known = the_set()
    changelog = read(CHANGELOG).split("## [Unreleased]")[1].split("\n## [")[0]
    roadmap = read(ROADMAP)
    labelled = line_starting(changelog, "- Labelled answers for one fault, `evals/judge/limits/`")
    judge = line_starting(changelog, "- Version 2 of the support judge, `--judge-version 2`")
    files = line_starting(changelog, "- Each version of a judge keeps its verdicts")
    item = line_starting(roadmap, "3. **Measure how often the answers leave out a limit")
    started = line_starting(roadmap, "- **Answers that leave out a limit**")

    assert numbers(labelled) == [
        str(known["answers"]),
        str(known["passages"]),
        str(known["same"]),
        str(known["statements"]),
        LIMITS,
        RECORD,
    ]
    assert numbers(judge) == [
        SECOND,
        SECOND,
        str(known["answers"]),
        str(len(second["caught"])),
        str(len(second["faults"])),
        FIRST,
        str(len(first["caught"])),
        str(len(second["supported"])),
        str(on_second["caught"]),
        str(on_second["faults"]),
        str(on_second["false_alarms"]),
        FIRST,
        str(on_first["caught"]),
        str(on_first["false_alarms"]),
        *scores,
        FIRST,
        LIMITS,
        RECORD,
    ]
    assert f"with {in_words(len(second['false_alarms']))} false alarm in" in judge
    assert numbers(files) == [SECOND, FIRST, SECOND, RECORD]
    assert f"`{SUPPORT_V2.rows_file}`, `{SUPPORT_V2.state_file}` and `{SUPPORT_V2.check_file}`" in (
        files
    )
    assert "\n[#59]: https://github.com/MumukshuPanchal/handbook-rag/issues/59\n" in read(CHANGELOG)
    assert f"\n[ADR {RECORD}]: docs/adr/{ADR.name}" in read(CHANGELOG)
    # The roadmap: what is left of issue 59, and what was started and is not finished.
    # The item of the hardening against planted instructions is item 2.
    assert line_starting(roadmap, "2. **Harden the assistant against planted instructions")
    # Item 3 gives the questions of the limit answers asked under both sets of instructions,
    # as the runs hold them (tests/evals/test_hardening_documents.py holds the figures).
    pairs, read_pairs = hardening_runs.limit_pairs(), hardening_runs.limit_pairs(read=True)
    goods = {v: sum(o == "good" for o in hardening_runs.limit_outcomes(v).values()) for v in (1, 2)}
    assert numbers(item) == [
        "3",
        LIMITS,
        SECOND,
        str(known["passages"]),
        SECOND,
        str(known["passages"]),
        SECOND,
        str(goods[2]),
        FIRST,
        str(goods[1]),
        str(len(pairs["better"])),
        f"{hardening_runs.sign_test(len(pairs['better']), len(pairs['worse'])):.3f}",
        str(len(read_pairs["better"])),
        str(len(read_pairs["worse"])),
        f"{hardening_runs.sign_test(len(read_pairs['better']), len(read_pairs['worse'])):.3f}",
        SECOND,
        FIRST,
        FIRST,
        RUBRIC,
    ]
    assert "a person's reading of the eight faulted answers" in item
    assert numbers(started) == [
        LIMITS,
        str(known["answers"]),
        str(known["passages"]),
        FIRST,
        str(len(first["caught"])),
        str(len(first["faults"])),
        SECOND,
        str(len(second["caught"])),
        str(len(second["supported"])),
        RECORD,
        FIRST,
        str(on_second["false_alarms"]),
        FIRST,
        str(on_first["false_alarms"]),
        *scores,
        FIRST,
        SECOND,
        "2",
    ]
    assert f"with {in_words(len(second['false_alarms']))} false alarm in" in started
    assert roadmap.index(started) > roadmap.index("Started since 1.0, and not finished:")
    assert "are neither counted in the assistant's own answers nor reduced" in started


def test_the_architecture_page_names_the_second_version() -> None:
    assert (
        "The first judge has a second version, written for the limit an answer leaves out. It "
        "keeps its verdicts apart, and no score in the results tables or in a run's summary is "
        "read from them"
    ) in read(ARCHITECTURE)


def test_the_make_target_judges_the_baseline_answers_with_the_check_of_version_2() -> None:
    makefile = read(MAKEFILE)
    (description, recipe) = re.findall(
        r"^eval-support-v2: ## (.+)\n\t(.+)$", makefile, flags=re.MULTILINE
    )[0]
    (check_recipe,) = re.findall(r"^eval-judge-v2: ## .+\n\t(.+)$", makefile, flags=re.MULTILINE)

    assert recipe.startswith("uv run handbook-rag eval-support --judge-version 2 ")
    assert "--variant baseline" in recipe
    # It quotes, and takes the verdicts of, the check that `make eval-judge-v2` makes.
    variant = check_recipe.split("--variant ")[1]
    assert f"--checked {limit_runs.FIRST_SET_CHECK_V2.relative_to(ROOT)}" in recipe
    assert limit_runs.FIRST_SET_CHECK_V2.name == variant
    assert "a real request only for an answer the check did not read" in description
    assert "eval-support-v2" in makefile.split("\n")[1].split()


# Links


def test_every_link_of_the_new_documents_leads_somewhere() -> None:
    for path in (ADR, LIMITS_CARD, FIRST_CARD, GUIDE, ROADMAP, ARCHITECTURE):
        text = read(path)
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if target.startswith(("http://", "https://")):
                continue
            file, _, anchor = target.partition("#")
            linked = (path.parent / file) if file else path
            assert linked.exists(), (path.name, target)
            if anchor:
                assert anchor in anchors(read(linked)), (path.name, target)


# Version 2's check on the first labelled answers, and its verdicts on the `baseline` answers


def test_version_2s_check_on_the_first_labelled_answers_is_whole_and_of_the_judge_built() -> None:
    second, first = limit_runs.first_set_check(2), limit_runs.first_set_check(1)
    saved = json.loads((limit_runs.FIRST_SET_CHECK_V2 / "run.json").read_text(encoding="utf-8"))

    assert second["judged"] == second["labelled"] == first["labelled"]
    assert second["summary"]["criterion"] == "support"
    assert second["summary"]["config"] == saved["config"] == describe_judge(Settings(), SUPPORT_V2)
    assert second["faults"] == first["faults"]
    assert second["not_faults"] == first["not_faults"]


def test_the_documents_give_version_2s_check_on_the_first_labelled_answers() -> None:
    for text in (guide_section(), read(ADR)):
        line = line_starting(text, limit_runs.FIRST_SET_LINE)
        assert numbers(line) == first_set_line_numbers(), line
    for path in (README, GUIDE, ADR, ROADMAP):
        assert limit_runs.NOT_YET_CHECKED not in read(path), path.name


def test_the_baseline_verdicts_of_version_2_are_of_the_judge_built_and_quote_its_check() -> None:
    state = json.loads((limit_runs.BASELINE / SUPPORT_V2.state_file).read_text(encoding="utf-8"))
    check = json.loads((limit_runs.BASELINE / SUPPORT_V2.check_file).read_text(encoding="utf-8"))
    judged = {row["case"] for row in limit_runs.rows(limit_runs.BASELINE_VERDICTS[2])}

    assert state["config"] == describe_judge(Settings(), SUPPORT_V2)
    assert check["config"] == state["config"]
    assert check["run"] == str(limit_runs.FIRST_SET_CHECK_V2.relative_to(ROOT))
    assert judged == {row["case"] for row in limit_runs.rows(limit_runs.BASELINE_VERDICTS[1])}


def test_the_documents_give_version_2s_score_of_the_baseline_answers() -> None:
    for text in (guide_section(), read(ADR)):
        line = line_starting(text, limit_runs.BASELINE_LINE)
        assert numbers(line) == baseline_line_numbers(), line
    for path in (README, GUIDE, ADR, ROADMAP):
        assert limit_runs.NOT_YET_RECORDED not in read(path), path.name


def _cases(path: Path) -> dict[str, dict[str, Any]]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return {row["case"]: row for row in rows}


def test_four_sentences_a_review_could_change_unnoticed() -> None:
    """A second review changed one word in each of these and no test failed.

    Each is held to the file it speaks of, so that the sentence cannot name another count,
    another case or another statement than the files do.
    """
    readme = read(README)
    for check in (
        "(evals/runs/judge/baseline)",
        "(evals/runs/judge/v2)",
        "(evals/runs/judge-correct",
    ):
        assert check in readme, f"the judges table names the check {check}"
    assert "under three sets of instructions" in readme

    first_set = _cases(ROOT / "evals" / "judge" / "support" / "cases.jsonl")
    assert {"q104", "q104-v1"} <= set(first_set)
    assert "It passes the same statement in the copy `q104-v1`" in read(GUIDE)

    assert "none of its three alarms for a limit is confirmed by the labels" in read(ADR)

    statements = {
        statement["number"]: statement["expected"]
        for statement in _cases(ROOT / "evals" / "judge" / "limits" / "cases.jsonl")["lim-083"][
            "statements"
        ]
    }
    assert statements == {1: "unsupported", 2: "supported"}
    assert "Statement 1 of `lim-083` is labelled unsupported" in read(LIMITS_CARD)
