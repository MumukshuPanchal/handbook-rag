"""The adversarial cases in the repository are what their card says they are.

Only the last test needs the corpus. It builds it from the handbook snapshot, as the
commands do, so it is marked slow and is skipped where there is no snapshot, as in CI.
That a passage still reads as it did is also checked by the command, before it spends a
request on the case.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.config import Settings
from handbook_rag.evals.adversarial import (
    INJECTED,
    KINDS,
    OUT_OF_SCOPE,
    WITH_OWN_CANARY,
    AdversarialCase,
    find_canaries,
    instruction_canaries,
    key_fact_stated,
    load_cases,
    load_readings,
    load_replies,
    planted_passages,
    reply_fingerprint,
    summarise_adversarial,
)
from handbook_rag.evals.dataset import load_golden_set
from handbook_rag.evals.support import passage_hash
from handbook_rag.generate.markers import MARKER_SYSTEM_PROMPT
from handbook_rag.ingest.build import build_corpus

ROOT = Path(__file__).parents[2]
CASES = ROOT / "evals" / "adversarial" / "cases.jsonl"
BASELINE = ROOT / "evals" / "runs" / "answers" / "baseline"
ADVERSARIAL_BASELINE = ROOT / "evals" / "runs" / "adversarial" / "baseline"
SNAPSHOT = ROOT / "data" / "handbook"
CANARIES = instruction_canaries(MARKER_SYSTEM_PROMPT)


def cases() -> list[AdversarialCase]:
    return load_cases(CASES)


def injected() -> list[AdversarialCase]:
    return [case for case in cases() if case.injected]


def baseline(name: str) -> dict[str, dict[str, Any]]:
    rows = map(json.loads, (BASELINE / name).read_text(encoding="utf-8").splitlines())
    return {row.get("question_id", row.get("prompt_id")): row for row in rows}


def test_the_cases_are_in_the_numbers_the_card_gives() -> None:
    counts = Counter((case.family, case.kind) for case in cases())

    assert counts == {
        (INJECTED, "override"): 4,
        (INJECTED, "append_link"): 4,
        (INJECTED, "reveal_instructions"): 4,
        (INJECTED, "forged_boundary"): 4,
        (INJECTED, "withhold"): 4,
        (OUT_OF_SCOPE, "general_knowledge"): 4,
        (OUT_OF_SCOPE, "task"): 4,
        (OUT_OF_SCOPE, "other_company"): 3,
        (OUT_OF_SCOPE, "direct_injection"): 5,
    }
    assert {kind for _, kind in counts} == {kind for kinds in KINDS.values() for kind in kinds}


def test_each_kind_is_planted_in_the_same_four_answered_golden_questions() -> None:
    golden = {
        question.id: question
        for question in load_golden_set(ROOT / "evals" / "golden" / "questions.jsonl")
    }
    by_kind: dict[str, list[str]] = {}
    for case in injected():
        assert case.source is not None
        by_kind.setdefault(case.kind, []).append(case.source)
        # The question is the golden question's, word for word.
        assert golden[case.source].type == "answerable", case.case_id
        assert case.question == golden[case.source].question, case.case_id

    sources = {tuple(sorted(found)) for found in by_kind.values()}
    assert len(sources) == 1
    (four,) = sources
    assert len(set(four)) == 4


def test_each_case_keeps_the_passages_the_baseline_gave_the_model_in_their_order() -> None:
    answers = baseline("answers.jsonl")
    for case in injected():
        assert case.source is not None
        record = answers[case.source]
        given = [(passage["number"], passage["chunk_id"]) for passage in record["passages"]]
        assert [(ref.number, ref.chunk_id) for ref in case.passages] == given, case.case_id
        # The evidence is where the baseline found it.
        holding = [passage["number"] for passage in record["passages"] if passage["covers"]]
        assert list(case.evidence) == holding, case.case_id


def test_the_questions_were_answered_correctly_from_their_evidence_in_the_baseline() -> None:
    answers, results = baseline("answers.jsonl"), baseline("results.jsonl")
    sections = set()
    for case in injected():
        assert case.source is not None
        grade = results[case.source]["grade"]
        assert grade["answered"] == grade["correct"] == grade["cites_evidence"] == 1.0
        cited = {
            number
            for statement in answers[case.source]["statements"]
            for number in statement["cites"]
        }
        assert cited & set(case.evidence), case.case_id
        evidence = answers[case.source]["passages"][case.evidence[0] - 1]
        sections.add(evidence["doc_id"].split("/")[0])
    # Four handbook sections, one for each question.
    assert len(sections) == 4


def test_half_of_each_kind_is_planted_in_the_evidence_and_all_withheld_text_is() -> None:
    in_evidence: Counter[str] = Counter()
    for case in injected():
        in_evidence[case.kind] += case.planted_in in case.evidence

    assert in_evidence == {
        "override": 2,
        "append_link": 2,
        "reveal_instructions": 2,
        "forged_boundary": 2,
        "withhold": 4,
    }


def test_every_canary_is_a_made_up_word_or_address_of_one_case_and_not_in_its_question() -> None:
    seen: list[str] = []
    for case in injected():
        if case.kind not in WITH_OWN_CANARY:
            assert case.canaries == (), case.case_id
            continue
        (canary,) = case.canaries
        seen.append(canary.casefold())
        assert not find_canaries(case.question, [canary]), case.case_id
        assert find_canaries(case.planted or "", [canary]), case.case_id
        if case.kind == "append_link":
            assert re.fullmatch(r"[a-z-]+\.example/[a-z]+/\d{4}", canary), case.case_id
        else:
            assert re.fullmatch(r"[A-Z]+-\d{4}", canary), case.case_id
    assert len(seen) == len(set(seen)) == 12
    # No two tokens share their number either, so none is a variant of another.
    assert len({re.sub(r"\D", "", canary) for canary in seen}) == 12


def test_no_question_or_planted_text_holds_a_phrase_of_the_instructions() -> None:
    for case in cases():
        assert not find_canaries(case.question, CANARIES), case.case_id
        assert not find_canaries(case.planted or "", CANARIES), case.case_id


def test_a_forged_boundary_closes_every_tag_the_passages_are_wrapped_in() -> None:
    for case in injected():
        if case.kind != "forged_boundary":
            continue
        closed = re.findall(
            r"<\s*/\s*(content|search_result|search_results)\s*>", case.planted or "", re.IGNORECASE
        )
        assert [tag.lower() for tag in closed] == ["content", "search_result", "search_results"]
        assert re.search(r"system", case.planted or "", re.IGNORECASE), case.case_id


def test_every_injected_case_has_the_key_fact_of_its_golden_question() -> None:
    # The loader refuses two key facts for one golden question; this asks for one at all.
    facts = {case.source: case.key_fact for case in injected()}

    assert len(facts) == 4
    assert all(fact is not None for fact in facts.values())


def test_each_key_fact_is_in_the_reference_answer_and_in_the_reply_judged_correct() -> None:
    golden = {
        question.id: question
        for question in load_golden_set(ROOT / "evals" / "golden" / "questions.jsonl")
    }
    answers = baseline("answers.jsonl")
    for case in injected():
        assert case.source is not None
        reply = "".join(s["text"] for s in answers[case.source]["statements"])
        assert key_fact_stated(case, golden[case.source].reference_answer), case.case_id
        # The first answer run's reply, which the second judge called correct.
        assert key_fact_stated(case, reply), case.case_id


def test_the_saved_adversarial_baseline_is_decided_by_the_rule_of_today() -> None:
    """Its results are what the code makes of its replies and readings now, with the key
    facts, so a figure read from it is a figure of the rule the documents describe."""
    cases = load_cases(CASES)
    report = summarise_adversarial(
        cases,
        load_replies(ADVERSARIAL_BASELINE),
        CANARIES,
        load_readings(ADVERSARIAL_BASELINE / "readings.jsonl"),
    )
    rows = map(json.loads, (ADVERSARIAL_BASELINE / "results.jsonl").read_text().splitlines())
    saved = {row["case"]: row for row in rows}

    assert report.final
    assert len(saved) == len(report.results) == len(cases)
    for result in report.results:
        row = saved[result.case.case_id]
        stated = key_fact_stated(result.case, result.record.text)
        assert [row["by_code"], row["reason"], row["outcome"]] == [
            result.by_code,
            result.reason,
            result.outcome,
        ], result.case.case_id
        assert row["reply_fingerprint"] == reply_fingerprint(result.record)
        expected = None if result.case.key_fact is None else stated is not None
        assert row["states_key_fact"] == expected, result.case.case_id
    summary = json.loads((ADVERSARIAL_BASELINE / "summary.json").read_text())
    assert set(summary["key_facts"]) == {case.source for case in injected()}


@pytest.mark.parametrize(
    ("run_name", "declined_in_the_gap_form"),
    [
        ("hardened-1", ["oos-task-4"]),
        (
            "hardened-2",
            ["oos-general-1", "oos-general-4", "oos-task-4", "oos-company-2", "oos-direct-4"],
        ),
    ],
)
def test_a_hardened_run_read_today_keeps_its_figures_and_its_models_own_declines(
    run_name: str, declined_in_the_gap_form: list[str]
) -> None:
    """Saved before the generator read a reply of the fixed form alone as a decline, these
    runs' replies of that kind went through the check for a reply that cites nothing. Read
    today, each is the model's decline, and no figure of the run moves."""
    run = ROOT / "evals" / "runs" / "adversarial" / run_name
    saved_summary = json.loads((run / "summary.json").read_text())
    report = summarise_adversarial(
        load_cases(CASES),
        load_replies(run),
        tuple(saved_summary["instruction_canaries"]),
        load_readings(run / "readings.jsonl"),
    )

    model_declines = sorted(r.case.case_id for r in report.results if r.record.model_text)
    assert model_declines == sorted(declined_in_the_gap_form)
    for result in report.results:
        if result.record.model_text:
            assert (result.record.checks, result.record.answered) == ((), False)
            assert result.record.model_text.startswith("The passages found do not say")
    checked = {r.case.case_id: [c.name for c in r.record.checks] for r in report.results}
    assert {case: names for case, names in checked.items() if names} == {
        "inj-withhold-b26": ["cites_nothing"]
    }
    for name, figures in report.figures.items():
        held = saved_summary["figures"][name]["held"]
        assert figures.held is not None, name
        assert held is not None, name
        assert (figures.held.mean, figures.held.count) == (held["mean"], held["count"]), name
        assert figures.held_by_code == saved_summary["figures"][name]["held_by_code"], name


@pytest.mark.slow
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="needs the handbook snapshot in data/handbook")
def test_every_passage_reads_as_it_did_and_holds_no_canary() -> None:
    # Built from the snapshot, as the commands build it. An export of the chunks that
    # was made before the chunking last changed must not be able to stand in for it.
    corpus = {
        chunk.chunk_id: chunk for chunk in build_corpus(Settings(data_dir=ROOT / "data")).chunks
    }
    for case in injected():
        for ref in case.passages:
            assert ref.chunk_id in corpus, f"{case.case_id}: passage {ref.chunk_id} is gone"
            assert passage_hash(corpus[ref.chunk_id].text) == ref.text_sha256, case.case_id
        # Refuses a passage that already holds one of the case's canaries, or a phrase of
        # the instructions, before the planted text is added.
        planted = planted_passages(case, corpus, CANARIES)
        assert sum(chunk.text != corpus[chunk.chunk_id].text for chunk in planted) == 1
