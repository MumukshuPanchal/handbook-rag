"""The golden set that ships with the repository must always load.

CI has no handbook snapshot, so it cannot check the evidence quotes against their pages;
``handbook-rag golden-check`` does that locally. What CI can check is the file itself.
"""

from pathlib import Path

from handbook_rag.evals.dataset import load_golden_set

GOLDEN_SET = Path(__file__).resolve().parents[2] / "evals" / "golden" / "questions.jsonl"


def test_committed_golden_set_loads_and_follows_the_drafting_guidelines() -> None:
    questions = load_golden_set(GOLDEN_SET)

    answerable = [question for question in questions if question.type == "answerable"]
    unanswerable = [question for question in questions if question.type == "unanswerable"]
    assert len(answerable) >= 100
    assert len(unanswerable) >= 30
    for question in answerable:
        for evidence in question.evidence:
            assert 40 <= len(evidence.quote) <= 300, question.id
            assert evidence.doc_id.endswith(".md"), question.id
    for question in unanswerable:
        assert question.note, f"{question.id}: say why the handbook cannot answer it"
