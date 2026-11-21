"""These tests load the real embedding model, so they are marked slow."""

import math

import pytest

from handbook_rag.index.embedder import SentenceTransformerEmbedder

pytestmark = pytest.mark.slow

PREFIX = "Represent this sentence for searching relevant passages: "


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", query_prefix=PREFIX)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_model_reports_its_shape(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.dimension == 384
    assert embedder.max_tokens == 512
    assert embedder.name == "BAAI/bge-small-en-v1.5"


def test_vectors_have_length_one(embedder: SentenceTransformerEmbedder) -> None:
    (vector,) = embedder.embed_documents(["Parental leave is sixteen weeks."])

    assert len(vector) == 384
    assert math.sqrt(sum(v * v for v in vector)) == pytest.approx(1.0, abs=1e-4)


def test_question_is_closer_to_its_answer_than_to_an_unrelated_passage(
    embedder: SentenceTransformerEmbedder,
) -> None:
    answer, unrelated = embedder.embed_documents(
        [
            "New parents can take up to sixteen weeks of fully paid leave.",
            "Rotate production database credentials every ninety days.",
        ]
    )
    question = embedder.embed_query("How much time off do I get when my baby is born?")

    assert cosine(question, answer) > cosine(question, unrelated) + 0.1


def test_query_prefix_is_applied_to_questions_only(embedder: SentenceTransformerEmbedder) -> None:
    text = "parental leave policy"
    (as_document,) = embedder.embed_documents([text])

    assert cosine(embedder.embed_query(text), as_document) < 0.999


def test_empty_input_gives_empty_output(embedder: SentenceTransformerEmbedder) -> None:
    assert embedder.embed_documents([]) == []
