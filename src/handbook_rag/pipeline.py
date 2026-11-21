"""The single entry point of the assistant: a question goes in, an answer comes out.

The command line, the HTTP API and the evaluation all call this function. Measuring the
same code path that users hit is what makes an evaluation result mean something.
"""

import dataclasses
from collections.abc import Callable

from handbook_rag.generate.claude import Generator
from handbook_rag.generate.models import Answer
from handbook_rag.retrieve.dense import Retriever


def answer_question(
    question: str,
    *,
    retriever: Retriever,
    generator: Generator,
    limit: int,
    section: str | None = None,
    on_text: Callable[[str], None] | None = None,
) -> Answer:
    """Retrieve the most relevant chunks for a question and answer from them."""
    retrieved = retriever.retrieve(question, limit=limit, section=section)
    answer = generator.generate(question, [result.chunk for result in retrieved], on_text=on_text)
    return dataclasses.replace(answer, retrieved=tuple(retrieved))
