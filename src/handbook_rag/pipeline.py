"""The single entry point of the assistant: a question goes in, an answer comes out.

The command line, the HTTP API and the evaluation all call this function. Measuring the
same code path that users hit is what makes an evaluation result mean something.

The function also opens the spans of an answer, so every caller gets the same trace:

    answer                          the question as a whole
      search                        the passages found, with their ids and scores
        retrieve.hybrid ...         the retriever's own spans
      generate_content <model>      the request to the language model
"""

import dataclasses
from collections.abc import Callable

from handbook_rag.generate.base import Generator
from handbook_rag.generate.models import Answer
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.tracing import (
    model_call,
    record_answer,
    record_query,
    record_reply,
    record_results,
    stage,
    trace_id,
)


def answer_question(
    question: str,
    *,
    retriever: Retriever,
    generator: Generator,
    limit: int,
    section: str | None = None,
    on_text: Callable[[str], None] | None = None,
    on_trace: Callable[[str], None] | None = None,
) -> Answer:
    """Retrieve the most relevant chunks for a question and answer from them.

    ``on_text`` receives the answer's text as the model writes it, from the thread this
    runs in. ``on_trace`` receives the id of the answer's trace before anything else
    happens, and only when the trace is recorded. A caller that reports a failure needs
    the id, and by then the spans are closed.
    """
    with stage("answer", section=section) as span:
        traced = trace_id(span)
        if traced is not None and on_trace is not None:
            on_trace(traced)
        record_query(span, question)
        with stage("search", limit=limit, section=section) as searching:
            record_query(searching, question)
            retrieved = retriever.retrieve(question, limit=limit, section=section)
            record_results(searching, retrieved)
        chunks = [result.chunk for result in retrieved]
        if chunks:
            with model_call(
                provider=generator.provider,
                operation=generator.operation,
                model=generator.model,
                passages=len(chunks),
            ) as calling:
                answer = generator.generate(question, chunks, on_text=on_text)
                record_reply(calling, answer)
        else:
            # With nothing to answer from, a generator declines without asking its model
            # (see generate/base.py). A span for a model call that was never made would
            # count as one, with no tokens, in a backend that sums model calls.
            answer = generator.generate(question, chunks, on_text=on_text)
        record_answer(span, answer)
    return dataclasses.replace(answer, retrieved=tuple(retrieved))
