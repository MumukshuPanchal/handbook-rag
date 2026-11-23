"""The single entry point of the assistant: a question goes in, an answer comes out.

The command line, the HTTP API and the evaluation all call this function. Measuring the
same code path that users hit is what makes an evaluation result mean something.

The function also opens the spans of an answer, so every caller gets the same trace:

    answer                          the question as a whole
      search                        the passages found, with their ids and scores
        retrieve.hybrid ...         the retriever's own spans
      generate_content <model>      a request to the language model, one span for each

An answer asked for on the command line or over HTTP may take several requests, to the
answer model and then to a fallback model (see ``generate/retrying.py``). Each one is a
span of its own, in the order they were sent.
"""

import dataclasses
from collections.abc import Callable, Sequence

from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.models import Answer
from handbook_rag.generate.retrying import RetryingGenerator, traced_request
from handbook_rag.ingest.models import Chunk
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.tracing import (
    record_answer,
    record_query,
    record_requests,
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

    When no answer can be written, the generation error leaves with the passages the
    search found, so that whoever reports it can show them without searching again.
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
        try:
            answer = _generate(generator, question, chunks, on_text)
        except GenerationError as failure:
            record_requests(span, requests=failure.requests, fallback=failure.fallback)
            # The search did its job, and what it found is still worth showing.
            failure.retrieved = tuple(retrieved)
            raise
        record_requests(span, requests=answer.requests, fallback=answer.fallback)
        record_answer(span, answer)
    return dataclasses.replace(answer, retrieved=tuple(retrieved))


def _generate(
    generator: Generator,
    question: str,
    chunks: Sequence[Chunk],
    on_text: Callable[[str], None] | None,
) -> Answer:
    """Ask the generator for an answer, with one span for each request to a model.

    A wrapper that may send several requests opens the span of each one itself, because
    only it knows how many it sends. Around any other generator, such as the one an
    evaluation uses, one call is one request, and its span is opened here.
    """
    if not chunks:
        # With nothing to answer from, a generator declines without asking its model
        # (see generate/base.py). A span for a model call that was never made would
        # count as one, with no tokens, in a backend that sums model calls, and the
        # answer counts no request either.
        declined = generator.generate(question, chunks, on_text=on_text)
        return dataclasses.replace(declined, requests=0, fallback=False)
    if isinstance(generator, RetryingGenerator):
        return generator.generate(question, chunks, on_text=on_text)
    return traced_request(generator, question, chunks, on_text=on_text)
