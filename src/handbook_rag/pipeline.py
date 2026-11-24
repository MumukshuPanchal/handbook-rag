"""The single entry point of the assistant: a question goes in, an answer comes out.

The command line, the HTTP API and the evaluation all call this function. Measuring the
same code path that users hit is what makes an evaluation result mean something.

The function also opens the spans of an answer, so every caller gets the same trace:

    answer                          the question as a whole
      cache.lookup                  what the answer cache did, when the caller has one
      search                        the passages found, with their ids and scores
        retrieve.hybrid ...         the retriever's own spans
      generate_content <model>      a request to the language model, one span for each

An answer asked for on the command line or over HTTP may take several requests, to the
answer model and then to a fallback model (see ``generate/retrying.py``). Each one is a
span of its own, in the order they were sent.

The command line and the HTTP API pass an answer cache (``cache/answers.py``). When it
holds an answer, there is no search and no request: the trace ends with the lookup. An
evaluation never passes one, because it measures what the model writes.
"""

import dataclasses
from collections.abc import Callable, Sequence

from handbook_rag.cache.answers import AnswerCache, Lookup
from handbook_rag.generate.base import GenerationError, Generator
from handbook_rag.generate.models import Answer
from handbook_rag.generate.retrying import RetryingGenerator, traced_request
from handbook_rag.ingest.models import Chunk
from handbook_rag.retrieve.dense import Retriever
from handbook_rag.tracing import (
    record_answer,
    record_cache_lookup,
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
    cache: AnswerCache | None = None,
    fresh: bool = False,
) -> Answer:
    """Retrieve the most relevant chunks for a question and answer from them.

    ``on_text`` receives the answer's text as the model writes it, from the thread this
    runs in. ``on_trace`` receives the id of the answer's trace before anything else
    happens, and only when the trace is recorded. A caller that reports a failure needs
    the id, and by then the spans are closed.

    When no answer can be written, the generation error leaves with the passages the
    search found, so that whoever reports it can show them without searching again.

    With a ``cache``, an answer stored for this question, or for one that means the
    same, is returned without a search or a request. It says that it is one
    (``Answer.cached``) and counts no request, and ``on_text`` receives its text as one
    piece. Otherwise the answer is written as always and then handed to the cache,
    which keeps it if it is one to keep. ``fresh`` skips the looking up and not the
    keeping: the new answer takes the place of what the cache held for the question.
    Without a ``cache``, which is the default and what an evaluation gets, none of this
    happens.
    """
    with stage("answer", section=section) as span:
        traced = trace_id(span)
        if traced is not None and on_trace is not None:
            on_trace(traced)
        record_query(span, question)
        lookup: Lookup | None = None
        if cache is not None:
            with stage("cache.lookup", mode=cache.mode) as looking:
                lookup = cache.lookup(question, section=section, read=not fresh)
                record_cache_lookup(looking, lookup)
            if lookup.answer is not None:
                if on_text is not None:
                    on_text(lookup.answer.text)
                record_requests(span, requests=0, fallback=False)
                record_answer(span, lookup.answer)
                return lookup.answer
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
    answer = dataclasses.replace(answer, retrieved=tuple(retrieved))
    if cache is not None and lookup is not None:
        # After the answer's span has closed: keeping an answer is no part of writing it.
        cache.store(lookup, answer)
    return answer


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
