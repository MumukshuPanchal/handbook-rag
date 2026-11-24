"""A real question to the HTTP API, answered by the real model and streamed over a socket.

The other tests of ``POST /ask`` use a stand-in model and an in-process client, which
hands over a stream only once it has ended. This one starts a real server on this
machine and reads the events as they arrive. It needs a key and spends one request, so
it does not run by default and never runs in CI: use ``make test-live``.

The search runs over two passages with stand-in models, as in the other live tests:
the model is what is being tried here, not the index.
"""

import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from handbook_rag import wiring
from handbook_rag.api import create_app
from handbook_rag.config import Settings
from handbook_rag.index.indexer import index_chunks
from handbook_rag.measuring import measuring
from tests.support import FakeEmbedder, FakeSparseEncoder, make_chunk

pytestmark = pytest.mark.live

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEAVE = make_chunk("New parents get sixteen weeks of paid leave.", heading_path=("Parental leave",))
EXPENSES = make_chunk(
    "A laptop can be expensed up to one thousand dollars.",
    doc_id="finance/expenses.md",
    title="Expenses",
)
# How long the server may take to start, and the answer to arrive.
START_SECONDS = 30.0
ANSWER_SECONDS = 120.0


@pytest.fixture
def settings() -> Settings:
    # Tests run in an empty directory, so the project's own .env file is named here. The
    # model is the default one, whatever that file selects for everyday use. Nothing is
    # sent to a trace backend, and no reranking model is loaded.
    settings = Settings(
        _env_file=PROJECT_ROOT / ".env",
        answer_model=Settings.model_fields["answer_model"].default,
        tracing=False,
        rerank=False,
    )
    if settings.gemini_api_key is None or not settings.gemini_api_key.get_secret_value():
        pytest.skip("no GEMINI_API_KEY in the project's .env file")
    return settings


@pytest.fixture
def server(settings: Settings) -> Iterator[str]:
    """The API on a free port of this machine, answering with the real model."""
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    store = wiring.make_memory_store("live")
    index_chunks([LEAVE, EXPENSES], embedder=embedder, sparse_encoder=sparse_encoder, store=store)
    retriever = wiring.make_retriever(
        settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )
    # No generator is passed: the application builds the real one from the settings.
    application = create_app(settings, retriever=retriever, store=store)
    listening = socket.socket()
    listening.bind(("127.0.0.1", 0))
    running = uvicorn.Server(uvicorn.Config(application, log_level="warning"))
    thread = threading.Thread(target=running.run, kwargs={"sockets": [listening]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + START_SECONDS
    while not running.started:
        assert thread.is_alive(), "the server stopped while starting"
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{listening.getsockname()[1]}"
    finally:
        running.should_exit = True
        thread.join(START_SECONDS)
        listening.close()


def read_events(lines: Iterator[str]) -> Iterator[tuple[str, Any]]:
    """The events of a stream of server-sent events, as their name and their parsed data."""
    name, data = "", ""
    for line in lines:
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data = line.removeprefix("data: ")
        elif line == "" and name:
            yield name, json.loads(data)
            name, data = "", ""


def test_a_real_question_is_answered_over_http_as_a_stream_of_events(server: str) -> None:
    # Every live test runs with the answer cache switched off (conftest.py): this one is
    # here to see what the model writes today.
    assert measuring()
    with (
        httpx.Client(timeout=ANSWER_SECONDS) as client,
        client.stream(
            "POST", f"{server}/ask", json={"query": "How long is parental leave?", "stream": True}
        ) as response,
    ):
        assert response.status_code == 200, response.read()
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["cache-control"] == "no-cache"
        events = list(read_events(response.iter_lines()))

    names = [name for name, _data in events]
    assert names[-1] == "answer", names
    assert set(names[:-1]) == {"text"}, "text first, as the model writes it"
    answer = events[-1][1]
    texts = "".join(data["text"] for name, data in events if name == "text")
    assert texts == "".join(segment["text"] for segment in answer["segments"])
    assert answer["answered"] is True
    assert "sixteen" in answer["answer"].lower() or "16" in answer["answer"]
    assert "[1]" in answer["answer"]
    assert [source["doc_id"] for source in answer["sources"]] == ["people/leave.md"]
    assert answer["stop_reason"] == "end_turn"
    assert (answer["cached"], answer["requests"]) == (None, 1), "the model wrote it just now"
    assert answer["model"].startswith("gemini-")
    assert answer["usage"]["input_tokens"] > 0
    assert answer["usage"]["output_tokens"] > 0
    assert 0 < answer["first_text_ms"] <= answer["took_ms"]
