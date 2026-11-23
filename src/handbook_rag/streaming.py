"""An answer written in a worker thread, read by a web request as a series of events.

The generators hand over text through a callback, from the thread they run in, and they
block that thread until the model has finished. A web request runs on an event loop,
which must never block. So each answer is written in a thread of its own, and every
thing that happens is put on a queue that the request reads: a piece of text, the
finished answer, or a failure.

The request can then wait for the first event before it decides what to send. A failure
that comes before any text becomes an ordinary HTTP error; anything after the first text
can only be sent as part of a stream that has already begun.
"""

import asyncio
import contextvars
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from handbook_rag.generate.models import Answer

# What a worker thread is called, so that it can be told apart in a thread dump.
THREAD_NAME = "handbook-rag-answer"

logger = logging.getLogger(__name__)

OnText = Callable[[str], None]
OnTrace = Callable[[str], None]
Writer = Callable[[OnText, OnTrace], Answer]
"""Writes one answer, passing its text and its trace id to the two callbacks it gets."""


@dataclass(frozen=True, slots=True)
class Text:
    """A piece of the answer as the model wrote it, with the citation markers taken out."""

    text: str


@dataclass(frozen=True, slots=True)
class Finished:
    """The whole answer, with its citations. Always the last event of a run that worked."""

    answer: Answer


@dataclass(frozen=True, slots=True)
class Failed:
    """What stopped the answer. Always the last event of a run that failed."""

    error: BaseException


Event = Text | Finished | Failed


class CallerGone(Exception):
    """Raised in the worker when nobody is listening any more, to abandon the model call."""


class Slots:
    """A fixed number of answers that may be written at the same time.

    Each answer holds a thread while the model writes, for seconds at a time. Without a
    limit, a burst of questions would start a thread for each, and all of them would
    wait on the model provider's own limit anyway.
    """

    def __init__(self, size: int) -> None:
        self._size = size
        self._used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Take a slot if one is free. Never waits: a full house is answered at once."""
        with self._lock:
            if self._used >= self._size:
                return False
            self._used += 1
            return True

    def give_back(self) -> None:
        with self._lock:
            self._used -= 1

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._used


class AnswerRun:
    """One answer being written in a worker thread, and the events it has produced so far.

    Start it with :meth:`start`, from the event loop that reads the events.
    """

    def __init__(self, write: Writer, slots: Slots, loop: asyncio.AbstractEventLoop) -> None:
        self._write = write
        self._slots = slots
        self._loop = loop
        self._events: asyncio.Queue[Event] = asyncio.Queue()
        self._stop = threading.Event()
        self.trace_id: str | None = None
        """The id of the answer's trace, once the worker knows it and if it is recorded."""
        self.first_text_at: float | None = None
        """When the first piece of text was read, by ``time.perf_counter``."""

    @classmethod
    def start(cls, write: Writer, slots: Slots) -> "AnswerRun":
        """Start writing an answer in a new thread, which holds a slot the caller has taken.

        The slot is given back when the thread ends, whatever happened in it, or here if
        the thread cannot be started.
        """
        run = cls(write, slots, asyncio.get_running_loop())
        # The worker runs in a copy of the caller's context. That copy holds the current
        # span, so the spans the worker opens belong to the caller's trace.
        context = contextvars.copy_context()
        thread = threading.Thread(
            target=context.run, args=(run._work,), name=THREAD_NAME, daemon=True
        )
        try:
            thread.start()
        except BaseException:
            slots.give_back()
            raise
        return run

    async def next(self) -> Event:
        """Wait for the next event. After a ``Finished`` or a ``Failed`` there is none."""
        event = await self._events.get()
        if isinstance(event, Text) and self.first_text_at is None:
            self.first_text_at = time.perf_counter()
        return event

    def stop(self) -> None:
        """Ask the worker to give up at its next piece of text.

        The model call is abandoned there: the callback raises, and the error leaves the
        generator the way any failure does, so it stops reading the model's reply. A
        model that is still thinking before its first words is stopped when they come.
        Once the run has ended this does nothing.
        """
        self._stop.set()

    def _work(self) -> None:
        try:
            outcome: Finished | Failed = Finished(self._write(self._on_text, self._on_trace))
        except BaseException as error:
            # Not raised here, where nobody would see it, but handed to the request,
            # which decides what the caller is told.
            outcome = Failed(error)
        # Given back before the last event is sent, so a caller who has read that event
        # finds the slot free again.
        self._slots.give_back()
        self._emit(outcome)

    def _on_text(self, text: str) -> None:
        if self._stop.is_set():
            logger.info("the caller went away; the answer is abandoned, trace %s", self.trace_id)
            raise CallerGone
        self._emit(Text(text))

    def _on_trace(self, traced: str) -> None:
        self.trace_id = traced

    def _emit(self, event: Event) -> None:
        """Put an event on the queue, from the worker thread, where only the loop may touch it."""
        try:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)
        except RuntimeError:
            # The loop has closed, as when the server shuts down. Nobody is left to read
            # the event, and the worker must still end quietly.
            logger.debug("an answer event was dropped: the event loop has closed")


def server_sent_event(name: str, data: str) -> bytes:
    """One event in the wire format of server-sent events: its name, its data, a blank line.

    The data has to be a single line. JSON is one when it is written without indentation,
    because a line break inside a string is written as ``\\n``.
    """
    if "\n" in data or "\r" in data:
        raise ValueError("The data of a server-sent event must be a single line.")
    return f"event: {name}\ndata: {data}\n\n".encode()
