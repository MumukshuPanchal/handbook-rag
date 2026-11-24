"""What an answer cache would hold after a saved answer run, by the cache's own rules.

The measurement of the cache simulates a cache that holds the golden questions. What it
holds *for* each of them decides which may be served to a question worded differently.
The first two runs took that from the golden set's label: an answer where the handbook
answers the question, a decline where it does not. A model does not follow the labels.
In the saved answer run it declined five questions the handbook answers, and to two it
does not answer it gave part of an answer and said in plain words what was missing.

This module takes what the answer model really wrote, from a saved answer run, and lets
the rules the cache itself applies decide (``cache/rules.py``): whether a reply is kept
at all, and whether it may be served by meaning. The rules are functions of a reply, so
no cache is needed here and none is built.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from handbook_rag.cache.rules import REASONS, kept_from_the_second_tier, storable
from handbook_rag.evals.answers import (
    ANSWERS_FILE,
    AnswerRecord,
    Passage,
    load_records,
    replies_to_current_labels,
)
from handbook_rag.evals.dataset import DatasetError, GoldenQuestion
from handbook_rag.generate.models import Answer, Citation, Segment
from handbook_rag.ingest.models import Chunk


@dataclass(frozen=True, slots=True)
class StoredReplies:
    """The replies of one answer run, sorted by what the cache would do with each."""

    source: str
    """The answer run they are from, as it was named to the command."""
    fingerprint: str
    """A short hash of the run's file of replies, so that a saved measurement names the
    replies it was made from."""
    held: dict[str, str | None]
    """For each golden question whose reply the cache keeps: ``None`` when the reply may
    be served to a question worded differently, and otherwise the reason it may not
    (one of ``cache.rules.REASONS``)."""
    not_kept: tuple[str, ...]
    """The questions whose reply the cache does not keep, such as one cut off at the
    length limit."""
    without_a_reply: tuple[str, ...]
    """The golden questions the run holds no reply for, or only a reply to the question
    as it was labelled before."""

    @property
    def may_be_candidates(self) -> dict[str, bool]:
        """Each kept question, and whether its reply may be served by meaning."""
        return {question_id: reason is None for question_id, reason in self.held.items()}

    def figures(self, golden: Sequence[GoldenQuestion]) -> dict[str, Any]:
        """What was held, as a saved measurement records it."""
        unanswerable = {question.id for question in golden if question.type == "unanswerable"}
        candidates = sorted(question_id for question_id, why in self.held.items() if why is None)
        return {
            "what": (
                "The simulated cache was filled with the replies of a saved answer run. "
                "Whether a reply is kept, and whether it may be served to a question worded "
                "differently, was decided by the answer cache's own rules on the reply "
                "itself, not by the golden set's label."
            ),
            "from": self.source,
            "fingerprint": self.fingerprint,
            "kept": len(self.held),
            "not_kept": list(self.not_kept),
            "without_a_reply": list(self.without_a_reply),
            "candidates": len(candidates),
            "kept_from_the_second_tier": {
                reason: sorted(
                    question_id for question_id, why in self.held.items() if why == reason
                )
                for reason in REASONS
            },
            # Where the reply and the label part ways.
            "candidates_the_handbook_does_not_answer": sorted(set(candidates) & unanswerable),
            "no_candidates_though_the_handbook_answers": sorted(
                question_id
                for question_id, why in self.held.items()
                if why is not None and question_id not in unanswerable
            ),
        }


def stored_replies(golden: Sequence[GoldenQuestion], run_dir: Path) -> StoredReplies:
    """Sort the replies of the answer run in ``run_dir`` by the cache's own rules."""
    path = run_dir / ANSWERS_FILE
    if not path.is_file():
        raise DatasetError(f"saved replies not found: {path}")
    current = replies_to_current_labels(golden, load_records(path))
    held: dict[str, str | None] = {}
    not_kept: list[str] = []
    for question in golden:
        if question.id not in current:
            continue
        answer = as_the_cache_is_handed_it(question, current[question.id])
        if storable(answer):
            held[question.id] = kept_from_the_second_tier(answer)
        else:
            not_kept.append(question.id)
    if not held:
        raise DatasetError(f"{path} holds no reply that an answer cache would keep")
    return StoredReplies(
        source=run_dir.as_posix(),
        fingerprint=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        held=held,
        not_kept=tuple(not_kept),
        without_a_reply=tuple(q.id for q in golden if q.id not in current),
    )


def as_the_cache_is_handed_it(question: GoldenQuestion, record: AnswerRecord) -> Answer:
    """A saved reply as the answer the pipeline would hand to the cache.

    An answer run sends each request once and has no fallback model, so the reply took
    one request and the answer model wrote it. The passages are named and not copied: a
    saved reply holds their ids, and the rules ask only whether a statement cites one.
    """
    passages = {passage.number: _named(passage) for passage in record.passages}
    return Answer(
        question=question.question,
        segments=tuple(
            Segment(
                statement.text,
                tuple(
                    Citation(passages[number]) for number in statement.cites if number in passages
                ),
            )
            for statement in record.statements
        ),
        answered=record.answered,
        model=record.model,
        stop_reason=record.stop_reason,
        requests=1,
    )


def _named(passage: Passage) -> Chunk:
    """A passage by its id and its page, without the text a saved reply does not hold."""
    return Chunk(
        chunk_id=passage.chunk_id,
        doc_id=passage.doc_id,
        title=passage.context_header,
        url="",
        section="",
        source_commit="",
        heading_path=(),
        start_char=0,
        end_char=0,
        text="",
        token_count=0,
    )
