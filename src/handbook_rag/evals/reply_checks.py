"""The checks of a reply in code, tried on the replies a run has saved, with no request.

The checks (``generate/checks.py``) are part of a version of the answering instructions,
and run when a reply is written. To learn what they would do to replies written before
they existed, such as those of the old instructions, they are made again here on the
saved replies, with the passages each reply was given. That costs no request, so the
checks can be tried on every saved run before a request is spent on new answers.

A saved reply keeps its statements and the numbers of the passages each cites, and names
the passages by chunk id. The text of a passage comes from the corpus. For a case of the
adversarial tests with a planted instruction, the passage it was planted in is given as
the model was given it, with the planted text, as the product would hold it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from handbook_rag.evals.answers import AnswerRecord
from handbook_rag.evals.dataset import DatasetError
from handbook_rag.generate.checks import CheckedReply, check_reply
from handbook_rag.generate.models import Citation, Segment
from handbook_rag.ingest.models import Chunk


@dataclass(frozen=True, slots=True)
class TriedReply:
    """One saved reply, and what the checks would have made of it."""

    reply_to: str
    record: AnswerRecord
    checked: CheckedReply


def given_passages(
    record: AnswerRecord,
    chunks: Mapping[str, Chunk],
    planted: Mapping[str, Sequence[Chunk]] | None = None,
) -> list[Chunk]:
    """The passages a saved reply was given, in order: those of ``planted`` for its case if
    it has any there, else read from the corpus by chunk id."""
    if planted is not None and record.question_id in planted:
        return list(planted[record.question_id])
    given: list[Chunk] = []
    for passage in record.passages:
        chunk = chunks.get(passage.chunk_id)
        if chunk is None:
            raise DatasetError(
                f"the reply to {record.question_id} was given passage {passage.chunk_id}, "
                "which is not in the corpus. The run was made with another snapshot or chunking."
            )
        given.append(chunk)
    return given


def try_checks(record: AnswerRecord, given: Sequence[Chunk]) -> TriedReply:
    """Make the checks on a saved reply as they are made on a reply being written."""
    segments = tuple(
        Segment(
            statement.text,
            tuple(Citation(given[number - 1]) for number in statement.cites),
        )
        for statement in record.statements
    )
    return TriedReply(
        record.question_id, record, check_reply(segments, given, answered=record.answered)
    )
