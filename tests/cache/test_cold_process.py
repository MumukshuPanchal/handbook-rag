"""The first lookup of a fresh process, with the real local models.

The ``ask`` command is a new process for each question. When its cache looks the
question up, the embedding model has embedded nothing yet, and the verifier is not even
loaded. The first time a process runs a model takes several times as long as any later
time, so a lookup that counted it would often be given up, and the command would then
spend a request it need not. The cache does not count what is slow only the first time:
the rule is tested with stand-ins in ``test_answers.py``, and here with the models
themselves.

Two processes are started, as two commands would be. The first stores one answer. The
second is as cold as a command is: it builds the embedding model and the lazy verifier
as ``ask`` builds them, and looks a rewording up with a limit far below what the two
models take the first time. The answers are kept in Qdrant's embedded mode in a
directory of the test, so that no server is needed. The models are, which makes this a
``slow`` test: it loads them from disk, or downloads them the first time.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import handbook_rag

pytestmark = pytest.mark.slow

# The limit of the lookup in the second process. The real one is 2 seconds, which a cold
# process passes only now and then. This one is passed every time, so the test does not
# depend on a slow moment of the machine it runs on.
LIMIT = 0.2
COLLECTION = "answers-of-a-fresh-process"
STORED = "How many weeks of parental leave do I get in the US?"
REWORDED = "In the US, how many weeks of parental leave am I entitled to?"

STORE = f"""
import sys

from qdrant_client import QdrantClient

from handbook_rag import wiring
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.cache.store import QdrantAnswerStore
from handbook_rag.config import Settings
from handbook_rag.generate.models import Answer, Citation, Segment
from handbook_rag.ingest.models import Chunk

settings = Settings()
client = QdrantClient(path=sys.argv[1])
cache = AnswerCache(
    QdrantAnswerStore(client, {COLLECTION!r}),
    embedder=wiring.make_embedder(settings),
    threshold=settings.cache_threshold,
)
assert cache.prepare()
passage = Chunk(
    chunk_id="leave#0",
    doc_id="people/leave.md",
    title="Leave",
    url="https://handbook.example/leave/",
    section="people",
    source_commit="0" * 40,
    heading_path=("Parental leave",),
    start_char=0,
    end_char=13,
    text="Sixteen weeks.",
    token_count=3,
)
answer = Answer(
    question={STORED!r},
    segments=(Segment("Sixteen weeks.", (Citation(passage, "Sixteen weeks."),)),),
    answered=True,
    model="stand-in",
    stop_reason="end_turn",
)
cache.store(cache.lookup({STORED!r}, read=False), answer)
print(cache.lookup({STORED!r}).tier)
client.close()
"""

ASK = f"""
import json
import sys
import time

from qdrant_client import QdrantClient

from handbook_rag import wiring
from handbook_rag.cache.answers import AnswerCache
from handbook_rag.cache.store import QdrantAnswerStore
from handbook_rag.cache.verify import LazyVerifier
from handbook_rag.config import Settings

settings = Settings()
# As `ask` builds them: the embedding model on the default device, and a verifier that
# is loaded only when a stored question is near the one asked.
embedder = wiring.make_embedder(settings)
verifier = LazyVerifier(lambda: wiring.make_verifier(settings), name=settings.cache_verifier_model)
took = {{}}


def timed(owner, call):
    original = getattr(owner, call)

    def measured(*arguments, **options):
        began = time.perf_counter()
        try:
            return original(*arguments, **options)
        finally:
            took.setdefault(call, []).append(time.perf_counter() - began)

    setattr(owner, call, measured)


timed(embedder, "embed_query")
timed(verifier, "load")
timed(verifier, "score")
client = QdrantClient(path=sys.argv[1])
cache = AnswerCache(
    QdrantAnswerStore(client, {COLLECTION!r}),
    embedder=embedder,
    verifier=verifier,
    threshold=settings.cache_threshold,
    candidates=settings.cache_candidates,
    floor=settings.cache_floor,
    lookup_seconds=float(sys.argv[2]),
)
assert cache.prepare()
lookups = []
for _ in range(2):
    began = time.perf_counter()
    found = cache.lookup({REWORDED!r})
    lookups.append(
        {{"tier": found.tier, "failure": found.failure, "seconds": time.perf_counter() - began}}
    )
print(json.dumps({{"lookups": lookups, "took": took, "device": embedder.device}}))
client.close()
"""


def a_fresh_process(script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run a script in a new Python process that imports this checkout of the package."""
    environment = {**os.environ, "PYTHONPATH": str(Path(handbook_rag.__file__).parents[1])}
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def test_a_fresh_process_is_served_by_meaning_though_its_models_run_for_the_first_time(
    tmp_path: Path,
) -> None:
    place = str(tmp_path / "answers")
    stored = a_fresh_process(STORE, place)
    assert stored.returncode == 0, stored.stderr
    assert stored.stdout.splitlines()[-1] == "exact", "the answer is there for the next process"

    asked = a_fresh_process(ASK, place, str(LIMIT))

    assert asked.returncode == 0, asked.stderr
    report = json.loads(asked.stdout.splitlines()[-1])
    first, _second = report["lookups"]
    embeddings, readings = report["took"]["embed_query"], report["took"]["score"]
    print(
        f"  on {report['device']}: the first embedding {embeddings[0] * 1000:.0f} ms, loading "
        f"the verifier {report['took']['load'][0] * 1000:.0f} ms, its first reading "
        f"{readings[0] * 1000:.0f} ms; the whole first lookup {first['seconds'] * 1000:.0f} ms"
    )
    if len(embeddings) > 1 and len(readings) > 1:
        print(
            f"  the second time: embedding {embeddings[1] * 1000:.0f} ms, reading "
            f"{readings[1] * 1000:.0f} ms"
        )
    if embeddings[0] + readings[0] <= LIMIT:
        pytest.skip(
            "this machine ran both models for the first time within the limit, so the "
            "lookup was never at risk and shows nothing"
        )
    # The two first runs alone took longer than a lookup may, and the question was still
    # served its stored answer: neither is counted, and no request would have been spent.
    assert (first["tier"], first["failure"]) == ("semantic", None)
    assert first["seconds"] > LIMIT
    # Both models ran once for that lookup, and the verifier was loaded once.
    assert len(report["took"]["load"]) == 1
