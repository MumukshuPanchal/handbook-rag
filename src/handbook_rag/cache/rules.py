"""The rules of the answer cache that need no store and no model.

What makes two questions the same question to the exact tier, which replies are kept,
and which of those may be served to a question worded differently. They stand apart
from the cache itself (``cache/answers.py``), so that the measurement of the cache can
apply the very same rules to saved replies (``evals/cache_replies.py``) without being
able to build a cache.
"""

import math
import unicodedata
import uuid
from typing import Final

from handbook_rag.cache.verify import Verification
from handbook_rag.generate.base import holds_the_decline, holds_the_gap_form, plain_words
from handbook_rag.generate.models import Answer

ANSWER_CACHE_VERSION = 2
"""Part of the name of every collection of cached answers (``wiring.py``). Raise it when
the code changes what an answer to a question is, or which answers are kept or may be
served by meaning, in a way no setting shows: how a model's reply is read into
statements and citations, how passages are found, or the rules in this module. Entries
written before are then left behind under their old name.

A test pins a hash of the modules that decide an answer together with this number
(``tests/test_answer_cache_version.py``). It fails when one of them changes, which is
the moment to ask whether the number has to go up. Whether it does is still a person's
judgement.

1. The first cache.
2. An answer that says in plain words that something is not in the handbook is no
   candidate of the second tier (:func:`says_something_is_missing`). Entries written
   before had no such mark.

Not raised for version 2 of the answering instructions (issue 66). Two rules changed with
it: the fixed form for a part of the question the passages do not answer keeps an answer
from the second tier as a decline in part does, and a reply that a check in code changed
is not kept at all. Neither can make an entry written before wrong, or servable where it
was not: the instructions of the current version are part of the collection's name
(``wiring.describe_answer_cache``), so the cache of version 2 starts empty, and the
entries of version 1 are never served by it.
"""

DECIDED_BY: Final = "entails_min"
"""The verifier's weaker direction decides, by its name in the measurement
(``evals/cache_pairs.py``). The measurement's rule chose it over the mean of the two
directions, and a test holds this to the saved run the default threshold comes from."""

# Why a stored reply is kept from the second tier, in the order they are asked.
DECLINE: Final = "decline"
"""The reply is not an answer: the model declined."""
DECLINES_IN_PART: Final = "declines_in_part"
"""An answer that holds the fixed sentence of a decline, or the fixed form for a part the
passages do not answer (``prompt.GAP_FORM``), for part of the question."""
SAYS_SOMETHING_IS_MISSING: Final = "says_something_is_missing"
"""An answer that says in words of its own that something is not in the handbook."""
CITES_NOTHING: Final = "cites_nothing"
"""An answer that cites no passage."""
REASONS: Final = (DECLINE, DECLINES_IN_PART, SAYS_SOMETHING_IS_MISSING, CITES_NOTHING)

# What a source is said not to do, and what is said not to have been done with a fact.
_NOT = ("does not", "do not", "did not", "doesn t", "don t", "didn t")
_NOT_BEING = ("not", "isn t", "aren t", "wasn t", "weren t")
# A word a model may put between the two: "does not explicitly state".
_HOW = ("", "explicitly ", "specifically ", "directly ", "clearly ")
_SAY = (
    "cover",
    "specify",
    "say",
    "state",
    "mention",
    "provide",
    "address",
    "list",
    "detail",
    "describe",
    "explain",
    "indicate",
    "define",
    # The instructions' own word: "If the search results do not contain the answer".
    "contain",
    "include",
    "give",
    "outline",
    "discuss",
)
_SAID = (
    "covered",
    "specified",
    "said",
    "stated",
    "mentioned",
    "provided",
    "addressed",
    "listed",
    "detailed",
    "described",
    "explained",
    "indicated",
    "defined",
    "contained",
    "included",
    "given",
    "outlined",
    "discussed",
    "found",
    "available",
    "clear",
)
# What there is said to be none of, and a word a model may put before it.
_NO = ("", "specific ", "further ", "additional ", "explicit ")
_LACKING = ("information", "mention", "detail", "details")
MISSING_PHRASES: Final = (
    *(f"{negation} {how}{verb}" for negation in _NOT for how in _HOW for verb in _SAY),
    *(f"{negation} {how}{state}" for negation in _NOT_BEING for how in _HOW for state in _SAID),
    *(f"no {kind}{what}" for kind in _NO for what in _LACKING),
    "not in the handbook",
    "not in the search results",
    "could not find",
    "cannot find",
    "can not find",
    "couldn t find",
    "can t find",
    "unable to find",
)
"""The phrases by which a reply is taken to say that something is not in the handbook,
as words in lower case with nothing between them but single spaces. An apostrophe is no
word, so "doesn't" is written "doesn t" here. See :func:`says_something_is_missing`.

They are the four phrases the model used in the saved answer run ("does not cover", "do
not cover", "do not specify", "not provided") and their relatives: other verbs of
saying, the instructions' own "do not contain", the shortened forms, and each with one
of a few adverbs in between. On the 165 replies of that run and the 17 answers of the
saved adversarial run, the relatives find nothing that the four do not."""

# Question marks, exclamation marks, full stops and the spaces between them, at the end.
_END_MARKS = "?!. "
_ID_NAMESPACE = uuid.UUID("b3a1c2d4-6e5f-4a7b-8c9d-0e1f2a3b4c5d")
_ALL = "all"


def comparable(question: str) -> str:
    """A question as the exact tier compares it.

    The rule, in this order:

    1. Unicode's composed form (NFC), so that a letter typed as one character and the
       same letter typed with a separate accent are one letter.
    2. Every run of white space becomes one space, and none is left at either end.
    3. Case does not count (``str.casefold``).
    4. Question marks, exclamation marks, full stops and spaces at the end are dropped.

    Nothing else is changed. Punctuation inside the question stays, and so does any
    other sign at its end: "What is C#?" and "What is C?" are two questions. Case can
    carry meaning, as in "US" and "us", and the exact tier does not see it.

    A question with nothing left, such as "?", has no comparable form: the empty text.
    """
    text = " ".join(unicodedata.normalize("NFC", question).split())
    # Folding the case of some letters gives a letter and a separate accent again.
    return unicodedata.normalize("NFC", text.casefold()).rstrip(_END_MARKS)


def scope_of(section: str | None) -> str:
    """Which part of the handbook a question was asked of, as an entry records it.

    An answer is written from the passages a search finds, and a search of one section
    finds others than a search of all of it. So the part is in an entry's key, and the
    tier by meaning looks among entries of the same part only.
    """
    return _ALL if section is None else f"section:{section}"


def entry_id(key: str, scope: str) -> str:
    """The id an answer to this question, asked of this part, is stored under."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{scope}\n{key}"))


def storable(answer: Answer) -> bool:
    """Whether a reply is kept: only a whole reply written by the answer model.

    Not kept:

    - A reply the fallback model wrote. Its answers are not evaluated (ADR 0018), and a
      stored one would be served long after the answer model is back.
    - A reply that did not come to a proper end: cut off at the length limit, withheld
      by the provider, or stopped for another reason.
    - A decline made without asking a model, because the search found nothing. That
      happens while an index is empty or half built, and it must not outlive that.
    - A reply with no text.
    - An answer that came from the cache itself.
    - A reply that a check in code changed or marked (``generate/checks.py``). The model
      wrote it from passages that may hold planted text, and a new answer to the same
      question is checked again, where a stored one would be served as it was.

    A decline the model wrote is a whole reply, and is kept. A failure raises before
    there is a reply, so nothing of it ever arrives here.
    """
    return (
        answer.cached is None
        and not answer.checks
        and not answer.fallback
        and answer.requests >= 1
        and answer.stop_reason == "end_turn"
        and bool(answer.text.strip())
    )


def says_something_is_missing(answer: Answer) -> bool:
    """Whether a reply says, in words of its own, that something is not in the handbook.

    The instructions ask the model for one fixed sentence when the passages hold no
    answer at all (:func:`handbook_rag.generate.base.holds_the_decline` finds that one).
    When they hold part of an answer, the instructions ask it to answer that part and to
    "say plainly what is not covered", in whatever words it chooses. Such a reply is a
    decline in part, and the part it could not find may be the very thing a neighbouring
    question asks. In the saved answer run the model wrote "the specific monetary amount
    allocated is not provided in the handbook" to one question. A neighbouring question
    asks for something else, and the verifier gave the two questions 0.989.

    There is no fixed form to look for, so this can only be a list of phrases
    (:data:`MISSING_PHRASES`): the ones the model used in the saved replies, and their
    relatives. Two things follow, and both are meant to be known:

    - **It errs towards keeping an answer out.** "Dental care is not covered" is a plain
      answer about a benefit, and it matches. It is then served to the very same
      question only, which costs a saving and nothing else.
    - **A reply can get past it.** A model that words the gap another way, "the
      handbook is silent on the amount", is not recognised, and its answer is a
      candidate like any other.

    Version 2 of the answering instructions gives the model one fixed form for this
    (``prompt.GAP_FORM``), which :func:`kept_from_the_second_tier` finds by its words
    alone, before this list is asked. The list stays for the answers that do not use the
    form: those of version 1, and a model that words the gap in its own way all the same.
    """
    said = f" {plain_words(answer.text)} "
    return any(f" {phrase} " in said for phrase in MISSING_PHRASES)


def kept_from_the_second_tier(answer: Answer) -> str | None:
    """Why a stored reply is served to the very same question only, or ``None`` if it may
    be served to a question worded differently. One of :data:`REASONS`, the first that
    holds.

    - **A decline.** The two near-misses the verifier scored highest in the first
      measurement were both a stored decline served to a question the handbook answers
      (``docs/evaluation.md``). The reader is then told the handbook lacks something it
      holds, and nothing in the reply shows it.
    - **An answer that holds the fixed sentence of a decline**, or the fixed form for a
      part the passages do not answer, for part of the question. No reply of the saved
      answer run holds either.
    - **An answer that says in plain words that something is missing**
      (:func:`says_something_is_missing`): what the model writes when part is missing.
    - **An answer that cites no passage.** It rests on nothing a reader can check, which
      is reason enough not to hand it to another question.

    Only what these four recognise is kept out. An answer that declines in part in words
    none of them knows is a candidate.
    """
    if not answer.answered:
        return DECLINE
    if holds_the_decline(answer.segments) or holds_the_gap_form(answer.segments):
        return DECLINES_IN_PART
    if says_something_is_missing(answer):
        return SAYS_SOMETHING_IS_MISSING
    if not answer.sources:
        return CITES_NOTHING
    return None


def servable_by_meaning(answer: Answer) -> bool:
    """Whether a stored reply may be served to a question worded differently.

    Not when one of the reasons of :func:`kept_from_the_second_tier` holds. Such a reply
    is still served to the very same question, by the exact tier.
    """
    return kept_from_the_second_tier(answer) is None


def deciding_score(entails: Verification) -> float:
    """The one number the threshold is compared with: the weaker of the two directions.

    It is high only when each question implies the other. :data:`DECIDED_BY` is its name
    in the measurement.

    A direction that is not a number makes the whole score not a number, which reaches
    no threshold. Left to ``min``, such a direction would be passed over without a word,
    and the other one would decide alone.
    """
    if math.isnan(entails.forward) or math.isnan(entails.backward):
        return math.nan
    return entails.smaller
