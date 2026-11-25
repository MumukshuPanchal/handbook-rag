"""The code that decides an answer, pinned together with the answer cache's version.

A cached answer is served only while the same question would still be answered the same
way. The settings that decide that are in the name of the cache's collection
(``wiring.describe_answer_cache``). Code decides it too, and no setting names code: how a
model's reply is read into statements and citations, how passages are found, and the
cache's own rules. For those there is ``ANSWER_CACHE_VERSION``, a number that a person
raises, and a person can forget.

This test fails whenever one of those modules changes, or one of the functions that
build a retriever and an answer model from the settings. That is the moment to ask, and
what to do then:

1. Ask whether an answer stored by the old code could now be wrong, or could now be
   served where the new code would not serve it.
2. If so, raise ``ANSWER_CACHE_VERSION`` in ``src/handbook_rag/cache/rules.py`` and say
   why in its list. Entries written before are then left behind.
3. Put the new hashes and the version here.

Whether the number has to go up stays a person's judgement. The test only makes sure the
question is asked. Docstrings, comments and the layout of the code are not hashed, so a
reworded explanation does not ask it.
"""

import ast
import hashlib
from collections.abc import Iterable
from pathlib import Path

import pytest

from handbook_rag.cache.rules import ANSWER_CACHE_VERSION

SOURCE = Path(__file__).parents[1] / "src" / "handbook_rag"

PINNED_VERSION = 2
# The modules between a question and its answer that no setting names, each with a hash
# of its code.
PINNED = {
    # Which replies are kept, and which may be served to a question worded differently.
    # Version 2 of the answering instructions (issue 66) added two rules: the fixed form
    # for a part the passages do not answer keeps an answer from the second tier, and a
    # reply a check in code changed is not kept. The version stays, as for generate/ below.
    "cache/rules.py": "95662f7de76bd1c3",
    # How a question is answered from the passages that were found. Follow-up questions
    # (issue 79, ADR 0024) changed it: with earlier questions and the rewriting step, the
    # question is rewritten first, and the lookup, the search, the answer and the entry
    # stored all take the question that came out, as they would take it typed. So an
    # entry is still the answer to its own question, by the same instructions and search.
    # Without earlier questions the code path is the one pinned before. A wrong rewrite
    # looks up and stores an answer to another question, which the reply names, as a
    # wrong question typed would. No stored answer becomes wrong, and none may be served
    # where it could not before: the version stays as it is. After the review, the entry
    # point leaves out an earlier question with no letter or digit, refuses a single string
    # in place of the earlier questions, and bounds how many the step reads: which earlier
    # questions reach the step, not how a question is answered. It stays.
    "pipeline.py": "415a1f1bce966f56",
    # How a model's reply is read into statements and citations, and checked.
    #
    # The answering instructions got versions (issue 66): a generator is given one, the
    # current one unless an evaluation names another, and refuses any other outside an
    # evaluation. With version 1 it sends the very text and layout it sent before, and
    # reads the reply as before.
    #
    # Version 2 became the current one: other instructions, passages wrapped in a boundary
    # made for each request, and the reply checked in code (checks.py), which can drop a
    # closing line, decline a reply that cites nothing, or mark one that is mostly a copy.
    # That changes what an answer is. The version of the cache stays all the same: the
    # current instructions are part of the collection's name (wiring.describe_answer_cache),
    # so the answers of version 2 go to a new, empty collection, and nothing stored under
    # version 1 is served under version 2. The layout is part of that name too, named with
    # a fixed code (wiring.answer_request_fingerprint): the code of each request is not.
    #
    # Then a reply under version 2 that is nothing but the fixed form for a missing part
    # became a decline by the model, shown as the fixed declining sentence (base.py), where
    # before the check for a reply that cites nothing declined it. Such a reply was never
    # stored, since a reply a check changed is not kept; now it is kept as a decline, which
    # is served to the very same question only. No stored answer becomes wrong: it stays.
    #
    # After a review, three changes to how a reply of version 2 is read and checked. A
    # reply in the fixed form that cites a passage stays an answer, where it was read as a
    # decline. A reply that is not an answer is shown as the fixed declining sentence
    # alone, with a check and its note, where it was shown as written. Every line after
    # the last citation that holds an address is dropped, where only the last line was.
    # The first stores such a reply as the answer it is, kept from the second tier as one
    # that declines in part, where it stored a decline. The other two make fewer replies
    # storable, never more: a reply a check changed is not kept.
    # What the code before could have stored under version 2 (a decline read from a cited
    # reply in the form, a decline with words beside it, an answer with an address above
    # its last line) is no more wrong now than when it was stored, and a decline is served
    # to the very same question only. Version 2 has not been released, so its collection
    # exists only where this branch was run, and `cache-clear` empties it. The time when
    # text was first passed on is recorded too, which changes no answer. It stays.
    #
    # After a second review: the stop after a digit ends a piece of the fixed form, so more
    # replies go to the check for a reply that cites nothing, and are not kept. A reply
    # whose every citation follows the fixed declining sentence is a decline, under either
    # version, and what the checks leave is decided again: such a reply is kept as a decline,
    # served to the very same question only, or not kept at all when a check changed it.
    # Before, it was kept as an answer that holds the declining sentence, which was also
    # served to the very same question only. The product no longer answers under version
    # 1 and an evaluation never has a cache, so the collection of version 1 is never read
    # again; version 2 is not released. It stays.
    "generate/base.py": "e5ed32ac306da474",
    "generate/checks.py": "3f3480b023da93b6",
    "generate/claude.py": "0a4a690b15c721b1",
    "generate/gemini.py": "52356590fc24d86b",
    "generate/markers.py": "db2add40ed28b909",
    "generate/models.py": "e2abfdf1b46b3281",
    # How passages are found: the question's vectors, the search, and what orders them.
    "index/embedder.py": "42df4fad32f4e53d",
    "index/sparse.py": "1852989448469c03",
    "index/store.py": "86a0c12a6e4089f6",
    "retrieve/dense.py": "cf0501885eedfb43",
    "retrieve/fusion.py": "b5ca10d9b6f801eb",
    "retrieve/hybrid.py": "01adfb6f4a8cdfeb",
    "retrieve/rerank.py": "82d23821dd25f851",
    "retrieve/sparse.py": "8da43720d6b1ac7d",
    # Of the search variants of issue 78, the limit on chunks of one page changed
    # hybrid.py and make_retriever and added this module. The number of candidates and the
    # fusion's constant change no code here: `eval-retrieval` sets the settings that
    # already held them, for one run. The limit is an option of `eval-retrieval` only and
    # off by default, and tests/retrieve/test_default_search.py pins the default search's
    # rankings. So no stored answer was found by another search than today's default, and
    # the version stays as it is.
    "retrieve/spread.py": "b5ebb11203841b0a",
}
# How the settings become what finds passages and what writes an answer. Of wiring.py
# only these functions are pinned, each by itself. The rest of that module connects to
# the store, builds the judges and the cache, and names what a run records: it changes
# with nearly every feature, and a change there changes no answer, or changes the name of
# the cache's collection, which starts an empty cache where everyone can see it.
PINNED_FUNCTIONS = {
    "wiring.py": {
        "make_embedder": "339a3818501b98e2",
        "make_sparse_encoder": "d5a50d71900c780a",
        "make_reranker": "218b8bc4eb480919",
        "make_retriever": "60c64124baf7cf64",
        # Pass on the version of the instructions (issue 66), as for generate/ above:
        # the service and the ask command name none and get the current one.
        "make_generator": "98556be4096d518b",
        "_model_generator": "ced800452a4d24ac",
        "_make_claude_generator": "353fa128aa1dea61",
    },
}


class _WithoutWordsForPeople(ast.NodeTransformer):
    """Drop every statement that is only a text: a docstring, or one under a name."""

    def visit_Expr(self, node: ast.Expr) -> ast.Expr | None:
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return node


def shape(node: object) -> object:
    """A piece of code as nested tuples, the same in every version of Python.

    Positions in the file are left out, and so is every field that is empty: a later
    Python that gives a node one more optional field then still sees the same code.
    """
    if isinstance(node, ast.AST):
        return (
            type(node).__name__,
            tuple(
                (name, shape(value))
                for name, value in ast.iter_fields(node)
                if value is not None and value != []
            ),
        )
    if isinstance(node, list):
        return tuple(shape(item) for item in node)
    return repr(node)


def tree_hash(code: ast.AST) -> str:
    """A short hash of a piece of parsed code, without the words that are there for people."""
    code = _WithoutWordsForPeople().visit(code)
    return hashlib.sha256(repr(shape(code)).encode("utf-8")).hexdigest()[:16]


def code_hash(source: str) -> str:
    return tree_hash(ast.parse(source))


def function_hashes(source: str, names: Iterable[str]) -> dict[str, str]:
    """The hash of each named function of a module, taken by itself."""
    functions = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    missing = sorted(set(names) - set(functions))
    assert not missing, f"pinned, and no longer a function of the module: {missing}"
    return {name: tree_hash(functions[name]) for name in names}


def test_the_modules_that_decide_an_answer_are_the_ones_the_version_was_set_for() -> None:
    now = {module: code_hash((SOURCE / module).read_text(encoding="utf-8")) for module in PINNED}
    changed = sorted(module for module in PINNED if now[module] != PINNED[module])

    assert not changed, (
        f"Code that decides an answer has changed: {', '.join(changed)}. Read the top of "
        "this file. If answers stored before could now be wrong, raise ANSWER_CACHE_VERSION "
        f"in cache/rules.py. Then pin: { {module: now[module] for module in changed} }"
    )
    assert ANSWER_CACHE_VERSION == PINNED_VERSION, (
        "The version was raised: pin it here too, with the hashes of the code it is for."
    )


def test_the_functions_that_build_a_retriever_and_an_answer_model_are_pinned_too() -> None:
    for module, pinned in PINNED_FUNCTIONS.items():
        now = function_hashes((SOURCE / module).read_text(encoding="utf-8"), pinned)
        changed = sorted(name for name in pinned if now[name] != pinned[name])

        assert not changed, (
            f"Code that decides an answer has changed in {module}: {', '.join(changed)}. Read "
            "the top of this file. If answers stored before could now be wrong, raise "
            "ANSWER_CACHE_VERSION in cache/rules.py. Then pin: "
            f"{ {name: now[name] for name in changed} }"
        )
    # The one that puts the search together from the settings is among them.
    assert "make_retriever" in PINNED_FUNCTIONS["wiring.py"]
    assert "wiring.py" not in PINNED, "the module as a whole is not pinned, and the file says why"


def test_a_function_is_pinned_by_itself_and_a_missing_one_is_named() -> None:
    module = "LIMIT = 3\n\ndef kept(x):\n    return x + LIMIT\n\ndef other(x):\n    return x\n"

    pinned = function_hashes(module, ["kept"])

    # A change beside the function does not ask the question, and a change in it does.
    assert function_hashes(module.replace("return x\n", "return -x\n"), ["kept"]) == pinned
    assert function_hashes(module.replace("LIMIT = 3", "LIMIT = 4"), ["kept"]) == pinned
    assert function_hashes(module.replace("x + LIMIT", "x - LIMIT"), ["kept"]) != pinned
    assert function_hashes(module.replace("def kept(x)", "def kept(x, y=1)"), ["kept"]) != pinned
    with pytest.raises(AssertionError, match="no longer a function of the module"):
        function_hashes(module, ["renamed"])


def test_every_module_of_the_search_and_of_reading_a_reply_is_pinned() -> None:
    # A new module in one of these packages decides an answer too, and has to be listed.
    for package, left_out in (
        ("retrieve", set()),
        # What a request costs, how it is laid out (hashed into the collection's name on
        # its own), and how it is sent again after a failure decide no answer.
        ("generate", {"pricing.py", "prompt.py", "retrying.py"}),
    ):
        modules = {path.name for path in (SOURCE / package).glob("*.py")} - {"__init__.py"}
        pinned = {module.split("/")[1] for module in PINNED if module.startswith(f"{package}/")}
        assert modules - pinned == left_out, package


def test_the_hash_is_of_the_code_and_not_of_the_words_around_it() -> None:
    plain = "def f(x):\n    return x + 1\n\nLIMIT = 3\n"
    explained = (
        '"""What this module is for."""\n\n'
        "def f(x):  # adds one\n"
        '    """Add one."""\n'
        "    return x + 1\n\n\n"
        "LIMIT = 3\n"
        '"""How many."""\n'
    )

    assert code_hash(explained) == code_hash(plain)
    assert code_hash(plain.replace("x + 1", "x + 2")) != code_hash(plain)
    assert code_hash(plain.replace("LIMIT = 3", "LIMIT = 4")) != code_hash(plain)
    # A text that is used is code.
    assert code_hash('NAME = "a"\n') != code_hash('NAME = "b"\n')
    assert len(code_hash(plain)) == 16
