"""The code that decides an answer, pinned together with the answer cache's version.

A cached answer is served only while the same question would still be answered the same
way. The settings that decide that are in the name of the cache's collection
(``wiring.describe_answer_cache``). Code decides it too, and no setting names code: how a
model's reply is read into statements and citations, how passages are found, and the
cache's own rules. For those there is ``ANSWER_CACHE_VERSION``, a number that a person
raises, and a person can forget.

This test fails whenever one of those modules changes. That is the moment to ask, and
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
from pathlib import Path

from handbook_rag.cache.rules import ANSWER_CACHE_VERSION

SOURCE = Path(__file__).parents[1] / "src" / "handbook_rag"

PINNED_VERSION = 2
# The modules between a question and its answer that no setting names, each with a hash
# of its code.
PINNED = {
    # Which replies are kept, and which may be served to a question worded differently.
    "cache/rules.py": "87fee71e1a0e98e3",
    # How a question is answered from the passages that were found.
    "pipeline.py": "bdd6ba2ef69fcad4",
    # How a model's reply is read into statements and citations.
    "generate/base.py": "0408beccbd6450f1",
    "generate/claude.py": "f370a5e4dd209266",
    "generate/gemini.py": "e62ce63965091596",
    "generate/markers.py": "915fe465e97fc690",
    "generate/models.py": "2a9bfbd98d0828a1",
    # How passages are found: the question's vectors, the search, and what orders them.
    "index/embedder.py": "42df4fad32f4e53d",
    "index/sparse.py": "1852989448469c03",
    "index/store.py": "86a0c12a6e4089f6",
    "retrieve/dense.py": "cf0501885eedfb43",
    "retrieve/fusion.py": "b5ca10d9b6f801eb",
    "retrieve/hybrid.py": "88568557c1520ab8",
    "retrieve/rerank.py": "82d23821dd25f851",
    "retrieve/sparse.py": "8da43720d6b1ac7d",
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


def code_hash(source: str) -> str:
    code = _WithoutWordsForPeople().visit(ast.parse(source))
    return hashlib.sha256(repr(shape(code)).encode("utf-8")).hexdigest()[:16]


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
