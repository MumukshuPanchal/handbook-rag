"""An evaluation measures the answer model, so no evaluation is ever handed the answer cache.

A cached answer costs no request and is the reply to another run, or to another
question. Scored, it would say nothing about the model.

What keeps a cache out is a switch (``handbook_rag.measuring``): every command that
measures runs inside it, and while it is on, building a cache and handing one to the
entry point of an answer both fail. ``tests/test_measuring.py`` and ``tests/test_cli.py``
test the switch and that each such command runs inside it.

This file reads the source, as a second net under the switch. It lists every use of the
entry point with whether a cache is passed there, however the call is written: by the
function's own name, by another name it was imported under, as an attribute of its
module, or with arguments that are not written out. A reader of source can still be got
round, which is why it is the second net and not the first.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[2] / "src" / "handbook_rag"
ENTRY_POINT = "answer_question"
# What builds an answer cache, or is one.
CACHE_NAMES = {
    "AnswerCache",
    "QdrantAnswerStore",
    "make_answer_cache",
    "make_answer_store",
    "make_retriever_and_cache",
}
# Every call of the entry point in the application, by the file and the command or
# function it stands in, and whether it passes a cache. A new call has to be added here,
# which is the moment to ask whether it is an evaluation.
CALLS = {
    ("api.py", "create_app"): True,
    ("cli.py", "ask"): True,
    ("cli.py", "eval_answers"): False,
    ("evals/adversarial.py", "asker"): False,
}


def top_level_functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [node for node in tree.body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)]


def names_in(node: ast.AST) -> set[str]:
    """Every name and attribute a piece of code mentions."""
    found = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    return found | {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}


def names_of_the_entry_point(tree: ast.Module) -> set[str]:
    """The names a module knows the entry point by: its own, and any it was imported as."""
    names = {ENTRY_POINT}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names |= {
                alias.asname or alias.name for alias in node.names if alias.name == ENTRY_POINT
            }
    return names


def is_the_entry_point(node: ast.AST, names: set[str]) -> bool:
    """Whether an expression names the entry point, by a name or as an attribute."""
    if isinstance(node, ast.Name):
        return node.id in names
    return isinstance(node, ast.Attribute) and node.attr == ENTRY_POINT


def uses_of_the_entry_point(
    source: str, where: str
) -> tuple[dict[tuple[str, str], bool], list[tuple[str, str]]]:
    """The calls of the entry point in one module, and the places that hand it on uncalled.

    A call is found with whether it passes a cache. One whose arguments are not all
    written out (``**options``) counts as passing one: nobody can read from the source
    that it does not.
    """
    tree = ast.parse(source)
    names = names_of_the_entry_point(tree)
    calls: dict[tuple[str, str], bool] = {}
    handed_on: list[tuple[str, str]] = []
    scopes: list[tuple[str, ast.AST]] = [(f.name, f) for f in top_level_functions(tree)]
    scopes.append(
        (
            "<module>",
            ast.Module(
                body=[
                    node
                    for node in tree.body
                    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                ],
                type_ignores=[],
            ),
        )
    )
    for scope, body in scopes:
        called = set()
        for node in ast.walk(body):
            if isinstance(node, ast.Call) and is_the_entry_point(node.func, names):
                called.add(id(node.func))
                keywords = {keyword.arg for keyword in node.keywords}
                assert (where, scope) not in calls, f"{where}: {scope} calls the entry point twice"
                # `fresh` goes with a cache and means nothing without one.
                assert ("fresh" in keywords) <= ("cache" in keywords), (where, scope)
                calls[(where, scope)] = "cache" in keywords or None in keywords
        for node in ast.walk(body):
            if (
                isinstance(node, ast.Name | ast.Attribute)
                and is_the_entry_point(node, names)
                and id(node) not in called
            ):
                handed_on.append((where, scope))
    return calls, handed_on


def every_use() -> tuple[dict[tuple[str, str], bool], list[tuple[str, str]]]:
    calls: dict[tuple[str, str], bool] = {}
    handed_on: list[tuple[str, str]] = []
    for path in sorted(SOURCE.rglob("*.py")):
        found, passed = uses_of_the_entry_point(
            path.read_text(encoding="utf-8"), path.relative_to(SOURCE).as_posix()
        )
        calls |= found
        handed_on += passed
    return calls, handed_on


def test_only_the_command_and_the_service_pass_a_cache_to_the_entry_point() -> None:
    calls, handed_on = every_use()

    assert calls == CALLS
    # Nowhere is the entry point given another name or handed to other code uncalled.
    assert handed_on == []


def test_a_call_is_found_however_it_is_written() -> None:
    """A check that found nothing in any source would pass the test above. This one finds."""
    written = {
        "by its name": "def f():\n    answer_question(q, cache=c)\n",
        "under another name": (
            "from handbook_rag.pipeline import answer_question as answer\n"
            "def f():\n    answer(q, cache=c)\n"
        ),
        "as an attribute of its module": (
            "from handbook_rag import pipeline\n"
            "def f():\n    pipeline.answer_question(q, cache=c)\n"
        ),
        "with arguments that are not written out": (
            "def f():\n    answer_question(q, **options)\n"
        ),
    }
    for how, source in written.items():
        calls, handed_on = uses_of_the_entry_point(source, "example.py")
        assert calls == {("example.py", "f"): True}, how
        assert handed_on == [], how

    without, _ = uses_of_the_entry_point("def f():\n    answer_question(q, limit=1)\n", "x.py")
    assert without == {("x.py", "f"): False}
    # Given another name by assignment, or handed to other code, it is no call and is named.
    for source in (
        "def f():\n    ask = answer_question\n    ask(q, cache=c)\n",
        "def f():\n    run(answer_question, cache=c)\n",
        "ask = answer_question\n",
        "from handbook_rag import pipeline\ndef f():\n    return pipeline.answer_question\n",
    ):
        calls, handed_on = uses_of_the_entry_point(source, "example.py")
        assert (calls, len(handed_on)) == ({}, 1), source


def test_no_evaluation_command_names_anything_that_builds_a_cache() -> None:
    """Each evaluation command, and every function of the module it reaches, however far."""
    tree = ast.parse((SOURCE / "cli.py").read_text(encoding="utf-8"))
    functions = {function.name: function for function in top_level_functions(tree)}
    evaluations = [name for name in functions if name.startswith(("eval_", "golden_"))]
    assert len(evaluations) >= 13, "the commands that measure are named eval_... and golden_..."

    reached: set[str] = set()
    to_read = list(evaluations)
    while to_read:
        name = to_read.pop()
        if name in reached:
            continue
        reached.add(name)
        to_read += sorted(names_in(functions[name]) & set(functions))

    # The commands that do use the cache are not among them, so the walk means something.
    assert not {"ask", "cache_status", "cache_clear", "index"} & reached
    assert len(reached) > len(evaluations), "the commands' own helpers are read too"
    for name in sorted(reached):
        function = functions[name]
        assert not names_in(function) & CACHE_NAMES, name
        assert "cache" not in {
            keyword.arg
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
        }, name


def test_no_module_of_the_evaluation_imports_the_answer_cache() -> None:
    modules = sorted((SOURCE / "evals").glob("*.py"))

    assert len(modules) > 10
    imports_the_rules = []
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        } | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        # The rules of the cache and its second check are measured by an evaluation,
        # and neither can serve an answer. The cache itself and its store can.
        assert not {"handbook_rag.cache.answers", "handbook_rag.cache.store"} & imported, path.name
        assert not names_in(tree) & CACHE_NAMES, path.name
        if "handbook_rag.cache.rules" in imported:
            imports_the_rules.append(path.name)
    assert imports_the_rules == ["cache_replies.py"]


def test_the_rules_an_evaluation_may_apply_cannot_build_a_cache() -> None:
    # What cache_replies.py imports holds functions of a reply and nothing that stores.
    tree = ast.parse((SOURCE / "cache" / "rules.py").read_text(encoding="utf-8"))
    imported = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert not {"handbook_rag.cache.answers", "handbook_rag.cache.store"} & imported
    assert not names_in(tree) & CACHE_NAMES


def test_the_retriever_an_evaluation_builds_comes_without_a_cache() -> None:
    tree = ast.parse((SOURCE / "wiring.py").read_text(encoding="utf-8"))
    (make_retriever,) = [f for f in top_level_functions(tree) if f.name == "make_retriever"]

    assert not names_in(make_retriever) & CACHE_NAMES
