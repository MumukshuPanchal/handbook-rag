"""An evaluation measures the answer model, so no evaluation is ever handed the answer cache.

A cached answer costs no request and is the reply to another run, or to another
question. Scored, it would say nothing about the model. ``tests/test_cli.py`` runs the
two evaluations that answer questions and checks that they build no cache. This file
reads the source instead, so that a path nobody ran is caught too: every call of the one
entry point is listed here, with whether it may pass a cache.
"""

import ast
from pathlib import Path

SOURCE = Path(__file__).parents[2] / "src" / "handbook_rag"
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


def calls_of_the_entry_point() -> dict[tuple[str, str], bool]:
    """Where ``answer_question`` is called, and whether a cache is passed there."""
    found: dict[tuple[str, str], bool] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in top_level_functions(tree):
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                if node.func.id != "answer_question":
                    continue
                keywords = {keyword.arg for keyword in node.keywords}
                place = (path.relative_to(SOURCE).as_posix(), function.name)
                assert place not in found, f"{place} calls the entry point twice"
                # `fresh` goes with a cache and means nothing without one.
                assert ("fresh" in keywords) <= ("cache" in keywords), place
                found[place] = "cache" in keywords
    return found


def test_only_the_command_and_the_service_pass_a_cache_to_the_entry_point() -> None:
    assert calls_of_the_entry_point() == CALLS


def test_no_evaluation_command_names_anything_that_builds_a_cache() -> None:
    """Each evaluation command, and every function of the module it reaches, however far."""
    tree = ast.parse((SOURCE / "cli.py").read_text(encoding="utf-8"))
    functions = {function.name: function for function in top_level_functions(tree)}
    evaluations = [name for name in functions if name.startswith("eval_")]
    assert len(evaluations) >= 9, "the evaluation commands are named eval_..."

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
        # The second check on two questions is measured by an evaluation. The cache is not.
        assert not {"handbook_rag.cache.answers", "handbook_rag.cache.store"} & imported, path.name
        assert not names_in(tree) & CACHE_NAMES, path.name


def test_the_retriever_an_evaluation_builds_comes_without_a_cache() -> None:
    tree = ast.parse((SOURCE / "wiring.py").read_text(encoding="utf-8"))
    (make_retriever,) = [f for f in top_level_functions(tree) if f.name == "make_retriever"]

    assert not names_in(make_retriever) & CACHE_NAMES
