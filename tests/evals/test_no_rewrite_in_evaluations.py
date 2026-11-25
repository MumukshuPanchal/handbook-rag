"""No evaluation of answers runs the rewriting step; only ``eval-follow-ups`` measures it.

An evaluation of answers measures the answer to a question as it was typed. A rewrite in
front of it would put another request, and another question, in front of what is scored.
What keeps it out is the switch an evaluation runs inside (``handbook_rag.measuring``):
while it is on, handing the step to the entry point of an answer fails. The pipeline's
tests hold that. This file reads the source, as a second net under the switch, the way
``test_no_answer_cache.py`` does for the cache: where the step is built, where it is
handed to the entry point, and that no evaluation command, nor any helper it reaches,
names any of it, except ``eval-follow-ups``, which measures the step by itself.
"""

import ast
from pathlib import Path

from tests.evals.test_no_answer_cache import (
    is_the_entry_point,
    names_in,
    names_of_the_entry_point,
    top_level_functions,
)

SOURCE = Path(__file__).parents[2] / "src" / "handbook_rag"
# What builds the rewriting step, runs it, or is it.
REWRITE_NAMES = {
    "make_rewriter",
    "make_follow_ups",
    "FollowUps",
    "standalone_question",
    "GeminiRewriter",
    "ClaudeRewriter",
}
# Every top-level function of the application that names one of them, by file. A new one
# has to be added here, which is the moment to ask whether it is an evaluation.
NAMED_IN = {
    ("api.py", "_follow_ups"),
    ("api.py", "create_app"),
    ("cli.py", "ask"),
    ("cli.py", "eval_follow_ups"),
    ("evals/follow_ups.py", "asker"),
    ("pipeline.py", "answer_question"),
    ("pipeline.py", "_searched_for"),
    ("wiring.py", "make_follow_ups"),
    ("wiring.py", "make_rewriter"),
}
# The calls of the entry point that hand it the step or earlier questions.
HANDED_THE_STEP = {("api.py", "create_app"), ("cli.py", "ask")}


def rewrite_names(tree: ast.Module) -> set[str]:
    """The names a module knows the step by: their own, and any it imported them as."""
    names = set(REWRITE_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names |= {
                alias.asname for alias in node.names if alias.name in REWRITE_NAMES and alias.asname
            }
    return names


def definitions(tree: ast.Module) -> dict[str, ast.AST]:
    """The top-level functions and classes of a module, the methods of its classes, by
    name, a method as ``Class.method``, and the code at the module's own level, as
    ``<module>``: it runs when the module is imported, for every command."""
    found: dict[str, ast.AST] = {
        "<module>": ast.Module(
            body=[
                node
                for node in tree.body
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            ],
            type_ignores=[],
        )
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found[node.name] = node
        if isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, ast.FunctionDef | ast.AsyncFunctionDef):
                    found[f"{node.name}.{member.name}"] = member
    return found


def test_the_step_is_named_only_where_the_list_says() -> None:
    named = set()
    for path in sorted(SOURCE.rglob("*.py")):
        where = path.relative_to(SOURCE).as_posix()
        if where.startswith("rewrite/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = rewrite_names(tree)
        for name, node in definitions(tree).items():
            if not isinstance(node, ast.ClassDef) and names_in(node) & names:
                named.add((where, name))

    assert named == NAMED_IN


def test_only_the_command_and_the_service_hand_the_step_to_the_entry_point() -> None:
    handed = set()
    for path in sorted(SOURCE.rglob("*.py")):
        where = path.relative_to(SOURCE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = names_of_the_entry_point(tree)
        for name, function in definitions(tree).items():
            if isinstance(function, ast.ClassDef):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and is_the_entry_point(node.func, names):
                    keywords = {keyword.arg for keyword in node.keywords}
                    if keywords & {"follow_ups", "earlier", None}:
                        handed.add((where, name))

    assert handed == HANDED_THE_STEP


def test_no_evaluation_command_but_eval_follow_ups_reaches_anything_that_rewrites() -> None:
    """Each evaluation command, and every function of the module it reaches, however far:
    eval-answers, eval-retrieval, the judges, the golden-set commands and the others."""
    tree = ast.parse((SOURCE / "cli.py").read_text(encoding="utf-8"))
    names = rewrite_names(tree)
    # Functions, and classes, whose methods a command reaches through the class's name.
    functions = {name: node for name, node in definitions(tree).items() if "." not in name}
    evaluations = [
        name
        for name in functions
        if name.startswith(("eval_", "golden_")) and name != "eval_follow_ups"
    ]
    for expected in ("eval_answers", "eval_retrieval", "eval_judge", "golden_check"):
        assert expected in evaluations, expected

    reached: set[str] = set()
    to_read = list(evaluations)
    while to_read:
        name = to_read.pop()
        if name in reached:
            continue
        reached.add(name)
        to_read += sorted(names_in(functions[name]) & set(functions))

    assert "ask" not in reached
    assert "eval_follow_ups" not in reached
    # The command line's own module-level code runs for every command, evaluations too.
    assert not names_in(definitions(tree)["<module>"]) & names
    for name in sorted(reached):
        function = functions[name]
        assert not names_in(function) & names, name
        assert not {"follow_ups", "earlier"} & {
            keyword.arg
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
        }, name
    # The one that measures the step builds its model and calls the step itself, never the
    # entry point of an answer.
    measuring_the_step = functions["eval_follow_ups"]
    assert "make_rewriter" in names_in(measuring_the_step)
    assert "answer_question" not in names_in(measuring_the_step)


def test_the_evaluation_of_follow_ups_calls_the_step_and_never_the_entry_point() -> None:
    source = (SOURCE / "evals" / "follow_ups.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert "standalone_question" in names_in(tree)
    assert "answer_question" not in names_in(tree)
    assert "make_follow_ups" not in names_in(tree)


def test_the_checks_above_find_a_rewrite_however_it_is_reached() -> None:
    """A check that found nothing in any source would pass the tests above. This one finds."""
    for source in (
        "def eval_answers():\n    make_rewriter(settings)\n",
        "def eval_answers():\n    wiring.make_follow_ups(settings)\n",
        "def eval_answers():\n    rewriter = GeminiRewriter(client, model=m)\n",
        # Under another name.
        "from handbook_rag.rewrite.base import standalone_question as sq\n"
        "def eval_answers():\n    sq(question, turns, rewriter=r)\n",
        "from handbook_rag.wiring import make_rewriter as mr\n"
        "def eval_answers():\n    mr(settings)\n",
    ):
        tree = ast.parse(source)
        (function,) = top_level_functions(tree)
        assert names_in(function) & rewrite_names(tree), source
    # In a method of a class, which a command reaches through the class.
    in_a_class = (
        "class Runner:\n    def go(self):\n        make_rewriter(settings)\n"
        "def eval_answers():\n    Runner().go()\n"
    )
    found = definitions(ast.parse(in_a_class))
    assert names_in(found["Runner.go"]) & REWRITE_NAMES
    # At the module's own level, where it runs on import.
    at_module_level = "from handbook_rag import wiring\nREWRITER = wiring.make_rewriter(s)\n"
    assert names_in(definitions(ast.parse(at_module_level))["<module>"]) & REWRITE_NAMES
    assert "Runner" in names_in(found["eval_answers"])
    assert names_in(found["Runner"]) & REWRITE_NAMES
    calls = "def f():\n    answer_question(q, earlier=e)\n    answer_question(q, **o)\n"
    tree = ast.parse(calls)
    names = names_of_the_entry_point(tree)
    keywords = [
        {keyword.arg for keyword in node.keywords}
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and is_the_entry_point(node.func, names)
    ]
    assert keywords == [{"earlier"}, {None}]
