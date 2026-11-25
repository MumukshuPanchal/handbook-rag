"""The demo page: served by the API, switched off by a setting, and careful with what it shows.

The page's script cannot run here. The project has no JavaScript toolchain, on purpose
(ADR 0016), so its behaviour in a browser is checked by hand. What can be checked without
a browser is checked on the files as served and as written: the headers, the content
security policy, the page's markup, and the script's source, which must hold no name
that would read text from the API as HTML or as code or take the reader to another page,
and must make links in one place only.
"""

import json
import re
from collections.abc import Callable, Iterator
from html.parser import HTMLParser
from importlib.resources import files
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from handbook_rag import demo, wiring
from handbook_rag.api import (
    MAX_EARLIER_QUESTIONS,
    MAX_QUERY_CHARS,
    AskResponse,
    CachedAnswer,
    CheckNote,
    SearchedQuestion,
    create_app,
)
from handbook_rag.config import Settings
from handbook_rag.demo import CONTENT_SECURITY_POLICY, FILES
from handbook_rag.index.indexer import index_chunks
from tests.support import FakeEmbedder, FakeSparseEncoder, make_chunk

GOLDEN = Path(__file__).parents[1] / "evals" / "golden" / "questions.jsonl"
# The page's header leads to the README's sections on how the system works and how it is
# measured, in the project's own repository.
REPOSITORY = "https://github.com/MumukshuPanchal/handbook-rag"
README_LINKS = [f"{REPOSITORY}#how-it-works", f"{REPOSITORY}#results"]
# The example questions on the page: answerable ones, and one the handbook does not answer.
EXAMPLES = 4


def application(settings: Settings | None = None) -> FastAPI:
    """The API over a one-chunk index with stand-in models, and no answer model."""
    settings = settings or Settings()
    embedder, sparse_encoder = FakeEmbedder(), FakeSparseEncoder()
    store = wiring.make_memory_store("demo")
    chunks = [make_chunk("New parents get sixteen weeks of paid leave.")]
    index_chunks(chunks, embedder=embedder, sparse_encoder=sparse_encoder, store=store)
    retriever = wiring.make_retriever(
        settings, store=store, embedder=embedder, sparse_encoder=sparse_encoder
    )
    return create_app(settings, retriever=retriever, store=store)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(application()) as test_client:
        yield test_client


def static(name: str) -> str:
    """A file of the page, as the installed package holds it."""
    return files("handbook_rag").joinpath("static", name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("path", "name", "content_type"),
    [
        ("/", "demo.html", "text/html; charset=utf-8"),
        ("/static/demo.js", "demo.js", "text/javascript; charset=utf-8"),
        ("/static/demo.css", "demo.css", "text/css; charset=utf-8"),
    ],
)
def test_the_page_and_its_files_are_served_with_their_type_and_the_security_headers(
    client: TestClient, path: str, name: str, content_type: str
) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"] == content_type
    assert response.text == static(name)
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-cache"


def test_the_page_is_on_by_default_and_the_setting_switches_it_and_only_it_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert Settings().api_demo is True

    monkeypatch.setenv("HANDBOOK_RAG_API_DEMO", "false")
    settings = Settings()
    with TestClient(application(settings)) as client:
        statuses = {path: client.get(path).status_code for path in FILES}
        search = client.post("/search", json={"query": "paid leave"})

    assert settings.api_demo is False
    assert statuses == {"/": 404, "/static/demo.js": 404, "/static/demo.css": 404}
    assert search.status_code == 200, "the rest of the API is still there"


def test_the_page_and_its_files_are_not_part_of_the_apis_documentation(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) == {"/health", "/search", "/ask"}
    assert client.get("/").status_code == 200, "the page is served all the same"


def test_the_page_and_its_files_take_no_other_method_than_get(client: TestClient) -> None:
    for path in FILES:
        for method in ("POST", "PUT", "DELETE"):
            assert client.request(method, path).status_code == 405, f"{method} {path}"


@pytest.mark.parametrize(("demo_on", "status"), [(True, {"GET": 200, "POST": 405}), (False, {})])
def test_no_request_to_the_pages_addresses_is_traced_while_a_search_is(
    monkeypatch: pytest.MonkeyPatch, demo_on: bool, status: dict[str, int]
) -> None:
    """A page load is not a question. The question asked on the page is traced as usual.

    The address alone decides: no request to the page's three addresses is traced, whatever
    its method, and also while the page is switched off and they answer 404.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: provider)
    monkeypatch.setattr("handbook_rag.tracing._provider", provider)

    with TestClient(application(Settings(api_demo=demo_on))) as client:
        exporter.clear()  # drop the trace of the search that warmed the service up
        for path in FILES:
            for method in ("GET", "POST"):
                assert client.request(method, path).status_code == status.get(method, 404)
        after_the_page = list(exporter.get_finished_spans())
        client.post("/search", json={"query": "paid leave"})
        after_a_search = exporter.get_finished_spans()

    assert after_the_page == []
    assert "POST /search" in {span.name for span in after_a_search}, "tracing was on"


def test_the_files_are_read_from_the_package_and_not_from_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The package's resources are swapped for a folder, and the page serves what it holds.

    Tests also run in an empty working directory (conftest.py), and the files are served
    there too: the test above compares them with the package's own.
    """
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "static").mkdir(parents=True)
    for name in ("demo.html", "demo.js", "demo.css"):
        (elsewhere / "static" / name).write_text(f"stand-in {name}", encoding="utf-8")
    asked: list[str] = []

    def stand_in(package: str) -> Path:
        asked.append(package)
        return elsewhere

    monkeypatch.setattr(demo, "files", stand_in)
    with TestClient(application()) as client:
        served = {path: client.get(path).text for path in FILES}

    assert asked == ["handbook_rag"]
    assert served == {
        "/": "stand-in demo.html",
        "/static/demo.js": "stand-in demo.js",
        "/static/demo.css": "stand-in demo.css",
    }


# The content security policy.


def directives(policy: str) -> dict[str, list[str]]:
    """A policy as each of its directives and the sources it allows."""
    parsed: dict[str, list[str]] = {}
    for directive in policy.split(";"):
        name, *sources = directive.split()
        assert name not in parsed, f"{name} is given twice"
        parsed[name] = sources
    return parsed


def test_the_policy_allows_nothing_inline_nothing_from_elsewhere_and_no_framing() -> None:
    policy = directives(CONTENT_SECURITY_POLICY)
    sources = {
        source
        for name, allowed in policy.items()
        if name != "require-trusted-types-for"
        for source in allowed
    }

    assert policy["default-src"] == ["'none'"]
    assert policy["frame-ancestors"] == ["'none'"]
    assert policy["form-action"] == ["'none'"]
    assert policy["base-uri"] == ["'none'"]
    # This server is allowed for what the page needs, and for nothing else.
    assert {name for name, allowed in policy.items() if "'self'" in allowed} == {
        "script-src",
        "style-src",
        "connect-src",
    }
    assert policy["script-src"] == policy["style-src"] == policy["connect-src"] == ["'self'"]
    # No other site, and none of the words that would let inline or generated code run.
    assert sources <= {"'self'", "'none'"}
    for loose in ("unsafe-inline", "unsafe-eval", "unsafe-hashes", "nonce-", "sha256-", "*"):
        assert loose not in CONTENT_SECURITY_POLICY
    assert policy["require-trusted-types-for"] == ["'script'"]


# The page's markup.


class Markup(HTMLParser):
    """The elements of a page with their attributes, and the text inside a few of them."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self.inline_script = ""
        self.examples: list[str] = []
        self.example_attributes: list[dict[str, str | None]] = []
        self.paragraphs: dict[str, str] = {}
        self._inside: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.elements.append((tag, attributes))
        if tag == "script":
            self._inside = "script"
        elif tag == "button" and "example" in (attributes.get("class") or "").split():
            self._inside = "example"
            self.examples.append("")
            self.example_attributes.append(attributes)
        elif tag == "p" and attributes.get("id"):
            self._inside = f"p#{attributes['id']}"
            self.paragraphs[str(attributes["id"])] = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "button", "p"):
            self._inside = None

    def handle_data(self, data: str) -> None:
        if self._inside == "script":
            self.inline_script += data
        elif self._inside == "example":
            self.examples[-1] += data
        elif self._inside is not None and self._inside.startswith("p#"):
            self.paragraphs[self._inside.removeprefix("p#")] += data


def page() -> Markup:
    markup = Markup()
    markup.feed(static("demo.html"))
    markup.close()
    return markup


def test_the_page_has_no_inline_script_or_style_and_loads_only_its_two_files() -> None:
    markup = page()
    tags = [tag for tag, _ in markup.elements]

    assert [attrs.get("src") for tag, attrs in markup.elements if tag == "script"] == [
        "/static/demo.js"
    ]
    assert markup.inline_script.strip() == "", "a script element holds no code of its own"
    assert [attrs.get("href") for tag, attrs in markup.elements if tag == "link"] == [
        "/static/demo.css"
    ]
    assert "style" not in tags
    for tag, attributes in markup.elements:
        assert "style" not in attributes, tag
        assert [name for name in attributes if name.startswith("on")] == [], tag
    # Nothing else loads anything, or sends a form anywhere. The icons are drawn in the
    # page's own markup, so the parts of an inline SVG that can fetch a file are refused too.
    loaders = {"img", "iframe", "frame", "object", "embed", "base", "use", "image", "feimage"}
    assert not (loaders | {"foreignobject"}) & set(tags)
    loading = {"src", "srcset", "data", "poster", "action", "formaction", "background"}
    for tag, attributes in markup.elements:
        if tag != "script":
            assert not loading & set(attributes), tag
        if tag not in ("a", "link"):
            assert not {"href", "xlink:href"} & set(attributes), tag
    assert [attrs for tag, attrs in markup.elements if attrs.get("http-equiv")] == []
    # The page's own links: two to the project's README, which leave the page in a new tab
    # and tell the page they lead to nothing about it, and one to the API's documentation,
    # on this server. No other.
    links = [attrs for tag, attrs in markup.elements if tag == "a"]
    assert [attrs.get("href") for attrs in links] == [*README_LINKS, "/docs"]
    for attrs in links[: len(README_LINKS)]:
        assert attrs.get("target") == "_blank"
        assert attrs.get("rel") == "noopener noreferrer"


def test_the_example_questions_are_real_questions_of_the_golden_set() -> None:
    """Each example is a question of the golden set, word for word, of the type it shows.

    The one marked as the decline is a question the handbook does not answer, and the page
    says so next to it, so that a visitor sees a decline on purpose. Every other example
    is answerable.
    """
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    golden = {row["question"]: row for row in rows}
    markup = page()
    examples = [text.strip() for text in markup.examples]
    marked = [
        "example-decline" in (attrs.get("class") or "").split()
        for attrs in markup.example_attributes
    ]

    assert len(examples) == len(set(examples)) == EXAMPLES
    assert marked.count(True) == 1, "one example shows a decline"
    for example, decline in zip(examples, marked, strict=True):
        assert example in golden, example
        assert golden[example]["type"] == ("unanswerable" if decline else "answerable"), example
    (decline_attributes,) = [
        attrs for attrs, decline in zip(markup.example_attributes, marked, strict=True) if decline
    ]
    note = str(decline_attributes.get("aria-describedby"))
    assert "not in the handbook" in markup.paragraphs[note]


def test_the_links_in_the_header_lead_to_sections_the_readme_has() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in readme.splitlines()
        if line.startswith("## ")
    }

    for link in README_LINKS:
        assert link.removeprefix(f"{REPOSITORY}#") in headings, link


def test_the_page_tells_the_reader_the_length_limit_before_they_reach_it() -> None:
    markup = page()
    (box,) = [attrs for tag, attrs in markup.elements if tag == "textarea"]

    assert box["maxlength"] == str(MAX_QUERY_CHARS)
    assert f"{MAX_QUERY_CHARS} characters" in markup.paragraphs["question-hint"]
    assert "question-hint" in str(box["aria-describedby"]).split()


def test_the_hint_tells_the_keys_that_ask_and_the_script_listens_for_them() -> None:
    """Enter asks, as in any chat box; Shift+Enter starts a new line; Ctrl+Enter and
    Cmd+Enter ask too. An Enter that ends a composition with an input method asks nothing.

    What each key does, and that an empty question, one over the limit, or one asked while
    an answer is being written is not sent, is run in tests/test_demo_behaviour.py.
    """
    markup = page()
    (box,) = [attrs for tag, attrs in markup.elements if tag == "textarea"]
    keys = body_of("isAskKey")

    assert "Enter asks" in markup.paragraphs["question-hint"]
    assert "Shift+Enter starts a new line" in markup.paragraphs["question-hint"]
    assert box["aria-keyshortcuts"] == "Enter Control+Enter Meta+Enter"
    assert 'event.key === "Enter"' in keys
    assert "!event.shiftKey" in keys
    assert "!event.isComposing" in keys
    assert "event.keyCode !== 229" in keys
    assert 'page.question.addEventListener("keydown"' in static("demo.js")


def test_the_stop_button_is_hidden_until_an_answer_is_being_written() -> None:
    by_id = {attrs["id"]: attrs for _, attrs in page().elements if attrs.get("id")}

    assert "hidden" in by_id["stop"], "hidden before the script runs, too"
    assert "page.stop.hidden = !busy;" in body_of("setBusy")


def test_the_steps_beside_the_welcome_and_folded_on_a_phone_are_the_same() -> None:
    html = static("demo.html")
    steps = re.findall(r'<ol class="steps">(.*?)</ol>', html, re.DOTALL)

    assert len(steps) == 2
    assert steps[0] == steps[1]
    assert "The handbook is searched" in steps[0].split("</li>")[0], "the search comes first"
    assert '<details class="how-fold">' in html


def test_the_question_has_a_label_and_the_state_of_the_request_is_announced() -> None:
    markup = page()
    by_id = {attrs["id"]: (tag, attrs) for tag, attrs in markup.elements if attrs.get("id")}
    labelled = [attrs.get("for") for tag, attrs in markup.elements if tag == "label"]

    assert [attrs.get("lang") for tag, attrs in markup.elements if tag == "html"] == ["en"]
    assert by_id["question"][0] == "textarea"
    assert labelled == ["question"]
    assert by_id["status"][1].get("role") == "status", "the state of the request"
    assert by_id["failure"][1].get("role") == "alert", "a failure, in words"
    # These two are the page's only regions that announce. The streamed text, which the
    # script puts in each turn, would be read piece by piece: the status line says that
    # the model is writing, and when the answer is complete. So no element of the page is
    # a live region, and the script makes none, by attribute or by role.
    announcing = {"status", "alert", "log", "marquee", "timer"}
    assert {
        str(attrs.get("id")) for _, attrs in markup.elements if attrs.get("role") in announcing
    } == {"status", "failure"}
    assert [attrs["id"] for _, attrs in markup.elements if attrs.get("aria-live")] == []
    assert "output" not in [tag for tag, _ in markup.elements], "an output element announces"
    # The script makes none either: no live attribute or property, no role, and no output
    # element, which the list of the elements it may make leaves out
    # (test_the_check_of_the_script_finds_each_kind_of_call_planted_in_it plants each).
    problems = problems_in(static("demo.js"))
    assert "a live region made by the script" not in problems
    assert "an element the page does not make" not in problems
    assert "output" not in ELEMENTS_MADE


def test_every_element_the_script_looks_up_is_on_the_page() -> None:
    markup = page()
    ids = {attrs["id"] for _, attrs in markup.elements if attrs.get("id")}
    looked_up = set(re.findall(r'getElementById\("([\w-]+)"\)', static("demo.js")))

    assert len(looked_up) > 10, "the script looks its elements up by id"
    assert looked_up <= ids, f"not on the page: {sorted(looked_up - ids)}"
    assert len(markup.examples) == EXAMPLES, 'the script looks for "button.example"'


# The script's source. It is read whole, comments and strings included: a line that looks
# like a comment can sit inside a template literal, where the ${...} on it runs. So the
# script's own comments do not name what it must not hold.


def function_body(source: str, name: str) -> range:
    """Where a top-level function is in the code: its declaration to its closing brace."""
    start = source.index(f"function {name}(")
    return range(start, source.index("\n}\n", start) + 2)


# What the script must never hold, by name, wherever the name stands. Each would let text
# from the API, or a string built from it, be read as markup, run as code, or take the
# reader to another page. The script has no reason to do any of these.
FORBIDDEN = {
    "innerHTML": r"\binnerHTML\b",
    "outerHTML": r"\bouterHTML\b",
    "insertAdjacentHTML": r"\binsertAdjacentHTML\b",
    "write or writeln": r"\bwrite(?:ln)?\b",
    "eval": r"\beval\b",
    "Function": r"\bFunction\b",
    "a constructor reached by name": r"\bconstructor\b",
    "a timer, which can run a string": r"\bset(?:Timeout|Interval)\b",
    "an inline event handler": r"\.on[a-z]+\s*=(?!=)|setAttribute\s*\(\s*[\"'`]on",
    "HTML parsed from a string": (
        r"\b(?:DOMParser|createContextualFragment|setHTML(?:Unsafe)?|parseHTMLUnsafe|srcdoc)\b"
    ),
    "an element made in a namespace": r"\bcreateElementNS\b",
    "a javascript: address": r"(?i)javascript:",
    "location": r"\blocation\b",
    "open": r"\bopen\b",
    "the Navigation API": r"\bnavigat(?:e|ion)\b",
    "a refresh by a meta element": r"(?i)\bhttp-?equiv\b",
    "the history": r"\bhistory\b",
    "an attribute made as a node": (
        r"\b(?:attributes|createAttribute|setNamedItem|setAttributeNode)\b"
    ),
    # A conversation is kept in the page's memory only (ADR 0024).
    "storage in the browser": r"\b(?:localStorage|sessionStorage|indexedDB|cookie|caches)\b",
}


# The elements the script may make, by name. The rules on the page's markup bind the HTML;
# the script builds most of the page, so these bind what it builds. Nothing here loads
# anything, runs anything or announces anything: no img, iframe, script, style, link,
# object, embed, form, output or svg. A link is made by sourceLink alone, below.
ELEMENTS_MADE = {
    "article", "button", "code", "dd", "details", "div", "dl", "dt", "h3", "h4",
    "li", "ol", "p", "span", "summary", "ul",
}  # fmt: skip


def elements_made_wrongly(source: str) -> bool:
    """Whether the script makes an element by a name not in the list, or by a name it
    does not write out: every call of make names its element in a string."""
    calls = re.findall(r"\bmake\(\s*([^,)]*)", source)
    named = [re.fullmatch(r"([\"'`])([a-z0-9]+)\1", call.strip()) for call in calls]
    # The one call that is no call: make's own declaration, `function make(tag, ...)`.
    unnamed = [call for call, name in zip(calls, named, strict=True) if name is None]
    if unnamed != ["tag"]:
        return True
    literal = re.findall(r"createElement\s*\(\s*([\"'`])([A-Za-z0-9]+)\1", source)
    names = [name.group(2) for name in named if name is not None] + [n for _, n in literal]
    return any(name.lower() not in ELEMENTS_MADE | {"a"} for name in names)


def problems_in(source: str) -> list[str]:
    """What a script holds that the demo page's script must not."""
    found = [name for name, pattern in FORBIDDEN.items() if re.search(pattern, source)]
    link, make = function_body(source, "sourceLink"), function_body(source, "make")
    # The streamed text is not announced: the script makes no live region, by attribute,
    # by property or by role. An output element is one too, and is not in the list.
    if re.search(r"(?i)\baria-?live\b|\.role\b|\[\s*[\"'`]role[\"'`]\s*\]", source):
        found.append("a live region made by the script")
    if elements_made_wrongly(source):
        found.append("an element the page does not make")
    # make is only called, by its own name: never kept in another name or called through
    # call or apply, where the list above could not read what it is asked for.
    if re.search(r"\bmake\b(?!\s*\()", source):
        found.append("make reached under another name")
    # Nothing set in bulk, where the name of what is set is not written beside it.
    if re.search(r"\bObject\s*\.\s*assign\b", source):
        found.append("properties set in bulk")
    # No style written by the script, in a style map or in a style sheet.
    if re.search(
        r"\b(?:attributeStyleMap|insertRule|adoptedStyleSheets|styleSheets|CSSStyleSheet)\b", source
    ):
        found.append("a style written by the script")
    # Nothing that loads or styles: the policy would refuse it, and the guard says so too.
    if re.search(
        r"\.(?:src|srcset|style)\b|\[\s*[\"'`](?:src|srcset|style|innerHTML|outerHTML)[\"'`]\s*\]",
        source,
    ):
        found.append("a source or a style set as a property")

    def outside(pattern: str, allowed: range) -> bool:
        return any(match.start() not in allowed for match in re.finditer(pattern, source))

    # An element is a link only once it has an address, and only sourceLink gives one.
    if outside(r"href", link):
        found.append("an address set outside sourceLink")
    # A link element, named as such, is made in sourceLink only. The browser takes the
    # name in either case, so "A" is a link too.
    if outside(r"(?i)\(\s*([\"'`])a\1\s*[,)]", link):
        found.append("a link element made outside sourceLink")
    # An element whose kind is not written out is made by make only, and the rule above
    # keeps make from being asked for a link.
    if outside(r"createElement\s*\(\s*(?![\"'`])", make):
        found.append("an element of unwritten kind made outside make")
    # Attributes are set as properties. By name, only the aria- attributes are set.
    if re.search(r"setAttribute(?!\s*\(\s*([\"'`])aria-[a-z]+\1)", source):
        found.append("an attribute other than aria- set by name")
    return found


def test_the_script_reads_nothing_as_html_or_code_and_makes_links_in_one_place() -> None:
    assert problems_in(static("demo.js")) == []


@pytest.mark.parametrize(
    ("line", "problem"),
    [
        ("page.answer.innerHTML = data.text;", "innerHTML"),
        ("page.answer.outerHTML = data.text;", "outerHTML"),
        ('page.answer.insertAdjacentHTML("beforeend", data.text);', "insertAdjacentHTML"),
        ("document.write(data.text);", "write or writeln"),
        ("document.writeln(data.text);", "write or writeln"),
        ("const doc = document; doc.write(data.text);", "write or writeln"),
        ("eval(data.text);", "eval"),
        ("(0, eval)(data.text);", "eval"),
        ("new Function(data.text)();", "Function"),
        ("const f = Function; f(data.text)();", "Function"),
        ("data.constructor.constructor(data.text)();", "a constructor reached by name"),
        ("page.ask.onclick = start;", "an inline event handler"),
        ('page.ask.setAttribute("onclick", data.text);', "an inline event handler"),
        ('setTimeout("start()", 10);', "a timer, which can run a string"),
        ("setInterval(data.text, 10);", "a timer, which can run a string"),
        ("new DOMParser().parseFromString(data.text, 'text/html');", "HTML parsed from a string"),
        ('document.createElementNS(namespace, "a");', "an element made in a namespace"),
        ('page.answer.append("javascript:alert(1)");', "a javascript: address"),
        ("document.location.assign(data.text);", "location"),
        ("top.location.replace(data.text);", "location"),
        ("location = data.text;", "location"),
        ("location.href = data.text;", "location"),
        ("window.open(data.text);", "open"),
        ("navigation.navigate(data.text);", "the Navigation API"),
        ('meta.httpEquiv = "refresh";', "a refresh by a meta element"),
        # A line that looks like a comment, inside a template literal: the ${...} runs.
        ("const x = `\n// ${page.status.innerHTML = data.text}\n`;", "innerHTML"),
        # A comment that names a forbidden call is caught too: comments are not skipped.
        ("// page.answer.innerHTML = data.text;", "innerHTML"),
        ("anchor.href = data.url;", "an address set outside sourceLink"),
        ('page.answer.append(make("a", "", data.text));', "a link element made outside sourceLink"),
        (
            "page.answer.append(document.createElement('a'));",
            "a link element made outside sourceLink",
        ),
        (
            'page.answer.append(document.createElement("A"));',
            "a link element made outside sourceLink",
        ),
        ("page.answer.setHTML(data.text);", "HTML parsed from a string"),
        ("history.go(-1);", "the history"),
        ("history.pushState(null, '', data.url);", "the history"),
        (
            "const a = document.createAttribute(name); link.attributes.setNamedItem(a);",
            "an attribute made as a node",
        ),
        (
            "page.answer.append(document.createElement(data.kind));",
            "an element of unwritten kind made outside make",
        ),
        (
            'page.answer.setAttribute("title", data.text);',
            "an attribute other than aria- set by name",
        ),
        ('localStorage.setItem("earlier", conversation.join());', "storage in the browser"),
        ("sessionStorage.conversation = question;", "storage in the browser"),
        ("document.cookie = question;", "storage in the browser"),
        ('indexedDB.open("conversation");', "storage in the browser"),
        ('turn.stream.ariaLive = "polite";', "a live region made by the script"),
        ('turn.stream.setAttribute("aria-live", "polite");', "a live region made by the script"),
        ('turn.stream.role = "status";', "a live region made by the script"),
        ('turn.body.append(make("output", "", data.text));', "an element the page does not make"),
        ('turn.body.append(make("img", "", ""));', "an element the page does not make"),
        ('turn.body.append(make("IFRAME", "", ""));', "an element the page does not make"),
        ('turn.body.append(make(data.kind, "", ""));', "an element the page does not make"),
        (
            'turn.body.append(document.createElement("script"));',
            "an element the page does not make",
        ),
        ("picture.src = data.url;", "a source or a style set as a property"),
        ("picture.srcset = data.url;", "a source or a style set as a property"),
        ("turn.body.style.backgroundImage = data.url;", "a source or a style set as a property"),
        ('turn.body["innerHTML"] = data.text;', "a source or a style set as a property"),
        ('const m = make; turn.body.append(m("img", "", ""));', "make reached under another name"),
        ('turn.body.append(make.call(null, "img", "", ""));', "make reached under another name"),
        ("Object.assign(picture, { src: data.url });", "properties set in bulk"),
        ('turn.body.attributeStyleMap.set("display", "none");', "a style written by the script"),
        ("document.styleSheets[0].insertRule(data.text);", "a style written by the script"),
    ],
)
def test_the_check_of_the_script_finds_each_kind_of_call_planted_in_it(
    line: str, problem: str
) -> None:
    """A check that found nothing in any script would pass the test above. This one finds."""
    assert problem in problems_in(f"{static('demo.js')}\n{line}\n")


def test_the_one_function_that_makes_a_link_checks_the_address_first() -> None:
    source = static("demo.js")
    span = function_body(source, "sourceLink")
    body = source[span.start : span.stop]

    checked = body.index('(url.startsWith("https://") || url.startsWith("http://"))')
    turned_away = body.index("if (!isWebAddress) {")
    made = body.index('document.createElement("a")')
    assert checked < turned_away < made
    assert body.count("createElement") == 1
    assert 'link.target = "_blank";' in body
    assert 'link.rel = "noopener noreferrer";' in body
    # The whole address is the link's title, so whoever points at the link sees it.
    assert "link.title = url;" in body


# An answer that came from the answer cache (ADR 0020).


def body_of(name: str) -> str:
    source = static("demo.js")
    span = function_body(source, name)
    return source[span.start : span.stop]


def body_of_constant(name: str) -> str:
    """A top-level constant of the script: its declaration to the semicolon that ends it."""
    source = static("demo.js")
    start = source.index(f"\nconst {name} =")
    return source[start : source.index(";\n", start) + 1]


def test_a_cached_answer_names_the_question_it_was_written_for_as_text() -> None:
    lines, render = body_of("cacheLines"), body_of("renderAnswer")

    assert "`It was written for the question: ${hit.question}`" in lines
    assert "No request was sent to the answer model." in lines
    assert "That judgement can be wrong." in lines, "said of a match by meaning"
    # Each line is given to make, which sets it as text. The stored question is a
    # question somebody typed, and is shown like everything else that comes from the API.
    assert 'note.append(make("p", "", line));' in render
    assert 'make("p", "cached-label", "From the cache")' in render
    assert ".cached {" in static("demo.css")
    # The search beside the answer is the new question's, not the one the answer was written
    # from. Each turn has its own fold of found passages, which the script builds, so the
    # words that say so are the script's and are put in every fold.
    assert "An answer that came from the cache ran no search" in body_of_constant("SEARCH_HINT")
    assert "SEARCH_HINT);" in body_of("addTurn")
    # The badge under the answer says it came from the cache, and from which tier.
    badges = body_of("answerBadges")
    assert '"From the cache"' in badges
    assert 'hit.tier === "exact" ? "exact match" : "matched by meaning"' in badges


# The notes of the checks in code that changed a reply or marked it (ADR 0025).


def test_the_page_reads_the_fields_the_api_sends_for_a_check() -> None:
    script = static("demo.js")

    read = set(re.findall(r"\bcheck\.(\w+)", script))
    assert read <= set(CheckNote.model_fields), read - set(CheckNote.model_fields)
    assert {"note", "action"} <= read
    assert "checks" in AskResponse.model_fields
    assert "answer.checks" in body_of("checkNotes")
    # A service older than the field sends none, and its answers are still shown.
    assert "checks" not in body_of("isAnswer")
    assert "Array.isArray(answer.checks)" in body_of("checkNotes")


def test_a_note_of_a_check_is_shown_as_text_and_read_out_with_the_outcome() -> None:
    box = body_of("checkBox")

    # Each note is given to make, which sets it as text, in an element of the fixed list,
    # under a class that is the script's own word.
    assert 'box.append(make("p", "check-note", note.note));' in box
    assert 'make("div", "check-notes")' in box
    # Read out with the outcome, through the status line, and no region of its own.
    assert 'const noted = notes.map((note) => ` Note: ${note.note}`).join("");' in body_of(
        "showAnswer"
    )
    assert "say(`${words}${from}${noted}`);" in body_of("showAnswer")
    assert ".check-notes {" in static("demo.css")


def test_the_page_reads_the_fields_the_api_sends_for_a_cached_answer() -> None:
    script = static("demo.js")

    assert set(re.findall(r"\bhit\.(\w+)", script)) == set(CachedAnswer.model_fields)
    assert "answer.cached" in body_of("cacheHit")
    assert "cached" in AskResponse.model_fields
    # A service older than the field sends none, and its answers are still shown.
    assert "cached" not in body_of("isAnswer")
    facts = body_of("cacheFacts")
    for label in ('"Served from"', '"Stored"', '"Similarity"', '"Second check"'):
        assert label in facts, label


# A conversation: the earlier questions sent with the next one (ADR 0024).


def test_the_question_searched_for_is_shown_as_text_when_it_is_not_the_one_asked() -> None:
    line, show = body_of("searchedLine"), body_of("showSearched")

    assert "`Searched for: ${searched.question}`" in line
    assert "searched.question !== searched.asked" in line
    # A follow-up that could not be rewritten says so: it was searched for as typed.
    assert 'searched.outcome === "failed" || searched.outcome === "unusable"' in line
    assert "as you typed it" in line
    # Set as text, like everything else that comes from the API, in the line of its own
    # turn: each turn has one, made empty and hidden with the turn.
    assert "turn.searchedLine.textContent = line;" in show
    assert 'event.name === "searched"' in body_of("readAnswer")
    # A follow-up that the service did not rewrite at all says so too.
    assert 'followUp && searched.outcome === "not_asked"' in line
    assert "this service did not rewrite the follow-up" in line
    assert "showSearched(turn, searched, followUp);" in body_of("readAnswer")


def test_the_page_reads_the_fields_the_api_sends_for_the_question_searched_for() -> None:
    script = static("demo.js")

    # The page's own element for the line is page.searched, which is not a field.
    read = set(re.findall(r"(?<!page\.)\bsearched\.(\w+)", script))
    assert read <= set(SearchedQuestion.model_fields)
    assert {"question", "asked", "outcome", "requests", "usage", "took_ms"} <= read
    assert "searched_for" in AskResponse.model_fields
    assert "answer.searched_for" in body_of("searchedFacts")
    # A service older than the field sends none, and its answers are still shown.
    assert "searched_for" not in body_of("isAnswer")


def test_only_the_questions_searched_for_are_sent_and_nothing_is_kept_in_the_browser() -> None:
    script = static("demo.js")
    ask, read = body_of("ask"), body_of("readAnswer")

    assert "body: JSON.stringify({ query: question, stream: true, earlier })," in ask
    assert "const earlier = conversation.slice(-EARLIER_SENT);" in ask
    # What is remembered of a turn is the question it searched for, or the one typed when
    # the service did not say: never the answer.
    assert "remember(searched === null ? question : searched.question);" in read
    assert script.count("remember(") == 2, "its definition and that one call"
    assert script.count("conversation.push(") == 1
    assert "conversation.push(question);" in body_of("remember")
    assert f"const EARLIER_SENT = {MAX_EARLIER_QUESTIONS};" in script
    # Reloading the page starts again: the conversation is a variable of the script.
    assert "const conversation = [];" in script
    assert problems_in(script) == []


def test_a_new_conversation_can_be_started_and_the_page_says_what_is_sent() -> None:
    markup = page()
    ids = {attrs["id"] for _, attrs in markup.elements if attrs.get("id")}
    html = static("demo.html")

    assert {"conversation-area", "conversation", "new-conversation"} <= ids
    assert "Only questions are sent, never answers." in html
    assert "reloading it starts a new conversation" in html
    # The same, in one short line that shows whenever there is a conversation: inside the
    # area the script shows then, and outside the fold that holds the list.
    start = html.index('<div id="conversation-area"')
    fold_end = html.index("</details>", start)
    area, after_fold = html[start:fold_end], html[fold_end : html.index('id="new-conversation"')]
    assert '<details class="context-fold">' in area
    notice = page().paragraphs["conversation-notice"]
    assert '<p id="conversation-notice"' in after_fold
    assert "Only questions are sent, never answers." in notice
    assert "kept on this page only" in notice
    assert "conversation.length = 0;" in body_of("forget")
    # The line about the question searched for belongs to each turn, which the script
    # builds: it is not an element of the page's markup any more.
    turn = body_of("addTurn")
    assert 'turn.searchedLine = make("p", "searched");' in turn
    assert "turn.searchedLine.hidden = true;" in turn


# The address under a source. The reader must see where a link leads: the line wraps, and
# the style sheet must not hide or cut it, its host included. The check below reads every
# rule of the sheet and fails on the ways of hiding or cutting text it knows of, set on the
# line, on an element or attribute selector that could reach it, or on the card, the found
# passage or an element around them. It does not compute the cascade, and it does not know
# every way a style sheet can hide text: a way it does not list gets past it.


def css_rules(sheet: str) -> list[tuple[str, dict[str, str]]]:
    """Every rule of a stylesheet, inside a media query or not: its selectors and its
    declarations, by property, with `!important` taken off the value. The frames of an
    animation are not rules for elements, and are left out."""
    sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.DOTALL)
    sheet = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*\s*\}", "", sheet)
    rules = []
    for selector, block in re.findall(r"([^{}]+)\{([^{}]*)\}", sheet):
        declarations = {}
        for declaration in block.split(";"):
            name, _, value = declaration.partition(":")
            value = value.lower().replace("!important", "").strip()
            if value:
                declarations[name.strip().lower()] = value
        rules.append((" ".join(selector.split()), declarations))
    return rules


def subjects(selector: str) -> list[str]:
    """The element each selector of a rule styles: the last compound of each, after the
    last combinator, with what is inside brackets kept whole. `.source-card > p` styles a
    `p`; `.source-address .x` styles the element inside the line, not the line."""
    found = []
    for part in selector.split(","):
        compounds = re.findall(r"(?:\[[^\]]*\]|[^\s>+~\[])+", part)
        if compounds:
            found.append(compounds[-1])
    return found


# The address line itself, by class; the card, the found passage, the fold that holds the
# whole address, and every element around them, by class; and the elements, by name, that
# the line is or that hold it. A rule for any other class or element, or for an id, cannot
# reach the line.
ADDRESS = {"source-address", "address-brief", "passage-address"}
AROUND = {
    "source-card", "source-cards", "sources-block", "passage-item", "passages", "passage",
    "search-fold", "turn-sources", "turn", "turns", "app",
}  # fmt: skip
HOLDERS = {"p", "li", "ol", "div", "details", "main", "body", "html", "*"}


def reach(compound: str) -> str | None:
    """Whether a rule for this compound styles the line ("line"), an element around it
    ("around"), something that may be either ("either"), or nothing that holds it (None)."""
    classes = set(re.findall(r"\.([\w-]+)", re.sub(r"\[[^\]]*\]", "", compound)))
    if "#" in re.sub(r"\[[^\]]*\]", "", compound):
        return None
    if classes:
        return "line" if classes & ADDRESS else "around" if classes & AROUND else None
    if compound == "[hidden]":
        return None  # the hidden attribute's own rule, which hides what is meant to be
    if "::" in compound:
        return None  # a box the style sheet adds before or after an element, not the element
    tag = re.match(r"[a-z][a-z0-9]*|\*", compound)
    return "either" if tag is None or tag.group(0) in HOLDERS else None


def wraps(value: str) -> bool:
    return value in ("normal", "pre-wrap", "pre-line", "break-spaces")


def clips(value: str) -> bool:
    return any(word in value.split() for word in ("paint", "strict", "content"))


# What hides or cuts text, set on the element itself or, through inheritance or through
# what it contains, on an element around it.
# A test of a value: whether it hides or cuts.
Test = Callable[[str], bool]
HIDES: dict[str, Test] = {
    "white-space": lambda value: not wraps(value),
    "visibility": lambda value: value in ("hidden", "collapse"),
    "display": lambda value: value == "none",
    "opacity": lambda value: value in ("0", "0.0", "0%"),
    "font-size": lambda value: value in ("0", "0px", "0rem", "0em"),
    "color": lambda value: value == "transparent",
    "contain": clips,
    "line-clamp": lambda value: True,
    "-webkit-line-clamp": lambda value: True,
}
CLIPS_AROUND: dict[str, Test] = {
    "overflow": lambda value: any(word in value.split() for word in ("hidden", "clip")),
    "overflow-x": lambda value: value in ("hidden", "clip"),
    "overflow-y": lambda value: value in ("hidden", "clip"),
}
# And what cuts it when set on the line, or on what may be the line.
CUTS_THE_LINE: dict[str, Test] = {
    "text-overflow": lambda value: True,
    "max-height": lambda value: value != "none",
    "height": lambda value: value != "auto",
    "overflow": lambda value: value != "visible",
    "overflow-x": lambda value: value != "visible",
    "overflow-y": lambda value: value != "visible",
    "clip-path": lambda value: value != "none",
    "clip": lambda value: value != "auto",
}
RULES_FOR: dict[str, dict[str, Test]] = {
    "line": {**HIDES, **CUTS_THE_LINE},
    "around": {**HIDES, **CLIPS_AROUND},
    "either": {**HIDES, **CUTS_THE_LINE},
}


def cuts_the_address(selector: str, declarations: dict[str, str]) -> list[str]:
    """What a rule does that could hide or cut the address line, by the ways it knows."""
    found = []
    for compound in subjects(selector):
        kind = reach(compound)
        if kind is not None:
            hides = RULES_FOR[kind]
            found += [
                f"{name}: {value} on {compound}"
                for name, value in declarations.items()
                if name in hides and hides[name](value)
            ]
    return found


def test_no_rule_of_the_stylesheet_can_cut_the_address_under_a_source() -> None:
    rules = css_rules(static("demo.css"))
    found = [
        problem
        for selector, declarations in rules
        for problem in cuts_the_address(selector, declarations)
    ]

    assert found == []
    # It wraps instead, so a long host shows whole on as many lines as it needs.
    (address,) = [d for selector, d in rules if selector == ".source-address"]
    assert address.get("overflow-wrap") == "anywhere"
    # The script puts the address in brief in this line, for the sources and the found
    # passages alike.
    assert 'const line = make("p", "source-address");' in body_of("addressLine")


@pytest.mark.parametrize(
    "rule",
    [
        ".source-address { overflow: hidden; }",
        ".source-address { text-overflow: ellipsis; white-space: nowrap; }",
        ".source-address { white-space: nowrap; }",
        "@media (max-width: 30rem) { .source-address { max-height: 1.4em; } }",
        ".address-brief { display: none; }",
        ".source-card { overflow: hidden; }",
        ".passage-item, .source-card { overflow: clip; }",
        ".source-card .source-address { visibility: hidden; }",
        ".source-address { -webkit-line-clamp: 1; }",
        # The ones the second review found getting past it.
        ".source-card > p { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
        ".source-card>p { white-space: nowrap; }",
        ".source-card { white-space: nowrap; }\n.turn { overflow: hidden; }",
        ".source-card { overflow: hidden !important; }",
        ".source-card { contain: paint; }",
        ".source-card { font-size: 0; }",
        ".source-address { color: transparent; }",
        ".source-card { opacity: 0; }",
        "p[class~=source-address] { white-space: nowrap; }",
        ".source-card [class~=source-address] { overflow: hidden; }",
        "li { white-space: nowrap; }",
        ".passage { display: none; }",
    ],
)
def test_the_check_of_the_stylesheet_finds_each_rule_that_could_cut_the_address(rule: str) -> None:
    rules = css_rules(f"{static('demo.css')}\n{rule}\n")

    assert [problem for selector, d in rules for problem in cuts_the_address(selector, d)] != []


# The notice of what is sent, the fold that holds the list, and the controls beside it. The
# script shows their area whenever there is a conversation (the node test runs that); what
# else decides whether a reader sees them is the markup and the style sheet, checked here.
NOTICE_CLASSES = {"context", "context-notice", "context-fold"}
NOTICE_IDS = {"conversation-area", "conversation-notice"}


def notice_problems(html: str, sheet: str) -> list[str]:
    """What keeps the notice of what is sent, or the words of its controls, from a reader."""
    found = []
    markup = Markup()
    markup.feed(html)
    markup.close()
    by_id = {str(attrs["id"]): attrs for _, attrs in markup.elements if attrs.get("id")}
    notice = by_id["conversation-notice"]
    if "hidden" in notice or notice.get("aria-hidden") == "true":
        found.append("the notice line is hidden in the markup")
    if by_id["conversation-area"].get("aria-hidden") == "true":
        found.append("the notice's area is hidden from a screen reader")
    # The words a phone does not show are still read out: the verb of the fold's control,
    # and the whole name of the new-conversation button.
    for tag, attrs in markup.elements:
        if "on-wide" in str(attrs.get("class") or "").split() and attrs.get("aria-hidden"):
            found.append(f"words shown on a larger screen only are hidden from a reader ({tag})")
    for selector, declarations in css_rules(sheet):
        for compound in subjects(selector):
            classes = set(re.findall(r"\.([\w-]+)", compound))
            ids = set(re.findall(r"#([\w-]+)", compound))
            if classes & NOTICE_CLASSES or ids & NOTICE_IDS:
                hidden = {
                    name: value
                    for name, value in declarations.items()
                    if (name in HIDES and HIDES[name](value))
                    or (name == "clip-path" and value != "none")
                    or (name in ("max-height", "height") and value in ("0", "0px"))
                }
                if hidden:
                    found.append(f"the notice or its fold is hidden: {hidden} on {compound}")
            if "on-wide" in classes and declarations.get("display") == "none":
                found.append("words shown on a larger screen only are not read on a phone")
    rules = css_rules(sheet)
    body = [d for selector, d in rules if ".context-body" in subjects(selector)]
    if not any("max-height" in d and d.get("overflow-y") in ("auto", "scroll") for d in body):
        found.append("the fold of what is sent has no height limit, or does not scroll")
    button = [d for selector, d in rules if "#new-conversation" in subjects(selector)]
    if not any(d.get("white-space") == "nowrap" for d in button):
        found.append("the new-conversation button may break onto two lines")
    return found


def test_the_notice_of_what_is_sent_and_its_controls_are_there_to_be_seen_and_heard() -> None:
    assert notice_problems(static("demo.html"), static("demo.css")) == []


NOTICE = '<p id="conversation-notice" class="context-notice"'
NOTICE_RULE = ".context-notice {\n  grid-area: notice;"
ON_WIDE = "  .on-wide {\n    position: absolute;"
NEW_BUTTON = "#new-conversation {\n  grid-area: new;\n  white-space: nowrap;\n}"
FOLD_BODY = "  max-height: min(40vh, 18rem);\n  margin-top: var(--s2);\n  overflow-y: auto;\n"
VERB = '<span id="conversation-verb" class="on-wide"'


@pytest.mark.parametrize(
    ("file", "old", "new"),
    [
        # The second review's faults.
        ("demo.css", FOLD_BODY, "  margin-top: var(--s2);\n"),
        ("demo.css", NOTICE_RULE, ".context-notice {\n  display: none;\n  grid-area: notice;"),
        ("demo.html", NOTICE + ">", NOTICE + " hidden>"),
        ("demo.css", ".context {\n  display: contents;\n}", ".context {\n  display: none;\n}"),
        ("demo.html", VERB + ">", VERB + ' aria-hidden="true">'),
        ("demo.css", ON_WIDE, "  .on-wide {\n    display: none;\n    position: absolute;"),
        ("demo.css", NEW_BUTTON, "#new-conversation {\n  grid-area: new;\n}"),
        # And some more.
        ("demo.css", NOTICE_RULE, ".context-notice {\n  visibility: hidden;\n  grid-area: notice;"),
        (
            "demo.css",
            NOTICE_RULE,
            ".context-notice {\n  opacity: 0 !important;\n  grid-area: notice;",
        ),
        ("demo.html", NOTICE + ">", NOTICE + ' aria-hidden="true">'),
    ],
)
def test_the_check_of_the_notice_finds_each_fault_planted_in_the_markup_or_the_style(
    file: str, old: str, new: str
) -> None:
    html, sheet = static("demo.html"), static("demo.css")
    text = html if file == "demo.html" else sheet
    assert text.count(old) == 1, old
    planted = text.replace(old, new)

    problems = notice_problems(*((planted, sheet) if file == "demo.html" else (html, planted)))
    assert problems != []
