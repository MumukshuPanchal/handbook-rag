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
from collections.abc import Iterator
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
from handbook_rag.api import MAX_QUERY_CHARS, create_app
from handbook_rag.config import Settings
from handbook_rag.demo import CONTENT_SECURITY_POLICY, FILES
from handbook_rag.index.indexer import index_chunks
from tests.support import FakeEmbedder, FakeSparseEncoder, make_chunk

GOLDEN = Path(__file__).parents[1] / "evals" / "golden" / "questions.jsonl"


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
    # Nothing else loads anything, or sends a form anywhere.
    assert not {"img", "iframe", "frame", "object", "embed", "base"} & set(tags)
    loading = {"src", "srcset", "data", "poster", "action", "formaction", "background"}
    for tag, attributes in markup.elements:
        if tag != "script":
            assert not loading & set(attributes), tag
    assert [attrs for tag, attrs in markup.elements if attrs.get("http-equiv")] == []
    # The page's one link of its own leads to the API's documentation, on this server.
    assert [attrs.get("href") for tag, attrs in markup.elements if tag == "a"] == ["/docs"]


def test_the_three_example_questions_are_real_questions_of_the_golden_set() -> None:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    golden = {row["question"]: row for row in rows}
    examples = [text.strip() for text in page().examples]

    assert len(examples) == len(set(examples)) == 3
    for example in examples:
        assert example in golden, example
        assert golden[example]["type"] == "answerable", example


def test_the_page_tells_the_reader_the_length_limit_before_they_reach_it() -> None:
    markup = page()
    (box,) = [attrs for tag, attrs in markup.elements if tag == "textarea"]

    assert box["maxlength"] == str(MAX_QUERY_CHARS)
    assert f"{MAX_QUERY_CHARS} characters" in markup.paragraphs["question-hint"]
    assert "question-hint" in str(box["aria-describedby"]).split()


def test_the_hint_tells_the_keys_that_ask_and_the_script_listens_for_them() -> None:
    markup = page()
    (box,) = [attrs for tag, attrs in markup.elements if tag == "textarea"]
    script = static("demo.js")

    assert "Ctrl+Enter" in markup.paragraphs["question-hint"]
    assert "Cmd+Enter" in markup.paragraphs["question-hint"]
    assert box["aria-keyshortcuts"] == "Control+Enter Meta+Enter"
    assert 'event.key === "Enter" && (event.ctrlKey || event.metaKey)' in script
    assert 'page.question.addEventListener("keydown"' in script


def test_the_question_has_a_label_and_the_state_of_the_request_is_announced() -> None:
    markup = page()
    by_id = {attrs["id"]: (tag, attrs) for tag, attrs in markup.elements if attrs.get("id")}
    labelled = [attrs.get("for") for tag, attrs in markup.elements if tag == "label"]

    assert [attrs.get("lang") for tag, attrs in markup.elements if tag == "html"] == ["en"]
    assert by_id["question"][0] == "textarea"
    assert labelled == ["question"]
    assert by_id["status"][1].get("role") == "status", "the state of the request"
    assert by_id["failure"][1].get("role") == "alert", "a failure, in words"
    # The streamed text would be read piece by piece. The status line says that the model
    # is writing, and when the answer is complete.
    assert "aria-live" not in by_id["stream"][1]
    assert [attrs["id"] for _, attrs in markup.elements if attrs.get("aria-live")] == []


def test_every_element_the_script_looks_up_is_on_the_page() -> None:
    markup = page()
    ids = {attrs["id"] for _, attrs in markup.elements if attrs.get("id")}
    looked_up = set(re.findall(r'getElementById\("([\w-]+)"\)', static("demo.js")))

    assert len(looked_up) > 10, "the script looks its elements up by id"
    assert looked_up <= ids, f"not on the page: {sorted(looked_up - ids)}"
    assert len(markup.examples) == 3, 'the script looks for "button.example"'


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
}


def problems_in(source: str) -> list[str]:
    """What a script holds that the demo page's script must not."""
    found = [name for name, pattern in FORBIDDEN.items() if re.search(pattern, source)]
    link, make = function_body(source, "sourceLink"), function_body(source, "make")

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
