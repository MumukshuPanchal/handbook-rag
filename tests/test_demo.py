"""The demo page: served by the API, switched off by a setting, and careful with what it shows.

The page's script cannot run here. The project has no JavaScript toolchain, on purpose
(ADR 0016), so its behaviour in a browser is checked by hand. What can be checked without
a browser is checked on the files as served and as written: the headers, the content
security policy, the page's markup, and the script's source, which must hold no name
that would read text from the API as HTML or as code or take the reader to another page,
and must make links in one place only.
"""

import hashlib
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


def static_bytes(name: str) -> bytes:
    return files("handbook_rag").joinpath("static", name).read_bytes()


# The page's typeface: the one file it loads besides its own three, and its licence.
FONT = "overpass-latin-wght-normal.woff2"
FONT_ADDRESS = f"/static/{FONT}"
FONT_SHA256 = "29217ef8adef5f6c059feb461bdb029348205fc233dc9b6ca8604b0d068ac086"
FONT_BYTES = 39_380
# The licence's text as it came with the font, byte for byte. The hook that trims spaces
# at the ends of lines leaves this file alone (.pre-commit-config.yaml).
LICENCE_SHA256 = "86e5ff25c701ec446d20b1a85b02ee6d36de8503a7288a4c948f5459809af1f0"


@pytest.mark.parametrize(
    ("path", "name", "content_type"),
    [
        ("/", "demo.html", "text/html; charset=utf-8"),
        ("/static/demo.js", "demo.js", "text/javascript; charset=utf-8"),
        ("/static/demo.css", "demo.css", "text/css; charset=utf-8"),
        (FONT_ADDRESS, FONT, "font/woff2"),
        ("/static/OFL.txt", "OFL.txt", "text/plain; charset=utf-8"),
    ],
)
def test_the_page_and_its_files_are_served_with_their_type_and_the_security_headers(
    client: TestClient, path: str, name: str, content_type: str
) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"] == content_type
    assert response.content == static_bytes(name)
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
    assert statuses == dict.fromkeys(FILES, 404)
    assert set(FILES) == {
        "/",
        "/static/demo.js",
        "/static/demo.css",
        FONT_ADDRESS,
        "/static/OFL.txt",
    }
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
    names = ("demo.html", "demo.js", "demo.css", FONT, "OFL.txt")
    for name in names:
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
        FONT_ADDRESS: f"stand-in {FONT}",
        "/static/OFL.txt": "stand-in OFL.txt",
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
    # This server is allowed for what the page needs, and for nothing else: its script, its
    # style, its requests to the API, and its one font file.
    assert {name for name, allowed in policy.items() if "'self'" in allowed} == {
        "script-src",
        "style-src",
        "connect-src",
        "font-src",
    }
    assert policy["script-src"] == policy["style-src"] == policy["connect-src"] == ["'self'"]
    assert policy["font-src"] == ["'self'"]
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
    # Under version 2 of the instructions the saved reply to that question says that the
    # passages do not say and adds GitLab's own figure, which the page shows as an answer.
    # So the note promises no decline: it says what is true of either reply.
    assert markup.paragraphs[note].strip() == (
        "The last one is not in the handbook: ask it to see how the assistant says so."
    )


def test_the_links_in_the_header_lead_to_sections_the_readme_has() -> None:
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in readme.splitlines()
        if line.startswith("## ")
    }

    for link in README_LINKS:
        assert link.removeprefix(f"{REPOSITORY}#") in headings, link


def test_the_header_says_what_the_page_is_in_a_line_that_fits() -> None:
    # The sentence beside the page's name: at 1280 pixels it fits on one line beside the
    # name and the two links, which the longer one did not.
    says = re.findall(r'<p class="band-says">(.*?)</p>', static("demo.html"))
    assert says == ["Answers from the public GitLab handbook, with their sources."]


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
    # Drawn as a line, so in the order things happen: a follow-up is rewritten first, a
    # stored answer is served in place of the rest, and then the search, the answer, or the
    # decline.
    order = [
        re.sub(r"<[^>]+>", "", strong) for strong in re.findall(r"<strong>(.*?)</strong>", steps[0])
    ]
    assert order == [
        "A follow-up is rewritten first.",
        "A stored answer",
        "The handbook is searched",
        "A model writes the answer",
        "Or it declines",
    ]
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
    assert 'make("p", "cached-label", "A stored answer.")' in render
    assert ".cached {" in static("demo.css")
    # The search beside the answer is the new question's, not the one the answer was written
    # from. Each turn has its own fold of found passages, which the script builds, so the
    # words that say so are the script's and are put in every fold.
    assert "An answer that came from the cache ran no search" in body_of_constant("SEARCH_HINT")
    assert "SEARCH_HINT);" in body_of("addTurn")
    # The line of stops says the answer was found stored, and for which kind of question.
    stops = body_of("routeStops")
    assert 'name: "Found stored"' in stops
    assert 'hit.tier === "exact" ? "an answer stored for the same question"' in stops


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


# The address line itself and what is inside it, by class; every element around it, by
# class and by id; and the elements, by name, that the line is, that it holds, or that hold
# it. The script builds most of these, so test_demo_behaviour.py reads them from the page
# the script really builds, in every state its run puts the page in, and fails when this
# list misses one: a new wrapper cannot be forgotten here. A rule whose subject is only of
# other classes or ids, or another element, cannot reach the line in that structure; what
# stands inside :not() and its kin does not count as narrowing a subject. This check reads
# selectors and does not compute the cascade: what holds every rule that could hide text,
# whatever its selector, is the list of rules further down
# (test_what_carries_meaning_is_set_only_in_the_listed_rules).
ADDRESS = {"source-address", "address-brief", "passage-address"}
AROUND = {
    # Built by the script, for each turn.
    "turn", "ended-answer", "ended-declined", "ended-reply", "ended-failed", "ended-stopped",
    "cols", "turn-sources", "no-sources", "sources-block", "source-cards", "source-card",
    "line-1", "line-2", "line-3", "line-4", "line-5", "line-other", "linked", "marked",
    "source-body", "passage", "search-fold", "nothing-more", "passages", "passage-item",
    # In the page's markup, around the list of turns.
    "turns", "app",
}  # fmt: skip
AROUND_IDS = {"turns"}
HOLDERS = {"p", "span", "li", "ol", "div", "details", "main", "body", "html", "*"}


def reach(
    compound: str,
    around: set[str] = AROUND,
    holders: set[str] = HOLDERS,
    ids: set[str] = AROUND_IDS,
) -> str | None:
    """Whether a rule for this compound styles the line ("line"), an element around it
    ("around"), something that may be either ("either"), or nothing that holds it (None).
    What stands around the line can be given, as the structure the script builds."""
    # What stands inside :not(), :is(), :has() or :where() does not narrow the element a
    # rule styles to those classes: p:not(.lede) styles every other paragraph. Such a
    # compound is read without them, so it reaches the line unless something else in it
    # does not.
    bare = re.sub(r"\[[^\]]*\]", "", compound)
    bare = re.sub(r":[\w-]+\((?:[^()]|\([^()]*\))*\)", "", bare)
    classes = set(re.findall(r"\.([\w-]+)", bare))
    named = set(re.findall(r"#([\w-]+)", bare))
    if named:
        return "around" if named & ids else None
    if classes:
        return "line" if classes & ADDRESS else "around" if classes & around else None
    if compound == "[hidden]":
        return None  # the hidden attribute's own rule, which hides what is meant to be
    if "::" in compound:
        return None  # a box the style sheet adds before or after an element, not the element
    tag = re.match(r"[a-z][a-z0-9]*|\*", bare)
    return "either" if tag is None or tag.group(0) in holders else None


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


def cuts_the_address(
    selector: str,
    declarations: dict[str, str],
    around: set[str] = AROUND,
    holders: set[str] = HOLDERS,
    ids: set[str] = AROUND_IDS,
) -> list[str]:
    """What a rule does that could hide or cut the address line, by the ways it knows."""
    found = []
    for compound in subjects(selector):
        kind = reach(compound, around, holders, ids)
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
        # The ones the review of the new look found getting past it: the wrapper of a
        # source's text, and the columns of a turn.
        ".source-body { overflow: hidden; max-height: 6em; }",
        ".source-body { display: -webkit-box; -webkit-line-clamp: 4; overflow: hidden; }",
        ".cols { overflow: hidden; }\n.source-body { white-space: nowrap; }",
        # And what else stands around the line now: an id in the markup, the colour of a
        # source's line, the state of the fold, and the spans inside the line.
        "#turns { overflow: hidden; }",
        ".line-3 { overflow: clip; }",
        ".nothing-more { display: none; }",
        "span { display: none; }",
        # The ones the second review found: what stands inside :not() does not narrow the
        # rule.
        "p:not(.lede) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
        "*:not(.sign) { white-space: nowrap; }",
        ":not(.band-says) { overflow: hidden; max-height: 1px; }",
        "p:is(:not(.lede)) { white-space: nowrap; }",
        "body:has(.turn) p:not(.lede) { overflow: hidden; height: 1px; }",
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


# The page's typeface. A font is a file that nobody reads, so it is pinned by its hash and
# its size, its licence travels with it, and the style sheet may name no other file or
# address: the page loads nothing but its own files, from its own server.


def test_the_font_file_is_the_one_pinned_and_its_licence_is_beside_it() -> None:
    font = static_bytes(FONT)

    assert len(font) == FONT_BYTES
    assert hashlib.sha256(font).hexdigest() == FONT_SHA256
    assert font[:4] == b"wOF2", "a WOFF2 file"
    licence = static("OFL.txt")
    assert hashlib.sha256(static_bytes("OFL.txt")).hexdigest() == LICENCE_SHA256
    assert "SIL OPEN FONT LICENSE Version 1.1" in licence
    assert "The Overpass Project Authors" in licence


def unescaped(sheet: str) -> str:
    """A style sheet with its comments taken out and its escapes resolved, as a browser
    reads it: `\\75 rl(` is `url(`."""
    sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.DOTALL)
    sheet = re.sub(r"\\([0-9a-fA-F]{1,6})\s?", lambda hit: chr(int(hit.group(1), 16)), sheet)
    return re.sub(r"\\(.)", r"\1", sheet)


def font_problems(sheet: str) -> list[str]:
    """What a style sheet names besides the page's one font file. It reads the sheet as a
    browser does, in either case and with escapes resolved, and finds: an address in a
    `url()` or `src()` other than the font's, an `@import`, a `local()` font, an image made
    by `image-set()`, `image()`, `cross-fade()` or `element()`, a second font face, and a
    family other than Overpass and the system's own in `font-family` or in the `font`
    shorthand, which may only be `inherit`. In a browser the policy stands behind it:
    `default-src 'none'` and `font-src 'self'` refuse anything from another origin, whatever
    the sheet names."""
    found = []
    sheet = unescaped(sheet)
    low = sheet.lower()
    addresses = [
        next(part for part in parts if part) if any(parts) else ""
        for parts in re.findall(r"url\(\s*(?:\"([^\"]*)\"|'([^']*)'|([^)\"'\s]*))\s*\)", low)
    ]
    if addresses != [FONT_ADDRESS] or low.count("url(") != 1:
        found.append(f"the sheet names a file or an address but the font: {addresses}")
    loads = r"@import\b|@namespace\b|(?<!\w)(?:src|local|image-set|image|cross-fade|element)\("
    if re.search(loads, low):
        found.append("the sheet imports, names a local font, or makes an image")
    faces = re.findall(r"@font-face\s*\{([^}]*)\}", low)
    if len(faces) != 1 or 'font-family: "overpass"' not in faces[0]:
        found.append("the sheet has another font face than Overpass")
    for selector, declarations in css_rules(sheet):
        family = declarations.get("font-family")
        if family is not None and family not in ('"overpass"', "var(--font)", "inherit"):
            found.append(f"a font family other than the page's on {selector}: {family}")
        if declarations.get("font", "inherit") != "inherit":
            found.append(
                f"the font shorthand on {selector}, which names a family: {declarations['font']}"
            )
    stacks = re.findall(r"--font\s*:\s*([^;}]+)", low)
    system = {"system-ui", "-apple-system", '"segoe ui"', "roboto", "arial", "sans-serif"}
    for stack in stacks:
        names = [name.strip() for name in stack.split(",")]
        if names[0] != '"overpass"' or not set(names[1:]) <= system:
            found.append(
                f"the page's font stack names a font not Overpass nor the system's: {stack}"
            )
    if len(stacks) != 1:
        found.append(f"the page's font stack is set {len(stacks)} times")
    return found


def test_the_style_sheet_names_no_file_or_address_but_the_one_font() -> None:
    assert font_problems(static("demo.css")) == []


@pytest.mark.parametrize(
    "rule",
    [
        '@font-face { font-family: "Inter"; src: url("https://fonts.example/inter.woff2"); }',
        '@import url("https://fonts.example/css?family=Inter");',
        '.band { background-image: url("/static/band.png"); }',
        '.band { font-family: "Helvetica Neue", sans-serif; }',
        '@font-face { font-family: "Overpass"; src: local("Overpass"); }',
        # The ones the review of the new look found getting past it.
        '.band { background: URL("https://evil.example/x.png"); }',
        ".band { background: Url(/static/x.png); }",
        '@IMPORT "https://evil.example/x.css";',
        '.band { background: IMAGE-SET("https://evil.example/x.png" 1x); }',
        ".band { background: \\75 rl(https://evil.example/x.png); }",
        '.band { font: 700 17px "Comic Sans MS", serif; }',
        '.band { font-family: "Comic Sans MS" }',
        # The ones the second review found: a quoted address that holds a bracket, and a
        # function under a vendor's prefix.
        '.band { background: url("https://evil.example/x).png"); }',
        '.band { background: -webkit-image-set("https://e.example/x.png" 1x); }',
        # And the newer function that takes an address, and a second stack.
        '.band { background: src("https://evil.example/x.png"); }',
        ':root { --font: "Comic Sans MS", serif; }',
    ],
)
def test_the_check_of_the_fonts_finds_each_rule_planted_in_the_style_sheet(rule: str) -> None:
    assert font_problems(f"{static('demo.css')}\n{rule}\n") != []


# The look's own rules, which the review of the page held it to: colour means source, so
# every colour is a token; no capitals for labels, no pills, no shadows.


def look_problems(sheet: str) -> list[str]:
    found = []
    sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.DOTALL)
    for selector, declarations in css_rules(sheet):
        for name, value in declarations.items():
            if re.search(r"#[0-9a-f]{3,8}\b|\brgba?\(|\bhsla?\(", value) and not name.startswith(
                "--"
            ):
                found.append(f"a colour written outside the tokens: {name}: {value} on {selector}")
            if name == "text-transform" and "uppercase" in value:
                found.append(f"capitals for a label on {selector}")
            if name == "box-shadow":
                found.append(f"a shadow on {selector}")
            if name == "border-radius" and value not in ("0", "50%"):
                found.append(f"a rounded box or a pill on {selector}: {value}")
    return found


def test_the_look_keeps_its_own_rules() -> None:
    assert look_problems(static("demo.css")) == []


@pytest.mark.parametrize(
    "rule",
    [
        ".band { background: #4f46e5; }",
        ".badge { text-transform: uppercase; letter-spacing: 0.08em; }",
        ".source-card { box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); }",
        ".badge { border-radius: 999px; }",
        ".answer-card { color: rgb(20, 20, 20); }",
    ],
)
def test_the_check_of_the_look_finds_each_rule_planted_in_the_style_sheet(rule: str) -> None:
    assert look_problems(f"{static('demo.css')}\n{rule}\n") != []


# Colour means source: it is the one meaning the look carries. Each line class takes its own
# line colour and a numeral colour, no two lines share a colour in either theme, the
# numeral on each line reads at 4.5:1 at least in either theme, worked out here from the
# tokens, and the bullets and strokes take the colour of their line. That the script gives
# each citation and its source the class of its number is run in node.


def css_blocks(sheet: str) -> list[tuple[str, str, dict[str, str]]]:
    """Every rule of a style sheet with the condition of the media query it sits in, or ""
    for none, in the order of the sheet."""
    sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.DOTALL)
    sheet = re.sub(r"@keyframes[^{]*\{(?:[^{}]*\{[^{}]*\})*\s*\}", "", sheet)
    found: list[tuple[str, str, dict[str, str]]] = []
    at = 0
    while at < len(sheet):
        media = re.compile(r"@media([^{]*)\{").search(sheet, at)
        plain = sheet[at : len(sheet) if media is None else media.start()]
        found += [("", selector, declared) for selector, declared in css_rules(plain)]
        if media is None:
            break
        depth, end = 1, media.end()
        while depth and end < len(sheet):
            depth += {"{": 1, "}": -1}.get(sheet[end], 0)
            end += 1
        condition = " ".join(media.group(1).split())
        found += [
            (condition, sel, declared) for sel, declared in css_rules(sheet[media.end() : end - 1])
        ]
        at = end
    return found


def merged(sheet: str) -> dict[tuple[str, str], dict[str, str]]:
    """The declarations of each selector in each media condition, a later one winning."""
    rules: dict[tuple[str, str], dict[str, str]] = {}
    for condition, selector, declared in css_blocks(sheet):
        rules.setdefault((condition, selector), {}).update(declared)
    return rules


DARK = "(prefers-color-scheme: dark)"


def theme_tokens(sheet: str) -> dict[str, dict[str, str]]:
    """The values of the custom properties of :root, in the light theme and the dark."""
    light: dict[str, str] = {}
    dark: dict[str, str] = {}
    for condition, selector, declared in css_blocks(sheet):
        if selector == ":root" and condition in ("", DARK):
            (dark if condition == DARK else light).update(declared)
    return {"light": light, "dark": {**light, **dark}}


def resolved(value: str, tokens: dict[str, str]) -> str | None:
    """A colour as a hex code, through the custom properties it names."""
    for _ in range(5):
        named = re.fullmatch(r"var\((--[\w-]+)\)", value.strip())
        if named is None:
            break
        value = tokens.get(named.group(1), "")
    return value if re.fullmatch(r"#[0-9a-f]{6}", value) else None


def contrast(one: str, other: str) -> float:
    """The contrast ratio of two colours, as WCAG 2 works it out."""

    def luminance(colour: str) -> float:
        channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    high, low = sorted((luminance(one), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


LINES = ["line-1", "line-2", "line-3", "line-4", "line-5", "line-other"]


def line_colour_problems(sheet: str) -> list[str]:
    found = []
    rules = merged(sheet)
    for theme, tokens in theme_tokens(sheet).items():
        colours = {}
        for line in LINES:
            declared = rules.get(("", f".{line}"), {})
            own = "var(--other)" if line == "line-other" else f"var(--l{line[-1]})"
            if declared.get("--c") != own:
                found.append(f"{line} does not take its own colour, {own}: {declared.get('--c')}")
            colour = resolved(declared.get("--c", ""), tokens)
            numeral = resolved(declared.get("--on-c", ""), tokens)
            if colour is None or numeral is None:
                found.append(f"{line} has no colour or no numeral colour in the {theme} theme")
                continue
            colours[line] = colour
            if contrast(numeral, colour) < 4.5:
                ratio = contrast(numeral, colour)
                found.append(f"the numeral on {line} reads at {ratio:.2f}:1 in the {theme} theme")
        if len(set(colours.values())) != len(colours):
            found.append(f"two lines share a colour in the {theme} theme: {colours}")
    for selector, wanted in (
        (".cite", {"background": "var(--c)", "color": "var(--on-c)"}),
        (".source-number", {"background": "var(--c)", "color": "var(--on-c)"}),
        (".source-card::before", {"background": "var(--c)"}),
    ):
        declared = rules.get(("", selector), {})
        for name, value in wanted.items():
            if declared.get(name) != value:
                found.append(
                    f"{selector} does not take its line's colour: {name}: {declared.get(name)}"
                )
    return found


def test_each_source_has_a_line_colour_of_its_own_and_its_numeral_reads() -> None:
    assert line_colour_problems(static("demo.css")) == []


@pytest.mark.parametrize(
    "rule",
    [
        ".line-4 { --on-c: var(--on-line); }",
        ":root { --on-l4: #ffffff; }",
        ":root { --l2: #d8261c; }",
        ".line-2 { --c: var(--l1); }",
        f"@media {DARK} {{ :root {{ --l5: #0a57c2; --l2: #0a57c2; }} }}",
        ".line-other { --c: var(--l5); }",
        ".cite { background: var(--l1); }",
        ".source-card::before { background: var(--rule-line); }",
        ".source-number { color: var(--text); }",
    ],
)
def test_the_check_of_the_line_colours_finds_each_rule_planted_in_the_style_sheet(
    rule: str,
) -> None:
    assert line_colour_problems(f"{static('demo.css')}\n{rule}\n") != []


# The other style rules that carry meaning: the one motion only for a reader who has not
# asked for less, a visible focus, the next stop filled, a decline's line dashed both ways,
# a bar where a line ends, and the dark theme's sign edged so that it stands out from the
# page.

REDUCED = "(prefers-reduced-motion: no-preference)"
NARROW = "(max-width: 40rem)"


def taken_away(value: str | None) -> bool:
    return value is not None and (
        value.split()[:1] in (["none"], ["0"], ["0px"]) or "none" in value.split()
    )


def meaning_problems(sheet: str) -> list[str]:
    found = []
    rules = merged(sheet)
    for condition, selector, declared in css_blocks(sheet):
        for name, value in declared.items():
            moving = name.startswith(("animation", "transition")) and value not in ("none", "0s")
            if moving and condition != REDUCED:
                found.append(f"motion outside {REDUCED}: {name}: {value} on {selector}")
        if "focus" in selector and (
            taken_away(declared.get("outline"))
            or taken_away(declared.get("outline-style"))
            or declared.get("outline-width") in ("0", "0px")
        ):
            # Only the question box takes its own outline away, because its frame draws the
            # focus instead.
            frame = rules.get(("", ".field:focus-within"), {}).get("outline", "")
            if selector != "textarea:focus-visible" or not re.match(r"[2-9]px solid", frame):
                found.append(f"the focus is not drawn on {selector}")
    focus = rules.get(("", ":focus-visible"), {}).get("outline", "")
    if not re.match(r"[2-9]px solid var\(--focus\)", focus):
        found.append(f"the focus has no visible outline: {focus}")
    if rules.get(("", ".station-next::after"), {}).get("background") != "var(--rule-line)":
        found.append("the next stop is not a filled ring")
    for condition in ("", NARROW):
        dashed = rules.get((condition, ".seg-dashed::before"), {}).get("background", "")
        if not (
            "repeating-linear-gradient" in dashed
            and "var(--off)" in dashed
            and "transparent" in dashed
        ):
            found.append(f"a decline's line is not dashed ({condition or 'wide'}): {dashed}")
    ring = rules.get(("", ".station::after"), {})
    bar = rules.get(("", ".station-end::after"), {})
    if (
        ring.get("content") != '""'
        or bar.get("background") != "var(--rule-line)"
        or "content" in bar
        or bar.get("display") == "none"
        or not {bar.get("width"), bar.get("height")} & {"28px"}
    ):
        found.append(f"no bar is drawn where a line ends: {bar}")
    # The words the fold's control adds to its own: to show the passages it holds, or, with
    # none left, what the search is.
    for selector, words in (
        (".search-fold > summary::after", '"show them" / ""'),
        (".search-fold[open] > summary::after", '"hide them" / ""'),
        (".search-fold.nothing-more > summary::after", '"about this search" / ""'),
    ):
        if rules.get(("", selector), {}).get("content") != words:
            found.append(f"the fold's control says the wrong thing: {selector}")
    dark = theme_tokens(sheet)["dark"]
    edge, page = resolved("var(--sign-edge)", dark), resolved("var(--page)", dark)
    if edge is None or page is None or contrast(edge, page) < 3:
        found.append("the dark theme's sign does not stand out from the page")
    border = rules.get(("", ".sign, .question-sign"), {}).get("border", "")
    if not re.fullmatch(r"[2-9]px solid var\(--sign-edge\)", border):
        found.append(f"a sign has no edge of its own: {border}")
    return found


def test_the_style_rules_that_carry_meaning_hold() -> None:
    assert meaning_problems(static("demo.css")) == []


@pytest.mark.parametrize(
    "rule",
    [
        # The eight style faults of the review of the new look, and more of each kind.
        ".station-next::after { animation: approach 1.2s ease-in-out infinite; }",
        ":focus-visible { outline: none; }",
        ".station-next::after { background: var(--ring-fill); }",
        ".seg-dashed::before { background: var(--off); }",
        ".station-end::after { background: transparent; }",
        f"@media {DARK} {{ :root {{ --sign-edge: #000000; }} }}",
        ".station { transition: color 0.2s; }",
        f"@media {NARROW} {{ .seg-dashed::before {{ background: var(--off); }} }}",
        ".cite:focus-visible { outline: 0; }",
        ".station-end::after { content: none; }",
        ".sign, .question-sign { border: 0; }",
        '.search-fold.nothing-more > summary::after { content: "Show them"; }',
    ],
)
def test_the_check_of_the_style_rules_that_carry_meaning_finds_each_rule_planted(rule: str) -> None:
    assert meaning_problems(f"{static('demo.css')}\n{rule}\n") != []


# The style rules that carry meaning, written out. Looking a rule up by its selector lets a
# rule the test did not think of get past, and the tests do not compute the cascade. So it
# is turned round: a property that carries meaning may be set only in a rule listed below,
# as written, with a word on why; any other rule that sets it fails, whatever its selector
# and wherever it stands. What carries meaning:
# - what can hide, cut or move text out of view (HIDING, and display: none, a font size
#   under 8 pixels or one this test cannot tell, a text colour that is not a token or reads
#   under 4.5:1 on the page or the strip, a negative offset, margin or spacing), since the
#   address line under a source must always be seen;
# - the focus (outline) and the edges (border), since a sign's edge in the dark theme and
#   the question box's frame stand in for what a reader must see;
# - motion (animation, transition), allowed only under reduced-motion no-preference;
# - every rule that draws a box before or after an element, since the stops, their rings,
#   the dashes of a decline, the bar where a line ends, the next stop and each source's
#   stroke are drawn that way;
# - every ground (background), since text on a ground of its own colour is gone;
# - a colour on anything that could be a citation or a source's bullet: a rule whose
#   subject has no class but theirs, or none.
# Custom properties are set only in :root, once in the light theme and once in the one dark
# block, and --c and --on-c only by the six line classes. A media query of a condition the
# test does not know fails, as do any other at-rule, a rule inside a rule, and !important
# outside the listed rules.

KNOWN_CONDITIONS = {"", DARK, "(min-width: 64rem)", NARROW, "(max-width: 30rem)", REDUCED}
KEYFRAMES = {"approach": "50% { transform: scale(0.7); }"}
HIDING = {
    "overflow", "overflow-x", "overflow-y", "overflow-block", "overflow-inline",
    "overflow-clip-margin", "text-overflow", "white-space", "white-space-collapse",
    "text-wrap", "text-wrap-mode", "max-height", "max-block-size", "height", "block-size",
    "max-width", "max-inline-size", "line-clamp", "-webkit-line-clamp", "-webkit-box-orient",
    "visibility", "opacity", "clip", "clip-path", "mask", "mask-image", "-webkit-mask",
    "-webkit-mask-image", "contain", "content-visibility", "text-indent", "transform",
    "scale", "zoom", "filter", "position", "content", "-webkit-text-fill-color",
    "mix-blend-mode",
}  # fmt: skip
OFFSETS = {"left", "right", "top", "bottom", "inset", "translate", "letter-spacing", "word-spacing"}
COLOURS = {
    "background", "background-color", "background-image", "color", "border-color",
    "outline-color", "fill", "stroke", "-webkit-text-fill-color", "text-decoration-color",
}  # fmt: skip
SOURCE_CLASSES = {"cite", "source-number", "source-card", "linked", "marked", *LINES}
PSEUDO = re.compile(r"::|:(?:before|after|first-line|first-letter)\b", re.IGNORECASE)

Declarations = list[tuple[str, str]]


def declarations(block: str) -> Declarations:
    """The declarations of a rule as written, in order, each value in lower case with its
    spaces collapsed, `!important` kept at its end."""
    found = []
    for declaration in block.split(";"):
        name, colon, value = declaration.partition(":")
        if colon and value.strip():
            value = " ".join(value.lower().split())
            value = re.sub(r"\s*!\s*important$", " !important", value)
            found.append((name.strip().lower(), value))
    return found


def sheet_rules(sheet: str) -> tuple[list[tuple[str, str, Declarations]], list[str]]:
    """Every rule of a style sheet with the condition of the media query it sits in, or ""
    for none, and its declarations as written; and what the sheet holds that the tests do
    not know: a media query of another condition or inside another, an at-rule other than
    `@font-face` and the one `@keyframes`, and a rule inside a rule."""
    rules: list[tuple[str, str, Declarations]] = []
    problems: list[str] = []

    def read(text: str, condition: str) -> None:
        at = 0
        while (opening := re.compile(r"([^{}]*)\{").search(text, at)) is not None:
            prelude = " ".join(opening.group(1).split())
            depth, end = 1, opening.end()
            while depth and end < len(text):
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                end += 1
            body = text[opening.end() : end - 1]
            at = end
            if prelude.lower().startswith("@media"):
                inner = " ".join(prelude[6:].split()).lower()
                if condition or inner not in KNOWN_CONDITIONS:
                    problems.append(f"a media query the tests do not know: {prelude}")
                read(body, inner)
            elif prelude.lower().startswith("@keyframes"):
                name = prelude[10:].strip()
                if condition or KEYFRAMES.get(name) != " ".join(body.split()):
                    problems.append(f"an animation the tests do not know: {prelude}")
            elif prelude.startswith("@") and prelude.lower() != "@font-face":
                problems.append(f"an at-rule the tests do not know: {prelude}")
            elif "{" in body:
                problems.append(f"a rule inside a rule: {prelude}")
            else:
                rules.append((condition, prelude, declarations(body)))
        if text[at:].strip():
            problems.append(f"text the tests cannot read as a rule: {text[at:].strip()[:60]}")

    read(unescaped(sheet), "")
    return rules, problems


def may_be_a_source_mark(selector: str) -> bool:
    """Whether a rule may style a citation, a source's bullet or a source's line: its
    subject has no class but theirs, or none. A selector with a function in it, such as
    :not() or :is(), is taken to reach them."""
    if "(" in selector:
        return True
    for compound in subjects(selector):
        positive = set(re.findall(r"\.([\w-]+)", re.sub(r"\[[^\]]*\]", "", compound)))
        if not positive or positive <= SOURCE_CLASSES:
            return True
    return False


def unreadable_colour(value: str, themes: dict[str, dict[str, str]]) -> bool:
    """Whether a colour of text may not read on the page: none, one that is not a token, or
    one that reads under 4.5:1 on the page or the strip in either theme. It is then right
    only on another ground, which a listed rule says."""
    if value in ("inherit", "currentcolor"):
        return False
    for tokens in themes.values():
        colour = resolved(value, tokens)
        if colour is None:
            return True
        for ground in (resolved("var(--page)", tokens), resolved("var(--ground)", tokens)):
            if ground is None or contrast(colour, ground) < 4.5:
                return True
    return False


def negative(value: str) -> bool:
    return bool(re.search(r"(?:^|\s)-\d|(?:^|\s)-\.\d", value))


def slight(value: str) -> bool:
    """Whether a negative letter spacing only tightens a heading: a hundredth of its size."""
    spacing = re.fullmatch(r"-(\d*\.?\d+)em", value)
    return spacing is not None and float(spacing.group(1)) <= 0.01


def too_small(value: str, themes: dict[str, dict[str, str]]) -> bool:
    """Whether a font size may be too small to read: under 8 pixels, taking an em as 16, or
    a size this test cannot tell, such as one in units of the window or worked out."""
    if value == "inherit":
        return False
    for tokens in themes.values():
        named = re.fullmatch(r"var\((--[\w-]+)\)", value)
        size = re.fullmatch(
            r"(\d*\.?\d+)(px|em|rem|%)?", tokens.get(named.group(1), "") if named else value
        )
        if size is None:
            return True
        scale = {"em": 16, "rem": 16, "%": 0.16}.get(size.group(2) or "px", 1)
        if float(size.group(1)) * scale < 8:
            return True
    return False


def carries_meaning(
    selector: str, name: str, value: str, themes: dict[str, dict[str, str]]
) -> str | None:
    """Which kind of meaning a declaration carries, or None."""
    if name.startswith("--"):
        return None  # held where it may be set, below
    if PSEUDO.search(selector):
        return "drawn before or after an element"
    hides = (
        name in HIDING
        or (name == "display" and value.split()[0] == "none")
        or (name == "font-size" and too_small(value, themes))
        or (
            (name in OFFSETS or name.startswith("margin"))
            and negative(value)
            and not (name == "letter-spacing" and slight(value))
        )
        or (name in ("color", "-webkit-text-fill-color") and unreadable_colour(value, themes))
    )
    if hides:
        return "can hide text"
    if name.startswith("outline"):
        return "the focus"
    if name.startswith(("animation", "transition")):
        return "motion"
    if name.startswith("border") and name != "border-radius":
        return "an edge"
    if name.startswith("background"):
        return "a ground"
    if name in COLOURS and may_be_a_source_mark(selector):
        return "a colour on what may be a source's mark"
    return None


def listed_rules() -> dict[tuple[str, str], Declarations]:
    """The rules of LISTED, by condition and selector, each with its declarations; each must
    come with a word on why it may set what it sets."""
    found: dict[tuple[str, str], Declarations] = {}
    entry = re.compile(
        r"/\*\s*(.*?)\s*\*/\s*(?:@media\s*(\(.*?\))\s+)?([^{}/]+?)\s*\{([^{}]*)\}", re.DOTALL
    )
    for why, condition, selector, block in entry.findall(LISTED):
        assert why.strip(), f"a listed rule with no word on why: {selector}"
        key = (" ".join((condition or "").split()).lower(), " ".join(selector.split()))
        assert key not in found, f"a rule listed twice: {key}"
        found[key] = declarations(block)
    return found


def meaning_rules_problems(sheet: str) -> list[str]:
    """Every way the sheet sets what carries meaning outside the listed rules, or differs from
    them, and every place a custom property is set but the two of the root and the line
    classes."""
    rules, found = sheet_rules(sheet)
    themes = theme_tokens(sheet)
    real: dict[tuple[str, str], Declarations] = {}
    for condition, selector, declared in rules:
        if selector.lower() == "@font-face":
            continue  # the font test reads it
        for name, value in declared:
            if carries_meaning(selector, name, value, themes) is not None:
                real.setdefault((condition, selector), []).append((name, value))
            elif value.endswith("!important"):
                found.append(f"!important on {name} in {selector}")
            if name.startswith("--"):
                line = selector in {f".{each}" for each in LINES} and condition == ""
                root = selector == ":root" and condition in ("", DARK)
                if not (root or (line and name in ("--c", "--on-c"))):
                    found.append(
                        f"a custom property set outside the root and the line classes: {name}"
                        f" in {condition} {selector}"
                    )
    roots = sorted((condition, selector) for condition, selector, _ in rules if selector == ":root")
    if roots != [("", ":root"), (DARK, ":root")]:
        found.append(f"the custom properties are not set in one root and one dark block: {roots}")
    for condition, selector, declared in rules:
        names = [name for name, _ in declared if name.startswith("--")]
        if len(names) != len(set(names)):
            found.append(f"a custom property set twice in {condition} {selector}")
    listed = listed_rules()
    for key in sorted(real.keys() | listed.keys()):
        if key not in listed:
            found.append(
                f"a rule the test does not list sets what carries meaning: {key}: {real[key]}"
            )
        elif key not in real:
            found.append(f"a listed rule is not in the sheet as written: {key}")
        elif real[key] != listed[key]:
            found.append(f"{key} differs from its listing: {real[key]} against {listed[key]}")
    return found


def test_what_carries_meaning_is_set_only_in_the_listed_rules() -> None:
    assert meaning_rules_problems(static("demo.css")) == []


@pytest.mark.parametrize(
    "rule",
    [
        # The second review's probes.
        "p:not(.lede) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }",
        ".cite.line-3 { --c: var(--l1); }",
        ".source-card.line-2 { --c: var(--l1); }",
        ".turn .line-4 { --on-c: #ffffff; }",
        "@media (prefers-color-scheme: dark) and (min-width: 1px) { :root { --l2: #d8261c; } }",
        "html { --l2: #d8261c; }",
        ":root:root { --l2: #d8261c; }",
        ".where-to { overflow: hidden; max-height: 2px; }",
        ".thread { overflow: hidden; white-space: nowrap; }",
        ".route .station-next::after { background: var(--ring-fill); }",
        "a:focus-visible { outline-color: transparent; }",
        ".search-fold > summary::after { margin-left: 0; }",
        "@media (max-width: 30rem) {"
        " .status { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; } }",
        # Attempts to get past the list, of the kinds it names and of others.
        ".cite.line-3 { background: var(--l1); }",
        "#turns button { background: var(--l1); }",
        "@supports (display: grid) { .source-body { overflow: hidden; } }",
        ".turn { & .source-address { visibility: hidden; } }",
        ".source-address { color: var(--page); }",
        ".source-address { position: absolute; left: -9999px; }",
        ".source-address:first-line { color: transparent; }",
        ".source-body { content-visibility: hidden; }",
        ".source-address { font-size: 0.3em; }",
        ".source-address { text-indent: -9999px; }",
        ".source-address { letter-spacing: -1em; }",
        ".source-body { background: var(--text); }",
        ".source-address { transform: scale(0); }",
        ".source-address { filter: opacity(0); }",
        ".source-address { margin-top: -999px; }",
        "@layer base { .source-body { overflow: hidden; } }",
        "@media screen { .source-body { overflow: hidden; } }",
        ".station-ahead { animation: approach 1s infinite; }",
        ".source-address { display: none !important; }",
        ".answer-text { --c: var(--l1); }",
        ".question-sign { border-color: var(--page); }",
        ".source-card::after { content: ; position: absolute; inset: 0; background: var(--page); }",
        ":root { --l2: #d8261c; --l2: #0a57c2; }",
        "@keyframes approach { 50% { transform: scale(0); } }",
        # Four that got past the list's first form, and closed: a named colour, a font size
        # in units of the window, one through a token, and a token changed in its place
        # (held by the test of the colours above).
        ".source-address { color: white; }",
        ".source-address { font-size: 0.4vw; }",
        ".source-address { font-size: var(--s1); }",
    ],
)
def test_the_check_of_the_listed_rules_finds_each_rule_planted(rule: str) -> None:
    assert meaning_rules_problems(f"{static('demo.css')}\n{rule}\n") != []


# The pairs of tokens the page puts text or a mark on, in both themes: text at 4.5:1 at
# least, a mark at 3:1. The listed rules hold where each colour may be used; this holds the
# colours themselves, so that a token changed in its place cannot make a text unreadable.
TEXT_PAIRS = [
    ("--text", "--page"), ("--text", "--ground"), ("--grey", "--page"), ("--grey", "--ground"),
    ("--sign-text", "--sign"), ("--band-says", "--sign"), ("--notice-text", "--notice"),
    ("--page", "--text"),
]  # fmt: skip
MARK_PAIRS = [
    ("--rule-line", "--ground"), ("--rule-line", "--page"), ("--off", "--ground"),
    ("--off", "--page"), ("--faint", "--ground"), ("--focus", "--page"),
    ("--sign-rule", "--page"),
]  # fmt: skip


def token_contrast_problems(sheet: str) -> list[str]:
    found = []
    for theme, tokens in theme_tokens(sheet).items():
        for pairs, least in ((TEXT_PAIRS, 4.5), (MARK_PAIRS, 3.0)):
            for one, other in pairs:
                front, back = resolved(f"var({one})", tokens), resolved(f"var({other})", tokens)
                if front is None or back is None or contrast(front, back) < least:
                    found.append(f"{one} on {other} reads under {least}:1 in the {theme} theme")
    return found


def test_the_colours_the_page_puts_text_and_marks_on_read() -> None:
    assert token_contrast_problems(static("demo.css")) == []


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("  --text: #000000;", "  --text: #ffffff;"),
        ("  --grey: #5f6368;", "  --grey: #a3a3a0;"),
        ("    --page: #1d1d1b;", "    --page: #f2f2ee;"),
        ("  --notice-text: #000000;", "  --notice-text: #ffffff;"),
        ("    --off: #93938c;", "    --off: #3a3a36;"),
        ("  --band-says: #cfcfcf;", "  --band-says: #333333;"),
    ],
)
def test_the_check_of_the_colours_finds_each_token_changed_in_its_place(old: str, new: str) -> None:
    sheet = static("demo.css")
    assert sheet.count(old) == 1, old

    assert token_contrast_problems(sheet.replace(old, new)) != []


# The rules that may set what carries meaning, as the sheet writes them, each after a word
# on why. A change to one of them is a change to this list, to be reviewed.
LISTED = """
/* ---- The base of the page ---- */

/* The hidden attribute's own rule: it hides what the script or the markup means to hide, and
   the script never sets it on an address line (the node run checks the line is shown). */
[hidden] { display: none !important; }

/* Every box measured with its border; the ::before and ::after here draw nothing. */
*, *::before, *::after { box-sizing: border-box; }

/* The page's own ground and text, the colours every other rule stands on. */
body { background: var(--page); color: var(--text); }

/* A link takes the colour of the text around it; no citation is a link. */
a { color: inherit; }

/* A button takes the colour of the text around it; a citation sets its own numeral colour on
   its line, by its class. */
button { color: inherit; }

/* A control that cannot be used now is dimmed; only buttons take aria-disabled, and an address
   line is no button. */
button[aria-disabled="true"] { opacity: 0.45; }

/* The focus, for every control: a heavy outline in the focus colour. */
:focus-visible { outline: 3px solid var(--focus); outline-offset: 2px; }

/* The screen reader's copy of a text, kept off the screen on purpose; inside the address line
   it holds the whole address, beside the brief one that is shown. */
.visually-hidden {
  position: absolute;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}

/* ---- The band at the top ---- */

/* White type on the black band, which holds no address line. */
.band { background: var(--sign); color: var(--sign-text); }

/* The band's measure; it narrows, it cuts nothing, and the band holds no address line. */
.band-inner { max-width: var(--page-width); }

/* The band's sentence, light grey on the black band, which holds no address. */
.band-says { color: var(--band-says); }

/* The band's two links, underlined in white on black. */
.band-nav a { border-bottom: 2px solid var(--sign-text); color: var(--sign-text); }

/* The focus on the black band, in white. */
.band :focus-visible { outline-color: var(--sign-text); }

/* ---- The page's column ---- */

/* The page's measure, around every turn; it narrows, it cuts nothing, and no overflow is
   hidden on it. */
.app { max-width: var(--page-width); }

/* The note for a browser without script, in a black frame. */
.noscript { border: 4px solid var(--text); }

/* ---- The signs ---- */

/* A sign: white on black with its edge, which in the dark theme stands out from the page, the
   type tightened by a hundredth; its text wraps, and it holds the question, not an address. */
.sign, .question-sign {
  position: relative;
  border: 2px solid var(--sign-edge);
  background: var(--sign);
  color: var(--sign-text);
  white-space: pre-wrap;
}

/* The rule above a sign, as a station sign has. */
.sign::before, .question-sign::before {
  content: "";
  position: absolute;
  top: -12px;
  right: 0;
  left: 0;
  height: 3px;
  background: var(--sign-rule);
}

/* ---- The welcome ---- */

/* The welcome's measure for reading; it narrows, it cuts nothing. */
.lede { max-width: 60ch; }

/* The board of destinations, under a heavy rule. */
.examples { border-top: 4px solid var(--rule-line); }

/* A destination: a button on no ground of its own, with a rule under it and its ring placed
   beside it. */
.example {
  position: relative;
  border: 0;
  border-bottom: 2px solid var(--ground);
  background: transparent;
}

/* A destination's ring. */
.example::before {
  content: "";
  flex: none;
  width: 20px;
  height: 20px;
  border: 5px solid var(--rule-line);
  border-radius: 50%;
  background: var(--page);
}

/* A destination pointed at, on the strip's ground. */
.example:hover { background: var(--ground); }

/* The note on plain text, at a measure for reading; it narrows, it cuts nothing. */
.plain-text-note { max-width: 60ch; }

/* A stop of "What happens when you ask", in grey, placed for its ring. */
.steps li { position: relative; color: var(--grey); }

/* The line between the stops of "What happens when you ask". */
.steps li::before {
  content: "";
  position: absolute;
  top: 6px;
  bottom: -6px;
  left: 7px;
  width: 6px;
  background: var(--rule-line);
}

/* No line after the last of those stops. */
.steps li:last-child::before { content: none; }

/* The ring of each of those stops. */
.steps li::after {
  content: "";
  position: absolute;
  top: 3px;
  left: 0;
  width: 20px;
  height: 20px;
  border: 5px solid var(--rule-line);
  border-radius: 50%;
  background: var(--page);
}

/* The name of each of those stops, in the text's colour. */
.steps strong { color: var(--text); }

/* The folded copy of those stops, shown on a phone only; it holds no address line. */
.how-fold { display: none; }

/* The strip of stops, on the strip's ground; it holds the stops' names, not an address. */
.route { background: var(--ground); }

/* ---- The line of stops ---- */

/* A stop, placed for its ring and its stroke. */
.station { position: relative; }

/* The stroke from a stop to the next. */
.station::before {
  content: "";
  position: absolute;
  top: 7px;
  right: 0;
  left: 0;
  height: 6px;
  background: var(--rule-line);
}

/* No stroke after the last stop. */
.seg-none::before { content: none; }

/* A faint stroke to a stop not reached yet. */
.seg-faint::before { top: 9px; height: 2px; background: var(--faint); }

/* The dashes to a stop the line does not serve: a decline. */
.seg-dashed::before {
  background: repeating-linear-gradient(90deg, var(--off) 0 10px, transparent 10px 18px);
}

/* A stop's ring. */
.station::after {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  width: 20px;
  height: 20px;
  border: 5px solid var(--rule-line);
  border-radius: 50%;
  background: var(--ring-fill);
}

/* The next stop, filled. */
.station-next::after { background: var(--rule-line); }

/* A stop not reached yet: a thin, faint ring. */
.station-ahead::after { border-width: 3px; border-color: var(--faint); }

/* A stop the line does not serve: a grey ring. */
.station-off::after { border-color: var(--off); }

/* The bar across the line where it ended short. */
.station-end::after {
  top: -4px;
  width: 6px;
  height: 28px;
  border: 0;
  border-radius: 0;
  background: var(--rule-line);
}

/* ---- The reply ---- */

/* The reply's text at a measure for reading, its line breaks kept; it wraps, and it holds no
   address line. */
.answer-text, .stream { max-width: 60ch; white-space: pre-wrap; }

/* The caret at the end of the text while it is written. */
.stream.streaming::after {
  content: "";
  display: inline-block;
  width: 0.5em;
  height: 1em;
  margin-left: 0.1em;
  background: var(--text);
  vertical-align: -0.12em;
}

/* Text that broke off, marked by a dashed edge. */
.stream.unfinished { border-left: 6px dashed var(--off); }

/* The bars that stand for the text until it comes, at a measure; they hold no address. */
.pending { max-width: 60ch; }

/* One bar that stands for text until it comes, on the strip's ground, its height its whole
   drawing; it holds no text. */
.skeleton-line { height: 12px; background: var(--ground); }

/* A citation: its number on its line's colour, in a round bullet of a fixed height that holds
   only the number. */
.cite { height: 1.45em; border: 0; background: var(--c); color: var(--on-c); }

/* A citation pointed at, or marked with its source: an outline round it. */
.cite:hover, .cite.linked { outline: 3px solid var(--text); outline-offset: 2px; }

/* A focused citation: its ring close, so that the full stop before it stays clear. */
.cite:focus-visible { outline-offset: 0; }

/* The line that says a follow-up was searched for as typed, at a measure, with an edge; it
   holds no address. */
.searched { max-width: 60ch; border-left: 6px solid var(--rule-line); }

/* The box of a stored answer, at a measure, with an edge; it holds no address. */
.cached { max-width: 60ch; border-left: 6px solid var(--rule-line); }

/* A notice or a failure card, placed for its sign. */
.notice, .error-card { position: relative; }

/* Why a reply is not an answer, at a measure; it holds no address. */
.notice-why { max-width: 60ch; }

/* The ring of the sign for "no service". */
.notice-decline::before {
  content: "";
  position: absolute;
  top: 2px;
  left: 0;
  width: 34px;
  height: 34px;
  border: 5px solid var(--rule-line);
  border-radius: 50%;
}

/* The bar across the ring of that sign. */
.notice-decline::after {
  content: "";
  position: absolute;
  top: 16px;
  left: -4px;
  width: 42px;
  height: 6px;
  background: var(--rule-line);
  transform: rotate(-45deg);
}

/* The upright of the sign where a line ended. */
.notice-unfinished::before, .error-card::before, .turn-note::before {
  content: "";
  position: absolute;
  top: 2px;
  left: 14px;
  width: 6px;
  height: 22px;
  background: var(--rule-line);
}

/* The foot of that sign. */
.notice-unfinished::after, .error-card::after, .turn-note::after {
  content: "";
  position: absolute;
  top: 22px;
  left: 0;
  width: 34px;
  height: 6px;
  background: var(--rule-line);
}

/* What failed, at a measure; it holds no address. */
.error-text { max-width: 60ch; }

/* "Ask again", in a black frame, on no ground of its own. */
.again { border: 3px solid var(--text); background: transparent; }

/* "Ask again" pointed at: the page's colour on the black it fills with. */
.again:hover { background: var(--text); color: var(--page); }

/* The note of a stop by the reader, placed for its sign. */
.turn-note { position: relative; }

/* A check's notes, black on the yellow of a notice, at a measure; they hold no address. */
.check-notes { max-width: 56ch; background: var(--notice); color: var(--notice-text); }

/* The timetable under "All details", at a measure, under a rule; it holds no address. */
.facts { max-width: 60ch; border-top: 4px solid var(--rule-line); }

/* A label of the timetable, in grey. */
.facts dt { color: var(--grey); }

/* The button that copies the trace id, in a frame, on no ground of its own. */
.copy { border: 2px solid var(--text); background: transparent; }

/* A count with nothing in it takes no room; the count is not an address. */
.side-count:empty { display: none; }

/* ---- The sources ---- */

/* A source's line, placed for its stroke. */
.source-card { position: relative; }

/* A source's stroke, in its line's colour, down its entry. */
.source-card::before {
  content: "";
  position: absolute;
  top: 34px;
  bottom: 0;
  left: 14px;
  width: 6px;
  background: var(--c);
}

/* The last source's stroke stops short of the end. */
.source-card:last-child::before { bottom: 20px; }

/* A source's bullet: its number on its line's colour, of a fixed height that holds only the
   number. */
.source-number { height: 34px; background: var(--c); color: var(--on-c); }

/* A source's bullet when its citation is pointed at or pressed: an outline round it. */
.source-card.linked .source-number, .source-card.marked .source-number {
  outline: 3px solid var(--text);
  outline-offset: 2px;
}

/* A source brought into view by its citation: the strip's ground, and a wide outline of the
   same, round the text, cutting nothing. */
.source-card.marked .source-body {
  background: var(--ground);
  outline: 8px solid var(--ground);
}

/* A source's title, in the text's colour. */
.source-path a { color: var(--text); }

/* The mark between two headings above a passage, read as a space. */
.crumb + .crumb::before { content: " \u203a "; content: " \u203a " / " "; white-space: pre; }

/* A piece of the passage, cut to three lines on purpose: the address is its own line after it,
   in the same source's body, and the whole passage is folded away below. */
.source-snippet {
  overflow: hidden;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 3;
  line-clamp: 3;
}

/* The whole passage, on the strip's ground, its line breaks kept; it wraps. */
.passage-text { background: var(--ground); white-space: pre-wrap; }

/* A source's place before it comes: its stroke and bullet in the strip's colour. */
.source-card.placeholder::before, .source-card.placeholder .source-number {
  background: var(--ground);
}

/* ---- The fold of the page's own search ---- */

/* The words the fold's control adds when it holds passages, apart from its own words. */
.search-fold > summary::after {
  content: "Show them";
  content: "Show them" / "";
  margin-left: 0.4em;
  color: var(--text);
  font-weight: 700;
  text-decoration: underline;
  white-space: nowrap;
}

/* The same, when the fold is open. */
.search-fold[open] > summary::after { content: "Hide them"; content: "Hide them" / ""; }

/* The words the control adds when the fold holds no passage: what the search is. */
.search-fold.nothing-more > summary::after {
  content: "About this search";
  content: "About this search" / "";
}

/* The same, when the fold is open. */
.search-fold.nothing-more[open] > summary::after { content: "Hide"; content: "Hide" / ""; }

/* The state of the search with nothing to say takes no room; it is not an address. */
.search-state:empty { display: none; }

/* A passage found, placed for its grey ring. */
.passage-item { position: relative; }

/* The grey ring of a passage found and not cited: a stop the line does not serve. */
.passage-item::before {
  content: "";
  position: absolute;
  top: 2px;
  left: -38px;
  width: 18px;
  height: 18px;
  border: 4px solid var(--off);
  border-radius: 50%;
  background: var(--page);
}

/* A passage's title, in the text's colour. */
.passage-title a { color: var(--text); }

/* ---- The question box ---- */

/* The dock of the question box, on the page's ground, under the conversation; it holds no
   address. */
.composer { position: sticky; background: var(--page); }

/* The row above the box, at the box's width; it holds no address. */
.context-bar { max-width: 800px; }

/* "Start a new conversation" on one line; it is a button, not an address. */
#new-conversation { white-space: nowrap; }

/* The list of what is sent, on the page's ground: it scrolls within a height, so that its
   control stays on the screen; it holds questions, not addresses. */
.context-body {
  max-height: min(40vh, 18rem);
  overflow-y: auto;
  border: 3px solid var(--text);
  background: var(--page);
}

/* A plain button, in a frame, on no ground of its own. */
.ghost { border: 3px solid var(--text); background: transparent; }

/* A plain button pointed at: the page's colour on the black it fills with. */
.ghost:hover { background: var(--text); color: var(--page); }

/* The question box: one heavy frame on the page's ground, at its width; it holds the question,
   not an address. */
.field { max-width: 800px; border: 4px solid var(--text); background: var(--page); }

/* The focus of the question box, drawn round its frame. */
.field:focus-within { outline: 3px solid var(--focus); outline-offset: 3px; }

/* The question, growing to a height and then scrolling; no frame of its own, on the box's
   ground. */
textarea { max-height: 12rem; border: 0; background: transparent; color: var(--text); }

/* The placeholder of the question, in grey. */
textarea::placeholder { color: var(--grey); opacity: 1; }

/* The question's own outline gives way to the frame's, drawn by the rule above. */
textarea:focus-visible { outline: none; }

/* Ask and Stop, with no frame of their own. */
.send, .stop { border: 0; }

/* Ask: the page's colour on black. */
.send { background: var(--text); color: var(--page); }

/* Stop, on the page's ground, set off from the question by a heavy rule. */
.stop { border-left: 4px solid var(--text); background: var(--page); }

/* The row under the box, at the box's width; it holds no address. */
.composer-meta { max-width: 800px; }

/* The shorter words for a phone, not shown on a larger screen; they are a button's, not an
   address. */
.on-narrow { display: none; }

/* The footer's measure; it narrows, it cuts nothing. */
.site-footer { max-width: var(--page-width); }

/* ---- On a narrow screen: the stops run down the page ---- */

/* The steps beside the welcome give way to their folded copy. */
@media (max-width: 40rem) .how { display: none; }

/* The stroke from a stop to the next, downwards. */
@media (max-width: 40rem) .station::before {
  top: 0;
  right: auto;
  bottom: 0;
  left: 7px;
  width: 6px;
  height: auto;
}

/* The faint stroke, downwards. */
@media (max-width: 40rem) .seg-faint::before { left: 9px; width: 2px; }

/* The dashes of a decline, downwards. */
@media (max-width: 40rem) .seg-dashed::before {
  background: repeating-linear-gradient(180deg, var(--off) 0 10px, transparent 10px 18px);
}

/* The bar where the line ended, across the downward line. */
@media (max-width: 40rem) .station-end::after {
  top: 7px;
  left: -4px;
  width: 28px;
  height: 6px;
}

/* ---- On a phone ---- */

/* The words a phone does not show, kept for a screen reader; they are a control's words, not
   an address. */
@media (max-width: 30rem) .on-wide {
  position: absolute;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}

/* The hint under the box, kept for a screen reader; it is not an address. */
@media (max-width: 30rem) #question-hint {
  position: absolute;
  height: 1px;
  overflow: hidden;
  clip-path: inset(50%);
  white-space: nowrap;
}

/* ---- The one motion ---- */

/* The next stop's ring shrinks and grows, only for a reader who has not asked for less motion. */
@media (prefers-reduced-motion: no-preference) .station-next::after {
  animation: approach 1.2s ease-in-out infinite;
}
"""


# The layout the reviews asked for, where it is not a matter of meaning: the fold of the
# page's own search set in under source lines and not when there are none, its grey stops
# keeping their room, and on a phone the status line on a row of its own, the full width.


def layout_problems(sheet: str) -> list[str]:
    found = []
    rules = merged(sheet)
    if rules.get(("", ".search-fold"), {}).get("margin") != "4px 0 0 46px":
        found.append("the fold is not set in under the source lines")
    if rules.get(("", ".no-sources .search-fold"), {}).get("margin-left") != "0":
        found.append("the fold is set in where there are no source lines")
    if rules.get(("", ".no-sources .passages"), {}).get("padding-left") != "46px":
        found.append("the grey stops have no room where there are no source lines")
    phone = rules.get(("(max-width: 30rem)", ".composer-meta"), {})
    areas = phone.get("grid-template-areas", "").replace('"', "").split()
    if areas != ["count", "status"] or phone.get("grid-template-columns") != "minmax(0, 1fr)":
        found.append("on a phone the status line does not have a row of its own, the full width")
    return found


def test_the_layout_the_reviews_asked_for_holds() -> None:
    assert layout_problems(static("demo.css")) == []


@pytest.mark.parametrize(
    "rule",
    [
        ".no-sources .search-fold { margin-left: 46px; }",
        ".no-sources .passages { padding-left: 0; }",
        '@media (max-width: 30rem) { .composer-meta { grid-template-areas: "status count"; } }',
        ".search-fold { margin: 4px 0 0; }",
    ],
)
def test_the_check_of_the_layout_finds_each_rule_planted(rule: str) -> None:
    assert layout_problems(f"{static('demo.css')}\n{rule}\n") != []
