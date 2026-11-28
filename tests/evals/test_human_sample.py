"""Drawing a sample for a person to label, the page they label on, and what could tell a label.

The sets here are small ones made by the tests (``human_fixtures``). No label in them, and
no label a test gives as a person's, is anybody's.
"""

import base64
import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from handbook_rag.evals.human_sample import (
    BATCH_FILE,
    Batch,
    BatchError,
    Candidate,
    DrawnBatch,
    Plan,
    SampleError,
    Sets,
    candidates,
    draw,
    item_hash,
    items_text,
    key_entries,
    key_hash,
    leaks,
    make_batch,
    ordered,
    read_batch,
    read_batches,
    redraw,
    runs,
    split,
    telling,
    write_key,
)
from handbook_rag.evals.label_page import (
    PAGE_DATA_KEYS,
    SCRIPT,
    STYLE,
    content_security_policy,
    how_to_label,
    inert_json,
    page_leaks,
    render_label_page,
    telling_words_in,
    write_batch,
)
from tests.evals import human_fixtures as fixtures

PLAN = Plan()
RUBRICS = {"support": "Support rubric.", "correct": "Correct rubric."}


@pytest.fixture
def sets(tmp_path: Path) -> Sets:
    return fixtures.make_project(tmp_path)


def drawn_batch(sets: Sets, number: int = 1, earlier: list[Candidate] | None = None) -> DrawnBatch:
    return make_batch(
        number=number,
        seed=7,
        plan=PLAN,
        sets=sets,
        earlier=earlier or [],
        sources={"sets": "stand-in"},
    )


def written(tmp_path: Path, sets: Sets, number: int = 1) -> Batch:
    root = tmp_path / "batches"
    earlier = [c for d in redraw(read_batches(root), sets) for c in d.picked]
    write_batch(root, drawn_batch(sets, number, earlier), RUBRICS, sets)
    return read_batch(root / f"batch-{number}")


# What is drawn


def test_disputes_come_from_the_saved_checks_the_readers_and_the_set_s_unclear(
    sets: Sets,
) -> None:
    assert dict(sets.disputes) == {
        ("support", "q009", 2): ("unclear",),
        ("support", "q011", 1): ("readers",),
        ("support", "q013", 1): ("support_v1",),
        ("support", "q015", 1): ("support_v2",),
        ("correct", "q041-v1", 1): ("correct",),
        ("correct", "q060", 1): ("correct",),
    }


def test_only_what_the_rules_allow_can_be_drawn(sets: Sets) -> None:
    pool = candidates(sets)
    by_case = {(c.part, c.case_id, c.statement): c for c in pool}

    assert ("support", "q005", 2) not in by_case, "no_claim is not drawn unless disputed"
    assert ("support", "q001-v1", 2) not in by_case, "a supported statement of a copy is not"
    assert by_case[("support", "q009", 2)].stratum == "disputed", "unclear, so disputed"
    assert by_case[("support", "q013", 1)].stratum == "disputed"
    assert by_case[("support", "q007", 2)].stratum == "unsupported"
    assert by_case[("support", "q001-v1", 1)].kind == "number"
    assert by_case[("support", "q002", 1)].shape == "one"
    assert by_case[("support", "q003", 1)].shape == "several"
    assert by_case[("correct", "q041-v1", 1)].stratum == "disputed"
    assert by_case[("correct", "q043-v1", 1)].stratum == "not_correct"


def test_the_same_seed_draws_the_same_batch_and_another_seed_another(sets: Sets) -> None:
    pool = candidates(sets)

    first = draw(pool, earlier=[], batch=1, seed=7, plan=PLAN)
    again = draw(list(reversed(pool)), earlier=[], batch=1, seed=7, plan=PLAN)
    other = draw(pool, earlier=[], batch=1, seed=8, plan=PLAN)

    assert first == again, "the order of the pool does not matter"
    assert first != other


def test_the_disputed_items_come_first_as_many_as_there_are(sets: Sets) -> None:
    picked = draw(candidates(sets), earlier=[], batch=1, seed=7, plan=PLAN)
    groups = Counter((c.part, c.stratum) for c in picked)

    # Four disputed statements and two disputed answers exist, fewer than the plan allows.
    assert groups[("support", "disputed")] == 4
    assert groups[("correct", "disputed")] == 2
    # The rest fills the places: 12 statements half and half, 6 answers half and half.
    assert groups[("support", "unsupported")] == groups[("support", "supported")] == 6
    assert groups[("correct", "not_correct")] == groups[("correct", "correct")] == 3


def test_disputed_places_spread_over_the_sorts_of_dispute(sets: Sets) -> None:
    for seed in range(20):
        picked = draw(
            candidates(sets),
            earlier=[],
            batch=1,
            seed=seed,
            plan=Plan(disputed_statements=3, disputed_answers=1),
        )
        sorts = [c.kind for c in picked if c.part == "support" and c.stratum == "disputed"]
        assert len(sorts) == len(set(sorts)) == 3, (seed, sorts)


def test_an_odd_place_goes_to_the_side_the_disputed_items_hold_fewer_of() -> None:
    def disputed(*labels: str) -> list[Candidate]:
        return [
            Candidate("support", f"c{n}", 1, f"q{n}", "x", label, "disputed", "one")
            for n, label in enumerate(labels)
        ]

    faults = frozenset({"unsupported"})
    assert split(10, disputed("supported", "unclear"), faults) == (5, 5)
    assert split(11, disputed("supported", "no_claim"), faults) == (6, 5)
    assert split(11, disputed("unsupported", "unsupported", "supported"), faults) == (5, 6)
    assert split(11, disputed("unclear"), faults) == (6, 5), "a tie goes to the faults"


def test_answers_not_correct_hold_a_partial_and_an_incorrect(sets: Sets) -> None:
    for seed in range(30):
        picked = draw(candidates(sets), earlier=[], batch=1, seed=seed, plan=PLAN)
        labels = {c.expected for c in picked if c.stratum == "not_correct"}
        assert labels == {"partial", "incorrect"}, seed


def test_supported_statements_match_the_faulted_ones_in_the_shape_of_their_answer(
    sets: Sets,
) -> None:
    for seed in range(30):
        picked = draw(candidates(sets), earlier=[], batch=1, seed=seed, plan=PLAN)
        shapes = {
            group: Counter(c.shape for c in picked if c.stratum == group)
            for group in ("unsupported", "supported")
        }
        assert shapes["unsupported"] == shapes["supported"], seed


def test_no_question_is_drawn_twice_in_a_batch_or_in_a_later_one(sets: Sets) -> None:
    pool = candidates(sets)
    small = Plan(statements=6, answers=3)
    earlier: list[Candidate] = []
    for batch in (1, 2, 3):
        picked = draw(pool, earlier=earlier, batch=batch, seed=7, plan=small)
        questions = [c.question for c in picked]
        assert len(set(questions)) == len(questions), batch
        assert not set(questions) & {c.question for c in earlier}, batch
        earlier += picked


def test_a_later_batch_counts_what_earlier_ones_drew_and_passes_over_a_kind_that_ran_out(
    sets: Sets,
) -> None:
    pool = [
        c
        for c in candidates(sets)
        if not (c.kind == "number" and c.case_id != "q001-v1") and c.case_id != "q007"
    ]
    plan = Plan(statements=16, disputed_statements=0, answers=0)
    first = draw(pool, earlier=[], batch=1, seed=7, plan=plan)
    second = draw(pool, earlier=first, batch=2, seed=7, plan=plan)

    assert sorted(c.kind for c in first if c.stratum == "unsupported") == sorted(
        fixtures.SUPPORT_KINDS
    )
    kinds = Counter(c.kind for c in second if c.stratum == "unsupported")
    assert "number" not in kinds, "its one statement was drawn in batch 1"
    assert len(kinds) == 7
    assert sorted(kinds.values()) == [1, 1, 1, 1, 1, 1, 2]


def test_a_group_that_cannot_be_filled_is_refused_with_what_to_lower(sets: Sets) -> None:
    pool = candidates(sets)
    earlier: list[Candidate] = []
    with pytest.raises(SampleError) as refused:  # noqa: PT012 - batches until one runs short
        for batch in range(1, 8):
            earlier += draw(pool, earlier=earlier, batch=batch, seed=7, plan=PLAN)

    message = str(refused.value)
    left = re.search(r"needs (\d+) (.+?), and only (\d+) are left", message)
    assert left, message
    assert int(left.group(3)) < int(left.group(1))
    assert re.search(r"lower --(statements|answers)\.$", message)


def test_a_plan_that_is_not_whole_numbers_is_refused() -> None:
    with pytest.raises(SampleError):
        Plan(statements=-1)
    with pytest.raises(SampleError):
        Plan(statements=0, answers=0)


# The order


def test_telling_orders_are_those_that_group_or_keep_apart_large_groups() -> None:
    assert telling(list("aaabbb"))
    assert telling(list("ababab"))
    assert not telling(list("aabbab"))
    assert not telling(list("ab")), "groups too small to check"
    assert not telling(list("aaab"))
    assert not telling(list("aaaa"))
    assert runs(list("aabba")) == 3
    assert runs([]) == 0


@pytest.mark.parametrize(
    "groups",
    [
        ("unsupported",) * 1 + ("supported",) * 1,
        ("a",) * 3 + ("b",),
        ("a",) * 2 + ("b",) * 2,
        ("a",) * 5,
        (),
        ("a",) * 3 + ("b",) * 3,
        ("a",) * 4 + ("b",) * 4 + ("c",) * 3,
    ],
)
def test_ordering_always_ends_and_never_gives_a_telling_order(groups: tuple[str, ...]) -> None:
    picked = [
        Candidate("support", f"c{n}", 1, f"q{n}", "k", "x", g, "one") for n, g in enumerate(groups)
    ]
    for seed in range(25):
        order = [c.stratum for _, c in ordered(picked, batch=1, seed=seed)]
        assert sorted(order) == sorted(groups)
        assert not telling(order)


def test_a_group_of_one_is_not_kept_from_the_ends_by_the_order() -> None:
    picked = [
        Candidate("support", f"c{n}", 1, f"q{n}", "k", "x", g, "one")
        for n, g in enumerate(("a", "a", "a", "b"))
    ]
    places = {
        [c.stratum for _, c in ordered(picked, batch=1, seed=seed)].index("b") for seed in range(60)
    }

    assert places == {0, 1, 2, 3}, "a small group's place is left to the shuffle"


def test_the_order_is_shuffled_and_never_telling(sets: Sets) -> None:
    for seed in range(30):
        drawn = make_batch(number=1, seed=seed, plan=PLAN, sets=sets, earlier=[], sources={})
        entries = key_entries(drawn.key)
        for part in ("support", "correct"):
            order = [entries[i["item"]].stratum for i in drawn.items if i["part"] == part]
            assert not telling(order), (seed, part)
        ids = [item["item"] for item in drawn.items]
        assert ids == [f"A{n:02d}" for n in range(1, 17)] + [f"B{n:02d}" for n in range(1, 9)]


# What the person is shown, and what tells a label


def test_an_item_holds_what_is_shown_and_its_key_holds_the_rest(sets: Sets) -> None:
    drawn = drawn_batch(sets)
    item = next(i for i in drawn.items if i["part"] == "support")
    entry = drawn.key["items"][item["item"]]
    labelled = sets.case("support", entry["case"])
    marked = next(s for s in labelled.case.statements if s.number == entry["statement"])

    assert item["marked"] == entry["statement"]
    assert [s["text"] for s in item["statements"]] == [s.text for s in labelled.case.statements]
    assert [p["number"] for p in item["passages"]] == list(marked.cites)
    titles = {p.number: p.title for p in labelled.case.passages}
    assert [p["title"] for p in item["passages"]] == [titles[n] for n in marked.cites]
    assert item["sha256"] == item_hash(item)
    assert set(entry) == {
        "part",
        "case",
        "statement",
        "question",
        "stratum",
        "kind",
        "disputes",
        "expected",
    }


def test_a_drawn_batch_tells_no_label(sets: Sets) -> None:
    for seed in range(10):
        drawn = make_batch(number=1, seed=seed, plan=PLAN, sets=sets, earlier=[], sources={})
        page = render_label_page(drawn, RUBRICS)
        shown = items_text(drawn.items) + page + how_to_label(drawn)
        assert leaks(drawn.items, key_entries(drawn.key), sets, shown=shown) == [], seed
        assert page_leaks(page, drawn, RUBRICS) == [], seed


def _first_copy(items: list[dict[str, Any]], key: dict[str, Any]) -> dict[str, Any]:
    return next(i for i in items if key["items"][i["item"]]["case"].endswith("-v1"))


LEAKS: dict[str, Callable[[list[dict[str, Any]], dict[str, Any]], object]] = {
    "a label of the set": lambda items, key: items[0].update(expected="unsupported"),
    "a kind of case": lambda items, key: items[3].update(kind="number"),
    "a reader's verdict": lambda items, key: items[1]["statements"][0].update(readers={}),
    "a judge's verdict": lambda items, key: next(i for i in items if i.get("passages"))["passages"][
        0
    ].update(verdict="supported"),
    "an id of the sets in the text": lambda items, key: items[5].update(
        question=items[5]["question"] + " (q012-v1)"
    ),
    "a space after a copy's question (S19)": lambda items, key: _first_copy(items, key).update(
        question=_first_copy(items, key)["question"] + " "
    ),
    "a space after a copy's statement (S20)": lambda items, key: _first_copy(items, key)[
        "statements"
    ][0].update(text=_first_copy(items, key)["statements"][0]["text"] + " "),
    "a kind named in a copy's text": lambda items, key: _first_copy(items, key)["statements"][
        0
    ].update(text=_first_copy(items, key)["statements"][0]["text"][:-1] + " (uncited)."),
    "a group named in a text": lambda items, key: items[-1].update(
        answer=items[-1]["answer"] + " Disputed."
    ),
    "words other than the set's": lambda items, key: items[2]["statements"][0].update(
        text=items[2]["statements"][0]["text"].replace("days", "weeks")
    ),
}


@pytest.mark.parametrize("leak", LEAKS)
def test_each_kind_of_leak_in_the_items_is_found(sets: Sets, leak: str) -> None:
    drawn = drawn_batch(sets)
    items = copy.deepcopy(drawn.items)
    LEAKS[leak](items, drawn.key)

    found = leaks(items, key_entries(drawn.key), sets, shown=items_text(items))

    assert found, leak


def test_an_item_id_of_the_sets_is_found(sets: Sets) -> None:
    drawn = drawn_batch(sets)
    items = copy.deepcopy(drawn.items)
    entries = key_entries(drawn.key)
    entries["q004-v1"] = entries[items[4]["item"]]
    items[4]["item"] = "q004-v1"

    found = leaks(items, entries, sets, shown=items_text(items))

    assert "item 'q004-v1' does not have the neutral id of its place" in found
    assert "the id q004-v1 of a labelled set is in what is shown" in found


def _reordered(sets: Sets, arrange: Callable[[list[str]], list[int]]) -> list[str]:
    drawn = drawn_batch(sets)
    entries = key_entries(drawn.key)
    part_a = [i for i in drawn.items if i["part"] == "support"]
    order = arrange([entries[i["item"]].stratum for i in part_a])
    renamed = {}
    items = []
    for place, index in enumerate(order, start=1):
        item = dict(part_a[index])
        new_id = f"A{place:02d}"
        renamed[new_id] = entries[item["item"]]
        item["item"] = new_id
        items.append(item)
    items += [i for i in drawn.items if i["part"] == "correct"]
    renamed.update({k: v for k, v in entries.items() if k.startswith("B")})
    return leaks(items, renamed, sets, shown="")


def test_an_order_grouped_by_label_is_found(sets: Sets) -> None:
    found = _reordered(sets, lambda strata: sorted(range(len(strata)), key=strata.__getitem__))

    assert "the order of part A shows its groups" in found


def test_an_order_that_keeps_every_two_of_a_group_apart_is_found(sets: Sets) -> None:
    def alternate(strata: list[str]) -> list[int]:
        by_group: dict[str, list[int]] = {}
        for index, stratum in enumerate(strata):
            by_group.setdefault(stratum, []).append(index)
        lines = sorted(by_group.values(), key=len, reverse=True)
        # Deal one from each group in turn, largest first: no two of a group side by side.
        dealt: list[int] = []
        while any(lines):
            for line in lines:
                if line and (not dealt or strata[dealt[-1]] != strata[line[0]]):
                    dealt.append(line.pop(0))
        return dealt

    found = _reordered(sets, alternate)

    assert "the order of part A shows its groups" in found


# The page


def test_a_page_that_carries_more_data_is_found(sets: Sets) -> None:
    drawn = drawn_batch(sets)
    page = render_label_page(drawn, RUBRICS)
    groups = inert_json([e["stratum"] for e in drawn.key["items"].values()])
    planted = page.replace('"rubrics":', f'"groups":{groups},"rubrics":', 1)

    assert "groups" in " ".join(page_leaks(planted, drawn, RUBRICS))


def test_a_page_with_a_comment_or_a_word_about_the_make_up_is_found(sets: Sets) -> None:
    drawn = drawn_batch(sets)
    page = render_label_page(drawn, RUBRICS)
    labels = " ".join(e["expected"] for e in drawn.key["items"].values())

    commented = page.replace("<main", f"<!-- {labels} --><main", 1)
    worded = page.replace("<main", "<p>Half of each part is labelled with a fault.</p><main", 1)

    assert "the page holds a comment" in page_leaks(commented, drawn, RUBRICS)
    assert {"the page says 'half'", "the page says 'fault'"} <= set(
        page_leaks(worded, drawn, RUBRICS)
    )


def test_steps_that_tell_the_make_up_are_not_written(
    tmp_path: Path, sets: Sets, monkeypatch: pytest.MonkeyPatch
) -> None:
    def telling_steps(drawn: DrawnBatch) -> str:
        return how_to_label(drawn) + "\nHalf of each part is labelled with a fault.\n"

    monkeypatch.setattr("handbook_rag.evals.label_page.how_to_label", telling_steps)

    with pytest.raises(SampleError, match="the steps say 'half'"):
        write_batch(tmp_path, drawn_batch(sets), RUBRICS, sets)
    assert not (tmp_path / "batch-1").exists()


def test_the_page_and_the_steps_say_none_of_the_words_about_the_make_up(sets: Sets) -> None:
    drawn = drawn_batch(sets)

    assert telling_words_in(how_to_label(drawn)) == []
    assert telling_words_in(SCRIPT) == []


def test_a_batch_that_would_tell_a_label_is_not_written(tmp_path: Path, sets: Sets) -> None:
    drawn = drawn_batch(sets)
    drawn.items[0]["question"] += " See q002."

    with pytest.raises(SampleError, match="would tell a label"):
        write_batch(tmp_path, drawn, RUBRICS, sets)
    assert not (tmp_path / "batch-1").exists()


HOSTILE = '</script><img src=x onerror=alert(1)><a href="https://example.com/x">x</a> & //'


def hostile_batch(sets: Sets) -> DrawnBatch:
    drawn = drawn_batch(sets)
    drawn.items[0]["statements"][0]["text"] = HOSTILE
    drawn.items[0]["passages"][0]["text"] = HOSTILE
    drawn.items[-1]["answer"] = HOSTILE + "\u2028"
    return drawn


def data_block(page: str) -> tuple[str, str]:
    """The JSON data of a page, and the page without it."""
    match = re.search(r'<script type="application/json" id="batch-data">(.*?)</script>', page, re.S)
    assert match
    return match.group(1), page[: match.start(1)] + page[match.end(1) :]


def test_text_from_the_sets_goes_in_as_data_and_never_as_markup(sets: Sets) -> None:
    drawn = hostile_batch(sets)
    page = render_label_page(drawn, RUBRICS)
    data, rest = data_block(page)

    assert "<" not in data
    assert ">" not in data
    assert "&" not in data
    assert "//" not in data
    assert "\u2028" not in data
    assert json.loads(data)["items"] == drawn.items, "it reads back as exactly what was drawn"
    assert set(json.loads(data)) == PAGE_DATA_KEYS
    assert "onerror" not in rest
    assert "example.com" not in rest
    assert page.count("</script>") == 2


def test_the_page_names_no_other_origin_and_loads_nothing(sets: Sets) -> None:
    page = render_label_page(hostile_batch(sets), RUBRICS)
    _, rest = data_block(page)

    for found in ("://", "//", "url(", "@import", " src=", " href=", "<link", "<iframe"):
        assert found not in rest, found
    for call in ("fetch(", "XMLHttpRequest", "WebSocket", "EventSource", "sendBeacon", "import("):
        assert call not in SCRIPT, call


def test_the_page_puts_text_on_screen_as_text_only() -> None:
    for call in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "Function(",
    ):
        assert call not in SCRIPT, call
    assert 'setAttribute("on' not in SCRIPT
    assert ".src" not in SCRIPT


def test_the_policy_allows_only_the_page_s_own_style_and_script(sets: Sets) -> None:
    page = render_label_page(drawn_batch(sets), RUBRICS)

    def digest(text: str) -> str:
        return base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()

    policy = content_security_policy()
    assert policy.startswith("default-src 'none'; ")
    assert f"script-src 'sha256-{digest(SCRIPT)}'" in policy
    assert f"style-src 'sha256-{digest(STYLE)}'" in policy
    assert f"<script>{SCRIPT}</script>" in page
    assert f"<style>{STYLE}</style>" in page
    assert 'http-equiv="Content-Security-Policy"' in page
    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy


def test_inert_json_reads_back_as_the_same_text() -> None:
    text = "</script> <!-- a & b > c / d \u2028 \u2029"

    assert json.loads(inert_json({"t": text})) == {"t": text}


# A batch on disk, and its key


def test_a_batch_is_written_without_its_key(tmp_path: Path, sets: Sets) -> None:
    batch = written(tmp_path, sets)
    files = sorted(path.name for path in batch.directory.iterdir())
    meta = json.loads((batch.directory / BATCH_FILE).read_text())

    assert files == ["HOW-TO-LABEL.md", "batch.json", "items.jsonl", "label.html"]
    assert set(meta) == {"batch", "seed", "plan", "sources", "items_sha256", "joins_sha256"}
    assert len(meta["joins_sha256"]) == 64
    for path in batch.directory.iterdir():
        text = path.read_text()
        for word in ('"expected"', '"stratum"', '"kind"', '"disputes"', '"case"'):
            assert word not in text, (path.name, word)
        for case_id in sets.case_ids():
            assert not re.search(rf"(?<![\w-]){re.escape(case_id)}(?![\w-])", text), path.name


def test_a_batch_is_read_back_as_it_was_written(tmp_path: Path, sets: Sets) -> None:
    batch = written(tmp_path, sets)

    assert len(batch.items) == 24
    assert batch.seed == 7
    assert batch.plan == PLAN


def test_the_key_drawn_again_has_the_hash_written_when_the_batch_was_drawn(
    tmp_path: Path, sets: Sets
) -> None:
    batch = written(tmp_path, sets)

    (again,) = redraw([batch], sets)

    assert key_hash(again.key) == batch.joins_sha256
    assert again.items == list(batch.items.values())


def test_a_key_that_cannot_be_drawn_again_is_refused(tmp_path: Path, sets: Sets) -> None:
    batch = written(tmp_path, sets)
    support, correct = fixtures.load_cases(tmp_path)
    relabelled = Sets(support, correct, {k: v for k, v in sets.disputes.items() if k[1] != "q013"})

    with pytest.raises(BatchError, match="drawn again is not the one drawn first"):
        redraw([batch], relabelled)


def test_the_key_is_written_only_beside_labels_and_only_as_drawn(
    tmp_path: Path, sets: Sets
) -> None:
    batch = written(tmp_path, sets)
    (again,) = redraw([batch], sets)

    with pytest.raises(BatchError, match="its key waits for them"):
        write_key(batch, again.key)
    batch.labels_path.write_text("{}")
    with pytest.raises(BatchError, match="does not have the hash"):
        write_key(batch, {**again.key, "seed": 8})
    write_key(batch, again.key)

    assert json.loads(batch.key_path.read_text()) == again.key


def test_a_second_batch_shares_no_question_with_the_first(tmp_path: Path, sets: Sets) -> None:
    written(tmp_path, sets)
    written(tmp_path, sets, number=2)
    first, second = redraw(read_batches(tmp_path / "batches"), sets)

    assert not {c.question for c in first.picked} & {c.question for c in second.picked}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda d: (d / "items.jsonl").write_text((d / "items.jsonl").read_text() + "\n"),
            "changed",
        ),
        (lambda d: (d / "batch.json").unlink(), "missing"),
        (
            lambda d: (d / "batch.json").write_text(
                (d / "batch.json").read_text().replace('"batch": 1', '"batch": 2')
            ),
            "of batch 2",
        ),
    ],
)
def test_a_batch_that_was_changed_on_disk_is_refused(
    tmp_path: Path, sets: Sets, change: Callable[[Path], object], message: str
) -> None:
    batch = written(tmp_path, sets)
    change(batch.directory)

    with pytest.raises(BatchError, match=message):
        read_batch(batch.directory)


def test_an_item_edited_together_with_the_hash_of_its_file_is_refused(
    tmp_path: Path, sets: Sets
) -> None:
    batch = written(tmp_path, sets)
    items_path, meta_path = batch.directory / "items.jsonl", batch.directory / BATCH_FILE
    edited = items_path.read_text().replace("People in team", "Persons in team", 1)
    items_path.write_text(edited)
    meta = json.loads(meta_path.read_text())
    meta["items_sha256"] = hashlib.sha256(edited.encode()).hexdigest()[:16]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(BatchError, match="is not what was drawn"):
        read_batch(batch.directory)
