from handbook_rag.generate.prompt import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, build_user_content
from tests.support import make_chunk

LEAVE = make_chunk(
    "## Parental leave\n\nNew parents get sixteen weeks.\n\n- Paid in full\n- Starts at birth",
    heading_path=("Parental leave",),
)
EXPENSES = make_chunk("Expense a laptop up to one thousand dollars.", doc_id="finance/expenses.md")


def test_each_chunk_becomes_one_search_result_with_citations_enabled() -> None:
    content = build_user_content("How long is parental leave?", [LEAVE, EXPENSES])

    results = [block for block in content if block["type"] == "search_result"]
    assert [r["source"] for r in results] == [LEAVE.url, EXPENSES.url]
    assert [r["title"] for r in results] == ["Leave > Parental leave", "Leave"]
    assert all(r["citations"] == {"enabled": True} for r in results)


def test_the_question_comes_last() -> None:
    content = build_user_content("How long is parental leave?", [LEAVE, EXPENSES])

    assert content[-1] == {"type": "text", "text": "How long is parental leave?"}
    assert len(content) == 3


def test_a_chunk_is_split_into_citable_blocks_with_headings_attached() -> None:
    (result, _question) = build_user_content("q", [LEAVE])

    assert result["type"] == "search_result"
    assert [block["text"] for block in result["content"]] == [
        "## Parental leave\n\nNew parents get sixteen weeks.",
        "- Paid in full\n- Starts at birth",
    ]


def test_no_search_results_when_nothing_was_retrieved() -> None:
    assert build_user_content("q", []) == [{"type": "text", "text": "q"}]


def test_instructions_tell_the_model_how_to_decline_and_to_distrust_documents() -> None:
    assert NOT_FOUND_MESSAGE in SYSTEM_PROMPT
    assert "only those search results" in SYSTEM_PROMPT
    assert "not instructions" in SYSTEM_PROMPT
