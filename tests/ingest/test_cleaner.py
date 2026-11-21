from handbook_rag.ingest.cleaner import clean_markdown, split_front_matter


def test_front_matter_is_separated_from_the_body() -> None:
    fields, body = split_front_matter(
        '---\ntitle: "Expenses"\ndescription: How to\n---\n\nBody text.\n'
    )

    assert fields == {"title": "Expenses", "description": "How to"}
    assert body.strip() == "Body text."


def test_page_without_front_matter_is_all_body() -> None:
    assert split_front_matter("Just text.\n") == ({}, "Just text.\n")


def test_invalid_front_matter_is_tolerated() -> None:
    fields, body = split_front_matter("---\ntitle: [unclosed\n---\nBody.\n")

    assert fields == {}
    assert body == "Body.\n"


def test_front_matter_that_is_not_a_mapping_is_ignored() -> None:
    assert split_front_matter("---\n- just\n- a list\n---\nBody.\n")[0] == {}


def test_html_comments_are_removed_even_across_lines() -> None:
    assert clean_markdown("Before.\n<!-- hidden\nnote -->\nAfter.") == "Before.\n\nAfter."


def test_shortcode_tags_are_dropped_and_their_content_kept() -> None:
    cleaned = clean_markdown(
        '{{% alert title="Important" color="warning" %}}\nSubmit within 30 days.\n{{% /alert %}}'
    )

    assert cleaned == "**Important**\nSubmit within 30 days."


def test_shortcode_caption_that_is_already_bold_is_not_wrapped_again() -> None:
    assert clean_markdown('{{% card header="**Compensation**" %}}') == "**Compensation**"


def test_self_closing_shortcodes_disappear() -> None:
    assert clean_markdown('Watch this.\n\n{{< youtube "VsgToca4oCw" >}}\n\nThen read on.') == (
        "Watch this.\n\nThen read on."
    )


def test_label_shortcode_keeps_its_name() -> None:
    assert clean_markdown('{{< label name="Visibility: Audit" color="#E24329" >}}') == (
        "**Visibility: Audit**"
    )


def test_links_keep_their_text_and_lose_their_target() -> None:
    assert clean_markdown('See [the policy](/handbook/finance/policy/ "Policy") today.') == (
        "See the policy today."
    )


def test_link_with_a_shortcode_as_target_keeps_its_text() -> None:
    assert clean_markdown('Read [this page]({{% ref "engaging#issues" %}}).') == "Read this page."


def test_reference_links_and_their_definitions() -> None:
    assert clean_markdown("See [the guide][guide].\n\n[guide]: https://example.com/guide") == (
        "See the guide."
    )


def test_images_are_removed() -> None:
    assert clean_markdown("Chart: ![quarterly chart](/images/q3.png) above.") == "Chart: above."


def test_autolinks_keep_the_address() -> None:
    assert clean_markdown("Go to <https://example.com/form>.") == "Go to https://example.com/form."


def test_html_tags_are_stripped_and_their_text_kept() -> None:
    cleaned = clean_markdown(
        "<details>\n<summary><strong>More</strong></summary>\n\nHidden text.\n</details>"
    )

    assert cleaned == "More\n\nHidden text."


def test_line_breaks_inside_a_table_keep_the_row_on_one_line() -> None:
    assert clean_markdown("| Laptop<br>Monitor | $1,000 |") == "| Laptop Monitor | $1,000 |"


def test_placeholders_in_angle_brackets_are_not_mistaken_for_html() -> None:
    assert clean_markdown("Replace <username> with your handle.") == (
        "Replace <username> with your handle."
    )


def test_inline_code_is_left_exactly_as_written() -> None:
    assert clean_markdown("Use `<div>` or `{{< alert >}}` here.") == (
        "Use `<div>` or `{{< alert >}}` here."
    )


def test_fenced_code_is_left_exactly_as_written() -> None:
    page = "```html\n<div>{{< keep >}}</div>   <!-- not here: comments are global -->\n```"

    assert "<div>{{< keep >}}</div>" in clean_markdown(page)


def test_indented_fenced_code_inside_a_list_is_left_alone() -> None:
    page = "- Step one:\n\n    ```bash\n    echo <b>hi</b>\n    ```"

    assert "echo <b>hi</b>" in clean_markdown(page)


def test_a_fence_is_not_closed_by_a_longer_info_line() -> None:
    page = "````markdown\n```python\n<b>still code</b>\n```\n````\n\n<b>prose</b>"

    cleaned = clean_markdown(page)

    assert "<b>still code</b>" in cleaned
    assert cleaned.endswith("prose")


def test_heading_anchors_and_kramdown_attributes_are_removed() -> None:
    assert clean_markdown("## Limits {#limits}\n\nText.\n{: .no_toc}\n\n[[_TOC_]]") == (
        "## Limits\n\nText."
    )


def test_list_indentation_is_preserved() -> None:
    assert clean_markdown("- parent\n  - child with [link](/x)\n    - grandchild") == (
        "- parent\n  - child with link\n    - grandchild"
    )


def test_extra_blank_lines_are_collapsed() -> None:
    assert clean_markdown("One.\n\n\n\n\nTwo.\n") == "One.\n\nTwo."


def test_padded_table_separator_is_shortened() -> None:
    table = "| Item | Limit |\n|" + "-" * 400 + "|:" + "-" * 300 + ":|\n| Laptop | 1000 |"

    assert clean_markdown(table) == "| Item | Limit |\n| --- | :---: |\n| Laptop | 1000 |"


def test_horizontal_rule_is_not_mistaken_for_a_table_separator() -> None:
    assert clean_markdown("Above.\n\n---\n\nBelow.") == "Above.\n\n---\n\nBelow."


def test_table_row_made_of_dashes_and_text_is_left_alone() -> None:
    assert clean_markdown("| - | Pipelines |") == "| - | Pipelines |"


def test_link_whose_text_is_inline_code_loses_its_target() -> None:
    assert clean_markdown("Ask in [`#help-desk`](https://chat.example.com/messages/C123).") == (
        "Ask in `#help-desk`."
    )
