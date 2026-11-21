"""Turn raw handbook Markdown into clean text for retrieval.

The handbook is written for a website generator, so pages carry things that are noise for
search: YAML front matter, template shortcodes, HTML, link targets and images. Cleaning
keeps the words a person would read and the Markdown structure the chunker relies on
(headings, lists, tables, code blocks).

Code is never touched: fenced blocks and inline code spans pass through unchanged, since
angle brackets and braces there are content, not markup.
"""

import re
from typing import Any

import yaml

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?\n)?---[ \t]*(?:\n|\Z)", re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_INLINE_CODE = re.compile(r"(`+)(?!`).+?(?<!`)\1(?!`)")
_LINK_DEFINITION = re.compile(r"^ {0,3}\[[^\]]+\]:\s+\S+.*$")

# {{< name args >}} or {{% name args %}}, opening, closing or self-closing.
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/)?\s*([\w./-]+)([^}]*?)\s*/?[%>]\}\}")
_CAPTION_ATTRIBUTE = re.compile(r'\b(title|header|summary|name)\s*=\s*"([^"]*)"')

# [`code`](target): the link text holds inline code, so it is handled before code spans
# are set aside.
_LINK_AROUND_CODE = re.compile(
    r"\[((?:[^\[\]`]|`[^`]*`)*`[^`]*`(?:[^\[\]`]|`[^`]*`)*)\]\((?:[^()]|\([^()]*\))*\)"
)
_TABLE_SEPARATOR_CELL = re.compile(r"^\s*(:?)-+(:?)\s*$")

_IMAGE = re.compile(r"!\[[^\]]*\]\((?:[^()]|\([^()]*\))*\)")
_INLINE_LINK = re.compile(r"\[((?:[^\[\]]|\[[^\[\]]*\])*)\]\((?:[^()]|\([^()]*\))*\)")
_REFERENCE_LINK = re.compile(r"\[((?:[^\[\]]|\[[^\[\]]*\])+)\]\[[^\]]*\]")
_AUTOLINK = re.compile(r"<((?:https?://|mailto:)[^<>\s]+)>")

_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_CELL_END = re.compile(r"</t[dh]\s*>", re.IGNORECASE)
_LINE_ENDING_TAG = re.compile(r"</(?:tr|p|div|summary|li|h[1-6])\s*>", re.IGNORECASE)
# Only real HTML tag names are stripped. A placeholder such as <username> in prose is kept.
_HTML_TAG = re.compile(
    r"</?(?:a|abbr|b|blockquote|button|center|code|col|colgroup|dd|del|details|div|dl|dt|em|"
    r"embed|figcaption|figure|font|footer|form|h[1-6]|header|hr|i|iframe|img|input|ins|kbd|"
    r"label|li|mark|nav|object|ol|p|path|pre|s|section|small|source|span|strike|strong|sub|"
    r"summary|sup|svg|table|tbody|td|tfoot|th|thead|tr|u|ul|video)\b[^<>]*>",
    re.IGNORECASE,
)
_KRAMDOWN_ATTRIBUTES = re.compile(r"\{:\s*[^{}]*\}")
_HEADING_ANCHOR = re.compile(r"\s*\{#[\w-]+\}\s*$")
_TABLE_OF_CONTENTS = re.compile(r"\[\[_TOC_\]\]")
_SPACES = re.compile(r"[ \t]{2,}")
_BLANK_LINES = re.compile(r"\n{3,}")


def split_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Separate the YAML block at the top of a page from its body.

    Returns the parsed fields and the body. A page without front matter, or with front
    matter that is not valid YAML, yields an empty dictionary.
    """
    match = _FRONT_MATTER.match(raw)
    if not match:
        return {}, raw
    try:
        fields = yaml.safe_load(match.group(1) or "")
    except yaml.YAMLError:
        fields = None
    return (fields if isinstance(fields, dict) else {}), raw[match.end() :]


def clean_markdown(body: str) -> str:
    """Remove website markup from a page body and return readable Markdown."""
    body = _HTML_COMMENT.sub("", body)
    body = _SCRIPT_OR_STYLE.sub("", body)

    cleaned: list[str] = []
    open_fence: str | None = None
    for line in body.splitlines():
        fence = _FENCE.match(line)
        if open_fence is not None:
            cleaned.append(line)
            if fence and _closes(fence, open_fence):
                open_fence = None
            continue
        if fence:
            open_fence = fence.group(1)
            cleaned.append(line)
            continue
        if _LINK_DEFINITION.match(line):
            continue
        cleaned.append(_shorten_table_separator(_clean_line(line)))

    text = "\n".join(cleaned)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _closes(fence: re.Match[str], open_fence: str) -> bool:
    """A fence closes with the same character, at least as many times, and nothing after."""
    marker, rest = fence.group(1), fence.group(2)
    return marker[0] == open_fence[0] and len(marker) >= len(open_fence) and not rest.strip()


def _clean_line(line: str) -> str:
    """Clean one line that is outside any fenced code block, leaving inline code alone."""
    # Indentation gives nested lists their structure, so it is kept as it is.
    content = line.lstrip(" \t")
    indent = line[: len(line) - len(content)]
    content = _LINK_AROUND_CODE.sub(r"\1", content)

    parts: list[str] = []
    position = 0
    for code in _INLINE_CODE.finditer(content):
        parts.append(_clean_prose(content[position : code.start()]))
        parts.append(code.group(0))
        position = code.end()
    parts.append(_clean_prose(content[position:]))
    cleaned = "".join(parts)

    if cleaned.startswith("#"):
        cleaned = _HEADING_ANCHOR.sub("", cleaned)
    return (indent + cleaned).rstrip()


def _shorten_table_separator(line: str) -> str:
    """Rewrite a table's separator row with three dashes per column.

    Authors pad these rows so the table lines up in an editor. A row of a thousand
    dashes carries no meaning but costs hundreds of tokens.
    """
    stripped = line.strip()
    if "|" not in stripped or "-" not in stripped or set(stripped) - set("|-: \t"):
        return line
    cells = stripped.strip("|").split("|")
    matches = [_TABLE_SEPARATOR_CELL.match(cell) for cell in cells]
    if not all(matches):
        return line
    indent = line[: len(line) - len(line.lstrip())]
    short = " | ".join(f"{m.group(1)}---{m.group(2)}" for m in matches if m)
    return f"{indent}| {short} |"


def _clean_prose(text: str) -> str:
    if not text:
        return text
    text = _SHORTCODE.sub(_replace_shortcode, text)
    text = _IMAGE.sub("", text)
    text = _INLINE_LINK.sub(r"\1", text)
    text = _REFERENCE_LINK.sub(r"\1", text)
    text = _AUTOLINK.sub(r"\1", text)
    text = _LINE_BREAK.sub(" ", text)
    text = _CELL_END.sub(" | ", text)
    text = _LINE_ENDING_TAG.sub(" ", text)
    text = _HTML_TAG.sub("", text)
    text = _KRAMDOWN_ATTRIBUTES.sub("", text)
    text = _TABLE_OF_CONTENTS.sub("", text)
    # Removed markup leaves runs of spaces behind.
    return _SPACES.sub(" ", text)


def _replace_shortcode(match: re.Match[str]) -> str:
    """Drop a shortcode tag, keeping any caption it carries.

    ``{{% alert title="Important" %}}`` becomes ``**Important**``. The text between an
    opening and a closing tag is ordinary Markdown and stays where it is.
    """
    is_closing, name, arguments = match.group(1), match.group(2), match.group(3)
    if is_closing:
        return ""
    for attribute, value in _CAPTION_ATTRIBUTE.findall(arguments):
        if attribute == "name" and name != "label":
            continue
        caption = value.strip()
        if caption:
            return caption if "*" in caption else f"**{caption}**"
    return ""
