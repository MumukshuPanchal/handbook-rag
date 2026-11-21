# ADR 0009: Structure-aware chunking with a token budget

- Status: Accepted
- Date: 2026-11-21

## Context

Retrieval returns chunks, not pages. How a page is split decides what can be found and what the model gets to read.

Three properties of this corpus shaped the choice. All were measured on the pinned snapshot.

- **Pages vary enormously.** The median page is about 6,000 characters. The largest is over 160,000.
- **Authors already structure pages with headings.** There are about 50,000 sections with a median length of about 530 characters. One chunk per heading would give many chunks too small to mean anything alone.
- **The source is written for a website generator.** Pages carry template shortcodes, HTML and link targets, which are noise for search, and a large share of the content sits in tables.

The embedding model reads at most 512 tokens and silently ignores anything beyond that.

## Decision

Clean each page, then split it along its own structure, measuring size in the embedding model's tokens.

**Cleaning**

- Drop front matter, HTML comments, template shortcodes, HTML tags, images and link targets. Keep link text, shortcode captions and all Markdown structure. Code blocks and inline code are left untouched.
- Rewrite padded table separator rows with three dashes per column.
- Leave out personal pages about individual team members, and pages with under 200 characters of text.

**Chunking**

- A Markdown parser finds the blocks of a page, so a chunk never starts inside a code block or a table row unless that block alone exceeds the budget.
- Sections are packed into chunks of up to 320 tokens. A level-2 heading starts a new chunk. A piece under 60 tokens is merged into a neighbour.
- A section over the budget is cut at block boundaries, or at line boundaries inside a single oversized block. Each piece repeats up to 40 tokens from the end of the piece before it. A piece cut from the middle of a table also repeats the table's header.
- Every chunk records its heading path and its character offsets in the page. The text sent to the embedding model is prefixed with the page title and heading path.

**Result on the snapshot:** 3,846 pages become 38,881 chunks. The median chunk is 225 tokens, the 99th percentile is 336, and the largest is 396. None exceeds the model's limit.

## Alternatives considered

- **Fixed-size windows with overlap.** The simplest approach. It cuts through sentences, tables and code, and chunks carry no heading.
- **One chunk per heading.** Respects structure, but many sections are too small to stand alone and a few exceed the model's limit.
- **Semantic chunking**, which splits where the similarity between neighbouring sentences drops. It needs an embedding pass over every page before indexing, and its boundaries are hard to explain or reproduce. These pages already have structure written by their authors.
- **Model-written context for every chunk.** Some teams report better retrieval from it. It costs a model call per chunk, about 39,000 here. It can be tested later against this baseline.

## Consequences

- Chunks describe themselves: each carries its headings, and table pieces keep their header.
- The budget, the minimum and the overlap are settings, and milestone M3 compares values. Changing any of them changes every chunk identifier, so indexes built with different settings never mix.
- The budget is approximate by design. A chunk may exceed 320 tokens by up to 25% when that avoids leaving a tiny fragment.
- Known losses:
  - Content pulled in by `include` shortcodes, about 290 uses, lives outside the snapshot and is missing.
  - Link targets and images are dropped, so an answer can name a form but cannot link to it.
  - Tables generated from data files, such as team lists, are absent.
