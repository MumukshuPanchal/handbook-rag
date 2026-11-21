# ADR 0002: Use the GitLab handbook as the corpus

- Status: Proposed
- Date: 2026-11-21

## Context

The assistant needs a real company handbook to answer questions from. The corpus has to meet four requirements:

- an explicit licence that allows using the text and publishing short attributed excerpts in our evaluation set;
- a public source in Markdown in a git repository, so a snapshot can be pinned to a commit;
- enough pages, departments and overlapping topics that finding the right passage is hard;
- active maintenance, so that keeping an index up to date is a realistic problem.

We reviewed the licence and size of the public handbooks we could find. Figures come from repository listings on 2026-11-21.

| Handbook | Licence covering the text | Markdown pages | Maintained |
|---|---|---|---|
| GitLab | CC BY-SA 4.0, stated in the handbook itself | about 4,200 | Yes |
| PostHog | MIT, for the `contents/` folder | about 410 | Yes |
| TTS (US government) | CC0 public domain dedication | about 240 | No, archived |
| Sourcegraph | Unclear: an Apache-2.0 file, no statement about the text | about 880 | No, archived |
| Mattermost | CC BY-NC-SA 3.0 (non-commercial) | about 350 | Yes |
| Basecamp | None; all rights reserved | 16 | Yes |

## Decision

Use the GitLab handbook: the `content/handbook/` directory of <https://gitlab.com/gitlab-com/content-sites/handbook>, pinned to a single commit.

**Licence basis**, checked at commit `e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8`:

- The handbook's own "Handbook usage" page says: "Our handbook falls under the Attribution-ShareAlike 4.0 International license." ([source](https://gitlab.com/gitlab-com/content-sites/handbook/-/blob/e645aac6dc6fbb0b759b92ed32f016aaed5ef4b8/content/handbook/about/handbook-usage.md#L340))
- Contributions are accepted under terms that say: "All documentation content is licensed to this project under Creative Commons: CC BY-SA 4.0." ([source](https://gitlab.com/gitlab-org/dco/-/blob/master/README.md))
- Every published handbook page carries a CC BY-SA 4.0 badge.
- The repository's `LICENSE` file is MIT and, by its own heading, covers "the GitLab Software". The "All Rights Reserved" line in the site footer is produced by the site theme.

**How we use it:**

- A script downloads the Markdown files at the pinned commit. The handbook text is not stored in this repository.
- The evaluation set contains short quoted passages with their source path. That file is licensed CC BY-SA 4.0 and credits GitLab. The code stays under the MIT licence.
- Pages about individual team members (personal "README" pages) are left out of the index.
- The README names the source and its licence, and states that this project is not affiliated with or endorsed by GitLab.

## Alternatives considered

- **PostHog handbook.** The cleanest terms: an MIT grant for the content, written in the `LICENSE` file. It is about a tenth of the size, which would make retrieval too easy to show the effect of later improvements.
- **TTS handbook.** Public domain, so no conditions at all. Small, and no longer maintained.
- **Sourcegraph, Mattermost, Basecamp.** Unclear, non-commercial or absent licence terms.

## Consequences

- About 4,200 pages across 34 sections give a corpus where many pages cover related topics, so retrieval quality differences should be measurable.
- Share-alike applies to material derived from the text. The evaluation set therefore carries CC BY-SA 4.0, separately from the MIT-licensed code.
- The licence statement lives in the handbook text, not in the `LICENSE` file. The exact wording and commit are recorded here so the basis for our use stays verifiable if the upstream text changes.
- The handbook is widely read and may appear in a language model's training data. Answers are therefore judged against the retrieved passages, as described in ADR 0007.
- This record is our reading of the published terms, not legal advice.
