# ADR 0001: Record architecture decisions

- Status: Accepted
- Date: 2026-11-21

## Context

Design decisions made in conversation are lost within weeks. A reviewer reading the code can see what was built, but not which alternatives were rejected or why. Without that, the same debates repeat and nobody can tell whether a choice still fits once circumstances change.

## Decision

Every significant technical decision is written down as a short Architecture Decision Record (ADR) in `docs/adr/`.

- Files are numbered in order: `NNNN-short-title.md`.
- Each record has four parts: context, decision, alternatives considered, consequences.
- Status moves from Proposed to Accepted after review. An accepted record is not edited to change its meaning; a later decision gets a new record that supersedes the old one.

A decision is significant if reversing it later would mean rewriting more than one module, or if a reasonable engineer would ask "why did you choose this?".

## Alternatives considered

- **Explain decisions in pull request descriptions only.** Easy, but the reasoning is scattered across many pull requests and hard to find later.
- **One large design document that is edited over time.** Shows the current state well, but loses the history of what changed and why. We keep an architecture overview for the current state and use ADRs for the history.

## Consequences

- A few minutes of writing per decision.
- Trade-offs are reviewable before code is written.
- New contributors can read the records in order and understand how the design got here.
