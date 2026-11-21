# Contributing

## Setup

You need [uv](https://docs.astral.sh/uv/) and `make`. uv installs the right Python version for you.

```bash
make install   # install dependencies and git hooks
make check     # lint, type check and fast tests: everything CI runs
```

Run `make` on its own to list every command.

## Workflow

1. **Start from an issue.** Every feature or fix implements an issue with a "done when" checklist. A small documentation correction can go straight to a pull request.
2. **Create a branch** named `type/short-description`, for example `feat/markdown-chunker`.
3. **Commit in small steps** using [Conventional Commits](https://www.conventionalcommits.org/): `type: summary` in the imperative mood. A git hook checks the format.
4. **Open a pull request** that links the issue with `Closes #N`, explains what changed and why, and says how it was tested.
5. **CI must be green.** Pull requests are merged by fast-forward, so every commit lands on `main` unchanged. Make each commit a meaningful step with a clear message.

Commit types used here:

| Type | Use it for |
|---|---|
| `feat` | New behaviour |
| `fix` | A bug fix |
| `docs` | Documentation only |
| `test` | Tests only |
| `refactor` | Code change that keeps behaviour the same |
| `perf` | A performance improvement |
| `build` | Dependencies and packaging |
| `ci` | CI configuration |
| `chore` | Maintenance that fits none of the above |

## Conventions

- **Tests.** New behaviour comes with tests. Tests live in `tests/` and must not depend on the developer's machine, such as a local `.env` file or network access. A test that loads a real model is marked `slow` and runs with `make test-slow`; everything else runs offline in seconds, using small stand-ins for the model and Qdrant's in-memory mode.
- **Settings.** Configuration goes through `handbook_rag.config.Settings`. Nothing else reads environment variables. Every new setting is added to `.env.example`.
- **Secrets.** Nothing secret is committed. `.env` is ignored by git, and a hook scans each commit for credentials.
- **Decisions.** A change to the design gets a decision record in `docs/adr/`, in the same pull request. Records are numbered and not rewritten later; a new record supersedes an old one.
- **Results.** A claim about quality, speed or cost comes with the numbers and the command that produced them.
