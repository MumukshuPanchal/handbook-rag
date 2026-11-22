# syntax=docker/dockerfile:1@sha256:572cebbc2b15e6f84c077742952674889cb63cd5e9b80abdffcbe587b6f19183
#
# The HTTP API as a container image. Two stages: the first installs everything into a
# virtual environment, the second copies that environment and nothing else, so build
# tools and caches never reach the image that runs.
#
# Images are pinned to a digest, with the tag in a comment, for the same reason the CI
# actions are pinned to a commit: a moved tag cannot change what is built here.

# ---- build ------------------------------------------------------------------------
# python:3.12-slim-bookworm
FROM python@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df AS build

# uv installs exactly what the lock file lists. The version matches the one used for
# development. ghcr.io/astral-sh/uv:0.10.3
COPY --from=ghcr.io/astral-sh/uv@sha256:7a88d4c4e6f44200575000638453a5a381db0ae31ad5c3a51b14f8687c9d93a3 /uv /usr/local/bin/uv

WORKDIR /app
# Compile to bytecode once, here, so the service starts faster and never needs to write
# next to its own code. Copy files instead of linking them, because the download cache
# is not part of the image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, in a layer of their own. It is rebuilt only when uv.lock or
# pyproject.toml changes, not on every change to the code.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- run --------------------------------------------------------------------------
# python:3.12-slim-bookworm
FROM python@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

# git downloads the handbook snapshot (`handbook-rag snapshot`).
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# The service does not run as root, and it owns only what it has to write: the handbook
# snapshot and the model files. Its own code belongs to root and is read-only to it.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /home/app/.cache \
    && chown app:app /app/data /home/app/.cache

WORKDIR /app
COPY --from=build /app/.venv /app/.venv

# Inside a container the API must accept connections from outside it. Compose decides
# which address of the host that is published on.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HANDBOOK_RAG_API_HOST=0.0.0.0 \
    HF_HOME=/home/app/.cache/huggingface \
    FASTEMBED_CACHE_PATH=/home/app/.cache/fastembed

USER app
EXPOSE 8000
CMD ["handbook-rag", "serve"]
