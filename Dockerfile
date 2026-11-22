# syntax=docker/dockerfile:1
#
# The HTTP API as a container image. Two stages: the first installs everything into a
# virtual environment, the second copies that environment and nothing else, so build
# tools and caches never reach the image that runs.

# ---- build ------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS build

# uv installs exactly what the lock file lists. The version matches the one used for
# development.
COPY --from=ghcr.io/astral-sh/uv:0.10.3 /uv /usr/local/bin/uv

WORKDIR /app
# Compile to bytecode once, here, so the service starts faster. Copy files instead of
# linking them, because the download cache is not part of the image.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, in a layer of their own. It is rebuilt only when the lock file
# changes, not on every change to the code.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

# ---- run --------------------------------------------------------------------------
FROM python:3.12-slim-bookworm

# git downloads the handbook snapshot (`handbook-rag snapshot`).
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# The service does not run as root.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /home/app/.cache \
    && chown -R app:app /app /home/app/.cache

WORKDIR /app
COPY --from=build --chown=app:app /app/.venv /app/.venv

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
