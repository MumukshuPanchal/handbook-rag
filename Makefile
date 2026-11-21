.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-slow check snapshot chunks qdrant traces up index search ask serve eval-retrieval eval-compare eval-table eval-gate golden-check golden-review

help: ## List the available commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-10s %s\n", $$1, $$2}'

install: ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

lint: ## Check code style and common bugs, without changing files
	uv run ruff check .
	uv run ruff format --check .

format: ## Fix style problems that can be fixed automatically
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Check type annotations
	uv run mypy

test: ## Run the fast tests with coverage
	uv run pytest --cov --cov-report=term-missing

test-slow: ## Run the tests that download a model
	uv run pytest -m slow

check: lint typecheck test ## Run everything CI runs

snapshot: ## Download the handbook at the pinned commit into data/
	uv run handbook-rag snapshot

chunks: ## Clean and chunk the downloaded handbook, and print a summary
	uv run handbook-rag chunks

qdrant: ## Start the Qdrant vector store in Docker
	docker compose up --detach --wait qdrant

up: ## Start the whole system in Docker: API, vector store and trace backend
	docker compose up --detach --build

traces: ## Start the trace backend in Docker; its web page is at http://localhost:16686
	docker compose up --detach --wait jaeger

index: ## Embed the chunks and store them in Qdrant
	uv run handbook-rag index

search: ## Search the index, for example: make search q="how do I expense a laptop"
	uv run handbook-rag search "$(q)"

serve: ## Run the HTTP API at http://127.0.0.1:8000; its documentation is at /docs
	uv run handbook-rag serve

ask: ## Answer a question with citations, for example: make ask q="how do I expense a laptop"
	uv run handbook-rag ask "$(q)"

eval-retrieval: ## Score the default retrieval on the golden set, as the run named "current"
	uv run handbook-rag eval-retrieval --variant current

eval-compare: ## Compare two runs: make eval-compare base=baseline candidate=v1
	uv run handbook-rag eval-compare $(base) $(candidate)

eval-table: ## Print a Markdown table of every evaluation run
	uv run handbook-rag eval-table --metric hit@5 --metric hit@1 --metric mrr@10 --metric ndcg@10

eval-gate: ## The CI check: score the default retrieval on a slice of the handbook, in memory
	uv run handbook-rag eval-gate

golden-check: ## Check every golden set label against the corpus
	uv run handbook-rag golden-check

golden-review: ## Write a page for reviewing the golden set by eye
	uv run handbook-rag golden-review
