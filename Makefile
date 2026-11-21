.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check snapshot chunks

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

test: ## Run the tests with coverage
	uv run pytest --cov --cov-report=term-missing

check: lint typecheck test ## Run everything CI runs

snapshot: ## Download the handbook at the pinned commit into data/
	uv run handbook-rag snapshot

chunks: ## Clean and chunk the downloaded handbook, and print a summary
	uv run handbook-rag chunks
