.PHONY: help install lint typecheck test test-all run leaderboard plots clean

PYTHON ?= python
TASKS  ?= abercrombie,proa,nys_judicial_ethics
PROVIDERS ?= local-qwen0p5b
LIMIT  ?= 30

help:
	@echo "make install                       - install package + dev deps via uv"
	@echo "make lint                          - ruff check + format check"
	@echo "make typecheck                     - mypy strict"
	@echo "make test                          - unit tests (no slow/needs_provider)"
	@echo "make test-all                      - all tests"
	@echo "make run TASKS=... PROVIDERS=... LIMIT=N"
	@echo "  example: make run TASKS=abercrombie,proa PROVIDERS=local-qwen0p5b,anthropic-haiku LIMIT=30"
	@echo "make leaderboard                   - aggregate all completed runs into a leaderboard"
	@echo "make plots                         - regenerate result charts"

install:
	uv sync --all-extras

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

typecheck:
	uv run mypy src

test:
	uv run pytest -m "not slow and not needs_provider"

test-all:
	uv run pytest

run:
	uv run lbmm run --tasks $(TASKS) --providers $(PROVIDERS) --limit $(LIMIT)

leaderboard:
	uv run lbmm leaderboard

plots:
	uv run lbmm plots

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +


.PHONY: pdf test-artifacts
pdf:
	cd docs/_report && pandoc research_report.md -o ../research_report.pdf --pdf-engine=xelatex || echo "pandoc + xelatex required; see https://pandoc.org/installing.html"

test-artifacts:
	uv run python ../../_meta/retrofit.py "$(notdir $(CURDIR))" "$(notdir $(CURDIR))"
