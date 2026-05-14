.PHONY: help install install-dev setup-dev format lint type-check security test test-cov test-fast pre-commit pre-commit-update clean build dev-check quick-check info

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install the package in editable mode
	pip install -e .

install-dev:  ## Install dev + all-extras
	pip install -e ".[dev,all]"

setup-dev: install-dev  ## install-dev + pre-commit hooks
	pre-commit install

format:  ## Format code with black + isort
	black src/ tests/
	isort src/ tests/

lint:  ## Ruff + black --check + isort --check
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:  ## mypy on the package
	mypy src/mlpipe

security:  ## bandit scan
	bandit -r src/ -f txt

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src/mlpipe --cov-report=term-missing --cov-report=html

test-fast:  ## Skip slow + integration tests
	pytest tests/ -v -m "not slow and not integration"

pre-commit:  ## Run all pre-commit hooks
	pre-commit run --all-files

pre-commit-update:  ## Update pinned pre-commit hooks
	pre-commit autoupdate

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build sdist + wheel
	python -m build
	twine check dist/*

dev-check: format lint type-check security test  ## Full pre-push check

quick-check: lint test-fast  ## Fast iteration check

info:  ## Show dev tool versions
	@python --version
	@pip --version
	@pre-commit --version 2>/dev/null || echo 'pre-commit: not installed'
	@black --version 2>/dev/null || echo 'black: not installed'
	@ruff --version 2>/dev/null || echo 'ruff: not installed'
	@pytest --version 2>/dev/null || echo 'pytest: not installed'
	@mypy --version 2>/dev/null || echo 'mypy: not installed'
