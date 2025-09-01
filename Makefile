# HEP-ML-Templates Development Makefile
# ====================================

.PHONY: help install install-dev format lint type-check security test test-cov clean pre-commit setup-dev

help:  ## Show this help message
	@echo "HEP-ML-Templates Development Commands"
	@echo "====================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install:  ## Install the package in editable mode
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -e ".[dev,all]"

setup-dev: install-dev  ## Setup development environment
	pre-commit install
	@echo "✅ Development environment ready!"
	@echo "💡 Run 'make help' to see available commands"

# Code Quality
format:  ## Format code with black and isort
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatted"

lint:  ## Run all linters
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/
	@echo "✅ Linting complete"

type-check:  ## Run type checking with mypy
	mypy src/mlpipe
	@echo "✅ Type checking complete"

security:  ## Run security checks with bandit
	bandit -r src/ -f txt
	@echo "✅ Security scan complete"

# Testing
test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src/mlpipe --cov-report=term-missing --cov-report=html

test-fast:  ## Run fast tests only (skip slow/integration tests)
	pytest tests/ -v -m "not slow and not integration"

# Pre-commit
pre-commit:  ## Run all pre-commit hooks
	pre-commit run --all-files

pre-commit-update:  ## Update pre-commit hooks
	pre-commit autoupdate

# CI/CD Simulation
ci-local:  ## Run CI pipeline locally
	@echo "🔍 Running local CI simulation..."
	make lint
	make type-check
	make security
	make test-cov
	@echo "✅ Local CI simulation complete!"

# Pipeline Testing
test-pipelines:  ## Test all pipeline types
	@echo "🧪 Testing all pipelines..."
	cd test_runs && \
	for pipeline in xgb decision-tree ensemble neural torch autoencoder gnn; do \
		echo "Testing $$pipeline pipeline..."; \
		mlpipe install-local --target-dir ./test-$$pipeline pipeline-$$pipeline && \
		cd test-$$pipeline && pip install -e . && mlpipe run && cd ..; \
	done
	@echo "✅ All pipelines tested!"

# Cleanup
clean:  ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

# Documentation
docs-build:  ## Build documentation (if/when added)
	@echo "📚 Documentation building not yet implemented"

# Release
build:  ## Build package for distribution
	python -m build
	twine check dist/*
	@echo "✅ Package built and verified"

# Development workflow helpers
dev-check: format lint type-check security test  ## Run full development check
	@echo "✅ Full development check complete!"

quick-check: lint test-fast  ## Run quick checks for rapid iteration
	@echo "✅ Quick check complete!"

# Environment info
info:  ## Show development environment info
	@echo "HEP-ML-Templates Development Environment"
	@echo "======================================="
	@echo "Python: $$(python --version)"
	@echo "Pip: $$(pip --version)"
	@echo "Pre-commit: $$(pre-commit --version 2>/dev/null || echo 'Not installed')"
	@echo "Black: $$(black --version 2>/dev/null || echo 'Not installed')"
	@echo "Ruff: $$(ruff --version 2>/dev/null || echo 'Not installed')"
	@echo "Pytest: $$(pytest --version 2>/dev/null || echo 'Not installed')"
	@echo "MyPy: $$(mypy --version 2>/dev/null || echo 'Not installed')"
