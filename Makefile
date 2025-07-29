# Graph Hypernetwork Forge - Development Automation
.PHONY: help install install-dev test test-fast test-coverage lint format security clean docker-build docker-test benchmark docs

# Default target
help:
	@echo "Graph Hypernetwork Forge - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  setup-pre-commit Install pre-commit hooks"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test             Run all tests with coverage"
	@echo "  test-fast        Run tests without slow tests"
	@echo "  test-coverage    Generate HTML coverage report"
	@echo "  test-security    Run security tests"
	@echo "  test-performance Run performance benchmarks"
	@echo ""
	@echo "Code Quality Commands:"
	@echo "  lint             Run all linting tools"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run mypy type checking"
	@echo "  security         Run security analysis"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build     Build all Docker images"
	@echo "  docker-test      Run tests in Docker"
	@echo "  docker-dev       Start development environment"
	@echo "  docker-clean     Clean Docker images and volumes"
	@echo ""
	@echo "Documentation Commands:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean            Clean build artifacts and cache"
	@echo "  update-deps      Update all dependencies"
	@echo "  profile          Profile code performance"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e .[dev,security,docs,performance]

setup-pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

# Testing
test:
	pytest tests/ -v --cov=graph_hypernetwork_forge --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow" --cov=graph_hypernetwork_forge --cov-report=term-missing

test-coverage:
	pytest tests/ --cov=graph_hypernetwork_forge --cov-report=html
	@echo "Coverage report generated in htmlcov/"

test-security:
	pytest tests/ -v -m security

test-performance:
	pytest tests/ -v -m performance --benchmark-only --benchmark-sort=mean

# Code Quality
lint:
	@echo "Running flake8..."
	flake8 graph_hypernetwork_forge/ tests/
	@echo "Running black check..."
	black --check graph_hypernetwork_forge/ tests/
	@echo "Running isort check..."
	isort --check-only graph_hypernetwork_forge/ tests/
	@echo "Running mypy..."
	mypy graph_hypernetwork_forge/

format:
	@echo "Formatting with black..."
	black graph_hypernetwork_forge/ tests/
	@echo "Sorting imports with isort..."
	isort graph_hypernetwork_forge/ tests/

type-check:
	mypy graph_hypernetwork_forge/

security:
	@echo "Running bandit security scan..."
	bandit -r graph_hypernetwork_forge/ -f json -o bandit-report.json || true
	@echo "Running safety check..."
	safety check --json --output safety-report.json || true
	@echo "Security reports generated: bandit-report.json, safety-report.json"

# Docker
docker-build:
	docker-compose build

docker-test:
	docker-compose run --rm test

docker-dev:
	docker-compose up dev

docker-clean:
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Documentation
docs:
	@if [ -d "docs/source" ]; then \
		cd docs && make html; \
	else \
		echo "Sphinx documentation not set up yet. Run 'make setup-docs' first."; \
	fi

docs-serve:
	@if [ -d "docs/_build/html" ]; then \
		cd docs/_build/html && python -m http.server 8000; \
	else \
		echo "Documentation not built. Run 'make docs' first."; \
	fi

setup-docs:
	@mkdir -p docs/source
	@cd docs && sphinx-quickstart -q -p "Graph Hypernetwork Forge" -a "Daniel Schmidt" --ext-autodoc --ext-viewcode --makefile --batchfile source

# Utilities
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade pip
	pip install --upgrade -e .[dev,security,docs,performance]
	pre-commit autoupdate

profile:
	@echo "Profiling with py-spy (requires py-spy installation)..."
	@echo "Run your application and then: py-spy top --pid <PID>"
	@echo "Or use line_profiler: kernprof -l -v your_script.py"

# Development workflow shortcuts
dev-setup: install-dev setup-pre-commit
	@echo "Development environment setup complete!"

dev-check: format lint test-fast
	@echo "Development checks passed!"

ci-check: lint type-check test security
	@echo "CI checks completed!"

benchmark:
	@echo "Running performance benchmarks..."
	pytest tests/ -v -m performance --benchmark-only --benchmark-sort=mean --benchmark-json=benchmark.json

# Release preparation
prepare-release: clean ci-check docs
	@echo "Release preparation complete!"
	@echo "Don't forget to:"
	@echo "1. Update version in pyproject.toml"
	@echo "2. Update CHANGELOG.md"
	@echo "3. Create and push git tag"

# Dependency vulnerability check
check-vulnerabilities:
	@echo "Checking for known vulnerabilities..."
	safety check
	@echo "Checking outdated packages..."
	pip list --outdated