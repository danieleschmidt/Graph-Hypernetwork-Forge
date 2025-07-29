# Contributing to Graph Hypernetwork Forge

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/graph-hypernetwork-forge.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -e ".[dev]"`
5. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest`
4. Run linting: `black . && isort . && flake8`
5. Run type checking: `mypy graph_hypernetwork_forge`
6. Commit your changes: `git commit -m "Add: brief description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Code Style

- Follow PEP 8 (enforced by black and flake8)
- Use type hints for all functions
- Write docstrings for all public functions and classes
- Keep line length to 88 characters (black default)

## Testing

- Write tests for new functionality
- Maintain test coverage above 80%
- Use descriptive test names
- Tests should be fast and independent

## Commit Messages

Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests

## Pull Request Process

1. Update documentation if needed
2. Ensure all tests pass
3. Update CHANGELOG.md if applicable
4. Request review from maintainers

## Questions?

Open an issue or start a discussion for questions or suggestions.