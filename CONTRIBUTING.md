# Contributing to Graph Hypernetwork Forge

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/graph-hypernetwork-forge.git
   cd graph-hypernetwork-forge
   ```

2. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Run tests**:
   ```bash
   pytest
   ```

4. **Run code quality checks**:
   ```bash
   black .
   ruff check .
   mypy src/
   ```

5. **Commit and push**:
   ```bash
   git add .
   git commit -m "Add your feature description"
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** with:
   - Clear description of changes
   - Reference to related issues
   - Updated tests and documentation

## Coding Standards

- **Code Style**: Use Black for formatting (line length: 100)
- **Linting**: Follow Ruff recommendations
- **Type Hints**: Use type hints for all functions and methods
- **Documentation**: Add docstrings to all public functions/classes
- **Testing**: Write tests for new functionality

## Pull Request Guidelines

- Keep PRs focused on a single feature or bug fix
- Include tests for new functionality
- Update documentation as needed
- Ensure CI passes before requesting review
- Reference related issues using `Fixes #123` or `Relates to #123`

## Reporting Issues

- Use the GitHub issue tracker
- Provide minimal reproducible examples
- Include system information and dependency versions
- Check existing issues before creating new ones

## Questions?

- Join our [Discord community](https://discord.gg/your-invite)
- Email: hypernetwork-forge@yourdomain.com