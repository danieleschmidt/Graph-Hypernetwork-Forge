# Development Guide

## Setup Instructions

### Prerequisites
- Python 3.10+
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation

1. **Clone and setup**:
   ```bash
   git clone https://github.com/yourusername/graph-hypernetwork-forge.git
   cd graph-hypernetwork-forge
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

## Development Workflow

### Code Quality Tools

- **Black**: Code formatting (`black .`)
- **isort**: Import sorting (`isort .`)
- **flake8**: Linting (`flake8 graph_hypernetwork_forge tests`)
- **mypy**: Type checking (`mypy graph_hypernetwork_forge`)

### Testing

- **Run tests**: `pytest`
- **With coverage**: `pytest --cov=graph_hypernetwork_forge`
- **Specific test**: `pytest tests/test_specific.py::test_function`

### Pre-commit Hooks

Automatically run on commit:
- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Black formatting
- Import sorting
- Linting
- Type checking

## Project Structure

```
graph_hypernetwork_forge/
├── models/           # Core model implementations
├── data/            # Data loading and processing
├── utils/           # Utility functions
tests/               # Test suite
docs/                # Documentation
configs/             # Configuration files (future)
scripts/             # Training and evaluation scripts (future)
```

## Architecture Overview

1. **Text Encoder**: Processes node descriptions
2. **Hypernetwork**: Generates GNN weights
3. **Dynamic GNN**: Applies generated weights
4. **Zero-Shot Adapter**: Handles domain transfer

## Debugging Tips

- Use `pdb.set_trace()` for debugging
- Enable verbose logging: `PYTHONPATH=. python -m pytest -v -s`
- GPU debugging: `CUDA_LAUNCH_BLOCKING=1`

## Performance Profiling

```python
import cProfile
cProfile.run('your_function()', 'profile_output')
```

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.1.0`
4. Push: `git push origin v0.1.0`