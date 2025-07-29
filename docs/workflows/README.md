# GitHub Actions Workflows

This directory contains documentation for recommended GitHub Actions workflows.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Purpose**: Run tests, linting, and type checking on every push and PR.

**Triggers**:
- Push to main branch
- Pull requests to main branch

**Jobs**:
- **test**: Run pytest with coverage reporting
- **lint**: Run black, isort, flake8
- **typecheck**: Run mypy
- **security**: Run safety and bandit

**Matrix Strategy**:
- Python versions: 3.10, 3.11, 3.12
- OS: ubuntu-latest, windows-latest, macos-latest

### 2. Security Scanning (`security.yml`)

**Purpose**: Automated security vulnerability scanning.

**Schedule**: Daily at 3 AM UTC

**Scans**:
- Dependency vulnerabilities (pip-audit)
- Code security issues (bandit)
- Secret detection (truffleHog)
- Container scanning (if Docker images)

### 3. Release Automation (`release.yml`)

**Purpose**: Automated release process when version tags are pushed.

**Triggers**: Git tag push (v*)

**Steps**:
- Build package
- Run full test suite
- Create GitHub release
- Publish to PyPI (with manual approval)

## Integration Requirements

### Environment Variables
```yaml
env:
  PYTHON_VERSION: "3.10"
  PYTORCH_VERSION: "2.3.0"
```

### Secrets Setup
- `PYPI_API_TOKEN`: For package publishing
- `CODECOV_TOKEN`: For coverage reporting

### Cache Configuration
- pip cache: `~/.cache/pip`
- pytest cache: `.pytest_cache`
- mypy cache: `.mypy_cache`

## Workflow Templates

Create these workflows in `.github/workflows/`:

1. **ci.yml**: Primary CI pipeline
2. **security.yml**: Security scanning
3. **release.yml**: Release automation
4. **docs.yml**: Documentation building

## Manual Setup Required

Since GitHub Actions workflows cannot be automatically created, please:

1. Create `.github/workflows/` directory
2. Copy workflow templates from this documentation
3. Customize for your specific needs
4. Test workflows with a test PR

## Monitoring and Notifications

### Status Badges
Add to README.md:
```markdown
[![CI](https://github.com/yourusername/graph-hypernetwork-forge/workflows/CI/badge.svg)](https://github.com/yourusername/graph-hypernetwork-forge/actions)
[![Security](https://github.com/yourusername/graph-hypernetwork-forge/workflows/Security/badge.svg)](https://github.com/yourusername/graph-hypernetwork-forge/actions)
```

### Failure Notifications
Configure GitHub Actions to notify on failures:
- Email notifications
- Slack integration
- Discord webhooks

## Performance Optimization

### Caching Strategy
- Cache pip dependencies
- Cache pre-commit environments
- Cache model downloads (if applicable)

### Parallel Execution
- Run tests in parallel across Python versions
- Separate linting and testing jobs
- Use matrix builds for OS compatibility

## Best Practices

1. **Fail Fast**: Stop on first failure in CI
2. **Clear Outputs**: Use clear job and step names
3. **Secure Secrets**: Never log secret values
4. **Resource Limits**: Set appropriate timeouts
5. **Documentation**: Comment complex workflow steps