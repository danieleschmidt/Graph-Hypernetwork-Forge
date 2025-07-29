# CI/CD Workflow Documentation

This directory contains templates and documentation for GitHub Actions workflows.

## Required Workflows

### 1. Continuous Integration (`ci.yml`)

**Triggers**: Pull requests, pushes to main
**Jobs**:
- Code quality checks (black, ruff, mypy)
- Test suite execution across Python versions
- Coverage reporting
- Security scanning

**Required secrets**:
- `CODECOV_TOKEN` (optional, for coverage reporting)

### 2. Security Scanning (`security.yml`)

**Triggers**: Daily schedule, pull requests
**Jobs**:
- Dependency vulnerability scanning with `safety`
- SAST scanning with `bandit`
- License compliance checking
- Container scanning (if applicable)

### 3. Release (`release.yml`)

**Triggers**: Tagged releases
**Jobs**:
- Build and test package
- Publish to PyPI
- Create GitHub release with changelog
- Update documentation

**Required secrets**:
- `PYPI_API_TOKEN`
- `GITHUB_TOKEN` (automatically provided)

## Workflow Templates

Copy these templates to `.github/workflows/` and customize as needed:

### Basic CI Template
```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run quality checks
      run: make quality
    
    - name: Run tests
      run: make test-cov
```

### Security Template
```yaml
name: Security

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC
  pull_request:

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: pip install safety bandit
    
    - name: Run safety check
      run: safety check
    
    - name: Run bandit
      run: bandit -r src/
```

## Setup Instructions

1. **Create workflow files** in `.github/workflows/`
2. **Configure repository secrets** in Settings → Secrets and variables → Actions
3. **Set up branch protection** rules requiring CI to pass
4. **Configure status checks** for required workflows

## Best Practices

- **Pin action versions** for reproducibility
- **Use matrix builds** for multiple Python versions
- **Cache dependencies** to speed up workflows
- **Run security scans** on schedule and PRs
- **Fail fast** on critical security issues
- **Generate artifacts** for debugging failed builds

## Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Check trigger conditions
   - Verify branch names match
   - Ensure workflow file is in correct location

2. **Permission denied errors**
   - Check required secrets are configured
   - Verify token permissions
   - Review repository settings

3. **Test failures in CI but not locally**
   - Check environment differences
   - Verify all dependencies are specified
   - Review OS-specific issues

### Debugging Tips

- Use `actions/upload-artifact` to save logs
- Add debugging steps with `run: env` to check environment
- Use `continue-on-error: true` for non-critical steps
- Check Actions tab for detailed logs

## Security Considerations

- **Never commit secrets** to workflow files
- **Use least privilege** for token permissions
- **Review third-party actions** before use
- **Pin action versions** to prevent supply chain attacks
- **Regularly update** workflow dependencies