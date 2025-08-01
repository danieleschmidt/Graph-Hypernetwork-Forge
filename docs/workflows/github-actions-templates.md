# GitHub Actions Workflow Templates

This document provides ready-to-implement GitHub Actions workflows for the Graph Hypernetwork Forge project. Copy these templates to `.github/workflows/` to enable full CI/CD automation.

## Quick Start

1. Create `.github/workflows/` directory in your repository
2. Copy the workflow files below
3. Configure required secrets in repository settings
4. Enable branch protection rules

## Core Workflows

### 1. Main CI/CD Pipeline

**File**: `.github/workflows/ci.yml`

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*.*.*' ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily security scans

env:
  PYTHON_VERSION: '3.11'
  CACHE_VERSION: v1

jobs:
  # Code Quality & Testing Matrix
  test:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cache/pip
          ~/.cache/pre-commit
        key: ${{ runner.os }}-python-${{ matrix.python-version }}-${{ hashFiles('**/requirements*.txt', '**/pyproject.toml', '**/.pre-commit-config.yaml') }}-${{ env.CACHE_VERSION }}
        restore-keys: |
          ${{ runner.os }}-python-${{ matrix.python-version }}-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev-lock.txt
        pip install -e .
    
    - name: Run pre-commit hooks
      run: |
        pre-commit install
        pre-commit run --all-files --show-diff-on-failure
    
    - name: Run tests with coverage
      run: |
        pytest tests/ \
          --cov=graph_hypernetwork_forge \
          --cov-report=xml \
          --cov-report=term-missing \
          --cov-fail-under=80 \
          --benchmark-skip \
          -v
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      if: matrix.python-version == env.PYTHON_VERSION
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
        verbose: true

  # Security Scanning
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    permissions:
      security-events: write
      contents: read
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install security dependencies
      run: |
        pip install -r requirements-security-lock.txt
    
    - name: Run Bandit security scan
      run: |
        bandit -r graph_hypernetwork_forge/ -f json -o bandit-report.json || true
        bandit -r graph_hypernetwork_forge/ -f txt
    
    - name: Run Safety dependency scan
      run: |
        safety check --json --output safety-report.json || true
        safety check
    
    - name: Run Semgrep security scan
      uses: returntocorp/semgrep-action@v1
      with:
        config: auto
        generateSarif: "1"
      continue-on-error: true
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-reports-${{ github.sha }}
        path: |
          bandit-report.json
          safety-report.json
        retention-days: 30

  # Container Security & Build
  container:
    name: Container Security
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        target: production
        tags: |
          hypergnn:${{ github.sha }}
          hypergnn:latest
        load: true
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'hypergnn:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run container structure tests
      run: |
        curl -LO https://storage.googleapis.com/container-structure-test/latest/container-structure-test-linux-amd64
        chmod +x container-structure-test-linux-amd64
        ./container-structure-test-linux-amd64 test --image hypergnn:${{ github.sha }} --config container-structure-test.yaml || echo "No structure test config found"

  # Performance Benchmarks
  performance:
    name: Performance Tests
    runs-on: ubuntu-latest
    if: github.event_name != 'schedule'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        pip install -e .[dev]
        pip install pytest-benchmark memory-profiler
    
    - name: Run performance benchmarks
      run: |
        pytest tests/test_performance.py \
          --benchmark-json=benchmark.json \
          --benchmark-min-rounds=3 \
          --benchmark-only
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      if: github.ref == 'refs/heads/main'
      with:
        name: Python Benchmark
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '110%'
        fail-on-alert: true

  # Documentation Build
  docs:
    name: Documentation
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install documentation dependencies
      run: |
        pip install sphinx sphinx-rtd-theme
        pip install -e .
    
    - name: Build documentation
      run: |
        cd docs/
        make html
    
    - name: Upload documentation artifacts
      uses: actions/upload-artifact@v3
      with:
        name: documentation-${{ github.sha }}
        path: docs/_build/html/
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      if: github.ref == 'refs/heads/main'
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html

  # Release & Deploy
  release:
    name: Release
    runs-on: ubuntu-latest
    needs: [test, security, container, performance, docs]
    if: startsWith(github.ref, 'refs/tags/v')
    
    permissions:
      contents: write
      packages: write
      id-token: write
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Build package
      run: |
        pip install build
        python -m build
    
    - name: Generate SBOM
      run: |
        pip install cyclonedx-bom
        cyclonedx-py -o sbom.json
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: |
          dist/*
          sbom.json
        generate_release_notes: true
        draft: false
        prerelease: ${{ contains(github.ref, '-') }}
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      if: "!contains(github.ref, '-')"  # Only for stable releases
```

### 2. Dependency Updates Workflow

**File**: `.github/workflows/dependency-updates.yml`

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 3 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  update-dependencies:
    name: Update Dependencies
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Update pip-tools
      run: |
        pip install --upgrade pip pip-tools
    
    - name: Update requirements
      run: |
        pip-compile --upgrade requirements.in
        pip-compile --upgrade requirements-dev.in
        pip-compile --upgrade requirements-security.in
    
    - name: Run tests with updated dependencies
      run: |
        pip install -r requirements.txt -r requirements-dev-lock.txt
        pip install -e .
        pytest tests/ --maxfail=5
    
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update dependencies'
        title: 'chore: weekly dependency updates'
        body: |
          Automated dependency updates.
          
          Please review the changes and ensure all tests pass.
        branch: automated/dependency-updates
        delete-branch: true
```

### 3. Security Monitoring Workflow

**File**: `.github/workflows/security-monitoring.yml`

```yaml
name: Security Monitoring

on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:

jobs:
  vulnerability-scan:
    name: Vulnerability Scan
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run comprehensive security scan
      run: |
        docker run --rm -v "$PWD":/src \
          returntocorp/semgrep:latest \
          --config=auto /src
    
    - name: Dependency vulnerability scan
      run: |
        pip install safety
        safety check --json
    
    - name: Container security scan
      run: |
        docker build -t temp-scan .
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          aquasec/trivy:latest image temp-scan
    
    - name: Create security issue on findings
      if: failure()
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: 'Security Alert: Vulnerabilities Detected',
            body: 'Automated security scan detected potential vulnerabilities. Please review the failed workflow for details.',
            labels: ['security', 'automated']
          })
```

## Required Repository Secrets

Configure these in **Settings > Secrets and variables > Actions**:

### Essential Secrets
- `CODECOV_TOKEN`: For coverage reporting integration
- `WANDB_API_KEY`: For experiment tracking (if using Weights & Biases)

### Optional Secrets (for enhanced features)
- `DOCKER_HUB_USERNAME`: For container registry publishing
- `DOCKER_HUB_ACCESS_TOKEN`: For container registry authentication
- `SLACK_WEBHOOK_URL`: For notification integration
- `FOSSA_API_KEY`: For license compliance scanning

## Branch Protection Configuration

Apply these settings to the `main` branch:

### Required Status Checks
- `test (3.10)`
- `test (3.11)`
- `test (3.12)`
- `security`
- `container`
- `docs`

### Protection Rules
- ✅ Require pull request reviews before merging
- ✅ Require up-to-date branches before merging
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators
- ✅ Allow force pushes (disabled)
- ✅ Allow deletions (disabled)

## Environment Variables

### Repository Variables
Set in **Settings > Secrets and variables > Actions > Variables**:

- `PYTHON_DEFAULT_VERSION`: `3.11`
- `COVERAGE_THRESHOLD`: `80`
- `BENCHMARK_THRESHOLD`: `110`

## Workflow Optimization Tips

### Performance Optimization
1. **Caching Strategy**: Use aggressive caching for dependencies and pre-commit hooks
2. **Matrix Optimization**: Use `fail-fast: false` for comprehensive testing
3. **Conditional Execution**: Skip expensive jobs for documentation-only changes
4. **Parallel Execution**: Run independent jobs concurrently

### Security Best Practices
1. **Minimal Permissions**: Use specific permissions for each job
2. **Secret Management**: Never log secrets, use masked variables
3. **Container Security**: Scan all images before deployment
4. **Supply Chain**: Verify action checksums with dependabot

### Cost Optimization
1. **Conditional Workflows**: Skip unnecessary runs on specific file changes
2. **Efficient Runners**: Use appropriate runner sizes
3. **Caching**: Implement multi-level caching strategy
4. **Timeout Configuration**: Set reasonable timeouts for all jobs

## Integration with Existing Tools

### Makefile Integration
The workflows integrate with existing Makefile commands:

```yaml
- name: Run make targets
  run: |
    make lint
    make test
    make security-check
    make build
```

### Pre-commit Integration
Workflows use the same pre-commit configuration:

```yaml
- name: Run pre-commit hooks
  run: pre-commit run --all-files --show-diff-on-failure
```

### Docker Integration
Workflows use the existing Docker configuration:

```yaml
- name: Build with existing Dockerfile
  run: docker build -t hypergnn:test .
```

## Monitoring and Alerts

### Workflow Monitoring
- **Success Rate Tracking**: Monitor workflow success rates
- **Performance Metrics**: Track build times and test duration
- **Security Alerts**: Automated issue creation on security findings
- **Dependency Updates**: Automated PR creation for updates

### Notification Integration
Add Slack/Teams notifications:

```yaml
- name: Notify on failure
  if: failure()
  uses: rtCamp/action-slack-notify@v2
  env:
    SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK_URL }}
    SLACK_MESSAGE: 'CI/CD pipeline failed for ${{ github.ref }}'
```

## Migration Checklist

### Phase 1: Basic CI (Week 1)
- [ ] Create `.github/workflows/` directory
- [ ] Add main CI/CD workflow (`ci.yml`)
- [ ] Configure repository secrets
- [ ] Enable branch protection rules
- [ ] Test with sample PR

### Phase 2: Enhanced Security (Week 2)
- [ ] Add security monitoring workflow
- [ ] Configure vulnerability scanning
- [ ] Enable automated security alerts
- [ ] Add container security scanning

### Phase 3: Advanced Features (Week 3)
- [ ] Add performance benchmarking
- [ ] Configure automated releases
- [ ] Add dependency update automation
- [ ] Enable advanced monitoring

---

**Implementation Note**: These workflows are designed to integrate seamlessly with the existing excellent tooling in this repository. The configuration preserves all existing quality standards while adding comprehensive automation.