# Branch Protection Configuration

This document outlines the recommended branch protection rules for the Graph Hypernetwork Forge repository to ensure code quality, security, and proper review processes.

## Branch Protection Rules for `main`

### Required Settings

#### Pull Request Requirements
- **Require pull request reviews before merging**: ✅ Enabled
  - Required number of approvers: **2**
  - Dismiss stale PR approvals when new commits are pushed: ✅ Enabled
  - Require review from code owners: ✅ Enabled
  - Restrict pushes that create PR reviews: ✅ Enabled

#### Status Checks
**Require status checks to pass before merging**: ✅ Enabled
- Require branches to be up to date before merging: ✅ Enabled

**Required Status Checks:**
```
ci-test-3.10
ci-test-3.11
ci-test-3.12
security-scan
lint-and-format
type-check
container-security
dependency-scan
```

#### Additional Restrictions
- **Restrict pushes that create branches**: ✅ Enabled
- **Restrict pushes that update tags**: ✅ Enabled
- **Include administrators**: ✅ Enabled
- **Allow force pushes**: ❌ Disabled
- **Allow deletions**: ❌ Disabled

### Branch Protection Rules for `develop`

#### Pull Request Requirements
- **Require pull request reviews before merging**: ✅ Enabled
  - Required number of approvers: **1**
  - Dismiss stale PR approvals when new commits are pushed: ✅ Enabled

#### Status Checks
**Required Status Checks:**
```
ci-test-3.11
security-scan
lint-and-format
type-check
```

#### Additional Restrictions
- **Include administrators**: ❌ Disabled (for faster development)
- **Allow force pushes**: ❌ Disabled
- **Allow deletions**: ❌ Disabled

## CODEOWNERS Configuration

Create `.github/CODEOWNERS` file:

```gitignore
# Global ownership
* @danieleschmidt

# Security-related files require security team review
SECURITY.md @danieleschmidt @security-team
.github/workflows/security.yml @danieleschmidt @security-team
docs/workflows/security-*.md @danieleschmidt @security-team
scripts/security/ @danieleschmidt @security-team

# CI/CD configuration requires DevOps review
.github/workflows/ @danieleschmidt @devops-team
docker-compose*.yml @danieleschmidt @devops-team
Dockerfile* @danieleschmidt @devops-team
scripts/ci/ @danieleschmidt @devops-team

# Core ML code requires ML engineer review
graph_hypernetwork_forge/core/ @danieleschmidt @ml-team
graph_hypernetwork_forge/models/ @danieleschmidt @ml-team
graph_hypernetwork_forge/training/ @danieleschmidt @ml-team

# Documentation can be reviewed by documentation team
docs/ @danieleschmidt @docs-team
README.md @danieleschmidt @docs-team
*.md @danieleschmidt @docs-team

# Configuration files require careful review
pyproject.toml @danieleschmidt
requirements*.txt @danieleschmidt
setup.py @danieleschmidt
Makefile @danieleschmidt

# Test files require test maintainer review
tests/ @danieleschmidt @test-maintainers
```

## Repository Settings

### General Settings
- **Default branch**: `main`
- **Allow merge commits**: ✅ Enabled
- **Allow squash merging**: ✅ Enabled (recommended)
- **Allow rebase merging**: ❌ Disabled
- **Automatically delete head branches**: ✅ Enabled

### Security Settings
- **Enable vulnerability alerts**: ✅ Enabled
- **Enable automated security fixes**: ✅ Enabled
- **Enable private vulnerability reporting**: ✅ Enabled

### Features
- **Issues**: ✅ Enabled
- **Projects**: ✅ Enabled
- **Wiki**: ❌ Disabled (use docs/ instead)
- **Discussions**: ✅ Enabled

## Workflow Integration

### Required GitHub Actions

The following workflows must be configured and passing for branch protection:

#### 1. Main CI Workflow (`.github/workflows/ci.yml`)
```yaml
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: ci-test-${{ matrix.python-version }}
    # ... (see workflow templates)
```

#### 2. Security Workflow (`.github/workflows/security.yml`)
```yaml
name: Security Scan
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'

jobs:
  security-scan:
    name: security-scan
    # ... (see workflow templates)
```

#### 3. Linting Workflow (`.github/workflows/lint.yml`)
```yaml
name: Lint and Format
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-format:
    name: lint-and-format
    # ... (see workflow templates)
```

### Status Check Requirements

All status checks must pass before merging. The checks include:

1. **Unit Tests**: All test suites across Python versions
2. **Integration Tests**: End-to-end functionality tests
3. **Security Scans**: Vulnerability and security analysis
4. **Code Quality**: Linting, formatting, and type checking
5. **Container Security**: Docker image vulnerability scanning
6. **Dependency Analysis**: Security and license compliance

## Emergency Procedures

### Hotfix Process
For critical security fixes or production issues:

1. Create hotfix branch from `main`
2. Apply minimal necessary changes
3. Request emergency review from 2+ maintainers
4. Merge with temporary bypass (document reason)
5. Create follow-up PR for additional testing/documentation

### Branch Protection Bypass
Only repository administrators can bypass branch protection. This should be:

1. **Documented**: Create issue explaining the bypass reason
2. **Time-limited**: Re-enable protection immediately after
3. **Audited**: Review bypass usage monthly
4. **Justified**: Only for critical issues or emergencies

## Enforcement Policies

### Review Requirements
- **New contributors**: All PRs require 2 reviews
- **Core maintainers**: PRs require 1 review from another maintainer
- **Documentation changes**: 1 review from docs team sufficient
- **Security changes**: Always require security team review

### Quality Gates
All PRs must meet these minimum requirements:

- [ ] All CI checks pass
- [ ] Code coverage maintained (≥80%)
- [ ] No high-severity security vulnerabilities
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for user-facing changes)

### Merge Strategies
- **Feature branches**: Squash and merge (clean history)
- **Hotfixes**: Merge commit (preserve context)
- **Documentation**: Squash and merge
- **Dependencies**: Squash and merge

## Monitoring and Compliance

### Weekly Reviews
- Review failed status checks and common failures
- Analyze bypass usage and necessity
- Update required checks based on new workflows

### Monthly Audits
- Review CODEOWNERS effectiveness
- Assess review quality and feedback
- Update branch protection rules based on team growth

### Quarterly Assessments
- Full branch protection policy review
- Team permission and access review
- Security and compliance assessment

### Key Metrics
Track these metrics for branch protection effectiveness:

- **PR Review Time**: Average time from creation to approval
- **CI Failure Rate**: Percentage of PRs with failing checks
- **Security Issue Rate**: Number of security issues reaching main
- **Bypass Frequency**: Number of branch protection bypasses
- **Review Coverage**: Percentage of code reviewed by domain experts

## Troubleshooting

### Common Issues

#### Status Check Failures
```bash
# Check workflow status
gh run list --branch feature-branch

# Re-run failed checks
gh run rerun <run-id>

# View specific job logs
gh run view <run-id> --job <job-id>
```

#### CODEOWNERS Issues
```bash
# Validate CODEOWNERS syntax
gh api repos/:owner/:repo/codeowners/errors

# Test CODEOWNERS rules
gh api repos/:owner/:repo/contents/path/to/file \
  --jq '.owner.login'
```

#### Branch Protection Conflicts
```bash
# Check current protection status
gh api repos/:owner/:repo/branches/main/protection

# Validate required status checks
gh api repos/:owner/:repo/branches/main/protection/required_status_checks
```

### Support Contacts
- **CI/CD Issues**: DevOps team (@devops-team)
- **Security Issues**: Security team (@security-team)
- **Branch Protection**: Repository admins (@danieleschmidt)
- **CODEOWNERS**: Team leads