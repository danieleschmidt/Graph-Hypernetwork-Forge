# Required Manual Setup for GitHub Actions

Due to GitHub App permission limitations, the following workflow files need to be manually created by repository maintainers with write access.

## Required Actions

### 1. Create GitHub Actions Workflow Files

Copy the template files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create .github/workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
```

### 2. Configure Repository Settings

#### Branch Protection Rules
Go to Settings → Branches → Add rule for `main`:

- ✅ Require a pull request before merging
- ✅ Require approvals (1 reviewer minimum)
- ✅ Dismiss stale PR approvals when new commits are pushed
- ✅ Require review from code owners
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - Required status checks:
    - `Code Quality`
    - `Security Scan`
    - `Tests (ubuntu-latest, 3.11)`
    - `Build Package`
- ✅ Require conversation resolution before merging
- ✅ Restrict pushes that create files matching a path pattern
- ✅ Require linear history
- ✅ Include administrators

#### Repository Secrets
Add the following secrets in Settings → Secrets and variables → Actions:

**Required Secrets:**
- `GITHUB_TOKEN` - Automatically provided by GitHub
- `CODECOV_TOKEN` - For coverage reporting (get from codecov.io)

**Optional Secrets:**
- `SEMGREP_APP_TOKEN` - For enhanced security scanning
- `NPM_TOKEN` - If publishing to npm registry
- `DOCKER_REGISTRY_TOKEN` - For custom Docker registry
- `SLACK_WEBHOOK_URL` - For notifications

#### Repository Variables
Add the following variables in Settings → Secrets and variables → Actions:

- `PYTHON_VERSION` = `3.11`
- `NODE_VERSION` = `18`
- `REGISTRY` = `ghcr.io`

### 3. Enable GitHub Pages

Go to Settings → Pages:
- Source: Deploy from a branch
- Branch: `gh-pages` / `root`
- Enable after first documentation build

### 4. Configure Dependabot

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      
  - package-ecosystem: "github-actions"
    directory: "/.github/workflows"
    schedule:
      interval: "weekly"
```

### 5. Configure Code Owners

Create `.github/CODEOWNERS`:

```
# Global owners
* @danieleschmidt

# Core model components
/graph_hypernetwork_forge/ @danieleschmidt
/tests/ @danieleschmidt

# Documentation
/docs/ @danieleschmidt
*.md @danieleschmidt

# CI/CD and infrastructure
/.github/ @danieleschmidt
/docker/ @danieleschmidt
/scripts/ @danieleschmidt
```

### 6. Enable GitHub Security Features

Go to Settings → Security & analysis:

- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Code scanning (CodeQL)
- ✅ Secret scanning
- ✅ Secret scanning push protection

### 7. Configure Notifications

Go to Settings → Notifications:

Set up Slack/Discord webhooks for:
- Failed CI builds
- Security alerts
- New releases
- Pull request reviews

## Verification Checklist

After setup, verify the following work correctly:

- [ ] CI pipeline runs on pull requests
- [ ] Security scanning reports vulnerabilities
- [ ] Tests run on multiple Python versions
- [ ] Code coverage reports to Codecov
- [ ] Docker images build and push to registry
- [ ] Documentation builds and deploys
- [ ] Semantic release creates new versions
- [ ] Branch protection prevents direct pushes
- [ ] Dependabot creates dependency update PRs

## Troubleshooting

### Common Issues

**Issue**: Workflows don't trigger
- Check branch protection rules
- Verify workflow file syntax
- Check repository permissions

**Issue**: Tests fail on CI but pass locally
- Check Python version differences
- Verify all dependencies are locked
- Check environment variable differences

**Issue**: Docker build fails
- Verify Dockerfile syntax
- Check build context size
- Ensure secrets are properly configured

**Issue**: Documentation doesn't deploy
- Check GitHub Pages settings
- Verify docs build locally
- Check gh-pages branch exists

### Getting Help

1. Check workflow run logs in Actions tab
2. Review GitHub's documentation
3. Contact repository maintainers
4. Open an issue with CI/CD label

## Security Considerations

- Never commit secrets to the repository
- Use GitHub's secret scanning
- Regularly update dependencies
- Monitor security alerts
- Review dependency licenses
- Use SLSA compliance where possible

## Performance Optimization

- Use dependency caching
- Implement matrix builds efficiently
- Use self-hosted runners for GPU tests
- Optimize Docker layer caching
- Run expensive tests conditionally