# Advanced Security Configuration

## Automated Security Scanning

### Dependency Scanning

```bash
# Run security audit
make security

# Check for known vulnerabilities
safety check --json --output safety-report.json

# Bandit static analysis
bandit -r graph_hypernetwork_forge/ -f json
```

### Container Security

```bash
# Scan Docker images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image your-image:tag

# Check container best practices
docker run --rm -i hadolint/hadolint < Dockerfile
```

## SBOM Generation

### Software Bill of Materials

```bash
# Generate SBOM with syft
syft packages . -o spdx-json=sbom.spdx.json

# Generate Python requirements SBOM
pip-licenses --format=json --output-file=requirements-sbom.json
```

### Supply Chain Security

```bash
# Verify package signatures
pip install --require-hashes -r requirements.txt

# Check for malicious packages
python -m pip audit
```

## Security Testing Integration

### Test Security Markers

Use pytest markers for security testing:

```python
@pytest.mark.security
def test_input_validation():
    # Test malicious input handling
    pass

@pytest.mark.security
def test_authentication():
    # Test auth mechanisms
    pass
```

### Security Benchmarks

```bash
# Run security-specific tests
pytest tests/ -m security -v

# Generate security report
pytest tests/ -m security --html=security-report.html
```

## Runtime Security

### Environment Hardening

```bash
# Run with limited privileges
docker run --user 1000:1000 --read-only your-app

# Use security profiles
docker run --security-opt seccomp=default.json your-app
```

### Secrets Management

```bash
# Environment variables validation
set -u  # Exit on undefined variables

# Use secure secret storage
export WANDB_API_KEY="$(cat /run/secrets/wandb_key)"
```

## Compliance Framework

### SLSA Compliance Steps

1. **Source Integrity**: All code in version control
2. **Build Integrity**: Reproducible builds
3. **Dependency Tracking**: Complete SBOM
4. **Vulnerability Management**: Automated scanning

### Security Audit Checklist

- [ ] All dependencies scanned for vulnerabilities
- [ ] Container images scanned and hardened
- [ ] SBOM generated and up-to-date
- [ ] Security tests passing
- [ ] No secrets in code or config
- [ ] Access controls properly configured
- [ ] Logging and monitoring enabled

## Incident Response

### Security Incident Workflow

1. **Detection**: Automated alerts or manual discovery
2. **Assessment**: Determine severity and impact
3. **Containment**: Isolate affected systems
4. **Eradication**: Remove threat and vulnerabilities
5. **Recovery**: Restore services safely
6. **Lessons Learned**: Document and improve

### Emergency Contacts

- Security Team: security@terragonlabs.com
- On-call Engineer: +1-555-SECURITY
- Incident Commander: incidents@terragonlabs.com

## Security Tools Integration

### Required Tools

```bash
# Install security scanning tools
pip install safety bandit semgrep

# Container scanning
docker pull aquasec/trivy
docker pull hadolint/hadolint

# SBOM generation
go install github.com/anchore/syft/cmd/syft@latest
```

### Automation Scripts

Refer to `scripts/security/` directory for:
- `scan-dependencies.sh`: Automated dependency scanning
- `generate-sbom.sh`: SBOM generation
- `container-security.sh`: Container hardening
- `compliance-check.sh`: SLSA compliance validation