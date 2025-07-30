#!/bin/bash
# Container security scanning and hardening

set -euo pipefail

ECHO_PREFIX="[CONTAINER-SEC]"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SECURITY_DIR="${PROJECT_DIR}/security-reports"
IMAGE_NAME="graph-hypernetwork-forge"

echo "${ECHO_PREFIX} Starting container security assessment..."

# Create security reports directory
mkdir -p "${SECURITY_DIR}"

# Dockerfile security best practices check
echo "${ECHO_PREFIX} Checking Dockerfile security..."
if command -v hadolint &> /dev/null; then
    hadolint "${PROJECT_DIR}/Dockerfile" --format json > "${SECURITY_DIR}/hadolint-report.json" || {
        echo "${ECHO_PREFIX} Hadolint found Dockerfile issues - check ${SECURITY_DIR}/hadolint-report.json"
    }
    echo "${ECHO_PREFIX} Dockerfile security check completed"
else
    echo "${ECHO_PREFIX} Hadolint not found. Install with: docker pull hadolint/hadolint"
fi

# Container image vulnerability scanning
if docker images | grep -q "${IMAGE_NAME}"; then
    echo "${ECHO_PREFIX} Scanning container image for vulnerabilities..."
    
    # Trivy vulnerability scan
    if command -v trivy &> /dev/null; then
        trivy image --format json --output "${SECURITY_DIR}/trivy-image-report.json" "${IMAGE_NAME}:latest" || {
            echo "${ECHO_PREFIX} Trivy found vulnerabilities - check ${SECURITY_DIR}/trivy-image-report.json"
        }
        echo "${ECHO_PREFIX} Container vulnerability scan completed"
    else
        echo "${ECHO_PREFIX} Trivy not found. Install from: https://github.com/aquasecurity/trivy"
    fi
    
    # Container configuration analysis
    if command -v dive &> /dev/null; then
        echo "${ECHO_PREFIX} Analyzing container layers with dive..."
        dive "${IMAGE_NAME}:latest" --ci --json > "${SECURITY_DIR}/dive-analysis.json" || {
            echo "${ECHO_PREFIX} Dive analysis completed with warnings"
        }
    else
        echo "${ECHO_PREFIX} Dive not found. Install from: https://github.com/wagoodman/dive"
    fi
else
    echo "${ECHO_PREFIX} Container image '${IMAGE_NAME}' not found. Build it first with: docker-compose build"
fi

# Generate container security recommendations
echo "${ECHO_PREFIX} Generating security recommendations..."
cat > "${SECURITY_DIR}/container-security-checklist.md" << 'EOF'
# Container Security Checklist

## Build-time Security

- [ ] Use official, minimal base images
- [ ] Pin specific image versions (not `latest`)
- [ ] Multi-stage builds to reduce attack surface
- [ ] Non-root user for application execution
- [ ] Minimal package installation
- [ ] Remove package managers after installation
- [ ] Use .dockerignore to exclude sensitive files
- [ ] Regular base image updates

## Runtime Security

- [ ] Read-only root filesystem where possible
- [ ] Drop unnecessary capabilities
- [ ] Use security profiles (AppArmor/SELinux)
- [ ] Resource limits (CPU, memory)
- [ ] Network policies for container communication
- [ ] Secrets management (not environment variables)
- [ ] Regular container restarts
- [ ] Log aggregation and monitoring

## Recommended Docker Run Flags

```bash
# Security-hardened container execution
docker run \
  --user 1000:1000 \
  --read-only \
  --tmpfs /tmp \
  --tmpfs /var/tmp \
  --cap-drop ALL \
  --cap-add CHOWN \
  --cap-add SETUID \
  --cap-add SETGID \
  --security-opt no-new-privileges:true \
  --security-opt seccomp=default.json \
  --memory 2g \
  --cpus 1.0 \
  your-image:tag
```

## Docker-compose Security

```yaml
services:
  app:
    security_opt:
      - no-new-privileges:true
      - seccomp:default.json
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETUID
      - SETGID
    mem_limit: 2g
    cpus: 1.0
    user: "1000:1000"
```

## Vulnerability Management

- [ ] Regular image vulnerability scanning
- [ ] Automated base image updates
- [ ] CVE monitoring and alerting
- [ ] Security patch deployment pipeline
- [ ] Container runtime monitoring
EOF

echo "${ECHO_PREFIX} Container security assessment completed!"
echo "${ECHO_PREFIX} Reports generated in: ${SECURITY_DIR}/"
echo "${ECHO_PREFIX} Security checklist: ${SECURITY_DIR}/container-security-checklist.md"