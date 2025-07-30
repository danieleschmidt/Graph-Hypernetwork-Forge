#!/bin/bash
# Comprehensive dependency security scanning

set -euo pipefail

ECHO_PREFIX="[SECURITY-SCAN]"
SCAN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="${SCAN_DIR}/security-reports"

echo "${ECHO_PREFIX} Starting dependency security scan..."

# Create reports directory
mkdir -p "${REPORT_DIR}"

# Safety check for known vulnerabilities
echo "${ECHO_PREFIX} Running Safety vulnerability scan..."
safety check --json --output "${REPORT_DIR}/safety-report.json" || {
    echo "${ECHO_PREFIX} Safety scan found vulnerabilities - check ${REPORT_DIR}/safety-report.json"
}

# Bandit static analysis
echo "${ECHO_PREFIX} Running Bandit static analysis..."
bandit -r graph_hypernetwork_forge/ -f json -o "${REPORT_DIR}/bandit-report.json" || {
    echo "${ECHO_PREFIX} Bandit found security issues - check ${REPORT_DIR}/bandit-report.json"
}

# Pip audit for additional vulnerability checking
if command -v pip-audit &> /dev/null; then
    echo "${ECHO_PREFIX} Running pip-audit..."
    pip-audit --format=json --output="${REPORT_DIR}/pip-audit-report.json" || {
        echo "${ECHO_PREFIX} pip-audit found issues - check ${REPORT_DIR}/pip-audit-report.json"
    }
else
    echo "${ECHO_PREFIX} pip-audit not available, install with: pip install pip-audit"
fi

# Semgrep security analysis
if command -v semgrep &> /dev/null; then
    echo "${ECHO_PREFIX} Running Semgrep security analysis..."
    semgrep --config=auto --json --output="${REPORT_DIR}/semgrep-report.json" graph_hypernetwork_forge/ || {
        echo "${ECHO_PREFIX} Semgrep found issues - check ${REPORT_DIR}/semgrep-report.json"
    }
else
    echo "${ECHO_PREFIX} Semgrep not available, install with: pip install semgrep"
fi

echo "${ECHO_PREFIX} Security scan completed. Reports in: ${REPORT_DIR}"
echo "${ECHO_PREFIX} Review all JSON reports for security findings."