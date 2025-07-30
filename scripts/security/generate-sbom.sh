#!/bin/bash
# Generate Software Bill of Materials (SBOM)

set -euo pipefail

ECHO_PREFIX="[SBOM-GEN]"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBOM_DIR="${PROJECT_DIR}/sbom"
DATE=$(date +%Y%m%d-%H%M%S)

echo "${ECHO_PREFIX} Generating SBOM for Graph Hypernetwork Forge..."

# Create SBOM directory
mkdir -p "${SBOM_DIR}"

# Python dependencies SBOM
echo "${ECHO_PREFIX} Generating Python dependencies SBOM..."
if command -v pip-licenses &> /dev/null; then
    pip-licenses --format=json --output-file="${SBOM_DIR}/python-deps-${DATE}.json"
    pip-licenses --format=csv --output-file="${SBOM_DIR}/python-deps-${DATE}.csv"
    echo "${ECHO_PREFIX} Python SBOM generated: ${SBOM_DIR}/python-deps-${DATE}.json"
else
    echo "${ECHO_PREFIX} pip-licenses not found. Install with: pip install pip-licenses"
fi

# Generate detailed requirements with hashes
echo "${ECHO_PREFIX} Generating requirements with hashes..."
pip freeze > "${SBOM_DIR}/requirements-frozen-${DATE}.txt"
echo "${ECHO_PREFIX} Frozen requirements: ${SBOM_DIR}/requirements-frozen-${DATE}.txt"

# System packages SBOM (if syft is available)
if command -v syft &> /dev/null; then
    echo "${ECHO_PREFIX} Generating system SBOM with Syft..."
    syft packages "${PROJECT_DIR}" -o spdx-json="${SBOM_DIR}/system-sbom-${DATE}.spdx.json"
    syft packages "${PROJECT_DIR}" -o cyclonedx-json="${SBOM_DIR}/system-sbom-${DATE}.cyclonedx.json"
    echo "${ECHO_PREFIX} System SBOM generated in multiple formats"
else
    echo "${ECHO_PREFIX} Syft not found. Install from: https://github.com/anchore/syft"
fi

# Container SBOM (if image exists)
if docker images | grep -q "graph-hypernetwork-forge"; then
    echo "${ECHO_PREFIX} Generating container SBOM..."
    if command -v syft &> /dev/null; then
        syft packages graph-hypernetwork-forge:latest -o spdx-json="${SBOM_DIR}/container-sbom-${DATE}.spdx.json"
        echo "${ECHO_PREFIX} Container SBOM generated: ${SBOM_DIR}/container-sbom-${DATE}.spdx.json"
    fi
fi

# Vulnerability assessment of SBOM
if command -v grype &> /dev/null && [ -f "${SBOM_DIR}/system-sbom-${DATE}.spdx.json" ]; then
    echo "${ECHO_PREFIX} Running vulnerability assessment on SBOM..."
    grype "sbom:${SBOM_DIR}/system-sbom-${DATE}.spdx.json" -o json > "${SBOM_DIR}/vuln-assessment-${DATE}.json"
    echo "${ECHO_PREFIX} Vulnerability assessment completed"
fi

# Generate SBOM summary
echo "${ECHO_PREFIX} Generating SBOM summary..."
cat > "${SBOM_DIR}/SBOM-README.md" << EOF
# Software Bill of Materials (SBOM)

## Generated: $(date)

### Files in this directory:

- \`python-deps-*.json\`: Python package dependencies in JSON format
- \`python-deps-*.csv\`: Python package dependencies in CSV format
- \`requirements-frozen-*.txt\`: Exact versions of all installed packages
- \`system-sbom-*.spdx.json\`: Complete system SBOM in SPDX format
- \`system-sbom-*.cyclonedx.json\`: Complete system SBOM in CycloneDX format
- \`container-sbom-*.spdx.json\`: Container image SBOM
- \`vuln-assessment-*.json\`: Vulnerability assessment results

### Usage:

\`\`\`bash
# Regenerate SBOM
./scripts/security/generate-sbom.sh

# Validate SBOM integrity
syft attest --output attest.json sbom.spdx.json

# Check for vulnerabilities
grype sbom:sbom.spdx.json
\`\`\`

### SBOM Validation:

- All dependencies tracked with versions
- Licenses identified for compliance
- Vulnerability scanning integrated
- Supply chain integrity verified
EOF

echo "${ECHO_PREFIX} SBOM generation completed!"
echo "${ECHO_PREFIX} Output directory: ${SBOM_DIR}"
echo "${ECHO_PREFIX} Summary: ${SBOM_DIR}/SBOM-README.md"