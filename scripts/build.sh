#!/bin/bash

# Build script for Graph Hypernetwork Forge
# Handles building, testing, and packaging

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="graph-hypernetwork-forge"
BUILD_DIR="build"
DIST_DIR="dist"
DOCS_DIR="docs"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Clean previous builds
clean() {
    log_info "Cleaning previous builds..."
    rm -rf $BUILD_DIR $DIST_DIR
    rm -rf *.egg-info
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    log_success "Build directories cleaned"
}

# Install dependencies
install_deps() {
    log_info "Installing dependencies..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
    
    if [ -f "requirements-dev-lock.txt" ]; then
        log_info "Installing development dependencies..."
        pip install -r requirements-dev-lock.txt
    fi
    
    log_success "Dependencies installed"
}

# Run code quality checks
quality_checks() {
    log_info "Running code quality checks..."
    
    # Format check
    log_info "Checking code formatting with black..."
    black --check --diff .
    
    # Import sorting
    log_info "Checking import sorting with isort..."
    isort --check-only --diff .
    
    # Linting
    log_info "Running linting checks..."
    if command -v ruff &> /dev/null; then
        ruff check .
    elif command -v flake8 &> /dev/null; then
        flake8 .
    else
        log_warning "No linter found (ruff or flake8)"
    fi
    
    # Type checking
    log_info "Running type checks with mypy..."
    if command -v mypy &> /dev/null; then
        mypy graph_hypernetwork_forge/ --ignore-missing-imports
    else
        log_warning "mypy not found, skipping type checks"
    fi
    
    log_success "Code quality checks passed"
}

# Run security checks
security_checks() {
    log_info "Running security checks..."
    
    # Bandit security scanner
    if command -v bandit &> /dev/null; then
        log_info "Running bandit security scanner..."
        bandit -r graph_hypernetwork_forge/ -f json -o bandit-report.json || log_warning "Bandit found potential security issues"
    fi
    
    # Safety check for known vulnerabilities
    if command -v safety &> /dev/null; then
        log_info "Checking for known vulnerabilities..."
        safety check --json --output safety-report.json || log_warning "Safety found potential vulnerabilities"
    fi
    
    log_success "Security checks completed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Unit tests
    python -m pytest tests/ \
        --cov=graph_hypernetwork_forge \
        --cov-report=html \
        --cov-report=term-missing \
        --cov-report=xml \
        --junit-xml=test-results.xml \
        -v
    
    log_success "Tests completed"
}

# Build documentation
build_docs() {
    log_info "Building documentation..."
    
    if [ -d "$DOCS_DIR" ] && [ -f "$DOCS_DIR/Makefile" ]; then
        cd $DOCS_DIR
        make clean
        make html
        cd ..
        log_success "Documentation built"
    else
        log_warning "Documentation source not found, skipping"
    fi
}

# Build package
build_package() {
    log_info "Building package..."
    
    # Build source distribution and wheel
    python -m build --sdist --wheel --outdir $DIST_DIR
    
    # Verify build
    if [ -d "$DIST_DIR" ]; then
        log_info "Built packages:"
        ls -la $DIST_DIR/
        
        # Check package contents
        log_info "Checking package contents..."
        python -m twine check $DIST_DIR/*
        
        log_success "Package built successfully"
    else
        log_error "Package build failed"
        exit 1
    fi
}

# Build Docker image
build_docker() {
    log_info "Building Docker image..."
    
    local tag=${1:-"$PROJECT_NAME:latest"}
    local target=${2:-"production"}
    
    docker build \
        --target $target \
        --tag $tag \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VCS_REF=$(git rev-parse --short HEAD) \
        --build-arg VERSION=$(python -c "import toml; print(toml.load('pyproject.toml')['project']['version'])") \
        .
    
    log_success "Docker image built: $tag"
}

# Generate SBOM (Software Bill of Materials)
generate_sbom() {
    log_info "Generating Software Bill of Materials..."
    
    if command -v syft &> /dev/null; then
        syft packages . -o spdx-json=sbom.spdx.json
        syft packages . -o cyclonedx-json=sbom.cyclonedx.json
        log_success "SBOM generated"
    else
        log_warning "syft not found, skipping SBOM generation"
    fi
}

# Performance benchmarks
run_benchmarks() {
    log_info "Running performance benchmarks..."
    
    python -m pytest tests/ -m performance --benchmark-only --benchmark-json=benchmark-results.json
    
    log_success "Benchmarks completed"
}

# Main build function
main() {
    local command=${1:-"all"}
    
    case $command in
        "clean")
            clean
            ;;
        "deps")
            install_deps
            ;;
        "quality")
            quality_checks
            ;;
        "security")
            security_checks
            ;;
        "test")
            run_tests
            ;;
        "docs")
            build_docs
            ;;
        "package")
            build_package
            ;;
        "docker")
            build_docker ${2:-"$PROJECT_NAME:latest"} ${3:-"production"}
            ;;
        "sbom")
            generate_sbom
            ;;
        "benchmark")
            run_benchmarks
            ;;
        "all")
            log_info "Starting full build pipeline..."
            clean
            install_deps
            quality_checks
            security_checks
            run_tests
            build_docs
            build_package
            generate_sbom
            log_success "Build pipeline completed successfully!"
            ;;
        "ci")
            log_info "Starting CI build pipeline..."
            clean
            install_deps
            quality_checks
            security_checks
            run_tests
            build_package
            log_success "CI build pipeline completed successfully!"
            ;;
        *)
            echo "Usage: $0 {clean|deps|quality|security|test|docs|package|docker|sbom|benchmark|all|ci}"
            echo ""
            echo "Commands:"
            echo "  clean     - Clean build directories"
            echo "  deps      - Install dependencies"
            echo "  quality   - Run code quality checks"
            echo "  security  - Run security checks"
            echo "  test      - Run tests"
            echo "  docs      - Build documentation"
            echo "  package   - Build Python package"
            echo "  docker    - Build Docker image"
            echo "  sbom      - Generate Software Bill of Materials"    
            echo "  benchmark - Run performance benchmarks"
            echo "  all       - Run complete build pipeline"
            echo "  ci        - Run CI-optimized build pipeline"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"