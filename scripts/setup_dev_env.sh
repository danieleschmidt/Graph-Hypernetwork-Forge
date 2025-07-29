#!/bin/bash
# Graph Hypernetwork Forge - Development Environment Setup Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if Python 3.10+ is available
check_python_version() {
    log_info "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        log_error "Python not found. Please install Python 3.10 or higher."
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
        log_error "Python 3.10 or higher is required. Found: $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python $PYTHON_VERSION detected"
}

# Create virtual environment
create_virtual_env() {
    log_info "Setting up virtual environment..."
    
    if [[ ! -d "venv" ]]; then
        $PYTHON_CMD -m venv venv
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_virtual_env() {
    log_info "Activating virtual environment..."
    
    if [[ -f "venv/bin/activate" ]]; then
        source venv/bin/activate
        log_success "Virtual environment activated"
    elif [[ -f "venv/Scripts/activate" ]]; then
        source venv/Scripts/activate
        log_success "Virtual environment activated (Windows)"
    else
        log_error "Could not find virtual environment activation script"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install the package in development mode with all extras
    pip install -e .[dev,security,docs,performance]
    
    log_success "Dependencies installed"
}

# Setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    pre-commit install
    pre-commit install --hook-type commit-msg
    
    log_success "Pre-commit hooks installed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Test imports
    python -c "from graph_hypernetwork_forge import HyperGNN; print('✓ Core imports working')"
    
    # Test development tools
    black --version > /dev/null && echo "✓ Black available"
    isort --version > /dev/null && echo "✓ isort available"
    flake8 --version > /dev/null && echo "✓ flake8 available"
    mypy --version > /dev/null && echo "✓ mypy available"
    pytest --version > /dev/null && echo "✓ pytest available"
    
    log_success "Installation verified"
}

# Setup IDE configurations
setup_ide_configs() {
    log_info "Setting up IDE configurations..."
    
    # VS Code settings
    if [[ ! -d ".vscode" ]]; then
        mkdir -p .vscode
        
        cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/htmlcov": true
    }
}
EOF
        
        cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${workspaceFolder}/tests", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF
        
        log_success "VS Code configuration created"
    else
        log_warning "VS Code configuration already exists"
    fi
}

# Run initial tests
run_initial_tests() {
    log_info "Running initial test suite..."
    
    if pytest tests/ -v --tb=short; then
        log_success "All tests passed!"
    else
        log_warning "Some tests failed. This is normal for a new setup."
    fi
}

# Display next steps
show_next_steps() {
    log_success "Development environment setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Run tests: make test"
    echo "3. Check code quality: make lint"
    echo "4. Format code: make format"
    echo "5. Build documentation: make docs"
    echo ""
    echo "Available make targets:"
    echo "  make help          - Show all available commands"
    echo "  make dev-check     - Run quick development checks"
    echo "  make ci-check      - Run full CI checks"
    echo ""
    echo "Happy coding! 🚀"
}

# Main execution
main() {
    log_info "Starting Graph Hypernetwork Forge development environment setup..."
    
    check_python_version
    create_virtual_env
    activate_virtual_env
    install_dependencies
    setup_pre_commit
    verify_installation
    setup_ide_configs
    run_initial_tests
    show_next_steps
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "clean")
        log_info "Cleaning development environment..."
        rm -rf venv/
        rm -rf .vscode/
        rm -rf build/
        rm -rf dist/
        rm -rf *.egg-info/
        log_success "Development environment cleaned"
        ;;
    "update")
        log_info "Updating development environment..."
        activate_virtual_env
        pip install --upgrade pip
        pip install --upgrade -e .[dev,security,docs,performance]
        pre-commit autoupdate
        log_success "Development environment updated"
        ;;
    "test")
        activate_virtual_env
        run_initial_tests
        ;;
    "help")
        echo "Graph Hypernetwork Forge Development Environment Setup"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  setup    - Setup complete development environment (default)"
        echo "  clean    - Clean all development artifacts"
        echo "  update   - Update dependencies and tools"
        echo "  test     - Run test suite"
        echo "  help     - Show this help message"
        ;;
    *)
        log_error "Unknown command: $1"
        echo "Run '$0 help' for available commands"
        exit 1
        ;;
esac