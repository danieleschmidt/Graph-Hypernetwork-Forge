#!/bin/bash

# Post-create script for Graph Hypernetwork Forge dev container
# This script runs after the container is created to set up the development environment

set -e

echo "🚀 Setting up Graph Hypernetwork Forge development environment..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update && apt-get upgrade -y

# Install additional system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y \
    build-essential \
    git-lfs \
    htop \
    tree \
    jq \
    curl \
    wget \
    vim \
    nano \
    graphviz \
    graphviz-dev \
    pkg-config

# Set up Git LFS
echo "📁 Initializing Git LFS..."
git lfs install

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install development dependencies
if [ -f "requirements.txt" ]; then
    echo "📋 Installing project dependencies..."
    pip install -r requirements.txt
fi

if [ -f "requirements-dev-lock.txt" ]; then
    echo "🛠️ Installing development dependencies..."
    pip install -r requirements-dev-lock.txt
fi

# Install project in development mode
echo "🔗 Installing project in development mode..."
pip install -e .

# Install pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
if [ -f ".pre-commit-config.yaml" ]; then
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Create necessary directories
echo "📁 Creating development directories..."
mkdir -p \
    data/raw \
    data/processed \
    experiments \
    logs \
    models \
    notebooks \
    .cache/torch \
    .cache/transformers \
    .cache/sentence_transformers

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
python -m ipykernel install --user --name graph-hypernetwork-forge --display-name "Graph Hypernetwork Forge"

# Download pre-trained models (if specified)
echo "🤖 Downloading pre-trained models..."
python -c "
import sentence_transformers
try:
    model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
    print('✅ Downloaded all-mpnet-base-v2')
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2')
    print('✅ Downloaded all-MiniLM-L6-v2')
except Exception as e:
    print(f'⚠️ Error downloading models: {e}')
"

# Set up Git configuration if not already set
echo "⚙️ Configuring Git..."
if [ -z "$(git config --global user.name)" ]; then
    echo "Please configure Git with your name and email:"
    echo "git config --global user.name 'Your Name'"
    echo "git config --global user.email 'your.email@example.com'"
fi

# Create .env file from example if it doesn't exist
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "⚠️ Please update .env with your specific configuration"
fi

# Run initial tests to verify setup
echo "🧪 Running initial tests..."
if command -v pytest >/dev/null 2>&1; then
    python -m pytest --version
    echo "✅ pytest is available"
else
    echo "⚠️ pytest not found, please install development dependencies"
fi

# Display environment info
echo "📊 Environment Information:"
echo "Python version: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
echo "Working directory: $(pwd)"

echo "✅ Development environment setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "  make help           - Show available commands"
echo "  make test           - Run tests" 
echo "  make lint           - Run linting"
echo "  make format         - Format code"
echo "  make notebook       - Start Jupyter Lab"
echo "  make train          - Start training"
echo ""
echo "Happy coding! 🎉"