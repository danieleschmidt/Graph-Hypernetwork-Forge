# Multi-stage build for Graph Hypernetwork Forge
FROM python:3.13.7-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Development stage
FROM base as development

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt pyproject.toml ./

# Install dependencies including dev dependencies
RUN pip install --user -r requirements.txt && \
    pip install --user -e .[dev]

# Copy source code
COPY --chown=app:app . .

# Install package in development mode
RUN pip install --user -e .

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# Production stage
FROM base as production

# Copy requirements first for better caching
COPY --chown=app:app requirements.txt pyproject.toml ./

# Install only production dependencies
RUN pip install --user -r requirements.txt

# Copy source code
COPY --chown=app:app graph_hypernetwork_forge/ ./graph_hypernetwork_forge/
COPY --chown=app:app README.md LICENSE ./

# Install package
RUN pip install --user .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import graph_hypernetwork_forge; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "from graph_hypernetwork_forge import HyperGNN; print('Graph Hypernetwork Forge ready')"]

# Training stage for ML workloads
FROM production as training

# Install additional training dependencies
RUN pip install --user wandb jupyter notebook

# Create directories for data and models
RUN mkdir -p data models logs

# Expose Jupyter port
EXPOSE 8888

# Default command for training
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]