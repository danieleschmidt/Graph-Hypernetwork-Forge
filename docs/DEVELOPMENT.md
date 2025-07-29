# Development Guide

This guide covers development setup, architecture, and workflow for Graph Hypernetwork Forge.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/graph-hypernetwork-forge.git
cd graph-hypernetwork-forge
make install-dev

# Run tests and quality checks
make test
make quality
```

## Architecture Overview

### Core Components

1. **Text Encoder Module** (`src/graph_hypernetwork_forge/models/encoders/`)
   - Processes node descriptions into semantic embeddings
   - Supports various pre-trained language models
   - Handles domain-specific vocabulary

2. **Hypernetwork Core** (`src/graph_hypernetwork_forge/models/hypernetworks/`)
   - Maps text embeddings to GNN weight tensors
   - Generates layer-specific parameters dynamically
   - Enables zero-shot transfer capabilities

3. **Dynamic GNN** (`src/graph_hypernetwork_forge/models/gnns/`)
   - Applies generated weights for graph convolutions
   - Supports multiple GNN architectures (GCN, GAT, GraphSAGE)
   - Handles heterogeneous graph structures

4. **Data Processing** (`src/graph_hypernetwork_forge/data/`)
   - Knowledge graph loading and preprocessing
   - Text extraction and normalization
   - Batch processing and sampling strategies

## Development Workflow

### Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff with strict settings
- **Type Checker**: MyPy with strict mode
- **Import Sorting**: Ruff built-in sorting

### Testing Strategy

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Property Tests**: Use hypothesis for edge cases
- **Performance Tests**: Benchmark critical paths

### Configuration Management

Uses Hydra for configuration:

```yaml
# configs/base.yaml
model:
  text_encoder: "sentence-transformers/all-MiniLM-L6-v2"
  gnn_backbone: "GAT"
```

Override via command line:
```bash
python scripts/train.py model.gnn_backbone=GCN
```

## Performance Considerations

### Memory Management
- Use gradient checkpointing for large models
- Implement efficient batching strategies
- Monitor GPU memory usage

### Optimization Tips
- Cache text embeddings when possible
- Use mixed precision training
- Implement dynamic graph sampling

## Debugging

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable gradient checkpointing

2. **Slow Training**
   - Profile with PyTorch profiler
   - Check data loading bottlenecks
   - Optimize text encoding pipeline

3. **Poor Performance**
   - Verify data preprocessing
   - Check hyperparameter ranges
   - Validate model architecture

### Debugging Tools

- Use `pdb` for step-through debugging
- Enable detailed logging with `PYTHONPATH`
- Use TensorBoard for training visualization
- Profile with `py-spy` for production issues

## Adding New Features

### New GNN Architecture

1. Create new module in `src/graph_hypernetwork_forge/models/gnns/`
2. Implement base interface
3. Add configuration options
4. Write comprehensive tests
5. Update documentation

### New Text Encoder

1. Extend `CustomTextEncoder` base class
2. Handle tokenization and encoding
3. Add caching mechanisms
4. Test with various input formats

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release PR
5. Tag release after merge
6. Publish to PyPI (automated)

## Getting Help

- **Issues**: GitHub issue tracker
- **Discussions**: GitHub discussions
- **Discord**: Join our community server
- **Email**: hypernetwork-forge@yourdomain.com