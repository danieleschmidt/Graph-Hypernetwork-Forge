# Getting Started

This guide will help you get up and running with Graph Hypernetwork Forge quickly.

## Installation

### From Source

```bash
git clone https://github.com/yourusername/graph-hypernetwork-forge.git
cd graph-hypernetwork-forge
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e .[dev,security,docs,performance]

# Set up pre-commit hooks
pre-commit install
```

### Docker Installation

```bash
# Build the development image
docker-compose build dev

# Run development environment
docker-compose up dev
```

## Quick Example

Here's a simple example to get you started:

```python
import torch
from graph_hypernetwork_forge import GraphHypernetwork
from torch_geometric.data import Data

# Create sample graph data
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.randn(3, 16)  # 3 nodes, 16 features each
data = Data(x=x, edge_index=edge_index)

# Initialize the hypernetwork
model = GraphHypernetwork(
    text_encoder_dim=768,
    node_feature_dim=16,
    hidden_dim=128,
    num_layers=2,
    num_heads=4
)

# Define task description
task_description = "Node classification on a small social network"

# Generate weights and perform inference
with torch.no_grad():
    output = model(data, task_description)
    print(f"Output shape: {output.shape}")
```

## Configuration

Graph Hypernetwork Forge uses Hydra for configuration management. Create a config file:

```yaml
# config/experiment.yaml
model:
  text_encoder_dim: 768
  node_feature_dim: 16
  hidden_dim: 256
  num_layers: 3
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  weight_decay: 0.0001

data:
  dataset_name: "cora"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
```

## Training Your First Model

```python
from graph_hypernetwork_forge.training import Trainer
from graph_hypernetwork_forge.data import GraphDataModule
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config", config_name="experiment")
def train(cfg: DictConfig):
    # Initialize data module
    data_module = GraphDataModule(cfg.data)
    
    # Initialize model
    model = GraphHypernetwork(
        text_encoder_dim=cfg.model.text_encoder_dim,
        node_feature_dim=cfg.model.node_feature_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout
    )
    
    # Initialize trainer
    trainer = Trainer(cfg.training)
    
    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    train()
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand the system design
- Explore [Examples](examples.md) for more complex use cases
- Check out the [API Reference](api_reference.rst) for detailed documentation
- See [Development Guide](development.md) for contributing guidelines

## Common Issues

### CUDA Out of Memory

If you encounter CUDA out of memory errors:

```python
# Reduce batch size
cfg.training.batch_size = 16

# Use gradient accumulation
cfg.training.gradient_accumulation_steps = 2

# Use mixed precision training
cfg.training.use_amp = True
```

### Slow Training

For faster training:

```python
# Use multiple GPUs
cfg.training.gpus = [0, 1, 2, 3]

# Increase number of workers
cfg.data.num_workers = 8

# Use pre-trained text encoder
cfg.model.text_encoder_pretrained = True
```

### Memory Profiling

To monitor memory usage:

```bash
# Use memory profiler
python -m memory_profiler train.py

# Use PyTorch profiler
python train.py --profile
```