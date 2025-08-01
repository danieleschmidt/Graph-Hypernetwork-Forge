# ADR-0003: GNN Backend Abstraction Design

**Date**: 2025-08-01  
**Status**: Accepted  
**Deciders**: Daniel Schmidt, Core Development Team  

## Context and Problem Statement

The hypernetwork generates weights for Graph Neural Network layers, but different GNN architectures (GCN, GAT, GraphSAGE) have varying parameter structures, weight shapes, and computational patterns. We need a flexible abstraction that allows the hypernetwork to generate appropriate weights for any GNN backend while maintaining modularity and extensibility.

The abstraction must handle differences in:
- Weight tensor shapes and structures
- Layer-specific parameters (attention heads, aggregation methods)
- Activation functions and normalization schemes
- Forward pass implementations

## Decision Drivers

- **Modularity**: Easy swapping between different GNN architectures
- **Extensibility**: Simple addition of new GNN backends
- **Performance**: Minimal overhead from abstraction layer
- **Type Safety**: Clear interfaces and parameter validation
- **Maintainability**: Clean separation of concerns
- **PyTorch Geometric Compatibility**: Leverage existing implementations

## Considered Options

### Option 1: Inheritance-Based Hierarchy
- Create abstract base class with concrete implementations for each GNN type
- **Pros**: Clear inheritance structure, polymorphic behavior
- **Cons**: Rigid hierarchy, difficult to compose different features

### Option 2: Strategy Pattern with Plugins
- Use strategy pattern with plugin-based GNN implementations
- **Pros**: Runtime switching, easy plugin development
- **Cons**: Complex plugin system, runtime overhead

### Option 3: Factory Pattern with Registry (Chosen)
- Combine factory pattern with registry system for GNN backends
- **Pros**: Clean registration, easy testing, compile-time safety
- **Cons**: Requires registration boilerplate

### Option 4: Configuration-Driven Dynamic Loading
- Load GNN implementations dynamically based on configuration
- **Pros**: Maximum flexibility, no code changes for new backends
- **Cons**: Complex configuration, runtime errors, debugging difficulty

## Decision Outcome

**Chosen option**: Factory Pattern with Registry (Option 3)

**Rationale**: The factory pattern with registry system provides the optimal balance of flexibility, type safety, and maintainability. It allows clean separation of hypernetwork weight generation and GNN implementation while maintaining compile-time verification and easy testing.

### Positive Consequences
- **Clean Interface**: Well-defined contract between hypernetwork and GNN
- **Easy Extension**: Simple registration of new GNN backends
- **Type Safety**: Full TypeScript-style type checking with mypy
- **Testability**: Easy mocking and unit testing
- **Performance**: Minimal runtime overhead
- **PyTorch Geometric Integration**: Leverages existing optimized implementations

### Negative Consequences
- **Registration Boilerplate**: Requires explicit registration of backends
- **Import Management**: Need to manage imports for all backends
- **Documentation Overhead**: Must document each backend's weight requirements

## Implementation Notes

### Core Abstraction Interface
```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import torch
from torch_geometric.data import Data

class GNNBackend(ABC):
    """Abstract base class for GNN backend implementations."""
    
    @abstractmethod
    def get_weight_specs(self, input_dim: int, hidden_dim: int, 
                        output_dim: int, num_layers: int) -> Dict[str, Tuple[int, ...]]:
        """Return weight tensor specifications for hypernetwork generation."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                weights: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Forward pass with generated weights."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass
```

### Registry System
```python
class GNNRegistry:
    """Registry for GNN backend implementations."""
    
    _backends: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register GNN backend."""
        def decorator(backend_class):
            cls._backends[name] = backend_class
            return backend_class
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> GNNBackend:
        """Create GNN backend instance."""
        if name not in cls._backends:
            raise ValueError(f"Unknown GNN backend: {name}")
        return cls._backends[name](**kwargs)
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List available backend names."""
        return list(cls._backends.keys())
```

### Concrete Backend Implementation
```python
@GNNRegistry.register("gcn")
class GCNBackend(GNNBackend):
    """Graph Convolutional Network backend."""
    
    def __init__(self, dropout: float = 0.1, normalize: bool = True):
        self.dropout = dropout
        self.normalize = normalize
    
    def get_weight_specs(self, input_dim: int, hidden_dim: int,
                        output_dim: int, num_layers: int) -> Dict[str, Tuple[int, ...]]:
        """GCN weight specifications."""
        specs = {}
        
        # Input layer
        specs["layer_0_weight"] = (input_dim, hidden_dim)
        specs["layer_0_bias"] = (hidden_dim,)
        
        # Hidden layers
        for i in range(1, num_layers - 1):
            specs[f"layer_{i}_weight"] = (hidden_dim, hidden_dim)
            specs[f"layer_{i}_bias"] = (hidden_dim,)
        
        # Output layer
        specs[f"layer_{num_layers-1}_weight"] = (hidden_dim, output_dim)
        specs[f"layer_{num_layers-1}_bias"] = (output_dim,)
        
        return specs
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                weights: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """GCN forward pass with generated weights."""
        from torch_geometric.nn import GCNConv
        
        # Implementation using generated weights
        h = x
        for i, (weight_key, bias_key) in enumerate(self._get_layer_keys()):
            conv = GCNConv(weights[weight_key].shape[0], 
                          weights[weight_key].shape[1])
            
            # Apply generated weights
            conv.lin.weight.data = weights[weight_key].T
            conv.lin.bias.data = weights[bias_key]
            
            h = conv(h, edge_index)
            if i < len(self._get_layer_keys()) - 1:
                h = torch.relu(h)
                h = torch.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    @property
    def name(self) -> str:
        return "gcn"
```

### Hypernetwork Integration
```python
class DynamicGNN(nn.Module):
    """Dynamic GNN with pluggable backends."""
    
    def __init__(self, backend_name: str, input_dim: int, 
                 hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        
        self.backend = GNNRegistry.create(backend_name)
        self.weight_specs = self.backend.get_weight_specs(
            input_dim, hidden_dim, output_dim, num_layers
        )
        
        # Validate weight specifications
        self._validate_weight_specs()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                generated_weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with hypernetwork-generated weights."""
        
        # Validate generated weights match specifications
        self._validate_generated_weights(generated_weights)
        
        return self.backend.forward(x, edge_index, generated_weights)
```

### Weight Generation Interface
```python
class WeightGenerator(nn.Module):
    """Generates GNN weights from text embeddings."""
    
    def __init__(self, text_dim: int, weight_specs: Dict[str, Tuple[int, ...]]):
        super().__init__()
        self.weight_specs = weight_specs
        self.generators = nn.ModuleDict()
        
        # Create weight generators for each specification
        for name, shape in weight_specs.items():
            total_params = torch.prod(torch.tensor(shape)).item()
            self.generators[name] = nn.Sequential(
                nn.Linear(text_dim, total_params * 2),
                nn.ReLU(),
                nn.Linear(total_params * 2, total_params)
            )
    
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate weights from text embeddings."""
        weights = {}
        
        for name, shape in self.weight_specs.items():
            flat_weights = self.generators[name](text_embeddings)
            weights[name] = flat_weights.view(-1, *shape)
        
        return weights
```

## Backend Implementations

### Supported Backends

1. **GCN (Graph Convolutional Network)**
   - Simple message passing with symmetric normalization
   - Efficient for homophilic graphs

2. **GAT (Graph Attention Network)**
   - Multi-head attention mechanisms
   - Better for heterophilic graphs

3. **GraphSAGE (Graph Sample and Aggregate)**
   - Inductive learning capability
   - Sampling-based training

4. **GIN (Graph Isomorphism Network)**
   - Theoretical guarantees for graph isomorphism
   - Strong expressive power

### Extension Template
```python
@GNNRegistry.register("custom_gnn")
class CustomGNNBackend(GNNBackend):
    def __init__(self, **kwargs):
        # Initialize custom parameters
        pass
    
    def get_weight_specs(self, input_dim: int, hidden_dim: int,
                        output_dim: int, num_layers: int) -> Dict[str, Tuple[int, ...]]:
        # Define weight tensor specifications
        return {}
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                weights: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        # Implement forward pass
        return torch.zeros_like(x)
    
    @property
    def name(self) -> str:
        return "custom_gnn"
```

## Configuration Management

### Backend Configuration
```python
GNN_CONFIGS = {
    "gcn": {
        "class": "GCNBackend",
        "params": {"dropout": 0.1, "normalize": True}
    },
    "gat": {
        "class": "GATBackend", 
        "params": {"heads": 8, "dropout": 0.1, "concat": False}
    },
    "sage": {
        "class": "GraphSAGEBackend",
        "params": {"aggr": "mean", "normalize": True}
    }
}
```

### Runtime Selection
```python
def create_model(config: Dict) -> nn.Module:
    backend_name = config["gnn"]["backend"]
    backend_params = config["gnn"].get("params", {})
    
    gnn_backend = GNNRegistry.create(backend_name, **backend_params)
    return HyperGNN(
        text_encoder=config["text"]["encoder"],
        gnn_backend=gnn_backend,
        **config["model"]
    )
```

## Testing Strategy

### Unit Tests
- Weight specification validation
- Forward pass correctness
- Registry functionality
- Backend registration/creation

### Integration Tests
- End-to-end training with different backends
- Weight generation compatibility
- Performance benchmarking

### Backend-Specific Tests
```python
class TestGNNBackend:
    @pytest.mark.parametrize("backend_name", ["gcn", "gat", "sage"])
    def test_weight_specs(self, backend_name):
        backend = GNNRegistry.create(backend_name)
        specs = backend.get_weight_specs(10, 64, 2, 3)
        assert all(isinstance(shape, tuple) for shape in specs.values())
    
    def test_forward_pass(self, backend_name):
        # Test forward pass with mock weights
        pass
```

## Performance Considerations

### Memory Efficiency
- Lazy weight generation to reduce memory usage
- Weight sharing for similar node types
- Gradient checkpointing for large graphs

### Computational Efficiency
- Vectorized weight application
- Batch processing support
- CUDA kernel optimization where applicable

## Links and References

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Graph Convolutional Networks (Kipf & Welling)](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks (Veličković et al.)](https://arxiv.org/abs/1710.10903)
- [GraphSAGE (Hamilton et al.)](https://arxiv.org/abs/1706.02216)
- [Factory Pattern (Gang of Four)](https://en.wikipedia.org/wiki/Factory_method_pattern)

## Future Considerations

- **Automatic Backend Selection**: Choose optimal backend based on graph properties
- **Hybrid Architectures**: Combine multiple GNN types in single model
- **Custom Layer Support**: Support for user-defined GNN layers
- **Hardware Optimization**: Specialized implementations for different hardware