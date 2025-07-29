# Architecture Overview

## System Design

Graph Hypernetwork Forge implements a novel architecture where hypernetworks generate Graph Neural Network (GNN) weights dynamically from textual node descriptions.

## Core Components

### 1. Text Encoder Module (`graph_hypernetwork_forge.models.encoders`)

Transforms textual node descriptions into semantic embeddings:

```python
text → sentence_transformer → text_embeddings [batch_size, text_dim]
```

**Supported Encoders**:
- Sentence Transformers (default)
- BERT/RoBERTa variants
- Custom domain-specific encoders

### 2. Hypernetwork Core (`graph_hypernetwork_forge.models.hypernetworks`)

Maps text embeddings to GNN weight tensors:

```python
text_embeddings → hypernetwork → gnn_weights [layers, weight_matrices]
```

**Architecture**:
- Multi-layer perceptron with residual connections
- Generates weights for each GNN layer
- Supports different GNN architectures (GCN, GAT, GraphSAGE)

### 3. Dynamic GNN (`graph_hypernetwork_forge.models.gnns`)

Applies generated weights for graph convolutions:

```python
(node_features, edge_index, generated_weights) → gnn_forward → predictions
```

**Features**:
- Interchangeable GNN backbones
- Efficient weight application
- Supports batched inference

### 4. Zero-Shot Adapter (`graph_hypernetwork_forge.models.adapters`)

Handles domain transfer between training and inference:

```python
source_domain_weights → adaptation_mechanism → target_domain_weights
```

## Data Flow

1. **Input**: Knowledge graph with textual node metadata
2. **Encoding**: Text descriptions → semantic embeddings
3. **Weight Generation**: Embeddings → GNN parameters
4. **Graph Processing**: Apply weights to graph structure
5. **Output**: Node/edge predictions or representations

## Key Design Principles

### Modularity
- Swappable components (encoders, GNNs, hypernetworks)
- Clear interfaces between modules
- Extensible architecture

### Efficiency
- Batch processing support
- Memory-efficient weight generation
- GPU acceleration throughout

### Flexibility
- Multiple GNN backbone support
- Configurable hypernetwork architectures
- Domain-specific customization

## Memory and Computational Complexity

### Memory Usage
- Text embeddings: O(n × d_text)
- Generated weights: O(L × d_hidden²) where L = layers
- Graph processing: O(n × d_hidden + m) where m = edges

### Computational Complexity
- Text encoding: O(n × sequence_length)
- Weight generation: O(n × hypernetwork_params)
- GNN forward pass: O(m × d_hidden)

## Extension Points

### Custom Text Encoders
```python
class CustomEncoder(TextEncoder):
    def encode(self, texts: List[str]) -> torch.Tensor:
        # Custom encoding logic
        pass
```

### Custom Hypernetworks
```python
class CustomHypernetwork(BaseHypernetwork):
    def generate_weights(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Custom weight generation
        pass
```

### Custom GNN Backends
```python
class CustomGNN(BaseGNN):
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Custom graph processing
        pass
```

## Performance Considerations

- **Batch Size**: Optimal batch size depends on GPU memory
- **Text Length**: Longer descriptions increase encoding time
- **Graph Size**: Large graphs may require gradient checkpointing
- **Weight Caching**: Consider caching generated weights for repeated inference