"""Hypernetwork modules for the Graph Hypernetwork Forge.

This module contains hypernetwork classes that generate neural network weights
from text embeddings, originally part of hypergnn.py but extracted for modularity.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn


class WeightGenerator(nn.Module):
    """Base class for weight generators."""
    
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """Initialize weight generator.
        
        Args:
            text_dim: Text embedding dimension
            hidden_dim: Hidden dimension for generated weights
            num_layers: Number of layers in the generator
            dropout: Dropout probability
        """
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build generator network
        layers = []
        in_dim = text_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i == num_layers - 1 else hidden_dim * 2
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim
        
        self.generator = nn.Sequential(*layers)
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Generate weights from text embeddings.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            
        Returns:
            Generated weights [batch_size, hidden_dim]
        """
        return self.generator(text_embeddings)


class HyperNetwork(nn.Module):
    """Generates GNN weights from text embeddings."""
    
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        num_layers: int,
        gnn_type: str = "GAT",
        dropout: float = 0.1,
    ):
        """Initialize hypernetwork.
        
        Args:
            text_dim: Text embedding dimension
            hidden_dim: GNN hidden dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN backbone (GCN, GAT, SAGE)
            dropout: Dropout probability
        """
        super().__init__()
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.upper()
        self.dropout = dropout
        
        # Calculate weight dimensions based on GNN type
        self.weight_dims = self._calculate_weight_dimensions()
        
        # Hypernetwork layers
        self.hypernetwork = nn.ModuleDict()
        
        # Generate weights for each GNN layer
        for layer_idx in range(num_layers):
            layer_generators = nn.ModuleDict()
            
            for weight_name, weight_shape in self.weight_dims[layer_idx].items():
                # Create generator for this weight
                weight_size = torch.prod(torch.tensor(weight_shape)).item()
                
                generator = nn.Sequential(
                    nn.Linear(self.text_dim, self.text_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.text_dim * 2, self.text_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.text_dim, weight_size),
                )
                
                layer_generators[weight_name] = generator
            
            self.hypernetwork[f"layer_{layer_idx}"] = layer_generators
        
        # Weight normalization factors
        self.weight_scales = nn.ParameterDict()
        for layer_idx in range(num_layers):
            layer_scales = nn.ParameterDict()
            for weight_name in self.weight_dims[layer_idx].keys():
                layer_scales[weight_name] = nn.Parameter(torch.ones(1) * 0.1)
            self.weight_scales[f"layer_{layer_idx}"] = layer_scales
    
    def _calculate_weight_dimensions(self) -> List[Dict[str, Tuple[int, ...]]]:
        """Calculate weight dimensions for each layer."""
        weight_dims = []
        
        for layer_idx in range(self.num_layers):
            layer_dims = {}
            
            if layer_idx == 0:
                # First layer: input features to hidden
                in_dim = None  # Will be set dynamically
                out_dim = self.hidden_dim
            elif layer_idx == self.num_layers - 1:
                # Last layer: hidden to output
                in_dim = self.hidden_dim
                out_dim = None  # Will be set dynamically
            else:
                # Middle layers: hidden to hidden
                in_dim = self.hidden_dim
                out_dim = self.hidden_dim
            
            # Weight dimensions depend on GNN type
            if self.gnn_type == "GCN":
                layer_dims["weight"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["bias"] = (out_dim or self.hidden_dim,)
            elif self.gnn_type == "GAT":
                # For simplicity, assume single head GAT
                layer_dims["weight"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["att_weight"] = (2 * (out_dim or self.hidden_dim), 1)
                layer_dims["bias"] = (out_dim or self.hidden_dim,)
            elif self.gnn_type == "SAGE":
                layer_dims["lin_l"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["lin_r"] = (in_dim or self.hidden_dim, out_dim or self.hidden_dim)
                layer_dims["bias"] = (out_dim or self.hidden_dim,)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
            
            weight_dims.append(layer_dims)
        
        return weight_dims
    
    def forward(
        self, 
        text_embeddings: torch.Tensor,
        input_dim: int,
        output_dim: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate GNN weights from text embeddings.
        
        Args:
            text_embeddings: Text embeddings [num_nodes, text_dim]
            input_dim: Input feature dimension
            output_dim: Output dimension
            
        Returns:
            List of weight dictionaries for each layer
        """
        batch_size = text_embeddings.size(0)
        generated_weights = []
        
        for layer_idx in range(self.num_layers):
            layer_weights = {}
            layer_generators = self.hypernetwork[f"layer_{layer_idx}"]
            layer_scales = self.weight_scales[f"layer_{layer_idx}"]
            
            # Update dimensions for first and last layers
            updated_dims = self.weight_dims[layer_idx].copy()
            
            # Determine actual output dimension for this layer
            if layer_idx == self.num_layers - 1:
                actual_output_dim = output_dim
            else:
                actual_output_dim = self.hidden_dim
            
            # Update first layer input dimensions
            if layer_idx == 0:
                for key in updated_dims:
                    if key == "weight" or "lin_l" in key or "lin_r" in key:
                        shape = list(updated_dims[key])
                        shape[0] = input_dim
                        updated_dims[key] = tuple(shape)
            
            # Update output dimensions for all layers
            for key in updated_dims:
                if key == "weight" or "lin_l" in key or "lin_r" in key:
                    shape = list(updated_dims[key])
                    shape[1] = actual_output_dim
                    updated_dims[key] = tuple(shape)
                elif "bias" in key:
                    updated_dims[key] = (actual_output_dim,)
                elif key == "att_weight":
                    shape = list(updated_dims[key])
                    shape[0] = 2 * actual_output_dim  # 2 * output_dim for concatenated attention
                    updated_dims[key] = tuple(shape)
            
            for weight_name, weight_shape in updated_dims.items():
                # Generate weights for each node
                flat_weights = layer_generators[weight_name](text_embeddings)
                
                # Calculate expected output size
                target_elements = torch.prod(torch.tensor(weight_shape)).item()
                
                # Ensure correct output size
                if flat_weights.size(-1) != target_elements:
                    raise RuntimeError(f"Weight generator for {weight_name} in layer {layer_idx} "
                                     f"produces {flat_weights.size(-1)} elements, "
                                     f"but expected {target_elements} for shape {weight_shape}")
                
                # Reshape to proper weight shape
                weight_tensor = flat_weights.view(batch_size, *weight_shape)
                
                # Apply scaling
                scale = layer_scales[weight_name]
                weight_tensor = weight_tensor * scale
                
                layer_weights[weight_name] = weight_tensor
            
            generated_weights.append(layer_weights)
        
        return generated_weights


class SimpleWeightGenerator(WeightGenerator):
    """Simple weight generator for basic use cases."""
    
    def __init__(
        self,
        text_dim: int,
        weight_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
    ):
        """Initialize simple weight generator.
        
        Args:
            text_dim: Text embedding dimension
            weight_dim: Dimension of generated weights
            hidden_dim: Hidden dimension (defaults to 2 * text_dim)
            dropout: Dropout probability
        """
        if hidden_dim is None:
            hidden_dim = 2 * text_dim
            
        super().__init__(text_dim, weight_dim, num_layers=2, dropout=dropout)
        self.weight_dim = weight_dim
    
    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        """Generate weight vectors.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            
        Returns:
            Generated weights [batch_size, weight_dim]
        """
        weights = super().forward(text_embeddings)
        return weights[:, :self.weight_dim]  # Ensure correct dimension