"""Dynamic Graph Neural Network implementations.

GNN layers that use dynamically generated weights from the hypernetwork
instead of static learned parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.typing import Adj, OptTensor, PairTensor
from typing import Dict, Optional, Union, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class DynamicGCNLayer(MessagePassing):
    """Graph Convolution Layer with dynamic weights."""
    
    def __init__(
        self,
        hidden_dim: int,
        layer_idx: int,
        dropout: float = 0.1,
        bias: bool = True,
        add_self_loops: bool = True,
        normalize: bool = True,
    ):
        super().__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        # These will be set dynamically
        self.weight = None
        self.bias = None
        
    def set_weights(self, weights: Dict[str, torch.Tensor], batch_idx: int = 0):
        """Set dynamic weights for this layer."""
        weight_key = f"gcn_layer_{self.layer_idx}_weight"
        bias_key = f"gcn_layer_{self.layer_idx}_bias"
        
        if weight_key in weights:
            self.weight = weights[weight_key][batch_idx] if weights[weight_key].dim() > 2 else weights[weight_key]
        if bias_key in weights:
            self.bias = weights[bias_key][batch_idx] if weights[bias_key].dim() > 1 else weights[bias_key]
    
    def forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        if self.weight is None:
            raise ValueError("Weights must be set before forward pass")
            
        # Add self-loops to the adjacency matrix
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Apply linear transformation
        x = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            x = x + self.bias
        
        # Normalize node features
        if self.normalize:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = None
        
        # Propagate messages
        return self.propagate(edge_index, x=x, norm=norm)
    
    def message(self, x_j: torch.Tensor, norm: OptTensor) -> torch.Tensor:
        if norm is not None:
            return norm.view(-1, 1) * x_j
        return x_j


class DynamicGATLayer(MessagePassing):
    """Graph Attention Layer with dynamic weights."""
    
    def __init__(
        self,
        hidden_dim: int,
        layer_idx: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        assert hidden_dim % num_heads == 0
        self.head_dim = hidden_dim // num_heads
        
        # Dynamic weights
        self.weight = None
        self.bias = None
        self.att_src = None
        self.att_dst = None
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        
    def set_weights(self, weights: Dict[str, torch.Tensor], batch_idx: int = 0):
        """Set dynamic weights for this layer."""
        weight_key = f"gat_layer_{self.layer_idx}_weight"
        bias_key = f"gat_layer_{self.layer_idx}_bias"
        att_src_key = f"gat_layer_{self.layer_idx}_att_src"
        att_dst_key = f"gat_layer_{self.layer_idx}_att_dst"
        
        if weight_key in weights:
            self.weight = weights[weight_key][batch_idx] if weights[weight_key].dim() > 2 else weights[weight_key]
        if bias_key in weights:
            self.bias = weights[bias_key][batch_idx] if weights[bias_key].dim() > 1 else weights[bias_key]
        if att_src_key in weights:
            self.att_src = weights[att_src_key][batch_idx] if weights[att_src_key].dim() > 2 else weights[att_src_key]
        if att_dst_key in weights:
            self.att_dst = weights[att_dst_key][batch_idx] if weights[att_dst_key].dim() > 2 else weights[att_dst_key]
    
    def forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        if self.weight is None or self.att_src is None or self.att_dst is None:
            raise ValueError("All weights must be set before forward pass")
        
        H, C = self.num_heads, self.head_dim
        
        # Linear transformation
        x = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            x = x + self.bias
        
        # Reshape for multi-head attention
        x = x.view(-1, H, C)
        
        # Propagate messages
        out = self.propagate(edge_index, x=x, size=None)
        
        if self.concat:
            out = out.view(-1, self.hidden_dim)
        else:
            out = out.mean(dim=1)
        
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_index_i: torch.Tensor) -> torch.Tensor:
        # Compute attention coefficients
        alpha_src = (x_i * self.att_src.view(self.num_heads, self.head_dim)).sum(dim=-1)
        alpha_dst = (x_j * self.att_dst.view(self.num_heads, self.head_dim)).sum(dim=-1)
        alpha = alpha_src + alpha_dst
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Apply attention weights
        return x_j * alpha.unsqueeze(-1)


class DynamicSAGELayer(MessagePassing):
    """GraphSAGE Layer with dynamic weights."""
    
    def __init__(
        self,
        hidden_dim: int,
        layer_idx: int,
        aggr: str = 'mean',
        dropout: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__(aggr=aggr)
        
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.normalize = normalize
        
        # Dynamic weights
        self.self_weight = None
        self.neighbor_weight = None
        self.bias = None
        
    def set_weights(self, weights: Dict[str, torch.Tensor], batch_idx: int = 0):
        """Set dynamic weights for this layer."""
        self_key = f"sage_layer_{self.layer_idx}_self_weight"
        neighbor_key = f"sage_layer_{self.layer_idx}_neighbor_weight"
        bias_key = f"sage_layer_{self.layer_idx}_bias"
        
        if self_key in weights:
            self.self_weight = weights[self_key][batch_idx] if weights[self_key].dim() > 2 else weights[self_key]
        if neighbor_key in weights:
            self.neighbor_weight = weights[neighbor_key][batch_idx] if weights[neighbor_key].dim() > 2 else weights[neighbor_key]
        if bias_key in weights:
            self.bias = weights[bias_key][batch_idx] if weights[bias_key].dim() > 1 else weights[bias_key]
    
    def forward(self, x: torch.Tensor, edge_index: Adj) -> torch.Tensor:
        if self.self_weight is None or self.neighbor_weight is None:
            raise ValueError("Weights must be set before forward pass")
        
        # Self transformation
        x_self = torch.matmul(x, self.self_weight.t())
        
        # Neighbor aggregation and transformation
        x_neighbor = self.propagate(edge_index, x=x)
        x_neighbor = torch.matmul(x_neighbor, self.neighbor_weight.t())
        
        # Combine self and neighbor representations
        out = x_self + x_neighbor
        
        if self.bias is not None:
            out = out + self.bias
            
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        
        return out
    
    def message(self, x_j: torch.Tensor) -> torch.Tensor:
        return x_j


class DynamicGNN(nn.Module):
    """Dynamic Graph Neural Network using generated weights."""
    
    def __init__(
        self,
        backbone: str = "GAT",
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        activation: str = "relu",
        layer_norm: bool = True,
    ):
        super().__init__()
        
        self.backbone = backbone.upper()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        elif activation == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu
        
        # Build GNN layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if self.backbone == "GCN":
                layer = DynamicGCNLayer(
                    hidden_dim=hidden_dim,
                    layer_idx=i,
                    dropout=dropout,
                )
            elif self.backbone == "GAT":
                layer = DynamicGATLayer(
                    hidden_dim=hidden_dim,
                    layer_idx=i,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            elif self.backbone == "GRAPHSAGE":
                layer = DynamicSAGELayer(
                    hidden_dim=hidden_dim,
                    layer_idx=i,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone}")
            
            self.layers.append(layer)
        
        # Layer normalization
        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.layer_norms = None
            
        # Edge feature processing
        if edge_dim is not None:
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_encoder = None
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        edge_attr: Optional[torch.Tensor] = None,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        """Forward pass through dynamic GNN.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            weights: Generated weights from hypernetwork
            edge_attr: Optional edge attributes
            batch_idx: Batch index for weight selection
            
        Returns:
            Node embeddings after GNN processing
        """
        # Process edge attributes if provided
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Apply GNN layers
        for i, layer in enumerate(self.layers):
            # Set dynamic weights for this layer
            layer.set_weights(weights, batch_idx)
            
            # Apply layer
            x_new = layer(x, edge_index)
            
            # Apply activation (except for last layer)
            if i < len(self.layers) - 1:
                x_new = self.activation(x_new)
            
            # Apply layer normalization
            if self.layer_norms is not None:
                x_new = self.layer_norms[i](x_new)
            
            # Apply dropout
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # Residual connection
            if x.size() == x_new.size():
                x = x + x_new
            else:
                x = x_new
        
        return x


def get_gnn_backbone(
    backbone: str,
    hidden_dim: int,
    num_layers: int,
    **kwargs
) -> DynamicGNN:
    """Factory function to create GNN backbones.
    
    Args:
        backbone: GNN type ('GCN', 'GAT', 'GraphSAGE')
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        **kwargs: Additional arguments
        
    Returns:
        DynamicGNN instance
    """
    return DynamicGNN(
        backbone=backbone,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
    )


# Available GNN architectures
AVAILABLE_GNNS = {
    "GCN": "Graph Convolution Network",
    "GAT": "Graph Attention Network", 
    "GRAPHSAGE": "GraphSAGE Network",
}