"""Graph Neural Network modules for the Graph Hypernetwork Forge.

This module contains GNN implementations that can use dynamically generated weights,
originally part of hypergnn.py but extracted for better modularity.
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicGNN(nn.Module):
    """Dynamic GNN that uses generated weights."""
    
    def __init__(self, gnn_type: str = "GAT", dropout: float = 0.1):
        """Initialize dynamic GNN.
        
        Args:
            gnn_type: Type of GNN (GCN, GAT, SAGE)
            dropout: Dropout probability
        """
        super().__init__()
        self.gnn_type = gnn_type.upper()
        self.dropout = dropout
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        generated_weights: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass with generated weights.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            generated_weights: Generated weights for each layer
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        current_x = x
        
        for layer_idx, layer_weights in enumerate(generated_weights):
            # Apply GNN layer with generated weights
            if self.gnn_type == "GCN":
                current_x = self._gcn_layer(current_x, edge_index, layer_weights)
            elif self.gnn_type == "GAT":
                current_x = self._gat_layer(current_x, edge_index, layer_weights)
            elif self.gnn_type == "SAGE":
                current_x = self._sage_layer(current_x, edge_index, layer_weights)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
            
            # Apply activation and dropout (except for last layer)
            if layer_idx < len(generated_weights) - 1:
                current_x = F.relu(current_x)
                current_x = F.dropout(current_x, p=self.dropout, training=self.training)
        
        return current_x
    
    def _gcn_layer(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """GCN layer with dynamic weights."""
        # Simple GCN implementation
        # x: [num_nodes, in_dim], weight: [num_nodes, in_dim, out_dim]
        weight = weights["weight"]  # [num_nodes, in_dim, out_dim]
        bias = weights["bias"]      # [num_nodes, out_dim]
        
        # Apply linear transformation per node
        out = torch.bmm(x.unsqueeze(1), weight).squeeze(1)  # [num_nodes, out_dim]
        out = out + bias
        
        # Apply message passing (simplified aggregation)
        row, col = edge_index
        out_messages = out[row]  # Messages from source nodes
        
        # Aggregate messages at target nodes
        out_agg = torch.zeros_like(out)
        out_agg.scatter_add_(0, col.unsqueeze(1).expand(-1, out.size(1)), out_messages)
        
        # Add self-loops effect
        out_agg = out_agg + out
        
        return out_agg
    
    def _gat_layer(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """GAT layer with dynamic weights."""
        weight = weights["weight"]      # [num_nodes, in_dim, out_dim]
        att_weight = weights["att_weight"]  # [num_nodes, 2*out_dim, 1]
        bias = weights["bias"]          # [num_nodes, out_dim]
        
        # Linear transformation
        h = torch.bmm(x.unsqueeze(1), weight).squeeze(1)  # [num_nodes, out_dim]
        
        # Attention mechanism (simplified)
        row, col = edge_index
        h_i = h[row]  # Source nodes
        h_j = h[col]  # Target nodes
        
        # Attention scores
        att_input = torch.cat([h_i, h_j], dim=1)  # [num_edges, 2*out_dim]
        # Use source node attention weights for each edge
        source_att_weights = att_weight[row]  # [num_edges, 2*out_dim, 1]
        # Batch matrix multiplication for attention scores
        att_scores = torch.bmm(att_input.unsqueeze(1), source_att_weights).squeeze()  # [num_edges]
        att_scores = F.leaky_relu(att_scores, 0.2)
        
        # Apply softmax per target node
        att_weights = torch.zeros_like(att_scores)
        for node in torch.unique(col):
            mask = col == node
            att_weights[mask] = F.softmax(att_scores[mask], dim=0)
        
        # Aggregate with attention
        weighted_messages = h_i * att_weights.unsqueeze(1)
        out = torch.zeros_like(h)
        out.scatter_add_(0, col.unsqueeze(1).expand(-1, h.size(1)), weighted_messages)
        
        # Add bias
        out = out + bias
        
        return out
    
    def _sage_layer(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """SAGE layer with dynamic weights."""
        lin_l = weights["lin_l"]  # [num_nodes, in_dim, out_dim]
        lin_r = weights["lin_r"]  # [num_nodes, in_dim, out_dim] 
        bias = weights["bias"]    # [num_nodes, out_dim]
        
        # Self transformation
        out_self = torch.bmm(x.unsqueeze(1), lin_l).squeeze(1)
        
        # Neighbor aggregation
        row, col = edge_index
        neighbor_messages = x[row]  # Messages from neighbors
        
        # Mean aggregation
        out_neigh = torch.zeros_like(x)
        ones = torch.ones(neighbor_messages.size(0), device=x.device)
        neighbor_count = torch.zeros(x.size(0), device=x.device)
        
        out_neigh.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), neighbor_messages)
        neighbor_count.scatter_add_(0, col, ones)
        neighbor_count = neighbor_count.clamp(min=1).unsqueeze(1)
        out_neigh = out_neigh / neighbor_count
        
        # Transform neighbor features
        out_neigh = torch.bmm(out_neigh.unsqueeze(1), lin_r).squeeze(1)
        
        # Concatenate and add bias
        out = out_self + out_neigh + bias
        
        return out


class StaticGNN(nn.Module):
    """Traditional static GNN for baseline comparisons."""
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        gnn_type: str = "GAT",
        dropout: float = 0.1
    ):
        """Initialize static GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
            gnn_type: GNN type (GCN, GAT, SAGE)
            dropout: Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.upper()
        self.dropout = dropout
        
        # Build layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = hidden_dim
                
            if i == num_layers - 1:
                out_dim = output_dim
            else:
                out_dim = hidden_dim
            
            if self.gnn_type == "GCN":
                from torch_geometric.nn import GCNConv
                layer = GCNConv(in_dim, out_dim)
            elif self.gnn_type == "GAT":
                from torch_geometric.nn import GATConv
                layer = GATConv(in_dim, out_dim, dropout=dropout)
            elif self.gnn_type == "SAGE":
                from torch_geometric.nn import SAGEConv
                layer = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
            
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        current_x = x
        
        for i, layer in enumerate(self.layers):
            current_x = layer(current_x, edge_index)
            
            # Apply activation and dropout (except for last layer)
            if i < len(self.layers) - 1:
                current_x = F.relu(current_x)
                current_x = F.dropout(current_x, p=self.dropout, training=self.training)
        
        return current_x


class AdaptiveGNN(nn.Module):
    """Adaptive GNN that can switch between static and dynamic modes."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        gnn_type: str = "GAT",
        dropout: float = 0.1,
        adaptive: bool = True
    ):
        """Initialize adaptive GNN.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension  
            output_dim: Output dimension
            num_layers: Number of layers
            gnn_type: GNN type (GCN, GAT, SAGE)
            dropout: Dropout probability
            adaptive: Whether to use adaptive/dynamic weights
        """
        super().__init__()
        self.adaptive = adaptive
        
        if adaptive:
            self.gnn = DynamicGNN(gnn_type=gnn_type, dropout=dropout)
        else:
            self.gnn = StaticGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                gnn_type=gnn_type,
                dropout=dropout
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        generated_weights: List[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            generated_weights: Generated weights (required if adaptive=True)
            
        Returns:
            Node embeddings
        """
        if self.adaptive:
            if generated_weights is None:
                raise ValueError("Generated weights required for adaptive mode")
            return self.gnn(x, edge_index, generated_weights)
        else:
            return self.gnn(x, edge_index)