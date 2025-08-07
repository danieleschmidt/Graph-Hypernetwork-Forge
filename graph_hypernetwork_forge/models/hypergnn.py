"""Core HyperGNN model implementation."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, MessagePassing
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Encodes text descriptions into embeddings."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        freeze_encoder: bool = False,
    ):
        """Initialize text encoder.
        
        Args:
            model_name: Pre-trained model name or path
            embedding_dim: Output embedding dimension
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.freeze_encoder = freeze_encoder
        
        # Initialize encoder based on model type
        if "sentence-transformers" in model_name:
            self.encoder = SentenceTransformer(model_name)
            self.is_sentence_transformer = True
            # Get actual embedding dimension
            self.input_dim = self.encoder.get_sentence_embedding_dimension()
        else:
            # Use Hugging Face transformers
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            self.is_sentence_transformer = False
            self.input_dim = self.encoder.config.hidden_size
        
        # Projection layer if dimensions don't match
        if self.input_dim != embedding_dim:
            self.projection = nn.Linear(self.input_dim, embedding_dim)
        else:
            self.projection = nn.Identity()
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        if self.is_sentence_transformer:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts into embeddings.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Text embeddings [batch_size, embedding_dim]
        """
        if self.is_sentence_transformer:
            # Use sentence-transformers
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                embeddings = self.encoder.encode(
                    texts, convert_to_tensor=True, show_progress_bar=False
                )
        else:
            # Use transformers
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            )
            inputs = {k: v.to(next(self.encoder.parameters()).device) for k, v in inputs.items()}
            
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                outputs = self.encoder(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Apply projection
        embeddings = self.projection(embeddings)
        return embeddings


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
                    nn.Linear(text_dim, text_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(text_dim * 2, text_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(text_dim, weight_size),
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
            if layer_idx == 0:
                for key in updated_dims:
                    if "weight" in key or "lin_l" in key or "lin_r" in key:
                        shape = list(updated_dims[key])
                        shape[0] = input_dim
                        updated_dims[key] = tuple(shape)
            if layer_idx == self.num_layers - 1:
                for key in updated_dims:
                    if "weight" in key or "lin_l" in key or "lin_r" in key:
                        shape = list(updated_dims[key])
                        shape[1] = output_dim
                        updated_dims[key] = tuple(shape)
                    elif "bias" in key:
                        updated_dims[key] = (output_dim,)
                    elif "att_weight" in key:
                        shape = list(updated_dims[key])
                        shape[0] = 2 * output_dim
                        updated_dims[key] = tuple(shape)
            
            for weight_name, weight_shape in updated_dims.items():
                # Generate weights for each node
                flat_weights = layer_generators[weight_name](text_embeddings)
                
                # Calculate expected tensor size
                expected_size = batch_size * torch.prod(torch.tensor(weight_shape)).item()
                actual_size = flat_weights.numel()
                
                # Debug information for troubleshooting
                if actual_size != expected_size:
                    # Try to fix by reshaping the generator output
                    target_size = torch.prod(torch.tensor(weight_shape)).item()
                    if flat_weights.size(-1) != target_size:
                        # Add a linear layer to match dimensions if needed
                        if not hasattr(self, f'_dim_fix_{weight_name}_{layer_idx}'):
                            setattr(self, f'_dim_fix_{weight_name}_{layer_idx}', 
                                   nn.Linear(flat_weights.size(-1), target_size))
                        dim_fixer = getattr(self, f'_dim_fix_{weight_name}_{layer_idx}')
                        flat_weights = dim_fixer(flat_weights)
                
                # Reshape to proper weight shape and add batch dimension
                try:
                    weight_tensor = flat_weights.view(batch_size, *weight_shape)
                except RuntimeError as e:
                    # Fallback: reshape to correct total size and then to target shape
                    target_elements = torch.prod(torch.tensor(weight_shape)).item()
                    flat_weights = flat_weights[:, :target_elements]  # Truncate if too large
                    weight_tensor = flat_weights.view(batch_size, *weight_shape)
                
                # Apply scaling
                scale = layer_scales[weight_name]
                weight_tensor = weight_tensor * scale
                
                layer_weights[weight_name] = weight_tensor
            
            generated_weights.append(layer_weights)
        
        return generated_weights


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
        # Use mean attention weight across nodes for simplicity
        mean_att_weight = att_weight.mean(dim=0)  # [2*out_dim, 1]
        att_scores = torch.mm(att_input, mean_att_weight).squeeze()  # [num_edges]
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


class HyperGNN(nn.Module):
    """Main HyperGNN model that combines text encoder, hypernetwork, and dynamic GNN."""
    
    def __init__(
        self,
        text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone: str = "GAT",
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        freeze_text_encoder: bool = False,
    ):
        """Initialize HyperGNN model.
        
        Args:
            text_encoder: Text encoder model name
            gnn_backbone: GNN backbone type (GCN, GAT, SAGE)
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout probability
            freeze_text_encoder: Whether to freeze text encoder
        """
        super().__init__()
        self.text_encoder_name = text_encoder
        self.gnn_backbone = gnn_backbone
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize components
        self.text_encoder = TextEncoder(
            model_name=text_encoder,
            embedding_dim=hidden_dim,
            freeze_encoder=freeze_text_encoder,
        )
        
        self.hypernetwork = HyperNetwork(
            text_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_backbone,
            dropout=dropout,
        )
        
        self.dynamic_gnn = DynamicGNN(
            gnn_type=gnn_backbone,
            dropout=dropout,
        )
    
    def generate_weights(self, node_texts: List[str]) -> List[Dict[str, torch.Tensor]]:
        """Generate GNN weights from node texts.
        
        Args:
            node_texts: List of node text descriptions
            
        Returns:
            Generated weights for each GNN layer
        """
        # Encode texts
        text_embeddings = self.text_encoder(node_texts)
        
        # Generate weights (assume standard dimensions for now)
        input_dim = 128  # Default input dimension
        output_dim = 64  # Default output dimension
        
        weights = self.hypernetwork(text_embeddings, input_dim, output_dim)
        return weights
    
    def forward(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        node_texts: List[str],
    ) -> torch.Tensor:
        """Forward pass of HyperGNN.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            node_features: Node features [num_nodes, feature_dim]
            node_texts: List of node text descriptions
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Encode texts to get embeddings
        text_embeddings = self.text_encoder(node_texts)
        
        # Generate GNN weights
        input_dim = node_features.size(1)
        output_dim = self.hidden_dim  # Can be made configurable
        
        generated_weights = self.hypernetwork(
            text_embeddings, input_dim, output_dim
        )
        
        # Apply dynamic GNN
        node_embeddings = self.dynamic_gnn(
            node_features, edge_index, generated_weights
        )
        
        return node_embeddings
    
    def predict(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        node_texts: List[str],
    ) -> torch.Tensor:
        """Prediction interface for inference.
        
        Args:
            edge_index: Edge connectivity [2, num_edges]
            node_features: Node features [num_nodes, feature_dim]
            node_texts: List of node text descriptions
            
        Returns:
            Predictions [num_nodes, output_dim]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(edge_index, node_features, node_texts)
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "text_encoder": self.text_encoder_name,
            "gnn_backbone": self.gnn_backbone,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }
    
    @classmethod
    def from_config(cls, config: Dict) -> "HyperGNN":
        """Create model from configuration."""
        return cls(**config)