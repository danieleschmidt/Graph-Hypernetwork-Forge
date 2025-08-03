"""HyperGNN: Hypernetwork-based Graph Neural Network

Core implementation of the hypernetwork that generates GNN weights dynamically
from textual node descriptions for zero-shot transfer learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoTokenizer
import logging

from .encoders import TextEncoder, get_text_encoder
from .hypernetworks import WeightGenerator
from .gnns import DynamicGNN, get_gnn_backbone

logger = logging.getLogger(__name__)


class HyperGNN(nn.Module):
    """Hypernetwork-based Graph Neural Network for zero-shot knowledge graph reasoning.
    
    This model generates GNN weights dynamically from textual node descriptions,
    enabling zero-shot transfer to unseen knowledge graphs without retraining.
    
    Args:
        text_encoder: Text encoder model name or instance
        gnn_backbone: GNN architecture ('GCN', 'GAT', 'GraphSAGE')
        hidden_dim: Hidden dimension size
        num_layers: Number of GNN layers
        num_heads: Number of attention heads (for GAT)
        dropout: Dropout probability
        text_dim: Text embedding dimension (auto-detected if None)
        node_dim: Node feature dimension
        edge_dim: Edge feature dimension
        activation: Activation function
    """
    
    def __init__(
        self,
        text_encoder: Union[str, TextEncoder] = "sentence-transformers/all-MiniLM-L6-v2",
        gnn_backbone: str = "GAT",
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        text_dim: Optional[int] = None,
        node_dim: int = 256,
        edge_dim: Optional[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_backbone = gnn_backbone
        
        # Initialize text encoder
        if isinstance(text_encoder, str):
            self.text_encoder = get_text_encoder(text_encoder)
        else:
            self.text_encoder = text_encoder
            
        # Auto-detect text dimension
        if text_dim is None:
            text_dim = self.text_encoder.get_output_dim()
        self.text_dim = text_dim
        
        # Initialize hypernetwork for weight generation
        self.weight_generator = WeightGenerator(
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            gnn_type=gnn_backbone,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Initialize dynamic GNN
        self.dynamic_gnn = DynamicGNN(
            backbone=gnn_backbone,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim,
            activation=activation,
        )
        
        # Node feature projection
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
    def generate_weights(
        self, 
        node_texts: List[str],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Generate GNN weights from node textual descriptions.
        
        Args:
            node_texts: List of textual descriptions for each node
            device: Target device for tensors
            
        Returns:
            Dictionary containing generated GNN weights
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Encode text descriptions
        text_embeddings = self.text_encoder.encode(node_texts)
        if isinstance(text_embeddings, list):
            text_embeddings = torch.stack(text_embeddings)
        text_embeddings = text_embeddings.to(device)
        
        # Generate weights using hypernetwork
        weights = self.weight_generator(text_embeddings)
        
        return weights
    
    def forward(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        node_texts: Optional[List[str]] = None,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the HyperGNN.
        
        Args:
            edge_index: Graph edge indices [2, num_edges]
            node_features: Node feature matrix [num_nodes, node_dim]
            node_texts: Optional node text descriptions (if weights not provided)
            weights: Pre-generated GNN weights (if node_texts not provided)
            edge_attr: Optional edge attributes
            return_embeddings: Whether to return node embeddings
            
        Returns:
            Predictions or (predictions, embeddings) if return_embeddings=True
        """
        device = edge_index.device
        
        # Generate weights if not provided
        if weights is None:
            if node_texts is None:
                raise ValueError("Either weights or node_texts must be provided")
            weights = self.generate_weights(node_texts, device)
        
        # Project node features
        x = self.node_projection(node_features)
        
        # Apply dynamic GNN with generated weights
        node_embeddings = self.dynamic_gnn(
            x, edge_index, weights, edge_attr
        )
        
        # Generate predictions
        predictions = self.predictor(node_embeddings)
        
        if return_embeddings:
            return predictions, node_embeddings
        return predictions
    
    def zero_shot_inference(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        node_texts: List[str],
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Perform zero-shot inference on a new knowledge graph.
        
        Args:
            edge_index: Graph edge indices
            node_features: Node feature matrix
            node_texts: Textual descriptions for each node
            edge_attr: Optional edge attributes
            
        Returns:
            Predictions for the new graph
        """
        self.eval()
        with torch.no_grad():
            return self.forward(
                edge_index=edge_index,
                node_features=node_features,
                node_texts=node_texts,
                edge_attr=edge_attr,
            )
    
    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Get text embeddings for given texts."""
        return self.text_encoder.encode(texts)
    
    def save_pretrained(self, path: str) -> None:
        """Save model state and configuration."""
        config = {
            'text_encoder_name': self.text_encoder.model_name,
            'gnn_backbone': self.gnn_backbone,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'text_dim': self.text_dim,
        }
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': config,
        }
        
        torch.save(save_dict, f"{path}/model.pt")
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_pretrained(cls, path: str, device: Optional[torch.device] = None) -> 'HyperGNN':
        """Load pretrained model from path."""
        save_dict = torch.load(f"{path}/model.pt", map_location=device)
        config = save_dict['config']
        
        model = cls(**config)
        model.load_state_dict(save_dict['model_state_dict'])
        
        if device is not None:
            model = model.to(device)
            
        logger.info(f"Model loaded from {path}")
        return model