"""Hypernetwork modules for generating GNN weights from text embeddings.

The hypernetwork takes text embeddings as input and generates the weights
for the graph neural network layers, enabling dynamic weight generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class WeightGenerator(nn.Module):
    """Hypernetwork that generates GNN weights from text embeddings.
    
    This module takes text embeddings and generates all necessary weights
    for the specified GNN architecture (GCN, GAT, GraphSAGE).
    """
    
    def __init__(
        self,
        text_dim: int,
        hidden_dim: int,
        num_layers: int,
        gnn_type: str = "GAT",
        num_heads: int = 8,
        dropout: float = 0.1,
        weight_init: str = "xavier",
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.upper()
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Compute total parameter count for target GNN
        self.param_sizes = self._compute_parameter_sizes()
        self.total_params = sum(self.param_sizes.values())
        
        logger.info(f"Generating {self.total_params} parameters for {gnn_type}")
        
        # Hypernetwork architecture
        self.hypernet_dim = max(1024, text_dim * 2)
        
        # Text processing layers
        self.text_processor = nn.Sequential(
            nn.Linear(text_dim, self.hypernet_dim),
            nn.LayerNorm(self.hypernet_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hypernet_dim, self.hypernet_dim),
            nn.LayerNorm(self.hypernet_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Weight generation layers for each parameter type
        self.weight_generators = nn.ModuleDict()
        
        for param_name, param_size in self.param_sizes.items():
            self.weight_generators[param_name] = nn.Sequential(
                nn.Linear(self.hypernet_dim, self.hypernet_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hypernet_dim // 2, param_size),
            )
        
        # Weight normalization and scaling
        self.weight_scales = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1)) 
            for name in self.param_sizes.keys()
        })
        
        self._initialize_weights(weight_init)
        
    def _compute_parameter_sizes(self) -> Dict[str, int]:
        """Compute the number of parameters needed for each weight matrix."""
        sizes = {}
        
        if self.gnn_type == "GCN":
            # GCN parameters: Linear transformations for each layer
            for i in range(self.num_layers):
                if i == 0:
                    sizes[f"gcn_layer_{i}_weight"] = self.hidden_dim * self.hidden_dim
                else:
                    sizes[f"gcn_layer_{i}_weight"] = self.hidden_dim * self.hidden_dim
                sizes[f"gcn_layer_{i}_bias"] = self.hidden_dim
                
        elif self.gnn_type == "GAT":
            # GAT parameters: Attention weights and linear transformations
            for i in range(self.num_layers):
                head_dim = self.hidden_dim // self.num_heads
                
                # Linear transformation weights
                sizes[f"gat_layer_{i}_weight"] = self.hidden_dim * self.hidden_dim
                sizes[f"gat_layer_{i}_bias"] = self.hidden_dim
                
                # Attention weights (per head)
                sizes[f"gat_layer_{i}_att_src"] = self.num_heads * head_dim
                sizes[f"gat_layer_{i}_att_dst"] = self.num_heads * head_dim
                
        elif self.gnn_type == "GRAPHSAGE":
            # GraphSAGE parameters: Self and neighbor transformations
            for i in range(self.num_layers):
                sizes[f"sage_layer_{i}_self_weight"] = self.hidden_dim * self.hidden_dim
                sizes[f"sage_layer_{i}_neighbor_weight"] = self.hidden_dim * self.hidden_dim
                sizes[f"sage_layer_{i}_bias"] = self.hidden_dim
                
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
            
        return sizes
    
    def _initialize_weights(self, init_method: str = "xavier"):
        """Initialize hypernetwork weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif init_method == "normal":
                    nn.init.normal_(module.weight, std=0.02)
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate GNN weights from text embeddings.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            
        Returns:
            Dictionary of generated weight tensors
        """
        batch_size = text_embeddings.size(0)
        
        # Process text embeddings
        processed_text = self.text_processor(text_embeddings)
        
        # Generate weights for each parameter
        generated_weights = {}
        
        for param_name, param_size in self.param_sizes.items():
            # Generate flattened weights
            flat_weights = self.weight_generators[param_name](processed_text)
            
            # Apply scaling
            scale = self.weight_scales[param_name]
            flat_weights = flat_weights * scale
            
            # Reshape to proper dimensions
            if "bias" in param_name:
                weights = flat_weights.view(batch_size, -1)
            else:
                # For weight matrices, reshape appropriately
                if self.gnn_type == "GAT" and "att_" in param_name:
                    # Attention weights
                    weights = flat_weights.view(batch_size, self.num_heads, -1)
                else:
                    # Linear layer weights
                    sqrt_size = int(math.sqrt(param_size))
                    if sqrt_size * sqrt_size == param_size:
                        weights = flat_weights.view(batch_size, sqrt_size, sqrt_size)
                    else:
                        weights = flat_weights.view(batch_size, -1)
            
            generated_weights[param_name] = weights
            
        return generated_weights
    
    def generate_mean_weights(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate mean weights across batch for inference."""
        batch_weights = self.forward(text_embeddings)
        
        mean_weights = {}
        for param_name, weights in batch_weights.items():
            mean_weights[param_name] = weights.mean(dim=0)
            
        return mean_weights


class AdaptiveWeightGenerator(WeightGenerator):
    """Adaptive hypernetwork that adjusts generation based on text complexity."""
    
    def __init__(self, *args, adaptive_layers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.adaptive_layers = adaptive_layers
        
        # Text complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(self.text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
        # Adaptive weight generation paths
        self.adaptive_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hypernet_dim, self.hypernet_dim // 2),
                nn.ReLU(),
                nn.Linear(self.hypernet_dim // 2, self.hypernet_dim // 2),
            )
            for _ in range(adaptive_layers)
        ])
        
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate weights with adaptive complexity."""
        # Estimate text complexity
        complexity = self.complexity_estimator(text_embeddings)
        
        # Process text with adaptive depth
        processed_text = self.text_processor(text_embeddings)
        
        # Apply adaptive layers based on complexity
        for i, adaptive_layer in enumerate(self.adaptive_generators):
            # Use complexity as gating mechanism
            gate = torch.sigmoid(complexity * (i + 1))
            adaptive_output = adaptive_layer(processed_text)
            processed_text = processed_text + gate * adaptive_output
        
        # Generate weights using the adapted representations
        generated_weights = {}
        
        for param_name, param_size in self.param_sizes.items():
            flat_weights = self.weight_generators[param_name](processed_text)
            scale = self.weight_scales[param_name]
            flat_weights = flat_weights * scale
            
            # Reshape weights
            if "bias" in param_name:
                weights = flat_weights.view(text_embeddings.size(0), -1)
            else:
                if self.gnn_type == "GAT" and "att_" in param_name:
                    weights = flat_weights.view(text_embeddings.size(0), self.num_heads, -1)
                else:
                    sqrt_size = int(math.sqrt(param_size))
                    if sqrt_size * sqrt_size == param_size:
                        weights = flat_weights.view(text_embeddings.size(0), sqrt_size, sqrt_size)
                    else:
                        weights = flat_weights.view(text_embeddings.size(0), -1)
            
            generated_weights[param_name] = weights
            
        return generated_weights


class MultiModalWeightGenerator(WeightGenerator):
    """Weight generator that handles multiple modalities beyond text."""
    
    def __init__(
        self, 
        text_dim: int, 
        image_dim: Optional[int] = None,
        numerical_dim: Optional[int] = None,
        fusion_method: str = "concatenation",
        **kwargs
    ):
        self.image_dim = image_dim or 0
        self.numerical_dim = numerical_dim or 0
        self.fusion_method = fusion_method
        
        # Compute total input dimension
        total_dim = text_dim
        if self.image_dim > 0:
            total_dim += self.image_dim
        if self.numerical_dim > 0:
            total_dim += self.numerical_dim
            
        super().__init__(text_dim=total_dim, **kwargs)
        
        # Modality-specific processors
        if self.image_dim > 0:
            self.image_processor = nn.Sequential(
                nn.Linear(self.image_dim, text_dim),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.1)),
            )
            
        if self.numerical_dim > 0:
            self.numerical_processor = nn.Sequential(
                nn.Linear(self.numerical_dim, text_dim // 2),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.1)),
            )
            
        # Fusion layer
        if fusion_method == "cross_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=8,
                dropout=kwargs.get('dropout', 0.1),
                batch_first=True,
            )
    
    def forward(
        self, 
        text_embeddings: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
        numerical_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate weights from multi-modal inputs."""
        modalities = [text_embeddings]
        
        # Process additional modalities
        if image_embeddings is not None and self.image_dim > 0:
            processed_images = self.image_processor(image_embeddings)
            modalities.append(processed_images)
            
        if numerical_features is not None and self.numerical_dim > 0:
            processed_numerical = self.numerical_processor(numerical_features)
            modalities.append(processed_numerical)
        
        # Fuse modalities
        if self.fusion_method == "concatenation":
            fused_embeddings = torch.cat(modalities, dim=-1)
        elif self.fusion_method == "cross_attention":
            # Use text as query, others as key/value
            if len(modalities) > 1:
                keys_values = torch.stack(modalities[1:], dim=1)  # [batch, num_modalities, dim]
                queries = text_embeddings.unsqueeze(1)  # [batch, 1, dim]
                
                attended, _ = self.cross_attention(queries, keys_values, keys_values)
                fused_embeddings = attended.squeeze(1)
            else:
                fused_embeddings = text_embeddings
        else:
            # Simple addition
            fused_embeddings = sum(modalities)
        
        # Generate weights using parent class method
        return super().forward(fused_embeddings)