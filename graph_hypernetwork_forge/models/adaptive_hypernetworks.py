"""Advanced Adaptive Hypernetworks for Dynamic GNN Weight Generation.

This module implements breakthrough research in dimension-adaptive hypernetworks
that can dynamically generate GNN weights for varying graph sizes and schemas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math
import logging

# Enhanced logging and validation
try:
    from ..utils.logging_utils import get_logger
    from ..utils.exceptions import ValidationError, ModelError
    from ..utils.memory_utils import memory_management
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)
    class ValidationError(Exception): pass
    class ModelError(Exception): pass
    def memory_management(*args, **kwargs):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class DimensionAdapter(nn.Module):
    """Adaptive module that handles varying input/output dimensions."""
    
    def __init__(self, base_dim: int = 256, adaptation_layers: int = 3):
        """Initialize dimension adapter.
        
        Args:
            base_dim: Base dimension for internal processing
            adaptation_layers: Number of adaptation layers
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.adaptation_layers = adaptation_layers
        
        # Adaptive projection layers
        self.input_projectors = nn.ModuleDict()
        self.output_projectors = nn.ModuleDict()
        
        # Internal processing layers
        self.processing_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, base_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(base_dim * 2, base_dim),
                nn.LayerNorm(base_dim)
            ) for _ in range(adaptation_layers)
        ])
        
        logger.info(f"DimensionAdapter initialized with base_dim={base_dim}")
    
    def get_or_create_projector(self, projector_dict: nn.ModuleDict, 
                               from_dim: int, to_dim: int, 
                               name: str) -> nn.Module:
        """Get or create a projector for given dimensions.
        
        Args:
            projector_dict: Dictionary to store projectors
            from_dim: Input dimension
            to_dim: Output dimension
            name: Projector name
            
        Returns:
            Projector module
        """
        key = f"{from_dim}_{to_dim}"
        
        if key not in projector_dict:
            # Create new projector with dimension-aware initialization
            projector = nn.Sequential(
                nn.Linear(from_dim, max(from_dim, to_dim)),
                nn.ReLU(),
                nn.Linear(max(from_dim, to_dim), to_dim)
            )
            
            # Xavier initialization for stable training
            for layer in projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
            
            projector_dict[key] = projector
            logger.debug(f"Created {name} projector: {from_dim} -> {to_dim}")
        
        return projector_dict[key]
    
    def forward(self, x: torch.Tensor, 
                target_dim: Optional[int] = None) -> torch.Tensor:
        """Forward pass with dimension adaptation.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            target_dim: Target output dimension
            
        Returns:
            Adapted tensor [batch_size, target_dim or base_dim]
        """
        batch_size, input_dim = x.shape
        
        # Project to base dimension if needed
        if input_dim != self.base_dim:
            input_proj = self.get_or_create_projector(
                self.input_projectors, input_dim, self.base_dim, "input"
            )
            x = input_proj(x)
        
        # Process through adaptation layers
        for layer in self.processing_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
        
        # Project to target dimension if specified
        if target_dim is not None and target_dim != self.base_dim:
            output_proj = self.get_or_create_projector(
                self.output_projectors, self.base_dim, target_dim, "output"
            )
            x = output_proj(x)
        
        return x


class AttentionBasedWeightGenerator(nn.Module):
    """Attention-based weight generator for sophisticated parameter synthesis."""
    
    def __init__(self, text_dim: int, max_weight_size: int = 10000,
                 num_heads: int = 8, attention_layers: int = 4):
        """Initialize attention-based weight generator.
        
        Args:
            text_dim: Text embedding dimension
            max_weight_size: Maximum weight tensor size
            num_heads: Number of attention heads
            attention_layers: Number of attention layers
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.max_weight_size = max_weight_size
        self.num_heads = num_heads
        self.attention_layers = attention_layers
        
        # Multi-head attention for weight synthesis
        self.weight_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=text_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(attention_layers)
        ])
        
        # Layer normalization for each attention layer
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(text_dim) for _ in range(attention_layers)
        ])
        
        # Final projection layers for weight generation
        self.weight_projector = nn.Sequential(
            nn.Linear(text_dim, text_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(text_dim * 2, text_dim),
            nn.LayerNorm(text_dim)
        )
        
        # Dynamic weight size adaptation
        self.size_predictor = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.ReLU(),
            nn.Linear(text_dim // 2, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        logger.info(f"AttentionBasedWeightGenerator initialized with {attention_layers} layers")
    
    def forward(self, text_embeddings: torch.Tensor,
                target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate weights using attention mechanism.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            target_shape: Target weight tensor shape
            
        Returns:
            Generated weights [batch_size, *target_shape]
        """
        batch_size = text_embeddings.shape[0]
        target_size = torch.prod(torch.tensor(target_shape)).item()
        
        # Add sequence dimension for attention (treat each embedding as a sequence of 1)
        x = text_embeddings.unsqueeze(1)  # [batch_size, 1, text_dim]
        
        # Apply multi-head attention layers
        for attention, norm in zip(self.weight_attention, self.attention_norms):
            # Self-attention
            attn_output, _ = attention(x, x, x)
            
            # Residual connection and normalization
            x = norm(x + attn_output)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # [batch_size, text_dim]
        
        # Project to weight features
        weight_features = self.weight_projector(x)
        
        # Predict optimal size scaling factor
        size_factor = self.size_predictor(weight_features)
        
        # Generate base weights
        weight_generator = nn.Linear(self.text_dim, target_size)
        
        # Apply size-aware scaling
        weights = weight_generator(weight_features) * size_factor
        
        # Reshape to target shape
        weights = weights.view(batch_size, *target_shape)
        
        return weights


class HierarchicalWeightDecomposition(nn.Module):
    """Hierarchical weight decomposition for efficient parameter generation."""
    
    def __init__(self, text_dim: int, decomposition_rank: int = 64):
        """Initialize hierarchical weight decomposition.
        
        Args:
            text_dim: Text embedding dimension
            decomposition_rank: Rank for low-rank decomposition
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.decomposition_rank = decomposition_rank
        
        # Low-rank decomposition generators
        self.left_factor_generator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, decomposition_rank * text_dim)
        )
        
        self.right_factor_generator = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Linear(text_dim, decomposition_rank * text_dim)
        )
        
        # Bias generator
        self.bias_generator = nn.Sequential(
            nn.Linear(text_dim, text_dim // 2),
            nn.ReLU(),
            nn.Linear(text_dim // 2, text_dim)
        )
        
        logger.info(f"HierarchicalWeightDecomposition initialized with rank={decomposition_rank}")
    
    def forward(self, text_embeddings: torch.Tensor,
                target_shape: Tuple[int, int]) -> torch.Tensor:
        """Generate weights using hierarchical decomposition.
        
        Args:
            text_embeddings: Text embeddings [batch_size, text_dim]
            target_shape: Target weight matrix shape (input_dim, output_dim)
            
        Returns:
            Generated weight matrix [batch_size, input_dim, output_dim]
        """
        batch_size = text_embeddings.shape[0]
        input_dim, output_dim = target_shape
        
        # Generate low-rank factors
        left_factors = self.left_factor_generator(text_embeddings)
        right_factors = self.right_factor_generator(text_embeddings)
        
        # Reshape factors
        left_factors = left_factors.view(batch_size, input_dim, self.decomposition_rank)
        right_factors = right_factors.view(batch_size, self.decomposition_rank, output_dim)
        
        # Compute full weight matrix via matrix multiplication
        weights = torch.bmm(left_factors, right_factors)
        
        # Generate and add bias-like correction
        bias_correction = self.bias_generator(text_embeddings)
        bias_correction = bias_correction.unsqueeze(1).unsqueeze(2)
        weights = weights + bias_correction
        
        return weights


class AdaptiveDimensionHyperGNN(nn.Module):
    """Advanced HyperGNN with adaptive dimension handling and novel architectures."""
    
    def __init__(self, 
                 text_encoder_dim: int = 384,
                 base_hidden_dim: int = 256,
                 num_gnn_layers: int = 3,
                 gnn_type: str = "GAT",
                 use_attention_generator: bool = True,
                 use_hierarchical_decomposition: bool = True,
                 adaptation_layers: int = 3,
                 attention_heads: int = 8):
        """Initialize adaptive dimension HyperGNN.
        
        Args:
            text_encoder_dim: Text encoder output dimension
            base_hidden_dim: Base hidden dimension for processing
            num_gnn_layers: Number of GNN layers
            gnn_type: GNN backbone type
            use_attention_generator: Whether to use attention-based weight generation
            use_hierarchical_decomposition: Whether to use hierarchical decomposition
            adaptation_layers: Number of adaptation layers
            attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.text_encoder_dim = text_encoder_dim
        self.base_hidden_dim = base_hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.gnn_type = gnn_type.upper()
        
        # Dimension adapter for handling varying input/output sizes
        self.dimension_adapter = DimensionAdapter(
            base_dim=base_hidden_dim,
            adaptation_layers=adaptation_layers
        )
        
        # Text embedding processor
        self.text_processor = nn.Sequential(
            nn.Linear(text_encoder_dim, base_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(base_hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Advanced weight generators
        self.weight_generators = nn.ModuleDict()
        
        if use_attention_generator:
            self.weight_generators['attention'] = AttentionBasedWeightGenerator(
                text_dim=base_hidden_dim,
                num_heads=attention_heads,
                attention_layers=3
            )
        
        if use_hierarchical_decomposition:
            self.weight_generators['hierarchical'] = HierarchicalWeightDecomposition(
                text_dim=base_hidden_dim,
                decomposition_rank=min(64, base_hidden_dim // 4)
            )
        
        # Fallback standard generator
        self.weight_generators['standard'] = nn.ModuleDict()
        for layer_idx in range(num_gnn_layers):
            self.weight_generators['standard'][f'layer_{layer_idx}'] = nn.ModuleDict()
        
        # Schema adaptation mechanism
        self.schema_adapter = nn.Sequential(
            nn.Linear(base_hidden_dim, base_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(base_hidden_dim * 2, base_hidden_dim),
            nn.Sigmoid()  # Gate for schema-specific adaptations
        )
        
        # Meta-learning components for fast adaptation
        self.meta_learner = nn.Sequential(
            nn.Linear(base_hidden_dim, base_hidden_dim),
            nn.ReLU(),
            nn.Linear(base_hidden_dim, base_hidden_dim),
            nn.Tanh()
        )
        
        logger.info(f"AdaptiveDimensionHyperGNN initialized with {len(self.weight_generators)} generators")
    
    def _calculate_weight_shapes(self, input_dim: int, output_dim: int,
                                layer_idx: int, num_layers: int) -> Dict[str, Tuple[int, ...]]:
        """Calculate weight shapes for a specific layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            layer_idx: Current layer index
            num_layers: Total number of layers
            
        Returns:
            Dictionary of weight shapes
        """
        # Determine layer dimensions
        if layer_idx == 0:
            in_dim = input_dim
        else:
            in_dim = self.base_hidden_dim
        
        if layer_idx == num_layers - 1:
            out_dim = output_dim
        else:
            out_dim = self.base_hidden_dim
        
        # GNN-specific weight shapes
        shapes = {}
        
        if self.gnn_type == "GCN":
            shapes['weight'] = (in_dim, out_dim)
            shapes['bias'] = (out_dim,)
        
        elif self.gnn_type == "GAT":
            shapes['weight'] = (in_dim, out_dim)
            shapes['attention_weight'] = (2 * out_dim, 1)
            shapes['bias'] = (out_dim,)
        
        elif self.gnn_type == "SAGE":
            shapes['self_weight'] = (in_dim, out_dim)
            shapes['neighbor_weight'] = (in_dim, out_dim)
            shapes['bias'] = (out_dim,)
        
        else:
            raise ValidationError("gnn_type", self.gnn_type, "GCN, GAT, or SAGE")
        
        return shapes
    
    def generate_adaptive_weights(self, text_embeddings: torch.Tensor,
                                 input_dim: int, output_dim: int,
                                 generator_type: str = "attention") -> List[Dict[str, torch.Tensor]]:
        """Generate adaptive weights for all GNN layers.
        
        Args:
            text_embeddings: Processed text embeddings [num_nodes, base_hidden_dim]
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            generator_type: Type of weight generator to use
            
        Returns:
            List of weight dictionaries for each layer
        """
        batch_size = text_embeddings.shape[0]
        generated_weights = []
        
        with memory_management():
            for layer_idx in range(self.num_gnn_layers):
                layer_weights = {}
                
                # Calculate weight shapes for this layer
                weight_shapes = self._calculate_weight_shapes(
                    input_dim, output_dim, layer_idx, self.num_gnn_layers
                )
                
                # Apply schema adaptation
                schema_gates = self.schema_adapter(text_embeddings)
                adapted_embeddings = text_embeddings * schema_gates
                
                # Apply meta-learning adaptation
                meta_features = self.meta_learner(adapted_embeddings)
                enhanced_embeddings = adapted_embeddings + meta_features
                
                # Generate weights using specified generator
                if generator_type == "attention" and "attention" in self.weight_generators:
                    generator = self.weight_generators["attention"]
                    
                    for weight_name, weight_shape in weight_shapes.items():
                        # Generate weights with attention mechanism
                        weights = generator(enhanced_embeddings, weight_shape)
                        layer_weights[weight_name] = weights
                
                elif generator_type == "hierarchical" and "hierarchical" in self.weight_generators:
                    generator = self.weight_generators["hierarchical"]
                    
                    for weight_name, weight_shape in weight_shapes.items():
                        if len(weight_shape) == 2:
                            # Use hierarchical decomposition for 2D weights
                            weights = generator(enhanced_embeddings, weight_shape)
                        else:
                            # Fall back to standard generation for 1D weights
                            weight_size = torch.prod(torch.tensor(weight_shape)).item()
                            linear_gen = nn.Linear(self.base_hidden_dim, weight_size)
                            weights = linear_gen(enhanced_embeddings)
                            weights = weights.view(batch_size, *weight_shape)
                        
                        layer_weights[weight_name] = weights
                
                else:
                    # Use standard weight generation
                    for weight_name, weight_shape in weight_shapes.items():
                        weight_size = torch.prod(torch.tensor(weight_shape)).item()
                        
                        # Create generator if not exists
                        gen_key = f"layer_{layer_idx}_{weight_name}"
                        if gen_key not in self.weight_generators['standard'][f'layer_{layer_idx}']:
                            generator = nn.Sequential(
                                nn.Linear(self.base_hidden_dim, self.base_hidden_dim),
                                nn.ReLU(),
                                nn.Linear(self.base_hidden_dim, weight_size)
                            )
                            self.weight_generators['standard'][f'layer_{layer_idx}'][weight_name] = generator
                        
                        generator = self.weight_generators['standard'][f'layer_{layer_idx}'][weight_name]
                        weights = generator(enhanced_embeddings)
                        weights = weights.view(batch_size, *weight_shape)
                        layer_weights[weight_name] = weights
                
                generated_weights.append(layer_weights)
        
        return generated_weights
    
    def forward(self, text_embeddings: torch.Tensor,
                input_dim: int, output_dim: int,
                generator_type: str = "attention") -> List[Dict[str, torch.Tensor]]:
        """Forward pass of adaptive dimension HyperGNN.
        
        Args:
            text_embeddings: Raw text embeddings [num_nodes, text_encoder_dim]
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            generator_type: Type of weight generator to use
            
        Returns:
            Generated weights for all layers
        """
        # Process text embeddings to base dimension
        processed_embeddings = self.text_processor(text_embeddings)
        
        # Apply dimension adaptation
        adapted_embeddings = self.dimension_adapter(processed_embeddings)
        
        # Generate adaptive weights
        weights = self.generate_adaptive_weights(
            adapted_embeddings, input_dim, output_dim, generator_type
        )
        
        return weights
    
    def get_generator_types(self) -> List[str]:
        """Get available weight generator types.
        
        Returns:
            List of available generator types
        """
        return list(self.weight_generators.keys())
    
    def compute_model_complexity(self) -> Dict[str, int]:
        """Compute model complexity metrics.
        
        Returns:
            Dictionary of complexity metrics
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        generator_params = {}
        for gen_type, generator in self.weight_generators.items():
            if hasattr(generator, 'parameters'):
                generator_params[gen_type] = sum(p.numel() for p in generator.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'generator_parameters': generator_params,
            'adaptation_parameters': sum(p.numel() for p in self.dimension_adapter.parameters()),
        }


class ZeroShotDomainAdapter(nn.Module):
    """Zero-shot domain adaptation module for cross-domain transfer."""
    
    def __init__(self, base_dim: int = 256, num_domains: int = 10):
        """Initialize zero-shot domain adapter.
        
        Args:
            base_dim: Base feature dimension
            num_domains: Maximum number of domains to handle
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.num_domains = num_domains
        
        # Domain embedding space
        self.domain_embeddings = nn.Embedding(num_domains, base_dim)
        
        # Domain-invariant feature extractor
        self.domain_invariant_encoder = nn.Sequential(
            nn.Linear(base_dim, base_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(base_dim * 2, base_dim),
            nn.LayerNorm(base_dim)
        )
        
        # Domain-specific adapters
        self.domain_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim, base_dim),
                nn.ReLU(),
                nn.Linear(base_dim, base_dim),
                nn.Tanh()
            ) for _ in range(num_domains)
        ])
        
        # Cross-domain alignment
        self.alignment_projector = nn.Sequential(
            nn.Linear(base_dim * 2, base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)
        )
        
        logger.info(f"ZeroShotDomainAdapter initialized for {num_domains} domains")
    
    def forward(self, features: torch.Tensor, 
                source_domain_id: int = 0,
                target_domain_id: Optional[int] = None) -> torch.Tensor:
        """Adapt features for zero-shot domain transfer.
        
        Args:
            features: Input features [batch_size, base_dim]
            source_domain_id: Source domain identifier
            target_domain_id: Target domain identifier (None for domain-invariant)
            
        Returns:
            Adapted features [batch_size, base_dim]
        """
        batch_size = features.shape[0]
        
        # Extract domain-invariant features
        invariant_features = self.domain_invariant_encoder(features)
        
        if target_domain_id is None:
            # Return domain-invariant features for zero-shot scenarios
            return invariant_features
        
        # Get domain embeddings
        source_embed = self.domain_embeddings(torch.tensor(source_domain_id, device=features.device))
        target_embed = self.domain_embeddings(torch.tensor(target_domain_id, device=features.device))
        
        # Apply source domain adaptation
        source_adapted = self.domain_adapters[source_domain_id % self.num_domains](invariant_features)
        
        # Apply target domain adaptation
        target_adapted = self.domain_adapters[target_domain_id % self.num_domains](invariant_features)
        
        # Cross-domain alignment
        domain_diff = target_embed - source_embed
        domain_diff = domain_diff.unsqueeze(0).expand(batch_size, -1)
        
        aligned_features = torch.cat([target_adapted, domain_diff], dim=1)
        aligned_features = self.alignment_projector(aligned_features)
        
        return aligned_features