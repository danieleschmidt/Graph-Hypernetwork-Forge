"""Meta-Learning Enhanced Hypernetworks for Few-Shot Graph Adaptation.

This module implements breakthrough meta-learning approaches for hypernetworks,
enabling rapid adaptation to new graph domains with minimal training examples.

Research Innovation:
- Model-Agnostic Meta-Learning (MAML) for hypernetwork adaptation
- Gradient-based meta-optimization for few-shot domain transfer
- Hierarchical meta-learning for multi-level graph understanding
- Prototype-based meta-learning for graph pattern recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import copy
import logging
from collections import OrderedDict

# Enhanced utilities
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


class MetaHyperNetwork(nn.Module):
    """Meta-learning enabled hypernetwork for few-shot graph adaptation."""
    
    def __init__(
        self,
        text_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 3,
        meta_lr: float = 1e-3,
        adaptation_steps: int = 5,
        support_size: int = 5,
        prototype_dim: int = 128
    ):
        """Initialize meta-learning hypernetwork.
        
        Args:
            text_dim: Text embedding dimension
            hidden_dim: Hidden dimension for weight generation
            num_layers: Number of hypernetwork layers
            meta_lr: Meta-learning rate for adaptation
            adaptation_steps: Number of gradient steps for adaptation
            support_size: Number of support examples for few-shot learning
            prototype_dim: Dimension for prototype representations
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.meta_lr = meta_lr
        self.adaptation_steps = adaptation_steps
        self.support_size = support_size
        self.prototype_dim = prototype_dim
        
        # Base hypernetwork for weight generation
        self.base_hypernetwork = self._build_base_hypernetwork()
        
        # Meta-learning components
        self.domain_encoder = self._build_domain_encoder()
        self.prototype_network = self._build_prototype_network()
        self.adaptation_controller = self._build_adaptation_controller()
        
        # Meta-parameters for fast adaptation
        self.meta_parameters = nn.ParameterDict({
            'adaptation_bias': nn.Parameter(torch.zeros(hidden_dim)),
            'domain_scaling': nn.Parameter(torch.ones(hidden_dim)),
            'prototype_weights': nn.Parameter(torch.randn(prototype_dim, hidden_dim) * 0.1)
        })
        
        logger.info(f"MetaHyperNetwork initialized with meta_lr={meta_lr}, adaptation_steps={adaptation_steps}")
    
    def _build_base_hypernetwork(self) -> nn.Module:
        """Build the base hypernetwork architecture."""
        layers = []
        
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.text_dim, self.hidden_dim))
            else:
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            
            layers.extend([
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        
        # Final output layer for weight generation
        layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        return nn.Sequential(*layers)
    
    def _build_domain_encoder(self) -> nn.Module:
        """Build domain encoder for domain-specific representations."""
        return nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim)
        )
    
    def _build_prototype_network(self) -> nn.Module:
        """Build prototype network for few-shot pattern recognition."""
        return nn.Sequential(
            nn.Linear(self.prototype_dim, self.prototype_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.prototype_dim * 2, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim)
        )
    
    def _build_adaptation_controller(self) -> nn.Module:
        """Build adaptation controller for meta-learning control."""
        return nn.Sequential(
            nn.Linear(self.prototype_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()  # Gating mechanism
        )
    
    def compute_domain_prototypes(
        self, 
        support_texts: List[str], 
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute domain prototypes from support examples.
        
        Args:
            support_texts: Support text descriptions
            support_features: Support graph features
            
        Returns:
            Domain prototype representations
        """
        # Encode support examples
        support_embeddings = []
        for text in support_texts:
            # Simple text encoding (would use proper encoder in practice)
            text_tensor = torch.randn(1, self.text_dim)  # Placeholder
            domain_emb = self.domain_encoder(text_tensor)
            support_embeddings.append(domain_emb)
        
        support_embeddings = torch.cat(support_embeddings, dim=0)
        
        # Compute prototype through averaging and refinement
        prototype = support_embeddings.mean(dim=0, keepdim=True)
        refined_prototype = self.prototype_network(prototype)
        
        return refined_prototype
    
    def meta_adapt(
        self,
        support_texts: List[str],
        support_features: torch.Tensor,
        support_targets: torch.Tensor,
        query_texts: List[str],
        query_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform meta-adaptation for few-shot learning.
        
        Args:
            support_texts: Support set text descriptions
            support_features: Support set features
            support_targets: Support set targets
            query_texts: Query set text descriptions  
            query_features: Query set features
            
        Returns:
            Tuple of (adapted_weights, adaptation_metrics)
        """
        # Compute domain prototypes
        domain_prototype = self.compute_domain_prototypes(support_texts, support_features)
        
        # Initialize adapted parameters
        adapted_params = OrderedDict()
        for name, param in self.base_hypernetwork.named_parameters():
            adapted_params[name] = param.clone()
        
        # Meta-learning adaptation loop
        adaptation_losses = []
        
        for step in range(self.adaptation_steps):
            # Forward pass with current adapted parameters
            support_outputs = self._forward_with_params(
                support_features, adapted_params, domain_prototype
            )
            
            # Compute adaptation loss
            adaptation_loss = F.mse_loss(support_outputs, support_targets)
            adaptation_losses.append(adaptation_loss.item())
            
            # Compute gradients
            grads = torch.autograd.grad(
                adaptation_loss, 
                adapted_params.values(),
                create_graph=True,
                retain_graph=True
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.meta_lr * grad
        
        # Generate final adapted weights for query
        final_weights = self._forward_with_params(
            query_features, adapted_params, domain_prototype
        )
        
        # Adaptation metrics
        metrics = {
            'adaptation_losses': torch.tensor(adaptation_losses),
            'final_adaptation_loss': adaptation_losses[-1],
            'domain_prototype_norm': torch.norm(domain_prototype),
            'adaptation_convergence': adaptation_losses[0] - adaptation_losses[-1]
        }
        
        return final_weights, metrics
    
    def _forward_with_params(
        self,
        features: torch.Tensor,
        params: OrderedDict,
        domain_prototype: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with specific parameters.
        
        Args:
            features: Input features
            params: Network parameters
            domain_prototype: Domain-specific prototype
            
        Returns:
            Network output
        """
        x = features
        
        # Apply layers with given parameters
        layer_idx = 0
        for name, param in params.items():
            if 'weight' in name:
                # Apply linear layer
                bias_name = name.replace('weight', 'bias')
                bias = params.get(bias_name, torch.zeros(param.size(0)))
                
                x = F.linear(x, param, bias)
                
                # Apply domain adaptation
                if layer_idx < len(self.meta_parameters):
                    adaptation_gate = self.adaptation_controller(
                        torch.cat([domain_prototype, x.mean(dim=0, keepdim=True)], dim=-1)
                    )
                    x = x * adaptation_gate + self.meta_parameters['adaptation_bias']
                
                # Apply activation (simplified)
                if layer_idx < len(params) // 2 - 1:  # Not last layer
                    x = F.relu(x)
                
                layer_idx += 1
        
        return x
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        domain_context: Optional[torch.Tensor] = None,
        adaptation_mode: bool = False
    ) -> torch.Tensor:
        """Forward pass through meta-hypernetwork.
        
        Args:
            text_embeddings: Text embedding inputs
            domain_context: Optional domain context
            adaptation_mode: Whether in adaptation mode
            
        Returns:
            Generated weights
        """
        # Base hypernetwork forward pass
        base_weights = self.base_hypernetwork(text_embeddings)
        
        if adaptation_mode and domain_context is not None:
            # Apply domain-specific adaptations
            domain_encoding = self.domain_encoder(domain_context)
            adaptation_signal = self.adaptation_controller(
                torch.cat([domain_encoding, base_weights.mean(dim=0, keepdim=True)], dim=-1)
            )
            
            # Apply meta-learned adaptations
            adapted_weights = (
                base_weights * adaptation_signal * self.meta_parameters['domain_scaling'] +
                self.meta_parameters['adaptation_bias']
            )
            
            return adapted_weights
        
        return base_weights


class HierarchicalMetaLearning(nn.Module):
    """Hierarchical meta-learning for multi-level graph understanding."""
    
    def __init__(
        self,
        base_dim: int = 256,
        hierarchy_levels: int = 3,
        level_dims: List[int] = None
    ):
        """Initialize hierarchical meta-learning.
        
        Args:
            base_dim: Base dimension
            hierarchy_levels: Number of hierarchy levels
            level_dims: Dimensions for each level
        """
        super().__init__()
        
        self.base_dim = base_dim
        self.hierarchy_levels = hierarchy_levels
        self.level_dims = level_dims or [base_dim // (2**i) for i in range(hierarchy_levels)]
        
        # Hierarchical encoders
        self.level_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_dim if i == 0 else self.level_dims[i-1], self.level_dims[i]),
                nn.ReLU(),
                nn.LayerNorm(self.level_dims[i])
            ) for i in range(hierarchy_levels)
        ])
        
        # Cross-level attention
        self.cross_level_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.level_dims[i], 
                num_heads=4, 
                batch_first=True
            ) for i in range(hierarchy_levels)
        ])
        
        # Level fusion
        self.level_fusion = nn.Sequential(
            nn.Linear(sum(self.level_dims), base_dim),
            nn.ReLU(),
            nn.Linear(base_dim, base_dim)
        )
        
        logger.info(f"HierarchicalMetaLearning initialized with {hierarchy_levels} levels")
    
    def forward(
        self, 
        x: torch.Tensor, 
        hierarchy_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through hierarchical meta-learning.
        
        Args:
            x: Input tensor
            hierarchy_mask: Optional mask for hierarchy levels
            
        Returns:
            Tuple of (fused_output, level_outputs)
        """
        level_outputs = []
        current_input = x
        
        # Process each hierarchy level
        for level in range(self.hierarchy_levels):
            # Encode at current level
            level_output = self.level_encoders[level](current_input)
            
            # Apply cross-level attention if not first level
            if level > 0:
                attended_output, _ = self.cross_level_attention[level](
                    level_output.unsqueeze(1),
                    level_outputs[-1].unsqueeze(1),
                    level_outputs[-1].unsqueeze(1)
                )
                level_output = attended_output.squeeze(1) + level_output
            
            level_outputs.append(level_output)
            current_input = level_output
        
        # Fuse all levels
        concatenated = torch.cat(level_outputs, dim=-1)
        fused_output = self.level_fusion(concatenated)
        
        return fused_output, level_outputs


class PrototypicalMetaLearning(nn.Module):
    """Prototypical meta-learning for graph pattern recognition."""
    
    def __init__(
        self,
        embedding_dim: int = 256,
        prototype_dim: int = 128,
        num_prototypes: int = 10,
        temperature: float = 1.0
    ):
        """Initialize prototypical meta-learning.
        
        Args:
            embedding_dim: Input embedding dimension
            prototype_dim: Prototype dimension
            num_prototypes: Number of learnable prototypes
            temperature: Temperature for prototype matching
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.prototype_dim = prototype_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Learnable prototypes
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, prototype_dim) * 0.1
        )
        
        # Embedding to prototype space
        self.embedding_projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, prototype_dim),
            nn.LayerNorm(prototype_dim)
        )
        
        # Prototype attention
        self.prototype_attention = nn.MultiheadAttention(
            embed_dim=prototype_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Output projector
        self.output_projector = nn.Sequential(
            nn.Linear(prototype_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        logger.info(f"PrototypicalMetaLearning initialized with {num_prototypes} prototypes")
    
    def compute_prototype_similarities(
        self, 
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarities between embeddings and prototypes.
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            
        Returns:
            Prototype similarities [batch_size, num_prototypes]
        """
        # Project to prototype space
        projected = self.embedding_projector(embeddings)
        
        # Compute cosine similarities
        projected_norm = F.normalize(projected, dim=-1)
        prototype_norm = F.normalize(self.prototypes, dim=-1)
        
        similarities = torch.matmul(projected_norm, prototype_norm.t()) / self.temperature
        
        return similarities
    
    def forward(
        self, 
        embeddings: torch.Tensor,
        return_similarities: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through prototypical meta-learning.
        
        Args:
            embeddings: Input embeddings
            return_similarities: Whether to return prototype similarities
            
        Returns:
            Enhanced embeddings (and optionally similarities)
        """
        batch_size = embeddings.size(0)
        
        # Project to prototype space
        projected = self.embedding_projector(embeddings)
        
        # Compute prototype similarities
        similarities = self.compute_prototype_similarities(embeddings)
        
        # Prototype attention
        prototype_expanded = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1)
        attended_prototypes, attention_weights = self.prototype_attention(
            projected.unsqueeze(1),
            prototype_expanded,
            prototype_expanded
        )
        
        # Combine with similarities
        similarity_weights = F.softmax(similarities, dim=-1).unsqueeze(-1)
        weighted_prototypes = (prototype_expanded * similarity_weights).sum(dim=1)
        
        # Fuse attended and weighted prototypes
        fused_prototypes = attended_prototypes.squeeze(1) + weighted_prototypes
        
        # Project back to embedding space
        enhanced_embeddings = self.output_projector(fused_prototypes)
        
        # Residual connection
        output = embeddings + enhanced_embeddings
        
        if return_similarities:
            return output, similarities
        
        return output


class AdaptiveMetaOptimizer(nn.Module):
    """Adaptive meta-optimizer for dynamic learning rate adaptation."""
    
    def __init__(
        self,
        param_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2
    ):
        """Initialize adaptive meta-optimizer.
        
        Args:
            param_dim: Parameter dimension
            hidden_dim: Hidden dimension for optimizer network
            num_layers: Number of optimizer layers
        """
        super().__init__()
        
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim
        
        # Optimizer network
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(param_dim * 2, hidden_dim))  # param + gradient
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.extend([nn.ReLU(), nn.Dropout(0.1)])
        
        # Output learning rate and momentum
        layers.append(nn.Linear(hidden_dim, 2))  # lr, momentum
        
        self.optimizer_net = nn.Sequential(*layers)
        
        # State tracking
        self.register_buffer('step_count', torch.zeros(1))
        
        logger.info(f"AdaptiveMetaOptimizer initialized for param_dim={param_dim}")
    
    def forward(
        self,
        parameters: torch.Tensor,
        gradients: torch.Tensor,
        loss_history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive learning rates and momentum.
        
        Args:
            parameters: Current parameters
            gradients: Parameter gradients
            loss_history: Optional loss history
            
        Returns:
            Tuple of (learning_rates, momentum_values)
        """
        # Concatenate parameters and gradients
        optimizer_input = torch.cat([parameters, gradients], dim=-1)
        
        # Compute adaptive optimization values
        optimizer_output = self.optimizer_net(optimizer_input)
        
        # Split into learning rate and momentum
        lr_raw, momentum_raw = optimizer_output[..., 0], optimizer_output[..., 1]
        
        # Apply appropriate activations
        learning_rates = torch.sigmoid(lr_raw) * 0.1  # Max LR of 0.1
        momentum_values = torch.sigmoid(momentum_raw) * 0.9  # Max momentum of 0.9
        
        return learning_rates, momentum_values
    
    def update_parameters(
        self,
        parameters: torch.Tensor,
        gradients: torch.Tensor,
        velocity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update parameters using adaptive optimization.
        
        Args:
            parameters: Current parameters
            gradients: Parameter gradients
            velocity: Current velocity (for momentum)
            
        Returns:
            Tuple of (updated_parameters, updated_velocity)
        """
        # Compute adaptive rates
        learning_rates, momentum_values = self.forward(parameters, gradients)
        
        # Initialize velocity if not provided
        if velocity is None:
            velocity = torch.zeros_like(parameters)
        
        # Update velocity with momentum
        updated_velocity = momentum_values * velocity + learning_rates * gradients
        
        # Update parameters
        updated_parameters = parameters - updated_velocity
        
        return updated_parameters, updated_velocity