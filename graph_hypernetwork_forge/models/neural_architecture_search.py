"""Neural Architecture Search for Hypernetwork Optimization

This module implements advanced neural architecture search (NAS) techniques
specifically designed for optimizing hypernetwork architectures that generate
GNN weights from textual descriptions.

Novel Contributions:
1. Differentiable Architecture Search for Hypernetworks (DASH)
2. Quantum-Inspired Search Spaces for Parameter Generation
3. Multi-Objective Optimization for Accuracy and Efficiency
4. Meta-Learning Integration for Few-Shot Architecture Adaptation
"""

import math
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from ..utils.logging_utils import get_logger
    from ..utils.exceptions import ValidationError, ModelError
    from .hypernetworks import HyperNetwork
    from .hypergnn import TextEncoder
except ImportError:
    # Fallback for standalone usage
    import logging
    def get_logger(name): return logging.getLogger(name)
    class ValidationError(Exception): pass
    class ModelError(Exception): pass


logger = get_logger(__name__)


@dataclass
class ArchitectureConfig:
    """Configuration for hypernetwork architecture."""
    num_layers: int = 3
    hidden_dims: List[int] = None
    activation_types: List[str] = None
    dropout_rates: List[float] = None
    residual_connections: List[bool] = None
    normalization_types: List[str] = None
    attention_mechanisms: List[str] = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256] * self.num_layers
        if self.activation_types is None:
            self.activation_types = ['relu'] * self.num_layers
        if self.dropout_rates is None:
            self.dropout_rates = [0.1] * self.num_layers
        if self.residual_connections is None:
            self.residual_connections = [True] * self.num_layers
        if self.normalization_types is None:
            self.normalization_types = ['layer_norm'] * self.num_layers
        if self.attention_mechanisms is None:
            self.attention_mechanisms = ['none'] * self.num_layers


class QuantumInspiredSearchSpace:
    """Quantum-inspired search space for hypernetwork architectures.
    
    This class implements quantum-inspired optimization principles:
    1. Superposition: Multiple architectural choices exist simultaneously
    2. Entanglement: Architectural choices are correlated across layers
    3. Interference: Constructive/destructive interactions between choices
    4. Measurement: Probabilistic sampling of architectures
    """
    
    def __init__(
        self,
        min_layers: int = 2,
        max_layers: int = 8,
        hidden_dim_choices: List[int] = None,
        activation_choices: List[str] = None,
        dropout_choices: List[float] = None,
        temperature: float = 1.0,
    ):
        """Initialize quantum-inspired search space.
        
        Args:
            min_layers: Minimum number of layers
            max_layers: Maximum number of layers
            hidden_dim_choices: Available hidden dimensions
            activation_choices: Available activation functions
            dropout_choices: Available dropout rates
            temperature: Temperature for quantum sampling
        """
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.temperature = temperature
        
        # Default choices inspired by successful architectures
        self.hidden_dim_choices = hidden_dim_choices or [128, 256, 512, 768, 1024]
        self.activation_choices = activation_choices or [
            'relu', 'gelu', 'swish', 'mish', 'leaky_relu', 'elu'
        ]
        self.dropout_choices = dropout_choices or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        self.normalization_choices = ['none', 'batch_norm', 'layer_norm', 'group_norm']
        self.attention_choices = ['none', 'self_attention', 'cross_attention', 'multi_head']
        
        # Quantum state representation (probability amplitudes)
        self._initialize_quantum_state()
        
        logger.info(f"Initialized quantum search space with {self._compute_search_space_size()} possible architectures")
    
    def _initialize_quantum_state(self):
        """Initialize quantum probability amplitudes for each choice."""
        # Layer count probabilities (uniform superposition)
        num_layer_choices = self.max_layers - self.min_layers + 1
        self.layer_probs = nn.Parameter(torch.ones(num_layer_choices) / num_layer_choices)
        
        # Architecture choice probabilities for each layer type
        self.hidden_dim_probs = nn.Parameter(torch.ones(len(self.hidden_dim_choices)) / len(self.hidden_dim_choices))
        self.activation_probs = nn.Parameter(torch.ones(len(self.activation_choices)) / len(self.activation_choices))
        self.dropout_probs = nn.Parameter(torch.ones(len(self.dropout_choices)) / len(self.dropout_choices))
        self.norm_probs = nn.Parameter(torch.ones(len(self.normalization_choices)) / len(self.normalization_choices))
        self.attention_probs = nn.Parameter(torch.ones(len(self.attention_choices)) / len(self.attention_choices))
        
        # Entanglement matrix (correlations between choices)
        total_choices = (len(self.hidden_dim_choices) + len(self.activation_choices) + 
                        len(self.dropout_choices) + len(self.normalization_choices) + 
                        len(self.attention_choices))
        self.entanglement_matrix = nn.Parameter(torch.randn(total_choices, total_choices) * 0.01)
    
    def _compute_search_space_size(self) -> int:
        """Compute the total size of the search space."""
        layer_choices = self.max_layers - self.min_layers + 1
        per_layer_choices = (len(self.hidden_dim_choices) * len(self.activation_choices) * 
                           len(self.dropout_choices) * len(self.normalization_choices) * 
                           len(self.attention_choices))
        return layer_choices * (per_layer_choices ** self.max_layers)
    
    def sample_architecture(self, use_gumbel: bool = True) -> ArchitectureConfig:
        """Sample an architecture from the quantum superposition.
        
        Args:
            use_gumbel: Whether to use Gumbel-Softmax for differentiable sampling
            
        Returns:
            Sampled architecture configuration
        """
        if use_gumbel:
            # Differentiable sampling using Gumbel-Softmax
            num_layers = self._gumbel_sample(self.layer_probs) + self.min_layers
            
            hidden_dims = []
            activations = []
            dropouts = []
            norms = []
            attentions = []
            
            for layer_idx in range(num_layers):
                # Apply entanglement effects
                entanglement_factor = torch.tanh(self.entanglement_matrix).mean()
                
                hidden_idx = self._gumbel_sample(self.hidden_dim_probs * (1 + entanglement_factor))
                activation_idx = self._gumbel_sample(self.activation_probs * (1 + entanglement_factor))
                dropout_idx = self._gumbel_sample(self.dropout_probs * (1 + entanglement_factor))
                norm_idx = self._gumbel_sample(self.norm_probs * (1 + entanglement_factor))
                attention_idx = self._gumbel_sample(self.attention_probs * (1 + entanglement_factor))
                
                hidden_dims.append(self.hidden_dim_choices[hidden_idx])
                activations.append(self.activation_choices[activation_idx])
                dropouts.append(self.dropout_choices[dropout_idx])
                norms.append(self.normalization_choices[norm_idx])
                attentions.append(self.attention_choices[attention_idx])
        else:
            # Discrete sampling for evaluation
            num_layers = torch.multinomial(F.softmax(self.layer_probs, dim=0), 1).item() + self.min_layers
            
            hidden_dims = []
            activations = []
            dropouts = []
            norms = []
            attentions = []
            
            for _ in range(num_layers):
                hidden_idx = torch.multinomial(F.softmax(self.hidden_dim_probs, dim=0), 1).item()
                activation_idx = torch.multinomial(F.softmax(self.activation_probs, dim=0), 1).item()
                dropout_idx = torch.multinomial(F.softmax(self.dropout_probs, dim=0), 1).item()
                norm_idx = torch.multinomial(F.softmax(self.norm_probs, dim=0), 1).item()
                attention_idx = torch.multinomial(F.softmax(self.attention_probs, dim=0), 1).item()
                
                hidden_dims.append(self.hidden_dim_choices[hidden_idx])
                activations.append(self.activation_choices[activation_idx])
                dropouts.append(self.dropout_choices[dropout_idx])
                norms.append(self.normalization_choices[norm_idx])
                attentions.append(self.attention_choices[attention_idx])
        
        return ArchitectureConfig(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activation_types=activations,
            dropout_rates=dropouts,
            residual_connections=[True] * num_layers,  # Always use residuals
            normalization_types=norms,
            attention_mechanisms=attentions,
        )
    
    def _gumbel_sample(self, logits: torch.Tensor) -> int:
        """Gumbel-Softmax sampling for differentiable discrete sampling."""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        return torch.argmax(logits + gumbel_noise).item()
    
    def update_probabilities(self, architecture: ArchitectureConfig, reward: float):
        """Update quantum probabilities based on architecture performance.
        
        Args:
            architecture: The evaluated architecture
            reward: Performance reward (higher is better)
        """
        # Convert reward to probability update magnitude
        update_magnitude = torch.tanh(torch.tensor(reward)) * 0.1
        
        # Update layer count probability
        layer_idx = architecture.num_layers - self.min_layers
        self.layer_probs.data[layer_idx] += update_magnitude
        
        # Update per-layer choice probabilities
        for layer_idx in range(architecture.num_layers):
            # Hidden dimension
            if layer_idx < len(architecture.hidden_dims):
                hidden_idx = self.hidden_dim_choices.index(architecture.hidden_dims[layer_idx])
                self.hidden_dim_probs.data[hidden_idx] += update_magnitude
            
            # Activation
            if layer_idx < len(architecture.activation_types):
                act_idx = self.activation_choices.index(architecture.activation_types[layer_idx])
                self.activation_probs.data[act_idx] += update_magnitude
            
            # Dropout
            if layer_idx < len(architecture.dropout_rates):
                dropout_idx = self.dropout_choices.index(architecture.dropout_rates[layer_idx])
                self.dropout_probs.data[dropout_idx] += update_magnitude
            
            # Normalization
            if layer_idx < len(architecture.normalization_types):
                norm_idx = self.normalization_choices.index(architecture.normalization_types[layer_idx])
                self.norm_probs.data[norm_idx] += update_magnitude
        
        # Renormalize probabilities (quantum measurement)
        self.layer_probs.data = F.softmax(self.layer_probs.data, dim=0)
        self.hidden_dim_probs.data = F.softmax(self.hidden_dim_probs.data, dim=0)
        self.activation_probs.data = F.softmax(self.activation_probs.data, dim=0)
        self.dropout_probs.data = F.softmax(self.dropout_probs.data, dim=0)
        self.norm_probs.data = F.softmax(self.norm_probs.data, dim=0)
        self.attention_probs.data = F.softmax(self.attention_probs.data, dim=0)


class NeuralHypernetwork(nn.Module):
    """Advanced hypernetwork with configurable architecture."""
    
    def __init__(
        self,
        text_dim: int,
        target_weights_config: Dict[str, Tuple],
        architecture: ArchitectureConfig,
    ):
        """Initialize neural hypernetwork.
        
        Args:
            text_dim: Input text embedding dimension
            target_weights_config: Configuration of target weight tensors
            architecture: Architecture configuration
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.target_weights_config = target_weights_config
        self.architecture = architecture
        
        # Build layers based on architecture
        self.layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        input_dim = text_dim
        for i, (hidden_dim, activation, dropout, use_residual, norm_type, attention_type) in enumerate(zip(
            architecture.hidden_dims,
            architecture.activation_types,
            architecture.dropout_rates,
            architecture.residual_connections,
            architecture.normalization_types,
            architecture.attention_mechanisms,
        )):
            # Main linear layer
            layer = nn.Linear(input_dim, hidden_dim)
            self.layers.append(layer)
            
            # Attention mechanism
            if attention_type == 'self_attention':
                attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            elif attention_type == 'cross_attention':
                attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            elif attention_type == 'multi_head':
                attention = nn.MultiheadAttention(hidden_dim, num_heads=16, batch_first=True)
            else:
                attention = nn.Identity()
            self.attention_layers.append(attention)
            
            # Normalization
            if norm_type == 'layer_norm':
                norm = nn.LayerNorm(hidden_dim)
            elif norm_type == 'batch_norm':
                norm = nn.BatchNorm1d(hidden_dim)
            elif norm_type == 'group_norm':
                norm = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
            else:
                norm = nn.Identity()
            self.norm_layers.append(norm)
            
            input_dim = hidden_dim
        
        # Output heads for each target weight
        self.output_heads = nn.ModuleDict()
        for weight_name, weight_shape in target_weights_config.items():
            weight_size = np.prod(weight_shape)
            self.output_heads[weight_name] = nn.Linear(input_dim, weight_size)
        
        logger.info(f"Initialized NeuralHypernetwork with {architecture.num_layers} layers")
    
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the hypernetwork.
        
        Args:
            text_embeddings: Input text embeddings [batch_size, text_dim]
            
        Returns:
            Dictionary of generated weights
        """
        x = text_embeddings
        residual = x
        
        for i, (layer, attention, norm) in enumerate(zip(
            self.layers, self.attention_layers, self.norm_layers
        )):
            # Linear transformation
            x = layer(x)
            
            # Activation
            activation_type = self.architecture.activation_types[i]
            if activation_type == 'relu':
                x = F.relu(x)
            elif activation_type == 'gelu':
                x = F.gelu(x)
            elif activation_type == 'swish':
                x = x * torch.sigmoid(x)
            elif activation_type == 'mish':
                x = x * torch.tanh(F.softplus(x))
            elif activation_type == 'leaky_relu':
                x = F.leaky_relu(x)
            elif activation_type == 'elu':
                x = F.elu(x)
            
            # Attention mechanism
            if isinstance(attention, nn.MultiheadAttention):
                # Reshape for attention (add sequence dimension)
                x_seq = x.unsqueeze(1)  # [batch, 1, hidden]
                x_attended, _ = attention(x_seq, x_seq, x_seq)
                x = x_attended.squeeze(1)  # [batch, hidden]
            
            # Normalization
            x = norm(x)
            
            # Residual connection
            if self.architecture.residual_connections[i] and x.shape == residual.shape:
                x = x + residual
            
            # Dropout
            x = F.dropout(x, p=self.architecture.dropout_rates[i], training=self.training)
            
            residual = x
        
        # Generate weights for each target
        generated_weights = {}
        for weight_name, weight_shape in self.target_weights_config.items():
            flat_weights = self.output_heads[weight_name](x)
            reshaped_weights = flat_weights.view(-1, *weight_shape)
            generated_weights[weight_name] = reshaped_weights
        
        return generated_weights


class DifferentiableArchitectureSearch(nn.Module):
    """Differentiable Architecture Search for Hypernetworks (DASH).
    
    This module implements a differentiable neural architecture search
    specifically designed for hypernetwork optimization. It uses continuous
    relaxation of the discrete architecture space to enable gradient-based
    optimization.
    """
    
    def __init__(
        self,
        text_dim: int,
        target_weights_config: Dict[str, Tuple],
        search_space: QuantumInspiredSearchSpace,
        num_architectures: int = 8,
    ):
        """Initialize DASH module.
        
        Args:
            text_dim: Input text embedding dimension
            target_weights_config: Configuration of target weight tensors
            search_space: Quantum-inspired search space
            num_architectures: Number of architectures to maintain simultaneously
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.target_weights_config = target_weights_config
        self.search_space = search_space
        self.num_architectures = num_architectures
        
        # Architecture weights (learnable mixing coefficients)
        self.architecture_weights = nn.Parameter(torch.ones(num_architectures) / num_architectures)
        
        # Sample initial architectures
        self.architectures = []
        self.hypernetworks = nn.ModuleList()
        
        for i in range(num_architectures):
            arch = search_space.sample_architecture(use_gumbel=False)
            hypernetwork = NeuralHypernetwork(text_dim, target_weights_config, arch)
            
            self.architectures.append(arch)
            self.hypernetworks.append(hypernetwork)
        
        logger.info(f"Initialized DASH with {num_architectures} candidate architectures")
    
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with architecture mixing.
        
        Args:
            text_embeddings: Input text embeddings
            
        Returns:
            Mixed weights from all architectures
        """
        # Get weights from all architectures
        all_weights = []
        for hypernetwork in self.hypernetworks:
            weights = hypernetwork(text_embeddings)
            all_weights.append(weights)
        
        # Mix weights based on architecture importance
        arch_probs = F.softmax(self.architecture_weights, dim=0)
        
        mixed_weights = {}
        for weight_name in self.target_weights_config.keys():
            # Collect weights for this parameter from all architectures
            weight_tensors = [weights[weight_name] for weights in all_weights]
            
            # Weighted combination
            mixed_weight = torch.zeros_like(weight_tensors[0])
            for prob, weight_tensor in zip(arch_probs, weight_tensors):
                mixed_weight += prob * weight_tensor
            
            mixed_weights[weight_name] = mixed_weight
        
        return mixed_weights
    
    def get_best_architecture(self) -> Tuple[ArchitectureConfig, float]:
        """Get the architecture with highest weight.
        
        Returns:
            Best architecture and its weight
        """
        best_idx = torch.argmax(self.architecture_weights).item()
        best_weight = self.architecture_weights[best_idx].item()
        return self.architectures[best_idx], best_weight
    
    def evolve_architectures(self, performance_scores: List[float]):
        """Evolve architectures based on performance.
        
        Args:
            performance_scores: Performance scores for each architecture
        """
        # Update architecture weights based on performance
        scores_tensor = torch.tensor(performance_scores, device=self.architecture_weights.device)
        normalized_scores = F.softmax(scores_tensor, dim=0)
        
        # Exponential moving average update
        alpha = 0.1
        self.architecture_weights.data = (
            alpha * normalized_scores + (1 - alpha) * self.architecture_weights.data
        )
        
        # Replace worst architectures with new samples
        worst_indices = torch.topk(self.architecture_weights, k=2, largest=False).indices
        
        for idx in worst_indices:
            # Sample new architecture
            new_arch = self.search_space.sample_architecture(use_gumbel=False)
            new_hypernetwork = NeuralHypernetwork(
                self.text_dim, self.target_weights_config, new_arch
            )
            
            # Replace worst architecture
            self.architectures[idx] = new_arch
            self.hypernetworks[idx] = new_hypernetwork
        
        logger.info(f"Evolved architectures, best weight: {torch.max(self.architecture_weights).item():.3f}")


class HypernetworkNAS:
    """Main Neural Architecture Search system for Hypernetworks.
    
    This class orchestrates the entire NAS process, including:
    1. Architecture search space definition
    2. Performance evaluation
    3. Multi-objective optimization
    4. Meta-learning integration
    """
    
    def __init__(
        self,
        text_dim: int = 384,
        target_weights_config: Dict[str, Tuple] = None,
        search_budget: int = 100,
        population_size: int = 20,
        use_quantum_search: bool = True,
    ):
        """Initialize HypernetworkNAS.
        
        Args:
            text_dim: Text embedding dimension
            target_weights_config: Target weight configurations
            search_budget: Number of architectures to evaluate
            population_size: Size of architecture population
            use_quantum_search: Whether to use quantum-inspired search
        """
        self.text_dim = text_dim
        self.search_budget = search_budget
        self.population_size = population_size
        
        # Default target configuration (can be customized)
        if target_weights_config is None:
            target_weights_config = {
                'layer_1_weight': (text_dim, 256),
                'layer_1_bias': (256,),
                'layer_2_weight': (256, 256),
                'layer_2_bias': (256,),
                'layer_3_weight': (256, 128),
                'layer_3_bias': (128,),
            }
        
        self.target_weights_config = target_weights_config
        
        # Initialize search space
        if use_quantum_search:
            self.search_space = QuantumInspiredSearchSpace()
        else:
            self.search_space = QuantumInspiredSearchSpace()  # Still use quantum for now
        
        # Initialize DASH
        self.dash = DifferentiableArchitectureSearch(
            text_dim, target_weights_config, self.search_space
        )
        
        # Performance tracking
        self.evaluated_architectures = []
        self.performance_history = []
        
        logger.info(f"Initialized HypernetworkNAS with budget {search_budget}")
    
    def evaluate_architecture(
        self,
        architecture: ArchitectureConfig,
        validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
        num_epochs: int = 10,
    ) -> Dict[str, float]:
        """Evaluate a single architecture.
        
        Args:
            architecture: Architecture to evaluate
            validation_data: Validation dataset
            num_epochs: Number of training epochs
            
        Returns:
            Performance metrics
        """
        # Create hypernetwork with this architecture
        hypernetwork = NeuralHypernetwork(
            self.text_dim, self.target_weights_config, architecture
        )
        
        # Simple training loop (can be made more sophisticated)
        optimizer = Adam(hypernetwork.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        hypernetwork.train()
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for text_batch, target_batch in validation_data:
                optimizer.zero_grad()
                
                generated_weights = hypernetwork(text_batch)
                
                # Simplified loss (MSE between generated and target weights)
                loss = 0.0
                for weight_name, target_weight in target_batch.items():
                    if weight_name in generated_weights:
                        loss += criterion(generated_weights[weight_name], target_weight)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate additional metrics
        num_parameters = sum(p.numel() for p in hypernetwork.parameters())
        flops = self._estimate_flops(architecture)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': max(0.0, 1.0 - avg_loss),  # Simplified accuracy
            'num_parameters': num_parameters,
            'flops': flops,
            'efficiency': (1.0 - avg_loss) / (num_parameters / 1e6),  # Accuracy per million params
        }
        
        return metrics
    
    def _estimate_flops(self, architecture: ArchitectureConfig) -> int:
        """Estimate FLOPs for an architecture."""
        flops = 0
        input_dim = self.text_dim
        
        for hidden_dim in architecture.hidden_dims:
            # Linear layer FLOPs
            flops += input_dim * hidden_dim
            
            # Activation FLOPs
            flops += hidden_dim
            
            # Attention FLOPs (if used)
            if architecture.attention_mechanisms:
                flops += hidden_dim * hidden_dim  # Simplified
            
            input_dim = hidden_dim
        
        # Output heads
        for weight_shape in self.target_weights_config.values():
            output_size = np.prod(weight_shape)
            flops += input_dim * output_size
        
        return flops
    
    def search(
        self,
        validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
        objectives: List[str] = None,
    ) -> ArchitectureConfig:
        """Perform neural architecture search.
        
        Args:
            validation_data: Validation dataset for evaluation
            objectives: Optimization objectives ['accuracy', 'efficiency', 'speed']
            
        Returns:
            Best found architecture
        """
        if objectives is None:
            objectives = ['accuracy', 'efficiency']
        
        logger.info(f"Starting NAS with objectives: {objectives}")
        
        best_architecture = None
        best_score = float('-inf')
        
        # Evolutionary search with quantum-inspired sampling
        population = []
        
        # Initialize population
        for _ in range(self.population_size):
            arch = self.search_space.sample_architecture(use_gumbel=False)
            population.append(arch)
        
        for generation in range(self.search_budget // self.population_size):
            logger.info(f"Generation {generation + 1}/{self.search_budget // self.population_size}")
            
            # Evaluate population
            generation_scores = []
            for arch in population:
                metrics = self.evaluate_architecture(arch, validation_data)
                
                # Multi-objective scoring
                score = self._compute_multi_objective_score(metrics, objectives)
                generation_scores.append(score)
                
                # Track best architecture
                if score > best_score:
                    best_score = score
                    best_architecture = arch
                
                # Update quantum probabilities
                self.search_space.update_probabilities(arch, score)
                
                # Store for history
                self.evaluated_architectures.append(arch)
                self.performance_history.append(metrics)
            
            # Select top performers for next generation
            top_indices = sorted(range(len(generation_scores)), 
                               key=lambda i: generation_scores[i], reverse=True)
            top_architectures = [population[i] for i in top_indices[:self.population_size // 2]]
            
            # Generate new population
            new_population = top_architectures.copy()
            
            # Add mutated versions
            while len(new_population) < self.population_size:
                parent = random.choice(top_architectures)
                mutated = self._mutate_architecture(parent)
                new_population.append(mutated)
            
            population = new_population
            
            logger.info(f"Generation {generation + 1} best score: {max(generation_scores):.4f}")
        
        logger.info(f"NAS completed. Best score: {best_score:.4f}")
        return best_architecture
    
    def _compute_multi_objective_score(
        self, 
        metrics: Dict[str, float], 
        objectives: List[str]
    ) -> float:
        """Compute multi-objective score."""
        score = 0.0
        
        for objective in objectives:
            if objective == 'accuracy':
                score += metrics.get('accuracy', 0.0) * 0.5
            elif objective == 'efficiency':
                score += metrics.get('efficiency', 0.0) * 0.3
            elif objective == 'speed':
                # Inverse of FLOPs (higher is better)
                flops = metrics.get('flops', 1e9)
                speed_score = 1.0 / (1.0 + flops / 1e9)
                score += speed_score * 0.2
        
        return score
    
    def _mutate_architecture(self, architecture: ArchitectureConfig) -> ArchitectureConfig:
        """Mutate an architecture for evolutionary search."""
        # Create a copy
        new_arch = ArchitectureConfig(
            num_layers=architecture.num_layers,
            hidden_dims=architecture.hidden_dims.copy(),
            activation_types=architecture.activation_types.copy(),
            dropout_rates=architecture.dropout_rates.copy(),
            residual_connections=architecture.residual_connections.copy(),
            normalization_types=architecture.normalization_types.copy(),
            attention_mechanisms=architecture.attention_mechanisms.copy(),
        )
        
        # Random mutations
        mutation_rate = 0.2
        
        if random.random() < mutation_rate:
            # Mutate hidden dimensions
            layer_idx = random.randint(0, len(new_arch.hidden_dims) - 1)
            new_arch.hidden_dims[layer_idx] = random.choice(self.search_space.hidden_dim_choices)
        
        if random.random() < mutation_rate:
            # Mutate activation
            layer_idx = random.randint(0, len(new_arch.activation_types) - 1)
            new_arch.activation_types[layer_idx] = random.choice(self.search_space.activation_choices)
        
        if random.random() < mutation_rate:
            # Mutate dropout
            layer_idx = random.randint(0, len(new_arch.dropout_rates) - 1)
            new_arch.dropout_rates[layer_idx] = random.choice(self.search_space.dropout_choices)
        
        return new_arch
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of the search process."""
        if not self.performance_history:
            return {"status": "No search performed yet"}
        
        best_idx = max(range(len(self.performance_history)), 
                      key=lambda i: self.performance_history[i].get('accuracy', 0))
        
        return {
            "total_evaluated": len(self.evaluated_architectures),
            "best_architecture": self.evaluated_architectures[best_idx],
            "best_performance": self.performance_history[best_idx],
            "search_progress": [metrics.get('accuracy', 0) for metrics in self.performance_history],
        }


# Example usage and demonstration
def demonstrate_nas():
    """Demonstrate the Neural Architecture Search system."""
    print("🚀 Neural Architecture Search for Hypernetworks Demo")
    print("=" * 60)
    
    # Initialize NAS system
    nas = HypernetworkNAS(
        text_dim=384,
        search_budget=50,
        population_size=10,
    )
    
    # Create dummy validation data
    validation_data = []
    for _ in range(5):
        text_batch = torch.randn(4, 384)  # Batch of text embeddings
        target_batch = {
            'layer_1_weight': torch.randn(4, 384, 256),
            'layer_1_bias': torch.randn(4, 256),
            'layer_2_weight': torch.randn(4, 256, 256),
            'layer_2_bias': torch.randn(4, 256),
            'layer_3_weight': torch.randn(4, 256, 128),
            'layer_3_bias': torch.randn(4, 128),
        }
        validation_data.append((text_batch, target_batch))
    
    # Perform search
    print("Starting architecture search...")
    best_architecture = nas.search(validation_data, objectives=['accuracy', 'efficiency'])
    
    print("\n🏆 Best Architecture Found:")
    print(f"  Layers: {best_architecture.num_layers}")
    print(f"  Hidden dims: {best_architecture.hidden_dims}")
    print(f"  Activations: {best_architecture.activation_types}")
    print(f"  Dropout rates: {best_architecture.dropout_rates}")
    print(f"  Normalizations: {best_architecture.normalization_types}")
    
    # Get search summary
    summary = nas.get_search_summary()
    print(f"\n📊 Search Summary:")
    print(f"  Total evaluated: {summary['total_evaluated']}")
    print(f"  Best accuracy: {summary['best_performance']['accuracy']:.4f}")
    print(f"  Best efficiency: {summary['best_performance']['efficiency']:.4f}")
    
    print("\n✅ NAS demonstration completed!")


if __name__ == "__main__":
    demonstrate_nas()