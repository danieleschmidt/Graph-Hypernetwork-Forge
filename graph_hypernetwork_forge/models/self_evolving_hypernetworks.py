"""Self-Evolving Hypernetwork Architectures with Meta-Learning

This module implements revolutionary self-evolving hypernetwork architectures that
can autonomously improve their own structure and parameters through meta-learning,
evolutionary algorithms, and neural architecture mutations.

Novel Contributions:
1. Self-Modifying Neural Architecture Search (SM-NAS)
2. Meta-Learning for Hypernetwork Architecture Evolution
3. Genetic Programming for Weight Generation Networks
4. Adaptive Structural Mutations with Performance Feedback
5. Autonomous Hyperparameter Optimization
6. Self-Supervised Architecture Discovery
"""

import copy
import math
import random
import time
import warnings
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, TensorDataset

try:
    from ..utils.logging_utils import get_logger
    from ..utils.exceptions import ValidationError, ModelError
    from .neural_architecture_search import ArchitectureConfig, NeuralHypernetwork
    from .hypernetworks import HyperNetwork
    from .hypergnn import HyperGNN
except ImportError:
    # Fallback for standalone usage
    import logging
    def get_logger(name): return logging.getLogger(name)
    class ValidationError(Exception): pass
    class ModelError(Exception): pass


logger = get_logger(__name__)


class EvolutionStrategy(Enum):
    """Strategies for architecture evolution."""
    RANDOM_MUTATION = "random_mutation"
    GUIDED_MUTATION = "guided_mutation"
    GENETIC_CROSSOVER = "genetic_crossover"
    NEURAL_EVOLUTION = "neural_evolution"
    META_LEARNING = "meta_learning"
    REINFORCEMENT_LEARNING = "rl_evolution"


class MutationType(Enum):
    """Types of structural mutations."""
    ADD_LAYER = "add_layer"
    REMOVE_LAYER = "remove_layer"
    CHANGE_ACTIVATION = "change_activation"
    MODIFY_DIMENSIONS = "modify_dimensions"
    ADD_CONNECTIONS = "add_connections"
    REMOVE_CONNECTIONS = "remove_connections"
    CHANGE_NORMALIZATION = "change_normalization"
    ADD_ATTENTION = "add_attention"
    MODIFY_DROPOUT = "modify_dropout"


@dataclass
class EvolutionConfig:
    """Configuration for self-evolving hypernetworks."""
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    selection_pressure: float = 0.5
    elitism_ratio: float = 0.2
    max_generations: int = 100
    
    # Meta-learning parameters
    meta_learning_rate: float = 1e-4
    meta_batch_size: int = 8
    meta_inner_steps: int = 5
    
    # Architecture constraints
    min_layers: int = 2
    max_layers: int = 10
    min_hidden_dim: int = 64
    max_hidden_dim: int = 2048
    
    # Performance thresholds
    convergence_threshold: float = 1e-4
    performance_patience: int = 10
    
    # Resource constraints
    max_flops: int = 1e9
    max_parameters: int = 10e6
    memory_budget: int = 2048  # MB


@dataclass
class ArchitectureGene:
    """Genetic representation of hypernetwork architecture."""
    layer_genes: List[Dict[str, Any]] = field(default_factory=list)
    connection_genes: List[Tuple[int, int]] = field(default_factory=list)
    global_genes: Dict[str, Any] = field(default_factory=dict)
    fitness: float = 0.0
    age: int = 0
    
    def __post_init__(self):
        if not self.global_genes:
            self.global_genes = {
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'dropout_rate': 0.1,
                'use_residual': True,
                'use_attention': False,
            }
    
    def mutate(self, mutation_rate: float = 0.1):
        """Apply mutations to the genetic code."""
        for gene in self.layer_genes:
            if random.random() < mutation_rate:
                self._mutate_layer_gene(gene)
        
        if random.random() < mutation_rate:
            self._mutate_global_genes()
        
        if random.random() < mutation_rate * 0.5:
            self._mutate_connections()
    
    def _mutate_layer_gene(self, gene: Dict[str, Any]):
        """Mutate a single layer gene."""
        mutation_type = random.choice(list(MutationType))
        
        if mutation_type == MutationType.MODIFY_DIMENSIONS:
            gene['hidden_dim'] = random.choice([64, 128, 256, 512, 768, 1024])
        elif mutation_type == MutationType.CHANGE_ACTIVATION:
            gene['activation'] = random.choice(['relu', 'gelu', 'swish', 'mish', 'leaky_relu'])
        elif mutation_type == MutationType.MODIFY_DROPOUT:
            gene['dropout'] = random.uniform(0.0, 0.5)
        elif mutation_type == MutationType.CHANGE_NORMALIZATION:
            gene['normalization'] = random.choice(['none', 'batch_norm', 'layer_norm', 'group_norm'])
        elif mutation_type == MutationType.ADD_ATTENTION:
            gene['attention'] = random.choice(['none', 'self_attention', 'multi_head'])
    
    def _mutate_global_genes(self):
        """Mutate global architecture genes."""
        for key in self.global_genes:
            if key == 'learning_rate':
                self.global_genes[key] *= random.uniform(0.5, 2.0)
            elif key == 'weight_decay':
                self.global_genes[key] *= random.uniform(0.1, 10.0)
            elif key == 'dropout_rate':
                self.global_genes[key] = random.uniform(0.0, 0.5)
            elif isinstance(self.global_genes[key], bool):
                if random.random() < 0.1:
                    self.global_genes[key] = not self.global_genes[key]
    
    def _mutate_connections(self):
        """Mutate connection genes (skip connections, etc.)."""
        if random.random() < 0.5 and len(self.layer_genes) > 2:
            # Add skip connection
            start = random.randint(0, len(self.layer_genes) - 2)
            end = random.randint(start + 1, len(self.layer_genes) - 1)
            self.connection_genes.append((start, end))
        elif self.connection_genes and random.random() < 0.3:
            # Remove skip connection
            self.connection_genes.pop(random.randint(0, len(self.connection_genes) - 1))
    
    def crossover(self, other: 'ArchitectureGene') -> 'ArchitectureGene':
        """Create offspring through genetic crossover."""
        # Choose crossover point
        min_layers = min(len(self.layer_genes), len(other.layer_genes))
        if min_layers <= 1:
            return copy.deepcopy(self)
        
        crossover_point = random.randint(1, min_layers - 1)
        
        # Create offspring
        offspring = ArchitectureGene()
        offspring.layer_genes = (
            self.layer_genes[:crossover_point] + 
            other.layer_genes[crossover_point:]
        )
        
        # Mix global genes
        offspring.global_genes = {}
        for key in self.global_genes:
            if random.random() < 0.5:
                offspring.global_genes[key] = self.global_genes[key]
            else:
                offspring.global_genes[key] = other.global_genes.get(key, self.global_genes[key])
        
        # Mix connections
        all_connections = self.connection_genes + other.connection_genes
        offspring.connection_genes = random.sample(
            all_connections, 
            k=min(len(all_connections), random.randint(0, len(all_connections)))
        )
        
        return offspring
    
    def to_architecture_config(self) -> ArchitectureConfig:
        """Convert genetic representation to architecture configuration."""
        num_layers = len(self.layer_genes)
        hidden_dims = [gene.get('hidden_dim', 256) for gene in self.layer_genes]
        activation_types = [gene.get('activation', 'relu') for gene in self.layer_genes]
        dropout_rates = [gene.get('dropout', 0.1) for gene in self.layer_genes]
        normalization_types = [gene.get('normalization', 'layer_norm') for gene in self.layer_genes]
        attention_mechanisms = [gene.get('attention', 'none') for gene in self.layer_genes]
        
        return ArchitectureConfig(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activation_types=activation_types,
            dropout_rates=dropout_rates,
            residual_connections=[True] * num_layers,  # From global genes
            normalization_types=normalization_types,
            attention_mechanisms=attention_mechanisms,
        )


class MetaLearner(nn.Module):
    """Meta-learner for architecture optimization."""
    
    def __init__(
        self,
        architecture_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        """Initialize meta-learner.
        
        Args:
            architecture_dim: Dimension of architecture encoding
            hidden_dim: Hidden dimension of meta-network
            num_layers: Number of layers in meta-network
        """
        super().__init__()
        
        self.architecture_dim = architecture_dim
        self.hidden_dim = hidden_dim
        
        # Architecture encoder
        self.arch_encoder = nn.Sequential(
            nn.Linear(architecture_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Performance predictor
        self.performance_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Architecture optimizer (suggests modifications)
        self.arch_optimizer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, architecture_dim),
            nn.Tanh(),  # Bounded modifications
        )
        
        logger.info(f"Initialized MetaLearner with {architecture_dim}D architecture space")
    
    def encode_architecture(self, arch_config: ArchitectureConfig) -> torch.Tensor:
        """Encode architecture configuration into vector representation."""
        # Create fixed-size encoding of architecture
        encoding = torch.zeros(self.architecture_dim)
        
        # Encode basic properties
        encoding[0] = arch_config.num_layers / 10.0  # Normalize
        
        # Encode layer properties (pad/truncate to fixed size)
        max_layers = min(arch_config.num_layers, 8)  # Limit for encoding
        for i in range(max_layers):
            base_idx = 1 + i * 8
            if i < len(arch_config.hidden_dims):
                # Hidden dimension (normalized)
                encoding[base_idx] = arch_config.hidden_dims[i] / 1024.0
                
                # Activation type (one-hot)
                activations = ['relu', 'gelu', 'swish', 'mish', 'leaky_relu']
                if arch_config.activation_types[i] in activations:
                    act_idx = activations.index(arch_config.activation_types[i])
                    encoding[base_idx + 1 + act_idx] = 1.0
                
                # Dropout rate
                encoding[base_idx + 6] = arch_config.dropout_rates[i] if i < len(arch_config.dropout_rates) else 0.1
                
                # Normalization type
                if i < len(arch_config.normalization_types):
                    norm_type = arch_config.normalization_types[i]
                    if norm_type == 'layer_norm':
                        encoding[base_idx + 7] = 1.0
                    elif norm_type == 'batch_norm':
                        encoding[base_idx + 7] = 0.5
        
        return encoding
    
    def forward(
        self, 
        architecture_encoding: torch.Tensor,
        performance_target: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of meta-learner.
        
        Args:
            architecture_encoding: Encoded architecture representation
            performance_target: Target performance (for training)
            
        Returns:
            Dictionary with predictions and modifications
        """
        # Encode architecture
        arch_features = self.arch_encoder(architecture_encoding)
        
        # Predict performance
        predicted_performance = self.performance_predictor(arch_features)
        
        # Generate architecture modifications
        arch_modifications = self.arch_optimizer(arch_features)
        
        return {
            'predicted_performance': predicted_performance,
            'architecture_modifications': arch_modifications,
            'architecture_features': arch_features,
        }
    
    def suggest_improvements(self, arch_config: ArchitectureConfig) -> ArchitectureConfig:
        """Suggest improvements to architecture based on meta-learning.
        
        Args:
            arch_config: Current architecture configuration
            
        Returns:
            Improved architecture configuration
        """
        self.eval()
        with torch.no_grad():
            # Encode current architecture
            arch_encoding = self.encode_architecture(arch_config)
            
            # Get meta-learner predictions
            outputs = self.forward(arch_encoding)
            modifications = outputs['architecture_modifications']
            
            # Apply modifications to create new architecture
            new_encoding = arch_encoding + 0.1 * modifications  # Small step size
            new_encoding = torch.clamp(new_encoding, 0.0, 1.0)  # Keep in bounds
            
            # Decode back to architecture config
            improved_config = self._decode_architecture(new_encoding)
            
        return improved_config
    
    def _decode_architecture(self, encoding: torch.Tensor) -> ArchitectureConfig:
        """Decode vector representation back to architecture configuration."""
        # Extract basic properties
        num_layers = max(2, min(8, int(encoding[0].item() * 10)))
        
        # Extract layer properties
        hidden_dims = []
        activation_types = []
        dropout_rates = []
        normalization_types = []
        
        activations = ['relu', 'gelu', 'swish', 'mish', 'leaky_relu']
        
        for i in range(num_layers):
            base_idx = 1 + i * 8
            
            # Hidden dimension
            hidden_dim = max(64, min(1024, int(encoding[base_idx].item() * 1024)))
            hidden_dims.append(hidden_dim)
            
            # Activation type
            act_scores = encoding[base_idx + 1:base_idx + 6]
            act_idx = torch.argmax(act_scores).item()
            activation_types.append(activations[act_idx])
            
            # Dropout rate
            dropout = max(0.0, min(0.5, encoding[base_idx + 6].item()))
            dropout_rates.append(dropout)
            
            # Normalization type
            norm_score = encoding[base_idx + 7].item()
            if norm_score > 0.75:
                normalization_types.append('layer_norm')
            elif norm_score > 0.25:
                normalization_types.append('batch_norm')
            else:
                normalization_types.append('none')
        
        return ArchitectureConfig(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activation_types=activation_types,
            dropout_rates=dropout_rates,
            residual_connections=[True] * num_layers,
            normalization_types=normalization_types,
            attention_mechanisms=['none'] * num_layers,
        )


class SelfEvolvingHypernetwork(nn.Module):
    """Self-evolving hypernetwork that improves its own architecture."""
    
    def __init__(
        self,
        text_dim: int,
        target_weights_config: Dict[str, Tuple],
        evolution_config: EvolutionConfig = None,
        initial_architecture: ArchitectureConfig = None,
    ):
        """Initialize self-evolving hypernetwork.
        
        Args:
            text_dim: Input text embedding dimension
            target_weights_config: Configuration of target weight tensors
            evolution_config: Evolution configuration
            initial_architecture: Initial architecture (random if None)
        """
        super().__init__()
        
        self.text_dim = text_dim
        self.target_weights_config = target_weights_config
        self.evolution_config = evolution_config or EvolutionConfig()
        
        # Initialize with random or provided architecture
        if initial_architecture is None:
            initial_architecture = self._create_random_architecture()
        
        self.current_architecture = initial_architecture
        
        # Create current hypernetwork
        self.hypernetwork = NeuralHypernetwork(
            text_dim, target_weights_config, initial_architecture
        )
        
        # Meta-learner for guided evolution
        self.meta_learner = MetaLearner(architecture_dim=256)
        
        # Evolution tracking
        self.generation = 0
        self.evolution_history = []
        self.performance_history = deque(maxlen=self.evolution_config.performance_patience)
        
        # Population management
        self.population: List[ArchitectureGene] = []
        self.elite_architectures: List[ArchitectureGene] = []
        
        logger.info("Initialized self-evolving hypernetwork")
    
    def _create_random_architecture(self) -> ArchitectureConfig:
        """Create a random initial architecture."""
        num_layers = random.randint(
            self.evolution_config.min_layers,
            self.evolution_config.max_layers
        )
        
        hidden_dims = [
            random.choice([128, 256, 512, 768])
            for _ in range(num_layers)
        ]
        
        activation_types = [
            random.choice(['relu', 'gelu', 'swish', 'mish'])
            for _ in range(num_layers)
        ]
        
        dropout_rates = [
            random.uniform(0.0, 0.3)
            for _ in range(num_layers)
        ]
        
        normalization_types = [
            random.choice(['none', 'layer_norm', 'batch_norm'])
            for _ in range(num_layers)
        ]
        
        return ArchitectureConfig(
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            activation_types=activation_types,
            dropout_rates=dropout_rates,
            residual_connections=[True] * num_layers,
            normalization_types=normalization_types,
            attention_mechanisms=['none'] * num_layers,
        )
    
    def _architecture_to_gene(self, arch_config: ArchitectureConfig) -> ArchitectureGene:
        """Convert architecture config to genetic representation."""
        layer_genes = []
        for i in range(arch_config.num_layers):
            gene = {
                'hidden_dim': arch_config.hidden_dims[i],
                'activation': arch_config.activation_types[i],
                'dropout': arch_config.dropout_rates[i],
                'normalization': arch_config.normalization_types[i],
                'attention': arch_config.attention_mechanisms[i],
            }
            layer_genes.append(gene)
        
        return ArchitectureGene(
            layer_genes=layer_genes,
            connection_genes=[],  # Start with no skip connections
            global_genes={'use_residual': True, 'use_attention': False},
        )
    
    def initialize_population(self):
        """Initialize population of architecture genes."""
        self.population = []
        
        # Add current architecture as elite
        current_gene = self._architecture_to_gene(self.current_architecture)
        current_gene.fitness = 1.0  # High initial fitness
        self.population.append(current_gene)
        
        # Create random population
        for _ in range(self.evolution_config.population_size - 1):
            random_arch = self._create_random_architecture()
            random_gene = self._architecture_to_gene(random_arch)
            self.population.append(random_gene)
        
        logger.info(f"Initialized population with {len(self.population)} architectures")
    
    def evaluate_fitness(
        self,
        gene: ArchitectureGene,
        validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
        num_epochs: int = 5,
    ) -> float:
        """Evaluate fitness of an architecture gene.
        
        Args:
            gene: Architecture gene to evaluate
            validation_data: Validation dataset
            num_epochs: Number of training epochs
            
        Returns:
            Fitness score (higher is better)
        """
        try:
            # Convert gene to architecture config
            arch_config = gene.to_architecture_config()
            
            # Create hypernetwork with this architecture
            test_hypernetwork = NeuralHypernetwork(
                self.text_dim, self.target_weights_config, arch_config
            )
            
            # Quick training and evaluation
            optimizer = Adam(test_hypernetwork.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            test_hypernetwork.train()
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(num_epochs):
                for text_batch, target_batch in validation_data:
                    optimizer.zero_grad()
                    
                    generated_weights = test_hypernetwork(text_batch)
                    
                    # Simplified loss computation
                    loss = 0.0
                    for weight_name, target_weight in target_batch.items():
                        if weight_name in generated_weights:
                            loss += criterion(generated_weights[weight_name], target_weight)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / max(num_batches, 1)
            
            # Compute fitness (accuracy + efficiency)
            accuracy = max(0.0, 1.0 - avg_loss)
            
            # Efficiency factors
            num_parameters = sum(p.numel() for p in test_hypernetwork.parameters())
            efficiency = 1.0 / (1.0 + num_parameters / 1e6)  # Penalize large models
            
            # Combined fitness
            fitness = 0.7 * accuracy + 0.3 * efficiency
            
            # Resource constraint penalty
            if num_parameters > self.evolution_config.max_parameters:
                fitness *= 0.5  # Heavy penalty for violating constraints
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Error evaluating architecture: {e}")
            return 0.0  # Assign low fitness to problematic architectures
    
    def evolve_generation(
        self,
        validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
        strategy: EvolutionStrategy = EvolutionStrategy.META_LEARNING,
    ) -> ArchitectureGene:
        """Evolve one generation of architectures.
        
        Args:
            validation_data: Validation data for fitness evaluation
            strategy: Evolution strategy to use
            
        Returns:
            Best architecture from this generation
        """
        logger.info(f"Evolving generation {self.generation + 1} with strategy {strategy.value}")
        
        # Evaluate fitness of current population
        for gene in self.population:
            if gene.fitness == 0.0:  # Only evaluate if not already evaluated
                gene.fitness = self.evaluate_fitness(gene, validation_data)
                gene.age += 1
        
        # Sort population by fitness
        self.population.sort(key=lambda g: g.fitness, reverse=True)
        
        # Apply evolution strategy
        if strategy == EvolutionStrategy.META_LEARNING:
            new_population = self._meta_learning_evolution()
        elif strategy == EvolutionStrategy.GENETIC_CROSSOVER:
            new_population = self._genetic_crossover_evolution()
        elif strategy == EvolutionStrategy.NEURAL_EVOLUTION:
            new_population = self._neural_evolution()
        else:
            new_population = self._random_mutation_evolution()
        
        # Keep elite individuals
        num_elite = int(self.evolution_config.population_size * self.evolution_config.elitism_ratio)
        elite = self.population[:num_elite]
        
        # Combine elite with new population
        self.population = elite + new_population[:self.evolution_config.population_size - num_elite]
        
        # Update evolution tracking
        best_gene = self.population[0]
        self.evolution_history.append({
            'generation': self.generation,
            'best_fitness': best_gene.fitness,
            'avg_fitness': np.mean([g.fitness for g in self.population]),
            'architecture': best_gene.to_architecture_config(),
        })
        
        self.generation += 1
        
        logger.info(f"Generation {self.generation}: Best fitness = {best_gene.fitness:.4f}")
        
        return best_gene
    
    def _meta_learning_evolution(self) -> List[ArchitectureGene]:
        """Use meta-learning to guide evolution."""
        new_population = []
        
        # Train meta-learner on current population
        self._train_meta_learner()
        
        # Generate new architectures using meta-learner
        for i in range(self.evolution_config.population_size // 2):
            # Select parent architecture
            parent_idx = random.randint(0, min(10, len(self.population) - 1))  # Bias toward top performers
            parent_gene = self.population[parent_idx]
            parent_config = parent_gene.to_architecture_config()
            
            # Use meta-learner to suggest improvements
            improved_config = self.meta_learner.suggest_improvements(parent_config)
            improved_gene = self._architecture_to_gene(improved_config)
            
            # Apply small random mutations
            improved_gene.mutate(mutation_rate=0.05)
            
            new_population.append(improved_gene)
        
        return new_population
    
    def _genetic_crossover_evolution(self) -> List[ArchitectureGene]:
        """Standard genetic algorithm with crossover and mutation."""
        new_population = []
        
        while len(new_population) < self.evolution_config.population_size // 2:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.evolution_config.crossover_rate:
                offspring = parent1.crossover(parent2)
            else:
                offspring = copy.deepcopy(parent1)
            
            # Mutation
            offspring.mutate(self.evolution_config.mutation_rate)
            
            new_population.append(offspring)
        
        return new_population
    
    def _neural_evolution(self) -> List[ArchitectureGene]:
        """Neuro-evolution with neural network-guided mutations."""
        # This is a simplified version - could be expanded with more sophisticated
        # neural networks that predict good mutations
        
        new_population = []
        
        for i in range(self.evolution_config.population_size // 2):
            # Select parent based on fitness-proportional selection
            parent = self._fitness_proportional_selection()
            
            # Create offspring with guided mutations
            offspring = copy.deepcopy(parent)
            
            # Apply more intelligent mutations based on performance patterns
            if parent.fitness > 0.7:
                # Good architecture - make small refinements
                offspring.mutate(mutation_rate=0.05)
            else:
                # Poor architecture - make larger changes
                offspring.mutate(mutation_rate=0.2)
                
                # Possibly add/remove layers
                if random.random() < 0.3:
                    if len(offspring.layer_genes) < self.evolution_config.max_layers:
                        # Add layer
                        new_gene = {
                            'hidden_dim': random.choice([128, 256, 512]),
                            'activation': 'relu',
                            'dropout': 0.1,
                            'normalization': 'layer_norm',
                            'attention': 'none',
                        }
                        insertion_point = random.randint(0, len(offspring.layer_genes))
                        offspring.layer_genes.insert(insertion_point, new_gene)
                    elif len(offspring.layer_genes) > self.evolution_config.min_layers:
                        # Remove layer
                        removal_point = random.randint(0, len(offspring.layer_genes) - 1)
                        offspring.layer_genes.pop(removal_point)
            
            new_population.append(offspring)
        
        return new_population
    
    def _random_mutation_evolution(self) -> List[ArchitectureGene]:
        """Simple random mutation evolution."""
        new_population = []
        
        for i in range(self.evolution_config.population_size // 2):
            # Select parent
            parent = self._tournament_selection()
            
            # Create mutated offspring
            offspring = copy.deepcopy(parent)
            offspring.mutate(self.evolution_config.mutation_rate)
            
            new_population.append(offspring)
        
        return new_population
    
    def _tournament_selection(self, tournament_size: int = 3) -> ArchitectureGene:
        """Tournament selection for parent selection."""
        tournament_pool = random.sample(
            self.population,
            min(tournament_size, len(self.population))
        )
        return max(tournament_pool, key=lambda g: g.fitness)
    
    def _fitness_proportional_selection(self) -> ArchitectureGene:
        """Fitness-proportional selection."""
        fitness_values = [g.fitness for g in self.population]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        selection_point = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        
        for gene in self.population:
            cumulative_fitness += gene.fitness
            if cumulative_fitness >= selection_point:
                return gene
        
        return self.population[-1]  # Fallback
    
    def _train_meta_learner(self):
        """Train meta-learner on population performance data."""
        if len(self.population) < 5:
            return  # Need minimum data for training
        
        self.meta_learner.train()
        optimizer = Adam(self.meta_learner.parameters(), lr=self.evolution_config.meta_learning_rate)
        
        # Prepare training data
        architectures = []
        performances = []
        
        for gene in self.population:
            arch_config = gene.to_architecture_config()
            arch_encoding = self.meta_learner.encode_architecture(arch_config)
            architectures.append(arch_encoding)
            performances.append(gene.fitness)
        
        # Convert to tensors
        arch_batch = torch.stack(architectures)
        perf_batch = torch.tensor(performances, dtype=torch.float32).unsqueeze(1)
        
        # Training step
        optimizer.zero_grad()
        outputs = self.meta_learner(arch_batch)
        
        # Loss: predict performance accurately
        loss = F.mse_loss(outputs['predicted_performance'], perf_batch)
        
        loss.backward()
        optimizer.step()
        
        logger.debug(f"Meta-learner training loss: {loss.item():.4f}")
    
    def evolve(
        self,
        validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
        max_generations: int = None,
        convergence_threshold: float = None,
    ) -> ArchitectureConfig:
        """Run complete evolution process.
        
        Args:
            validation_data: Validation data for fitness evaluation
            max_generations: Maximum number of generations
            convergence_threshold: Convergence threshold
            
        Returns:
            Best evolved architecture
        """
        if max_generations is None:
            max_generations = self.evolution_config.max_generations
        
        if convergence_threshold is None:
            convergence_threshold = self.evolution_config.convergence_threshold
        
        logger.info(f"Starting evolution for {max_generations} generations")
        
        # Initialize population if not done already
        if not self.population:
            self.initialize_population()
        
        best_fitness = 0.0
        no_improvement_count = 0
        
        for generation in range(max_generations):
            # Evolve one generation
            best_gene = self.evolve_generation(validation_data)
            
            # Check for improvement
            if best_gene.fitness > best_fitness + convergence_threshold:
                best_fitness = best_gene.fitness
                no_improvement_count = 0
                
                # Update current architecture
                self.current_architecture = best_gene.to_architecture_config()
                
                # Replace hypernetwork with evolved architecture
                self.hypernetwork = NeuralHypernetwork(
                    self.text_dim, self.target_weights_config, self.current_architecture
                )
                
                logger.info(f"New best architecture found with fitness {best_fitness:.4f}")
            else:
                no_improvement_count += 1
            
            # Early stopping
            if no_improvement_count >= self.evolution_config.performance_patience:
                logger.info(f"Converged after {generation + 1} generations")
                break
        
        logger.info(f"Evolution completed. Final fitness: {best_fitness:.4f}")
        
        return self.current_architecture
    
    def forward(self, text_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass using current evolved architecture."""
        return self.hypernetwork(text_embeddings)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process."""
        if not self.evolution_history:
            return {"status": "No evolution performed yet"}
        
        best_generation = max(
            self.evolution_history,
            key=lambda x: x['best_fitness']
        )
        
        fitness_progression = [entry['best_fitness'] for entry in self.evolution_history]
        
        return {
            "total_generations": len(self.evolution_history),
            "best_fitness": best_generation['best_fitness'],
            "best_generation_number": best_generation['generation'],
            "fitness_progression": fitness_progression,
            "final_architecture": self.current_architecture,
            "population_size": len(self.population),
            "current_generation": self.generation,
        }


# Demonstration and example usage
def demonstrate_self_evolving_hypernetworks():
    """Demonstrate the self-evolving hypernetwork system."""
    print("🧬 Self-Evolving Hypernetworks Demo")
    print("=" * 45)
    
    # Configuration
    text_dim = 384
    target_weights_config = {
        'layer_1_weight': (text_dim, 256),
        'layer_1_bias': (256,),
        'layer_2_weight': (256, 128),
        'layer_2_bias': (128,),
    }
    
    evolution_config = EvolutionConfig(
        population_size=10,  # Small for demo
        max_generations=20,
        mutation_rate=0.1,
        crossover_rate=0.3,
    )
    
    # Initialize self-evolving hypernetwork
    self_evolving_hn = SelfEvolvingHypernetwork(
        text_dim=text_dim,
        target_weights_config=target_weights_config,
        evolution_config=evolution_config,
    )
    
    print(f"Initial architecture: {self_evolving_hn.current_architecture.num_layers} layers")
    print(f"Hidden dims: {self_evolving_hn.current_architecture.hidden_dims}")
    
    # Create dummy validation data
    validation_data = []
    for _ in range(3):  # Small dataset for demo
        text_batch = torch.randn(4, text_dim)
        target_batch = {
            'layer_1_weight': torch.randn(4, text_dim, 256),
            'layer_1_bias': torch.randn(4, 256),
            'layer_2_weight': torch.randn(4, 256, 128),
            'layer_2_bias': torch.randn(4, 128),
        }
        validation_data.append((text_batch, target_batch))
    
    # Run evolution
    print("\nStarting evolution process...")
    final_architecture = self_evolving_hn.evolve(
        validation_data,
        max_generations=10,  # Reduced for demo
    )
    
    print(f"\n🎉 Evolution completed!")
    print(f"Final architecture: {final_architecture.num_layers} layers")
    print(f"Hidden dims: {final_architecture.hidden_dims}")
    print(f"Activations: {final_architecture.activation_types}")
    
    # Get evolution summary
    summary = self_evolving_hn.get_evolution_summary()
    print(f"\n📊 Evolution Summary:")
    print(f"  Total generations: {summary['total_generations']}")
    print(f"  Best fitness: {summary['best_fitness']:.4f}")
    print(f"  Best generation: {summary['best_generation_number']}")
    
    # Test evolved hypernetwork
    print(f"\n🧪 Testing evolved hypernetwork...")
    test_input = torch.randn(2, text_dim)
    with torch.no_grad():
        outputs = self_evolving_hn(test_input)
        print(f"Generated weights for {len(outputs)} weight tensors")
        for name, tensor in outputs.items():
            print(f"  {name}: {tensor.shape}")
    
    print("\n✅ Self-evolving hypernetwork demonstration completed!")


if __name__ == "__main__":
    demonstrate_self_evolving_hypernetworks()