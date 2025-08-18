"""Adaptive optimization and auto-tuning for Graph Hypernetwork Forge."""

import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for basic operations
    class np:
        @staticmethod
        def array(data): return data
        @staticmethod
        def mean(data): return sum(data) / len(data)
        @staticmethod
        def std(data): 
            mean = sum(data) / len(data)
            return math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
        @staticmethod
        def percentile(data, q): return sorted(data)[int(len(data) * q / 100)]

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Mock scipy optimize
    class optimize:
        @staticmethod
        def minimize(func, x0, **kwargs):
            class Result:
                def __init__(self):
                    self.x = x0
                    self.fun = func(x0) if callable(func) else 0
                    self.success = True
            return Result()

try:
    from .logging_utils import get_logger
    from .exceptions import GraphHypernetworkError, ModelError
    from .production_monitoring import get_metrics_collector, get_performance_profiler
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name): return logging.getLogger(name)
    class GraphHypernetworkError(Exception): pass
    class ModelError(Exception): pass
    def get_metrics_collector(): 
        class MockCollector:
            def record_model_performance(self, *args): pass
        return MockCollector()
    def get_performance_profiler():
        class MockProfiler:
            def profile_operation(self, *args, **kwargs):
                class DummyContext:
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                return DummyContext()
        return MockProfiler()
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    GRID_SEARCH = "grid_search"                 # Exhaustive grid search
    RANDOM_SEARCH = "random_search"             # Random parameter sampling
    BAYESIAN = "bayesian"                       # Bayesian optimization
    GENETIC = "genetic"                         # Genetic algorithm
    SIMULATED_ANNEALING = "simulated_annealing" # Simulated annealing
    ADAPTIVE_LEARNING = "adaptive_learning"     # Online adaptive learning


class ParameterType(Enum):
    """Types of parameters for optimization."""
    CONTINUOUS = "continuous"    # Float values in range
    DISCRETE = "discrete"        # Integer values in range
    CATEGORICAL = "categorical"  # Categorical choices
    BOOLEAN = "boolean"          # True/False values


@dataclass
class ParameterSpec:
    """Parameter specification for optimization."""
    name: str
    param_type: ParameterType
    default_value: Any
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    choices: Optional[List[Any]] = None
    description: str = ""
    importance: float = 1.0  # 0.0-1.0, higher means more important
    
    def __post_init__(self):
        """Validate parameter specification."""
        if self.param_type in [ParameterType.CONTINUOUS, ParameterType.DISCRETE]:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"min_value and max_value required for {self.param_type}")
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError("choices required for categorical parameters")


@dataclass
class OptimizationResult:
    """Result from parameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    total_evaluations: int
    optimization_time: float
    convergence_reached: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceMetrics:
    """Comprehensive performance metrics collection and analysis."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize performance metrics.
        
        Args:
            window_size: Size of rolling window for metrics
        """
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.metric_stats = {}
        
        logger.info(f"Performance metrics initialized with window_size={window_size}")
    
    def record_performance(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        timestamp: float = None
    ):
        """Record performance measurement.
        
        Args:
            params: Parameters used for this measurement
            metrics: Performance metrics achieved
            timestamp: Optional timestamp
        """
        record = {
            'params': params.copy(),
            'metrics': metrics.copy(),
            'timestamp': timestamp or time.time()
        }
        
        self.metrics_history.append(record)
        self._update_stats()
    
    def get_performance_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate composite performance score.
        
        Args:
            weights: Weights for different metrics (higher is better)
            
        Returns:
            Weighted performance score
        """
        if not self.metrics_history:
            return 0.0
        
        latest = self.metrics_history[-1]['metrics']
        
        if weights is None:
            # Default weights favoring throughput and low latency
            weights = {
                'throughput': 0.4,
                'latency': -0.3,  # Negative because lower is better
                'memory_efficiency': 0.2,
                'accuracy': 0.1
            }
        
        score = 0.0
        for metric, value in latest.items():
            weight = weights.get(metric, 0.0)
            if metric in ['latency', 'memory_usage', 'error_rate']:
                # Lower is better for these metrics
                score -= weight * value
            else:
                # Higher is better for other metrics
                score += weight * value
        
        return score
    
    def _update_stats(self):
        """Update running statistics."""
        if len(self.metrics_history) < 2:
            return
        
        # Calculate statistics for each metric
        metrics_by_name = defaultdict(list)
        for record in self.metrics_history:
            for name, value in record['metrics'].items():
                metrics_by_name[name].append(value)
        
        self.metric_stats = {}
        for name, values in metrics_by_name.items():
            if NUMPY_AVAILABLE:
                self.metric_stats[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': min(values),
                    'max': max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                }
            else:
                # Basic statistics without numpy
                mean_val = sum(values) / len(values)
                self.metric_stats[name] = {
                    'mean': mean_val,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
    
    def get_pareto_frontier(self, metrics: List[str]) -> List[Dict[str, Any]]:
        """Find Pareto-optimal configurations.
        
        Args:
            metrics: List of metrics to consider (assumes higher is better)
            
        Returns:
            List of Pareto-optimal configurations
        """
        if len(metrics) < 2 or not self.metrics_history:
            return []
        
        pareto_configs = []
        
        for i, record_i in enumerate(self.metrics_history):
            is_dominated = False
            
            for j, record_j in enumerate(self.metrics_history):
                if i == j:
                    continue
                
                # Check if record_j dominates record_i
                dominates = True
                strictly_better_in_one = False
                
                for metric in metrics:
                    val_i = record_i['metrics'].get(metric, 0)
                    val_j = record_j['metrics'].get(metric, 0)
                    
                    if val_j < val_i:
                        dominates = False
                        break
                    elif val_j > val_i:
                        strictly_better_in_one = True
                
                if dominates and strictly_better_in_one:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_configs.append(record_i)
        
        return pareto_configs


class AdaptiveHyperparameterOptimizer:
    """Advanced hyperparameter optimization with multiple strategies."""
    
    def __init__(
        self,
        parameter_specs: List[ParameterSpec],
        strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE_LEARNING,
        max_evaluations: int = 100,
        early_stopping_patience: int = 20,
        optimization_timeout: float = 3600.0,  # 1 hour
    ):
        """Initialize hyperparameter optimizer.
        
        Args:
            parameter_specs: List of parameter specifications
            strategy: Optimization strategy to use
            max_evaluations: Maximum number of evaluations
            early_stopping_patience: Early stopping patience
            optimization_timeout: Maximum optimization time in seconds
        """
        self.parameter_specs = {spec.name: spec for spec in parameter_specs}
        self.strategy = strategy
        self.max_evaluations = max_evaluations
        self.early_stopping_patience = early_stopping_patience
        self.optimization_timeout = optimization_timeout
        
        # Optimization state
        self.evaluation_history = []
        self.best_score = float('-inf')
        self.best_params = None
        self.evaluations_since_improvement = 0
        
        # Strategy-specific state
        self.strategy_state = {}
        self._initialize_strategy()
        
        logger.info(f"Hyperparameter optimizer initialized with {strategy.value} strategy")
    
    def _initialize_strategy(self):
        """Initialize strategy-specific state."""
        if self.strategy == OptimizationStrategy.GENETIC:
            self.strategy_state = {
                'population_size': min(20, max(4, self.max_evaluations // 10)),
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'population': [],
                'generation': 0
            }
        elif self.strategy == OptimizationStrategy.SIMULATED_ANNEALING:
            self.strategy_state = {
                'temperature': 1.0,
                'cooling_rate': 0.95,
                'current_params': self._generate_random_params(),
                'current_score': None
            }
        elif self.strategy == OptimizationStrategy.ADAPTIVE_LEARNING:
            self.strategy_state = {
                'learning_rates': {name: 0.1 for name in self.parameter_specs},
                'momentum': {name: 0.0 for name in self.parameter_specs},
                'exploration_rate': 0.3,
                'exploitation_bonus': 0.1
            }
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        initial_params: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """Optimize hyperparameters using specified strategy.
        
        Args:
            objective_function: Function to optimize (higher is better)
            initial_params: Optional initial parameters
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        self.evaluation_history = []
        self.best_score = float('-inf')
        self.best_params = None
        self.evaluations_since_improvement = 0
        
        logger.info(f"Starting optimization with {self.strategy.value}")
        
        try:
            if self.strategy == OptimizationStrategy.GRID_SEARCH:
                result = self._optimize_grid_search(objective_function)
            elif self.strategy == OptimizationStrategy.RANDOM_SEARCH:
                result = self._optimize_random_search(objective_function)
            elif self.strategy == OptimizationStrategy.BAYESIAN:
                result = self._optimize_bayesian(objective_function)
            elif self.strategy == OptimizationStrategy.GENETIC:
                result = self._optimize_genetic(objective_function)
            elif self.strategy == OptimizationStrategy.SIMULATED_ANNEALING:
                result = self._optimize_simulated_annealing(objective_function)
            elif self.strategy == OptimizationStrategy.ADAPTIVE_LEARNING:
                result = self._optimize_adaptive_learning(objective_function)
            else:
                raise ValueError(f"Unknown optimization strategy: {self.strategy}")
            
            optimization_time = time.time() - start_time
            
            return OptimizationResult(
                best_params=self.best_params or {},
                best_score=self.best_score,
                optimization_history=self.evaluation_history,
                total_evaluations=len(self.evaluation_history),
                optimization_time=optimization_time,
                convergence_reached=self.evaluations_since_improvement < self.early_stopping_patience,
                metadata={
                    'strategy': self.strategy.value,
                    'strategy_state': self.strategy_state
                }
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise ModelError("HyperparameterOptimizer", "optimization", str(e))
    
    def _optimize_random_search(self, objective_function: Callable) -> None:
        """Optimize using random search strategy."""
        for evaluation in range(self.max_evaluations):
            if self._should_stop():
                break
            
            params = self._generate_random_params()
            score = self._evaluate_params(params, objective_function)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.evaluations_since_improvement = 0
            else:
                self.evaluations_since_improvement += 1
    
    def _optimize_adaptive_learning(self, objective_function: Callable) -> None:
        """Optimize using adaptive learning strategy."""
        state = self.strategy_state
        current_params = self._generate_random_params()
        
        for evaluation in range(self.max_evaluations):
            if self._should_stop():
                break
            
            # Exploration vs exploitation
            if random.random() < state['exploration_rate']:
                # Exploration: try random parameters
                test_params = self._generate_random_params()
            else:
                # Exploitation: improve current best parameters
                test_params = self._improve_params(current_params, state)
            
            score = self._evaluate_params(test_params, objective_function)
            
            # Update learning rates based on performance
            if score > self.best_score:
                self.best_score = score
                self.best_params = test_params.copy()
                self.evaluations_since_improvement = 0
                current_params = test_params.copy()
                
                # Increase learning rates for successful parameters
                self._update_learning_rates(test_params, increase=True)
            else:
                self.evaluations_since_improvement += 1
                
                # Decrease learning rates for unsuccessful parameters  
                self._update_learning_rates(test_params, increase=False)
            
            # Decay exploration rate over time
            state['exploration_rate'] *= 0.995
    
    def _optimize_genetic(self, objective_function: Callable) -> None:
        """Optimize using genetic algorithm."""
        state = self.strategy_state
        
        # Initialize population
        if not state['population']:
            state['population'] = [
                {
                    'params': self._generate_random_params(),
                    'score': None,
                    'age': 0
                }
                for _ in range(state['population_size'])
            ]
        
        # Evaluate initial population
        for individual in state['population']:
            if individual['score'] is None:
                individual['score'] = self._evaluate_params(
                    individual['params'], objective_function
                )
                
                if individual['score'] > self.best_score:
                    self.best_score = individual['score']
                    self.best_params = individual['params'].copy()
                    self.evaluations_since_improvement = 0
        
        # Evolution loop
        while len(self.evaluation_history) < self.max_evaluations and not self._should_stop():
            # Selection: tournament selection
            parents = self._tournament_selection(state['population'], 2)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i]['params'], parents[i+1]['params'])
                    
                    if random.random() < state['mutation_rate']:
                        child1 = self._mutate(child1)
                    if random.random() < state['mutation_rate']:
                        child2 = self._mutate(child2)
                    
                    offspring.extend([
                        {'params': child1, 'score': None, 'age': 0},
                        {'params': child2, 'score': None, 'age': 0}
                    ])
            
            # Evaluate offspring
            for individual in offspring:
                if len(self.evaluation_history) >= self.max_evaluations:
                    break
                
                individual['score'] = self._evaluate_params(
                    individual['params'], objective_function
                )
                
                if individual['score'] > self.best_score:
                    self.best_score = individual['score']
                    self.best_params = individual['params'].copy()
                    self.evaluations_since_improvement = 0
            
            # Replacement: keep best individuals
            all_individuals = state['population'] + offspring
            all_individuals.sort(key=lambda x: x['score'], reverse=True)
            state['population'] = all_individuals[:state['population_size']]
            
            # Age individuals
            for individual in state['population']:
                individual['age'] += 1
            
            state['generation'] += 1
    
    def _optimize_simulated_annealing(self, objective_function: Callable) -> None:
        """Optimize using simulated annealing."""
        state = self.strategy_state
        current_params = state['current_params']
        
        if state['current_score'] is None:
            state['current_score'] = self._evaluate_params(current_params, objective_function)
            
            if state['current_score'] > self.best_score:
                self.best_score = state['current_score']
                self.best_params = current_params.copy()
        
        for evaluation in range(1, self.max_evaluations):
            if self._should_stop():
                break
            
            # Generate neighbor
            candidate_params = self._generate_neighbor(current_params, state['temperature'])
            candidate_score = self._evaluate_params(candidate_params, objective_function)
            
            # Acceptance probability
            if candidate_score > state['current_score']:
                # Accept better solution
                current_params = candidate_params
                state['current_score'] = candidate_score
                
                if candidate_score > self.best_score:
                    self.best_score = candidate_score
                    self.best_params = candidate_params.copy()
                    self.evaluations_since_improvement = 0
            else:
                # Accept worse solution with probability
                delta = candidate_score - state['current_score']
                probability = math.exp(delta / state['temperature'])
                
                if random.random() < probability:
                    current_params = candidate_params
                    state['current_score'] = candidate_score
                else:
                    self.evaluations_since_improvement += 1
            
            # Cool down
            state['temperature'] *= state['cooling_rate']
            state['current_params'] = current_params
    
    def _optimize_bayesian(self, objective_function: Callable) -> None:
        """Optimize using Bayesian optimization (simplified)."""
        # Simplified Bayesian optimization using random search with exploitation
        # In practice, would use libraries like scikit-optimize or GPyOpt
        
        # Start with random exploration
        for evaluation in range(min(10, self.max_evaluations // 4)):
            params = self._generate_random_params()
            score = self._evaluate_params(params, objective_function)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.evaluations_since_improvement = 0
        
        # Exploitation phase: search around best parameters
        for evaluation in range(len(self.evaluation_history), self.max_evaluations):
            if self._should_stop():
                break
            
            if self.best_params and random.random() < 0.7:
                # Exploit: search near best parameters
                params = self._generate_neighbor(self.best_params, temperature=0.1)
            else:
                # Explore: random parameters
                params = self._generate_random_params()
            
            score = self._evaluate_params(params, objective_function)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.evaluations_since_improvement = 0
            else:
                self.evaluations_since_improvement += 1
    
    def _optimize_grid_search(self, objective_function: Callable) -> None:
        """Optimize using grid search."""
        # Generate grid points
        grid_points = self._generate_grid_points()
        
        for params in grid_points:
            if len(self.evaluation_history) >= self.max_evaluations:
                break
            if self._should_stop():
                break
            
            score = self._evaluate_params(params, objective_function)
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = params.copy()
                self.evaluations_since_improvement = 0
            else:
                self.evaluations_since_improvement += 1
    
    def _generate_grid_points(self) -> List[Dict[str, Any]]:
        """Generate grid search points."""
        # Simplified grid generation - in practice would be more sophisticated
        grid_points = []
        
        # For each parameter, create 5-10 grid points
        param_grids = {}
        for name, spec in self.parameter_specs.items():
            if spec.param_type == ParameterType.CONTINUOUS:
                num_points = min(10, int(self.max_evaluations ** (1/len(self.parameter_specs))))
                param_grids[name] = [
                    spec.min_value + i * (spec.max_value - spec.min_value) / (num_points - 1)
                    for i in range(num_points)
                ]
            elif spec.param_type == ParameterType.DISCRETE:
                param_grids[name] = list(range(spec.min_value, spec.max_value + 1))
            elif spec.param_type == ParameterType.CATEGORICAL:
                param_grids[name] = spec.choices
            elif spec.param_type == ParameterType.BOOLEAN:
                param_grids[name] = [True, False]
        
        # Generate Cartesian product (simplified)
        import itertools
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            grid_points.append(params)
            
            if len(grid_points) >= self.max_evaluations:
                break
        
        return grid_points
    
    def _generate_random_params(self) -> Dict[str, Any]:
        """Generate random parameter values."""
        params = {}
        
        for name, spec in self.parameter_specs.items():
            if spec.param_type == ParameterType.CONTINUOUS:
                params[name] = random.uniform(spec.min_value, spec.max_value)
            elif spec.param_type == ParameterType.DISCRETE:
                params[name] = random.randint(spec.min_value, spec.max_value)
            elif spec.param_type == ParameterType.CATEGORICAL:
                params[name] = random.choice(spec.choices)
            elif spec.param_type == ParameterType.BOOLEAN:
                params[name] = random.choice([True, False])
            else:
                params[name] = spec.default_value
        
        return params
    
    def _generate_neighbor(self, params: Dict[str, Any], temperature: float = 0.1) -> Dict[str, Any]:
        """Generate neighbor parameters for local search."""
        neighbor = params.copy()
        
        # Modify 1-2 parameters
        num_changes = random.randint(1, min(2, len(params)))
        changed_params = random.sample(list(params.keys()), num_changes)
        
        for name in changed_params:
            spec = self.parameter_specs[name]
            
            if spec.param_type == ParameterType.CONTINUOUS:
                # Gaussian perturbation
                range_size = spec.max_value - spec.min_value
                std_dev = temperature * range_size
                new_value = random.gauss(params[name], std_dev)
                neighbor[name] = max(spec.min_value, min(spec.max_value, new_value))
                
            elif spec.param_type == ParameterType.DISCRETE:
                # Random walk
                range_size = spec.max_value - spec.min_value
                max_step = max(1, int(temperature * range_size))
                step = random.randint(-max_step, max_step)
                new_value = params[name] + step
                neighbor[name] = max(spec.min_value, min(spec.max_value, new_value))
                
            elif spec.param_type == ParameterType.CATEGORICAL:
                if random.random() < temperature:
                    neighbor[name] = random.choice(spec.choices)
                    
            elif spec.param_type == ParameterType.BOOLEAN:
                if random.random() < temperature:
                    neighbor[name] = not params[name]
        
        return neighbor
    
    def _improve_params(self, params: Dict[str, Any], state: Dict) -> Dict[str, Any]:
        """Improve parameters using adaptive learning rates."""
        improved = params.copy()
        
        for name, spec in self.parameter_specs.items():
            learning_rate = state['learning_rates'][name]
            momentum = state['momentum'][name]
            
            if spec.param_type == ParameterType.CONTINUOUS:
                # Gradient-like update with momentum
                range_size = spec.max_value - spec.min_value
                update = learning_rate * range_size * (random.random() - 0.5)
                update = momentum * state.get(f'{name}_prev_update', 0) + (1 - momentum) * update
                
                new_value = params[name] + update
                improved[name] = max(spec.min_value, min(spec.max_value, new_value))
                state[f'{name}_prev_update'] = update
                
            elif spec.param_type == ParameterType.DISCRETE:
                # Discrete update
                if random.random() < learning_rate:
                    direction = 1 if random.random() < 0.5 else -1
                    new_value = params[name] + direction
                    improved[name] = max(spec.min_value, min(spec.max_value, new_value))
        
        return improved
    
    def _update_learning_rates(self, params: Dict[str, Any], increase: bool):
        """Update learning rates based on performance."""
        state = self.strategy_state
        factor = 1.1 if increase else 0.95
        
        for name in params:
            if name in state['learning_rates']:
                new_rate = state['learning_rates'][name] * factor
                state['learning_rates'][name] = max(0.01, min(0.5, new_rate))
    
    def _evaluate_params(self, params: Dict[str, Any], objective_function: Callable) -> float:
        """Evaluate parameters and record results."""
        try:
            score = objective_function(params)
            
            self.evaluation_history.append({
                'params': params.copy(),
                'score': score,
                'timestamp': time.time(),
                'evaluation_index': len(self.evaluation_history)
            })
            
            return score
            
        except Exception as e:
            logger.warning(f"Evaluation failed for params {params}: {e}")
            return float('-inf')
    
    def _should_stop(self) -> bool:
        """Check if optimization should stop."""
        if time.time() > getattr(self, '_start_time', 0) + self.optimization_timeout:
            return True
        
        return self.evaluations_since_improvement >= self.early_stopping_patience
    
    def _tournament_selection(self, population: List[Dict], tournament_size: int) -> List[Dict]:
        """Tournament selection for genetic algorithm."""
        selected = []
        
        for _ in range(len(population)):
            tournament = random.sample(population, min(tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x['score'])
            selected.append(winner)
        
        return selected
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover operation for genetic algorithm."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Uniform crossover
        for name in parent1:
            if random.random() < 0.5:
                child1[name] = parent2[name]
                child2[name] = parent1[name]
        
        return child1, child2
    
    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mutation operation for genetic algorithm."""
        mutated = params.copy()
        
        # Mutate each parameter with small probability
        for name, spec in self.parameter_specs.items():
            if random.random() < 0.1:  # Mutation probability
                if spec.param_type == ParameterType.CONTINUOUS:
                    range_size = spec.max_value - spec.min_value
                    mutation = random.gauss(0, 0.1 * range_size)
                    new_value = params[name] + mutation
                    mutated[name] = max(spec.min_value, min(spec.max_value, new_value))
                    
                elif spec.param_type == ParameterType.DISCRETE:
                    mutated[name] = random.randint(spec.min_value, spec.max_value)
                    
                elif spec.param_type == ParameterType.CATEGORICAL:
                    mutated[name] = random.choice(spec.choices)
                    
                elif spec.param_type == ParameterType.BOOLEAN:
                    mutated[name] = not params[name]
        
        return mutated


class AutoTuner:
    """Automated system tuning and optimization."""
    
    def __init__(self):
        """Initialize auto-tuner."""
        self.performance_metrics = PerformanceMetrics()
        self.optimization_history = []
        self.current_config = {}
        
        logger.info("Auto-tuner initialized")
    
    def auto_tune_model(
        self,
        model_factory: Callable,
        parameter_specs: List[ParameterSpec],
        evaluation_dataset: Any,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE_LEARNING
    ) -> Dict[str, Any]:
        """Automatically tune model hyperparameters.
        
        Args:
            model_factory: Function that creates model with given parameters
            parameter_specs: Parameters to optimize
            evaluation_dataset: Dataset for evaluation
            optimization_strategy: Optimization strategy to use
            
        Returns:
            Best configuration found
        """
        optimizer = AdaptiveHyperparameterOptimizer(
            parameter_specs=parameter_specs,
            strategy=optimization_strategy
        )
        
        def objective_function(params: Dict[str, Any]) -> float:
            """Objective function for hyperparameter optimization."""
            try:
                # Create model with parameters
                model = model_factory(**params)
                
                # Evaluate model performance
                metrics = self._evaluate_model(model, evaluation_dataset)
                
                # Record performance
                self.performance_metrics.record_performance(params, metrics)
                
                # Return composite score
                return self.performance_metrics.get_performance_score()
                
            except Exception as e:
                logger.warning(f"Model evaluation failed: {e}")
                return float('-inf')
        
        # Run optimization
        result = optimizer.optimize(objective_function)
        
        # Store results
        self.optimization_history.append(result)
        self.current_config = result.best_params
        
        logger.info(f"Auto-tuning completed. Best score: {result.best_score:.4f}")
        return result.best_params
    
    def _evaluate_model(self, model: Any, dataset: Any) -> Dict[str, float]:
        """Evaluate model performance on dataset."""
        # This would be implemented based on specific model and dataset types
        # For now, return mock metrics
        start_time = time.time()
        
        # Simulate model evaluation
        time.sleep(0.1)  # Simulate processing time
        
        processing_time = time.time() - start_time
        
        return {
            'latency': processing_time,
            'throughput': 1.0 / processing_time if processing_time > 0 else float('inf'),
            'accuracy': random.uniform(0.8, 0.95),  # Mock accuracy
            'memory_efficiency': random.uniform(0.7, 0.9),  # Mock memory efficiency
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if not self.optimization_history:
            return {'message': 'No optimization runs completed'}
        
        latest_result = self.optimization_history[-1]
        
        return {
            'current_config': self.current_config,
            'best_score': latest_result.best_score,
            'total_evaluations': latest_result.total_evaluations,
            'optimization_time': latest_result.optimization_time,
            'convergence_reached': latest_result.convergence_reached,
            'optimization_runs': len(self.optimization_history),
            'performance_stats': self.performance_metrics.metric_stats,
            'pareto_frontier': self.performance_metrics.get_pareto_frontier(['throughput', 'accuracy'])
        }


# Convenience functions
def create_parameter_specs_from_config(config: Dict[str, Any]) -> List[ParameterSpec]:
    """Create parameter specifications from configuration dictionary.
    
    Args:
        config: Configuration dictionary with parameter definitions
        
    Returns:
        List of parameter specifications
    """
    specs = []
    
    for name, param_config in config.items():
        if isinstance(param_config, dict):
            spec = ParameterSpec(
                name=name,
                param_type=ParameterType(param_config.get('type', 'continuous')),
                default_value=param_config.get('default'),
                min_value=param_config.get('min'),
                max_value=param_config.get('max'),
                choices=param_config.get('choices'),
                description=param_config.get('description', ''),
                importance=param_config.get('importance', 1.0)
            )
            specs.append(spec)
    
    return specs


# Global auto-tuner instance
_global_auto_tuner = None

def get_auto_tuner() -> AutoTuner:
    """Get global auto-tuner instance."""
    global _global_auto_tuner
    if _global_auto_tuner is None:
        _global_auto_tuner = AutoTuner()
    return _global_auto_tuner