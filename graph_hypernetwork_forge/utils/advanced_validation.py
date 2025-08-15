"""Advanced Validation Framework for Graph Hypernetwork Operations.

This module provides comprehensive validation, error handling, and robustness
mechanisms for all graph hypernetwork components.

Features:
- Multi-level validation with cascading checks
- Semantic graph validation and consistency checking
- Performance-aware validation with resource monitoring
- Adaptive validation based on context and confidence
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import logging
import warnings
from dataclasses import dataclass
from enum import Enum
import time
import psutil
import gc

# Enhanced utilities
try:
    from .logging_utils import get_logger
    from .exceptions import ValidationError, ModelError, DataError
    from .memory_utils import memory_management, get_memory_usage
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)
    class ValidationError(Exception): pass
    class ModelError(Exception): pass
    class DataError(Exception): pass
    def memory_management(*args, **kwargs):
        class DummyContext:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return DummyContext()
    def get_memory_usage(): return 0.0
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


class ValidationContext(Enum):
    """Validation context types."""
    TRAINING = "training"
    INFERENCE = "inference"
    RESEARCH = "research"
    PRODUCTION = "production"


@dataclass
class ValidationResult:
    """Validation result with detailed information."""
    is_valid: bool
    confidence: float
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    execution_time: float
    memory_usage: float


@dataclass
class ValidationConfig:
    """Configuration for validation operations."""
    level: ValidationLevel = ValidationLevel.STANDARD
    context: ValidationContext = ValidationContext.TRAINING
    enable_performance_checks: bool = True
    enable_semantic_checks: bool = True
    enable_statistical_checks: bool = True
    max_execution_time: float = 30.0
    max_memory_usage_gb: float = 8.0
    confidence_threshold: float = 0.95
    nan_tolerance: float = 1e-6
    inf_tolerance: float = 1e-6


class GraphValidator:
    """Comprehensive graph structure and content validator."""
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize graph validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.logger = get_logger(self.__class__.__name__)
        
        # Validation statistics
        self.validation_history = []
        self.error_patterns = {}
        
        self.logger.info(f"GraphValidator initialized with level={self.config.level.value}")
    
    def validate_node_features(
        self,
        node_features: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        feature_names: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate node features with comprehensive checks.
        
        Args:
            node_features: Node feature tensor
            expected_shape: Expected tensor shape
            feature_names: Optional feature names for detailed reporting
            
        Returns:
            Validation result
        """
        start_time = time.time()
        start_memory = get_memory_usage()
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Basic tensor validation
            if not isinstance(node_features, torch.Tensor):
                errors.append(f"Expected torch.Tensor, got {type(node_features)}")
                return ValidationResult(
                    is_valid=False, confidence=0.0, errors=errors, warnings=warnings,
                    metrics=metrics, execution_time=time.time() - start_time,
                    memory_usage=get_memory_usage() - start_memory
                )
            
            # Shape validation
            if expected_shape and node_features.shape != expected_shape:
                errors.append(f"Shape mismatch: expected {expected_shape}, got {node_features.shape}")
            
            # Dimension validation
            if node_features.dim() < 2:
                errors.append(f"Node features must be at least 2D, got {node_features.dim()}D")
            
            # Statistical validation
            if self.config.enable_statistical_checks:
                # NaN/Inf detection
                nan_count = torch.isnan(node_features).sum().item()
                inf_count = torch.isinf(node_features).sum().item()
                
                if nan_count > 0:
                    errors.append(f"Found {nan_count} NaN values in node features")
                
                if inf_count > 0:
                    errors.append(f"Found {inf_count} infinite values in node features")
                
                # Statistical properties
                mean_val = node_features.mean().item()
                std_val = node_features.std().item()
                min_val = node_features.min().item()
                max_val = node_features.max().item()
                
                metrics.update({
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'nan_count': nan_count,
                    'inf_count': inf_count
                })
                
                # Range validation
                if abs(mean_val) > 1000:
                    warnings.append(f"Large mean value detected: {mean_val}")
                
                if std_val > 100:
                    warnings.append(f"Large standard deviation detected: {std_val}")
                
                if max_val - min_val > 10000:
                    warnings.append(f"Large value range detected: [{min_val}, {max_val}]")
            
            # Feature-specific validation
            if feature_names:
                if len(feature_names) != node_features.size(-1):
                    warnings.append(
                        f"Feature name count ({len(feature_names)}) doesn't match "
                        f"feature dimension ({node_features.size(-1)})"
                    )
                
                metrics['feature_names'] = feature_names
            
            # Memory validation
            if self.config.enable_performance_checks:
                tensor_memory_gb = node_features.numel() * 4 / (1024**3)  # Assume float32
                metrics['tensor_memory_gb'] = tensor_memory_gb
                
                if tensor_memory_gb > self.config.max_memory_usage_gb:
                    warnings.append(f"Large tensor memory usage: {tensor_memory_gb:.2f}GB")
            
            # Semantic validation based on context
            if self.config.enable_semantic_checks:
                self._validate_semantic_properties(node_features, metrics, warnings)
            
            execution_time = time.time() - start_time
            memory_usage = get_memory_usage() - start_memory
            
            # Performance checks
            if execution_time > self.config.max_execution_time:
                warnings.append(f"Validation took {execution_time:.2f}s (limit: {self.config.max_execution_time}s)")
            
            # Compute confidence
            confidence = self._compute_confidence(errors, warnings, metrics)
            
            is_valid = len(errors) == 0 and confidence >= self.config.confidence_threshold
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=execution_time,
                memory_usage=memory_usage
            )
            
            self._update_validation_history(result)
            
            return result
            
        except Exception as e:
            errors.append(f"Validation failed with exception: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=time.time() - start_time,
                memory_usage=get_memory_usage() - start_memory
            )
    
    def validate_edge_structure(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        allow_self_loops: bool = True,
        allow_duplicate_edges: bool = True
    ) -> ValidationResult:
        """Validate edge structure and connectivity.
        
        Args:
            edge_index: Edge index tensor [2, num_edges]
            num_nodes: Number of nodes in the graph
            allow_self_loops: Whether self-loops are allowed
            allow_duplicate_edges: Whether duplicate edges are allowed
            
        Returns:
            Validation result
        """
        start_time = time.time()
        start_memory = get_memory_usage()
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Basic structure validation
            if not isinstance(edge_index, torch.Tensor):
                errors.append(f"Expected torch.Tensor, got {type(edge_index)}")
                return self._create_failed_result(errors, warnings, metrics, start_time, start_memory)
            
            if edge_index.dim() != 2:
                errors.append(f"Edge index must be 2D, got {edge_index.dim()}D")
                return self._create_failed_result(errors, warnings, metrics, start_time, start_memory)
            
            if edge_index.size(0) != 2:
                errors.append(f"Edge index first dimension must be 2, got {edge_index.size(0)}")
                return self._create_failed_result(errors, warnings, metrics, start_time, start_memory)
            
            num_edges = edge_index.size(1)
            metrics['num_edges'] = num_edges
            
            if num_edges == 0:
                warnings.append("Graph has no edges")
                return self._create_successful_result(warnings, metrics, start_time, start_memory)
            
            # Node index validation
            max_node_idx = edge_index.max().item()
            min_node_idx = edge_index.min().item()
            
            if min_node_idx < 0:
                errors.append(f"Negative node index found: {min_node_idx}")
            
            if max_node_idx >= num_nodes:
                errors.append(f"Node index {max_node_idx} exceeds graph size {num_nodes}")
            
            metrics.update({
                'max_node_idx': max_node_idx,
                'min_node_idx': min_node_idx,
                'num_nodes': num_nodes
            })
            
            # Self-loop validation
            if not allow_self_loops:
                self_loops = (edge_index[0] == edge_index[1]).sum().item()
                if self_loops > 0:
                    errors.append(f"Found {self_loops} self-loops (not allowed)")
                metrics['self_loops'] = self_loops
            
            # Duplicate edge validation
            if not allow_duplicate_edges:
                unique_edges = torch.unique(edge_index, dim=1)
                duplicate_count = num_edges - unique_edges.size(1)
                if duplicate_count > 0:
                    errors.append(f"Found {duplicate_count} duplicate edges (not allowed)")
                metrics['duplicate_edges'] = duplicate_count
            
            # Graph connectivity analysis
            if self.config.enable_semantic_checks:
                connectivity_metrics = self._analyze_graph_connectivity(edge_index, num_nodes)
                metrics.update(connectivity_metrics)
                
                # Check for connectivity issues
                if connectivity_metrics.get('num_components', 1) > 1:
                    warnings.append(f"Graph has {connectivity_metrics['num_components']} disconnected components")
                
                if connectivity_metrics.get('avg_degree', 0) < 1:
                    warnings.append(f"Low average degree: {connectivity_metrics['avg_degree']:.2f}")
            
            # Statistical validation
            if self.config.enable_statistical_checks:
                row, col = edge_index
                
                # Degree distribution
                in_degrees = torch.bincount(col, minlength=num_nodes)
                out_degrees = torch.bincount(row, minlength=num_nodes)
                
                metrics.update({
                    'avg_in_degree': in_degrees.float().mean().item(),
                    'avg_out_degree': out_degrees.float().mean().item(),
                    'max_in_degree': in_degrees.max().item(),
                    'max_out_degree': out_degrees.max().item(),
                    'degree_variance': in_degrees.float().var().item()
                })
                
                # Check for degree distribution anomalies
                if in_degrees.max().item() > num_nodes * 0.1:
                    warnings.append(f"High degree node detected: {in_degrees.max().item()} connections")
            
            execution_time = time.time() - start_time
            memory_usage = get_memory_usage() - start_memory
            
            confidence = self._compute_confidence(errors, warnings, metrics)
            is_valid = len(errors) == 0 and confidence >= self.config.confidence_threshold
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=execution_time,
                memory_usage=memory_usage
            )
            
            self._update_validation_history(result)
            return result
            
        except Exception as e:
            errors.append(f"Edge validation failed with exception: {e}")
            return self._create_failed_result(errors, warnings, metrics, start_time, start_memory)
    
    def validate_text_inputs(
        self,
        texts: List[str],
        min_length: int = 1,
        max_length: int = 1000,
        required_patterns: Optional[List[str]] = None
    ) -> ValidationResult:
        """Validate text inputs for hypernetwork processing.
        
        Args:
            texts: List of text descriptions
            min_length: Minimum text length
            max_length: Maximum text length
            required_patterns: Optional required patterns
            
        Returns:
            Validation result
        """
        start_time = time.time()
        start_memory = get_memory_usage()
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Basic type validation
            if not isinstance(texts, list):
                errors.append(f"Expected list, got {type(texts)}")
                return self._create_failed_result(errors, warnings, metrics, start_time, start_memory)
            
            if len(texts) == 0:
                errors.append("Empty text list provided")
                return self._create_failed_result(errors, warnings, metrics, start_time, start_memory)
            
            metrics['num_texts'] = len(texts)
            
            # Text content validation
            lengths = []
            empty_count = 0
            
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    errors.append(f"Text {i} is not a string: {type(text)}")
                    continue
                
                text_len = len(text.strip())
                lengths.append(text_len)
                
                if text_len == 0:
                    empty_count += 1
                    errors.append(f"Text {i} is empty")
                elif text_len < min_length:
                    warnings.append(f"Text {i} is too short: {text_len} < {min_length}")
                elif text_len > max_length:
                    warnings.append(f"Text {i} is too long: {text_len} > {max_length}")
            
            if lengths:
                metrics.update({
                    'avg_length': np.mean(lengths),
                    'min_length': min(lengths),
                    'max_length': max(lengths),
                    'std_length': np.std(lengths),
                    'empty_count': empty_count
                })
            
            # Pattern validation
            if required_patterns:
                import re
                pattern_matches = {pattern: 0 for pattern in required_patterns}
                
                for text in texts:
                    if isinstance(text, str):
                        for pattern in required_patterns:
                            if re.search(pattern, text):
                                pattern_matches[pattern] += 1
                
                metrics['pattern_matches'] = pattern_matches
                
                for pattern, count in pattern_matches.items():
                    if count == 0:
                        warnings.append(f"No texts match required pattern: {pattern}")
            
            # Content quality checks
            if self.config.enable_semantic_checks:
                quality_metrics = self._analyze_text_quality(texts)
                metrics.update(quality_metrics)
                
                if quality_metrics.get('avg_word_count', 0) < 3:
                    warnings.append("Texts appear to be very short (< 3 words average)")
                
                if quality_metrics.get('duplicate_ratio', 0) > 0.5:
                    warnings.append(f"High duplicate ratio: {quality_metrics['duplicate_ratio']:.2%}")
            
            execution_time = time.time() - start_time
            memory_usage = get_memory_usage() - start_memory
            
            confidence = self._compute_confidence(errors, warnings, metrics)
            is_valid = len(errors) == 0 and confidence >= self.config.confidence_threshold
            
            result = ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=execution_time,
                memory_usage=memory_usage
            )
            
            self._update_validation_history(result)
            return result
            
        except Exception as e:
            errors.append(f"Text validation failed with exception: {e}")
            return self._create_failed_result(errors, warnings, metrics, start_time, start_memory)
    
    def _validate_semantic_properties(
        self,
        tensor: torch.Tensor,
        metrics: Dict[str, Any],
        warnings: List[str]
    ) -> None:
        """Validate semantic properties of tensors."""
        try:
            # Check for common ML issues
            if tensor.dtype == torch.float32:
                # Check for gradient flow issues
                if hasattr(tensor, 'grad') and tensor.grad is not None:
                    grad_norm = torch.norm(tensor.grad).item()
                    metrics['grad_norm'] = grad_norm
                    
                    if grad_norm < 1e-8:
                        warnings.append("Very small gradient norm detected (vanishing gradients)")
                    elif grad_norm > 100:
                        warnings.append("Very large gradient norm detected (exploding gradients)")
                
                # Check for saturation
                if tensor.max().item() > 10 or tensor.min().item() < -10:
                    warnings.append("Potential activation saturation detected")
            
            # Check for rank deficiency
            if tensor.dim() == 2 and min(tensor.shape) > 1:
                try:
                    rank = torch.linalg.matrix_rank(tensor).item()
                    metrics['matrix_rank'] = rank
                    
                    if rank < min(tensor.shape) * 0.9:
                        warnings.append(f"Low rank matrix detected: {rank}/{min(tensor.shape)}")
                except:
                    pass  # Skip if computation fails
                    
        except Exception as e:
            warnings.append(f"Semantic validation failed: {e}")
    
    def _analyze_graph_connectivity(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Dict[str, Any]:
        """Analyze graph connectivity properties."""
        metrics = {}
        
        try:
            # Convert to adjacency representation
            row, col = edge_index
            
            # Compute degrees
            degrees = torch.bincount(torch.cat([row, col]), minlength=num_nodes)
            metrics['avg_degree'] = degrees.float().mean().item()
            
            # Estimate number of connected components (simplified)
            # In a real implementation, you'd use proper graph algorithms
            isolated_nodes = (degrees == 0).sum().item()
            metrics['isolated_nodes'] = isolated_nodes
            
            # Rough estimate of components
            if isolated_nodes == num_nodes:
                metrics['num_components'] = num_nodes
            elif edge_index.size(1) == 0:
                metrics['num_components'] = num_nodes
            else:
                # Simplified estimate
                metrics['num_components'] = max(1, isolated_nodes + 1)
            
            # Edge density
            max_edges = num_nodes * (num_nodes - 1) // 2
            actual_edges = edge_index.size(1)
            metrics['edge_density'] = actual_edges / max(max_edges, 1)
            
        except Exception as e:
            metrics['connectivity_error'] = str(e)
        
        return metrics
    
    def _analyze_text_quality(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text quality metrics."""
        metrics = {}
        
        try:
            word_counts = []
            unique_texts = set()
            
            for text in texts:
                if isinstance(text, str):
                    words = text.split()
                    word_counts.append(len(words))
                    unique_texts.add(text.lower().strip())
            
            if word_counts:
                metrics['avg_word_count'] = np.mean(word_counts)
                metrics['min_word_count'] = min(word_counts)
                metrics['max_word_count'] = max(word_counts)
            
            metrics['unique_texts'] = len(unique_texts)
            metrics['duplicate_ratio'] = 1 - len(unique_texts) / len(texts)
            
        except Exception as e:
            metrics['text_quality_error'] = str(e)
        
        return metrics
    
    def _compute_confidence(
        self,
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, Any]
    ) -> float:
        """Compute validation confidence score."""
        if errors:
            return 0.0
        
        # Start with high confidence
        confidence = 1.0
        
        # Reduce confidence based on warnings
        confidence -= len(warnings) * 0.05
        
        # Context-specific adjustments
        if self.config.context == ValidationContext.PRODUCTION:
            confidence -= len(warnings) * 0.05  # More strict in production
        
        # Metrics-based adjustments
        if 'nan_count' in metrics and metrics['nan_count'] > 0:
            confidence -= 0.3
        
        if 'inf_count' in metrics and metrics['inf_count'] > 0:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    def _create_failed_result(
        self,
        errors: List[str],
        warnings: List[str],
        metrics: Dict[str, Any],
        start_time: float,
        start_memory: float
    ) -> ValidationResult:
        """Create a failed validation result."""
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            execution_time=time.time() - start_time,
            memory_usage=get_memory_usage() - start_memory
        )
    
    def _create_successful_result(
        self,
        warnings: List[str],
        metrics: Dict[str, Any],
        start_time: float,
        start_memory: float
    ) -> ValidationResult:
        """Create a successful validation result."""
        confidence = self._compute_confidence([], warnings, metrics)
        return ValidationResult(
            is_valid=True,
            confidence=confidence,
            errors=[],
            warnings=warnings,
            metrics=metrics,
            execution_time=time.time() - start_time,
            memory_usage=get_memory_usage() - start_memory
        )
    
    def _update_validation_history(self, result: ValidationResult) -> None:
        """Update validation history for pattern analysis."""
        self.validation_history.append(result)
        
        # Keep only recent history
        if len(self.validation_history) > 1000:
            self.validation_history = self.validation_history[-1000:]
        
        # Update error patterns
        for error in result.errors:
            error_key = error[:50]  # First 50 chars as key
            self.error_patterns[error_key] = self.error_patterns.get(error_key, 0) + 1
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation history."""
        if not self.validation_history:
            return {'message': 'No validation history available'}
        
        recent_results = self.validation_history[-100:]  # Last 100 validations
        
        success_rate = sum(1 for r in recent_results if r.is_valid) / len(recent_results)
        avg_confidence = np.mean([r.confidence for r in recent_results])
        avg_execution_time = np.mean([r.execution_time for r in recent_results])
        
        return {
            'total_validations': len(self.validation_history),
            'recent_success_rate': success_rate,
            'avg_confidence': avg_confidence,
            'avg_execution_time': avg_execution_time,
            'common_error_patterns': dict(list(self.error_patterns.items())[:5])
        }


class ModelValidator:
    """Validator for model components and operations."""
    
    def __init__(self, config: ValidationConfig = None):
        """Initialize model validator.
        
        Args:
            config: Validation configuration
        """
        self.config = config or ValidationConfig()
        self.logger = get_logger(self.__class__.__name__)
    
    def validate_model_output(
        self,
        output: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_range: Optional[Tuple[float, float]] = None
    ) -> ValidationResult:
        """Validate model output tensor.
        
        Args:
            output: Model output tensor
            expected_shape: Expected output shape
            expected_range: Expected value range
            
        Returns:
            Validation result
        """
        start_time = time.time()
        start_memory = get_memory_usage()
        
        errors = []
        warnings = []
        metrics = {}
        
        try:
            # Basic validation
            if not isinstance(output, torch.Tensor):
                errors.append(f"Expected torch.Tensor, got {type(output)}")
                return ValidationResult(
                    is_valid=False, confidence=0.0, errors=errors, warnings=warnings,
                    metrics=metrics, execution_time=time.time() - start_time,
                    memory_usage=get_memory_usage() - start_memory
                )
            
            # Shape validation
            if expected_shape and output.shape != expected_shape:
                errors.append(f"Output shape mismatch: expected {expected_shape}, got {output.shape}")
            
            # Value validation
            nan_count = torch.isnan(output).sum().item()
            inf_count = torch.isinf(output).sum().item()
            
            if nan_count > 0:
                errors.append(f"Output contains {nan_count} NaN values")
            
            if inf_count > 0:
                errors.append(f"Output contains {inf_count} infinite values")
            
            # Range validation
            if expected_range:
                min_val = output.min().item()
                max_val = output.max().item()
                
                if min_val < expected_range[0]:
                    warnings.append(f"Output minimum {min_val} below expected range {expected_range}")
                
                if max_val > expected_range[1]:
                    warnings.append(f"Output maximum {max_val} above expected range {expected_range}")
            
            # Gradient validation
            if output.requires_grad and output.grad is not None:
                grad_nan = torch.isnan(output.grad).sum().item()
                grad_inf = torch.isinf(output.grad).sum().item()
                
                if grad_nan > 0:
                    errors.append(f"Output gradients contain {grad_nan} NaN values")
                
                if grad_inf > 0:
                    errors.append(f"Output gradients contain {grad_inf} infinite values")
                
                grad_norm = torch.norm(output.grad).item()
                metrics['grad_norm'] = grad_norm
                
                if grad_norm < 1e-8:
                    warnings.append("Very small gradient norm (vanishing gradients)")
                elif grad_norm > 100:
                    warnings.append("Very large gradient norm (exploding gradients)")
            
            # Statistical metrics
            metrics.update({
                'mean': output.mean().item(),
                'std': output.std().item(),
                'min': output.min().item(),
                'max': output.max().item(),
                'shape': list(output.shape),
                'dtype': str(output.dtype),
                'device': str(output.device)
            })
            
            execution_time = time.time() - start_time
            memory_usage = get_memory_usage() - start_memory
            
            confidence = 1.0 - len(warnings) * 0.1
            if errors:
                confidence = 0.0
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=execution_time,
                memory_usage=memory_usage
            )
            
        except Exception as e:
            errors.append(f"Model output validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                errors=errors,
                warnings=warnings,
                metrics=metrics,
                execution_time=time.time() - start_time,
                memory_usage=get_memory_usage() - start_memory
            )


def create_validator(
    level: str = "standard",
    context: str = "training",
    **kwargs
) -> GraphValidator:
    """Create a configured validator.
    
    Args:
        level: Validation level (minimal, standard, strict, paranoid)
        context: Validation context (training, inference, research, production)
        **kwargs: Additional configuration options
        
    Returns:
        Configured validator
    """
    config = ValidationConfig(
        level=ValidationLevel(level),
        context=ValidationContext(context),
        **kwargs
    )
    
    return GraphValidator(config)


def validate_hypergnn_inputs(
    edge_index: torch.Tensor,
    node_features: torch.Tensor,
    node_texts: List[str],
    validator: Optional[GraphValidator] = None
) -> Tuple[bool, List[str]]:
    """Convenience function for validating HyperGNN inputs.
    
    Args:
        edge_index: Edge connectivity tensor
        node_features: Node feature tensor
        node_texts: Node text descriptions
        validator: Optional validator instance
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    if validator is None:
        validator = create_validator()
    
    all_errors = []
    
    # Validate node features
    feature_result = validator.validate_node_features(node_features)
    if not feature_result.is_valid:
        all_errors.extend(feature_result.errors)
    
    # Validate edge structure
    edge_result = validator.validate_edge_structure(edge_index, node_features.size(0))
    if not edge_result.is_valid:
        all_errors.extend(edge_result.errors)
    
    # Validate text inputs
    text_result = validator.validate_text_inputs(node_texts)
    if not text_result.is_valid:
        all_errors.extend(text_result.errors)
    
    # Cross-validation
    if len(node_texts) != node_features.size(0):
        all_errors.append(
            f"Mismatch between node count ({node_features.size(0)}) "
            f"and text count ({len(node_texts)})"
        )
    
    return len(all_errors) == 0, all_errors