"""Advanced resilience patterns for production-ready HyperGNN systems.

This module implements sophisticated error recovery, circuit breakers,
retry mechanisms, and auto-healing capabilities for robust ML operations.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .logging_utils import get_logger
from .exceptions import GPUError, ModelError, NetworkError, ValidationError


logger = get_logger(__name__)


class HealthState(Enum):
    """Health states for circuit breaker pattern."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float
    error_type: str
    error_message: str
    retry_count: int = 0
    recovery_attempted: bool = False


class AdaptiveCircuitBreaker:
    """Intelligent circuit breaker that adapts to system conditions."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        success_threshold: int = 3,
        max_timeout: float = 300.0,
    ):
        """Initialize adaptive circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Base timeout before attempting recovery
            success_threshold: Consecutive successes needed to close circuit
            max_timeout: Maximum timeout duration
        """
        self.failure_threshold = failure_threshold
        self.base_recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.max_timeout = max_timeout
        
        self.failure_count = 0
        self.success_count = 0
        self.state = HealthState.HEALTHY
        self.last_failure_time = 0.0
        self.current_timeout = recovery_timeout
        
        self.error_history = deque(maxlen=100)
        self.lock = Lock()
        
        logger.info(f"Initialized AdaptiveCircuitBreaker: "
                   f"failure_threshold={failure_threshold}, "
                   f"recovery_timeout={recovery_timeout}s")
    
    def _should_attempt_call(self) -> bool:
        """Determine if call should be attempted based on circuit state."""
        with self.lock:
            current_time = time.time()
            
            if self.state == HealthState.HEALTHY:
                return True
            elif self.state == HealthState.DEGRADED:
                # Allow some calls through in degraded state
                return self.failure_count < self.failure_threshold // 2
            elif self.state == HealthState.FAILING:
                # Check if recovery timeout has passed
                if current_time - self.last_failure_time > self.current_timeout:
                    logger.info("Circuit breaker attempting recovery")
                    return True
                return False
            else:  # CRITICAL
                # Only attempt recovery after extended timeout
                if current_time - self.last_failure_time > self.max_timeout:
                    logger.warning("Circuit breaker attempting recovery from critical state")
                    return True
                return False
    
    def _record_success(self):
        """Record successful operation."""
        with self.lock:
            self.success_count += 1
            
            if self.state != HealthState.HEALTHY:
                if self.success_count >= self.success_threshold:
                    logger.info(f"Circuit breaker recovering: {self.success_count} consecutive successes")
                    self.state = HealthState.HEALTHY
                    self.failure_count = 0
                    self.current_timeout = self.base_recovery_timeout
    
    def _record_failure(self, error: Exception):
        """Record failed operation and update circuit state."""
        with self.lock:
            self.failure_count += 1
            self.success_count = 0
            self.last_failure_time = time.time()
            
            # Store error context
            error_ctx = ErrorContext(
                operation="circuit_breaker",
                timestamp=self.last_failure_time,
                error_type=type(error).__name__,
                error_message=str(error)
            )
            self.error_history.append(error_ctx)
            
            # Update state based on failure patterns
            if self.failure_count >= self.failure_threshold * 2:
                self.state = HealthState.CRITICAL
                self.current_timeout = min(self.current_timeout * 2, self.max_timeout)
            elif self.failure_count >= self.failure_threshold:
                self.state = HealthState.FAILING
                self.current_timeout = min(self.current_timeout * 1.5, self.max_timeout)
            elif self.failure_count >= self.failure_threshold // 2:
                self.state = HealthState.DEGRADED
            
            logger.warning(f"Circuit breaker state: {self.state.value}, "
                          f"failures: {self.failure_count}, "
                          f"timeout: {self.current_timeout}s")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not self._should_attempt_call():
            raise ModelError(
                "circuit_breaker", 
                "call_blocked",
                f"Circuit breaker is {self.state.value} - call blocked"
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "current_timeout": self.current_timeout,
            "error_rate": len([e for e in self.error_history 
                             if time.time() - e.timestamp < 60]) / min(len(self.error_history), 60)
        }


class ExponentialBackoffRetry:
    """Intelligent retry mechanism with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize exponential backoff retry.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries
            max_delay: Maximum delay between retries
            exponential_base: Base for exponential backoff
            jitter: Whether to add random jitter
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        
        logger.info(f"Initialized ExponentialBackoffRetry: "
                   f"max_retries={max_retries}, base_delay={base_delay}s")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            import random
            delay = delay * (0.5 + 0.5 * random.random())
        
        return delay
    
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function succeeded after {attempt} retries")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. "
                                 f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    break
        
        # All retries exhausted
        raise ModelError(
            "retry_mechanism",
            f"max_retries_exceeded_{self.max_retries}",
            f"Operation failed after {self.max_retries + 1} attempts. Last error: {last_exception}"
        )


class ResourceGuard:
    """Intelligent resource management and protection."""
    
    def __init__(
        self,
        memory_limit_gb: float = 4.0,
        cpu_limit_percent: float = 80.0,
        gpu_memory_limit_gb: float = 8.0,
    ):
        """Initialize resource guard.
        
        Args:
            memory_limit_gb: Maximum memory usage in GB
            cpu_limit_percent: Maximum CPU usage percentage
            gpu_memory_limit_gb: Maximum GPU memory usage in GB
        """
        self.memory_limit_gb = memory_limit_gb
        self.cpu_limit_percent = cpu_limit_percent
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        
        self.resource_history = deque(maxlen=60)  # 60 seconds of history
        
        logger.info(f"Initialized ResourceGuard: "
                   f"memory_limit={memory_limit_gb}GB, "
                   f"cpu_limit={cpu_limit_percent}%, "
                   f"gpu_limit={gpu_memory_limit_gb}GB")
    
    def check_resources(self) -> Dict[str, Any]:
        """Check current resource usage."""
        import psutil
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_gb = memory.used / (1024**3)
        memory_percent = memory.percent
        
        # GPU memory (if available)
        gpu_memory_gb = 0.0
        gpu_memory_percent = 0.0
        if torch.cuda.is_available():
            try:
                gpu_memory_bytes = torch.cuda.memory_allocated()
                gpu_memory_gb = gpu_memory_bytes / (1024**3)
                gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_percent = (gpu_memory_bytes / gpu_total_bytes) * 100
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        resource_info = {
            "cpu_percent": cpu_percent,
            "memory_gb": memory_gb,
            "memory_percent": memory_percent,
            "gpu_memory_gb": gpu_memory_gb,
            "gpu_memory_percent": gpu_memory_percent,
            "timestamp": time.time(),
        }
        
        self.resource_history.append(resource_info)
        return resource_info
    
    def is_resources_healthy(self) -> Tuple[bool, List[str]]:
        """Check if resource usage is within healthy limits."""
        resource_info = self.check_resources()
        warnings = []
        
        if resource_info["memory_gb"] > self.memory_limit_gb:
            warnings.append(f"Memory usage ({resource_info['memory_gb']:.2f}GB) "
                          f"exceeds limit ({self.memory_limit_gb}GB)")
        
        if resource_info["cpu_percent"] > self.cpu_limit_percent:
            warnings.append(f"CPU usage ({resource_info['cpu_percent']:.1f}%) "
                          f"exceeds limit ({self.cpu_limit_percent}%)")
        
        if resource_info["gpu_memory_gb"] > self.gpu_memory_limit_gb:
            warnings.append(f"GPU memory usage ({resource_info['gpu_memory_gb']:.2f}GB) "
                          f"exceeds limit ({self.gpu_memory_limit_gb}GB)")
        
        return len(warnings) == 0, warnings
    
    @contextmanager
    def resource_protection(self):
        """Context manager for resource-protected execution."""
        # Check resources before execution
        healthy, warnings = self.is_resources_healthy()
        if not healthy:
            logger.warning(f"Starting execution with resource warnings: {warnings}")
        
        start_time = time.time()
        start_resources = self.check_resources()
        
        try:
            yield
        finally:
            # Check resources after execution
            end_time = time.time()
            end_resources = self.check_resources()
            
            duration = end_time - start_time
            memory_delta = end_resources["memory_gb"] - start_resources["memory_gb"]
            
            logger.info(f"Resource usage during execution ({duration:.2f}s): "
                       f"Memory delta: {memory_delta:+.3f}GB, "
                       f"Final CPU: {end_resources['cpu_percent']:.1f}%, "
                       f"Final GPU: {end_resources['gpu_memory_gb']:.2f}GB")


class AutoHealingManager:
    """Advanced auto-healing system for ML models."""
    
    def __init__(self, model: nn.Module):
        """Initialize auto-healing manager.
        
        Args:
            model: PyTorch model to monitor and heal
        """
        self.model = model
        self.initial_state = self._capture_model_state()
        self.healing_strategies = {}
        self.healing_history = []
        
        self._register_default_strategies()
        
        logger.info("Initialized AutoHealingManager with default healing strategies")
    
    def _capture_model_state(self) -> Dict[str, Any]:
        """Capture current model state for healing purposes."""
        return {
            "parameters": {name: param.clone().detach() 
                         for name, param in self.model.named_parameters()},
            "buffers": {name: buffer.clone().detach() 
                       for name, buffer in self.model.named_buffers()},
            "training": self.model.training,
        }
    
    def _register_default_strategies(self):
        """Register default healing strategies."""
        self.healing_strategies.update({
            "nan_weights": self._heal_nan_weights,
            "exploding_gradients": self._heal_exploding_gradients,
            "dead_neurons": self._heal_dead_neurons,
            "memory_overflow": self._heal_memory_overflow,
        })
    
    def _heal_nan_weights(self) -> bool:
        """Heal NaN weights by reinitializing them."""
        healed = False
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                logger.warning(f"Healing NaN weights in parameter: {name}")
                if name in self.initial_state["parameters"]:
                    param.data.copy_(self.initial_state["parameters"][name])
                else:
                    # Reinitialize with Xavier/Kaiming
                    if param.dim() >= 2:
                        nn.init.xavier_uniform_(param)
                    else:
                        nn.init.zeros_(param)
                healed = True
        return healed
    
    def _heal_exploding_gradients(self) -> bool:
        """Heal exploding gradients by clipping."""
        max_norm = 1.0
        total_norm = 0.0
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > max_norm:
            logger.warning(f"Healing exploding gradients: norm={total_norm:.4f}")
            clip_coef = max_norm / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
            return True
        
        return False
    
    def _heal_dead_neurons(self) -> bool:
        """Heal dead neurons by reinitializing them."""
        healed = False
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Check for dead neurons (weights and gradients close to zero)
                weight_mean = module.weight.abs().mean(dim=1)
                dead_mask = weight_mean < 1e-6
                
                if dead_mask.any():
                    dead_count = dead_mask.sum().item()
                    logger.warning(f"Healing {dead_count} dead neurons in {name}")
                    
                    # Reinitialize dead neurons
                    with torch.no_grad():
                        for i in range(len(dead_mask)):
                            if dead_mask[i]:
                                nn.init.xavier_uniform_(module.weight[i:i+1])
                                if module.bias is not None:
                                    nn.init.zeros_(module.bias[i:i+1])
                    
                    healed = True
        
        return healed
    
    def _heal_memory_overflow(self) -> bool:
        """Heal memory overflow by clearing caches."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache for memory healing")
                return True
        except Exception as e:
            logger.error(f"Failed to clear CUDA cache: {e}")
        
        return False
    
    def diagnose_and_heal(self) -> Dict[str, Any]:
        """Diagnose model issues and attempt healing."""
        diagnosis = {
            "issues_found": [],
            "healing_attempted": [],
            "healing_successful": [],
            "timestamp": time.time(),
        }
        
        # Diagnose issues
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                diagnosis["issues_found"].append("nan_weights")
            if torch.isinf(param).any():
                diagnosis["issues_found"].append("inf_weights")
        
        # Check gradients if available
        if any(p.grad is not None for p in self.model.parameters()):
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), float('inf'))
            if total_norm > 10.0:
                diagnosis["issues_found"].append("exploding_gradients")
        
        # Attempt healing for found issues
        for issue in set(diagnosis["issues_found"]):
            if issue in self.healing_strategies:
                diagnosis["healing_attempted"].append(issue)
                try:
                    success = self.healing_strategies[issue]()
                    if success:
                        diagnosis["healing_successful"].append(issue)
                except Exception as e:
                    logger.error(f"Healing strategy {issue} failed: {e}")
        
        # Record healing history
        self.healing_history.append(diagnosis)
        
        if diagnosis["healing_successful"]:
            logger.info(f"Successfully healed: {diagnosis['healing_successful']}")
        
        return diagnosis


def resilient_model_call(
    circuit_breaker: Optional[AdaptiveCircuitBreaker] = None,
    retry_strategy: Optional[ExponentialBackoffRetry] = None,
    resource_guard: Optional[ResourceGuard] = None,
    auto_healer: Optional[AutoHealingManager] = None,
):
    """Decorator for resilient model calls with comprehensive protection."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Resource protection
            if resource_guard:
                with resource_guard.resource_protection():
                    return _execute_with_resilience(
                        func, args, kwargs, circuit_breaker, retry_strategy, auto_healer
                    )
            else:
                return _execute_with_resilience(
                    func, args, kwargs, circuit_breaker, retry_strategy, auto_healer
                )
        
        return wrapper
    return decorator


def _execute_with_resilience(
    func: Callable,
    args: Tuple,
    kwargs: Dict,
    circuit_breaker: Optional[AdaptiveCircuitBreaker],
    retry_strategy: Optional[ExponentialBackoffRetry],
    auto_healer: Optional[AutoHealingManager],
) -> Any:
    """Execute function with resilience strategies."""
    
    def protected_call():
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Attempt auto-healing if available
            if auto_healer and isinstance(e, (ModelError, GPUError)):
                logger.info("Attempting auto-healing due to model error")
                healing_result = auto_healer.diagnose_and_heal()
                
                if healing_result["healing_successful"]:
                    logger.info("Auto-healing successful, retrying operation")
                    return func(*args, **kwargs)
            
            raise
    
    # Apply circuit breaker protection
    if circuit_breaker:
        if retry_strategy:
            return circuit_breaker.call(retry_strategy.retry, protected_call)
        else:
            return circuit_breaker.call(protected_call)
    elif retry_strategy:
        return retry_strategy.retry(protected_call)
    else:
        return protected_call()


class ResilientModelWrapper(nn.Module):
    """Wrapper that adds resilience features to any PyTorch model."""
    
    def __init__(
        self,
        model: nn.Module,
        enable_circuit_breaker: bool = True,
        enable_auto_healing: bool = True,
        enable_resource_guard: bool = True,
    ):
        """Initialize resilient model wrapper.
        
        Args:
            model: Base PyTorch model to wrap
            enable_circuit_breaker: Enable circuit breaker protection
            enable_auto_healing: Enable auto-healing capabilities
            enable_resource_guard: Enable resource usage monitoring
        """
        super().__init__()
        self.base_model = model
        
        # Initialize resilience components
        self.circuit_breaker = AdaptiveCircuitBreaker() if enable_circuit_breaker else None
        self.retry_strategy = ExponentialBackoffRetry()
        self.resource_guard = ResourceGuard() if enable_resource_guard else None
        self.auto_healer = AutoHealingManager(model) if enable_auto_healing else None
        
        logger.info(f"Initialized ResilientModelWrapper with features: "
                   f"circuit_breaker={enable_circuit_breaker}, "
                   f"auto_healing={enable_auto_healing}, "
                   f"resource_guard={enable_resource_guard}")
    
    @resilient_model_call()
    def forward(self, *args, **kwargs):
        """Forward pass with resilience protection."""
        return self.base_model(*args, **kwargs)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        status = {
            "timestamp": time.time(),
            "overall_health": "healthy",
        }
        
        if self.circuit_breaker:
            circuit_metrics = self.circuit_breaker.get_health_metrics()
            status["circuit_breaker"] = circuit_metrics
            if circuit_metrics["state"] != "healthy":
                status["overall_health"] = "degraded"
        
        if self.resource_guard:
            healthy, warnings = self.resource_guard.is_resources_healthy()
            status["resources"] = {
                "healthy": healthy,
                "warnings": warnings,
                "current": self.resource_guard.check_resources()
            }
            if not healthy:
                status["overall_health"] = "degraded"
        
        if self.auto_healer:
            status["auto_healing"] = {
                "healing_history": self.auto_healer.healing_history[-5:],  # Last 5 events
                "total_healings": len(self.auto_healer.healing_history),
            }
        
        return status
    
    def __getattr__(self, name):
        """Delegate attribute access to base model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)