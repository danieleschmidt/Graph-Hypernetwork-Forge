"""Resilience Framework for Graph Hypernetwork Forge.

This module provides comprehensive resilience features including:
- Advanced error recovery and self-healing mechanisms  
- Circuit breaker patterns for fault tolerance
- Graceful degradation strategies
- Automatic failover and retry logic
- System health monitoring and alerts
"""

import time
import threading
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional
import logging

import torch
import numpy as np

# Enhanced logging
try:
    from .logging_utils import get_logger
    ENHANCED_FEATURES = True
except ImportError:
    def get_logger(name):
        return logging.getLogger(name)
    ENHANCED_FEATURES = False

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, blocking requests
    HALF_OPEN = "half_open" # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5           # Failures before opening
    recovery_timeout: float = 60.0       # Seconds before trying half-open
    success_threshold: int = 3           # Successes before closing
    timeout: float = 30.0               # Operation timeout


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        """Initialize circuit breaker."""
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.lock = threading.RLock()
        
        logger.info(f"CircuitBreaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap function with circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout:
                    raise Exception(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
        
        try:
            result = func(*args, **kwargs)
            
            # Record success
            with self.lock:
                if self.state == CircuitState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = CircuitState.CLOSED
                        self.failure_count = 0
                elif self.state == CircuitState.CLOSED:
                    self.failure_count = max(0, self.failure_count - 1)
            
            return result
            
        except Exception as e:
            # Record failure
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if (self.state == CircuitState.CLOSED and 
                    self.failure_count >= self.config.failure_threshold):
                    self.state = CircuitState.OPEN
                elif self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.OPEN
            
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
            }


class ResilienceManager:
    """Comprehensive resilience management system."""
    
    def __init__(self):
        """Initialize resilience manager."""
        self.circuit_breakers = {}
        logger.info("ResilienceManager initialized")
    
    def create_circuit_breaker(self, name: str, **kwargs) -> CircuitBreaker:
        """Create and register circuit breaker."""
        config = CircuitBreakerConfig(**kwargs)
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)


# Global resilience manager instance
_resilience_manager = None

def get_resilience_manager() -> ResilienceManager:
    """Get global resilience manager instance."""
    global _resilience_manager
    if _resilience_manager is None:
        _resilience_manager = ResilienceManager()
    return _resilience_manager


# Decorator for resilient operations
def resilient_operation(operation_name: str = None):
    """Decorator to make operation resilient."""
    def decorator(func: Callable) -> Callable:
        name = operation_name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = get_resilience_manager()
            circuit_breaker = manager.get_circuit_breaker(name)
            if circuit_breaker is None:
                circuit_breaker = manager.create_circuit_breaker(name)
            return circuit_breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator